"""Validate the DP-Training note using 8 local CPU/Gloo ranks.

Three optimizer steps compare a 64-sample reference batch with:
  1. DDP synchronizing every microbatch;
  2. DDP using no_sync for the first 3 of 4 microbatches;
  3. An educational sharded Adam implementation using real ReduceScatter/AllGather.

The sharded implementation is not Megatron, DeepSpeed, or FSDP. All numeric
checks use float64 and do not measure GPU performance or mixed-precision memory.
"""

from contextlib import nullcontext
from datetime import timedelta
import argparse
import json
import os
from pathlib import Path
import platform
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


WORLD = 8
ACCUMULATION = 4
MICROBATCH = 2
STEPS = 3
MAX_NORM = 0.25
DTYPE = torch.float64


def new_model():
    model = nn.Sequential(nn.Linear(2, 3), nn.ReLU(), nn.Linear(3, 2)).to(dtype=DTYPE)
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., -1., 1.], [1., 1., -1.]], dtype=DTYPE).T)
        model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor([[1., -1.], [-2., 1.], [1., 2.]], dtype=DTYPE).T)
        model[2].bias.zero_()
    return model


def data(step):
    ids = torch.arange(WORLD * ACCUMULATION * MICROBATCH, dtype=DTYPE)
    x = torch.stack(((ids % 11 + 1) / 10, ((ids * 3 + step) % 13 + 1) / 10), dim=1)
    y = torch.stack((0.2 * x[:, 0] - 0.3 * x[:, 1], 0.1 + 0.4 * x[:, 0]), dim=1)
    return x, y


def loss_fn(prediction, target):
    return 0.5 * (prediction - target).square().sum(dim=-1).mean()


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def flat_grads(model):
    return torch.cat([p.grad.detach().reshape(-1) for p in model.parameters()])


def flat_state(model, optimizer, name):
    return torch.cat([optimizer.state[p][name].reshape(-1) for p in model.parameters()])


def check(a, b):
    torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    return (a - b).abs().max().item()


def new_optimizer(params):
    return torch.optim.Adam(params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, foreach=False)


def reference_step(model, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    x, y = data(step)
    loss = loss_fn(model(x), y)
    loss.backward()
    gradient = flat_grads(model).clone()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM, foreach=False)
    optimizer.step()
    return gradient, norm.item(), loss.item()


def mean_hook(state, bucket):
    # A custom DDP hook must implement the world-size averaging itself.
    state['calls'] += 1
    state['elements'] += bucket.buffer().numel()
    buffer = bucket.buffer().div_(WORLD)
    return dist.all_reduce(buffer, async_op=True).get_future().then(lambda future: future.value()[0])


def run_ddp(rank, delayed):
    model, reference = new_model(), new_model()
    ddp = DDP(model, bucket_cap_mb=25)
    counter = {'calls': 0, 'elements': 0}
    ddp.register_comm_hook(counter, mean_hook)
    optimizer = new_optimizer(model.parameters())
    ref_optimizer = new_optimizer(reference.parameters())
    max_grad_error = max_param_error = max_state_error = 0.0
    traces = []
    for step in range(STEPS):
        reference_gradient, reference_norm, reference_loss = reference_step(reference, ref_optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        x, y = data(step)
        params_before = flat_params(model).clone()
        calls_before = counter['calls']
        per_microbatch_calls = []
        for micro in range(ACCUMULATION):
            start = (rank * ACCUMULATION + micro) * MICROBATCH
            context = ddp.no_sync() if delayed and micro < ACCUMULATION - 1 else nullcontext()
            with context:
                loss = loss_fn(ddp(x[start:start + MICROBATCH]), y[start:start + MICROBATCH])
                (loss / ACCUMULATION).backward()
            check(flat_params(model), params_before)
            per_microbatch_calls.append(counter['calls'] - calls_before)
        max_grad_error = max(max_grad_error, check(flat_grads(model), reference_gradient))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM, foreach=False)
        assert abs(norm.item() - reference_norm) < 1e-11
        optimizer.step()
        max_param_error = max(max_param_error, check(flat_params(model), flat_params(reference)))
        for name in ['exp_avg', 'exp_avg_sq']:
            max_state_error = max(max_state_error, check(flat_state(model, optimizer, name),
                                                       flat_state(reference, ref_optimizer, name)))
        for p in model.parameters():
            assert optimizer.state[p]['step'].item() == step + 1
        expected = [0, 0, 0, 1] if delayed else [1, 2, 3, 4]
        assert per_microbatch_calls == expected, per_microbatch_calls
        traces.append({'step': step + 1, 'cumulative_gradient_collectives_by_microbatch': per_microbatch_calls,
                       'global_gradient_norm': norm.item(), 'reference_loss_before_step': reference_loss})
    return {'max_gradient_abs_error': max_grad_error, 'max_parameter_abs_error': max_param_error,
            'max_adam_state_abs_error': max_state_error, 'gradient_allreduce_calls': counter['calls'],
            'gradient_elements_presented_to_collectives': counter['elements'], 'trace': traces}


def padded(tensor, size):
    result = torch.zeros(size, dtype=DTYPE)
    result[:tensor.numel()].copy_(tensor)
    return result


def copy_flat_to_model(flat, model):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            count = param.numel()
            param.copy_(flat[offset:offset + count].view_as(param))
            offset += count


def run_sharded_adam(rank):
    model, reference = new_model(), new_model()
    count = flat_params(model).numel()
    per_rank = (count + WORLD - 1) // WORLD
    total = per_rank * WORLD
    start, end = rank * per_rank, (rank + 1) * per_rank
    shard = nn.Parameter(padded(flat_params(model), total)[start:end].clone())
    optimizer = new_optimizer([shard])
    ref_optimizer = new_optimizer(reference.parameters())
    max_grad_error = max_param_error = max_state_error = max_norm_error = 0.0
    first_local_gradient = None
    for step in range(STEPS):
        reference_gradient, reference_norm, _ = reference_step(reference, ref_optimizer, step)
        model.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        x, y = data(step)
        for micro in range(ACCUMULATION):
            offset = (rank * ACCUMULATION + micro) * MICROBATCH
            (loss_fn(model(x[offset:offset + MICROBATCH]), y[offset:offset + MICROBATCH]) / ACCUMULATION).backward()
        local_gradient = flat_grads(model)
        if step == 0:
            first_local_gradient = local_gradient[0].item()
        gradient_shard = torch.empty(per_rank, dtype=DTYPE)
        dist.reduce_scatter_tensor(gradient_shard, padded(local_gradient, total), op=dist.ReduceOp.SUM)
        gradient_shard.div_(WORLD)
        max_grad_error = max(max_grad_error, check(gradient_shard, padded(reference_gradient, total)[start:end]))

        # Each gradient element has exactly one owner; sum disjoint contributions.
        norm_squared = gradient_shard.square().sum()
        dist.all_reduce(norm_squared, op=dist.ReduceOp.SUM)
        norm = norm_squared.sqrt()
        max_norm_error = max(max_norm_error, abs(norm.item() - reference_norm))
        assert max_norm_error < 1e-11
        clip_coefficient = torch.clamp(MAX_NORM / (norm + 1e-6), max=1.0)
        shard.grad = gradient_shard.mul(clip_coefficient)
        optimizer.step()

        gathered = torch.empty(total, dtype=DTYPE)
        dist.all_gather_into_tensor(gathered, shard.detach())
        copy_flat_to_model(gathered[:count], model)
        max_param_error = max(max_param_error, check(flat_params(model), flat_params(reference)))
        for name in ['exp_avg', 'exp_avg_sq']:
            expected = padded(flat_state(reference, ref_optimizer, name), total)[start:end]
            max_state_error = max(max_state_error, check(optimizer.state[shard][name], expected))
        assert optimizer.state[shard]['step'].item() == step + 1
    return {'parameter_count': count, 'padded_parameter_count': total,
            'valid_owned_elements': max(0, min(end, count) - start),
            'first_step_first_parameter_local_gradient': first_local_gradient,
            'max_gradient_abs_error': max_grad_error, 'max_parameter_abs_error': max_param_error,
            'max_adam_state_abs_error': max_state_error, 'max_global_norm_abs_error': max_norm_error,
            'algorithm_collectives_per_step': {'gradient_reduce_scatter': 1, 'norm_allreduce': 1,
                                              'parameter_allgather': 1}}


def worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank,
                            world_size=WORLD, timeout=timedelta(seconds=120))
    try:
        results = {'rank': rank, 'sample_ids': list(range(rank * 8, (rank + 1) * 8)),
                   'ddp_sync_each_microbatch': run_ddp(rank, delayed=False),
                   'ddp_sync_last_microbatch': run_ddp(rank, delayed=True),
                   'educational_sharded_adam': run_sharded_adam(rank)}
        ranks = [None] * WORLD if rank == 0 else None
        dist.gather_object(results, ranks, dst=0)
        if rank == 0:
            assert sorted(i for r in ranks for i in r['sample_ids']) == list(range(64))
            report = {'status': 'PASS', 'environment': {'python': platform.python_version(),
                      'torch': torch.__version__, 'backend': 'gloo', 'device': 'cpu', 'dtype': str(DTYPE)},
                      'configuration': {'world_size': WORLD, 'microbatch_size_per_rank': MICROBATCH,
                      'microbatches_per_step': ACCUMULATION, 'global_batch_size': 64,
                      'optimizer_steps': STEPS, 'max_gradient_norm': MAX_NORM},
                      'scope': 'Actual CPU DDP and educational ReduceScatter/sharded-Adam/AllGather, not GPU or framework ZeRO execution.',
                      'ranks': ranks}
            Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='dp-training-') as tmp:
        mp.start_processes(worker, args=(str(Path(tmp) / 'rendezvous'), str(args.json.resolve())),
                           nprocs=WORLD, join=True, start_method='fork')
    report = json.loads(args.json.read_text())
    print(json.dumps({'status': report['status'], 'configuration': report['configuration'],
                      'rank_0': report['ranks'][0]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
