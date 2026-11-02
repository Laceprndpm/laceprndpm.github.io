"""CPU checks for training/basic-train.md; requires PyTorch, no GPU required.

Run from the note repository root:
    python assets/basic-train/verify_basic_train.py
    python assets/basic-train/verify_basic_train.py --json /tmp/basic-train-results.json
"""

import argparse
import copy
import json
import platform
from pathlib import Path

import torch


DTYPE = torch.float64


def tensor(value, requires_grad=False):
    return torch.tensor(value, dtype=DTYPE, requires_grad=requires_grad)


def new_params():
    return {
        "W1": tensor([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0]], True),
        "b1": tensor([0.0, 0.0, 0.0], True),
        "W2": tensor([[1.0, -1.0], [-2.0, 1.0], [1.0, 2.0]], True),
        "b2": tensor([0.0, 0.0], True),
    }


def forward(p, x):
    z = x @ p["W1"] + p["b1"]
    h = z.relu()
    return h @ p["W2"] + p["b2"]


def loss_fn(p, x, y):
    return 0.5 * (forward(p, x) - y).square().sum(dim=-1).mean()


def close(actual, expected, atol=1e-12):
    torch.testing.assert_close(actual, torch.as_tensor(expected, dtype=actual.dtype),
                               atol=atol, rtol=1e-12)


def hand_calculation():
    p = new_params()
    x, y = tensor([[1.0, 2.0]], True), tensor([[0.0, 0.0]])
    loss = loss_fn(p, x, y)
    close(forward(p, x), [[1.0, -2.0]])
    close(loss, 2.5)
    loss.backward()
    expected = {"W1": [[3.0, -4.0, 0.0], [6.0, -8.0, 0.0]],
                "b1": [3.0, -4.0, 0.0],
                "W2": [[3.0, -6.0], [1.0, -2.0], [0.0, 0.0]], "b2": [1.0, -2.0]}
    for name in p:
        close(p[name].grad, expected[name])
    close(x.grad, [[7.0, -1.0]])

    # Independent finite differences, away from ReLU's nondifferentiable point.
    max_error = 0.0
    with torch.no_grad():
        for param in p.values():
            flat = param.view(-1)
            for i in range(flat.numel()):
                old = flat[i].item()
                flat[i] = old + 1e-6
                plus = loss_fn(p, x, y).item()
                flat[i] = old - 1e-6
                minus = loss_fn(p, x, y).item()
                flat[i] = old
                numerical = (plus - minus) / 2e-6
                max_error = max(max_error, abs(numerical - param.grad.view(-1)[i].item()))
    assert max_error < 1e-8

    optimizer = torch.optim.SGD(list(p.values()), lr=0.01)
    optimizer.step()
    close(p["W1"], [[0.97, -0.96, 1.0], [0.94, 1.08, -1.0]])
    close(p["b1"], [-0.03, 0.04, 0.0])
    close(p["W2"], [[0.97, -0.94], [-2.01, 1.02], [1.0, 2.0]])
    close(p["b2"], [-0.01, 0.02])
    close(forward(p, x), [[0.233, -1.366]])
    close(loss_fn(p, x, y), 0.9601225)
    return {"gradients": expected, "input_gradient": x.grad.tolist(),
            "loss_before": loss.item(), "loss_after_sgd": loss_fn(p, x, y).item(),
            "finite_difference_max_abs_error": max_error}


def accumulation():
    # Uneven microbatches (1 + 2) reveal incorrect averaging of batch means.
    x = tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 3.0]])
    y = tensor([[0.0, 0.0], [1.0, 0.0], [-1.0, 1.0]])
    full, split, wrong = new_params(), new_params(), new_params()
    loss_fn(full, x, y).backward()
    for start, end in [(0, 1), (1, 3)]:
        loss = loss_fn(split, x[start:end], y[start:end])
        (loss * ((end - start) / len(x))).backward()
        (loss_fn(wrong, x[start:end], y[start:end]) / 2).backward()
    for name in full:
        close(split[name].grad, full[name].grad)
    max_wrong_error = max((wrong[n].grad - full[n].grad).abs().max().item() for n in full)
    assert max_wrong_error > 0.1

    opt_full = torch.optim.Adam(list(full.values()), lr=0.01, foreach=False)
    opt_split = torch.optim.Adam(list(split.values()), lr=0.01, foreach=False)
    opt_full.step()
    opt_split.step()
    for name in full:
        close(split[name], full[name])
        for key in ["step", "exp_avg", "exp_avg_sq"]:
            close(opt_split.state[split[name]][key], opt_full.state[full[name]][key])
    return {"microbatch_sizes": [1, 2], "correct_normalization_matches": True,
            "wrong_mean_of_means_max_gradient_error": max_wrong_error,
            "adam_parameters_and_states_match": True}


def adam_states():
    p = new_params()
    x, y = tensor([[1.0, 2.0]]), tensor([[0.0, 0.0]])
    optimizer = torch.optim.Adam(list(p.values()), lr=0.01, betas=(0.9, 0.999),
                                 eps=1e-8, foreach=False)
    manual = {name: {"param": param.detach().clone(), "m": torch.zeros_like(param),
                     "v": torch.zeros_like(param)} for name, param in p.items()}
    rows = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss_fn(p, x, y).backward()
        g = p["W2"].grad[0, 0].item()
        for name, param in p.items():
            s = manual[name]
            s["m"] = 0.9 * s["m"] + 0.1 * param.grad
            s["v"] = 0.999 * s["v"] + 0.001 * param.grad.square()
            s["param"] -= 0.01 * (s["m"] / (1 - 0.9**step)) / (
                (s["v"] / (1 - 0.999**step)).sqrt() + 1e-8)
        optimizer.step()
        for name, param in p.items():
            close(param, manual[name]["param"])
            close(optimizer.state[param]["exp_avg"], manual[name]["m"])
            close(optimizer.state[param]["exp_avg_sq"], manual[name]["v"])
            close(optimizer.state[param]["step"], step)
        state = optimizer.state[p["W2"]]
        m, v = state["exp_avg"][0, 0].item(), state["exp_avg_sq"][0, 0].item()
        rows.append({"step": step, "gradient_W2_0": g, "m": m, "v": v,
                     "m_hat": m / (1 - 0.9**step), "v_hat": v / (1 - 0.999**step),
                     "weight_W2_0": p["W2"][0, 0].item(),
                     "loss_after_step": loss_fn(p, x, y).item()})

    # Same weights and fresh gradients; only the optimizer history differs.
    restored = {name: param.detach().clone().requires_grad_() for name, param in p.items()}
    reset = {name: param.detach().clone().requires_grad_() for name, param in p.items()}
    restored_opt = torch.optim.Adam(list(restored.values()), lr=0.01, foreach=False)
    reset_opt = torch.optim.Adam(list(reset.values()), lr=0.01, foreach=False)
    restored_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for params, opt in [(restored, restored_opt), (reset, reset_opt)]:
        loss_fn(params, x, y).backward()
        opt.step()
    gap = max((restored[n] - reset[n]).abs().max().item() for n in restored)
    assert gap > 1e-6
    return {"two_step_trace": rows, "reset_optimizer_weight_difference": gap}


def adamw_decay():
    p = tensor([1.0], True)
    p.grad = tensor([0.0])
    optimizer = torch.optim.AdamW([p], lr=0.01, weight_decay=0.1, foreach=False)
    optimizer.step()
    close(p, [0.999])
    close(optimizer.state[p]["exp_avg"], [0.0])
    close(optimizer.state[p]["exp_avg_sq"], [0.0])
    return {"weight_before": 1.0, "weight_after_zero_gradient_step": p.item()}


def optimizer_comparison():
    x, y = tensor([[1.0, 2.0]]), tensor([[0.0, 0.0]])
    traces = {}
    for kind in ["sgd", "momentum", "adam"]:
        p = new_params()
        if kind == "sgd":
            optimizer = torch.optim.SGD(list(p.values()), lr=0.01, foreach=False)
        elif kind == "momentum":
            optimizer = torch.optim.SGD(list(p.values()), lr=0.01, momentum=0.9,
                                        dampening=0.0, nesterov=False, foreach=False)
        else:
            optimizer = torch.optim.Adam(list(p.values()), lr=0.01, betas=(0.9, 0.999),
                                         eps=1e-8, foreach=False)
        manual = {name: {"param": param.detach().clone(), "u": torch.zeros_like(param),
                         "m": torch.zeros_like(param), "v": torch.zeros_like(param)}
                  for name, param in p.items()}
        rows = []
        for step in (1, 2):
            optimizer.zero_grad(set_to_none=True)
            loss_fn(p, x, y).backward()
            g = p["W2"].grad[0, 0].item()
            old_weight = p["W2"][0, 0].item()
            for name, param in p.items():
                state = manual[name]
                if kind == "sgd":
                    direction = param.grad
                elif kind == "momentum":
                    state["u"] = 0.9 * state["u"] + param.grad
                    direction = state["u"]
                else:
                    state["m"] = 0.9 * state["m"] + 0.1 * param.grad
                    state["v"] = 0.999 * state["v"] + 0.001 * param.grad.square()
                    direction = (state["m"] / (1 - 0.9**step)) / (
                        (state["v"] / (1 - 0.999**step)).sqrt() + 1e-8)
                state["param"] -= 0.01 * direction
            optimizer.step()
            for name, param in p.items():
                close(param, manual[name]["param"])
                if kind == "momentum":
                    close(optimizer.state[param]["momentum_buffer"], manual[name]["u"])
                elif kind == "adam":
                    close(optimizer.state[param]["exp_avg"], manual[name]["m"])
                    close(optimizer.state[param]["exp_avg_sq"], manual[name]["v"])
            if kind == "sgd":
                assert not optimizer.state
            new_weight = p["W2"][0, 0].item()
            rows.append({"step": step, "gradient_W2_0": g,
                         "subtracted_update": old_weight - new_weight,
                         "direction": (old_weight - new_weight) / 0.01,
                         "weight_W2_0": new_weight,
                         "loss_after_step": loss_fn(p, x, y).item()})
        traces[kind] = rows
    close(tensor(traces["momentum"][1]["gradient_W2_0"]), 0.65706)
    close(tensor(traces["momentum"][1]["direction"]), 3.35706)
    close(tensor(traces["momentum"][1]["weight_W2_0"]), 0.9364294)
    return traces


def momentum_and_rmsprop_examples():
    parameter = tensor([1.0], True)
    optimizer = torch.optim.SGD([parameter], lr=0.01, momentum=0.9, foreach=False)
    reversal = []
    for gradient in [3.0, -3.0]:
        old_weight = parameter.detach().clone()
        parameter.grad = tensor([gradient])
        optimizer.step()
        reversal.append({"gradient": gradient,
                         "momentum_buffer": optimizer.state[parameter]["momentum_buffer"].item(),
                         "subtracted_update": (old_weight - parameter).item()})
    close(optimizer.state[parameter]["momentum_buffer"], [-0.3])

    parameter = tensor([1.0, 1.0], True)
    optimizer = torch.optim.RMSprop([parameter], lr=0.01, alpha=0.9, eps=1e-8,
                                    momentum=0.0, centered=False, foreach=False)
    expected = parameter.detach().clone()
    average = torch.zeros_like(parameter)
    rmsprop = []
    for gradients in [[1.0, 10.0], [2.0, -20.0]]:
        parameter.grad = tensor(gradients)
        average = 0.9 * average + 0.1 * parameter.grad.square()
        expected -= 0.01 * parameter.grad / (average.sqrt() + 1e-8)
        optimizer.step()
        close(parameter, expected)
        close(optimizer.state[parameter]["square_avg"], average)
        rmsprop.append({"gradient": gradients, "square_avg": average.tolist(),
                        "parameter": parameter.detach().tolist()})
    return {"momentum_reversal": reversal, "rmsprop_manual_matches": rmsprop}


def coupled_l2_vs_adamw():
    output = {}
    for kind in ["adam_l2", "adamw"]:
        parameter = tensor([1.0], True)
        parameter.grad = tensor([0.0])
        factory = torch.optim.Adam if kind == "adam_l2" else torch.optim.AdamW
        optimizer = factory([parameter], lr=0.01, weight_decay=0.1,
                            betas=(0.9, 0.999), eps=1e-8, foreach=False)
        optimizer.step()
        state = optimizer.state[parameter]
        if kind == "adam_l2":
            close(parameter, [1.0 - 0.01 * 0.1 / (0.1 + 1e-8)])
            close(state["exp_avg"], [0.01])
            close(state["exp_avg_sq"], [0.00001])
        else:
            close(parameter, [0.999])
            close(state["exp_avg"], [0.0])
            close(state["exp_avg_sq"], [0.0])
        output[kind] = {"weight": parameter.item(), "m": state["exp_avg"].item(),
                        "v": state["exp_avg_sq"].item()}
    fixed_denominators = tensor([1.0, 10.0])
    l2_contribution = 0.01 * 0.1 * tensor([1.0, 1.0]) / fixed_denominators
    close(l2_contribution, [0.001, 0.0001])
    output["fixed_denominator_l2_contribution"] = l2_contribution.tolist()
    return output


def master_weights():
    traces = {}
    for dtype in [torch.float16, torch.bfloat16]:
        low = torch.tensor(1.0, dtype=dtype)
        main = torch.tensor(1.0, dtype=torch.float32)
        delta = torch.tensor(1e-4, dtype=torch.float32)
        rows = []
        for step in range(1, 21):
            # Even FP32 subtraction loses the update if its result is stored in low precision.
            low = (low.float() - delta).to(dtype)
            main -= delta
            model = main.to(dtype)
            if step in [1, 2, 3, 10, 20]:
                rows.append({"step": step, "low_only": low.item(), "main_fp32": main.item(),
                             "model_from_main": model.item()})
        assert low.item() == 1.0
        assert model.item() < 1.0
        traces[str(dtype)] = rows

    # Underflow and loss scaling use a separate example from update rounding.
    tiny = torch.tensor(1e-8, dtype=torch.float32)
    unscaled = tiny.half().float().item()
    recovered = (tiny.mul(1024).half().float() / 1024).item()
    assert unscaled == 0.0
    assert abs(recovered - 1e-8) / 1e-8 < 0.01
    return {"rounding_traces": traces,
            "tiny_gradient": {"without_scaling": unscaled, "scaled_then_unscaled": recovered}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "device": "cpu", "reference_dtype": str(DTYPE)},
              "hand_calculation": hand_calculation(), "accumulation": accumulation(),
              "adam": adam_states(), "adamw": adamw_decay(), "precision": master_weights(),
              "optimizer_comparison": optimizer_comparison(),
              "momentum_and_rmsprop": momentum_and_rmsprop_examples(),
              "coupled_l2_vs_adamw": coupled_l2_vs_adamw(),
              "status": "PASS"}
    output = json.dumps(report, indent=2, ensure_ascii=False)
    if args.json:
        args.json.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
