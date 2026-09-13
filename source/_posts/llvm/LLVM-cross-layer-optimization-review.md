---
title: LLVM 跨层优化与诊断复盘
date: 2026-09-13 18:33:36 +08:00
tags:
  - LLVM
  - AI-Infra
categories:
  - 编译器
mathjax: true
---

# LLVM 跨层优化与诊断复盘

**面向读者：** 在算子优化中，想知道“这个优化为什么编译器做不了”的算子工程师。本文沿 IR、分析结果、调度与性能追踪原因，帮助读者判断编译器缺少什么信息、在哪一步受阻，以及应该修改哪一层。

**核心问题与结论：** 一项优化没有发生或没有提速，阻塞点在哪里？先定位行为首次出现的层，再区分 Legality、Profitability、Transformation 与 Pipeline Interaction，最后用单变量实验检查结构、正确性与性能。允许更多重排不保证最终更快。

**范围与依据：** 本文为原理教程与案例复盘，覆盖 C/C++ 语义、LLVM IR、Analysis、Transformation、MIR 和后端调度。命令按 LLVM 22 风格整理；案例保留原稿中的 IR/MIR 观察和性能记录，本轮结构整理没有重新执行实验。通用概念参考文末官方资料，案例所缺版本及附件另行占位。

阅读顺序：第 1–8 节建立概念与工具，第 9–10 节连接案例，第 11–13 节整理排查方法和未解问题。已有基础的读者可先看两个案例，再按需回查前文。

## 目录

- [1. LLVM 优化全链路](#1-llvm-优化全链路)
- [2. LLVM IR 核心计算模型](#2-llvm-ir-核心计算模型)
- [3. 常见 Analysis](#3-常见-analysis)
- [4. 常见优化与实现手段](#4-常见优化与实现手段)
- [5. 如何定位优化发生或失败](#5-如何定位优化发生或失败)
- [6. LLVM Pass 的结构与契约](#6-llvm-pass-的结构与契约)
- [7. Backend、MIR 与性能解释](#7-backendmir-与性能解释)
- [8. Clang、opt、llc 常用命令](#8-clangoptllc-常用命令)
- [9. 案例一：Alias、restrict 与并行归约](#9-案例一aliasrestrict-与并行归约)
- [10. 案例二：MMO、调度 DAG 与启发式失配](#10-案例二mmo调度-dag-与启发式失配)
- [11. 通用排查清单](#11-通用排查清单)
- [12. 一句话术语表](#12-一句话术语表)
- [13. 可迁移的方法与未解问题](#13-可迁移的方法与未解问题)
- [14. 参考与复现材料](#14-参考与复现材料)

---

# 1. LLVM 优化全链路

## 1.1 总体链路

```text
C/C++ Source
  ↓ Clang Frontend
LLVM IR（前端生成）
  ↓ Middle-end Analysis / Transformation
LLVM IR（优化后）
  ↓ Instruction Selection / Lowering
Machine IR（虚拟寄存器）
  ↓ pre-RA MachineScheduler
Machine IR（已调度，仍是虚拟寄存器）
  ↓ Register Allocation
Machine IR（物理寄存器，可能有 spill/reload）
  ↓ post-RA scheduling / pseudo expansion / packetization
MCInst / Assembly
  ↓ Assembler
Object File
  ↓ Linker
Executable / Shared Library
```

## 1.2 每层负责什么

| 层 | 主要对象 | 主要问题 |
|---|---|---|
| Clang Frontend | AST、初始 LLVM IR | 源语言语义如何编码成 IR 属性、metadata 和控制流？ |
| Middle-end | LLVM IR | 哪些变换合法、是否值得、由哪个 Pass 执行？ |
| Instruction Selection | SelectionDAG / GlobalISel、MIR | 通用 IR 操作选择成哪些目标 opcode？ |
| pre-RA Scheduler | SUnit、SDep、ReadyQ | 在虚拟寄存器约束下如何重排以隐藏延迟？ |
| Register Allocation | live range、物理寄存器 | 如何映射寄存器；是否发生 spill/reload？ |
| post-RA Backend | 物理 MIR、bundle | 在固定寄存器后做局部调度、pseudo 展开和打包。 |
| MC / Assembler / Linker | 指令编码、符号、重定位 | 如何形成目标文件和最终可执行文件？ |

## 1.3 “行为最早出现在哪层”

同一现象可能有不同的“首次出现”位置：

| 问题 | 最早可见位置 |
|---|---|
| 源码没有表达 `restrict` 契约 | C/C++ / Clang Frontend |
| AA 返回 `MayAlias` | LLVM IR 的 AA 查询 |
| 内存身份在 lowering 中丢失 | ISel 后 MIR 的 MMO |
| `MayAlias` 真正变成不可跨越的调度边 | pre-RA MachineScheduler DAG |
| 调度后寄存器压力转化为 spill | Register Allocation |
| 最终 bundle、周期和硬件 stall | Assembly / simulator / hardware |

因此不能只说“问题在 LLVM”。必须回答：

1. 根因在哪层产生？
2. 哪层首次能观察到？
3. 哪个消费者把它变成实际限制？
4. 最终性能在哪层被验证？

---

# 2. LLVM IR 核心计算模型

## 2.1 Module、Function、BasicBlock、Instruction

- `Module`：一个 LLVM 编译单元，包含函数、全局变量、声明和 metadata。
- `Function`：参数、属性和若干 BasicBlock 的集合。
- `BasicBlock`：单入口、以 terminator 结束的直线指令序列。
- `Instruction`：既是操作，也是产生 SSA 值的 `Value`。

## 2.2 CFG

CFG（Control Flow Graph）以 BasicBlock 为节点，以 `br`、`switch` 等控制转移为边。

CFG回答：

- 哪些路径可能执行？
- 某个块的前驱和后继是谁？
- 一个定义是否在所有到达路径上都可用？
- 循环的 header、latch、exit 在哪里？

注意：MachineScheduler DAG 中的蓝色虚线“control dependency”不是 CFG 分支边；它泛指非 `Data` 的调度依赖。

## 2.3 SSA

SSA（Static Single Assignment）要求每个 SSA 名字只定义一次：

```llvm
%x = add i32 %a, %b
%y = mul i32 %x, 4
```

循环中的“更新”通过新值和 `phi` 表达，而不是覆盖旧变量。

## 2.4 Phi

`phi` 根据控制流来自哪个前驱块选择值：

```llvm
%i = phi i32 [ 0, %entry ], [ %next, %latch ]
```

它表达的是块入口处的值合流，不是一条普通运行时 move。

## 2.5 Use-Def

Use-Def 链连接“值在哪里定义”和“在哪里使用”。显式寄存器依赖可直接沿 SSA 边追踪；内存依赖不能仅靠 Use-Def，需要 AA、MemorySSA 或其他内存分析。

## 2.6 Dominance

若从函数入口到 B 的每条路径都经过 A，则 A dominate B。一个普通 SSA 定义必须 dominate 它的所有使用；循环回边合流通常需要 `phi`。

## 2.7 GEP

`getelementptr` 只计算地址，不读取内存：

```llvm
%p = getelementptr float, ptr %base, i64 %i
%v = load float, ptr %p
```

它基于类型和索引计算偏移；`inbounds` 还附带更强的地址范围契约。

## 2.8 显式依赖与隐式内存依赖

```llvm
%old = load i32, ptr %x
store i32 7, ptr %p
%now = load i32, ptr %x
```

两次 load 没有 SSA 数据边，但若 `%p` 可能 alias `%x`，store 就可能 clobber 第二次 load。因此优化器必须通过 AA 判断。

## 2.9 常见属性与语义

| 属性/标记 | 含义 |
|---|---|
| `noalias` | 约束基于该参数/返回值派生出的相关内存访问；不是“地址在全世界没有别名”。 |
| `readonly` | 不通过该参数路径写内存；不保证其他 alias 路径不会写。 |
| `writeonly` | 通过该参数只写不读。 |
| `nonnull` | 该指针不能为 null。 |
| `dereferenceable(N)` | 至少有 N 字节可以被安全解引用。 |
| `noundef` | 值不能含未定义位；与 `nonnull` 无关。 |
| `volatile` | 每次访问本身可观察，通常不能被删除或合并。 |
| `reassoc` | 允许浮点运算重新结合，可能改变舍入、NaN、符号零等行为。 |
| TBAA metadata | 用源语言类型规则辅助 alias 判断。 |

LLVM verifier主要检查类型、SSA支配、Phi、CFG和属性放置等 IR 结构合法性；它不会证明你的 Transformation 与原程序语义严格等价。若 Pass 主动增加 `reassoc` 或 `noalias`，正确性责任在 Pass 和调用契约。

---

# 3. 常见 Analysis

Analysis提供事实，Transformation消费事实。Analysis通常不是固定处于“Loop前或后”的一次性 Pass；关键是消费者在什么 IR 状态下查询它。

| Analysis | 回答的问题 | 常见消费者 |
|---|---|---|
| DominatorTree | 哪个块/指令支配另一个？ | LICM、GVN、循环分析、SSA维护 |
| LoopInfo | 哪些块构成 natural loop；header/latch/exit 是什么？ | Loop Pass、Vectorizer、Unroll |
| ScalarEvolution | 循环变量、trip count、地址递推和范围能否符号化？ | Unroll、Vectorizer、LoopAccessAnalysis |
| Alias Analysis | 两个 `MemoryLocation` 是 No/May/Must/PartialAlias？ | GVN、LICM、Vectorizer、DSE |
| MemorySSA | 哪个 MemoryDef 可能到达并 clobber 某次 MemoryUse？ | LICM、DSE、GVN 等内存优化 |
| LoopAccessAnalysis | 循环中的内存访问是否可并行，是否需要 runtime check？ | LoopVectorizer |
| TargetTransformInfo | 某种 IR 操作在目标机器上的成本如何？ | Vectorizer、Unroll、InstCombine 等 |
| BlockFrequencyInfo | 哪些路径热，块预计执行多少次？ | Inlining、布局、收益模型 |

## 3.1 Alias Analysis

AA结果：

- `NoAlias`：两个位置不重叠；
- `MayAlias`：无法证明不重叠；不是证明它们一定重叠；
- `MustAlias`：指向同一位置；
- `PartialAlias`：部分范围重叠。

工程上最常见的观察方式：

```bash
opt -passes=aa-eval \
  -print-all-alias-modref-info \
  -disable-output test.ll
```

正式 Pass 中使用：

```cpp
AAResults &AA = FAM.getResult<AAManager>(F);
AliasResult R = AA.alias(LocA, LocB);
```

不能只凭 IR 中没有 `noalias` 就断言 `MayAlias`，因为 BasicAA、GEP、TBAA、范围等仍可能证明 `NoAlias`。

## 3.2 MemorySSA

MemorySSA把内存行为抽象成：

- `MemoryDef`：可能改变内存状态的写或调用；
- `MemoryUse`：读取内存；
- `MemoryPhi`：控制流合流处的内存状态合并。

它适合回答“这个 load 的最近潜在 clobber 是谁”，但最终仍可能调用 AA 判断两个位置是否真正冲突。

## 3.3 LoopInfo 与 ScalarEvolution

LoopInfo识别循环结构；ScalarEvolution进一步将递推表示为 add recurrence，例如：

```text
{0,+,1}<loop>
```

表示从0开始、每次迭代加1。若编译器无法推导 trip count、范围或地址递推，Unroll和Vectorizer的 legality/profitability 都可能受限。

## 3.4 Analysis 的生命周期

Transformation修改 IR 后：

- 返回 `PreservedAnalyses::all()`：声明所有分析仍有效；
- 返回 `PreservedAnalyses::none()`：声明分析结果全部失效；
- 也可以只保留确认未被破坏的分析。

错误地保留失效分析可能导致后续 Pass 使用陈旧事实。

---

# 4. 常见优化与实现手段

每项优化统一从五个问题理解：

```text
目标
→ 所需 Analysis
→ Legality
→ Transformation
→ Profitability
```

## 4.1 CFG 优化

| 手段 | 作用 | 常见条件/风险 |
|---|---|---|
| Constant Folding | 编译期计算常量表达式。 | 操作语义必须允许折叠。 |
| SCCP | 在 SSA 和可达 CFG 上传播常量。 | 只能沿被证明可达的边传播。 |
| SimplifyCFG | 折叠常量分支、合并块、删除空跳转。 | 不能改变可观察副作用和异常路径。 |
| Jump Threading | 根据已知条件复制/穿越块，跳过无用分支。 | 受代码膨胀和条件可证明性限制。 |
| Loop Simplify / LCSSA | 规范化循环结构，便于后续 Loop Pass。 | 多为规范化，不直接保证提速。 |

## 4.2 标量与数据流优化

| 手段 | 作用 |
|---|---|
| `mem2reg` | 将可提升的栈变量变为 SSA register 和 Phi。 |
| SROA | 将 aggregate/alloca 拆成更小的标量对象。 |
| InstCombine | 使用局部代数规则规范化和折叠指令。 |
| EarlyCSE / GVN | 消除可证明等价的重复计算或重复 load。 |
| DCE / ADCE | 删除无用且无副作用的指令。 |
| DSE | 删除其结果必然被覆盖、未被观察的 store。 |

## 4.3 Code Motion

- **Hoist**：把循环或分支内不变计算提前；典型 Pass 是 LICM。
- **Sink**：把计算延后到真正需要它的路径，缩短 live range 或避免无用执行。
- **Load hoist**：除支配关系外，还必须证明移动过程中没有可能 clobber 它的 store/call。

Hoist/sink不是单纯“把文本顺序换一下”；必须同时满足控制等价、内存安全、异常/poison语义和收益条件。

## 4.4 Loop 优化

| 手段 | 核心效果 | 常见限制 |
|---|---|---|
| Loop Rotate | 将循环变成适合优化的 canonical 结构。 | CFG变化、额外入口/出口。 |
| Loop Unroll | 复制循环体，减少分支并暴露 ILP。 | 代码膨胀、寄存器压力、余数循环。 |
| Unroll-and-Jam | 展开外层并融合内层副本。 | 完美/适合的嵌套结构与合法内存依赖。 |
| Loop Interchange | 交换嵌套循环顺序。 | 依赖方向必须允许；局部性和成本模型要获益。 |
| Loop Unswitch | 将循环不变量条件移到循环外，生成多个版本。 | 代码膨胀。 |
| Loop Distribution | 将一个循环拆成多个循环。 | 必须保持跨语句依赖。 |
| Loop Fusion | 合并相邻循环，改善局部性并减少控制开销。 | trip count、边界和依赖必须兼容。 |
| Loop Vectorization | 将跨迭代标量操作变为向量操作。 | 归约语义、alias、trip count、target cost。 |
| SLP Vectorization | 在基本块内组合等构的标量操作。 | lane结构、依赖和成本。 |
| Software Pipelining | 交叠不同迭代的操作。 | 通常要求真实 machine loop、可计算依赖和资源模型。 |

### Unroll不等于并行

普通展开可能仍保留一条串行 accumulator 链：

```text
acc0 → add0 → add1 → add2 → add3
```

只有在整数语义允许，或浮点指令具有 `reassoc`/fast-math授权时，才能改造成多条独立归约链：

```text
acc0 → add0
acc1 → add1
acc2 → add2
acc3 → add3
       ↓ tree reduction
```

浮点重新结合并非严格 bitwise 等价；它通常提高吞吐，也可能改变舍入误差，不能以“平均精度可能更好”为由绕过语言语义。

## 4.5 内存优化

- Load forwarding：用已知 store 值替代后续 load。
- Load CSE：合并未被 clobber 的重复 load。
- Runtime versioning：运行前检查指针范围，无危险重叠时走优化版本。
- Prefetch/preload：提前发起 load 隐藏延迟，但会拉长 live range。
- Store merging / memcpy optimization：合并相邻访问或识别库函数模式。

`restrict/noalias` 能补充 legality 证据，但不会自动保证优化有收益。

## 4.6 函数间优化

- Inlining：把调用体复制到调用点，暴露常量和跨函数优化机会。
- IPSCCP：跨函数传播常量。
- Devirtualization：将可证明目标的间接调用变为直接调用。
- Function attributes inference：推导 `readonly`、`nofree`、`nounwind` 等属性。

Inlining既能暴露优化，也会增加代码尺寸和寄存器压力。

---

# 5. 如何定位优化发生或失败

## 5.1 四分法

### Legality

变换是否保持语言和 IR 契约？常见阻塞：

- `MayAlias`；
- 未知 call 的 ModRef；
- volatile/atomic/fence；
- 浮点不允许 reassociation；
- CFG、dominance、异常和 poison 语义；
- 循环依赖或无法计算 trip count。

### Profitability

合法，但成本模型认为不值得。常见原因：

- 代码膨胀；
- 寄存器压力和 spill 风险；
- target latency/throughput；
- 循环太短；
- 分支概率、profile和热度不足。

### Transformation

条件满足且决定执行后，Pass实际怎样修改 IR/MIR：复制块、创建 Phi、插入 runtime check、重写 Use-Def或生成新循环版本。

### Pipeline Interaction

目标 Pass可能：

- 根本未进入 pipeline；
- 因 IR形态不匹配而未触发；
- 已优化但被后续 Pass 合并、撤销或改写；
- 在观察前证据或目标指令已经被删除。

## 5.2 标准诊断流程

```text
1. 构造最小复现
2. 找行为首次出现的层
3. 确认消费该事实的 Pass
4. 查 legality 证据
5. 查 profitability 分支/评分
6. 只修改一个条件
7. 比较 IR/MIR/DAG/ASM
8. 验证 correctness
9. 测量真实性能
10. 写回归测试
```

## 5.3 如何具体到源码分支

工具由粗到细：

1. `-Rpass` / `-Rpass-missed` / `-Rpass-analysis`：先判断 Pass 给出的公开理由；
2. `-debug-pass-manager`：确认 Pass 是否运行；
3. `-print-before/after` 或 `-print-changed`：定位 IR首次变化；
4. `-debug-only=<pass>`：查看内部判断与候选评分；
5. Debug build + GDB：在候选 legality/profitability 分支打断点，查看哪一行返回 false。

没有任何单一工具能同时回答“是否 MayAlias”和“它是否就是本次优化失败原因”。需要 Analysis结果和消费方决策两类证据。

---

# 6. LLVM Pass 的结构与契约

## 6.1 Pass的输入和输出

Function Pass典型接口：

```cpp
PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
```

- 输入对象：当前 `Function` 和 Analysis Manager；
- 可读取：BasicBlock、Instruction、CFG、Analysis结果；
- 可输出：诊断信息，或直接修改当前 IR；
- 返回值：哪些 Analysis结果在修改后仍然有效。

Pass通常不是“输入一个 Function，返回另一个 Function”；它原地修改 IR，并返回分析保留契约。

## 6.2 Analysis Pass 与 Transformation Pass

- Analysis：计算并缓存事实，不应修改 IR。
- Transformation：消费事实并修改 IR。
- Utility：如 `IRBuilder`、LoopUnroll工具函数，提供变换实现，但本身不一定是独立 Pass。

## 6.3 最小 New Pass Manager 插件

```cpp
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

class InspectPass : public PassInfoMixin<InspectPass> {
public:
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM) {
    errs() << "Function: " << F.getName() << "\n";
    for (BasicBlock &BB : F)
      for (Instruction &I : BB)
        errs() << I << "\n";
    return PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "IRStarter", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name != "inspect-ir")
                    return false;
                  FPM.addPass(InspectPass());
                  return true;
                });
          }};
}
```

## 6.4 `PreservedAnalyses`

```cpp
return PreservedAnalyses::all();  // 没有破坏任何分析事实
return PreservedAnalyses::none(); // 修改可能使所有分析失效
```

不是“想丢弃 Analysis 就返回 none”，而是向 Pass Manager报告：本次修改后哪些缓存仍可信。

## 6.5 Options 与包装 Pass

`LoopUnrollOptions`这类对象用于把策略传给通用变换实现，例如 count、partial、runtime、peeling：

- `Count`：目标展开倍数；
- `Partial`：trip count不能整除时，是否生成部分展开和余数处理；
- `Runtime`：trip count仅运行时已知时，是否生成runtime remainder逻辑；
- `Peeling`：先单独执行若干次迭代，使后续循环满足更好条件。

Options不是都必须显式设置；未设置项使用默认策略。包装 `LoopUnrollPass` 的价值是复用LLVM维护的CFG、Phi、LCSSA和余数循环变换，而不是手工复制块。

---

# 7. Backend、MIR 与性能解释

## 7.1 为什么需要 MIR

LLVM IR仍是目标无关、无限SSA值的抽象。MIR加入了后端必须处理的约束：

- 目标 opcode；
- register class和subregister；
- implicit def/use、regmask、calling convention；
- MachineMemOperand；
- sched class、latency和processor resource；
- pseudo instruction、bundle和目标特定状态。

## 7.2 SSA值与虚拟寄存器

- LLVM SSA值表达目标无关的数据流；
- MIR虚拟寄存器已经属于目标register class，但数量尚不受物理寄存器容量限制；
- Register Allocation将虚拟寄存器live range映射到有限物理寄存器，必要时插入spill/reload。

虚拟寄存器阶段使Instruction Selection和MachineScheduler能够先在目标指令语义下工作，而无需过早固定物理寄存器。

## 7.3 MIR常见信息

```text
%16:vrm2 = PseudoVLE32_V_M2 ...
  :: (load unknown-size from %ir.12, align 4, !tbaa !22)
```

其中包含：

- `%16:vrm2`：虚拟寄存器和register class；
- `PseudoVLE32_V_M2`：目标pseudo opcode；
- `killed/dead/implicit`：liveness和隐式寄存器信息；
- `debug-location`：源码映射；
- `:: (...)`：MMO。

## 7.4 `mayLoad/mayStore` 与 MMO

- `mayLoad/mayStore`：opcode级保守属性，表示这类指令可能读/写内存；
- MMO：当前这条MachineInstr具体访问什么对象、范围、对齐，以及是否volatile/atomic。

```text
mayStore=1 + 精确 MMO
→ scheduler/AA有机会证明与其他对象独立

mayStore=1 + 无 MMO
→ 不知道写到哪里
→ 可能被视为unknown/global ordered memory
→ 建立过宽Order/Barrier边
```

## 7.5 MachineScheduler DAG

每条可调度MachineInstr通常对应一个SUnit；SDep表示依赖：

| SDep | 含义 |
|---|---|
| Data | RAW真依赖，consumer等待producer结果。 |
| Anti | WAR反依赖。 |
| Output | WAW输出依赖。 |
| Order | 内存、Barrier、Cluster或其他顺序约束。 |

图形DAG中：

- 默认实线通常是Data边；
- 蓝色虚线表示所有非Data依赖，不只表示CFG控制流；
- 青色虚线表示artificial依赖。

颜色只能证明“有非Data顺序约束”；精确区分 `MayAliasMem`、`MustAliasMem`、`Barrier` 需要文本DAG或调试日志。

## 7.6 READY

某SUnit进入ReadyQ表示其尚未满足的前驱依赖已经解除，可以成为调度候选。READY不表示一定立即issue；它还要与其他候选比较：

- critical path；
- latency reduction；
- processor resource；
- register pressure；
- target-specific score；
- top-down/bottom-up边界策略。

因此：

```text
未READY → legality/dependency问题
已READY但未选 → profitability/heuristic/resource问题
```

## 7.7 Latency、Throughput和资源

- Latency：producer结果到consumer可用所需的依赖距离；
- Throughput/issue rate：同类指令能多频繁发射；
- Resource occupancy：执行单元被占用多久；
- ReleaseAtCycles：调度模型中资源预订区间；
- Hazard recognizer：表达简单资源计数无法描述的目标约束。

ISA通常定义指令功能和架构状态，不统一规定每条指令的cycle延迟。Latency主要由具体微架构决定，编译器TableGen和模拟器必须分别校准到同一硬件事实。

## 7.8 pre-RA与post-RA

- pre-RA Scheduler：有虚拟寄存器自由度，可以较大范围重排，但只能估计未来物理寄存器压力；
- RA：固定物理寄存器并可能插入spill/reload；
- post-RA Scheduler：寄存器已固定，信息更真实，但自由度更小，不能轻易延长或重构live range。

Preload通常优先在pre-RA调度考虑：此时既能看到目标指令和MMO，又有足够重排空间。IR层负责提供独立性，post-RA更多做有限修整。

---

# 8. Clang、opt、llc 常用命令

以下命令按LLVM 22风格整理；具体隐藏选项以当前build的 `--help-hidden` 为准。

## 8.1 Clang：源码到IR

```bash
# 前端IR，尽量不运行Middle-end优化
clang -O2 -Xclang -disable-llvm-passes \
  -S -emit-llvm test.c -o raw.ll

# 优化后的IR
clang -O2 -S -emit-llvm test.c -o opt.ll

# 保留易读名称
clang -O2 -fno-discard-value-names \
  -S -emit-llvm test.c -o test.ll

# 向量化诊断
clang -O3 \
  -Rpass=loop-vectorize \
  -Rpass-missed=loop-vectorize \
  -Rpass-analysis=loop-vectorize \
  test.c -c
```

## 8.2 `opt`：Middle-end

```bash
# 运行默认O2 pipeline
opt -passes='default<O2>' -S raw.ll -o optimized.ll

# 运行单个或组合Pass
opt -passes='function(mem2reg,instcombine,simplifycfg)' \
  -S test.ll -o changed.ll

# 加载外部Pass Plugin
opt -load-pass-plugin=build/IRStarter.so \
  -passes='function(mem2reg,inspect-ir),verify' \
  -S test.ll -o changed.ll

# 查看Pass Manager执行顺序
opt -passes='default<O2>' -debug-pass-manager \
  -disable-output test.ll

# 查看某Pass前后IR
opt -passes='default<O2>' \
  -print-before=loop-vectorize \
  -print-after=loop-vectorize \
  -filter-print-funcs=foo \
  -disable-output test.ll

# AA
opt -passes=aa-eval \
  -print-all-alias-modref-info \
  -disable-output test.ll

# 常见Analysis结构
opt -passes='print<domtree>' -disable-output test.ll
opt -passes='print<loops>' -disable-output test.ll
opt -passes='print<scalar-evolution>' -disable-output test.ll
opt -passes='print<memoryssa>' -disable-output test.ll

# 验证IR结构
opt -passes=verify -disable-output changed.ll
```

## 8.3 `llc`：IR到MIR/汇编/目标文件

```bash
LLC=/opt/build-rv-assert/bin/llc

# RISC-V汇编
$LLC test.ll \
  -mtriple=riscv64-unknown-linux-gnu \
  -mcpu=generic-rv64 \
  -mattr=+m,+a,+f,+d,+c,+v \
  -target-abi=lp64d \
  -O2 -o test.s

# ISel结束后的MIR
$LLC test.ll \
  -mtriple=riscv64-unknown-linux-gnu \
  -mcpu=generic-rv64 \
  -mattr=+m,+a,+f,+d,+c,+v \
  -target-abi=lp64d \
  -O2 -stop-after=finalize-isel \
  -o test.mir

# 对已有MIR只运行MachineScheduler
$LLC modified.mir \
  -run-pass=machine-scheduler \
  -verify-machineinstrs \
  -o scheduled.mir

# 图形化查看MachineScheduler DAG
$LLC modified.mir \
  -run-pass=machine-scheduler \
  -view-misched-dags \
  -misched-only-func=streams \
  -misched-only-block=2 \
  -o scheduled.mir

# 文本DAG，识别具体边类型
$LLC modified.mir \
  -run-pass=machine-scheduler \
  -misched-print-dags \
  -debug-only=machine-scheduler \
  -o scheduled.mir 2> scheduler.log

# 直接生成目标文件
$LLC test.ll -filetype=obj -o test.o
```

`-view-misched-dags`和`-debug-only`通常要求带Assertions的LLVM build。图形查看器需要Graphviz/xdot。

## 8.4 目标文件与汇编检查

```bash
llvm-objdump -d test.o
llvm-readobj --file-headers --sections --symbols test.o
llvm-nm test.o
```

汇编文本不等于目标文件：目标文件还包含机器码、符号表、section、重定位和调试信息；链接器再解析跨文件符号与重定位，形成可执行文件或共享库。

`llvm-mca`可以根据LLVM目标调度模型静态估计吞吐、资源压力和关键路径，但它不是模拟器，更不等于真实硬件周期：

```bash
llvm-mca -mtriple=riscv64 \
  -mcpu=<具体CPU> kernel.s
```

---

# 9. 案例一：Alias、restrict 与并行归约

## 9.1 `const&`不表示底层对象immutable

```cpp
int may_alias(const int &x, int *p) {
  int old = x;
  *p = 7;
  int now = x;
  return now - old;
}
```

Clang可以给 `%x` 参数标记 `readonly`，但 `%p` 仍可能alias `%x`，因此IR保留两次load：

```text
load x
store p
load x
```

AA得到 `MayAlias`。未知call则通常得到 `ModRef`，同样阻止复用第一次load。

## 9.2 `restrict/noalias`

C++使用Clang/GCC扩展：

```cpp
int non_alias(const int &__restrict x,
              int *__restrict p);
```

Clang可生成参数 `noalias`。若调用者违反该契约，使两条受约束访问冲突，程序行为未定义。

`noalias`是访问行为契约，不是简单的 `%a != %b`，也不是“该指针永远没有任何alias”。descriptor本身的 `noalias` 通常不会自动传播给从其字段load出来的任意地址。

## 9.3 Softmax向量化

无 `restrict` 时，LLVM仍可通过loop versioning生成：

```text
runtime pointer check
├─ 无危险重叠 → vector loop
└─ 可能重叠   → scalar loop
```

有 `restrict/noalias` 后，静态证明删除runtime alias check。结论不是“没有restrict就不能向量化”，而是：

```text
无restrict：运行时补证明
有restrict：编译期直接提供证明
```

## 9.4 四路展开与归约链

循环metadata中的：

```llvm
!{!"llvm.loop.unroll.count", i32 4}
```

只是向LoopUnroll提供策略信息。通用Unroll Pass负责复制循环体和生成余数循环；它不必自动把浮点accumulator拆成四条独立链。

普通浮点 `fadd` 要保持原结合顺序。加入：

```llvm
fadd reassoc float ...
```

后，编译器或自定义Pass才可以把单链改写为四个Phi accumulator，并在退出处tree reduction。实验最终IR出现：

```text
acc0 += a[i+0]
acc1 += a[i+1]
acc2 += a[i+2]
acc3 += a[i+3]
result = (acc0 + acc1) + (acc2 + acc3)
```

这暴露出四条独立浮点依赖链，允许隐藏约4-cycle ALU latency。语义授权来自 `reassoc`/fast-math，而不是LLVM verifier自动证明数值等价。

## 9.5 本案例的能力闭环

```text
IR观察两次load/归约链
→ aa-eval确认MayAlias/NoAlias
→ restrict或reassoc补充契约
→ LoopUnroll执行CFG变换
→ 自定义Pass拆分accumulator
→ verify验证IR结构
→ MIR/汇编检查是否形成独立链
```

---

# 10. 案例二：MMO、调度 DAG 与启发式失配

## 10.1 第一阶段：错误global barrier

DLC的 `setIA` 实际写隐藏IA地址生成状态，不是普通VMEM store。但旧建模为：

```text
set_ia_of_subcore
→ VS_setIAsub
→ mayStore=1
→ 最终MI没有MMO
→ hasOrderedMemoryRef()保守成立
→ scheduler建立global BarrierChain
```

结果是普通VMEM load/store即使已有DLC pack和NoAlias证明，也不能跨越 `setIA`。这是legality问题：后续节点根本尚未READY。

## 10.2 PSV + MMO修复

使用 `IAPGX0/IAPGX1` PseudoSourceValue表达隐藏状态：

```text
setIA(n)       : store IAPGXn
indexed load   : load VMEM + load IAPGXn
indexed store  : store VMEM + load IAPGXn
```

这样可以：

- 保留同一IA channel的set/use/overwrite顺序；
- 区分IA0和IA1；
- 允许普通VMEM和Vector ALU跨越setIA；
- 避免直接清除 `mayStore` 所造成的正确性风险。

这是“用更精确的内存身份收窄依赖”，不是绕过依赖。

## 10.3 RISC-V受控复现

在上游RISC-V向量MIR中建立单变量实验：

1. `precise.mir`保留第一条 `PseudoVSE32` 的store MMO；
2. `global.mir`只删除该MMO，opcode和其他指令不变；
3. 分别运行 `machine-scheduler` 并查看DAG。

修改版出现从store到后续load的蓝色虚线。它证明新增了非Data的顺序依赖。图形颜色本身不能区分 `MayAliasMem` 和 `Barrier`，精确类型应由文本DAG确认。

这个实验将DLC现象还原为通用LLVM机制：

```text
内存身份模糊
→ 后端无法证明独立
→ Scheduler DAG新增Order边
→ 合法重排空间缩小
```

## 10.4 第二阶段：已READY但启发式不选

PSV+MMO解除global barrier后，后续indexed load已经可以preload：

```text
load g0
load g1
load g2
consume g0
load g3
```

在第四组选择点：

- g3 load已经在ReadyQ，因此不是legality失败；
- VX pressure显著超过目标limit；
- scheduler优先选择能消费并释放已有VX的 `M_Permute`，而不是继续延长g3 load的live range。

这属于profitability/pressure heuristic决策。

## 10.5 反事实与性能

手工四链版本得到新鲜hooked profile：

```text
control：251404 / 251398 cycles
four-chain：248554 / 248548 cycles
correctness：max diff = 0
```

以XYS0计算：

```text
(251404 - 248554) / 251404 ≈ 1.13%
```

这证明在该具体热点和输入上，现有局部pressure heuristic选择并非最优。

但解除另一组Raw/Folded假alias后，scheduler将静态pending从8扩张到13，spill/reload从928增至984，性能反而回退约1.136%。因此：

```text
解除错误legality边
≠ 越激进调度越快
≠ 应全局降低register-pressure权重
```

真正open point是：如何使目标heuristic选择有收益的四路窗口，同时阻止无收益的13路过度调度。

## 10.6 范围勘误

早期报告将部分手写R/FoldA issue/retire结构称为“Q group”，并把MachinePipeliner纳入主因讨论。后续重新定位后，应以以下结论为准：

- 整体8组pending来自源码手写流水；
- MachineScheduler只负责其中局部preload和指令选择顺序；
- MachinePipeliner不是本次preload问题的主角；
- Raw/Folded alias真实存在，但不是前三组indexed-load preload的第一阻塞。

---

# 11. 通用排查清单

## 11.1 Middle-end

- [ ] 最小复现是否保留问题？
- [ ] 原始IR和优化后IR差异首次出现在哪个Pass？
- [ ] Pass是否实际进入pipeline？
- [ ] AA/ModRef/MemorySSA给出什么事实？
- [ ] 是事实不成立，还是LLVM无法证明？
- [ ] 是否缺少 `noalias`、memory effects、range、fast-math等契约？
- [ ] 合法后是否被cost model拒绝？
- [ ] 后续Pass是否改写了结果？

## 11.2 Backend

- [ ] ISel选择了哪个opcode和sched class？
- [ ] MIR中MMO是否存在且身份精确？
- [ ] SDep是Data、Anti、Output还是Order？
- [ ] 节点尚未READY，还是READY但没被选？
- [ ] latency、resource和hazard来自哪个模型？
- [ ] pre-RA pressure预测是否与post-RA spill一致？
- [ ] 最终汇编是否保留预期并行链？
- [ ] simulator cycle、编译器latency和硬件cycle是否被混用？

## 11.3 因果验证

- [ ] 是否只改变一个变量？
- [ ] 是否同时保存before/after IR、MIR、DAG和ASM？
- [ ] 是否运行 verifier和correctness测试？
- [ ] 是否证明运行时加载了新对象？
- [ ] 是否报告负结果、spill和资源转移？
- [ ] 是否把“允许优化”和“实际提速”分开？

---

# 12. 一句话术语表

| 术语 | 一句话解释 |
|---|---|
| CFG | 以BasicBlock为节点、控制转移为边的图。 |
| SSA | 每个名字只定义一次，通过Phi表达控制流合流。 |
| Use-Def | 从一次定义追踪到所有使用，或从使用追踪回定义。 |
| Phi | 在BasicBlock入口根据前驱选择SSA值。 |
| Dominance | A在所有到达B的路径上都出现，则A支配B。 |
| GEP | 按类型和索引计算地址，不访问内存。 |
| AA | 判断两个内存位置的潜在重叠关系。 |
| ModRef | 判断调用/指令可能修改或读取哪些内存。 |
| MemorySSA | 用MemoryDef/Use/Phi描述内存状态的到达关系。 |
| LoopInfo | 识别natural loop及其header、latch和exit。 |
| ScalarEvolution | 符号化循环递推、范围和trip count。 |
| Legality | 变换是否保持所有有效程序的语义。 |
| Profitability | 合法变换是否值得执行。 |
| Loop Versioning | 运行时检查后选择优化版或保守版循环。 |
| Lowering | 将高层/目标无关操作逐步变为目标可实现操作。 |
| SelectionDAG | 一种用于合法化、指令选择和早期调度的DAG表示。 |
| MIR | 带目标opcode、寄存器类、MMO和调度信息的机器级IR。 |
| Virtual Register | 已受目标register class约束、尚未映射物理寄存器的临时寄存器。 |
| Live Range | 一个值从定义到最后使用期间必须保持可用的范围。 |
| RA | 将虚拟寄存器分配到有限物理寄存器。 |
| Spill/Reload | 物理寄存器不足时把值保存到栈并重新加载。 |
| MMO | 描述MachineInstr具体内存访问对象、范围和属性。 |
| SUnit | MachineScheduler DAG中的可调度节点。 |
| SDep | SUnit之间的Data/Anti/Output/Order依赖。 |
| READY | 所需前驱已满足，节点进入调度候选集合。 |
| Latency | producer结果到consumer可用之间的依赖距离。 |
| Throughput | 同类指令持续发射的速率。 |
| Hazard | 由资源、流水线或目标特定规则造成的发射限制。 |
| TableGen | LLVM用于声明指令、寄存器和调度模型并生成C++表的DSL。 |
| Pseudo Instruction | 编译器内部机器指令，后续展开为真实指令或特殊编码。 |
| PSV | 用于表达不对应普通IR指针的目标特定伪内存对象。 |
| `noalias` | 对相关访问路径的非别名契约，不是全局地址唯一性。 |
| `readonly` | 不通过该参数路径写，不保证底层对象不会被其他路径修改。 |
| `noundef` | 值不能包含未定义位。 |
| `reassoc` | 授权浮点重新结合，可能改变严格IEEE结果。 |
| `undef` | 每次使用都可选择任意合法位模式的未指定值。 |
| Poison | 由某些非法运算或违反IR契约产生、会沿数据流传播的特殊语义。 |
| `freeze` | 将 `undef`/Poison固定为一次选择的普通值，阻止继续不稳定传播。 |
| Strict Aliasing | 源语言限制哪些类型的lvalue可以合法访问同一对象，前端常用TBAA编码。 |

---

# 13. 可迁移的方法与未解问题

## 13.1 已形成的能力

```text
提出真实性能问题
→ 定位到IR或MIR
→ 查询Analysis事实
→ 区分legality/profitability
→ 找到实际消费方和依赖边
→ 修改契约、Pass或Backend模型
→ 验证IR/MIR/ASM
→ correctness
→ simulator/hardware performance
```

这比“记住某个Pass名字”更可迁移：同一方法可以用于LICM、Vectorizer、Unroll、preload、MachineScheduler和目标特定hidden state。

## 13.2 结课五问

面对新问题，应能独立回答：

1. 行为最早出现在哪个编译阶段？
2. LLVM掌握了哪些事实，还缺少哪些事实？
3. 变换是否合法，为什么值得或不值得？
4. 最终机器码为何这样生成，性能瓶颈在哪里？
5. 应修改源码、IR、Analysis、Pass、Backend，还是交给Runtime/Hardware？

## 13.3 保留的Open Points

- 精确复算DLC scheduler候选评分，找到四路收益与13路过度调度之间的控制量；
- 评估是否需要目标特定preload/window策略，而非全局调整通用pressure权重；
- 将RISC-V MMO单变量实验固化为最小MIR regression test；
- Runtime/JIT、MLIR和更高级自动调优在后续独立阶段展开，不阻塞本课程结课。

---

## 最终速记

```text
IR说明“程序和证明”
Analysis说明“编译器知道什么”
Pass说明“如何改变程序”
MIR说明“目标机器上变成什么”
Scheduler DAG说明“为什么能或不能重排”
RA说明“自由度最终付出了多少寄存器代价”
Assembly说明“最终生成了什么”
Simulator/Hardware说明“它实际上是否更快”
```

---

# 14. 参考与复现材料

| 材料 | 内容 | 状态与原因 |
| --- | --- | --- |
| 案例的精确工具链与目标版本 | — | 待补：原稿没有完整 commit、构建选项及目标配置，LLVM 22 风格命令不能替代案例版本记录 |
| RISC-V 单变量实验附件 | — | 待补：已描述 precise/global MIR 的差别，但尚未随文提供文件及文本 DAG 输出 |
| 性能实验与正确性记录附件 | — | 待补：现稿列出了汇总值，未附完整输入、执行命令和原始 profile；需整理可提供的材料 |

第 8 节保留通用诊断命令，尚不能替代上述案例的完整复现流程。以下官方资料用于核对概念与接口：


- [LLVM Language Reference](https://llvm.org/docs/LangRef.html)
- [LLVM Alias Analysis](https://llvm.org/docs/AliasAnalysis.html)
- [LLVM Loop Terminology](https://llvm.org/docs/LoopTerminology.html)
- [LLVM New Pass Manager](https://llvm.org/docs/NewPassManager.html)
- [Writing an LLVM New PM Pass](https://llvm.org/docs/WritingAnLLVMNewPMPass.html)
- [LLVM Code Generator](https://llvm.org/docs/CodeGenerator.html)
- [MIR Language Reference](https://llvm.org/docs/MIRLangRef.html)
