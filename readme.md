# RankDSL

**[Title]** RankDSL: Verifiable Constrained Re-ranking by Compiling Preferences into Executable Ranking Programs





## Pipline

把 ICL rerank 从“让 LLM 直接排列表”改成“让 LLM 生成可解析的 Ranking DSL 程序 + 证明”，再由确定性求解器执行。

核心爆点：**稳定、可复现、可审计地满足硬约束**，直接解决 ICL “玄学+不可控”。

把 ICL 重排从“玄学文本输出”变成“可执行 DSL 程序 + 可验证约束”，让稳定性、可复现性和可审计性一起上台阶；同时保留 ICL 的动态规则注入优势。





[Problem Statement]

LLM 对Listwisec重排最常见做法是把候选 item 文本拼进 prompt，让 LLM 输出排序或分数；但示例敏感、约束不稳定、输出不可验证。

我们定义输入：

- 给定用户上下文 $u$、
- 召回结果候选集 $C=\{i_1..i_n\}$ 
- 基础打分 $s(u,i)$（来自召回模型）
- 对于公平/多样性/安全/曝光配额等约束集合 $\mathcal{G}$（人为要求）

输出 top-K 排序 $\pi =\{i_1..i_n\}$。

目标:

- 最大化**相关性**同时满足 $\mathcal{G}$，

- 输出过程**可检查、可复现**（同输入同随机种子得到近似一致结果）。



要点：

- 将 LLM 输出限制为可执行 DSL（而非直接排序）+ verifier，可显著降低不同示例/顺序导致的方差，并提高硬约束满足率。
- 在约束组合发生变化（训练/构造时未见过的新配额组合）时，RankDSL 的泛化优于 finetune reranker/LTR，因为约束在推理时可注入并被求解器严格执行。

**把 LLM 变成“编译器”而不是“打分器”**。ICL 的不可替代性在于：新规则/新偏好是**运行时文本输入**，无需参数更新即可编译成程序；而传统 LTR/finetune 需要重新训练或很难保证硬约束。Verifier + 求解器把“语言不确定性”隔离在程序生成阶段，输出可审计。

LLM-as-ranker、constrained ranking、multi-objective re-ranking、program-of-thought/structured prompting、verifier。差异：不是让 LLM 学会排序，而是让 LLM **编译出可验证的排序程序**，把可控性从“软提示”提升到“硬执行”。贡献闭环：方法（DSL+verifier+求解）+ 鲁棒性/可复现评估 + 可审计输出。

关键不是“让 LLM 更会排序”，而是**改变接口**：让 LLM 生成“排序程序/权重/规则”，由确定性执行器产出排序并验证约束。
ICL 的不可替代性在于：每次请求可注入不同规则/偏好/合规条款（动态约束），无需大规模 finetune 就能迁移到新业务规则；传统 LTR 需要重新训练/特征工程才能跟上规则变化。



DSL 的构建过程：

$(u, C_{small}, \mathcal{G})\rightarrow \text{DSL}$

DSL 由我们定义（JSON/EBNF），包含：

- 目标项（例如融合分数）
- 分组定义（genre/provider/敏感类目）、
- 约束（最少/最多、覆盖、去重、安全过滤）、
- tie-break 规则。并通过规则组合**合成**大量约束配置（可开源）。



这个步骤存疑：

> **选择策略**：按“约束类型签名”（如：配额约束+多样性）检索最匹配 exemplars；再做覆盖：不同约束模式至少各 1 个，避免单一示例诱导。
>
> **Prompt 结构**：few-shot “编译示例”：输入=自然语言偏好/约束 + 候选摘要（结构化字段），输出=DSL；并要求 LLM 输出**仅 DSL**。



**LLM 输出**：生成 DSL + 可选“约束证明草稿”（例如每条约束对应的计数/理由），但最终以 DSL 执行为准。

**Verifier/修复**：静态检查（语法、字段、约束可满足性）+ 动态检查（在 (C) 上是否可行）。失败则把错误信息回灌给 LLM 做 1–2 轮修复。

**融合与校准**：DSL 中允许调用 (s(u,i)) 与简单可解释特征（新颖度、重复度）；并做温度标定将 LLM 建议的权重映射到稳定范围。

**成本控制**：LLM 仅生成一次 DSL（与 (n) 弱相关），求解用贪心/ILP；对频繁出现的约束签名缓存 DSL 模板。



伪代码：

```python
Input: 
    u, 
    candidate set C (size n), 
    base scores s(u,i), 
    constraints text T 
    
# constraint-signature + coverage
S_ex = SelectExemplars(T, u, C)

# output DSL only
dsl0 = LLM_Compile(prompt(S_ex, u, C, T))  

for t in {1..R}:
    ok, err = VerifyDSL(dsl0, schema, C)
    if ok: break
    dsl0 = LLM_Repair(prompt_with_error(S_ex, u, C, T, err))
	# deterministic greedy/ILP
    pi = SolveRankingProgram(dsl0, C, s)         
return top-K pi
```



**DSL 的构建方式1**

用公开数据构造“上下文→偏好→排序规则”的样例：从历史交互生成弱标签（点击/评分/停留），并把 item 属性/文本转为规则可引用字段；再合成少量“硬约束”样例（例如需要覆盖不同类别/避免敏感内容）。先用检索（embedding）选相似上下文样例，再做覆盖重排（覆盖 item 属性、约束类型多样）。

系统提示规定 RankDSL 语法（如 `score = a*rel + b*novelty - c*toxicity`，`constraints: at_least(k, category)`），给 3–6 个演示：输入字段、输出 DSL、执行后的 top-K。

允许输出 RankDSL JSON；禁止自然语言解释（解释另开字段可选）。

将 DSL 产生的 score 与 base reranker 分数做线性/对数融合，并用小规模校准集学习融合权重（只需几十到几百条）。



DSL 构建方式2

RankDSL 的最优格式应该是：基于 JSON 的声明式约束配置单。LM 只负责填写“规则参数”，完全不接触候选列表 $C$ 的实际数据，也不进行任何循环操作。不涉及Few shot。
比如输出格式

> {
> "meta": {
>  "user_intent_summary": "用户想要找适合家庭观看的喜剧，不要恐怖片", // 强制 CoT：先总结，防跑偏
>  "focus_metrics": ["relevance", "safety"]
> },
> "definitions": [
>  // 步骤 1: 定义集合 (利用 LLM 的语义理解能力)
>  // 类似于 SQL 的 WHERE，但更灵活
>  {
>    "group_id": "high_quality_comedy",
>    "filter_expression": "genre == 'Comedy' and rating >= 4.0"
>  },
>  {
>    "group_id": "horror_content",
>    "filter_expression": "genre == 'Horror' or tags contains 'scary'"
>  },
>  {
>    "group_id": "promoted_items",
>    "filter_expression": "is_promoted == True"
>  }
> ],
> "constraints": {
>  // 步骤 2: 硬约束 (Hard Constraints) -> 传给 ILP/CSP 求解器
>  "filters": [
>    { "action": "exclude", "target_group": "horror_content" } // 直接剔除
>  ],
>  "quotas": [
>    // 核心亮点：滑动窗口或全局配额
>    { "target_group": "promoted_items", "min_count": 2, "max_count": 4 }, 
>    { "target_group": "high_quality_comedy", "min_ratio": 0.5 }
>  ],
>  "diversity": [
>    // 避免连续出现同一作者/类别
>    { "attribute": "author_id", "window_size": 3, "max_repetition": 1 } 
>  ]
> },
> "adjustments": {
>  // 步骤 3: 软调整 (Soft Adjustments) -> 调整 s(u,i) 基础分
>  "boost_rules": [
>    { "target_group": "high_quality_comedy", "factor": 1.2 }
>  ]
> }
> }





[Evaluation Plan]

数据集：

MovieLens-1M/20M（可用 genre 做多样性/配额；可做用户侧属性公平）

Amazon Reviews 子集（Books/Electronics；类目/品牌做 group）

MIND（新闻标题/类目做安全与多样性）



Baseline：

传统：LambdaMART / Transformer reranker；经典 constrained ranking（贪心/ILP/流式）与多目标重排（线性加权、MMR 等）

LLM：prompt-only listwise；ICL+random exemplars；ICL+retrieval；“LLM 给每个 item 打分”方案



指标：

NDCG@K/HR@K + 约束满足率（硬约束 0/1）

多样性(ILD/coverage) + 公平曝光差异

成本（tokens、LLM 调用数、延迟 proxy）

稳定性（不同 seed/示例顺序的方差）





## Reference

[Reference 1]

Anka: A Domain-Specific Language for Reliable LLM Code Generation 

URL: https://arxiv.org/pdf/2512.23214v1

> 核心发现: 专门为 LLM 设计的 DSL 在特定任务上比通用语言（如 Python）错误率更低。
>
> - Canonical Form（唯一规范形式）: 每个操作只有一种写法，减少 LLM 的决策空间（例如：不要同时允许 `df.query` 和 `df.loc`，只允许 `FILTER ...`）。
> - Verbose Keywords（冗长关键字）: 使用英语单词（如 `FILTER`, `AGGREGATE`）而非符号，利用 LLM 的语言能力。
> - Explicit Naming（显式命名）: 强制中间变量命名，防止 LLM 在长推理中丢失上下文。
>
> 启示: RankDSL 不应追求“极简”，而应追求“啰嗦且唯一”。



[Reference 2]

Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs

URL：https://arxiv.org/pdf/2502.19411

> **核心发现**: 代码结构能增强 LLM 的推理能力。PoT 证明了将“计算”从“推理”中分离（LLM 生成代码，解释器执行）能显著提高数学和逻辑任务的准确性。
>
> **启示**: RankDSL 不应让 LLM 直接输出排序结果（文本），而应输出**“排序逻辑的程序”**（代码），由外部引擎执行打分和排序。



[Reference 3]

Program of Thoughts Prompting: Disentangling Computation from Reasoning for Numerical Reasoning Tasks

URL：

> **核心发现**: 代码结构能增强 LLM 的推理能力。PoT 证明了将“计算”从“推理”中分离（LLM 生成代码，解释器执行）能显著提高数学和逻辑任务的准确性。
>
> **启示**: RankDSL 不应让 LLM 直接输出排序结果（文本），而应输出**“排序逻辑的程序”**（代码），由外部引擎执行打分和排序。



[Reference 4]

Ranking LLM-Generated Loop Invariants for Program Verification

> **核心发现**: LLM 生成形式化规范（Specification）的能力可以通过“生成-验证-重排”循环来提升。
>
> **启示**: RankDSL 必须设计配套的 **Verifier（验证器）**。如果 LLM 生成的 DSL 违反了硬约束（如语法错误或类型错误），Verifier 应报错并反馈给 LLM 进行 Self-Repair。









