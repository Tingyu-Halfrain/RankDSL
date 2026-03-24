# RankDSL Operation Guide

本文档按“先进入仓库根目录 `mmm/RankDSL` 再执行命令”的口径写。下面所有路径都是相对当前仓库根目录的相对路径，不再依赖本地绝对路径。

先进入项目：

```bash
cd path/to/mmm/RankDSL
```

为了减少路径问题，仓库里现在提供了两个包装脚本：

- `./run_sasrec_recall.sh`
- `./run_rankdsl_experiment.sh`

这两个脚本都会先切到脚本所在目录，再调用对应的 Python 入口，所以在 `mmm/RankDSL` 内直接执行即可。

如果你的环境禁止直接执行工作区脚本，也可以写成：

```bash
bash run_rankdsl_experiment.sh --help
```

如果你想直接走 DeepSeek API，现在也可以用包装命令：

```bash
./run_rankdsl_experiment.sh deepseek --help
```

## 1. 当前代码结构

- `core/`
  - DSL 解析、约束校验、求解器。
- `data/`
  - 数据读取与候选导出。
  - `ml1m_reader.py` 负责用户画像、历史文本、item metadata。
  - `recbole_export.py` 负责从 `SASRec` 导出 top-k candidates。
- `experiments/`
  - 请求构造、candidate 构造、baseline、实验 runner。
- `evaluation/`
  - 指标计算与聚合。
- `llm/`
  - Claude/API 客户端和 prompt。
- `run_rankdsl_experiment.py`
  - 主实验入口。
- `export_ml1m_sasrec_candidates.py`
  - 真实 `SASRec top-k` 候选导出入口。

## 2. 最快自检：先跑 stub 版本

这一步不依赖 `recbole`，也不依赖真实 API。

先跑单测：

```bash
PYTHONPATH=.. python -m unittest tests.test_rankdsl_core
```

再跑一个小规模 smoke test：

```bash
./run_rankdsl_experiment.sh \
  --scenario-size 1 \
  --candidate-topn 12 \
  --max-eval-users 0 \
  --requests /tmp/rankdsl_requests.jsonl \
  --candidates /tmp/rankdsl_candidates.jsonl \
  --output /tmp/rankdsl_smoke.json \
  --llm-mode stub
```

成功后你会得到：

- `/tmp/rankdsl_requests.jsonl`
- `/tmp/rankdsl_candidates.jsonl`
- `/tmp/rankdsl_smoke.json`

建议先看 summary：

```bash
python - <<'PY'
import json
with open('/tmp/rankdsl_smoke.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(json.dumps(data['summary'], ensure_ascii=False, indent=2))
PY
```

## 3. 真实实验前的依赖

如果你要跑真实 `SASRec` 候选和 Claude API，需要这些依赖：

- `torch`
- `recbole`
- `numpy`
- `openai`

如果你当前环境还没有它们，先安装到你自己的 conda / venv 环境里。

`RankDSL` 现在不强依赖 `pytest`，因为测试已经改成 `unittest`。

## 4. 第一步：训练 ML-1M 的 SASRec

先确认这两个文件存在：

- `configs/sasrec_ml1m.yaml`
- `dataset/ml-1m/*`

然后训练：

```bash
./run_sasrec_recall.sh \
  --dataset ml-1m \
  --config configs/sasrec_ml1m.yaml \
  --save_name SASRec_ml1m_top20.pth
```

训练完成后，期望 checkpoint 在：

```bash
saved_ckpt/SASRec_ml1m_top20.pth
```

注意：

- 这个脚本会自动把工作目录切到仓库根目录，所以在 `mmm/RankDSL` 内直接执行即可。
- 如果你想改 GPU，用 `configs/sasrec_ml1m.yaml` 里的 `gpu_id`。

## 5. 第二步：导出真实 SASRec top-20 candidates

训练完成后，导出真实候选：

```bash
python export_ml1m_sasrec_candidates.py \
  --config configs/sasrec_ml1m.yaml \
  --checkpoint saved_ckpt/SASRec_ml1m_top20.pth \
  --dataset-dir dataset/ml-1m \
  --output outputs/ml1m_candidates_sasrec.jsonl \
  --topk 20
```

输出文件：

- `outputs/ml1m_candidates_sasrec.jsonl`
- `outputs/ml1m_suitable_requests_600.jsonl`

这个文件导出后会自动补齐：

- `title`
- `genre`
- `dominant_genre`
- `release_year`
- `base_score`

同时脚本会额外生成一个经过可行性筛选的 request 文件：

- `outputs/ml1m_suitable_requests_600.jsonl`

默认配置下，这个文件会按 6 个场景各导出 100 条，总计 600 条 request：

- `filter_horror`
- `quota_comedy`
- `quota_children`
- `quota_comedy_filter_horror`
- `diversity_dominant_genre`
- `quota_children_diversity_filter_horror`

这里的 `600` 指的是 `6 个场景 x 每个场景 100 条可行 request`，不是 `600 个互不重复的用户`。

筛选标准来自 `data/select_suitable_users.py`，默认阈值是：

- `filter_horror`: 候选池里至少有 `1` 个 `Horror`，并且至少有 `10` 个非 `Horror` 候选，确保 top-10 可排出“完全无 Horror”的结果。
- `quota_comedy`: 候选池里至少有 `3` 个 `Comedy`。
- `quota_children`: 候选池里至少有 `2` 个 `Children's`。
- `quota_comedy_filter_horror`: 同时满足 `Comedy >= 3`、`Horror >= 1`、`non-Horror >= 10`。
- `diversity_dominant_genre`: 候选池里 `distinct_dominant_genres >= 3`。
- `quota_children_diversity_filter_horror`: 同时满足 `Children's >= 2`、`Horror >= 1`、`non-Horror >= 10`、`distinct_dominant_genres >= 3`。

除了这些阈值之外，脚本还会对每条 request 的 `reference_dsl` 调用 `verify_dsl(...)` 做可行性检查；只有 verifier 通过的样本才会被写入 `outputs/ml1m_suitable_requests_600.jsonl`。

如果 checkpoint 不存在，脚本会直接报错并提示你先训练。

## 6. 第三步：使用预构造 suitable requests

现在推荐直接使用上一步自动生成的：

```bash
outputs/ml1m_suitable_requests_600.jsonl
```

这个文件已经包含：

- `user_profile`
- `user_summary`
- `history_text`
- `constraint_text`
- `target_item_id`
- `candidates`
- `candidate_stats`

并且 `run_rankdsl_experiment.py` 支持通过 `--requests-file` 直接消费这个文件，不再触发自动 request 重建。

如果你只想快速试论文设置里的一次 compile，可以额外传：

```bash
--num-paraphrases 1
```

默认仍然是 `3`。

## 7. 第四步：跑真实 Claude API 实验

先设置环境变量，或者创建 `.env.deepseek`。不要把 key 直接写进命令历史里。

推荐方式是新建仓库根目录下的 `.env.deepseek`：

```bash
cat > .env.deepseek <<'EOF'
DEEPSEEK_API_KEY=sk-xxxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EOF
```

脚本 `./run_rankdsl_experiment.sh deepseek ...` 会自动按下面顺序加载：

1. `.env.deepseek` 或 `$DEEPSEEK_ENV_FILE`
2. `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`
3. `RANKDSL_API_KEY` / `RANKDSL_BASE_URL`

如果你更喜欢手动 export，也可以：

```bash
export RANKDSL_API_KEY="cr_e8fdb7d247ccec1edfed8ade3fb489b560a5fca09c0f5e4ffb591b68ffeb3b67"
export RANKDSL_BASE_URL="https://cursor.scihub.edu.kg/api/v1"
```

```bash
export RANKDSL_API_KEY="sk-88eaede7830d4200bdc72765074cb705"
export RANKDSL_BASE_URL="https://api.deepseek.com/v1"
```
cat > .env.deepseek <<'EOF'
DEEPSEEK_API_KEY=sk-88eaede7830d4200bdc72765074cb705
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
EOF

如果你已经在 `AIresearcher/testAPI.py` 里验证过接口，可以直接复用那套配置；这里只是建议改成环境变量，不要把 key 再复制到新代码里。

然后跑实验：

```bash
./run_rankdsl_experiment.sh deepseek \
  --dataset-dir dataset/ml-1m \
  --requests-file outputs/ml1m_suitable_requests_600.jsonl \
  --candidates outputs/ml1m_candidates_sasrec.jsonl \
  --output outputs/experiment_results_api.json \
  --llm-log-path outputs/llm_interactions.jsonl \
  --candidate-topn 20 \
  --max-eval-users 0 \
  --num-paraphrases 1 \
  --save-results \
  --load-from-cache
```

如果你想临时覆盖默认模型，也可以显式传：

```bash
./run_rankdsl_experiment.sh deepseek --model deepseek-chat ...
```

说明：

- `--requests-file` 会直接使用预构造的 suitable request 文件，不再按 `scenario-size` 自动生成样本。
- `--num-paraphrases` 默认是 `3`；如果你要提速做 ablation，可改成 `1`。
- `--save-results` 会把每个 request 的完整结果保存到 `results/saved_rankings/{request_id}.json`，后续改指标不需要重新调用 LLM。
- `--load-from-cache` 会优先读取 `results/saved_rankings/` 里的文件，命中时不再跑 LLM。
- 默认会评估全部符合条件的用户；`--max-eval-users 0` 表示不设上限。
- `deepseek` 子命令会自动补上 `--llm-mode api`，并默认使用 `deepseek-chat`。
- `--llm-log-path` 会把每次 LLM 的输入 messages 和原始输出追加写到 JSONL，方便排查 JSON 解析失败。
- `--llm-parse-log-path` 会把 JSON 提取与解析细节单独写到 JSONL，包括 `raw_preview`、候选 JSON 起始位置、提取出的 JSON 片段预览、parse error、verifier error。
- compile 失败时，`RankDSL` 现在会自动 fallback 到 `base_recall` ranking，而不是直接记成空 ranking 和硬 0。

例如：

```bash
rg 'diversity_dominant_genre-005' outputs/llm_parse_debug.jsonl
```

## 8. 第五步：看结果

结果文件：

- `outputs/experiment_results_api.json`

建议先只看 summary：

```bash
python - <<'PY'
import json
with open('outputs/experiment_results_api.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(json.dumps(data['summary'], ensure_ascii=False, indent=2))
PY
```

重点看这些字段：

- `base_recall`
- `prompt_only_direct_rerank`
- `rankdsl_greedy`
- `rankdsl_ilp`
- `compile_success_rate`
- `canonical_program_agreement`
- `rankdsl_constraint_satisfaction_variance`
- `prompt_only_constraint_satisfaction_variance`

## 9. V2 指标说明

除了原来的整体 `constraint_satisfaction` 和 `genre_coverage`，现在还支持更细粒度的离线指标：

- `filter_satisfaction`
- `quota_satisfaction`
- `diversity_satisfaction`
- `sliding_window_ok_rate`
- `ild_score_avg`
- `ndcg_only_successful`

单条 ranking 里还会额外记录：

- `filter_ok`
- `quota_status`
- `sliding_window_ok`
- `max_rep_in_any_window`
- `sliding_violation_rate`
- `ild_score`

这些指标会直接写进缓存结果文件，方便后处理。

## 10. 离线分析流程

第一次真实跑实验时建议把缓存一起打开：

```bash
./run_rankdsl_experiment.sh \
  --dataset-dir dataset/ml-1m \
  --requests-file outputs/ml1m_suitable_requests_600.jsonl \
  --candidates outputs/ml1m_candidates_sasrec.jsonl \
  --output outputs/experiment_results_api.json \
  --llm-mode api \
  --save-results
```

之后如果你只是改指标，不想重新调用 LLM：

```bash
./run_rankdsl_experiment.sh \
  --dataset-dir dataset/ml-1m \
  --requests-file outputs/ml1m_suitable_requests_600.jsonl \
  --candidates outputs/ml1m_candidates_sasrec.jsonl \
  --output outputs/experiment_results_api.json \
  --llm-mode api \
  --load-from-cache
```

批量离线重算细粒度 summary：

```bash
python analysis/compute_metrics_offline.py \
  --input-dir results/saved_rankings \
  --output summary_detailed.json
```

可视化分析入口：

- `analysis/inspect_results.ipynb`

## 8A. Amazon-Books：训练、在线语义 enrichment、候选导出、实验

`Amazon-Books` 和 `ML-1M` 不同的点在于：

- 没有可靠的用户画像字段，所以 `user_profile` 会更弱。
- item 语义更依赖 `title + categories + 互联网查询到的书籍描述`。
- 当前代码支持一个可选的在线 enrichment 步骤，优先查 `Open Library`，失败再查 `Google Books`。

### 8A.1 训练 Amazon-Books SASRec

```bash
./run_sasrec_recall.sh \
  --dataset amazon-books \
  --config configs/sasrec_amazonbooks.yaml \
  --save_name SASRec_amazonbooks_top50.pth
```

checkpoint 期望路径：

```bash
saved_ckpt/SASRec_amazonbooks_top50.pth
```

### 8A.2 在线抓取书籍语义

这一步会访问互联网，给一批 `amazon-books.item` 里的书补 `description / authors / categories / published_date`。

先抓一小批调试：

```bash
python enrich_amazon_books_semantics.py \
  --item-path dataset/amazon-books/amazon-books.item \
  --output outputs/amazon_books_semantics.jsonl \
  --offset 0 \
  --limit 500 \
  --sleep-seconds 0.2
```

如果接口稳定，再分批扩展。

说明：

- 这一步默认用 `Open Library Books API`。
- 如果 `Open Library` 没命中，会回退到 `Google Books Volumes API`。
- 因为 `amazon-books` 很大，不建议一次抓完整个库；推荐分批缓存。

### 8A.3 导出 Amazon-Books SASRec top-20 candidates

```bash
python export_amazonbooks_sasrec_candidates.py \
  --config configs/sasrec_amazonbooks.yaml \
  --checkpoint saved_ckpt/SASRec_amazonbooks_top50.pth \
  --dataset-dir dataset/amazon-books \
  --semantic-cache outputs/amazon_books_semantics.jsonl \
  --output outputs/amazon_books_candidates_sasrec.jsonl \
  --topk 20
```

输出文件：

- `outputs/amazon_books_candidates_sasrec.jsonl`

### 8A.4 跑 Amazon-Books 实验

```bash
./run_rankdsl_experiment.sh \
  --dataset-dir dataset/amazon-books \
  --semantic-cache outputs/amazon_books_semantics.jsonl \
  --requests outputs/amazon_books_requests.jsonl \
  --candidates outputs/amazon_books_candidates_sasrec.jsonl \
  --output outputs/amazon_books_experiment_results.json \
  --scenario-size 50 \
  --candidate-topn 20 \
  --max-eval-users 100 \
  --llm-mode api \
  --model claude-opus-4-6
```

当前 `Amazon-Books` 场景默认是：

- `filter_expensive`
- `quota_mystery`
- `quota_scifi`
- `quota_mystery_filter_expensive`
- `diversity_dominant_category`
- `quota_scifi_diversity_filter_expensive`

## 9. 推荐操作顺序

建议严格按这个顺序来：

1. `unittest` 通过。
2. 跑 `stub smoke`。
3. 训练 `ML-1M SASRec`。
4. 导出 `SASRec candidates`。
5. 跑一小批真实 API 实验：

```bash
./run_rankdsl_experiment.sh \
  --dataset-dir dataset/ml-1m \
  --requests outputs/ml1m_requests_small.jsonl \
  --candidates outputs/ml1m_candidates_sasrec.jsonl \
  --output outputs/experiment_results_small.json \
  --scenario-size 2 \
  --candidate-topn 20 \
  --max-eval-users 20 \
  --llm-mode api
```

6. 小批确认没问题后，再跑完整 `scenario-size 50`。

## 10. 常见问题

### 10.1 `No module named recbole`

说明环境没装 `recbole`。先安装依赖，再训练和导出候选。

### 10.2 `Checkpoint not found`

说明你还没有训练出：

```bash
saved_ckpt/SASRec_ml1m_top50.pth
```

先执行第 4 节的训练命令。

### 10.3 实验很慢

这是正常的，原因有两个：

- `RankDSL + ILP` 现在是无外部 ILP 库的 exact fallback，`top-20 -> top-10` 会慢。
- API 模式下每个 request 最多会经历 `compile + repair + direct rerank`。

如果只是调试：

- 把 `--scenario-size` 改小，比如 `1` 或 `2`
- 把 `--candidate-topn` 先降到 `12`
- 把 `--max-eval-users` 先降到 `20`

完整论文实验再恢复到：

- `scenario-size 50`
- `candidate-topn 20`

### 10.4 API 能通，但 compile success 低

先看：

- `results[*].rankdsl.runs[*].compile_error`

通常先从这三处改：

- `llm/prompts.py`
- `core/dsl_parser.py`
- `core/verifier.py`

## 11. 现在这套代码的边界

当前已经完成：

- RankDSL v1 核心闭环
- ML-1M 语义读取
- stub / api 双模式
- request 自动生成
- popularity / SASRec 两种 candidate 输入路径
- 统一评估与 summary 输出

当前还没完成：

- 真正的外部 ILP 求解器版本
- 公平曝光实验
- Amazon-Books 迁移实验
- 更复杂的 few-shot exemplar 检索

## 12. 你现在最应该执行的命令

如果你只是验证工程通路：

```bash
PYTHONPATH=.. python -m unittest tests.test_rankdsl_core
./run_rankdsl_experiment.sh --scenario-size 1 --candidate-topn 12 --max-eval-users 100 --llm-mode stub --requests /tmp/rankdsl_requests.jsonl --candidates /tmp/rankdsl_candidates.jsonl --output /tmp/rankdsl_smoke.json
```

如果你要进入真实实验：

```bash
./run_sasrec_recall.sh --dataset ml-1m --config configs/sasrec_ml1m.yaml --save_name SASRec_ml1m_top50.pth
python export_ml1m_sasrec_candidates.py --config configs/sasrec_ml1m.yaml --checkpoint saved_ckpt/SASRec_ml1m_top50.pth --dataset-dir dataset/ml-1m --output outputs/ml1m_candidates_sasrec.jsonl --topk 20
./run_rankdsl_experiment.sh --dataset-dir dataset/ml-1m --requests outputs/ml1m_requests.jsonl --candidates outputs/ml1m_candidates_sasrec.jsonl --output outputs/experiment_results_api.json --scenario-size 50 --candidate-topn 20 --max-eval-users 100 --llm-mode api --model claude-opus-4-6
```
