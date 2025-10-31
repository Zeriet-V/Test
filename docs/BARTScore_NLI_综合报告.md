BARTScore 与 NLI 幻觉检测综合报告

## 范围与数据

- 覆盖两条路线：BARTScore 与 NLI（DeBERTa NLI、检索增强 NLI）。
- 数据集规模（示例）：17,790（有幻觉 7,664 / 43.08%，无幻觉 10,126 / 56.92%）。
- 评估指标：Precision、Recall、F1，混淆矩阵，任务（Summary/QA/Data2txt）与幻觉类型（Conflict/Baseless，Evident/Subtle）拆解。

## 模型与原理

### BARTScore

- 基础模型：`facebook/bart-large-cnn`。
- 思路：用条件对数似然衡量生成文本与原文一致性。
  - 记 `score = -loss(P(generated | source))`，分数越低 → 一致性越差 → 越可能有幻觉。
- 判定：score < threshold → 幻觉。
- 两个实现版本：
  - 原版：单向评分（source→target），全局统一阈值。
  - 改进版：任务特定阈值（Summary/QA/Data2txt 不同）+ 双向评分（forward/backward）+ 置信度（两向分数一致性）。

### NLI（自然语言推理）

- 基础模型：`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`。
- 思路：把原文视为前提（premise），生成文本视为假设（hypothesis），判断蕴含/中立/矛盾。
- 判定：
  - entailment_score < t_entail 或 contradiction_score > t_contra → 幻觉（组合逻辑）。
  - 句子级：对生成文本分句逐句做 NLI，整体用“最差句”聚合（min entailment 或 max contradiction）。
- 长文本：对 premise 做 token 级 chunking 滑窗聚合，避免截断丢证据。
- 可选：检索增强（RAG for NLI），先用 SentenceTransformer 选取相关证据再做 NLI。

## 方法尝试与关键改进

### BARTScore

- 原版（统一阈值）
  - 优点：简单鲁棒、召回高；在 Data2txt 表现优。
  - 阈值：统一阈值 -1.8649；验证集优化后推荐 -1.82 附近（F1 最优点）。
- 改进版（任务特定阈值 + 双向 + 置信度）
  - 经验设阈（mean+offset）在验证中导致整体 F1 下滑、召回明显下降，Data2txt 召回大幅下滑。
  - 结论：任务特定阈值需基于验证集独立优化，双向评分未带来稳定收益。
- 阈值优化（验证集网格搜索，步长 0.01）
  - 最优统一阈值约 -1.82，F1≈65.87%，与原全局阈值接近。

### NLI（DeBERTa）

- 分句改进：从脆弱正则到 `nltk/spacy` 或改进正则，避免缩写/小数点误切。
- 聚合改进：从“平均分”改为“最差句”分数（有一句错即整体风险），显著提升区分度与阈值可用性。
- 判定逻辑：加入组合判定（高矛盾或低蕴含且中立高/轻度矛盾）。
- 长上下文：前提 chunking + 聚合（取最大支持/最大反对，同时保留最差句）。
- 检索增强：先检索最相关证据后再做 NLI，缓解长文本截断与噪声干扰。

## 完整实验流程（标准三步法）

### 流程概述

**第一步：验证集测试** → **第二步：阈值优化** → **第三步：测试集评估**

这是标准的机器学习流程，确保阈值不在测试集上泄漏，保证评估的公正性。

---

### 第一步：在验证集上运行检测器

使用**默认阈值**在验证集上运行，获得初步结果和分数分布。

#### BARTScore（验证集）

```bash
cd /home/xgq/Test/detectors/bartscore_methods

# 原版（统一阈值，默认 -1.8649）
python bartscore_detector.py \
  --gpu 0 \
  --input ../../data/validation_set.jsonl \
  --output bartscore_validation_results.jsonl

# 改进版（任务特定阈值）
python bartscore_detector_improved.py \
  --gpu 0 \
  --input ../../data/validation_set.jsonl \
  --output bartscore_validation_results_improved.jsonl
```

**输出**：
- `bartscore_validation_results.jsonl`：每个样本的 BARTScore、是否检测为幻觉
- `bartscore_validation_results_report.txt`：统计报告（TP/FP/FN/TN、任务/标签分布）

#### NLI（验证集）

```bash
cd /home/xgq/Test/detectors

# DeBERTa-NLI（句子级 + 最差句聚合，默认阈值 entailment<0.5）
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_validation_results.jsonl \
  --use-entailment \
  --sentence-level

# 文档级（contradiction>0.5）
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_val_contradiction.jsonl \
  --threshold-contra 0.5
```

**输出**：
- `nli_validation_results.jsonl`：每个样本的 entailment/contradiction 分数、最差句、是否检测为幻觉
- `nli_validation_results_report.txt`：统计报告

---

### 第二步：基于验证集优化阈值

使用验证集结果，通过网格搜索找到**最优阈值**（最大化 F1 或其他目标指标）。

#### BARTScore 阈值优化

```bash
cd /home/xgq/Test/detectors/bartscore_methods

# 统一阈值优化
python bartscore_threshold_optimizer.py \
  --results bartscore_validation_results.jsonl \
  --output bartscore_threshold_opt_report.txt
```

**输出**：
- `bartscore_threshold_opt_report.txt`：最优阈值、F1 曲线、Top 10 阈值排名
- 示例结果：**最优阈值 -1.82，F1 65.87%**

#### NLI 阈值优化

```bash
cd /home/xgq/Test/detectors/nli_methods

# Entailment 阈值优化
python nli_threshold_optimizer.py \
  --results nli_validation_results.jsonl \
  --use-entailment \
  --output nli_threshold_opt_report.txt

# Contradiction 阈值优化
python nli_threshold_optimizer.py \
  --results nli_val_contradiction.jsonl \
  --output nli_threshold_opt_contradiction_report.txt
```

**输出**：
- `nli_threshold_opt_report.txt`：最优阈值、F1 曲线、Precision-Recall 权衡
- 示例结果：**最优阈值 0.54（entailment），F1 60.73%**

---

### 第三步：用最优阈值在测试集上评估

使用从验证集获得的**最优阈值**，在测试集上进行最终评估。

#### BARTScore（测试集，使用最优阈值）

```bash
cd /home/xgq/Test/detectors/bartscore_methods

# 使用优化后的统一阈值 -1.82
python bartscore_detector.py \
  --gpu 0 \
  --input ../../data/test_set.jsonl \
  --output bartscore_test_final_results.jsonl \
  --threshold -1.82
```

**输出**：
- `bartscore_test_final_results.jsonl`：测试集检测结果
- `bartscore_test_final_results_report.txt`：最终性能报告（这是**最终可报告的结果**）

#### NLI（测试集，使用最优阈值）

```bash
cd /home/xgq/Test/detectors

# 使用优化后的 entailment 阈值（如 0.54）
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/test_set.jsonl \
  --output nli_test_final_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold-entail 0.54
```

**输出**：
- `nli_test_final_results.jsonl`：测试集检测结果
- `nli_test_final_results_report.txt`：最终性能报告

---

### 流程总结表

| 步骤 | 数据集 | 目的 | 输出 |
|------|--------|------|------|
| 1. 验证集测试 | `validation_set.jsonl` | 用默认阈值获取分数分布 | `*_validation_results.jsonl` + report |
| 2. 阈值优化 | 验证集结果 | 网格搜索最优阈值（最大化 F1） | `*_threshold_opt_report.txt`（含最优阈值） |
| 3. 测试集评估 | `test_set.jsonl` | 用最优阈值评估最终性能 | `*_test_final_results.jsonl` + report |

---

### 关键原则

1. **阈值不能在测试集上调**：必须在验证集上优化，测试集仅用于最终评估。
2. **网格搜索策略**：步长 0.01，搜索范围根据验证集分数分布自动确定。
3. **目标函数**：通常最大化 F1；也可根据业务需求调整（如优先召回或准确率）。
4. **复现性**：记录每步的阈值和结果文件，确保实验可复现。

## 结果汇总（关键指标）

### BARTScore（数据集：17,790 样本）

- 原版（统一阈值 -1.8649；参考最优 -1.82）
  - 整体：Precision 53.94%，Recall 85.87%，F1 66.26。
  - 任务：Summary F1 45.25；QA F1 53.54；Data2txt F1 81.19。
  - 结论：召回高、对明显幻觉与 Data2txt 表现好，但误报偏高。
- 改进版（任务阈值 + 双向 + 置信度）
  - 整体：Precision 49.98%，Recall 64.29%，F1 56.24（下降）。
  - 优点：假阳性减少、Summary F1 小幅提升。
  - 缺点：整体与 Data2txt 召回显著下降（从≈99%至≈58%）。
- 阈值优化（验证集）
  - 统一阈值最优约 -1.82，F1≈65.87%，与原阈值接近。

### NLI（多配置对比）

- 句子级（entailment<0.5 判负，最差句聚合）
  - 整体：Precision 43.59%，Recall 99.93%，F1 60.71。
- 文档级（contradiction>0.5 判负）
  - 整体：Precision 43.04%，Recall 92.23%，F1 58.69。
- 额外对比（一次报告）
  - spacy 分句：F1≈60.75；正则分句：F1≈60.73；文档级：F1≈60.35（以当次阈值配置为准）。
- 备注：早期关闭句子级/不当阈值设定在 QA 子集上出现 F1≈8 的极差结果，已被后续改进替代。

## 适用场景与建议

- 若重召回/初筛：BARTScore 原版（统一阈值 -1.82 附近），可接受较高假阳性并配合复核。
- 若重解释与定位：NLI 句子级 + 最差句聚合，输出问题句与矛盾/蕴含分数，便于追溯。
- 长文本/证据稀疏：优先“检索增强 NLI”。
- 组合策略（推荐）：
  - 阶段一：BARTScore 原版高召回筛查。
  - 阶段二：NLI 句子级复核与定位，控制误报并提升可解释性。

## 其他可选实验

### 检索增强 NLI（快速测试）

适用于长文本场景，先检索相关证据再做 NLI。

```bash
cd /home/xgq/Test/detectors/nli_methods

# 在验证集子集上快速测试（500 样本）
python nli_with_retrieval.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_test.jsonl \
  --sample-size 500 \
  --top-k 3

# 完整验证集
python nli_with_retrieval.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_validation_results.jsonl \
  --top-k 3
```

### 不同 Top-K 值对比测试

```bash
cd /home/xgq/Test/detectors/nli_methods

python nli_with_retrieval.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --sample-size 500 \
  --full-test  # 自动测试 k=1,2,3,5,10
```

## 阈值优化要点

- BARTScore：统一阈值使用验证集网格搜索（步长 0.01）最大化 F1，推荐 -1.82 附近；若做任务特定阈值，需对每个任务在验证集独立优化，避免经验偏置。
- NLI：结合业务权衡误报/漏报，调节 `t_entail` 与 `t_contra`；在句子级使用“最差句”聚合能提升区分度与可追溯性。

## 结论

- 当前最佳单一方案：BARTScore 原版统一阈值（F1≈66，召回≈86%），适合高召回初筛；
- 可解释复核/定位：NLI 句子级 + 最差句聚合（F1≈58–61），并支持检索增强；
- 实践推荐：先 BARTScore 初筛，再 NLI 复核，基于验证集联合调参以平衡 Precision/Recall 与审核成本。


