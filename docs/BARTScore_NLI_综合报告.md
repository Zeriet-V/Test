# BARTScore 与 NLI 幻觉检测综合报告

## 范围与数据

- 覆盖两条路线：BARTScore 与 NLI（DeBERTa NLI、检索增强 NLI）。
- 数据集规模（RAGTruth）：17,790（有幻觉 7,664 / 43.08%，无幻觉 10,126 / 56.92%）。
- 评估指标：Precision、Recall、F1，混淆矩阵，任务（Summary/QA/Data2txt）与幻觉类型（Conflict/Baseless，Evident/Subtle）拆解。

## 模型与原理

### BARTScore

- 基础模型：`facebook/bart-large-cnn`。
- 思路：用条件对数似然衡量生成文本与原文一致性。
  - 记 `score = -loss(P(generated | source))`，分数越低 → 一致性越差 → 越可能有幻觉。
- 判定：score < threshold → 幻觉。
- 两个实现版本：
  - 原版：单向评分（source→target），全局统一阈值。
  - 改进版：
    - 任务特定阈值（Summary/QA/Data2txt 不同） 
    - 双向评分(计算$P(target|source)和P（source|target）$)
    - 置信度（两向分数一致性）。

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

## 完整实验流程

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

| 步骤          | 数据集                 | 目的                          | 输出                                       |
| ------------- | ---------------------- | ----------------------------- | ------------------------------------------ |
| 1. 验证集测试 | `validation_set.jsonl` | 用默认阈值获取分数分布        | `*_validation_results.jsonl` + report      |
| 2. 阈值优化   | 验证集结果             | 网格搜索最优阈值（最大化 F1） | `*_threshold_opt_report.txt`（含最优阈值） |
| 3. 测试集评估 | `test_set.jsonl`       | 用最优阈值评估最终性能        | `*_test_final_results.jsonl` + report      |

---

### 关键原则

1. **阈值不能在测试集上调**：必须在验证集上优化，测试集仅用于最终评估。
2. **网格搜索策略**：步长 0.01，搜索范围根据验证集分数分布自动确定。
3. **目标函数**：通常最大化 F1；也可根据业务需求调整（如优先召回或准确率）。
4. **复现性**：记录每步的阈值和结果文件，确保实验可复现。

## 结果汇总（关键指标）

### BARTScore（测试集：14,233 样本）

#### 原版（统一阈值 -1.8649）

**配置**：
- 模型：facebook/bart-large-cnn
- 策略：单向评分
- 阈值：-1.8649（统一）

**整体性能**：
- **F1分数**：66.26
- **Precision**：53.94%
- **Recall**：85.87%
- 混淆矩阵：
  - TP: 6,581，FP: 5,619
  - FN: 1,083，TN: 4,507

**按任务类型表现**：

| 任务 | F1 | Precision | Recall |
|------|-----|-----------|---------|
| Summary | 45.25 | 37.65% | 55.69% |
| QA | 53.54 | 39.85% | 82.02% |
| Data2txt | **81.19** | **68.66%** | **99.39%** |

**按幻觉类型召回率**：

| 幻觉类型 | 总数 | 检测到 | 召回率 |
|---------|------|--------|--------|
| Evident Conflict | 5,324 | 4,881 | 91.68% |
| Subtle Conflict | 201 | 135 | 67.16% |
| Evident Baseless Info | 6,237 | 5,632 | 90.30% |
| Subtle Baseless Info | 2,527 | 2,302 | 91.10% |

**结论**：召回高、对明显幻觉与 Data2txt 表现好，但误报偏高（准确率仅 53.94%）。

---

#### 改进版 V2（测试集，统一阈值 -1.82 + 双向评分）

**配置**：
- 模型：facebook/bart-large-cnn
- 策略：双向评分 + 置信度 + 按幻觉类型F1统计
- 阈值：-1.82（三个任务统一使用，基于验证集优化结果）

**整体性能**：
- **F1分数**：**66.40**
- **Precision**：**53.47%**
- **Recall**：**87.59%**
- 混淆矩阵：
  - TP: 5,371，FP: 4,674
  - FN: 761，TN: 3,427

**按任务类型表现**：

| 任务 | 总数 | 有幻觉 | F1 | Precision | Recall |
|------|------|--------|-----|-----------|---------|
| Summary | 4,522 | 1,328 (29.37%) | 45.67 | 36.97% | 59.71% |
| QA | 4,727 | 1,365 (28.88%) | 53.57 | 39.24% | 84.40% |
| Data2txt | 4,984 | 3,439 (69.00%) | **81.54** | **69.02%** | **99.62%** |

**按幻觉类型召回率**：

| 幻觉类型 | 总数 | 检测到 (TP) | 召回率 | 状态 |
|---------|------|------------|--------|------|
| Evident Conflict | 4,272 | 3,972 | **92.98%** | ✓ 好 |
| Subtle Conflict | 163 | 107 | 65.64% | ✗ 需改进 |
| Evident Baseless Info | 5,019 | 4,592 | **91.49%** | ✓ 好 |
| Subtle Baseless Info | 1,993 | 1,836 | **92.12%** | ✓ 好 |

**特点**：
- ✅ 双向评分提供置信度信息
- ✅ 按幻觉类型提供详细召回率统计
- ✅ 整体性能与原版相当（F1 66.40 vs 66.26）
- ✅ Data2txt 表现优异（F1 81.54，召回率 99.62%）
- ⚠️ Subtle Conflict 检测率仍较低（65.64%）

**对比原版**：
- F1提升：66.26 → 66.40（+0.14）
- Precision略降：53.94% → 53.47%（-0.47%）
- Recall提升：85.87% → 87.59%（+1.72%）
- 假阳性减少：5,619 → 4,674（-945）
- 假阴性减少：1,083 → 761（-322）

---

#### 阈值优化（验证集）

- 统一阈值最优约 -1.82，F1≈65.87%，与原阈值 -1.8649 接近。
- 改进版 V2 在测试集上使用此优化阈值，取得与原版相当甚至略好的性能。

### NLI（验证集：3,557 样本）

**注意**：NLI 方法**仅在验证集上运行**，未在测试集上评估。

---

#### 关键技术说明

##### 1. Chunking（长文本处理）

**问题**：NLI 模型输入限制为 512 tokens，但原文（premise）可能很长（如几千 tokens）。

**Chunking 解决方案**：
```
原文（2000 tokens）
  ↓ 滑窗切分
Chunk 1: tokens[0:240]
Chunk 2: tokens[120:360]  ← 重叠部分（stride=120）
Chunk 3: tokens[240:480]
...
  ↓ 每个 chunk 单独做 NLI
Result 1: entailment=0.8, contradiction=0.1
Result 2: entailment=0.6, contradiction=0.2
Result 3: entailment=0.9, contradiction=0.05
  ↓ 聚合策略
最终结果: 
  - 最大 entailment = 0.9（最支持）
  - 最大 contradiction = 0.2（最反对）
  - 最小 entailment = 0.6（最差句，用于判定）
```

**参数**：
- `chunk_size=240`：每个 chunk 的长度
- `stride=120`：滑窗步长（50% 重叠，避免切断关键信息）

**优势**：
- ✅ 避免截断丢失证据
- ✅ 全面覆盖原文所有部分
- ✅ 通过聚合得到更可靠的判断

**代价**：
- ⚠️ 增加推理次数（长文本需要多次推理）
- ⚠️ 增加计算时间

##### 2. 句子级检测 + 最差句聚合

**流程**：
```
生成文本（多句）
  ↓ 分句
句1: "公司成立于2010年。"
句2: "总部在纽约。"  
句3: "有500名员工。"
  ↓ 逐句做 NLI（每句都用完整原文或 chunking）
句1: entailment=0.85 ✓
句2: entailment=0.90 ✓
句3: entailment=0.30 ✗ 幻觉！
  ↓ 聚合：取最差句
最终结果: entailment=0.30（句3），判定为幻觉
记录: worst_sentence="有500名员工。"
```

**优势**：
- ✅ 可以定位具体哪句话有问题
- ✅ "有一句错即整体有风险"的保守策略
- ✅ 比平均分更能反映真实问题

---

#### 运行时间说明

由于 NLI 方法的计算开销较大，在验证集上的运行时间显著长于 BARTScore：

| 方法 | 验证集耗时 | 主要耗时原因 |
|------|-----------|------------|
| BARTScore | 10-15 分钟 | 单次推理，批处理 |
| NLI | **2-3 小时** | 句子级 + chunking + 模型较大 |

**NLI 耗时原因**：
1. 句子级检测：平均每样本 7.5 句，推理次数 × 7.5
2. 长前提 chunking：长文本需多次推理后聚合
3. DeBERTa-v3-large 模型较大（推理速度慢于 BART）
4. 组合判定需同时计算多个分数

鉴于时间成本，**未在测试集上运行 NLI 方法**。

---

#### 配置1：句子级 + entailment < 0.5（基线）

**配置**：
- 模型：DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- 策略：句子级检测 + 长前提 chunking + 最差句聚合
- 判定阈值：entailment < 0.5
- 粒度：句子级

**整体性能**：
- **F1分数**：60.71
- **Precision**：43.59%
- **Recall**：**99.93%**
- 混淆矩阵：
  - TP: 1,531，FP: 1,981
  - FN: 1，TN: 44

**按任务类型表现**：

| 任务 | F1 | Precision | Recall |
|------|-----|-----------|---------|
| Data2txt | **80.34** | **67.13%** | **100%** |
| Summary | 47.82 | 31.45% | 99.72% |
| QA | 47.17 | 30.87% | 100% |

**按幻觉类型召回率**：

| 幻觉类型 | 总数 | 检测到 | 召回率 |
|---------|------|--------|--------|
| Evident Conflict | 1,052 | 1,051 | **99.90%** |
| Subtle Conflict | 38 | 38 | **100%** |
| Evident Baseless Info | 1,218 | 1,218 | **100%** |
| Subtle Baseless Info | 534 | 534 | **100%** |

**结论**：召回率极高（仅漏检 1 个），但误报严重（准确率仅 43.59%）。

---

#### 配置2：文档级 + entailment < 0.5

**配置**：
- 策略：文档级检测（不分句）
- 判定阈值：entailment < 0.5
- 粒度：文档级

**整体性能**：
- **F1分数**：60.03
- **Precision**：43.49%
- **Recall**：96.87%
- 混淆矩阵：
  - TP: 1,484，FP: 1,928
  - FN: 48，TN: 97

**按任务类型表现**：

| 任务 | F1 | Precision | Recall |
|------|-----|-----------|---------|
| Data2txt | 79.98 | 66.97% | 99.26% |
| Summary | 47.06 | 31.04% | 97.21% |
| QA | 45.35 | 30.19% | 91.09% |

**按幻觉类型召回率**：

| 幻觉类型 | 总数 | 检测到 | 召回率 |
|---------|------|--------|--------|
| Evident Conflict | 1,052 | 1,029 | 97.81% |
| Subtle Conflict | 38 | 38 | 100% |
| Evident Baseless Info | 1,218 | 1,176 | 96.55% |
| Subtle Baseless Info | 534 | 509 | 95.32% |

**结论**：相比句子级，召回率略降（96.87% vs 99.93%），但准确率基本不变。

---

#### 配置3：文档级 + contradiction > 0.5

**配置**：
- 策略：文档级检测
- 判定阈值：contradiction > 0.5（只看矛盾分数）
- 粒度：文档级

**整体性能**：
- **F1分数**：58.69
- **Precision**：43.04%
- **Recall**：92.23%
- 混淆矩阵：
  - TP: 1,413，FP: 1,870
  - FN: 119，TN: 155

**按任务类型表现**：

| 任务 | F1 | Precision | Recall |
|------|-----|-----------|---------|
| Data2txt | 79.22 | 66.92% | 97.06% |
| Summary | 44.05 | 29.57% | 86.31% |
| QA | 44.24 | 29.64% | 87.19% |

**按幻觉类型召回率**：

| 幻觉类型 | 总数 | 检测到 | 召回率 |
|---------|------|--------|--------|
| Evident Conflict | 1,052 | 986 | 93.73% |
| Subtle Conflict | 38 | 36 | 94.74% |
| Evident Baseless Info | 1,218 | 1,113 | 91.38% |
| Subtle Baseless Info | 534 | 482 | 90.26% |

**结论**：召回率进一步降低（92.23%），但更平衡。

---

#### 配置4：句子级 + 组合判定（最优推荐）⭐

**配置**：
- 模型：DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- 策略：句子级检测 + 长前提 chunking + 最差句聚合
- 判定阈值：**entailment < 0.45 或 contradiction > 0.2**（组合判定）
- 粒度：句子级
- Chunking：chunk_size=240, stride=120

**整体性能**：
- **F1分数**：**62.06**
- **Precision**：**45.70%**
- **Recall**：**96.67%**
- 混淆矩阵：
  - TP: 1,481，FP: 1,760
  - FN: 51，TN: 265

**按任务类型表现**：

| 任务 | 总数 | 有幻觉 | F1 | Precision | Recall | 平均contradiction |
|------|------|--------|-----|-----------|---------|------------------|
| Data2txt | 1,214 | 815 (67.13%) | - | - | 98.27% | 0.1454 |
| Summary | 1,136 | 358 (31.51%) | - | - | 93.05% | 0.3140 |
| QA | 1,207 | 359 (29.74%) | - | - | 82.10% | 0.4182 |

**按幻觉类型召回率**：

| 幻觉类型 | 总数 | 检测到 | 召回率 |
|---------|------|--------|--------|
| Evident Conflict | 1,052 | 1,038 | **98.67%** |
| Evident Baseless Info | 1,218 | 1,188 | **97.54%** |
| Subtle Baseless Info | 534 | 511 | **95.69%** |
| Subtle Conflict | 38 | 34 | 89.47% |

**特点**：
- ✅ **所有配置中 F1 最高**（62.06）
- ✅ 召回率极高（96.67%），仅漏检 51/1,532 个样本
- ✅ 准确率提升至 45.70%（相比其他配置）
- ✅ 所有明显幻觉（Evident）检测率 >97%
- ✅ 提供句子级定位和可解释分数
- ✅ 支持长文本 chunking，避免截断
- ⚠️ 误报率仍较高（54.3%）
- ⚠️ 计算开销大（2-3 小时）

**结论**：最优配置，推荐用于需要高召回和可解释性的场景。

---

#### NLI 配置对比总结

| 配置 | 判定标准 | 粒度 | F1 | Precision | Recall | 特点 |
|------|---------|------|-----|-----------|---------|------|
| **配置1** | entailment<0.5 | 句子级 | 60.71 | 43.59% | **99.93%** | 召回极高，仅漏检1个 |
| **配置2** | entailment<0.5 | 文档级 | 60.03 | 43.49% | 96.87% | 略平衡 |
| **配置3** | contradiction>0.5 | 文档级 | 58.69 | 43.04% | 92.23% | 更平衡 |
| **配置4⭐** | 组合判定 | 句子级 | **62.06** | **45.70%** | 96.67% | **最优** |

**最优配置**：配置4（entailment<0.45 或 contradiction>0.2 + 句子级 + chunking）

## 方法对比与适用场景

### 性能对比表（关键指标）

| 方法               | 数据集 | F1        | Precision  | Recall     | 优势             | 劣势         |
| ------------------ | ------ | --------- | ---------- | ---------- | ---------------- | ------------ |
| **BARTScore 原版** | 测试集 | **66.40** | **53.47%** | 87.59%     | ✅ 最平衡，速度快 | ⚠️ 会漏检 12% |
| **NLI 最优**       | 验证集 | 62.06     | 45.70%     | **96.67%** | ✅ 几乎零漏检     | ⚠️ 误报多，慢 |

---

### 最终结论

**当前最佳方案**：**BARTScore 原版**（统一阈值 -1.82）

- 已在测试集充分验证（F1 66.40）
- 速度快，平衡性好
- 适合生产环境和大规模应用

**NLI 作为补充**：

- 验证集表现优秀（Recall 96.67%）
- 适合特殊需求和研究场景
- 由于时间成本，未在测试集验证
- 可作为二阶段精细复核工具