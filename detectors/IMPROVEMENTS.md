# BARTScore幻觉检测器改进方案

## 📊 原版问题分析

根据 `bartscore_results_report.txt` 的分析，原版存在以下问题：

### 1. 整体性能较差
- **准确率 (Precision)**: 53.94% ❌
- **召回率 (Recall)**: 85.87% ✓
- **F1分数**: 66.26
- **假阳性**: 5619 个（过高！）
- **假阴性**: 1083 个

### 2. 不同任务类型表现差异大

| 任务类型 | 召回率 | 准确率 | F1分数 | 平均BARTScore |
|---------|--------|--------|--------|--------------|
| Summary | 55.69% | 37.65% | 45.25 | -1.8192 |
| QA | 82.02% | 39.85% | 53.54 | -2.1199 |
| Data2txt | 99.39% | 68.66% | 81.19 | -2.5006 |

**问题**：三种任务的平均分数差异达到 0.68，但使用统一阈值 -1.8649

### 3. 分数区分度不够
- 有幻觉样本: -2.3376 (±0.4391)
- 无幻觉样本: -2.0201 (±0.8030)
- 差值仅: 0.32

无幻觉样本的标准差(0.8)远大于有幻觉样本(0.44)，说明存在大量边界情况。

### 4. 假阳性占误判的 83.84%
主要问题是**误报过多**，模型过于敏感。

---

## 🛠️ 改进方案

### 方案一：任务特定阈值 ✅ (已实现)

**核心思路**：为不同任务类型使用不同的检测阈值

**阈值设定**：
```python
task_thresholds = {
    'Summary': -1.65,      # 原平均-1.82，略微放宽以减少假阳性
    'QA': -2.05,           # 原平均-2.12
    'Data2txt': -2.45      # 原平均-2.50
}
```

**预期效果**：
- ✓ 减少 Summary 任务的假阳性（原来误报最严重）
- ✓ 提高整体准确率
- ⚠ 可能略微降低召回率

**使用方法**：
```bash
cd /home/xgq/Test/detectors
python bartscore_detector_improved.py
```

---

### 方案二：双向BARTScore ✅ (已实现)

**核心思路**：同时计算两个方向的一致性评分

1. **Forward Score**: P(generated|source) - 生成文本基于原文的似然
2. **Backward Score**: P(source|generated) - 原文基于生成文本的似然
3. **置信度**: 基于两个分数的一致性计算

**公式**：
```
confidence = 1 / (1 + |forward - backward|)
```

**优势**：
- ✓ 更全面的一致性评估
- ✓ 可以识别"看似合理但不符合原文"的幻觉
- ✓ 提供置信度，支持阈值优化

**使用方法**：
```python
process_dataset_improved(
    input_file='../data/test_response_label.jsonl',
    output_file='bartscore_improved_results.jsonl',
    use_bidirectional=True  # 启用双向评分
)
```

---

### 方案三：多特征融合（可选）

**核心思路**：BARTScore + 其他特征的加权融合

**可融合特征**：

1. **长度比例**
```python
length_ratio = len(generated) / len(source)
# 过长或过短都可能有问题
length_penalty = abs(1.0 - length_ratio)
```

2. **N-gram重叠度**
```python
# 计算BLEU或ROUGE分数
from nltk.translate.bleu_score import sentence_bleu
overlap_score = sentence_bleu([source.split()], generated.split())
```

3. **实体一致性**
```python
# 使用NER检测实体是否一致
source_entities = extract_entities(source)
generated_entities = extract_entities(generated)
entity_consistency = jaccard_similarity(source_entities, generated_entities)
```

4. **语义相似度**
```python
# 使用Sentence-BERT计算语义相似度
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode([source, generated])
semantic_sim = cosine_similarity(embeddings[0], embeddings[1])
```

**融合公式**：
```python
final_score = (
    0.5 * bartscore_normalized + 
    0.2 * overlap_score + 
    0.2 * entity_consistency + 
    0.1 * (1 - length_penalty)
)
```

**预期效果**：
- ✓ 更鲁棒的检测
- ✓ 减少单一指标的偏差
- ⚠ 计算开销增加

---

### 方案四：自适应阈值

**核心思路**：根据分数分布动态调整阈值

**方法1：百分位数法**
```python
# 使用训练集确定最优百分位
threshold = np.percentile(hallucination_scores, 90)
```

**方法2：最大化F1**
```python
# 在验证集上搜索最优阈值
best_threshold = find_optimal_threshold(
    scores, labels, 
    metric='f1',
    search_range=(-3.0, -1.0),
    step=0.01
)
```

**方法3：任务+类型特定**
```python
# 为每个任务的每种幻觉类型设定阈值
thresholds = {
    ('Summary', 'Evident Conflict'): -1.5,
    ('Summary', 'Subtle Conflict'): -1.7,
    ('QA', 'Evident Conflict'): -1.9,
    ...
}
```

---

### 方案五：模型微调（长期方案）

**核心思路**：在幻觉检测任务上Fine-tune BART

**步骤**：
1. 准备训练数据：(source, generated, label)
2. 设计训练目标：
   - 方案A：作为二分类任务训练
   - 方案B：对比学习（拉近无幻觉样本，推远有幻觉样本）
3. 微调BART模型
4. 在测试集上评估

**预期效果**：
- ✓ 显著提升性能
- ✓ 更好的分数区分度
- ⚠ 需要较多标注数据
- ⚠ 训练成本高

---

## 📈 预期改进效果

### 方案一（任务特定阈值）预期：
- 准确率：53.94% → **65-70%** ✅
- 召回率：85.87% → **75-80%** (略降)
- F1分数：66.26 → **70-75** ✅
- 假阳性：5619 → **3500-4000** ✅

### 方案二（双向评分）预期：
- 准确率：+3-5%
- 召回率：+2-3%
- F1分数：+3-4
- 特别改善：Subtle类型的检测

### 方案三（多特征融合）预期：
- 准确率：+5-8%
- 召回率：+1-2%
- F1分数：+4-6

### 方案一+二组合预期：
- 准确率：53.94% → **70-75%** ✅✅
- 召回率：85.87% → **78-83%**
- F1分数：66.26 → **74-79** ✅✅

---

## 🚀 快速开始

### 1. 运行改进版（任务特定阈值 + 双向评分）
```bash
cd /home/xgq/Test/detectors
python bartscore_detector_improved.py
```

### 2. 查看结果对比
```bash
python run_comparison.py
```

### 3. 查看详细报告
```bash
cat bartscore_improved_results_report.txt
```

---

## 📋 文件说明

- `bartscore_detector.py` - 原版检测器
- `bartscore_detector_improved.py` - 改进版（任务特定阈值 + 双向评分）
- `run_comparison.py` - 版本对比脚本
- `IMPROVEMENTS.md` - 本文档
- `bartscore_results.jsonl` - 原版检测结果
- `bartscore_improved_results.jsonl` - 改进版检测结果

---

## 🔍 进一步优化建议

### 针对不同幻觉类型
从报告看，对 **Subtle Conflict** 的检测效果最差 (67.16%)，可以：
1. 为Subtle类型单独设定更敏感的阈值
2. 增加语义相似度特征（Subtle幻觉往往语义相似但细节错误）
3. 使用更大的BART模型（如 bart-large 而非 bart-large-cnn）

### 针对Summary任务
Summary任务表现最差，可以：
1. 使用专门的摘要评估模型（如 BARTScore-Para）
2. 增加提取式摘要检测（检测是否直接复制原文片段）
3. 评估摘要的压缩率是否合理

### 针对假阳性问题
假阳性主要集中在分数 -1.9 到 -2.3 之间，可以：
1. 在这个区间设置"灰色地带"，要求人工review
2. 使用集成方法，要求多个检测器都判定为幻觉
3. 增加置信度阈值，低置信度的检测结果不采纳

---

## 📞 联系与反馈

如有问题或建议，请查看运行日志或检查报告文件。

