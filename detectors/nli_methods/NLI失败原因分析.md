# NLI 失败的根本原因分析

## 🔴 核心问题：数据格式不匹配

### 发现的问题

**矛盾分数异常高**: 平均 0.9743

这意味着模型认为**几乎所有样本都是矛盾/中立**（无论是否真的有幻觉）。

---

## 🔍 根本原因

### Data2txt 任务的格式问题

**输入到 NLI 模型的内容**:
```
Premise（原文）: 
  "{'name': 'Jack in the Box', 'address': '6875 Hollister Ave', 
   'city': 'Goleta', 'state': 'CA', 'categories': 'Restaurants...}"

Hypothesis（生成文本）:
  "Sure! Here's an objective overview of Jack in the Box based 
   on the provided structured data: Jack in the Box is a fast 
   food restaurant located in Goleta, CA..."
```

**NLI 模型的视角**:
- Premise 看起来像 **Python代码/JSON**
- Hypothesis 是 **自然英语**
- 格式完全不匹配！
- 模型判断: 这两个东西没有关系 → **neutral/contradiction**
- 矛盾分数: 0.95-0.99

**结果**: 
- 所有 Data2txt 样本的矛盾分数都很高（0.95+）
- 无论是否有幻觉
- 无法区分

---

## 📊 数据组成分析

你的验证集：
```
Summary: 1136 样本 (32%)
QA: 1207 样本 (34%)
Data2txt: 1214 样本 (34%)  ← 问题源头
```

**Data2txt 的影响**:
- 1214 个样本都被误判
- 占总样本的 34%
- 拉低整体准确率
- 导致 43% 的准确率

---

## ✅ 解决方案

### 方案1: 排除 Data2txt（验证NLI是否有效）

只在 **Summary + QA** 上测试：

**步骤1**: 筛选数据
```bash
cd /home/xgq/Test/detectors
python filter_summary_qa_only.py
```

**输出**: 
- `validation_summary_qa_conflict.jsonl`
- `test_summary_qa_conflict.jsonl`

**步骤2**: 测试
```bash
cd nli_methods
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_summary_qa_conflict.jsonl \
  --use-contradiction
```

**预期**: 
- 准确率: 43% → **60-80%** (显著提升！)
- 因为 Summary/QA 的原文是自然语言

---

### 方案2: 修复 Data2txt 的原文格式

将结构化数据转换为自然语言：

**当前**:
```python
text_original = "{'name': 'Jack in the Box', 'address': '...'}"
```

**修复后**:
```python
text_original = "Name: Jack in the Box. Address: 6875 Hollister Ave. 
                 City: Goleta. State: CA. Categories: Restaurants, Fast Food..."
```

这在你的代码中应该已经做了（Data2txt部分），但可能没有生效。

---

### 方案3: 使用不同的检测策略

**针对不同任务**:
- Summary/QA: 用 NLI（自然语言）
- Data2txt: 用 BARTScore 或其他方法（结构化数据）

---

## 📊 预期改进

### 全数据集（当前）
```
包含 Data2txt: 
  准确率: 28-43%  ← Data2txt 全部误判
  区分度: 0.014
```

### 只 Summary/QA（方案1）
```
排除 Data2txt:
  准确率: 60-80%  ← 应该显著提升！
  区分度: 0.20-0.40
```

---

## 🎯 诊断总结

### 为什么 NLI 把所有样本都判为幻觉？

1. **Data2txt 格式问题** (34%的数据)
   - 字典格式 vs 自然语言
   - 模型无法理解
   - 全部高矛盾分数

2. **长文本截断** (剩余66%的数据)
   - Summary/QA 也很长
   - 被截断后信息丢失
   - 部分误判

3. **阈值被迫很高**
   - 因为矛盾分数普遍高
   - 最优阈值=0.89（极高）
   - 还是误报很多

---

## 🚀 立即验证

**测试假设**：排除 Data2txt 后，NLI 在 Summary/QA 上应该表现好

```bash
cd /home/xgq/Test/detectors

# 1. 筛选 Summary/QA 数据
python filter_summary_qa_only.py

# 2. 在 Summary/QA 子集上测试
cd nli_methods
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_summary_qa_conflict.jsonl \
  --output nli_val_summary_qa.jsonl \
  --use-contradiction

# 3. 查看准确率
cat nli_val_summary_qa_report.txt | grep "准确率"
```

**如果准确率提升到 60%+** → NLI 有效，但只对 Summary/QA 有效！

**如果还是 <50%** → NLI 真的不适合，用 BARTScore。

---

**现在运行筛选工具吧！** 看看去掉 Data2txt 后，NLI 能否正常工作！
