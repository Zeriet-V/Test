# 验证集使用指南

## 🎯 什么是验证集？

在机器学习中，数据集通常分为三部分：

```
完整数据集
├── 训练集 (Training Set) - 60%     # 训练模型
├── 验证集 (Validation Set) - 20%   # 调参、优化阈值 ⭐
└── 测试集 (Test Set) - 20%         # 最终评估
```

**你的情况**：
- 你用的是**预训练模型**（BART、DeBERTa），不需要训练集
- 所以只需要分割为：**验证集 + 测试集**

---

## 📊 你的数据集现状

当前你只有一个文件：
```
test_response_label.jsonl - 17,790 样本
  ├── 有幻觉: 7,664 (43.08%)
  └── 无幻觉: 10,126 (56.92%)
```

**问题**：
- ❌ 你在这个数据集上**同时**优化阈值和评估性能
- ❌ 这导致**过拟合**（阈值是针对这个特定数据集优化的）
- ❌ 性能指标可能**虚高**

---

## ✅ 正确做法

### 方案 A: 数据分割（推荐）

将 `test_response_label.jsonl` 分为两部分：

```
原始数据 (17,790)
├── 验证集 (3,558 = 20%)   # 用于优化阈值
└── 测试集 (14,232 = 80%)  # 用于最终评估
```

**步骤**：

#### 1. 分割数据
```bash
cd /home/xgq/Test/detectors

# 分割数据集 (80% 测试, 20% 验证)
python split_dataset.py \
  --input ../data/test_response_label.jsonl \
  --val-ratio 0.2 \
  --output-val ../data/validation_set.jsonl \
  --output-test ../data/test_set.jsonl
```

**输出**：
```
✓ 验证集: validation_set.jsonl (3,558 样本)
✓ 测试集: test_set.jsonl (14,232 样本)
```

#### 2. 在验证集上优化阈值
```bash
# NLI 检测器在验证集上运行
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/validation_set.jsonl \
  --output nli_validation_results.jsonl \
  --use-entailment \
  --sentence-level

# 优化阈值
python nli_threshold_optimizer.py \
  --results nli_validation_results.jsonl \
  --use-entailment

# 输出: 最优阈值 = 0.3500
```

#### 3. 在测试集上评估
```bash
# 使用找到的最优阈值
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_set.jsonl \
  --output nli_test_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.35  # 验证集优化的值
```

---

### 方案 B: 使用原数据（快速但不严格）

如果你只是想快速实验，可以直接在整个数据集上优化阈值：

```bash
# 在完整数据集上运行
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_response_label.jsonl \
  --output nli_results.jsonl \
  --use-entailment \
  --sentence-level

# 基于结果优化阈值
python nli_threshold_optimizer.py \
  --results nli_results.jsonl \
  --use-entailment

# 得到最优阈值（如 0.3500）
```

**问题**：
- ⚠️ 阈值在同一数据集上优化和评估
- ⚠️ 性能指标可能过于乐观
- ⚠️ 不符合学术标准

**适用于**：
- 快速原型验证
- 内部测试
- 不需要发表论文

---

### 方案 C: 交叉验证（最严格）

将数据分为 5 折，轮流作为验证集：

```
Fold 1: 用 Fold 2-5 验证, Fold 1 测试
Fold 2: 用 Fold 1,3-5 验证, Fold 2 测试
...
最终结果: 5次测试的平均值
```

这个最严格但也最耗时。

---

## 📋 推荐的工作流程

### 情况1: 你需要严格的评估（如论文、报告）

✅ **使用方案 A（数据分割）**

```bash
# 1. 分割数据
python split_dataset.py

# 2. 在验证集优化阈值
python nli_deberta_detector.py --input ../data/validation_set.jsonl --gpu 0 --use-entailment --sentence-level
python nli_threshold_optimizer.py --results nli_deberta_results.jsonl

# 3. 在测试集评估（使用最优阈值）
python nli_deberta_detector.py --input ../data/test_set.jsonl --gpu 0 --threshold 0.35 --use-entailment --sentence-level
```

---

### 情况2: 你只是快速实验

✅ **使用方案 B（直接运行）**

```bash
# 直接在全部数据上运行
python nli_deberta_detector.py --gpu 0 --input ../data/test_response_label.jsonl --use-entailment --sentence-level

# 优化阈值
python nli_threshold_optimizer.py --results nli_deberta_results.jsonl

# 记录结果即可
```

---

## 🔍 当前 BARTScore 的做法

从 `threshold_optimizer.py` 看：
```python
sample_size=1000  # 只用了 1000 个样本
thresholds = np.linspace(..., 100)  # 只测试 100 个阈值点
```

**问题**：
1. ❌ **样本太少**：1000 vs 17790（只用了5.6%）
2. ❌ **步长太大**：100个点，步长约0.06（vs NLI的0.01）
3. ❌ **没有明确分离验证集和测试集**
4. ❌ **阈值 -1.8649 可能不是真正的最优**

**这就是为什么**：
- BARTScore 性能可能还能提升
- 阈值优化不够精确
- 存在过拟合风险

---

## 💡 建议

### 对于 NLI（新方法）
✅ **严格按标准流程**：
```bash
# 1. 分割数据（一次性操作）
python split_dataset.py

# 2. 验证集优化
python nli_deberta_detector.py --input ../data/validation_set.jsonl --use-entailment --sentence-level --gpu 0
python nli_threshold_optimizer.py --results nli_deberta_results.jsonl

# 3. 测试集评估
python nli_deberta_detector.py --input ../data/test_set.jsonl --threshold [最优值] --use-entailment --sentence-level --gpu 0
```

### 对于 BARTScore（可选改进）
如果你想重新优化 BARTScore 的阈值，可以：
```bash
# 使用完整验证集，更精细的步长
# （需要修改 threshold_optimizer.py 的步长参数）
```

---

## 📂 生成的文件

运行 `split_dataset.py` 后会生成：
```
/home/xgq/Test/data/
├── test_response_label.jsonl      # 原始文件（保留）
├── validation_set.jsonl            # 新：验证集 (20%)
└── test_set.jsonl                  # 新：测试集 (80%)
```

---

## ⚠️ 重要原则

### 黄金法则
```
验证集: 可以反复使用，用于调参
测试集: 只能用一次，用于最终评估

❌ 永远不要在测试集上调参！
❌ 永远不要根据测试集结果修改阈值！
```

### 你目前的情况

**BARTScore**：
- 可能在全部17790样本上优化了阈值
- 然后又在同样的17790样本上评估
- → 存在**信息泄露**

**应该怎么做**：
1. 分割数据
2. 在验证集优化阈值
3. 在测试集评估（只看一次结果）
4. 报告测试集的性能

---

## 🚀 立即操作

我建议你现在：

```bash
cd /home/xgq/Test/detectors

# 1. 分割数据集（一次性）
python split_dataset.py

# 2. 然后告诉我，我们在验证集上优化阈值
```

这样才能得到**真正可靠**的性能评估！

要不要现在就分割数据集？


