# 完整的幻觉检测工作流程

## 📋 标准流程（严格评估）

### 步骤0: 数据准备

**当前情况**：
- 你只有 `test_response_label.jsonl` (17,790 样本)
- 需要分割为：**验证集** + **测试集**

**为什么需要分割？**
- ✅ 验证集用于调参、优化阈值（可以反复使用）
- ✅ 测试集用于最终评估（只能用一次）
- ✅ 避免过拟合，性能评估更可靠

---

## 🚀 完整操作流程

### 第一步：分割数据集（一次性操作）

```bash
cd /home/xgq/Test/detectors

# 分割数据：80% 测试集, 20% 验证集
python split_dataset.py \
  --input ../data/test_response_label.jsonl \
  --val-ratio 0.2 \
  --output-val ../data/validation_set.jsonl \
  --output-test ../data/test_set.jsonl
```

**输出**：
```
✓ 验证集: validation_set.jsonl (3,558 样本, 43% 有幻觉)
✓ 测试集: test_set.jsonl (14,232 样本, 43% 有幻觉)
```

---

### 第二步：在验证集上优化阈值

#### 方案A: NLI-DeBERTa（推荐）

```bash
# 1. 在验证集上运行 NLI
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/validation_set.jsonl \
  --output nli_validation_results.jsonl \
  --use-entailment \
  --sentence-level

# 2. 优化阈值
python nli_threshold_optimizer.py \
  --results nli_validation_results.jsonl \
  --use-entailment \
  --output nli_threshold_opt

# 输出示例:
# 最优阈值: 0.3500
# F1分数: 72.45%
# 准确率: 68.32%
# 召回率: 77.18%
```

#### 方案B: BARTScore

```bash
# 1. 在验证集上运行 BARTScore
cd /home/xgq/Test/detectors/bartscore_methods
python bartscore_detector.py \
  --gpu 0 \
  --input ../../data/validation_set.jsonl \
  --output bartscore_validation_results.jsonl \
  --threshold -2.0  # 随便设一个值，后面会优化

# 2. 优化阈值（改进版）
cd /home/xgq/Test/detectors
python bartscore_threshold_optimizer.py \
  --results bartscore_methods/bartscore_validation_results.jsonl \
  --step 0.01 \
  --output bartscore_threshold_opt

# 输出示例:
# 最优阈值: -1.8734
# F1分数: 67.83%
# 准确率: 55.21%
# 召回率: 86.45%
```

**可选：为每个任务优化**
```bash
python bartscore_threshold_optimizer.py \
  --results bartscore_methods/bartscore_validation_results.jsonl \
  --step 0.01 \
  --optimize-by-task \
  --output bartscore_task_threshold_opt

# 输出:
# Summary: -1.7234 (F1=48.56%)
# QA: -2.1045 (F1=54.32%)
# Data2txt: -2.5123 (F1=82.45%)
```

---

### 第三步：在测试集上评估（最终评估）

使用验证集找到的**最优阈值**，在测试集上运行一次：

#### NLI-DeBERTa（使用最优阈值，如0.35）

```bash
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_set.jsonl \
  --output nli_test_final_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.35  # 验证集优化得到的值
```

#### BARTScore（使用最优阈值，如-1.8734）

```bash
cd /home/xgq/Test/detectors/bartscore_methods
python bartscore_detector.py \
  --gpu 0 \
  --input ../../data/test_set.jsonl \
  --output bartscore_test_final_results.jsonl \
  --threshold -1.8734  # 验证集优化得到的值
```

---

### 第四步：报告测试集性能

**重要**: 只报告测试集上的性能！

```
最终性能（测试集，14,232样本）:
  NLI-DeBERTa:
    - F1分数: 71.23%
    - 准确率: 67.45%
    - 召回率: 75.67%
  
  BARTScore:
    - F1分数: 66.89%
    - 准确率: 54.32%
    - 召回率: 85.23%
```

---

## 📊 不同场景的推荐流程

### 场景1: 快速实验（不需要发表）

**简化流程**：
```bash
# 直接在全部数据上运行和优化
python nli_deberta_detector.py --gpu 0 --use-entailment --sentence-level
python nli_threshold_optimizer.py --results nli_deberta_results.jsonl
```

**注意**: 性能可能虚高，但快速

---

### 场景2: 论文/正式报告（需要严格评估）

**标准流程**：
```bash
# 1. 分割数据（一次性）
python split_dataset.py

# 2. 验证集优化
python nli_deberta_detector.py --input validation_set.jsonl --gpu 0 --use-entailment --sentence-level
python nli_threshold_optimizer.py --results nli_deberta_results.jsonl

# 3. 测试集评估（只运行一次！）
python nli_deberta_detector.py --input test_set.jsonl --threshold [最优值] --gpu 0 --use-entailment --sentence-level

# 4. 报告测试集性能
```

---

### 场景3: 方法对比（BARTScore vs NLI）

```bash
# 1. 分割数据
python split_dataset.py

# 2A. BARTScore - 验证集优化
cd bartscore_methods
python bartscore_detector.py --input ../../data/validation_set.jsonl --gpu 0
cd ..
python bartscore_threshold_optimizer.py --results bartscore_methods/bartscore_validation_results.jsonl

# 2B. NLI - 验证集优化
python nli_deberta_detector.py --input ../data/validation_set.jsonl --gpu 0 --use-entailment --sentence-level
python nli_threshold_optimizer.py --results nli_deberta_results.jsonl

# 3A. BARTScore - 测试集评估
cd bartscore_methods
python bartscore_detector.py --input ../../data/test_set.jsonl --threshold [最优值] --gpu 0

# 3B. NLI - 测试集评估
cd ..
python nli_deberta_detector.py --input ../data/test_set.jsonl --threshold [最优值] --gpu 0 --use-entailment --sentence-level

# 4. 对比测试集性能
```

---

## ⚠️ 关键原则

### ✅ 正确做法

1. **数据分离**
   ```
   原始数据 → 验证集 + 测试集
   ```

2. **验证集用途**
   - 优化阈值 ✓
   - 调整超参数 ✓
   - 选择模型 ✓
   - 可以反复使用 ✓

3. **测试集用途**
   - 只用于最终评估 ✓
   - 只运行一次 ✓
   - 不能根据结果调参 ✓
   - 用于报告性能 ✓

### ❌ 错误做法

1. **在测试集上调参**
   ```python
   # 错误！
   在测试集上试了10个阈值，选最好的
   ```

2. **验证集和测试集混用**
   ```python
   # 错误！
   在全部数据上优化阈值，然后在全部数据上评估
   ```

3. **看了测试集结果后修改**
   ```python
   # 错误！
   看到测试集F1只有60%，调整阈值到65%
   ```

---

## 📈 预期改进

### BARTScore 阈值优化改进

| 方法 | 样本数 | 步长 | 预期最优阈值 | 预期F1 |
|------|--------|------|--------------|--------|
| 旧方法 | 1,000 | ~0.06 | -1.8649 | 66.26 |
| **新方法** | **3,558** | **0.01** | **-1.87xx** | **66-68** |

**改进点**：
- ✅ 样本量增加 3.5 倍
- ✅ 步长精细 6 倍
- ✅ 阈值更精确
- ✅ 预期 F1 提升 0.5-2%

---

## 🎯 快速命令参考

### 完整流程（复制粘贴即可）

```bash
cd /home/xgq/Test/detectors

# ============= 步骤1: 分割数据 =============
python split_dataset.py

# ============= 步骤2: NLI优化 =============
# 验证集运行
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/validation_set.jsonl \
  --output nli_val_results.jsonl \
  --use-entailment \
  --sentence-level

# 优化阈值
python nli_threshold_optimizer.py \
  --results nli_val_results.jsonl \
  --use-entailment

# 记录最优阈值（如 0.35）

# 测试集评估
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_set.jsonl \
  --output nli_test_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.35

# ============= 步骤3: BARTScore优化 =============
# 验证集运行
cd bartscore_methods
python bartscore_detector.py \
  --gpu 1 \
  --input ../../data/validation_set.jsonl \
  --output bartscore_val_results.jsonl

# 优化阈值
cd ..
python bartscore_threshold_optimizer.py \
  --results bartscore_methods/bartscore_val_results.jsonl \
  --step 0.01

# 记录最优阈值（如 -1.8734）

# 测试集评估
cd bartscore_methods
python bartscore_detector.py \
  --gpu 1 \
  --input ../../data/test_set.jsonl \
  --output bartscore_test_results.jsonl \
  --threshold -1.8734
```

---

## 📂 文件结构

```
/home/xgq/Test/
├── data/
│   ├── test_response_label.jsonl  # 原始数据（保留）
│   ├── validation_set.jsonl       # 新：验证集 (20%, 3,558)
│   └── test_set.jsonl             # 新：测试集 (80%, 14,232)
│
├── detectors/
│   ├── split_dataset.py                      # 数据分割工具
│   ├── nli_deberta_detector.py               # NLI检测器
│   ├── nli_threshold_optimizer.py            # NLI阈值优化器
│   ├── bartscore_threshold_optimizer.py      # BARTScore阈值优化器（新）
│   │
│   └── bartscore_methods/
│       └── bartscore_detector.py             # BARTScore检测器
```

---

## 💡 关键要点

### 1. 验证集 vs 测试集

| 数据集 | 用途 | 使用次数 | 可以调参? |
|--------|------|----------|-----------|
| **验证集** | 优化阈值、选模型 | 多次 | ✓ 可以 |
| **测试集** | 最终评估 | 一次 | ✗ 不可以 |

### 2. BARTScore 旧方法的问题

```python
# threshold_optimizer.py (旧版)
sample_size=1000          # ❌ 太少
thresholds = np.linspace(..., 100)  # ❌ 步长太大
# 没有明确的验证集/测试集分离    # ❌ 
```

### 3. 改进版的优势

```python
# bartscore_threshold_optimizer.py (新版)
使用完整验证集              # ✅ 3,558 样本
step=0.01                  # ✅ 精细步长
明确的验证/测试分离         # ✅ 
```

---

## 📊 预期结果

### 验证集优化后

**NLI-DeBERTa** (use_entailment + sentence_level):
- 最优阈值: 0.30-0.40
- 验证集 F1: 70-78%
- 测试集 F1: 68-76% (预期)

**BARTScore** (精细优化):
- 最优阈值: -1.85 到 -1.90
- 验证集 F1: 66-68%
- 测试集 F1: 65-67% (预期)

---

## 🎯 立即开始

**我建议现在：**

```bash
cd /home/xgq/Test/detectors

# 步骤1: 分割数据集（5秒钟）
python split_dataset.py
```

这样你就有了：
- ✅ `validation_set.jsonl` - 用于调参
- ✅ `test_set.jsonl` - 用于最终评估

然后我们可以分别优化 NLI 和 BARTScore 的阈值，得到真正可靠的性能评估！

要不要现在就分割？


