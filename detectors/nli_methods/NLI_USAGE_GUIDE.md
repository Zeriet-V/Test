# NLI 改进版使用指南

## 🎯 三大改进

根据专业建议，NLI 检测器已进行三大关键改进：

### ✅ 修正 A: 非蕴含即幻觉（最推荐）
**原方法**: 只有 `contradiction` 才是幻觉  
**改进**: `contradiction` + `neutral` 都是幻觉，只有 `entailment` 是正常

**判定逻辑**:
```python
# 原方法（不推荐）
if contradiction_score > 0.5:  # 只检测矛盾
    判定为幻觉

# 改进方法（推荐）✓
if entailment_score < 0.5:  # 蕴含分数不够高
    判定为幻觉  # 包括矛盾和中立
```

**优势**:
- ✅ 更严格，漏检更少
- ✅ 符合NLI语义（只有蕴含才是一致）
- ✅ 预期召回率提升 15-25%

---

### ✅ 修正 B: 句子级检测
**原方法**: 将整个文档对直接送入 NLI  
**问题**: NLI 模型在句子对上训练，不适合长文档

**改进**: 将生成文本拆分为句子，逐句检测

**检测逻辑**:
```python
生成文本: "句子1. 句子2. 句子3."

检测过程:
  sentence_1 vs 原文 → entailment_score: 0.85 ✓
  sentence_2 vs 原文 → entailment_score: 0.25 ✗ 幻觉！
  sentence_3 vs 原文 → entailment_score: 0.90 ✓

结果: 只要有一句是幻觉，整个样本判定为幻觉
```

**优势**:
- ✅ 避免"抓大放小"问题
- ✅ 更精细的检测
- ✅ 预期准确率提升 10-15%

---

### ✅ 修正 C: 阈值校准
**原方法**: 阈值硬编码为 0.5  
**问题**: 0.5 是"拍脑袋"的数字

**改进**: 在验证集上网格搜索最优阈值

**优化流程**:
```
1. 在验证集上运行，保存所有分数
2. 测试 0.05 到 0.95 的阈值（步长0.01）
3. 计算每个阈值的 F1 分数
4. 选择 F1 最高的阈值（如 0.35）
5. 在测试集上使用最优阈值
```

**预期提升**: F1 分数提升 5-10%

---

## 🚀 使用方法

### 方法1: 推荐配置（改进版）

```bash
cd /home/xgq/Test/detectors

# 使用改进方法：非蕴含即幻觉 + 句子级检测
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_response_label.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.5
```

**说明**:
- `--use-entailment`: 使用蕴含分数（推荐）
- `--sentence-level`: 句子级检测（推荐）
- `--threshold 0.5`: 默认阈值（后续可优化）

---

### 方法2: 原始方法（对比用）

```bash
# 只检测矛盾，文档级
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_response_label.jsonl \
  --use-contradiction \
  --threshold 0.5
```

---

### 方法3: 阈值优化工作流（最佳实践）

#### 步骤1: 在验证集上运行，保存分数
```bash
# 使用任意阈值先运行一遍
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/validation_set.jsonl \
  --output nli_validation_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.5
```

#### 步骤2: 优化阈值
```bash
# 在验证结果上优化阈值
python nli_threshold_optimizer.py \
  --results nli_validation_results.jsonl \
  --use-entailment \
  --output threshold_opt

# 输出:
# 最优阈值: 0.3500
# F1分数: 72.45%
# 准确率: 68.32%
# 召回率: 77.18%
```

#### 步骤3: 使用最优阈值在测试集上运行
```bash
# 使用找到的最优阈值（例如 0.35）
python nli_deberta_detector.py \
  --gpu 0 \
  --input ../data/test_response_label.jsonl \
  --output nli_test_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.35
```

---

## 📊 预期性能对比

| 配置 | F1分数 | 准确率 | 召回率 | 推荐度 |
|------|--------|--------|--------|--------|
| 原方法 (contradiction, 文档级) | 55-65 | 50-60% | 60-70% | ⭐⭐ |
| 改进A (entailment, 文档级) | 60-70 | 55-65% | 70-80% | ⭐⭐⭐⭐ |
| 改进A+B (entailment, 句子级) | 65-75 | 60-70% | 75-85% | ⭐⭐⭐⭐⭐ |
| 改进A+B+C (优化阈值) | **70-80** | **65-75%** | **75-85%** | ⭐⭐⭐⭐⭐ 最佳 |

---

## 💡 使用示例

### 示例1: 基础使用（改进版）
```python
from nli_deberta_detector import DeBERTaNLIDetector

detector = DeBERTaNLIDetector(gpu_id=0)

source = "公司成立于2010年"
generated = "公司成立于2015年"

# 使用蕴含分数
has_hall, result = detector.detect_hallucination(
    source, 
    generated,
    threshold=0.5,
    use_entailment=True  # 推荐
)

print(f"蕴含分数: {result['entailment_score']:.4f}")
print(f"是否幻觉: {has_hall}")  # True (因为蕴含分数很低)
```

### 示例2: 句子级检测
```python
source = "公司A成立于2010年。总部在北京。"
generated = "公司A成立于2010年。总部在上海。CEO是张三。"

# 句子级检测
has_hall, result = detector.detect_hallucination(
    source,
    generated,
    threshold=0.5,
    use_entailment=True,
    sentence_level=True  # 启用句子级
)

# 查看每句的结果
for sent_result in result['sentence_results']:
    print(f"句子: {sent_result['sentence']}")
    print(f"  蕴含分数: {sent_result['entailment_score']:.4f}")
    print(f"  是否幻觉: {sent_result['is_hallucination']}")

# 输出:
# 句子: 公司A成立于2010年
#   蕴含分数: 0.9500  ✓
#   是否幻觉: False
# 句子: 总部在上海
#   蕴含分数: 0.1200  ✗ 幻觉（矛盾）
#   是否幻觉: True
# 句子: CEO是张三
#   蕴含分数: 0.2800  ✗ 幻觉（无根据）
#   是否幻觉: True
```

---

## 📋 命令行参数完整列表

```bash
python nli_deberta_detector.py --help
```

**主要参数**:
- `--gpu N` : 指定GPU ID
- `--input FILE` : 输入文件路径
- `--output FILE` : 输出文件路径
- `--threshold FLOAT` : 阈值（默认0.5）
- `--use-entailment` : 使用蕴含分数（推荐，默认开启）
- `--use-contradiction` : 使用矛盾分数（不推荐）
- `--sentence-level` : 启用句子级检测（推荐）
- `--model NAME` : 模型名称

---

## 🆚 配置对比

### 配置1: 原始方法
```bash
--use-contradiction  # 只检测矛盾
# 文档级（默认）
```
- 召回率低（漏检多）
- 对长文本不准确
- 不推荐 ⭐⭐

### 配置2: 改进A
```bash
--use-entailment  # 非蕴含即幻觉
# 文档级（默认）
```
- 召回率提升
- 仍有长文本问题
- 推荐 ⭐⭐⭐⭐

### 配置3: 改进A+B（推荐）
```bash
--use-entailment  # 非蕴含即幻觉
--sentence-level  # 句子级检测
```
- 召回率和准确率都提升
- 解决长文本问题
- 强烈推荐 ⭐⭐⭐⭐⭐

### 配置4: 改进A+B+C（最佳）
```bash
# 先优化阈值
python nli_threshold_optimizer.py --results validation.jsonl

# 使用最优阈值
--use-entailment
--sentence-level
--threshold 0.35  # 优化后的值
```
- 性能最优
- 需要验证集
- 最佳实践 ⭐⭐⭐⭐⭐

---

## ⚠️ 重要提示

### 1. 阈值方向不同
```
use_entailment=True:   score < threshold → 幻觉
use_entailment=False:  score > threshold → 幻觉
```
注意方向相反！

### 2. 句子级检测会更慢
- 文档级: 1次模型推理
- 句子级 (3句话): 3次模型推理
- 权衡: 准确性 vs 速度

### 3. 必须先优化阈值
- 默认0.5不一定最优
- 建议在验证集上优化
- 可能最优阈值是 0.3 或 0.7

---

## 📂 输出文件

### 检测结果 (nli_deberta_results.jsonl)
```json
{
  "id": "12345",
  "task_type": "Summary",
  "has_label": true,
  "label_types": ["Evident Conflict"],
  "nli_label": "neutral",
  "entailment_score": 0.2500,
  "contradiction_score": 0.3500,
  "neutral_score": 0.4000,
  "detected": true,
  "threshold": 0.5
}
```

### 句子级结果（如果启用）
```json
{
  "sentence_results": [
    {
      "sentence": "句子1",
      "entailment_score": 0.85,
      "is_hallucination": false
    },
    {
      "sentence": "句子2",
      "entailment_score": 0.25,
      "is_hallucination": true
    }
  ],
  "num_sentences": 2,
  "num_hallucination_sentences": 1
}
```

---

## 🎓 最佳实践总结

1. ✅ **使用蕴含分数** (`--use-entailment`)
2. ✅ **启用句子级检测** (`--sentence-level`)
3. ✅ **在验证集上优化阈值** (使用 `nli_threshold_optimizer.py`)
4. ✅ **用优化后的阈值运行测试集**

**完整流程**:
```bash
# 1. 验证集运行
python nli_deberta_detector.py --gpu 0 --input validation.jsonl --output val_results.jsonl --use-entailment --sentence-level

# 2. 优化阈值
python nli_threshold_optimizer.py --results val_results.jsonl --use-entailment

# 3. 测试集运行（使用最优阈值，如0.35）
python nli_deberta_detector.py --gpu 0 --input test.jsonl --output test_results.jsonl --use-entailment --sentence-level --threshold 0.35
```

---

**预期性能**: F1分数 70-80%（相比原方法提升 15-25%）

Good luck! 🎉

