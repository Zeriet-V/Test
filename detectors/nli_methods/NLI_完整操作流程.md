# NLI-DeBERTa 幻觉检测完整操作流程

## 🎯 使用模型
**MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli**

**优势**：
- ✅ 在多个NLI数据集上训练（MNLI + FEVER + ANLI + LING + WANLI）
- ✅ 泛化能力更强
- ✅ 对幻觉检测更鲁棒
- ✅ 约1.5GB，首次下载需5-10分钟

---

## 📋 完整操作流程

### ✅ 步骤0: 数据准备（已完成）

```
✓ 验证集: /home/xgq/Test/data/validation_set.jsonl (3,557 样本)
✓ 测试集: /home/xgq/Test/data/test_set.jsonl (14,233 样本)
```

---

### 🔧 步骤1: 在验证集上运行（优化阈值）

```bash
cd /home/xgq/Test/detectors

# 运行 NLI 检测器（使用改进方法）
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_validation_results.jsonl \
  --use-entailment \
  --sentence-level
```

**参数说明**：
- `--gpu 0`: 使用GPU 0
- `--input`: 验证集路径
- `--output`: 输出文件名
- `--use-entailment`: 使用蕴含分数判定（推荐）⭐
- `--sentence-level`: 句子级检测（推荐）⭐

**首次运行**会自动下载模型：
```
本地无缓存，开始在线下载...
（首次下载约1.3GB，需要几分钟）
Downloading: 100%|████████| 1.5G/1.5G [06:00<00:00]
✓ 在线下载并加载成功！
```

**运行时间**：约10-20分钟（3,557样本 + 句子级检测）

**输出**：
- `nli_validation_results.jsonl` - 每个样本的检测结果
- `nli_validation_results_report.txt` - 详细报告

---

### 📊 步骤2: 优化阈值（找最优值）

```bash
python nli_threshold_optimizer.py \
  --results nli_validation_results.jsonl \
  --use-entailment \
  --output nli_threshold_opt
```

**运行时间**：<1分钟

**输出示例**：
```
最优阈值: 0.3500
F1分数: 72.45%
准确率: 68.32%
召回率: 77.18%

✓ 报告已保存到: nli_threshold_opt_report.txt
✓ 分析图已保存到: nli_threshold_opt_analysis.png
```

**查看报告**：
```bash
cat nli_threshold_opt_report.txt
```

**重点关注**：
- 最优阈值（如 0.35）
- Top 10 阈值列表
- Precision-Recall 权衡

---

### 🎯 步骤3: 在测试集上评估（最终结果）

使用步骤2找到的最优阈值（假设是 0.35）：

```bash
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/test_set.jsonl \
  --output nli_test_final_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.35
```

**⚠️ 重要**：
- 只运行一次！
- 不要根据结果调整阈值！
- 这是最终报告的性能

**运行时间**：约40-80分钟（14,233样本 + 句子级检测）

**输出**：
- `nli_test_final_results.jsonl` - 测试集检测结果
- `nli_test_final_results_report.txt` - 最终报告 ⭐

---

### 📈 步骤4: 查看最终报告

```bash
cat nli_test_final_results_report.txt
```

**关键指标**：
```
【整体性能指标】
  准确率 (Precision): XX.XX%
  召回率 (Recall): XX.XX%
  F1分数: XX.XX

【按任务类型统计】
◆ Summary 任务: F1=XX.XX
◆ QA 任务: F1=XX.XX
◆ Data2txt 任务: F1=XX.XX

【按幻觉类型统计】
◆ Evident Conflict: 检测率 XX.XX%
◆ Subtle Conflict: 检测率 XX.XX%
◆ Evident Baseless Info: 检测率 XX.XX%
◆ Subtle Baseless Info: 检测率 XX.XX%
```

---

## 🆚 对比 BARTScore（可选）

如果你想对比 NLI 和 BARTScore 的性能：

### BARTScore 在验证集优化

```bash
cd /home/xgq/Test/detectors/bartscore_methods

# 1. 验证集运行
python bartscore_detector.py \
  --gpu 1 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output bartscore_validation_results.jsonl

# 2. 优化阈值（改进版）
cd ..
python bartscore_threshold_optimizer.py \
  --results bartscore_methods/bartscore_validation_results.jsonl \
  --step 0.01 \
  --output bartscore_threshold_opt

# 3. 测试集评估（使用最优阈值）
cd bartscore_methods
python bartscore_detector.py \
  --gpu 1 \
  --input /home/xgq/Test/data/test_set.jsonl \
  --output bartscore_test_final_results.jsonl \
  --threshold [步骤2得到的最优值]
```

---

## 📊 预期性能

### NLI-DeBERTa (改进方法)
**配置**: entailment判定 + 句子级检测 + 优化阈值

**预期指标** (测试集):
- F1分数: **70-78%**
- 准确率: 65-73%
- 召回率: 75-85%

**特别擅长**:
- Evident Conflict: 85-95%
- Subtle Conflict: 70-85%

### BARTScore (改进优化)
**配置**: 统一阈值，步长0.01精细优化

**预期指标** (测试集):
- F1分数: **66-68%**
- 准确率: 54-57%
- 召回率: 84-87%

---

## ⏱️ 时间估算

### 完整流程总时间：

```
步骤1: 验证集运行
  - 首次下载模型: 5-10分钟（一次性）
  - NLI检测: 10-20分钟（3,557样本，句子级）
  
步骤2: 优化阈值
  - <1分钟

步骤3: 测试集评估
  - NLI检测: 40-80分钟（14,233样本，句子级）

总计: 约1-2小时
```

**优化建议**：
- 步骤1和BARTScore可以同时在两张GPU上运行
- 句子级检测慢，但准确率更高

---

## 🚀 现在就开始！

### 完整命令（复制粘贴即可）

```bash
cd /home/xgq/Test/detectors

# ============= 步骤1: 验证集运行 =============
echo "步骤1: 在验证集上运行 NLI 检测器..."
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_validation_results.jsonl \
  --use-entailment \
  --sentence-level

# 等待完成（10-20分钟）...

# ============= 步骤2: 优化阈值 =============
echo "步骤2: 优化阈值..."
python nli_threshold_optimizer.py \
  --results nli_validation_results.jsonl \
  --use-entailment \
  --output nli_threshold_opt

# 查看最优阈值
echo "查看最优阈值："
grep "最优阈值" nli_threshold_opt_report.txt

# 假设得到最优阈值 0.35

# ============= 步骤3: 测试集评估 =============
echo "步骤3: 在测试集上评估（使用最优阈值）..."
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/test_set.jsonl \
  --output nli_test_final_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold 0.35

# ============= 步骤4: 查看最终结果 =============
echo "步骤4: 查看最终报告..."
cat nli_test_final_results_report.txt
```

---

## 💡 关键配置说明

### 模型
```
MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
```
- 在5个NLI数据集上训练
- 泛化能力强
- 特别适合幻觉检测

### 判定方法
```
--use-entailment  # 蕴含分数 < threshold → 幻觉
```
- 将 contradiction + neutral 都视为幻觉
- 比只检测矛盾更严格
- 预期召回率提升 20%+

### 检测粒度
```
--sentence-level  # 句子级检测
```
- 逐句检测，避免"抓大放小"
- 任一句有幻觉 → 整体幻觉
- 预期准确率提升 10-15%

### 阈值
```
先在验证集上优化，再在测试集上使用
```
- 不是硬编码的 0.5
- 数据驱动，F1最优
- 预期提升 5-10%

---

## 📊 预期vs BARTScore

| 指标 | BARTScore | NLI-DeBERTa改进版 | 提升 |
|------|-----------|-------------------|------|
| F1分数 | 66.26 | **70-78** | +5-12 ⭐ |
| 准确率 | 53.94% | **65-73%** | +11-19% |
| 召回率 | 85.87% | 75-85% | 持平 |
| Conflict检测 | 90% | **90-95%** | +0-5% |
| Baseless检测 | 90% | 60-75% | -15-30% |

**结论**：
- NLI 在**准确率**上显著优于 BARTScore
- NLI 特别擅长检测**矛盾型幻觉**
- BARTScore 对**无根据信息**检测更好

---

## 📁 输出文件

运行完成后会生成：

```
/home/xgq/Test/detectors/
├── nli_validation_results.jsonl          # 验证集检测结果
├── nli_validation_results_report.txt     # 验证集报告
├── nli_threshold_opt_report.txt          # 阈值优化报告 ⭐
├── nli_threshold_opt_analysis.png        # 阈值分析图
├── nli_test_final_results.jsonl          # 测试集结果
└── nli_test_final_results_report.txt     # 最终报告 ⭐⭐
```

**最重要的文件**：
1. `nli_threshold_opt_report.txt` - 告诉你最优阈值
2. `nli_test_final_results_report.txt` - 最终性能（用于论文/报告）

---

## ⚠️ 注意事项

1. **首次运行会下载模型**（约1.5GB，5-10分钟）
2. **句子级检测比较慢**（但准确率高）
3. **测试集只能运行一次**（不能看结果后调整）
4. **GPU显存需求**：约6-8GB

如果显存不足，可以用更小的模型：
```bash
--model microsoft/deberta-base-mnli  # 400MB，显存需求3-4GB
```

---

## 🎯 立即开始

**执行这个命令开始步骤1**：

```bash
cd /home/xgq/Test/detectors && python nli_deberta_detector.py --gpu 0 --input /home/xgq/Test/data/validation_set.jsonl --output nli_validation_results.jsonl --use-entailment --sentence-level
```

运行后等待10-20分钟，完成后告诉我，我们进行步骤2（优化阈值）！

---

**快速参考**：
```bash
# 完整三步（依次执行）
cd /home/xgq/Test/detectors

# 步骤1（10-20分钟）
python nli_deberta_detector.py --gpu 0 --input /home/xgq/Test/data/validation_set.jsonl --output nli_validation_results.jsonl --use-entailment --sentence-level

# 步骤2（<1分钟）
python nli_threshold_optimizer.py --results nli_validation_results.jsonl --use-entailment

# 步骤3（40-80分钟，使用步骤2得到的阈值）
python nli_deberta_detector.py --gpu 0 --input /home/xgq/Test/data/test_set.jsonl --output nli_test_final_results.jsonl --use-entailment --sentence-level --threshold [最优值]
```

Good luck! 🎉

