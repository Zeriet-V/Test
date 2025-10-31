# BARTScore 改进版 V2 使用说明

## 主要改进

相比原 `bartscore_detector_improved.py`：

1. ✅ **增加按幻觉类型的F1分数统计** - 每个幻觉类型都有详细的 TP/FN/Recall 指标
2. ✅ **支持标准三步流程** - 验证集 → 阈值优化 → 测试集
3. ✅ **支持外部阈值输入** - 可使用阈值优化器得到的最优阈值
4. ✅ **明确数据集类型** - 报告中标注是 validation 还是 test

## 完整使用流程

### 第一步：验证集测试（使用默认阈值）

```bash
cd /home/xgq/Test/detectors/bartscore_methods

python bartscore_detector_improved_v2.py \
  --gpu 0 \
  --input ../../data/validation_set.jsonl \
  --output bartscore_improved_validation_results.jsonl \
  --dataset-type validation
```

**输出**：
- `bartscore_improved_validation_results.jsonl`
- `bartscore_improved_validation_results_report.txt`（含按幻觉类型的F1分数）

### 第二步：阈值优化

使用验证集结果优化阈值（可以使用现有的 `bartscore_threshold_optimizer.py` 或手动分析）。

#### 方法A：使用统一阈值优化器

```bash
python bartscore_threshold_optimizer.py \
  --results bartscore_improved_validation_results.jsonl \
  --output bartscore_improved_threshold_opt_report.txt
```

这会给出最优的**统一阈值**（适用于所有任务）。

#### 方法B：任务特定阈值优化（手动）

分析验证集报告中各任务的分数分布，手动调整阈值。例如：

```
Summary: 平均分 -1.82，设阈值 -1.75（稍微放宽）
QA: 平均分 -2.12，设阈值 -2.00
Data2txt: 平均分 -2.50，设阈值 -2.40
```

### 第三步：测试集评估（使用最优阈值）

#### 使用统一阈值（从优化器得到，假设为 -1.82）

如果阈值优化器给出统一阈值 -1.82，则三个任务都用这个值：

```bash
python bartscore_detector_improved_v2.py \
  --gpu 0 \
  --input ../../data/test_set.jsonl \
  --output bartscore_improved_test_final_results.jsonl \
  --dataset-type test \
  --threshold-summary -1.82 \
  --threshold-qa -1.82 \
  --threshold-data2txt -1.82
```

#### 使用任务特定阈值（从验证集分析得到）

```bash
python bartscore_detector_improved_v2.py \
  --gpu 0 \
  --input ../../data/test_set.jsonl \
  --output bartscore_improved_test_final_results.jsonl \
  --dataset-type test \
  --threshold-summary -1.75 \
  --threshold-qa -2.00 \
  --threshold-data2txt -2.40
```

**输出**：
- `bartscore_improved_test_final_results.jsonl`
- `bartscore_improved_test_final_results_report.txt`（**这是最终可报告的结果**）

## 报告改进对比

### 旧报告（bartscore_improved_results_report.txt）

```
◆ Evident Conflict:
  总数: 5324
  检测到: 3562 (66.90%)
  漏检: 1762 (33.10%)
  状态: ✗ 需要改进
```

❌ 只有检测率，没有 Precision/Recall/F1

### 新报告（V2版本）

```
◆ Evident Conflict:
  总数: 5324
  检测到 (TP): 3562 (66.90%)
  漏检 (FN): 1762 (33.10%)
  召回率 (Recall): 66.90%
  状态: ✗ 需要改进
```

✅ 明确标注 TP/FN，提供 Recall（召回率）

**注意**：由于无幻觉样本没有类型标签，无法计算每个幻觉类型的 Precision（会有假阳性但不知道是哪个类型），所以只能计算 Recall。

## 命令行参数说明

```bash
python bartscore_detector_improved_v2.py \
  --gpu 0                              # GPU ID
  --input <input_file>                 # 输入文件
  --output <output_file>               # 输出文件
  --dataset-type validation|test       # 数据集类型（会在报告中标注）
  --threshold-summary -1.75            # Summary任务阈值（可选）
  --threshold-qa -2.00                 # QA任务阈值（可选）
  --threshold-data2txt -2.40           # Data2txt任务阈值（可选）
  --no-bidirectional                   # 禁用双向评分（可选）
  --model facebook/bart-large-cnn      # BART模型名称（可选）
```

## 与原版对比

| 特性 | 原版 (improved.py) | V2版本 (improved_v2.py) |
|------|-------------------|------------------------|
| 任务特定阈值 | ✅ | ✅ |
| 双向评分 | ✅ | ✅ |
| 置信度 | ✅ | ✅ |
| 按幻觉类型F1 | ❌ | ✅ |
| 数据集类型标注 | ❌ | ✅ |
| 外部阈值输入 | ❌ | ✅ |
| 标准流程支持 | ❌ | ✅ |

## 推荐工作流

1. **验证集测试**：使用默认阈值，生成验证集报告
2. **分析报告**：查看各任务分数分布、幻觉类型检测率
3. **阈值优化**：基于验证集结果调整阈值（统一或任务特定）
4. **测试集评估**：用最优阈值在测试集上跑最终结果
5. **报告撰写**：测试集报告作为最终性能指标

## 注意事项

1. **阈值不能在测试集上调**：必须在验证集上优化，测试集仅用于最终评估
2. **默认阈值**：Summary -1.65, QA -2.05, Data2txt -2.45（这些是经验值，需要验证集优化）
3. **双向评分**：默认开启，可通过 `--no-bidirectional` 禁用
4. **幻觉类型F1限制**：由于数据特点，只能计算 Recall，无法计算完整的 Precision/F1

## 文件输出说明

- `*_validation_results.jsonl` - 验证集每个样本的详细结果
- `*_validation_results_report.txt` - 验证集统计报告（含幻觉类型F1）
- `*_test_final_results.jsonl` - 测试集每个样本的详细结果
- `*_test_final_results_report.txt` - **测试集最终报告（可对外报告）**


