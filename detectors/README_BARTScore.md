# BARTScore 幻觉检测器使用说明

## 文件说明

### 1. `bartscore_detector.py` - 主检测器
BARTScore幻觉检测的主要实现文件。

**功能：**
- 使用BARTScore评估生成文本与原文的一致性
- 支持自动或手动指定阈值
- 生成详细的检测报告
- 按任务类型和幻觉类型统计

### 2. `threshold_optimizer.py` - 阈值优化器
独立的阈值分析工具，用于确定最优检测阈值。

**功能：**
- 分析BARTScore分数分布
- 基于F1分数确定最优阈值
- 生成详细的阈值分析报告
- 可视化有幻觉/无幻觉样本的分数统计

---

## 使用方法

### 方法1: 先分析阈值，再运行检测（推荐）

**第一步：确定最优阈值**
```bash
cd detectors
python threshold_optimizer.py
```

这会：
- 随机采样1000条数据
- 分析有幻觉和无幻觉样本的分数分布
- 计算最优阈值（基于F1分数）
- 生成 `threshold_report.txt` 报告

**第二步：使用确定的阈值运行检测**
修改 `bartscore_detector.py` 中的阈值：
```python
process_dataset_bartscore(
    input_file='../data/test_response_label.jsonl',
    output_file='bartscore_results.jsonl',
    threshold=-2.5,  # 使用第一步确定的阈值
    auto_threshold=False
)
```

然后运行：
```bash
python bartscore_detector.py
```

---

### 方法2: 自动确定阈值（一步完成）

直接运行主检测器，它会自动分析阈值：
```bash
cd detectors
python bartscore_detector.py
```

默认配置会：
- 先分析1000条样本确定最优阈值
- 然后使用该阈值对全部数据进行检测

---

### 方法3: 手动指定阈值

如果您已经知道合适的阈值：
```python
process_dataset_bartscore(
    input_file='../data/test_response_label.jsonl',
    output_file='bartscore_results.jsonl',
    threshold=-3.0,  # 手动指定
    auto_threshold=False  # 关闭自动阈值
)
```

---

## 参数说明

### `threshold_optimizer.py`

```python
analyze_score_distribution(
    input_file='test_response_label.jsonl',  # 输入数据文件
    sample_size=1000,                        # 分析样本数量
    model_name='facebook/bart-large-cnn',    # BART模型
    batch_size=4                             # 批处理大小
)
```

### `bartscore_detector.py`

```python
process_dataset_bartscore(
    input_file='test_response_label.jsonl',  # 输入数据文件
    output_file='bartscore_results.jsonl',   # 输出结果文件
    threshold=None,                          # 阈值（None时自动确定）
    model_name='facebook/bart-large-cnn',    # BART模型
    batch_size=4,                            # 批处理大小
    auto_threshold=True,                     # 是否自动确定阈值
    threshold_sample_size=1000               # 阈值分析样本数
)
```

---

## 输出文件

### 1. `threshold_report.txt`
阈值分析报告，包含：
- 有幻觉/无幻觉样本的分数统计
- 推荐的最优阈值
- 预期性能指标

### 2. `bartscore_results.jsonl`
检测结果，每行格式：
```json
{
    "id": "0",
    "task_type": "Summary",
    "has_label": false,
    "label_types": [],
    "bartscore": -2.3456,
    "detected": false
}
```

### 3. `bartscore_results_report.txt`
详细的检测报告，包含：
- 总体性能指标（准确率、召回率、F1分数）
- 按任务类型统计
- 按幻觉类型统计
- BARTScore分数分析

---

## 阈值选择建议

### 阈值的含义
- BARTScore分数越低，表示生成文本与原文一致性越差
- **分数 < 阈值** → 判定为幻觉
- **分数 ≥ 阈值** → 判定为无幻觉

### 阈值调优策略

1. **保守策略（高准确率）**：使用较低的阈值
   - 减少误报，提高准确率
   - 可能漏检一些幻觉，降低召回率

2. **激进策略（高召回率）**：使用较高的阈值
   - 尽可能检测所有幻觉，提高召回率
   - 可能误报较多，降低准确率

3. **平衡策略（最优F1）**：使用自动确定的阈值
   - 平衡准确率和召回率
   - 最大化F1分数

---

## 性能优化建议

1. **使用GPU**：BART模型较大，建议使用GPU加速
   ```bash
   # 确认GPU可用
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **调整批处理大小**：
   - GPU显存充足：`batch_size=8` 或更大
   - GPU显存不足：`batch_size=2` 或 `batch_size=1`
   - CPU运行：`batch_size=1`

3. **减少阈值分析样本数**：
   - 快速测试：`threshold_sample_size=500`
   - 标准分析：`threshold_sample_size=1000`
   - 精确分析：`threshold_sample_size=2000`

---

## 常见问题

**Q1: 显存不足怎么办？**
- 降低 `batch_size` 到 1
- 使用更小的模型：`facebook/bart-base`

**Q2: CPU运行太慢怎么办？**
- 减少 `threshold_sample_size`
- 只分析部分数据
- 考虑使用GPU

**Q3: 检测效果不好怎么办？**
- 调整阈值参数
- 增加阈值分析样本数
- 尝试不同的BART模型

**Q4: 如何选择合适的阈值？**
- 运行 `threshold_optimizer.py` 查看分数分布
- 根据业务需求选择准确率或召回率优先
- 使用自动确定的阈值作为起点

---

## 示例输出

### 阈值分析示例
```
【BARTScore阈值分析】开始分析 1000 条数据
================================================================================

【分数分布分析】
有幻觉样本: 450
无幻觉样本: 550

有幻觉样本分数:
  平均: -4.2341
  中位数: -4.1234
  标准差: 1.2345
  范围: [-8.1234, -1.5678]

无幻觉样本分数:
  平均: -2.1234
  中位数: -2.0123
  标准差: 0.8765
  范围: [-4.5678, -0.1234]

【阈值建议】
方法1 - 均值中点: -3.1788
方法2 - 最优F1: -3.1234 (F1=0.8567)

【最优阈值 -3.1234 的性能】
  准确率 (Precision): 0.8234
  召回率 (Recall): 0.8901
  F1分数: 0.8567
  准确度 (Accuracy): 0.8456
```

### 检测结果示例
```
【BARTScore检测性能摘要】
================================================================================
阈值: -3.1234

准确率 (Precision): 82.34%
召回率 (Recall): 89.01%
F1分数: 0.8567

BARTScore统计:
  平均分数: -3.1245
  有幻觉: -4.2341
  无幻觉: -2.1234
================================================================================
```






