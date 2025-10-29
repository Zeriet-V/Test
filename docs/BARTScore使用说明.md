# BARTScore 幻觉检测方法说明

## 方法原理

**BARTScore** 是一个基于BART的文本生成评估指标，可以用来评估生成文本与参考文本的一致性。

### 核心思想
- 使用预训练的BART模型计算生成文本相对于原文的**对数似然**
- **分数越高** = 生成文本与原文越一致 = 越不可能是幻觉
- **分数越低** = 生成文本与原文不一致 = 可能存在幻觉

### 与其他方法的对比

| 方法 | 原理 | 优势 | 劣势 |
|------|------|------|------|
| **SVO** | 句法结构对比 | 可检测逻辑矛盾 | 提取失败率高，召回率极低(3%) |
| **N-gram** | 词汇重叠度 | 召回率高(95.59%) | 无法检测逻辑矛盾 |
| **BARTScore** | 语义一致性评估 | 理解语义，可检测多种幻觉 | 需要GPU，计算较慢 |

### BARTScore的优势

1. **语义理解**：基于预训练语言模型，理解语义而非仅看词汇
2. **端到端**：无需人工特征工程，直接评估一致性
3. **通用性**：可以检测多种类型的幻觉（矛盾、无依据等）
4. **成熟度**：BART是成熟的预训练模型，效果有保证

## 环境要求

### 必需库
```bash
pip install torch transformers tqdm numpy
```

### GPU推荐
- BARTScore使用BART模型，建议使用GPU
- CPU运行会很慢（可能需要数小时处理17790个样本）
- 至少需要8GB显存（使用bart-large-cnn）

### 模型选择
- `facebook/bart-large-cnn`: 推荐，适合摘要任务
- `facebook/bart-large`: 通用模型，更大但效果可能更好
- 首次运行会自动下载模型（约1.6GB）

## 使用方法

### 1. 基本使用

```bash
python bartscore_detector.py
```

### 2. 调整参数

编辑 `bartscore_detector.py` 文件，修改以下参数：

```python
process_dataset_bartscore(
    input_file='test_response_label.jsonl',
    output_file='bartscore_results.jsonl',
    threshold=-3.0,  # 阈值：越低越严格
    model_name='facebook/bart-large-cnn',
    batch_size=4  # 根据显存调整
)
```

### 3. 阈值选择

阈值需要根据实际数据调整，建议步骤：

1. **先运行一次**，查看分数分布
2. **查看报告**中的平均分数：
   - 有幻觉样本的平均分数
   - 无幻觉样本的平均分数
3. **选择合适阈值**：
   - 两者之间的某个值
   - 或使用ROC曲线找最佳阈值

**经验值参考**：
- `-2.0` ~ `-4.0` 是常见范围
- 越低越严格（减少误报，但可能漏检）
- 越高越宽松（提高召回率，但可能误报）

## 输出说明

### 1. 结果文件 (bartscore_results.jsonl)

每行一个JSON对象：
```json
{
  "id": "样本ID",
  "task_type": "任务类型",
  "has_label": true/false,
  "label_types": ["幻觉类型"],
  "bartscore": -3.245,  // BARTScore分数
  "detected": true/false  // 是否检测为幻觉
}
```

### 2. 报告文件 (bartscore_results_report.txt)

包含：
- 总体性能指标（准确率、召回率、F1）
- BARTScore分数统计
- 按任务类型的性能
- 按幻觉类型的检测率

## 预期效果

### 理论优势
相比N-gram和SVO，BARTScore应该能：
- ✓ 检测逻辑矛盾（语义理解）
- ✓ 检测无依据信息（一致性评估）
- ✓ 减少误报（理解改写和同义替换）

### 性能预期
- **准确率**：预期 > 60%（比N-gram的49.45%更高）
- **召回率**：预期 > 80%（可能略低于N-gram的95.59%）
- **F1分数**：预期 > 70（比N-gram的65.18更高）

## 故障排查

### 问题1：显存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：
- 减小batch_size（改为1或2）
- 使用更小的模型
- 使用CPU（会很慢）

### 问题2：模型下载失败
```
OSError: Can't load weights for 'facebook/bart-large-cnn'
```
**解决方案**：
- 检查网络连接
- 使用镜像源：
```python
model_name = 'facebook/bart-large-cnn'
# 或使用国内镜像
```

### 问题3：运行太慢
**解决方案**：
- 确保使用GPU
- 增大batch_size（如果显存够）
- 使用更小的数据集测试

## 下一步

运行完成后，对比三种方法的性能：

```bash
# 创建对比脚本
python compare_all_methods.py
```

对比内容：
- SVO vs N-gram vs BARTScore
- 准确率、召回率、F1分数
- 各自的优势场景

## 进阶：调参建议

### 1. 阈值优化
运行以下脚本找最佳阈值：
```python
# 创建 find_best_threshold.py
# 遍历不同阈值，找F1最高的
```

### 2. 句子级检测
对每个句子单独评分，而非整篇文档：
```python
scorer.score_sentence_level(source_text, generated_text)
```

### 3. 双向评分
同时计算：
- 生成文本 → 原文（是否一致）
- 原文 → 生成文本（是否缺失信息）

## 参考文献

- Yuan et al. (2021). "BARTScore: Evaluating Generated Text as Text Generation"
- BART: Lewis et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
