# 检索增强 NLI 使用说明

## 文件说明

- **`nli_retrieval_detector.py`** - 新版检索增强 NLI 检测器
  - ✅ 与 `nli_deberta_detector.py` 相同的配置和报告格式
  - ✅ 便于直接对比检索增强的效果
  - ✅ 支持句子级检测 + 最差句聚合

- **`nli_with_retrieval.py`** - 旧版检索增强实现
  - 仅供参考，建议使用新版

## 核心思路

### 标准 NLI vs 检索增强 NLI

#### 标准 NLI（nli_deberta_detector.py）
```
原文（可能很长） + 生成文本 → NLI
问题：长文本会被截断，丢失证据
```

#### 检索增强 NLI（nli_retrieval_detector.py）
```
1. 对生成文本分句
2. 对每个句子，从原文中检索最相关的 top-k 句（SentenceTransformer）
3. 用检索到的证据 + 该句子做 NLI
4. 最差句聚合

优势：
- 避免长文本截断
- 聚焦相关证据
- 理论上可提升准确率
```

## 使用方法

### 在验证集上运行（与标准 NLI 相同配置）

```bash
cd /home/xgq/Test/detectors/nli_methods

python nli_retrieval_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_validation_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold-entail 0.45 \
  --threshold-contra 0.2 \
  --top-k 3
```

**注意**：使用与标准 NLI 相同的阈值（0.45/0.2），便于公平对比。

### 参数说明

```bash
--gpu 0                      # GPU ID
--input <file>               # 输入文件
--output <file>              # 输出文件
--threshold-entail 0.45      # entailment 阈值（与标准NLI相同）
--threshold-contra 0.2       # contradiction 阈值（与标准NLI相同）
--use-entailment             # 使用 entailment 判定
--sentence-level             # 启用句子级检测（推荐）
--top-k 3                    # 检索 top-k 个相关句子（默认3）
--retrieval-model <model>    # 检索模型（默认 all-MiniLM-L6-v2）
```

## 输出文件

### 结果文件
- `nli_retrieval_validation_results.jsonl` - 每个样本的检测结果
  - 包含 entailment/contradiction 分数
  - 包含句子数和幻觉句子数
  - 包含检索 top-k 信息

### 报告文件
- `nli_retrieval_validation_results_report.txt` - 统计报告
  - 格式与 `nli_deberta_detector.py` 生成的报告完全相同
  - 便于直接对比

## 对比实验

### 步骤1：运行标准 NLI（对照组）

```bash
cd /home/xgq/Test/detectors

python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_validation_fixed_cpu.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold-entail 0.45 \
  --threshold-contra 0.2
```

**结果**：
- F1: 62.06
- Precision: 45.70%
- Recall: 96.67%

### 步骤2：运行检索增强 NLI（实验组）

```bash
cd /home/xgq/Test/detectors/nli_methods

python nli_retrieval_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_validation_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold-entail 0.45 \
  --threshold-contra 0.2 \
  --top-k 3
```

### 步骤3：对比结果

对比两个报告文件：
- `nli_validation_fixed_cpu_report.txt`（标准 NLI）
- `nli_retrieval_validation_results_report.txt`（检索增强）

**关注指标**：
- F1 是否提升？
- Precision 是否提升？（检索应该能减少误报）
- Recall 是否保持？
- 各任务类型表现差异

## Top-K 参数调优（可选）

测试不同的 top-k 值：

```bash
# Top-K = 1（最相关的1句）
python nli_retrieval_detector.py --gpu 0 --top-k 1 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_k1.jsonl

# Top-K = 3（默认）
python nli_retrieval_detector.py --gpu 0 --top-k 3 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_k3.jsonl

# Top-K = 5
python nli_retrieval_detector.py --gpu 0 --top-k 5 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_k5.jsonl
```

然后对比各自的报告，找到最优的 top-k 值。

## 预期效果

### 可能的改进
- ✅ **Precision 提升**：聚焦相关证据，减少噪声干扰，降低误报
- ✅ **长文本任务表现更好**：避免截断丢失关键证据
- ✅ **QA 任务可能提升明显**：因为可以精准检索到相关段落

### 可能的问题
- ⚠️ **Recall 可能略降**：如果检索遗漏关键证据，会导致漏检
- ⚠️ **计算时间增加**：增加了 SentenceTransformer 编码和相似度计算
- ⚠️ **检索质量依赖**：如果原文分句不好或相似度计算不准，效果会打折扣

## 运行时间估算

- **标准 NLI**：验证集 2-3 小时
- **检索增强 NLI**：验证集 **3-4 小时**（增加约 30-50%）

额外耗时来源：
1. SentenceTransformer 对原文所有句子编码
2. 对每个生成句子进行相似度计算
3. Top-K 选择和排序

## 依赖安装

```bash
pip install sentence-transformers
```

## 注意事项

1. **相同阈值**：使用与标准 NLI 相同的阈值（0.45/0.2），确保公平对比
2. **相同配置**：句子级检测 + 组合判定
3. **报告格式**：与标准 NLI 完全相同，便于直接比较数字
4. **GPU 内存**：同时加载 NLI 模型和检索模型，需要足够的 GPU 内存（约 4-6GB）

