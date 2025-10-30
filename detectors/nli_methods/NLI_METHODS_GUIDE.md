# NLI 方法幻觉检测指南

## 🎯 什么是 NLI 方法？

**NLI (Natural Language Inference)** - 自然语言推理，是一个判断两段文本之间逻辑关系的任务。

### 核心思想

在幻觉检测中，NLI 的应用方式：
```
原文（Source）     →  Premise (前提)
生成文本（Generated）→  Hypothesis (假设)

NLI 模型判断关系：
- Entailment (蕴含)：生成文本可以从原文推导出 → ✅ 无幻觉
- Neutral (中立)：无法判断，需要更多信息  → ⚠️ 可能有幻觉
- Contradiction (矛盾)：生成文本与原文矛盾  → ❌ 有幻觉
```

### 优势
✅ 直接检测矛盾和不一致  
✅ DeBERTa 在 NLI 任务上表现优异  
✅ 可解释性强（知道具体是什么关系）  
✅ 特别适合检测 **Conflict** 类型的幻觉  

### 局限
⚠️ 对 Baseless Info (无根据信息) 检测较弱  
⚠️ 需要明确的逻辑关系  
⚠️ 长文本可能超过模型最大长度 (512 tokens)  

---

## 🚀 快速开始

### 1. 运行检测
```bash
cd /home/xgq/Test/detectors

# 使用默认设置（DeBERTa-v3-large, 阈值0.5）
python nli_deberta_detector.py --gpu 0 --input ../data/test_response_label.jsonl

# 自定义阈值
python nli_deberta_detector.py --gpu 0 --threshold 0.3

# 使用更小的模型（更快但可能不够准确）
python nli_deberta_detector.py --gpu 0 --model microsoft/deberta-v3-base-mnli
```

### 2. 查看结果
```bash
# 查看报告
cat nli_deberta_results_report.txt

# 查看详细结果
head -5 nli_deberta_results.jsonl
```

---

## 🔧 模型选择

### 推荐模型

| 模型 | 大小 | 性能 | 速度 | 推荐度 |
|------|------|------|------|--------|
| **microsoft/deberta-v3-large-mnli** | 1.3GB | ⭐⭐⭐⭐⭐ | 中等 | ⭐⭐⭐⭐⭐ 推荐 |
| microsoft/deberta-v3-base-mnli | 500MB | ⭐⭐⭐⭐ | 快 | ⭐⭐⭐⭐ |
| microsoft/deberta-large-mnli | 1.4GB | ⭐⭐⭐⭐ | 中等 | ⭐⭐⭐ |

### 首次运行
首次运行会自动下载模型（约1.3GB），需要5-10分钟：
```bash
# 会看到
本地无缓存，开始在线下载...
（首次下载约1.3GB，需要几分钟）
Downloading: 100%|████████| 1.3G/1.3G [05:00<00:00]
✓ 在线下载并加载成功！
```

之后运行会直接使用缓存，秒速加载。

---

## 📊 阈值设置

### 什么是阈值？
```
矛盾分数 (contradiction_score) > threshold → 判定为幻觉
```

### 推荐设置

| 阈值 | 效果 | 适用场景 |
|------|------|----------|
| **0.5** | 平衡 | 默认推荐，准确率和召回率均衡 |
| 0.3 | 敏感 | 提高召回率，更多检测，可能误报增加 |
| 0.7 | 保守 | 提高准确率，只检测明确的矛盾 |

### 如何调优？
1. 先用默认 0.5 运行
2. 查看报告中的矛盾分数分布
3. 根据需求调整：
   - 如果漏检多（假阴性高）→ 降低阈值（如 0.3）
   - 如果误报多（假阳性高）→ 提高阈值（如 0.7）

---

## 💡 使用示例

### 示例1：基础使用
```python
from nli_deberta_detector import DeBERTaNLIDetector

# 初始化
detector = DeBERTaNLIDetector(
    model_name='microsoft/deberta-v3-large-mnli',
    gpu_id=0
)

# 检测幻觉
source = "The company was founded in 2010 in San Francisco."
generated = "The company was established in 2015 in New York."

has_hallucination, result = detector.detect_hallucination(source, generated)

print(f"有幻觉: {has_hallucination}")
print(f"矛盾分数: {result['contradiction_score']:.4f}")
print(f"NLI 标签: {result['label']}")
# 输出：
# 有幻觉: True
# 矛盾分数: 0.8756
# NLI 标签: contradiction
```

### 示例2：获取详细分数
```python
result = detector.predict(source, generated)

print("各类别概率:")
for label, score in result['scores'].items():
    print(f"  {label}: {score:.4f}")

# 输出：
# 各类别概率:
#   contradiction: 0.8756
#   neutral: 0.1123
#   entailment: 0.0121
```

### 示例3：批量处理
```python
samples = [
    ("Paris is in France.", "Paris is in Germany."),
    ("The sky is blue.", "The sky appears blue during daytime."),
    ("It rained yesterday.", "The weather was sunny all week.")
]

for source, generated in samples:
    has_hall, result = detector.detect_hallucination(source, generated)
    print(f"原文: {source}")
    print(f"生成: {generated}")
    print(f"判定: {'幻觉' if has_hall else '正常'} ({result['label']})")
    print(f"矛盾分数: {result['contradiction_score']:.4f}\n")
```

---

## 📈 预期性能

基于类似数据集的经验：

### 预期指标
- **准确率**: 60-75%
- **召回率**: 70-85%
- **F1分数**: 65-78

### 不同幻觉类型的表现
| 幻觉类型 | 预期检测率 |
|----------|------------|
| **Evident Conflict** | 85-95% ⭐⭐⭐⭐⭐ |
| **Subtle Conflict** | 65-80% ⭐⭐⭐⭐ |
| Evident Baseless Info | 45-60% ⭐⭐⭐ |
| Subtle Baseless Info | 35-50% ⭐⭐ |

**结论**: NLI 方法特别擅长检测 **矛盾型** 幻觉！

---

## 🆚 与 BARTScore 对比

### BARTScore
- **原理**: 基于生成模型的困惑度
- **优势**: 检测所有类型的不一致
- **劣势**: 可解释性差，阈值难调
- **F1**: 66.26 (原版)

### NLI (DeBERTa)
- **原理**: 直接判断逻辑关系
- **优势**: 
  - ✅ 对矛盾检测非常准确
  - ✅ 可解释性强（知道是entailment/neutral/contradiction）
  - ✅ 阈值语义清晰
- **劣势**: 
  - ⚠️ 对无根据信息检测弱
- **预期F1**: 65-78

### 适用场景对比

| 场景 | BARTScore | NLI-DeBERTa |
|------|-----------|-------------|
| 检测事实矛盾 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 检测无根据信息 | ⭐⭐⭐⭐ | ⭐⭐ |
| 检测所有幻觉 | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| 需要可解释性 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 处理速度 | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 🔍 结果文件说明

### nli_deberta_results.jsonl
每行一个JSON对象：
```json
{
  "id": "12345",
  "task_type": "Summary",
  "has_label": true,
  "label_types": ["Evident Conflict"],
  "nli_label": "contradiction",
  "contradiction_score": 0.8756,
  "entailment_score": 0.0121,
  "neutral_score": 0.1123,
  "detected": true,
  "threshold": 0.5
}
```

### nli_deberta_results_report.txt
详细报告，包含：
- 整体性能指标（准确率、召回率、F1）
- 按任务类型的性能
- 按幻觉类型的检测率
- 矛盾分数分布统计

---

## ⚠️ 注意事项

### 1. 文本长度限制
DeBERTa 最大输入长度为 512 tokens（约400-450词）。超长文本会被截断。

**解决方案**：
- 使用摘要或关键句子
- 分段检测后综合判断

### 2. GPU 内存
- DeBERTa-large 需要约 6GB GPU 内存
- 如果显存不足，使用 base 版本

### 3. 首次运行慢
- 首次需要下载模型（1.3GB）
- 后续运行会使用缓存，快很多

### 4. 阈值需要调优
- 默认 0.5 可能不是最优
- 建议在验证集上优化

---

## 🎓 进阶用法

### 1. 集成方法
结合 BARTScore 和 NLI：
```python
# BARTScore 检测整体一致性
# NLI 检测具体矛盾
if nli_contradiction > 0.5 or bartscore < -2.0:
    # 判定为幻觉
```

### 2. 双向 NLI
```python
# 正向: source → generated
forward = detector.predict(source, generated)

# 反向: generated → source  
backward = detector.predict(generated, source)

# 如果任一方向显示矛盾，则判定为幻觉
if forward['contradiction_score'] > 0.5 or backward['contradiction_score'] > 0.5:
    has_hallucination = True
```

### 3. 句子级检测
对长文本，分句检测：
```python
import nltk
sentences = nltk.sent_tokenize(generated_text)

for sent in sentences:
    result = detector.predict(source, sent)
    if result['contradiction_score'] > 0.5:
        print(f"可能有幻觉的句子: {sent}")
```

---

## 📞 故障排除

### 问题1: 下载模型失败
```bash
# 确认镜像设置
python -c "import os; print(os.environ.get('HF_ENDPOINT'))"

# 应该输出: https://hf-mirror.com
```

### 问题2: GPU 内存不足
```python
# 使用更小的模型
python nli_deberta_detector.py --model microsoft/deberta-v3-base-mnli
```

### 问题3: 结果不理想
- 检查阈值设置
- 查看报告中的分数分布
- 尝试不同的阈值

---

## 📚 参考资料

- [DeBERTa 论文](https://arxiv.org/abs/2006.03654)
- [MNLI 数据集](https://cims.nyu.edu/~sbowman/multinli/)
- [Hugging Face 模型库](https://huggingface.co/microsoft/deberta-v3-large-mnli)

---

## 🚀 下一步

1. **运行检测**
   ```bash
   python nli_deberta_detector.py --gpu 0
   ```

2. **查看报告**
   ```bash
   cat nli_deberta_results_report.txt
   ```

3. **与 BARTScore 对比**
   - 比较 F1 分数
   - 分析不同幻觉类型的检测率
   - 决定使用哪种方法或组合使用

Good luck! 🎉

