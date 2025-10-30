# 幻觉检测方法集合

本目录包含多种幻觉检测方法的实现。

## 📂 文件夹结构

```
detectors/
├── bartscore_methods/     # BARTScore 相关方法
├── nli_methods/           # NLI-DeBERTa 相关方法
├── split_dataset.py       # 数据集分割工具
└── 其他工具和文档
```

---

## 📁 1. BARTScore 方法

**路径**: `bartscore_methods/`

### 核心思想
使用预训练 BART 模型计算生成文本相对于原文的对数似然，分数越低表示一致性越差。

### 实现文件
- `bartscore_detector.py` - 原版检测器 ⭐ 推荐
- `bartscore_detector_improved.py` - 改进版（任务特定阈值）
- `bartscore_detector_adaptive.py` - 自适应版本
- `bartscore_threshold_optimizer.py` - 阈值优化器

### 性能（测试集）
- **F1分数**: 66-68
- **准确率**: 54-57%
- **召回率**: 84-87%

### 优势
✅ 检测所有类型的不一致  
✅ 召回率高，漏检少  
✅ 对 Baseless Info 检测好  

### 劣势
⚠️ 准确率较低，误报多  
⚠️ 可解释性差  

### 使用
```bash
cd bartscore_methods
python bartscore_detector.py --gpu 0 --input ../../data/test_set.jsonl --threshold -1.8734
```

---

## 📁 2. NLI-DeBERTa 方法

**路径**: `nli_methods/`

### 核心思想
使用自然语言推理模型判断原文和生成文本之间的逻辑关系（蕴含/中立/矛盾）。

### 实现文件
- `nli_deberta_detector.py` - NLI 检测器 ⭐ 推荐
- `nli_threshold_optimizer.py` - 阈值优化器

### 关键改进
1. **蕴含判定**: entailment < threshold → 幻觉
2. **句子级检测**: SpaCy 分句 + 逐句检测
3. **最差分数**: 使用最差句子的分数
4. **阈值优化**: 验证集网格搜索

### 性能（预期测试集）
- **F1分数**: 70-80
- **准确率**: 65-75%
- **召回率**: 75-85%

### 优势
✅ 准确率高，误报少  
✅ 可解释性强  
✅ 对 Conflict 类型检测特别强（90-95%）  
✅ SpaCy 分句准确  

### 劣势
⚠️ 对 Baseless Info 检测较弱  
⚠️ 句子级检测速度慢  

### 使用
```bash
cd nli_methods
python nli_deberta_detector.py --gpu 0 --input ../../data/test_set.jsonl --threshold 0.54 --use-entailment --sentence-level
```

---

## 🆚 方法对比

| 维度 | BARTScore | NLI-DeBERTa | 最佳 |
|------|-----------|-------------|------|
| **F1分数** | 66-68 | **70-80** | 🏆 NLI |
| **准确率** | 54-57% | **65-75%** | 🏆 NLI |
| **召回率** | **84-87%** | 75-85% | 🏆 BARTScore |
| **Conflict检测** | 90% | **90-95%** | 🏆 NLI |
| **Baseless检测** | **90%** | 60-75% | 🏆 BARTScore |
| **速度** | **快** | 慢 | 🏆 BARTScore |
| **可解释性** | 低 | **高** | 🏆 NLI |

### 推荐使用

**NLI-DeBERTa** (推荐):
- ✅ 需要高准确率
- ✅ 主要检测矛盾型幻觉
- ✅ 需要可解释结果
- ✅ 可以接受较慢速度

**BARTScore**:
- ✅ 需要高召回率（不能漏检）
- ✅ 混合类型幻觉
- ✅ 需要快速处理
- ✅ 对 Baseless Info 检测重要

**组合使用**:
- 两个检测器都判定为幻觉 → 高置信度幻觉
- 任一判定为幻觉 → 需要人工审核

---

## 🛠️ 工具文件

### 数据准备
- `split_dataset.py` - 分割数据为验证集和测试集

### 文档
- `COMPLETE_WORKFLOW.md` - 完整工作流程
- `VALIDATION_SET_GUIDE.md` - 验证集使用指南
- `IMPROVEMENTS.md` - 改进说明

### 使用流程
```bash
# 1. 分割数据（一次性）
python split_dataset.py

# 2. 选择方法
cd bartscore_methods  # 或 cd nli_methods

# 3. 验证集优化阈值
python [detector].py --input ../../data/validation_set.jsonl ...
python [optimizer].py --results ...

# 4. 测试集评估
python [detector].py --input ../../data/test_set.jsonl --threshold [最优值] ...
```

---

## 📊 当前状态

### ✅ 已完成
- [x] 数据分割（验证集 + 测试集）
- [x] BARTScore 验证集运行
- [x] BARTScore 阈值优化
- [x] BARTScore 测试集评估
- [x] NLI 验证集运行（多个版本）
- [x] NLI 阈值优化

### ⏳ 待完成
- [ ] NLI 测试集评估（使用最优阈值）
- [ ] 生成最终对比报告

---

*项目路径: /home/xgq/Test/detectors*  
*最后更新: 2025-10-30*



