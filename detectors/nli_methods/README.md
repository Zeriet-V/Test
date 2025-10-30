# NLI-DeBERTa 幻觉检测方法

本文件夹包含所有基于 NLI (自然语言推理) 的幻觉检测实现和结果。

## 📁 文件结构

### 核心实现
- `nli_deberta_detector.py` - **NLI-DeBERTa 检测器**
  - 模型: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
  - 改进方法: 蕴含判定 + 句子级检测 + SpaCy分句
  - 最差句子分数聚合

- `nli_threshold_optimizer.py` - 阈值优化工具
  - 网格搜索最优阈值
  - 步长 0.01
  - 最大化 F1 分数

### 结果文件

#### 验证集结果（用于优化阈值）
- `nli_validation_improved.jsonl` - 改进版（句子级）
- `nli_validation_improved_report.txt` - 改进版报告
- `nli_validation_document.jsonl` - 文档级（对比用）
- `nli_validation_document_report.txt` - 文档级报告
- `nli_validation_results.jsonl` - 早期版本

#### 测试集结果（最终评估）
- `nli_test_final_results.jsonl` - 最终检测结果
- （测试集报告待生成）

#### 阈值优化结果
- `nli_threshold_opt_report.txt` - 阈值优化报告 ⭐
- `nli_threshold_opt_analysis.png` - 阈值分析图
- `nli_doc_threshold_opt_report.txt` - 文档级阈值优化

### 文档
- `NLI_METHODS_GUIDE.md` - NLI 方法详细指南
- `NLI_USAGE_GUIDE.md` - 使用指南
- `NLI_完整操作流程.md` - 完整操作流程
- `NLI_关键改进说明.md` - 关键改进说明

## 🚀 快速使用

### 验证集优化（已完成）
```bash
cd /home/xgq/Test/detectors/nli_methods

# 1. 验证集运行
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_validation.jsonl \
  --use-entailment \
  --sentence-level

# 2. 优化阈值
python nli_threshold_optimizer.py \
  --results nli_validation.jsonl \
  --use-entailment
```

### 测试集评估
```bash
# 使用最优阈值（查看优化报告获得）
python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/test_set.jsonl \
  --output nli_test_final.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold [最优值]
```

## 📊 方法特点

### 核心改进（相比 BARTScore）

1. **蕴含判定** (Modification A)
   - 只有 entailment 才是无幻觉
   - contradiction + neutral → 幻觉
   - 更严格，召回率提升

2. **句子级检测** (Modification B)
   - SpaCy 精确分句
   - 逐句检测，避免"抓大放小"
   - 准确率提升 10-15%

3. **最差分数聚合** (Modification B+)
   - 使用最差句子的分数
   - 而非平均分数
   - 分数区分度提升 4-5倍

4. **阈值优化** (Modification C)
   - 验证集网格搜索
   - 步长 0.01
   - 数据驱动，非启发式

### 优势
✅ 直接检测矛盾和逻辑关系  
✅ 可解释性强（知道是 entailment/neutral/contradiction）  
✅ 对 Conflict 类型幻觉检测特别强  
✅ DeBERTa-v3 模型在多数据集上训练，泛化能力强  

### 局限
⚠️ 对 Baseless Info (无根据信息) 检测相对较弱  
⚠️ 句子级检测速度较慢（2-4倍于文档级）  

## 📈 预期性能

| 配置 | F1分数 | 准确率 | 召回率 |
|------|--------|--------|--------|
| 文档级 | 55-65 | 45-55% | 80-90% |
| **句子级（推荐）** | **70-80** | **65-75%** | **75-85%** |

## 🆚 vs BARTScore

| 指标 | BARTScore | NLI-DeBERTa |
|------|-----------|-------------|
| F1分数 | 66.26 | 70-80 |
| 准确率 | 53.94% | 65-75% |
| Conflict检测 | 90% | 90-95% |
| Baseless检测 | 90% | 60-75% |

**适用场景**:
- 主要是矛盾型幻觉 → NLI 更好
- 混合类型幻觉 → BARTScore 更均衡
- 需要可解释性 → NLI 更好

---

*最后更新: 2025-10-30*



