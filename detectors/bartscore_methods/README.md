# BARTScore 幻觉检测方法

本文件夹包含所有 BARTScore 相关的实现和结果。

## 📁 文件结构

### 核心实现
- `bartscore_detector.py` - **原版 BARTScore** 检测器（推荐使用）
  - 统一阈值 -1.8649
  - 单向评分
  - F1分数: 66.26
  
- `bartscore_detector_improved.py` - **改进版 BARTScore** 检测器
  - 任务特定阈值
  - 双向评分 + 置信度
  - F1分数: 56.24（性能下降）
  
- `bartscore_detector_adaptive.py` - 自适应版本（实验性）

### 结果文件
- `bartscore_improved_results.jsonl` - 改进版检测结果
- `bartscore_improved_results_report.txt` - 改进版详细报告

### 文档
- `BARTSCORE_SUMMARY.md` - **完整对比分析**（推荐阅读）
  - 两版本方法详解
  - 性能对比
  - 阈值来源说明
  - 改进建议

## 🚀 快速使用

### 运行原版（推荐）
```bash
cd /home/xgq/Test/detectors/bartscore_methods
python bartscore_detector.py --gpu 0 --input ../../data/test_response_label.jsonl
```

### 运行改进版
```bash
python bartscore_detector_improved.py --gpu 0 --input ../../data/test_response_label.jsonl
```

## 📊 性能对比

| 版本 | F1分数 | 准确率 | 召回率 | 推荐度 |
|------|--------|--------|--------|--------|
| 原版 | **66.26** | 53.94% | 85.87% | ⭐⭐⭐⭐⭐ |
| 改进版 | 56.24 | 49.98% | 64.29% | ⭐⭐ |

**结论**: 原版性能更好，推荐使用。

## 💡 主要发现

1. **统一阈值** 优于 任务特定阈值（在当前实现下）
2. **双向评分** 没有带来性能提升
3. **阈值必须通过验证集优化**，不能用启发式方法
4. **简单方法更可靠**

详见 `BARTSCORE_SUMMARY.md`

---

*最后更新: 2025-10-30*

