#!/bin/bash
# 快速修复 NLI 准确率低的问题
# 方案1: 改用矛盾判定

echo "============================================================"
echo "NLI 准确率优化 - 方案1: 矛盾判定"
echo "============================================================"
echo ""
echo "当前问题:"
echo "  entailment判定 → 准确率 43%, 误报过多"
echo ""
echo "解决方案:"
echo "  contradiction判定 → 预期准确率 60-75%"
echo ""
echo "============================================================"
echo ""

cd /home/xgq/Test/detectors/nli_methods

# 验证集运行
echo "步骤1: 验证集运行（使用矛盾判定）"
echo "预计时间: 5-10分钟"
echo ""

python nli_deberta_detector.py \
  --gpu 0 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_val_contradiction.jsonl \
  --use-contradiction

if [ $? -ne 0 ]; then
    echo "✗ 运行失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "步骤2: 优化阈值"
echo "============================================================"
echo ""

python nli_threshold_optimizer.py \
  --results nli_val_contradiction.jsonl \
  --use-contradiction \
  --output nli_contradiction_threshold_opt

if [ $? -ne 0 ]; then
    echo "✗ 优化失败"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ 优化完成！"
echo "============================================================"
echo ""
echo "查看结果:"
echo "  cat nli_contradiction_threshold_opt_report.txt"
echo ""
echo "如果准确率提升到 60%+，则在测试集上运行:"
echo "  python nli_deberta_detector.py \\"
echo "    --gpu 0 \\"
echo "    --input ../../data/test_set.jsonl \\"
echo "    --output nli_test_contradiction.jsonl \\"
echo "    --use-contradiction \\"
echo "    --threshold [最优值]"
echo ""


