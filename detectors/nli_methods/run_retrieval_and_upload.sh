#!/bin/bash

# 检索增强 NLI 检测 + 自动上传到 GitHub
# 使用方法: bash run_retrieval_and_upload.sh

set -e  # 遇到错误立即退出

echo "========================================"
echo "检索增强 NLI 检测 + 自动 Git 上传"
echo "========================================"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# 第一步：运行检索增强 NLI 检测
echo "【步骤 1/3】运行检索增强 NLI 检测..."
echo "预计耗时: 3-4 小时"
echo "----------------------------------------"

cd /home/xgq/Test/detectors/nli_methods

python3 nli_retrieval_detector.py \
  --gpu 1 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_validation_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold-entail 0.45 \
  --threshold-contra 0.2 \
  --top-k 3

echo ""
echo "✓ 检测完成！"
echo ""

# 记录检测结束时间
DETECT_END=$(date +%s)
DETECT_TIME=$((DETECT_END - START_TIME))
echo "检测耗时: $((DETECT_TIME / 60)) 分钟"
echo ""

# 第二步：检查结果文件
echo "【步骤 2/3】检查生成的文件..."
echo "----------------------------------------"

if [ -f "nli_retrieval_validation_results.jsonl" ]; then
    echo "✓ nli_retrieval_validation_results.jsonl ($(wc -l < nli_retrieval_validation_results.jsonl) 行)"
else
    echo "✗ 结果文件未生成！"
    exit 1
fi

if [ -f "nli_retrieval_validation_results_report.txt" ]; then
    echo "✓ nli_retrieval_validation_results_report.txt"
    echo ""
    echo "--- 报告摘要 ---"
    grep -E "(F1|Precision|Recall):" nli_retrieval_validation_results_report.txt | head -5
    echo ""
else
    echo "✗ 报告文件未生成！"
    exit 1
fi

# 第三步：上传到 GitHub
echo "【步骤 3/3】上传结果到 GitHub..."
echo "----------------------------------------"

cd /home/xgq/Test

# 检查 git 状态
echo "当前 git 状态:"
git status --short

echo ""
echo "添加文件到 git..."

# 添加结果文件
git add detectors/nli_methods/nli_retrieval_validation_results.jsonl
git add detectors/nli_methods/nli_retrieval_validation_results_report.txt

# 添加新的检测器和文档
git add detectors/nli_methods/nli_retrieval_detector.py
git add detectors/nli_methods/检索增强NLI使用说明.md

# 添加综合报告（如果有修改）
git add docs/BARTScore_NLI_综合报告.md 2>/dev/null || true

# 提交
COMMIT_MSG="添加检索增强NLI验证集结果 - Top-K=3, F1结果见报告"
echo ""
echo "提交信息: $COMMIT_MSG"
git commit -m "$COMMIT_MSG"

# 推送
echo ""
echo "推送到 GitHub..."
git push origin main || git push origin master

# 记录结束时间
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

echo ""
echo "========================================"
echo "✓ 全部完成！"
echo "========================================"
echo "总耗时: $((TOTAL_TIME / 60)) 分钟 ($((TOTAL_TIME / 3600)) 小时)"
echo "检测耗时: $((DETECT_TIME / 60)) 分钟"
echo "上传耗时: $((TOTAL_TIME - DETECT_TIME)) 秒"
echo ""
echo "GitHub 仓库已更新，可在本地查看结果："
echo "  - nli_retrieval_validation_results.jsonl"
echo "  - nli_retrieval_validation_results_report.txt"
echo ""
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

