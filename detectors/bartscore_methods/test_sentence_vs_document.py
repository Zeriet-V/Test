"""
对比测试：句子级检测 vs 文档级检测
快速在验证集上测试两种方法的性能差异
"""

import os
import sys

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

print("=" * 80)
print("句子级 vs 文档级检测对比测试")
print("=" * 80)

# 参数
validation_file = '/home/xgq/Test/data/validation_set.jsonl'
gpu_id = 0

print(f"\n配置:")
print(f"  验证集: {validation_file}")
print(f"  GPU: {gpu_id}")
print(f"  模型: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
print("=" * 80)

# 测试1: 文档级检测
print("\n【测试1: 文档级检测】")
print("不分句，直接将整个生成文本与原文比较")
print("-" * 80)

output_doc = 'nli_validation_document_level.jsonl'

cmd_doc = f"""python nli_deberta_detector.py \\
  --gpu {gpu_id} \\
  --input {validation_file} \\
  --output {output_doc} \\
  --use-entailment"""

print(f"运行命令:\n{cmd_doc}\n")
print("开始运行...")

import subprocess
result = subprocess.run(cmd_doc, shell=True)

if result.returncode != 0:
    print("\n✗ 文档级检测失败")
    sys.exit(1)

print("✓ 文档级检测完成")

# 测试2: 句子级检测
print("\n" + "=" * 80)
print("【测试2: 句子级检测】")
print("将生成文本分句，逐句检测")
print("-" * 80)

output_sent = 'nli_validation_sentence_level.jsonl'

cmd_sent = f"""python nli_deberta_detector.py \\
  --gpu {gpu_id} \\
  --input {validation_file} \\
  --output {output_sent} \\
  --use-entailment \\
  --sentence-level"""

print(f"运行命令:\n{cmd_sent}\n")
print("开始运行...")

result = subprocess.run(cmd_sent, shell=True)

if result.returncode != 0:
    print("\n✗ 句子级检测失败")
    sys.exit(1)

print("✓ 句子级检测完成")

# 对比结果
print("\n" + "=" * 80)
print("【对比分析】")
print("=" * 80)

# 优化阈值
print("\n优化文档级阈值...")
cmd_opt_doc = f"python nli_threshold_optimizer.py --results {output_doc} --use-entailment --output nli_opt_doc"
subprocess.run(cmd_opt_doc, shell=True, stdout=subprocess.DEVNULL)

print("优化句子级阈值...")
cmd_opt_sent = f"python nli_threshold_optimizer.py --results {output_sent} --use-entailment --output nli_opt_sent"
subprocess.run(cmd_opt_sent, shell=True, stdout=subprocess.DEVNULL)

# 读取结果
import json

def read_optimization_results(report_file):
    """读取优化报告"""
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取关键指标
    import re
    threshold_match = re.search(r'最优阈值:\s*([\d.]+)', content)
    f1_match = re.search(r'F1分数:\s*([\d.]+)%', content)
    precision_match = re.search(r'准确率:\s*([\d.]+)%', content)
    recall_match = re.search(r'召回率:\s*([\d.]+)%', content)
    
    return {
        'threshold': float(threshold_match.group(1)) if threshold_match else None,
        'f1': float(f1_match.group(1)) if f1_match else None,
        'precision': float(precision_match.group(1)) if precision_match else None,
        'recall': float(recall_match.group(1)) if recall_match else None
    }

doc_results = read_optimization_results('nli_opt_doc_report.txt')
sent_results = read_optimization_results('nli_opt_sent_report.txt')

print("\n" + "=" * 80)
print("性能对比（验证集）")
print("=" * 80)

print(f"\n{'指标':<20} {'文档级':<15} {'句子级':<15} {'差异':<15}")
print("-" * 70)

metrics = [
    ('最优阈值', 'threshold', ''),
    ('F1分数', 'f1', '%'),
    ('准确率', 'precision', '%'),
    ('召回率', 'recall', '%')
]

for name, key, unit in metrics:
    doc_val = doc_results[key]
    sent_val = sent_results[key]
    
    if doc_val and sent_val:
        diff = sent_val - doc_val
        diff_str = f"{diff:+.2f}{unit}"
        print(f"{name:<20} {doc_val:.2f}{unit:<12} {sent_val:.2f}{unit:<12} {diff_str:<15}")

print("\n" + "=" * 80)
print("结论")
print("=" * 80)

if sent_results['f1'] and doc_results['f1']:
    if sent_results['f1'] > doc_results['f1']:
        improvement = sent_results['f1'] - doc_results['f1']
        print(f"\n✓ 句子级检测更好！F1提升 {improvement:.2f}%")
        print(f"  推荐使用: 句子级检测（--sentence-level）")
        print(f"  最优阈值: {sent_results['threshold']:.4f}")
    else:
        decline = doc_results['f1'] - sent_results['f1']
        print(f"\n✓ 文档级检测更好！F1高 {decline:.2f}%")
        print(f"  推荐使用: 文档级检测（不加 --sentence-level）")
        print(f"  最优阈值: {doc_results['threshold']:.4f}")

print("\n详细报告:")
print(f"  文档级: nli_opt_doc_report.txt")
print(f"  句子级: nli_opt_sent_report.txt")

print("\n" + "=" * 80)

