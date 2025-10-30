"""
NLI 阈值优化器
在验证集上找到最优阈值（修正C）

使用方法:
1. 先运行一遍验证集，不设阈值，保存所有分数
2. 网格搜索最优阈值，最大化 F1 分数
3. 使用最优阈值在测试集上运行
"""

import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt


def load_results(result_file):
    """
    加载NLI检测结果
    
    :return: list of dicts
    """
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def optimize_threshold(results, score_key='entailment_score', use_entailment=True):
    """
    网格搜索最优阈值
    
    :param results: 检测结果列表
    :param score_key: 使用的分数key
    :param use_entailment: True=基于蕴含分数优化, False=基于矛盾分数优化
    :return: best_threshold, best_f1, all_results
    """
    # 提取真实标签和分数
    y_true = [r['has_label'] for r in results]
    
    if use_entailment:
        scores = [r['entailment_score'] for r in results]
    else:
        scores = [r['contradiction_score'] for r in results]
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(y_true)}")
    print(f"  有幻觉: {sum(y_true)}")
    print(f"  无幻觉: {len(y_true) - sum(y_true)}")
    print(f"  分数范围: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  平均分数: {np.mean(scores):.4f}")
    
    # 网格搜索
    if use_entailment:
        # 蕴含分数：低于阈值判定为幻觉
        thresholds = np.arange(0.05, 0.95, 0.01)
    else:
        # 矛盾分数：高于阈值判定为幻觉
        thresholds = np.arange(0.05, 0.95, 0.01)
    
    best_threshold = 0.5
    best_f1 = 0
    all_results = []
    
    print(f"\n网格搜索阈值 (从 {thresholds[0]:.2f} 到 {thresholds[-1]:.2f}, 步长 0.01)...")
    
    for threshold in thresholds:
        # 根据阈值生成预测
        if use_entailment:
            y_pred = [score < threshold for score in scores]
        else:
            y_pred = [score > threshold for score in scores]
        
        # 计算指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt and not yp)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and not yp)
        
        all_results.append({
            'threshold': threshold,
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, all_results


def plot_threshold_analysis(all_results, best_threshold, output_file='threshold_analysis.png', use_entailment=True):
    """
    绘制阈值分析图
    """
    thresholds = [r['threshold'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1s = [r['f1'] * 100 for r in all_results]  # 转换为百分比
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1s, label='F1 Score', linewidth=2, linestyle='--')
    
    # 标记最优点
    best_result = [r for r in all_results if r['threshold'] == best_threshold][0]
    plt.scatter([best_threshold], [best_result['f1'] * 100], 
                color='red', s=100, zorder=5, label=f'Best F1={best_result["f1"]*100:.2f}%')
    plt.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.5)
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    
    if use_entailment:
        plt.title('Threshold Optimization (Entailment Score < threshold → Hallucination)', fontsize=14)
    else:
        plt.title('Threshold Optimization (Contradiction Score > threshold → Hallucination)', fontsize=14)
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"\n✓ 分析图已保存到: {output_file}")


def generate_report(best_threshold, best_f1, all_results, output_file='threshold_optimization_report.txt', use_entailment=True):
    """
    生成优化报告
    """
    best_result = [r for r in all_results if r['threshold'] == best_threshold][0]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NLI 阈值优化报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【优化策略】\n")
        if use_entailment:
            f.write("  判定标准: entailment_score < threshold → 幻觉\n")
            f.write("  优势: 将 contradiction 和 neutral 都视为幻觉，更严格\n")
        else:
            f.write("  判定标准: contradiction_score > threshold → 幻觉\n")
            f.write("  优势: 只检测明确的矛盾\n")
        f.write(f"  搜索范围: [0.05, 0.95]\n")
        f.write(f"  步长: 0.01\n")
        f.write(f"  优化目标: 最大化 F1 分数\n\n")
        
        f.write("【最优结果】\n")
        f.write(f"  最优阈值: {best_threshold:.4f}\n")
        f.write(f"  F1分数: {best_result['f1']*100:.2f}%\n")
        f.write(f"  准确率: {best_result['precision']:.2f}%\n")
        f.write(f"  召回率: {best_result['recall']:.2f}%\n\n")
        
        f.write(f"  混淆矩阵:\n")
        f.write(f"    真阳性 (TP): {best_result['tp']}\n")
        f.write(f"    假阳性 (FP): {best_result['fp']}\n")
        f.write(f"    假阴性 (FN): {best_result['fn']}\n")
        f.write(f"    真阴性 (TN): {best_result['tn']}\n\n")
        
        # 阈值附近的结果
        f.write("【阈值附近的表现】\n")
        idx = [r['threshold'] for r in all_results].index(best_threshold)
        start_idx = max(0, idx - 5)
        end_idx = min(len(all_results), idx + 6)
        
        f.write(f"{'阈值':<10} {'准确率':<10} {'召回率':<10} {'F1分数':<10}\n")
        f.write("-" * 50 + "\n")
        for r in all_results[start_idx:end_idx]:
            marker = " ⭐" if r['threshold'] == best_threshold else ""
            f.write(f"{r['threshold']:<10.4f} {r['precision']:<10.2f} {r['recall']:<10.2f} {r['f1']*100:<10.2f}{marker}\n")
        
        f.write("\n")
        
        # Top 10 阈值
        f.write("【Top 10 阈值】\n")
        sorted_results = sorted(all_results, key=lambda x: x['f1'], reverse=True)[:10]
        f.write(f"{'排名':<6} {'阈值':<10} {'F1分数':<10} {'准确率':<10} {'召回率':<10}\n")
        f.write("-" * 60 + "\n")
        for i, r in enumerate(sorted_results, 1):
            f.write(f"{i:<6} {r['threshold']:<10.4f} {r['f1']*100:<10.2f} {r['precision']:<10.2f} {r['recall']:<10.2f}\n")
        
        f.write("\n")
        f.write("【使用建议】\n")
        f.write(f"  在测试集上运行时，使用阈值: {best_threshold:.4f}\n\n")
        f.write("  运行命令:\n")
        if use_entailment:
            f.write(f"  python nli_deberta_detector.py --gpu 0 --threshold {best_threshold:.4f} --use-entailment\n")
        else:
            f.write(f"  python nli_deberta_detector.py --gpu 0 --threshold {best_threshold:.4f} --use-contradiction\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ 报告已保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='NLI 阈值优化器')
    parser.add_argument('--results', type=str, required=True, help='NLI检测结果文件 (.jsonl)')
    parser.add_argument('--use-entailment', action='store_true', default=True,
                        help='基于蕴含分数优化（推荐）')
    parser.add_argument('--use-contradiction', dest='use_entailment', action='store_false',
                        help='基于矛盾分数优化')
    parser.add_argument('--output', type=str, default='threshold_optimization', help='输出文件前缀')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("NLI 阈值优化器")
    print("=" * 80)
    print(f"结果文件: {args.results}")
    print(f"优化策略: {'entailment_score < threshold' if args.use_entailment else 'contradiction_score > threshold'}")
    print("=" * 80)
    
    # 加载结果
    print("\n加载检测结果...")
    results = load_results(args.results)
    print(f"✓ 加载了 {len(results)} 个样本")
    
    # 优化阈值
    best_threshold, best_f1, all_results = optimize_threshold(
        results,
        use_entailment=args.use_entailment
    )
    
    print(f"\n" + "=" * 80)
    print(f"【最优结果】")
    print(f"=" * 80)
    print(f"最优阈值: {best_threshold:.4f}")
    print(f"F1分数: {best_f1*100:.2f}%")
    
    best_result = [r for r in all_results if r['threshold'] == best_threshold][0]
    print(f"准确率: {best_result['precision']:.2f}%")
    print(f"召回率: {best_result['recall']:.2f}%")
    print("=" * 80)
    
    # 生成报告
    report_file = f"{args.output}_report.txt"
    generate_report(best_threshold, best_f1, all_results, report_file, args.use_entailment)
    
    # 绘制分析图
    try:
        plot_file = f"{args.output}_analysis.png"
        plot_threshold_analysis(all_results, best_threshold, plot_file, args.use_entailment)
    except Exception as e:
        print(f"\n⚠ 绘图失败: {str(e)}")
        print("  （可能是缺少matplotlib，不影响优化结果）")
    
    print(f"\n✓ 优化完成！")
    print(f"\n下一步：使用最优阈值运行测试集")
    if args.use_entailment:
        print(f"  python nli_deberta_detector.py --gpu 0 --threshold {best_threshold:.4f} --use-entailment --input ../data/test_response_label.jsonl")
    else:
        print(f"  python nli_deberta_detector.py --gpu 0 --threshold {best_threshold:.4f} --use-contradiction --input ../data/test_response_label.jsonl")

