"""
BARTScore 阈值优化器（改进版）
在验证集上精确搜索最优阈值

使用方法:
1. 先在验证集上运行 BARTScore，保存所有分数
2. 网格搜索最优阈值（步长0.01），最大化 F1 分数
3. 使用最优阈值在测试集上运行
"""

import json
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt


def load_bartscore_results(result_file):
    """
    加载 BARTScore 检测结果
    
    :return: list of dicts
    """
    results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results


def optimize_threshold(results, min_threshold=-4.0, max_threshold=-0.5, step=0.01):
    """
    网格搜索最优阈值
    
    :param results: 检测结果列表
    :param min_threshold: 最小阈值
    :param max_threshold: 最大阈值
    :param step: 步长（默认0.01，和NLI一致）
    :return: best_threshold, best_f1, all_results
    """
    # 提取真实标签和分数
    y_true = [r['has_label'] for r in results]
    scores = [r['bartscore'] for r in results]
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(y_true)}")
    print(f"  有幻觉: {sum(y_true)} ({sum(y_true)/len(y_true)*100:.2f}%)")
    print(f"  无幻觉: {len(y_true) - sum(y_true)} ({(len(y_true)-sum(y_true))/len(y_true)*100:.2f}%)")
    print(f"  BARTScore范围: [{min(scores):.4f}, {max(scores):.4f}]")
    print(f"  平均分数: {np.mean(scores):.4f}")
    
    # 分析有/无幻觉的分数分布
    hall_scores = [s for s, label in zip(scores, y_true) if label]
    no_hall_scores = [s for s, label in zip(scores, y_true) if not label]
    
    print(f"\n分数分布:")
    print(f"  有幻觉样本: 均值={np.mean(hall_scores):.4f}, 标准差={np.std(hall_scores):.4f}")
    print(f"  无幻觉样本: 均值={np.mean(no_hall_scores):.4f}, 标准差={np.std(no_hall_scores):.4f}")
    print(f"  区分度: {abs(np.mean(hall_scores) - np.mean(no_hall_scores)):.4f}")
    
    # 网格搜索
    thresholds = np.arange(min_threshold, max_threshold, step)
    
    best_threshold = None
    best_f1 = 0
    all_results = []
    
    print(f"\n网格搜索阈值 (从 {min_threshold:.2f} 到 {max_threshold:.2f}, 步长 {step})...")
    print(f"  总共测试 {len(thresholds)} 个阈值点")
    
    for threshold in thresholds:
        # BARTScore: 分数越低越可能是幻觉
        y_pred = [score < threshold for score in scores]
        
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
    
    print(f"✓ 搜索完成！")
    
    return best_threshold, best_f1, all_results


def optimize_by_task(results, min_threshold=-4.0, max_threshold=-0.5, step=0.01):
    """
    为每个任务类型单独优化阈值
    
    :return: dict of {task_type: (threshold, f1, results)}
    """
    task_types = set(r['task_type'] for r in results)
    task_optimal_thresholds = {}
    
    print(f"\n" + "=" * 80)
    print("为每个任务类型单独优化阈值")
    print("=" * 80)
    
    for task_type in sorted(task_types):
        print(f"\n优化 {task_type} 任务...")
        task_results = [r for r in results if r['task_type'] == task_type]
        
        if len(task_results) < 10:
            print(f"  样本太少 ({len(task_results)})，跳过")
            continue
        
        threshold, f1, task_all_results = optimize_threshold(
            task_results, min_threshold, max_threshold, step
        )
        
        task_optimal_thresholds[task_type] = {
            'threshold': threshold,
            'f1': f1,
            'all_results': task_all_results,
            'sample_count': len(task_results)
        }
        
        print(f"  ✓ {task_type}: 最优阈值={threshold:.4f}, F1={f1*100:.2f}%")
    
    return task_optimal_thresholds


def plot_threshold_analysis(all_results, best_threshold, output_file='bartscore_threshold_analysis.png'):
    """
    绘制阈值分析图
    """
    thresholds = [r['threshold'] for r in all_results]
    precisions = [r['precision'] for r in all_results]
    recalls = [r['recall'] for r in all_results]
    f1s = [r['f1'] * 100 for r in all_results]
    
    plt.figure(figsize=(14, 6))
    
    # 子图1: 指标曲线
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions, label='Precision', linewidth=2, color='blue')
    plt.plot(thresholds, recalls, label='Recall', linewidth=2, color='green')
    plt.plot(thresholds, f1s, label='F1 Score', linewidth=2.5, linestyle='--', color='red')
    
    # 标记最优点
    best_result = [r for r in all_results if abs(r['threshold'] - best_threshold) < 0.001][0]
    plt.scatter([best_threshold], [best_result['f1'] * 100], 
                color='red', s=150, zorder=5, marker='*',
                label=f'Best: F1={best_result["f1"]*100:.2f}%@{best_threshold:.4f}')
    plt.axvline(x=best_threshold, color='red', linestyle=':', alpha=0.5)
    
    plt.xlabel('BARTScore Threshold', fontsize=12)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Threshold Optimization: BARTScore < threshold → Hallucination', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 子图2: Precision-Recall 曲线
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions, linewidth=2, color='purple')
    plt.scatter([best_result['recall']], [best_result['precision']], 
                color='red', s=150, zorder=5, marker='*',
                label=f'Best Point')
    
    plt.xlabel('Recall (%)', fontsize=12)
    plt.ylabel('Precision (%)', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=13)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 分析图已保存到: {output_file}")


def generate_report(best_threshold, best_f1, all_results, task_thresholds=None,
                    output_file='bartscore_threshold_optimization_report.txt'):
    """
    生成优化报告
    """
    best_result = [r for r in all_results if abs(r['threshold'] - best_threshold) < 0.001][0]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BARTScore 阈值优化报告（改进版）\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【优化策略】\n")
        f.write("  判定标准: BARTScore < threshold → 幻觉\n")
        f.write(f"  搜索范围: 根据数据分布自动确定\n")
        f.write(f"  步长: 0.01 (精细搜索)\n")
        f.write(f"  优化目标: 最大化 F1 分数\n")
        f.write(f"  数据集: 验证集\n\n")
        
        f.write("【统一阈值 - 最优结果】\n")
        f.write(f"  最优阈值: {best_threshold:.4f}\n")
        f.write(f"  F1分数: {best_result['f1']*100:.2f}%\n")
        f.write(f"  准确率: {best_result['precision']:.2f}%\n")
        f.write(f"  召回率: {best_result['recall']:.2f}%\n\n")
        
        f.write(f"  混淆矩阵:\n")
        f.write(f"    真阳性 (TP): {best_result['tp']}\n")
        f.write(f"    假阳性 (FP): {best_result['fp']}\n")
        f.write(f"    假阴性 (FN): {best_result['fn']}\n")
        f.write(f"    真阴性 (TN): {best_result['tn']}\n\n")
        
        # 如果有任务特定阈值
        if task_thresholds:
            f.write("=" * 80 + "\n")
            f.write("【任务特定阈值 - 最优结果】\n")
            f.write("=" * 80 + "\n\n")
            
            for task_type, task_data in sorted(task_thresholds.items()):
                f.write(f"◆ {task_type} 任务:\n")
                f.write(f"  样本数: {task_data['sample_count']}\n")
                f.write(f"  最优阈值: {task_data['threshold']:.4f}\n")
                f.write(f"  F1分数: {task_data['f1']*100:.2f}%\n")
                
                # 找到该任务的最优结果
                task_best = [r for r in task_data['all_results'] 
                            if abs(r['threshold'] - task_data['threshold']) < 0.001][0]
                f.write(f"  准确率: {task_best['precision']:.2f}%\n")
                f.write(f"  召回率: {task_best['recall']:.2f}%\n\n")
        
        # 阈值附近的结果
        f.write("=" * 80 + "\n")
        f.write("【阈值附近的表现】\n")
        f.write("=" * 80 + "\n\n")
        
        idx = next(i for i, r in enumerate(all_results) if abs(r['threshold'] - best_threshold) < 0.001)
        start_idx = max(0, idx - 10)
        end_idx = min(len(all_results), idx + 11)
        
        f.write(f"{'阈值':<12} {'准确率':<12} {'召回率':<12} {'F1分数':<12}\n")
        f.write("-" * 60 + "\n")
        for r in all_results[start_idx:end_idx]:
            marker = " ⭐ 最优" if abs(r['threshold'] - best_threshold) < 0.001 else ""
            f.write(f"{r['threshold']:<12.4f} {r['precision']:<12.2f} {r['recall']:<12.2f} {r['f1']*100:<12.2f}{marker}\n")
        
        f.write("\n")
        
        # Top 10 阈值
        f.write("【Top 10 阈值（按F1排序）】\n")
        sorted_results = sorted(all_results, key=lambda x: x['f1'], reverse=True)[:10]
        f.write(f"{'排名':<6} {'阈值':<12} {'F1分数':<12} {'准确率':<12} {'召回率':<12}\n")
        f.write("-" * 70 + "\n")
        for i, r in enumerate(sorted_results, 1):
            f.write(f"{i:<6} {r['threshold']:<12.4f} {r['f1']*100:<12.2f} {r['precision']:<12.2f} {r['recall']:<12.2f}\n")
        
        f.write("\n")
        
        # 对比旧阈值
        f.write("=" * 80 + "\n")
        f.write("【与原阈值对比】\n")
        f.write("=" * 80 + "\n\n")
        
        old_threshold = -1.8649
        old_result = min(all_results, key=lambda x: abs(x['threshold'] - old_threshold))
        
        f.write(f"原阈值 ({old_threshold:.4f}):\n")
        f.write(f"  F1分数: {old_result['f1']*100:.2f}%\n")
        f.write(f"  准确率: {old_result['precision']:.2f}%\n")
        f.write(f"  召回率: {old_result['recall']:.2f}%\n\n")
        
        f.write(f"新阈值 ({best_threshold:.4f}):\n")
        f.write(f"  F1分数: {best_result['f1']*100:.2f}%\n")
        f.write(f"  准确率: {best_result['precision']:.2f}%\n")
        f.write(f"  召回率: {best_result['recall']:.2f}%\n\n")
        
        f1_improve = (best_result['f1'] - old_result['f1']) * 100
        f.write(f"改进: F1 {'提升' if f1_improve > 0 else '下降'} {abs(f1_improve):.2f}%\n")
        
        f.write("\n")
        f.write("【使用建议】\n")
        f.write(f"  统一阈值: {best_threshold:.4f}\n")
        
        if task_thresholds:
            f.write(f"\n  任务特定阈值:\n")
            for task, data in sorted(task_thresholds.items()):
                f.write(f"    {task}: {data['threshold']:.4f} (F1={data['f1']*100:.2f}%)\n")
        
        f.write(f"\n  测试集运行命令:\n")
        f.write(f"  cd /home/xgq/Test/detectors/bartscore_methods\n")
        f.write(f"  python bartscore_detector.py --gpu 0 \\\n")
        f.write(f"         --input ../../data/test_set.jsonl \\\n")
        f.write(f"         --threshold {best_threshold:.4f}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print(f"✓ 报告已保存到: {output_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BARTScore 阈值优化器（改进版）')
    parser.add_argument('--results', type=str, required=True, 
                        help='BARTScore检测结果文件 (.jsonl)')
    parser.add_argument('--min-threshold', type=float, default=-4.0,
                        help='最小阈值（默认-4.0）')
    parser.add_argument('--max-threshold', type=float, default=-0.5,
                        help='最大阈值（默认-0.5）')
    parser.add_argument('--step', type=float, default=0.01,
                        help='步长（默认0.01）')
    parser.add_argument('--optimize-by-task', action='store_true',
                        help='为每个任务类型单独优化阈值')
    parser.add_argument('--output', type=str, default='bartscore_threshold_opt',
                        help='输出文件前缀')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("BARTScore 阈值优化器（改进版）")
    print("=" * 80)
    print(f"结果文件: {args.results}")
    print(f"优化策略: BARTScore < threshold → 幻觉")
    print(f"步长: {args.step} (精细搜索)")
    print("=" * 80)
    
    # 加载结果
    print("\n加载 BARTScore 检测结果...")
    results = load_bartscore_results(args.results)
    print(f"✓ 加载了 {len(results)} 个样本")
    
    # 优化统一阈值
    print("\n" + "=" * 80)
    print("优化统一阈值")
    print("=" * 80)
    
    best_threshold, best_f1, all_results = optimize_threshold(
        results,
        min_threshold=args.min_threshold,
        max_threshold=args.max_threshold,
        step=args.step
    )
    
    print(f"\n" + "=" * 80)
    print(f"【最优统一阈值】")
    print(f"=" * 80)
    print(f"阈值: {best_threshold:.4f}")
    print(f"F1分数: {best_f1*100:.2f}%")
    
    best_result = [r for r in all_results if abs(r['threshold'] - best_threshold) < 0.001][0]
    print(f"准确率: {best_result['precision']:.2f}%")
    print(f"召回率: {best_result['recall']:.2f}%")
    print("=" * 80)
    
    # 任务特定优化（可选）
    task_thresholds = None
    if args.optimize_by_task:
        task_thresholds = optimize_by_task(
            results,
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            step=args.step
        )
    
    # 生成报告
    report_file = f"{args.output}_report.txt"
    generate_report(best_threshold, best_f1, all_results, task_thresholds, report_file)
    
    # 绘制分析图
    try:
        plot_file = f"{args.output}_analysis.png"
        plot_threshold_analysis(all_results, best_threshold, plot_file)
    except Exception as e:
        print(f"\n⚠ 绘图失败: {str(e)}")
        print("  （不影响优化结果）")
    
    print(f"\n✓ 优化完成！")
    print(f"\n" + "=" * 80)
    print("下一步：使用最优阈值在测试集上运行")
    print("=" * 80)
    print(f"\ncd /home/xgq/Test/detectors/bartscore_methods")
    print(f"python bartscore_detector.py --gpu 0 \\")
    print(f"       --input ../../data/test_set.jsonl \\")
    print(f"       --output bartscore_test_results.jsonl \\")
    print(f"       --threshold {best_threshold:.4f}")

