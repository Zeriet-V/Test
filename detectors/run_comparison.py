"""
BARTScore检测器对比脚本
比较原版和改进版的性能差异
"""

import json
import numpy as np

def load_results(file_path):
    """加载检测结果"""
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            results.append(json.loads(line))
    return results

def calculate_metrics(results):
    """计算性能指标"""
    tp = fp = fn = tn = 0
    
    for r in results:
        has_label = r['has_label']
        detected = r['detected']
        
        if has_label and detected:
            tp += 1
        elif has_label and not detected:
            fn += 1
        elif not has_label and detected:
            fp += 1
        else:
            tn += 1
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) * 100 if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }

def calculate_task_metrics(results):
    """按任务类型计算指标"""
    task_results = {}
    
    for r in results:
        task = r['task_type']
        if task not in task_results:
            task_results[task] = []
        task_results[task].append(r)
    
    task_metrics = {}
    for task, task_data in task_results.items():
        task_metrics[task] = calculate_metrics(task_data)
    
    return task_metrics

def compare_versions(original_file, improved_file):
    """对比两个版本的性能"""
    print("\n" + "=" * 80)
    print("BARTScore检测器版本对比分析")
    print("=" * 80)
    
    # 加载结果
    print("\n正在加载结果...")
    original_results = load_results(original_file)
    improved_results = load_results(improved_file)
    
    print(f"原版结果数: {len(original_results)}")
    print(f"改进版结果数: {len(improved_results)}")
    
    # 计算整体指标
    original_metrics = calculate_metrics(original_results)
    improved_metrics = calculate_metrics(improved_results)
    
    # 打印对比
    print("\n" + "=" * 80)
    print("【整体性能对比】")
    print("=" * 80)
    print(f"\n{'指标':<20} {'原版':<20} {'改进版':<20} {'变化':<20}")
    print("-" * 80)
    
    metrics_names = [
        ('准确率 (Precision)', 'precision', '%'),
        ('召回率 (Recall)', 'recall', '%'),
        ('F1分数', 'f1', ''),
        ('准确度 (Accuracy)', 'accuracy', '%'),
        ('真阳性 (TP)', 'tp', ''),
        ('假阳性 (FP)', 'fp', ''),
        ('假阴性 (FN)', 'fn', ''),
        ('真阴性 (TN)', 'tn', '')
    ]
    
    for name, key, unit in metrics_names:
        orig_val = original_metrics[key]
        impr_val = improved_metrics[key]
        
        if unit == '%':
            diff = impr_val - orig_val
            diff_str = f"{diff:+.2f}{unit}"
            print(f"{name:<20} {orig_val:.2f}{unit:<17} {impr_val:.2f}{unit:<17} {diff_str:<20}")
        else:
            diff = int(impr_val - orig_val)
            diff_str = f"{diff:+d}"
            print(f"{name:<20} {int(orig_val):<20} {int(impr_val):<20} {diff_str:<20}")
    
    # 按任务类型对比
    print("\n" + "=" * 80)
    print("【按任务类型对比】")
    print("=" * 80)
    
    original_task_metrics = calculate_task_metrics(original_results)
    improved_task_metrics = calculate_task_metrics(improved_results)
    
    for task in ['Summary', 'QA', 'Data2txt']:
        if task in original_task_metrics and task in improved_task_metrics:
            print(f"\n◆ {task} 任务:")
            print(f"{'指标':<20} {'原版':<15} {'改进版':<15} {'变化':<15}")
            print("-" * 65)
            
            orig = original_task_metrics[task]
            impr = improved_task_metrics[task]
            
            print(f"{'准确率':<20} {orig['precision']:.2f}%{'':<10} {impr['precision']:.2f}%{'':<10} {impr['precision']-orig['precision']:+.2f}%")
            print(f"{'召回率':<20} {orig['recall']:.2f}%{'':<10} {impr['recall']:.2f}%{'':<10} {impr['recall']-orig['recall']:+.2f}%")
            print(f"{'F1分数':<20} {orig['f1']:.2f}{'':<12} {impr['f1']:.2f}{'':<12} {impr['f1']-orig['f1']:+.2f}")
            print(f"{'假阳性 (FP)':<20} {int(orig['fp']):<15} {int(impr['fp']):<15} {int(impr['fp']-orig['fp']):+d}")
            print(f"{'假阴性 (FN)':<20} {int(orig['fn']):<15} {int(impr['fn']):<15} {int(impr['fn']-orig['fn']):+d}")
    
    # 总结
    print("\n" + "=" * 80)
    print("【改进效果总结】")
    print("=" * 80)
    
    improvements = []
    regressions = []
    
    if improved_metrics['precision'] > original_metrics['precision']:
        improvements.append(f"✓ 准确率提升 {improved_metrics['precision']-original_metrics['precision']:.2f}%")
    else:
        regressions.append(f"✗ 准确率下降 {original_metrics['precision']-improved_metrics['precision']:.2f}%")
    
    if improved_metrics['recall'] > original_metrics['recall']:
        improvements.append(f"✓ 召回率提升 {improved_metrics['recall']-original_metrics['recall']:.2f}%")
    else:
        regressions.append(f"✗ 召回率下降 {original_metrics['recall']-improved_metrics['recall']:.2f}%")
    
    if improved_metrics['f1'] > original_metrics['f1']:
        improvements.append(f"✓ F1分数提升 {improved_metrics['f1']-original_metrics['f1']:.2f}")
    else:
        regressions.append(f"✗ F1分数下降 {original_metrics['f1']-improved_metrics['f1']:.2f}")
    
    if improved_metrics['fp'] < original_metrics['fp']:
        improvements.append(f"✓ 假阳性减少 {int(original_metrics['fp']-improved_metrics['fp'])} 个")
    else:
        regressions.append(f"✗ 假阳性增加 {int(improved_metrics['fp']-original_metrics['fp'])} 个")
    
    if improved_metrics['fn'] < original_metrics['fn']:
        improvements.append(f"✓ 假阴性减少 {int(original_metrics['fn']-improved_metrics['fn'])} 个")
    else:
        regressions.append(f"✗ 假阴性增加 {int(improved_metrics['fn']-original_metrics['fn'])} 个")
    
    if improvements:
        print("\n改进点:")
        for imp in improvements:
            print(f"  {imp}")
    
    if regressions:
        print("\n退步点:")
        for reg in regressions:
            print(f"  {reg}")
    
    # 整体评价
    print("\n整体评价:")
    if improved_metrics['f1'] > original_metrics['f1']:
        print(f"  ✓ 改进版整体性能更好 (F1: {improved_metrics['f1']:.2f} vs {original_metrics['f1']:.2f})")
    else:
        print(f"  ✗ 原版整体性能更好 (F1: {original_metrics['f1']:.2f} vs {improved_metrics['f1']:.2f})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # 对比原版和改进版
    compare_versions(
        original_file='bartscore_results.jsonl',
        improved_file='bartscore_improved_results.jsonl'
    )

