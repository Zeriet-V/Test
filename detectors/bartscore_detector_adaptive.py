"""
BARTScore 自适应阈值检测器
针对不同任务类型使用不同阈值
"""

# 复制 bartscore_detector.py 的内容，但修改阈值部分
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bartscore_detector import BARTScorer, process_dataset_bartscore
import torch

# 任务类型特定阈值
TASK_THRESHOLDS = {
    'Summary': -2.2,     # 降低阈值，减少Summary的高误报率
    'QA': -2.5,          # 大幅降低，QA误报严重
    'Data2txt': -2.0     # Data2txt表现好，略微调整
}

def process_dataset_bartscore_adaptive(input_file='test_response_label.jsonl', 
                                       output_file='bartscore_results_adaptive.jsonl',
                                       model_name='facebook/bart-large-cnn',
                                       batch_size=4):
    """
    使用自适应阈值的BARTScore检测
    根据任务类型自动选择最优阈值
    """
    import json
    from tqdm import tqdm
    import numpy as np
    
    print(f"\n【BARTScore自适应阈值检测器】")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"任务类型特定阈值:")
    for task, thresh in TASK_THRESHOLDS.items():
        print(f"  {task}: {thresh:.4f}")
    print("=" * 80)
    
    # 初始化BARTScore
    scorer = BARTScorer(model_name=model_name)
    
    # 统计数据
    total_count = 0
    has_hallucination_count = 0
    no_hallucination_count = 0
    detected_count = 0
    
    # 性能统计
    stats = {
        'has_label_detected': 0,
        'has_label_not_detected': 0,
        'no_label_detected': 0,
        'no_label_not_detected': 0
    }
    
    # 按任务类型统计
    task_stats = {
        'Summary': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'scores': [], 'threshold': TASK_THRESHOLDS['Summary']},
        'QA': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'scores': [], 'threshold': TASK_THRESHOLDS['QA']},
        'Data2txt': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'scores': [], 'threshold': TASK_THRESHOLDS['Data2txt']}
    }
    
    # 分数统计
    all_scores = []
    hallucination_scores = []
    no_hallucination_scores = []
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        
        for line in tqdm(lines, desc="自适应阈值检测"):
            total_count += 1
            data = json.loads(line)
            
            # 解析数据
            test_data = data.get('test', '')
            task_type = data.get('task_type', 'Unknown')
            
            # 处理不同任务类型
            if isinstance(test_data, dict):
                if task_type == 'QA':
                    question = test_data.get('question', '')
                    passages = test_data.get('passages', '')
                    text_original = f"{question} {passages}"
                elif task_type == 'Data2txt':
                    parts = []
                    if 'name' in test_data: parts.append(f"Name: {test_data['name']}")
                    if 'address' in test_data: parts.append(f"Address: {test_data['address']}")
                    if 'city' in test_data and 'state' in test_data: 
                        parts.append(f"Location: {test_data['city']}, {test_data['state']}")
                    if 'categories' in test_data: parts.append(f"Categories: {test_data['categories']}")
                    if 'business_stars' in test_data: parts.append(f"Rating: {test_data['business_stars']} stars")
                    if 'review_info' in test_data and isinstance(test_data['review_info'], list):
                        for i, review in enumerate(test_data['review_info'][:3], 1):
                            if isinstance(review, dict) and 'review_text' in review:
                                parts.append(f"Review {i}: {review['review_text']}")
                    text_original = ' '.join(parts)
                else:
                    text_original = str(test_data)
            else:
                text_original = test_data
            
            text_generated = data.get('response', '')
            label_types = data.get('label_types', [])
            
            if not isinstance(text_original, str): text_original = str(text_original)
            if not isinstance(text_generated, str): text_generated = str(text_generated)
            
            if not text_original.strip() or not text_generated.strip():
                continue
            
            has_label = len(label_types) > 0
            if has_label:
                has_hallucination_count += 1
            else:
                no_hallucination_count += 1
            
            try:
                # 计算BARTScore
                score = scorer.score(text_original, text_generated)
                
                # 使用任务类型特定阈值
                threshold = TASK_THRESHOLDS.get(task_type, -2.2)
                detected = score < threshold
                
                if detected:
                    detected_count += 1
                
                # 记录分数
                all_scores.append(score)
                if has_label:
                    hallucination_scores.append(score)
                else:
                    no_hallucination_scores.append(score)
                
                # 更新性能统计
                if has_label and detected:
                    stats['has_label_detected'] += 1
                elif has_label and not detected:
                    stats['has_label_not_detected'] += 1
                elif not has_label and detected:
                    stats['no_label_detected'] += 1
                else:
                    stats['no_label_not_detected'] += 1
                
                # 更新任务类型统计
                if task_type in task_stats:
                    task_stats[task_type]['total'] += 1
                    task_stats[task_type]['scores'].append(score)
                    if has_label: task_stats[task_type]['has_label'] += 1
                    if detected: task_stats[task_type]['detected'] += 1
                    if has_label and detected: task_stats[task_type]['true_positive'] += 1
                    elif has_label and not detected: task_stats[task_type]['false_negative'] += 1
                    elif not has_label and detected: task_stats[task_type]['false_positive'] += 1
                
                # 保存结果
                result = {
                    'id': data.get('id', ''),
                    'task_type': task_type,
                    'has_label': has_label,
                    'label_types': label_types,
                    'bartscore': float(score),
                    'threshold_used': float(threshold),
                    'detected': detected
                }
                
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n处理错误 (ID: {data.get('id', 'unknown')}): {str(e)}")
                continue
    
    # 生成报告
    report_file = output_file.replace('.jsonl', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BARTScore自适应阈值检测报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【检测策略】\n")
        f.write("  方法: 自适应阈值BARTScore\n")
        f.write("  任务类型特定阈值:\n")
        for task, thresh in TASK_THRESHOLDS.items():
            f.write(f"    {task}: {thresh:.4f}\n")
        f.write("\n")
        
        # 性能指标
        tp = stats['has_label_detected']
        fp = stats['no_label_detected']
        fn = stats['has_label_not_detected']
        tn = stats['no_label_not_detected']
        
        precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f.write("【整体性能指标】\n")
        f.write(f"  准确率 (Precision): {precision:.2f}%\n")
        f.write(f"  召回率 (Recall): {recall:.2f}%\n")
        f.write(f"  F1分数: {f1:.2f}\n\n")
        
        # 按任务类型统计
        f.write("【按任务类型统计】\n")
        for task, stats_data in task_stats.items():
            if stats_data['total'] > 0:
                task_recall = stats_data['true_positive'] / stats_data['has_label'] * 100 if stats_data['has_label'] > 0 else 0
                task_precision = stats_data['true_positive'] / stats_data['detected'] * 100 if stats_data['detected'] > 0 else 0
                task_f1 = 2 * task_precision * task_recall / (task_precision + task_recall) if (task_precision + task_recall) > 0 else 0
                
                f.write(f"\n{task}任务 (阈值: {stats_data['threshold']:.4f}):\n")
                f.write(f"  准确率: {task_precision:.2f}%\n")
                f.write(f"  召回率: {task_recall:.2f}%\n")
                f.write(f"  F1分数: {task_f1:.2f}\n")
                f.write(f"  TP: {stats_data['true_positive']}, FP: {stats_data['false_positive']}, FN: {stats_data['false_negative']}\n")
    
    print(f"\n自适应阈值检测完成！")
    print(f"准确率: {precision:.2f}%")
    print(f"召回率: {recall:.2f}%")
    print(f"F1分数: {f1:.2f}")
    print(f"\n详细报告: {report_file}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    process_dataset_bartscore_adaptive(
        input_file='../data/test_response_label.jsonl',
        output_file='bartscore_results_adaptive.jsonl',
        model_name='facebook/bart-large-cnn',
        batch_size=4
    )





