"""
BARTScore阈值优化器
专门用于分析BARTScore分数分布，确定最优阈值
"""

import torch
import json
import numpy as np
from tqdm import tqdm
from bartscore_detector import BARTScorer


def analyze_score_distribution(input_file='test_response_label.jsonl', 
                                sample_size=1000,
                                model_name='facebook/bart-large-cnn',
                                batch_size=4):
    """
    分析BARTScore分数分布，确定最优阈值
    
    :param input_file: 输入数据文件
    :param sample_size: 分析样本数量
    :param model_name: BART模型名称
    :param batch_size: 批处理大小
    :return: (最优阈值, 有幻觉分数列表, 无幻觉分数列表)
    """
    print(f"\n【BARTScore阈值分析】开始分析 {sample_size} 条数据")
    print("=" * 80)
    
    # 初始化BARTScore
    scorer = BARTScorer(model_name=model_name)
    
    hallucination_scores = []
    no_hallucination_scores = []
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        
        # 随机采样
        import random
        random.seed(42)  # 固定随机种子
        sampled_lines = random.sample(lines, min(sample_size, len(lines)))
        
        for line in tqdm(sampled_lines, desc="阈值分析"):
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
            
            try:
                # 计算BARTScore
                score = scorer.score(text_original, text_generated)
                
                if has_label:
                    hallucination_scores.append(score)
                else:
                    no_hallucination_scores.append(score)
                    
            except Exception as e:
                print(f"\n处理错误 (ID: {data.get('id', 'unknown')}): {str(e)}")
                continue
    
    # 分析分数分布
    print(f"\n【分数分布分析】")
    print(f"有幻觉样本: {len(hallucination_scores)}")
    print(f"无幻觉样本: {len(no_hallucination_scores)}")
    
    if not hallucination_scores or not no_hallucination_scores:
        print("警告: 无法分析分数分布，使用默认阈值 -3.0")
        return -3.0, hallucination_scores, no_hallucination_scores
    
    hallucination_mean = np.mean(hallucination_scores)
    hallucination_std = np.std(hallucination_scores)
    hallucination_min = np.min(hallucination_scores)
    hallucination_max = np.max(hallucination_scores)
    hallucination_median = np.median(hallucination_scores)
    
    no_hallucination_mean = np.mean(no_hallucination_scores)
    no_hallucination_std = np.std(no_hallucination_scores)
    no_hallucination_min = np.min(no_hallucination_scores)
    no_hallucination_max = np.max(no_hallucination_scores)
    no_hallucination_median = np.median(no_hallucination_scores)
    
    print(f"\n有幻觉样本分数:")
    print(f"  平均: {hallucination_mean:.4f}")
    print(f"  中位数: {hallucination_median:.4f}")
    print(f"  标准差: {hallucination_std:.4f}")
    print(f"  范围: [{hallucination_min:.4f}, {hallucination_max:.4f}]")
    
    print(f"\n无幻觉样本分数:")
    print(f"  平均: {no_hallucination_mean:.4f}")
    print(f"  中位数: {no_hallucination_median:.4f}")
    print(f"  标准差: {no_hallucination_std:.4f}")
    print(f"  范围: [{no_hallucination_min:.4f}, {no_hallucination_max:.4f}]")
    
    # 计算最优阈值
    best_threshold = find_optimal_threshold(hallucination_scores, no_hallucination_scores)
    
    return best_threshold, hallucination_scores, no_hallucination_scores


def find_optimal_threshold(hallucination_scores, no_hallucination_scores):
    """
    基于F1分数找到最优阈值
    
    :param hallucination_scores: 有幻觉样本的分数列表
    :param no_hallucination_scores: 无幻觉样本的分数列表
    :return: 最优阈值
    """
    hallucination_mean = np.mean(hallucination_scores)
    no_hallucination_mean = np.mean(no_hallucination_scores)
    
    # 方法1: 使用两个分布的均值中点
    threshold_mean = (hallucination_mean + no_hallucination_mean) / 2
    
    # 方法2: 使用ROC曲线的最优点（基于F1分数）
    hallucination_min = np.min(hallucination_scores)
    hallucination_max = np.max(hallucination_scores)
    no_hallucination_min = np.min(no_hallucination_scores)
    no_hallucination_max = np.max(no_hallucination_scores)
    
    thresholds = np.linspace(
        min(hallucination_min, no_hallucination_min), 
        max(hallucination_max, no_hallucination_max), 
        100
    )
    
    best_threshold = threshold_mean
    best_f1 = 0
    best_precision = 0
    best_recall = 0
    
    for thresh in thresholds:
        tp = sum(1 for score in hallucination_scores if score < thresh)
        fp = sum(1 for score in no_hallucination_scores if score < thresh)
        fn = sum(1 for score in hallucination_scores if score >= thresh)
        tn = sum(1 for score in no_hallucination_scores if score >= thresh)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_precision = precision
            best_recall = recall
    
    print(f"\n【阈值建议】")
    print(f"方法1 - 均值中点: {threshold_mean:.4f}")
    print(f"方法2 - 最优F1: {best_threshold:.4f} (F1={best_f1:.4f})")
    
    # 验证最优阈值
    tp = sum(1 for score in hallucination_scores if score < best_threshold)
    fp = sum(1 for score in no_hallucination_scores if score < best_threshold)
    fn = sum(1 for score in hallucination_scores if score >= best_threshold)
    tn = sum(1 for score in no_hallucination_scores if score >= best_threshold)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    
    print(f"\n【最优阈值 {best_threshold:.4f} 的性能】")
    print(f"  准确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall): {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  准确度 (Accuracy): {accuracy:.4f}")
    print(f"  真阳性 (TP): {tp}")
    print(f"  假阳性 (FP): {fp}")
    print(f"  假阴性 (FN): {fn}")
    print(f"  真阴性 (TN): {tn}")
    
    return best_threshold


def save_threshold_report(threshold, hallucination_scores, no_hallucination_scores, 
                         output_file='threshold_report.txt'):
    """
    保存阈值分析报告
    
    :param threshold: 最优阈值
    :param hallucination_scores: 有幻觉样本分数
    :param no_hallucination_scores: 无幻觉样本分数
    :param output_file: 输出报告文件
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BARTScore 阈值分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"分析样本数: {len(hallucination_scores) + len(no_hallucination_scores)}\n")
        f.write(f"有幻觉样本: {len(hallucination_scores)}\n")
        f.write(f"无幻觉样本: {len(no_hallucination_scores)}\n\n")
        
        f.write("【分数统计】\n")
        f.write(f"有幻觉样本:\n")
        f.write(f"  平均: {np.mean(hallucination_scores):.4f}\n")
        f.write(f"  中位数: {np.median(hallucination_scores):.4f}\n")
        f.write(f"  标准差: {np.std(hallucination_scores):.4f}\n")
        f.write(f"  范围: [{np.min(hallucination_scores):.4f}, {np.max(hallucination_scores):.4f}]\n\n")
        
        f.write(f"无幻觉样本:\n")
        f.write(f"  平均: {np.mean(no_hallucination_scores):.4f}\n")
        f.write(f"  中位数: {np.median(no_hallucination_scores):.4f}\n")
        f.write(f"  标准差: {np.std(no_hallucination_scores):.4f}\n")
        f.write(f"  范围: [{np.min(no_hallucination_scores):.4f}, {np.max(no_hallucination_scores):.4f}]\n\n")
        
        f.write("【最优阈值】\n")
        f.write(f"推荐阈值: {threshold:.4f}\n")
        f.write(f"判断规则: BARTScore < {threshold:.4f} 判定为幻觉\n\n")
        
        # 计算该阈值下的性能
        tp = sum(1 for score in hallucination_scores if score < threshold)
        fp = sum(1 for score in no_hallucination_scores if score < threshold)
        fn = sum(1 for score in hallucination_scores if score >= threshold)
        tn = sum(1 for score in no_hallucination_scores if score >= threshold)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
        
        f.write("【预期性能】\n")
        f.write(f"准确率 (Precision): {precision:.2%}\n")
        f.write(f"召回率 (Recall): {recall:.2%}\n")
        f.write(f"F1分数: {f1:.4f}\n")
        f.write(f"准确度 (Accuracy): {accuracy:.2%}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("建议: 在 bartscore_detector.py 中使用 threshold={:.4f}\n".format(threshold))
        f.write("=" * 80 + "\n")
    
    print(f"\n阈值分析报告已保存到: {output_file}")


if __name__ == "__main__":
    print("=" * 80)
    print("BARTScore 阈值优化器")
    print("=" * 80)
    
    # 分析1000条数据确定最优阈值
    optimal_threshold, hall_scores, no_hall_scores = analyze_score_distribution(
        input_file='../data/test_response_label.jsonl',
        sample_size=1000,
        model_name='facebook/bart-large-cnn',
        batch_size=4
    )
    
    # 保存报告
    save_threshold_report(
        threshold=optimal_threshold,
        hallucination_scores=hall_scores,
        no_hallucination_scores=no_hall_scores,
        output_file='threshold_report.txt'
    )
    
    print(f"\n" + "=" * 80)
    print(f"✓ 建议的最优阈值: {optimal_threshold:.4f}")
    print("=" * 80)
    
    print(f"\n【使用建议】")
    print(f"1. 在 bartscore_detector.py 中使用:")
    print(f"   process_dataset_bartscore(threshold={optimal_threshold:.4f})")
    print(f"2. 或者启用自动阈值:")
    print(f"   process_dataset_bartscore(auto_threshold=True)")
    print(f"3. 查看详细报告: threshold_report.txt")

