"""
加权融合幻觉检测器 (Weighted Fusion Detector)
融合 SVO 和 N-gram，使用加权评分机制
策略: 根据两种方法的检测结果计算置信度分数，超过阈值才标记
"""

import spacy
import json
from tqdm import tqdm

# 从 hybrid_detector.py 复用函数
from hybrid_detector import (
    get_phrase, extract_svos, compare_svos,
    get_ngrams, build_ngram_set, check_sentence_novelty,
    has_critical_ngrams, detect_ngram_hallucinations
)

def weighted_fusion_detect(svo_detected, ngram_detected, svo_weight=0.3, ngram_weight=0.7, threshold=0.7):
    """
    加权融合检测
    
    :param svo_detected: SVO是否检测到
    :param ngram_detected: N-gram是否检测到
    :param svo_weight: SVO权重（默认0.3，因为SVO准确但召回率低）
    :param ngram_weight: N-gram权重（默认0.7，因为N-gram召回率高）
    :param threshold: 阈值（默认0.7，超过此分数才标记为幻觉）
    :return: (是否检测到, 置信度分数)
    """
    score = 0.0
    
    if svo_detected:
        score += svo_weight
    
    if ngram_detected:
        score += ngram_weight
    
    detected = score >= threshold
    
    return detected, score

def process_dataset_weighted(input_file='test_response_label.jsonl', 
                             output_file='weighted_fusion_results.jsonl',
                             svo_weight=0.3, 
                             ngram_weight=0.7, 
                             threshold=0.7):
    """
    加权融合检测器
    
    :param svo_weight: SVO权重（0-1），默认0.3
    :param ngram_weight: N-gram权重（0-1），默认0.7
    :param threshold: 检测阈值（0-1），默认0.7
    """
    print(f"\n【加权融合检测器】开始处理数据集: {input_file}")
    print("=" * 80)
    print(f"加权策略: SVO权重={svo_weight}, N-gram权重={ngram_weight}")
    print(f"检测阈值: {threshold} (分数≥{threshold}才标记为幻觉)")
    print("=" * 80)
    print("\n策略说明:")
    print(f"  - 仅N-gram检测到: 分数={ngram_weight} → {'检测到' if ngram_weight >= threshold else '未达标'}")
    print(f"  - 仅SVO检测到: 分数={svo_weight} → {'检测到' if svo_weight >= threshold else '未达标'}")
    print(f"  - 两者都检测到: 分数={svo_weight + ngram_weight} → {'检测到' if (svo_weight + ngram_weight) >= threshold else '未达标'}")
    print(f"  - 两者都未检测到: 分数=0.0 → 未检测到")
    print("=" * 80)
    
    # 加载spaCy模型
    print("\n加载 spaCy 英文模型...")
    try:
        nlp = spacy.load("en_core_web_lg")
        print("spaCy English model 'en_core_web_lg' loaded successfully.")
    except OSError:
        print("Error: 'en_core_web_lg' model not found.")
        exit()
    
    # 统计数据
    total_count = 0
    has_hallucination_count = 0
    no_hallucination_count = 0
    
    # 检测统计
    weighted_detected = 0
    svo_only_detected = 0       # 仅SVO达标
    ngram_only_detected = 0     # 仅N-gram达标
    both_detected = 0           # 两者都检测到（高置信度）
    
    # 分数分布统计
    score_distribution = {
        '0.0': 0,           # 两者都未检测到
        f'{svo_weight}': 0, # 仅SVO
        f'{ngram_weight}': 0, # 仅N-gram
        f'{svo_weight + ngram_weight}': 0  # 两者都有
    }
    
    # 性能统计
    stats = {
        'has_label_detected': 0,
        'has_label_not_detected': 0,
        'no_label_detected': 0,
        'no_label_not_detected': 0
    }
    
    # 按任务类型统计
    task_stats = {
        'Summary': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0},
        'QA': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0},
        'Data2txt': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0}
    }
    
    # 按幻觉标签类型统计
    label_stats = {
        'Evident Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Evident Baseless Info': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Baseless Info': {'total': 0, 'detected': 0, 'samples': []}
    }
    
    # N-gram参数
    NGRAM_N_VALUES = [2]
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        
        for line in tqdm(lines, desc="加权融合检测"):
            total_count += 1
            data = json.loads(line)
            
            # 解析数据（与hybrid_detector相同）
            test_data = data.get('test', '')
            task_type = data.get('task_type', 'Unknown')
            
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
                doc_original = nlp(text_original)
                doc_generated = nlp(text_generated)
                
                # ============ SVO 检测 ============
                svos_original = extract_svos(doc_original)
                svos_generated = extract_svos(doc_generated)
                contradictions = compare_svos(svos_original, svos_generated)
                svo_detected = len(contradictions) > 0
                
                # ============ N-gram 检测 ============
                if task_type == 'Summary':
                    NGRAM_THRESHOLD = 0.65
                elif task_type == 'QA':
                    NGRAM_THRESHOLD = 0.55
                elif task_type == 'Data2txt':
                    NGRAM_THRESHOLD = 0.45
                else:
                    NGRAM_THRESHOLD = 0.55
                
                ground_truth_set = build_ngram_set(doc_original, 
                                                   n_values=NGRAM_N_VALUES, 
                                                   use_lemma=True, 
                                                   filter_stop=True, 
                                                   filter_punct=True)
                
                detection_results = detect_ngram_hallucinations(
                    doc_generated, 
                    ground_truth_set, 
                    threshold=NGRAM_THRESHOLD, 
                    nlp=nlp,
                    n_values=NGRAM_N_VALUES, 
                    use_lemma=True, 
                    filter_stop=True, 
                    filter_punct=True
                )
                
                ngram_detected = detection_results['detected']
                
                # ============ 加权融合 ============
                detected, confidence_score = weighted_fusion_detect(
                    svo_detected, 
                    ngram_detected, 
                    svo_weight, 
                    ngram_weight, 
                    threshold
                )
                
                # 统计分数分布
                score_key = f'{confidence_score:.1f}'
                if score_key in score_distribution:
                    score_distribution[score_key] += 1
                
                # 统计检测来源
                if detected:
                    weighted_detected += 1
                    if svo_detected and ngram_detected:
                        both_detected += 1
                    elif svo_detected:
                        svo_only_detected += 1
                    elif ngram_detected:
                        ngram_only_detected += 1
                
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
                    if has_label: task_stats[task_type]['has_label'] += 1
                    if detected: task_stats[task_type]['detected'] += 1
                    if has_label and detected: task_stats[task_type]['true_positive'] += 1
                    elif has_label and not detected: task_stats[task_type]['false_negative'] += 1
                    elif not has_label and detected: task_stats[task_type]['false_positive'] += 1
                
                # 更新幻觉标签类型统计
                for label_type in label_types:
                    if label_type in label_stats:
                        label_stats[label_type]['total'] += 1
                        if detected:
                            label_stats[label_type]['detected'] += 1
                        if len(label_stats[label_type]['samples']) < 5:
                            label_stats[label_type]['samples'].append(data.get('id', ''))
                
                # 保存结果
                result = {
                    'id': data.get('id', ''),
                    'task_type': task_type,
                    'has_label': has_label,
                    'label_types': label_types,
                    'svo_detected': svo_detected,
                    'ngram_detected': ngram_detected,
                    'confidence_score': confidence_score,
                    'weighted_detected': detected,
                    'svo_contradictions': contradictions,
                    'ngram_details': detection_results['details']
                }
                
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n处理错误: {str(e)}")
                continue
    
    # ============ 生成报告 ============
    report_file = output_file.replace('.jsonl', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("加权融合幻觉检测详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【检测策略】\n")
        f.write("  融合方法: SVO (逻辑矛盾) + N-gram (无依据信息)\n")
        f.write("  融合逻辑: 加权评分 - 根据置信度分数判断\n")
        f.write(f"  权重配置: SVO={svo_weight}, N-gram={ngram_weight}\n")
        f.write(f"  检测阈值: {threshold} (分数≥{threshold}才标记为幻觉)\n")
        f.write("  N-gram配置: 2-gram + 自适应阈值 + NER二次校验\n\n")
        
        f.write("【评分规则】\n")
        f.write(f"  仅N-gram检测到: {ngram_weight} → {'✓ 达标' if ngram_weight >= threshold else '✗ 未达标'}\n")
        f.write(f"  仅SVO检测到: {svo_weight} → {'✓ 达标' if svo_weight >= threshold else '✗ 未达标'}\n")
        f.write(f"  两者都检测到: {svo_weight + ngram_weight} → ✓ 高置信度\n")
        f.write(f"  两者都未检测到: 0.0 → ✗ 未检测到\n\n")
        
        f.write("【总体统计】\n")
        f.write(f"  总数据量: {total_count}\n")
        f.write(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 加权检测到: {weighted_detected} ({weighted_detected/total_count*100:.2f}%)\n\n")
        
        f.write("【检测来源分析】\n")
        f.write(f"  两者都检测到（高置信度）: {both_detected}\n")
        f.write(f"  仅N-gram达标: {ngram_only_detected}\n")
        f.write(f"  仅SVO达标: {svo_only_detected}\n")
        f.write(f"  总计检测: {weighted_detected}\n\n")
        
        f.write("【置信度分数分布】\n")
        for score, count in sorted(score_distribution.items(), key=lambda x: float(x[0])):
            f.write(f"  分数 {score}: {count} 个样本\n")
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
        f.write(f"  ✓ 真阳性 (TP): {tp}\n")
        f.write(f"  ✗ 假阴性 (FN): {fn}\n")
        f.write(f"  ✗ 假阳性 (FP): {fp}\n")
        f.write(f"  ✓ 真阴性 (TN): {tn}\n\n")
        f.write(f"  准确率 (Precision): {precision:.2f}%\n")
        f.write(f"  召回率 (Recall): {recall:.2f}%\n")
        f.write(f"  F1分数: {f1:.2f}\n\n")
        
        # 按任务类型统计
        f.write("=" * 80 + "\n")
        f.write("【按任务类型统计】\n")
        f.write("=" * 80 + "\n\n")
        
        for task, stats_data in task_stats.items():
            if stats_data['total'] > 0:
                task_recall = stats_data['true_positive'] / stats_data['has_label'] * 100 if stats_data['has_label'] > 0 else 0
                task_precision = stats_data['true_positive'] / stats_data['detected'] * 100 if stats_data['detected'] > 0 else 0
                
                f.write(f"◆ {task} 任务:\n")
                f.write(f"  总数: {stats_data['total']}\n")
                f.write(f"  - 有幻觉数据: {stats_data['has_label']} ({stats_data['has_label']/stats_data['total']*100:.2f}%)\n")
                f.write(f"  - 检测到幻觉: {stats_data['detected']} ({stats_data['detected']/stats_data['total']*100:.2f}%)\n\n")
                f.write(f"  性能表现:\n")
                f.write(f"    ✓ 成功检测 (TP): {stats_data['true_positive']}\n")
                f.write(f"    ✗ 漏检 (FN): {stats_data['false_negative']}\n")
                f.write(f"    ✗ 误报 (FP): {stats_data['false_positive']}\n")
                f.write(f"    召回率: {task_recall:.2f}%\n")
                f.write(f"    准确率: {task_precision:.2f}%\n\n")
        
        # 按幻觉类型统计
        f.write("=" * 80 + "\n")
        f.write("【按幻觉类型统计】\n")
        f.write("=" * 80 + "\n\n")
        
        for label_type, label_data in label_stats.items():
            if label_data['total'] > 0:
                detection_rate = label_data['detected'] / label_data['total'] * 100
                miss_count = label_data['total'] - label_data['detected']
                miss_rate = 100 - detection_rate
                
                f.write(f"◆ {label_type}:\n")
                f.write(f"  总数: {label_data['total']}\n")
                f.write(f"  检测到: {label_data['detected']} ({detection_rate:.2f}%)\n")
                f.write(f"  漏检: {miss_count} ({miss_rate:.2f}%)\n")
                f.write(f"  状态: {'✓ 检测效果好' if detection_rate >= 80 else '✗ 需要改进'}\n")
                f.write(f"  样本ID (前5个): {label_data['samples']}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"结果已保存到: {output_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    
    # 打印性能摘要
    print("\n" + "=" * 80)
    print("【加权融合性能摘要】")
    print("=" * 80)
    print(f"权重配置: SVO={svo_weight}, N-gram={ngram_weight}, 阈值={threshold}")
    print(f"\n准确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall): {recall:.2f}%")
    print(f"F1分数: {f1:.2f}")
    print(f"\n检测分布:")
    print(f"  - 高置信度（两者都检测到）: {both_detected}")
    print(f"  - 仅N-gram达标: {ngram_only_detected}")
    print(f"  - 仅SVO达标: {svo_only_detected}")
    print("=" * 80)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total_count,
        'detected': weighted_detected
    }


if __name__ == "__main__":
    print("\n【加权融合检测器】")
    print("=" * 80)
    print("推荐配置：")
    print("  方案1（平衡）: SVO=0.3, N-gram=0.7, 阈值=0.7")
    print("    → N-gram为主，SVO辅助提升置信度")
    print("  方案2（保守）: SVO=0.4, N-gram=0.6, 阈值=0.8")
    print("    → 要求两者都检测到才标记，提高准确率")
    print("  方案3（激进）: SVO=0.2, N-gram=0.8, 阈值=0.6")
    print("    → N-gram主导，SVO仅作参考")
    print("=" * 80)
    
    # 使用推荐的平衡配置
    process_dataset_weighted(
        input_file='test_response_label.jsonl',
        output_file='weighted_fusion_results.jsonl',
        svo_weight=0.3,
        ngram_weight=0.7,
        threshold=0.7
    )
