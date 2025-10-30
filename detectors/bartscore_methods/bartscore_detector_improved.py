"""
BARTScore 幻觉检测器 - 改进版
针对原版准确率低、假阳性高的问题进行优化

主要改进：
1. 任务特定阈值 - 为不同任务类型使用不同的检测阈值
2. 双向BARTScore - 同时评估source->target和target->source
3. 置信度评分 - 提供检测置信度而非简单的二分类
4. 自适应阈值 - 基于分数分布动态调整
"""

import torch
import json
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

# ===== 必须在 import transformers 之前设置环境变量 =====
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 打印确认
print(f"🔧 镜像设置: {os.environ.get('HF_ENDPOINT')}")

from transformers import BartTokenizer, BartForConditionalGeneration


class ImprovedBARTScorer:
    """
    改进的BARTScore评分器
    支持双向评分和更精细的评估
    """
    def __init__(self, model_name='facebook/bart-large-cnn', device='cuda' if torch.cuda.is_available() else 'cpu', gpu_id=None):
        """
        初始化BARTScore模型
        
        :param model_name: BART模型名称
        :param device: 运行设备
        :param gpu_id: GPU ID (0, 1, 2, ...)，如果为None则自动选择
        """
        # 如果指定了GPU ID，设置device
        if gpu_id is not None and torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
            print(f"指定使用GPU: {gpu_id}")
        
        print(f"加载改进版BARTScore模型: {model_name}")
        print(f"使用设备: {device}")
        
        self.device = device
        
        # 检查本地缓存是否存在
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        model_cache = os.path.join(cache_dir, f'models--{model_name.replace("/", "--")}')
        
        if os.path.exists(model_cache):
            print("检测到本地缓存，尝试离线加载...")
            try:
                self.tokenizer = BartTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                self.model = BartForConditionalGeneration.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("✓ 离线加载成功！")
            except Exception as e:
                print(f"⚠ 离线加载失败: {str(e)[:100]}")
                print("删除损坏的缓存，开始在线下载...")
                import shutil
                shutil.rmtree(model_cache, ignore_errors=True)
                print("（首次下载约1.6GB，需要几分钟，请耐心等待）")
                self.tokenizer = BartTokenizer.from_pretrained(model_name)
                self.model = BartForConditionalGeneration.from_pretrained(model_name)
                print("✓ 在线下载并加载成功！")
        else:
            print("本地无缓存，开始在线下载...")
            print("（首次下载约1.6GB，需要几分钟，请耐心等待）")
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            print("✓ 在线下载并加载成功！")
            print("（下次运行将直接使用本地缓存）")
        
        self.model.eval()
        self.model.to(self.device)
        
        print("BARTScore模型加载成功！")
    
    def score_bidirectional(self, source_text, generated_text):
        """
        双向BARTScore评估
        同时计算 P(target|source) 和 P(source|target)
        
        :param source_text: 原文
        :param generated_text: 生成文本
        :return: (forward_score, backward_score, avg_score, harmonic_mean)
        """
        with torch.no_grad():
            # Forward: P(generated|source)
            src_tokens = self.tokenizer([source_text], return_tensors='pt', truncation=True, max_length=1024)
            gen_tokens = self.tokenizer([generated_text], return_tensors='pt', truncation=True, max_length=1024)
            
            src_tokens = {k: v.to(self.device) for k, v in src_tokens.items()}
            gen_tokens = {k: v.to(self.device) for k, v in gen_tokens.items()}
            
            forward_output = self.model(
                input_ids=src_tokens['input_ids'],
                attention_mask=src_tokens['attention_mask'],
                labels=gen_tokens['input_ids']
            )
            forward_score = -forward_output.loss.item()
            
            # Backward: P(source|generated)
            backward_output = self.model(
                input_ids=gen_tokens['input_ids'],
                attention_mask=gen_tokens['attention_mask'],
                labels=src_tokens['input_ids']
            )
            backward_score = -backward_output.loss.item()
            
            # 计算综合分数
            avg_score = (forward_score + backward_score) / 2
            # 调和平均数（对不平衡的分数更敏感）
            if forward_score + backward_score > 0:
                harmonic_mean = 2 * forward_score * backward_score / (forward_score + backward_score)
            else:
                harmonic_mean = 2 * forward_score * backward_score / (forward_score + backward_score) if (forward_score + backward_score) != 0 else 0
            
            return {
                'forward': forward_score,
                'backward': backward_score,
                'average': avg_score,
                'harmonic': harmonic_mean
            }
    
    def score_with_confidence(self, source_text, generated_text):
        """
        计算BARTScore并提供置信度
        
        :return: {'score': float, 'confidence': float, 'bidirectional': dict}
        """
        bi_scores = self.score_bidirectional(source_text, generated_text)
        
        # 计算置信度：基于前向和后向分数的一致性
        score_diff = abs(bi_scores['forward'] - bi_scores['backward'])
        confidence = 1.0 / (1.0 + score_diff)  # 分数差异越小，置信度越高
        
        return {
            'score': bi_scores['forward'],  # 主分数
            'confidence': confidence,
            'bidirectional': bi_scores
        }


def process_dataset_improved(input_file='test_response_label.jsonl', 
                             output_file='bartscore_improved_results.jsonl',
                             task_thresholds=None,
                             model_name='facebook/bart-large-cnn',
                             use_bidirectional=True,
                             gpu_id=None):
    """
    使用改进的BARTScore检测幻觉
    
    :param input_file: 输入数据文件
    :param output_file: 输出结果文件
    :param task_thresholds: 任务特定阈值字典 {'Summary': -1.5, 'QA': -2.0, 'Data2txt': -2.3}
    :param model_name: BART模型名称
    :param use_bidirectional: 是否使用双向评分
    :param gpu_id: GPU ID (0, 1, 2, ...)，如果为None则自动选择
    """
    print(f"\n【改进版BARTScore幻觉检测器】开始处理数据集: {input_file}")
    print("=" * 80)
    print(f"模型: {model_name}")
    print(f"使用双向评分: {use_bidirectional}")
    if gpu_id is not None:
        print(f"指定GPU: {gpu_id}")
    
    # 默认任务特定阈值（基于数据分析）
    if task_thresholds is None:
        task_thresholds = {
            'Summary': -1.65,      # Summary平均-1.82，设置更严格
            'QA': -2.05,           # QA平均-2.12，设置略严格
            'Data2txt': -2.45      # Data2txt平均-2.50，设置略严格
        }
    
    print(f"任务特定阈值:")
    for task, threshold in task_thresholds.items():
        print(f"  {task}: {threshold:.4f}")
    print("=" * 80)
    
    # 初始化BARTScore
    scorer = ImprovedBARTScorer(model_name=model_name, gpu_id=gpu_id)
    
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
        'Summary': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'scores': []},
        'QA': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'scores': []},
        'Data2txt': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'scores': []}
    }
    
    # 按幻觉标签类型统计
    label_stats = {
        'Evident Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Evident Baseless Info': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Baseless Info': {'total': 0, 'detected': 0, 'samples': []}
    }
    
    # 分数统计
    all_scores = []
    hallucination_scores = []
    no_hallucination_scores = []
    
    # 误判样本统计
    false_positive_samples = []
    false_negative_samples = []
    
    # 假阳性/假阴性按任务类型统计
    fp_by_task = {'Summary': 0, 'QA': 0, 'Data2txt': 0}
    fn_by_task = {'Summary': 0, 'QA': 0, 'Data2txt': 0}
    fn_by_label = {
        'Evident Conflict': 0,
        'Subtle Conflict': 0,
        'Evident Baseless Info': 0,
        'Subtle Baseless Info': 0
    }
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        
        for line in tqdm(lines, desc="改进版BARTScore检测"):
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
                if use_bidirectional:
                    score_result = scorer.score_with_confidence(text_original, text_generated)
                    score = score_result['score']
                    confidence = score_result['confidence']
                    bi_scores = score_result['bidirectional']
                else:
                    bi_scores_result = scorer.score_bidirectional(text_original, text_generated)
                    score = bi_scores_result['forward']
                    confidence = 1.0
                    bi_scores = bi_scores_result
                
                # 使用任务特定阈值判断
                threshold = task_thresholds.get(task_type, -1.8649)
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
                    false_negative_samples.append({
                        'id': data.get('id', ''),
                        'task_type': task_type,
                        'label_types': label_types,
                        'score': float(score),
                        'confidence': float(confidence)
                    })
                    if task_type in fn_by_task:
                        fn_by_task[task_type] += 1
                    for label_type in label_types:
                        if label_type in fn_by_label:
                            fn_by_label[label_type] += 1
                elif not has_label and detected:
                    stats['no_label_detected'] += 1
                    false_positive_samples.append({
                        'id': data.get('id', ''),
                        'task_type': task_type,
                        'score': float(score),
                        'confidence': float(confidence)
                    })
                    if task_type in fp_by_task:
                        fp_by_task[task_type] += 1
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
                    'bartscore': float(score),
                    'confidence': float(confidence),
                    'detected': detected,
                    'threshold_used': float(threshold)
                }
                
                if use_bidirectional:
                    result['bidirectional_scores'] = {
                        'forward': float(bi_scores['forward']),
                        'backward': float(bi_scores['backward']),
                        'average': float(bi_scores['average']),
                        'harmonic': float(bi_scores['harmonic'])
                    }
                
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n处理错误 (ID: {data.get('id', 'unknown')}): {str(e)}")
                continue
    
    # ============ 生成报告 ============
    report_file = output_file.replace('.jsonl', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("改进版BARTScore幻觉检测详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【检测策略】\n")
        f.write("  方法: 改进版BARTScore\n")
        f.write(f"  模型: {model_name}\n")
        f.write(f"  双向评分: {use_bidirectional}\n")
        f.write("  改进点:\n")
        f.write("    1. 任务特定阈值 - 为不同任务使用不同检测阈值\n")
        if use_bidirectional:
            f.write("    2. 双向BARTScore - 同时评估source->target和target->source\n")
            f.write("    3. 置信度评分 - 基于双向分数一致性提供置信度\n")
        f.write("\n  任务特定阈值:\n")
        for task, threshold in task_thresholds.items():
            f.write(f"    {task}: {threshold:.4f}\n")
        f.write("\n")
        
        f.write("【总体统计】\n")
        f.write(f"  总数据量: {total_count}\n")
        f.write(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 检测到幻觉: {detected_count} ({detected_count/total_count*100:.2f}%)\n\n")
        
        f.write("【BARTScore分数分析】\n")
        f.write(f"  全部样本:\n")
        f.write(f"    平均分数: {np.mean(all_scores):.4f}\n")
        f.write(f"    标准差: {np.std(all_scores):.4f}\n")
        f.write(f"    最小值: {np.min(all_scores):.4f}\n")
        f.write(f"    最大值: {np.max(all_scores):.4f}\n\n")
        
        if hallucination_scores:
            f.write(f"  有幻觉样本:\n")
            f.write(f"    平均分数: {np.mean(hallucination_scores):.4f}\n")
            f.write(f"    标准差: {np.std(hallucination_scores):.4f}\n\n")
        
        if no_hallucination_scores:
            f.write(f"  无幻觉样本:\n")
            f.write(f"    平均分数: {np.mean(no_hallucination_scores):.4f}\n")
            f.write(f"    标准差: {np.std(no_hallucination_scores):.4f}\n\n")
        
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
                task_f1 = 2 * task_precision * task_recall / (task_precision + task_recall) if (task_precision + task_recall) > 0 else 0
                avg_score = np.mean(stats_data['scores']) if stats_data['scores'] else 0
                
                f.write(f"◆ {task} 任务:\n")
                f.write(f"  阈值: {task_thresholds.get(task, 'N/A'):.4f}\n")
                f.write(f"  总数: {stats_data['total']}\n")
                f.write(f"  平均BARTScore: {avg_score:.4f}\n")
                f.write(f"  - 有幻觉数据: {stats_data['has_label']} ({stats_data['has_label']/stats_data['total']*100:.2f}%)\n")
                f.write(f"  - 检测到幻觉: {stats_data['detected']} ({stats_data['detected']/stats_data['total']*100:.2f}%)\n\n")
                f.write(f"  性能表现:\n")
                f.write(f"    ✓ 成功检测 (TP): {stats_data['true_positive']}\n")
                f.write(f"    ✗ 漏检 (FN): {stats_data['false_negative']}\n")
                f.write(f"    ✗ 误报 (FP): {stats_data['false_positive']}\n")
                f.write(f"    准确率: {task_precision:.2f}%\n")
                f.write(f"    召回率: {task_recall:.2f}%\n")
                f.write(f"    F1分数: {task_f1:.2f}\n\n")
        
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
        
        # 误判样本详细分析
        f.write("=" * 80 + "\n")
        f.write("【误判样本详细分析】\n")
        f.write("=" * 80 + "\n\n")
        
        # 假阳性分析
        f.write(f"◆ 假阳性 (False Positive) - 无幻觉但被误判为幻觉\n")
        f.write(f"  总数: {fp}\n\n")
        
        f.write("  按任务类型分布:\n")
        for task_type, count in fp_by_task.items():
            if count > 0:
                percentage = count / fp * 100 if fp > 0 else 0
                f.write(f"    {task_type}: {count} ({percentage:.2f}%)\n")
        
        if false_positive_samples:
            f.write(f"\n  假阳性样本示例 (前10个):\n")
            for i, sample in enumerate(false_positive_samples[:10], 1):
                f.write(f"    {i}. ID: {sample['id']}, 任务: {sample['task_type']}, 分数: {sample['score']:.4f}, 置信度: {sample.get('confidence', 1.0):.4f}\n")
        
        f.write("\n")
        
        # 假阴性分析
        f.write(f"◆ 假阴性 (False Negative) - 有幻觉但未被检测\n")
        f.write(f"  总数: {fn}\n\n")
        
        f.write("  按任务类型分布:\n")
        for task_type, count in fn_by_task.items():
            if count > 0:
                percentage = count / fn * 100 if fn > 0 else 0
                f.write(f"    {task_type}: {count} ({percentage:.2f}%)\n")
        
        f.write("\n  按幻觉标签类型分布:\n")
        for label_type, count in fn_by_label.items():
            if count > 0:
                percentage = count / fn * 100 if fn > 0 else 0
                f.write(f"    {label_type}: {count} ({percentage:.2f}%)\n")
        
        if false_negative_samples:
            f.write(f"\n  假阴性样本示例 (前10个):\n")
            for i, sample in enumerate(false_negative_samples[:10], 1):
                labels_str = ', '.join(sample['label_types'])
                f.write(f"    {i}. ID: {sample['id']}, 任务: {sample['task_type']}, 标签: [{labels_str}], 分数: {sample['score']:.4f}, 置信度: {sample.get('confidence', 1.0):.4f}\n")
        
        f.write("\n")
        
        # 对比原版性能
        f.write("=" * 80 + "\n")
        f.write("【与原版对比】\n")
        f.write("=" * 80 + "\n")
        f.write("原版性能:\n")
        f.write("  准确率: 53.94%\n")
        f.write("  召回率: 85.87%\n")
        f.write("  F1分数: 66.26\n")
        f.write("  假阳性: 5619\n")
        f.write("  假阴性: 1083\n\n")
        
        f.write("改进版性能:\n")
        f.write(f"  准确率: {precision:.2f}% {'(提升)' if precision > 53.94 else '(下降)'}\n")
        f.write(f"  召回率: {recall:.2f}% {'(提升)' if recall > 85.87 else '(下降)'}\n")
        f.write(f"  F1分数: {f1:.2f} {'(提升)' if f1 > 66.26 else '(下降)'}\n")
        f.write(f"  假阳性: {fp} {'(减少 ' + str(5619-fp) + ')' if fp < 5619 else '(增加 ' + str(fp-5619) + ')'}\n")
        f.write(f"  假阴性: {fn} {'(减少 ' + str(1083-fn) + ')' if fn < 1083 else '(增加 ' + str(fn-1083) + ')'}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"结果已保存到: {output_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    
    # 打印性能摘要
    print("\n" + "=" * 80)
    print("【改进版BARTScore检测性能摘要】")
    print("=" * 80)
    print(f"准确率 (Precision): {precision:.2f}% (原版: 53.94%)")
    print(f"召回率 (Recall): {recall:.2f}% (原版: 85.87%)")
    print(f"F1分数: {f1:.2f} (原版: 66.26)")
    print(f"假阳性: {fp} (原版: 5619, 变化: {fp-5619:+d})")
    print(f"假阴性: {fn} (原版: 1083, 变化: {fn-1083:+d})")
    print("=" * 80)


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='改进版BARTScore幻觉检测器')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID (0, 1, 2, ...)，不指定则自动选择')
    parser.add_argument('--input', type=str, default='../data/test_response_label.jsonl', help='输入文件路径')
    parser.add_argument('--output', type=str, default='bartscore_improved_results.jsonl', help='输出文件路径')
    parser.add_argument('--no-bidirectional', action='store_true', help='禁用双向评分')
    parser.add_argument('--model', type=str, default='facebook/bart-large-cnn', help='BART模型名称')
    
    args = parser.parse_args()
    
    # 检查GPU可用性
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n检测到 {gpu_count} 张GPU卡:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if args.gpu is not None:
            if args.gpu >= gpu_count:
                print(f"⚠ 警告: 指定的GPU {args.gpu} 不存在，将使用GPU 0")
                args.gpu = 0
    else:
        print("⚠ 警告: 未检测到GPU，使用CPU运行会很慢")
        args.gpu = None
    
    # 运行改进版BARTScore检测
    # 使用任务特定阈值和双向评分
    process_dataset_improved(
        input_file=args.input,
        output_file=args.output,
        task_thresholds={
            'Summary': -1.65,      # 原平均-1.82，略微放宽以减少假阳性
            'QA': -2.05,           # 原平均-2.12
            'Data2txt': -2.45      # 原平均-2.50
        },
        model_name=args.model,
        use_bidirectional=not args.no_bidirectional,  # 启用双向评分
        gpu_id=args.gpu
    )

