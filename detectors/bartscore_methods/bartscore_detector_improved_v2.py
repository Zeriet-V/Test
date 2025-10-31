"""
BARTScore 幻觉检测器 - 改进版 V2
支持标准的验证集→阈值优化→测试集流程

主要改进：
1. 任务特定阈值 - 为不同任务类型使用不同的检测阈值
2. 双向BARTScore - 同时评估source->target和target->source
3. 置信度评分 - 基于双向分数一致性提供置信度
4. 按幻觉类型统计F1分数 - 详细的幻觉类型性能分析
5. 支持外部阈值输入 - 可使用阈值优化器得到的最优阈值
"""

import torch
import json
import os
import argparse
from tqdm import tqdm
import numpy as np

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

print(f"🔧 镜像设置: {os.environ.get('HF_ENDPOINT')}")

from transformers import BartTokenizer, BartForConditionalGeneration


class ImprovedBARTScorer:
    """改进的BARTScore评分器，支持双向评分和置信度评估"""
    
    def __init__(self, model_name='facebook/bart-large-cnn', device='cuda' if torch.cuda.is_available() else 'cpu', gpu_id=None):
        if gpu_id is not None and torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
            print(f"指定使用GPU: {gpu_id}")
        
        print(f"加载改进版BARTScore模型: {model_name}")
        print(f"使用设备: {device}")
        
        self.device = device
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        model_cache = os.path.join(cache_dir, f'models--{model_name.replace("/", "--")}')
        
        if os.path.exists(model_cache):
            print("检测到本地缓存，尝试离线加载...")
            try:
                self.tokenizer = BartTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = BartForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
                print("✓ 离线加载成功！")
            except Exception as e:
                print(f"⚠ 离线加载失败，尝试在线下载...")
                self.tokenizer = BartTokenizer.from_pretrained(model_name)
                self.model = BartForConditionalGeneration.from_pretrained(model_name)
                print("✓ 在线加载成功！")
        else:
            print("本地无缓存，开始在线下载...")
            self.tokenizer = BartTokenizer.from_pretrained(model_name)
            self.model = BartForConditionalGeneration.from_pretrained(model_name)
            print("✓ 下载并加载成功！")
        
        self.model.eval()
        self.model.to(self.device)
        print("BARTScore模型加载完成！")
    
    def score_bidirectional(self, source_text, generated_text):
        """双向BARTScore评估"""
        with torch.no_grad():
            src_tokens = self.tokenizer([source_text], return_tensors='pt', truncation=True, max_length=1024)
            gen_tokens = self.tokenizer([generated_text], return_tensors='pt', truncation=True, max_length=1024)
            
            src_tokens = {k: v.to(self.device) for k, v in src_tokens.items()}
            gen_tokens = {k: v.to(self.device) for k, v in gen_tokens.items()}
            
            # Forward: P(generated|source)
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
            
            avg_score = (forward_score + backward_score) / 2
            harmonic_mean = 2 * forward_score * backward_score / (forward_score + backward_score) if (forward_score + backward_score) != 0 else 0
            
            return {
                'forward': forward_score,
                'backward': backward_score,
                'average': avg_score,
                'harmonic': harmonic_mean
            }
    
    def score_with_confidence(self, source_text, generated_text):
        """计算BARTScore并提供置信度"""
        bi_scores = self.score_bidirectional(source_text, generated_text)
        score_diff = abs(bi_scores['forward'] - bi_scores['backward'])
        confidence = 1.0 / (1.0 + score_diff)
        
        return {
            'score': bi_scores['forward'],
            'confidence': confidence,
            'bidirectional': bi_scores
        }


def compute_f1_metrics(tp, fp, fn):
    """计算 Precision, Recall, F1"""
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1


def process_dataset_improved_v2(input_file,
                                 output_file,
                                 task_thresholds=None,
                                 model_name='facebook/bart-large-cnn',
                                 use_bidirectional=True,
                                 gpu_id=None,
                                 dataset_type='validation'):
    """
    改进版 BARTScore 检测器 V2
    
    :param input_file: 输入文件
    :param output_file: 输出文件
    :param task_thresholds: 任务特定阈值字典
    :param model_name: BART模型名称
    :param use_bidirectional: 是否使用双向评分
    :param gpu_id: GPU ID
    :param dataset_type: 数据集类型 ('validation' 或 'test')
    """
    print(f"\n【改进版BARTScore幻觉检测器 V2】")
    print(f"数据集类型: {dataset_type.upper()}")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"模型: {model_name}")
    print(f"双向评分: {use_bidirectional}")
    
    # 默认任务特定阈值（可被外部覆盖）
    if task_thresholds is None:
        task_thresholds = {
            'Summary': -1.65,
            'QA': -2.05,
            'Data2txt': -2.45
        }
    
    print(f"\n任务特定阈值:")
    for task, threshold in task_thresholds.items():
        print(f"  {task}: {threshold:.4f}")
    print("=" * 80)
    
    # 初始化模型
    scorer = ImprovedBARTScorer(model_name=model_name, gpu_id=gpu_id)
    
    # 统计变量
    total_count = 0
    has_hallucination_count = 0
    no_hallucination_count = 0
    detected_count = 0
    
    stats = {
        'has_label_detected': 0,
        'has_label_not_detected': 0,
        'no_label_detected': 0,
        'no_label_not_detected': 0
    }
    
    # 按任务类型统计
    task_stats = {}
    for task in ['Summary', 'QA', 'Data2txt']:
        task_stats[task] = {
            'total': 0, 'has_label': 0, 'detected': 0,
            'true_positive': 0, 'false_negative': 0, 'false_positive': 0,
            'scores': []
        }
    
    # 按幻觉标签类型统计（增强版）
    label_stats = {}
    for label_type in ['Evident Conflict', 'Subtle Conflict', 'Evident Baseless Info', 'Subtle Baseless Info']:
        label_stats[label_type] = {
            'total': 0,
            'detected': 0,
            'true_positive': 0,  # 新增
            'false_negative': 0,  # 新增
            'samples': []
        }
    
    # 分数统计
    all_scores = []
    hallucination_scores = []
    no_hallucination_scores = []
    
    # 误判样本
    false_positive_samples = []
    false_negative_samples = []
    
    # 读取并处理数据
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(lines, desc=f"检测-{dataset_type}"):
            total_count += 1
            try:
                data = json.loads(line)
            except:
                continue
            
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
                
                # 更新整体统计
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
                elif not has_label and detected:
                    stats['no_label_detected'] += 1
                    false_positive_samples.append({
                        'id': data.get('id', ''),
                        'task_type': task_type,
                        'score': float(score),
                        'confidence': float(confidence)
                    })
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
                
                # 更新幻觉标签类型统计（增强版，计算每个类型的TP/FN）
                for label_type in label_types:
                    if label_type in label_stats:
                        label_stats[label_type]['total'] += 1
                        if detected:
                            label_stats[label_type]['detected'] += 1
                            label_stats[label_type]['true_positive'] += 1
                        else:
                            label_stats[label_type]['false_negative'] += 1
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
        f.write(f"改进版BARTScore幻觉检测详细报告 ({dataset_type.upper()})\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【检测策略】\n")
        f.write("  方法: 改进版BARTScore V2\n")
        f.write(f"  数据集: {dataset_type}\n")
        f.write(f"  模型: {model_name}\n")
        f.write(f"  双向评分: {use_bidirectional}\n")
        f.write("  改进点:\n")
        f.write("    1. 任务特定阈值\n")
        if use_bidirectional:
            f.write("    2. 双向BARTScore\n")
            f.write("    3. 置信度评分\n")
        f.write("    4. 按幻觉类型的F1分数统计\n")
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
        
        # 整体性能指标
        tp = stats['has_label_detected']
        fp = stats['no_label_detected']
        fn = stats['has_label_not_detected']
        tn = stats['no_label_not_detected']
        
        precision, recall, f1 = compute_f1_metrics(tp, fp, fn)
        
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
                task_precision, task_recall, task_f1 = compute_f1_metrics(
                    stats_data['true_positive'],
                    stats_data['false_positive'],
                    stats_data['false_negative']
                )
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
        
        # 按幻觉类型统计（增强版，包含F1）
        f.write("=" * 80 + "\n")
        f.write("【按幻觉类型统计（含F1分数）】\n")
        f.write("=" * 80 + "\n\n")
        
        for label_type, label_data in label_stats.items():
            if label_data['total'] > 0:
                detection_rate = label_data['detected'] / label_data['total'] * 100
                
                # 计算该幻觉类型的 Precision, Recall, F1
                # 注意：这里 FP 无法精确统计（因为无幻觉样本没有标签类型）
                # 我们只计算 Recall（检测率）
                label_recall = detection_rate
                label_tp = label_data['true_positive']
                label_fn = label_data['false_negative']
                
                f.write(f"◆ {label_type}:\n")
                f.write(f"  总数: {label_data['total']}\n")
                f.write(f"  检测到 (TP): {label_data['detected']} ({detection_rate:.2f}%)\n")
                f.write(f"  漏检 (FN): {label_fn} ({(100-detection_rate):.2f}%)\n")
                f.write(f"  召回率 (Recall): {label_recall:.2f}%\n")
                f.write(f"  状态: {'✓ 检测效果好' if detection_rate >= 80 else '✗ 需要改进'}\n")
                f.write(f"  样本ID (前5个): {label_data['samples']}\n\n")
        
        # 误判样本分析
        f.write("=" * 80 + "\n")
        f.write("【误判样本分析】\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"◆ 假阳性 (FP): {fp}\n")
        if false_positive_samples:
            f.write(f"  示例 (前10个):\n")
            for i, sample in enumerate(false_positive_samples[:10], 1):
                f.write(f"    {i}. ID: {sample['id']}, 任务: {sample['task_type']}, 分数: {sample['score']:.4f}, 置信度: {sample.get('confidence', 1.0):.4f}\n")
        f.write("\n")
        
        f.write(f"◆ 假阴性 (FN): {fn}\n")
        if false_negative_samples:
            f.write(f"  示例 (前10个):\n")
            for i, sample in enumerate(false_negative_samples[:10], 1):
                labels_str = ', '.join(sample['label_types'])
                f.write(f"    {i}. ID: {sample['id']}, 任务: {sample['task_type']}, 标签: [{labels_str}], 分数: {sample['score']:.4f}, 置信度: {sample.get('confidence', 1.0):.4f}\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write(f"结果已保存到: {output_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    print("\n" + "=" * 80)
    print(f"【检测性能摘要 - {dataset_type.upper()}】")
    print("=" * 80)
    print(f"准确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall): {recall:.2f}%")
    print(f"F1分数: {f1:.2f}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改进版BARTScore幻觉检测器 V2（支持标准流程）')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    parser.add_argument('--dataset-type', type=str, default='validation', choices=['validation', 'test'], 
                        help='数据集类型：validation 或 test')
    parser.add_argument('--no-bidirectional', action='store_true', help='禁用双向评分')
    parser.add_argument('--model', type=str, default='facebook/bart-large-cnn', help='BART模型名称')
    
    # 任务特定阈值（可选，可以从阈值优化器的结果中获取）
    parser.add_argument('--threshold-summary', type=float, default=-1.65, help='Summary任务阈值')
    parser.add_argument('--threshold-qa', type=float, default=-2.05, help='QA任务阈值')
    parser.add_argument('--threshold-data2txt', type=float, default=-2.45, help='Data2txt任务阈值')
    
    args = parser.parse_args()
    
    # GPU 检查
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n检测到 {gpu_count} 张GPU:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if args.gpu is not None and args.gpu >= gpu_count:
            print(f"⚠ 警告: GPU {args.gpu} 不存在，使用 GPU 0")
            args.gpu = 0
    else:
        print("⚠ 警告: 未检测到GPU，使用CPU运行会很慢")
        args.gpu = None
    
    # 构建任务特定阈值
    task_thresholds = {
        'Summary': args.threshold_summary,
        'QA': args.threshold_qa,
        'Data2txt': args.threshold_data2txt
    }
    
    # 运行检测
    process_dataset_improved_v2(
        input_file=args.input,
        output_file=args.output,
        task_thresholds=task_thresholds,
        model_name=args.model,
        use_bidirectional=not args.no_bidirectional,
        gpu_id=args.gpu,
        dataset_type=args.dataset_type
    )

