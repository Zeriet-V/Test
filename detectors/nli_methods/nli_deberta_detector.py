"""
DeBERTa-NLI 幻觉检测器
使用自然语言推理(NLI)模型检测生成文本中的幻觉

原理：
- 将原文作为 premise（前提）
- 将生成文本作为 hypothesis（假设）
- NLI模型判断关系：entailment（蕴含）、neutral（中立）、contradiction（矛盾）
- contradiction 表示有幻觉，entailment 表示一致

优势：
- 直接检测矛盾和不一致
- DeBERTa 在 NLI 任务上表现优异
- 可解释性强
"""

import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

print(f"🔧 镜像设置: {os.environ.get('HF_ENDPOINT')}")


class DeBERTaNLIDetector:
    """
    基于 DeBERTa 的 NLI 幻觉检测器
    """
    
    def __init__(self, 
                 model_name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 gpu_id=None):
        """
        初始化 NLI 检测器
        
        :param model_name: 模型名称（推荐）
            - MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (推荐，多数据集训练，约1.5GB)
            - microsoft/deberta-large-mnli (标准，约1.4GB)
            - microsoft/deberta-base-mnli (较小，约400MB)
            - cross-encoder/nli-deberta-v3-large (DeBERTa-v3)
        :param device: 设备
        :param gpu_id: GPU ID
        """
        if gpu_id is not None and torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
            print(f"指定使用GPU: {gpu_id}")
        
        print(f"加载 DeBERTa-NLI 模型: {model_name}")
        print(f"使用设备: {device}")
        
        self.device = device
        self.model_name = model_name
        
        # 检查本地缓存
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        model_cache = os.path.join(cache_dir, f'models--{model_name.replace("/", "--")}')
        
        try:
            if os.path.exists(model_cache):
                print("检测到本地缓存，尝试离线加载...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    local_files_only=True
                )
                print("✓ 离线加载成功！")
            else:
                print("本地无缓存，开始在线下载...")
                print("（首次下载约1.3GB，需要几分钟）")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("✓ 在线下载并加载成功！")
        except Exception as e:
            print(f"⚠ 离线加载失败: {str(e)[:100]}")
            print("尝试在线下载...")
            import shutil
            if os.path.exists(model_cache):
                shutil.rmtree(model_cache, ignore_errors=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("✓ 在线下载并加载成功！")
        
        self.model.eval()
        self.model.to(self.device)
        
        # 标签映射（DeBERTa-MNLI 的标签顺序）
        # 0: contradiction, 1: neutral, 2: entailment
        self.label_mapping = {
            0: 'contradiction',  # 矛盾
            1: 'neutral',        # 中立
            2: 'entailment'      # 蕴含
        }
        
        print("DeBERTa-NLI 模型加载成功！")
    
    def predict(self, premise, hypothesis):
        """
        预测 NLI 关系
        
        :param premise: 前提（原文）
        :param hypothesis: 假设（生成文本）
        :return: {'label': str, 'scores': dict, 'contradiction_score': float}
        """
        # Tokenize
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        # 获取预测标签
        pred_label_id = torch.argmax(probs).item()
        pred_label = self.label_mapping[pred_label_id]
        
        # 各类别的概率
        scores = {
            'contradiction': probs[0].item(),
            'neutral': probs[1].item(),
            'entailment': probs[2].item()
        }
        
        return {
            'label': pred_label,
            'scores': scores,
            'contradiction_score': scores['contradiction'],
            'entailment_score': scores['entailment']
        }
    
    def detect_hallucination(self, source_text, generated_text, threshold=0.5, 
                            use_entailment=True, sentence_level=False):
        """
        检测幻觉
        
        :param source_text: 原文
        :param generated_text: 生成文本
        :param threshold: 阈值
        :param use_entailment: True=使用蕴含分数(推荐), False=使用矛盾分数
        :param sentence_level: 是否进行句子级检测
        :return: bool, dict
        """
        if sentence_level:
            return self._detect_sentence_level(source_text, generated_text, threshold, use_entailment)
        
        result = self.predict(source_text, generated_text)
        
        # 判断是否有幻觉
        if use_entailment:
            # 修正A: 蕴含分数不够高 = 有幻觉
            has_hallucination = result['entailment_score'] < threshold
        else:
            # 原方法: 矛盾分数高 = 有幻觉
            has_hallucination = result['contradiction_score'] > threshold
        
        return has_hallucination, result
    
    def _detect_sentence_level(self, source_text, generated_text, threshold, use_entailment):
        """
        句子级检测（修正B）
        
        :return: bool, dict with sentence-level results
        """
        # 改进1: 使用更健壮的分句方法
        # 优先级: SpaCy > NLTK > 正则表达式
        sentences = None
        
        # 方法1: SpaCy（最准确）
        try:
            import spacy
            if not hasattr(self, '_spacy_nlp'):
                # 只加载一次，缓存起来
                try:
                    # 只加载分句器，禁用其他组件以提速
                    self._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
                    # 确保有分句功能
                    if 'sentencizer' not in self._spacy_nlp.pipe_names and 'parser' not in self._spacy_nlp.pipe_names:
                        self._spacy_nlp.add_pipe('sentencizer')
                except OSError:
                    self._spacy_nlp = None
            
            if self._spacy_nlp:
                doc = self._spacy_nlp(generated_text)
                sentences = [sent.text.strip() for sent in doc.sents]
        except Exception:
            # 任何SpaCy错误都跳过
            pass
        
        # 方法2: NLTK（备用）
        if sentences is None:
            try:
                import nltk
                try:
                    sentences = nltk.sent_tokenize(generated_text)
                except LookupError:
                    # 跳过 punkt 下载，直接用正则
                    sentences = None
            except ImportError:
                pass
        
        # 方法3: 改进的正则表达式（最后备用）
        if sentences is None:
            import re
            sentences = re.split(r'(?<=[.!?])\s+', generated_text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            # 如果没有句子，回退到整体检测
            return self.detect_hallucination(source_text, generated_text, threshold, use_entailment, False)
        
        sentence_results = []
        any_hallucination = False
        
        for sent in sentences:
            result = self.predict(source_text, sent)
            
            if use_entailment:
                is_hallucination = result['entailment_score'] < threshold
            else:
                is_hallucination = result['contradiction_score'] > threshold
            
            sentence_results.append({
                'sentence': sent,
                'label': result['label'],
                'entailment_score': result['entailment_score'],
                'contradiction_score': result['contradiction_score'],
                'is_hallucination': is_hallucination
            })
            
            if is_hallucination:
                any_hallucination = True
        
        # 改进2: 使用"最差句子分数"而非"平均分数"
        # 找到最差的那个句子（entailment分数最低，或contradiction分数最高）
        if use_entailment:
            # 找entailment分数最低的句子（最可能是幻觉的）
            worst_sentence = min(sentence_results, key=lambda x: x['entailment_score'])
            worst_entailment = worst_sentence['entailment_score']
            worst_contradiction = worst_sentence['contradiction_score']
        else:
            # 找contradiction分数最高的句子
            worst_sentence = max(sentence_results, key=lambda x: x['contradiction_score'])
            worst_entailment = worst_sentence['entailment_score']
            worst_contradiction = worst_sentence['contradiction_score']
        
        # 计算 neutral 分数
        worst_neutral = 1.0 - worst_entailment - worst_contradiction
        
        # 同时也保留平均分数（用于统计分析）
        avg_entailment = np.mean([r['entailment_score'] for r in sentence_results])
        avg_contradiction = np.mean([r['contradiction_score'] for r in sentence_results])
        avg_neutral = 1.0 - avg_entailment - avg_contradiction
        
        aggregated_result = {
            'label': 'hallucination_detected' if any_hallucination else 'entailment',
            'scores': {
                'contradiction': worst_contradiction,  # 使用最差句子的分数
                'neutral': worst_neutral,
                'entailment': worst_entailment,
            },
            'entailment_score': worst_entailment,  # 最差句子的蕴含分数
            'contradiction_score': worst_contradiction,  # 最差句子的矛盾分数
            'avg_entailment_score': avg_entailment,  # 平均分数（仅供参考）
            'avg_contradiction_score': avg_contradiction,
            'worst_sentence': worst_sentence['sentence'],  # 最差的句子内容
            'sentence_results': sentence_results,
            'num_sentences': len(sentences),
            'num_hallucination_sentences': sum(r['is_hallucination'] for r in sentence_results)
        }
        
        return any_hallucination, aggregated_result


def process_dataset_nli(input_file='../data/test_response_label.jsonl',
                        output_file='nli_deberta_results.jsonl',
                        model_name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                        threshold=0.5,
                        use_entailment=True,
                        sentence_level=False,
                        gpu_id=None):
    """
    使用 DeBERTa-NLI 检测数据集中的幻觉
    
    :param input_file: 输入文件
    :param output_file: 输出文件
    :param model_name: 模型名称
    :param threshold: 矛盾分数阈值
    :param gpu_id: GPU ID
    """
    print(f"\n【DeBERTa-NLI 幻觉检测器】开始处理数据集: {input_file}")
    print("=" * 80)
    print(f"模型: {model_name}")
    if use_entailment:
        print(f"蕴含阈值: {threshold} (entailment_score < {threshold} 判定为幻觉) ✓ 推荐")
    else:
        print(f"矛盾阈值: {threshold} (contradiction_score > {threshold} 判定为幻觉)")
    print(f"句子级检测: {'开启' if sentence_level else '关闭'}")
    if gpu_id is not None:
        print(f"指定GPU: {gpu_id}")
    print("=" * 80)
    
    # 初始化检测器
    detector = DeBERTaNLIDetector(model_name=model_name, gpu_id=gpu_id)
    
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
        'Summary': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'contradiction_scores': []},
        'QA': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'contradiction_scores': []},
        'Data2txt': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'contradiction_scores': []}
    }
    
    # 按幻觉标签类型统计
    label_stats = {
        'Evident Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Evident Baseless Info': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Baseless Info': {'total': 0, 'detected': 0, 'samples': []}
    }
    
    # 分数统计
    all_contradiction_scores = []
    hallucination_contradiction_scores = []
    no_hallucination_contradiction_scores = []
    
    # 误判样本
    false_positive_samples = []
    false_negative_samples = []
    
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
        
        for line in tqdm(lines, desc="NLI检测"):
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
                # NLI 检测
                detected, nli_result = detector.detect_hallucination(
                    text_original, 
                    text_generated, 
                    threshold,
                    use_entailment=use_entailment,
                    sentence_level=sentence_level
                )
                
                if detected:
                    detected_count += 1
                
                contradiction_score = nli_result['contradiction_score']
                
                # 记录分数
                all_contradiction_scores.append(contradiction_score)
                if has_label:
                    hallucination_contradiction_scores.append(contradiction_score)
                else:
                    no_hallucination_contradiction_scores.append(contradiction_score)
                
                # 更新性能统计
                if has_label and detected:
                    stats['has_label_detected'] += 1
                elif has_label and not detected:
                    stats['has_label_not_detected'] += 1
                    false_negative_samples.append({
                        'id': data.get('id', ''),
                        'task_type': task_type,
                        'label_types': label_types,
                        'contradiction_score': float(contradiction_score),
                        'predicted_label': nli_result['label']
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
                        'contradiction_score': float(contradiction_score),
                        'predicted_label': nli_result['label']
                    })
                    if task_type in fp_by_task:
                        fp_by_task[task_type] += 1
                else:
                    stats['no_label_not_detected'] += 1
                
                # 更新任务类型统计
                if task_type in task_stats:
                    task_stats[task_type]['total'] += 1
                    task_stats[task_type]['contradiction_scores'].append(contradiction_score)
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
                    'nli_label': nli_result['label'],
                    'contradiction_score': float(contradiction_score),
                    'entailment_score': float(nli_result['entailment_score']),
                    'neutral_score': float(nli_result['scores'].get('neutral', 0.0)),  # 安全访问
                    'detected': detected,
                    'threshold': threshold
                }
                
                # 如果是句子级检测，添加额外信息
                if sentence_level and 'sentence_results' in nli_result:
                    result['num_sentences'] = nli_result.get('num_sentences', 0)
                    result['num_hallucination_sentences'] = nli_result.get('num_hallucination_sentences', 0)
                
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n处理错误 (ID: {data.get('id', 'unknown')}): {str(e)}")
                continue
    
    # ============ 生成报告 ============
    report_file = output_file.replace('.jsonl', '_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DeBERTa-NLI 幻觉检测详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【检测策略】\n")
        f.write("  方法: DeBERTa-NLI - 基于自然语言推理的幻觉检测\n")
        f.write(f"  模型: {model_name}\n")
        if use_entailment:
            f.write(f"  检测阈值: {threshold} (entailment_score < {threshold} 判定为幻觉)\n")
            f.write("  判定标准: 非蕴含即幻觉 (contradiction + neutral → 幻觉)\n")
        else:
            f.write(f"  检测阈值: {threshold} (contradiction_score > {threshold} 判定为幻觉)\n")
            f.write("  判定标准: 矛盾即幻觉\n")
        f.write(f"  句子级检测: {'开启' if sentence_level else '关闭'}\n")
        f.write("  原理: 将原文作为premise，生成文本作为hypothesis，检测逻辑关系\n\n")
        
        f.write("【总体统计】\n")
        f.write(f"  总数据量: {total_count}\n")
        f.write(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 检测到幻觉: {detected_count} ({detected_count/total_count*100:.2f}%)\n\n")
        
        f.write("【矛盾分数分析】\n")
        f.write(f"  全部样本:\n")
        f.write(f"    平均矛盾分数: {np.mean(all_contradiction_scores):.4f}\n")
        f.write(f"    标准差: {np.std(all_contradiction_scores):.4f}\n")
        f.write(f"    最小值: {np.min(all_contradiction_scores):.4f}\n")
        f.write(f"    最大值: {np.max(all_contradiction_scores):.4f}\n\n")
        
        if hallucination_contradiction_scores:
            f.write(f"  有幻觉样本:\n")
            f.write(f"    平均矛盾分数: {np.mean(hallucination_contradiction_scores):.4f}\n")
            f.write(f"    标准差: {np.std(hallucination_contradiction_scores):.4f}\n\n")
        
        if no_hallucination_contradiction_scores:
            f.write(f"  无幻觉样本:\n")
            f.write(f"    平均矛盾分数: {np.mean(no_hallucination_contradiction_scores):.4f}\n")
            f.write(f"    标准差: {np.std(no_hallucination_contradiction_scores):.4f}\n\n")
        
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
                avg_score = np.mean(stats_data['contradiction_scores']) if stats_data['contradiction_scores'] else 0
                
                f.write(f"◆ {task} 任务:\n")
                f.write(f"  总数: {stats_data['total']}\n")
                f.write(f"  平均矛盾分数: {avg_score:.4f}\n")
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
        
        f.write("=" * 80 + "\n")
        f.write(f"结果已保存到: {output_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n详细报告已保存到: {report_file}")
    
    # 打印性能摘要
    print("\n" + "=" * 80)
    print("【DeBERTa-NLI 检测性能摘要】")
    print("=" * 80)
    print(f"矛盾阈值: {threshold}")
    print(f"\n准确率 (Precision): {precision:.2f}%")
    print(f"召回率 (Recall): {recall:.2f}%")
    print(f"F1分数: {f1:.2f}")
    print(f"\n矛盾分数统计:")
    print(f"  平均分数: {np.mean(all_contradiction_scores):.4f}")
    print(f"  有幻觉: {np.mean(hallucination_contradiction_scores):.4f}")
    print(f"  无幻觉: {np.mean(no_hallucination_contradiction_scores):.4f}")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DeBERTa-NLI 幻觉检测器')
    parser.add_argument('--gpu', type=int, default=None, help='指定GPU ID')
    parser.add_argument('--input', type=str, default='../data/test_response_label.jsonl', help='输入文件路径')
    parser.add_argument('--output', type=str, default='nli_deberta_results.jsonl', help='输出文件路径')
    parser.add_argument('--threshold', type=float, default=0.5, help='阈值（默认0.5）')
    parser.add_argument('--use-entailment', action='store_true', default=True, 
                        help='使用蕴含分数判定（推荐，默认开启）')
    parser.add_argument('--use-contradiction', dest='use_entailment', action='store_false',
                        help='使用矛盾分数判定（不推荐）')
    parser.add_argument('--sentence-level', action='store_true', 
                        help='启用句子级检测（推荐）')
    parser.add_argument('--model', type=str, 
                        default='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', 
                        help='模型名称 (默认: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)')
    
    args = parser.parse_args()
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"\n检测到 {gpu_count} 张GPU卡:")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        
        if args.gpu is not None:
            if args.gpu >= gpu_count:
                print(f"⚠ 警告: GPU {args.gpu} 不存在，将使用GPU 0")
                args.gpu = 0
    else:
        print("⚠ 警告: 未检测到GPU")
        args.gpu = None
    
    # 运行检测
    process_dataset_nli(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        threshold=args.threshold,
        use_entailment=args.use_entailment,
        sentence_level=args.sentence_level,
        gpu_id=args.gpu
    )

