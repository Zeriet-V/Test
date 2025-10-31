#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检索增强 NLI 幻觉检测器
与 nli_deberta_detector.py 相同的配置和报告格式，便于直接对比检索增强的效果

主要特点:
1. 使用 SentenceTransformer 从长原文中检索最相关的证据
2. 用检索到的证据 + 生成文本做 NLI
3. 句子级检测 + 最差句聚合
4. 与标准 NLI 检测器相同的报告格式
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

# -------- utils --------
def safe_mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def compute_f1_from_binary(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


class RetrievalAugmentedNLIDetector:
    """检索增强 NLI 检测器"""
    
    def __init__(self,
                 nli_model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                 retrieval_model='sentence-transformers/all-MiniLM-L6-v2',
                 device=None,
                 local_files_only=False,
                 max_length=512,
                 top_k=3):
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.top_k = top_k
        self.max_length = max_length
        
        print(f"加载检索增强NLI模型: {nli_model}")
        print(f"检索模型: {retrieval_model}")
        print(f"设备: {self.device}, Top-K: {self.top_k}")
        
        # 加载 NLI 模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model, local_files_only=local_files_only)
            self.model = AutoModelForSequenceClassification.from_pretrained(nli_model, local_files_only=local_files_only)
        except Exception as e:
            print(f"本地加载失败，尝试在线下载...")
            self.tokenizer = AutoTokenizer.from_pretrained(nli_model, local_files_only=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(nli_model, local_files_only=False)
        
        self.model.to(self.device)
        self.model.eval()
        
        # 加载 Sentence Transformer
        try:
            from sentence_transformers import SentenceTransformer
            self.retrieval_model = SentenceTransformer(retrieval_model, device=self.device)
            print("✓ 检索模型加载成功")
        except ImportError:
            print("✗ 错误: sentence-transformers 未安装")
            print("请运行: pip install sentence-transformers")
            raise
        
        # NLI 标签映射
        id2label_raw = getattr(getattr(self.model, "config", None), "id2label", None)
        self.id2label = {}
        if isinstance(id2label_raw, dict):
            for k, v in id2label_raw.items():
                try:
                    idx = int(k)
                    self.id2label[idx] = str(v).lower()
                except:
                    pass
        else:
            self.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        
        print("Detected id2label mapping:", self.id2label)
        
        self.canonical = {'entailment': None, 'neutral': None, 'contradiction': None}
        for idx, lab in self.id2label.items():
            if 'enta' in lab:
                self.canonical['entailment'] = idx
            elif 'neut' in lab:
                self.canonical['neutral'] = idx
            elif 'contra' in lab or 'contrad' in lab:
                self.canonical['contradiction'] = idx
    
    def _safe_score(self, probs, label_name):
        idx = self.canonical.get(label_name)
        if idx is not None and idx < len(probs):
            return float(probs[idx])
        fallback = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        i = fallback[label_name]
        return float(probs[i]) if i < len(probs) else 0.0
    
    def split_into_sentences(self, text):
        """分句"""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def retrieve_relevant_evidence(self, source_text, query_text, top_k=None):
        """从原文中检索最相关的证据"""
        if top_k is None:
            top_k = self.top_k
        
        source_sentences = self.split_into_sentences(source_text)
        
        if not source_sentences or len(source_sentences) <= top_k:
            return source_text
        
        try:
            from sentence_transformers import util
            
            # 编码
            source_embeddings = self.retrieval_model.encode(source_sentences, convert_to_tensor=True)
            query_embedding = self.retrieval_model.encode(query_text, convert_to_tensor=True)
            
            # 计算相似度
            similarities = util.cos_sim(query_embedding, source_embeddings)[0]
            
            # 选择 top-k
            top_k_indices = torch.topk(similarities, min(top_k, len(source_sentences))).indices
            
            # 按原顺序组合
            top_sentences = [source_sentences[i] for i in sorted(top_k_indices.cpu().numpy())]
            evidence = ' '.join(top_sentences)
            
            return evidence
        except Exception as e:
            print(f"检索失败: {str(e)[:50]}, 使用原文")
            return source_text
    
    def predict_single(self, premise, hypothesis):
        """单次 NLI 推理"""
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        
        scores = {
            'entailment': self._safe_score(probs, 'entailment'),
            'neutral': self._safe_score(probs, 'neutral'),
            'contradiction': self._safe_score(probs, 'contradiction')
        }
        
        return {
            'label': max(scores, key=scores.get),
            'scores': scores,
            'entailment_score': scores['entailment'],
            'contradiction_score': scores['contradiction']
        }
    
    def detect_hallucination(self, premise, hypothesis,
                             use_entailment=True,
                             threshold_entail=0.45,
                             threshold_contra=0.2,
                             sentence_level=True):
        """
        检测幻觉（句子级）
        对每个生成的句子检索相关证据后做 NLI
        """
        if not sentence_level:
            # 文档级：直接检索证据
            evidence = self.retrieve_relevant_evidence(premise, hypothesis)
            res = self.predict_single(evidence, hypothesis)
            is_h = self._decide_from_scores(res['scores'], use_entailment, threshold_entail, threshold_contra)
            return is_h, res
        
        # 句子级检测
        generated_sentences = self.split_into_sentences(hypothesis)
        
        if not generated_sentences:
            evidence = self.retrieve_relevant_evidence(premise, hypothesis)
            res = self.predict_single(evidence, hypothesis)
            is_h = self._decide_from_scores(res['scores'], use_entailment, threshold_entail, threshold_contra)
            return is_h, res
        
        sentence_results = []
        any_hallucination = False
        
        for sent in generated_sentences:
            # 为每个句子检索证据
            evidence = self.retrieve_relevant_evidence(premise, sent)
            result = self.predict_single(evidence, sent)
            result['evidence_used'] = evidence[:200]  # 记录使用的证据（截断）
            
            is_h = self._decide_from_scores(result['scores'], use_entailment, threshold_entail, threshold_contra)
            result['is_hallucination'] = is_h
            sentence_results.append(result)
            
            if is_h:
                any_hallucination = True
        
        # 聚合：使用最差句子
        if use_entailment:
            worst = min(sentence_results, key=lambda x: x['entailment_score'])
        else:
            worst = max(sentence_results, key=lambda x: x['contradiction_score'])
        
        worst['sentence_results'] = sentence_results
        worst['num_sentences'] = len(generated_sentences)
        worst['num_hallucination_sentences'] = sum(1 for r in sentence_results if r['is_hallucination'])
        
        return any_hallucination, worst
    
    def _decide_from_scores(self, scores, use_entailment, t_entail, t_contra):
        """判定逻辑"""
        ent = scores.get('entailment', 0.0)
        contra = scores.get('contradiction', 0.0)
        neut = scores.get('neutral', max(0.0, 1.0 - ent - contra))
        
        if contra > t_contra:
            return True
        if use_entailment and ent < t_entail:
            if neut > 0.6 or contra > 0.05:
                return True
        return False


def process_dataset_retrieval_nli(input_file,
                                   output_file,
                                   model_name,
                                   retrieval_model='sentence-transformers/all-MiniLM-L6-v2',
                                   threshold_entail=0.45,
                                   threshold_contra=0.2,
                                   use_entailment=True,
                                   sentence_level=True,
                                   top_k=3,
                                   gpu_id=None,
                                   local_files_only=False):
    
    # 设备选择
    if gpu_id is not None and torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    detector = RetrievalAugmentedNLIDetector(
        nli_model=model_name,
        retrieval_model=retrieval_model,
        device=device,
        local_files_only=local_files_only,
        top_k=top_k
    )
    
    print("\n开始处理数据集:", input_file)
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
    
    total = 0
    detected_count = 0
    has_label_count = 0
    no_label_count = 0
    
    stats = {'has_label_detected': 0, 'has_label_not_detected': 0, 
             'no_label_detected': 0, 'no_label_not_detected': 0}
    
    all_contra_scores = []
    hall_contra_scores = []
    nohall_contra_scores = []
    
    task_stats = {}
    label_stats = {}
    
    # Main loop
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(lines, desc="检索增强NLI检测"):
            total += 1
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
                    premise = f"{question} {passages}"
                elif task_type == 'Data2txt':
                    parts = []
                    if 'name' in test_data: parts.append(f"Name: {test_data['name']}")
                    if 'address' in test_data: parts.append(f"Address: {test_data['address']}")
                    if 'city' in test_data and 'state' in test_data:
                        parts.append(f"Location: {test_data['city']}, {test_data['state']}")
                    if 'categories' in test_data: parts.append(f"Categories: {test_data['categories']}")
                    if 'business_stars' in test_data: parts.append(f"Rating: {test_data['business_stars']} stars")
                    if 'review_info' in test_data and isinstance(test_data['review_info'], list):
                        for i_r, review in enumerate(test_data['review_info'][:3], 1):
                            if isinstance(review, dict) and 'review_text' in review:
                                parts.append(f"Review {i_r}: {review['review_text']}")
                    premise = ' '.join(parts)
                else:
                    premise = json.dumps(test_data, ensure_ascii=False)
            else:
                premise = str(test_data)
            
            hypothesis = str(data.get('response', ''))
            
            if not premise.strip() or not hypothesis.strip():
                continue
            
            has_label = bool(data.get('label_types'))
            if has_label:
                has_label_count += 1
            else:
                no_label_count += 1
            
            # 检测
            detected, nli_res = detector.detect_hallucination(
                premise, hypothesis,
                use_entailment=use_entailment,
                threshold_entail=threshold_entail,
                threshold_contra=threshold_contra,
                sentence_level=sentence_level
            )
            
            if detected:
                detected_count += 1
            
            contra_score = nli_res.get('contradiction_score', 0.0)
            all_contra_scores.append(contra_score)
            if has_label:
                hall_contra_scores.append(contra_score)
            else:
                nohall_contra_scores.append(contra_score)
            
            # 更新统计
            if has_label and detected:
                stats['has_label_detected'] += 1
            elif has_label and not detected:
                stats['has_label_not_detected'] += 1
            elif not has_label and detected:
                stats['no_label_detected'] += 1
            else:
                stats['no_label_not_detected'] += 1
            
            # 任务统计
            if task_type not in task_stats:
                task_stats[task_type] = {'total': 0, 'has_label': 0, 'detected': 0, 'contradiction_scores': []}
            task_stats[task_type]['total'] += 1
            task_stats[task_type]['contradiction_scores'].append(contra_score)
            if has_label:
                task_stats[task_type]['has_label'] += 1
            if detected:
                task_stats[task_type]['detected'] += 1
            
            # 幻觉类型统计
            for lab in data.get('label_types', []):
                if lab not in label_stats:
                    label_stats[lab] = {'total': 0, 'detected': 0, 'samples': []}
                label_stats[lab]['total'] += 1
                if detected:
                    label_stats[lab]['detected'] += 1
                if len(label_stats[lab]['samples']) < 5:
                    label_stats[lab]['samples'].append(data.get('id', ''))
            
            # 写入结果
            out = {
                'id': data.get('id', ''),
                'task_type': task_type,
                'has_label': has_label,
                'label_types': data.get('label_types', []),
                'nli_label': nli_res.get('label', ''),
                'contradiction_score': float(contra_score),
                'entailment_score': float(nli_res.get('entailment_score', 0.0)),
                'neutral_score': float(nli_res.get('scores', {}).get('neutral', 0.0)),
                'detected': detected,
                'thresholds': {'entailment': threshold_entail, 'contradiction': threshold_contra},
                'retrieval_top_k': top_k
            }
            
            if sentence_level and 'sentence_results' in nli_res:
                out['num_sentences'] = nli_res.get('num_sentences', 0)
                out['num_hallucination_sentences'] = nli_res.get('num_hallucination_sentences', 0)
            
            fout.write(json.dumps(out, ensure_ascii=False) + '\n')
    
    # 计算指标
    tp = stats['has_label_detected']
    fp = stats['no_label_detected']
    fn = stats['has_label_not_detected']
    tn = stats['no_label_not_detected']
    
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 生成报告
    report_file = output_file.replace('.jsonl', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as rf:
        rf.write("=" * 80 + "\n")
        rf.write("检索增强 NLI 幻觉检测报告\n")
        rf.write("=" * 80 + "\n\n")
        rf.write(f"模型: {model_name}\n")
        rf.write(f"检索模型: {retrieval_model}\n")
        rf.write(f"设备: {detector.device}\n")
        rf.write(f"检索 Top-K: {top_k}\n")
        rf.write(f"句子级检测: {sentence_level}\n")
        rf.write(f"判定阈值 (entailment < t): t_entail={threshold_entail}\n")
        rf.write(f"判定阈值 (contradiction > t): t_contra={threshold_contra}\n\n")
        
        rf.write("【总体统计】\n")
        rf.write(f"  总样本: {total}\n")
        rf.write(f"  有标签 (被标注为含幻觉): {has_label_count}\n")
        rf.write(f"  无标签: {no_label_count}\n")
        rf.write(f"  检测到幻觉: {detected_count}\n\n")
        
        rf.write("【分数统计】\n")
        rf.write(f"  全部 contradiction 平均: {safe_mean(all_contra_scores):.4f}\n")
        rf.write(f"  有幻觉样本 contradiction 平均: {safe_mean(hall_contra_scores):.4f}\n")
        rf.write(f"  无幻觉样本 contradiction 平均: {safe_mean(nohall_contra_scores):.4f}\n\n")
        
        rf.write("【性能指标】\n")
        rf.write(f"  TP: {tp}\n")
        rf.write(f"  FP: {fp}\n")
        rf.write(f"  FN: {fn}\n")
        rf.write(f"  TN: {tn}\n\n")
        rf.write(f"  Precision: {precision:.2f}%\n")
        rf.write(f"  Recall: {recall:.2f}%\n")
        rf.write(f"  F1: {f1:.2f}\n\n")
        
        rf.write("按任务类型统计:\n")
        for t, s in task_stats.items():
            rf.write(f"  - {t}: total={s['total']}, has_label={s['has_label']}, detected={s['detected']}, mean_contra={safe_mean(s['contradiction_scores']):.4f}\n")
        
        rf.write("\n按幻觉类型统计:\n")
        for lab, s in label_stats.items():
            rf.write(f"  - {lab}: total={s['total']}, detected={s['detected']}, sample_ids={s['samples']}\n")
        
        rf.write("\n结果文件: " + output_file + "\n")
        rf.write("=" * 80 + "\n")
    
    print("\n==== Summary ====")
    print(f"Total: {total}")
    print(f"Precision: {precision:.2f}%  Recall: {recall:.2f}%  F1: {f1:.2f}")
    print(f"Report written to: {report_file}")
    print(f"Results jsonl: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检索增强 NLI 幻觉检测器")
    parser.add_argument('--gpu', type=int, default=None, help='GPU id')
    parser.add_argument('--input', type=str, default='../data/validation_set.jsonl', help='输入文件')
    parser.add_argument('--output', type=str, default='nli_retrieval_validation_results.jsonl', help='输出文件')
    parser.add_argument('--model', type=str, default='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', help='NLI模型')
    parser.add_argument('--retrieval-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='检索模型')
    parser.add_argument('--threshold-entail', type=float, default=0.45, help='entailment 阈值')
    parser.add_argument('--threshold-contra', type=float, default=0.2, help='contradiction 阈值')
    parser.add_argument('--use-entailment', action='store_true', default=True, help='使用 entailment 判定')
    parser.add_argument('--no-sentence-level', dest='sentence_level', action='store_false', help='禁用句子级检测')
    parser.add_argument('--top-k', type=int, default=3, help='检索 top-k 个相关句子')
    parser.add_argument('--local-only', action='store_true', help='仅使用本地缓存')
    
    args = parser.parse_args()
    
    # GPU 检查
    if args.gpu is not None and torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            print(f"警告: GPU {args.gpu} 不存在，改为 GPU 0")
            args.gpu = 0
    
    process_dataset_retrieval_nli(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        retrieval_model=args.retrieval_model,
        threshold_entail=args.threshold_entail,
        threshold_contra=args.threshold_contra,
        use_entailment=args.use_entailment,
        sentence_level=args.sentence_level,
        top_k=args.top_k,
        gpu_id=args.gpu,
        local_files_only=args.local_only
    )

