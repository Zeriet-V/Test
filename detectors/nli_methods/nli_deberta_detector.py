#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
nli_deberta_detector_fixed.py
修正版 DeBERTa-NLI 幻觉检测器（完整）
主要改动：
 - 自动读取 model.config.id2label（不再硬编码）
 - 支持长前提 chunking + 聚合（避免截断丢证据）
 - 批量化推理（可选）
 - 简单阈值自动校准（--calibrate）
 - 改进判定逻辑（entailment/contradiction/neutral 组合）
"""

import os
import json
import argparse
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -------- utils --------
def safe_mean(xs):
    return float(np.mean(xs)) if xs else 0.0

def compute_f1_from_binary(y_true, y_pred):
    # y_true, y_pred: lists of 0/1
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def find_best_threshold(y_true, entailment_scores):
    # search entailment threshold t: predict hallucination if entailment < t
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, 101):
        preds = [1 if s < t else 0 for s in entailment_scores]
        _, _, f1 = compute_f1_from_binary(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1

# -------- Detector Class --------
class DeBERTaNLIDetector:
    def __init__(self,
                 model_name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                 device=None,
                 local_files_only=False,
                 max_length=512,
                 chunk_size=240,
                 chunk_stride=120,
                 batch_size=16):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.batch_size = batch_size

        print(f"加载模型: {model_name} (local_files_only={local_files_only}) 到设备 {self.device}")
        # load tokenizer & model (try local first if requested)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=local_files_only)
        except Exception as e:
            print(f"模型加载出错（尝试在线重新下载）: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=False)

        self.model.to(self.device)
        self.model.eval()

        # parse id2label canonical mapping (lowercased)
        id2label_raw = getattr(getattr(self.model, "config", None), "id2label", None)
        if id2label_raw is None:
            # fallback: try model.config.label2id
            id2label_raw = getattr(getattr(self.model, "config", None), "label2id", None)

        # build normalized id->label map
        self.id2label = {}
        if isinstance(id2label_raw, dict):
            # ensure keys are ints
            for k, v in id2label_raw.items():
                try:
                    idx = int(k)
                except:
                    # if keys are labels mapping to ints (label2id), invert
                    try:
                        idx = int(v)
                        label_name = str(k).lower()
                        self.id2label[idx] = label_name
                    except:
                        continue
                else:
                    self.id2label[idx] = str(v).lower()
        else:
            # unknown; fallback assume standard order entailment, neutral, contradiction
            print("Warning: 未能从 model.config.id2label 中解析映射，使用默认顺序 (entailment, neutral, contradiction)")
            self.id2label = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

        print("Detected id2label mapping:", self.id2label)

        # try to find canonical ids
        self.canonical = {'entailment': None, 'neutral': None, 'contradiction': None}
        for idx, lab in self.id2label.items():
            if 'enta' in lab:
                self.canonical['entailment'] = idx
            elif 'neut' in lab:
                self.canonical['neutral'] = idx
            elif 'contra' in lab or 'contrad' in lab:
                self.canonical['contradiction'] = idx
        print("Canonical id mapping:", self.canonical)

    def _safe_score(self, probs, label_name):
        idx = self.canonical.get(label_name)
        if idx is not None and idx < len(probs):
            return float(probs[idx])
        # fallback: try common order
        fallback = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        i = fallback[label_name]
        return float(probs[i]) if i < len(probs) else 0.0

    def predict_single(self, premise, hypothesis, truncation='only_first'):
        """
        对一对文本进行单次 NLI 推理（small: 会进行 tokenization -> model）
        truncation: 'only_first' 表示只截断前提(premise)，避免截断 hypothesis
        """
        inputs = self.tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=truncation,
            max_length=self.max_length,
            padding='max_length'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()

        # build scores dict using canonical mapping
        scores = {
            'entailment': self._safe_score(probs, 'entailment'),
            'neutral': self._safe_score(probs, 'neutral'),
            'contradiction': self._safe_score(probs, 'contradiction')
        }

        pred_id = int(np.argmax(probs))
        pred_label_raw = self.id2label.get(pred_id, str(pred_id))
        # normalize predicted label to canonical names if possible
        if 'enta' in pred_label_raw:
            pred_norm = 'entailment'
        elif 'neut' in pred_label_raw:
            pred_norm = 'neutral'
        elif 'contra' in pred_label_raw or 'contrad' in pred_label_raw:
            pred_norm = 'contradiction'
        else:
            pred_norm = pred_label_raw

        return {
            'label': pred_norm,
            'scores': scores,
            'entailment_score': scores['entailment'],
            'contradiction_score': scores['contradiction'],
            'raw_probs': probs.tolist()
        }

    def predict_by_chunks(self, premise, hypothesis, chunk_size=None, stride=None):
        """
        对长前提进行 chunking 并聚合：
        - 切分 premise 的 token ids（不重复地滑窗截断）
        - 对每个 chunk 调用 predict_single
        - 聚合策略：取最大 entailment（最支持）、最大 contradiction（最反对）；同时保留最差句子（最小 entailment）
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if stride is None:
            stride = self.chunk_stride

        # tokenize premise into ids without special tokens
        enc = self.tokenizer(premise, add_special_tokens=False)
        input_ids = enc.get('input_ids', [])
        L = len(input_ids)
        if L == 0:
            # fallback to single predict
            return self.predict_single(premise, hypothesis)

        # if short enough, single call with truncation='only_first'
        if L <= chunk_size:
            return self.predict_single(premise, hypothesis)

        best_ent = -1.0
        best_contra = -1.0
        worst_ent = 1.0
        best_res = None
        # create chunks (non-overlapping or overlapping depending on stride)
        for start in range(0, L, stride):
            end = start + chunk_size
            chunk_ids = input_ids[start:end]
            # decode chunk
            chunk_text = self.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            res = self.predict_single(chunk_text, hypothesis, truncation='only_first')
            ent = res['entailment_score']
            contra = res['contradiction_score']
            if ent > best_ent:
                best_ent = ent
            if contra > best_contra:
                best_contra = contra
            if ent < worst_ent:
                worst_ent = ent
            # stop if we've covered the end
            if end >= L:
                break

        # compute neutral approx
        approx_neutral = max(0.0, 1.0 - best_ent - best_contra)
        aggregated = {
            'label': 'aggregated',
            'scores': {'entailment': best_ent, 'neutral': approx_neutral, 'contradiction': best_contra},
            'entailment_score': best_ent,
            'contradiction_score': best_contra
        }
        return aggregated

    def detect_hallucination(self, premise, hypothesis,
                             use_entailment=True,
                             threshold_entail=0.45,
                             threshold_contra=0.2,
                             sentence_level=False,
                             chunking=True):
        """
        主检测入口：
        - sentence_level: 若 True，则对 hypothesis 按句子分割并逐句检测（返回句子级结果）
        - chunking: 若 True，则对长前提进行 chunking 聚合
        - use_entailment: True -> 以 entailment < threshold_entail 作为怀疑触发之一
        - threshold_contra: 若 contradiction > threshold_contra 立即判断为幻觉
        返回: (bool, result_dict)
        """
        # 如果启用了句子级检测：把生成文本分句
        if sentence_level:
            # 简单的正则切句，避免依赖外部包
            import re
            sents = re.split(r'(?<=[.!?])\s+', hypothesis)
            sents = [s.strip() for s in sents if s.strip()]
            sentence_results = []
            any_hall = False
            for s in sents:
                if chunking:
                    res = self.predict_by_chunks(premise, s)
                else:
                    res = self.predict_single(premise, s)
                # 判定
                is_h = self._decide_from_scores(res['scores'], use_entailment, threshold_entail, threshold_contra)
                sentence_results.append({
                    'sentence': s,
                    'label': res.get('label', ''),
                    'entailment_score': res['entailment_score'],
                    'contradiction_score': res['contradiction_score'],
                    'is_hallucination': is_h
                })
                if is_h:
                    any_hall = True
            # aggregated choose worst sentence (min entailment)
            if sentence_results:
                worst = min(sentence_results, key=lambda x: x['entailment_score'])
                aggregated = {
                    'label': 'hallucination_detected' if any_hall else 'entailment',
                    'scores': {
                        'entailment': worst['entailment_score'],
                        'contradiction': worst['contradiction_score'],
                        'neutral': max(0.0, 1.0 - worst['entailment_score'] - worst['contradiction_score'])
                    },
                    'entailment_score': worst['entailment_score'],
                    'contradiction_score': worst['contradiction_score'],
                    'worst_sentence': worst['sentence'],
                    'sentence_results': sentence_results,
                    'num_sentences': len(sentence_results),
                    'num_hallucination_sentences': sum(1 for r in sentence_results if r['is_hallucination'])
                }
                return any_hall, aggregated
            else:
                # fallback
                pass

        # non-sentence-level:
        if chunking:
            res = self.predict_by_chunks(premise, hypothesis)
        else:
            res = self.predict_single(premise, hypothesis)

        is_h = self._decide_from_scores(res['scores'], use_entailment, threshold_entail, threshold_contra)
        return is_h, res

    def _decide_from_scores(self, scores, use_entailment, t_entail, t_contra):
        """
        改进判定逻辑（组合判断）：
        - 若 contradiction > t_contra -> hallucination
        - 否则若 use_entailment 且 entailment < t_entail 且 (neutral 高或 contradiction 中等) -> hallucination
        - 否则 not hallucination
        """
        ent = scores.get('entailment', 0.0)
        contra = scores.get('contradiction', 0.0)
        neut = scores.get('neutral', max(0.0, 1.0 - ent - contra))

        if contra > t_contra:
            return True
        if use_entailment and ent < t_entail:
            # if neutral is very high or small contradiction exists, consider hallucination
            if neut > 0.6 or contra > 0.05:
                return True
        return False

# -------- Dataset processing --------
def process_dataset_nli(input_file,
                        output_file,
                        model_name,
                        threshold_entail=0.45,
                        threshold_contra=0.2,
                        use_entailment=True,
                        sentence_level=False,
                        gpu_id=None,
                        local_files_only=False,
                        calibrate=False,
                        chunking=True):
    # device selection
    if gpu_id is not None and torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    detector = DeBERTaNLIDetector(model_name=model_name,
                                  device=device,
                                  local_files_only=local_files_only)

    print("\n开始处理数据集:", input_file)
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    total = 0
    detected_count = 0
    has_label_count = 0
    no_label_count = 0

    stats = {'has_label_detected': 0, 'has_label_not_detected': 0, 'no_label_detected': 0, 'no_label_not_detected': 0}
    false_positive_samples = []
    false_negative_samples = []

    all_contra_scores = []
    hall_contra_scores = []
    nohall_contra_scores = []

    task_stats = {}
    label_stats = {}

    # For calibration
    calib_y = []
    calib_entail_scores = []

    # Diagnostic first-N samples
    DIAG_N = min(20, len(lines))
    print("\n--- Diagnostic (前 {} 条样本) ---".format(DIAG_N))
    for i in range(DIAG_N):
        try:
            j = json.loads(lines[i])
            premise = j.get('test', '')
            if isinstance(premise, dict):
                # try to stringify
                premise = json.dumps(premise, ensure_ascii=False)
            hyp = j.get('response', '')
            debug_res = detector.predict_by_chunks(str(premise), str(hyp))
            print(f"[diag {i}] len_prem={len(str(premise))} len_hyp={len(str(hyp))} -> scores={debug_res.get('scores')}")
        except Exception as e:
            print("diag error:", e)

    # Main loop
    with open(output_file, 'w', encoding='utf-8') as fout:
        for line in tqdm(lines, desc="NLI检测"):
            total += 1
            try:
                data = json.loads(line)
            except:
                continue
            test_data = data.get('test', '')
            task_type = data.get('task_type', 'Unknown')
            if isinstance(test_data, dict):
                # try to compose a reasonable premise string (adapt to common structures)
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

            # detection
            detected, nli_res = detector.detect_hallucination(
                premise, hypothesis,
                use_entailment=use_entailment,
                threshold_entail=threshold_entail,
                threshold_contra=threshold_contra,
                sentence_level=sentence_level,
                chunking=chunking
            )

            if detected:
                detected_count += 1

            contra_score = nli_res.get('contradiction_score', 0.0)
            all_contra_scores.append(contra_score)
            if has_label:
                hall_contra_scores.append(contra_score)
            else:
                nohall_contra_scores.append(contra_score)

            # update stats
            if has_label and detected:
                stats['has_label_detected'] += 1
            elif has_label and not detected:
                stats['has_label_not_detected'] += 1
                false_negative_samples.append({
                    'id': data.get('id', ''),
                    'task_type': task_type,
                    'label_types': data.get('label_types', []),
                    'contradiction_score': float(contra_score),
                    'predicted_label': nli_res.get('label', '')
                })
            elif not has_label and detected:
                stats['no_label_detected'] += 1
                false_positive_samples.append({
                    'id': data.get('id', ''),
                    'task_type': task_type,
                    'contradiction_score': float(contra_score),
                    'predicted_label': nli_res.get('label', '')
                })
            else:
                stats['no_label_not_detected'] += 1

            # per-task
            if task_type not in task_stats:
                task_stats[task_type] = {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0, 'contradiction_scores': []}
            task_stats[task_type]['total'] += 1
            task_stats[task_type]['contradiction_scores'].append(contra_score)
            if has_label:
                task_stats[task_type]['has_label'] += 1
            if detected:
                task_stats[task_type]['detected'] += 1
            if has_label and detected:
                task_stats[task_type]['true_positive'] += 1
            elif has_label and not detected:
                task_stats[task_type]['false_negative'] += 1
            elif not has_label and detected:
                task_stats[task_type]['false_positive'] += 1

            # per-label-types
            for lab in data.get('label_types', []):
                if lab not in label_stats:
                    label_stats[lab] = {'total': 0, 'detected': 0, 'samples': []}
                label_stats[lab]['total'] += 1
                if detected:
                    label_stats[lab]['detected'] += 1
                if len(label_stats[lab]['samples']) < 5:
                    label_stats[lab]['samples'].append(data.get('id', ''))

            # write result
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
                'thresholds': {'entailment': threshold_entail, 'contradiction': threshold_contra}
            }

            if sentence_level and 'sentence_results' in nli_res:
                out['num_sentences'] = nli_res.get('num_sentences', 0)
                out['num_hallucination_sentences'] = nli_res.get('num_hallucination_sentences', 0)

            fout.write(json.dumps(out, ensure_ascii=False) + '\n')

            # for calibration gathering
            if calibrate:
                # y_true: 1 if has_label (we treat any label_types as hallucination)
                calib_y.append(1 if has_label else 0)
                calib_entail_scores.append(float(nli_res.get('entailment_score', 0.0)))

    # compute metrics
    tp = stats['has_label_detected']
    fp = stats['no_label_detected']
    fn = stats['has_label_not_detected']
    tn = stats['no_label_not_detected']
    precision, recall, f1 = compute_f1_from_binary(
        [1]*tp + [0]*fp + [1]*fn + [0]*tn,
        [1]*tp + [1]*fp + [0]*fn + [0]*tn
    )  # alternative simple computation
    # better compute from counts:
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # calibration if requested
    best_t, best_f1 = None, None
    if calibrate and calib_y and calib_entail_scores:
        best_t, best_f1 = find_best_threshold(calib_y, calib_entail_scores)

    # report
    report_file = output_file.replace('.jsonl', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as rf:
        rf.write("="*80 + "\n")
        rf.write("DeBERTa-NLI 幻觉检测报告（修正版）\n")
        rf.write("="*80 + "\n\n")
        rf.write(f"模型: {model_name}\n")
        rf.write(f"设备: {detector.device}\n")
        rf.write(f"id2label mapping: {detector.id2label}\n")
        rf.write(f"canonical mapping: {detector.canonical}\n")
        rf.write(f"句子级检测: {sentence_level}\n")
        rf.write(f"长前提 chunking: {chunking} (chunk_size={detector.chunk_size}, stride={detector.chunk_stride})\n")
        rf.write(f"判定阈值 (entailment < t -> hallucination): t_entail={threshold_entail}\n")
        rf.write(f"判定阈值 (contradiction > t -> hallucination): t_contra={threshold_contra}\n")
        if best_t is not None:
            rf.write(f"自动校准最好阈值（基于验证集）: entailment_t = {best_t} (F1={best_f1:.4f})\n")
        rf.write("\n【总体统计】\n")
        rf.write(f"  总样本: {total}\n")
        rf.write(f"  有标签 (被标注为含幻觉): {has_label_count}\n")
        rf.write(f"  无标签: {no_label_count}\n")
        rf.write(f"  检测到幻觉: {detected_count}\n\n")
        rf.write("【分数统计】\n")
        rf.write(f"  全部 contradiction 平均: {safe_mean(all_contra_scores):.4f}\n")
        rf.write(f"  有幻觉样本 contradiction 平均: {safe_mean(hall_contra_scores):.4f}\n")
        rf.write(f"  无幻觉样本 contradiction 平均: {safe_mean(nohall_contra_scores):.4f}\n\n")
        rf.write("【性能指标】\n")
        rf.write(f"  TP: {tp}\n  FP: {fp}\n  FN: {fn}\n  TN: {tn}\n\n")
        rf.write(f"  Precision: {precision:.2f}%\n")
        rf.write(f"  Recall: {recall:.2f}%\n")
        rf.write(f"  F1: {f1:.2f}\n\n")
        rf.write("按任务类型统计（前若干）:\n")
        for t, s in list(task_stats.items())[:20]:
            rf.write(f"  - {t}: total={s['total']}, has_label={s['has_label']}, detected={s['detected']}, mean_contra={safe_mean(s['contradiction_scores']):.4f}\n")
        rf.write("\n按幻觉类型统计（前若干）:\n")
        for lab, s in label_stats.items():
            rf.write(f"  - {lab}: total={s['total']}, detected={s['detected']}, sample_ids={s['samples']}\n")

        rf.write("\n结果文件: " + output_file + "\n")
        rf.write("="*80 + "\n")

    print("\n==== Summary ====")
    print(f"Total: {total}")
    print(f"Precision: {precision:.2f}%  Recall: {recall:.2f}%  F1: {f1:.2f}")
    if best_t is not None:
        print(f"Calibrated best entailment threshold: {best_t} (F1={best_f1:.4f})")
    print(f"Report written to: {report_file}")
    print(f"Results jsonl: {output_file}")

# -------- CLI --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeBERTa-NLI 幻觉检测器（修正版）")
    parser.add_argument('--gpu', type=int, default=None, help='GPU id (optional)')
    parser.add_argument('--input', type=str, default='../data/test_response_label.jsonl', help='输入 jsonl 文件')
    parser.add_argument('--output', type=str, default='nli_deberta_results_fixed.jsonl', help='输出 jsonl 文件')
    parser.add_argument('--model', type=str, default='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli', help='模型名')
    parser.add_argument('--threshold-entail', type=float, default=0.45, help='entailment 阈值 (entailment < t -> 怀疑)')
    parser.add_argument('--threshold-contra', type=float, default=0.20, help='contradiction 阈值 (contradiction > t -> 怀疑)')
    parser.add_argument('--use-entailment', action='store_true', default=True, help='使用 entailment 判定')
    parser.add_argument('--no-chunk', dest='chunking', action='store_false', help='禁用长前提 chunking')
    parser.add_argument('--sentence-level', action='store_true', help='启用句子级检测')
    parser.add_argument('--calibrate', action='store_true', help='在输入集上进行阈值校准（使用 label_types 判定真值）')
    parser.add_argument('--local-only', action='store_true', help='仅使用本地缓存加载模型，禁用下载')
    args = parser.parse_args()

    # GPU selection
    if args.gpu is not None and torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            print(f"警告: GPU {args.gpu} 不存在，改为 GPU 0")
            args.gpu = 0
    else:
        if args.gpu is not None:
            print("指定 GPU 无效或未检测到 CUDA，使用 CPU/默认设备。")
            args.gpu = None

    process_dataset_nli(
        input_file=args.input,
        output_file=args.output,
        model_name=args.model,
        threshold_entail=args.threshold_entail,
        threshold_contra=args.threshold_contra,
        use_entailment=args.use_entailment,
        sentence_level=args.sentence_level,
        gpu_id=args.gpu,
        local_files_only=args.local_only,
        calibrate=args.calibrate,
        chunking=args.chunking
    )
