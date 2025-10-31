"""
基于检索的 NLI 幻觉检测器
Retrieval-Augmented NLI

核心思路:
1. 使用 SentenceTransformer 从长原文中检索最相关的证据
2. 用短证据 + 生成文本做 NLI
3. 避免长文本截断，保留相关信息

优势:
- 解决长文本问题
- 聚焦相关证据
- 预期准确率大幅提升
"""

import torch
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import re

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'

print(f"🔧 镜像设置: {os.environ.get('HF_ENDPOINT')}")


class RetrievalAugmentedNLI:
    """
    基于检索的 NLI 检测器
    """
    
    def __init__(self,
                 nli_model='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                 retrieval_model='sentence-transformers/all-MiniLM-L6-v2',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 gpu_id=None):
        """
        初始化
        
        :param nli_model: NLI 模型
        :param retrieval_model: 检索模型（SentenceTransformer）
        :param device: 设备
        :param gpu_id: GPU ID
        """
        if gpu_id is not None and torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
            print(f"指定使用GPU: {gpu_id}")
        
        self.device = device
        
        print(f"\n加载模型...")
        print(f"  NLI模型: {nli_model}")
        print(f"  检索模型: {retrieval_model}")
        print(f"  设备: {device}")
        
        # 加载 NLI 模型
        print("\n1. 加载 NLI 模型...")
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        nli_cache = os.path.join(cache_dir, f'models--{nli_model.replace("/", "--")}')
        
        try:
            if os.path.exists(nli_cache):
                self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model, local_files_only=True)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model, local_files_only=True)
                print("  ✓ NLI模型加载成功（离线）")
            else:
                self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
                self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
                print("  ✓ NLI模型加载成功（在线）")
        except Exception as e:
            print(f"  ⚠ 离线加载失败，尝试在线...")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model)
            print("  ✓ NLI模型加载成功")
        
        self.nli_model.eval()
        self.nli_model.to(self.device)
        
        # 加载 Sentence Transformer
        print("\n2. 加载检索模型...")
        try:
            from sentence_transformers import SentenceTransformer
            self.retrieval_model = SentenceTransformer(retrieval_model, device=self.device)
            print("  ✓ 检索模型加载成功")
        except ImportError:
            print("  ✗ 错误: sentence-transformers 未安装")
            print("  请运行: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"  ⚠ 离线加载失败: {str(e)[:100]}")
            print("  尝试在线下载...")
            self.retrieval_model = SentenceTransformer(retrieval_model, device=self.device)
            print("  ✓ 检索模型加载成功")
        
        print("\n✓ 所有模型加载完成！")
        
        # NLI 标签映射（官方正确映射 - 已修复）
        # 0: entailment, 1: neutral, 2: contradiction
        self.label_mapping = {
            0: 'entailment',     # 蕴含
            1: 'neutral',        # 中立
            2: 'contradiction'   # 矛盾
        }
    
    def split_into_sentences(self, text):
        """
        将文本分句
        """
        try:
            import spacy
            if not hasattr(self, '_spacy_nlp'):
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])
                    if 'sentencizer' not in self._spacy_nlp.pipe_names and 'parser' not in self._spacy_nlp.pipe_names:
                        self._spacy_nlp.add_pipe('sentencizer')
                except:
                    self._spacy_nlp = None
            
            if self._spacy_nlp:
                doc = self._spacy_nlp(text)
                return [sent.text.strip() for sent in doc.sents]
        except:
            pass
        
        # 备用：正则表达式
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s) > 10]
    
    def retrieve_relevant_evidence(self, source_text, generated_text, top_k=3, max_evidence_tokens=150):
        """
        从原文中检索最相关的证据
        
        :param source_text: 原文（可能很长）
        :param generated_text: 生成文本
        :param top_k: 检索top-k个最相关的句子
        :param max_evidence_tokens: 证据最大token数
        :return: 检索到的证据文本
        """
        # 1. 将原文分句
        source_sentences = self.split_into_sentences(source_text)
        
        if not source_sentences:
            # 如果分句失败，直接截断
            return source_text[:max_evidence_tokens * 4]  # 粗略估计
        
        if len(source_sentences) <= top_k:
            # 如果句子数量少，直接返回全部
            return ' '.join(source_sentences)
        
        # 2. 计算相似度
        try:
            source_embeddings = self.retrieval_model.encode(source_sentences, convert_to_tensor=True)
            generated_embedding = self.retrieval_model.encode(generated_text, convert_to_tensor=True)
            
            # 计算余弦相似度
            from sentence_transformers import util
            similarities = util.cos_sim(generated_embedding, source_embeddings)[0]
            
            # 3. 选择top-k最相关的句子
            top_k_indices = torch.topk(similarities, min(top_k, len(source_sentences))).indices
            
            # 4. 按原文顺序组合证据
            top_sentences = [source_sentences[i] for i in sorted(top_k_indices.cpu().numpy())]
            evidence = ' '.join(top_sentences)
            
            # 5. 如果证据太长，再截断
            words = evidence.split()
            if len(words) > max_evidence_tokens:
                evidence = ' '.join(words[:max_evidence_tokens])
            
            return evidence
            
        except Exception as e:
            print(f"  检索失败: {str(e)[:50]}, 使用截断")
            return source_text[:max_evidence_tokens * 4]
    
    def nli_predict(self, premise, hypothesis):
        """
        NLI 预测
        """
        inputs = self.nli_tokenizer(
            premise,
            hypothesis,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
        
        pred_label_id = torch.argmax(probs).item()
        pred_label = self.label_mapping[pred_label_id]
        
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
    
    def detect_hallucination(self, source_text, generated_text, task_type='Unknown',
                             threshold=0.5, use_entailment=True, 
                             retrieve_evidence=True, top_k=3):
        """
        [!! 关键修正 !!] - 逐句检测
        """
        
        # 1. 将 "生成文本" 拆分为句子
        generated_sentences = self.split_into_sentences(generated_text)
        
        if not generated_sentences:
            # 如果分句失败，回退到全文检测（作为保险）
            return self.nli_predict(source_text, generated_text)

        sentence_results = []
        any_hallucination = False

        for sent in generated_sentences:
            premise = source_text  # 默认前提是原文
            
            # 2. 对 "每一句" 进行检索（如果开启）
            if retrieve_evidence:
                premise = self.retrieve_relevant_evidence(source_text, sent, top_k=top_k)
            
            # 3. 对 "每一句" 进行NLI验证
            result = self.nli_predict(premise, sent)
            result['evidence_used'] = premise
            result['evidence_retrieved'] = retrieve_evidence

            if use_entailment:
                is_hallucination = result['entailment_score'] < threshold
            else:
                is_hallucination = result['contradiction_score'] > threshold
            
            result['is_hallucination'] = is_hallucination
            sentence_results.append(result)
            
            if is_hallucination:
                any_hallucination = True
        
        # 4. 聚合结果（使用"最差句子"逻辑）
        if not sentence_results: # 再次检查
            return False, self.nli_predict(source_text, generated_text)
            
        if any_hallucination:
            # 找最能证明"幻觉"的句子
            if use_entailment:
                worst_result = min(sentence_results, key=lambda x: x['entailment_score'])
            else:
                worst_result = max(sentence_results, key=lambda x: x['contradiction_score'])
        else:
            # 找最能证明"无幻觉"的句子
            if use_entailment:
                worst_result = max(sentence_results, key=lambda x: x['entailment_score'])
            else:
                worst_result = min(sentence_results, key=lambda x: x['contradiction_score'])

        # 添加句子级详情
        worst_result['sentence_level_details'] = {
            'num_sentences': len(generated_sentences),
            'num_hallucination_sentences': sum(1 for r in sentence_results if r['is_hallucination']),
            'all_sentence_results': sentence_results 
        }

        return any_hallucination, worst_result


def quick_test(input_file, output_file, gpu_id=0, sample_size=500, top_k=3):
    """
    快速测试检索增强的效果
    """
    print("=" * 80)
    print("基于检索的 NLI 检测器 - 快速测试")
    print("=" * 80)
    print(f"样本数: {sample_size}")
    print(f"Top-K 检索: {top_k}")
    print("=" * 80)
    
    # 初始化
    detector = RetrievalAugmentedNLI(gpu_id=gpu_id)
    
    # 统计
    total = 0
    tp = fp = fn = tn = 0
    retrieved_count = 0
    
    contradiction_scores_hall = []
    contradiction_scores_no_hall = []
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()[:sample_size]
        
        for line in tqdm(lines, desc="检索增强NLI"):
            total += 1
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
            has_label = len(data.get('label_types', [])) > 0
            
            if not isinstance(text_original, str): text_original = str(text_original)
            if not isinstance(text_generated, str): text_generated = str(text_generated)
            
            if not text_original.strip() or not text_generated.strip():
                continue
            
            try:
                # 检索增强 NLI
                detected, result = detector.detect_hallucination(
                    text_original,
                    text_generated,
                    task_type,
                    threshold=0.5,
                    use_entailment=True,      # [!! 修正3 !!] - 明确使用蕴含
                    retrieve_evidence=True,
                    top_k=top_k
                )
                if result['evidence_retrieved']:
                    retrieved_count += 1
                
                # 统计
                if has_label and detected: tp += 1
                elif has_label and not detected: fn += 1
                elif not has_label and detected: fp += 1
                else: tn += 1
                
                # 记录分数
                if has_label:
                    contradiction_scores_hall.append(result['contradiction_score'])
                else:
                    contradiction_scores_no_hall.append(result['contradiction_score'])
                
                # 保存结果
                fout.write(json.dumps({
                    'id': data.get('id'),
                    'task_type': task_type,
                    'has_label': has_label,
                    'detected': detected,
                    'contradiction_score': result['contradiction_score'],
                    'entailment_score': result['entailment_score'],
                    'nli_label': result['label'],
                    'evidence_retrieved': result['evidence_retrieved'],
                    'evidence': result.get('evidence_used', '')[:200]  # 只保存前200字符
                }, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n错误 (ID: {data.get('id')}): {str(e)[:100]}")
                continue
    
    # 计算指标
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 分数分布
    if contradiction_scores_hall and contradiction_scores_no_hall:
        hall_mean = np.mean(contradiction_scores_hall)
        no_hall_mean = np.mean(contradiction_scores_no_hall)
        separation = abs(hall_mean - no_hall_mean)
    else:
        hall_mean = no_hall_mean = separation = 0
    
    print("\n" + "=" * 80)
    print("【检索增强 NLI 结果】")
    print("=" * 80)
    print(f"样本数: {total}")
    print(f"检索样本: {retrieved_count} ({retrieved_count/total*100:.1f}%)")
    print(f"\n性能指标:")
    print(f"  准确率: {precision:.2f}%")
    print(f"  召回率: {recall:.2f}%")
    print(f"  F1分数: {f1:.2f}")
    print(f"\n混淆矩阵:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    print(f"\n分数分布:")
    print(f"  有幻觉样本: {hall_mean:.4f}")
    print(f"  无幻觉样本: {no_hall_mean:.4f}")
    print(f"  区分度: {separation:.4f}")
    print("=" * 80)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'separation': separation
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='检索增强 NLI 检测器')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--input', type=str, 
                        default='/home/xgq/Test/data/validation_set.jsonl',
                        help='输入文件')
    parser.add_argument('--output', type=str,
                        default='nli_retrieval_test.jsonl',
                        help='输出文件')
    parser.add_argument('--sample-size', type=int, default=500,
                        help='测试样本数（默认500）')
    parser.add_argument('--top-k', type=int, default=3,
                        help='检索top-k个相关句子（默认3）')
    parser.add_argument('--full-test', action='store_true',
                        help='测试不同的top-k值')
    
    args = parser.parse_args()
    
    if args.full_test:
        # 测试不同的 top-k 值
        print("=" * 80)
        print("测试不同的 Top-K 值")
        print("=" * 80)
        
        results = {}
        for k in [1, 2, 3, 5, 10]:
            print(f"\n\n{'='*80}")
            print(f"测试 Top-K = {k}")
            print("=" * 80)
            
            output = f"nli_retrieval_test_k{k}.jsonl"
            result = quick_test(
                input_file=args.input,
                output_file=output,
                gpu_id=args.gpu,
                sample_size=args.sample_size,
                top_k=k
            )
            results[k] = result
        
        # 总结
        print("\n\n" + "=" * 80)
        print("【Top-K 对比总结】")
        print("=" * 80)
        print(f"{'Top-K':<10} {'准确率':<12} {'召回率':<12} {'F1':<10} {'区分度':<10}")
        print("-" * 60)
        for k, r in results.items():
            print(f"{k:<10} {r['precision']:<12.2f} {r['recall']:<12.2f} {r['f1']:<10.2f} {r['separation']:<10.4f}")
        
        best_k = max(results.items(), key=lambda x: x[1]['f1'])[0]
        print(f"\n✓ 最佳 Top-K: {best_k} (F1={results[best_k]['f1']:.2f}%)")
        
    else:
        # 单次测试
        quick_test(
            input_file=args.input,
            output_file=args.output,
            gpu_id=args.gpu,
            sample_size=args.sample_size,
            top_k=args.top_k
        )
        
        print(f"\n下一步:")
        print(f"1. 如果准确率提升到 55%+，在完整验证集上运行")
        print(f"2. 优化阈值")
        print(f"3. 在测试集上评估")

