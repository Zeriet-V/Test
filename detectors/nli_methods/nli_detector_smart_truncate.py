"""
NLI 检测器 - 智能截断版本
解决输入文本过长导致的准确率低问题

核心改进:
1. 智能截取原文关键部分（而非全部）
2. 对不同任务使用不同截取策略
3. 确保输入在 NLI 模型的有效范围内
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

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def smart_truncate_source(source_text, generated_text, task_type, max_length=400):
    """
    智能截取原文
    
    策略:
    - Summary: 保留原文开头和与生成文本相关的部分
    - QA: 优先保留question和相关passage片段
    - Data2txt: 保留最相关的字段
    
    :param source_text: 原文
    :param generated_text: 生成文本
    :param task_type: 任务类型
    :param max_length: 最大字符数
    :return: 截取后的原文
    """
    if len(source_text) <= max_length:
        return source_text
    
    # 策略1: 保留开头 + 与生成文本相关的部分
    # 简单实现: 取前 max_length 字符
    # 高级实现: 可以用 TF-IDF 找最相关的句子
    
    if task_type == 'Summary':
        # 摘要任务: 保留原文开头部分
        return source_text[:max_length]
    
    elif task_type == 'QA':
        # QA任务: 尝试保留 question 和部分 passage
        # 假设格式是 "question passage"
        parts = source_text.split(maxsplit=1)
        if len(parts) == 2:
            question, passage = parts
            # 保留完整question + 部分passage
            if len(question) < max_length:
                remaining = max_length - len(question) - 1
                return question + " " + passage[:remaining]
        return source_text[:max_length]
    
    elif task_type == 'Data2txt':
        # Data2txt: 保留前面的关键字段
        return source_text[:max_length]
    
    else:
        return source_text[:max_length]


class SmartNLIDetector:
    """
    智能 NLI 检测器（带截断）
    """
    
    def __init__(self, 
                 model_name='MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 gpu_id=None,
                 max_source_length=400):
        """
        :param max_source_length: 原文最大长度（字符数）
        """
        if gpu_id is not None and torch.cuda.is_available():
            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
            print(f"指定使用GPU: {gpu_id}")
        
        print(f"加载 DeBERTa-NLI 模型: {model_name}")
        print(f"使用设备: {device}")
        print(f"原文最大长度: {max_source_length} 字符")
        
        self.device = device
        self.model_name = model_name
        self.max_source_length = max_source_length
        
        # 加载模型
        cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
        model_cache = os.path.join(cache_dir, f'models--{model_name.replace("/", "--")}')
        
        try:
            if os.path.exists(model_cache):
                print("检测到本地缓存，尝试离线加载...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                print("✓ 离线加载成功！")
            else:
                print("本地无缓存，开始在线下载...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("✓ 在线下载并加载成功！")
        except Exception as e:
            print(f"⚠ 离线加载失败，尝试在线下载...")
            import shutil
            if os.path.exists(model_cache):
                shutil.rmtree(model_cache, ignore_errors=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("✓ 在线下载并加载成功！")
        
        self.model.eval()
        self.model.to(self.device)
        
        # 标签映射
        self.label_mapping = {
            0: 'contradiction',
            1: 'neutral',
            2: 'entailment'
        }
        
        print("DeBERTa-NLI 模型加载成功！")
    
    def predict(self, premise, hypothesis, task_type='Unknown'):
        """
        预测 NLI 关系（带智能截断）
        """
        # 智能截取原文
        premise_truncated = smart_truncate_source(premise, hypothesis, task_type, self.max_source_length)
        
        # Tokenize
        inputs = self.tokenizer(
            premise_truncated,
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
            'entailment_score': scores['entailment'],
            'premise_truncated': len(premise_truncated) < len(premise)
        }
    
    def detect_hallucination(self, source_text, generated_text, task_type, threshold=0.5, use_contradiction=True):
        """
        检测幻觉
        
        :param use_contradiction: True=使用矛盾分数, False=使用蕴含分数
        """
        result = self.predict(source_text, generated_text, task_type)
        
        if use_contradiction:
            has_hallucination = result['contradiction_score'] > threshold
        else:
            has_hallucination = result['entailment_score'] < threshold
        
        return has_hallucination, result


# 简化版处理函数（仅用于测试）
def test_smart_truncate(input_file, output_file, gpu_id=0, max_source_length=400):
    """
    测试智能截断版本
    """
    print(f"\n【智能截断 NLI 检测器】")
    print("=" * 80)
    print(f"原文最大长度: {max_source_length} 字符")
    print("=" * 80)
    
    detector = SmartNLIDetector(gpu_id=gpu_id, max_source_length=max_source_length)
    
    # 统计
    total = 0
    tp = fp = fn = tn = 0
    truncated_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        lines = fin.readlines()
        
        for line in tqdm(lines[:500], desc="测试中（前500样本）"):  # 先测试500个
            total += 1
            data = json.loads(line)
            
            # 解析数据
            test_data = data.get('test', '')
            task_type = data.get('task_type', 'Unknown')
            
            # 简化的文本提取
            if isinstance(test_data, dict):
                text_original = str(test_data)
            else:
                text_original = test_data
            
            text_generated = data.get('response', '')
            has_label = len(data.get('label_types', [])) > 0
            
            if not text_original or not text_generated:
                continue
            
            try:
                detected, result = detector.detect_hallucination(
                    text_original, text_generated, task_type,
                    threshold=0.5, use_contradiction=True
                )
                
                if result['premise_truncated']:
                    truncated_count += 1
                
                if has_label and detected: tp += 1
                elif has_label and not detected: fn += 1
                elif not has_label and detected: fp += 1
                else: tn += 1
                
                fout.write(json.dumps({
                    'id': data.get('id'),
                    'has_label': has_label,
                    'detected': detected,
                    'contradiction_score': result['contradiction_score'],
                    'truncated': result['premise_truncated']
                }, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n错误: {str(e)[:100]}")
                continue
    
    # 计算指标
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n结果（前500样本）:")
    print(f"  准确率: {precision:.2f}%")
    print(f"  召回率: {recall:.2f}%")
    print(f"  F1分数: {f1:.2f}")
    print(f"  截断样本: {truncated_count}/{total} ({truncated_count/total*100:.1f}%)")


if __name__ == "__main__":
    # 测试不同的截断长度
    for max_len in [200, 400, 800, 1200]:
        print(f"\n\n{'='*80}")
        print(f"测试 max_length = {max_len}")
        print("=" * 80)
        
        test_smart_truncate(
            input_file='/home/xgq/Test/data/validation_set.jsonl',
            output_file=f'test_truncate_{max_len}.jsonl',
            gpu_id=0,
            max_source_length=max_len
        )


