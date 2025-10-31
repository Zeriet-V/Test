#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分析验证集中生成文本的长度分布
"""

import json
import numpy as np
from transformers import AutoTokenizer
import re

def split_into_sentences(text):
    """分句"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def analyze_text_length(input_file):
    """分析文本长度"""
    
    # 加载 tokenizer
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
        local_files_only=True
    )
    
    print(f"分析文件: {input_file}\n")
    
    # 统计变量
    response_tokens = []
    response_chars = []
    response_sentences = []
    sentence_tokens = []  # 单句的 token 长度
    
    task_stats = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"总样本数: {len(lines)}")
    
    for line in lines:
        try:
            data = json.loads(line)
            response = data.get('response', '')
            task_type = data.get('task_type', 'Unknown')
            
            if not response:
                continue
            
            # 字符长度
            char_len = len(response)
            response_chars.append(char_len)
            
            # Token 长度
            tokens = tokenizer.encode(response, add_special_tokens=False)
            token_len = len(tokens)
            response_tokens.append(token_len)
            
            # 句子数
            sentences = split_into_sentences(response)
            num_sentences = len(sentences)
            response_sentences.append(num_sentences)
            
            # 每个句子的 token 长度
            for sent in sentences:
                sent_tokens = tokenizer.encode(sent, add_special_tokens=False)
                sentence_tokens.append(len(sent_tokens))
            
            # 按任务统计
            if task_type not in task_stats:
                task_stats[task_type] = {
                    'count': 0,
                    'tokens': [],
                    'sentences': []
                }
            task_stats[task_type]['count'] += 1
            task_stats[task_type]['tokens'].append(token_len)
            task_stats[task_type]['sentences'].append(num_sentences)
            
        except Exception as e:
            continue
    
    # 打印统计结果
    print("\n" + "=" * 80)
    print("【生成文本长度统计】")
    print("=" * 80)
    
    print("\n整体统计（Response）:")
    print(f"  样本数: {len(response_tokens)}")
    print(f"\n  Token 长度:")
    print(f"    平均: {np.mean(response_tokens):.2f}")
    print(f"    中位数: {np.median(response_tokens):.2f}")
    print(f"    最小值: {np.min(response_tokens)}")
    print(f"    最大值: {np.max(response_tokens)}")
    print(f"    标准差: {np.std(response_tokens):.2f}")
    print(f"    95分位: {np.percentile(response_tokens, 95):.2f}")
    print(f"    99分位: {np.percentile(response_tokens, 99):.2f}")
    
    print(f"\n  字符长度:")
    print(f"    平均: {np.mean(response_chars):.2f}")
    print(f"    最大值: {np.max(response_chars)}")
    
    print(f"\n  句子数:")
    print(f"    平均: {np.mean(response_sentences):.2f}")
    print(f"    中位数: {np.median(response_sentences):.0f}")
    print(f"    最大值: {np.max(response_sentences)}")
    
    # 超长样本统计
    over_512 = sum(1 for t in response_tokens if t > 512)
    over_400 = sum(1 for t in response_tokens if t > 400)
    over_300 = sum(1 for t in response_tokens if t > 300)
    
    print(f"\n  超长样本:")
    print(f"    > 300 tokens: {over_300} ({over_300/len(response_tokens)*100:.2f}%)")
    print(f"    > 400 tokens: {over_400} ({over_400/len(response_tokens)*100:.2f}%)")
    print(f"    > 512 tokens: {over_512} ({over_512/len(response_tokens)*100:.2f}%)")
    
    # 单句长度统计
    print("\n" + "=" * 80)
    print("【单句长度统计】")
    print("=" * 80)
    print(f"  总句子数: {len(sentence_tokens)}")
    print(f"  平均 token 长度: {np.mean(sentence_tokens):.2f}")
    print(f"  中位数: {np.median(sentence_tokens):.2f}")
    print(f"  最大值: {np.max(sentence_tokens)}")
    print(f"  95分位: {np.percentile(sentence_tokens, 95):.2f}")
    print(f"  99分位: {np.percentile(sentence_tokens, 99):.2f}")
    
    over_200_sent = sum(1 for t in sentence_tokens if t > 200)
    over_100_sent = sum(1 for t in sentence_tokens if t > 100)
    
    print(f"\n  超长单句:")
    print(f"    > 100 tokens: {over_100_sent} ({over_100_sent/len(sentence_tokens)*100:.2f}%)")
    print(f"    > 200 tokens: {over_200_sent} ({over_200_sent/len(sentence_tokens)*100:.2f}%)")
    
    # 按任务类型统计
    print("\n" + "=" * 80)
    print("【按任务类型统计】")
    print("=" * 80)
    
    for task, stats in task_stats.items():
        if stats['count'] > 0:
            print(f"\n◆ {task}:")
            print(f"  样本数: {stats['count']}")
            print(f"  平均 token 长度: {np.mean(stats['tokens']):.2f}")
            print(f"  最大 token 长度: {np.max(stats['tokens'])}")
            print(f"  平均句子数: {np.mean(stats['sentences']):.2f}")
            
            over_512_task = sum(1 for t in stats['tokens'] if t > 512)
            if over_512_task > 0:
                print(f"  > 512 tokens: {over_512_task} ({over_512_task/stats['count']*100:.2f}%)")
    
    # 检索后的长度估算
    print("\n" + "=" * 80)
    print("【检索增强后的长度估算】")
    print("=" * 80)
    print("\n假设原文平均每句 30 tokens，检索 top-3：")
    print(f"  检索证据长度: 3 × 30 = 90 tokens")
    print(f"  单句生成文本: {np.mean(sentence_tokens):.2f} tokens")
    print(f"  总长度（证据+生成句）: {90 + np.mean(sentence_tokens):.2f} tokens")
    print(f"  是否超过 512？ {'否 ✓' if (90 + np.mean(sentence_tokens)) < 512 else '是 ✗'}")
    
    print("\n最坏情况（99分位单句 + top-3 证据）：")
    worst_sent = np.percentile(sentence_tokens, 99)
    print(f"  检索证据: 90 tokens")
    print(f"  99分位单句: {worst_sent:.2f} tokens")
    print(f"  总长度: {90 + worst_sent:.2f} tokens")
    print(f"  是否超过 512？ {'否 ✓' if (90 + worst_sent) < 512 else '是 ✗'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, 
                        default='/home/xgq/Test/data/validation_set.jsonl',
                        help='输入文件')
    
    args = parser.parse_args()
    
    analyze_text_length(args.input)

