#!/usr/bin/env python3
import json
import re

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def analyze_length(input_file):
    response_word_counts = []
    response_char_counts = []
    response_sentence_counts = []
    sentence_word_counts = []
    
    task_stats = {'Summary': [], 'QA': [], 'Data2txt': []}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                response = data.get('response', '')
                task_type = data.get('task_type', 'Unknown')
                
                if not response:
                    continue
                
                # 字符和词数统计（粗略估算 tokens ≈ words * 1.3）
                char_len = len(response)
                word_len = len(response.split())
                response_word_counts.append(word_len)
                response_char_counts.append(char_len)
                
                # 句子数
                sentences = split_into_sentences(response)
                num_sentences = len(sentences)
                response_sentence_counts.append(num_sentences)
                
                # 单句词数
                for sent in sentences:
                    sent_words = len(sent.split())
                    sentence_word_counts.append(sent_words)
                
                # 按任务统计
                if task_type in task_stats:
                    task_stats[task_type].append(word_len)
                
            except:
                continue
    
    # 计算统计量
    def stats(data):
        if not data:
            return {}
        data_sorted = sorted(data)
        n = len(data_sorted)
        return {
            'mean': sum(data) / len(data),
            'median': data_sorted[n//2],
            'min': min(data),
            'max': max(data),
            'p95': data_sorted[int(n * 0.95)],
            'p99': data_sorted[int(n * 0.99)]
        }
    
    print("=" * 80)
    print("【生成文本（Response）长度统计】")
    print("=" * 80)
    
    s = stats(response_word_counts)
    print(f"\n词数统计（估算 tokens ≈ 词数 × 1.3）:")
    print(f"  样本数: {len(response_word_counts)}")
    print(f"  平均词数: {s['mean']:.2f} (估算 tokens: {s['mean']*1.3:.2f})")
    print(f"  中位数: {s['median']} (估算 tokens: {s['median']*1.3:.0f})")
    print(f"  最小值: {s['min']} (估算 tokens: {s['min']*1.3:.0f})")
    print(f"  最大值: {s['max']} (估算 tokens: {s['max']*1.3:.0f})")
    print(f"  95分位: {s['p95']:.0f} (估算 tokens: {s['p95']*1.3:.0f})")
    print(f"  99分位: {s['p99']:.0f} (估算 tokens: {s['p99']*1.3:.0f})")
    
    # 超长样本统计（估算）
    over_512 = sum(1 for w in response_word_counts if w * 1.3 > 512)
    over_400 = sum(1 for w in response_word_counts if w * 1.3 > 400)
    over_300 = sum(1 for w in response_word_counts if w * 1.3 > 300)
    
    print(f"\n  超长样本（估算）:")
    print(f"    > 300 tokens: {over_300} ({over_300/len(response_word_counts)*100:.2f}%)")
    print(f"    > 400 tokens: {over_400} ({over_400/len(response_word_counts)*100:.2f}%)")
    print(f"    > 512 tokens: {over_512} ({over_512/len(response_word_counts)*100:.2f}%)")
    
    # 句子数统计
    s_sent = stats(response_sentence_counts)
    print(f"\n句子数统计:")
    print(f"  平均句子数: {s_sent['mean']:.2f}")
    print(f"  中位数: {s_sent['median']}")
    print(f"  最大值: {s_sent['max']}")
    
    print("\n" + "=" * 80)
    print("【单句长度统计】")
    print("=" * 80)
    
    s_single = stats(sentence_word_counts)
    print(f"\n词数统计（估算 tokens ≈ 词数 × 1.3）:")
    print(f"  总句子数: {len(sentence_word_counts)}")
    print(f"  平均词数: {s_single['mean']:.2f} (估算 tokens: {s_single['mean']*1.3:.2f})")
    print(f"  中位数: {s_single['median']} (估算 tokens: {s_single['median']*1.3:.0f})")
    print(f"  最大值: {s_single['max']} (估算 tokens: {s_single['max']*1.3:.0f})")
    print(f"  95分位: {s_single['p95']:.0f} (估算 tokens: {s_single['p95']*1.3:.0f})")
    print(f"  99分位: {s_single['p99']:.0f} (估算 tokens: {s_single['p99']*1.3:.0f})")
    
    over_200_sent = sum(1 for w in sentence_word_counts if w * 1.3 > 200)
    over_100_sent = sum(1 for w in sentence_word_counts if w * 1.3 > 100)
    
    print(f"\n  超长单句（估算）:")
    print(f"    > 100 tokens: {over_100_sent} ({over_100_sent/len(sentence_word_counts)*100:.2f}%)")
    print(f"    > 200 tokens: {over_200_sent} ({over_200_sent/len(sentence_word_counts)*100:.2f}%)")
    
    # 按任务类型
    print("\n" + "=" * 80)
    print("【按任务类型统计】")
    print("=" * 80)
    
    for task, word_counts in task_stats.items():
        if word_counts:
            s_task = stats(word_counts)
            print(f"\n◆ {task}:")
            print(f"  样本数: {len(word_counts)}")
            print(f"  平均词数: {s_task['mean']:.2f} (估算 tokens: {s_task['mean']*1.3:.2f})")
            print(f"  最大词数: {s_task['max']} (估算 tokens: {s_task['max']*1.3:.0f})")
            over_512_task = sum(1 for w in word_counts if w * 1.3 > 512)
            if over_512_task > 0:
                print(f"  > 512 tokens: {over_512_task} ({over_512_task/len(word_counts)*100:.2f}%)")
    
    # 检索后估算
    print("\n" + "=" * 80)
    print("【检索增强后的长度估算】")
    print("=" * 80)
    print("\n假设：")
    print("  - 原文平均每句 25-30 words (32-39 tokens)")
    print("  - 检索 top-3 句: 3 × 30 = 90 tokens")
    print(f"  - 平均单句生成文本: {s_single['mean']:.2f} words ({s_single['mean']*1.3:.2f} tokens)")
    print(f"\n总长度（检索证据 + 单句生成）:")
    print(f"  平均: 90 + {s_single['mean']*1.3:.2f} = {90 + s_single['mean']*1.3:.2f} tokens")
    print(f"  99分位: 90 + {s_single['p99']*1.3:.2f} = {90 + s_single['p99']*1.3:.2f} tokens")
    print(f"  是否安全（< 512）？ {'✓ 是' if (90 + s_single['p99']*1.3) < 512 else '✗ 否'}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else '/home/xgq/Test/data/validation_set.jsonl'
    analyze_length(input_file)

