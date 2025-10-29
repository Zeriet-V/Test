"""
分析假阴性样本特征
找出为什么这些有幻觉的样本未被检测到
"""

import json

# 从报告中提取的假阴性样本ID
fn_sample_ids = ['2', '5', '10', '28', '29', '33', '40', '44', '46', '51']

print("=" * 80)
print("假阴性样本详细分析")
print("=" * 80)

with open('../data/test_response_label.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        if data['id'] in fn_sample_ids:
            print(f"\n{'='*80}")
            print(f"样本ID: {data['id']}")
            print(f"任务类型: {data['task_type']}")
            print(f"幻觉标签: {data['label_types']}")
            print(f"\n原文:")
            test_data = data.get('test', '')
            if isinstance(test_data, dict):
                print(str(test_data)[:500])
            else:
                print(str(test_data)[:500])
            print(f"\n生成文本:")
            print(str(data.get('response', ''))[:500])
            print(f"\n{'='*80}")
            
            input("按Enter查看下一个样本...")

print("\n分析完成！")
print("\n观察要点：")
print("1. 这些样本的生成文本是否与原文很相似？")
print("2. 幻觉内容是否很微妙？")
print("3. 是否有特定的幻觉模式？")





