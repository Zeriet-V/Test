import json
from collections import Counter

# 读取 response.jsonl 文件并统计 label_type
label_type_count = 0
label_types = []

with open('response.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        labels = data.get('labels', [])
        for label in labels:
            if 'label_type' in label:
                label_type_count += 1
                label_types.append(label['label_type'])

# 输出结果
print(f"总共有 {label_type_count} 个 label_type")
print(f"\nlabel_type 分布:")
counter = Counter(label_types)
for label_type, count in counter.most_common():
    print(f"  {label_type}: {count}")
