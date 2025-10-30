"""
筛选 Summary 和 QA 任务的 Conflict 数据
排除 Data2txt（格式问题）
"""

import json


def filter_summary_qa_conflict(input_file, output_file):
    """
    筛选 Summary 和 QA 任务中的 Conflict 样本
    """
    print("=" * 80)
    print("筛选 Summary/QA 任务的 Conflict 数据")
    print("=" * 80)
    
    summary_conflict = []
    qa_conflict = []
    summary_no_label = []
    qa_no_label = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            task_type = data.get('task_type')
            label_types = data.get('label_types', [])
            
            # 只保留 Summary 和 QA
            if task_type not in ['Summary', 'QA']:
                continue
            
            has_conflict = any('Conflict' in label for label in label_types)
            
            if has_conflict:
                if task_type == 'Summary':
                    summary_conflict.append(data)
                else:
                    qa_conflict.append(data)
            elif not label_types:  # 无标签
                if task_type == 'Summary':
                    summary_no_label.append(data)
                else:
                    qa_no_label.append(data)
    
    # 合并
    all_samples = summary_conflict + qa_conflict + summary_no_label + qa_no_label
    
    # 保存
    with open(output_file, 'w', encoding='utf-8') as f:
        for data in all_samples:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 统计
    total_conflict = len(summary_conflict) + len(qa_conflict)
    total_no_label = len(summary_no_label) + len(qa_no_label)
    total = total_conflict + total_no_label
    
    print(f"\n【筛选结果】")
    print(f"  Summary Conflict: {len(summary_conflict)}")
    print(f"  QA Conflict: {len(qa_conflict)}")
    print(f"  Summary 无标签: {len(summary_no_label)}")
    print(f"  QA 无标签: {len(qa_no_label)}")
    print(f"\n  总计:")
    print(f"    有Conflict: {total_conflict} ({total_conflict/total*100:.2f}%)")
    print(f"    无幻觉: {total_no_label} ({total_no_label/total*100:.2f}%)")
    print(f"    总样本: {total}")
    
    print(f"\n✓ 已保存到: {output_file}")
    
    return total_conflict, total_no_label


if __name__ == "__main__":
    print("\n处理验证集...")
    val_conflict, val_no = filter_summary_qa_conflict(
        '/home/xgq/Test/data/validation_set.jsonl',
        '/home/xgq/Test/data/validation_summary_qa_conflict.jsonl'
    )
    
    print("\n" + "=" * 80)
    print("\n处理测试集...")
    test_conflict, test_no = filter_summary_qa_conflict(
        '/home/xgq/Test/data/test_set.jsonl',
        '/home/xgq/Test/data/test_summary_qa_conflict.jsonl'
    )
    
    print("\n" + "=" * 80)
    print("【完成】")
    print("=" * 80)
    print(f"\n验证集 (Summary+QA Conflict): {val_conflict + val_no} 样本")
    print(f"测试集 (Summary+QA Conflict): {test_conflict + test_no} 样本")
    
    print("\n下一步:")
    print("  cd /home/xgq/Test/detectors/nli_methods")
    print("  python nli_deberta_detector.py --gpu 0 \\")
    print("         --input /home/xgq/Test/data/validation_summary_qa_conflict.jsonl \\")
    print("         --use-contradiction")

