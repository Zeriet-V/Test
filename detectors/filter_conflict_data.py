"""
筛选矛盾型幻觉数据
专门测试 NLI 在 Conflict 类型上的表现
"""

import json
from collections import defaultdict


def filter_conflict_samples(input_file, output_conflict, output_stats='conflict_data_stats.txt'):
    """
    筛选包含 Conflict 标签的样本
    
    :param input_file: 输入文件
    :param output_conflict: 输出文件（只包含Conflict）
    """
    print("=" * 80)
    print("筛选矛盾型幻觉数据")
    print("=" * 80)
    print(f"输入: {input_file}")
    print(f"输出: {output_conflict}")
    print("=" * 80)
    
    # 统计
    total_count = 0
    has_conflict = 0
    no_conflict = 0
    no_label = 0
    
    conflict_samples = []
    no_conflict_samples = []
    no_label_samples = []
    
    # 按类型统计
    label_counts = defaultdict(int)
    task_counts = defaultdict(int)
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_count += 1
            data = json.loads(line)
            
            label_types = data.get('label_types', [])
            task_type = data.get('task_type', 'Unknown')
            
            # 检查是否包含 Conflict
            has_conflict_label = any('Conflict' in label for label in label_types)
            
            if label_types:  # 有标签
                if has_conflict_label:
                    has_conflict += 1
                    conflict_samples.append(data)
                    task_counts[f"{task_type}_conflict"] += 1
                else:
                    no_conflict += 1
                    no_conflict_samples.append(data)
                    task_counts[f"{task_type}_baseless"] += 1
                
                # 统计每种标签
                for label in label_types:
                    label_counts[label] += 1
            else:  # 无标签
                no_label += 1
                no_label_samples.append(data)
                task_counts[f"{task_type}_no_label"] += 1
    
    # 保存 Conflict 样本 + 无幻觉样本（用于完整评估）
    with open(output_conflict, 'w', encoding='utf-8') as f:
        # 1. 保存所有 Conflict 样本
        for data in conflict_samples:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # 2. 保存所有无幻觉样本
        for data in no_label_samples:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 保存统计报告
    with open(output_stats, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("矛盾型幻觉数据统计\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【总体统计】\n")
        f.write(f"  总样本数: {total_count}\n")
        f.write(f"  - 包含 Conflict: {has_conflict} ({has_conflict/total_count*100:.2f}%)\n")
        f.write(f"  - 只有 Baseless: {no_conflict} ({no_conflict/total_count*100:.2f}%)\n")
        f.write(f"  - 无标签: {no_label} ({no_label/total_count*100:.2f}%)\n\n")
        
        f.write("【幻觉类型分布】\n")
        for label, count in sorted(label_counts.items()):
            f.write(f"  {label}: {count}\n")
        
        f.write("\n【按任务类型统计】\n")
        for task, count in sorted(task_counts.items()):
            f.write(f"  {task}: {count}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"✓ Conflict 数据已保存到: {output_conflict}\n")
        f.write("=" * 80 + "\n")
    
    # 打印统计
    print(f"\n【筛选结果】")
    print(f"  总样本: {total_count}")
    print(f"  包含 Conflict: {has_conflict} ({has_conflict/total_count*100:.2f}%)")
    print(f"  只有 Baseless: {no_conflict} ({no_conflict/total_count*100:.2f}%)")
    print(f"  无标签: {no_label} ({no_label/total_count*100:.2f}%)")
    
    print(f"\n【Conflict 数据集组成】")
    print(f"  有幻觉(Conflict): {has_conflict}")
    print(f"  无幻觉: {no_label}")
    print(f"  总计: {has_conflict + no_label}")
    print(f"  幻觉比例: {has_conflict/(has_conflict+no_label)*100:.2f}%")
    
    print(f"\n✓ Conflict 数据已保存到: {output_conflict}")
    print(f"✓ 统计报告已保存到: {output_stats}")
    
    return has_conflict, no_label


def create_conflict_validation_test(validation_file, test_file, output_dir='./'):
    """
    为验证集和测试集分别创建 Conflict 子集
    """
    print("\n" + "=" * 80)
    print("创建 Conflict 子集（验证集 + 测试集）")
    print("=" * 80)
    
    # 验证集
    print("\n处理验证集...")
    val_conflict, val_no_label = filter_conflict_samples(
        validation_file,
        output_conflict=output_dir + 'validation_conflict_only.jsonl',
        output_stats=output_dir + 'validation_conflict_stats.txt'
    )
    
    # 测试集
    print("\n处理测试集...")
    test_conflict, test_no_label = filter_conflict_samples(
        test_file,
        output_conflict=output_dir + 'test_conflict_only.jsonl',
        output_stats=output_dir + 'test_conflict_stats.txt'
    )
    
    # 总结
    print("\n" + "=" * 80)
    print("【Conflict 子集创建完成】")
    print("=" * 80)
    print(f"\n验证集 Conflict 子集:")
    print(f"  文件: validation_conflict_only.jsonl")
    print(f"  有Conflict: {val_conflict}")
    print(f"  无幻觉: {val_no_label}")
    print(f"  总计: {val_conflict + val_no_label}")
    
    print(f"\n测试集 Conflict 子集:")
    print(f"  文件: test_conflict_only.jsonl")
    print(f"  有Conflict: {test_conflict}")
    print(f"  无幻觉: {test_no_label}")
    print(f"  总计: {test_conflict + test_no_label}")
    
    print("\n下一步:")
    print("1. 在 Conflict 验证集上测试 NLI:")
    print("   cd /home/xgq/Test/detectors/nli_methods")
    print("   python nli_deberta_detector.py --gpu 0 --input validation_conflict_only.jsonl --use-contradiction")
    print("\n2. 优化阈值")
    print("\n3. 在 Conflict 测试集上评估")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='筛选矛盾型幻觉数据')
    parser.add_argument('--validation', type=str,
                        default='/home/xgq/Test/data/validation_set.jsonl',
                        help='验证集路径')
    parser.add_argument('--test', type=str,
                        default='/home/xgq/Test/data/test_set.jsonl',
                        help='测试集路径')
    parser.add_argument('--output-dir', type=str,
                        default='/home/xgq/Test/data/',
                        help='输出目录')
    
    args = parser.parse_args()
    
    create_conflict_validation_test(
        validation_file=args.validation,
        test_file=args.test,
        output_dir=args.output_dir
    )


