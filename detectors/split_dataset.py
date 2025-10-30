"""
数据集分割工具
将 test_response_label.jsonl 分割为验证集和测试集

标准做法：
- 验证集: 用于优化阈值、调参
- 测试集: 用于最终评估，不能用于调参

推荐比例：
- 验证集: 20-30%
- 测试集: 70-80%
"""

import json
import os
import random
from collections import defaultdict


def split_dataset(input_file, 
                  val_ratio=0.2,
                  output_val='validation_set.jsonl',
                  output_test='test_set.jsonl',
                  stratified=True,
                  random_seed=42):
    """
    分割数据集
    
    :param input_file: 输入文件
    :param val_ratio: 验证集比例（0-1）
    :param output_val: 验证集输出文件
    :param output_test: 测试集输出文件
    :param stratified: 是否分层采样（保持有/无幻觉比例一致）
    :param random_seed: 随机种子
    """
    print("=" * 80)
    print("数据集分割工具")
    print("=" * 80)
    print(f"输入文件: {input_file}")
    print(f"验证集比例: {val_ratio*100:.1f}%")
    print(f"测试集比例: {(1-val_ratio)*100:.1f}%")
    print(f"分层采样: {stratified}")
    print(f"随机种子: {random_seed}")
    print("=" * 80)
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 加载数据
    print("\n加载数据...")
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"✓ 加载了 {len(data)} 个样本")
    
    # 统计
    has_label_count = sum(1 for d in data if len(d.get('label_types', [])) > 0)
    no_label_count = len(data) - has_label_count
    
    print(f"  - 有幻觉: {has_label_count} ({has_label_count/len(data)*100:.2f}%)")
    print(f"  - 无幻觉: {no_label_count} ({no_label_count/len(data)*100:.2f}%)")
    
    # 按任务类型统计
    task_counts = defaultdict(int)
    for d in data:
        task_counts[d.get('task_type', 'Unknown')] += 1
    
    print(f"\n按任务类型:")
    for task, count in task_counts.items():
        print(f"  {task}: {count}")
    
    # 处理输出路径
    # 如果输出路径不是绝对路径，则保存到输入文件同目录
    if not os.path.isabs(output_val):
        input_dir = os.path.dirname(os.path.abspath(input_file))
        output_val = os.path.join(input_dir, output_val)
    
    if not os.path.isabs(output_test):
        input_dir = os.path.dirname(os.path.abspath(input_file))
        output_test = os.path.join(input_dir, output_test)
    
    print(f"\n输出路径:")
    print(f"  验证集: {output_val}")
    print(f"  测试集: {output_test}")
    
    # 分割数据
    if stratified:
        print("\n使用分层采样（保持比例一致）...")
        
        # 按有无幻觉分组
        has_label_data = [d for d in data if len(d.get('label_types', [])) > 0]
        no_label_data = [d for d in data if len(d.get('label_types', [])) == 0]
        
        # 打乱
        random.shuffle(has_label_data)
        random.shuffle(no_label_data)
        
        # 分割
        val_size_has = int(len(has_label_data) * val_ratio)
        val_size_no = int(len(no_label_data) * val_ratio)
        
        val_data = has_label_data[:val_size_has] + no_label_data[:val_size_no]
        test_data = has_label_data[val_size_has:] + no_label_data[val_size_no:]
        
        # 打乱验证集和测试集
        random.shuffle(val_data)
        random.shuffle(test_data)
        
    else:
        print("\n使用随机分割...")
        random.shuffle(data)
        val_size = int(len(data) * val_ratio)
        val_data = data[:val_size]
        test_data = data[val_size:]
    
    # 保存
    print(f"\n保存数据...")
    with open(output_val, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ 验证集: {output_val} ({len(val_data)} 样本)")
    
    with open(output_test, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"✓ 测试集: {output_test} ({len(test_data)} 样本)")
    
    # 验证比例
    val_has_label = sum(1 for d in val_data if len(d.get('label_types', [])) > 0)
    test_has_label = sum(1 for d in test_data if len(d.get('label_types', [])) > 0)
    
    print(f"\n【分割结果验证】")
    print(f"验证集:")
    print(f"  总数: {len(val_data)}")
    print(f"  有幻觉: {val_has_label} ({val_has_label/len(val_data)*100:.2f}%)")
    print(f"  无幻觉: {len(val_data)-val_has_label} ({(len(val_data)-val_has_label)/len(val_data)*100:.2f}%)")
    
    print(f"\n测试集:")
    print(f"  总数: {len(test_data)}")
    print(f"  有幻觉: {test_has_label} ({test_has_label/len(test_data)*100:.2f}%)")
    print(f"  无幻觉: {len(test_data)-test_has_label} ({(len(test_data)-test_has_label)/len(test_data)*100:.2f}%)")
    
    # 统计任务类型分布
    print(f"\n【任务类型分布】")
    val_tasks = defaultdict(int)
    test_tasks = defaultdict(int)
    
    for d in val_data:
        val_tasks[d.get('task_type', 'Unknown')] += 1
    for d in test_data:
        test_tasks[d.get('task_type', 'Unknown')] += 1
    
    print(f"{'任务类型':<15} {'验证集':<10} {'测试集':<10}")
    print("-" * 40)
    for task in val_tasks.keys():
        print(f"{task:<15} {val_tasks[task]:<10} {test_tasks[task]:<10}")
    
    print("\n" + "=" * 80)
    print("✓ 数据集分割完成！")
    print("=" * 80)
    print("\n下一步:")
    print("1. 在验证集上优化阈值:")
    print(f"   python nli_deberta_detector.py --gpu 0 --input {output_val}")
    print(f"   python nli_threshold_optimizer.py --results nli_deberta_results.jsonl")
    print("\n2. 使用最优阈值在测试集上评估:")
    print(f"   python nli_deberta_detector.py --gpu 0 --input {output_test} --threshold [最优值]")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集分割工具')
    parser.add_argument('--input', type=str, 
                        default='../data/test_response_label.jsonl',
                        help='输入文件')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='验证集比例 (默认: 0.2 = 20%%)')
    parser.add_argument('--output-val', type=str, 
                        default='validation_set.jsonl',
                        help='验证集输出文件（默认保存在输入文件同目录）')
    parser.add_argument('--output-test', type=str,
                        default='test_set.jsonl',
                        help='测试集输出文件（默认保存在输入文件同目录）')
    parser.add_argument('--no-stratified', action='store_true',
                        help='禁用分层采样（默认启用）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    split_dataset(
        input_file=args.input,
        val_ratio=args.val_ratio,
        output_val=args.output_val,
        output_test=args.output_test,
        stratified=not args.no_stratified,
        random_seed=args.seed
    )

