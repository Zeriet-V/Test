"""
预测AND逻辑（交叉验证）的性能表现
基于SVO和N-gram的现有结果估算
"""

import json

def predict_and_logic_performance():
    """
    基于现有SVO和N-gram结果，预测AND逻辑的性能
    """
    print("=" * 80)
    print("AND逻辑（交叉验证）性能预测")
    print("=" * 80)
    print("\n【策略说明】")
    print("AND逻辑：只有当SVO和N-gram都检测到时，才标记为幻觉")
    print("目标：降低N-gram的误报，提高准确率")
    print()
    
    # 读取SVO结果
    svo_results = {}
    with open('spacy_results.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_id = data.get('id', '')
            has_label = data.get('has_label', False)
            detected = len(data.get('contradictions', [])) > 0
            svo_results[data_id] = {'has_label': has_label, 'detected': detected}
    
    # 读取N-gram结果
    ngram_results = {}
    with open('ngram_results_v2.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data_id = data.get('id', '')
            has_label = data.get('has_label', False)
            detected = data.get('detected', False)
            ngram_results[data_id] = {'has_label': has_label, 'detected': detected}
    
    # 统计
    total = 0
    has_label_count = 0
    no_label_count = 0
    
    # AND逻辑统计
    and_detected = 0
    and_tp = 0  # 真阳性
    and_fp = 0  # 假阳性
    and_fn = 0  # 假阴性
    and_tn = 0  # 真阴性
    
    # 各方法检测统计
    svo_only = 0
    ngram_only = 0
    both_detected = 0
    neither = 0
    
    # 对比N-gram单独使用
    ngram_tp = 0
    ngram_fp = 0
    ngram_fn = 0
    ngram_tn = 0
    
    # 遍历所有样本
    for data_id in svo_results.keys():
        if data_id not in ngram_results:
            continue
        
        total += 1
        has_label = svo_results[data_id]['has_label']
        svo_det = svo_results[data_id]['detected']
        ngram_det = ngram_results[data_id]['detected']
        
        if has_label:
            has_label_count += 1
        else:
            no_label_count += 1
        
        # AND逻辑：两者都检测到才标记
        and_det = svo_det and ngram_det
        
        if and_det:
            and_detected += 1
        
        # 统计检测来源
        if svo_det and ngram_det:
            both_detected += 1
        elif svo_det:
            svo_only += 1
        elif ngram_det:
            ngram_only += 1
        else:
            neither += 1
        
        # AND逻辑混淆矩阵
        if has_label and and_det:
            and_tp += 1
        elif has_label and not and_det:
            and_fn += 1
        elif not has_label and and_det:
            and_fp += 1
        else:
            and_tn += 1
        
        # N-gram单独使用混淆矩阵
        if has_label and ngram_det:
            ngram_tp += 1
        elif has_label and not ngram_det:
            ngram_fn += 1
        elif not has_label and ngram_det:
            ngram_fp += 1
        else:
            ngram_tn += 1
    
    # 计算性能指标
    and_precision = and_tp / (and_tp + and_fp) * 100 if (and_tp + and_fp) > 0 else 0
    and_recall = and_tp / (and_tp + and_fn) * 100 if (and_tp + and_fn) > 0 else 0
    and_f1 = 2 * and_precision * and_recall / (and_precision + and_recall) if (and_precision + and_recall) > 0 else 0
    
    ngram_precision = ngram_tp / (ngram_tp + ngram_fp) * 100 if (ngram_tp + ngram_fp) > 0 else 0
    ngram_recall = ngram_tp / (ngram_tp + ngram_fn) * 100 if (ngram_tp + ngram_fn) > 0 else 0
    ngram_f1 = 2 * ngram_precision * ngram_recall / (ngram_precision + ngram_recall) if (ngram_precision + ngram_recall) > 0 else 0
    
    # 打印结果
    print("【数据集统计】")
    print(f"总样本数: {total}")
    print(f"有幻觉样本: {has_label_count} ({has_label_count/total*100:.2f}%)")
    print(f"无幻觉样本: {no_label_count} ({no_label_count/total*100:.2f}%)")
    print()
    
    print("【各方法检测统计】")
    print(f"仅SVO检测到: {svo_only}")
    print(f"仅N-gram检测到: {ngram_only}")
    print(f"两者都检测到: {both_detected}")
    print(f"两者都未检测到: {neither}")
    print()
    
    print("=" * 80)
    print("【性能对比：N-gram单独 vs AND逻辑（交叉验证）】")
    print("=" * 80)
    print()
    
    print(f"{'指标':<20} {'N-gram单独':<20} {'AND逻辑':<20} {'变化':<20}")
    print("-" * 80)
    print(f"{'检测数量':<20} {ngram_tp + ngram_fp:<20} {and_tp + and_fp:<20} {(and_tp + and_fp) - (ngram_tp + ngram_fp):<20}")
    print(f"{'真阳性(TP)':<20} {ngram_tp:<20} {and_tp:<20} {and_tp - ngram_tp:<20}")
    print(f"{'假阳性(FP)':<20} {ngram_fp:<20} {and_fp:<20} {and_fp - ngram_fp:<20}")
    print(f"{'假阴性(FN)':<20} {ngram_fn:<20} {and_fn:<20} {and_fn - ngram_fn:<20}")
    print(f"{'真阴性(TN)':<20} {ngram_tn:<20} {and_tn:<20} {and_tn - ngram_tn:<20}")
    print()
    print(f"{'准确率(Precision)':<20} {ngram_precision:>18.2f}% {and_precision:>18.2f}% {and_precision - ngram_precision:>+18.2f}%")
    print(f"{'召回率(Recall)':<20} {ngram_recall:>18.2f}% {and_recall:>18.2f}% {and_recall - ngram_recall:>+18.2f}%")
    print(f"{'F1分数':<20} {ngram_f1:>18.2f} {and_f1:>18.2f} {and_f1 - ngram_f1:>+18.2f}")
    print()
    
    print("=" * 80)
    print("【关键发现】")
    print("=" * 80)
    print()
    
    # 准确率提升
    precision_gain = and_precision - ngram_precision
    recall_loss = ngram_recall - and_recall
    fp_reduction = ngram_fp - and_fp
    fp_reduction_rate = fp_reduction / ngram_fp * 100 if ngram_fp > 0 else 0
    
    print(f"1. 误报控制效果:")
    print(f"   ✓ 假阳性(FP)减少: {fp_reduction} 个 ({fp_reduction_rate:.1f}%)")
    print(f"   ✓ 准确率提升: {precision_gain:+.2f}%")
    print(f"   → AND逻辑有效过滤了 {fp_reduction} 个N-gram的误报")
    print()
    
    print(f"2. 召回率代价:")
    print(f"   ✗ 召回率降低: {recall_loss:.2f}%")
    print(f"   ✗ 漏检增加: {and_fn - ngram_fn} 个")
    print(f"   → 这些是SVO未检测到但N-gram检测到的样本")
    print()
    
    print(f"3. F1分数变化:")
    if and_f1 > ngram_f1:
        print(f"   ✓ F1提升: {and_f1 - ngram_f1:+.2f}")
        print(f"   → 准确率提升超过召回率损失，整体效果更好")
    elif and_f1 >= ngram_f1 * 0.98:
        print(f"   ≈ F1基本持平: {and_f1 - ngram_f1:+.2f}")
        print(f"   → 准确率提升与召回率损失基本平衡")
    else:
        print(f"   ✗ F1降低: {and_f1 - ngram_f1:+.2f}")
        print(f"   → 召回率损失超过准确率提升，整体效果变差")
    print()
    
    print(f"4. 交叉验证效率:")
    print(f"   两者都检测到: {both_detected} (AND逻辑标记)")
    print(f"   仅N-gram检测到: {ngram_only} (被排除，可能是误报)")
    print(f"   仅SVO检测到: {svo_only} (被排除，SVO贡献有限)")
    print(f"   → AND逻辑排除了 {ngram_only + svo_only} 个单一方法检测")
    print()
    
    print("=" * 80)
    print("【建议】")
    print("=" * 80)
    print()
    
    if and_f1 > ngram_f1 + 1:
        print("✓ 推荐使用AND逻辑（交叉验证）")
        print(f"  - F1分数提升明显 ({and_f1 - ngram_f1:+.2f})")
        print(f"  - 准确率显著提高 ({precision_gain:+.2f}%)")
        print(f"  - 召回率损失可接受 ({recall_loss:.2f}%)")
    elif precision_gain > 10 and recall_loss < 5:
        print("✓ 推荐使用AND逻辑（交叉验证）")
        print(f"  - 准确率大幅提升 ({precision_gain:+.2f}%)")
        print(f"  - 召回率损失很小 ({recall_loss:.2f}%)")
        print(f"  - 有效控制误报")
    elif and_precision > 60 and and_recall > 90:
        print("✓ 推荐使用AND逻辑（交叉验证）")
        print(f"  - 准确率达到 {and_precision:.2f}%")
        print(f"  - 召回率保持在 {and_recall:.2f}%")
        print(f"  - 平衡性良好")
    else:
        print("⚠ AND逻辑效果有限")
        print(f"  - F1分数变化: {and_f1 - ngram_f1:+.2f}")
        print(f"  - 准确率提升: {precision_gain:+.2f}%")
        print(f"  - 召回率损失: {recall_loss:.2f}%")
        print()
        print("原因分析:")
        print(f"  - SVO检测能力弱（仅检测到 {both_detected + svo_only} 个）")
        print(f"  - 两者重叠少（仅 {both_detected} 个同时检测到）")
        print(f"  - AND逻辑过于严格，排除了太多N-gram的正确检测")
        print()
        print("建议:")
        print("  1. 继续使用N-gram单独检测")
        print("  2. 或改进SVO方法提高检测能力")
        print("  3. 或考虑使用加权策略而非严格AND")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    predict_and_logic_performance()
