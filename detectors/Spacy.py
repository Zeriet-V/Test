import spacy
import json
from tqdm import tqdm

def get_phrase(token):
    """
    获取以token为中心的短语，包括其修饰词
    对于中文，主要收集复合词和修饰词
    """
    # 收集所有修饰当前token的子节点
    phrase_tokens = [token]
    
    for child in token.children:
        # 收集常见的修饰关系：复合词、限定词、数量词等
        if child.dep_ in ['compound', 'nummod', 'amod', 'det', 'nmod']:
            phrase_tokens.append(child)
    
    # 按照在句子中的位置排序
    phrase_tokens.sort(key=lambda t: t.i)
    return ' '.join([t.text for t in phrase_tokens])

def extract_svos(doc):
    """
    从 spaCy Doc 对象中提取 SVO (主-谓-宾) 元组。
    返回一个列表，每个元素是 (subject, verb_lemma, object, negation)
    """
    svos = []

    for token in doc:
        # 我们寻找句子的核心动词 (ROOT 且词性为 VERB)
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            verb = token.lemma_  # 获取动词的词元（例如 "收购"）
            neg = False
            subject_text = ""
            object_text = ""

            # 遍历动词的子节点
            for child in token.children:
                # 寻找主语 (nsubj)
                if child.dep_ == 'nsubj':
                    # 获取包含修饰词的完整短语
                    subject_text = get_phrase(child)
                
                # 寻找宾语 (dobj)
                elif child.dep_ == 'dobj':
                    object_text = get_phrase(child)
                
                # 检查是否有否定词 (neg)
                elif child.dep_ == 'neg':
                    neg = True
            
            # 只有当主语和宾语都存在时，才添加
            if subject_text and object_text:
                svos.append((subject_text, verb, object_text, neg))
                
    return svos

def compare_svos(svos_orig, svos_gen):
    """
    比较原文 (orig) 和生成 (gen) 的 SVO 列表，查找矛盾。
    """
    contradictions = []

    for g_s, g_v, g_o, g_neg in svos_gen:
        # 遍历生成内容中的每一个 SVO
        
        # 标志位，看是否在原文中找到了相关陈述
        found_match_in_orig = False

        for o_s, o_v, o_o, o_neg in svos_orig:
            
            # 1. 检查主谓宾是否完全颠倒 (例如 "A 收购 B" vs "B 收购 A")
            if g_s == o_o and g_o == o_s and g_v == o_v:
                contradictions.append(
                    f"[Role Reversal]: Generated '{g_s} {g_v} {g_o}' "
                    f"has reversed roles compared to original '{o_s} {o_v} {o_o}'."
                )
                found_match_in_orig = True
                continue # 继续检查下一个生成SVO

            # 2. 检查主语和谓语是否匹配
            # (我们假设主语和谓语相同，是在讨论同一件事)
            if g_s == o_s and g_v == o_v:
                found_match_in_orig = True # 找到了相关陈述
                
                # 2a. 检查宾语是否矛盾
                if g_o != o_o:
                    contradictions.append(
                        f"[Object Contradiction]: Generated says '{g_s} {g_v} {g_o}', "
                        f"but original says '{o_s} {o_v} {o_o}'."
                    )
                
                # 2b. 检查否定词是否矛盾
                if g_neg != o_neg:
                    contradictions.append(
                        f"[Negation Contradiction]: Generated '{g_s} {g_v} {g_o}' (negation={g_neg}) "
                        f"contradicts original '{o_s} {o_v} {o_o}' (negation={o_neg})."
                    )
        
        # # 3. 检查"无依据信息"：如果生成了一个SVO，但在原文中完全没找到相关主谓
        # if not found_match_in_orig:
        #     contradictions.append(
        #         f"[Baseless Information]: Generated '{g_s} {g_v} {g_o}' "
        #         f"has no supporting statement in original text."
        #     )
            
    return contradictions

try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy English model 'en_core_web_lg' loaded successfully.")
except OSError:
    print("Error: 'en_core_web_lg' model not found.")
    print("Please run: python -m spacy download en_core_web_lg")
    exit()


def process_dataset(input_file='test_response_label.jsonl', output_file='spacy_results.jsonl'):
    """
    处理数据集，检测幻觉
    """
    print(f"\n开始处理数据集: {input_file}")
    print("=" * 60)
    
    # 统计数据
    total_count = 0
    has_hallucination_count = 0  # 数据集标注有幻觉
    no_hallucination_count = 0   # 数据集标注无幻觉
    detected_contradictions = 0  # 我们检测到矛盾
    
    # 分类统计
    stats = {
        'has_label_detected': 0,      # 有标签且检测到矛盾
        'has_label_not_detected': 0,  # 有标签但未检测到矛盾
        'no_label_detected': 0,       # 无标签但检测到矛盾（误报）
        'no_label_not_detected': 0    # 无标签且未检测到矛盾（正确）
    }
    
    # 按任务类型统计
    task_stats = {
        'Summary': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0},
        'QA': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0},
        'Data2txt': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0}
    }
    
    # 按幻觉标签类型统计
    label_stats = {
        'Evident Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Evident Baseless Info': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Baseless Info': {'total': 0, 'detected': 0, 'samples': []}
    }
    
    # SVO提取失败统计
    svo_stats = {
        'orig_svo_zero': 0,      # 原文SVO=0
        'gen_svo_zero': 0,       # 生成SVO=0
        'both_svo_zero': 0,      # 两者都=0
        'orig_svo_success': 0,   # 原文SVO≥1
        'gen_svo_success': 0     # 生成SVO≥1
    }
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # 先统计总行数
        lines = fin.readlines()
        total_lines = len(lines)
        fin.seek(0)
        
        for line in tqdm(lines, desc="处理数据"):
            total_count += 1
            data = json.loads(line)
            
            # 获取原文和生成文本
            test_data = data.get('test', '')
            task_type = data.get('task_type', 'Unknown')
            
            # 根据任务类型处理不同格式的test字段
            if isinstance(test_data, dict):
                if task_type == 'QA':
                    # QA任务：合并question和passages
                    question = test_data.get('question', '')
                    passages = test_data.get('passages', '')
                    text_original = f"{question} {passages}"
                elif task_type == 'Data2txt':
                    # Data2txt任务：将结构化数据转换为文本
                    parts = []
                    # 基本信息
                    if 'name' in test_data:
                        parts.append(f"Name: {test_data['name']}")
                    if 'address' in test_data:
                        parts.append(f"Address: {test_data['address']}")
                    if 'city' in test_data and 'state' in test_data:
                        parts.append(f"Location: {test_data['city']}, {test_data['state']}")
                    if 'categories' in test_data:
                        parts.append(f"Categories: {test_data['categories']}")
                    if 'business_stars' in test_data:
                        parts.append(f"Rating: {test_data['business_stars']} stars")
                    
                    # 评论信息
                    if 'review_info' in test_data and isinstance(test_data['review_info'], list):
                        for i, review in enumerate(test_data['review_info'][:3], 1):  # 只取前3条
                            if isinstance(review, dict) and 'review_text' in review:
                                parts.append(f"Review {i}: {review['review_text']}")
                    
                    text_original = ' '.join(parts)
                else:
                    # 其他字典类型：简单转换为字符串
                    text_original = str(test_data)
            else:
                # Summary任务：直接使用字符串
                text_original = test_data
            
            text_generated = data.get('response', '')
            label_types = data.get('label_types', [])
            
            # 确保都是字符串格式
            if not isinstance(text_original, str):
                text_original = str(text_original)
            if not isinstance(text_generated, str):
                text_generated = str(text_generated)
            
            # 跳过空文本
            if not text_original.strip() or not text_generated.strip():
                continue
            
            # 判断是否有标签（有幻觉）
            has_label = len(label_types) > 0
            if has_label:
                has_hallucination_count += 1
            else:
                no_hallucination_count += 1
            
            # 使用spaCy检测矛盾
            try:
                doc_original = nlp(text_original)
                doc_generated = nlp(text_generated)
                
                svos_original = extract_svos(doc_original)
                svos_generated = extract_svos(doc_generated)
                
                # 统计SVO提取情况
                orig_svo_count = len(svos_original)
                gen_svo_count = len(svos_generated)
                
                if orig_svo_count == 0:
                    svo_stats['orig_svo_zero'] += 1
                else:
                    svo_stats['orig_svo_success'] += 1
                
                if gen_svo_count == 0:
                    svo_stats['gen_svo_zero'] += 1
                else:
                    svo_stats['gen_svo_success'] += 1
                
                if orig_svo_count == 0 and gen_svo_count == 0:
                    svo_stats['both_svo_zero'] += 1
                
                contradictions = compare_svos(svos_original, svos_generated)
                
                # 判断是否检测到矛盾
                detected = len(contradictions) > 0
                if detected:
                    detected_contradictions += 1
                
                # 更新分类统计
                if has_label and detected:
                    stats['has_label_detected'] += 1
                elif has_label and not detected:
                    stats['has_label_not_detected'] += 1
                elif not has_label and detected:
                    stats['no_label_detected'] += 1
                else:
                    stats['no_label_not_detected'] += 1
                
                # 更新任务类型统计
                if task_type in task_stats:
                    task_stats[task_type]['total'] += 1
                    if has_label:
                        task_stats[task_type]['has_label'] += 1
                    if detected:
                        task_stats[task_type]['detected'] += 1
                    if has_label and detected:
                        task_stats[task_type]['true_positive'] += 1
                    elif has_label and not detected:
                        task_stats[task_type]['false_negative'] += 1
                    elif not has_label and detected:
                        task_stats[task_type]['false_positive'] += 1
                
                # 更新幻觉标签类型统计
                for label_type in label_types:
                    if label_type in label_stats:
                        label_stats[label_type]['total'] += 1
                        if detected:
                            label_stats[label_type]['detected'] += 1
                        # 保存样本（最多5个）
                        if len(label_stats[label_type]['samples']) < 5:
                            label_stats[label_type]['samples'].append({
                                'id': data['id'],
                                'detected': detected,
                                'contradictions_count': len(contradictions)
                            })
                
                # 保存结果
                result = {
                    'id': data['id'],
                    'source_id': data['source_id'],
                    'task_type': data['task_type'],
                    'label_types': label_types,
                    'has_label': has_label,
                    'svos_original': svos_original,
                    'svos_generated': svos_generated,
                    'contradictions': contradictions,
                    'detected': detected
                }
                
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n处理ID {data['id']} 时出错: {e}")
                continue
    
    # 打印详细统计报告
    print("\n" + "=" * 80)
    print("幻觉检测完成！详细统计报告")
    print("=" * 80)
    
    # 1. 总体统计
    print(f"\n【总体统计】")
    print(f"  总数据量: {total_count}")
    print(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)")
    print(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)")
    print(f"  - 检测到矛盾: {detected_contradictions} ({detected_contradictions/total_count*100:.2f}%)")
    
    # 2. SVO提取统计
    print(f"\n【SVO提取统计】")
    print(f"  原文SVO提取:")
    print(f"    ✓ 成功提取 (≥1个SVO): {svo_stats['orig_svo_success']} ({svo_stats['orig_svo_success']/total_count*100:.2f}%)")
    print(f"    ✗ 提取失败 (0个SVO): {svo_stats['orig_svo_zero']} ({svo_stats['orig_svo_zero']/total_count*100:.2f}%)")
    print(f"  生成文本SVO提取:")
    print(f"    ✓ 成功提取 (≥1个SVO): {svo_stats['gen_svo_success']} ({svo_stats['gen_svo_success']/total_count*100:.2f}%)")
    print(f"    ✗ 提取失败 (0个SVO): {svo_stats['gen_svo_zero']} ({svo_stats['gen_svo_zero']/total_count*100:.2f}%)")
    print(f"  两者都提取失败: {svo_stats['both_svo_zero']} ({svo_stats['both_svo_zero']/total_count*100:.2f}%)")
    print(f"\n  ⚠ 提取失败影响: 原文SVO=0时无法进行矛盾比对")
    
    # 3. 整体性能指标
    if has_hallucination_count > 0:
        precision = stats['has_label_detected'] / detected_contradictions if detected_contradictions > 0 else 0
        recall = stats['has_label_detected'] / has_hallucination_count
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n【整体性能指标】")
        print(f"  ✓ 真阳性 (True Positive): {stats['has_label_detected']} - 有幻觉且成功检测")
        print(f"  ✗ 假阴性 (False Negative): {stats['has_label_not_detected']} - 有幻觉但未检测到")
        print(f"  ✗ 假阳性 (False Positive): {stats['no_label_detected']} - 无幻觉但误报")
        print(f"  ✓ 真阴性 (True Negative): {stats['no_label_not_detected']} - 无幻觉且正确")
        print(f"\n  准确率 (Precision): {precision*100:.2f}%")
        print(f"  召回率 (Recall): {recall*100:.2f}%")
        print(f"  F1分数: {f1*100:.2f}%")
    
    # 3. 按任务类型统计
    print(f"\n{'=' * 80}")
    print("【按任务类型统计】")
    print(f"{'=' * 80}")
    
    for task_type in ['Summary', 'QA', 'Data2txt']:
        if task_stats[task_type]['total'] > 0:
            t_stats = task_stats[task_type]
            print(f"\n◆ {task_type} 任务:")
            print(f"  总数: {t_stats['total']}")
            print(f"  - 有幻觉数据: {t_stats['has_label']} ({t_stats['has_label']/t_stats['total']*100:.2f}%)")
            print(f"  - 检测到矛盾: {t_stats['detected']} ({t_stats['detected']/t_stats['total']*100:.2f}%)")
            
            if t_stats['has_label'] > 0:
                task_recall = t_stats['true_positive'] / t_stats['has_label'] * 100
                print(f"\n  性能表现:")
                print(f"    ✓ 成功检测 (TP): {t_stats['true_positive']}")
                print(f"    ✗ 漏检 (FN): {t_stats['false_negative']}")
                print(f"    ✗ 误报 (FP): {t_stats['false_positive']}")
                print(f"    召回率: {task_recall:.2f}%")
                
                if t_stats['detected'] > 0:
                    task_precision = t_stats['true_positive'] / t_stats['detected'] * 100
                    print(f"    准确率: {task_precision:.2f}%")
    
    # 4. 按幻觉标签类型统计
    print(f"\n{'=' * 80}")
    print("【按幻觉标签类型统计】")
    print(f"{'=' * 80}")
    
    for label_type in ['Evident Conflict', 'Subtle Conflict', 'Evident Baseless Info', 'Subtle Baseless Info']:
        if label_stats[label_type]['total'] > 0:
            l_stats = label_stats[label_type]
            detection_rate = l_stats['detected'] / l_stats['total'] * 100
            
            print(f"\n◆ {label_type}:")
            print(f"  总数: {l_stats['total']}")
            print(f"  检测到: {l_stats['detected']} ({detection_rate:.2f}%)")
            print(f"  漏检: {l_stats['total'] - l_stats['detected']} ({100-detection_rate:.2f}%)")
            
            # 显示检测状态
            if detection_rate > 50:
                status = "✓ 检测效果好"
            elif detection_rate > 20:
                status = "⚠ 检测效果一般"
            else:
                status = "✗ 检测效果差"
            print(f"  状态: {status}")
            
            # 显示样本
            if l_stats['samples']:
                print(f"  样本ID (前5个): {[s['id'] for s in l_stats['samples']]}")
    
    # 5. 关键发现
    print(f"\n{'=' * 80}")
    print("【关键发现】")
    print(f"{'=' * 80}")
    
    # 找出检测率最高和最低的任务类型
    task_recalls = {}
    for task_type, t_stats in task_stats.items():
        if t_stats['has_label'] > 0:
            task_recalls[task_type] = t_stats['true_positive'] / t_stats['has_label'] * 100
    
    if task_recalls:
        best_task = max(task_recalls, key=task_recalls.get)
        worst_task = min(task_recalls, key=task_recalls.get)
        print(f"\n1. 任务类型表现:")
        print(f"   ✓ 最佳: {best_task} (召回率 {task_recalls[best_task]:.2f}%)")
        print(f"   ✗ 最差: {worst_task} (召回率 {task_recalls[worst_task]:.2f}%)")
    
    # 找出检测率最高和最低的标签类型
    label_rates = {}
    for label_type, l_stats in label_stats.items():
        if l_stats['total'] > 0:
            label_rates[label_type] = l_stats['detected'] / l_stats['total'] * 100
    
    if label_rates:
        best_label = max(label_rates, key=label_rates.get)
        worst_label = min(label_rates, key=label_rates.get)
        print(f"\n2. 幻觉类型检测表现:")
        print(f"   ✓ 最易检测: {best_label} ({label_rates[best_label]:.2f}%)")
        print(f"   ✗ 最难检测: {worst_label} ({label_rates[worst_label]:.2f}%)")
    
    # Conflict vs Baseless 对比
    conflict_total = label_stats['Evident Conflict']['total'] + label_stats['Subtle Conflict']['total']
    conflict_detected = label_stats['Evident Conflict']['detected'] + label_stats['Subtle Conflict']['detected']
    baseless_total = label_stats['Evident Baseless Info']['total'] + label_stats['Subtle Baseless Info']['total']
    baseless_detected = label_stats['Evident Baseless Info']['detected'] + label_stats['Subtle Baseless Info']['detected']
    
    if conflict_total > 0 and baseless_total > 0:
        conflict_rate = conflict_detected / conflict_total * 100
        baseless_rate = baseless_detected / baseless_total * 100
        print(f"\n3. 矛盾类 vs 无依据类:")
        print(f"   矛盾类 (Conflict): {conflict_detected}/{conflict_total} ({conflict_rate:.2f}%)")
        print(f"   无依据类 (Baseless): {baseless_detected}/{baseless_total} ({baseless_rate:.2f}%)")
        print(f"   → SVO方法对{'矛盾类' if conflict_rate > baseless_rate else '无依据类'}检测效果更好")
    
    print(f"\n{'=' * 80}")
    print(f"结果已保存到: {output_file}")
    print("=" * 80)
    
    # 6. 保存详细报告到文件
    report_file = output_file.replace('.jsonl', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("幻觉检测详细报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 总体统计
        f.write("【总体统计】\n")
        f.write(f"  总数据量: {total_count}\n")
        f.write(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 检测到矛盾: {detected_contradictions} ({detected_contradictions/total_count*100:.2f}%)\n\n")
        
        # SVO提取统计
        f.write("【SVO提取统计】\n")
        f.write(f"  原文SVO提取:\n")
        f.write(f"    ✓ 成功提取 (≥1个SVO): {svo_stats['orig_svo_success']} ({svo_stats['orig_svo_success']/total_count*100:.2f}%)\n")
        f.write(f"    ✗ 提取失败 (0个SVO): {svo_stats['orig_svo_zero']} ({svo_stats['orig_svo_zero']/total_count*100:.2f}%)\n")
        f.write(f"  生成文本SVO提取:\n")
        f.write(f"    ✓ 成功提取 (≥1个SVO): {svo_stats['gen_svo_success']} ({svo_stats['gen_svo_success']/total_count*100:.2f}%)\n")
        f.write(f"    ✗ 提取失败 (0个SVO): {svo_stats['gen_svo_zero']} ({svo_stats['gen_svo_zero']/total_count*100:.2f}%)\n")
        f.write(f"  两者都提取失败: {svo_stats['both_svo_zero']} ({svo_stats['both_svo_zero']/total_count*100:.2f}%)\n\n")
        f.write(f"  ⚠ 关键影响:\n")
        f.write(f"    - 原文SVO=0时，无法进行矛盾比对（只能检测结构性矛盾）\n")
        f.write(f"    - 这是导致召回率低的主要原因\n\n")
        
        # 整体性能指标
        if has_hallucination_count > 0:
            f.write("【整体性能指标】\n")
            f.write(f"  ✓ 真阳性 (True Positive): {stats['has_label_detected']} - 有幻觉且成功检测\n")
            f.write(f"  ✗ 假阴性 (False Negative): {stats['has_label_not_detected']} - 有幻觉但未检测到\n")
            f.write(f"  ✗ 假阳性 (False Positive): {stats['no_label_detected']} - 无幻觉但误报\n")
            f.write(f"  ✓ 真阴性 (True Negative): {stats['no_label_not_detected']} - 无幻觉且正确\n\n")
            f.write(f"  准确率 (Precision): {precision*100:.2f}%\n")
            f.write(f"  召回率 (Recall): {recall*100:.2f}%\n")
            f.write(f"  F1分数: {f1*100:.2f}%\n\n")
        
        # 按任务类型统计
        f.write("=" * 80 + "\n")
        f.write("【按任务类型统计】\n")
        f.write("=" * 80 + "\n\n")
        
        for task_type in ['Summary', 'QA', 'Data2txt']:
            if task_stats[task_type]['total'] > 0:
                t_stats = task_stats[task_type]
                f.write(f"◆ {task_type} 任务:\n")
                f.write(f"  总数: {t_stats['total']}\n")
                f.write(f"  - 有幻觉数据: {t_stats['has_label']} ({t_stats['has_label']/t_stats['total']*100:.2f}%)\n")
                f.write(f"  - 检测到矛盾: {t_stats['detected']} ({t_stats['detected']/t_stats['total']*100:.2f}%)\n")
                
                if t_stats['has_label'] > 0:
                    task_recall = t_stats['true_positive'] / t_stats['has_label'] * 100
                    f.write(f"\n  性能表现:\n")
                    f.write(f"    ✓ 成功检测 (TP): {t_stats['true_positive']}\n")
                    f.write(f"    ✗ 漏检 (FN): {t_stats['false_negative']}\n")
                    f.write(f"    ✗ 误报 (FP): {t_stats['false_positive']}\n")
                    f.write(f"    召回率: {task_recall:.2f}%\n")
                    
                    if t_stats['detected'] > 0:
                        task_precision = t_stats['true_positive'] / t_stats['detected'] * 100
                        f.write(f"    准确率: {task_precision:.2f}%\n")
                f.write("\n")
        
        # 按幻觉标签类型统计
        f.write("=" * 80 + "\n")
        f.write("【按幻觉标签类型统计】\n")
        f.write("=" * 80 + "\n\n")
        
        for label_type in ['Evident Conflict', 'Subtle Conflict', 'Evident Baseless Info', 'Subtle Baseless Info']:
            if label_stats[label_type]['total'] > 0:
                l_stats = label_stats[label_type]
                detection_rate = l_stats['detected'] / l_stats['total'] * 100
                
                f.write(f"◆ {label_type}:\n")
                f.write(f"  总数: {l_stats['total']}\n")
                f.write(f"  检测到: {l_stats['detected']} ({detection_rate:.2f}%)\n")
                f.write(f"  漏检: {l_stats['total'] - l_stats['detected']} ({100-detection_rate:.2f}%)\n")
                
                if detection_rate > 50:
                    status = "✓ 检测效果好"
                elif detection_rate > 20:
                    status = "⚠ 检测效果一般"
                else:
                    status = "✗ 检测效果差"
                f.write(f"  状态: {status}\n")
                
                if l_stats['samples']:
                    f.write(f"  样本ID (前5个): {[s['id'] for s in l_stats['samples']]}\n")
                f.write("\n")
        
        # 关键发现
        f.write("=" * 80 + "\n")
        f.write("【关键发现】\n")
        f.write("=" * 80 + "\n\n")
        
        if task_recalls:
            f.write("1. 任务类型表现:\n")
            f.write(f"   ✓ 最佳: {best_task} (召回率 {task_recalls[best_task]:.2f}%)\n")
            f.write(f"   ✗ 最差: {worst_task} (召回率 {task_recalls[worst_task]:.2f}%)\n\n")
        
        if label_rates:
            f.write("2. 幻觉类型检测表现:\n")
            f.write(f"   ✓ 最易检测: {best_label} ({label_rates[best_label]:.2f}%)\n")
            f.write(f"   ✗ 最难检测: {worst_label} ({label_rates[worst_label]:.2f}%)\n\n")
        
        if conflict_total > 0 and baseless_total > 0:
            f.write("3. 矛盾类 vs 无依据类:\n")
            f.write(f"   矛盾类 (Conflict): {conflict_detected}/{conflict_total} ({conflict_rate:.2f}%)\n")
            f.write(f"   无依据类 (Baseless): {baseless_detected}/{baseless_total} ({baseless_rate:.2f}%)\n")
            f.write(f"   → SVO方法主要检测矛盾类幻觉\n\n")
        
        # SVO提取失败的影响
        f.write("4. SVO提取失败的影响:\n")
        f.write(f"   原文提取失败率: {svo_stats['orig_svo_zero']/total_count*100:.2f}%\n")
        f.write(f"   生成提取失败率: {svo_stats['gen_svo_zero']/total_count*100:.2f}%\n")
        f.write(f"   → 这是导致召回率低的根本原因！\n")
        f.write(f"   → 原文SVO=0时，无法进行任何矛盾比对\n")
        f.write(f"   → 约{svo_stats['orig_svo_zero']/total_count*100:.0f}%的数据因SVO提取失败而无法检测\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"数据结果文件: {output_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"详细报告已保存到: {report_file}")


if __name__ == "__main__":
    # 处理完整数据集（包含有幻觉和无幻觉的数据）
    process_dataset('test_response_label.jsonl', 'spacy_results.jsonl')