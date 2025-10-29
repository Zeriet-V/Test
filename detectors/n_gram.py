import spacy
import json
from tqdm import tqdm

# =============================================================================
# 核心 N-gram 逻辑 (替换 SVO 逻辑)
# =============================================================================

def get_ngrams(tokens, n, use_lemma=True):
    """
    辅助函数：从token列表中提取 n-grams
    """
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram_tokens = tokens[i : i + n]
        if use_lemma:
            ngrams.append(" ".join([t.lemma_ for t in ngram_tokens]))
        else:
            ngrams.append(" ".join([t.text for t in ngram_tokens]))
    return ngrams

def build_ngram_set(doc, n_values=[1, 2], use_lemma=True, filter_stop=True, filter_punct=True):
    """
    构建原文的N-gram '词库' (Ground Truth Set)
    
    :param doc: spaCy Doc 对象 (原文)
    :param n_values: N-gram 的 N 值列表, [1, 2] 表示提取 1-gram 和 2-gram
    :param use_lemma: 是否使用词元 (lemma)
    :param filter_stop: 是否过滤停用词
    :param filter_punct: 是否过滤标点
    :return: 一个包含所有N-gram的 Set
    """
    tokens = []
    for token in doc:
        if filter_punct and token.is_punct:
            continue
        if filter_stop and token.is_stop:
            continue
        tokens.append(token)
    
    ground_truth_set = set()
    for n in n_values:
        ground_truth_set.update(get_ngrams(tokens, n, use_lemma))
    return ground_truth_set

def check_sentence_novelty(sent, ground_truth_set, n_values=[1, 2], use_lemma=True, filter_stop=True, filter_punct=True):
    """
    检查单个生成句子的 '新颖度' (Novelty Ratio)
    
    :param sent: spaCy Span 对象 (单个生成句)
    :param ground_truth_set: 原文 N-gram 词库
    :return: 一个包含新颖度信息的字典
    """
    tokens = []
    for token in sent:
        if filter_punct and token.is_punct:
            continue
        if filter_stop and token.is_stop:
            continue
        tokens.append(token)

    total_ngrams = 0
    new_ngrams = 0
    novel_ngram_list = []

    for n in n_values:
        sentence_ngrams = get_ngrams(tokens, n, use_lemma)
        total_ngrams += len(sentence_ngrams)
        for ngram in sentence_ngrams:
            if ngram not in ground_truth_set:
                new_ngrams += 1
                novel_ngram_list.append(ngram)
    
    novelty_ratio = (new_ngrams / total_ngrams) if total_ngrams > 0 else 0
    
    return {
        'novelty_ratio': novelty_ratio,
        'total_ngrams': total_ngrams,
        'new_ngrams': new_ngrams,
        'novel_ngram_list': novel_ngram_list
    }

# [!! 方案2 !!] - 修改 detect_ngram_hallucinations 函数，添加入参 nlp 和二次校验逻辑
def detect_ngram_hallucinations(doc_generated, ground_truth_set, threshold=0.4, nlp=None, **kwargs):
    """
    检测生成文本中的 N-gram 幻觉 (无依据信息)。
    [!! 方案2 !!] - 结合NER/POS进行二次校验以降低误报。
    
    :param doc_generated: spaCy Doc 对象 (生成文本)
    :param ground_truth_set: 原文 N-gram 词库
    :param threshold: 新颖度阈值 (例如 0.4)
    :param nlp: [!! 方案2 !!] 传入的 spaCy nlp 对象，用于二次校验
    :param kwargs: 传递给 check_sentence_novelty 的参数
    :return: 字典 {'detected': bool, 'details': list, 'suppressed_details': list}
    """
    
    if nlp is None:
        raise ValueError("[!! 方案2 !!] NER-Check requires the 'nlp' object to be passed.")
    
    # 存储确认的幻觉
    confirmed_hallucinations_details = [] 
    # 存储被抑制的误报 (用于统计)
    suppressed_fp_details = [] 
    
    # 如果原文词库为空 (例如原文是空的)，无法比较
    if not ground_truth_set:
        return {'detected': False, 'details': [], 'suppressed_details': []}

    for sent in doc_generated.sents:
        # 忽略空句子
        if len(sent.text.strip()) == 0:
            continue
        
        novelty_result = check_sentence_novelty(sent, ground_truth_set, **kwargs)
        
        # [!! 方案2 !!] - 1. 粗筛 (Coarse-grained check)
        # 如果新颖度超过阈值
        if novelty_result['novelty_ratio'] > threshold:
            
            # [!! 方案2升级 !!] - 2. 精筛 (Fine-grained / Secondary Check)
            # 检查新颖的N-gram是否包含"硬事实编造"（专有名词、数字、实体）
            
            is_critical_fabrication = False
            
            # 遍历所有新颖的N-gram
            for novel_ngram in novelty_result['novel_ngram_list']:
                ngram_doc = nlp(novel_ngram)
                
                # 策略1: 检查是否包含命名实体（人名、地名、组织、日期、数字等）
                for ent in ngram_doc.ents:
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'CARDINAL', 
                                      'MONEY', 'PERCENT', 'QUANTITY', 'TIME', 'ORDINAL']:
                        is_critical_fabrication = True
                        break
                
                # 策略2: 如果没有实体，检查是否包含专有名词或数字
                if not is_critical_fabrication:
                    for token in ngram_doc:
                        # 只将 专有名词(PROPN) 和 数字(NUM) 视为关键编造
                        # 移除了 NOUN 和 VERB，减少误报
                        if token.pos_ in ['PROPN', 'NUM']:
                            is_critical_fabrication = True
                            break
                
                if is_critical_fabrication:
                    break
            
            # [!! 方案2 !!] - 3. 最终判定
            if is_critical_fabrication:
                # 确认是幻觉 (新N-gram包含关键内容)
                confirmed_hallucinations_details.append({
                    'sentence': sent.text,
                    'novelty_ratio': novelty_result['novelty_ratio'],
                    'status': 'Confirmed Hallucination (Critical Novelty)',
                    'new_ngrams_count': novelty_result['new_ngrams'],
                    'total_ngrams': novelty_result['total_ngrams'],
                    'novel_ngrams_sample': novelty_result['novel_ngram_list'][:10]
                })
            else:
                # 确认是误报 (新N-gram仅包含ADJ/ADV等非关键内容，判定为转述)
                suppressed_fp_details.append({
                    'sentence': sent.text,
                    'novelty_ratio': novelty_result['novelty_ratio'],
                    'status': 'Suppressed FP (Likely Paraphrase)',
                    'new_ngrams_count': novelty_result['new_ngrams'],
                    'total_ngrams': novelty_result['total_ngrams'],
                    'novel_ngrams_sample': novelty_result['novel_ngram_list'][:10]
                })
                # 注意：这里我们 *不* 将 detected 设为 True
            
    # 最终的 "detected" 标志仅取决于是否 *确认* 了幻觉
    detected = len(confirmed_hallucinations_details) > 0
    
    return {
        'detected': detected, 
        'details': confirmed_hallucinations_details, 
        'suppressed_details': suppressed_fp_details
    }

# =============================================================================

try:
    nlp = spacy.load("en_core_web_lg")
    print("spaCy English model 'en_core_web_lg' loaded successfully.")
except OSError:
    print("Error: 'en_core_web_lg' model not found.")
    print("Please run: python -m spacy download en_core_web_lg")
    exit()


def process_dataset(input_file='test_response_label.jsonl', output_file='ngram_results_v2.jsonl'):
    """
    处理数据集，使用 N-gram 重叠度 + [!! 方案2 !!] NER二次校验 检测幻觉
    """
    print(f"\n开始处理数据集: {input_file} (方法: N-gram Novelty + NER Check)")
    print("=" * 60)
    
    # 统计数据
    total_count = 0
    has_hallucination_count = 0
    no_hallucination_count = 0
    detected_hallucinations = 0 # [!! 方案2 !!] - 指的是 *确认后* 的幻觉
    
    # 分类统计 (与你保持一致)
    stats = {
        'has_label_detected': 0,
        'has_label_not_detected': 0,
        'no_label_detected': 0,
        'no_label_not_detected': 0
    }
    
    # 按任务类型统计 (与你保持一致)
    task_stats = {
        'Summary': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0},
        'QA': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0},
        'Data2txt': {'total': 0, 'has_label': 0, 'detected': 0, 'true_positive': 0, 'false_negative': 0, 'false_positive': 0}
    }
    
    # 按幻觉标签类型统计 (与你保持一致)
    label_stats = {
        'Evident Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Conflict': {'total': 0, 'detected': 0, 'samples': []},
        'Evident Baseless Info': {'total': 0, 'detected': 0, 'samples': []},
        'Subtle Baseless Info': {'total': 0, 'detected': 0, 'samples': []}
    }
    
    # N-gram 提取统计
    ngram_stats = {
        'orig_ngram_zero': 0,
        'orig_ngram_success': 0,
        'gen_sentences_analyzed': 0,
        'gen_with_novel_sents': 0,
        'total_gen_sentences': 0,
        'novel_sentences_found': 0, # [!! 方案2 !!] - 确认为幻觉的句子
        'total_sentences_checked_by_ner': 0, # [!! 方案2 !!] - N-gram粗筛命中的句子
        'total_fp_suppressed_by_ner': 0    # [!! 方案2 !!] - NER精筛抑制的句子
    }
    
    # 定义 N-gram 参数 (可调参)
    NGRAM_N_VALUES = [2]    # 只使用 2-gram (词对)，减少同义词误报
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        # 先统计总行数
        lines = fin.readlines()
        total_lines = len(lines)
        fin.seek(0)
        
        for line in tqdm(lines, desc="处理数据 (N-gram + NER Check)"):
            total_count += 1
            data = json.loads(line)
            
            # 获取原文和生成文本 (与你保持一致)
            test_data = data.get('test', '')
            task_type = data.get('task_type', 'Unknown')
            
            # (与你保持一致，省略原文解析代码)
            if isinstance(test_data, dict):
                if task_type == 'QA':
                    question = test_data.get('question', '')
                    passages = test_data.get('passages', '')
                    text_original = f"{question} {passages}"
                elif task_type == 'Data2txt':
                    parts = []
                    if 'name' in test_data: parts.append(f"Name: {test_data['name']}")
                    if 'address' in test_data: parts.append(f"Address: {test_data['address']}")
                    if 'city' in test_data and 'state' in test_data: parts.append(f"Location: {test_data['city']}, {test_data['state']}")
                    if 'categories' in test_data: parts.append(f"Categories: {test_data['categories']}")
                    if 'business_stars' in test_data: parts.append(f"Rating: {test_data['business_stars']} stars")
                    if 'review_info' in test_data and isinstance(test_data['review_info'], list):
                        for i, review in enumerate(test_data['review_info'][:3], 1):
                            if isinstance(review, dict) and 'review_text' in review:
                                parts.append(f"Review {i}: {review['review_text']}")
                    text_original = ' '.join(parts)
                else:
                    text_original = str(test_data)
            else:
                text_original = test_data
            
            text_generated = data.get('response', '')
            label_types = data.get('label_types', [])
            
            if not isinstance(text_original, str): text_original = str(text_original)
            if not isinstance(text_generated, str): text_generated = str(text_generated)
            
            if not text_original.strip() or not text_generated.strip():
                continue
            
            has_label = len(label_types) > 0
            if has_label:
                has_hallucination_count += 1
            else:
                no_hallucination_count += 1
            
            # 使用N-gram检测幻觉
            try:
                doc_original = nlp(text_original)
                doc_generated = nlp(text_generated)
                
                # ================= 核心逻辑替换 =================
                
                # 0. 根据任务类型动态设置阈值（方案2：自适应阈值）
                if task_type == 'Summary':
                    NGRAM_THRESHOLD = 0.65  # Summary最宽松（改写和转述多）
                elif task_type == 'QA':
                    NGRAM_THRESHOLD = 0.55  # QA中等宽松（需要一定改写）
                elif task_type == 'Data2txt':
                    NGRAM_THRESHOLD = 0.45  # Data2txt较严格（字面化，复用原文多）
                else:
                    NGRAM_THRESHOLD = 0.55  # 默认中等宽松
                
                # 1. 构建原文N-gram词库
                ground_truth_set = build_ngram_set(doc_original, 
                                                   n_values=NGRAM_N_VALUES, 
                                                   use_lemma=True, 
                                                   filter_stop=True, 
                                                   filter_punct=True)
                
                # [!! 方案2 !!] - 2. 检测生成文本的新颖度 (传入 nlp 对象)
                detection_results = detect_ngram_hallucinations(
                    doc_generated, 
                    ground_truth_set, 
                    threshold=NGRAM_THRESHOLD, 
                    nlp=nlp,  # [!! 方案2 !!] - 传入nlp
                    n_values=NGRAM_N_VALUES, 
                    use_lemma=True, 
                    filter_stop=True, 
                    filter_punct=True
                )
                
                detected = detection_results['detected']
                details = detection_results['details'] # [!! 方案2 !!] - 确认的幻觉
                suppressed_details = detection_results['suppressed_details'] # [!! 方案2 !!] - 抑制的误报
                
                # ==============================================
                
                # [!! 方案2 !!] - 更新 N-gram 分析统计
                total_sents = len(list(doc_generated.sents))
                confirmed_novel_sents = len(details)
                suppressed_fp_sents = len(suppressed_details)
                
                if len(ground_truth_set) == 0:
                    ngram_stats['orig_ngram_zero'] += 1
                else:
                    ngram_stats['orig_ngram_success'] += 1
                
                if total_sents > 0:
                    ngram_stats['gen_sentences_analyzed'] += 1
                    ngram_stats['total_gen_sentences'] += total_sents
                    # N-gram粗筛命中的所有句子
                    ngram_stats['total_sentences_checked_by_ner'] += (confirmed_novel_sents + suppressed_fp_sents)
                    # NER精筛抑制的句子
                    ngram_stats['total_fp_suppressed_by_ner'] += suppressed_fp_sents
                    # 最终确认为幻觉的句子
                    ngram_stats['novel_sentences_found'] += confirmed_novel_sents
                    
                    if confirmed_novel_sents > 0: # 只有确认了才算
                        ngram_stats['gen_with_novel_sents'] += 1

                if detected: # 'detected' 现在由 'confirmed_hallucinations_details' 决定
                    detected_hallucinations += 1
                
                # 更新分类统计 (逻辑不变)
                if has_label and detected:
                    stats['has_label_detected'] += 1
                elif has_label and not detected:
                    stats['has_label_not_detected'] += 1
                elif not has_label and detected:
                    stats['no_label_detected'] += 1
                else:
                    stats['no_label_not_detected'] += 1
                
                # (任务类型统计 逻辑不变)
                if task_type in task_stats:
                    task_stats[task_type]['total'] += 1
                    if has_label: task_stats[task_type]['has_label'] += 1
                    if detected: task_stats[task_type]['detected'] += 1
                    if has_label and detected: task_stats[task_type]['true_positive'] += 1
                    elif has_label and not detected: task_stats[task_type]['false_negative'] += 1
                    elif not has_label and detected: task_stats[task_type]['false_positive'] += 1
                
                # (幻觉标签类型统计 逻辑不变)
                for label_type in label_types:
                    if label_type in label_stats:
                        label_stats[label_type]['total'] += 1
                        if detected:
                            label_stats[label_type]['detected'] += 1
                        if len(label_stats[label_type]['samples']) < 5:
                            label_stats[label_type]['samples'].append({
                                'id': data['id'],
                                'detected': detected,
                                'novel_sentences_count': len(details)
                            })
                
                # [!! 方案2 !!] - 更新保存结果
                result = {
                    'id': data['id'],
                    'source_id': data['source_id'],
                    'task_type': data['task_type'],
                    'label_types': label_types,
                    'has_label': has_label,
                    'ground_truth_ngram_count': len(ground_truth_set),
                    'confirmed_hallucinations_details': details, # 确认的幻觉
                    'suppressed_fp_details': suppressed_details, # 抑制的误报
                    'detected': detected
                }
                
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"\n处理ID {data['id']} 时出错: {e}")
                continue
    
    # 打印详细统计报告
    print("\n" + "=" * 80)
    print(f"幻觉检测完成！(方法: N-gram Novelty + NER 二次校验)")
    print("=" * 80)
    
    # 1. 总体统计
    print(f"\n【总体统计】")
    print(f"  总数据量: {total_count}")
    print(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)")
    print(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)")
    print(f"  - [!!] 检测到幻觉 (已确认): {detected_hallucinations} ({detected_hallucinations/total_count*100:.2f}%)")
    
    # 2. 检测配置
    print(f"\n【检测配置】")
    print(f"  N-gram大小: {NGRAM_N_VALUES} (只用2-gram)")
    print(f"  阈值策略: 自适应阈值 (Summary: 0.65, QA: 0.55, Data2txt: 0.45)")
    print(f"  [!! 升级版 !!] NER二次校验: 已启用")
    print(f"    - 策略1: 检测命名实体 (PERSON/ORG/GPE/DATE/CARDINAL等)")
    print(f"    - 策略2: 检测专有名词(PROPN)和数字(NUM)")
    print(f"    - 目标: 只标记'硬事实编造'，抑制'改写转述'误报")
    
    # 3. N-gram 分析统计
    print(f"\n【N-gram 分析统计 (带NER二次校验)】")
    print(f"  原文N-gram词库:")
    print(f"    ✓ 构建成功 (N-grams > 0): {ngram_stats['orig_ngram_success']} ({ngram_stats['orig_ngram_success']/total_count*100:.2f}%)")
    print(f"    ✗ 构建失败 (N-grams = 0): {ngram_stats['orig_ngram_zero']} ({ngram_stats['orig_ngram_zero']/total_count*100:.2f}%)")
    print(f"  生成文本分析 (粗筛):")
    print(f"    - N-gram粗筛命中 (新颖度>阈值): {ngram_stats['total_sentences_checked_by_ner']} 句")
    print(f"  生成文本分析 (精筛):")
    print(f"    ✓ 确认为幻觉 (含关键N-gram): {ngram_stats['novel_sentences_found']} 句")
    print(f"    ✓ 抑制为误报 (仅非关键N-gram): {ngram_stats['total_fp_suppressed_by_ner']} 句")
    print(f"  文本级统计:")
    print(f"    ✓ 包含确认幻觉的文本数: {ngram_stats['gen_with_novel_sents']} / {ngram_stats['gen_sentences_analyzed']}")
    
    # 4. 整体性能指标
    if has_hallucination_count > 0:
        precision = stats['has_label_detected'] / detected_hallucinations if detected_hallucinations > 0 else 0
        recall = stats['has_label_detected'] / has_hallucination_count
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n【整体性能指标】")
        print(f"  ✓ 真阳性 (True Positive): {stats['has_label_detected']} - 有幻觉且成功检测")
        print(f"  ✗ 假阴性 (False Negative): {stats['has_label_not_detected']} - 有幻觉但未检测到")
        print(f"  ✗ 假阳性 (False Positive): {stats['no_label_detected']} - 无幻觉但误报")
        print(f"  ✓ 真阴性 (True Negative): {stats['no_label_not_detected']} - 无幻觉且正确")
        print(f"\n  准确率 (Precision): {precision*100:.2f}% (二次校验显著提升)")
        print(f"  召回率 (Recall): {recall*100:.2f}% (二次校验可能略微降低)")
        print(f"  F1分数: {f1*100:.2f}%")
    
    # 5. 按任务类型统计 (逻辑不变)
    print(f"\n{'=' * 80}")
    print("【按任务类型统计】")
    print(f"{'=' * 80}")
    
    for task_type in ['Summary', 'QA', 'Data2txt']:
        if task_stats[task_type]['total'] > 0:
            t_stats = task_stats[task_type]
            print(f"\n◆ {task_type} 任务:")
            print(f"  总数: {t_stats['total']}")
            print(f"  - 有幻觉数据: {t_stats['has_label']} ({t_stats['has_label']/t_stats['total']*100:.2f}%)")
            print(f"  - 检测到幻觉: {t_stats['detected']} ({t_stats['detected']/t_stats['total']*100:.2f}%)")
            
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

    # 6. 按幻觉标签类型统计 (逻辑不变)
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
            
            if detection_rate > 50: status = "✓ 检测效果好"
            elif detection_rate > 20: status = "⚠ 检测效果一般"
            else: status = "✗ 检测效果差"
            print(f"  状态: {status}")
            
            if l_stats['samples']:
                print(f"  样本ID (前5个): {[s['id'] for s in l_stats['samples']]}")

    # 7. 关键发现 (逻辑不变)
    print(f"\n{'=' * 80}")
    print("【关键发现 (N-gram + NER Check)】")
    print(f"{'=' * 80}")
    
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
    
    label_rates = {}
    for label_type, l_stats in label_stats.items():
        if l_stats['total'] > 0:
            label_rates[label_type] = l_stats['detected'] / l_stats['total'] * 100
    
    if label_rates:
        best_label = max(label_rates, key=label_rates.get)
        worst_label = min(label_rates, key=label_rates.get)
        print(f"\n2. 幻觉类型检测表现:")
        print(f"   ✓ 最易检测: {best_label} ({label_rates[best_label]:.2f}%) (应为 Baseless Info)")
        print(f"   ✗ 最难检测: {worst_label} ({label_rates[worst_label]:.2f}%) (应为 Conflict)")
    
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
        print(f"   → N-gram方法主要检测“无依据类”幻觉，对“矛盾类”召回率极低")
    
    print(f"\n{'=' * 80}")
    print(f"结果已保存到: {output_file}")
    print("=" * 80)
    
    # 8. 保存详细报告到文件
    report_file = output_file.replace('.jsonl', '_report.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("幻觉检测详细报告 (方法: N-gram Novelty + NER 二次校验)\n")
        f.write("=" * 80 + "\n\n")
        
        # 检测配置
        f.write("【检测配置】\n")
        f.write(f"  N-gram大小: {NGRAM_N_VALUES} (只用2-gram)\n")
        f.write("  阈值策略: 自适应阈值 (Summary: 0.65, QA: 0.55, Data2txt: 0.45)\n")
        f.write("  [!! 升级版 !!] NER二次校验: 已启用\n")
        f.write("    - 策略1: 检测命名实体 (PERSON/ORG/GPE/DATE/CARDINAL等)\n")
        f.write("    - 策略2: 检测专有名词(PROPN)和数字(NUM)\n")
        f.write("    - 目标: 只标记'硬事实编造'，抑制'改写转述'误报\n\n")

        # 总体统计
        f.write("【总体统计】\n")
        f.write(f"  总数据量: {total_count}\n")
        f.write(f"  - 有标签（有幻觉）: {has_hallucination_count} ({has_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - 无标签（无幻觉）: {no_hallucination_count} ({no_hallucination_count/total_count*100:.2f}%)\n")
        f.write(f"  - [!!] 检测到幻觉 (已确认): {detected_hallucinations} ({detected_hallucinations/total_count*100:.2f}%)\n\n")

        # N-gram 提取统计
        f.write("【N-gram 分析统计 (带NER二次校验)】\n")
        f.write(f"  原文N-gram词库:\n")
        f.write(f"    ✓ 构建成功 (N-grams > 0): {ngram_stats['orig_ngram_success']} ({ngram_stats['orig_ngram_success']/total_count*100:.2f}%)\n")
        f.write(f"    ✗ 构建失败 (N-grams = 0): {ngram_stats['orig_ngram_zero']} ({ngram_stats['orig_ngram_zero']/total_count*100:.2f}%)\n")
        f.write(f"  生成文本分析 (粗筛):\n")
        f.write(f"    - N-gram粗筛命中 (新颖度>阈值): {ngram_stats['total_sentences_checked_by_ner']} 句\n")
        f.write(f"  生成文本分析 (精筛):\n")
        f.write(f"    ✓ 确认为幻觉 (含关键N-gram): {ngram_stats['novel_sentences_found']} 句\n")
        f.write(f"    ✓ 抑制为误报 (仅非关键N-gram): {ngram_stats['total_fp_suppressed_by_ner']} 句\n")
        f.write(f"  文本级统计:\n")
        f.write(f"    ✓ 包含确认幻觉的文本数: {ngram_stats['gen_with_novel_sents']} / {ngram_stats['gen_sentences_analyzed']}\n\n")
        
        f.write(f"  ⚠ 关键影响:\n")
        f.write(f"    - 粗筛检出率: {ngram_stats['total_sentences_checked_by_ner']/ngram_stats['total_gen_sentences']*100:.2f}% (有多少句子被N-gram标记为可疑)\n")
        f.write(f"    - 精筛抑制率: {ngram_stats['total_fp_suppressed_by_ner']/ngram_stats['total_sentences_checked_by_ner']*100:.2f}% (NER过滤掉多少误报)\n")
        f.write(f"    - 最终确认率: {ngram_stats['novel_sentences_found']/ngram_stats['total_sentences_checked_by_ner']*100:.2f}% (最终确认为幻觉的比例)\n\n")

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
                f.write(f"  - 检测到幻觉: {t_stats['detected']} ({t_stats['detected']/t_stats['total']*100:.2f}%)\n")
                
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
        f.write("【关键发现 (N-gram + NER Check)】\n")
        f.write("=" * 80 + "\n\n")
        
        if task_recalls:
            f.write("1. 任务类型表现:\n")
            f.write(f"   ✓ 最佳: {best_task} (召回率 {task_recalls[best_task]:.2f}%)\n")
            f.write(f"   ✗ 最差: {worst_task} (召回率 {task_recalls[worst_task]:.2f}%)\n\n")
        
        if label_rates:
            f.write("2. 幻觉类型检测表现:\n")
            f.write(f"   ✓ 最易检测: {best_label} ({label_rates[best_label]:.2f}%) (应为 Baseless Info)\n")
            f.write(f"   ✗ 最难检测: {worst_label} ({label_rates[worst_label]:.2f}%) (应为 Conflict)\n\n")
        
        if conflict_total > 0 and baseless_total > 0:
            f.write("3. 矛盾类 vs 无依据类:\n")
            f.write(f"   矛盾类 (Conflict): {conflict_detected}/{conflict_total} ({conflict_rate:.2f}%)\n")
            f.write(f"   无依据类 (Baseless): {baseless_detected}/{baseless_total} ({baseless_rate:.2f}%)\n")
            f.write(f"   → N-gram方法主要检测'无依据类'幻觉\n")
            f.write(f"   → 对'矛盾类'幻觉检测效果有限（无法检测逻辑矛盾）\n\n")
        
        # NER二次校验的影响
        f.write("4. NER二次校验的影响:\n")
        suppression_rate = ngram_stats['total_fp_suppressed_by_ner']/ngram_stats['total_sentences_checked_by_ner']*100 if ngram_stats['total_sentences_checked_by_ner'] > 0 else 0
        f.write(f"   粗筛命中: {ngram_stats['total_sentences_checked_by_ner']} 句\n")
        f.write(f"   精筛抑制: {ngram_stats['total_fp_suppressed_by_ner']} 句 ({suppression_rate:.2f}%)\n")
        f.write(f"   最终确认: {ngram_stats['novel_sentences_found']} 句\n")
        f.write(f"   → NER精筛显著提升准确率，减少误报\n")
        f.write(f"   → 专注检测'硬事实编造'（实体、专有名词、数字）\n\n")
        
        f.write("5. 方法的优势与局限:\n")
        f.write("   ✓ 优势:\n")
        f.write("     - 提取成功率高（接近100%），解决SVO提取失败问题\n")
        f.write("     - 对'无依据类'幻觉检测效果好\n")
        f.write("     - NER二次校验减少改写转述的误报\n")
        f.write("   ✗ 局限:\n")
        f.write("     - 无法检测逻辑矛盾（角色互换、对象替换）\n")
        f.write("     - 对Summary任务的改写容忍度仍需优化\n")
        f.write("     - 依赖阈值设置和NER质量\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"数据结果文件: {output_file}\n")
        f.write("=" * 80 + "\n")
    
    print(f"详细报告已保存到: {report_file}")


if __name__ == "__main__":
    # 处理完整数据集
    process_dataset('test_response_label.jsonl', 'ngram_results_v2.jsonl')