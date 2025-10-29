"""
评估方法函数库
根据UniversalDataset类的不同属性来进行评估
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
import numpy as np
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from universal_dataset import UniversalDataset, TaskType, LabelType
from transformers import AutoModelForSequenceClassification

# ============ Vectara 幻觉检测评估器 ============

class VectaraEvaluator:
    """Vectara 幻觉检测评估器
    
    使用 Vectara 的幻觉评估模型来检测生成文本中的幻觉
    模型会对 (source, summary) 对进行评估，返回幻觉分数
    """
    
    def __init__(self, model_name: str = 'vectara/hallucination_evaluation_model'):
        """初始化评估器
        
        Args:
            model_name: 模型名称，默认为 'vectara/hallucination_evaluation_model'
        """
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """加载 Vectara 幻觉检测模型"""
        if self.model is None:
            print(f"正在加载 Vectara 模型: {self.model_name}")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            print("模型加载完成")
    
    def evaluate(self, datasets: List[UniversalDataset]) -> Dict[str, Any]:
        """评估数据集
        
        Args:
            datasets: UniversalDataset 对象列表
            
        Returns:
            包含评估结果的字典
        """
        # 确保模型已加载
        self.load_model()
        
        # 准备数据对 List[Tuple[str, str]]
        # 第一个元素是 source，第二个元素是 summary（模型生成的）
        pairs = []
        valid_indices = []  # 记录有效数据的索引
        
        for idx, data in enumerate(datasets):
            source = data.source if data.source else ""
            summary = data.summary if data.summary else ""
            
            # 只有当 source 和 summary 都存在时才添加
            if source and summary:
                pairs.append((source, summary))
                valid_indices.append(idx)
            else:
                print(f"警告: 数据 {data.id} 缺少 source 或 summary 字段，跳过")
        
        if not pairs:
            return {
                'error': '没有有效的数据对可以评估',
                'total_samples': len(datasets),
                'valid_samples': 0
            }
        
        # 使用模型进行预测
        print(f"正在评估 {len(pairs)} 个数据对...")
        predictions = self.model.predict(pairs)
        
        # 将预测结果转换为 numpy 数组（如果还不是的话）
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions.cpu() if hasattr(predictions, 'cpu') else predictions)
        
        # 将预测结果写回到 UniversalDataset 对象中
        for i, idx in enumerate(valid_indices):
            datasets[idx].model_prediction = float(predictions[i])
        
        # 计算评估指标
        threshold = 0.5
        
        # 1. Hallucination Rate（幻觉率）：幻觉评分低于0.5的汇总百分比
        hallucination_count = int(np.sum(predictions < threshold))
        hallucination_rate = (hallucination_count / len(pairs)) * 100 if pairs else 0
        
        # 2. Factual Consistency Rate（事实一致率）：1 - 幻觉率
        factual_consistency_rate = 100 - hallucination_rate
        
        # 3. Answer Rate（回答率）：非空摘要的百分比
        total_samples = len(datasets)
        answer_rate = (len(pairs) / total_samples) * 100 if total_samples > 0 else 0
        
        # 4. Average Summary Length（平均摘要长度）：生成的摘要的平均字数
        summary_lengths = []
        for idx in valid_indices:
            summary = datasets[idx].summary
            if summary:
                # 计算字数（按空格分割）
                word_count = len(summary.split())
                summary_lengths.append(word_count)
        
        avg_summary_length = float(np.mean(summary_lengths)) if summary_lengths else 0
        
        results = {
            'hallucination_rate': round(hallucination_rate, 2),  # 百分比
            'factual_consistency_rate': round(factual_consistency_rate, 2),  # 百分比
            'answer_rate': round(answer_rate, 2),  # 百分比
            'average_summary_length': round(avg_summary_length, 2),  # 平均字数
        }
        
        return results
    
    def evaluate_single(self, source: str, summary: str) -> float:
        """评估单个数据对
        
        Args:
            source: 源文本
            summary: 生成的摘要/文本
            
        Returns:
            幻觉分数（0-1之间，越高表示越可能存在幻觉）
        """
        # 确保模型已加载
        self.load_model()
        
        # 准备数据对
        pairs = [(source, summary)]
        
        # 进行预测
        predictions = self.model.predict(pairs)
        
        # 返回分数
        if isinstance(predictions, np.ndarray):
            return float(predictions[0])
        elif hasattr(predictions, 'cpu'):
            return float(predictions[0].cpu().item())
        else:
            return float(predictions[0])


# ============ HaluEval 摘要任务评估器 ============

class HaluEvalEvaluator:
    """HaluEval 摘要任务评估器
    
    评估模型对摘要中幻觉的判断能力
    模型需要判断摘要是否包含幻觉（Yes/No），然后与真实标签对比
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def extract_judgement(self, model_output: str) -> str:
        """从模型输出中提取判断结果（Yes/No）
        
        Args:
            model_output: 模型的原始输出文本
            
        Returns:
            提取的判断结果："Yes", "No", 或 "failed"
        """
        if not model_output:
            return "failed"
        
        # 移除句号
        ans = str(model_output).replace(".", "")
        
        # 判断逻辑（与 evaluate.py 中的逻辑一致）
        has_yes = "Yes" in ans
        has_no = "No" in ans
        
        if (has_yes and has_no) or (not has_yes and not has_no):
            return "failed"
        elif has_yes:
            return "Yes"
        elif has_no:
            return "No"
        else:
            return "failed"
    
    def evaluate(self, datasets: List[UniversalDataset]) -> Dict[str, Any]:
        """评估数据集
        
        Args:
            datasets: UniversalDataset 对象列表，每个对象应包含：
                     - ground_truth: 真实标签（"Yes" 或 "No"）
                     - model_prediction: 模型的原始输出
                     
        Returns:
            包含评估结果的字典
        """
        if not datasets:
            return {
                'error': '数据集为空',
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }
        
        correct = 0
        incorrect = 0
        failed = 0
        
        for idx, data in enumerate(datasets):
            # 获取真实标签
            ground_truth = data.ground_truth
            if ground_truth is None:
                ground_truth = data.label
            
            if ground_truth is None:
                print(f"警告: 数据 {data.id} 缺少 ground_truth 或 label 字段，跳过")
                failed += 1
                continue
            
            # 标准化 ground_truth
            ground_truth = str(ground_truth)
            
            # 获取模型预测
            model_output = data.model_prediction
            if model_output is None:
                print(f"警告: 数据 {data.id} 缺少 model_prediction 字段，跳过")
                failed += 1
                continue
            
            # 提取判断结果
            judgement = self.extract_judgement(model_output)
            
            # 判断是否正确
            if judgement == "failed":
                failed += 1
                print(f'样本 {idx} 判断失败（无法提取 Yes/No）')
            elif judgement == ground_truth:
                correct += 1
                print(f'样本 {idx} 判断正确')
            else:
                incorrect += 1
                print(f'样本 {idx} 判断错误（预测: {judgement}, 真实: {ground_truth}）')
        
        # 计算准确率（失败的样本计入 incorrect）
        total_samples = len(datasets)
        total_incorrect = incorrect + failed
        accuracy = (correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        results = {
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': round(accuracy, 2),  # 百分比
        }
        
        print(f'\n评估完成：')
        print(f'正确: {correct}')
        print(f'错误: {incorrect}')
        print(f'准确率: {accuracy:.2f}%')
        
        return results
    
    def evaluate_with_details(self, datasets: List[UniversalDataset]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """评估数据集并返回详细结果
        
        Args:
            datasets: UniversalDataset 对象列表
                     
        Returns:
            (评估指标字典, 详细结果列表)
        """
        if not datasets:
            return {
                'error': '数据集为空',
                'correct': 0,
                'incorrect': 0,
                'accuracy': 0.0
            }, []
        
        correct = 0
        incorrect = 0
        failed = 0
        detailed_results = []
        
        for idx, data in enumerate(datasets):
            # 获取真实标签
            ground_truth = data.ground_truth
            if ground_truth is None:
                ground_truth = data.label
            
            # 获取模型预测
            model_output = data.model_prediction
            
            # 提取判断结果
            if model_output is None:
                judgement = "failed"
                status = "no_prediction"
                failed += 1
            else:
                judgement = self.extract_judgement(model_output)
                
                if judgement == "failed":
                    status = "extraction_failed"
                    failed += 1
                elif judgement == str(ground_truth):
                    status = "correct"
                    correct += 1
                else:
                    status = "incorrect"
                    incorrect += 1
            
            # 记录详细结果
            detail = {
                'id': data.id,
                'document': data.source if data.source else data.context,
                'summary': data.summary,
                'ground_truth': str(ground_truth),
                'model_output': str(model_output) if model_output else None,
                'judgement': judgement,
                'status': status
            }
            detailed_results.append(detail)
        
        # 计算准确率
        total_samples = len(datasets)
        accuracy = (correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        metrics = {
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': round(accuracy, 2),
        }
        
        return metrics, detailed_results


# ============ RAGTruth 幻觉检测评估器 ============

class RAGTruthEvaluator:
    """RAGTruth 幻觉检测评估器
    
    评估模型对 QA、Summary 和 Data2txt 任务中幻觉的检测能力
    通过比较 is_halu 和 pred_halu，计算 recall、precision 和 f1 score
    """
    
    def __init__(self):
        """初始化评估器"""
        pass
    
    def extract_is_halu(self, data: UniversalDataset) -> bool:
        """提取真实的幻觉标签
        
        Args:
            data: UniversalDataset 对象
            
        Returns:
            是否存在幻觉的布尔值
        """
        # 优先使用 is_hallucination 字段
        if data.is_hallucination is not None:
            return bool(data.is_hallucination)
        
        # 否则通过 labels 或 hallucination_list 判断
        if data.labels and len(data.labels) > 0:
            return True
        
        if data.hallucination_list and len(data.hallucination_list) > 0:
            return True
        
        return False
    
    def extract_pred_halu(self, pred: Any) -> bool:
        """从预测结果中提取幻觉判断
        
        Args:
            pred: 模型预测结果（可以是字典、字符串等）
            
        Returns:
            是否预测为幻觉的布尔值
        """
        if pred is None:
            return False
        
        # 如果 pred 是字典格式
        if isinstance(pred, dict):
            hallucination_list = pred.get('hallucination list', [])
            if hallucination_list is None:
                return False
            return len(hallucination_list) > 0
        
        # 如果 pred 是字符串，尝试解析为 JSON
        if isinstance(pred, str):
            try:
                pred_dict = json.loads(pred)
                hallucination_list = pred_dict.get('hallucination list', [])
                if hallucination_list is None:
                    return False
                return len(hallucination_list) > 0
            except:
                # 如果无法解析，返回 False
                return False
        
        # 其他情况返回 False
        return False
    
    def evaluate(self, datasets: List[UniversalDataset]) -> Dict[str, Any]:
        """评估数据集
        
        Args:
            datasets: UniversalDataset 对象列表
                     
        Returns:
            包含评估结果的字典
        """
        if not datasets:
            return {
                'error': '数据集为空',
                'total_samples': 0
            }
        
        # 提取 is_halu 和 pred_halu
        is_halu_list = []
        pred_halu_list = []
        task_types = []
        
        for data in datasets:
            is_halu = self.extract_is_halu(data)
            pred_halu = self.extract_pred_halu(data.model_prediction)
            
            is_halu_list.append(is_halu)
            pred_halu_list.append(pred_halu)
            task_types.append(data.task_type.value if hasattr(data.task_type, 'value') else str(data.task_type))
        
        # 计算整体指标
        overall_metrics = self._calculate_metrics(is_halu_list, pred_halu_list, "Overall")
        
        # 按任务类型计算指标
        task_metrics = {}
        for task in ['QA', 'Summary', 'Data2txt']:
            # 筛选当前任务类型的样本
            task_indices = [i for i, t in enumerate(task_types) if t == task]
            
            if task_indices:
                task_is_halu = [is_halu_list[i] for i in task_indices]
                task_pred_halu = [pred_halu_list[i] for i in task_indices]
                
                task_metrics[task] = self._calculate_metrics(task_is_halu, task_pred_halu, task)
        
        # 组合结果
        results = {
            'total_samples': len(datasets),
            'overall': overall_metrics,
            'by_task': task_metrics
        }
        
        # 打印结果
        self._print_results(results)
        
        return results
    
    def _calculate_metrics(self, y_true: List[bool], y_pred: List[bool], label: str) -> Dict[str, float]:
        """计算评估指标
        
        Args:
            y_true: 真实标签列表
            y_pred: 预测标签列表
            label: 标签名称（用于打印）
            
        Returns:
            包含 recall、precision、f1 的字典
        """
        if len(y_true) == 0:
            return {
                'recall': 0.0,
                'precision': 0.0,
                'f1': 0.0,
                'sample_count': 0
            }
        
        # 转换为整数
        y_true_int = [int(y) for y in y_true]
        y_pred_int = [int(y) for y in y_pred]
        
        recall = recall_score(y_true_int, y_pred_int, zero_division=0)
        precision = precision_score(y_true_int, y_pred_int, zero_division=0)
        f1 = f1_score(y_true_int, y_pred_int, zero_division=0)
        
        return {
            'recall': round(recall, 3),
            'precision': round(precision, 3),
            'f1': round(f1, 3),
            'sample_count': len(y_true)
        }
    
    def _print_results(self, results: Dict[str, Any]):
        """打印评估结果
        
        Args:
            results: 评估结果字典
        """
        print(f"\n{'='*60}")
        print(f"RAGTruth 幻觉检测评估结果")
        print(f"{'='*60}")
        
        # 打印整体结果
        overall = results['overall']
        print(f"\n整体 - Case recall/precision/f1: {overall['recall']:.3f}, {overall['precision']:.3f}, {overall['f1']:.3f}")
        print(f"  总样本数: {overall['sample_count']}")
        
        # 打印各任务类型结果
        by_task = results['by_task']
        for task in ['QA', 'Summary', 'Data2txt']:
            if task in by_task:
                metrics = by_task[task]
                print(f"\n{task} - Case recall/precision/f1: {metrics['recall']:.3f}, {metrics['precision']:.3f}, {metrics['f1']:.3f}")
                print(f"  样本数: {metrics['sample_count']}")
        
        print(f"\n{'='*60}\n")
    
    def evaluate_with_details(self, datasets: List[UniversalDataset]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """评估数据集并返回详细结果
        
        Args:
            datasets: UniversalDataset 对象列表
                     
        Returns:
            (评估指标字典, 详细结果列表)
        """
        if not datasets:
            return {
                'error': '数据集为空',
                'total_samples': 0
            }, []
        
        # 提取数据并记录详细结果
        is_halu_list = []
        pred_halu_list = []
        task_types = []
        detailed_results = []
        
        for data in datasets:
            is_halu = self.extract_is_halu(data)
            pred_halu = self.extract_pred_halu(data.model_prediction)
            
            is_halu_list.append(is_halu)
            pred_halu_list.append(pred_halu)
            task_types.append(data.task_type.value if hasattr(data.task_type, 'value') else str(data.task_type))
            
            # 记录详细结果
            detail = {
                'id': data.id,
                'task_type': task_types[-1],
                'question': data.question,
                'source': data.source,
                'response': data.response,
                'is_halu': is_halu,
                'pred_halu': pred_halu,
                'correct': is_halu == pred_halu,
                'hallucination_list': data.hallucination_list,
                'model_prediction': data.model_prediction
            }
            detailed_results.append(detail)
        
        # 计算整体指标
        overall_metrics = self._calculate_metrics(is_halu_list, pred_halu_list, "Overall")
        
        # 按任务类型计算指标
        task_metrics = {}
        for task in ['QA', 'Summary', 'Data2txt']:
            task_indices = [i for i, t in enumerate(task_types) if t == task]
            
            if task_indices:
                task_is_halu = [is_halu_list[i] for i in task_indices]
                task_pred_halu = [pred_halu_list[i] for i in task_indices]
                
                task_metrics[task] = self._calculate_metrics(task_is_halu, task_pred_halu, task)
        
        # 组合结果
        metrics = {
            'total_samples': len(datasets),
            'overall': overall_metrics,
            'by_task': task_metrics
        }
        
        return metrics, detailed_results


# ============ HallusionBench VQA 评估器 ============

class HallusionBenchEvaluator:
    """HallusionBench VQA 评估器
    
    评估视觉问答任务中的幻觉检测能力
    根据 gt_answer_details 和 model_prediction 来评估正确性
    只考虑 visual_input=1 和 visual_input=2 的数据
    """
    
    def __init__(self, model_correctness_entry: str = "model_correctness"):
        """初始化评估器
        
        Args:
            model_correctness_entry: 存储模型正确性判断的字段名
        """
        self.model_correctness_entry = model_correctness_entry
    
    def assign_correctness(self, data_arr: List[UniversalDataset]) -> List[UniversalDataset]:
        """分配正确性标签
        
        Args:
            data_arr: UniversalDataset 对象列表
            
        Returns:
            添加了正确性标签的数据列表
        """
        for data in data_arr:
            correctness = data.metadata.get(self.model_correctness_entry, 0)
            correctness = int(correctness)
            
            assert correctness in [0, 1, 2], f"correctness 必须是 0、1 或 2，当前值: {correctness}"
            
            # 只有 correctness == 1 才算正确
            data.accuracy = 1.0 if correctness == 1 else 0.0
            
            # 存储到 metadata 中
            data.metadata['correct'] = int(data.accuracy)
        
        return data_arr
    
    def get_eval_fig(self, data: List[UniversalDataset]) -> Dict[str, Any]:
        """计算每张图的准确率（一致性测试）
        
        Args:
            data: UniversalDataset 对象列表
            
        Returns:
            包含图像级别评估结果的字典
        """
        eval_fig_dict = {}
        
        for r in data:
            name = "_".join([r.category, r.subcategory, str(r.set_id), str(r.figure_id)])
            correct_count = r.metadata.get('correct', 0)
            
            if name in eval_fig_dict:
                c, t = eval_fig_dict[name]
                eval_fig_dict[name] = (c + correct_count, t + 1)
            else:
                eval_fig_dict[name] = (correct_count, 1)
        
        eval_fig_stat = {
            'note': 'all accuracy per image (consistency test)',
            'total': len(eval_fig_dict),
            'correct': 0,
            'wrong': 0,
            'inconsistent': 0,
            'score': 0.0
        }
        
        for v in eval_fig_dict.values():
            if v[0] == v[1]:  # 所有问题都答对
                eval_fig_stat['correct'] += 1
            elif v[0] == 0:  # 所有问题都答错
                eval_fig_stat['wrong'] += 1
            else:  # 部分对部分错
                eval_fig_stat['inconsistent'] += 1
            eval_fig_stat['score'] += (v[0] / v[1])
        
        if eval_fig_stat['total'] > 0:
            eval_fig_stat['score'] = eval_fig_stat['score'] / eval_fig_stat['total']
        
        return eval_fig_stat
    
    def get_eval_all(self, data: List[UniversalDataset]) -> Dict[str, Any]:
        """计算每个问题的准确率
        
        Args:
            data: UniversalDataset 对象列表
            
        Returns:
            包含问题级别评估结果的字典
        """
        eval_all_dict = {}
        eval_all_stat = {
            'LH': 0,  # Language Hallucination
            'VI': 0,  # Vision Inadequate
            'Mix': 0  # Mixed
        }
        
        for r in data:
            question_id = r.metadata.get('question_id', '')
            name = "_".join([r.category, r.subcategory, str(r.set_id), 
                           str(r.figure_id), str(question_id)])
            
            assert name not in eval_all_dict, f"重复的问题 ID: {name}"
            
            correct_count = r.metadata.get('correct', 0)
            eval_all_dict[name] = correct_count
            
            # 根据类别和正确性判断错误类型
            correctness = r.metadata.get(self.model_correctness_entry, 0)
            correctness_str = str(correctness)
            
            if r.category == "VD":  # Visual Dependent
                if str(r.figure_id) == "0":
                    if correctness_str in ["0", "2"]:
                        eval_all_stat['VI'] += 1
                else:
                    if correctness_str == "0":
                        eval_all_stat['Mix'] += 1
                    elif correctness_str == "2":
                        eval_all_stat['VI'] += 1
            else:  # VS (Visual Supplement)
                # 只处理 visual_input == 1 or 2 的情况
                if correctness_str == "0":
                    eval_all_stat['Mix'] += 1
                elif correctness_str == "2":
                    eval_all_stat['VI'] += 1
        
        eval_all_stat['note'] = 'all accuracy per question'
        eval_all_stat['total'] = len(eval_all_dict)
        eval_all_stat['correct'] = sum(eval_all_dict.values())
        eval_all_stat['wrong'] = eval_all_stat['total'] - eval_all_stat['correct']
        
        return eval_all_stat
    
    def get_eval_pair_all(self, data: List[UniversalDataset]) -> Dict[str, Any]:
        """计算每个问题对的准确率
        
        Args:
            data: UniversalDataset 对象列表
            
        Returns:
            包含问题对级别评估结果的字典
        """
        # 收集原始问题（figure_id == 0）的正确性
        orig_correctness = {}
        for r in data:
            if str(r.figure_id) == "0":
                question_id = r.metadata.get('question_id', '')
                key = "_".join([r.category, r.subcategory, str(r.set_id), str(question_id)])
                orig_correctness[key] = r.metadata.get(self.model_correctness_entry, 0)
        
        get_eval_pair_dict = {}
        get_analysis_pair_dict = {}
        counter = 0
        lh_counter = 0
        vi_counter = 0
        both_counter = 0
        
        for r in data:
            question_id = r.metadata.get('question_id', '')
            name = "_".join([r.category, r.subcategory, str(r.set_id), str(question_id)])
            correct_count = r.metadata.get('correct', 0)
            
            if name in get_eval_pair_dict:
                c, t = get_eval_pair_dict[name]
                get_eval_pair_dict[name] = (c + correct_count, t + 1)
            else:
                get_eval_pair_dict[name] = (correct_count, 1)
            counter += 1
            
            # 分析错误类型 (LH, VI)
            analysis = (0, 0)
            correctness = str(r.metadata.get(self.model_correctness_entry, 0))
            same = str(r.metadata.get('same', '0'))
            
            if str(r.figure_id) == "0":  # 原始问题
                if r.category == "VD":
                    if correctness in ["0", "2"]:
                        analysis = (0, 1)  # VI
                # VS 类别不处理 figure_id == 0 的情况（已过滤 visual_input=0）
            else:  # 修改后的问题
                key = "_".join([r.category, r.subcategory, str(r.set_id), str(question_id)])
                orig_c = str(orig_correctness.get(key, 0))
                
                if r.category == "VD":
                    if orig_c == "1" and correctness == "0":
                        if same == "1":
                            analysis = (1, 1)  # Mixed
                        else:
                            analysis = (0, 1)  # VI
                    elif orig_c == "1" and correctness == "2":
                        analysis = (0, 1)  # VI
                    elif correctness in ["0", "2"]:
                        analysis = (0, 1)  # VI
                else:  # VS (只有 visual_input == 1 or 2)
                    visual_input = str(r.visual_input_type) if r.visual_input_type else "1"
                    
                    if orig_c == "2":
                        if correctness in ["0", "2"]:
                            analysis = (0, 1)  # VI
                    elif orig_c == "1":
                        if correctness == "2":
                            analysis = (0, 1)  # VI
                        elif correctness == "0":
                            if visual_input == "1":
                                analysis = (0, 1)  # VI
                            elif visual_input == "2":
                                if same == "1":
                                    analysis = (1, 0)  # LH
                                else:
                                    analysis = (0, 1)  # VI
            
            # 统计错误类型
            if analysis[0] > 0 and analysis[1] > 0:
                both_counter += 1
            elif analysis[0] > 0:
                lh_counter += 1
            elif analysis[1] > 0:
                vi_counter += 1
            
            if name in get_analysis_pair_dict:
                lh, vi = get_analysis_pair_dict[name]
                get_analysis_pair_dict[name] = (lh + analysis[0], vi + analysis[1])
            else:
                get_analysis_pair_dict[name] = analysis
        
        eval_all_pair_stat = {
            'note': 'all accuracy per question pair',
            'total': len(get_eval_pair_dict),
            'total_q': counter,
            'correct': 0,
            'wrong': 0,
            'LH': 0,
            'VI': 0,
            'Mix': 0,
            'LH_cg': lh_counter,
            'VI_cg': vi_counter,
            'Mix_cg': both_counter
        }
        
        for k in get_eval_pair_dict.keys():
            v = get_eval_pair_dict[k]
            a = get_analysis_pair_dict[k]
            
            if v[0] == v[1]:
                eval_all_pair_stat['correct'] += 1
            else:
                eval_all_pair_stat['wrong'] += 1
            
            if a[0] > 0 and a[1] > 0:
                eval_all_pair_stat['Mix'] += 1
            elif a[0] > 0:
                eval_all_pair_stat['LH'] += 1
            elif a[1] > 0:
                eval_all_pair_stat['VI'] += 1
        
        assert eval_all_pair_stat['wrong'] == (eval_all_pair_stat['Mix'] + 
                                                 eval_all_pair_stat['LH'] + 
                                                 eval_all_pair_stat['VI'])
        
        return eval_all_pair_stat
    
    def get_eval_pair_easy(self, data: List[UniversalDataset]) -> Dict[str, Any]:
        """计算简单问题对的准确率（visual_input != 2）
        
        Args:
            data: UniversalDataset 对象列表
            
        Returns:
            包含简单问题对评估结果的字典
        """
        get_eval_pair_dict = {}
        counter = 0
        
        for r in data:
            visual_input = str(r.visual_input_type) if r.visual_input_type else "0"
            if visual_input == "2":
                continue
            
            question_id = r.metadata.get('question_id', '')
            name = "_".join([r.category, r.subcategory, str(r.set_id), str(question_id)])
            correct_count = r.metadata.get('correct', 0)
            
            if name in get_eval_pair_dict:
                c, t = get_eval_pair_dict[name]
                get_eval_pair_dict[name] = (c + correct_count, t + 1)
            else:
                get_eval_pair_dict[name] = (correct_count, 1)
            counter += 1
        
        eval_all_pair_stat = {
            'note': 'easy accuracy per question pair',
            'total': len(get_eval_pair_dict),
            'total_q': counter,
            'correct': 0,
            'wrong': 0
        }
        
        for v in get_eval_pair_dict.values():
            if v[0] == v[1]:
                eval_all_pair_stat['correct'] += 1
            else:
                eval_all_pair_stat['wrong'] += 1
        
        return eval_all_pair_stat
    
    def get_eval_pair_hard(self, data: List[UniversalDataset]) -> Dict[str, Any]:
        """计算困难问题对的准确率（visual_input == 2）
        
        Args:
            data: UniversalDataset 对象列表
            
        Returns:
            包含困难问题对评估结果的字典
        """
        get_eval_pair_dict = {}
        counter = 0
        
        for r in data:
            visual_input = str(r.visual_input_type) if r.visual_input_type else "0"
            if visual_input != "2":
                continue
            
            question_id = r.metadata.get('question_id', '')
            name = "_".join([r.category, r.subcategory, str(r.set_id), str(question_id)])
            correct_count = r.metadata.get('correct', 0)
            
            if name in get_eval_pair_dict:
                c, t = get_eval_pair_dict[name]
                get_eval_pair_dict[name] = (c + correct_count, t + 1)
            else:
                get_eval_pair_dict[name] = (correct_count, 1)
            counter += 1
        
        eval_all_pair_stat = {
            'note': 'hard accuracy per question pair',
            'total': len(get_eval_pair_dict),
            'total_q': counter,
            'correct': 0,
            'wrong': 0
        }
        
        for v in get_eval_pair_dict.values():
            if v[0] == v[1]:
                eval_all_pair_stat['correct'] += 1
            else:
                eval_all_pair_stat['wrong'] += 1
        
        return eval_all_pair_stat
    
    def yes_ratio_stats(self, data: List[UniversalDataset]) -> Dict[str, float]:
        """计算 Yes/No 偏差统计
        
        Args:
            data: UniversalDataset 对象列表
            
        Returns:
            包含偏差统计的字典
        """
        yes_gt = []
        yes_pred = []
        fp_sample = []
        
        for i in data:
            gt_answer = i.gt_answer
            correct = i.metadata.get('correct', 0)
            
            # gt_answer 转为整数（假设 "0" 表示 No, "1" 表示 Yes）
            yes_gt.append(int(gt_answer) if gt_answer else 0)
            
            # 预测是否与真实答案一致
            yes_pred.append(int(correct == int(gt_answer)) if gt_answer else 0)
            
            # 收集错误预测的样本
            if correct == 0:
                fp_sample.append(i)
        
        fp = [int(i.gt_answer) if i.gt_answer else 0 for i in fp_sample]
        
        stats = {}
        if len(yes_pred) > 0:
            stats['diff'] = round(sum(yes_pred) / len(yes_pred) - sum(yes_gt) / len(yes_gt), 4)
        else:
            stats['diff'] = 0.0
        
        if len(fp) > 0:
            stats['fp'] = round((len(fp) - sum(fp)) / len(fp), 4)
        else:
            stats['fp'] = 0.0
        
        return stats
    
    def evaluate(self, datasets: List[UniversalDataset]) -> Dict[str, Any]:
        """评估数据集
        
        Args:
            datasets: UniversalDataset 对象列表，应包含：
                     - category: 类别（VD 或 VS）
                     - subcategory: 子类别
                     - set_id: 集合ID
                     - figure_id: 图像ID
                     - question: 问题文本
                     - gt_answer: 真实答案（"0" 或 "1"）
                     - gt_answer_details: 真实答案详情
                     - model_prediction: 模型预测
                     - metadata[model_correctness_entry]: 模型正确性（0/1/2）
                     
        Returns:
            包含评估结果的字典
        """
        if not datasets:
            return {
                'error': '数据集为空',
                'total_samples': 0
            }
        
        # 分离 VD 和 VS 类别
        data_vd = [d for d in datasets if d.category == 'VD']
        data_vs = [d for d in datasets if d.category == 'VS']
        
        # 分配正确性标签
        data_vd = self.assign_correctness(data_vd)
        data_vs = self.assign_correctness(data_vs)
        data = data_vd + data_vs
        
        # 计算各项指标
        # 1. 每个问题的准确率
        all_data = self.get_eval_all(data)
        all_vd = self.get_eval_all(data_vd)
        all_vs = self.get_eval_all(data_vs)
        
        # 2. 每个问题对的准确率
        all_data_pair = self.get_eval_pair_all(data)
        easy = self.get_eval_pair_easy(data)
        hard = self.get_eval_pair_hard(data)
        all_vd_pair = self.get_eval_pair_all(data_vd)
        easy_vd = self.get_eval_pair_easy(data_vd)
        hard_vd = self.get_eval_pair_hard(data_vd)
        all_vs_pair = self.get_eval_pair_all(data_vs)
        easy_vs = self.get_eval_pair_easy(data_vs)
        hard_vs = self.get_eval_pair_hard(data_vs)
        
        # 3. 每张图的准确率
        fig_all = self.get_eval_fig(data)
        fig_vd = self.get_eval_fig(data_vd)
        fig_vs = self.get_eval_fig(data_vs)
        
        # 4. Yes/No 偏差统计
        stats = self.yes_ratio_stats(data)
        
        # 计算关键指标
        q_acc = round(100 * all_data['correct'] / all_data['total'], 2) if all_data['total'] > 0 else 0.0
        pair_acc = round(100 * all_data_pair['correct'] / all_data_pair['total'], 2) if all_data_pair['total'] > 0 else 0.0
        figure_acc = round(100 * fig_all['correct'] / fig_all['total'], 2) if fig_all['total'] > 0 else 0.0
        easy_acc = round(100 * easy['correct'] / easy['total'], 2) if easy['total'] > 0 else 0.0
        hard_acc = round(100 * hard['correct'] / hard['total'], 2) if hard['total'] > 0 else 0.0
        
        # 组合结果
        results = {
            'total_samples': len(datasets),
            'total_vd': len(data_vd),
            'total_vs': len(data_vs),
            
            # 主要指标
            'accuracy_per_question': q_acc,
            'accuracy_per_question_pair': pair_acc,
            'accuracy_per_figure': figure_acc,
            'accuracy_easy': easy_acc,
            'accuracy_hard': hard_acc,
            
            # 每个问题的准确率（按类别）
            'per_question': {
                'VD': round(100 * all_vd['correct'] / all_vd['total'], 2) if all_vd['total'] > 0 else 0.0,
                'VS': round(100 * all_vs['correct'] / all_vs['total'], 2) if all_vs['total'] > 0 else 0.0,
                'Overall': q_acc
            },
            
            # 每个问题对的准确率（按类别和难度）
            'per_question_pair': {
                'VD': {
                    'Easy': round(100 * easy_vd['correct'] / easy_vd['total'], 2) if easy_vd['total'] > 0 else 0.0,
                    'Hard': round(100 * hard_vd['correct'] / hard_vd['total'], 2) if hard_vd['total'] > 0 else 0.0,
                    'Total': round(100 * all_vd_pair['correct'] / all_vd_pair['total'], 2) if all_vd_pair['total'] > 0 else 0.0
                },
                'VS': {
                    'Easy': round(100 * easy_vs['correct'] / easy_vs['total'], 2) if easy_vs['total'] > 0 else 0.0,
                    'Hard': round(100 * hard_vs['correct'] / hard_vs['total'], 2) if hard_vs['total'] > 0 else 0.0,
                    'Total': round(100 * all_vs_pair['correct'] / all_vs_pair['total'], 2) if all_vs_pair['total'] > 0 else 0.0
                },
                'Overall': {
                    'Easy': easy_acc,
                    'Hard': hard_acc,
                    'Total': pair_acc
                }
            },
            
            # 每张图的准确率（按类别）
            'per_figure': {
                'VD': {
                    'correct': round(100 * fig_vd['correct'] / fig_vd['total'], 2) if fig_vd['total'] > 0 else 0.0,
                    'wrong': round(100 * (fig_vd['inconsistent'] + fig_vd['wrong']) / fig_vd['total'], 2) if fig_vd['total'] > 0 else 0.0,
                    'score': round(fig_vd['score'], 4)
                },
                'VS': {
                    'correct': round(100 * fig_vs['correct'] / fig_vs['total'], 2) if fig_vs['total'] > 0 else 0.0,
                    'wrong': round(100 * (fig_vs['inconsistent'] + fig_vs['wrong']) / fig_vs['total'], 2) if fig_vs['total'] > 0 else 0.0,
                    'score': round(fig_vs['score'], 4)
                },
                'Overall': {
                    'correct': figure_acc,
                    'wrong': round(100 * (fig_all['inconsistent'] + fig_all['wrong']) / fig_all['total'], 2) if fig_all['total'] > 0 else 0.0,
                    'score': round(fig_all['score'], 4)
                }
            },
            
            # 统计信息
            'statistics': {
                'easy_questions': f"{easy_vd['total_q']} (VD) + {easy_vs['total_q']} (VS)",
                'hard_questions': f"{hard_vd['total_q']} (VD) + {hard_vs['total_q']} (VS)",
                'total_questions': all_data_pair['total_q'],
                'vd_figures': fig_vd['total'],
                'vs_figures': fig_vs['total'],
                'total_figures': fig_all['total']
            },
            
            # 偏差和一致性测试
            'bias_and_consistency': {
                'yes_no_bias_pct_diff': stats['diff'],
                'yes_no_bias_fp_ratio': stats['fp'],
                'consistency_correct': round(100 * fig_all['correct'] / fig_all['total'], 2) if fig_all['total'] > 0 else 0.0,
                'consistency_inconsistent': round(100 * fig_all['inconsistent'] / fig_all['total'], 2) if fig_all['total'] > 0 else 0.0,
                'consistency_wrong': round(100 * fig_all['wrong'] / fig_all['total'], 2) if fig_all['total'] > 0 else 0.0,
                'LH': round(100 * all_data_pair['LH_cg'] / (all_data_pair['LH_cg'] + all_data_pair['VI_cg'] + all_data_pair['Mix_cg']), 2) if (all_data_pair['LH_cg'] + all_data_pair['VI_cg'] + all_data_pair['Mix_cg']) > 0 else 0.0,
                'VI': round(100 * all_data_pair['VI_cg'] / (all_data_pair['LH_cg'] + all_data_pair['VI_cg'] + all_data_pair['Mix_cg']), 2) if (all_data_pair['LH_cg'] + all_data_pair['VI_cg'] + all_data_pair['Mix_cg']) > 0 else 0.0,
                'Mixed': round(100 * all_data_pair['Mix_cg'] / (all_data_pair['LH_cg'] + all_data_pair['VI_cg'] + all_data_pair['Mix_cg']), 2) if (all_data_pair['LH_cg'] + all_data_pair['VI_cg'] + all_data_pair['Mix_cg']) > 0 else 0.0
            }
        }
        
        # 打印结果
        self._print_results(results)
        
        return results
    
    def _print_results(self, results: Dict[str, Any]):
        """打印评估结果（使用 PrettyTable 格式）
        
        Args:
            results: 评估结果字典
        """
        try:
            from prettytable import PrettyTable
            use_prettytable = True
        except ImportError:
            use_prettytable = False
        
        print(f"\n{'='*80}")
        print(f"HallusionBench VQA 评估结果")
        print(f"{'='*80}")
        
        # 问题统计
        print("\n##### 问题统计 #####")
        stats = results['statistics']
        print(f"简单问题: {stats['easy_questions']}")
        print(f"困难问题: {stats['hard_questions']}")
        print(f"总问题数: {stats['total_questions']}")
        
        # 图像统计
        print("\n##### 图像统计 #####")
        print(f"视觉依赖图像 (VD): {stats['vd_figures']}")
        print(f"视觉补充图像 (VS): {stats['vs_figures']}")
        print(f"总图像数: {stats['total_figures']}")
        
        # 主要指标
        print("\n##### 排行榜指标 #####")
        if use_prettytable:
            table = PrettyTable()
            table.field_names = ["", "qAcc (问题对)", "fAcc (图像)", "easy aAcc", "hard aAcc", "aAcc (问题)"]
            table.add_row([
                "评估结果",
                results['accuracy_per_question_pair'],
                results['accuracy_per_figure'],
                results['accuracy_easy'],
                results['accuracy_hard'],
                results['accuracy_per_question']
            ])
            print(table)
        else:
            print(f"问题对准确率 (qAcc): {results['accuracy_per_question_pair']}")
            print(f"图像准确率 (fAcc): {results['accuracy_per_figure']}")
            print(f"简单问题准确率 (easy aAcc): {results['accuracy_easy']}")
            print(f"困难问题准确率 (hard aAcc): {results['accuracy_hard']}")
            print(f"问题准确率 (aAcc): {results['accuracy_per_question']}")
        
        # 每个问题的准确率
        print("\n##### 每个问题的准确率 #####")
        if use_prettytable:
            table = PrettyTable()
            table.field_names = ["", "Total"]
            pq = results['per_question']
            table.add_row(["VD", pq['VD']])
            table.add_row(["VS", pq['VS']])
            table.add_row(["Overall", pq['Overall']])
            print(table)
        else:
            pq = results['per_question']
            print(f"VD: {pq['VD']}")
            print(f"VS: {pq['VS']}")
            print(f"Overall: {pq['Overall']}")
        
        # 每个问题对的准确率
        print("\n##### 每个问题对的准确率 #####")
        if use_prettytable:
            table = PrettyTable()
            table.field_names = ["", "Easy", "Hard", "Total"]
            pqp = results['per_question_pair']
            table.add_row(["VD", pqp['VD']['Easy'], pqp['VD']['Hard'], pqp['VD']['Total']])
            table.add_row(["VS", pqp['VS']['Easy'], pqp['VS']['Hard'], pqp['VS']['Total']])
            table.add_row(["Overall", pqp['Overall']['Easy'], pqp['Overall']['Hard'], pqp['Overall']['Total']])
            print(table)
        else:
            pqp = results['per_question_pair']
            print(f"VD - Easy: {pqp['VD']['Easy']}, Hard: {pqp['VD']['Hard']}, Total: {pqp['VD']['Total']}")
            print(f"VS - Easy: {pqp['VS']['Easy']}, Hard: {pqp['VS']['Hard']}, Total: {pqp['VS']['Total']}")
            print(f"Overall - Easy: {pqp['Overall']['Easy']}, Hard: {pqp['Overall']['Hard']}, Total: {pqp['Overall']['Total']}")
        
        # 偏差和一致性测试
        print("\n##### 偏差和一致性测试 #####")
        if use_prettytable:
            table = PrettyTable()
            table.field_names = ["Yes/No偏差(差值)", "Yes/No偏差(FP比)", "一致性(正确)", 
                               "一致性(不一致)", "一致性(错误)", "LH", "VI", "Mixed"]
            bc = results['bias_and_consistency']
            table.add_row([
                bc['yes_no_bias_pct_diff'],
                bc['yes_no_bias_fp_ratio'],
                bc['consistency_correct'],
                bc['consistency_inconsistent'],
                bc['consistency_wrong'],
                bc['LH'],
                bc['VI'],
                bc['Mixed']
            ])
            print(table)
        else:
            bc = results['bias_and_consistency']
            print(f"Yes/No 偏差 (百分比差值): {bc['yes_no_bias_pct_diff']}")
            print(f"Yes/No 偏差 (FP 比率): {bc['yes_no_bias_fp_ratio']}")
            print(f"一致性测试 - 正确: {bc['consistency_correct']}, 不一致: {bc['consistency_inconsistent']}, 错误: {bc['consistency_wrong']}")
            print(f"错误类型 - LH: {bc['LH']}, VI: {bc['VI']}, Mixed: {bc['Mixed']}")
        
        print(f"\n{'='*80}\n")