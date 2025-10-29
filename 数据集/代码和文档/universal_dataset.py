"""
通用数据集类设计
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import random


class TaskType(Enum):
    """任务类型枚举"""
    QA = "QA"  # 问答任务
    SUMMARY = "Summary"  # 摘要任务
    DATA2TXT = "Data2txt"  # 数据到文本
    VQA = "VQA"  # 视觉问答

class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"  # 纯文本
    MULTIMODAL = "multimodal"  # 多模态


class LabelType(Enum):
    """标签类型枚举"""
    BINARY = "binary"  # 二分类 (0/1, Yes/No)


@dataclass
class UniversalDataset:
    """
    通用数据集类
    
    设计原则：
    1. 可重复的参数共用一个类属性
    2. 不能共用或功能完全不同的需要新增类属性
    3. 一个数据集不需要用到所有类属性
    4. 不同数据集的同一类属性功能必须一致
    """
    
    # ============ 基础标识属性 ============
    id: str = ""  # 样本唯一标识符
    dataset_name: str = ""  # 数据集名称
    split: str = "test"
    
    # ============ 任务相关属性 ============
    task_type: TaskType = TaskType.QA  # 任务类型
    modality: ModalityType = ModalityType.TEXT  # 模态类型
    
    # ============ 输入相关属性 ============
    # 文本输入
    question: Optional[str] = None  # 问题文本（用于QA、VQA等任务）
    context: Optional[str] = None  # 上下文文本（用于RAG、摘要等）
    source: Optional[str] = None  # 源文本（用于摘要、翻译等）
    
    # 视觉输入
    image_path: Optional[str] = None  # 图像文件路径
    visual_input_type: Optional[str] = None
    
    # ============ 输出/答案相关属性 ============
    response: Optional[str] = None  # 模型响应
    summary: Optional[str] = None  # 摘要文本
    
    # ============ 标注相关属性 ============
    # 真实标签
    ground_truth: Optional[Union[str, int, float, List]] = None  # 真实答案（通用）
    gt_answer: Optional[str] = None  # 真实答案（文本形式）
    gt_answer_details: Optional[str] = None  # 真实答案详情
    
    # 标签信息
    label_type: LabelType = LabelType.BINARY  # 标签类型
    labels: Optional[List[str]] = None  # 标签列表（用于多标签任务）
    label: Optional[Union[str, int, float]] = None  # 单一标签
    
    # 幻觉相关标注
    is_hallucination: Optional[bool] = None  # 是否存在幻觉
    hallucination_type: Optional[str] = None  # 幻觉类型
    hallucination_list: Optional[List[str]] = None  # 幻觉列表（具体的幻觉片段）
    
    # ============ 分类/层级属性 ============
    category: Optional[str] = None  # 主类别 (如: VD/VS)
    subcategory: Optional[str] = None  # 子类别 (如: chart/map/table)
    
    # ============ 分组/配对属性 ============
    set_id: Optional[str] = None  # 集合ID（用于将多个样本分组）
    figure_id: Optional[str] = None  # 图像ID
    group_id: Optional[str] = None  # 分组ID
    
    # ============ 预测/评估相关属性 ============
    model_prediction: Optional[Union[str, int, float, List]] = None  # 模型预测
    
    # ============ 评估指标相关属性 ============
    accuracy: Optional[float] = None  # 准确率
    f1_score: Optional[float] = None  # F1分数
    precision: Optional[float] = None  # 精确率
    recall: Optional[float] = None  # 召回率
    
    # ============ 元数据属性 ============
    sample_note: Optional[str] = None  # 样本备注
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外的元数据


# ============ 数据集适配器 ============

class HallusionBenchAdapter(DatasetAdapter):
    """HallusionBench数据集适配器"""
    
    @staticmethod
    def adapt(data: Dict[str, Any], dataset_name: str = "HallusionBench") -> UniversalDataset:
        # 提取 question_id 用于生成 id 和存储到 metadata
        question_id = data.get('question_id', '')
        
        return UniversalDataset(
            id=f"{question_id}",
            dataset_name=dataset_name,
            task_type=TaskType.VQA,
            modality=ModalityType.MULTIMODAL,
            
            # 输入
            question=data.get('question'),
            image_path=data.get('filename'),
            visual_input_type=data.get('visual_input'),
            
            # 标注
            gt_answer=data.get('gt_answer'),
            gt_answer_details=data.get('gt_answer_details'),
            label_type=LabelType.BINARY,
            
            # 分类
            category=data.get('category'),
            subcategory=data.get('subcategory'),
            
            # 分组
            set_id=data.get('set_id'),
            figure_id=data.get('figure_id'),
            
            # 备注
            sample_note=data.get('sample_note'),
            
            # 预测
            model_prediction=data.get('model_prediction'),
            
            # 元数据（存储 question_id 用于评估）
            metadata={
                'question_id': question_id
            }
        )


class HaluEvalAdapter(DatasetAdapter):
    """HaluEval数据集适配器"""
    
    @staticmethod
    def adapt(data: Dict[str, Any], dataset_name: str = "HaluEval") -> UniversalDataset:
        # 随机选择使用幻觉摘要还是正确摘要
        use_hallucinated = random.random() > 0.5
        
        if use_hallucinated:
            summary = data.get('hallucinated_summary')
            ground_truth = "Yes"
            is_hallucination = True
        else:
            summary = data.get('right_summary')
            ground_truth = "No"
            is_hallucination = False
            
        return UniversalDataset(
            id=data.get('id', ''),
            dataset_name=dataset_name,
            task_type=TaskType.SUMMARY,
            modality=ModalityType.TEXT,
            
            # 输入
            source=data.get('document'),
            context=data.get('document'),
            summary=summary,
            
            # 标注
            ground_truth=ground_truth,
            gt_answer=data.get('right_summary'),  # 正确摘要作为参考
            is_hallucination=is_hallucination,
            hallucination_type="summarization",
            
            # 标签信息
            label_type=LabelType.BINARY,
            label=ground_truth,
            
            # 元数据
            metadata={
                'right_summary': data.get('right_summary'),
                'hallucinated_summary': data.get('hallucinated_summary'),
                'selected_summary_type': 'hallucinated' if use_hallucinated else 'right'
            }
        )


class RAGTruthRawAdapter(DatasetAdapter):
    """RAGTruth原始数据适配器
    
    注意：此适配器用于处理原始的source_info.jsonl和response.jsonl文件。
    """
    
    @staticmethod
    def adapt(data: Dict[str, Any], dataset_name: str = "RAGTruth") -> UniversalDataset:
        # 处理标签格式转换（来自response.jsonl）
        labels = data.get('labels', [])
        hallucination_list = []
        format_label = {'baseless info': [], 'conflict': []}
        
        if labels:
            # 从labels中提取具体的幻觉文本片段，并按类型分类
            for label in labels:
                text = label.get('text', '')
                if text:
                    hallucination_list.append(text)
                    label_type = label.get('label_type', '').lower()
                    if 'baseless' in label_type:
                        format_label['baseless info'].append(text)
                    elif 'conflict' in label_type:
                        format_label['conflict'].append(text)
        
        # 处理任务类型（来自source_info.jsonl）
        task_type = data.get('task_type', 'QA')
        try:
            task_type_enum = TaskType(task_type)
        except:
            task_type_enum = TaskType.QA  # 默认为QA任务
        
        # 根据任务类型提取reference和question
        source_info = data.get('source_info', {})
        question = None
        reference = None
        
        if task_type == 'QA':
            # QA任务：source_info是字典，包含question和passages
            if isinstance(source_info, dict):
                question = source_info.get('question')
                reference = source_info.get('passages')
        elif task_type == 'Summary':
            # Summary任务：source_info是字符串（原始文档）
            reference = source_info if isinstance(source_info, str) else str(source_info)
        elif task_type == 'Data2txt':
            # Data2txt任务：source_info是结构化数据
            reference = str(source_info)
        
        # 确定幻觉类型
        has_baseless = format_label.get('baseless info', [])
        has_conflict = format_label.get('conflict', [])
        
        if has_baseless and has_conflict:
            hallucination_type = "both"
        elif has_baseless:
            hallucination_type = "baseless"
        elif has_conflict:
            hallucination_type = "conflict"
        else:
            hallucination_type = "none"
        
        # 构建通用数据集实例
        return UniversalDataset(
            id=str(data.get('id', '')),
            dataset_name=dataset_name,
            split=data.get('split', 'train'),
            task_type=task_type_enum,
            modality=ModalityType.TEXT,
            
            # 输入相关
            question=question,              # 仅QA任务有此字段
            source=reference,               # 参考文本/源文档
            context=reference,              # 与source保持一致
            response=data.get('response'),  # 模型生成的响应（来自response.jsonl）
            
            # 标注相关
            labels=labels,
            is_hallucination=len(labels) > 0,
            hallucination_list=hallucination_list,
            label_type=LabelType.BINARY,
            
            # 幻觉类型分类
            hallucination_type=hallucination_type,
            
            # 分组相关
            group_id=data.get('source_id'),  # 使用source_id作为分组ID
            
            # 元数据
            metadata={
                'model': data.get('model'),
                'temperature': data.get('temperature'),
                'quality': data.get('quality'),
                'source_id': data.get('source_id'),
                'source': data.get('source'),  # 数据来源（如CNN/DM, MARCO等）
                'prompt': data.get('prompt'),  # 原始prompt
                'format_label': format_label,
                'raw_source_info': source_info
            },
            
            # 预测相关（用于评估时）
            model_prediction=data.get('pred')
        )

class VectaraAdapter(DatasetAdapter):
    """Vectara数据集适配器"""
    
    @staticmethod
    def adapt(data: Dict[str, Any], dataset_name: str = "Vectara") -> UniversalDataset:
        # 从Excel数据中提取相关信息
        source_text = data.get('source', '')
        
        return UniversalDataset(
            id=data.get('id', ''),
            dataset_name=dataset_name,
            task_type=TaskType.SUMMARY,
            modality=ModalityType.TEXT,
            
            # 输入 - 主要使用source字段作为源文本
            source=source_text,
        )

