# GPU选择使用指南

## 🎮 查看可用GPU

运行任意检测器时会自动显示：
```bash
python bartscore_detector_improved.py
```

输出示例：
```
检测到 2 张GPU卡:
  GPU 0: NVIDIA GeForce RTX 3090
  GPU 1: NVIDIA GeForce RTX 3080
```

## 🚀 使用方法

### 方法一：命令行参数指定（推荐）

#### 1. 改进版检测器

```bash
# 使用GPU 0
python bartscore_detector_improved.py --gpu 0

# 使用GPU 1
python bartscore_detector_improved.py --gpu 1

# 不指定，自动选择第一张可用GPU
python bartscore_detector_improved.py
```

**完整参数示例：**
```bash
python bartscore_detector_improved.py \
    --gpu 1 \
    --input ../data/test_response_label.jsonl \
    --output bartscore_improved_results.jsonl
```

**查看所有参数：**
```bash
python bartscore_detector_improved.py --help
```

输出：
```
usage: bartscore_detector_improved.py [-h] [--gpu GPU] [--input INPUT] [--output OUTPUT] 
                                       [--no-bidirectional] [--model MODEL]

改进版BARTScore幻觉检测器

optional arguments:
  -h, --help         show this help message and exit
  --gpu GPU          指定GPU ID (0, 1, 2, ...)，不指定则自动选择
  --input INPUT      输入文件路径
  --output OUTPUT    输出文件路径
  --no-bidirectional 禁用双向评分
  --model MODEL      BART模型名称
```

#### 2. 原版检测器

```bash
# 使用GPU 0
python bartscore_detector.py --gpu 0

# 使用GPU 1
python bartscore_detector.py --gpu 1

# 指定阈值和批处理大小
python bartscore_detector.py \
    --gpu 1 \
    --threshold -1.8649 \
    --batch-size 8
```

**查看所有参数：**
```bash
python bartscore_detector.py --help
```

---

### 方法二：环境变量设置

**Linux/Mac:**
```bash
# 只使用GPU 0
export CUDA_VISIBLE_DEVICES=0
python bartscore_detector_improved.py

# 只使用GPU 1
export CUDA_VISIBLE_DEVICES=1
python bartscore_detector_improved.py

# 使用GPU 0和1（多卡）
export CUDA_VISIBLE_DEVICES=0,1
python bartscore_detector_improved.py

# 临时设置（仅本次运行）
CUDA_VISIBLE_DEVICES=1 python bartscore_detector_improved.py
```

**Windows (PowerShell):**
```powershell
# 设置环境变量
$env:CUDA_VISIBLE_DEVICES="1"
python bartscore_detector_improved.py

# 或者临时设置
$env:CUDA_VISIBLE_DEVICES="1"; python bartscore_detector_improved.py
```

**Windows (CMD):**
```cmd
set CUDA_VISIBLE_DEVICES=1
python bartscore_detector_improved.py
```

---

### 方法三：在Python代码中设置

如果你需要在脚本中直接调用函数：

```python
import torch
import os

# 方式1: 设置环境变量（必须在import torch之前）
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# 方式2: 使用gpu_id参数
from bartscore_detector_improved import process_dataset_improved

process_dataset_improved(
    input_file='../data/test_response_label.jsonl',
    output_file='bartscore_improved_results.jsonl',
    gpu_id=1  # 指定使用GPU 1
)
```

---

## 📊 多GPU并行（高级）

如果想同时使用两张卡处理不同任务：

**终端1（使用GPU 0）：**
```bash
python bartscore_detector_improved.py \
    --gpu 0 \
    --input ../data/test_response_label_part1.jsonl \
    --output results_part1.jsonl
```

**终端2（使用GPU 1）：**
```bash
python bartscore_detector_improved.py \
    --gpu 1 \
    --input ../data/test_response_label_part2.jsonl \
    --output results_part2.jsonl
```

---

## 🔍 监控GPU使用情况

**实时监控：**
```bash
# 每秒刷新一次
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

**查看特定GPU：**
```bash
nvidia-smi -i 0  # 查看GPU 0
nvidia-smi -i 1  # 查看GPU 1
```

**查看GPU内存使用：**
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

---

## ⚠️ 常见问题

### 1. 如何选择使用哪张GPU？

**查看GPU负载：**
```bash
nvidia-smi
```

选择负载较低、显存剩余较多的GPU。

### 2. GPU显存不足怎么办？

**方法1：清空GPU缓存**
```python
import torch
torch.cuda.empty_cache()
```

**方法2：减小batch size**
```bash
python bartscore_detector.py --batch-size 2  # 从4减少到2
```

**方法3：使用显存更大的GPU**
```bash
python bartscore_detector_improved.py --gpu 1  # 如果GPU 1显存更大
```

### 3. 如何验证正在使用哪张GPU？

运行程序时会打印：
```
指定使用GPU: 1
加载改进版BARTScore模型: facebook/bart-large-cnn
使用设备: cuda:1
```

同时在另一个终端运行：
```bash
watch -n 1 nvidia-smi
```

观察哪张GPU的显存占用增加。

### 4. 两张卡性能不同，如何选择？

**查看GPU性能：**
```bash
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv
```

一般选择：
- 显存更大的GPU（用于大模型）
- 算力更强的GPU（看CUDA Cores数量）
- 负载更低的GPU（避免与其他任务冲突）

---

## 💡 最佳实践

### 推荐配置（双卡场景）

**GPU 0**: 运行改进版检测器（计算量更大，需要双向评分）
```bash
CUDA_VISIBLE_DEVICES=0 python bartscore_detector_improved.py
```

**GPU 1**: 运行原版检测器或其他实验
```bash
CUDA_VISIBLE_DEVICES=1 python bartscore_detector.py
```

### 性能优化建议

1. **显存充足**：增大batch size提速
   ```bash
   python bartscore_detector.py --gpu 0 --batch-size 8
   ```

2. **显存紧张**：减小batch size
   ```bash
   python bartscore_detector.py --gpu 0 --batch-size 2
   ```

3. **长时间运行**：使用nohup后台运行
   ```bash
   nohup python bartscore_detector_improved.py --gpu 1 > output.log 2>&1 &
   ```

4. **同时运行多个任务**：使用screen或tmux
   ```bash
   # 终端1
   screen -S detector1
   python bartscore_detector_improved.py --gpu 0
   # Ctrl+A+D 退出
   
   # 终端2
   screen -S detector2
   python bartscore_detector.py --gpu 1
   # Ctrl+A+D 退出
   
   # 查看所有会话
   screen -ls
   
   # 恢复会话
   screen -r detector1
   ```

---

## 📝 快速参考

| 场景 | 命令 |
|------|------|
| 使用GPU 0 | `python script.py --gpu 0` |
| 使用GPU 1 | `python script.py --gpu 1` |
| 自动选择 | `python script.py` |
| 环境变量方式 | `CUDA_VISIBLE_DEVICES=1 python script.py` |
| 查看GPU | `nvidia-smi` |
| 监控GPU | `watch -n 1 nvidia-smi` |
| 后台运行 | `nohup python script.py --gpu 1 > log.txt 2>&1 &` |

---

## 🔗 相关文件

- `bartscore_detector.py` - 原版检测器（支持 --gpu 参数）
- `bartscore_detector_improved.py` - 改进版检测器（支持 --gpu 参数）
- `IMPROVEMENTS.md` - 性能改进说明
- `GPU_USAGE.md` - 本文档

如有问题，请查看程序输出日志或使用 `--help` 查看帮助。

