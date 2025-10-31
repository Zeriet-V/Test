#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
检索增强 NLI 检测 + 自动上传到 GitHub
运行方法: python3 run_retrieval_and_upload.py
"""

import subprocess
import os
import time
from datetime import datetime

def run_command(cmd, description):
    """运行命令并显示输出"""
    print(f"\n{'='*80}")
    print(f"【{description}】")
    print('='*80)
    print(f"命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n✗ 命令执行失败！退出码: {result.returncode}")
        return False
    
    print(f"\n✓ {description} 完成！")
    return True

def main():
    start_time = time.time()
    
    print("=" * 80)
    print("检索增强 NLI 检测 + 自动 Git 上传")
    print("=" * 80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 第一步：运行检索增强 NLI 检测
    print("【步骤 1/3】运行检索增强 NLI 检测...")
    print("预计耗时: 3-4 小时")
    print("可以去喝杯咖啡了 ☕\n")
    
    os.chdir('/home/xgq/Test/detectors/nli_methods')
    
    detect_cmd = """python3 nli_retrieval_detector.py \
  --gpu 1 \
  --input /home/xgq/Test/data/validation_set.jsonl \
  --output nli_retrieval_validation_results.jsonl \
  --use-entailment \
  --sentence-level \
  --threshold-entail 0.45 \
  --threshold-contra 0.2 \
  --top-k 3"""
    
    if not run_command(detect_cmd, "检索增强 NLI 检测"):
        return
    
    detect_end = time.time()
    detect_time = int((detect_end - start_time) / 60)
    print(f"\n检测耗时: {detect_time} 分钟 ({detect_time/60:.1f} 小时)")
    
    # 第二步：检查结果文件
    print("\n【步骤 2/3】检查生成的文件...")
    
    result_files = [
        'nli_retrieval_validation_results.jsonl',
        'nli_retrieval_validation_results_report.txt'
    ]
    
    for file in result_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            if file.endswith('.jsonl'):
                with open(file, 'r') as f:
                    lines = len(f.readlines())
                print(f"  ✓ {file} ({lines} 行, {size/1024:.1f} KB)")
            else:
                print(f"  ✓ {file} ({size/1024:.1f} KB)")
        else:
            print(f"  ✗ {file} 未找到！")
            return
    
    # 显示报告摘要
    print("\n--- 报告性能摘要 ---")
    with open('nli_retrieval_validation_results_report.txt', 'r') as f:
        for line in f:
            if 'Precision:' in line or 'Recall:' in line or 'F1:' in line:
                print(f"  {line.strip()}")
    
    # 第三步：上传到 GitHub
    print("\n【步骤 3/3】上传结果到 GitHub...")
    
    os.chdir('/home/xgq/Test')
    
    # Git 状态
    print("\n当前 git 状态:")
    subprocess.run("git status --short", shell=True)
    
    # 添加文件
    print("\n添加文件到 git...")
    
    files_to_add = [
        'detectors/nli_methods/nli_retrieval_validation_results.jsonl',
        'detectors/nli_methods/nli_retrieval_validation_results_report.txt',
        'detectors/nli_methods/nli_retrieval_detector.py',
        'detectors/nli_methods/检索增强NLI使用说明.md',
        'docs/BARTScore_NLI_综合报告.md'
    ]
    
    for file in files_to_add:
        if os.path.exists(file):
            subprocess.run(f"git add {file}", shell=True)
            print(f"  ✓ 已添加: {file}")
    
    # 提交
    commit_msg = f"添加检索增强NLI验证集结果 - Top-K=3 - {datetime.now().strftime('%Y-%m-%d')}"
    print(f"\n提交信息: {commit_msg}")
    
    result = subprocess.run(
        ['git', 'commit', '-m', commit_msg],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ Git 提交成功")
    else:
        if "nothing to commit" in result.stdout or "nothing to commit" in result.stderr:
            print("ℹ 没有新的改动需要提交")
        else:
            print(f"⚠ 提交可能失败: {result.stderr}")
    
    # 推送
    print("\n推送到 GitHub...")
    
    push_result = subprocess.run(
        "git push origin main 2>&1 || git push origin master 2>&1",
        shell=True,
        capture_output=True,
        text=True
    )
    
    if push_result.returncode == 0 or "up-to-date" in push_result.stdout.lower():
        print("✓ 推送成功！")
    else:
        print(f"⚠ 推送可能失败: {push_result.stderr}")
        print("请手动检查并推送")
    
    # 总结
    end_time = time.time()
    total_time = int((end_time - start_time) / 60)
    
    print("\n" + "=" * 80)
    print("✓ 全部完成！")
    print("=" * 80)
    print(f"总耗时: {total_time} 分钟 ({total_time/60:.1f} 小时)")
    print(f"检测耗时: {detect_time} 分钟 ({detect_time/60:.1f} 小时)")
    print(f"上传耗时: {total_time - detect_time} 分钟")
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n已上传文件:")
    for file in files_to_add:
        if os.path.exists(f'/home/xgq/Test/{file}'):
            print(f"  - {file}")
    print("\n现在可以在本地 git pull 查看结果！")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n用户中断执行")
    except Exception as e:
        print(f"\n\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()

