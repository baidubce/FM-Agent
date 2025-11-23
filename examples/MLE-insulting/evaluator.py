#!/usr/bin/env python3
"""
evaluator_mlebench.py ─ 支持自动运行 user_code.py 生成 submission.csv 并评测
────────────────────────────────────────────
"""
import re
import os
import time
import subprocess
import json
import traceback
import numpy as np
import shutil
import glob
import ast
from pathlib import Path
from datetime import datetime
from typing import List


# 存储得到奖牌的 submission_csv
def move_and_rename_file(source_path, target_path, backup=True):
    """
    移动文件并重命名
    
    Args:
        source_path: 源文件路径
        target_path: 目标文件路径（包含新文件名）
    
    Returns:
        bool: 操作是否成功
    """
    """
    安全的文件移动和重命名，包含备份功能
    
    Args:
        source_path: 源文件路径
        target_path: 目标文件路径
        backup: 是否创建备份
    
    Returns:
        bool: 操作是否成功
    """
    try:
        source = Path(source_path)
        target = Path(target_path)
        
        # 验证源文件
        if not source.exists():
            print(f"错误: 源文件不存在 {source_path}")
            return False
        
        if not source.is_file():
            print(f"错误: 源路径不是文件 {source_path}")
            return False
        
        # 如果目标文件已存在，处理冲突
        if target.exists():
            if backup:
                # 创建备份
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = target.parent / f"{target.stem}_backup_{timestamp}{target.suffix}"
                shutil.move(str(target), str(backup_path))
                print(f"📦 已创建备份: {backup_path}")
            else:
                # 直接覆盖
                target.unlink()
        
        # 确保目标目录存在
        target.parent.mkdir(parents=True, exist_ok=True)
        
        # 执行移动操作
        shutil.move(str(source), str(target))
        
        # 验证操作成功
        if target.exists() and not source.exists():
            print(f"✅ 成功: {source_path} -> {target_path}")
            return True
        else:
            print("❌ 操作可能未完全成功")
            return False
            
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        return False
    

# 从文件名中提取submission的idx编号
def extract_submission_idx(filename) -> int:
    """
    从文件名中提取submission的idx编号
    
    Args:
        filename (str): 文件名，可以是完整路径或单纯文件名
        
    Returns:
        int: 提取到的idx编号，如果没有找到则返回-1
    """
    # 获取纯文件名（去掉路径）
    basename = os.path.basename(filename)
    
    # 定义正则表达式模式，匹配 submission_{数字}.csv 或 submission.csv
    pattern = r'^submission(?:_(\d+))?\.csv$'
    
    # 进行匹配
    match = re.match(pattern, basename)
    
    if match:
        idx_str = match.group(1)
        if idx_str is not None:
            try:
                return int(idx_str)
            except ValueError:
                return -1
        else:
            return -1  # submission.csv 的情况
    else:
        return -1  # 不匹配任何模式


def robust_loads(json_str):
    """
    尝试用 json.loads 解析，如果失败则用 ast.literal_eval 兜底
    """
    try:
        return json.loads(json_str)
    except Exception:
        try:
            return ast.literal_eval(json_str)
        except Exception:
            raise  # 最后还是抛出原始异常


def run_python_script(
    py_path: str, competition_name: str, forced_submission_dir: str, gpu_rank: int = 0, timeout=3*60*60
):
    """
    运行用户提交的 python 脚本，确保在其所在目录下执行，并监控GPU峰值显存占用。
    返回 submission.csv 路径（若生成成功），否则报错
    """
    work_dir = os.path.dirname(py_path)
    py_basename = os.path.basename(py_path)

    t0 = time.time()
    try:
        proc = subprocess.Popen(
            f"python3 {py_path}",
            # cwd=work_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
            shell=True,
        )
        out, err = proc.communicate(timeout=timeout)
        t1 = time.time()
        print(f"proc err:{err[-2000:]}")

        # 检查是否生成 submission.csv
        submission_csv_path = os.path.join(forced_submission_dir, competition_name)
        os.makedirs(submission_csv_path, exist_ok=True)
        sub_csv = os.path.join(submission_csv_path, "submission.csv")

        if not os.path.exists(sub_csv):
            return {
                "success": False,
                "error_info": {"run_stderr": err[-1000:] + "\nsubmission_csv not found!"},
                "wall_time": t1 - t0,
                "submission_csv": None,
            }
        return {
            "success": True,
            "error_info": {},
            "wall_time": t1 - t0,
            "submission_csv": sub_csv,
        }

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return {
            "success": False,
            "error_info": {"timeout": "python script run timeout"},
            "wall_time": timeout,
            "submission_csv": None,
        }
    except Exception as ex:
        return {
            "success": False,
            "error_info": {
                "exception": str(ex),
                "exception_type": type(ex).__name__,
                "traceback": traceback.format_exc(),
            },
            "wall_time": 0.0,
            "submission_csv": None,
        }


def run_grader(sub_csv, competition, data_dir, timeout=3600*4):
    """
    调用 mlebench grade-sample 命令，捕获并解析输出，返回评测 json dict。
    自动处理 grader 无输出、输出非 json、json 解析失败等情况。
    兼容非标准 JSON（如单引号、None 等）。
    """
    cmd = [
        "mlebench",
        "grade-sample",
        sub_csv,
        competition ,
        "--data-dir", data_dir
    ]
    try:
        t0 = time.time()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        out, err = proc.communicate(timeout=timeout)
        t1 = time.time()

        # 尝试从stdout和stderr中提取json
        def extract_json(txt):
            m = re.search(r'(\{[\s\S]*?\})', txt)
            return m.group(1) if m else None

        print(f"err:{err[-2000:]}")
        json_str = extract_json(out) or extract_json(err)
        json_from = 'stdout' if extract_json(out) else ('stderr' if extract_json(err) else None)

        if not json_str:
            return {
                "validity": 0.0,
                "score": "null",
                "medal": "none",
                "above_median": False,
                "error_info": {
                    "grader_stdout": out[-1000:],
                    "grader_stderr": err[-1000:],
                    "reason": "No JSON output found in grader stdout or stderr"
                },
                "meta_data": {},
                "eval_wall_time": t1 - t0,
            }

        # 尝试解析json，优先标准json，失败后用ast.literal_eval兜底
        try:
            res = robust_loads(json_str)
        except Exception as ex:
            return {
                "validity": 0.0,
                "score": 0.0,
                "medal": "none",
                "above_median": False,
                "error_info": {
                    "grader_stdout": out[-1000:],
                    "grader_stderr": err[-1000:],
                    "json_str": json_str,
                    "exception": str(ex),
                    "traceback": traceback.format_exc(),
                    "reason": f"JSON decode error from {json_from}"
                },
                "meta_data": {},
                "eval_wall_time": t1 - t0,
                "competition_report": {}, 
            }

        print(f"res:\n{res}")
        # 校验得分是否正常
        try:
            score = float(res.get("score", 0.0))
        except Exception as ex:
            return {
                "validity": 0.0,
                "score": "null",
                "medal": "none",
                "above_median": False,
                "error_info": {
                    "grader_stdout": out[-1000:],
                    "grader_stderr": err[-1000:],
                    "json_str": json_str,
                    "exception": str(ex),
                    "traceback": traceback.format_exc(),
                    "reason": f"JSON decode error from {json_from}"
                },
                "meta_data": {},
                "eval_wall_time": t1 - t0,
                "competition_report": res, 
            }

        # 成功提取与解析
        return {
            "validity": float(res.get("valid_submission", False)),
            "score": float(res.get("score", 0.0)),
            "medal": (
                "gold" if res.get("gold_medal") else
                "silver" if res.get("silver_medal") else
                "bronze" if res.get("bronze_medal") else
                "none"
            ),
            "above_median": bool(res.get("above_median", False)),
            "error_info": {},
            "meta_data": res,
            "eval_wall_time": t1 - t0,
            "competition_report": res, 
        }

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return {
            "validity": 0.0,
            "score": 0.0,
            "medal": "none",
            "above_median": False,
            "error_info": {"timeout": "grading process timeout"},
            "meta_data": {},
            "eval_wall_time": timeout,
            "competition_report": {}, 
        }
    except Exception as ex:
        return {
            "validity": 0.0,
            "score": 0.0,
            "medal": "none",
            "above_median": False,
            "error_info": {
                "exception": str(ex),
                "traceback": traceback.format_exc(),
            },
            "meta_data": {},
            "eval_wall_time": 0.0,
            "competition_report": {}, 
        }
    

def evaluate(
    path_user_py, 
    competition_name="detecting-insults-in-social-commentary", 
    forced_submission_dir="/tmp/mle_save_dir/", 
    gpu_rank=0, 
    data_dir="./", 
    timeout=2*60*60, 
):
    # 检查当前文件夹下是否已经有submission_csv，存在就删除，避免使用之前的结果
    submission_csv_path = os.path.join(forced_submission_dir, competition_name)
    os.makedirs(submission_csv_path, exist_ok=True)
    # 匹配所有以 submission 开头，以 .csv 结尾的文件
    sub_csv_paths: List[str] = glob.glob(os.path.join(submission_csv_path, "submission*.csv"))
    for sub_csv_path in sub_csv_paths:
        if sub_csv_path and os.path.exists(sub_csv_path):
            os.remove(sub_csv_path)

    lower_is_better = False
    step1 = run_python_script(path_user_py, competition_name, forced_submission_dir, gpu_rank, timeout=timeout)
    print(f"step1:{step1}")
    if not step1["success"]:        
        return {
            "validity": 0.0,
            "score": 0,
            "combined_score": 0.0,  # 新增
            "medal": "none",
            "above_median": False,
            "error_info": {"run_error": step1["error_info"]},
            "meta_data": {},
            "eval_wall_time": step1["wall_time"],
        }
    print("------------------ step1 run successfully ---------------------")

    try:
        sub_csv_path = step1["submission_csv"]
        # 评估预测文件
        grader_res = run_grader(sub_csv_path, competition_name, data_dir, timeout=timeout)
        grader_res["eval_wall_time"] += step1["wall_time"]
        
        def scale_score_with_medal(raw_score, medal:str = "none"):
            # 调整方向：越好越大
            score = -raw_score if lower_is_better else raw_score  
            # 用 sigmoid 压缩到 (0,1)
            combined_score = float(1 / (1 + np.exp(-score)))
            if medal == "none":
                return combined_score
            elif medal == "bronze":
                return combined_score + 1.0
            elif medal == "silver":
                return combined_score + 2.0
            elif medal == "gold":
                return combined_score + 3.0
            return 
            
        # 先判断有效性，无效直接得分为0
        if grader_res["validity"] <= 0:
            grader_res["combined_score"] = 0.0
        else:
            grader_res["combined_score"] = scale_score_with_medal(
                grader_res["score"], grader_res.get("medal", "none"))

    finally:
        # 安全删除所有的 submission.csv
        try:
            sub_csv_folder = os.path.join("/".join(sub_csv_path.split("/")[:-1]), "submission_save")
            if not os.path.exists(sub_csv_folder):
                os.makedirs(sub_csv_folder)
            
            program_save_folder = os.path.join("/".join(sub_csv_path.split("/")[:-1]), "program_save")
            if not os.path.exists(program_save_folder):
                os.makedirs(program_save_folder)
            
            # 其余文件直接删除
            for sub_csv_path in sub_csv_paths:
                if sub_csv_path and os.path.exists(sub_csv_path):
                    os.remove(sub_csv_path)

        except Exception as e:
            # 删除失败也不影响评测流程，只打印警告
            print(f"Warning: Failed to remove {sub_csv_path}: {str(e)}")
        
    return grader_res


if __name__ == "__main__":
    result = evaluate("init.py",)
