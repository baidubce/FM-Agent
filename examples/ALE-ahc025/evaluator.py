import subprocess
from pathlib import Path
import time
import random
import tempfile
import re
import os
import numpy as np

import uuid
 
 
def run_algorithm(
    input_file: Path,
    code_file: Path,
    language: str,
    timeout: int,
):
    """
    """
    start_at_total = time.perf_counter()
    output_path: Path | None = None
    proc = None
    temp_executable_path: Path | None = None
 
    if not input_file.exists():
        return None, (time.perf_counter() - start_at_total), f"输入文件不存在: {input_file}"
    if not code_file.exists():
        return None, (time.perf_counter() - start_at_total), f"算法代码文件不存在: {code_file}"
 
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".txt", encoding="utf-8"
        ) as temp_output_file:
            output_path = Path(temp_output_file.name)
 
        run_command_parts: list[str] = []
 
        if language == "Python":
            run_command_parts = ["python", str(code_file)]
 
        elif language == "C++17":
            temp_executable_name = f"temp_exec_{uuid.uuid4().hex}"
            temp_executable_path = Path(tempfile.gettempdir()) / temp_executable_name
 
            compile_command_parts = [
                "g++",
                str(code_file),
                "-o",
                str(temp_executable_path),
                "-std=c++17",
                "-O2",
                "-lboost_system",
                "-lboost_filesystem",
                "-lboost_thread",
                "-lboost_program_options",
                "-lboost_date_time",
                "-lboost_regex",
                "-lgmp",
                "-lgmpxx",
            ]
 
            try:
                compile_proc = subprocess.run(
                    compile_command_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
 
                if compile_proc.returncode != 0:
                    error_message = (
                        f"C++ compilation Error (Exit Code: {compile_proc.returncode})。\n"
                        f"compile stdout:\n{compile_proc.stdout}\n"
                        f"compile stderr:\n{compile_proc.stderr}"
                    )
                    return None, (time.perf_counter() - start_at_total), error_message
 
            except FileNotFoundError:
                error_message = "编译失败：找不到 'g++' 命令，请检查 C++ 编译器是否已安装并配置在 PATH 中。"
                return None, (time.perf_counter() - start_at_total), error_message
 
            run_command_parts = [str(temp_executable_path)]
 
        else:
            error_message = f"不支持的语言: {language}"
            return None, (time.perf_counter() - start_at_total), error_message
 
        start_at_exec = time.perf_counter()
 
        with open(input_file, "r", encoding="utf-8") as stdin_handle:
            with open(output_path, "w", encoding="utf-8") as stdout_handle:
                proc = subprocess.Popen(
                    run_command_parts,
                    stdin=stdin_handle,
                    stdout=stdout_handle,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                )
                _, stderr_output = proc.communicate(timeout=2 * timeout)
 
        execution_time = time.perf_counter() - start_at_exec
 
        if proc.returncode != 0:
            error_message = (
                f"Algorithm Execution Error (Exit Code: {proc.returncode})。\n"
                f"standard error output:\n{stderr_output}"
            )
            if output_path.exists():
                output_path.unlink()
            return None, execution_time, error_message
 
        return output_path, execution_time, None
 
    except FileNotFoundError:
        error_message = (
            f"执行失败：找不到 '{run_command_parts[0]}' 命令，请检查 Python 是否已安装并配置在 PATH 中。"
        )
        if output_path and output_path.exists():
            output_path.unlink()
 
        return None, (time.perf_counter() - start_at_total), error_message
 
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        error_message = "Algorithm Execution Timeout."
        return None, (time.perf_counter() - start_at_total), error_message
 
    except Exception as e:
        error_message = f"Run error: {e}"
        if output_path and output_path.exists():
            output_path.unlink()
        return None, (time.perf_counter() - start_at_total), error_message
 
    finally:
        if temp_executable_path and temp_executable_path.exists():
            try:
                temp_executable_path.unlink()
            except OSError as e:
                print(f"警告: 无法删除临时可执行文件 {temp_executable_path}: {e}")
 
 
def run_algorithm_reactive(
    input_file: Path,
    code_file: Path,
    evaluator_file: Path,
    language: str,
    timeout: int,
):
    """
    """
 
    tester_path = str(evaluator_file)
 
    input_file_handle = None
    output_file_handle = None
    output_temp_file_path_str = None
 
    proc = None
 
    temp_executable_path: Path | None = None
 
    score = None
    run_time = None
    run_error = None
 
    try:
        output_temp_file_obj = tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".out", encoding="utf-8"
        )
        output_temp_file_path_str = output_temp_file_obj.name
        output_file_handle = output_temp_file_obj
 
        input_file_handle = open(input_file, "r", encoding="utf-8")
 
        executable_to_run: Path | str
 
        if "c++" in language.lower():
            temp_executable_name = f"temp_reactive_exec_{uuid.uuid4().hex}"
            temp_executable_path = Path(tempfile.gettempdir()) / temp_executable_name
 
            compile_command_parts = [
                "g++",
                str(code_file),
                "-o",
                str(temp_executable_path),
                "-std=c++17",
                "-O2",
                "-I",
                "/usr/include/eigen3",
                "-lboost_system",
                "-lboost_filesystem",
                "-lboost_thread",
                "-lboost_program_options",
                "-lboost_date_time",
                "-lboost_regex",
                "-lgmp",
                "-lgmpxx",
            ]
 
            compile_result = subprocess.run(
                compile_command_parts, check=False, capture_output=True, text=True
            )
 
            if compile_result.returncode != 0:
                run_error = (
                    f"C++ compilation failed with exit code {compile_result.returncode}.\n"
                    f"Stdout:\n{compile_result.stdout}\n"
                    f"Stderr:\n{compile_result.stderr}"
                )
                return score, run_time, run_error
 
            executable_to_run = str(temp_executable_path)
 
        elif language.lower() == "python":
            executable_to_run = str(code_file)
        else:
            raise ValueError(
                f"Unsupported language: {language}. Please use 'Python' or 'C++'."
            )
 
        command_args_for_tester = []
        if language.lower() == "python":
            command_args_for_tester.append("python")
            command_args_for_tester.append(executable_to_run)
        elif "c++" in language.lower():
            command_args_for_tester.append(executable_to_run)
 
        command_to_run = [tester_path] + command_args_for_tester
 
        start_at = time.perf_counter()
 
        proc = subprocess.Popen(
            command_to_run,
            stdin=input_file_handle,
            stdout=output_file_handle,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout_output, stderr_output = proc.communicate(timeout=2 * timeout)
 
        run_time = time.perf_counter() - start_at
 
        if proc.returncode != 0:
            run_error = (
                f"Command failed with exit code {proc.returncode}.\n"
                f"Stderr:\n{stderr_output}\n"
                f"Stdout:\n{stdout_output}"
            )
 
        pattern = r"Score\s*[:=]\s*(\d+(\.\d+)?)"
        match_stderr = None
        if stderr_output:
            match_stderr = re.search(pattern, stderr_output, re.IGNORECASE)
        match_stdout = None
        if stdout_output:
            match_stdout = re.search(pattern, stdout_output, re.IGNORECASE)
        match = match_stderr or match_stdout
        if not match and not run_error:
            run_error = f"No score found. Output was:\n{stderr_output}\n{stdout_output}"
        
        score = float(match.group(1)) if match else 0.0
 
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
        run_error = "Command timed out."
        run_time = time.perf_counter() - start_at
 
    except FileNotFoundError as e:
        run_error = (
            f"Executable not found. Check paths: "
            f"{tester_path} or 'python' interpreter/compiler. Error: {e}"
        )
        run_time = time.perf_counter() - start_at
    except ValueError as e:
        run_error = f"Configuration error: {e}"
        run_time = time.perf_counter() - start_at
    except Exception as e:
        run_error = f"An unexpected error occurred: {e}"
        run_time = time.perf_counter() - start_at
    finally:
        if input_file_handle:
            input_file_handle.close()
        if output_file_handle:
            output_file_handle.close()
            if output_temp_file_path_str and os.path.exists(output_temp_file_path_str):
                os.remove(output_temp_file_path_str)
 
        if temp_executable_path and os.path.exists(temp_executable_path):
            try:
                os.remove(temp_executable_path)
            except OSError as e:
                print(f"警告: 无法删除临时可执行文件 {temp_executable_path}: {e}")
 
    return score, run_time, run_error
 
 
def judge_score(input_file, output_file, evaluator_file):
    """
    """
    command = [str(evaluator_file), str(input_file), str(output_file)]
    proc = None
    try:
        start_at = time.perf_counter()
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        stdout_output, stderr_output = proc.communicate(timeout=100)
 
        execution_time = time.perf_counter() - start_at
 
        if proc.returncode != 0:
            error_message = (
                f"Evaluation error (exit code {proc.returncode}):\n{stderr_output}"
            )
            return None, 0.0, error_message
 
        pattern = r"Score\s*[:=]\s*(\d+(\.\d+)?)"
        match_stderr = None
        if stderr_output:
            match_stderr = re.search(pattern, stderr_output, re.IGNORECASE)
        match_stdout = None
        if stdout_output:
            match_stdout = re.search(pattern, stdout_output, re.IGNORECASE)
 
        match = match_stderr or match_stdout
        if not match:
            error_message = f"No score found. Output was:\n{stderr_output}\n{stdout_output}"
            return 0.0, execution_time, error_message
 
        score = float(match.group(1)) if match else 0.0
        return score, execution_time, None
 
    except subprocess.TimeoutExpired:
        if proc:
            proc.kill()
            proc.communicate()
        error_message = "Evaluation timed out."
        return None, 0.0, error_message
    except ValueError:
        error_message = f"Evaluation error: wrong format"
        return None, 0.0, error_message
    except Exception as e:
        error_message = f"Evaluation error: {e}"
        return None, 0.0, error_message
    finally:
        pass
 
 
def evaluate_one(
    code_file,
    input_file,
    evaluator_file,
    language="c++",
    timeout=10,
    score_type="maximize",
    problem_type="batch",
):
    """
    """
    final_result = {
        "validity": 0.0,
        "score": 0.0,
        "combined_score": 0.0,
        "run_time": 0.0,
        "evaluate_time": 0.0,
        "error_info": {},
    }
 
    user_output_file = None
    try:
        if problem_type == "batch":
            user_output_file, run_time, run_error = run_algorithm(
                input_file, code_file, language, timeout
            )
 
            if run_error:
                final_result["error_info"]["run_error"] = f"Solution runtime error: {run_error}"
                return final_result
 
            score, evaluate_time, judge_error = judge_score(
                input_file, user_output_file, evaluator_file
            )
 
            if judge_error:
                final_result["error_info"][
                    "evaluator_error"
                ] = f"Solution evaluation error: {judge_error}"
                if score is None:
                    final_result["validity"] = 0.0
                    return final_result
                final_result["score"] = 0.0

            final_result["score"] = score if score is not None else 0.0
            final_result["run_time"] = run_time
            final_result["evaluate_time"] = evaluate_time
            final_result["validity"] = 1.0
            return final_result
        else:
            score, run_time, run_error = run_algorithm_reactive(
                input_file, code_file, evaluator_file, language, timeout
            )
 
            if run_error:
                final_result["error_info"]["run_error"] = f"Solution runtime error: {run_error}"
                return final_result
 
            final_result["score"] = score
            final_result["run_time"] = run_time
            final_result["evaluate_time"] = 0
            final_result["validity"] = 1.0
            return final_result
 
    finally:
        if user_output_file and user_output_file.exists():
            user_output_file.unlink()
 
 
def evaluate(
    path_user_py,
    input_dir="./ahc025/in",
    evaluator_file="./ahc025/tester",
    timeout=2.0,
    score_type="minimize",
    problem_type="reactive"
):
    input_dir = Path(input_dir)
    code_file = Path(path_user_py)
    result = subprocess.run(['chmod', '+x', evaluator_file], check=True, capture_output=True, text=True)
    print(f"成功运行 'chmod +x {evaluator_file}'。")
 
    final_result = {
        "validity": 0.0,
        "combined_score": 0.0,
        "score_list": [],
        "score": 0.0,
        "total_run_time": 0.0,
        "avg_run_time": 0.0,
        "max_run_time": 0.0,
        "error_info": {},
    }
    file_num = 0
 
    input_files_to_process = [f for f in input_dir.iterdir() if f.is_file()]
 
    if not input_files_to_process:
        print(f"No files found in {input_dir}")
        return final_result
 
    for input_file in input_files_to_process:
        try:
            result = evaluate_one(
                code_file,
                input_file,
                evaluator_file,
                timeout,
                score_type,
                problem_type,
            )
        except Exception as exc:
            print(f"'{input_file.name}' generated an exception: {exc}")
            final_result['validity'] = 0 
            final_result['error_info'] = {"evaluation_exception": str(exc)}
            return final_result
        else:
            for k, v in result["error_info"].items():
                if v not in final_result["error_info"].get(k, ""):
                    final_result["error_info"][k] = final_result["error_info"].get(k, "") + "\n" + v
                
            final_result['validity'] += result["validity"]
            final_result["score"] += result["score"]
            final_result["total_run_time"] += result["run_time"]
            final_result["max_run_time"] = max(final_result["max_run_time"], result["run_time"])
            final_result['score_list'].append(result["score"])
                    
            file_num += 1
 
    if file_num > 0:
        final_result["avg_run_time"] = final_result["total_run_time"] / file_num
        final_result["score"] /= file_num
        final_result['validity'] /= file_num
    else:
        final_result["avg_run_time"] = 0.0
    if final_result["validity"] != 1 or final_result["score"] == 0:
        final_result["combined_score"] = 0
    else:
        if target is not None:
            if score_type == "maximize":
                final_result["combined_score"] = final_result["score"] / target
            else: 
                final_result["combined_score"] = 1 - (final_result["score"]-target) / final_result["score"]
        else:
            if score_type == "maximize":
                final_result["combined_score"] = final_result["score"]
            else:
                final_result["combined_score"] = float(10000000 /  np.log10(100 + final_result["score"]))
 
    return final_result
 
 
if __name__ == "__main__":
    result = evaluate(
        "best_program.py"
    )
    print(result)