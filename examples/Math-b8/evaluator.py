"""
B8 evaluator
"""
import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import os
import subprocess
import tempfile
import sys
import pickle
from scipy.spatial.distance import pdist


class TimeoutError(Exception):
    """
    TimeoutError
    """
    pass


def run_with_timeout(program_path, timeout_seconds=100):
    """
    Run the program in a separate process with timeout
    using a simple subprocess approach

    Args:
        program_path: Path to the program file
        timeout_seconds: Maximum execution time in seconds

    Returns:
        Dictionary with keys:
        - 'success': bool, whether execution was successful
        - 'result': the result if successful, None otherwise
        - 'error_info': error message if failed, empty string otherwise
    """
    # Create a temporary file to execute
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp_file:
        # Write a script that executes the program and saves results
        script = f"""
import sys
import numpy as np
import os
import pickle
import traceback

# Add the directory to sys.path
sys.path.insert(0, os.path.dirname('{program_path}'))

try:
    # Import the program
    spec = __import__('importlib.util').util.spec_from_file_location("program", '{program_path}')
    program = __import__('importlib.util').util.module_from_spec(spec)
    spec.loader.exec_module(program)
    
    # Run the construction function
    result = program.run_construction()

    # Save results to a file
    results = {{
        'result': result
    }}

    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump(results, f)
    
except Exception as e:
    # If an error occurs, save the error instead
    print(f"Error in subprocess: {{str(e)}}")
    traceback.print_exc()
    with open('{temp_file.name}.results', 'wb') as f:
        pickle.dump({{'error_info': str(e)}}, f)
"""
        temp_file.write(script.encode())
        temp_file_path = temp_file.name

    results_path = f"{temp_file_path}.results"

    try:
        # Run the script with timeout
        process = subprocess.Popen(
            [sys.executable, temp_file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode

            if exit_code != 0:
                return {
                    'success': False,
                    'result': None,
                    'error_info': f"Process exited with code {exit_code}"
                }

            # Load the results
            if os.path.exists(results_path):
                with open(results_path, "rb") as f:
                    results = pickle.load(f)

                # Check if an error was returned
                if "error_info" in results:
                    return {
                        'success': False,
                        'result': None,
                        'error_info': f"Program execution failed: {results['error_info']}"
                    }

                return {
                    'success': True,
                    'result': results["result"],
                    'error_info': ""
                }
            else:
                return {
                    'success': False,
                    'result': None,
                    'error_info': "Results file not found"
                }

        except subprocess.TimeoutExpired:
            # Kill the process if it times out
            process.kill()
            process.wait()
            return {
                'success': False,
                'result': None,
                'error_info': f"Process timed out after {timeout_seconds} seconds"
            }

    except Exception as e:
        return {
            'success': False,
            'result': None,
            'error_info': f"Unexpected error: {str(e)}"
        }
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if os.path.exists(results_path):
            os.unlink(results_path)


def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0


def verify_points(points):
    """Verify that the points are valid and calculate ratio_squared"""
    try:
        if points is None:
            return None, None, "Invalid points format, the points are None"
        
        # MODIFICATION: Convert list-like input to a NumPy array
        # This will handle lists of lists, etc., and is efficient if already an array
        points = np.asarray(points)

        # Now we can safely use .shape
        if  len(points) != 16 or points.shape[1] != 2:
            return None, None, f"Invalid points format, the shape of points is not (16, 2), got {points.shape}"
        
        # Calculate pairwise distances
        distances = pdist(points)
        
        if len(distances) == 0:
            return None, None, "No pairwise distances calculated"
        
        min_distance = np.min(distances)
        max_distance = np.max(distances)
        
        if min_distance <= 0:
            return None, None, "Minimum distance is zero or negative"
        
        ratio = max_distance / min_distance
        ratio_squared = ratio ** 2
        
        return ratio, ratio_squared, None
        
    except Exception as e:
        # This will catch errors from np.asarray (e.g., ragged list), .shape, or pdist
        return None, None, f"Error in verification: {str(e)}"


def evaluate(path_user_py):
    """
    Evaluate the 2D point configuration optimization program by running it once
    and checking how close it gets to the target ratio.

    Args:
        path_user_py: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Target ratio squared from AlphaEvolve
    TARGET_RATIO_SQUARED = 12.889266112
    
    try:
        start_time = time.time()

        # Run with timeout
        result = run_with_timeout(path_user_py, timeout_seconds=1000)

        # Handle different result formats
        if result['success']:
            # MODIFICATION: Removed the 'isinstance' check.
            # We now pass the result (which could be list or ndarray)
            # directly to verify_points, which will handle conversion.
            points = result['result']
            
            # Verify the points and calculate ratio_squared
            ratio, ratio_squared, error = verify_points(points)
            
            if error is None:
                print(f"Got points and calculated ratio_squared: {ratio_squared:.6f}")
                
                end_time = time.time()
                execution_time = end_time - start_time

                # Calculate combined score: ratio_squared / target_ratio_squared
                # Lower is better, so we invert the ratio
                combined_score = float(TARGET_RATIO_SQUARED / ratio_squared) if ratio_squared > 0 else 0.0
                
                # Cap the score at 1.0 (when we achieve the target or better)
                combined_score = min(combined_score, 1.0)

                return {
                    "valid": 1.0,
                    "validity": 1.0,
                    "combined_score": combined_score,
                    "ratio_squared": ratio_squared,
                    "ratio": ratio,
                    "points": "to do, save the valid points",
                    "execution_time": execution_time,
                    "error_info": "",
                }
            else:
                # This block now handles failures from verify_points
                print(f"Point verification failed: {error}, the input was: {points}")
                return {
                    "valid": 0.0,
                    "validity": 0.0,
                    "combined_score": 0.0,
                    "points": None,
                    "error_info": f"Point verification failed: {error}",
                }
        else:
            print(f"Program execution failed or timed out: {result['error_info']}")
            return {
                "valid": 0.0,
                "validity": 0.0,
                "combined_score": 0.0,
                "points": None,
                "error_info": f"Program execution failed or timed out: {result['error_info']}",
            }
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "valid": 0.0,
            "validity": 0.0,
            "combined_score": 0.0,
            "points": None,
            "error_info": f"Evaluation failed completely: {str(e)}",
        }


if __name__ == "__main__":
    print(evaluate("best_program.py"))