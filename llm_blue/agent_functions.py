# agent_functions.py

import os
import time
import asyncio
import subprocess
import tempfile
import psutil
from typing import List, Tuple, Dict
import re
import ast
from langchain_core.output_parsers import StrOutputParser

async def get_optimization_techniques(llm, user_code: str, number_times: int) -> List[str]:
    """
    Generate a list of `number_times` code optimization techniques for `user_code`
    using the provided llm.
    """
    try:
        prompt = f"""Generate a list of {number_times} code optimization techniques for the following code:

{user_code}

Each technique should be specific and actionable. 
Return only the list of techniques, one per line, with no additional text."""
        response = await llm.ainvoke(prompt)
        if not response or not response.content:
            return ["Basic optimization"]  # fallback
        techniques = response.content.strip().split('\n')
        techniques = [technique.strip('- ').strip().lstrip('0123456789. ') 
                      for technique in techniques if technique.strip()]
        return techniques[:number_times]
    except Exception as e:
        print(f"Error generating optimization techniques: {str(e)}")
        return ["Basic optimization"]

def get_subsets(techniques: List[str]) -> List[List[str]]:
    """
    Returns all subsets (combinations) of the given list of techniques.
    """
    subsets = []
    def subset_helper(i: int, arr: List[str]):
        if i == len(techniques):
            subsets.append(arr)
            return
        subset_helper(i + 1, arr)
        subset_helper(i + 1, arr + [techniques[i]])
    subset_helper(0, [])
    return subsets

async def improve_code(llm, code: str, technique_subset: List[str]) -> str:
    """
    Use the provided llm to improve the code according to the given subset of optimization techniques.
    """
    techniques = ', '.join(technique_subset)
    prompt = f"""
Please improve the following code using the following optimization technique(s): {techniques}.
Make sure the code is complete and can run independently if put into a temporary file and executed using subprocess.
Include all necessary variable definitions.

{code}

ONLY RETURN CODE, NO OTHER TEXT.
ONLY RETURN THE RAW PYTHON CODE WITHOUT ANY MARKDOWN, AND NOTHING ELSE.
Ensure all variables are properly defined before use.
Keep any print statements or output generation from the original code.
"""
    response = await llm.ainvoke(prompt)
    parser = StrOutputParser()
    improved_code = parser.invoke(response.content)
    return improved_code

def run_code(code: str) -> Tuple[str, float, float]:
    """
    Run the provided code by writing it to a temporary file and executing it.
    Returns a tuple of (output, execution_time in seconds, peak_memory in MB).
    """
    process = None
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        start_time = time.time()
        peak_memory = 0.0

        process = subprocess.Popen(
            ['python', temp_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        try:
            ps_process = psutil.Process(process.pid)
            while process.poll() is None:
                try:
                    mem = ps_process.memory_info().rss / (1024 * 1024)  # bytes to MB
                    peak_memory = max(peak_memory, mem)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                time.sleep(0.1)
            stdout, stderr = process.communicate()
            execution_time = time.time() - start_time
            if process.returncode == 0:
                output = stdout
            else:
                output = f"Error:\n{stderr}"
        except psutil.NoSuchProcess:
            output = "Process terminated unexpectedly"
            execution_time = time.time() - start_time

    except Exception as e:
        output = f"Exception occurred while running the code: {str(e)}"
        execution_time = 0.0
        peak_memory = 0.0

    finally:
        if process and process.poll() is None:
            try:
                process.kill()
            except Exception:
                pass
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    return output, execution_time, peak_memory

async def process_improvement(llm, original_code: str, technique_subset: List[str]) -> Tuple[str, str, float, float, str]:
    """
    Process a single code improvement and its execution.
    Returns a tuple: (improved_code, output, execution_time, peak_memory, technique_description)
    """
    technique_desc = " + ".join(technique_subset) if technique_subset else "Original (No optimization)"
    if not technique_subset:
        code_output, execution_time, peak_memory = run_code(original_code)
        return original_code, code_output, execution_time, peak_memory, technique_desc
    improved_code = await improve_code(llm, original_code, technique_subset)
    code_output, execution_time, peak_memory = run_code(improved_code)
    return improved_code, code_output, execution_time, peak_memory, technique_desc

async def parallel_optimize(llm, user_code: str, optimization_techniques: List[str]) -> List[Dict]:
    """
    Run all possible combinations of optimizations concurrently using asyncio.
    """
    technique_subsets = get_subsets(optimization_techniques)
    tasks = [
        process_improvement(llm, user_code, subset)
        for subset in technique_subsets
    ]
    results_data = await asyncio.gather(*tasks)
    results = []
    for improved_code, output, execution_time, peak_memory, technique_desc in results_data:
        results.append({
            'code': improved_code,
            'output': output,
            'execution_time': execution_time,
            'memory_usage': peak_memory,
            'techniques': technique_desc
        })
    results.sort(key=lambda x: len(x['techniques'].split(" + ")))
    return results

def prepare_performance_data(execution_times: List[float], memory_usages: List[float], techniques_applied: List[str]) -> Dict:
    """
    Prepare performance data in a structured dictionary.
    """
    return {
        'Technique Applied': techniques_applied,
        'Execution Time (seconds)': execution_times,
        'Memory Usage (MB)': memory_usages
    }
