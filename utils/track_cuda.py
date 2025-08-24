import torch
import functools
import inspect

def track_cuda_memory(tag=None):
    """
    装饰器：追踪 GPU 显存变化，显示调用函数名、前后使用的显存
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated() / (1024 ** 2)
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            after = torch.cuda.memory_allocated() / (1024 ** 2)
            delta = after - before

            print(f"[显存追踪] {tag or func.__name__:<40} | +{delta:.2f} MB | now: {after:.2f} MB")
            return result
        return wrapper
    return decorator


# extract_memory_trace.py

def extract_memory_trace(log_path, output_path=None):
    """
    从日志文件中提取以 "[显存追踪] training step" 开头的行。

    Args:
        log_path (str): 输入日志文件路径。
        output_path (str, optional): 如果指定，则将结果写入此文件。
    """
    prefix = "[显存追踪] parallel_beam_search_decode"
    extracted_lines = []

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith(prefix):
                extracted_lines.append(line)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            f_out.writelines(extracted_lines)
        print(f"提取的日志已保存到 {output_path}")
    else:
        for line in extracted_lines:
            print(line.strip())

# 示例用法
if __name__ == "__main__":
    log_file = "logs/train_2docoder.log"          # 替换成你的日志文件路径
    output_file = "logs/train_2docoder_parell.txt"        # 如果你想把结果保存到新文件
    extract_memory_trace(log_file, output_file)
