import os
import time
import torch

from vllm import LLM, SamplingParams

# 1. Programmatically configure the profiling engine parameters
llm = LLM(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    enforce_eager=True,       # Crucial: Disables CUDA graphs to reveal internal routines
    #kv_cache_dtype="turboquant_4bit_nc",     # Forces your FP8 quantization paths
    kv_cache_dtype="fp8",     # Forces your FP8 quantization paths
    gpu_memory_utilization=0.85,
    profiler_config={
        "profiler": "torch",  # Uses PyTorch Kineto tracer
        "torch_profiler_dir": "./vllm_traces", # Destination directory
        "torch_profiler_with_stack": True,     # Collects Python source lines
        "torch_profiler_record_shapes": True,  # Collects Tensor dimensions
    }
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=5)
prompt = ["Trace this execution path."]

# 2. Warmup Run 
# Clears out initial memory allocations so they don't corrupt the timeline
print("Running engine warmup...")
llm.generate(prompt, sampling_params)
time.sleep(1) # Allow asynchronous GPU queues to settle

# 3. Programmatically catch the target execution trace
print("Starting programmatic trace collection...")
llm.start_profile()  # Enables the underlying hooks

torch.cuda.profiler.start()

# This call will capture exactly 1 Prefill step and 5 Decode steps
outputs = llm.generate(prompt, sampling_params)
print("outputs::", outputs)

torch.cuda.profiler.stop()

llm.stop_profile()   # Flushes and writes the trace out to the file system
print("Trace captured successfully!")
