# 1. Select the GPU you want to use (e.g., GPU 0)
export CUDA_VISIBLE_DEVICES=0

# 2. Set the GPU compute mode to EXCLUSIVE_PROCESS. 
# This ensures that only the MPS control daemon can create a direct context on the GPU.
# All other user programs will be forced to connect through the MPS server.
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS

# 3. Define where the MPS daemon will store its pipe and log files
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# 4. Start the MPS server in the background
nvidia-cuda-mps-control -d
