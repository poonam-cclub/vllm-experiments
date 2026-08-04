# Tell the control daemon to quit
echo quit | nvidia-cuda-mps-control

# Restore the GPU compute mode to DEFAULT (allowing direct contexts again)
sudo nvidia-smi -i 0 -c DEFAULT
