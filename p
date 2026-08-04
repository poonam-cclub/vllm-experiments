-- Build type: Release
-- Target device: cuda
-- Found python matching: /home/piyus/KV/.venv/bin/python.
CMake Error at /home/piyus/KV/.venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.4/Modules/CMakeDetermineCompilerId.cmake:966 (message):
  Compiling the CUDA compiler identification source file
  "CMakeCUDACompilerId.cu" failed.

  Compiler: /home/piyus/cuda-12.4/bin/nvcc

  Build flags:

  Id flags: --keep;--keep-dir;tmp -v



  The output was:

  2

  #$ _NVVM_BRANCH_=nvvm

  #$ _NVVM_BRANCH_SUFFIX_=

  #$ _SPACE_=

  #$ _CUDART_=cudart

  #$ _HERE_=/home/piyus/cuda-12.4/bin

  #$ _THERE_=/home/piyus/cuda-12.4/bin

  #$ _TARGET_SIZE_=

  #$ _TARGET_DIR_=

  #$ _TARGET_DIR_=targets/x86_64-linux

  #$ TOP=/home/piyus/cuda-12.4/bin/..

  #$ NVVMIR_LIBRARY_DIR=/home/piyus/cuda-12.4/bin/../nvvm/libdevice

  #$
  LD_LIBRARY_PATH=/home/piyus/cuda-12.4/bin/../lib:/home/piyus/cuda-12.4/lib64:

  #$
  PATH=/home/piyus/cuda-12.4/bin/../nvvm/bin:/home/piyus/cuda-12.4/bin:/home/piyus/KV/.venv/bin:/home/piyus/cuda-12.4/bin:/home/piyus/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin

  #$ INCLUDES="-I/home/piyus/cuda-12.4/bin/../targets/x86_64-linux/include"

  #$ LIBRARIES=
  "-L/home/piyus/cuda-12.4/bin/../targets/x86_64-linux/lib/stubs"
  "-L/home/piyus/cuda-12.4/bin/../targets/x86_64-linux/lib"

  #$ CUDAFE_FLAGS=

  #$ PTXAS_FLAGS=

  #$ rm tmp/a_dlink.reg.c

  #$ gcc -D__CUDA_ARCH_LIST__=520 -D__NV_LEGACY_LAUNCH -E -x c++ -D__CUDACC__
  -D__NVCC__ "-I/home/piyus/cuda-12.4/bin/../targets/x86_64-linux/include"
  -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=4
  -D__CUDACC_VER_BUILD__=99 -D__CUDA_API_VER_MAJOR__=12
  -D__CUDA_API_VER_MINOR__=4 -D__NVCC_DIAG_PRAGMA_SUPPORT__=1 -include
  "cuda_runtime.h" -m64 "CMakeCUDACompilerId.cu" -o
  "tmp/CMakeCUDACompilerId.cpp4.ii"

  #$ cudafe++ --c++17 --gnu_version=130100 --display_error_number
  --orig_src_file_name "CMakeCUDACompilerId.cu" --orig_src_path_name
  "/home/piyus/KV/vllm/cmake-build-release/CMakeFiles/4.4.0/CompilerIdCUDA/CMakeCUDACompilerId.cu"
  --allow_managed --m64 --parse_templates --gen_c_file_name
  "tmp/CMakeCUDACompilerId.cudafe1.cpp" --stub_file_name
  "CMakeCUDACompilerId.cudafe1.stub.c" --gen_module_id_file
  --module_id_file_name "tmp/CMakeCUDACompilerId.module_id"
  "tmp/CMakeCUDACompilerId.cpp4.ii"

  /usr/include/x86_64-linux-gnu/bits/floatn-common.h(214): error: invalid
  combination of type specifiers

    typedef float _Float32;
                  ^



  /usr/include/x86_64-linux-gnu/bits/floatn-common.h(251): error: invalid
  combination of type specifiers

    typedef double _Float64;
                   ^



  /usr/include/x86_64-linux-gnu/bits/floatn-common.h(268): error: invalid
  combination of type specifiers

    typedef double _Float32x;
                   ^



  /usr/include/x86_64-linux-gnu/bits/floatn-common.h(285): error: invalid
  combination of type specifiers

    typedef long double _Float64x;
                        ^



  4 errors detected in the compilation of "CMakeCUDACompilerId.cu".

  # --error 0x2 --





Call Stack (most recent call first):
  /home/piyus/KV/.venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.4/Modules/CMakeDetermineCompilerId.cmake:8 (CMAKE_DETERMINE_COMPILER_ID_BUILD)
  /home/piyus/KV/.venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.4/Modules/CMakeDetermineCompilerId.cmake:53 (__determine_compiler_id_test)
  /home/piyus/KV/.venv/lib/python3.12/site-packages/cmake/data/share/cmake-4.4/Modules/CMakeDetermineCUDACompiler.cmake:163 (CMAKE_DETERMINE_COMPILER_ID)
  /home/piyus/KV/.venv/lib/python3.12/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake:56 (enable_language)
  /home/piyus/KV/.venv/lib/python3.12/site-packages/torch/share/cmake/Caffe2/Caffe2Config.cmake:86 (include)
  /home/piyus/KV/.venv/lib/python3.12/site-packages/torch/share/cmake/Torch/TorchConfig.cmake:62 (find_package)
  CMakeLists.txt:112 (find_package)


-- Configuring incomplete, errors occurred!
