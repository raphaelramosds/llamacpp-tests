# Guidelines 

## Install llama-cpp-python with args

    sudo apt update && sudo apt install -y cmake python3-dev python3-pip git build-essential

Using CUDA cores

    sudo apt update && sudo apt install -y nvidia-cuda-toolkit 
    CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 \
        pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

Using Basic Linear Algebra Subprograms

    sudo apt update && sudo apt install -y libopenblas-dev
    CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
        pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir

## GGUF models

- [The Bloke - Llama 2 7B Chat - GGUF](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF#provided-files)