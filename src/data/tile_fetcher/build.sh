#!/usr/bin/env bash
set -e

CUDA=/usr/local/cuda-12.4

$CUDA/bin/nvcc -std=c++17 \
    -I$CUDA/include \
    test1_hw_backend.cpp \
    -L$CUDA/lib64 -lnvjpeg -lcudart \
    -o test1_hw_backend

$CUDA/bin/nvcc -std=c++17 \
    -I$CUDA/include \
    test2_probe_default.cpp \
    -L$CUDA/lib64 -lnvjpeg -lcudart \
    -o test2_probe_default