#include <cuda_runtime.h>
#include <iostream>

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    std::cout << "CUDA device count: " << count << "\n";

    for (int i = 0; i < count; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        std::cout << "Device " << i << ": " << p.name
                  << "  compute capability " << p.major << "." << p.minor << "\n";
    }
}