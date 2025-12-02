#include <cuda_runtime.h>
#include <iostream>

int main() {
    int v = -1;

    // 161 = cudaDevAttrHardwareJpegSupport
    auto st = cudaDeviceGetAttribute(&v, (cudaDeviceAttr)161, 0);
    std::cout << "cudaDeviceGetAttribute status=" << st
              << " hardwareJpegSupport=" << v << "\n";

    return 0;
}