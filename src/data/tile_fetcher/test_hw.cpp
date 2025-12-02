#include <nvjpeg.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaSetDevice(0);

    nvjpegHandle_t h = nullptr;
    nvjpegStatus_t st = nvjpegCreateEx(
        NVJPEG_BACKEND_HARDWARE,
        nullptr,
        nullptr,
        0,
        &h
    );

    if (st == NVJPEG_STATUS_SUCCESS) {
        std::cout << "hardware backend available\n";
        nvjpegDestroy(h);
        return 0;
    }

    if (st == NVJPEG_STATUS_ARCH_MISMATCH) {
        std::cout << "hardware backend not supported; trying DEVICE variant\n";
        st = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE_DEVICE, nullptr, nullptr, 0, &h);

        if (st == NVJPEG_STATUS_SUCCESS) {
            std::cout << "hardware backend DEVICE available\n";
            nvjpegDestroy(h);
            return 0;
        }

        std::cout << "DEVICE variant also failed; status " << st << "\n";
        return 1;
    }

    std::cout << "nvjpegCreateEx error code " << st << "\n";
    return 2;
}
