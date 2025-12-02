#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <cstdio>

int main() {
    for (int dev = 0; dev < 8; dev++) {
        cudaDeviceProp p{};
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) break;

        cudaSetDevice(dev);
        printf("=== Device %d: %s cc %d.%d ===\n",
               dev, p.name, p.major, p.minor);

        nvjpegDevAllocator_t da{nullptr, nullptr};
        nvjpegPinnedAllocator_t pa{nullptr, nullptr};
        nvjpegHandle_t h{};

        nvjpegStatus_t st =
            nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE,
                           &da, &pa,
                           NVJPEG_FLAGS_DEFAULT,
                           &h);

        printf("nvjpegCreateEx(HARDWARE) status %d\n", int(st));
        if (st == NVJPEG_STATUS_SUCCESS) nvjpegDestroy(h);
    }
    return 0;
}