#include <torch/extension.h>
#include <nvjpeg.h>

#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <chrono>

namespace {

nvjpegHandle_t g_handle = nullptr;
nvjpegJpegState_t g_state = nullptr;
bool g_initialized = false;

static double g_read_seconds = 0.0;
static double g_info_seconds = 0.0;
static double g_decode_seconds = 0.0;

inline double now() {
    return std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now().time_since_epoch()
    ).count();
}

void check_nvjpeg(nvjpegStatus_t status, const char* msg) {
    if (status != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("nvJPEG error in ") + msg +
                                 ": " + std::to_string(status));
    }
}

void init_nvjpeg_once() {
    if (g_initialized) return;
    check_nvjpeg(nvjpegCreateSimple(&g_handle), "nvjpegCreateSimple");
    check_nvjpeg(nvjpegJpegStateCreate(g_handle, &g_state), "nvjpegJpegStateCreate");
    g_initialized = true;
}

std::vector<unsigned char> read_file_bytes(const std::string& path) {
    double t0 = now();

    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Failed to open file: " + path);

    std::streamsize size = f.tellg();
    if (size <= 0) throw std::runtime_error("Empty or invalid file: " + path);

    std::vector<unsigned char> buf(size);
    f.seekg(0, std::ios::beg);
    if (!f.read(reinterpret_cast<char*>(buf.data()), size))
        throw std::runtime_error("Failed to read file: " + path);

    double t1 = now();
    g_read_seconds += (t1 - t0);
    return buf;
}

} // namespace


torch::Tensor decode_batch(const std::vector<std::string>& paths) {
    init_nvjpeg_once();
    int N = paths.size();
    if (N == 0) {
        return torch::empty({0,3,1,1}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    }

    // --- INFO TIMER ---
    auto buf0 = read_file_bytes(paths[0]);
    int nComponents = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];

    double t0 = now();
    check_nvjpeg(
        nvjpegGetImageInfo(
            g_handle,
            buf0.data(),
            buf0.size(),
            &nComponents,
            &subsampling,
            widths,
            heights
        ),
        "nvjpegGetImageInfo"
    );
    g_info_seconds += (now() - t0);

    int H = heights[0];
    int W = widths[0];

    auto out_uint8 =
        torch::empty({N,3,H,W}, torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    for (int i = 0; i < N; ++i) {
        auto buf = read_file_bytes(paths[i]);
        torch::Tensor img_tensor = out_uint8[i];

        nvjpegImage_t desc;
        for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
            desc.channel[c] = nullptr;
            desc.pitch[c] = 0;
        }
        for (int c = 0; c < 3; c++) {
            desc.channel[c] = img_tensor[c].data_ptr<unsigned char>();
            desc.pitch[c] = W * sizeof(unsigned char);
        }

        double d0 = now();
        check_nvjpeg(
            nvjpegDecode(
                g_handle,
                g_state,
                buf.data(),
                buf.size(),
                NVJPEG_OUTPUT_RGB,
                &desc,
                0
            ),
            "nvjpegDecode"
        );
        g_decode_seconds += (now() - d0);
    }

    auto out_float = out_uint8.to(torch::kFloat32).div_(255.0);
    return out_float;
}

double get_read_time()   { return g_read_seconds; }
double get_info_time()   { return g_info_seconds; }
double get_decode_time() { return g_decode_seconds; }

void reset_all_timers() {
    g_read_seconds = 0.0;
    g_info_seconds = 0.0;
    g_decode_seconds = 0.0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_batch", &decode_batch);
    m.def("get_read_time", &get_read_time);
    m.def("get_info_time", &get_info_time);
    m.def("get_decode_time", &get_decode_time);
    m.def("reset_all", &reset_all_timers);
}
