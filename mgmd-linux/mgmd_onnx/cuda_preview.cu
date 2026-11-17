#include <cuda_runtime.h>

namespace {
constexpr int cell_width = 17, line_height = 32, text_columns = 128;

__global__ void render(const unsigned char *src, size_t pitch, uchar3 *dst,
                       int width, int height, const float *boxes, int box_count,
                       const unsigned char *atlas, const unsigned char *text,
                       int lines, int columns) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t pixel = static_cast<size_t>(y) * width + x;
    if (x >= 16 && x < 40 + columns * cell_width &&
        y >= 16 && y < 64 + line_height * (lines - 1)) {
        int alpha = 0;
        if (x >= 28 && y >= 22) {
            const int tx = x - 28, ty = y - 22;
            const int col = tx / cell_width, row = ty / line_height;
            if (col < columns && row < lines) {
                const int ch = text[row * text_columns + col];
                alpha = atlas[(ch * line_height + ty % line_height) * cell_width
                              + tx % cell_width];
            }
        }
        const unsigned char fg = 18 + (237 * alpha + 127) / 255;
        dst[pixel] = make_uchar3(fg, fg, fg);
        return;
    }
    for (int i = 0; i < box_count; ++i) {
        const float *box = boxes + i * 4;
        if ((x >= box[0] && x <= box[2] &&
             (fabsf(y - box[1]) <= 1 || fabsf(y - box[3]) <= 1)) ||
            (y >= box[1] && y <= box[3] &&
             (fabsf(x - box[0]) <= 1 || fabsf(x - box[2]) <= 1))) {
            dst[pixel] = make_uchar3(0, 255, 0);
            return;
        }
    }
    const unsigned char *bgr = src + y * pitch + x * 3;
    dst[pixel] = make_uchar3(bgr[0], bgr[1], bgr[2]);
}
}  // namespace

extern "C" int mgmd_render_preview(const unsigned char *src, size_t pitch, uchar3 *dst,
                                   int width, int height, const float *boxes, int box_count,
                                   const unsigned char *atlas, const unsigned char *text,
                                   int lines, int columns, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    render<<<grid, block, 0, stream>>>(src, pitch, dst, width, height, boxes, box_count,
                                       atlas, text, lines, columns);
    return cudaGetLastError();
}
