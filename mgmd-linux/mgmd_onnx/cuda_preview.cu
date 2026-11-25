#include <cuda_runtime.h>

namespace {
constexpr int glyph_width = 24, line_height = 28, text_columns = 128;
constexpr int hud_left = 12, hud_top = 10;

__device__ unsigned char blend(unsigned char source, uchar2 coverage) {
    return (source * (255 - coverage.y) + 127) / 255 + coverage.x;
}

__device__ uchar3 draw_hud(uchar3 color, int x, int y, const uchar2 *atlas,
                           const short *layout, int lines, int hud_width) {
    x -= hud_left;
    y -= hud_top;
    if (x < 0 || x >= hud_width || y < 0 || y >= lines * line_height) return color;
    const short *row = layout + (y / line_height) * (2 * text_columns + 1);
    const short *ends = row + 1 + text_columns;
    const int count = row[0];
    if (count == 0 || x >= ends[count - 1]) return color;

    int first = 0, last = count;
    // Find the glyph whose right edge is the first one beyond this pixel.
    while (first < last) {
        const int middle = (first + last) / 2;
        if (x < ends[middle]) last = middle;
        else first = middle + 1;
    }
    const int char_x = x - (first ? ends[first - 1] : 0);
    if (char_x >= glyph_width) return color;
    const int code = row[1 + first];
    const uchar2 coverage = atlas[(code * line_height + y % line_height) * glyph_width + char_x];
    if (coverage.y == 0) return color;
    return make_uchar3(blend(color.x, coverage), blend(color.y, coverage), blend(color.z, coverage));
}

__global__ void render(const unsigned char *src, size_t pitch, uchar3 *dst,
                       int width, int height, const float *boxes, int box_count,
                       const uchar2 *atlas, const short *layout, int lines, int hud_width) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const size_t pixel = static_cast<size_t>(y) * width + x;
    const unsigned char *bgr = src + y * pitch + x * 3;
    uchar3 color = make_uchar3(bgr[0], bgr[1], bgr[2]);
    for (int i = 0; i < box_count; ++i) {
        const float *box = boxes + i * 4;
        if ((x >= box[0] && x <= box[2] &&
             (fabsf(y - box[1]) <= 1 || fabsf(y - box[3]) <= 1)) ||
            (y >= box[1] && y <= box[3] &&
             (fabsf(x - box[0]) <= 1 || fabsf(x - box[2]) <= 1))) {
            color = make_uchar3(0, 255, 0);
            break;
        }
    }
    dst[pixel] = draw_hud(color, x, y, atlas, layout, lines, hud_width);
}
}  // namespace

extern "C" int mgmd_render_preview_hud(const unsigned char *src, size_t pitch, uchar3 *dst,
                                       int width, int height, const float *boxes, int box_count,
                                       const uchar2 *atlas, const short *layout,
                                       int lines, int hud_width, cudaStream_t stream) {
    const dim3 block(32, 8);
    const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    render<<<grid, block, 0, stream>>>(src, pitch, dst, width, height, boxes, box_count,
                                       atlas, layout, lines, hud_width);
    return cudaGetLastError();
}
