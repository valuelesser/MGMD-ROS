// Experimental single-class LAD postprocessing and BUF region measurements.
#include <cuda_runtime.h>
#include <climits>
#include <math_constants.h>

extern "C" int mgmd_host_pointer(void **device, void *host) {
    return cudaHostGetDevicePointer(device, host, 0);
}

struct Region { int area, first, left, top, right, bottom; };

__global__ void initialize(Region *regions, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) regions[i] = {0, INT_MAX, INT_MAX, INT_MAX, 0, 0};
}

__global__ void accumulate(const int *labels, int pixels, int width, Region *regions) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pixels || labels[i] == 0) return;
    Region *r = regions + labels[i];
    int x = i % width, y = i / width;
    atomicAdd(&r->area, 1);
    atomicMin(&r->first, i);
    atomicMin(&r->left, x);
    atomicMin(&r->top, y);
    atomicMax(&r->right, x);
    atomicMax(&r->bottom, y);
}

extern "C" int mgmd_regions(const int *labels, int width, int height, Region *regions,
                            int capacity, cudaStream_t stream) {
    initialize<<<(capacity+255)/256,256,0,stream>>>(regions, capacity);
    accumulate<<<(width*height+255)/256,256,0,stream>>>(labels, width*height, width, regions);
    return cudaGetLastError();
}

// Keep the same greedy, score-ordered top 300 as YOLOv5, then choose the
// rounded, clipped box nearest the crop centre. One kernel handles one ROI.
__global__ void select_box(const float *raw, const long long *order, int n,
                           float threshold, float *out) {
    float4 kept[300];
    int count = 0;
    float best = CUDART_INF_F;
    for (int k = 0; k < n && count < 300; ++k) {
        const float *p = raw + 6*order[k];
        float score = p[4]*p[5];
        if (score <= threshold) break;
        if (!(p[4] > threshold && score > threshold)) continue;
        float4 box = make_float4(p[0]-p[2]/2, p[1]-p[3]/2, p[0]+p[2]/2, p[1]+p[3]/2);
        bool suppressed = false;
        for (int j = 0; j < count; ++j) {
            float4 b = kept[j];
            float inter = fmaxf(0, fminf(b.z,box.z)-fmaxf(b.x,box.x)) *
                          fmaxf(0, fminf(b.w,box.w)-fmaxf(b.y,box.y));
            float area = (box.z-box.x)*(box.w-box.y) + (b.z-b.x)*(b.w-b.y);
            if (inter / (area-inter) > .45f) { suppressed = true; break; }
        }
        if (suppressed) continue;
        kept[count++] = box;
        float coords[] = {box.x, box.y, box.z, box.w};
        for (int j=0; j<4; ++j) coords[j] = fminf(320, fmaxf(0, nearbyintf(coords[j])));
        float x = (coords[0]+coords[2])*.5f-160, y = (coords[1]+coords[3])*.5f-160;
        float distance = x*x+y*y;
        if (distance < best) {
            best = distance;
            for (int j=0; j<4; ++j) out[j] = coords[j];
            out[4] = score;
        }
    }
}

extern "C" int mgmd_select_box(const float *raw, const long long *order, int n,
                               float threshold, float *out, cudaStream_t stream) {
    select_box<<<1,1,0,stream>>>(raw, order, n, threshold, out);
    return cudaGetLastError();
}
