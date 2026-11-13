// Block-based union-find for binary eight-connected components.
// Each 2x2 tile's foreground pixels are mutually eight-connected.
#include <cuda_runtime.h>

__device__ int root(int* parent, int i) {
    for (;;) {
        int next = atomicAdd(parent + i, 0);
        if (next == i) return i;
        i = next;
    }
}

__device__ void join(int* parent, int a, int b) {
    for (;;) {
        a = root(parent, a);
        b = root(parent, b);
        if (a == b) return;
        int hi = max(a, b), lo = min(a, b);
        if (atomicCAS(parent + hi, hi, lo) == hi) return;
    }
}

__global__ void initialize(int* parent, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) parent[i] = i;
}

__global__ void merge(const unsigned char* mask, int w, int h, int tiles_x, int* parent) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h || !mask[y*w+x]) return;
    int tile = (y/2)*tiles_x+x/2;
    for (int dy = -1; dy <= 0; ++dy) {
        for (int dx = -1; dx <= (dy == -1 ? 1 : -1); ++dx) {
            int xx = x+dx, yy = y+dy;
            if (xx < 0 || xx >= w || yy < 0) continue;
            int other = (yy/2)*tiles_x+xx/2;
            if (other != tile && mask[yy*w+xx]) join(parent, tile, other);
        }
    }
}

__global__ void resolve(const unsigned char* mask, int w, int h, int tiles_x,
                        int* parent, int* labels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h)
        labels[y*w+x] = mask[y*w+x] ? root(parent, (y/2)*tiles_x+x/2)+1 : 0;
}

extern "C" int mgmd_buf_ccl(const unsigned char* mask, int w, int h, int* parent,
                           int* labels, cudaStream_t stream) {
    int tiles_x = (w+1)/2, tiles = tiles_x*((h+1)/2);
    dim3 block(32, 8), grid((w+31)/32, (h+7)/8);
    initialize<<<(tiles+255)/256, 256, 0, stream>>>(parent, tiles);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    merge<<<grid, block, 0, stream>>>(mask, w, h, tiles_x, parent);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    resolve<<<grid, block, 0, stream>>>(mask, w, h, tiles_x, parent, labels);
    return cudaGetLastError();
}
