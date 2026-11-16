/* Rectangular assignment follows SciPy's Crouse solver, including tie ordering.
 * SciPy source: scipy/optimize/rectangular_lsap/rectangular_lsap.cpp, v1.10.1.
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>

struct Box { double x1, y1, x2, y2; };

struct Track {
    double value[4], velocity[3];
    // Three independent position/velocity covariance blocks, plus aspect ratio.
    double covariance[3][4], aspect_variance;
    Box history[4];
    int64_t history_frame[4], predicted_at, id;
    int hits, misses, age;
    double d, score;
};

struct Control {
    int active, rows, cols, error;
    int64_t next_id;
};

struct Workspace {
    Control *ctl;
    Track *tracks;
    Box *predictions;
    double *cost, *u, *v, *dist;
    int *row_map, *col_map, *match, *owner, *path, *row_to_col, *col_to_row, *remaining;
    uint8_t *seen_rows, *seen_cols;
    int track_capacity, det_capacity;
};

struct Result {
    Box *boxes;
    int64_t *ids;
    double *details;  // score, D, vx, vy
    int *lifecycle;   // hits, misses, age
    int64_t *assigned_ids;
    int *count, *status;
};

__device__ double width(Box b) { return b.x2 - b.x1; }
__device__ double height(Box b) { return b.y2 - b.y1; }
__device__ double diagonal(Box b) { return sqrt(width(b) * width(b) + height(b) * height(b)); }

__device__ double center_distance(Box a, Box b)
{
    const double x = (a.x1 + a.x2) / 2 - (b.x1 + b.x2) / 2;
    const double y = (a.y1 + a.y2) / 2 - (b.y1 + b.y2) / 2;
    return sqrt(x * x + y * y);
}

__device__ double overlap(Box a, Box b)
{
    const double iw = fmax(0.0, fmin(a.x2, b.x2) - fmax(a.x1, b.x1));
    const double ih = fmax(0.0, fmin(a.y2, b.y2) - fmax(a.y1, b.y1));
    const double area = iw * ih;
    const double total = fmax(0.0, width(a)) * fmax(0.0, height(a))
                       + fmax(0.0, width(b)) * fmax(0.0, height(b)) - area;
    return total > 0 ? area / total : 0;
}

__device__ void measurement(Box b, double *z)
{
    const double w = fmax(width(b), 1e-6), h = fmax(height(b), 1e-6);
    z[0] = (b.x1 + b.x2) / 2;
    z[1] = (b.y1 + b.y2) / 2;
    z[2] = w * h;
    z[3] = w / h;
}

__device__ Box predict(Track &t, int64_t frame)
{
    const int64_t dt = frame - t.predicted_at;
    const int steps = dt < 1 ? 1 : (dt > 30 ? 30 : int(dt));
    for (int i = 0; i < steps; ++i) {
        if (t.value[2] + t.velocity[2] <= 0)
            t.velocity[2] = 0;
        for (int axis = 0; axis < 3; ++axis) {
            t.value[axis] += t.velocity[axis];
            double *p = t.covariance[axis];
            const double right = p[1] + p[3];
            p[0] = ((p[0] + p[2]) + right) + 1.0;
            p[1] = right;
            p[2] += p[3];
            p[3] += axis == 2 ? .0001 : .01;
        }
        t.aspect_variance += 1;
    }
    t.predicted_at = frame;
    const double area = fmax(t.value[2], 1e-6);
    const double w = sqrt(area * fmax(t.value[3], 1e-6));
    const double h = area / fmax(w, 1e-6);
    return {t.value[0] - w / 2, t.value[1] - h / 2,
            t.value[0] + w / 2, t.value[1] + h / 2};
}

__device__ void update(Track &t, Box box)
{
    double z[4];
    measurement(box, z);
    for (int axis = 0; axis < 3; ++axis) {
        double *p = t.covariance[axis];
        const double inverse = 1 / (p[0] + (axis == 2 ? 10 : 1));
        const double k0 = p[0] * inverse, k1 = p[2] * inverse;
        const double residual = z[axis] - t.value[axis];
        t.value[axis] += k0 * residual;
        t.velocity[axis] += k1 * residual;
        p[2] -= k1 * p[0];
        p[3] -= k1 * p[1];
        p[0] *= 1 - k0;
        p[1] *= 1 - k0;
    }
    const double k = t.aspect_variance * (1 / (t.aspect_variance + 10));
    t.value[3] += k * (z[3] - t.value[3]);
    t.aspect_variance *= 1 - k;
}

__device__ void append(Track &t, int64_t frame, Box box, double score)
{
    const int slot = t.hits % 4;
    t.history[slot] = box;
    t.history_frame[slot] = frame;
    ++t.hits;
    t.misses = 0;
    t.score = score;
    t.d = 0;
    if (t.hits < 3)
        return;
    const int count = min(4, t.hits);
    const int first = (t.hits - count) % 4;
    int older = first;
    for (int i = 0; i < count - 1; ++i) {
        const int candidate = (first + i) % 4;
        if (t.history_frame[candidate] <= frame - 3 &&
            t.history_frame[candidate] > t.history_frame[older])
            older = candidate;
    }
    const Box previous = t.history[older];
    const double diameter = (diagonal(previous) + diagonal(box)) / 2;
    if (t.history_frame[older] < frame && diameter > 0)
        t.d = center_distance(previous, box) / diameter;
}

__global__ void initialize(Workspace w)
{
    *w.ctl = {0, 0, 0, 0, 1};
}

__global__ void begin(Workspace w, const int *count, int supplied, Result out)
{
    *out.count = 0;
    const int n = count ? *count : supplied;
    if (n < 0 || n > supplied || n > w.det_capacity)
        w.ctl->error = 1;
    w.ctl->rows = w.ctl->active;
    w.ctl->cols = n;
    *out.status = w.ctl->error;
}

__global__ void predict_and_map(Workspace w, int64_t frame)
{
    if (w.ctl->error)
        return;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w.ctl->rows) {
        w.row_map[i] = i;
        w.match[i] = -1;
        w.predictions[i] = predict(w.tracks[i], frame);
    }
    if (i < w.ctl->cols) {
        w.col_map[i] = i;
        w.owner[i] = -1;
    }
}

__global__ void costs(Workspace w, const Box *detections, bool secondary)
{
    if (w.ctl->error)
        return;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const int rows = w.ctl->rows, cols = w.ctl->cols;
    if (p >= rows * cols)
        return;
    const bool transpose = cols < rows;
    const int row = transpose ? p % rows : p / cols;
    const int col = transpose ? p / rows : p % cols;
    const Box a = w.predictions[w.row_map[row]], b = detections[w.col_map[col]];
    if (secondary) {
        const double radius = diagonal(a) + diagonal(b); // association factor 2.0
        const double d = radius > 0 ? center_distance(a, b) / radius : CUDART_INF;
        w.cost[p] = d <= 1 ? d : 1e6;
    } else {
        w.cost[p] = -overlap(a, b);
    }
}

struct Choice {
    double distance;
    int rank;
    int position;
};

__device__ Choice closer(Choice a, Choice b)
{
    return b.distance < a.distance || (b.distance == a.distance && b.rank < a.rank) ? b : a;
}

__device__ Choice warp_min(Choice best)
{
    const int lane = threadIdx.x & 31;
    for (int offset = 16; offset; offset >>= 1) {
        Choice other{
            __shfl_down_sync(0xffffffff, best.distance, offset),
            __shfl_down_sync(0xffffffff, best.rank, offset),
            __shfl_down_sync(0xffffffff, best.position, offset)
        };
        if (lane + offset < 32)
            best = closer(best, other);
    }
    return best;
}

constexpr int solver_threads = 128;

__global__ void solve(Workspace w, bool secondary)
{
    if (w.ctl->error || !w.ctl->rows || !w.ctl->cols)
        return;
    const int rows = w.ctl->rows, cols = w.ctl->cols;
    const bool transpose = cols < rows;
    const int nr = min(rows, cols), nc = max(rows, cols);
    const int thread = threadIdx.x;
    const int lane = thread & 31;
    __shared__ Choice warps[solver_threads / 32];
    __shared__ int free_columns, row, sink;
    __shared__ double minimum;
    for (int i = thread; i < nr; i += solver_threads) {
        w.u[i] = 0;
        w.row_to_col[i] = -1;
    }
    for (int j = thread; j < nc; j += solver_threads) {
        w.v[j] = 0;
        w.col_to_row[j] = -1;
    }
    __syncthreads();
    for (int start = 0; start < nr; ++start) {
        for (int i = thread; i < nr; i += solver_threads)
            w.seen_rows[i] = 0;
        for (int j = thread; j < nc; j += solver_threads) {
            w.seen_cols[j] = 0;
            w.dist[j] = CUDART_INF;
            w.remaining[j] = nc - j - 1;
        }
        if (thread == 0) {
            free_columns = nc;
            row = start;
            sink = -1;
            minimum = 0;
        }
        __syncthreads();
        while (sink < 0) {
            const Choice empty{CUDART_INF, INT_MAX, -1};
            Choice best = empty;
            if (thread == 0)
                w.seen_rows[row] = 1;
            for (int p = thread; p < free_columns; p += solver_threads) {
                const int col = w.remaining[p];
                const double candidate = minimum + w.cost[row * nc + col] - w.u[row] - w.v[col];
                if (candidate < w.dist[col]) {
                    w.dist[col] = candidate;
                    w.path[col] = row;
                }
                // SciPy prefers a free sink; among tied sinks it keeps the last
                // remaining position, and among occupied columns the first.
                const int rank = w.col_to_row[col] < 0 ? -p - 1 : p;
                best = closer(best, {w.dist[col], rank, p});
            }
            best = warp_min(best);
            if (lane == 0)
                warps[thread / 32] = best;
            __syncthreads();
            if (thread < 32) {
                best = warp_min(lane < solver_threads / 32 ? warps[lane] : empty);
                if (thread == 0) {
                    if (best.position < 0 || !isfinite(best.distance)) {
                        w.ctl->error = 2;
                    } else {
                        minimum = best.distance;
                        const int col = w.remaining[best.position];
                        const int owner = w.col_to_row[col];
                        if (owner < 0)
                            sink = col;
                        else
                            row = owner;
                        w.seen_cols[col] = 1;
                        w.remaining[best.position] = w.remaining[--free_columns];
                    }
                }
            }
            __syncthreads();
            if (w.ctl->error)
                return;
        }
        if (thread == 0)
            w.u[start] += minimum;
        for (int i = thread; i < nr; i += solver_threads)
            if (i != start && w.seen_rows[i])
                w.u[i] += minimum - w.dist[w.row_to_col[i]];
        for (int j = thread; j < nc; j += solver_threads)
            if (w.seen_cols[j])
                w.v[j] -= minimum - w.dist[j];
        __syncthreads();
        if (thread == 0) {
            for (int col = sink;;) {
                const int i = w.path[col];
                const int old = w.row_to_col[i];
                w.row_to_col[i] = col;
                w.col_to_row[col] = i;
                if (i == start)
                    break;
                col = old;
            }
        }
        __syncthreads();
    }
    for (int i = thread; i < nr; i += solver_threads) {
        const int j = w.row_to_col[i];
        const int row = transpose ? j : i, col = transpose ? i : j;
        const double value = w.cost[i * nc + j];
        if (secondary ? value <= 1 : value <= -.3) {
            const int track = w.row_map[row], det = w.col_map[col];
            w.match[track] = det;
            w.owner[det] = track;
        }
    }
}

__global__ void unmatched(Workspace w)
{
    if (w.ctl->error)
        return;
    int rows = 0, cols = 0;
    for (int r = 0; r < w.ctl->rows; ++r)
        if (w.match[r] < 0)
            w.row_map[rows++] = r;
    for (int c = 0; c < w.ctl->cols; ++c)
        if (w.owner[c] < 0)
            w.col_map[cols++] = c;
    w.ctl->rows = rows;
    w.ctl->cols = cols;
}

__global__ void update_tracks(Workspace w, const Box *boxes, const double *scores, int64_t frame)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (w.ctl->error || i >= w.ctl->active)
        return;
    Track &t = w.tracks[i];
    const int d = w.match[i];
    if (d >= 0) {
        update(t, boxes[d]);
        append(t, frame, boxes[d], scores[d]);
    } else {
        ++t.misses;
        ++t.age;
    }
}

__global__ void finish(Workspace w, const Box *boxes, int64_t frame, Result out)
{
    if (w.ctl->error) {
        *out.status = w.ctl->error;
        return;
    }
    int active = 0;
    for (int i = 0; i < w.ctl->active; ++i) {
        if (w.match[i] >= 0)
            out.assigned_ids[w.match[i]] = w.tracks[i].id;
        if (w.tracks[i].misses <= 3) {
            if (i != active)
                w.tracks[active] = w.tracks[i];
            ++active;
        }
    }
    int output_count = 0;
    for (int i = 0; i < active; ++i) {
        const Track &t = w.tracks[i];
        if (t.misses || t.hits < 3 || t.d < .02)
            continue;
        const int p = output_count++;
        out.boxes[p] = t.history[(t.hits - 1) % 4];
        out.ids[p] = t.id;
        out.details[4 * p] = t.score;
        out.details[4 * p + 1] = t.d;
        out.details[4 * p + 2] = t.velocity[0];
        out.details[4 * p + 3] = t.velocity[1];
        out.lifecycle[3 * p] = t.hits;
        out.lifecycle[3 * p + 1] = t.misses;
        out.lifecycle[3 * p + 2] = t.age;
    }
    int new_tracks = 0;
    for (int i = 0; i < w.ctl->cols; ++i)
        new_tracks += w.owner[w.col_map[i]] < 0;
    if (active + new_tracks > w.track_capacity) {
        w.ctl->error = 1;
        *out.status = 1;
        return;
    }
    // col_map retains the columns unmatched by the first round, in input order.
    for (int i = 0; i < w.ctl->cols; ++i) {
        const int det = w.col_map[i];
        if (w.owner[det] >= 0)
            continue;
        Track &t = w.tracks[active++];
        t = {};
        t.id = w.ctl->next_id++;
        out.assigned_ids[det] = t.id;
        t.predicted_at = frame;
        t.age = 1;
        measurement(boxes[det], t.value);
        for (int axis = 0; axis < 3; ++axis) {
            t.covariance[axis][0] = 10;
            t.covariance[axis][3] = 10000;
        }
        t.aspect_variance = 10;
        append(t, frame, boxes[det], 0);
    }
    w.ctl->active = active;
    *out.count = output_count;
}

extern "C" int mgmd_tf_step(
    Workspace w, const Box *boxes, const double *scores, const int *count,
    int supplied, int64_t frame, Result out, cudaStream_t stream)
{
    const int threads = 128;
    const int width = w.track_capacity > w.det_capacity ? w.track_capacity : w.det_capacity;
    const int grid = (width + threads - 1) / threads;
    const int pairs = (w.track_capacity * w.det_capacity + threads - 1) / threads;
    begin<<<1, 1, 0, stream>>>(w, count, supplied, out);
    predict_and_map<<<grid, threads, 0, stream>>>(w, frame);
    costs<<<pairs, threads, 0, stream>>>(w, boxes, false);
    solve<<<1, solver_threads, 0, stream>>>(w, false);
    unmatched<<<1, 1, 0, stream>>>(w);
    costs<<<pairs, threads, 0, stream>>>(w, boxes, true);
    solve<<<1, solver_threads, 0, stream>>>(w, true);
    update_tracks<<<grid, threads, 0, stream>>>(w, boxes, scores, frame);
    finish<<<1, 1, 0, stream>>>(w, boxes, frame, out);
    return cudaGetLastError();
}

extern "C" int mgmd_tf_reset(Workspace w, cudaStream_t stream)
{
    initialize<<<1, 1, 0, stream>>>(w);
    return cudaGetLastError();
}

extern "C" size_t mgmd_tf_track_size() { return sizeof(Track); }
extern "C" size_t mgmd_tf_control_size() { return sizeof(Control); }
