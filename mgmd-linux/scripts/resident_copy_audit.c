/* LD_PRELOAD probe for scoped CUDA runtime copies on the calling thread. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stddef.h>
#include <stdio.h>

static __thread const char *scope;
void audit_copy_scope(const char *label) { scope = label; }

int cudaMemcpyAsync(void *dst, const void *src, size_t bytes, int kind, void *stream) {
    typedef int (*copy_fn)(void *, const void *, size_t, int, void *);
    copy_fn copy = (copy_fn)dlsym(RTLD_NEXT, "cudaMemcpyAsync");
    if (scope) fprintf(stderr, "%s async kind=%d bytes=%zu\n", scope, kind, bytes);
    return copy(dst, src, bytes, kind, stream);
}

int cudaStreamSynchronize(void *stream) {
    typedef int (*sync_fn)(void *);
    sync_fn sync = (sync_fn)dlsym(RTLD_NEXT, "cudaStreamSynchronize");
    if (scope) fprintf(stderr, "%s wait=stream\n", scope);
    return sync(stream);
}

int cudaEventSynchronize(void *event) {
    typedef int (*sync_fn)(void *);
    sync_fn sync = (sync_fn)dlsym(RTLD_NEXT, "cudaEventSynchronize");
    if (scope) fprintf(stderr, "%s wait=event\n", scope);
    return sync(event);
}

int cudaMemcpy(void *dst, const void *src, size_t bytes, int kind) {
    typedef int (*copy_fn)(void *, const void *, size_t, int);
    copy_fn copy = (copy_fn)dlsym(RTLD_NEXT, "cudaMemcpy");
    if (scope) fprintf(stderr, "%s sync kind=%d bytes=%zu\n", scope, kind, bytes);
    return copy(dst, src, bytes, kind);
}
