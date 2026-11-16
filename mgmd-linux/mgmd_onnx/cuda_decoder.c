/* A borrowed CUDA NV12 frame remains valid until the next read or close. */
#include <stdint.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/mem.h>

typedef struct {
    AVFormatContext *format;
    AVCodecContext *codec;
    AVPacket *packet;
    AVFrame *frame;
    int stream;
    int draining;
} Decoder;

typedef struct {
    uintptr_t y, uv;
    int width, height, y_stride, uv_stride;
    int color_space, full_range, time_num, time_den;
    int64_t pts;
} FrameView;

static enum AVPixelFormat cuda_format(AVCodecContext *ctx, const enum AVPixelFormat *formats)
{
    (void)ctx;
    for (; *formats != AV_PIX_FMT_NONE; ++formats)
        if (*formats == AV_PIX_FMT_CUDA)
            return *formats;
    return AV_PIX_FMT_NONE;
}

void mgmd_decoder_close(Decoder *d)
{
    if (!d)
        return;
    av_frame_free(&d->frame);
    av_packet_free(&d->packet);
    avcodec_free_context(&d->codec);
    avformat_close_input(&d->format);
    av_free(d);
}

void mgmd_decoder_error(int code, char *message, size_t size)
{
    av_strerror(code, message, size);
}

int mgmd_decoder_open(const char *path, const char *device, Decoder **result)
{
    Decoder *d = av_mallocz(sizeof(*d));
    if (!d)
        return AVERROR(ENOMEM);
    AVDictionary *options = NULL;
    const AVCodec *codec = NULL;
    int err;
    if ((err = avformat_open_input(&d->format, path, NULL, NULL)) < 0 ||
        (err = avformat_find_stream_info(d->format, NULL)) < 0)
        goto fail;
    d->stream = av_find_best_stream(d->format, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if ((err = d->stream) < 0)
        goto fail;
    d->codec = avcodec_alloc_context3(codec);
    d->frame = av_frame_alloc();
    d->packet = av_packet_alloc();
    err = AVERROR(ENOMEM);
    if (!d->codec || !d->frame || !d->packet)
        goto fail;
    if ((err = avcodec_parameters_to_context(d->codec,
             d->format->streams[d->stream]->codecpar)) < 0 ||
        (err = av_dict_set(&options, "current_ctx", "1", 0)) < 0 ||
        (err = av_hwdevice_ctx_create(&d->codec->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                      device, options, 0)) < 0)
        goto fail;
    av_dict_free(&options);
    d->codec->get_format = cuda_format;
    if ((err = avcodec_open2(d->codec, codec, NULL)) < 0)
        goto fail;
    *result = d;
    return 0;
fail:
    av_dict_free(&options);
    mgmd_decoder_close(d);
    return err;
}

int mgmd_decoder_next(Decoder *d, FrameView *view)
{
    av_frame_unref(d->frame);
    for (;;) {
        int err = avcodec_receive_frame(d->codec, d->frame);
        if (err == 0)
            break;
        if (err == AVERROR_EOF)
            return 0;
        if (err != AVERROR(EAGAIN) || d->draining)
            return err;
        while ((err = av_read_frame(d->format, d->packet)) >= 0) {
            if (d->packet->stream_index == d->stream)
                break;
            av_packet_unref(d->packet);
        }
        if (err == AVERROR_EOF) {
            d->draining = 1;
            err = avcodec_send_packet(d->codec, NULL);
        } else if (err >= 0) {
            err = avcodec_send_packet(d->codec, d->packet);
            av_packet_unref(d->packet);
        }
        if (err < 0)
            return err;
    }
    AVFrame *f = d->frame;
    const AVHWFramesContext *hw = f->hw_frames_ctx ?
        (const AVHWFramesContext *)f->hw_frames_ctx->data : NULL;
    if (!hw || f->format != AV_PIX_FMT_CUDA || hw->sw_format != AV_PIX_FMT_NV12 ||
        !f->data[0] || !f->data[1] || f->linesize[0] < f->width ||
        f->linesize[1] < f->width || ((f->width | f->height) & 1) ||
        f->best_effort_timestamp == AV_NOPTS_VALUE)
        return AVERROR_INVALIDDATA;
    AVRational tb = d->format->streams[d->stream]->time_base;
    *view = (FrameView){
        (uintptr_t)f->data[0], (uintptr_t)f->data[1],
        f->width, f->height, f->linesize[0], f->linesize[1],
        f->colorspace, f->color_range == AVCOL_RANGE_JPEG,
        tb.num, tb.den, f->best_effort_timestamp
    };
    return 1;
}
