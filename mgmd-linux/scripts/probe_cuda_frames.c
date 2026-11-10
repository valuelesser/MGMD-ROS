/* Verify libav returns CUDA AVFrames without downloading pixel data. */
#include <inttypes.h>
#include <stdio.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixdesc.h>

static enum AVPixelFormat choose_cuda(AVCodecContext *ctx, const enum AVPixelFormat *formats)
{
    (void)ctx;
    for (; *formats != AV_PIX_FMT_NONE; ++formats)
        if (*formats == AV_PIX_FMT_CUDA)
            return *formats;
    return AV_PIX_FMT_NONE;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        fprintf(stderr, "usage: %s video.mp4\n", argv[0]);
        return 2;
    }
    AVFormatContext *input = NULL;
    AVCodecContext *decoder = NULL;
    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    int count = 0;
    int64_t first_pts = AV_NOPTS_VALUE, last_pts = AV_NOPTS_VALUE;
    int err = AVERROR(ENOMEM);
    if (!packet || !frame)
        goto done;
    if ((err = avformat_open_input(&input, argv[1], NULL, NULL)) < 0 ||
        (err = avformat_find_stream_info(input, NULL)) < 0)
        goto done;
    int stream = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
    if ((err = stream) < 0)
        goto done;
    const AVCodec *codec = avcodec_find_decoder(input->streams[stream]->codecpar->codec_id);
    if (!codec) {
        err = AVERROR_DECODER_NOT_FOUND;
        goto done;
    }
    decoder = avcodec_alloc_context3(codec);
    if (!decoder) {
        err = AVERROR(ENOMEM);
        goto done;
    }
    if ((err = avcodec_parameters_to_context(decoder, input->streams[stream]->codecpar)) < 0 ||
        (err = av_hwdevice_ctx_create(&decoder->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                      NULL, NULL, 0)) < 0)
        goto done;
    decoder->get_format = choose_cuda;
    if ((err = avcodec_open2(decoder, codec, NULL)) < 0)
        goto done;
    while (count < 120 && (err = av_read_frame(input, packet)) >= 0) {
        if (packet->stream_index == stream) {
            if ((err = avcodec_send_packet(decoder, packet)) < 0)
                goto done;
            while (count < 120 && (err = avcodec_receive_frame(decoder, frame)) == 0) {
                AVHWFramesContext *hw = frame->hw_frames_ctx ?
                    (AVHWFramesContext *)frame->hw_frames_ctx->data : NULL;
                int64_t pts = frame->best_effort_timestamp;
                if (!hw || frame->format != AV_PIX_FMT_CUDA || !frame->data[0] ||
                    pts == AV_NOPTS_VALUE || (count && pts <= last_pts)) {
                    err = AVERROR_INVALIDDATA;
                    goto done;
                }
                if (count == 0) {
                    first_pts = pts;
                    printf("device=cuda sw_format=%s width=%d height=%d stride=%d\n",
                           av_get_pix_fmt_name(hw->sw_format), frame->width, frame->height,
                           frame->linesize[0]);
                }
                last_pts = pts;
                ++count;
                av_frame_unref(frame);
            }
            if (count < 120 && err != AVERROR(EAGAIN))
                goto done;
        }
        av_packet_unref(packet);
    }
    printf("frames=%d first_pts=%" PRId64 " last_pts=%" PRId64 "\n", count, first_pts, last_pts);
    err = count == 120 ? 0 : AVERROR_INVALIDDATA;
done:
    if (err < 0)
        fprintf(stderr, "CUDA frame probe: %s\n", av_err2str(err));
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    avformat_close_input(&input);
    return err < 0;
}
