/*
 * Copyright (c) 2025 Lynne <dev@lynne.ee>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "vulkan_decode.h"
#include "hwaccel_internal.h"

#include "apv_decode.h"
#include "libavutil/vulkan_spirv.h"
#include "libavutil/mem.h"

extern const char *ff_source_common_comp;
extern const char *ff_source_apv_decode_comp;
extern const char *ff_source_apv_idct_comp;

const FFVulkanDecodeDescriptor ff_vk_dec_apv_desc = {
    .codec_id         = AV_CODEC_ID_APV,
    .decode_extension = FF_VK_EXT_PUSH_DESCRIPTOR,
    .queue_flags      = VK_QUEUE_COMPUTE_BIT,
};

typedef struct APVVulkanDecodePicture {
    FFVulkanDecodePicture vp;

    AVBufferRef *frame_data_buf;
    uint32_t    *frame_data;
    int          tile_num;
} APVVulkanDecodePicture;

typedef struct APVVulkanDecodeContext {
    FFVulkanShader decode;
    FFVulkanShader idct;

    FFVkBuffer    clut_buf;
    AVBufferPool *frame_data_pool;
} APVVulkanDecodeContext;

typedef struct DecodePushData {
    VkDeviceAddress tile_data;
    int tile_count[2];
    int log2_chroma_sub[2];
    int components;
} DecodePushData;

static int vk_apv_start_frame(AVCodecContext          *avctx,
                              const AVBufferRef       *buffer_ref,
                              av_unused const uint8_t *buffer,
                              av_unused uint32_t       size)
{
    int err;
    APVDecodeContext *apv = avctx->priv_data;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx = dec->shared_ctx;
    APVVulkanDecodeContext *apvvk = ctx->sd_ctx;

    APVVulkanDecodePicture *apvvp = apv->hwaccel_picture_private;
    FFVulkanDecodePicture *vp = &apvvp->vp;

    /* Host map the input tile data if supported */
    if (0 && ctx->s.extensions & FF_VK_EXT_EXTERNAL_HOST_MEMORY)
        ff_vk_host_map_buffer(&ctx->s, &vp->slices_buf, buffer_ref->data,
                              buffer_ref,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    /* Allocate frame data buffer */
    int fd_size = (2*4*APV_MAX_TILE_COUNT)*APV_MAX_NUM_COMP +
                  (64 + APV_MAX_TILE_COUNT)*APV_MAX_NUM_COMP +
                  (APV_MAX_TILE_COLS + 1 + APV_MAX_TILE_ROWS + 1)*2;

    err = ff_vk_get_pooled_buffer(&ctx->s, &apvvk->frame_data_pool,
                                  &apvvp->frame_data_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, fd_size,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (err < 0)
        return err;

    /* Frame data */
    FFVkBuffer *frame_data = (FFVkBuffer *)apvvp->frame_data_buf->data;
    uint8_t *fd = frame_data->mapped_mem;

    fd += 2*4*APV_MAX_TILE_COUNT*APV_MAX_NUM_COMP; /* Tile offsets go first */

    /* per-component qmatrix and QPs */
    for (int i = 0; i < APV_MAX_NUM_COMP; i++)
        memcpy(fd + 64*i,
               apv->cur_raw_frame->frame_header.quantization_matrix.q_matrix[i],
               64);
    fd += 64*APV_MAX_NUM_COMP;

    for (int i = 0; i < APV_MAX_NUM_COMP; i++) {
        for (int j = 0; j < APV_MAX_TILE_COUNT; j++)
            fd[j] = apv->cur_raw_frame->tile[j].tile_header.tile_qp[i];
        fd += APV_MAX_TILE_COUNT;
    }

    /* tile col/row offset */
    memcpy(fd, apv->tile_info.col_starts, (APV_MAX_TILE_COLS+1)*2);
    fd += (APV_MAX_TILE_COLS+1)*2;
    memcpy(fd, apv->tile_info.row_starts, (APV_MAX_TILE_ROWS+1)*2);

    /* Prepare frame to be used */
    err = ff_vk_decode_prepare_frame_sdr(dec, apv->output_frame, vp, 1,
                                         FF_VK_REP_NATIVE, 0);
    if (err < 0)
        return err;

    return 0;
}

static int vk_apv_decode_slice(AVCodecContext *avctx,
                               const uint8_t  *data,
                               uint32_t        size)
{
    APVDecodeContext *apv = avctx->priv_data;

    APVVulkanDecodePicture *apvvp = apv->hwaccel_picture_private;
    FFVulkanDecodePicture *vp = &apvvp->vp;

    FFVkBuffer *frame_data = (FFVkBuffer *)apvvp->frame_data_buf->data;
    FFVkBuffer *slices_buf = vp->slices_buf ? (FFVkBuffer *)vp->slices_buf->data : NULL;

    if (slices_buf && slices_buf->host_ref) {
        AV_WN32(frame_data->mapped_mem + (2*apvvp->tile_num + 0)*sizeof(uint32_t),
                data - slices_buf->mapped_mem);
        AV_WN32(frame_data->mapped_mem + (2*apvvp->tile_num + 1)*sizeof(uint32_t),
                size);

        apvvp->tile_num++;
    } else {
        int err = ff_vk_decode_add_slice(avctx, vp, data, size, 0,
                                         &apvvp->tile_num,
                                         (const uint32_t **)&apvvp->frame_data);
        if (err < 0)
            return err;

        AV_WN32(frame_data->mapped_mem + (2*(apvvp->tile_num - 1) + 0)*sizeof(uint32_t),
                apvvp->frame_data[apvvp->tile_num - 1]);
        AV_WN32(frame_data->mapped_mem + (2*(apvvp->tile_num - 1) + 1)*sizeof(uint32_t),
                size);
    }

    return 0;
}

static int vk_apv_end_frame(AVCodecContext *avctx)
{
    int err;
    APVDecodeContext *apv = avctx->priv_data;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx = dec->shared_ctx;
    APVVulkanDecodeContext *apvvk = ctx->sd_ctx;
    FFVulkanFunctions *vk = &ctx->s.vkfn;

    APVVulkanDecodePicture *apvvp = apv->hwaccel_picture_private;
    FFVulkanDecodePicture *vp = &apvvp->vp;

    FFVkBuffer *slices_buf = (FFVkBuffer *)vp->slices_buf->data;
    FFVkBuffer *frame_data_buf = (FFVkBuffer *)apvvp->frame_data_buf->data;

    AVHWFramesContext *hwfc = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
    enum AVPixelFormat sw_format = hwfc->sw_format;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(sw_format);

    VkImageMemoryBarrier2 img_bar[8];
    int nb_img_bar = 0;

    FFVkExecContext *exec = ff_vk_exec_get(&ctx->s, &ctx->exec_pool);
    ff_vk_exec_start(&ctx->s, exec);

    /* Make sure the buffer is flushed */
    RET(ff_vk_flush_buffer(&ctx->s, frame_data_buf, 0, frame_data_buf->size, 1));

    /* Prepare deps */
    RET(ff_vk_exec_add_dep_frame(&ctx->s, exec, apv->output_frame,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));

    err = ff_vk_exec_mirror_sem_value(&ctx->s, exec, &vp->sem, &vp->sem_value,
                                      apv->output_frame);
    if (err < 0)
        return err;

    RET(ff_vk_exec_add_dep_buf(&ctx->s, exec, &vp->slices_buf, 1, 0));
    vp->slices_buf = NULL;
    RET(ff_vk_exec_add_dep_buf(&ctx->s, exec, &apvvp->frame_data_buf, 1, 0));
    apvvp->frame_data_buf = NULL;

    ff_vk_frame_barrier(&ctx->s, exec, apv->output_frame, img_bar, &nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_ACCESS_2_TRANSFER_WRITE_BIT,
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pImageMemoryBarriers = img_bar,
        .imageMemoryBarrierCount = nb_img_bar,
    });
    nb_img_bar = 0;

    /* Setup push data */
    DecodePushData pd = (DecodePushData) {
        .tile_data = slices_buf->address,
        .tile_count = { apv->tile_info.tile_cols, apv->tile_info.tile_rows },
        .log2_chroma_sub = { desc->log2_chroma_w, desc->log2_chroma_h },
        .components = desc->nb_components,
    };

    /* Decoding */
    ff_vk_shader_update_img_array(&ctx->s, exec, &apvvk->decode,
                                  apv->output_frame, vp->view.out,
                                  0, 0,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_NULL_HANDLE);
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, &apvvk->decode,
                                    0, 1, 0,
                                    frame_data_buf,
                                    0, frame_data_buf->size,
                                    VK_FORMAT_UNDEFINED);

    ff_vk_exec_bind_shader(&ctx->s, exec, &apvvk->decode);
    ff_vk_shader_update_push_const(&ctx->s, exec, &apvvk->decode,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pd), &pd);

    vk->CmdDispatch(exec->buf,
                    apv->tile_info.tile_cols, apv->tile_info.tile_rows,
                    desc->nb_components);

    /* iDCT */
    ff_vk_shader_update_img_array(&ctx->s, exec, &apvvk->idct,
                                  apv->output_frame, vp->view.out,
                                  0, 0,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_NULL_HANDLE);
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, &apvvk->idct,
                                    0, 1, 0,
                                    frame_data_buf,
                                    0, frame_data_buf->size,
                                    VK_FORMAT_UNDEFINED);

    ff_vk_exec_bind_shader(&ctx->s, exec, &apvvk->idct);
    ff_vk_shader_update_push_const(&ctx->s, exec, &apvvk->idct,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pd), &pd);

    vk->CmdDispatch(exec->buf,
                    apv->tile_info.tile_cols, apv->tile_info.tile_rows,
                    desc->nb_components);

    err = ff_vk_exec_submit(&ctx->s, exec);
    if (err < 0)
        return err;

fail:
    return 0;
}

static void add_push_data(FFVulkanShader *shd)
{
    av_bprintf(&shd->src, "#define APV_MAX_NUM_COMP %i\n"         ,APV_MAX_NUM_COMP);
    av_bprintf(&shd->src, "#define APV_MAX_TILE_COUNT %i\n"     ,APV_MAX_TILE_COUNT);
    av_bprintf(&shd->src, "#define APV_MAX_TILE_COLS %i\n"       ,APV_MAX_TILE_COLS);
    av_bprintf(&shd->src, "#define APV_MAX_TILE_ROWS %i\n"       ,APV_MAX_TILE_ROWS);
    av_bprintf(&shd->src, "#define APV_MIN_TRANS_COEFF %i\n"   ,APV_MIN_TRANS_COEFF);
    av_bprintf(&shd->src, "#define APV_MAX_TRANS_COEFF %i\n"   ,APV_MAX_TRANS_COEFF);
    av_bprintf(&shd->src, "#define APV_VLC_LUT_BITS %i\n"         ,APV_VLC_LUT_BITS);
    av_bprintf(&shd->src, "#define APV_VLC_LUT_SIZE %i\n"         ,APV_VLC_LUT_SIZE);
    av_bprintf(&shd->src, "#define APV_TR_SIZE %i\n"                   ,APV_TR_SIZE);
    av_bprintf(&shd->src, "#define APV_BLK_COEFFS %i\n"             ,APV_BLK_COEFFS);
    av_bprintf(&shd->src, "#define APV_MB_SIZE ivec2(%i, %i)\n",
               APV_MB_WIDTH, APV_MB_HEIGHT);
    GLSLC(0,                                                                       );
    GLSLC(0, layout(push_constant, scalar) uniform pushConstants {                 );
    GLSLC(1,     u8buf tile_data;                                                  );
    GLSLC(1,     ivec2 tile_count;                                                 );
    GLSLC(1,     ivec2 log2_chroma_sub;                                            );
    GLSLC(1,     int components;                                                   );
    GLSLC(0, };                                                                    );
    GLSLC(0,                                                                       );
    ff_vk_shader_add_push_const(shd, 0, sizeof(DecodePushData),
                                VK_SHADER_STAGE_COMPUTE_BIT);
}

static int init_decode_shader(AVCodecContext *avctx, FFVulkanContext *s,
                              FFVkExecPool *pool, FFVkSPIRVCompiler *spv,
                              FFVulkanShader *shd)
{
    int err;
    FFVulkanDescriptorSetBinding *desc_set;
    AVHWFramesContext *dec_frames_ctx;
    dec_frames_ctx = (AVHWFramesContext *)avctx->hw_frames_ctx->data;

    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;

    RET(ff_vk_shader_init(s, shd, "apv_decode",
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          (const char *[]) { "GL_EXT_buffer_reference",
                                             "GL_EXT_buffer_reference2" }, 2,
                          1, 1, 1,
                          0));

    /* Common codec header */
    GLSLD(ff_source_common_comp);

    add_push_data(shd);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name       = "dst",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .dimensions = 2,
            .mem_layout = ff_vk_shader_rep_fmt(dec_frames_ctx->sw_format,
                                               FF_VK_REP_NATIVE),
            .mem_quali  = "writeonly",
            .elems      = av_pix_fmt_count_planes(dec_frames_ctx->sw_format),
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .name        = "frame_data_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .mem_quali   = "readonly",
            .buf_content =
            "uvec2 tile_offset[APV_MAX_NUM_COMP][APV_MAX_TILE_ROWS][APV_MAX_TILE_COLS];\n"
        "    uint8_t q_matrix[APV_MAX_NUM_COMP][8][8];\n"
        "    uint8_t tile_qp[APV_MAX_NUM_COMP][APV_MAX_TILE_ROWS][APV_MAX_TILE_COLS];\n"
        "    uint16_t tile_col[APV_MAX_TILE_COLS + 1];\n"
        "    uint16_t tile_row[APV_MAX_TILE_ROWS + 1];"
        }
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 2, 0, 0));

    GLSLC(0, struct SingleCLUTEntry {                                              );
    GLSLC(1,     uint16_t result;                                                  );
    GLSLC(1,     uint8_t consume;                                                  );
    GLSLC(1,     uint8_t more;                                                     );
    GLSLC(0, };                                                                    );
    GLSLC(0,                                                                       );
    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "vlc_tab_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .mem_quali   = "readonly",
            .buf_content = "SingleCLUTEntry single_lut[6][APV_VLC_LUT_SIZE];",
        },
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 1, 1, 0));

    GLSLD(ff_source_apv_decode_comp);

    RET(spv->compile_shader(s, spv, shd, &spv_data, &spv_len, "main",
                            &spv_opaque));
    RET(ff_vk_shader_link(s, shd, spv_data, spv_len, "main"));

    RET(ff_vk_shader_register_exec(s, pool, shd));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int init_idct_shader(AVCodecContext *avctx, FFVulkanContext *s,
                            FFVkExecPool *pool, FFVkSPIRVCompiler *spv,
                            FFVulkanShader *shd)
{
    int err;
    FFVulkanDescriptorSetBinding *desc_set;
    AVHWFramesContext *dec_frames_ctx;
    dec_frames_ctx = (AVHWFramesContext *)avctx->hw_frames_ctx->data;

    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;

    RET(ff_vk_shader_init(s, shd, "apv_idct",
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          (const char *[]) { "GL_EXT_buffer_reference",
                                             "GL_EXT_buffer_reference2" }, 2,
                          1, 1, 1,
                          0));

    /* Common codec header */
    GLSLD(ff_source_common_comp);

    add_push_data(shd);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name       = "dst",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .dimensions = 2,
            .mem_layout = ff_vk_shader_rep_fmt(dec_frames_ctx->sw_format,
                                               FF_VK_REP_NATIVE),
            .elems      = av_pix_fmt_count_planes(dec_frames_ctx->sw_format),
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .name        = "frame_data_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .mem_quali   = "readonly",
            .buf_content =
            "uvec2 tile_offset[APV_MAX_NUM_COMP][APV_MAX_TILE_ROWS][APV_MAX_TILE_COLS];\n"
        "    uint8_t q_matrix[APV_MAX_NUM_COMP][8][8];\n"
        "    uint8_t tile_qp[APV_MAX_NUM_COMP][APV_MAX_TILE_ROWS][APV_MAX_TILE_COLS];\n"
        "    uint16_t tile_col[APV_MAX_TILE_COLS + 1];\n"
        "    uint16_t tile_row[APV_MAX_TILE_ROWS + 1];"
        },
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 2, 0, 0));

    GLSLD(ff_source_apv_idct_comp);

    RET(spv->compile_shader(s, spv, shd, &spv_data, &spv_len, "main",
                            &spv_opaque));
    RET(ff_vk_shader_link(s, shd, spv_data, spv_len, "main"));

    RET(ff_vk_shader_register_exec(s, pool, shd));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static void vk_decode_apv_uninit(FFVulkanDecodeShared *ctx)
{
    APVVulkanDecodeContext *apvvk = ctx->sd_ctx;

    ff_vk_shader_free(&ctx->s, &apvvk->decode);
    ff_vk_shader_free(&ctx->s, &apvvk->idct);

    av_buffer_pool_uninit(&apvvk->frame_data_pool);

    ff_vk_free_buf(&ctx->s, &apvvk->clut_buf);

    av_freep(&apvvk);
}

static int vk_decode_apv_init(AVCodecContext *avctx)
{
    int err;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;

    FFVkSPIRVCompiler *spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(avctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    err = ff_vk_decode_init(avctx);
    if (err < 0)
        return err;

    FFVulkanDecodeShared *ctx = dec->shared_ctx;
    APVVulkanDecodeContext *apvvk = ctx->sd_ctx = av_mallocz(sizeof(*apvvk));
    if (!apvvk) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->sd_ctx_free = &vk_decode_apv_uninit;

    RET(init_decode_shader(avctx, &ctx->s, &ctx->exec_pool,
                           spv, &apvvk->decode));

    RET(init_idct_shader(avctx, &ctx->s, &ctx->exec_pool,
                         spv, &apvvk->idct));

    /* CLUT for decoding */
    size_t buf_size = 6*APV_VLC_LUT_SIZE*4;
    APVSingleVLCLUTEntry *buf_mapped;
    RET(ff_vk_create_buf(&ctx->s, &apvvk->clut_buf,
                         buf_size,
                         NULL, NULL,
                         VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                         VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT));
    RET(ff_vk_map_buffer(&ctx->s, &apvvk->clut_buf, (void *)&buf_mapped, 0));

    memcpy(buf_mapped, &ff_apv_decode_lut.single_lut,
           sizeof(ff_apv_decode_lut.single_lut));

    RET(ff_vk_unmap_buffer(&ctx->s, &apvvk->clut_buf, 1));

    ff_vk_shader_update_desc_buffer(&ctx->s, &ctx->exec_pool.contexts[0],
                                    &apvvk->decode,
                                    1, 0, 0,
                                    &apvvk->clut_buf,
                                    0, buf_size,
                                    VK_FORMAT_UNDEFINED);

fail:
    spv->uninit(&spv);

    return err;
}

static void vk_apv_free_frame_priv(AVRefStructOpaque _hwctx, void *data)
{
    AVHWDeviceContext *dev_ctx = _hwctx.nc;

    APVVulkanDecodePicture *apvvp = data;
    FFVulkanDecodePicture *vp = &apvvp->vp;

    ff_vk_decode_free_frame(dev_ctx, vp);

    av_buffer_unref(&vp->slices_buf);
    av_buffer_unref(&apvvp->frame_data_buf);
}

const FFHWAccel ff_apv_vulkan_hwaccel = {
    .p.name                = "apv_vulkan",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_APV,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    .start_frame           = &vk_apv_start_frame,
    .decode_slice          = &vk_apv_decode_slice,
    .end_frame             = &vk_apv_end_frame,
    .free_frame_priv       = &vk_apv_free_frame_priv,
    .frame_priv_data_size  = sizeof(APVVulkanDecodePicture),
    .init                  = &vk_decode_apv_init,
    .update_thread_context = &ff_vk_update_thread_context,
    .decode_params         = &ff_vk_params_invalidate,
    .flush                 = &ff_vk_decode_flush,
    .uninit                = &ff_vk_decode_uninit,
    .frame_params          = &ff_vk_frame_params,
    .priv_data_size        = sizeof(FFVulkanDecodeContext),
    .caps_internal         = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
