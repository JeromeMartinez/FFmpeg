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

#include "jpeg2000dec.h"
#include "libavutil/vulkan_spirv.h"
#include "libavutil/mem.h"

#define RGB_LINECACHE 2

extern const char *ff_source_common_comp;

const FFVulkanDecodeDescriptor ff_vk_dec_jpeg2000_desc = {
    .codec_id         = AV_CODEC_ID_JPEG2000,
    .decode_extension = FF_VK_EXT_PUSH_DESCRIPTOR,
    .queue_flags      = VK_QUEUE_COMPUTE_BIT,
};

typedef struct JPEG2000VulkanDecodePicture {
    FFVulkanDecodePicture vp;

    AVBufferRef *slice_state;
    uint32_t plane_state_size;
    uint32_t slice_state_size;
    uint32_t slice_data_size;

    AVBufferRef *slice_offset_buf;
    uint32_t    *slice_offset;
    int          slice_num;

    AVBufferRef *slice_status_buf;
    int crc_checked;
} JPEG2000VulkanDecodePicture;

typedef struct JPEG2000VulkanDecodeContext {
    AVBufferRef *intermediate_frames_ref[2]; /* 16/32 bit */

    FFVulkanShader setup;
    FFVulkanShader reset[2]; /* AC/Golomb */
    FFVulkanShader decode[2][2][2]; /* 16/32 bit, AC/Golomb, Normal/RGB */

    FFVkBuffer rangecoder_static_buf;
    FFVkBuffer quant_buf;
    FFVkBuffer crc_tab_buf;

    AVBufferPool *slice_state_pool;
    AVBufferPool *slice_offset_pool;
    AVBufferPool *slice_status_pool;
} JPEG2000VulkanDecodeContext;

typedef struct JPEG2000VkParameters {
} JPEG2000VkParameters;

static void add_push_data(FFVulkanShader *shd)
{
    GLSLC(0, layout(push_constant, scalar) uniform pushConstants {  );
    GLSLC(0, };                                                     );
    ff_vk_shader_add_push_const(shd, 0, sizeof(JPEG2000VkParameters),
                                VK_SHADER_STAGE_COMPUTE_BIT);
}

static int vk_jpeg2000_start_frame(AVCodecContext          *avctx,
                                   const AVBufferRef       *buffer_ref,
                                   av_unused const uint8_t *buffer,
                                   av_unused uint32_t       size)
{
    #if 0
    int err;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx = dec->shared_ctx;
    JPEG2000VulkanDecodeContext *jv = ctx->sd_ctx;
    Jpeg2000DecoderContext *j = avctx->priv_data;

    JPEG2000VulkanDecodePicture *jp = j->hwaccel_picture_private;
    FFVulkanDecodePicture *vp = &jp->vp;

    AVHWFramesContext *hwfc = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
    enum AVPixelFormat sw_format = hwfc->sw_format;

    int max_contexts;
    int is_rgb = 0;

    jp->slice_num = 0;

    max_contexts = 0;
    for (int i = 0; i < j->quant_table_count; i++)
        max_contexts = FFMAX(j->context_count[i], max_contexts);

    /* Allocate slice buffer data */
    if (j->ac == AC_GOLOMB_RICE)
        jp->plane_state_size = 8;
    else
        jp->plane_state_size = CONTEXT_SIZE;

    jp->plane_state_size *= max_contexts;
    jp->slice_state_size = jp->plane_state_size*j->plane_count;

    jp->slice_data_size = 256; /* Overestimation for the SliceContext struct */
    jp->slice_state_size += jp->slice_data_size;
    jp->slice_state_size = FFALIGN(jp->slice_state_size, 8);

    jp->crc_checked = j->ec && (avctx->err_recognition & AV_EF_CRCCHECK);

    /* Host map the input slices data if supported */
    if (ctx->s.extensions & FF_VK_EXT_EXTERNAL_HOST_MEMORY)
        ff_vk_host_map_buffer(&ctx->s, &vp->slices_buf, buffer_ref->data,
                              buffer_ref,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);

    /* Allocate slice state data */
    if (j->picture.f->flags & AV_FRAME_FLAG_KEY) {
        err = ff_vk_get_pooled_buffer(&ctx->s, &jv->slice_state_pool,
                                      &jp->slice_state,
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                      NULL, j->slice_count*jp->slice_state_size,
                                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (err < 0)
            return err;
    } else {
        JPEG2000VulkanDecodePicture *jpl = j->hwaccel_last_picture_private;
        jp->slice_state = av_buffer_ref(fpl->slice_state);
        if (!jp->slice_state)
            return AVERROR(ENOMEM);
    }

    /* Allocate slice offsets buffer */
    err = ff_vk_get_pooled_buffer(&ctx->s, &jv->slice_offset_pool,
                                  &jp->slice_offset_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, 2*j->slice_count*sizeof(uint32_t),
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (err < 0)
        return err;

    /* Allocate slice status buffer */
    err = ff_vk_get_pooled_buffer(&ctx->s, &jv->slice_status_pool,
                                  &jp->slice_status_buf,
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                  VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                  NULL, 2*j->slice_count*sizeof(uint32_t),
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (err < 0)
        return err;

    /* Prepare frame to be used */
    err = ff_vk_decode_prepare_frame_sdr(dec, j->picture.f, vp, 1,
                                         FF_VK_REP_NATIVE, 0);
    if (err < 0)
        return err;

    /* Create a temporaty frame for RGB */
    if (is_rgb) {
        vp->dpb_frame = av_frame_alloc();
        if (!vp->dpb_frame)
            return AVERROR(ENOMEM);

        err = av_hwframe_get_buffer(jv->intermediate_frames_ref[j->use32bit],
                                    vp->dpb_frame, 0);
        if (err < 0)
            return err;
    }
    #endif
    return 0;
}

static int vk_jpeg2000_decode_slice(AVCodecContext *avctx,
                                    const uint8_t  *data,
                                    uint32_t        size)
{
    Jpeg2000DecoderContext *j = avctx->priv_data;

    JPEG2000VulkanDecodePicture *jp = j->hwaccel_picture_private;
    FFVulkanDecodePicture *vp = &jp->vp;

    FFVkBuffer *slice_offset = (FFVkBuffer *)jp->slice_offset_buf->data;
    FFVkBuffer *slices_buf = vp->slices_buf ? (FFVkBuffer *)vp->slices_buf->data : NULL;

    if (slices_buf && slices_buf->host_ref) {
        AV_WN32(slice_offset->mapped_mem + (2*jp->slice_num + 0)*sizeof(uint32_t),
                data - slices_buf->mapped_mem);
        AV_WN32(slice_offset->mapped_mem + (2*jp->slice_num + 1)*sizeof(uint32_t),
                size);

        jp->slice_num++;
    } else {
        int err = ff_vk_decode_add_slice(avctx, vp, data, size, 0,
                                         &jp->slice_num,
                                         (const uint32_t **)&jp->slice_offset);
        if (err < 0)
            return err;

        AV_WN32(slice_offset->mapped_mem + (2*(jp->slice_num - 1) + 0)*sizeof(uint32_t),
                jp->slice_offset[jp->slice_num - 1]);
        AV_WN32(slice_offset->mapped_mem + (2*(jp->slice_num - 1) + 1)*sizeof(uint32_t),
                size);
    }

    return 0;
}

static int vk_jpeg2000_end_frame(AVCodecContext *avctx)
{
    #if 0
    int err;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx = dec->shared_ctx;
    FFVulkanFunctions *vk = &ctx->s.vkfn;

    Jpeg2000DecoderContext *j = avctx->priv_data;
    JPEG2000VulkanDecodeContext *jv = ctx->sd_ctx;
    FFv1VkParameters pd;
    FFv1VkResetParameters pd_reset;

    AVHWFramesContext *hwfc = (AVHWFramesContext *)avctx->hw_frames_ctx->data;
    enum AVPixelFormat sw_format = hwfc->sw_format;

    int bits = j->avctx->bits_per_raw_sample > 0 ? j->avctx->bits_per_raw_sample : 8;
    int is_rgb = !(j->colorspace == 0 && sw_format != AV_PIX_FMT_YA8) &&
                 !(sw_format == AV_PIX_FMT_YA8);
    int color_planes = av_pix_fmt_desc_get(avctx->sw_pix_fmt)->nb_components;

    FFVulkanShader *reset_shader;
    FFVulkanShader *decode_shader;

    JPEG2000VulkanDecodePicture *jp = j->hwaccel_picture_private;
    FFVulkanDecodePicture *vp = &jp->vp;

    FFVkBuffer *slices_buf = (FFVkBuffer *)vp->slices_buf->data;
    FFVkBuffer *slice_state = (FFVkBuffer *)jp->slice_state->data;
    FFVkBuffer *slice_offset = (FFVkBuffer *)jp->slice_offset_buf->data;
    FFVkBuffer *slice_status = (FFVkBuffer *)jp->slice_status_buf->data;

    VkImageView rct_image_views[AV_NUM_DATA_POINTERS];

    AVFrame *decode_dst = is_rgb ? vp->dpb_frame : j->picture.f;
    VkImageView *decode_dst_view = is_rgb ? rct_image_views : vp->view.out;

    VkImageMemoryBarrier2 img_bar[37];
    int nb_img_bar = 0;
    VkBufferMemoryBarrier2 buf_bar[8];
    int nb_buf_bar = 0;

    FFVkExecContext *exec = ff_vk_exec_get(&ctx->s, &ctx->exec_pool);
    ff_vk_exec_start(&ctx->s, exec);

    /* Prepare deps */
    RET(ff_vk_exec_add_dep_frame(&ctx->s, exec, j->picture.f,
                                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT));

    err = ff_vk_exec_mirror_sem_value(&ctx->s, exec, &vp->sem, &vp->sem_value,
                                      j->picture.f);
    if (err < 0)
        return err;

    if (is_rgb) {
        RET(ff_vk_create_imageviews(&ctx->s, exec, rct_image_views,
                                    vp->dpb_frame, FF_VK_REP_NATIVE));
        RET(ff_vk_exec_add_dep_frame(&ctx->s, exec, vp->dpb_frame,
                                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_2_CLEAR_BIT));
        ff_vk_frame_barrier(&ctx->s, exec, decode_dst, img_bar, &nb_img_bar,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_ACCESS_2_TRANSFER_WRITE_BIT,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_QUEUE_FAMILY_IGNORED);
    }

    if (!(j->picture.f->flags & AV_FRAME_FLAG_KEY)) {
        JPEG2000VulkanDecodePicture *jpl = j->hwaccel_last_picture_private;
        FFVulkanDecodePicture *vpl = &fpl->vp;

        /* Wait on the previous frame */
        RET(ff_vk_exec_add_dep_wait_sem(&ctx->s, exec, vpl->sem, vpl->sem_value,
                                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT));
    }

    RET(ff_vk_exec_add_dep_buf(&ctx->s, exec, &jp->slice_state, 1, 1));
    RET(ff_vk_exec_add_dep_buf(&ctx->s, exec, &jp->slice_status_buf, 1, 1));
    RET(ff_vk_exec_add_dep_buf(&ctx->s, exec, &vp->slices_buf, 1, 0));
    vp->slices_buf = NULL;
    RET(ff_vk_exec_add_dep_buf(&ctx->s, exec, &jp->slice_offset_buf, 1, 0));
    jp->slice_offset_buf = NULL;

    /* Entry barrier for the slice state */
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = slice_state->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = slice_state->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = slice_state->buf,
        .offset = 0,
        .size = jp->slice_data_size*j->slice_count,
    };

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pImageMemoryBarriers = img_bar,
        .imageMemoryBarrierCount = nb_img_bar,
        .pBufferMemoryBarriers = buf_bar,
        .bufferMemoryBarrierCount = nb_buf_bar,
    });
    slice_state->stage = buf_bar[0].dstStageMask;
    slice_state->access = buf_bar[0].dstAccessMask;
    nb_buf_bar = 0;
    nb_img_bar = 0;

    /* Setup shader */
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, &jv->setup,
                                    1, 0, 0,
                                    slice_state,
                                    0, jp->slice_data_size*j->slice_count,
                                    VK_FORMAT_UNDEFINED);
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, &jv->setup,
                                    1, 1, 0,
                                    slice_offset,
                                    0, 2*j->slice_count*sizeof(uint32_t),
                                    VK_FORMAT_UNDEFINED);
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, &jv->setup,
                                    1, 2, 0,
                                    slice_status,
                                    0, 2*j->slice_count*sizeof(uint32_t),
                                    VK_FORMAT_UNDEFINED);

    ff_vk_exec_bind_shader(&ctx->s, exec, &jv->setup);
    pd = (FFv1VkParameters) {
        .slice_data = slices_buf->address,
        .slice_state  = slice_state->address + j->slice_count*jp->slice_data_size,

        .img_size[0] = j->picture.f->width,
        .img_size[1] = j->picture.f->height,
        .chroma_shift[0] = j->chroma_h_shift,
        .chroma_shift[1] = j->chroma_v_shift,

        .plane_state_size = jp->plane_state_size,
        .crcref = j->crcref,
        .rct_offset = 1 << bits,

        .bits_per_raw_sample = bits,
        .quant_table_count = j->quant_table_count,
        .version = j->version,
        .micro_version = j->micro_version,
        .key_frame = j->picture.f->flags & AV_FRAME_FLAG_KEY,
        .planes = av_pix_fmt_count_planes(sw_format),
        .codec_planes = j->plane_count,
        .color_planes = color_planes,
        .transparency = j->transparency,
        .planar_rgb = ff_vk_mt_is_np_rgb(sw_format) &&
                      (ff_vk_count_images((AVVkFrame *)j->picture.f->data[0]) > 1),
        .colorspace = j->colorspace,
        .ec = j->ec,
        .golomb = j->ac == AC_GOLOMB_RICE,
        .check_crc = !!(avctx->err_recognition & AV_EF_CRCCHECK),
    };
    for (int i = 0; i < j->quant_table_count; i++)
        pd.extend_lookup[i] = (j->quant_tables[i][3][127] != 0) ||
                              (j->quant_tables[i][4][127] != 0);


    /* For some reason the C FFv1 encoder/decoder treats these differently */
    if (sw_format == AV_PIX_FMT_GBRP10 || sw_format == AV_PIX_FMT_GBRP12 ||
        sw_format == AV_PIX_FMT_GBRP14)
        memcpy(pd.fmt_lut, (int [4]) { 2, 1, 0, 3 }, 4*sizeof(int));
    else if (sw_format == AV_PIX_FMT_X2BGR10)
        memcpy(pd.fmt_lut, (int [4]) { 0, 2, 1, 3 }, 4*sizeof(int));
    else
        ff_vk_set_perm(sw_format, pd.fmt_lut, 0);

    ff_vk_shader_update_push_const(&ctx->s, exec, &jv->setup,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pd), &pd);

    vk->CmdDispatch(exec->buf, j->num_h_slices, j->num_v_slices, 1);

    if (is_rgb) {
        AVVkFrame *vkf = (AVVkFrame *)vp->dpb_frame->data[0];
        for (int i = 0; i < color_planes; i++)
            vk->CmdClearColorImage(exec->buf, vkf->img[i], VK_IMAGE_LAYOUT_GENERAL,
                                   &((VkClearColorValue) { 0 }),
                                   1, &((VkImageSubresourceRange) {
                                       .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                                       .levelCount = 1,
                                       .layerCount = 1,
                                   }));
    }

    /* Reset shader */
    reset_shader = &jv->reset[j->ac == AC_GOLOMB_RICE];
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, reset_shader,
                                    1, 0, 0,
                                    slice_state,
                                    0, jp->slice_data_size*j->slice_count,
                                    VK_FORMAT_UNDEFINED);

    ff_vk_exec_bind_shader(&ctx->s, exec, reset_shader);

    pd_reset = (FFv1VkResetParameters) {
        .slice_state = slice_state->address + j->slice_count*jp->slice_data_size,
        .plane_state_size = jp->plane_state_size,
        .codec_planes = j->plane_count,
        .key_frame = j->picture.j->flags & AV_FRAME_FLAG_KEY,
        .version = j->version,
        .micro_version = j->micro_version,
    };
    for (int i = 0; i < j->quant_table_count; i++)
        pd_reset.context_count[i] = j->context_count[i];

    ff_vk_shader_update_push_const(&ctx->s, exec, reset_shader,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pd_reset), &pd_reset);

    /* Sync between setup and reset shaders */
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = slice_state->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = slice_state->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = slice_state->buf,
        .offset = 0,
        .size = jp->slice_data_size*j->slice_count,
    };
    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pImageMemoryBarriers = img_bar,
        .imageMemoryBarrierCount = nb_img_bar,
        .pBufferMemoryBarriers = buf_bar,
        .bufferMemoryBarrierCount = nb_buf_bar,
    });
    slice_state->stage = buf_bar[0].dstStageMask;
    slice_state->access = buf_bar[0].dstAccessMask;
    nb_buf_bar = 0;
    nb_img_bar = 0;

    vk->CmdDispatch(exec->buf, j->num_h_slices, j->num_v_slices,
                    j->plane_count);

    /* Decode */
    decode_shader = &jv->decode[j->use32bit][j->ac == AC_GOLOMB_RICE][is_rgb];
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, decode_shader,
                                    1, 0, 0,
                                    slice_state,
                                    0, jp->slice_data_size*j->slice_count,
                                    VK_FORMAT_UNDEFINED);
    ff_vk_shader_update_img_array(&ctx->s, exec, decode_shader,
                                  decode_dst, decode_dst_view,
                                  1, 1,
                                  VK_IMAGE_LAYOUT_GENERAL,
                                  VK_NULL_HANDLE);
    ff_vk_shader_update_desc_buffer(&ctx->s, exec, decode_shader,
                                    1, 2, 0,
                                    slice_status,
                                    0, 2*j->slice_count*sizeof(uint32_t),
                                    VK_FORMAT_UNDEFINED);
    if (is_rgb)
        ff_vk_shader_update_img_array(&ctx->s, exec, decode_shader,
                                      j->picture.f, vp->view.out,
                                      1, 3,
                                      VK_IMAGE_LAYOUT_GENERAL,
                                      VK_NULL_HANDLE);

    ff_vk_exec_bind_shader(&ctx->s, exec, decode_shader);
    ff_vk_shader_update_push_const(&ctx->s, exec, decode_shader,
                                   VK_SHADER_STAGE_COMPUTE_BIT,
                                   0, sizeof(pd), &pd);

    /* Sync between reset and decode shaders */
    buf_bar[nb_buf_bar++] = (VkBufferMemoryBarrier2) {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask = slice_state->stage,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = slice_state->access,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                         VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = slice_state->buf,
        .offset = jp->slice_data_size*j->slice_count,
        .size = j->slice_count*(jp->slice_state_size - jp->slice_data_size),
    };

    /* Input frame barrier */
    ff_vk_frame_barrier(&ctx->s, exec, j->picture.f, img_bar, &nb_img_bar,
                        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                        VK_ACCESS_SHADER_WRITE_BIT |
                        (!is_rgb ? VK_ACCESS_SHADER_READ_BIT : 0),
                        VK_IMAGE_LAYOUT_GENERAL,
                        VK_QUEUE_FAMILY_IGNORED);
    if (is_rgb)
        ff_vk_frame_barrier(&ctx->s, exec, vp->dpb_frame, img_bar, &nb_img_bar,
                            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                            VK_IMAGE_LAYOUT_GENERAL,
                            VK_QUEUE_FAMILY_IGNORED);

    vk->CmdPipelineBarrier2(exec->buf, &(VkDependencyInfo) {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pImageMemoryBarriers = img_bar,
        .imageMemoryBarrierCount = nb_img_bar,
        .pBufferMemoryBarriers = buf_bar,
        .bufferMemoryBarrierCount = nb_buf_bar,
    });
    slice_state->stage = buf_bar[0].dstStageMask;
    slice_state->access = buf_bar[0].dstAccessMask;
    nb_img_bar = 0;
    nb_buf_bar = 0;

    vk->CmdDispatch(exec->buf, j->num_h_slices, j->num_v_slices, 1);

    err = ff_vk_exec_submit(&ctx->s, exec);
    if (err < 0)
        return err;

    /* We don't need the temporary frame after decoding */
    av_frame_free(&vp->dpb_frame);
    #endif

fail:
    return 0;
}

static void define_shared_code(FFVulkanShader *shd)
{
    av_bprintf(&shd->src, "#define RGB_LINECACHE %i\n"                   ,RGB_LINECACHE);

//    GLSLD(ff_source_rangecoder_comp);
//    GLSLD(ff_source_ffv1_common_comp);
}

#if 0
static int init_setup_shader(Jpeg2000DecoderContext *f, FFVulkanContext *s,
                             FFVkExecPool *pool, FFVkSPIRVCompiler *spv,
                             FFVulkanShader *shd)
{
    int err;
    FFVulkanDescriptorSetBinding *desc_set;

    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;

    RET(ff_vk_shader_init(s, shd, "ffv1_dec_setup",
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          (const char *[]) { "GL_EXT_buffer_reference",
                                             "GL_EXT_buffer_reference2" }, 2,
                          1, 1, 1,
                          0));

    /* Common codec header */
    GLSLD(ff_source_common_comp);

    add_push_data(shd);

    av_bprintf(&shd->src, "#define MAX_QUANT_TABLES %i\n", MAX_QUANT_TABLES);
    av_bprintf(&shd->src, "#define MAX_CONTEXT_INPUTS %i\n", MAX_CONTEXT_INPUTS);
    av_bprintf(&shd->src, "#define MAX_QUANT_TABLE_SIZE %i\n", MAX_QUANT_TABLE_SIZE);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "rangecoder_static_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "uint8_t zero_one_state[512];",
        },
        {
            .name        = "crc_ieee_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "uint32_t crc_ieee[256];",
        },
        {
            .name        = "quant_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "int16_t quant_table[MAX_QUANT_TABLES]"
                           "[MAX_CONTEXT_INPUTS][MAX_QUANT_TABLE_SIZE];",
        },
    };

    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 3, 1, 0));

    define_shared_code(shd, 0 /* Irrelevant */);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "slice_data_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "SliceContext slice_ctx",
            .buf_elems   = j->max_slice_count,
        },
        {
            .name        = "slice_offsets_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_quali   = "readonly",
            .buf_content = "u32vec2 slice_offsets",
            .buf_elems   = 2*j->max_slice_count,
        },
        {
            .name        = "slice_status_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_quali   = "writeonly",
            .buf_content = "uint32_t slice_status",
            .buf_elems   = 2*j->max_slice_count,
        },
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 3, 0, 0));

    GLSLD(ff_source_ffv1_dec_setup_comp);

    RET(spv->compile_shader(s, spv, shd, &spv_data, &spv_len, "main",
                            &spv_opaque));
    RET(ff_vk_shader_link(s, shd, spv_data, spv_len, "main"));

    RET(ff_vk_shader_register_exec(s, pool, shd));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int init_reset_shader(Jpeg2000DecoderContext *f, FFVulkanContext *s,
                             FFVkExecPool *pool, FFVkSPIRVCompiler *spv,
                             FFVulkanShader *shd, int ac)
{
    int err;
    FFVulkanDescriptorSetBinding *desc_set;

    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    int wg_dim = FFMIN(s->props.properties.limits.maxComputeWorkGroupSize[0], 1024);

    RET(ff_vk_shader_init(s, shd, "ffv1_dec_reset",
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          (const char *[]) { "GL_EXT_buffer_reference",
                                             "GL_EXT_buffer_reference2" }, 2,
                          wg_dim, 1, 1,
                          0));

    if (ac == AC_GOLOMB_RICE)
        av_bprintf(&shd->src, "#define GOLOMB\n");

    /* Common codec header */
    GLSLD(ff_source_common_comp);

    GLSLC(0, layout(push_constant, scalar) uniform pushConstants {             );
    GLSLF(1,    uint context_count[%i];                                        ,MAX_QUANT_TABLES);
    GLSLC(1,    u8buf slice_state;                                             );
    GLSLC(1,    uint plane_state_size;                                         );
    GLSLC(1,    uint8_t codec_planes;                                          );
    GLSLC(1,    uint8_t key_frame;                                             );
    GLSLC(1,    uint8_t version;                                               );
    GLSLC(1,    uint8_t micro_version;                                         );
    GLSLC(1,    uint8_t padding[1];                                            );
    GLSLC(0, };                                                                );
    ff_vk_shader_add_push_const(shd, 0, sizeof(FFv1VkResetParameters),
                                VK_SHADER_STAGE_COMPUTE_BIT);

    av_bprintf(&shd->src, "#define MAX_QUANT_TABLES %i\n", MAX_QUANT_TABLES);
    av_bprintf(&shd->src, "#define MAX_CONTEXT_INPUTS %i\n", MAX_CONTEXT_INPUTS);
    av_bprintf(&shd->src, "#define MAX_QUANT_TABLE_SIZE %i\n", MAX_QUANT_TABLE_SIZE);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "rangecoder_static_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "uint8_t zero_one_state[512];",
        },
        {
            .name        = "quant_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "int16_t quant_table[MAX_QUANT_TABLES]"
                           "[MAX_CONTEXT_INPUTS][MAX_QUANT_TABLE_SIZE];",
        },
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 2, 1, 0));

    define_shared_code(shd, 0 /* Bit depth irrelevant for the reset shader */);
    if (ac == AC_GOLOMB_RICE)
        GLSLD(ff_source_ffv1_vlc_comp);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "slice_data_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .mem_quali   = "readonly",
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "SliceContext slice_ctx",
            .buf_elems   = j->max_slice_count,
        },
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 1, 0, 0));

    GLSLD(ff_source_ffv1_reset_comp);

    RET(spv->compile_shader(s, spv, shd, &spv_data, &spv_len, "main",
                            &spv_opaque));
    RET(ff_vk_shader_link(s, shd, spv_data, spv_len, "main"));

    RET(ff_vk_shader_register_exec(s, pool, shd));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int init_decode_shader(Jpeg2000DecoderContext *f, FFVulkanContext *s,
                              FFVkExecPool *pool, FFVkSPIRVCompiler *spv,
                              FFVulkanShader *shd,
                              AVHWFramesContext *dec_frames_ctx,
                              AVHWFramesContext *out_frames_ctx,
                              int use32bit, int ac, int rgb)
{
    int err;
    FFVulkanDescriptorSetBinding *desc_set;

    uint8_t *spv_data;
    size_t spv_len;
    void *spv_opaque = NULL;
    int use_cached_reader = ac != AC_GOLOMB_RICE &&
                            s->driver_props.driverID == VK_DRIVER_ID_MESA_RADV;

    RET(ff_vk_shader_init(s, shd, "ffv1_dec",
                          VK_SHADER_STAGE_COMPUTE_BIT,
                          (const char *[]) { "GL_EXT_buffer_reference",
                                             "GL_EXT_buffer_reference2" }, 2,
                          use_cached_reader ? CONTEXT_SIZE : 1, 1, 1,
                          0));

    if (ac == AC_GOLOMB_RICE)
        av_bprintf(&shd->src, "#define GOLOMB\n");

    if (rgb)
        av_bprintf(&shd->src, "#define RGB\n");

    if (use_cached_reader)
        av_bprintf(&shd->src, "#define CACHED_SYMBOL_READER 1\n");

    /* Common codec header */
    GLSLD(ff_source_common_comp);

    add_push_data(shd);

    av_bprintf(&shd->src, "#define MAX_QUANT_TABLES %i\n", MAX_QUANT_TABLES);
    av_bprintf(&shd->src, "#define MAX_CONTEXT_INPUTS %i\n", MAX_CONTEXT_INPUTS);
    av_bprintf(&shd->src, "#define MAX_QUANT_TABLE_SIZE %i\n", MAX_QUANT_TABLE_SIZE);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "rangecoder_static_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "uint8_t zero_one_state[512];",
        },
        {
            .name        = "quant_buf",
            .type        = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_layout  = "scalar",
            .buf_content = "int16_t quant_table[MAX_QUANT_TABLES]"
                           "[MAX_CONTEXT_INPUTS][MAX_QUANT_TABLE_SIZE];",
        },
    };

    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 2, 1, 0));

    define_shared_code(shd, use32bit);
    if (ac == AC_GOLOMB_RICE)
        GLSLD(ff_source_ffv1_vlc_comp);

    desc_set = (FFVulkanDescriptorSetBinding []) {
        {
            .name        = "slice_data_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .buf_content = "SliceContext slice_ctx",
            .buf_elems   = j->max_slice_count,
        },
        {
            .name       = "dec",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .dimensions = 2,
            .mem_layout = ff_vk_shader_rep_fmt(dec_frames_ctx->sw_format,
                                               FF_VK_REP_NATIVE),
            .elems      = av_pix_fmt_count_planes(dec_frames_ctx->sw_format),
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .name        = "slice_status_buf",
            .type        = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .stages      = VK_SHADER_STAGE_COMPUTE_BIT,
            .mem_quali   = "writeonly",
            .buf_content = "uint32_t slice_status",
            .buf_elems   = 2*j->max_slice_count,
        },
        {
            .name       = "dst",
            .type       = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .dimensions = 2,
            .mem_layout = ff_vk_shader_rep_fmt(out_frames_ctx->sw_format,
                                               FF_VK_REP_NATIVE),
            .mem_quali  = "writeonly",
            .elems      = av_pix_fmt_count_planes(out_frames_ctx->sw_format),
            .stages     = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    RET(ff_vk_shader_add_descriptor_set(s, shd, desc_set, 3 + rgb, 0, 0));

    GLSLD(ff_source_ffv1_dec_comp);

    RET(spv->compile_shader(s, spv, shd, &spv_data, &spv_len, "main",
                            &spv_opaque));
    RET(ff_vk_shader_link(s, shd, spv_data, spv_len, "main"));

    RET(ff_vk_shader_register_exec(s, pool, shd));

fail:
    if (spv_opaque)
        spv->free_shader(spv, &spv_opaque);

    return err;
}

static int init_indirect(AVCodecContext *avctx, FFVulkanContext *s,
                         AVBufferRef **dst, enum AVPixelFormat sw_format)
{
    int err;
    AVHWFramesContext *frames_ctx;
    AVVulkanFramesContext *vk_frames;
    Jpeg2000DecoderContext *j = avctx->priv_data;

    *dst = av_hwframe_ctx_alloc(s->device_ref);
    if (!(*dst))
        return AVERROR(ENOMEM);

    frames_ctx = (AVHWFramesContext *)((*dst)->data);
    frames_ctx->format    = AV_PIX_FMT_VULKAN;
    frames_ctx->sw_format = sw_format;
    frames_ctx->width     = s->frames->width;
    frames_ctx->height    = j->num_v_slices*RGB_LINECACHE;

    vk_frames = frames_ctx->hwctx;
    vk_frames->tiling    = VK_IMAGE_TILING_OPTIMAL;
    vk_frames->img_flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
    vk_frames->usage     = VK_IMAGE_USAGE_STORAGE_BIT |
                           VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    err = av_hwframe_ctx_init(*dst);
    if (err < 0) {
        av_log(avctx, AV_LOG_ERROR, "Unable to initialize frame pool with format %s: %s\n",
               av_get_pix_fmt_name(sw_format), av_err2str(err));
        av_buffer_unref(dst);
        return err;
    }

    return 0;
}
#endif

static void vk_decode_jpeg2000_uninit(FFVulkanDecodeShared *ctx)
{
    JPEG2000VulkanDecodeContext *jv = ctx->sd_ctx;

    #if 0
    ff_vk_shader_free(&ctx->s, &jv->setup);

    for (int i = 0; i < 2; i++) /* 16/32 bit */
        av_buffer_unref(&jv->intermediate_frames_ref[i]);

    for (int i = 0; i < 2; i++) /* AC/Golomb */
        ff_vk_shader_free(&ctx->s, &jv->reset[i]);

    for (int i = 0; i < 2; i++) /* 16/32 bit */
        for (int j = 0; j < 2; j++) /* AC/Golomb */
            for (int k = 0; k < 2; k++) /* Normal/RGB */
                ff_vk_shader_free(&ctx->s, &jv->decode[i][j][k]);

    ff_vk_free_buf(&ctx->s, &jv->quant_buf);
    ff_vk_free_buf(&ctx->s, &jv->rangecoder_static_buf);
    ff_vk_free_buf(&ctx->s, &jv->crc_tab_buf);

    av_buffer_pool_uninit(&jv->slice_state_pool);
    av_buffer_pool_uninit(&jv->slice_offset_pool);
    av_buffer_pool_uninit(&jv->slice_status_pool);
    #endif

    av_freep(&jv);
}

static int vk_decode_jpeg2000_init(AVCodecContext *avctx)
{
    int err;
    Jpeg2000DecoderContext *j = avctx->priv_data;
    FFVulkanDecodeContext *dec = avctx->internal->hwaccel_priv_data;
    FFVulkanDecodeShared *ctx = NULL;
    JPEG2000VulkanDecodeContext *jv;
    FFVkSPIRVCompiler *spv;

    /* TODO filter out unsupported here */

    spv = ff_vk_spirv_init();
    if (!spv) {
        av_log(avctx, AV_LOG_ERROR, "Unable to initialize SPIR-V compiler!\n");
        return AVERROR_EXTERNAL;
    }

    err = ff_vk_decode_init(avctx);
    if (err < 0)
        return err;
    ctx = dec->shared_ctx;

    jv = ctx->sd_ctx = av_mallocz(sizeof(*jv));
    if (!jv) {
        err = AVERROR(ENOMEM);
        goto fail;
    }

    ctx->sd_ctx_free = &vk_decode_jpeg2000_uninit;

    #if 0
    /* Intermediate frame pool for RCT */
    for (int i = 0; i < 2; i++) { /* 16/32 bit */
        RET(init_indirect(avctx, &ctx->s, &jv->intermediate_frames_ref[i],
                          i ? AV_PIX_FMT_GBRAP32 : AV_PIX_FMT_GBRAP16));
    }

    /* Setup shader */
    RET(init_setup_shader(f, &ctx->s, &ctx->exec_pool, spv, &jv->setup));

    /* Reset shaders */
    for (int i = 0; i < 2; i++) { /* AC/Golomb */
        RET(init_reset_shader(f, &ctx->s, &ctx->exec_pool,
                              spv, &jv->reset[i], !i ? AC_RANGE_CUSTOM_TAB : 0));
    }

    /* Decode shaders */
    for (int i = 0; i < 2; i++) { /* 16/32 bit */
        for (int j = 0; j < 2; j++) { /* AC/Golomb */
            for (int k = 0; k < 2; k++) { /* Normal/RGB */
                AVHWFramesContext *dec_frames_ctx;
                dec_frames_ctx = k ? (AVHWFramesContext *)jv->intermediate_frames_ref[i]->data :
                                     (AVHWFramesContext *)avctx->hw_frames_ctx->data;
                RET(init_decode_shader(f, &ctx->s, &ctx->exec_pool,
                                       spv, &jv->decode[i][j][k],
                                       dec_frames_ctx,
                                       (AVHWFramesContext *)avctx->hw_frames_ctx->data,
                                       i,
                                       !j ? AC_RANGE_CUSTOM_TAB : AC_GOLOMB_RICE,
                                       k));
            }
        }
    }

    /* Range coder data */
    RET(ff_ffv1_vk_init_state_transition_data(&ctx->s,
                                              &jv->rangecoder_static_buf,
                                              f));

    /* Quantization table data */
    RET(ff_ffv1_vk_init_quant_table_data(&ctx->s,
                                         &jv->quant_buf,
                                         f));

    /* CRC table buffer */
    RET(ff_ffv1_vk_init_crc_table_data(&ctx->s,
                                       &jv->crc_tab_buf,
                                       f));

    /* Update setup global descriptors */
    RET(ff_vk_shader_update_desc_buffer(&ctx->s, &ctx->exec_pool.contexts[0],
                                        &jv->setup, 0, 0, 0,
                                        &jv->rangecoder_static_buf,
                                        0, jv->rangecoder_static_buf.size,
                                        VK_FORMAT_UNDEFINED));
    RET(ff_vk_shader_update_desc_buffer(&ctx->s, &ctx->exec_pool.contexts[0],
                                        &jv->setup, 0, 1, 0,
                                        &jv->crc_tab_buf,
                                        0, jv->crc_tab_buf.size,
                                        VK_FORMAT_UNDEFINED));

    /* Update decode global descriptors */
    for (int i = 0; i < 2; i++) { /* 16/32 bit */
        for (int j = 0; j < 2; j++) { /* AC/Golomb */
            for (int k = 0; k < 2; k++) { /* Normal/RGB */
                RET(ff_vk_shader_update_desc_buffer(&ctx->s, &ctx->exec_pool.contexts[0],
                                                    &jv->decode[i][j][k], 0, 0, 0,
                                                    &jv->rangecoder_static_buf,
                                                    0, jv->rangecoder_static_buf.size,
                                                    VK_FORMAT_UNDEFINED));
                RET(ff_vk_shader_update_desc_buffer(&ctx->s, &ctx->exec_pool.contexts[0],
                                                    &jv->decode[i][j][k], 0, 1, 0,
                                                    &jv->quant_buf,
                                                    0, jv->quant_buf.size,
                                                    VK_FORMAT_UNDEFINED));
            }
        }
    }
    #endif

fail:
    spv->uninit(&spv);

    return err;
}

static void vk_jpeg2000_free_frame_priv(AVRefStructOpaque _hwctx, void *data)
{
    AVHWDeviceContext *dev_ctx = _hwctx.nc;
    AVVulkanDeviceContext *hwctx = dev_ctx->hwctx;

    JPEG2000VulkanDecodePicture *jp = data;
    FFVulkanDecodePicture *vp = &jp->vp;
    FFVkBuffer *slice_status = (FFVkBuffer *)jp->slice_status_buf->data;

    ff_vk_decode_free_frame(dev_ctx, vp);

    av_buffer_unref(&vp->slices_buf);

    #if 0
    av_buffer_unref(&jp->slice_state);
    av_buffer_unref(&jp->slice_offset_buf);
    av_buffer_unref(&jp->slice_status_buf);
    #endif
}

const FFHWAccel ff_jpeg2000_vulkan_hwaccel = {
    .p.name                = "jpeg2000_vulkan",
    .p.type                = AVMEDIA_TYPE_VIDEO,
    .p.id                  = AV_CODEC_ID_JPEG2000,
    .p.pix_fmt             = AV_PIX_FMT_VULKAN,
    .start_frame           = &vk_jpeg2000_start_frame,
    .decode_slice          = &vk_jpeg2000_decode_slice,
    .end_frame             = &vk_jpeg2000_end_frame,
    .free_frame_priv       = &vk_jpeg2000_free_frame_priv,
    .frame_priv_data_size  = sizeof(JPEG2000VulkanDecodePicture),
    .init                  = &vk_decode_jpeg2000_init,
    .update_thread_context = &ff_vk_update_thread_context,
    .decode_params         = &ff_vk_params_invalidate,
    .flush                 = &ff_vk_decode_flush,
    .uninit                = &ff_vk_decode_uninit,
    .frame_params          = &ff_vk_frame_params,
    .priv_data_size        = sizeof(FFVulkanDecodeContext),
    .caps_internal         = HWACCEL_CAP_ASYNC_SAFE | HWACCEL_CAP_THREAD_SAFE,
};
