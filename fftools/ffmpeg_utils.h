/*
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

#ifndef FFTOOLS_FFMPEG_UTILS_H
#define FFTOOLS_FFMPEG_UTILS_H

#include "libavutil/thread.h"
#include <stdint.h>

#include "libavutil/common.h"
#include "libavutil/frame.h"
#include "libavutil/rational.h"

#include "libavcodec/packet.h"


typedef struct SideDataStorage {
    enum AVPacketSideDataType *types;
    uint8_t **datas;
    int *sizes;
    int nb_elems;
} SideDataStorage;

typedef struct SideDataNode {
    SideDataStorage sd;
    struct SideDataNode *next;
} SideDataNode;

typedef struct SideDataQueue {
    SideDataNode *head;
    SideDataNode *tail;
    pthread_mutex_t lock;
    pthread_cond_t cond;
} SideDataQueue;

#define SD_OST_MAX 8
extern SideDataQueue sd_queues[SD_OST_MAX][1]; //TODO: by stream_index
extern void* sd_ost[SD_OST_MAX];

typedef struct Timestamp {
    int64_t    ts;
    AVRational tb;
} Timestamp;

/**
 * Merge two return codes - return one of the error codes if at least one of
 * them was negative, 0 otherwise.
 */
static inline int err_merge(int err0, int err1)
{
    // prefer "real" errors over EOF
    if ((err0 >= 0 || err0 == AVERROR_EOF) && err1 < 0)
        return err1;
    return (err0 < 0) ? err0 : FFMIN(err1, 0);
}

/**
 * Wrapper calling av_frame_side_data_clone() in a loop for all source entries.
 * It does not clear dst beforehand. */
static inline int clone_side_data(AVFrameSideData ***dst, int *nb_dst,
                                  AVFrameSideData * const *src, int nb_src,
                                  unsigned int flags)
{
    for (int i = 0; i < nb_src; i++) {
        int ret = av_frame_side_data_clone(dst, nb_dst, src[i], flags);
        if (ret < 0)
            return ret;
    }

    return 0;
}

#endif // FFTOOLS_FFMPEG_UTILS_H
