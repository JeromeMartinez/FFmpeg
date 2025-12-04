/*
 * Copyright (c) 2006 Smartjog S.A.S, Baptiste Coudurier <baptiste.coudurier@gmail.com>
 * Copyright (c) 2011-2012 Smartjog S.A.S, Clément Bœsch <clement.boesch@smartjog.com>
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

/**
 * @file
 * Timecode helpers
 * @see https://en.wikipedia.org/wiki/SMPTE_time_code
 * @see http://www.dropframetimecode.org
 */

#include <stdio.h>
#include "common.h"
#include "timecode.h"
#include "timecode_internal.h"
#include "log.h"
#include "error.h"

int av_timecode_adjust_ntsc_framenum2(int framenum, int fps)
{
    /* only works for multiples of NTSC 29.97 */
    int drop_frames = 0;
    int d, m, frames_per_10mins;

    if (fps && fps % 30 == 0) {
        drop_frames = fps / 30 * 2;
        frames_per_10mins = fps / 30 * 17982;
    } else
        return framenum;

    d = framenum / frames_per_10mins;
    m = framenum % frames_per_10mins;

    return framenum + 9U * drop_frames * d + drop_frames * ((m - drop_frames) / (frames_per_10mins / 10));
}

uint32_t av_timecode_get_smpte_from_framenum(const AVTimecode *tc, int framenum)
{
    unsigned fps = tc->fps;
    int drop = !!(tc->flags & AV_TIMECODE_FLAG_DROPFRAME);
    int hh, mm, ss, ff;

    framenum += tc->start;
    if (drop)
        framenum = av_timecode_adjust_ntsc_framenum2(framenum, tc->fps);
    ff = framenum % fps;
    ss = framenum / fps      % 60;
    mm = framenum / (fps*60LL) % 60;
    hh = framenum / (fps*3600LL) % 24;
    return av_timecode_get_smpte(tc->rate, drop, hh, mm, ss, ff);
}

uint32_t av_timecode_get_smpte(AVRational rate, int drop, int hh, int mm, int ss, int ff)
{
    uint32_t tc = 0;

    /* For SMPTE 12-M timecodes, frame count is a special case if > 30 FPS.
       See SMPTE ST 12-1:2014 Sec 12.1 for more info. */
    if (av_cmp_q(rate, (AVRational) {30, 1}) == 1) {
        if (ff % 2 == 1) {
            if (av_cmp_q(rate, (AVRational) {50, 1}) == 0)
                tc |= (1 << 7);
            else
                tc |= (1 << 23);
        }
        ff /= 2;
    }

    hh = hh % 24;
    mm = av_clip(mm, 0, 59);
    ss = av_clip(ss, 0, 59);
    ff = ff % 40;

    tc |= drop << 30;
    tc |= (ff / 10) << 28;
    tc |= (ff % 10) << 24;
    tc |= (ss / 10) << 20;
    tc |= (ss % 10) << 16;
    tc |= (mm / 10) << 12;
    tc |= (mm % 10) << 8;
    tc |= (hh / 10) << 4;
    tc |= (hh % 10);

    return tc;
}

char *av_timecode_make_string(const AVTimecode *tc, char *buf, int framenum_arg)
{
    unsigned fps = tc->fps;
    int drop = tc->flags & AV_TIMECODE_FLAG_DROPFRAME;
    int hh, mm, ss, ff, ff_len, neg = 0;
    int64_t framenum = framenum_arg;

    framenum += tc->start;
    if (drop)
        framenum = av_timecode_adjust_ntsc_framenum2(framenum, fps);
    if (framenum < 0) {
        framenum = -framenum;
        neg = tc->flags & AV_TIMECODE_FLAG_ALLOWNEGATIVE;
    }
    ff = framenum % fps;
    ss = framenum / fps        % 60;
    mm = framenum / (fps*60LL) % 60;
    hh = framenum / (fps*3600LL);
    if (tc->flags & AV_TIMECODE_FLAG_24HOURSMAX)
        hh = hh % 24;
    ff_len = fps > 10000 ? 5 : fps > 1000 ? 4 : fps > 100 ? 3 : fps > 10 ? 2 : 1;
    snprintf(buf, AV_TIMECODE_STR_SIZE, "%s%02d:%02d:%02d%c%0*d",
             neg ? "-" : "",
             hh, mm, ss, drop ? ';' : ':', ff_len, ff);
    return buf;
}

char *av_timecode_make_smpte_tc_string2(char *buf, AVRational rate, uint32_t tcsmpte, int prevent_df, int skip_field)
{
    unsigned hh, mm, ss, ff, drop;
    ff_timecode_set_smpte(&drop, &hh, &mm, &ss, &ff, rate, tcsmpte, prevent_df, skip_field);

    snprintf(buf, AV_TIMECODE_STR_SIZE, "%02u:%02u:%02u%c%02u",
             hh, mm, ss, drop ? ';' : ':', ff);
    return buf;

}

char *av_timecode_make_smpte_tc_string(char *buf, uint32_t tcsmpte, int prevent_df)
{
    return av_timecode_make_smpte_tc_string2(buf, (AVRational){30, 1}, tcsmpte, prevent_df, 1);
}

char *av_timecode_make_mpeg_tc_string(char *buf, uint32_t tc25bit)
{
    snprintf(buf, AV_TIMECODE_STR_SIZE,
             "%02"PRIu32":%02"PRIu32":%02"PRIu32"%c%02"PRIu32,
             tc25bit>>19 & 0x1f,              // 5-bit hours
             tc25bit>>13 & 0x3f,              // 6-bit minutes
             tc25bit>>6  & 0x3f,              // 6-bit seconds
             tc25bit     & 1<<24 ? ';' : ':', // 1-bit drop flag
             tc25bit     & 0x3f);             // 6-bit frames
    return buf;
}

static int check_fps(int fps)
{
    int i;
    static const int supported_fps[] = {
        24, 25, 30, 48, 50, 60, 100, 120, 150,
    };

    for (i = 0; i < FF_ARRAY_ELEMS(supported_fps); i++)
        if (fps == supported_fps[i])
            return 0;
    return -1;
}

static int check_timecode(void *log_ctx, AVTimecode *tc)
{
    if ((int)tc->fps <= 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Valid timecode frame rate must be specified. Minimum value is 1\n");
        return AVERROR(EINVAL);
    }
    if ((tc->flags & AV_TIMECODE_FLAG_DROPFRAME) && tc->fps % 30 != 0) {
        av_log(log_ctx, AV_LOG_ERROR, "Drop frame is only allowed with multiples of 30000/1001 FPS\n");
        return AVERROR(EINVAL);
    }
    if (check_fps(tc->fps) < 0) {
        av_log(log_ctx, AV_LOG_WARNING, "Using non-standard frame rate %d/%d\n",
               tc->rate.num, tc->rate.den);
    }
    return 0;
}

static int fps_from_frame_rate(AVRational rate)
{
    if (!rate.den || !rate.num)
        return -1;
    return (rate.num + rate.den/2LL) / rate.den;
}

int av_timecode_check_frame_rate(AVRational rate)
{
    return check_fps(fps_from_frame_rate(rate));
}

int av_timecode_init(AVTimecode *tc, AVRational rate, int flags, int frame_start, void *log_ctx)
{
    memset(tc, 0, sizeof(*tc));
    tc->start = frame_start;
    tc->flags = flags;
    tc->rate  = rate;
    tc->fps   = fps_from_frame_rate(rate);
    return check_timecode(log_ctx, tc);
}

int av_timecode_init_from_components(AVTimecode *tc, AVRational rate, int flags, int hh, int mm, int ss, int ff, void *log_ctx)
{
    int ret;

    memset(tc, 0, sizeof(*tc));
    tc->flags = flags;
    tc->rate  = rate;
    tc->fps   = fps_from_frame_rate(rate);

    ret = check_timecode(log_ctx, tc);
    if (ret < 0)
        return ret;

    tc->start = (hh*3600 + mm*60 + ss) * tc->fps + ff;
    if (tc->flags & AV_TIMECODE_FLAG_DROPFRAME) { /* adjust frame number */
        int tmins = 60*hh + mm;
        tc->start -= (tc->fps / 30 * 2) * (tmins - tmins/10);
    }
    return 0;
}

int av_timecode_init_from_string(AVTimecode *tc, AVRational rate, const char *str, void *log_ctx)
{
    char c;
    int hh, mm, ss, ff, flags;

    if (sscanf(str, "%d:%d:%d%c%d", &hh, &mm, &ss, &c, &ff) != 5) {
        av_log(log_ctx, AV_LOG_ERROR, "Unable to parse timecode, "
                                      "syntax: hh:mm:ss[:;.]ff\n");
        return AVERROR_INVALIDDATA;
    }
    flags = c != ':' ? AV_TIMECODE_FLAG_DROPFRAME : 0; // drop if ';', '.', ...

    return av_timecode_init_from_components(tc, rate, flags, hh, mm, ss, ff, log_ctx);
}

static int bcd_to_int(int bcd) {
    return ((bcd >> 4) & 0xF) * 10 + (bcd & 0xF);
}

uint64_t av_timecode_expand_to_64bit(uint32_t tc32)
{
    uint64_t tc64 = 0;

    int hours_bcd   =  tc32        & 0x3F;  // bits 0-5
    int bgf1        = (tc32 >> 6)  & 0x1;   // bit 6
    int bgf2        = (tc32 >> 7)  & 0x1;   // bit 7
    int minutes_bcd = (tc32 >> 8)  & 0x7F;  // bits 8-14
    int bgf0        = (tc32 >> 15) & 0x1;   // bit 15
    int seconds_bcd = (tc32 >> 16) & 0x7F;  // bits 16-22
    int frames_bcd  = (tc32 >> 24) & 0x3F;  // bits 24-29
    int drop        = (tc32 >> 30) & 0x1;   // bit 30
    int color       = (tc32 >> 31) & 0x1;   // bit 31

    int hours   = bcd_to_int(hours_bcd);
    int minutes = bcd_to_int(minutes_bcd);
    int seconds = bcd_to_int(seconds_bcd);
    int frames  = bcd_to_int(frames_bcd);

    // Units and tens
    int uh = hours   % 10, th = hours   / 10;
    int um = minutes % 10, tm = minutes / 10;
    int us = seconds % 10, ts = seconds / 10;
    int uf = frames  % 10, tf = frames  / 10;

    // Assign bits as per RFC 5484 layout
    tc64 |= (uint64_t)(uf & 0xF) << 0;   // Units of frames
    tc64 |= 0                    << 4;   // First binary group
    tc64 |= (uint64_t)(tf & 0x3) << 8;   // Tens of frames (2 bits)
    tc64 |= (uint64_t)drop       << 10;  // Drop frame flag
    tc64 |= (uint64_t)color      << 11;  // Color frame flag
    tc64 |= 0                    << 12;  // Second binary group

    tc64 |= (uint64_t)(us & 0xF) << 16;  // Units of seconds
    tc64 |= 0                    << 20;  // Third binary group
    tc64 |= (uint64_t)(ts & 0x7) << 24;  // Tens of seconds
    tc64 |= 0                    << 27;  // Polarity correction
    tc64 |= 0                    << 28;  // Fourth binary group

    tc64 |= (uint64_t)(um & 0xF) << 32;  // Units of minutes
    tc64 |= (uint64_t)0          << 36;  // Fifth binary group
    tc64 |= (uint64_t)(tm & 0x7) << 40;  // Tens of minutes
    tc64 |= (uint64_t)bgf0       << 43;  // BGF0
    tc64 |= (uint64_t)0          << 44;  // Sixth binary group

    tc64 |= (uint64_t)(uh & 0xF) << 48;  // Units of hours
    tc64 |= (uint64_t)0          << 52;  // Seventh binary group
    tc64 |= (uint64_t)(th & 0x3) << 56;  // Tens of hours
    tc64 |= (uint64_t)bgf1       << 58;  // BGF1
    tc64 |= (uint64_t)bgf2       << 59;  // BGF2
    tc64 |= (uint64_t)0          << 60;  // Eighth binary group

    return tc64;
}

static int int_to_bcd(int val) {
    return ((val / 10) << 4) | (val % 10);
}

uint32_t av_timecode_parse_from_64bit(uint64_t tc64)
{
    uint32_t tc32 = 0;

    int uf   = (tc64 >> 0)  & 0xF;   // Ones of frames
    int tf   = (tc64 >> 8)  & 0x3;   // Tens of frames
    int drop = (tc64 >> 10) & 0x1;
    int color= (tc64 >> 11) & 0x1;

    int us   = (tc64 >> 16) & 0xF;   // Ones of seconds
    int ts   = (tc64 >> 24) & 0x7;   // Tens of seconds

    int um   = (tc64 >> 32) & 0xF;   // Ones of minutes
    int tm   = (tc64 >> 40) & 0x7;   // Tens of minutes
    int bgf0 = (tc64 >> 43) & 0x1;

    int uh   = (tc64 >> 48) & 0xF;   // Ones of hours
    int th   = (tc64 >> 56) & 0x3;   // Tens of hours
    int bgf1 = (tc64 >> 58) & 0x1;
    int bgf2 = (tc64 >> 59) & 0x1;

    int hours   = int_to_bcd(th * 10 + uh);
    int minutes = int_to_bcd(tm * 10 + um);
    int seconds = int_to_bcd(ts * 10 + us);
    int frames  = int_to_bcd(tf * 10 + uf);

    // Assemble tc32 using the FFmpeg SMPTE 32-bit format
    tc32 |= (hours & 0x3F)        << 0;   // bits 0-5: hours (BCD)
    tc32 |= (bgf1  & 0x1)         << 6;   // bit 6: BGF1
    tc32 |= (bgf2  & 0x1)         << 7;   // bit 7: BGF2 (or FIELD)
    tc32 |= (minutes & 0x7F)      << 8;   // bits 8-14: minutes (BCD)
    tc32 |= (bgf0  & 0x1)         << 15;  // bit 15: BGF0
    tc32 |= (seconds & 0x7F)      << 16;  // bits 16-22: seconds (BCD)
    tc32 |= 0                     << 23;  // bit 23: FIELD (set to 0)
    tc32 |= (frames  & 0x3F)      << 24;  // bits 24-29: frames (BCD)
    tc32 |= (drop    & 0x1)       << 30;  // bit 30: drop frame flag
    tc32 |= (color   & 0x1)       << 31;  // bit 31: color frame flag

    return tc32;
}

int av_timecode_add_to_side_data(AVFormatContext *ctx, AVPacket *pkt, unsigned flags, uint64_t tc, uint32_t id, const char *title)
{
    const int add_block_timecode_count = 4; //TODO
    size_t sd_size = 0;
    uint8_t *sd = av_packet_get_side_data(pkt, AV_PKT_DATA_S12M_TIMECODE, &sd_size);
    uint8_t count = sd ? *((uint8_t*) (sd + 4)) : 0;
    if (!count) {
        sd_size = 8 + add_block_timecode_count * (8 + 4 + 16);
        sd = av_packet_new_side_data(pkt, AV_PKT_DATA_S12M_TIMECODE, sd_size);
        uint32_t *sd_allocated = (uint32_t*)sd;
        *sd_allocated = sd_size;
        uint8_t *item_size = sd + 5;
        *item_size = 8 + 4 + 16;
        count = 0;
    }

    if (count >= add_block_timecode_count) {
        av_log(ctx, AV_LOG_DEBUG, "There are more timecodes in the block than the count indicated in the track header, extra timecodes are ignored.\n");
    }
    else if (sd) {
        av_log(ctx, AV_LOG_DEBUG, "Reading SMPTE timecode from BlockAdditional: 0x%016lX (RFC 5484)\n", tc);

        // header: 0-3 allocated size; 4 count; 5 byte size per item; 6 flags; 7 reserved
        // flags: 0 id present; 1 title present
        // content: 8 bytes tc; 4 bytes id; 16 bytes 0 terminated or 16 bytes UTF8 title

        count++;
        uint8_t *sd_count = (uint8_t*) (sd + 4);
        *sd_count = count;
        uint8_t *sd_base = sd + 8 + (8 + 4 + 16) * (count - 1);
        uint64_t *sd_tc = (uint64_t*)sd_base;
        uint32_t *sd_id = (uint32_t*)(sd_base + 8);
        char* sd_title = (char*)(sd_base + 8 + 4);
        *sd_tc = tc;
        *sd_id = id;
        size_t title_size = strlen(title);
        memcpy(sd_title, title, title_size);
        memset(sd_title + title_size, 0, 16 - title_size);
    }
    
    return 0;
}