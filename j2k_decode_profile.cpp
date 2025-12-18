#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>

#include "openjpeg.h"

/* Wall-clock timing helper */
static double get_wall_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1e6;
}

static void error_callback(const char* msg, void* client_data) {
    (void)client_data;
    fputs(msg, stderr);
}

static void warning_callback(const char* msg, void* client_data) {
    (void)client_data;
    fputs(msg, stderr);
}

static void info_callback(const char* msg, void* client_data) {
    (void)client_data;
    (void)msg;
}

static unsigned char clamp_u8(int v) {
    if (v < 0) {
        return 0;
    }
    if (v > 255) {
        return 255;
    }
    return (unsigned char)v;
}

static int write_pnm_u8(const char* path, const opj_image_t* image) {
    if (!image || image->numcomps < 1) {
        return 0;
    }

    const int width = (int)(image->x1 - image->x0);
    const int height = (int)(image->y1 - image->y0);
    if (width <= 0 || height <= 0) {
        return 0;
    }

    const int numcomps = (image->numcomps >= 3) ? 3 : 1;

    for (int c = 0; c < numcomps; c++) {
        if (image->comps[c].w != (OPJ_UINT32)width || image->comps[c].h != (OPJ_UINT32)height) {
            fprintf(stderr, "Unsupported: subsampled components\n");
            return 0;
        }
        if (image->comps[c].prec > 16) {
            fprintf(stderr, "Unsupported: precision > 16\n");
            return 0;
        }
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot open output: %s\n", path);
        return 0;
    }

    if (numcomps == 3) {
        fprintf(fp, "P6\n%d %d\n255\n", width, height);
    } else {
        fprintf(fp, "P5\n%d %d\n255\n", width, height);
    }

    const size_t pixels = (size_t)width * (size_t)height;
    const int rshift = (image->numcomps >= 3) ? ((int)image->comps[0].prec - 8) : 0;
    const int gshift = (image->numcomps >= 3) ? ((int)image->comps[1].prec - 8) : 0;
    const int bshift = (image->numcomps >= 3) ? ((int)image->comps[2].prec - 8) : 0;
    const int shift0 = ((int)image->comps[0].prec - 8);

    if (numcomps == 3) {
        /* Allocate buffer for all pixels (3 bytes per pixel for RGB) */
        size_t total_bytes = pixels * 3;
        unsigned char* buf = (unsigned char*)malloc(total_bytes);
        if (!buf) {
            fclose(fp);
            return 0;
        }

        /* Process pixels in parallel with OpenMP */
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pixels; i++) {
            int r = image->comps[0].data[i];
            int g = image->comps[1].data[i];
            int b = image->comps[2].data[i];

            if (rshift > 0) r >>= rshift;
            if (gshift > 0) g >>= gshift;
            if (bshift > 0) b >>= bshift;

            size_t out_idx = i * 3;
            buf[out_idx]     = clamp_u8(r);
            buf[out_idx + 1] = clamp_u8(g);
            buf[out_idx + 2] = clamp_u8(b);
        }

        if (fwrite(buf, 1, total_bytes, fp) != total_bytes) {
            free(buf);
            fclose(fp);
            return 0;
        }
        free(buf);
    } else {
        /* Allocate buffer for all pixels (1 byte per pixel for grayscale) */
        unsigned char* buf = (unsigned char*)malloc(pixels);
        if (!buf) {
            fclose(fp);
            return 0;
        }

        /* Process pixels in parallel with OpenMP */
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pixels; i++) {
            int v = image->comps[0].data[i];
            if (shift0 > 0) {
                v >>= shift0;
            }
            buf[i] = clamp_u8(v);
        }

        if (fwrite(buf, 1, pixels, fp) != pixels) {
            free(buf);
            fclose(fp);
            return 0;
        }
        free(buf);
    }

    fclose(fp);
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.j2k> <output.pgm|output.ppm>\n", argv[0]);
        return 2;
    }

    const char* in_path = argv[1];
    const char* out_path = argv[2];

    double t_load_start = get_wall_time();

    opj_dparameters_t parameters;
    opj_set_default_decoder_parameters(&parameters);

    double t_load_end = get_wall_time();
    double t_create_start = get_wall_time();

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    if (!codec) {
        return 1;
    }

    double t_create_end = get_wall_time();
    double t_setup_start = get_wall_time();

    opj_set_error_handler(codec, error_callback, NULL);
    opj_set_warning_handler(codec, warning_callback, NULL);
    opj_set_info_handler(codec, info_callback, NULL);

    opj_stream_t* stream = opj_stream_create_default_file_stream(in_path, OPJ_TRUE);
    if (!stream) {
        opj_destroy_codec(codec);
        return 1;
    }

    if (!opj_setup_decoder(codec, &parameters)) {
        fprintf(stderr, "opj_setup_decoder failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    double t_setup_end = get_wall_time();
    double t_readhdr_start = get_wall_time();

    opj_image_t* image = NULL;
    if (!opj_read_header(stream, codec, &image)) {
        fprintf(stderr, "opj_read_header failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    double t_readhdr_end = get_wall_time();
    double t_decode_start = get_wall_time();

    if (!opj_decode(codec, stream, image)) {
        fprintf(stderr, "opj_decode failed\n");
        opj_image_destroy(image);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    double t_decode_end = get_wall_time();
    double t_end_start = get_wall_time();

    if (!opj_end_decompress(codec, stream)) {
        fprintf(stderr, "opj_end_decompress failed\n");
        opj_image_destroy(image);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    double t_end_end = get_wall_time();
    double t_write_start = get_wall_time();

    int ok = write_pnm_u8(out_path, image);

    double t_write_end = get_wall_time();
    
    /* Capture image info before cleanup */
    OPJ_UINT32 img_width = image ? (image->x1 - image->x0) : 0;
    OPJ_UINT32 img_height = image ? (image->y1 - image->y0) : 0;

    double t_cleanup_start = get_wall_time();

    opj_image_destroy(image);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);

    double t_cleanup_end = get_wall_time();

    /* Calculate totals and percentages */
    double t_setup = (t_load_end - t_load_start) + (t_create_end - t_create_start) + (t_setup_end - t_setup_start);
    double t_readhdr = t_readhdr_end - t_readhdr_start;
    double t_decode = t_decode_end - t_decode_start;
    double t_end = t_end_end - t_end_start;
    double t_write = t_write_end - t_write_start;
    double t_cleanup = t_cleanup_end - t_cleanup_start;
    double t_total = t_write_end - t_load_start;

    /* Print profiling results */
    fprintf(stdout, "=== DECODING PROFILING RESULTS ===\n");
    fprintf(stdout, "Input file: %s\n", in_path);
    fprintf(stdout, "Image size: %ux%u\n", img_width, img_height);
    fprintf(stdout, "====================================\n");
    fprintf(stdout, "Setup (params + create + setup): %.4f s (%5.1f%%)\n", t_setup, t_total > 0 ? 100.0 * t_setup / t_total : 0);
    fprintf(stdout, "Read header:                     %.4f s (%5.1f%%)\n", t_readhdr, t_total > 0 ? 100.0 * t_readhdr / t_total : 0);
    fprintf(stdout, "Decode (main):                   %.4f s (%5.1f%%)\n", t_decode, t_total > 0 ? 100.0 * t_decode / t_total : 0);
    fprintf(stdout, "End decompress:                  %.4f s (%5.1f%%)\n", t_end, t_total > 0 ? 100.0 * t_end / t_total : 0);
    fprintf(stdout, "Write output:                    %.4f s (%5.1f%%)\n", t_write, t_total > 0 ? 100.0 * t_write / t_total : 0);
    fprintf(stdout, "Cleanup:                         %.4f s (%5.1f%%)\n", t_cleanup, t_total > 0 ? 100.0 * t_cleanup / t_total : 0);
    fprintf(stdout, "====================================\n");
    fprintf(stdout, "TOTAL TIME:                      %.4f s\n", t_total);
    fprintf(stdout, "====================================\n");

    if (!ok) {
        fprintf(stderr, "Failed to write output PNM\n");
    }

    return ok ? 0 : 1;
}
