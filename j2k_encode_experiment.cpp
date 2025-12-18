/**
 * JPEG2000 Encoder with configurable OpenMP optimizations
 * 
 * Compile with different flags to enable/disable optimizations:
 *   -DPIXEL_PARALLEL=1    Enable parallel pixel conversion (default: 1)
 *   -DT1_PARALLEL=1       Enable parallel T1 encoding (default: 1)
 *   -DNUM_THREADS=N       Set number of threads (default: OMP_NUM_THREADS)
 * 
 * Examples:
 *   # All optimizations enabled (default)
 *   g++ -O3 -fopenmp j2k_encode_experiment.cpp ...
 * 
 *   # Only T1 parallel, no pixel parallel
 *   g++ -O3 -fopenmp -DPIXEL_PARALLEL=0 j2k_encode_experiment.cpp ...
 * 
 *   # No OpenMP at all (baseline)
 *   g++ -O3 -DPIXEL_PARALLEL=0 -DT1_PARALLEL=0 j2k_encode_experiment.cpp ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#include "openjpeg.h"

extern double opj_clock(void);

// Default optimization settings
#ifndef PIXEL_PARALLEL
#define PIXEL_PARALLEL 1
#endif

#ifndef T1_PARALLEL
#define T1_PARALLEL 1
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 0  // 0 = use OMP_NUM_THREADS or max available
#endif

typedef struct {
    double total_time;
    double load_image_time;
    double pixel_convert_time;
    double setup_time;
    double compress_start_time;
    double encode_time;
    double compress_end_time;
} profile_times_t;

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

static int read_non_comment_token(FILE* fp, char* buf, size_t buf_size) {
    int ch;
    do {
        ch = fgetc(fp);
        if (ch == EOF) return 0;
        if (isspace(ch)) continue;
        if (ch == '#') {
            while ((ch = fgetc(fp)) != EOF && ch != '\n');
            continue;
        }
        ungetc(ch, fp);
        break;
    } while (1);

    size_t len = 0;
    while ((ch = fgetc(fp)) != EOF) {
        if (isspace(ch) || ch == '#') {
            if (ch == '#') {
                while ((ch = fgetc(fp)) != EOF && ch != '\n');
            }
            break;
        }
        if (len + 1 < buf_size) {
            buf[len++] = (char)ch;
        } else {
            return 0;
        }
    }
    buf[len] = '\0';
    return len > 0;
}

static opj_image_t* load_pnm_as_image(const char* path, profile_times_t* prof) {
    double t_start = opj_clock();
    
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open input: %s\n", path);
        return NULL;
    }

    (void)setvbuf(fp, NULL, _IOFBF, 4 * 1024 * 1024);

    char tok[64];
    if (!read_non_comment_token(fp, tok, sizeof(tok))) {
        fclose(fp);
        return NULL;
    }

    int is_ppm = 0;
    if (strcmp(tok, "P5") == 0) {
        is_ppm = 0;
    } else if (strcmp(tok, "P6") == 0) {
        is_ppm = 1;
    } else {
        fprintf(stderr, "Only binary PNM P5/P6 supported (got %s)\n", tok);
        fclose(fp);
        return NULL;
    }

    if (!read_non_comment_token(fp, tok, sizeof(tok))) { fclose(fp); return NULL; }
    int width = atoi(tok);

    if (!read_non_comment_token(fp, tok, sizeof(tok))) { fclose(fp); return NULL; }
    int height = atoi(tok);

    if (!read_non_comment_token(fp, tok, sizeof(tok))) { fclose(fp); return NULL; }
    int maxval = atoi(tok);

    if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) {
        fprintf(stderr, "Unsupported PNM header (w=%d h=%d maxval=%d)\n", width, height, maxval);
        fclose(fp);
        return NULL;
    }

    const int numcomps = is_ppm ? 3 : 1;
    opj_image_cmptparm_t cmptparm[3];
    memset(cmptparm, 0, sizeof(cmptparm));
    for (int i = 0; i < numcomps; i++) {
        cmptparm[i].dx = 1;
        cmptparm[i].dy = 1;
        cmptparm[i].w = (OPJ_UINT32)width;
        cmptparm[i].h = (OPJ_UINT32)height;
        cmptparm[i].prec = 8;
        cmptparm[i].sgnd = 0;
    }

    OPJ_COLOR_SPACE cs = is_ppm ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY;
    opj_image_t* image = opj_image_create(numcomps, cmptparm, cs);
    if (!image) {
        fclose(fp);
        return NULL;
    }

    image->x0 = 0;
    image->y0 = 0;
    image->x1 = width;
    image->y1 = height;

    const size_t pixels = (size_t)width * (size_t)height;
    const size_t bytes_per_pixel = (size_t)numcomps;
    const size_t data_size = pixels * bytes_per_pixel;

    unsigned char* data = (unsigned char*)malloc(data_size);
    if (!data) {
        opj_image_destroy(image);
        fclose(fp);
        return NULL;
    }

    size_t nread = fread(data, 1, data_size, fp);
    fclose(fp);
    if (nread != data_size) {
        fprintf(stderr, "Unexpected EOF reading pixel data\n");
        free(data);
        opj_image_destroy(image);
        return NULL;
    }

    double t_file_done = opj_clock();
    prof->load_image_time = t_file_done - t_start;

    // Pixel conversion - optionally parallel
    double t_convert_start = opj_clock();
    
    if (is_ppm) {
#if PIXEL_PARALLEL
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < pixels; i++) {
            image->comps[0].data[i] = data[i * 3 + 0];
            image->comps[1].data[i] = data[i * 3 + 1];
            image->comps[2].data[i] = data[i * 3 + 2];
        }
    } else {
#if PIXEL_PARALLEL
        #pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < pixels; i++) {
            image->comps[0].data[i] = data[i];
        }
    }

    free(data);
    prof->pixel_convert_time = opj_clock() - t_convert_start;
    
    return image;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.ppm> <output.j2k>\n", argv[0]);
        fprintf(stderr, "\nCompile-time options:\n");
        fprintf(stderr, "  PIXEL_PARALLEL=%d (pixel conversion)\n", PIXEL_PARALLEL);
        fprintf(stderr, "  T1_PARALLEL=%d (T1 encoding)\n", T1_PARALLEL);
        fprintf(stderr, "  NUM_THREADS=%d (0=auto)\n", NUM_THREADS);
        return 2;
    }

    const char* in_path = argv[1];
    const char* out_path = argv[2];

    // Determine thread count
    int num_threads = NUM_THREADS;
    if (num_threads == 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    printf("=== OpenMP Optimization Experiment ===\n");
    printf("PIXEL_PARALLEL: %s\n", PIXEL_PARALLEL ? "ENABLED" : "DISABLED");
    printf("T1_PARALLEL:    %s\n", T1_PARALLEL ? "ENABLED" : "DISABLED");
    printf("NUM_THREADS:    %d\n", num_threads);
    printf("======================================\n\n");

    profile_times_t prof = {0};
    double total_start = opj_clock();

    // Load image
    opj_image_t* image = load_pnm_as_image(in_path, &prof);
    if (!image) {
        return 1;
    }

    printf("Image: %dx%d, %d components\n", 
           image->x1 - image->x0, image->y1 - image->y0, image->numcomps);

    // Setup encoder
    double t_setup_start = opj_clock();
    
    opj_cparameters_t parameters;
    opj_set_default_encoder_parameters(&parameters);
    parameters.tcp_rates[0] = 0;
    parameters.tcp_numlayers = 1;
    parameters.cp_disto_alloc = 1;
    parameters.numresolution = 6;

    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    if (!codec) {
        opj_image_destroy(image);
        return 1;
    }

    opj_set_error_handler(codec, error_callback, NULL);
    opj_set_warning_handler(codec, warning_callback, NULL);
    opj_set_info_handler(codec, info_callback, NULL);

    // Set T1 parallel encoding
#if T1_PARALLEL
    if (!opj_codec_set_threads(codec, num_threads)) {
        fprintf(stderr, "Warning: Failed to set %d threads for T1\n", num_threads);
    }
#else
    // Force single-threaded T1 encoding
    opj_codec_set_threads(codec, 1);
#endif

    opj_stream_t* stream = opj_stream_create_default_file_stream(out_path, OPJ_FALSE);
    if (!stream) {
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }

    if (!opj_setup_encoder(codec, &parameters, image)) {
        fprintf(stderr, "opj_setup_encoder failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof.setup_time = opj_clock() - t_setup_start;

    // Compress
    double t_compress_start = opj_clock();
    if (!opj_start_compress(codec, image, stream)) {
        fprintf(stderr, "opj_start_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof.compress_start_time = opj_clock() - t_compress_start;

    double t_encode_start = opj_clock();
    if (!opj_encode(codec, stream)) {
        fprintf(stderr, "opj_encode failed\n");
        opj_end_compress(codec, stream);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof.encode_time = opj_clock() - t_encode_start;

    double t_end_start = opj_clock();
    if (!opj_end_compress(codec, stream)) {
        fprintf(stderr, "opj_end_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof.compress_end_time = opj_clock() - t_end_start;

    prof.total_time = opj_clock() - total_start;

    // Print results
    printf("\n=== Profiling Results ===\n");
    printf("File I/O:         %.4fs (%5.1f%%)\n", prof.load_image_time, 
           100.0 * prof.load_image_time / prof.total_time);
    printf("Pixel Convert:    %.4fs (%5.1f%%) [PIXEL_PARALLEL=%d]\n", prof.pixel_convert_time,
           100.0 * prof.pixel_convert_time / prof.total_time, PIXEL_PARALLEL);
    printf("Setup:            %.4fs (%5.1f%%)\n", prof.setup_time,
           100.0 * prof.setup_time / prof.total_time);
    printf("Compress Start:   %.4fs (%5.1f%%)\n", prof.compress_start_time,
           100.0 * prof.compress_start_time / prof.total_time);
    printf("Encode (T1+DWT):  %.4fs (%5.1f%%) [T1_PARALLEL=%d]\n", prof.encode_time,
           100.0 * prof.encode_time / prof.total_time, T1_PARALLEL);
    printf("Compress End:     %.4fs (%5.1f%%)\n", prof.compress_end_time,
           100.0 * prof.compress_end_time / prof.total_time);
    printf("=========================\n");
    printf("TOTAL:            %.4fs\n", prof.total_time);
    printf("=========================\n");

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);

    return 0;
}
