#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>
#include <ctype.h>
#include <omp.h>

#include "openjpeg.h"
#include "profile_times.h"

/* Declare internal timing function from OpenJPEG */
extern double opj_clock(void);

/* External timing data from tcd.cpp */
extern struct TimingData global_timing;

// Profiling structure
typedef struct {
    double total_time;
    double load_image_time;
    double setup_time;
    double compress_start_time;
    double encode_time;
    double compress_end_time;
} profile_times_t;

static profile_times_t prof_times = {0};

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
        if (ch == EOF) {
            return 0;
        }
        if (isspace(ch)) {
            continue;
        }
        if (ch == '#') {
            while ((ch = fgetc(fp)) != EOF && ch != '\n') {
            }
            continue;
        }
        ungetc(ch, fp);
        break;
    } while (1);

    size_t len = 0;
    while ((ch = fgetc(fp)) != EOF) {
        if (isspace(ch) || ch == '#') {
            if (ch == '#') {
                while ((ch = fgetc(fp)) != EOF && ch != '\n') {
                }
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

static opj_image_t* load_pnm_as_image(const char* path) {
    double t_start = opj_clock();
    
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open input: %s\n", path);
        return NULL;
    }

    /* Large buffer for testing: 4MB */
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

    if (!read_non_comment_token(fp, tok, sizeof(tok))) {
        fclose(fp);
        return NULL;
    }
    int width = atoi(tok);

    if (!read_non_comment_token(fp, tok, sizeof(tok))) {
        fclose(fp);
        return NULL;
    }
    int height = atoi(tok);

    if (!read_non_comment_token(fp, tok, sizeof(tok))) {
        fclose(fp);
        return NULL;
    }
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

    if (is_ppm) {
        /* Parallel pixel copy with OpenMP */
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pixels; i++) {
            image->comps[0].data[i] = data[i * 3 + 0];
            image->comps[1].data[i] = data[i * 3 + 1];
            image->comps[2].data[i] = data[i * 3 + 2];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pixels; i++) {
            image->comps[0].data[i] = data[i];
        }
    }

    free(data);
    
    double t_end = opj_clock();
    prof_times.load_image_time = t_end - t_start;
    
    return image;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input.pgm|input.ppm> <output.j2k>\n", argv[0]);
        return 2;
    }

    double total_start = opj_clock();

    const char* in_path = argv[1];
    const char* out_path = argv[2];

    opj_image_t* image = load_pnm_as_image(in_path);
    if (!image) {
        return 1;
    }

    double t_start, t_end;
    
    t_start = opj_clock();
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

    // Set number of threads for parallel encoding (auto-detect max threads)
    int num_threads = omp_get_max_threads();
    const char* env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads) {
        int t = atoi(env_threads);
        if (t > 0) num_threads = t;
    }
    if (num_threads > 0) {
        if (!opj_codec_set_threads(codec, num_threads)) {
            fprintf(stderr, "Warning: Failed to set %d threads\n", num_threads);
        } else {
            fprintf(stdout, "Using %d threads for encoding\n", num_threads);
        }
    }

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
    t_end = opj_clock();
    prof_times.setup_time = t_end - t_start;

    t_start = opj_clock();
    if (!opj_start_compress(codec, image, stream)) {
        fprintf(stderr, "opj_start_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    t_end = opj_clock();
    prof_times.compress_start_time = t_end - t_start;

    t_start = opj_clock();
    if (!opj_encode(codec, stream)) {
        fprintf(stderr, "opj_encode failed\n");
        opj_end_compress(codec, stream);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    t_end = opj_clock();
    prof_times.encode_time = t_end - t_start;

    t_start = opj_clock();
    if (!opj_end_compress(codec, stream)) {
        fprintf(stderr, "opj_end_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    t_end = opj_clock();
    prof_times.compress_end_time = t_end - t_start;

    double total_end = opj_clock();
    prof_times.total_time = total_end - total_start;

    // Open profiling log file
    FILE* log_fp = fopen("profiling_results.txt", "a");
    FILE* outputs[2] = {stdout, log_fp};
    
    for (int out_idx = 0; out_idx < 2; out_idx++) {
        FILE* fp = outputs[out_idx];
        if (!fp) continue;
        
        fprintf(fp, "\n=== ENCODING PROFILING RESULTS ===\n");
        fprintf(fp, "Input file: %s\n", in_path);
        fprintf(fp, "Image size: %dx%d\n", image->x1 - image->x0, image->y1 - image->y0);
        fprintf(fp, "====================================\n");
        fprintf(fp, "Load image:       %8.4f s (%5.1f%%)\n", 
               prof_times.load_image_time, 
               100.0 * prof_times.load_image_time / prof_times.total_time);
        fprintf(fp, "Setup encoder:    %8.4f s (%5.1f%%)\n", 
               prof_times.setup_time,
               100.0 * prof_times.setup_time / prof_times.total_time);
        fprintf(fp, "Start compress:   %8.4f s (%5.1f%%)\n", 
               prof_times.compress_start_time,
               100.0 * prof_times.compress_start_time / prof_times.total_time);
        fprintf(fp, "------------------------------------\n");
        fprintf(fp, "Encode (main):    %8.4f s (%5.1f%%)\n", 
               prof_times.encode_time,
               100.0 * prof_times.encode_time / prof_times.total_time);
        fprintf(fp, "  ├─ DC shift:    %8.4f s (%5.1f%%)\n",
               global_timing.dc_shift_time,
               100.0 * global_timing.dc_shift_time / prof_times.total_time);
        fprintf(fp, "  ├─ MCT:         %8.4f s (%5.1f%%)\n",
               global_timing.mct_time,
               100.0 * global_timing.mct_time / prof_times.total_time);
        fprintf(fp, "  ├─ DWT:         %8.4f s (%5.1f%%)\n",
               global_timing.dwt_time,
               100.0 * global_timing.dwt_time / prof_times.total_time);
        fprintf(fp, "  ├─ T1 (quant):  %8.4f s (%5.1f%%)\n",
               global_timing.t1_time,
               100.0 * global_timing.t1_time / prof_times.total_time);
        fprintf(fp, "  ├─ Rate alloc:  %8.4f s (%5.1f%%)\n",
               global_timing.rate_time,
               100.0 * global_timing.rate_time / prof_times.total_time);
        fprintf(fp, "  └─ T2 (stream): %8.4f s (%5.1f%%)\n",
               global_timing.t2_time,
               100.0 * global_timing.t2_time / prof_times.total_time);
        fprintf(fp, "------------------------------------\n");
        fprintf(fp, "End compress:     %8.4f s (%5.1f%%)\n", 
               prof_times.compress_end_time,
               100.0 * prof_times.compress_end_time / prof_times.total_time);
        fprintf(fp, "====================================\n");
        fprintf(fp, "TOTAL TIME:       %8.4f s\n", prof_times.total_time);
        fprintf(fp, "====================================\n\n");
    }
    
    if (log_fp) {
        fclose(log_fp);
        printf("Profiling results saved to: profiling_results.txt\n");
    }

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);

    return 0;
}
