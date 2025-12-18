#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#include <omp.h>

#include "openjpeg.h"
#include "profile_times.h"

extern double opj_clock(void);
extern struct TimingData global_timing;

typedef struct {
    double total_time;
    double load_image_time;
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
    prof->load_image_time = opj_clock() - t_start;
    
    return image;
}

int encode_single_image(const char* in_path, const char* out_path, int rank) {
    profile_times_t prof_times = {0};
    double total_start = opj_clock();

    // Reset global timing
    memset(&global_timing, 0, sizeof(global_timing));

    opj_image_t* image = load_pnm_as_image(in_path, &prof_times);
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
    prof_times.setup_time = opj_clock() - t_start;

    t_start = opj_clock();
    if (!opj_start_compress(codec, image, stream)) {
        fprintf(stderr, "opj_start_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof_times.compress_start_time = opj_clock() - t_start;

    t_start = opj_clock();
    if (!opj_encode(codec, stream)) {
        fprintf(stderr, "opj_encode failed\n");
        opj_end_compress(codec, stream);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof_times.encode_time = opj_clock() - t_start;

    t_start = opj_clock();
    if (!opj_end_compress(codec, stream)) {
        fprintf(stderr, "opj_end_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        opj_image_destroy(image);
        return 1;
    }
    prof_times.compress_end_time = opj_clock() - t_start;

    prof_times.total_time = opj_clock() - total_start;

    printf("[Rank %d] Encoded: %s -> %s\n", rank, in_path, out_path);
    printf("[Rank %d] Total: %.4fs, T1: %.4fs (%.1f%%), DWT: %.4fs (%.1f%%)\n",
           rank, prof_times.total_time,
           global_timing.t1_time, 100.0 * global_timing.t1_time / prof_times.total_time,
           global_timing.dwt_time, 100.0 * global_timing.dwt_time / prof_times.total_time);

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);
    opj_image_destroy(image);

    return 0;
}

int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s <input1.ppm> [input2.ppm ...]\n", argv[0]);
            fprintf(stderr, "Images will be distributed across %d MPI ranks\n", size);
        }
        MPI_Finalize();
        return 2;
    }

    int num_images = argc - 1;
    
    if (rank == 0) {
        printf("=== MPI + OpenMP J2K Encoder ===\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Total images: %d\n", num_images);
        printf("==================================\n\n");
    }

    double start_time = MPI_Wtime();

    // Distribute images across ranks
    for (int img_idx = rank + 1; img_idx < argc; img_idx += size) {
        const char* in_path = argv[img_idx];
        
        // Generate output filename
        char out_path[512];
        snprintf(out_path, sizeof(out_path), "output/mpi_rank%d_%d.j2k", rank, img_idx);
        
        encode_single_image(in_path, out_path, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\n==================================\n");
        printf("Total wall time: %.4f seconds\n", end_time - start_time);
        printf("==================================\n");
    }

    MPI_Finalize();
    return 0;
}
