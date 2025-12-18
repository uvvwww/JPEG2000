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

    /* Improve throughput on large PPM/PGM (best-effort) */
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

    // Optional: enable tiling via environment variables
    const char* env_tile_w = getenv("J2K_TILE_W");
    const char* env_tile_h = getenv("J2K_TILE_H");
    if (env_tile_w && env_tile_h) {
        int tile_w = atoi(env_tile_w);
        int tile_h = atoi(env_tile_h);
        if (tile_w > 0 && tile_h > 0) {
            parameters.tile_size_on = OPJ_TRUE;
            parameters.cp_tdx = tile_w;
            parameters.cp_tdy = tile_h;
            if (rank == 0) {
                printf("Using tiling: %dx%d (env)\n", tile_w, tile_h);
            }
        }
    }

    // Optional: override codeblock size for granularity
    const char* env_cblkw = getenv("J2K_CBLKW");
    const char* env_cblkh = getenv("J2K_CBLKH");
    if (env_cblkw && env_cblkh) {
        int cblkw = atoi(env_cblkw);
        int cblkh = atoi(env_cblkh);
        if (cblkw > 0 && cblkh > 0) {
            parameters.cblockw_init = cblkw;
            parameters.cblockh_init = cblkh;
            if (rank == 0) {
                printf("Using codeblock: %dx%d (env)\n", cblkw, cblkh);
            }
        }
    }

    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    if (!codec) {
        opj_image_destroy(image);
        return 1;
    }

    opj_set_error_handler(codec, error_callback, NULL);
    opj_set_warning_handler(codec, warning_callback, NULL);
    opj_set_info_handler(codec, info_callback, NULL);

    // Set number of threads for parallel encoding per rank
    int num_threads = omp_get_max_threads();
    const char* env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads) {
        int t = atoi(env_threads);
        if (t > 0) num_threads = t;
    }
    if (num_threads > 0) {
        if (!opj_codec_set_threads(codec, num_threads)) {
            if (rank == 0) fprintf(stderr, "Warning: Failed to set %d threads\n", num_threads);
        } else {
            if (rank == 0) fprintf(stdout, "Using %d threads for encoding per rank\n", num_threads);
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

// Encode from already-loaded image in memory (for tile parallel mode)
static int encode_single_image_from_memory(opj_image_t* image, const char* out_path, int rank) {
    profile_times_t prof_times = {0};
    double t_start, total_start = opj_clock();

    t_start = opj_clock();
    opj_cparameters_t parameters;
    opj_set_default_encoder_parameters(&parameters);
    parameters.tcp_rates[0] = 0;
    parameters.tcp_numlayers = 1;
    parameters.cp_disto_alloc = 1;
    parameters.numresolution = 6;

    opj_codec_t* codec = opj_create_compress(OPJ_CODEC_J2K);
    if (!codec) {
        return 1;
    }

    opj_set_error_handler(codec, error_callback, NULL);
    opj_set_warning_handler(codec, warning_callback, NULL);
    opj_set_info_handler(codec, info_callback, NULL);

    // Set number of threads for parallel encoding per rank
    int num_threads = omp_get_max_threads();
    const char* env_threads = getenv("OMP_NUM_THREADS");
    if (env_threads) {
        int t = atoi(env_threads);
        if (t > 0) num_threads = t;
    }
    if (num_threads > 0) {
        if (!opj_codec_set_threads(codec, num_threads)) {
            if (rank == 0) fprintf(stderr, "Warning: Failed to set %d threads\n", num_threads);
        }
    }

    opj_stream_t* stream = opj_stream_create_default_file_stream(out_path, OPJ_FALSE);
    if (!stream) {
        opj_destroy_codec(codec);
        return 1;
    }

    if (!opj_setup_encoder(codec, &parameters, image)) {
        fprintf(stderr, "opj_setup_encoder failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }
    prof_times.setup_time = opj_clock() - t_start;

    t_start = opj_clock();
    if (!opj_start_compress(codec, image, stream)) {
        fprintf(stderr, "opj_start_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }
    prof_times.compress_start_time = opj_clock() - t_start;

    t_start = opj_clock();
    if (!opj_encode(codec, stream)) {
        fprintf(stderr, "opj_encode failed\n");
        opj_end_compress(codec, stream);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }
    prof_times.encode_time = opj_clock() - t_start;

    t_start = opj_clock();
    if (!opj_end_compress(codec, stream)) {
        fprintf(stderr, "opj_end_compress failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }
    prof_times.compress_end_time = opj_clock() - t_start;

    prof_times.total_time = opj_clock() - total_start;

    printf("[Rank %d] Encoded tile -> %s\n", rank, out_path);
    printf("[Rank %d] Total: %.4fs, T1: %.4fs (%.1f%%), DWT: %.4fs (%.1f%%)\n",
           rank, prof_times.total_time,
           global_timing.t1_time, 100.0 * global_timing.t1_time / prof_times.total_time,
           global_timing.dwt_time, 100.0 * global_timing.dwt_time / prof_times.total_time);

    opj_stream_destroy(stream);
    opj_destroy_codec(codec);

    return 0;
}

// Define TILE_PARALLEL to enable tile-based parallelization of a single image
// Otherwise, use multi-image parallelization (default)
// Compile with: -DTILE_PARALLEL to enable tile mode

#ifdef TILE_PARALLEL

// Tile-based parallel: split single image across MPI ranks
int main(int argc, char** argv) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage (TILE_PARALLEL mode): %s <input.ppm> <output_prefix>\n", argv[0]);
            fprintf(stderr, "Each of %d MPI ranks will encode a horizontal tile\n", size);
        }
        MPI_Finalize();
        return 2;
    }

    const char* in_path = argv[1];
    const char* out_prefix = argv[2];
    
    if (rank == 0) {
        printf("=== MPI + OpenMP J2K Encoder (TILE_PARALLEL MODE) ===\n");
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads per rank: %d\n", omp_get_max_threads());
        printf("Input: %s\n", in_path);
        printf("Each rank encodes a horizontal tile (row-based split)\n");
        printf("======================================================\n\n");
    }

    double start_time = MPI_Wtime();

    opj_image_t* full_image = NULL;
    int width = 0, height = 0, numcomps = 0;

    // Rank 0 reads the full image
    if (rank == 0) {
        profile_times_t dummy_prof = {0,0,0,0,0,0};
        full_image = load_pnm_as_image(in_path, &dummy_prof);
        if (!full_image) {
            fprintf(stderr, "[Rank 0] Failed to load image\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        width = full_image->x1 - full_image->x0;
        height = full_image->y1 - full_image->y0;
        numcomps = full_image->numcomps;
        printf("[Rank 0] Loaded image: %dx%d, %d components\n", width, height, numcomps);
    }

    // Broadcast image dimensions
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numcomps, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate tile height for each rank
    int tile_height = height / size;
    int remainder = height % size;
    int my_tile_height = tile_height + (rank < remainder ? 1 : 0);
    int my_start_row = rank * tile_height + (rank < remainder ? rank : remainder);

    printf("[Rank %d] Tile: rows %d to %d (height=%d)\n", 
           rank, my_start_row, my_start_row + my_tile_height - 1, my_tile_height);

    // Create tile image structure
    opj_image_cmptparm_t cmptparms[3];
    memset(cmptparms, 0, sizeof(cmptparms));
    for (int i = 0; i < numcomps; i++) {
        cmptparms[i].dx = 1;
        cmptparms[i].dy = 1;
        cmptparms[i].w = width;
        cmptparms[i].h = my_tile_height;
        cmptparms[i].x0 = 0;
        cmptparms[i].y0 = 0;
        cmptparms[i].prec = 8;
        cmptparms[i].bpp = 8;
        cmptparms[i].sgnd = 0;
    }

    opj_image_t* tile_image = opj_image_create(numcomps, cmptparms, 
                                                numcomps == 3 ? OPJ_CLRSPC_SRGB : OPJ_CLRSPC_GRAY);
    if (!tile_image) {
        fprintf(stderr, "[Rank %d] Failed to create tile image\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    tile_image->x0 = 0;
    tile_image->y0 = 0;
    tile_image->x1 = width;
    tile_image->y1 = my_tile_height;

    // Distribute tile data
    int pixels_per_tile = width * my_tile_height;
    
    if (rank == 0) {
        // Rank 0: copy own tile
        for (int c = 0; c < numcomps; c++) {
            for (int row = 0; row < my_tile_height; row++) {
                memcpy(&tile_image->comps[c].data[row * width],
                       &full_image->comps[c].data[row * width],
                       width * sizeof(OPJ_INT32));
            }
        }
        
        // Send tiles to other ranks
        for (int r = 1; r < size; r++) {
            int r_tile_height = tile_height + (r < remainder ? 1 : 0);
            int r_start_row = r * tile_height + (r < remainder ? r : remainder);
            int r_pixels = width * r_tile_height;
            
            for (int c = 0; c < numcomps; c++) {
                MPI_Send(&full_image->comps[c].data[r_start_row * width],
                         r_pixels, MPI_INT, r, c, MPI_COMM_WORLD);
            }
        }
        
        opj_image_destroy(full_image);
    } else {
        // Other ranks: receive tile data
        for (int c = 0; c < numcomps; c++) {
            MPI_Recv(tile_image->comps[c].data, pixels_per_tile, MPI_INT,
                     0, c, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Each rank encodes its own tile
    char out_path[512];
    snprintf(out_path, sizeof(out_path), "%s_tile%d.j2k", out_prefix, rank);
    
    encode_single_image_from_memory(tile_image, out_path, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("\n======================================================\n");
        printf("Total wall time: %.4f seconds\n", end_time - start_time);
        printf("Output files: %s_tile0.j2k to %s_tile%d.j2k\n", 
               out_prefix, out_prefix, size - 1);
        printf("======================================================\n");
    }

    opj_image_destroy(tile_image);
    MPI_Finalize();
    return 0;
}

#else

// Default: Multi-image parallel (each rank encodes different images)
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

#endif  // TILE_PARALLEL
