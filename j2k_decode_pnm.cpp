#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "openjpeg.h"

/* Declare internal timing function from OpenJPEG */
extern double opj_clock(void);

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
    
    if (numcomps == 3) {
        // Allocate buffer for entire image to enable single fwrite
        unsigned char* buffer = (unsigned char*)malloc(pixels * 3);
        if (!buffer) {
            fclose(fp);
            return 0;
        }
        
        int rshift = (int)image->comps[0].prec - 8;
        int gshift = (int)image->comps[1].prec - 8;
        int bshift = (int)image->comps[2].prec - 8;
        
        for (size_t i = 0; i < pixels; i++) {
            int r = image->comps[0].data[i];
            int g = image->comps[1].data[i];
            int b = image->comps[2].data[i];
            
            if (rshift > 0) r >>= rshift;
            if (gshift > 0) g >>= gshift;
            if (bshift > 0) b >>= bshift;
            
            buffer[i * 3 + 0] = clamp_u8(r);
            buffer[i * 3 + 1] = clamp_u8(g);
            buffer[i * 3 + 2] = clamp_u8(b);
        }
        
        size_t written = fwrite(buffer, 1, pixels * 3, fp);
        free(buffer);
        if (written != pixels * 3) {
            fclose(fp);
            return 0;
        }
    } else {
        // Allocate buffer for grayscale
        unsigned char* buffer = (unsigned char*)malloc(pixels);
        if (!buffer) {
            fclose(fp);
            return 0;
        }
        
        int shift = (int)image->comps[0].prec - 8;
        for (size_t i = 0; i < pixels; i++) {
            int v = image->comps[0].data[i];
            if (shift > 0) v >>= shift;
            buffer[i] = clamp_u8(v);
        }
        
        size_t written = fwrite(buffer, 1, pixels, fp);
        free(buffer);
        if (written != pixels) {
            fclose(fp);
            return 0;
        }
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

    opj_dparameters_t parameters;
    opj_set_default_decoder_parameters(&parameters);

    opj_codec_t* codec = opj_create_decompress(OPJ_CODEC_J2K);
    if (!codec) {
        return 1;
    }

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

    opj_image_t* image = NULL;
    if (!opj_read_header(stream, codec, &image)) {
        fprintf(stderr, "opj_read_header failed\n");
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    double t_start = opj_clock();

    if (!opj_decode(codec, stream, image)) {
        fprintf(stderr, "opj_decode failed\n");
        opj_image_destroy(image);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    if (!opj_end_decompress(codec, stream)) {
        fprintf(stderr, "opj_end_decompress failed\n");
        opj_image_destroy(image);
        opj_stream_destroy(stream);
        opj_destroy_codec(codec);
        return 1;
    }

    double t_end = opj_clock();
    fprintf(stdout, "Decoding time: %f seconds\n", t_end - t_start);

    int ok = write_pnm_u8(out_path, image);
    if (!ok) {
        fprintf(stderr, "Failed to write output PNM\n");
    }

    opj_image_destroy(image);
    opj_stream_destroy(stream);
    opj_destroy_codec(codec);

    return ok ? 0 : 1;
}
