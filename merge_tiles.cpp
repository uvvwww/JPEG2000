/**
 * Merge horizontal tile PPM files into a single PPM
 * Usage: merge_tiles <output.ppm> <tile0.ppm> <tile1.ppm> ...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

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

struct TileInfo {
    int width;
    int height;
    int maxval;
    int is_ppm;
    unsigned char* data;
};

static TileInfo* load_ppm(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open: %s\n", path);
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
        fprintf(stderr, "Only P5/P6 supported (got %s)\n", tok);
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
        fprintf(stderr, "Invalid PNM header\n");
        fclose(fp);
        return NULL;
    }

    size_t bytes_per_pixel = is_ppm ? 3 : 1;
    size_t data_size = (size_t)width * height * bytes_per_pixel;

    unsigned char* data = (unsigned char*)malloc(data_size);
    if (!data) {
        fclose(fp);
        return NULL;
    }

    size_t nread = fread(data, 1, data_size, fp);
    fclose(fp);

    if (nread != data_size) {
        fprintf(stderr, "Incomplete read: expected %zu, got %zu\n", data_size, nread);
        free(data);
        return NULL;
    }

    TileInfo* info = (TileInfo*)malloc(sizeof(TileInfo));
    info->width = width;
    info->height = height;
    info->maxval = maxval;
    info->is_ppm = is_ppm;
    info->data = data;

    return info;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <output.ppm> <tile0.ppm> <tile1.ppm> ...\n", argv[0]);
        fprintf(stderr, "Merges horizontal tiles (stacked vertically) into one image\n");
        return 1;
    }

    const char* out_path = argv[1];
    int num_tiles = argc - 2;

    printf("Merging %d tiles into %s\n", num_tiles, out_path);

    // Load all tiles
    TileInfo** tiles = (TileInfo**)malloc(num_tiles * sizeof(TileInfo*));
    int total_height = 0;
    int width = 0;
    int is_ppm = 0;

    for (int i = 0; i < num_tiles; i++) {
        tiles[i] = load_ppm(argv[i + 2]);
        if (!tiles[i]) {
            fprintf(stderr, "Failed to load tile: %s\n", argv[i + 2]);
            return 1;
        }
        printf("  Tile %d: %s (%dx%d)\n", i, argv[i + 2], tiles[i]->width, tiles[i]->height);

        if (i == 0) {
            width = tiles[i]->width;
            is_ppm = tiles[i]->is_ppm;
        } else {
            if (tiles[i]->width != width) {
                fprintf(stderr, "Width mismatch: tile %d has width %d, expected %d\n", 
                        i, tiles[i]->width, width);
                return 1;
            }
            if (tiles[i]->is_ppm != is_ppm) {
                fprintf(stderr, "Format mismatch between tiles\n");
                return 1;
            }
        }
        total_height += tiles[i]->height;
    }

    printf("Output image: %dx%d\n", width, total_height);

    // Write merged output
    FILE* fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot create output: %s\n", out_path);
        return 1;
    }

    fprintf(fp, "%s\n%d %d\n255\n", is_ppm ? "P6" : "P5", width, total_height);

    size_t bytes_per_pixel = is_ppm ? 3 : 1;
    for (int i = 0; i < num_tiles; i++) {
        size_t tile_size = (size_t)tiles[i]->width * tiles[i]->height * bytes_per_pixel;
        fwrite(tiles[i]->data, 1, tile_size, fp);
        free(tiles[i]->data);
        free(tiles[i]);
    }
    free(tiles);

    fclose(fp);
    printf("Merged successfully: %s\n", out_path);

    return 0;
}
