#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>

#define THRESH 100

typedef struct {
    unsigned char *data;
    int Width;
    int Height;
} Image;

Image *CreateNewImage(int width, int height) {
    Image *image = (Image *)malloc(sizeof(Image));
    if (image == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    image->Width = width;
    image->Height = height;
    image->data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (image->data == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    return image;
}

void FreeImage(Image *image) {
    free(image->data);
    free(image);
}

void rgb2gray(const unsigned char *rgbData, unsigned char *grayData, int width, int height) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            unsigned char gray = (unsigned char)(0.21 * rgbData[(y * width + x) * 3] +
                                                  0.72 * rgbData[(y * width + x) * 3 + 1] +
                                                  0.07 * rgbData[(y * width + x) * 3 + 2]);
            grayData[y * width + x] = gray;
        }
    }
}

Image *SobelEdgeDetection(const unsigned char *grayData, int width, int height) {
    static const int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    static const int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    Image *result = CreateNewImage(width, height);
    omp_set_num_threads(4);
    #pragma omp parallel for shared(grayData, result, sobel_x, sobel_y, width, height) default(none)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int pixel_value = grayData[(y + j) * width + (x + i)];
                    gx += pixel_value * sobel_x[j + 1][i + 1];
                    gy += pixel_value * sobel_y[j + 1][i + 1];
                }
            }
            result->data[y * width + x] = (gx * gx + gy * gy) < THRESH * THRESH ? 0 : 255;
        }
    }

    return result;
}

Image *LoadImage(const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error: Unable to open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    char magic_number[3];
    fscanf(file, "%2s", magic_number);
    int width, height, max_value;
    fscanf(file, "%d %d %d", &width, &height, &max_value);
    fgetc(file); // Consume newline

    unsigned char *rgbData;
    if (magic_number[1] == '6') { // PPM binary format
        rgbData = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
        if (!rgbData) {
            printf("Error: Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        fread(rgbData, sizeof(unsigned char), width * height * 3, file);
    } else if (magic_number[1] == '3') { // PPM ASCII format
        rgbData = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));
        if (!rgbData) {
            printf("Error: Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < width * height * 3; ++i) {
            fscanf(file, "%hhu", &rgbData[i]);
            fgetc(file); // Consume whitespace
        }
    } else {
        printf("Error: Unsupported image format\n");
        exit(EXIT_FAILURE);
    }

    fclose(file);

    unsigned char *grayData = (unsigned char *)malloc(width * height * sizeof(unsigned char));
    if (!grayData) {
        printf("Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }
    rgb2gray(rgbData, grayData, width, height);

    Image *image = CreateNewImage(width, height);
    image->data = grayData;

    free(rgbData);

    return image;
}

void SaveImage(const char *filename, Image *image) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        printf("Error: Unable to open file %s for writing\n", filename);
        exit(EXIT_FAILURE);
    }

    fprintf(file, "P5\n%d %d\n255\n", image->Width, image->Height);
    fwrite(image->data, 1, image->Width * image->Height, file);

    fclose(file);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Image *input_image;
    clock_t start, end;
    if (rank == 0) {
        input_image = LoadImage("flowers.ppm");
    }

    int width, height;
    MPI_Bcast(&input_image->Width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&input_image->Height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    width = input_image->Width;
    height = input_image->Height;

    int chunk_height = height / num_procs;
    int chunk_size = width * chunk_height;

    unsigned char *local_data = (unsigned char *)malloc(chunk_size * sizeof(unsigned char));
    if (!local_data) {
        printf("Error: Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    MPI_Scatter(input_image->data, chunk_size, MPI_UNSIGNED_CHAR, local_data, chunk_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    Image *local_image = CreateNewImage(width, chunk_height);
    local_image->data = local_data;

    start = clock();
    Image *edges = SobelEdgeDetection(local_image->data, width, chunk_height);
    end = clock();

    unsigned char *gathered_data = NULL;
    if (rank == 0) {
        gathered_data = (unsigned char *)malloc(width * height * sizeof(unsigned char));
        if (!gathered_data) {
            printf("Error: Memory allocation failed\n");
            exit(EXIT_FAILURE);
        }
    }

    MPI_Gather(edges->data, chunk_size, MPI_UNSIGNED_CHAR, gathered_data, chunk_size, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Image *output_image = CreateNewImage(width, height);
        output_image->data = gathered_data;
        SaveImage("sobelHybrid.pgm", output_image);
        FreeImage(output_image);
    }

    FreeImage(edges);
    FreeImage(local_image);

    if (rank == 0) {
        double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
        printf("Execution Time: %f seconds\n", time_taken);
    }

    MPI_Finalize();

    return 0;
}
