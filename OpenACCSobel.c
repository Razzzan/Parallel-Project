%%writefile sobel_openacc.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <openacc.h>

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

void rgb2gray(Image *image, unsigned char *rgbData) {
    for (int i = 0; i < image->Width * image->Height; ++i) {
        // Convert RGB to grayscale using luminosity method
        unsigned char gray = (unsigned char)(0.21 * rgbData[i * 3] + 0.72 * rgbData[i * 3 + 1] + 0.07 * rgbData[i * 3 + 2]);
        image->data[i] = gray;
    }
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

    Image *image = CreateNewImage(width, height);
    rgb2gray(image, rgbData);

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

void sobel_edge_detection_openacc(Image *input_image, Image *output_image) {
    int width = input_image->Width;
    int height = input_image->Height;
    unsigned char *in_data = input_image->data;
    unsigned char *out_data = output_image->data;

    // Define Sobel filter kernels
    const int dx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1} };
    const int dy[3][3] = { {1, 2, 1}, {0, 0, 0}, {-1, -2, -1} };

    #pragma acc data copyin(in_data[0:width*height]) copyout(out_data[0:width*height])
    {
        #pragma acc parallel loop collapse(2) 
        for (int row = 1; row < height-1; ++row) {
            for (int col = 1; col < width-1; ++col) {
                int sum_x = 0;
                int sum_y = 0;

                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        unsigned char pixel = in_data[(row + i) * width + (col + j)];
                        sum_x += pixel * dx[i + 1][j + 1];
                        sum_y += pixel * dy[i + 1][j + 1];
                    }
                }

                int magnitude = (int)sqrt(sum_x * sum_x + sum_y * sum_y);
                magnitude = (magnitude > 255) ? 255 : magnitude; 
                out_data[row * width + col] = (unsigned char)magnitude;
            }
        }
    }
}

int main() {
    const char *input_filename = "test.ppm";
    const char *output_filename = "output.pgm";

    Image *input_image = LoadImage(input_filename);

    Image *output_image = CreateNewImage(input_image->Width, input_image->Height);

    clock_t start = clock();

    sobel_edge_detection_openacc(input_image, output_image);

    clock_t end = clock();

    SaveImage(output_filename, output_image);

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution Time: %f seconds\n", time_taken);

    FreeImage(input_image);
    FreeImage(output_image);

    return 0;
}
