#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define THRESH 100

typedef struct {
    unsigned char *data; 
    int Width;           
    int Height;          
} Image;

// Function to create a new image
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

// Function to free image data
void FreeImage(Image *image) {
    free(image->data);
    free(image);
}

// Function to convert RGB to grayscale
void rgb2gray(Image *image, unsigned char *rgbData) {
    for (int i = 0; i < image->Width * image->Height; ++i) {
        // Convert RGB to grayscale using luminosity method
        unsigned char gray = (unsigned char)(0.21 * rgbData[i * 3] + 0.72 * rgbData[i * 3 + 1] + 0.07 * rgbData[i * 3 + 2]);
        image->data[i] = gray;
    }
}

// Function to load an image from file
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
            fgetc(file); 
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

// Function to save an image to file in PGM format
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

__global__ void optimized_kernel(unsigned char *d_in, unsigned char *d_out, int width, int height, int widthStep) {
    __shared__ unsigned char tile[18][18]; 

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int local_col = threadIdx.x + 1;
    int local_row = threadIdx.y + 1;

    if (col < width && row < height) {
        tile[local_row][local_col] = d_in[row * widthStep + col];
    }
    __syncthreads();

    if (col < width && row < height) {
        int dx[3][3] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
        int dy[3][3] = { 1, 2, 1, 0, 0, 0, -1, -2, -1 };

        int sum_x = 0;
        int sum_y = 0;

        for (int m = -1; m <= 1; m++) {
            for (int n = -1; n <= 1; n++) {
                sum_x += tile[local_row + m][local_col + n] * dx[m + 1][n + 1];
                sum_y += tile[local_row + m][local_col + n] * dy[m + 1][n + 1];
            }
        }

        int sum = abs(sum_x) + abs(sum_y);
        sum = min(255, sum); 

        d_out[row * widthStep + col] = sum;
    }
}

int main() {
    Image *input_image = LoadImage("flowers.ppm");
    int width = input_image->Width;
    int height = input_image->Height;

    Image *output_image = CreateNewImage(width, height);

    unsigned char *d_in, *d_out;
    size_t imageSize = width * height * sizeof(unsigned char);
    cudaMalloc((void**)&d_in, imageSize);
    cudaMalloc((void**)&d_out, imageSize);

    cudaMemcpy(d_in, input_image->data, imageSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    clock_t start = clock();
    optimized_kernel<<<grid, block>>>(d_in, d_out, width, height, width);
    cudaDeviceSynchronize(); 
    clock_t end = clock();

    cudaMemcpy(output_image->data, d_out, imageSize, cudaMemcpyDeviceToHost);

    SaveImage("sobelGPU_optimized2.pgm", output_image);

    cudaFree(d_in);
    cudaFree(d_out);
    FreeImage(input_image);
    FreeImage(output_image);

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Execution Time: %f seconds\n", time_taken);

    return 0;
}
