#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "img_helper.cuh"
#include "gputimer.h"

#include <stdio.h>
#include <math.h>
#include <string>

#define MAX_MSE 0.00001f


//cudaError_t pool(int* out, const int* in, unsigned int size);
cudaError_t pool(unsigned char* image_out, unsigned char* image_in, unsigned int width, unsigned int size, unsigned int threads_per_block, struct GpuTimer* timer, float* timeElapsed);

__global__ void poolKernel(unsigned char* out, unsigned char* in, unsigned int width, unsigned int size)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = (idx/4) % width;
    int y = (int)idx / (width * 4);
    int out_idx = (idx / 4) + x + (idx % 4);
    if (y % 2 == 0 && idx%8 < 4 && idx < size && out_idx < size/4) {
        out[out_idx] = max(max(in[idx], in[idx + 4]), max(in[idx + (4*width)], in[idx + (4*width) + 4]));
    }
}


// Helper function for using CUDA to perform 2 x 2 max pooling on a single image channel.
cudaError_t pool(unsigned char* out, unsigned char* in, unsigned int width, unsigned int size, unsigned int threads_per_block, struct GpuTimer* timer, float* timeElapsed)
{
    unsigned char* dev_in = 0;
    unsigned char* dev_out = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_out, (size / 4) * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }
  
    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in, in, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU and time it
    timer->Start();
    poolKernel << < (size + threads_per_block - 1) / threads_per_block, threads_per_block >> > (dev_out, dev_in, width, size);
    timer->Stop();

    // record the computation time
    *timeElapsed = timer->Elapsed();

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(out, dev_out, (size / 4) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_out);
    cudaFree(dev_in);

    return cudaStatus;
}


float get_MSE(char* input_filename_1, char* input_filename_2)
{
    unsigned error1, error2;
    unsigned char* image1, * image2;
    unsigned width1, height1, width2, height2;

    error1 = lodepng_decode32_file(&image1, &width1, &height1, input_filename_1);
    error2 = lodepng_decode32_file(&image2, &width2, &height2, input_filename_2);
    if (error1) printf("error %u: %s\n", error1, lodepng_error_text(error1));
    if (error2) printf("error %u: %s\n", error2, lodepng_error_text(error2));
    if (width1 != width2) printf("images do not have same width\n");
    if (height1 != height2) printf("images do not have same height\n");

    // process image
    float im1, im2, diff, sum, MSE;
    sum = 0;
    for (int i = 0; i < width1 * height1; i++) {
        im1 = (float)image1[i];
        im2 = (float)image2[i];
        diff = im1 - im2;
        sum += diff * diff;
    }
    MSE = sqrt(sum) / (width1 * height1);

    free(image1);
    free(image2);

    return MSE;
}


int main(int argc, char** argv) {
    if (argc != 5) {
        printf("Usage: ./pool <input_image> <output_image> <gold_standard_output_filename> <num_threads>");
        exit(1);
    }

    // load command args
    char* input_filename = argv[1];
    char* output_filename = argv[2];
    char* gold_standard_output_filename = argv[3];
    unsigned int NUM_THREADS = std::stoi(argv[4]);

    if (NUM_THREADS > 1024) {
        printf("Maximum number of threads can be 1024. Aborting...");
        exit(1);
    }

    float totalTimeElapsed = 0.0;

    // repeat the operation 10 times
    for (int i = 0; i < 10; i++) {
        // initialize timer
        float timeElapsed = 0.0;
        struct GpuTimer* timer = new GpuTimer();

        // load the image
        unsigned int width, height, size;
        unsigned char* image = { 0 };
        //unsigned char* image = (unsigned char*) malloc(size * sizeof(unsigned char));
        load_image(input_filename, &image, &width, &height, &size);

        // perform pooling
        unsigned char* image_out = (unsigned char*)malloc((size / 4) * sizeof(unsigned char));
        pool(image_out, image, width, size, NUM_THREADS, timer, &timeElapsed);
        printf("Pooling computation time: %f\n", timeElapsed);
        totalTimeElapsed += timeElapsed;

        // save the image
        save_image(output_filename, image_out, width / 2, height / 2);

        // compare the generated image to the provided correct output image
        printf("MSE between the generated output and given output: %f\n", get_MSE(output_filename, gold_standard_output_filename));

        // free memory
        free(image);
        free(image_out);
    }

    // obtain the average total time for performing the pooling operation
    printf("Average computation time for preforming pooling: %f\n", totalTimeElapsed/10.0);
}