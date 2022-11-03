
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "gputimer.h"
#include "img_helper.cuh"
#include "weight_matrix.cuh"

#include <stdio.h>
#include <math.h>
#include <string>


// function declaration
cudaError_t convolve(unsigned char* out, unsigned char* in, float[][3] w, unsigned int width, unsigned int height, unsigned int size, unsigned int out_size, unsigned int threads_per_block, struct GpuTimer* timer, float* timeElapsed);


__global__ void convolveKernel(unsigned char* out, unsigned char* in, float[][3] w, unsigned int width, unsigned int height, unsigned int size)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);
    int x = ((int)(idx / 4) % width);
    int y = (int)(idx / (width * 4));
    //int out_idx = (idx/4) - width + (x-1);
    int out_idx = ((idx/width) - 1) * (4 * (width - 2)) + (x - 1);
    if (x != 0 && y != 0 && x != width-1 && y != height-1 && idx < size && idx%4 != 3) {
        unsigned char top = idx - (4 * width);
        unsigned char top_left = top - 4;
        unsigned char top_right = top + 4;
        unsigned char middle = idx;
        unsigned char middle_left = middle - 4;
        unsigned char middle_right = middle + 4;
        unsigned char bottom = idx + (4 * width);
        unsigned char bottom_left = bottom - 4;
        unsigned char bottom_right = bottom + 4;

        unsigned char o = w[0][0] * in[top_left]       + w[0][1] * in[top]     + w[0][2] * in[top_right] + 
                          w[1][0] * in[middle_left]    + w[1][1] * in[middle]  + w[1][2] * in[middle_right] +
                          w[2][0] * in[bottom_left]    + w[2][2] * in[bottom]  + w[2][2] * in[bottom_right];
        out[out_idx] = max(0, min(255, o));
    }
    else {
        out[out_idx] = in[idx];
    }
}


// Helper function for using CUDA to add vectors in parallel.
cudaError_t convolve(unsigned char* out, unsigned char* in, float[][3] w, unsigned int width, unsigned int height, unsigned int size, unsigned int out_size, unsigned int threads_per_block, struct GpuTimer* timer, float* timeElapsed)
{
    unsigned char* dev_in = 0;
    unsigned char* dev_out = 0;
    float* dev_w = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (input image, weight matrix, output image)
    cudaStatus = cudaMalloc((void**)&dev_out, (out_size) * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_in, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_w, (3 * 3) * sizeof(float));
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

    cudaStatus = cudaMemcpy(dev_w, w, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU and time it
    timer->Start();
    convolveKernel << < (size + threads_per_block - 1) / threads_per_block, threads_per_block >> > (dev_out, dev_in, dev_w, width, height, size);
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
    cudaStatus = cudaMemcpy(out, dev_out, (out_size) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_out);
    cudaFree(dev_in);
    cudaFree(dev_w);

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


int main(int argc, char** argv)
{
    if (argc != 5) {
        printf("Usage: ./convolve <input_image> <output_image> <gold_standard_output_filename> <num_threads>");
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
        unsigned int width, height, size, out_size;
        unsigned char* image = { 0 };
        //unsigned char* image = (unsigned char*) malloc(size * sizeof(unsigned char));
        load_image(input_filename, &image, &width, &height, &size);
        out_size = (width - 2) * (height - 2);

        // perform pooling
        unsigned char* image_out = (unsigned char*)malloc((out_size) * sizeof(unsigned char));
        convolve(image_out, image, w, width, height, size, out_size, NUM_THREADS, timer, &timeElapsed);
        printf("Pooling computation time: %f\n", timeElapsed);
        totalTimeElapsed += timeElapsed;

        // save the image
        save_image(output_filename, image_out, width - 2, height - 2);

        // compare the generated image to the provided correct output image
        printf("MSE between the generated output and given output: %f\n", get_MSE(output_filename, gold_standard_output_filename));

        // free memory
        free(image);
        free(image_out);
    }

    // obtain the average total time for performing the pooling operation
    printf("Average computation time for preforming pooling: %f\n", totalTimeElapsed / 10.0);
}