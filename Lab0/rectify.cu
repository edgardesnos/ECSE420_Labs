
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.cuh"

#include <stdio.h>
#include <string>

#define MAX_MSE 0.00001f

float memsettime;
cudaEvent_t start, stop;


cudaError_t rectifyWithCuda(unsigned char *outputImage, unsigned char *inputImage, unsigned int size, unsigned int threads);

__global__ void rectifyKernel(unsigned char *outputImage, unsigned char *inputImage)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    outputImage[i] = max(inputImage[i], 127);
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


int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Usage: ./rectify <name of input png> <name of output png> < # threads>\n");
        return 1;
    }
    char *inputFilename = argv[1];
    char *outputFilename = argv[2];
    const int threads = atoi(argv[3]);

    unsigned error;
    unsigned char *inputImage, *outputImage;
    unsigned width, height;

    error = lodepng_decode32_file(&inputImage, &width, &height, inputFilename);
    if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
    outputImage = (unsigned char*) malloc(width * height * 4 * sizeof(unsigned char));


    const int arraySize = width * height * 4;
    // Rectify image in parallel.
    cudaError_t cudaStatus = rectifyWithCuda(outputImage, inputImage, arraySize, threads);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rectifyWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    lodepng_encode32_file(outputFilename, outputImage, width, height);

    free(inputImage);
    free(outputImage);

    //char* rectifiedFilename = "./Test Images/Test_1_rectified.png";

    //// get mean squared error between image1 and image2
    //float MSE = get_MSE(rectifiedFilename, outputFilename);

    //if (MSE < MAX_MSE) {
    //    printf("Images are equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
    //}
    //else {
    //    printf("Images are NOT equal (MSE = %f, MAX_MSE = %f)\n", MSE, MAX_MSE);
    //}

    return 0;
}

// Helper function for using CUDA to rectify image in parallel.
cudaError_t rectifyWithCuda(unsigned char *outputImage, unsigned char *inputImage, unsigned int size, unsigned int threads)
{
    unsigned char *dev_inputImage = 0;
    unsigned char *dev_outputImage = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for 2 images (one input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_outputImage, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inputImage, size * sizeof(unsigned char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input image from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_inputImage, inputImage, size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    // Launch a kernel on the GPU with specified number of threads for each element.
    cudaEventRecord(start, 0);
    rectifyKernel<<<(size + threads - 1)/threads, threads>>>(dev_outputImage, dev_inputImage);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop);
    printf(" *** CUDA execution time: %f *** \n", memsettime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "rectifyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rectifyKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output image from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(outputImage, dev_outputImage, size * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_outputImage);
    cudaFree(dev_inputImage);
    
    return cudaStatus;
}
