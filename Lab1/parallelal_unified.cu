#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_helper.cuh"

#include <stdio.h>
#include <string>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

float memsettime_unified;
cudaEvent_t start_unified, stop_unified;

cudaError_t logicGateCudaUnified(bool* output, bool* a, bool* b, char* gate, unsigned int size);

__global__ void logicGateKernelUnified(bool* output, bool* a, bool* b, char* gate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    switch (gate[i])
    {
    case AND:
        output[i] = a[i] && b[i];
        break;
    case OR:
        output[i] = a[i] || b[i];
        break;
    case NAND:
        output[i] = !(a[i] && b[i]);
        break;
    case NOR:
        output[i] = !(a[i] || b[i]);
        break;
    case XOR:
        output[i] = (a[i] || b[i]) && (!a[i] || !b[i]);
        break;
    case XNOR:
        output[i] = !((a[i] || b[i]) && (!a[i] || !b[i]));
        break;
    default:
        printf("Error: Gate not specified.\n");
    }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t logicGateCudaUnified(bool* output, bool* a, bool* b, char* gate, unsigned int size)
{
    cudaError_t cudaStatus;
    int num_blocks = (size / 1024) + 1;
    int num_threads = (size / num_blocks) + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        return cudaStatus;
    }

    cudaEventRecord(start_unified, 0);
    logicGateKernelUnified<<<num_blocks, num_threads>>>(output, a, b, gate);
    cudaEventRecord(stop_unified, 0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "logicGateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching logicGateKernel!\n", cudaStatus);
        return cudaStatus;
    }
}


int main(int argc, char* argv[])
{
    // perform input validation
    if (argc != 4) {
        printf("Usage: ./parallelal_unified <input_file_name> <input_file_length> <output_file_name>\n");
        return 1;
    }

    // parse input arguments
    char* input_filename = argv[1];
    char* output_filename = argv[3];
    unsigned int size = std::stoi(argv[2]);

    // dynamically allocate memory for the arrays
    bool* a = (bool*)calloc(size, sizeof(bool));
    bool* b = (bool*)calloc(size, sizeof(bool));
    char* gate = (char*)calloc(size, sizeof(char));
    bool* output = (bool*)calloc(size, sizeof(bool));

    // initialize timer primitives
    cudaEventCreate(&start_unified);
    cudaEventCreate(&stop_unified);

    // allocate unified memory
    cudaMallocManaged((void**)&output, size * sizeof(bool));
    cudaMallocManaged((void**)&a, size * sizeof(bool));
    cudaMallocManaged((void**)&b, size * sizeof(bool));
    cudaMallocManaged((void**)&gate, size * sizeof(char));

    // load data from file into the arrays
    load_data(input_filename, a, b, gate, size);

    // Execute in parallel.
    cudaError_t cudaStatus = logicGateCudaUnified(output, a, b, gate, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "logicGateCuda failed!");
        return 1;
    }

    // save output data into a file
    save_data(output_filename, output, size);

    // report the CUDA kernel execution time
    cudaEventSynchronize(stop_unified);
    cudaEventElapsedTime(&memsettime_unified, start_unified, stop_unified);
    printf("\n *** CUDA kernel execution time with unified memory allocation: %f *** \n", memsettime_unified);
    cudaEventDestroy(start_unified);
    cudaEventDestroy(stop_unified);

    cudaFree(output);
    cudaFree(a);
    cudaFree(b);
    cudaFree(gate);

    cudaDeviceReset();
    return 0;
}