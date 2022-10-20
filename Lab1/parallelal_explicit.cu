
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "data_helper.cuh"

#include <stdio.h>
#include <string>
#include <chrono>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5


float memsettime_explicit;
cudaEvent_t start_explicit, stop_explicit;

cudaError_t logicGateCudaExplicit(bool* output, bool* a, bool* b, char* gate, unsigned int size);

__global__ void logicGateKernelExplicit(bool* output, bool* a, bool* b, char* gate)
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
cudaError_t logicGateCudaExplicit(bool* output, bool* a, bool* b, char* gate, unsigned int size)
{
    bool* dev_a = 0;
    bool* dev_b = 0;
    char* dev_gate = 0;
    bool* dev_output = 0;
    cudaError_t cudaStatus;

    int num_blocks = (size / 1024) + 1;
    int num_threads = (size / num_blocks) + 1;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // initialize variables for timing data migration
    std::chrono::high_resolution_clock::time_point data_transfer_start, data_transfer_end;
    double data_migration_time;

    data_transfer_start = std::chrono::high_resolution_clock::now();
    // Allocate GPU buffers for four vectors (three input, one output)
    cudaMalloc((void**)&dev_output, size * sizeof(bool));
    cudaMalloc((void**)&dev_a, size * sizeof(bool));
    cudaMalloc((void**)&dev_b, size * sizeof(bool));
    cudaMalloc((void**)&dev_gate, size * sizeof(char));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_gate, gate, size * sizeof(char), cudaMemcpyHostToDevice);
    data_transfer_end = std::chrono::high_resolution_clock::now();
    
    // calculate and report the time taken for data migration
    data_migration_time = std::chrono::duration_cast<std::chrono::nanoseconds>(data_transfer_end - data_transfer_start).count() / 1000000.;
    printf("Time taken for data migration (host -> source) i.e., cudaMalloc + cudaMemcpy is: %f\n", data_migration_time);

    cudaEventRecord(start_explicit, 0);
    logicGateKernelExplicit<<< num_blocks, num_threads >>>(dev_output, dev_a, dev_b, dev_gate);
    cudaEventRecord(stop_explicit, 0);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "logicGateKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching logicGateKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output, dev_output, size * sizeof(bool), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_output);
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_gate);

    return cudaStatus;
}


int main(int argc, char* argv[])
{
    // perform input validation
    if (argc != 4) {
        printf("Usage: ./parallelal_explicit <input_file_name> <input_file_length> <output_file_name>\n");
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

    // load data from file into the arrays
    load_data(input_filename, a, b, gate, size);

    // initialize timer primitives
    cudaEventCreate(&start_explicit);
    cudaEventCreate(&stop_explicit);

    // Execute in parallel.
    cudaError_t cudaStatus = logicGateCudaExplicit(output, a, b, gate, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "logicGateCuda failed!");
        return 1;
    }

    // save output data into a file
    save_data(output_filename, output, size);

    // report the CUDA kernel execution time
    cudaEventSynchronize(stop_explicit);
    cudaEventElapsedTime(&memsettime_explicit, start_explicit, stop_explicit);
    printf("\n *** CUDA kernel execution time with explicit memory allocation: %f *** \n", memsettime_explicit);
    cudaEventDestroy(start_explicit);
    cudaEventDestroy(stop_explicit);

    cudaDeviceReset();
    return 0;
}