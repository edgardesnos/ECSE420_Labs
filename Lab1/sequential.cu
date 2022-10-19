
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

cudaError_t logicGateCuda(bool *output, bool *a, bool *b, char *gate, unsigned int size);
bool gateOutput(bool a, bool b, char gate);

__global__ void logicGateKernel(bool *output, bool *a, bool *b, char *gate)
{
    int i = threadIdx.x;
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

void logicGateSequential(bool* output, bool* a, bool* b, char* gate, int length) 
{
    for (int i=0; i<length; i++) 
    {
        output[i] = gateOutput(a[i], b[i], gate[i]);
    }
}

bool gateOutput(bool a, bool b, char gate)
{
    bool output;
    switch (gate)
    {
    case AND:
        output = a && b;
        break;
    case OR:
        output = a || b;
        break;
    case NAND:
        output = !(a && b);
        break;
    case NOR:
        output = !(a || b);
        break;
    case XOR:
        output = (a || b) && (!a || !b);
        break;
    case XNOR:
        output = !((a || b) && (!a || !b));
        break;
    default:
        printf("Error: Gate not specified.\n");
    }
    return output;
}


int main(int argc, char* argv[])
{
    if (argc != 4) {
        printf("Usage: ./sequential <input_file_name> <input_file_length> <output_file_name>\n");
        return 1;
    }

    //Open file input_file_name with length input_file_length
    //Read contents of file into arrays a, b and gate

    const int arraySize = 4;
    bool a[arraySize] = { 0, 1, 0, 1 };
    bool b[arraySize] = { 0, 0, 1, 1 };
    char gate[arraySize] = { XNOR, AND, OR, NOR };
    bool output[arraySize] = { 0 };

    //logicGateSequential(output, a, b, gate, arraySize);

    // Execute in parallel.
    cudaError_t cudaStatus = logicGateCuda(output, a, b, gate, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "logicGateCuda failed!");
        return 1;
    }

    printf("Output array : {%d,%d,%d,%d}\n",
        output[0], output[1], output[2], output[3]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t logicGateCuda(bool *output, bool *a, bool *b, char *gate, unsigned int size)
{
    bool *dev_a = 0;
    bool *dev_b = 0;
    char *dev_gate = 0;
    bool *dev_output = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_output, size * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(bool));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_gate, size * sizeof(char));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(bool), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_gate, gate, size * sizeof(char), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    logicGateKernel<<<1, size>>>(dev_output, dev_a, dev_b, dev_gate);

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
