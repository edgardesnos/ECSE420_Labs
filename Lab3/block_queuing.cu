#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "io_helper.cuh"
#include "compare.cuh"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5


/*
*
* Function to compute the output given a logical gate and the corresponding two input values.
*/
__device__ int gate_solver_parallel_block(int gateType, int inp1, int inp2) {
	int output;
	switch (gateType)
	{
	case AND:
		output = inp1 && inp2;
		break;
	case OR:
		output = inp1 || inp2;
		break;
	case NAND:
		output = !(inp1 && inp2);
		break;
	case NOR:
		output = !(inp1 || inp2);
		break;
	case XOR:
		output = (inp1 || inp2) && (!inp1 || !inp2);
		break;
	case XNOR:
		output = !((inp1 || inp2) && (!inp1 || !inp2));
		break;
	default:
		printf("Error: Gate not specified.\n");
		output = 0;
	}
	return output;
}

__global__ void blockQueuingKernel(int *nextLevelNodes, int *nodePtrs, int *nodeNeighbors, int *nodeInfo, int *currLevelNodes, int *numNextLevelNodes, float elementsPerThread)
{
    int idx = threadIdx.x + (blockIdx.x * blockDim.x);

    // iterate over all the nodes assigned to the current thread
    for (int i = std::floor(idx * elementsPerThread); i < std::floor((idx + 1) * elementsPerThread); i++) {

        // obtain the node index from the current level nodes list
        int node = currLevelNodes[i];

        // iterate over all neighbors of the current level node selected
        for (int neighborIdx = nodePtrs[node]; neighborIdx < nodePtrs[node + 1]; neighborIdx++) {

            // obtain the neighbor node index from the node neighbors list
            int neighbor = nodeNeighbors[neighborIdx];

            // if this neighbor node has not yet been visited
            if (nodeInfo[neighbor*4] == 0) {

            	// set the node as visited
            	nodeInfo[neighbor*4] = 1;

            	// compute the node output
            	nodeInfo[neighbor*4 + 3] = gate_solver_parallel_block(nodeInfo[neighbor*4 + 1], nodeInfo[neighbor*4 + 2], nodeInfo[node*4 + 3]);

            	// store the node in nextLevelNodes -> make sure to use atomic addition instead of the ++ operator
                // Add to the memory block if space else to the global memory
                nextLevelNodes[atomicAdd(numNextLevelNodes, 1)] = neighbor;
            }
        }
    }
    // Once done with all elements, transfer block memory to global memory
}


// Helper function for using CUDA to perform global queuing
cudaError_t blockQueuingHelper(
    int *nextLevelNodes, int *nodePtrs, int *nodeNeighbors, int *nodeInfo, int *currLevelNodes,
    int blockSize, int numBlock,
    int *numNextLevelNodes, int nodePtrs_size, int nodeNeighbors_size, int nodeInfo_size, int currLevelNodes_size
)
{
    int* dev_nextLevelNodes = 0;
    int* dev_nodePtrs = 0;
    int* dev_nodeNeighbors = 0;
    int* dev_nodeInfo = 0;
    int* dev_currLevelNodes = 0;
    int* dev_numNextLevelNodes;
    cudaError_t cudaStatus;

    float elementsPerThread = (float) currLevelNodes_size / (blockSize * numBlock);

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_nextLevelNodes, nodeInfo_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nodePtrs, nodePtrs_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nodeNeighbors, nodeNeighbors_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_nodeInfo, nodeInfo_size * 4 * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_currLevelNodes, currLevelNodes_size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_numNextLevelNodes, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_nodePtrs, nodePtrs, nodePtrs_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_nodeNeighbors, nodeNeighbors, nodeNeighbors_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_nodeInfo, nodeInfo, nodeInfo_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_currLevelNodes, currLevelNodes, currLevelNodes_size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_numNextLevelNodes, numNextLevelNodes, sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    blockQueuingKernel <<<numBlock, blockSize>>>(dev_nextLevelNodes, dev_nodePtrs, dev_nodeNeighbors, dev_nodeInfo, dev_currLevelNodes, dev_numNextLevelNodes, elementsPerThread);


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "blockQueuingKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching blockQueuingKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(nextLevelNodes, dev_nextLevelNodes, nodeInfo_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(nodeInfo, dev_nodeInfo, nodeInfo_size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(numNextLevelNodes, dev_numNextLevelNodes, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_nextLevelNodes);
    cudaFree(dev_nodePtrs);
    cudaFree(dev_nodeNeighbors);
    cudaFree(dev_nodeInfo);
    cudaFree(dev_currLevelNodes);
    
    return cudaStatus;
}


int main(int argc, char** argv)
{
    // validate input arguments
    if (argc != 9) {
        printf("Usage: ./block_queuing <blockSize> <numBlock> <inp1_file> <inp2_file> <inp3_file> <inp4_file> <nodeOutput_output_file> <nextLevelNodes_output_file>\n");
        exit(1);
    }

    // store the arguments
    int blockSize = std::stoi(argv[1]);
    int numBlock = std::stoi(argv[2]);
    char* inp1_filepath = argv[3];
    char* inp2_filepath = argv[4];
    char* inp3_filepath = argv[5];
    char* inp4_filepath = argv[6];
    char* nodeOutput_filepath = argv[7];
    char* nextLevelNodes_filepath = argv[8];

    // load the data from the files into arrays
    int nodePtrs_size, nodeNeighbors_size, nodeInfo_size, currLevelNodes_size;
    int* nodePtrs = input_reader(inp1_filepath, &nodePtrs_size);
    int* nodeNeighbors = input_reader(inp2_filepath, &nodeNeighbors_size);
    int* nodeInfo = input_reader_multiple(inp3_filepath, &nodeInfo_size);  // nodeInfo format -> visited, nodeGate, nodeInput, nodeOutput
    int* currLevelNodes = input_reader(inp4_filepath, &currLevelNodes_size);

    // initialize output variables
    int* nextLevelNodes = (int*)calloc(nodeInfo_size, sizeof(int));
    int numNextLevelNodes = 0;

    cudaError_t cudaStatus = blockQueuingHelper(
        nextLevelNodes, nodePtrs, nodeNeighbors, nodeInfo, currLevelNodes,
        blockSize, numBlock,
        &numNextLevelNodes, nodePtrs_size, nodeNeighbors_size, nodeInfo_size, currLevelNodes_size
    );
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "blockQueuingHelper failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    // store the output in appropriate files
    int* nodeOutput = (int*)calloc(nodeInfo_size, sizeof(int));
    for (int i = 0; i < nodeInfo_size; i++) nodeOutput[i] = nodeInfo[i * 4 + 3];
    output_writer(nodeOutput_filepath, nodeOutput, nodeInfo_size);
    output_writer(nextLevelNodes_filepath, nextLevelNodes, numNextLevelNodes);

    // compare the results using the helper scripts provided
    printf("\nComparing the output files from the program with the solution files\n");
    printf("Comparing nodeOutput file: \n");
    compareFiles(nodeOutput_filepath, "./Lab3/Output/sol_nodeOutput.raw");
    printf("\nComparing nextLevelNodes file: \n");
    compareNextLevelNodeFiles(nextLevelNodes_filepath, "./Lab3/Output/sol_nextLevelNodes.raw");

    return 0;
}