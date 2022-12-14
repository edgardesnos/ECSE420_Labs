#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "io_helper.cuh"
#include "compare.cuh"
#include <chrono>


/*
*
* Function to compute the output given a logical gate and the corresponding two input values.
*/
int gate_solver(int gateType, int inp1, int inp2) {
	int output;
	switch (gateType) {
	case 0:  // AND
		output = inp1 && inp2;
		break;
	case 1:  // OR
		output = inp1 || inp2;
		break;
	case 2:  // NAND
		output = !(inp1 && inp2);
		break;
	case 3:  // NOR
		output = !(inp1 || inp2);
		break;
	case 4:  // XOR
		output = (inp1 || inp2) && (!inp1 || !inp2);
		break;
	case 5:  // XNOR
		output = !((inp1 || inp2) && (!inp1 || !inp2));
		break;
	default:
		printf("Error: Gate %d not specified.\n", gateType);
		output = -1;
	}
	return output;
}


int mainSeq(int argc, char **argv) {
	// validate input arguments
	if (argc != 7) {
		printf("Usage: ./sequential <inp1_file> <inp2_file> <inp3_file> <inp4_file> <nodeOutput_output_file> <nextLevelNodes_output_file>\n");
		exit(1);
	}

	// store the arguments
	char* inp1_filepath = argv[1];
	char* inp2_filepath = argv[2];
	char* inp3_filepath = argv[3];
	char* inp4_filepath = argv[4];
	char* nodeOutput_filepath = argv[5];
	char* nextLevelNodes_filepath = argv[6];

	// load the data from the files into arrays
	int nodePtrs_size, nodeNeighbors_size, nodeInfo_size, currLevelNodes_size;
	int* nodePtrs = input_reader(inp1_filepath, &nodePtrs_size);
	int* nodeNeighbors = input_reader(inp2_filepath, &nodeNeighbors_size);
	int* nodeInfo = input_reader_multiple(inp3_filepath, &nodeInfo_size);  // nodeInfo format -> visited, nodeGate, nodeInput, nodeOutput
	int* currLevelNodes = input_reader(inp4_filepath, &currLevelNodes_size);

	// initialize output variables
	int numNextLevelNodes = 0;
	int* nextLevelNodes = (int*)calloc(nodeInfo_size, sizeof(int));

	// initialize timer variables
	std::chrono::high_resolution_clock::time_point start, end;
	double exec_time, total_exec_time = 0.;

	// start the timer
	start = std::chrono::high_resolution_clock::now();

	// start the sequential algorithm
	for (int i = 0; i < currLevelNodes_size; i++) {
		// obtain the current node in the queue
		int node = currLevelNodes[i];

		// loop over all neighbors
		for (int neighborIdx = nodePtrs[node]; neighborIdx < nodePtrs[node + 1]; neighborIdx++) {
			int neighbor = nodeNeighbors[neighborIdx];
			// if this neighbor node has not yet been visited
			if (nodeInfo[neighbor*4] == 0) {
				// set the node as visited
				nodeInfo[neighbor*4] = 1;
				// compute the node output
				nodeInfo[neighbor*4 + 3] = gate_solver(nodeInfo[neighbor*4 + 1], nodeInfo[neighbor*4 + 2], nodeInfo[node*4 + 3]);
				// store the node in nextLevelNodes
				nextLevelNodes[numNextLevelNodes++] = neighbor;  // nextLevelNodes[atomicAdd(&numNextLevelNodes, 1) + 1] = neighbor;
			}
		}
	}

	// end timer
	end = std::chrono::high_resolution_clock::now();

	// print the runtime
	exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.;
	printf("\nThe runtime for sequential execution is: %f ms\n", exec_time);

	// store the output in appropriate files
	int* nodeOutput = (int*)calloc(nodeInfo_size, sizeof(int));
	for (int i = 0; i < nodeInfo_size; i++) nodeOutput[i] = nodeInfo[i*4 + 3];
	output_writer(nodeOutput_filepath, nodeOutput, nodeInfo_size);
	output_writer(nextLevelNodes_filepath, nextLevelNodes, numNextLevelNodes);

	// compare the results using the helper scripts provided
	printf("\nComparing the output files from the program with the solution files\n");
	printf("Comparing nodeOutput file: \n");
	compareFiles(nodeOutput_filepath, "./Lab3/Output/sol_nodeOutput.raw");
	printf("\nComparing nextLevelNodes file: \n");
	compareNextLevelNodeFiles(nextLevelNodes_filepath, "./Lab3/Output/sol_nextLevelNodes.raw");

}