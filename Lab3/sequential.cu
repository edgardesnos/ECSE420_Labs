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
* Functions to read the contents of the csv input files provided for this assignment.
*/
int* input_reader_multiple(char* filename, int *size) {
	FILE* fp = fopen(filename, "r");
	char buffer[100];
	int idx = 0;
	int* out;

	while (fgets(buffer, 100, fp) != NULL) {
		char* token = strtok(buffer, ",");
		while (token != NULL) {
			if (idx == 0) {
				*size = std::stoi(token);
				out = (int*)calloc(*size * 4, sizeof(int));
			}
			else {
				out[idx - 1] = std::stoi(token);
			}
			idx++;
			token = strtok(NULL, ",");
		}
	}
	fclose(fp);
	return out;
}

int* input_reader(char* filename, int* size) {
	FILE* fp = fopen(filename, "r");
	char buffer[100];
	int idx = 0;
	int* out;

	while (fgets(buffer, 100, fp) != NULL) {
		if (idx == 0) {
			*size = std::stoi(buffer);
			out = (int*)calloc(*size, sizeof(int));
		}
		else {
			out[idx - 1] = std::stoi(buffer);
		}
		idx++;
	}
	fclose(fp);
	return out;
}


/*
*
* Function to compute the output given a logical gate and the corresponding two input values.
*/
int gate_solver(int gateType, int inp1, int inp2) {
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


/*
*
* Function to write the output file as instructed in the assignment
*/
void output_writer(char *filename, int* arr, int size) {
	FILE* fp = fopen(filename, "wt");
	fprintf(fp, "%d\n", size);
	for (int i = 0; i < size; i++) fprintf(fp, "%d\n", arr[i]);
	fclose(fp);
}


int main(int argc, char **argv) {
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

	// start the sequential algorithm
	for (int i = 0; i < currLevelNodes_size; i++) {
		// obtain the current node in the queue
		int node = currLevelNodes[i];

		// loop over all neighbors
		for (int neighborIdx = nodePtrs[node]; neighborIdx < nodePtrs[node + 1]; neighborIdx++) {
			int neighbor = nodeNeighbors[neighborIdx];
			// if this neighbor node has not yet been visited
			if (nodeInfo[neighbor] == 0) {
				// set the node as visited
				nodeInfo[neighbor] = 1;
				// compute the node output
				nodeInfo[neighbor + 3] = gate_solver(nodeInfo[neighbor + 1], nodeInfo[neighbor + 2], nodeInfo[node + 3]);
				// store the node in nextLevelNodes
				nextLevelNodes[numNextLevelNodes++] = neighbor;
			}
		}
	}

	// store the output in appropriate files
	int* nodeOutput = (int*)calloc(nodeInfo_size, sizeof(int));
	for (int i = 0; i < nodeInfo_size; i++) nodeOutput[i] = nodeInfo[i + 3];
	output_writer(nodeOutput_filepath, nodeOutput, nodeInfo_size);
	output_writer(nextLevelNodes_filepath, nextLevelNodes, numNextLevelNodes);
}