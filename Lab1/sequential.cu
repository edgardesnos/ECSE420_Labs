
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

bool gateOutput(bool a, bool b, char gate);


void logicGateSequential(bool* output, bool* a, bool* b, char* gate, int length) 
{
    for (int i=0; i<length; i++) 
    {
        output[i] = gateOutput(a[i], b[i], gate[i]);
    }
}

bool gateOutput(bool a, bool b, char gate)
{
    bool output = 0;
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

    //Execute sequentially
    logicGateSequential(output, a, b, gate, size);

    printf("Output array : {%d,%d,%d,%d}\n",output[0], output[1], output[2], output[3]);

    // save output data into a file
    save_data(output_filename, output, size);

    return 0;
}
