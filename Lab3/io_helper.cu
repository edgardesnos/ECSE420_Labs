#include "io_helper.cuh"


/*
*
* Functions to read the contents of the csv input files provided for this assignment.
*/
int* input_reader_multiple(char* filename, int* size) {
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
* Function to write the output file as instructed in the assignment
*/
void output_writer(char* filename, int* arr, int size) {
    FILE* fp = fopen(filename, "wt");
    fprintf(fp, "%d\n", size);
    for (int i = 0; i < size; i++) fprintf(fp, "%d\n", arr[i]);
    fclose(fp);
}