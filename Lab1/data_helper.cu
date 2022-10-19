#include <stdio.h>
#include <stdlib.h>
#include "data_helper.cuh"

#define BUFFER_LEN 50


void load_data(char *filename, bool *a1, bool *a2, char *a3, unsigned int size) {
	// open the file and obtain the file pointer
	FILE *f = fopen(filename, "r");

	if (f == NULL) {
		printf("Unexpected error occured in reading from the file %s\n", filename);
		exit(1);
	}

	char buffer[BUFFER_LEN];
	int idx = 0;

	// read data from the file and populate the two arrays
	while (fgets(buffer, BUFFER_LEN, f) != NULL) {
		int a1_num, a2_num;
		sscanf(buffer, "%d,%d,%d", &a1_num, &a2_num, a3 + idx);
		a1[idx] = (bool)a1_num;
		a2[idx] = (bool)a2_num;
		idx++;
	}

	// close the file stream
	fclose(f);
}

void save_data(char* filename, bool* output_array, unsigned int output_array_size) {
	// create or open the file and obtain it's file pointer
	FILE* f = fopen(filename, "w");

	if (f == NULL) {
		printf("Unexpected error occured in writing to the file %s\n", filename);
		exit(1);
	}

	// write data from the provided array into the file
	for (unsigned int i = 0; i < output_array_size; i++) {
		fprintf(f, "%d\n", (int)output_array[i]);
	}

	// close the file stream
	fclose(f);
}