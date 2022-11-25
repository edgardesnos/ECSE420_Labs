#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string>

int* input_reader_multiple(char* filename, int* size);
int* input_reader(char* filename, int* size);
void output_writer(char* filename, int* arr, int size);