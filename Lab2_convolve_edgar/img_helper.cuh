#pragma once

#include "lodepng.cuh"
#include <stdio.h>
#include <stdlib.h>

void load_image(char* input_filename, unsigned char** image, unsigned int* width, unsigned int* height, unsigned int* size);
void save_image(char* output_filename, unsigned char* image, unsigned int width, unsigned int size);