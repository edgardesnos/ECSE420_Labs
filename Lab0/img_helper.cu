#include "img_helper.cuh"

void load_image(char* input_filename, unsigned char** image, unsigned int* width, unsigned int* height, unsigned int *size) {
	unsigned error;

	error = lodepng_decode32_file(image, width, height, input_filename);
	if(error) printf("Error %u: %s\n", error, lodepng_error_text(error));
	*size = (*width) * (*height) * 4;
}

void save_image(char* output_filename, unsigned char* image, unsigned int width, unsigned int height) {
	lodepng_encode32_file(output_filename, image, width, height);
}