#ifndef DATA_HELPER_H
#define DATA_HELPER_H


/*
* Loads data from a CSV into the provided arrays
*/
void load_data(char *filename, bool *a1, bool *a2, char *a3, unsigned int size);

/*
* Writes the output array on to the specified file
*/
void save_data(char *filename, bool *output_array, unsigned int size);

#endif  // DATA_HELPER_H