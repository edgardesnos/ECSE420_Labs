#include "compare.cuh"


void sort(int* pointer, int size) {
    //get from https://stackoverflow.com/questions/13012594/sorting-with-pointers-instead-of-indexes
    int* i, * j, temp;
    for (i = pointer; i < pointer + size; i++) {
        for (j = i + 1; j < pointer + size; j++) {
            if (*j < *i) {
                temp = *j;
                *j = *i;
                *i = temp;
            }
        }
    }
}


void compareNextLevelNodeFiles(char* file_name1, char* file_name2)
{


    FILE* fp_1 = fopen(file_name1, "r");
    if (fp_1 == NULL) {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    FILE* fp_2 = fopen(file_name2, "r");
    if (fp_2 == NULL) {
        fprintf(stderr, "Couldn't open file for reading\n");
        exit(1);
    }

    int counter = 0;
    int len_1;
    int len_2;
    int length_file_1 = fscanf(fp_1, "%d", &len_1);
    int length_file_2 = fscanf(fp_2, "%d", &len_2);

    if (length_file_1 != length_file_2) {
        fprintf(stderr, "Wrong file length\n");
        exit(1);
    }
    int* input1 = (int*)malloc(len_1 * sizeof(int));
    int* input2 = (int*)malloc(len_2 * sizeof(int));




    int temp1;
    int temp2;

    while ((fscanf(fp_1, "%d", &temp1) == 1) && (fscanf(fp_2, "%d", &temp2) == 1)) {
        (input1)[counter] = temp1;
        (input2)[counter] = temp2;
        counter++;
    }

    sort(input1, len_1);
    sort(input2, len_2);

    for (int i = 0; i < len_1; i++) {
        if (input1[i] != input2[i]) {
            fprintf(stderr, "Something goes wrong\n");
            exit(1);
        }
    }

    fprintf(stderr, "No errors!\n");
}


void compareFiles(char* file_name1, char* file_name2)
{
    //get from https://www.tutorialspoint.com/c-program-to-compare-two-files-and-report-mismatches
    FILE* fp1 = fopen(file_name1, "r");
    FILE* fp2 = fopen(file_name2, "r");
    // fetching character of two file 
    // in two variable ch1 and ch2 
    char ch1 = getc(fp1);
    char ch2 = getc(fp2);

    // error keeps track of number of errors 
    // pos keeps track of position of errors 
    // line keeps track of error line 
    int error = 0, pos = 0, line = 1;

    // iterate loop till end of file 
    while (ch1 != EOF && ch2 != EOF)
    {
        pos++;

        // if both variable encounters new 
        // line then line variable is incremented 
        // and pos variable is set to 0 
        if (ch1 == '\n' && ch2 == '\n')
        {
            line++;
            pos = 0;
        }

        // if fetched data is not equal then 
        // error is incremented 
        if (ch1 != ch2)
        {
            error++;
            printf("Line Number : %d \tError"
                " Position : %d \tDifferences: %c -> %c\n", line, pos, ch2, ch1);
        }

        // fetching character until end of file 
        ch1 = getc(fp1);
        ch2 = getc(fp2);
    }

    printf("Total Errors : %d\t", error);
}