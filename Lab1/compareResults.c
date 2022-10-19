#include <stdio.h>
#include <stdlib.h>


void compareFiles(char *file_name1, char *file_name2) 
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
               " Position : %d \n", line, pos); 
        } 
  
        // fetching character until end of file 
        ch1 = getc(fp1); 
        ch2 = getc(fp2); 
    } 
  
    printf("Total Errors : %d\t", error); 
} 

int main(int argc, char *argv[]){

    if( argc < 3) {
      printf("Require two files\n");
      exit(1);
      
   }
compareFiles(argv[1], argv[2]);
}
