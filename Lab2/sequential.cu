#include <stdio.h>
#include <math.h>
#include <string>
#include <chrono>

int mainSequential(int argc, char** argv)
{

    if (argc != 2) {
        printf("Usage: ./sequential <num_iterations>");
        exit(1);
    }

    // load command args
    int T = std::stoi(argv[1]);

    const int N = 4;

    // Init u arrays
    float u2[N][N] = { 0 };
    float u1[N][N] = { 0 };
    float u[N][N] = { 0 };

    // constants
    float n = 0.0002;
    float p = 0.5;
    float G = 0.75;

    // Hit coordinates
    int hit_i = 2;
    int hit_j = 2;

    // Add the drum hit
    u1[hit_i][hit_j] = 1;

    printf("Size of grid: %d nodes\n", N*N);

    // initialize timer variables
    std::chrono::high_resolution_clock::time_point start, end;
    double exec_time, total_exec_time = 0.;

    // start the timer
    start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < T; k++) {
        // Interior elements
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                u[i][j] = (p * (u1[i - 1][j] + u1[i + 1][j] + u1[i][j - 1] + u1[i][j + 1] - 4 * u1[i][j]) + 2 * u1[i][j] - (1 - n) * u2[i][j]) / (1 + n);
            }
        }
        // Side elements
        for (int i = 1; i < N - 1; i++) {
            u[0][i] = G * u[1][i];
            u[N - 1][i] = G * u[N - 2][i];
            u[i][0] = G * u[i][1];
            u[i][N - 1] = G * u[i][N - 2];
        }
        // Corner elements
        u[0][0] = G * u[1][0];
        u[N - 1][0] = G * u[N - 2][0];
        u[0][N - 1] = G * u[0][N - 2];
        u[N - 1][N - 1] = G * u[N - 1][N - 2];
        // Copy elements from u to u1 and u1 to u2
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                u2[i][j] = u1[i][j];
                u1[i][j] = u[i][j];
            }
        }
        printf("(0,0): %f (0,1): %f (0,2): %f (0,3): %f\n", u[0][0], u[0][1], u[0][2], u[0][3]);
        printf("(1,0): %f (1,1): %f (1,2): %f (1,3): %f\n", u[1][0], u[1][1], u[1][2], u[1][3]);
        printf("(2,0): %f (2,1): %f (2,2): %f (2,3): %f\n", u[2][0], u[2][1], u[2][2], u[2][3]);
        printf("(3,0): %f (3,1): %f (3,2): %f (3,3): %f\n", u[3][0], u[3][1], u[3][2], u[3][3]);
        printf("\n");
    }

    // end timer
    end = std::chrono::high_resolution_clock::now();

    // print the runtime
    exec_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1000000.;
    printf("\nThe runtime for sequential execution is: %f ms\n", exec_time);

    return 0;
}
