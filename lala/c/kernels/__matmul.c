#include <stdio.h>

void matmul_int(int rrows, int rcols, int lcols, int rhs[rrows][rcols], int lhs[rcols][lcols], int res[rrows][lcols]){

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (rrows + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > rrows){
            end = rrows;
        }


        for (int i = start; i < end; ++i) {
            for (int j = 0; j < lcols; ++j) {
                for (int k = 0; k < rcols; ++k) {
                    res[i][j] += rhs[i][k] * lhs[k][j];
                }
                
                
            }
        }
    }
}


void matmul_float(int rrows, int rcols, int lcols, float rhs[rrows][rcols], float lhs[rcols][lcols], float res[rrows][lcols]){

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (rrows + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > rrows){
            end = rrows;
        }


        for (int i = start; i < end; ++i) {
            for (int j = 0; j < lcols; ++j) {
                for (int k = 0; k < rcols; ++k) {
                    res[i][j] += rhs[i][k] * lhs[k][j];
                }
                
            }
        }
    }
}




void matmul_single_int(int rrows, int rcols, int lcols, int rhs[rrows][rcols], int lhs[rcols][lcols], int res[rrows][lcols]){

    for (int i = 0; i < rrows; ++i) {
        for (int j = 0; j < lcols; ++j) {
            for (int k = 0; k < rcols; ++k) {
                res[i][j] += rhs[i][k] * lhs[k][j];
            }
        }
    }
}

