#include <omp.h>
#include <stdio.h>

void matmul(int rrows, int rcols, int lcols, int rhs[rrows][rcols], int lhs[rcols][lcols], int res[rrows][lcols]){
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
                int sum = 0;
                
                for (int k = 0; k < rcols; ++k) {
                    sum += rhs[i][k] * lhs[k][j];
                }
                res[i][j] = sum;
            }
        }
    }
}



void mul(int rows, int cols, int *rhs, int *lhs, int *res){

     #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (rows * cols + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > (rows * cols)){
            end = rows * cols;
        }

        for (int i = start; i < end; ++i) {
            res[i] = rhs[i] * lhs[i];
        }
    }

}
