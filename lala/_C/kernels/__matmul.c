

#include <stdio.h>

void matmul_int(int* lhs, int* rhs, int* s1, int* s2){

}

void matmul_float(int lhs_rows, int lhs_cols, int rhs_cols, float *lhs, float *rhs, float *res){
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (lhs_rows + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > lhs_rows)
            end = lhs_rows;

        for (int row = start; row < end; row++) {
            for (int col=0; col < rhs_cols; col++){
                for (int elem=0; elem < lhs_cols; elem++)
                    res[row*rhs_cols + col] += lhs[row*lhs_cols + elem] * rhs[elem*rhs_cols + col];
            }   
        }
    }
    return;

}



