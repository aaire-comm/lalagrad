//transpose two dimentions given the stride and the dims

void transpose_int(int* t, int dim0, int dim1, int* strides, int dims, int size){
    return;
}



void transpose_float(float* lhs, float* res, int lhs_rows, int lhs_cols){
     #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (lhs_rows + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > lhs_rows){
            end = lhs_rows;
        }

        for (int row = start; row < end; row++) {
            for (int col=0; col < lhs_cols; col++){
                res[col*lhs_rows + row] = lhs[row*lhs_cols + col];
            }
        }
    }
    return;
}


