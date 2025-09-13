
void add_scalar(int rows, int cols, int rhs[rows][cols], int scalar, int res[rows][cols]){
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (rows + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > rows){
            end = rows;
        }

        for (int i = start; i < end; ++i) {
                for (int j = 0; j < cols; ++j) {
                    res[i][j] = rhs[i][j] + scalar;
                }
            }
    }
}

void sub_scalar(int rows, int cols, int* rhs, int scalar, int* res){
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (rows + cols + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > (rows + cols)){
            end = rows;
        }

        for (int i = start; i < end; ++i) {
                    res[i][j] = rhs[i][j] + scalar;
            }
    }
}