void relu_int(int* blk, int* res, int size){
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (size + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > size){
            end = size;
        }

        for (int i = start; i < end; ++i)
                res[i] = blk[i] ? blk[i] > 0 : 0;
    }
}

void relu_floatint(float* blk, float* res, int size){
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int rows_per_thread = (size + nthreads-1) / nthreads;
        int start  = tid * rows_per_thread;
        int end = start + rows_per_thread;

        if (end > size){
            end = size;
        }


        for (int i = start; i < end; ++i) 
                res[i] = blk[i] ? blk[i] > 0 : 0;
    }
}