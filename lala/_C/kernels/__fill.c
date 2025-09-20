//fill a tensor with specific value

void fill_int(int* blk, int value, int size){
 
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

        for (int i = start; i < end; ++i) {
            blk[i] =  value;
        }
    }   
}

void fill_float(float* blk, float value, int size){
 
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

        for (int i = start; i < end; ++i) {
            blk[i] =  value;
        }

        
    }   
}