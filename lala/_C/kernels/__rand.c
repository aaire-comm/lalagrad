//Uses libc random number generator function 
//libc is included in lib_tensor.c

#ifndef RAND_MAX
RAND_MAX = 10000;
#endif


void rand_int(int* blk, int seed, int size){
 
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

        srand(seed);
        for (int i = start; i < end; ++i) {
            blk[i] =  rand();
        }
    }   
}


void rand_int_in_range(int* blk, int seed, int bottom, int top, int size){
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

        srand(seed);
        for (int i = start; i < end; ++i) {
            blk[i] =  bottom + (top - bottom) * ((double)rand() / (double)RAND_MAX);
        }
    } 
}


//this is a kernel for random float in 0 - 1 range
void rand_(float* blk, int seed, int size){
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

        srand(seed);
        for (int i = start; i < end; ++i) {
            blk[i] =  2.0 * rand() / (double)RAND_MAX - 1.0;
        }
    } 

}