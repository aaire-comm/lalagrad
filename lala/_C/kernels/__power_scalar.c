//tensor power by scalar loop through and do (element ** scalar)

void power_scalar_int(int* rhs, float scalar, int* res, int size){
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
            res[i] = pow(rhs[i],  scalar);
        }
    }
}


void power_scalar_float(float* rhs, float scalar, float* res, int size){
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
            res[i] = pow(rhs[i],  scalar);
        }
    }
}
