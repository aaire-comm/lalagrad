//REMEMBER in c casting to float is not just reinterpreting the int bits as floats
//we instead need to convert the c int values to IEEE 754 single-precision float format for the corresponding int value

void cast_int_float(int* lhs, float* res, int size){
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
            res[i] = (float)lhs[i];
        }
    }
}