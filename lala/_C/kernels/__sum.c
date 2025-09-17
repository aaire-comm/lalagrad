void sum_int(int *t, int* res, int size){

    int MAX_THEAD_NUM = omp_get_max_threads();
    //Allocate accumuator per thread (Avoid race)
    int per_thread_acc[MAX_THEAD_NUM]; 

    //init all accumulators to ZERO
    for(int thread=0; thread < MAX_THEAD_NUM; thread++){
        per_thread_acc[thread] = 0;
    }

    //do the same for the final result
    res[0] = 0;


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
        
        for (int i = start; i < end; i++) {
            per_thread_acc[tid] += t[i];
        }
    }

    //remember this only runs in the main thread (openMP joins threads by default)
    //go over all the thread accumulators and add up to final result

    for(int thread=0; thread < MAX_THEAD_NUM; thread++){
        res[0] += per_thread_acc[thread];
    }


}



