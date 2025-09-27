float* batch_matmul_float(
    float* lhs, float* rhs,
    int* lhs_shape, int* lhs_stride , int* rhs_shape, int* rhs_stride,
    int dims
){
    
    //set dims with 0 stride to 1 so ops along that dim only happen once
    for (int i=0; i < dims; i++){
        if (lhs_stride[i] == 0)
        lhs_shape[i] = 1;
        if (rhs_stride[i] == 0)
        rhs_shape[i] = 1;
    }
    
    int total_matmuls = 1;
    //calcutate the total num of matmuls to be done
    for (int dim = dims-3; dim > -1 ; dim--){
        total_matmuls *= lhs_shape[dim];
    }
    
    
    //get the rows and cols of the matrices
    int lhs_rows = lhs_shape[dims-2];
    int lhs_cols = lhs_shape[dims-1]; //REMEMBER: this is equal to rhs_rows
    int rhs_cols = rhs_shape[dims-1];

    int res_size = total_matmuls * lhs_rows * rhs_cols;

    //allocate a mem for res
    float* res_ = (float*)malloc(res_size);

    
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();
        
        int per_thread = (total_matmuls + nthreads-1) / nthreads;
        int start  = tid * per_thread;

        
        int end = start + per_thread;

        
        if (end > total_matmuls){
            end = total_matmuls;
        }

        for (int matmul = start; matmul < end; matmul++) {
            float* lhs_offset = lhs + matmul * lhs_cols * lhs_rows;
            float* rhs_offset = rhs + matmul * rhs_cols * lhs_cols;
            float* res_offset = res_ + matmul * rhs_cols * lhs_rows;

            for (int row = 0; row < lhs_rows; row++) {
                for (int col=0; col < rhs_cols; col++){
                    for (int elem=0; elem < lhs_cols; elem++){
                        res_offset[row*rhs_cols + col] += lhs_offset[row*lhs_cols + elem] * rhs_offset[elem*rhs_cols + col];
                        printf("%f  %f\n", lhs_offset[row*lhs_cols + elem], rhs_offset[elem*rhs_cols + col]);
                    }
                }   
            }
        
        }
        

    }

    for(int i=0; i < res_size; i++){
        res_[i] = 1.0;
    }

    return res_;

}

