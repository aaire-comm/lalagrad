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

    //the response size is just whatever number of matmuls you have to do * size of one matrice in result
    int res_size = total_matmuls * lhs_rows * rhs_cols;

    //allocate a mem for res
    float* res = (float*)malloc(res_size);
    if (res == NULL){
        printf("Error allocating memory");
        return NULL;
    }

    
    

    return res;

}

