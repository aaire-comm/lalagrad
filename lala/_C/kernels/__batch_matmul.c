//This is a mixed radic digit calcultor
//it basicaly get the nth sequence of digits counting in mixed base according to 
//out matrice shape (this shape is the same for both input matrices)
#define MIXED_REDIX_DIGITS(n, stride, size) (n/stride) % size

float* batch_matmul_float(
    float* lhs, float* rhs,
    int* common_shape, int* lhs_stride , int* rhs_stride, int* res_stride,
    int lhs_rows, int lhs_cols, int rhs_cols,
    int common_dims
){

    //calculate strides
    //this strides are used for counting not offset calculation
    int strides[common_dims];
    strides[common_dims-1] = 1;

    for (int i=0; i < common_dims; i++)
        if(lhs_stride[i] == 0 && rhs_stride[i]==0)
            common_shape[i] = 1;

    for (int dim=common_dims-2; dim >= 0; dim-- )
        strides[dim] = strides[dim+1] * common_shape[dim + 1];

    //calculate total number of matmuls to do
    int total_matmuls = 1;
    for (int i=0; i < common_dims; i++)
        total_matmuls *= common_shape[i];

    int res_size = total_matmuls * lhs_rows * rhs_cols * 4;
    float* res= (float*)malloc(res_size);

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int matmul_per_thread = (total_matmuls + nthreads-1) / nthreads;
        int start  = tid * matmul_per_thread;
        int end = start + matmul_per_thread;

        if (end > total_matmuls){
            end = total_matmuls;
        }
        //assign
        int counter[common_dims];
        for (int i=0; i < common_dims; i++)
            counter[i] = 0;

        int lhs_base_offset;
        int rhs_base_offset;
        int res_base_offset;

        for (int matmul=start; matmul < end; matmul++){
            //get the size allong each dim to get that current matrice
            for (int i=0; i < common_dims; i++)
                counter[i] = MIXED_REDIX_DIGITS(matmul, strides[i], common_shape[i]);

            lhs_base_offset = 0;
            rhs_base_offset = 0;
            res_base_offset = 0;

            for (int dim=0; dim < common_dims; dim++){
                lhs_base_offset += counter[dim] * lhs_stride[dim];
                rhs_base_offset += counter[dim] * rhs_stride[dim];
                res_base_offset += counter[dim] * res_stride[dim];
                
            }
            
            float* lhs_offset = lhs + lhs_base_offset;
            float* rhs_offset = rhs + rhs_base_offset;
            float* res_offset = res + res_base_offset;
            // printf("Thread: %d, matmul: %d, lhs_off: %d, rhs_off: %d, res_off: %d, lhs[0]: %f, rhs[0]: %f\n", tid, matmul, lhs_base_offset, rhs_base_offset, res_base_offset, lhs_offset[0], rhs_offset[0]);
            float tmp;
            for (int row = 0; row < lhs_rows; row++) {
                for (int col=0; col < rhs_cols; col++){
                    tmp = 0;
                    for (int elem=0; elem < lhs_cols; elem++){
                        tmp += lhs_offset[row*lhs_cols + elem] * rhs_offset[elem*rhs_cols + col];
                        // printf("%p, %04f = %p X %04f\n", res_offset+ row*rhs_cols + col,  tmp, lhs_offset + row*lhs_cols + elem, rhs_offset[elem*rhs_cols + col]);
                    }
                    // printf("Writing at %p: valueof %f\n", res_offset + row*rhs_cols + col, tmp);
                    res_offset[row*rhs_cols + col] = tmp;
                }   
            }
        }

    }

    /*
    PRINT THE RESULT MATRICES
    for (int i=0; i < total_matmuls; i++){
        printf("\n");
        for (int row = 0; row < lhs_rows; row++) {
            printf("[");
            for (int col=0; col < rhs_cols; col++)
                    printf("%f, ", res[i*lhs_rows*lhs_cols + row*lhs_rows + col]);

            }
            printf("]\n");
        }
        printf("\n");
    */
    return res;
    
}


//This is a mixed radic digit calcultor
//it basicaly get the nth sequence of digits counting in mixed base according to 
//out matrice shape (this shape is the same for both input matrices)
#define MIXED_REDIX_DIGITS(n, stride, size) (n/stride) % size

int* batch_matmul_int(
    int* lhs, int* rhs,
    int* common_shape, int* lhs_stride , int* rhs_stride, int* res_stride,
    int lhs_rows, int lhs_cols, int rhs_cols,
    int common_dims
){

    //calculate strides
    //this strides are used for counting not offset calculation
    int strides[common_dims];
    strides[common_dims-1] = 1;

    for (int i=0; i < common_dims; i++)
        if(lhs_stride[i] == 0 && rhs_stride[i]==0)
            common_shape[i] = 1;

    for (int dim=common_dims-2; dim >= 0; dim-- )
        strides[dim] = strides[dim+1] * common_shape[dim + 1];

    //calculate total number of matmuls to do
    int total_matmuls = 1;
    for (int i=0; i < common_dims; i++)
        total_matmuls *= common_shape[i];

    int res_size = total_matmuls * lhs_rows * rhs_cols * 4;
    int* res = (int*)malloc(res_size);

    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int matmul_per_thread = (total_matmuls + nthreads-1) / nthreads;
        int start  = tid * matmul_per_thread;
        int end = start + matmul_per_thread;

        if (end > total_matmuls){
            end = total_matmuls;
        }
        //assign
        int counter[common_dims];
        for (int i=0; i < common_dims; i++)
            counter[i] = 0;

        int lhs_base_offset;
        int rhs_base_offset;
        int res_base_offset;

        for (int matmul=start; matmul < end; matmul++){
            //get the size allong each dim to get that current matrice
            for (int i=0; i < common_dims; i++)
                counter[i] = MIXED_REDIX_DIGITS(matmul, strides[i], common_shape[i]);

            lhs_base_offset = 0;
            rhs_base_offset = 0;
            res_base_offset = 0;

            for (int dim=0; dim < common_dims; dim++){
                lhs_base_offset += counter[dim] * lhs_stride[dim];
                rhs_base_offset += counter[dim] * rhs_stride[dim];
                res_base_offset += counter[dim] * res_stride[dim];
                
            }
            
            int* lhs_offset = lhs + lhs_base_offset;
            int* rhs_offset = rhs + rhs_base_offset;
            int* res_offset = res + res_base_offset;
            // printf("Thread: %d, matmul: %d, lhs_off: %d, rhs_off: %d, res_off: %d, lhs[0]: %f, rhs[0]: %f\n", tid, matmul, lhs_base_offset, rhs_base_offset, res_base_offset, lhs_offset[0], rhs_offset[0]);
            int tmp;
            for (int row = 0; row < lhs_rows; row++) {
                for (int col=0; col < rhs_cols; col++){
                    tmp = 0;
                    for (int elem=0; elem < lhs_cols; elem++){
                        tmp += lhs_offset[row*lhs_cols + elem] * rhs_offset[elem*rhs_cols + col];
                        // printf("%p, %04f = %p X %04f\n", res_offset+ row*rhs_cols + col,  tmp, lhs_offset + row*lhs_cols + elem, rhs_offset[elem*rhs_cols + col]);
                    }
                    // printf("Writing at %p: valueof %f\n", res_offset + row*rhs_cols + col, tmp);
                    res_offset[row*rhs_cols + col] = tmp;
                }   
            }
        }

    }

    /*
    PRINT THE RESULT MATRICES
    for (int i=0; i < total_matmuls; i++){
        printf("\n");
        for (int row = 0; row < lhs_rows; row++) {
            printf("[");
            for (int col=0; col < rhs_cols; col++)
                    printf("%f, ", res[i*lhs_rows*lhs_cols + row*lhs_rows + col]);

            }
            printf("]\n");
        }
        printf("\n");
    */
    return res;
    
}

