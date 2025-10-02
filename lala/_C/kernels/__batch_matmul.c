//This is a mixed radic digit calcultor
//it basicaly get the nth sequence of digits counting in mixed base according to 
//out matrice shape (this shape is the same for both input matrices)
#define MIXED_REDIX_DIGITS(n, stride, size) (n/stride) % size

float* batch_matmul_float(
    float* lhs, float* rhs, float* res,
    int* res_shape, int* res_strides,
    int* lhs_shape, int* lhs_stride ,
    int* rhs_shape, int* rhs_stride,
    int dims
){

    int lhs_rows = lhs_shape[dims-2];
    int lhs_cols = lhs_shape[dims-1];
    int rhs_cols = rhs_shape[dims-1];


    //set all sizes allong dims where both operands have 0 stride to 1
    //this lets us do the matmul allong that shape just once and view will handle the rest
    for (int i=0; i < dims; i++)
        if(lhs_stride[i] == 0 && rhs_stride[i]==0)
            res_shape[i] = 1;

    
    //calculate total number of matmuls to do
    int work_items = 1;
    for (int i=0; i < dims; i++)
        work_items *= res_shape[i];
    
    #pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int tid = omp_get_thread_num();

        //A Work Group is a group of work items that are done by a single thread
        int work_group = (work_items + nthreads-1) / nthreads; 

        //for each thread we calculate the start and end work groups
        int start  = tid * work_group;
        int end = start + work_group;
        
        if (end > work_items)
            end = work_items;

        int counter[dims];
        for (int i=0; i < dims; i++)
            counter[i] = 0;

        int lhs_base_offset;
        int rhs_base_offset;
        // printf("Thread %d: %d of total %d\n", tid, end - start, work_items);
        float* lhs_offset = NULL;
        float* rhs_offset = NULL;
        float tmp;
        
        for (int item=start; item < end; item++){
            lhs_base_offset = 0;
            rhs_base_offset = 0;
            tmp = 0;
            // printf("Thread %d: %d of total %d\n", tid, end - start, work_items);

            //get the size allong each dim to get that current matrice
            for (int i=0; i < dims; i++)
                counter[i] = MIXED_REDIX_DIGITS(item, res_strides[i], res_shape[i]);

            //mode the pointers to the current matrice we are working on
            for (int i=0; i < dims-2; i++){
                lhs_base_offset += counter[i] * lhs_stride[i];
                rhs_base_offset += counter[i] * rhs_stride[i];
            }

            //get to the row we are woking on
            lhs_base_offset += counter[dims-2] * lhs_stride[dims-2];
            rhs_base_offset += counter[dims-1] * rhs_stride[dims-1];

            lhs_offset = lhs + lhs_base_offset;
            rhs_offset = rhs + rhs_base_offset;
            
            //float* res_offset = res + item;
            // printf("lhs_off: %d, rhs_off: %d, res_off: %d\n", lhs_base_offset, rhs_base_offset, item);

            //A Single Work Item for matmul
            printf("item: %d [ ", item);
            for (int elem=0; elem < lhs_cols; elem++){
                printf("+ %f * %f", lhs_offset[elem], rhs_offset[elem*rhs_cols*rhs_stride[dims-1]]);
                tmp += lhs_offset[elem] * rhs_offset[elem*rhs_cols*rhs_stride[dims-1]];
            }
            printf(" = %f ]\n", tmp);


            res[item] = tmp;
        }
        
    }

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

    // for (int i=0; i < common_dims; i++)
    //     if(lhs_stride[i] == 0 && rhs_stride[i]==0)
    //         common_shape[i] = 1;

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

        int work_group = (total_matmuls + nthreads-1) / nthreads;
        int start  = tid * work_group;
        int end = start + work_group;

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
            printf("Writing to base offset: %d", res_base_offset);
            
            int tmp;
            for (int row = 0; row < lhs_rows; row++) {
                for (int col=0; col < rhs_cols; col++){
                    tmp = 0;
                    for (int elem=0; elem < lhs_cols; elem++)
                        tmp += lhs_offset[row*lhs_cols + elem] * rhs_offset[elem*rhs_cols + col];
                    res_offset[row*rhs_cols + col] = tmp;
                }   
            }
        }

    }

    return res;
    
}

