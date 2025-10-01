
//functional kernels (exposed as lala.functional)
void relu_int(int* blk, int* res, int size);

//tensor op with scalar ops
void add_scalar_int(int* rhs, int scalar, int* res, int size);
void mul_scalar_int(int* rhs, int scalar, int* res, int size);
void div_scalar_int(int* rhs, int scalar, int* res, int size);
void sub_scalar_int(int* rhs, int scalar, int* res, int size);
void mul_scalar_float(float* rhs, float scalar, float* res, int size); 



//tensor with tensor  ops
void add_float(float *rhs, float* lhs, float* res, int size);
void add_int(int *rhs, int* lhs, int* res, int size);
void sub_float(float *rhs, float* lhs, float* res, int size);
void sub_int(int *rhs, int* lhs, int* res, int size);
void div_int(int *rhs, int* lhs, int* res, int size);
void mul_int(int *rhs, int* lhs, int* res, int size);
void mul_float(float *rhs, float* lhs, float* res, int size);
void matmul_int(int* lhs, int* rhs,  int* s1, int* s2);
// void matmul_float(float* lhs, float* rhs, int* s1, int* s2);
// void matmul_float(float* lhs, float* rhs, float* res, int lhs_rows, int lhs_cols, int rhs_cols );
// void matmul_float(int lhs_rows, int lhs_cols, int rhs_cols, float lhs[lhs_rows][lhs_cols], float rhs[lhs_cols][rhs_cols], float res[lhs_rows][rhs_cols]);
void matmul_float(int lhs_rows, int lhs_cols, int rhs_cols, float *lhs, float *rhs, float *res);
// float* batch_matmul_float( float* lhs, float* rhs, int* common_shape, int* lhs_stride , int* rhs_stride, int* res_stride, int lhs_rows, int lhs_cols, int rhs_cols, int common_dims );
float* batch_matmul_float( float* lhs, float* rhs, float* res, int* res_shape, int* res_strides, int* lhs_shape, int* lhs_stride , int* rhs_shape,int* rhs_stride, int dims);
int* batch_matmul_int( int* lhs, int* rhs, int* common_shape, int* lhs_stride , int* rhs_stride, int* res_stride, int lhs_rows, int lhs_cols, int rhs_cols, int common_dims );


void power_float(float *rhs, float* lhs, float* res, int size);
void power_scalar_int(int* rhs, float scalar, int* res, int size);
void power_scalar_float(float* rhs, float scalar, float* res, int size);



//Scala reduce ops  (this operations reduce an array to an int)
void sum_int(int *t, int* res, int size);
void sum_float(float *t, float* res, int size);
void mean_float(float *t, float* res, int size);
void mean_int(float *t, float* res, int size);





//Memory ops
void fill_int(int* blk, int value, int size);
void fill_float(float* blk, float value, int size);
void transpose_int(int* t, int dim0, int dim1, int* strides, int dims, int size);
// void transpose_float(float* t, int dim0, int dim1, int* strides, int dims, int size);
void transpose_float(float* lhs, float* res, int lhs_rows, int lhs_cols);

void cast_int_float(int* lhs, float* res, int size);

//Memory ops from libc  (handled in setup)
void* malloc(size_t size);
void free(void* ptr);
void* memset(void* ptr, int v, size_t size);
void *memcpy(void *dest, const void *src, size_t n);




