
//functional kernels (exposed as lala.functional)
void relu_int(int* blk, int* res, int size);

//tensor op with scalar ops
void add_scalar_int(int* rhs, int scalar, int* res, int size);
void mul_scalar_int(int* rhs, int scalar, int* res, int size);
void div_scalar_int(int* rhs, int scalar, int* res, int size);
void sub_scalar_int(int* rhs, int scalar, int* res, int size);

//tensor with tensor  ops
void sub_int(int *rhs, int* lhs, int* res, int size);
void div_int(int *rhs, int* lhs, int* res, int size);
void mul_int(int *rhs, int* lhs, int* res, int size);
void add_int(int *rhs, int* lhs, int* res, int size);


//Memory ops
void fill_int(int* blk, int value, int size);

//Memory ops from libc
void* malloc(size_t size);
void free(void* ptr);
void* memset(void* ptr, int v, size_t size);

