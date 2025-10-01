#include <omp.h>
#include <stdlib.h>
#include <math.h>


#include "../kernels/__relu.c"
#include "../kernels/__fill.c"
#include "../kernels/__mul_scalar.c"
#include "../kernels/__rand.c"
#include "../kernels/__div_scalar.c"
#include "../kernels/__sub.c"
#include "../kernels/__sub_scalar.c"
#include "../kernels/__div.c"
#include "../kernels/__mul.c"
#include "../kernels/__add.c"
#include "../kernels/__matmul.c"
#include "../kernels/__sum.c"
#include "../kernels/__mean.c"

#include "../kernels/__transpose.c"
#include "../kernels/__power.c"
#include "../kernels/__power_scalar.c"
#include "../kernels/__cast_to_float.c"
#include "../kernels/__batch_matmul.c"