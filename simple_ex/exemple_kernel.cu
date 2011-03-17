#include <starpu.h>
#include <starpu_cuda.h>

static __global__ void myf(int *dMatA)
{
  int tidy = threadIdx.y;

  dMatA[ tidy ] = dMatA[ tidy ]  * 5;
}

extern "C" void my_codelet_gpu(void *descr[], void *_args)
{
  unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
  int *sub = (int *)STARPU_VECTOR_GET_PTR(descr[0]);

  dim3 dimGrid(1,1);
  dim3 dimBlock(nx,nx);

  myf<<<dimGrid, dimBlock, 0, starpu_cuda_get_local_stream()>>>(sub);
 
  cudaStreamSynchronize(starpu_cuda_get_local_stream());
}
