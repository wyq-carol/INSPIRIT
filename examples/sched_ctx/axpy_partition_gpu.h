#pragma once


__device__ static uint get_smid(void) {
#if defined(__CUDACC__)
  uint ret;
  asm("mov.u32 %0, %smid;" : "=r"(ret) );
  return ret;
#else
  return 0;
#endif
}


#define __P_HKARGS    dimGrid,     active_blocks     ,occupancy,               block_assignment_d,   mapping_start
#define __P_KARGS dim3 blocks, int active_blocks, int occupancy, unsigned int* block_assignment, int mapping_start

#define __P_DARGS blocks,blockid

#define __P_BEGIN							\
__shared__ unsigned int block_start;					\
int smid = get_smid();							\
if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)		\
  {									\
    block_start = atomicDec(&block_assignment[smid],0xDEADBEEF);	\
  }									\
__syncthreads();							\
									\
if(block_start > active_blocks)						\
  {									\
    return;								\
  }									

#define __P_LOOPXY							\
  dim3 blockid;								\
  blockid.z = 0;							\
									\
  int gridDim_sum = blocks.x*blocks.y;					\
  int startBlock = block_start + (smid - mapping_start) * occupancy;	\
									\
  for(int blockid_sum = startBlock; blockid_sum < gridDim_sum; blockid_sum +=active_blocks) \
    {									\
  blockid.x = blockid_sum % blocks.x;					\
  blockid.y = blockid_sum / blocks.x;

#define __P_LOOPEND }
// Needed if shared memory is used
#define __P_LOOPEND_SAFE __syncthreads(); }

#define __P_LOOPX							\
  dim3 blockid;								\
  blockid.z = 0;							\
  blockid.y = 0;							\
  int gridDim_sum = blocks.x;						\
  int startBlock = (smid-mapping_start) + block_start*(active_blocks/occupancy); \
									\
  for(int blockid_sum = startBlock; blockid_sum < gridDim_sum; blockid_sum +=active_blocks) \
    {									\
  blockid.x = blockid_sum;


  //  int startBlock = block_start + (smid - mapping_start) * occupancy; \


//////////// HOST side functions


template <typename F>
static void buildPartitionedBlockMapping(F cudaFun, int threads, int shmem, int mapping_start, int allocation,
				  int &width, int &active_blocks, unsigned int *block_assignment_d,cudaStream_t current_stream = cudaStreamPerThread)
{
  int occupancy;
  int nb_SM = 13; //TODO: replace with call
  int mapping_end = mapping_start + allocation - 1; // exclusive
  unsigned int block_assignment[15];
  
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy,cudaFun,threads,shmem);
  //occupancy = 4;
  width = occupancy * nb_SM; // Physical wrapper grid size. Fits GPU exactly
  active_blocks = occupancy*allocation; // The total number of blocks doing work

  for(int i = 0; i < mapping_start; i++)
    block_assignment[i] = (unsigned) -1;

  for(int i = mapping_start; i <= mapping_end; i++)
    {
      block_assignment[i] = occupancy - 1;
    }

  for(int i = mapping_end+1; i < nb_SM; i++)
    block_assignment[i] = (unsigned) -1;

  cudaMemcpyAsync((void*)block_assignment_d,block_assignment,sizeof(block_assignment),cudaMemcpyHostToDevice, current_stream);
  //cudaMemcpy((void*)block_assignment_d,block_assignment,sizeof(block_assignment),cudaMemcpyHostToDevice);
}



#define __P_HOSTSETUP(KERNEL,GRIDDIM,BLOCKSIZE,SHMEMSIZE,MAPPING_START,MAPPING_END,STREAM)	\
  unsigned int* block_assignment_d; cudaMalloc((void**) &block_assignment_d,15*sizeof(unsigned int)); \
  int width = 0;							\
  int active_blocks = 0;						\
  buildPartitionedBlockMapping(KERNEL,BLOCKSIZE,SHMEMSIZE,(MAPPING_START),(MAPPING_END)-(MAPPING_START), \
			       width, active_blocks, block_assignment_d,STREAM); \
  int occupancy = active_blocks/((MAPPING_END)-(MAPPING_START));		\
  dim3 dimGrid = (GRIDDIM);\
  int mapping_start = (MAPPING_START);
