#include <stdio.h>
#include <malloc.h>
#include <starpu.h>

static inline void my_codelet_cpu(void *descr[], void *_args)
{
  unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
  float *sub = (float *)STARPU_VECTOR_GET_PTR(descr[0]);

  unsigned i;

  for(i = 0; i < nx; i++){
    sub[i] *= 5;
  }
}

extern void my_codelet_gpu(void *descr[], __attribute__ ((unused)) void *_args);


static starpu_codelet cl = 
  {
    .where = STARPU_CPU|STARPU_CUDA,
    .cpu_func = my_codelet_cpu,
    .cuda_func = my_codelet_gpu,
    .nbuffers = 1
  };

void print_vect(int *vect, int size){
  unsigned i;
  for(i = 0; i < size; i++)
    printf("%d ", vect[i]);
  printf("\n");
  
}

int main(int argc, char **argv)
{
  srand(time(NULL));
  int *mat;
  unsigned size = 20, children = 5;
  mat = (int *)malloc(size*sizeof(int));

  unsigned i;
  for(i = 0; i < size; i++)
    {
      mat[i] = random()% 10 + 1;
    }

  print_vect(mat, size);

  //  struct starpu_conf conf;
  //conf.sched_policy_name = "heft-tm";
  //conf.ncpus = -1;
  // printf("got here \n");
  starpu_init(NULL);

  starpu_data_handle dataA;
  starpu_vector_data_register(&dataA, 0, (uintptr_t)mat, size, sizeof(mat[00]));

  struct starpu_data_filter f =
    {
      .filter_func = starpu_block_filter_func_vector,
      .nchildren = children,
      .get_nchildren = NULL,
      .get_child_ops = NULL
    };
  starpu_data_partition(dataA, &f);

  struct starpu_sched_ctx sched_ctx;
  int procs[] = {1, 2, 3};
  starpu_create_sched_ctx(&sched_ctx, "heft", procs, 3);

  unsigned j;
  for(j = 0; j < children; j++){
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl;
    task->synchronous = 1;
    task->callback_func = NULL;
    task->buffers[0].handle = starpu_data_get_sub_data(dataA, 1, j);
    task->buffers[0].mode = STARPU_RW;
    task->name = "first 1 2 3";  
    starpu_task_submit_to_ctx(task, &sched_ctx);
  }

  int procs_to_remove[]={1,3};
  starpu_remove_workers_from_sched_ctx(procs_to_remove, 2, &sched_ctx);

  printf("procs removed \n");

  for(j = 0; j < children; j++){
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl;
    task->synchronous = 1;
    task->callback_func = NULL;
    task->buffers[0].handle = starpu_data_get_sub_data(dataA, 1, j);
    task->buffers[0].mode = STARPU_RW;
    task->name = "first 2";  
    starpu_task_submit_to_ctx(task, &sched_ctx);
  }

  int procs_to_add[]={1, 4, 5};
  starpu_add_workers_to_sched_ctx(procs_to_add, 2, &sched_ctx);

  printf("procs add \n");

  for(j = 0; j < children; j++){
    struct starpu_task *task = starpu_task_create();
    task->cl = &cl;
    task->synchronous = 1;
    task->callback_func = NULL;
    task->buffers[0].handle = starpu_data_get_sub_data(dataA, 1, j);
    task->buffers[0].mode = STARPU_RW;
    task->name = "first 1 2 4 5";  
    starpu_task_submit_to_ctx(task, &sched_ctx);
  }


  struct starpu_sched_ctx sched_ctx2;
  int procs2[]={3, 4, 5, 6, 7};
  starpu_create_sched_ctx(&sched_ctx2, "random", procs2, 5);

  for(j = 0; j < children; j++){
    struct starpu_task *task3 = starpu_task_create();
    task3->cl = &cl;
    task3->synchronous = 1;
    task3->callback_func = NULL;
    task3->buffers[0].handle = starpu_data_get_sub_data(dataA, 1, j);
    task3->buffers[0].mode = STARPU_RW;
    task3->name = "third 3 4 5 6 7";
    starpu_task_submit_to_ctx(task3, &sched_ctx2);
  }

  for(j = 0; j < children; j++){
    struct starpu_task *task2 = starpu_task_create();
    task2->cl = &cl;
    task2->synchronous = 1;
    task2->callback_func = NULL;
    task2->buffers[0].handle = starpu_data_get_sub_data(dataA, 1, j);
    task2->buffers[0].mode = STARPU_RW;
    task2->name = "anything";
    starpu_task_submit(task2);
  }
  
  printf("wait for all \n");
  starpu_task_wait_for_all();
  starpu_data_unpartition(dataA, 0);

  printf("data unregister  \n");
  starpu_data_unregister(dataA);
  
  printf("the end \n");
  starpu_shutdown();

  print_vect(mat, size);
  
  return 0;
}
