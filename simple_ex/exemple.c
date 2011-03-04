#include <starpu.h>
#include <stdio.h>
#include <time.h>
#include <malloc.h>

static inline void my_codelet(void *descr[], void *_args)
{
  unsigned nx = STARPU_VECTOR_GET_NX(descr[0]);
  float *sub = (float *)STARPU_VECTOR_GET_PTR(descr[0]);

  unsigned i;
  for(i = 0; i < nx; i++){
    sub[i] *= 5;
  }
}

static starpu_codelet cl = 
  {
    .where = STARPU_CPU,
    .cpu_func = my_codelet,
    .nbuffers = 1
  };

void print_vect(float *vect, int size){
  unsigned i;
  for(i = 0; i < size; i++)
    printf("%4.1f ", vect[i]);
  printf("\n");
  
}
int main(int argc, char **argv)
{
  srand(time(NULL));
  float *mat;
  unsigned size = 20, children = 5;
  mat = malloc(size*sizeof(float));

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
  starpu_vector_data_register(&dataA, 0, (uintptr_t)mat, size, sizeof(float));

  starpu_data_set_sequential_consistency_flag(dataA, 0);
  struct starpu_data_filter f;
  f.filter_func = starpu_vector_list_filter_func;
  f.nchildren = children;
  f.get_nchildren = NULL;
  f.get_child_ops = NULL;
  int len[] = {4, 4, 4, 4, 4};
  f.filter_arg_ptr = len;
  
  starpu_data_partition(dataA, &f);
  starpu_data_map_filters(dataA, 1, &f);

  struct starpu_sched_ctx sched_ctx;
  int procs[]={1, 2, 3};
  starpu_create_sched_ctx(&sched_ctx, "random", procs, 3);

  struct starpu_task *task = starpu_task_create();
  task->cl = &cl;
  task->buffers[0].handle = starpu_data_get_sub_data(dataA, 0);
  task->buffers[0].mode = STARPU_R;
  task->name = "first 1 2 3";
  starpu_task_submit_to_ctx(task, &sched_ctx);

  struct starpu_sched_ctx sched_ctx2;
  int procs2[]={4, 5, 6, 7};
  starpu_create_sched_ctx(&sched_ctx2, "random", procs2, 4);

  struct starpu_task *task3 = starpu_task_create();
  task3->cl = &cl;
  task3->buffers[0].handle = starpu_data_get_sub_data(dataA, 0);
  task3->buffers[0].mode = STARPU_R;
  task3->name = "third 4 5 6 7";
  starpu_task_submit_to_ctx(task3, &sched_ctx2);


  struct starpu_task *task2 = starpu_task_create();
  task2->cl = &cl;
  task2->buffers[0].handle = starpu_data_get_sub_data(dataA, 0);
  task2->buffers[0].mode = STARPU_R;
  task2->name = "second > 7";
  starpu_task_submit(task2);

  
  starpu_task_wait_for_all();
 
  starpu_data_unpartition(dataA, 0);
  
  starpu_data_unregister(dataA);
  
  starpu_shutdown();

  print_vect(mat, size);
  
  return 0;
}
