#include <string.h>
#include <stdlib.h>
#include "forf.h"
#include "libcforf.h"

#define LENV_SIZE 100

#define DSTACK_SIZE 200
#define CSTACK_SIZE 500
#define MEMORY_SIZE 10
#define MAX_FUNCS  60


int main(int argc, char *argv[]) {
  struct forf_env env;
  // the next num_of_funcs triplets are the (# ins, # effects, # outs)
  // the next func_slots args are the function slot values
  // the last args are the function definitions
  int num_static_args = 1 + // first arg is the exe path
                        1 + // second arg is the forf program to be run
                        1 + // third arg is the random seed value
                        MEMORY_SIZE + // the next MEMORY_SIZE args are the memory values
                        1 + // the next arg is the number of function slots
                        1; // the next arg is the number of functions
                        // The number args depend on the inputs
  int func_slots_index = 1 + 1 + 1 + MEMORY_SIZE;
  int num_funcs_index = 1 + 1 + 1 + MEMORY_SIZE + 1;

  if (argc < num_static_args) {
    fprintf(stderr, "Too few args\n");
    return 1;
  }
  int func_slots = atoi(argv[func_slots_index]);
  int num_funcs = atoi(argv[num_funcs_index]);

  if (num_funcs > MAX_FUNCS) {
    fprintf(stderr, "Too many funcs; can only handle %d functions\n", MAX_FUNCS);
    return 1;
  }

  int num_args_for_funcs_triplets = num_static_args + (num_funcs * 3);
  if (argc < num_args_for_funcs_triplets) {
    fprintf(stderr, "Too few args; not enough for func triplets\n");
    return 1;
  }

  struct forf_func funcs[num_funcs];
  struct slots   input_slots[num_funcs];
  struct effects effects[num_funcs];
  struct slots   output_slots[num_funcs];

  int total_num_args = num_args_for_funcs_triplets + func_slots;

  size_t num_slot_refs = 0;
  size_t num_effects = 0;
  for (int i=0; i < num_funcs; i++) {
    char *ptr;
    funcs[i].input_slots = &input_slots[i];
    funcs[i].effects = &effects[i];
    funcs[i].output_slots = &output_slots[i];
    input_slots[i].size = strtol(argv[num_static_args + (i*3)], &ptr, 10);
    effects[i].size = strtol(argv[num_static_args + (i*3) + 1], &ptr, 10);
    output_slots[i].size = strtol(argv[num_static_args + (i*3) + 2], &ptr, 10);
    total_num_args += 1 + input_slots[i].size + (effects[i].size * 2) + output_slots[i].size;
    num_slot_refs += input_slots[i].size + output_slots[i].size;
    num_effects += effects[i].size;
  }

//  num_args += func_slots;
  if (argc < total_num_args) {
    fprintf(stderr, "Too few args; not enough for all args\n");
    return 1;
  }

//  int num_funcs = atoi(argv[num_args]);
//  num_args += num_funcs * 3;

//  fprintf(stdout, "input: '%s' len: %d", argv[1], a);
  char *ptr;
  unsigned long rand_seed = strtoul(argv[2], &ptr, 10);


  struct forf_value cmdvals[CSTACK_SIZE];
  struct forf_value datavals[DSTACK_SIZE];
  long memvals[MEMORY_SIZE];
  long slotvals[func_slots];

  for (int i=0; i < MEMORY_SIZE; i++) {
    char *ptr;
    memvals[i] = strtol(argv[i+3], &ptr, 10);
  }

  for (int i=0; i < func_slots; i++) {
    char *ptr;
    slotvals[i] = strtol(argv[num_args_for_funcs_triplets+i], &ptr, 10);
  }

  char* tokens[num_funcs];
  long slot_refs[num_slot_refs];
  struct effect effect_list[num_effects];
  int func_def_start_index = num_args_for_funcs_triplets + func_slots;
  int cur_func_start_index = func_def_start_index;
  int cur_slot_ref_index = 0;
  int cur_effect_index = 0;
  for (int i = 0; i < num_funcs; ++i) {
    char *ptr;
    tokens[i] = argv[cur_func_start_index];
    cur_func_start_index += 1;

    input_slots[i].slots = &slot_refs[cur_slot_ref_index];
    for (int j = 0; j < input_slots[i].size; ++j) {
      slot_refs[cur_slot_ref_index] = strtol(argv[cur_func_start_index], &ptr, 10);
      cur_slot_ref_index += 1;
      cur_func_start_index += 1;
    }

    effects[i].effects = &effect_list[cur_effect_index];
    for (int j = 0; j < input_slots[i].size; ++j) {
      effect_list[cur_effect_index].value = strtol(argv[cur_func_start_index], &ptr, 10);
      effect_list[cur_effect_index].slot = strtol(argv[cur_func_start_index+1], &ptr, 10);
      cur_effect_index += 2;
      cur_func_start_index += 2;
    }

    output_slots[i].slots = &slot_refs[cur_slot_ref_index];
    for (int j = 0; j < output_slots[i].size; ++j) {
      slot_refs[cur_slot_ref_index] = strtol(argv[cur_func_start_index], &ptr, 10);
      cur_slot_ref_index += 1;
      cur_func_start_index += 1;
    }
  }

//  long in_slots[] = {0};
//  struct slots   input_slots;
//  forf_slots_init(&input_slots, in_slots, 1);
//
//  struct effect effect[] = {{23, 1}};
//  struct effects effects = {1, effect};
//
//  long out_slots[] = {2};
//  struct slots   output_slots;
//  forf_slots_init(&output_slots, out_slots, 1);
//
//  struct forf_func test_func = {&input_slots, &effects, &output_slots};

  struct forf_lexical_env extra_procs_lexical_env[] = {
//          {"random", forf_type_proc, forf_random},
          {NULL, forf_type_proc, NULL}
  };

  struct forf_lexical_env extra_funcs_lexical_env[num_funcs+1];

  for (int i = 0; i < num_funcs; ++i) {
    extra_funcs_lexical_env[i].name = tokens[i];
    extra_funcs_lexical_env[i].type = forf_type_function;
    extra_funcs_lexical_env[i].v.f = &funcs[i];
  }

  extra_funcs_lexical_env[num_funcs].name = NULL;
  extra_funcs_lexical_env[num_funcs].v.p = NULL;
//          {"test", forf_type_function, .v.f=&test_func},

  struct forf_lexical_env lenv[LENV_SIZE];
  if ((!forf_extend_lexical_env(lenv, forf_base_lexical_env, LENV_SIZE) ||
       !forf_extend_lexical_env(lenv, extra_procs_lexical_env, LENV_SIZE) ||
       !forf_extend_lexical_env(lenv, extra_funcs_lexical_env, LENV_SIZE))) {
    fprintf(stderr, "Unable to initialize lexical environment.\n");
    return 1;
  }

  struct forf_stack cmd;
  struct forf_stack data;
  struct forf_memory mem;
  struct forf_memory slots;

  forf_stack_init(&cmd, cmdvals, CSTACK_SIZE);
  forf_stack_init(&data, datavals, DSTACK_SIZE);
  forf_memory_init(&mem, memvals, MEMORY_SIZE);
  forf_memory_init(&slots, slotvals, func_slots);
  env.rand_seed = &rand_seed;
  forf_env_init(&env, lenv, &data, &cmd, &mem, &slots, NULL);
  env.error = forf_error_none;

  forf_run(argv[1], extra_funcs_lexical_env, &env);

  fprintf(stdout, "{\"error_code\": %d, \"rand\": %ld, \"mem\": [", env.error, rand_seed);

  for (int i=0; i < MEMORY_SIZE; i++) {
    fprintf(stdout, "%ld", memvals[i]);
    if (i != MEMORY_SIZE-1) {
      fprintf(stdout, ", ");
    }
  }
  fprintf(stdout, "], \"slots\": [");

  for (int i=0; i < func_slots; i++) {
    fprintf(stdout, "%ld", slotvals[i]);
    if (i != func_slots-1) {
      fprintf(stdout, ", ");
    }
  }
  fprintf(stdout, "]}\n");

  return 0;
}