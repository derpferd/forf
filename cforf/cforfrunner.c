#include <string.h>
#include <stdlib.h>
#include "forf.h"

#define LENV_SIZE 100

#define DSTACK_SIZE 200
#define CSTACK_SIZE 500
#define MEMORY_SIZE 10


int main(int argc, char *argv[]) {
  struct forf_env env;
  if (argc < 2 + MEMORY_SIZE) {
    fprintf(stderr, "Too few args\n");
    return 1;
  }



//  fprintf(stdout, "input: '%s' len: %d", argv[1], a);

  struct forf_lexical_env lenv[LENV_SIZE];
  if (!forf_extend_lexical_env(lenv, forf_base_lexical_env, LENV_SIZE)) {
      fprintf(stderr, "Unable to initialize lexical environment.\n");
      return 1;
  }

  struct forf_value cmdvals[CSTACK_SIZE];
  struct forf_value datavals[DSTACK_SIZE];
  long memvals[MEMORY_SIZE];

  for (int i=0; i < MEMORY_SIZE; i++) {
    char *ptr;
    memvals[i] = strtol(argv[i+2], &ptr, 10);
  }

  struct forf_stack cmd;
  struct forf_stack data;
  struct forf_memory mem;

  forf_stack_init(&cmd, cmdvals, CSTACK_SIZE);
  forf_stack_init(&data, datavals, DSTACK_SIZE);
  forf_memory_init(&mem, memvals, MEMORY_SIZE);
  forf_env_init(&env, lenv, &data, &cmd, &mem, NULL);

  forf_parse_string(&env, argv[1]);
  forf_eval(&env);

  fprintf(stdout, "{\"error_code\": %d, \"mem\": [", env.error);

  for (int i=0; i < MEMORY_SIZE; i++) {
    fprintf(stdout, "%ld", memvals[i]);
    if (i != MEMORY_SIZE-1) {
      fprintf(stdout, ", ");
    }
  }
  fprintf(stdout, "]}\n");

  return 0;
}