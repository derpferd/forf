#include "libcforf.h"

#define LENV_SIZE 100

void forf_run(char* code, struct forf_lexical_env* funcs_lexical_env, struct forf_env* env) {
  struct forf_lexical_env lenv[LENV_SIZE];
  lenv[0].name = NULL;
  if ((!forf_extend_lexical_env(lenv, forf_base_lexical_env, LENV_SIZE) ||
       !forf_extend_lexical_env(lenv, funcs_lexical_env, LENV_SIZE))) {
    env->error = forf_error_init;
    return;
  }
  env->lenv = lenv;

  forf_parse_string(env, code);
  if (!env->error) {
    forf_eval(env);
  }
}
