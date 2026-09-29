#include "network.h"
#include "play.h"
#include "train.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[])
{
  srand(time(NULL));

  char  *weights_fn  = "weights.bin";
  size_t max_iters   = 10000;
  size_t max_steps   = 2048;
  size_t max_epchs   = 15;
  bool   verbose     = false;
  bool   bench       = false;
  bool   playout_n   = false;
  bool   playout_p   = false;
  float  alpha       = 0.005f;
  float  beta        = 0.9f;
  float  epsilon     = 0.2f;
  float  gamma       = 0.95f;
  gamma             *= gamma;
  gamma             *= gamma;
  float lambda       = 0.99f;
  lambda            *= lambda;
  lambda            *= lambda;
  float c1           = 1.0f;
  float c2           = 0.05f;

  struct Network *n = network_new();
  if (!n) goto fail_n;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-b") == 0) {
      bench = true;
    } else if (strcmp(argv[i], "-c1") == 0) {
      c1 = atof(argv[++i]);
    } else if (strcmp(argv[i], "-c2") == 0) {
      c2 = atof(argv[++i]);
    } else if (strcmp(argv[i], "-e") == 0) {
      max_epchs = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-i") == 0) {
      max_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-l") == 0) {
      if (argc > i + 1) weights_fn = argv[++i];
      network_load(n, weights_fn);
    } else if (strcmp(argv[i], "-s") == 0) {
      max_steps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "-n") == 0) {
      playout_n = true;
    } else if (strcmp(argv[i], "-p") == 0) {
      playout_p = true;
    }
  }

  if (bench) {
    network_benchmark();
  } else if (playout_n) {
    if (playout(n, false, verbose) != 0) goto fail;
  } else if (playout_p) {
    if (playout(n, true, true) != 0) goto fail;
  } else {
    if (train(n, weights_fn, max_iters, max_steps, max_epchs, alpha, beta,
              gamma, epsilon, lambda, c1, c2, verbose) != 0)
      goto fail;
  }

  network_free(n);
  return 0;

fail:
  network_free(n);
fail_n:
  return 1;
}
