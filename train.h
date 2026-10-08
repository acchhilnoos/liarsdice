#ifndef TRAIN_H
#define TRAIN_H

#include "network.h"
#include <stdlib.h>

int train(struct Network *n, const char *weights_fn, size_t max_iters,
          size_t max_steps, size_t max_epchs, float alpha, float beta,
          float epsilon, float gamma, float lambda, float c1, float c2,
          bool verbose);

#endif
