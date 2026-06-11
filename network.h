#ifndef NETWORK_H
#define NETWORK_H

#include "game.h"
#include "tensor.h"

// WARN: MUST CHANGE WITH NUM_FACES,PLAYERS IN game.h
/*
 * last last count x1
 * last last face  x1
 * last count      x1
 * last face       x1
 * turn            x1
 * opp 1-3 left    x3
 * hand 1-6s       x6
 */
#define NUM_INPUTS 14
/*
 * [count: 1-20] x [face: 1-6] x120
 * challenge                   x1
 */
#define NUM_OUTPUTS 121
#define NUM_LAYERS 4
#define MAX_BATCH_SIZE 1

struct Network {
  struct Tensor ks[NUM_LAYERS];
  struct Tensor bs[NUM_LAYERS];
  struct Tensor as[NUM_LAYERS];

  struct Tensor m_ks[NUM_LAYERS];
  struct Tensor m_bs[NUM_LAYERS];
};

struct Network *network_new(void);
void network_free(struct Network *n);

void network_zero_grad(struct Network *n);

void network_forward(struct Network *n, const struct Tensor *inputs,
                     const struct Game *g);
void network_backward(struct Network *n, struct Tensor *inputs,
                      const struct Tensor *loss);

void network_sgd(struct Network *n, float alpha);

void network_save(struct Network *n, const char *path);
void network_load(struct Network *n, const char *path);

void network_benchmark(void);

#endif
