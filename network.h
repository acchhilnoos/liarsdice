#ifndef NETWORK_H
#define NETWORK_H

#include "game.h"
#include "tensor.h"
#include "config.h"

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
                      const struct Tensor *loss_p, float loss_v);

void network_sgd(struct Network *n, float alpha, float beta);

void network_peek(const struct Network *n);

void network_save(struct Network *n, const char *path);
void network_load(struct Network *n, const char *path);

void network_benchmark(void);

#endif
