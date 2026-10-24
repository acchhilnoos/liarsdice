#ifndef NETWORK_H
#define NETWORK_H

#include "game.h"

struct Network;

struct Network *network_new(void);
void            network_free(struct Network *n);

void network_zero_grad(struct Network *n);

void network_forward(const struct Network *n, const struct Tensor *in,
                     const struct Tensor *h_prev, const struct Game *g,
                     struct Tensor activs[NUM_TOTAL_LAYERS]);
void network_backward(struct Network *n, struct Tensor *in,
                      struct Tensor       *h_prev,
                      struct Tensor        activs[NUM_TOTAL_LAYERS],
                      const struct Tensor *loss_p, float loss_v);

void network_sgd(struct Network *n, float alpha, float beta);

void network_peek(const struct Network *n, const struct Game *g,
                  const struct Tensor *hp);
int  network_save(const struct Network *n, const char *path);
int  network_load(struct Network *n, const char *path);
void network_benchmark(void);

#endif
