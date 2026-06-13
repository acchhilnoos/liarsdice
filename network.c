#include "network.h"
#include "game.h"
#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct Network *network_new(void) {
  struct Network *n = malloc(sizeof(*n));

  // kernels: [ky, kx, ic, oc]
  tensor_init(&n->ks[0], 1, 1, NUM_INPUTS, 256);
  tensor_init(&n->ks[1], 1, 1, 256, 256);
  tensor_init(&n->ks[2], 1, 1, 256, 256);
  tensor_init(&n->ks[3], 1, 1, 256, NUM_OUTPUTS);
  tensor_init(&n->ks[4], 1, 1, 256, 1);

  for (size_t i = 0; i < NUM_LAYERS; i++) {
    struct Tensor *ks = &n->ks[i];
    struct Tensor *bs = &n->bs[i];
    struct Tensor *as = &n->as[i];

    // biases: [1, 1, 1, oc]
    tensor_init(bs, 1, 1, 1, ks->c);

    // activations: [MAX_BATCH_SIZE, y, x, c]
    tensor_init(as, 1, 1, 1, ks->c);
    as->n = 1;

    // momenta
    tensor_init(&n->m_ks[i], ks->n, ks->y, ks->x, ks->c);
    tensor_init(&n->m_bs[i], bs->n, bs->y, bs->x, bs->c);

    float fan_in = (float)(ks->n * ks->y * ks->x);
    float limit  = sqrtf(6.0f / (fan_in + ks->c));

    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++)
      ks->buf[j] = (((float)rand() / RAND_MAX) * 2.0f - 1.0f) * limit;
  }

  return n;
}

void network_free(struct Network *n) {
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    tensor_free(&n->ks[i]);
    tensor_free(&n->bs[i]);
    tensor_free(&n->as[i]);
    tensor_free(&n->m_ks[i]);
    tensor_free(&n->m_bs[i]);
  }
  free(n);
}

void network_zero_grad(struct Network *n) {
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    tensor_zero_grad(&n->ks[i]);
    tensor_zero_grad(&n->bs[i]);
    tensor_zero_grad(&n->as[i]);
  }
}

void network_forward(struct Network *n, const struct Tensor *inputs,
                     const struct Game *g) {
  tensor_conv(inputs, &n->ks[0], &n->bs[0], &n->as[0], 0, 1);
  tensor_relu(&n->as[0]);

  tensor_conv(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1], 0, 1);
  tensor_relu(&n->as[1]);

  tensor_conv(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2], 0, 1);
  tensor_relu(&n->as[2]);

  tensor_conv(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3], 0, 1);
  for (size_t b = 0; b < n->as[3].n; b++) {
    for (size_t i = 0; i < 20; i++)
      for (size_t j = 0; j < 6; j++)
        if (!legal(g, i + 1, j + 1))
          n->as[3].buf[b * 121 + i * 6 + j] = -FLT_MAX;
    if (g->d1bid.c == 0)
      n->as[3].buf[b * 121 + 120] = -FLT_MAX;
  }
  tensor_softmax(&n->as[3]);

  tensor_conv(&n->as[2], &n->ks[4], &n->bs[4], &n->as[4], 0, 1);
  tensor_tanh(&n->as[4]);
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      const struct Tensor *loss_p, float loss_v) {
  for (size_t i = 0; i < NUM_LAYERS; i++)
    tensor_zero_grad(&n->as[i]);

  n->as[4].grad[0] = loss_v;
  memcpy(n->as[3].grad, loss_p->buf,
         tensor_size(loss_p) * sizeof(*loss_p->buf));

  tensor_tanh_grad(&n->as[4]);
  tensor_conv_grad(&n->as[2], &n->ks[4], &n->bs[4], &n->as[4], 0, 1);

  tensor_softmax_grad(&n->as[3]);
  tensor_conv_grad(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3], 0, 1);

  tensor_relu_grad(&n->as[2]);
  tensor_conv_grad(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2], 0, 1);

  tensor_relu_grad(&n->as[1]);
  tensor_conv_grad(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1], 0, 1);

  tensor_relu_grad(&n->as[0]);
  tensor_conv_grad(inputs, &n->ks[0], &n->bs[0], &n->as[0], 0, 1);
}

void network_sgd(struct Network *n, float alpha, float beta) {
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++) {
      n->m_ks[i].buf[j] = beta * n->m_ks[i].buf[j] + n->ks[i].grad[j];
      n->ks[i].buf[j] -= alpha * n->m_ks[i].buf[j];
    }
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++) {
      n->m_bs[i].buf[j] = beta * n->m_bs[i].buf[j] + n->bs[i].grad[j];
      n->bs[i].buf[j] -= alpha * n->m_bs[i].buf[j];
    }
  }
}

void network_save(struct Network *n, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f)
    return;
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    fwrite(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fwrite(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
    fwrite(n->m_ks[i].buf, sizeof(float), tensor_size(&n->m_ks[i]), f);
    fwrite(n->m_bs[i].buf, sizeof(float), tensor_size(&n->m_bs[i]), f);
  }
  fclose(f);
}

void network_load(struct Network *n, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return;
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    fread(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fread(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
    fread(n->m_ks[i].buf, sizeof(float), tensor_size(&n->m_ks[i]), f);
    fread(n->m_bs[i].buf, sizeof(float), tensor_size(&n->m_bs[i]), f);
  }
  fclose(f);
}

void network_benchmark(void) {
  printf("Benchmarking (1000 iterations)...\n");

  struct Network *n = network_new();
  struct Game    *g = game_new();

  struct Tensor inputs, loss_p;
  float         loss_v;
  tensor_init(&inputs, 1, 1, 1, NUM_INPUTS);
  tensor_init(&loss_p, 1, 1, 1, NUM_OUTPUTS);
  get_canonical(g, &inputs);
  for (size_t i = 0; i < tensor_size(&loss_p); i++)
    loss_p.buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  loss_v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  for (int i = 0; i < 100; i++) {
    network_forward(n, &inputs, g);
    network_backward(n, &inputs, &loss_p, loss_v);
  }

  clock_t start = clock();
  for (int i = 0; i < 1000; i++) {
    network_forward(n, &inputs, g);
    network_backward(n, &inputs, &loss_p, loss_v);
  }
  clock_t end = clock();

  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Time for 1000 Forward+Backward passes: %f seconds\n", time_spent);
  printf("Average time per pass: %f ms\n", (time_spent / 1000.0) * 1000.0);
  printf("Estimated Evals per Second: %.2f\n\n", 1000.0 / time_spent);

  tensor_free(&loss_p);
  tensor_free(&inputs);
  network_free(n);
  free(g);
}
