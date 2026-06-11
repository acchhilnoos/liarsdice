#include "network.h"
#include "game.h"
#include "tensor.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct Network *network_new(void) {
  struct Network *n = malloc(sizeof(*n));

  // kernels: [ky, kx, ic, oc]
  tensor_init(&n->ks[0], 1, 1, NUM_INPUTS, 512);
  tensor_init(&n->ks[1], 1, 1, 512, 512);
  tensor_init(&n->ks[2], 1, 1, 512, 256);
  tensor_init(&n->ks[3], 1, 1, 256, NUM_OUTPUTS);

  // biases: [1, 1, 1, oc]
  for (size_t i = 0; i < NUM_LAYERS; i++)
    tensor_init(&n->bs[i], 1, 1, 1, n->ks[i].c);

  // activations: [MAX_BATCH_SIZE, y, x, c]
  tensor_init(&n->as[0], MAX_BATCH_SIZE, 1, 1, 512);
  tensor_init(&n->as[1], MAX_BATCH_SIZE, 1, 1, 512);
  tensor_init(&n->as[2], MAX_BATCH_SIZE, 1, 1, 256);
  tensor_init(&n->as[3], MAX_BATCH_SIZE, 1, 1, NUM_OUTPUTS);

  for (int i = 0; i < NUM_LAYERS; i++)
    n->as[i].n = 1;

  for (size_t i = 0; i < NUM_LAYERS; i++) {
    struct Tensor *ks = &n->ks[i];
    struct Tensor *bs = &n->bs[i];

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
          n->as[3].buf[b * 121 + i * 6 + j] = 0;
    if (g->d1bid.c == 0)
      n->as[3].buf[b * 121 + 120] = 0;
  }
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      const struct Tensor *loss) {
  for (size_t i = 0; i < NUM_LAYERS; i++)
    tensor_zero_grad(&n->as[i]);

  memcpy(n->as[3].grad, loss->buf, tensor_size(loss) * sizeof(*loss->buf));

  tensor_conv_grad(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3], 0, 1);

  tensor_relu_grad(&n->as[2]);
  tensor_conv_grad(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2], 0, 1);

  tensor_relu_grad(&n->as[1]);
  tensor_conv_grad(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1], 0, 1);

  tensor_relu_grad(&n->as[0]);
  tensor_conv_grad(inputs, &n->ks[0], &n->bs[0], &n->as[0], 0, 1);
}

void network_sgd(struct Network *n, float alpha) {
  float m = 0.9f;
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++) {
      n->m_ks[i].buf[j] = m * n->m_ks[i].buf[j] + n->ks[i].grad[j];
      n->ks[i].buf[j] -= alpha * n->m_ks[i].buf[j];
    }
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++) {
      n->m_bs[i].buf[j] = m * n->m_bs[i].buf[j] + n->bs[i].grad[j];
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

  struct Tensor inputs, loss;
  tensor_init(&inputs, 1, 1, 1, NUM_INPUTS);
  tensor_init(&loss, 1, 1, 1, NUM_OUTPUTS);
  get_canonical(g, &inputs);
  for (size_t i = 0; i < tensor_size(&loss); i++)
    loss.buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  for (int i = 0; i < 100; i++) {
    network_forward(n, &inputs, g);
    network_backward(n, &inputs, &loss);
  }

  clock_t start = clock();
  for (int i = 0; i < 1000; i++) {
    network_forward(n, &inputs, g);
    network_backward(n, &inputs, &loss);
  }
  clock_t end = clock();

  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Time for 1000 Forward+Backward passes: %f seconds\n", time_spent);
  printf("Average time per pass: %f ms\n", (time_spent / 1000.0) * 1000.0);
  printf("Estimated Evals per Second: %.2f\n\n", 1000.0 / time_spent);

  tensor_free(&loss);
  tensor_free(&inputs);
  network_free(n);
  free(g);
}
