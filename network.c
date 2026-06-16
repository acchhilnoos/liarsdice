#include "network.h"
#include "config.h"
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

  tensor_init(&n->ks[0], NUM_INPUTS, 128);
  tensor_init(&n->ks[1], 128, 128);

  tensor_init(&n->ks[2], 128, 128);
  tensor_init(&n->ks[3], 128, NUM_POL_OUT);

  tensor_init(&n->ks[4], 128, 128);
  tensor_init(&n->ks[5], 128, 1);

  tensor_init(&n->ks[6], 128, 128);
  tensor_init(&n->ks[7], 128, NUM_FACES);

  for (size_t i = 0; i < NUM_LAYERS; i++) {
    struct Tensor *ks = &n->ks[i];
    struct Tensor *bs = &n->bs[i];
    struct Tensor *as = &n->as[i];

    tensor_init(bs, 1, ks->x);
    tensor_init(as, 1, ks->x);

    tensor_init(&n->m_ks[i], ks->y, ks->x);
    tensor_init(&n->m_bs[i], bs->y, bs->x);

    float limit = sqrtf(6.0f / (ks->y + ks->x));
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
  tensor_fc(inputs, &n->ks[0], &n->bs[0], &n->as[0]);
  tensor_relu(&n->as[0]);
  tensor_fc(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1]);
  tensor_relu(&n->as[1]);

  tensor_fc(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2]);
  tensor_relu(&n->as[2]);
  tensor_fc(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3]);
  for (size_t i = 0; i < NUM_PLAYERS * 5; i++)
    for (size_t j = 0; j < NUM_FACES; j++)
      if (!legal(g, i + 1, j + 1))
        n->as[3].buf[i * NUM_FACES + j] = -FLT_MAX;
  if (g->last.c == 0)
    n->as[3].buf[CHALLENGE_IDX] = -FLT_MAX;
  tensor_softmax(&n->as[3]);

  tensor_fc(&n->as[1], &n->ks[4], &n->bs[4], &n->as[4]);
  tensor_relu(&n->as[4]);
  tensor_fc(&n->as[4], &n->ks[5], &n->bs[5], &n->as[5]);
  // tensor_tanh(&n->as[5]);

  tensor_fc(&n->as[1], &n->ks[6], &n->bs[6], &n->as[6]);
  tensor_relu(&n->as[6]);
  tensor_fc(&n->as[6], &n->ks[7], &n->bs[7], &n->as[7]);
  tensor_softmax(&n->as[7]);
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      const struct Tensor *loss_p, float loss_v,
                      const struct Tensor *loss_c) {
  for (size_t i = 0; i < NUM_LAYERS; i++)
    tensor_zero_grad(&n->as[i]);

  memcpy(n->as[7].grad, loss_c->buf,
         tensor_size(loss_c) * sizeof(*loss_c->buf));
  n->as[5].grad[0] = loss_v;
  memcpy(n->as[3].grad, loss_p->buf,
         tensor_size(loss_p) * sizeof(*loss_p->buf));

  tensor_softmax_grad(&n->as[7]);
  tensor_fc_grad(&n->as[6], &n->ks[7], &n->bs[7], &n->as[7]);
  tensor_relu_grad(&n->as[6]);
  tensor_fc_grad(&n->as[1], &n->ks[6], &n->bs[6], &n->as[6]);

  // tensor_tanh_grad(&n->as[5]);
  tensor_fc_grad(&n->as[4], &n->ks[5], &n->bs[5], &n->as[5]);
  tensor_relu_grad(&n->as[4]);
  tensor_fc_grad(&n->as[1], &n->ks[4], &n->bs[4], &n->as[4]);

  tensor_softmax_grad(&n->as[3]);
  tensor_fc_grad(&n->as[2], &n->ks[3], &n->bs[3], &n->as[3]);
  tensor_relu_grad(&n->as[2]);
  tensor_fc_grad(&n->as[1], &n->ks[2], &n->bs[2], &n->as[2]);

  tensor_relu_grad(&n->as[1]);
  tensor_fc_grad(&n->as[0], &n->ks[1], &n->bs[1], &n->as[1]);
  tensor_relu_grad(&n->as[0]);
  tensor_fc_grad(inputs, &n->ks[0], &n->bs[0], &n->as[0]);
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

void network_peek(const struct Network *n) {
  const char *RED    = "\033[31m";
  const char *YELLOW = "\033[33m";
  const char *GREEN  = "\033[32m";
  const char *DIM    = "\033[2m";
  const char *BOLD   = "\033[1m";
  const char *RESET  = "\033[0m";

  float min_val = n->as[POL_HEAD].buf[0];
  float max_val = n->as[POL_HEAD].buf[0];
  for (size_t i = 0; i < NUM_POL_OUT; i++) {
    if (n->as[POL_HEAD].buf[i] < min_val)
      min_val = n->as[POL_HEAD].buf[i];
    if (n->as[POL_HEAD].buf[i] > max_val)
      max_val = n->as[POL_HEAD].buf[i];
  }

  float range = max_val - min_val;

  for (size_t i = 0; i < NUM_FACES; i++) {
    for (size_t j = 0; j < NUM_PLAYERS * 5; j++) {
      float       val = n->as[POL_HEAD].buf[j * NUM_FACES + i];
      const char *color;
      if (range > 1e-8f) {
        float norm = (val - min_val) / range;
        if (norm > 0.66f)
          color = GREEN;
        else if (norm > 0.33f)
          color = YELLOW;
        else if (norm > 0.01f)
          color = RED;
        else
          color = DIM;
      } else {
        color = GREEN;
      }
      printf("%s%6.3f%s", color, val, RESET);
    }
    printf("\n");
  }

  float       doubt_val = n->as[POL_HEAD].buf[CHALLENGE_IDX];
  const char *doubt_color;
  if (range > 1e-8f) {
    float norm = (doubt_val - min_val) / range;
    if (norm > 0.66f)
      doubt_color = GREEN;
    else if (norm > 0.33f)
      doubt_color = YELLOW;
    else if (norm > 0.01f)
      doubt_color = RED;
    else
      doubt_color = DIM;
  } else {
    doubt_color = GREEN;
  }
  printf("%s%6.3f %s(challenge)%s\n", doubt_color, doubt_val, BOLD, RESET);
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

  struct Tensor inputs, loss_p, loss_c;
  float         loss_v;
  tensor_init(&inputs, 1, NUM_INPUTS);
  tensor_init(&loss_p, 1, NUM_POL_OUT);
  tensor_init(&loss_c, 1, NUM_FACES);

  get_canonical(g, &inputs);
  loss_v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (size_t i = 0; i < tensor_size(&loss_p); i++)
    loss_p.buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (size_t i = 0; i < tensor_size(&loss_c); i++)
    loss_c.buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  for (int i = 0; i < 100; i++) {
    network_forward(n, &inputs, g);
    network_backward(n, &inputs, &loss_p, loss_v, &loss_c);
  }

  clock_t start = clock();
  for (int i = 0; i < 1000; i++) {
    network_forward(n, &inputs, g);
    network_backward(n, &inputs, &loss_p, loss_v, &loss_c);
  }
  clock_t end = clock();

  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Time for 1000 Forward+Backward passes: %f seconds\n", time_spent);
  printf("Average time per pass: %f ms\n", (time_spent / 1000.0) * 1000.0);
  printf("Estimated Evals per Second: %.2f\n\n", 1000.0 / time_spent);

  tensor_free(&loss_c);
  tensor_free(&loss_p);
  tensor_free(&inputs);
  network_free(n);
  free(g);
}
