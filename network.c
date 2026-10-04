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

static inline int layer_init(struct Network *n, size_t i, size_t y, size_t x)
{
  if (tensor_init(&n->ks[i], y, x) != 0) goto fail_ks;
  if (tensor_init(&n->m_ks[i], y, x) != 0) goto fail_m_ks;
  if (tensor_init(&n->bs[i], 1, x) != 0) goto fail_bs;
  if (tensor_init(&n->m_bs[i], 1, x) != 0) goto fail_m_bs;
  if (tensor_init(&n->as[i], 1, x) != 0) goto fail_as;
  return 0;

fail_as:
  tensor_free(&n->m_bs[i]);
fail_m_bs:
  tensor_free(&n->bs[i]);
fail_bs:
  tensor_free(&n->m_ks[i]);
fail_m_ks:
  tensor_free(&n->ks[i]);
fail_ks:
  for (size_t j = 0; j < i; j++) {
    tensor_free(&n->ks[j]);
    tensor_free(&n->m_ks[j]);
    tensor_free(&n->bs[j]);
    tensor_free(&n->m_bs[j]);
    tensor_free(&n->as[j]);
  }
  return -1;
}

struct Network *network_new(void)
{
  struct Network *n = malloc(sizeof(*n));
  if (!n) goto fail_n;

  for (size_t i = 0; i < NUM_LAYERS; i++)
    if (layer_init(n, i,
                   /**
                    * i            y        x
                    * 0            n_in     s_hidden
                    * [1, i_pol-1] s_hidden s_hidden
                    * i_pol        s_hidden s_pol
                    * i_val        s_hidden s_val
                    */
                   i == 0 ? NUM_INPUTS : SIZE_HIDDEN,
                   i == POL_HEAD_IDX   ? SIZE_POL
                   : i == VAL_HEAD_IDX ? 1
                                       : SIZE_HIDDEN) != 0)
      goto fail_l;

  for (size_t i = 0; i < NUM_LAYERS; i++) {
    struct Tensor *ks = &n->ks[i];

    // uniform Kaiming He
    float limit = sqrtf(6.0f / ks->y);
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++) {
      // U(0,1) -> U(0,2) -> U(-1,1) -> U(-l,l)
      ks->buf[j] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * limit;
    }
  }

  return n;

fail_l:
  free(n);
fail_n:
  return NULL;
}

void network_free(struct Network *n)
{
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    tensor_free(&n->ks[i]);
    tensor_free(&n->bs[i]);
    tensor_free(&n->as[i]);
    tensor_free(&n->m_ks[i]);
    tensor_free(&n->m_bs[i]);
  }
  free(n);
}

void network_zero_grad(struct Network *n)
{
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    tensor_zero_grad(&n->ks[i]);
    tensor_zero_grad(&n->bs[i]);
    tensor_zero_grad(&n->as[i]);
  }
}

void network_forward(struct Network *n, const struct Tensor *inputs,
                     const struct Game *g)
{
  tensor_fc(inputs, &n->ks[0], &n->bs[0], &n->as[0]);
  tensor_relu(&n->as[0]);

  for (size_t i = 0; i < POL_HEAD_IDX - 1; i++) {
    tensor_fc(&n->as[i], &n->ks[i + 1], &n->bs[i + 1], &n->as[i + 1]);
    tensor_relu(&n->as[i + 1]);
  }

  tensor_fc(&n->as[POL_HEAD_IDX - 1], &n->ks[POL_HEAD_IDX],
            &n->bs[POL_HEAD_IDX], &n->as[POL_HEAD_IDX]);
  for (size_t i = 0; i < NUM_TOTAL_DICE; i++)
    for (size_t j = 0; j < NUM_FACES; j++)
      if (!legal(g, i + 1, j + 1))
        n->as[POL_HEAD_IDX].buf[i * NUM_FACES + j] = -FLT_MAX;
  if (g->last.c == 0) n->as[POL_HEAD_IDX].buf[CHALLENGE_IDX] = -FLT_MAX;
  tensor_softmax(&n->as[POL_HEAD_IDX]);

  tensor_fc(&n->as[VAL_HEAD_IDX - 2], &n->ks[VAL_HEAD_IDX],
            &n->bs[VAL_HEAD_IDX], &n->as[VAL_HEAD_IDX]);
  // tensor_tanh(&n->as[4]);
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      const struct Tensor *loss_p, float loss_v)
{
  for (size_t i = 0; i < NUM_LAYERS; i++) tensor_zero_grad(&n->as[i]);

  n->as[VAL_HEAD_IDX].grad[0] = loss_v;
  // tensor_tanh_grad(&n->as[4]);
  tensor_fc_grad(&n->as[VAL_HEAD_IDX - 2], &n->ks[VAL_HEAD_IDX],
                 &n->bs[VAL_HEAD_IDX], &n->as[VAL_HEAD_IDX]);

  memcpy(n->as[POL_HEAD_IDX].grad, loss_p->buf,
         tensor_size(loss_p) * sizeof(*loss_p->buf));
  tensor_softmax_grad(&n->as[POL_HEAD_IDX]);
  tensor_fc_grad(&n->as[POL_HEAD_IDX - 1], &n->ks[POL_HEAD_IDX],
                 &n->bs[POL_HEAD_IDX], &n->as[POL_HEAD_IDX]);

  for (size_t i = POL_HEAD_IDX - 1; i > 0; i--) {
    tensor_relu_grad(&n->as[i]);
    tensor_fc_grad(&n->as[i - 1], &n->ks[i], &n->bs[i], &n->as[i]);
  }

  tensor_relu_grad(&n->as[0]);
  tensor_fc_grad(inputs, &n->ks[0], &n->bs[0], &n->as[0]);
}

void network_sgd(struct Network *n, float alpha, float beta)
{
  float norm = 0.0f;
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++)
      norm += n->ks[i].grad[j] * n->ks[i].grad[j];
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++)
      norm += n->bs[i].grad[j] * n->bs[i].grad[j];
  }
  norm        = sqrtf(norm);
  float scale = (norm > 1.0f) ? (1.0f / norm) : 1.0f;

  for (size_t i = 0; i < NUM_LAYERS; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++) {
      float grad         = n->ks[i].grad[j] * scale;
      n->m_ks[i].buf[j]  = beta * n->m_ks[i].buf[j] + grad;
      n->ks[i].buf[j]   -= alpha * n->m_ks[i].buf[j];
    }
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++) {
      float grad         = n->bs[i].grad[j] * scale;
      n->m_bs[i].buf[j]  = beta * n->m_bs[i].buf[j] + grad;
      n->bs[i].buf[j]   -= alpha * n->m_bs[i].buf[j];
    }
  }
}

void network_peek(const struct Network *n)
{
  const char *RED    = "\033[31m";
  const char *YELLOW = "\033[33m";
  const char *GREEN  = "\033[32m";
  const char *DIM    = "\033[2m";
  const char *BOLD   = "\033[1m";
  const char *RESET  = "\033[0m";

  float min_val = n->as[POL_HEAD_IDX].buf[0];
  float max_val = n->as[POL_HEAD_IDX].buf[0];
  for (size_t i = 0; i < SIZE_POL; i++) {
    if (n->as[POL_HEAD_IDX].buf[i] < min_val)
      min_val = n->as[POL_HEAD_IDX].buf[i];
    if (n->as[POL_HEAD_IDX].buf[i] > max_val)
      max_val = n->as[POL_HEAD_IDX].buf[i];
  }

  float range = max_val - min_val;

  for (size_t i = 0; i < NUM_FACES; i++) {
    for (size_t j = 0; j < NUM_TOTAL_DICE; j++) {
      float       val = n->as[POL_HEAD_IDX].buf[j * NUM_FACES + i];
      const char *color;
      if (range > 1e-8f) {
        float norm = (val - min_val) / range;
        if (norm > 0.66f) color = GREEN;
        else if (norm > 0.33f) color = YELLOW;
        else if (norm >= 0.001f) color = RED;
        else color = DIM;
      } else {
        color = GREEN;
      }
      printf("%s%6.3f%s", color, val, RESET);
    }
    printf("\n");
  }

  float       doubt_val = n->as[POL_HEAD_IDX].buf[CHALLENGE_IDX];
  const char *doubt_color;
  if (range > 1e-8f) {
    float norm = (doubt_val - min_val) / range;
    if (norm > 0.66f) doubt_color = GREEN;
    else if (norm > 0.33f) doubt_color = YELLOW;
    else if (norm >= 0.001f) doubt_color = RED;
    else doubt_color = DIM;
  } else {
    doubt_color = GREEN;
  }
  printf("%s%6.3f %s%s/ ", doubt_color, doubt_val, BOLD, RESET);

  printf("est.v: %6.3f / ", n->as[VAL_HEAD_IDX].buf[0]);
  printf("\n");
}

void network_save(struct Network *n, const char *path)
{
  FILE *f = fopen(path, "wb");
  if (!f) return;
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    fwrite(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fwrite(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
    fwrite(n->m_ks[i].buf, sizeof(float), tensor_size(&n->m_ks[i]), f);
    fwrite(n->m_bs[i].buf, sizeof(float), tensor_size(&n->m_bs[i]), f);
  }
  fclose(f);
}

void network_load(struct Network *n, const char *path)
{
  FILE *f = fopen(path, "rb");
  if (!f) return;
  for (size_t i = 0; i < NUM_LAYERS; i++) {
    fread(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fread(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
    fread(n->m_ks[i].buf, sizeof(float), tensor_size(&n->m_ks[i]), f);
    fread(n->m_bs[i].buf, sizeof(float), tensor_size(&n->m_bs[i]), f);
  }
  fclose(f);
}

void network_benchmark(void)
{
  printf("Benchmarking (1000 iterations)...\n");

  struct Network *n = network_new();
  struct Game    *g = game_new();

  struct Tensor inputs, loss_p;
  float         loss_v;
  tensor_init(&inputs, 1, NUM_INPUTS);
  tensor_init(&loss_p, 1, SIZE_POL);

  get_canonical(g, &inputs);
  loss_v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (size_t i = 0; i < tensor_size(&loss_p); i++)
    loss_p.buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

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
