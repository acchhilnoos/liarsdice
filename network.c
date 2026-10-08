#include "network.h"
#include "config.h"
#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct Network {
  struct Tensor ks[NUM_TOTAL_LAYERS];
  struct Tensor m_ks[NUM_TOTAL_LAYERS];
  struct Tensor bs[NUM_TOTAL_LAYERS];
  struct Tensor m_bs[NUM_TOTAL_LAYERS];
};

static inline int layer_init(struct Network *n, size_t i, size_t y, size_t x)
{
  if (tensor_init(&n->ks[i], y, x) != 0) goto fail_ks;
  if (tensor_init(&n->m_ks[i], y, x) != 0) goto fail_m_ks;
  if (tensor_init(&n->bs[i], 1, x) != 0) goto fail_bs;
  if (tensor_init(&n->m_bs[i], 1, x) != 0) goto fail_m_bs;
  return 0;

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
  }
  return 1;
}

struct Network *network_new(void)
{
  struct Network *n = malloc(sizeof(*n));
  if (!n) goto fail_n;

  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++)
    if (layer_init(n, i,
                   /**
                    * i                 y        x
                    * 0                 n_in     s_hidden
                    * [1, i_gru-1]      s_hidden s_hidden
                    * i_gru             s_hidden s_gru
                    * [i_gru+1,i_pol-1] s_gru    s_gru
                    * i_pol             s_gru    s_pol
                    * i_val             s_gru    1
                    */
                   i == 0            ? SIZE_INPUT
                   : i <= GRU_WR_IDX ? SIZE_HIDDEN
                                     : SIZE_GRU,
                   i < GRU_WR_IDX ? SIZE_HIDDEN
                   : i < POL_IDX  ? SIZE_GRU
                   : i < VAL_IDX  ? SIZE_POL
                                  : 1) != 0)
      goto fail_l;

  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
    struct Tensor *k     = &n->ks[i];
    float          limit = sqrtf(6.0f / k->y);

    for (size_t j = 0; j < tensor_size(k); j++)
      k->buf[j] = (2.0f * (float)rand() / RAND_MAX - 1.0f) * limit;
  }

  return n;

fail_l:
  free(n);
fail_n:
  return NULL;
}

void network_free(struct Network *n)
{
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
    tensor_free(&n->ks[i]);
    tensor_free(&n->bs[i]);
    tensor_free(&n->m_ks[i]);
    tensor_free(&n->m_bs[i]);
  }
  free(n);
}

void network_zero_grad(struct Network *n)
{
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
    tensor_zero_grad(&n->ks[i]);
    tensor_zero_grad(&n->bs[i]);
  }
}

void network_forward(const struct Network *n, const struct Tensor *in,
                     const struct Tensor *h_prev, const struct Game *g,
                     struct Tensor activs[NUM_TOTAL_LAYERS])
{
  tensor_fc(in, &n->ks[0], &n->bs[0], &activs[0]);
  tensor_relu(&activs[0]);

  for (size_t i = 1; i < NUM_MLP_LAYERS; i++) {
    tensor_fc(&activs[i - 1], &n->ks[i], &n->bs[i], &activs[i]);
    tensor_relu(&activs[i]);
  }

  tensor_gru(&activs[NUM_MLP_LAYERS - 1], h_prev, &n->ks[GRU_WR_IDX],
             &n->ks[GRU_UR_IDX], &n->bs[GRU_BR_IDX], &activs[GRU_R_IDX],
             &n->ks[GRU_WZ_IDX], &n->ks[GRU_UZ_IDX], &n->bs[GRU_BZ_IDX],
             &activs[GRU_Z_IDX], &n->ks[GRU_WH_IDX], &n->ks[GRU_UH_IDX],
             &n->bs[GRU_BH_IDX], &activs[GRU_HT_IDX], &activs[GRU_H_IDX]);

  tensor_fc(&activs[GRU_H_IDX], &n->ks[POL_IDX], &n->bs[POL_IDX],
            &activs[POL_IDX]);
  for (size_t i = 0; i < NUM_TOTAL_DICE; i++)
    for (size_t j = 0; j < NUM_FACES; j++)
      if (!legal(g, i + 1, j + 1))
        activs[POL_IDX].buf[i * NUM_FACES + j] = -FLT_MAX;
  if (g->last.c == 0) activs[POL_IDX].buf[CHALLENGE_IDX] = -FLT_MAX;
  tensor_softmax(&activs[POL_IDX]);

  tensor_fc(&activs[GRU_H_IDX], &n->ks[VAL_IDX], &n->bs[VAL_IDX],
            &activs[VAL_IDX]);
}

void network_backward(struct Network *n, struct Tensor *inputs,
                      struct Tensor       *h_prev,
                      struct Tensor        activs[NUM_TOTAL_LAYERS],
                      const struct Tensor *loss_p, float loss_v)
{
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) tensor_zero_grad(&activs[i]);

  activs[VAL_IDX].grad[0] = loss_v;
  memcpy(activs[POL_IDX].grad, loss_p->buf,
         tensor_size(loss_p) * sizeof(*loss_p->buf));

  tensor_fc_grad(&activs[GRU_H_IDX], &n->ks[VAL_IDX], &n->bs[VAL_IDX],
                 &activs[VAL_IDX]);

  tensor_softmax_grad(&activs[POL_IDX]);
  tensor_fc_grad(&activs[GRU_H_IDX], &n->ks[POL_IDX], &n->bs[POL_IDX],
                 &activs[POL_IDX]);

  tensor_gru_grad(&activs[NUM_MLP_LAYERS - 1], h_prev, &n->ks[GRU_WR_IDX],
                  &n->ks[GRU_UR_IDX], &n->bs[GRU_BR_IDX], &activs[GRU_R_IDX],
                  &n->ks[GRU_WZ_IDX], &n->ks[GRU_UZ_IDX], &n->bs[GRU_BZ_IDX],
                  &activs[GRU_Z_IDX], &n->ks[GRU_WH_IDX], &n->ks[GRU_UH_IDX],
                  &n->bs[GRU_BH_IDX], &activs[GRU_HT_IDX], &activs[GRU_H_IDX]);

  for (size_t i = NUM_MLP_LAYERS - 1; i > 0; i--) {
    tensor_relu_grad(&activs[i]);
    tensor_fc_grad(&activs[i - 1], &n->ks[i], &n->bs[i], &activs[i]);
  }

  tensor_relu_grad(&activs[0]);
  tensor_fc_grad(inputs, &n->ks[0], &n->bs[0], &activs[0]);
}

void network_sgd(struct Network *n, float alpha, float beta)
{
  float norm = 0.0f;
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
    for (size_t j = 0; j < tensor_size(&n->ks[i]); j++)
      norm += n->ks[i].grad[j] * n->ks[i].grad[j];
    for (size_t j = 0; j < tensor_size(&n->bs[i]); j++)
      norm += n->bs[i].grad[j] * n->bs[i].grad[j];
  }
  norm        = sqrtf(norm);
  float scale = (norm > 1.0f) ? (1.0f / norm) : 1.0f;

  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
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

void network_peek(const struct Network *n, const struct Game *g,
                  const struct Tensor *hp)
{
  const char *RED    = "\033[31m";
  const char *YELLOW = "\033[33m";
  const char *GREEN  = "\033[32m";
  const char *DIM    = "\033[2m";
  const char *BOLD   = "\033[1m";
  const char *RESET  = "\033[0m";

  struct Tensor inputs;
  struct Tensor activs[NUM_TOTAL_LAYERS];
  tensor_init(&inputs, 1, SIZE_INPUT);
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++)
    tensor_init(activs + i, 1,
                i < GRU_WR_IDX ? SIZE_HIDDEN
                : i < POL_IDX  ? SIZE_GRU
                : i < VAL_IDX  ? SIZE_POL
                               : 1);

  get_canonical(g, &inputs);
  float val;
  network_forward(n, &inputs, hp, g, activs);

  float *pol     = activs[0].buf;
  float  min_val = FLT_MAX, max_val = -FLT_MAX;
  for (size_t i = 0; i < SIZE_POL; i++) {
    if (pol[i] < min_val) min_val = pol[i];
    if (pol[i] > max_val) max_val = pol[i];
  }

  printf("%sNetwork Peek:%s  est.v: %6.3f\n", BOLD, RESET, val);
  float range = max_val - min_val;
  for (size_t c = 1; c <= NUM_TOTAL_DICE; c++) {
    printf("%s%2zu%s ", BOLD, c, RESET);
    for (size_t f = 1; f <= NUM_FACES; f++) {
      size_t      idx = (c - 1) * NUM_FACES + (f - 1);
      float       p   = pol[idx];
      const char *color;
      if (range > 1e-8f) {
        float norm = (p - min_val) / range;
        if (norm > 0.66f) color = GREEN;
        else if (norm > 0.33f) color = YELLOW;
        else if (norm >= 0.001f) color = RED;
        else color = DIM;
      } else {
        color = GREEN;
      }
      printf("%s%6.3f%s", color, p, RESET);
    }
    printf("\n");
  }
  printf("%sCH%s ", BOLD, RESET);
  float       cp = pol[CHALLENGE_IDX];
  const char *ccolor =
      (range > 1e-8f && (cp - min_val) / range > 0.33f) ? GREEN : RESET;
  printf("%s%6.3f%s\n", ccolor, cp, RESET);

  tensor_free(&inputs);
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) tensor_free(activs + i);
}

int network_save(const struct Network *n, const char *path)
{
  FILE *f = fopen(path, "wb");
  if (!f) return 1;
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
    fwrite(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fwrite(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
    fwrite(n->m_ks[i].buf, sizeof(float), tensor_size(&n->m_ks[i]), f);
    fwrite(n->m_bs[i].buf, sizeof(float), tensor_size(&n->m_bs[i]), f);
  }
  fclose(f);
  return 0;
}

int network_load(struct Network *n, const char *path)
{
  FILE *f = fopen(path, "rb");
  if (!f) return 1;
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) {
    fread(n->ks[i].buf, sizeof(float), tensor_size(&n->ks[i]), f);
    fread(n->bs[i].buf, sizeof(float), tensor_size(&n->bs[i]), f);
    fread(n->m_ks[i].buf, sizeof(float), tensor_size(&n->m_ks[i]), f);
    fread(n->m_bs[i].buf, sizeof(float), tensor_size(&n->m_bs[i]), f);
  }
  fclose(f);
  return 0;
}

void network_benchmark(void)
{
  printf("Benchmarking (1000 iterations)...\n");

  struct Network *n = network_new();
  struct Game    *g = game_new();

  struct Tensor inputs, loss_p, hp;
  float         loss_v;
  tensor_init(&inputs, 1, SIZE_INPUT);
  tensor_init(&loss_p, 1, SIZE_POL);
  tensor_init(&hp, 1, SIZE_GRU);

  for (size_t i = 0; i < SIZE_INPUT; i++) inputs.buf[i] = (i + 1) * 0.01f;
  for (size_t i = 0; i < SIZE_GRU; i++) hp.buf[i] = (i + 1) * 0.005f;

  get_canonical(g, &inputs);
  loss_v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (size_t i = 0; i < tensor_size(&loss_p); i++)
    loss_p.buf[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  struct Tensor activs[NUM_TOTAL_LAYERS];
  for (size_t i = 0; i < NUM_MLP_LAYERS; i++)
    tensor_init(&activs[i], 1, SIZE_HIDDEN);
  for (size_t i = GRU_WR_IDX; i <= GRU_H_IDX; i++)
    tensor_init(&activs[i], 1, SIZE_GRU);
  tensor_init(&activs[VAL_IDX], 1, 1);
  tensor_init(&activs[POL_IDX], 1, SIZE_POL);

  for (int i = 0; i < 100; i++) {
    network_forward(n, &inputs, &hp, g, activs);
    network_backward(n, &inputs, &hp, activs, &loss_p, loss_v);
  }

  clock_t start = clock();
  for (int i = 0; i < 1000; i++) {
    network_forward(n, &inputs, &hp, g, activs);
    network_backward(n, &inputs, &hp, activs, &loss_p, loss_v);
  }
  clock_t end = clock();

  double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
  printf("Time for 1000 Forward+Backward passes: %f seconds\n", time_spent);
  printf("Average time per pass: %f ms\n", (time_spent / 1000.0) * 1000.0);
  printf("Estimated Evals per Second: %.2f\n\n", 1000.0 / time_spent);

  for (size_t i = 0; i < NUM_MLP_LAYERS; i++) tensor_free(&activs[i]);
  for (size_t i = GRU_WR_IDX; i <= GRU_H_IDX; i++) tensor_free(&activs[i]);
  tensor_free(&activs[VAL_IDX]);
  tensor_free(&activs[POL_IDX]);
  tensor_free(&loss_p);
  tensor_free(&inputs);
  tensor_free(&hp);
  network_free(n);
  free(g);
}
