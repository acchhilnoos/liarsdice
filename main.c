#include "config.h"
#include "game.h"
#include "network.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void random_playout(void) {
  struct Game *g = game_new();

  size_t alive;
  do {
    while (1) {
      size_t c, f;

      if (g->d1bid.c == g->total_left) {
        challenge(g);
        break;
      }

      do {
        c = rand() % (g->total_left - g->d1bid.c) + g->d1bid.c + 1;
        f = rand() % NUM_FACES + 1;
      } while (!legal(g, c, f));

      bid(g, c, f);
    }

    alive = 0;
    for (size_t i = 0; i < NUM_PLAYERS; i++)
      if (g->dice_left[i] != 0)
        alive++;
  } while (alive > 1);

  free(g);
}

void network_playout(struct Network *n) {
  struct Game  *g = game_new();
  struct Tensor inputs;
  tensor_init(&inputs, 1, NUM_INPUTS);

  size_t alive;
  do {
    game_print(g);

    while (1) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / (RAND_MAX + 1.0f);
      float s = 0.0f;
      for (size_t i = 0; i < NUM_OUTPUTS; i++) {
        s += n->as[POL_HEAD].buf[i];
        if (s > r) {
          a = i;
          break;
        }
      }
      network_peek(n);
      if (a == CHALLENGE_IDX) {
        challenge(g);
        break;
      } else {
        size_t p = g->p;
        bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);
        printf("(%zu,%2zu,%zu)\n", p + 1, (a / NUM_FACES) + 1,
               (a % NUM_FACES) + 1);
      }
    }
    printf("\n");

    alive = 0;
    for (size_t i = 0; i < NUM_PLAYERS; i++)
      if (g->dice_left[i] != 0)
        alive++;
  } while (alive > 1);

  tensor_free(&inputs);
  free(g);
}

int main(int argc, char *argv[]) {
  srand(time(NULL));
  size_t max_iters = 1000;
  size_t max_steps = 200;
  size_t max_epchs = 10;
  float  alpha     = 0.01f;
  float  beta      = 0.9f;
  float  epsilon   = 0.2f;
  float  gamma     = 0.95f;
  float  lambda    = 0.99f;
  float  c1        = 1.0f;
  float  c2        = 0.1f;
  gamma *= gamma;
  gamma *= gamma;
  lambda *= lambda;
  lambda *= lambda;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "bench") == 0) {
      network_benchmark();
      return 0;
    }
    if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random") == 0) {
      random_playout();
      return 0;
    }
    if (strcmp(argv[i], "-e") == 0)
      max_epchs = atoi(argv[++i]);
    if (strcmp(argv[i], "-i") == 0)
      max_iters = atoi(argv[++i]);
    if (strcmp(argv[i], "-s") == 0)
      max_steps = atoi(argv[++i]);
  }

  /* --- --- */

  struct Game    *g = game_new();
  struct Network *n = network_new();
  struct Tensor   inputs;
  struct Tensor   loss_p;
  tensor_init(&inputs, 1, NUM_INPUTS);
  tensor_init(&loss_p, 1, NUM_OUTPUTS);

  struct Step {
    float  state[NUM_INPUTS];
    size_t a;
    float  r;
    float  v;
    float  log_pi;
    size_t c, f, total_left;
    bool   terminal;
  };
  struct Step *step_buf = calloc(max_steps, sizeof(*step_buf));
  size_t       step_n   = 0;

  float  *as   = calloc(max_steps, sizeof(*as));
  float  *rs   = calloc(max_steps, sizeof(*rs));
  size_t *idxs = calloc(max_steps, sizeof(*idxs));

  for (size_t iter = 0; iter < max_iters; iter++) {
    step_n = 0;

    while (step_n < max_steps) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / (RAND_MAX + 1.0f);
      float s = 0.0f;
      for (size_t i = 0; i < NUM_OUTPUTS; i++) {
        s += n->as[POL_HEAD].buf[i];
        if (s > r) {
          a = i;
          break;
        }
      }
      if (g->p == 0) {
        struct Step *step = &step_buf[step_n];
        memcpy(&step->state, inputs.buf, tensor_size(&inputs) * sizeof(float));
        step->a          = a;
        step->r          = 0.0f;
        step->v          = n->as[VAL_HEAD].buf[0];
        step->log_pi     = logf(n->as[POL_HEAD].buf[a]);
        step->c          = g->d1bid.c;
        step->f          = g->d1bid.f;
        step->total_left = g->total_left;
        step->terminal   = false;

        if (a == CHALLENGE_IDX)
          if (challenge(g))
            step->r = 0.5f;
          else
            step->r = -0.5f;
        else
          bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);

        step_n++;
      } else {
        if (a == CHALLENGE_IDX)
          challenge(g);
        else
          bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);
      }

      size_t alive = 0;
      for (size_t i = 0; i < NUM_PLAYERS; i++)
        if (g->dice_left[i] != 0)
          alive++;
      if (alive == 1) {
        step_buf[step_n - 1].terminal = true;
        if (g->dice_left[0] != 0)
          step_buf[step_n - 1].r = 1.0f;
        else
          step_buf[step_n - 1].r = -1.0f;
        free(g);
        g = game_new();
      }
    }

    /* --- --- */

    float d    = 0.0f;
    float a    = 0.0f;
    float r    = 0.0f;
    float mean = 0.0f;
    float std  = 0.0f;

    for (size_t i = 1; i <= max_steps; i++) {
      struct Step *step = &step_buf[max_steps - i];
      if (i == 1 || step->terminal) {
        d = step->r - step->v;
        a = d;
      } else {
        d = step->r + gamma * step_buf[max_steps - i + 1].v - step->v;
        a = d + gamma * lambda * a;
      }
      r = a + step->v;

      as[max_steps - i] = a;
      rs[max_steps - i] = r;
      mean += a;

      idxs[i - 1] = i - 1;
    }
    mean /= max_steps;

    for (size_t i = 0; i < max_steps; i++)
      std += powf(as[i] - mean, 2);
    std = sqrtf(std / max_steps + 1e-8f);
    for (size_t i = 0; i < max_steps; i++)
      as[i] = (as[i] - mean) / std;

    /* --- --- */

    struct Game g_temp = {0};

    for (size_t epch = 0; epch < max_epchs; epch++) {
      for (size_t i = 1; i < max_steps; i++) {
        size_t r = rand() % (i + 1);
        size_t t = idxs[i];
        idxs[i]  = idxs[r];
        idxs[r]  = t;
      }

      for (size_t batch = 0; batch < max_steps; batch += MAX_BATCH_SIZE) {
        network_zero_grad(n);

        for (size_t i = 0; i < MAX_BATCH_SIZE && batch + i < max_steps; i++) {
          size_t       idx  = idxs[batch + i];
          struct Step *step = &step_buf[idx];

          g_temp.total_left = step->total_left;
          g_temp.d1bid.c    = step->c;
          g_temp.d1bid.f    = step->f;

          memcpy(inputs.buf, step->state, sizeof(step->state));
          network_forward(n, &inputs, &g_temp);

          float pi_old  = expf(step->log_pi);
          float pi_new  = n->as[POL_HEAD].buf[step->a];
          float r_t     = pi_new / pi_old;
          float dl_clip = as[idx] > 0
                              ? (r_t > 1 + epsilon ? 0 : -as[idx] / pi_old)
                              : (r_t < 1 - epsilon ? 0 : -as[idx] / pi_old);

          float dl_vf = (n->as[VAL_HEAD].buf[0] - rs[idx]);

          tensor_zero(&loss_p);

          for (size_t j = 0; j < NUM_OUTPUTS; j++)
            loss_p.buf[j] =
                c2 * (logf(fmaxf(n->as[POL_HEAD].buf[j], 1e-10f)) + 1);
          loss_p.buf[step->a] += dl_clip;

          network_backward(n, &inputs, &loss_p, c1 * dl_vf);
        }
        network_sgd(n, alpha / MAX_BATCH_SIZE, beta);
      }
    }

    if (iter % 50 == 0)
      network_playout(n);
  }

  free(idxs);
  free(rs);
  free(as);
  free(step_buf);
  tensor_free(&loss_p);
  tensor_free(&inputs);
  network_free(n);
  free(g);
  return 0;
}
