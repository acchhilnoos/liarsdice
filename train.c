#include "train.h"
#include "network.h"
#include "play.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

struct Step {
  float  in[NUM_INPUTS];
  size_t d[NUM_FACES];
  size_t a;
  float  r;
  float  v;
  float  pi;
  bool   terminal;
};

int train(struct Network *n, const char *weights_fn, size_t max_iters,
          size_t max_steps, size_t max_epchs, float alpha, float beta,
          float epsilon, float gamma, float lambda, float c1, float c2,
          bool verbose)
{
  struct Game *g = game_new();
  if (!g) goto fail_g;
  struct Tensor inputs;
  struct Tensor loss_p;
  struct Tensor loss_c;
  if (tensor_init(&inputs, 1, NUM_INPUTS) != 0) goto fail_is;
  if (tensor_init(&loss_p, 1, SIZE_POL) != 0) goto fail_lp;
  if (tensor_init(&loss_c, 1, NUM_FACES) != 0) goto fail_lc;

  struct Step *step_buf = calloc(max_steps, sizeof(*step_buf));
  if (!step_buf) goto fail_s;
  size_t step_n = 0;

  float *as = calloc(max_steps, sizeof(*as));
  if (!as) goto fail_as;
  float *vs = calloc(max_steps, sizeof(*vs));
  if (!as) goto fail_vs;
  size_t *idxs = calloc(max_steps, sizeof(*idxs));
  if (!as) goto fail_idxs;

  for (size_t iter = 0; iter < max_iters; iter++) {

    /* --- playouts --- */

    step_n = 0;

    while (step_n < max_steps) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / (RAND_MAX + 1.0f), s = 0.0f;
      float sum = 0.0f;
      for (size_t i = 0; i < SIZE_POL; i++) sum += n->as[POL_HEAD_IDX].buf[i];
      r *= sum;
      for (size_t i = 0; i < SIZE_POL; i++) {
        s += n->as[POL_HEAD_IDX].buf[i];
        if (s > r) {
          a = i;
          break;
        }
      }

      if (g->p == 0) {
        struct Step *step = &step_buf[step_n];
        memcpy(&step->in, inputs.buf, tensor_size(&inputs) * sizeof(float));
        memcpy(&step->d, g->game_counts, sizeof(g->game_counts));
        step->a        = a;
        step->r        = 0.0f;
        step->v        = n->as[VAL_HEAD_IDX].buf[0];
        step->pi       = n->as[POL_HEAD_IDX].buf[a];
        step->terminal = false;

        if (a == CHALLENGE_IDX)
          // player challenge good (encourage plausible challenges)
          if (challenge(g)) step->r = 0.5f;
          // player challenge bad (discourage implausible challenges)
          else step->r = -0.5f;
        else bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);

        step_n++;
      } else {
        if (a == CHALLENGE_IDX)
          if (g->last.p == 0)
            // player challenged good (discourage implausible high bids)
            if (challenge(g)) step_buf[step_n - 1].r = -0.5;
            // player challenged bad (encourage plausible high bids)
            else step_buf[step_n - 1].r = 0.5;
          else challenge(g);
        else bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);
      }

      size_t alive = 0;
      for (size_t i = 0; i < NUM_PLAYERS; i++)
        if (g->player_rem[i] != 0) alive++;
      if (alive == 1) {
        step_buf[step_n - 1].terminal = true;
        // player win
        if (g->player_rem[0] != 0) step_buf[step_n - 1].r += 1.0f;
        // player loss
        else step_buf[step_n - 1].r += -1.0f;
        game_restart(g);
      }
    }

    /* --- value assignment --- */

    float d    = 0.0f;
    float a    = 0.0f;
    float v    = 0.0f;
    float mean = 0.0f;
    float std  = 0.0f;

    for (size_t i = 0; i < max_steps; i++) {
      size_t       idx  = max_steps - i - 1;
      struct Step *step = &step_buf[idx];

      /**
       * d_t = r_t + gv(s_{t+1})- v(s_t)
       * A_t = d_t + gl(d_{t+1}) + ggll(d_{t+2}) + ...
       *     = d_t + glA_{t+1}
       */
      if (i == 0) {
        d = step->r - step->v;
        a = d;
      } else {
        d = step->r + gamma * step_buf[idx + 1].v - step->v;
        a = d + gamma * lambda * a;
      }
      // v_targ(s_t) = v(s_t) + A_t
      v = step->v + a;

      as[idx] = a;
      vs[idx] = v;
      idxs[i] = i;

      mean += a;
    }
    mean /= max_steps;

    for (size_t i = 0; i < max_steps; i++) std += powf(as[i] - mean, 2);
    std = sqrtf(std / max_steps);
    for (size_t i = 0; i < max_steps; i++) as[i] = (as[i] - mean) / std;

    /* --- backprop  --- */

    struct Game g_temp = {0};

    for (size_t epch = 0; epch < max_epchs; epch++) {
      for (size_t i = 1; i < max_steps; i++) {
        size_t x = rand() % (i + 1);
        size_t y = idxs[i];
        idxs[i]  = idxs[x];
        idxs[x]  = y;
      }

      for (size_t batch = 0; batch < max_steps; batch += MAX_BATCH_SIZE) {
        network_zero_grad(n);

        for (size_t i = 0; i < MAX_BATCH_SIZE && batch + i < max_steps; i++) {
          size_t       idx  = idxs[batch + i];
          struct Step *step = &step_buf[idx];

          g_temp.game_rem =
              (size_t)(step->in[NUM_FACES + NUM_PLAYERS + 2] * NUM_TOTAL_DICE +
                       0.5f);
          g_temp.last.c =
              (size_t)(step->in[NUM_FACES] * g_temp.game_rem + 0.5f);
          g_temp.last.f = (size_t)(step->in[NUM_FACES + 1] * NUM_FACES + 0.5f);

          memcpy(inputs.buf, step->in, sizeof(step->in));
          network_forward(n, &inputs, &g_temp);

          float pi_old = step->pi;
          float pi_new = fmaxf(n->as[POL_HEAD_IDX].buf[step->a], 1e-10f);
          float r_t    = pi_new / pi_old;
          /**
           * L_clip = min(rA, clip(r, 1-e, 1+e)A)
           *           / min(rA, (1+e)A)   A > 0
           *        = <                0   A = 0
           *           \ min((1-e)A, rA)   A < 0
           */
          float dl_clip = as[idx] > 0
                              ? (r_t > 1 + epsilon ? 0 : -as[idx] / pi_old)
                              : (r_t < 1 - epsilon ? 0 : -as[idx] / pi_old);

          // L_vf = (v_new(s) - v_targ(s))^2
          float dl_vf = (n->as[VAL_HEAD_IDX].buf[0] - vs[idx]);

          tensor_zero(&loss_p);
          tensor_zero(&loss_c);

          // S = -sum plogp
          for (size_t j = 0; j < SIZE_POL; j++)
            loss_p.buf[j] =
                c2 * (logf(fmaxf(n->as[POL_HEAD_IDX].buf[j], 1e-10f)) + 1);
          loss_p.buf[step->a] += dl_clip;

          network_backward(n, &inputs, &loss_p, c1 * dl_vf);
        }
        network_sgd(n, alpha / MAX_BATCH_SIZE, beta);
      }
    }

    if (iter % 100 == 0) {
      playout(n, false, verbose);
      network_save(n, weights_fn);
      printf("iteration %zu complete\n", iter);
    }
  }

  free(vs);
  free(as);
  free(step_buf);
  tensor_free(&loss_c);
  tensor_free(&loss_p);
  tensor_free(&inputs);
  free(g);
  return 0;

fail_idxs:
  free(vs);
fail_vs:
  free(as);
fail_as:
  free(step_buf);
fail_s:
  tensor_free(&loss_c);
fail_lc:
  tensor_free(&loss_p);
fail_lp:
  tensor_free(&inputs);
fail_is:
  free(g);
fail_g:
  return 1;
}
