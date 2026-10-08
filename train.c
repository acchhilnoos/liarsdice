#include "train.h"
#include "play.h"
#include <stdio.h>
#include <string.h>

struct Step {
  // TODO: struct of arrays?
  float  in[SIZE_INPUT];
  float  h_prev[SIZE_HIDDEN];
  size_t d[NUM_FACES];
  // TODO: use instead of as, vs, idxs?
  size_t act;
  float  adv;
  float  rwd;
  float  val;
  float  pi;
  bool   terminal;
};

int train(struct Network *n, const char *weights_fn, size_t max_iters,
          size_t max_steps, size_t max_epchs, float alpha, float beta,
          float epsilon, float gamma, float lambda, float c1, float c2,
          bool verbose)
{
  int exit = 1;

  struct Game *g = game_new();
  if (!g) goto fail_g;

  struct Tensor inputs;
  struct Tensor loss_p;
  struct Tensor loss_c;
  struct Tensor h_prev;
  struct Tensor hidden[NUM_PLAYERS];
  struct Tensor activs[NUM_TOTAL_LAYERS];

  if (tensor_init(&inputs, 1, SIZE_INPUT) != 0) goto fail_is;
  if (tensor_init(&loss_p, 1, SIZE_POL) != 0) goto fail_lp;
  if (tensor_init(&loss_c, 1, NUM_FACES) != 0) goto fail_lc;
  if (tensor_init(&h_prev, 1, SIZE_HIDDEN) != 0) goto fail_hp;

  for (size_t i = 0; i < NUM_PLAYERS; i++)
    if (tensor_init(hidden + i, 1, SIZE_HIDDEN) != 0) {
      for (size_t j = 0; j < i; j++) tensor_free(hidden + j);
      goto fail_h;
    }

  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++)
    if (tensor_init(activs + i, 1,
                    i < GRU_WR_IDX ? SIZE_HIDDEN
                    : i < POL_IDX  ? SIZE_GRU
                    : i < VAL_IDX  ? SIZE_POL
                                   : 1) != 0) {
      for (size_t j = 0; j < 1; j++) tensor_free(activs + j);
      goto fail_activs;
    }

  struct Step *step_buf = calloc(max_steps, sizeof(*step_buf));
  if (!step_buf) goto fail_s;

  size_t step_n = 0;

  for (size_t iter = 0; iter < max_iters; iter++) {

    /* --- playouts --- */

    step_n = 0;

    while (step_n < max_steps) {
      size_t a = 0;

      get_canonical(g, &inputs);

      if (g->p == 0)
        memcpy(step_buf[step_n].h_prev, hidden[0].buf,
               SIZE_HIDDEN * sizeof(float));

      network_forward(n, &inputs, hidden + g->p, g, activs);
      memcpy(hidden[g->p].buf, activs[GRU_H_IDX].buf,
             SIZE_HIDDEN * sizeof(*hidden[g->p].buf));

      float one = 0.0f;
      for (size_t i = 0; i < SIZE_POL; i++) one += activs[POL_IDX].buf[i];

      float r = (float)rand() / (RAND_MAX + 1.0f) * one, sum = 0.0f;
      for (size_t i = 0; i < SIZE_POL; i++) {
        sum += activs[POL_IDX].buf[i];
        if (sum > r) {
          a = i;
          break;
        }
      }

      if (g->p == 0) {
        struct Step *step = &step_buf[step_n++];

        *step = (struct Step){.act      = a,
                              .rwd      = 0.0f,
                              .val      = activs[VAL_IDX].buf[0],
                              .pi       = activs[POL_IDX].buf[a],
                              .terminal = false};

        memcpy(&step->in, inputs.buf, tensor_size(&inputs) * sizeof(float));
        memcpy(&step->d, g->game_counts, sizeof(g->game_counts));
      }

      if (a == CHALLENGE_IDX) {
        if (g->p == 0) {
          step_buf[step_n].rwd =
              challenge(g) ? R_CHALLENGE_GOOD : R_CHALLENGE_BAD;
          step_buf[step_n].terminal = true;
        } else if (g->last.p == 0) {
          step_buf[step_n - 1].rwd =
              challenge(g) ? R_CHALLENGED_GOOD : R_CHALLENGED_BAD;
          step_buf[step_n - 1].terminal = true;
        } else {
          challenge(g);
          step_buf[step_n - 1].terminal = true;
        }
        for (size_t i = 0; i < NUM_PLAYERS; i++) tensor_zero(hidden + i);
      } else bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);

      size_t alive = 0;
      for (size_t i = 0; i < NUM_PLAYERS; i++)
        if (g->player_rem[i] != 0) alive++;

      if (alive == 1) {
        if (g->player_rem[0] != 0) step_buf[step_n - 1].rwd += R_WIN;
        else step_buf[step_n - 1].rwd += R_LOSS;
        game_restart(g);
      }
    }

    /* --- value assignment --- */

    float d    = 0.0f;
    float a    = 0.0f;
    float mean = 0.0f;
    float std  = 0.0f;

    for (size_t i = 1; i <= max_steps; i++) {
      size_t       idx  = max_steps - i;
      struct Step *step = &step_buf[idx];

      /**
       * d_t = r_t + gv(s_{t+1})- v(s_t)
       * A_t = d_t + gl(d_{t+1}) + ggll(d_{t+2}) + ...
       *     = d_t + glA_{t+1}
       */
      if (i == 1 || step->terminal) {
        d = step->rwd - step->val;
        a = d;
      } else {
        d = step->rwd + gamma * step_buf[idx + 1].val - step->val;
        a = d + gamma * lambda * a;
      }
      step->adv = a;
      // v_targ(s_t) = v(s_t) + A_t
      step->val = step->val + a;

      mean += a;
    }
    mean /= max_steps;

    for (size_t i = 0; i < max_steps; i++)
      std += powf(step_buf[i].adv - mean, 2);
    std = sqrtf(std / max_steps);
    for (size_t i = 0; i < max_steps; i++)
      step_buf[i].adv = (step_buf[i].adv - mean) / std;

    /* --- backprop  --- */

    struct Game g_temp = {0};

    for (size_t epch = 0; epch < max_epchs; epch++) {
      for (size_t batch = 0; batch < max_steps; batch += MAX_BATCH_SIZE) {
        tensor_zero_grad(&h_prev);
        network_zero_grad(n);

        for (size_t i = (max_steps < batch + MAX_BATCH_SIZE
                             ? max_steps - 1
                             : batch + MAX_BATCH_SIZE - 1);
             i-- > batch;) {
          struct Step *step = &step_buf[i];

          memcpy(inputs.buf, step->in, sizeof(step->in));
          memcpy(h_prev.buf, step->h_prev, sizeof(step->h_prev));

          network_forward(n, &inputs, &h_prev, &g_temp, activs);

          float pi_old = step->pi;
          float pi_new = activs[POL_IDX].buf[step->act];
          float r_t    = pi_new / pi_old;
          /**
           * L_clip = min(rA, clip(r, 1-e, 1+e)A)
           *           / min(rA, (1+e)A)   A > 0
           *        = <                0   A = 0
           *           \ min((1-e)A, rA)   A < 0
           */
          float dl_clip = step->adv > 0
                              ? (r_t > 1 + epsilon ? 0 : -step->adv / pi_old)
                              : (r_t < 1 - epsilon ? 0 : -step->adv / pi_old);

          // L_vf = (v_new(s) - v_targ(s))^2
          float dl_vf = (activs[VAL_IDX].buf[0] - step->val);

          tensor_zero(&loss_p);
          tensor_zero(&loss_c);

          // S = -sum plogp
          for (size_t j = 0; j < SIZE_POL; j++)
            loss_p.buf[j] = c2 * (logf(activs[POL_IDX].buf[j]) + 1);
          loss_p.buf[step->act] += dl_clip;

          network_backward(n, &inputs, &h_prev, activs, &loss_p, c1 * dl_vf);
          if (i > 0 && step_buf[i - 1].terminal) tensor_zero_grad(&h_prev);
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

  exit = 0;

  free(step_buf);
fail_s:
  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) tensor_free(activs + i);
fail_activs:
  for (size_t i = 0; i < NUM_PLAYERS; i++) tensor_free(hidden + i);
fail_h:
  tensor_free(&h_prev);
fail_hp:
  tensor_free(&loss_c);
fail_lc:
  tensor_free(&loss_p);
fail_lp:
  tensor_free(&inputs);
fail_is:
  free(g);
fail_g:
  return exit;
}
