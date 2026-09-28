#include "config.h"
#include "game.h"
#include "network.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void playout(struct Network *n, bool human, bool verbose)
{
  struct Game  *g = game_new();
  struct Tensor inputs;
  tensor_init(&inputs, 1, NUM_INPUTS);

  size_t p_human = human ? rand() % NUM_PLAYERS : NUM_PLAYERS;
  size_t alive;
  size_t round = 0;
  do {
    printf("--- Round %zu ---\n", ++round);
    game_print(g, p_human);

    while (1) {
      size_t a = 0;

      if (g->p == p_human) {
        size_t c, f;
        printf("Your turn ([count face] or [%d 1] to challenge): ",
               NUM_PLAYERS * 5 + 1);
        if (scanf("%zu %zu", &c, &f) == 2) {
          a = (c - 1) * NUM_FACES + (f - 1);
          if (!legal(g, c, f) && (a != CHALLENGE_IDX || g->last.c == 0))
            continue;
        } else continue;
      } else {
        get_canonical(g, &inputs);
        network_forward(n, &inputs, g);

        float r = (float)rand() / (RAND_MAX + 1.0f), s = 0.0f;
        float sum = 0.0f;
        for (size_t i = 0; i < NUM_POL_OUT; i++) sum += n->as[POL_HEAD].buf[i];
        r *= sum;
        for (size_t i = 0; i < NUM_POL_OUT; i++) {
          s += n->as[POL_HEAD].buf[i];
          if (s > r) {
            a = i;
            break;
          }
        }

        if (!human && verbose) network_peek(n);
      }
      if (a == CHALLENGE_IDX) {
        if (human) game_print(g, NUM_PLAYERS);
        size_t p     = g->p;
        bool   chall = challenge(g);
        printf("p%zu challenge: %s\n\n", p + 1, chall ? "good" : "bad");
        break;
      } else {
        printf("p%zu: %2zux%2zu\n", g->p + 1, (a / NUM_FACES) + 1,
               (a % NUM_FACES) + 1);
        bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);
      }
    }

    alive = 0;
    for (size_t i = 0; i < NUM_PLAYERS; i++)
      if (g->player_rem[i] != 0) alive++;
  } while (alive > 1);

  tensor_free(&inputs);
  free(g);
}

int main(int argc, char *argv[])
{
  srand(time(NULL));

  char  *weights_fn  = "weights.bin";
  size_t max_iters   = 10000;
  size_t max_steps   = 2048;
  size_t max_epchs   = 15;
  bool   verbose     = false;
  bool   playout_n   = false;
  bool   playout_p   = false;
  float  alpha       = 0.005f;
  float  beta        = 0.9f;
  float  epsilon     = 0.2f;
  float  gamma       = 0.95f;
  gamma             *= gamma;
  gamma             *= gamma;
  float lambda       = 0.99f;
  lambda            *= lambda;
  lambda            *= lambda;
  float c1           = 1.0f;
  float c2           = 0.05f;

  struct Game    *g = game_new();
  struct Network *n = network_new();
  struct Tensor   inputs;
  struct Tensor   loss_p;
  struct Tensor   loss_c;
  tensor_init(&inputs, 1, NUM_INPUTS);
  tensor_init(&loss_p, 1, NUM_POL_OUT);
  tensor_init(&loss_c, 1, NUM_FACES);

  struct Step {
    float  in[NUM_INPUTS];
    size_t d[NUM_FACES];
    size_t a;
    float  r;
    float  v;
    float  pi;
    bool   terminal;
  };
  struct Step *step_buf = calloc(max_steps, sizeof(*step_buf));
  size_t       step_n   = 0;

  float  *as   = calloc(max_steps, sizeof(*as));
  float  *vs   = calloc(max_steps, sizeof(*vs));
  size_t *idxs = calloc(max_steps, sizeof(*idxs));

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-b") == 0) {
      network_benchmark();
      goto free;
    } else if (strcmp(argv[i], "-c1") == 0) {
      c1 = atof(argv[++i]);
    } else if (strcmp(argv[i], "-c2") == 0) {
      c2 = atof(argv[++i]);
    } else if (strcmp(argv[i], "-e") == 0) {
      max_epchs = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-i") == 0) {
      max_iters = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-l") == 0) {
      if (argc > i + 1) weights_fn = argv[++i];
      network_load(n, weights_fn);
    } else if (strcmp(argv[i], "-s") == 0) {
      max_steps = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-v") == 0) {
      verbose = true;
    } else if (strcmp(argv[i], "-n") == 0) {
      playout_n = true;
    } else if (strcmp(argv[i], "-p") == 0) {
      playout_p = true;
    } else {
      goto free;
    }
  }

  if (playout_n) {
    playout(n, false, verbose);
    goto free;
  }
  if (playout_p) {
    playout(n, true, true);
    goto free;
  }

  /* --- --- */

  for (size_t iter = 0; iter < max_iters; iter++) {
    step_n = 0;

    while (step_n < max_steps) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / (RAND_MAX + 1.0f), s = 0.0f;
      float sum = 0.0f;
      for (size_t i = 0; i < NUM_POL_OUT; i++) sum += n->as[POL_HEAD].buf[i];
      r *= sum;
      for (size_t i = 0; i < NUM_POL_OUT; i++) {
        s += n->as[POL_HEAD].buf[i];
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
        step->v        = n->as[VAL_HEAD].buf[0];
        step->pi       = n->as[POL_HEAD].buf[a];
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

    /* --- --- */

    float d    = 0.0f;
    float a    = 0.0f;
    float v    = 0.0f;
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
      if (i == 1) {
        d = step->r - step->v;
        a = d;
      } else {
        d = step->r + gamma * step_buf[idx + 1].v - step->v;
        a = d + gamma * lambda * a;
      }
      // v_targ(s_t) = v(s_t) + A_t
      v = step->v + a;

      as[idx]     = a;
      vs[idx]     = v;
      idxs[i - 1] = i - 1;

      mean += a;
    }
    mean /= max_steps;

    for (size_t i = 0; i < max_steps; i++) std += powf(as[i] - mean, 2);
    std = sqrtf(std / max_steps);
    for (size_t i = 0; i < max_steps; i++) as[i] = (as[i] - mean) / std;

    /* --- --- */

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

          g_temp.game_rem = (size_t)(step->in[NUM_FACES + NUM_PLAYERS + 2] *
                                         (NUM_PLAYERS * 5) +
                                     0.5f);
          g_temp.last.c =
              (size_t)(step->in[NUM_FACES] * g_temp.game_rem + 0.5f);
          g_temp.last.f = (size_t)(step->in[NUM_FACES + 1] * NUM_FACES + 0.5f);

          memcpy(inputs.buf, step->in, sizeof(step->in));
          network_forward(n, &inputs, &g_temp);

          float pi_old = step->pi;
          float pi_new = fmaxf(n->as[POL_HEAD].buf[step->a], 1e-10f);
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

          /* L_vf = (v_new(s) - v_targ(s))^2 */
          float dl_vf = (n->as[VAL_HEAD].buf[0] - vs[idx]);

          tensor_zero(&loss_p);
          tensor_zero(&loss_c);

          /* S = -sum plogp */
          for (size_t j = 0; j < NUM_POL_OUT; j++)
            loss_p.buf[j] =
                c2 * (logf(fmaxf(n->as[POL_HEAD].buf[j], 1e-10f)) + 1);
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

free:
  free(idxs);
  free(vs);
  free(as);
  free(step_buf);
  tensor_free(&loss_c);
  tensor_free(&loss_p);
  tensor_free(&inputs);
  network_free(n);
  free(g);
  return 0;
}
