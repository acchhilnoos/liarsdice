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
        f = rand() % 6 + 1;
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
  tensor_init(&inputs, 1, 1, 1, NUM_INPUTS);

  size_t alive;
  do {
    game_print(g);

    while (1) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / RAND_MAX;
      float s = 0.0f;
      for (size_t i = 0; i < NUM_OUTPUTS; i++) {
        s += n->as[POL_HEAD].buf[i];
        if (s > r) {
          a = i;
          break;
        }
      }
      if (a == 120) {
        challenge(g);
        break;
      } else
        bid(g, (a / 6) + 1, (a % 6) + 1);
    }

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
  size_t iters   = 1000;
  size_t steps   = 200;
  size_t epchs   = 10;
  float  alpha   = 0.01f;
  float  beta    = 0.9f;
  float  epsilon = 0.2f;
  float  gamma   = 0.95f;
  float  lambda  = 0.99f;
  float  c1      = 1.0f;
  float  c2      = 0.01f;

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
      epchs = atoi(argv[++i]);
    if (strcmp(argv[i], "-i") == 0)
      iters = atoi(argv[++i]);
    if (strcmp(argv[i], "-s") == 0)
      steps = atoi(argv[++i]);
  }

  /* --- --- */

  struct Game    *g = game_new();
  struct Network *n = network_new();
  struct Tensor   inputs;
  struct Tensor   loss_p;
  tensor_init(&inputs, 1, 1, 1, NUM_INPUTS);
  tensor_init(&loss_p, 1, 1, 1, NUM_OUTPUTS);

  struct {
    float  state[NUM_INPUTS];
    size_t a;
    float  r;
    float  v;
    float  log_pi;
    bool   terminal;
  } buf[steps];
  size_t buf_n = 0;

  float  as[steps];
  float  rs[steps];
  size_t idxs[steps];

  for (size_t iter = 0; iter < iters; iter++) {
    buf_n     = 0;
    bool flag = false;

    while (buf_n < steps) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / RAND_MAX;
      float s = 0.0f;
      for (size_t i = 0; i < NUM_OUTPUTS; i++) {
        s += n->as[POL_HEAD].buf[i];
        if (s > r) {
          a = i;
          break;
        }
      }

      if (g->p == 0) {
        memcpy(&buf[buf_n].state, inputs.buf,
               tensor_size(&inputs) * sizeof(float));
        buf[buf_n].a      = a;
        buf[buf_n].r      = 0.0f;
        buf[buf_n].v      = n->as[VAL_HEAD].buf[0];
        buf[buf_n].log_pi = logf(n->as[POL_HEAD].buf[a]);

        if (a == 120)
          if (challenge(g))
            buf[buf_n].r = 0.5f;
          else
            buf[buf_n].r = -0.5f;
        else
          bid(g, (a / 6) + 1, (a % 6) + 1);

        buf_n++;
        flag = true;
      } else if (flag) {
        if (a == 120)
          if (challenge(g)) {
            buf[buf_n - 1].r = -0.2f;
          } else {
            buf[buf_n - 1].r = 0.2f;
          }
        else
          bid(g, (a / 6) + 1, (a % 6) + 1);

        flag = false;
      } else {
        if (a == 120)
          challenge(g);
        else
          bid(g, (a / 6) + 1, (a % 6) + 1);
      }

      size_t alive = 0;
      for (size_t i = 0; i < NUM_PLAYERS; i++)
        if (g->dice_left[i] != 0)
          alive++;
      if (alive == 1) {
        buf[buf_n - 1].terminal = true;
        if (g->dice_left[0] != 0)
          buf[buf_n - 1].r = 1.0f;
        else
          buf[buf_n - 1].r = -1.0f;
        free(g);
        g = game_new();
      }
    }

    /* --- --- */

    float d   = 0.0f;
    float a   = 0.0f;
    float r   = 0.0f;
    float sum = 0.0f;

    for (size_t step = 1; step <= steps; step++) {
      if (step == 1 || buf[steps - step].terminal) {
        d = buf[steps - step].r - buf[steps - step].v;
        a = d;
      } else {
        d = buf[steps - step].r + gamma * buf[steps - step + 1].v -
            buf[steps - step].v;
        a = d + gamma * lambda * a;
      }
      r = a + buf[steps - step].v;

      as[steps - step] = a;
      rs[steps - step] = r;
      sum += a;

      idxs[step - 1] = step - 1;
    }

    /* --- --- */

    struct Game g_temp = {0};

    for (size_t epch = 0; epch < epchs; epch++) {
      for (size_t i = 1; i < steps; i++) {
        size_t r = rand() % (i + 1);
        size_t t = idxs[i];
        idxs[i]  = idxs[r];
        idxs[r]  = t;
      }

      for (size_t batch = 0; batch < steps; batch += MAX_BATCH_SIZE) {
        network_zero_grad(n);

        for (size_t i = 0; i < MAX_BATCH_SIZE && batch + i < steps; i++) {
          size_t idx = idxs[batch + i];

          g_temp.total_left = buf[idx].state[5] + buf[idx].state[6] +
                              buf[idx].state[7] + buf[idx].state[14];
          g_temp.d1bid =
              (struct Bid){.c = buf[idx].state[2], .f = buf[idx].state[3]};

          memcpy(inputs.buf, buf[idx].state, sizeof(buf[idx].state));
          network_forward(n, &inputs, &g_temp);

          float pi_old  = expf(buf[idx].log_pi);
          float pi_new  = n->as[POL_HEAD].buf[buf[idx].a];
          float r_t     = pi_new / pi_old;
          float dl_clip = as[idx] > 0
                              ? (r_t > 1 + epsilon ? 0 : as[idx] / pi_old)
                              : (r_t < 1 - epsilon ? 0 : as[idx] / pi_old);

          float dl_vf = (n->as[VAL_HEAD].buf[0] - rs[idx]);

          float dl_s = -logf(pi_new) - 1;

          tensor_zero(&loss_p);
          loss_p.buf[buf[idx].a] = dl_clip + c2 * dl_s;
          network_backward(n, &inputs, &loss_p, c1 * dl_vf);
        }
        network_sgd(n, alpha / MAX_BATCH_SIZE, beta);
      }
    }
    // if (iter % 10 == 0)
    //   network_playout(n);
  }

  tensor_free(&loss_p);
  tensor_free(&inputs);
  network_free(n);
  free(g);
  return 0;
}
