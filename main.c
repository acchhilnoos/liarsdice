#include "game.h"
#include "network.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char *argv[]) {
  srand(time(NULL));
  size_t iters   = 1000;
  size_t steps   = 200;
  size_t epchs   = 10;
  bool   verbose = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "bench") == 0) {
      network_benchmark();
      return 0;
    }
    if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--random") == 0) {
      struct Game *g = game_new();

      size_t alive;
      do {
        if (verbose)
          game_print(g);
        while (1) {
          size_t c, f;

          if (g->d1bid.c == g->total_left && g->d1bid.f == 6) {
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
      return 0;
    }
    if (strcmp(argv[i], "-e") == 0)
      epchs = atoi(argv[++i]);
    if (strcmp(argv[i], "-i") == 0)
      iters = atoi(argv[++i]);
    if (strcmp(argv[i], "-s") == 0)
      steps = atoi(argv[++i]);
    if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0)
      verbose = true;
  }

  struct Game    *g = game_new();
  struct Network *n = network_new();
  struct Tensor   inputs;
  tensor_init(&inputs, 1, 1, 1, NUM_INPUTS);
  struct {
    float  state[NUM_INPUTS];
    size_t a;
    float  r;
    float  v;
    float  log_pi;
    bool   terminal;
  } buf[steps];
  size_t buf_n = 0;

  for (size_t i = 0; i < iters; i++) {
    buf_n     = 0;
    bool flag = false;

    while (buf_n < steps) {
      size_t a = 0;

      get_canonical(g, &inputs);
      network_forward(n, &inputs, g);

      float r = (float)rand() / RAND_MAX;
      float s = 0.0f;
      for (size_t j = 0; j < NUM_OUTPUTS; j++) {
        s += n->as[3].buf[j];
        if (s > r) {
          a = j;
          break;
        }
      }

      if (g->p == 0) {
        memcpy(&buf[buf_n].state, inputs.buf,
               tensor_size(&inputs) * sizeof(float));
        buf[buf_n].a      = a;
        buf[buf_n].v      = n->as[4].buf[0];
        buf[buf_n].log_pi = log2f(n->as[3].buf[a]);

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
      for (size_t j = 0; j < NUM_PLAYERS; j++)
        if (g->dice_left[j] != 0)
          alive++;
      if (alive == 1) {
        if (g->dice_left[0] != 0)
          buf[buf_n - 1].r = 1.0f;
        else
          buf[buf_n - 1].r = -1.0f;
        free(g);
        g = game_new();
      }
    }
  }

  tensor_free(&inputs);
  network_free(n);
  free(g);
  return 0;
}
