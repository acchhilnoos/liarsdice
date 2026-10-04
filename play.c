#include "play.h"
#include "network.h"
#include "stdlib.h"
#include <stdio.h>

int playout(struct Network *n, bool human, bool verbose)
{
  struct Game *g = game_new();
  if (!g) goto fail_g;

  struct Tensor inputs;
  if (tensor_init(&inputs, 1, NUM_INPUTS) != 0) goto fail_i;

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
               NUM_TOTAL_DICE + 1);
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
        for (size_t i = 0; i < SIZE_POL; i++) sum += n->as[POL_HEAD_IDX].buf[i];
        r *= sum;
        for (size_t i = 0; i < SIZE_POL; i++) {
          s += n->as[POL_HEAD_IDX].buf[i];
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
  return 0;

fail_i:
  free(g);
fail_g:
  return 1;
}
