#include "play.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int playout(struct Network *n, bool human, bool verbose)
{
  int exit = 1;

  struct Game *g = game_new();
  if (!g) goto fail_g;

  struct Tensor inputs;
  struct Tensor hidden[NUM_PLAYERS];
  struct Tensor activs[NUM_TOTAL_LAYERS];

  if (tensor_init(&inputs, 1, SIZE_INPUT) != 0) goto fail_i;

  for (size_t i = 0; i < NUM_PLAYERS; i++)
    if (tensor_init(hidden + i, 1, SIZE_HIDDEN) != 0)
      for (size_t j = 0; j < i; j++) {
        tensor_free(hidden + j);
        goto fail_h;
      }

  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++)
    if (tensor_init(activs + i, 1,
                    i < GRU_WR_IDX ? SIZE_HIDDEN
                    : i < POL_IDX  ? SIZE_GRU
                    : i < VAL_IDX  ? SIZE_POL
                                   : 1) != 0)
      for (size_t j = 0; j < i; j++) {
        tensor_free(activs + j);
        goto fail_a;
      }

  size_t p_human = human ? rand() % NUM_PLAYERS : NUM_PLAYERS;
  size_t round   = 0;

  while (1) {
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

        if (!human && verbose) network_peek(n, g, hidden + g->p);
      }

      if (a == CHALLENGE_IDX) {
        if (human) game_print(g, NUM_PLAYERS);
        printf("p%zu challenge: %s\n\n", g->p + 1,
               challenge(g) ? "good" : "bad");
        for (size_t i = 0; i < NUM_PLAYERS; i++) tensor_zero(hidden + i);
        break;
      } else {
        printf("p%zu: %2zux%2zu\n", g->p + 1, (a / NUM_FACES) + 1,
               (a % NUM_FACES) + 1);
        bid(g, (a / NUM_FACES) + 1, (a % NUM_FACES) + 1);
      }
    }

    size_t alive = 0;
    for (size_t i = 0; i < NUM_PLAYERS; i++)
      if (g->player_rem[i] != 0) alive++;
    if (alive == 1) break;
  }

  exit = 0;

  for (size_t i = 0; i < NUM_TOTAL_LAYERS; i++) tensor_free(activs + i);
fail_a:
  for (size_t i = 0; i < NUM_PLAYERS; i++) tensor_free(hidden + i);
fail_h:
  tensor_free(&inputs);
fail_i:
  free(g);
fail_g:
  return exit;
}
