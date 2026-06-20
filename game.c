#include "game.h"
#include "config.h"
#include "tensor.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define cftoidx(C, F) ((C - 1) * NUM_FACES + (F - 1))

#define advance(G)                                                             \
  do {                                                                         \
    do                                                                         \
      (G)->p = ((G)->p + 1) % NUM_PLAYERS;                                     \
    while ((G)->player_rem[(G)->p] == 0);                                      \
    (G)->turn++;                                                               \
  } while (0)

#define advance_if_inactive(G)                                                 \
  do {                                                                         \
    while ((G)->player_rem[(G)->p] == 0)                                       \
      (G)->p = ((G)->p + 1) % NUM_PLAYERS;                                     \
    (G)->turn++;                                                               \
  } while (0)

void roll(struct Game *g) {
  memset(g->bids, 0, sizeof(g->bids));
  memset(g->player_counts, 0, sizeof(g->player_counts));
  memset(g->game_counts, 0, sizeof(g->game_counts));

  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    for (size_t j = 0; j < g->player_rem[i]; j++) {
      int r = rand() % NUM_FACES;
      g->player_counts[i][r] += 1;
      g->game_counts[r] += 1;
    }
  }

  g->turn = 0;
  g->last = (struct Bid){0};
}

struct Game *game_new(void) {
  struct Game *g = malloc(sizeof(*g));
  if (!g)
    return NULL;

  *g = (struct Game){0};

  for (size_t i = 0; i < NUM_PLAYERS; i++)
    g->player_rem[i] = 5;
  g->game_rem = 5 * NUM_PLAYERS;
  roll(g);

  return g;
}

void game_restart(struct Game *g) {
  *g = (struct Game){0};
  for (size_t i = 0; i < NUM_PLAYERS; i++)
    g->player_rem[i] = 5;
  g->game_rem = 5 * NUM_PLAYERS;
  roll(g);
}

bool legal(const struct Game *g, size_t c, size_t f) {
  return c <= g->game_rem && ((g->last.c == 0 && g->last.f == 0) ||
                              (cftoidx(c, f) > cftoidx(g->last.c, g->last.f)));
}

void bid(struct Game *g, size_t c, size_t f) {
  g->bids[f - 1] = c;
  g->last        = (struct Bid){.p = g->p, .c = c, .f = f};

  advance(g);
}

bool challenge(struct Game *g) {
  bool good = false;

  size_t sum = g->game_counts[g->last.f - 1];
  if (g->last.f != 1)
    sum += g->game_counts[0];

  if (sum < g->last.c) {
    good = true;
    g->p = g->last.p;
  }

  g->player_rem[g->p]--;
  g->game_rem--;

  advance_if_inactive(g);

  roll(g);
  return good;
}

void get_canonical(const struct Game *g, struct Tensor *t) {
  size_t idx = 0;

  /* max bid of face */
  for (size_t i = 0; i < NUM_FACES; i++)
    t->buf[idx++] = (float)g->bids[i] / g->game_rem;

  /* last bid */
  t->buf[idx++] = (float)g->last.c / g->game_rem;
  t->buf[idx++] = (float)g->last.f / NUM_FACES;

  /* players' # remaining dice */
  for (size_t i = 0; i < NUM_PLAYERS; i++)
    t->buf[idx++] =
        (float)g->player_rem[(g->p + i) % NUM_PLAYERS] / g->game_rem;

  /* # remaining dice */
  t->buf[idx++] = (float)g->game_rem / (NUM_PLAYERS * 5);

  /* player hand */
  for (size_t i = 0; i < NUM_FACES; i++)
    t->buf[idx++] = (float)g->player_counts[g->p][i] / g->player_rem[g->p];
}

void game_print(const struct Game *g) {
  printf("            1  2  3  4  5  6\n");

  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    printf("player %zu: ", i + 1);
    for (size_t j = 0; j < NUM_FACES; j++) {
      if (g->player_counts[i][j] != 0)
        printf("%3zu", g->player_counts[i][j]);
      else
        printf("   ");
    }
    printf("  |%3zu\n", g->player_rem[i]);
  }

  for (size_t i = 0; i < 34; i++)
    printf("-");

  printf("\n  totals: ");
  for (size_t i = 0; i < NUM_FACES; i++) {
    if (g->game_counts[i] != 0)
      printf("%3zu", g->game_counts[i]);
    else
      printf("   ");
  }
  printf("  |%3zu\n", g->game_rem);
}
