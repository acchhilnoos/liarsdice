#include "game.h"
#include "tensor.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define UNREACHABLE(cond)                                                      \
  do {                                                                         \
    if (cond) {                                                                \
      fprintf(stderr, "unreachable at %s:%d (%s)!", __FILE_NAME__, __LINE__,   \
              __FUNCTION__);                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

void roll(struct Game *g) {
  g->d1bid = (struct Bid){0};
  g->d2bid = (struct Bid){0};
  g->turn  = 0;
  for (size_t i = 0; i < NUM_FACES; i++)
    g->total_dice[i] = 0;

  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    for (size_t j = 0; j < NUM_FACES; j++)
      g->dice[i][j] = 0;

    for (size_t j = 0; j < g->dice_left[i]; j++) {
      int r = rand() % NUM_FACES;
      g->dice[i][r] += 1;
      g->total_dice[r] += 1;
    }
  }
}

struct Game *game_new(void) {
  struct Game *g = malloc(sizeof(*g));
  if (!g)
    return NULL;

  *g = (struct Game){0};

  for (size_t i = 0; i < NUM_PLAYERS; i++)
    g->dice_left[i] = 5;
  g->total_left = 5 * NUM_PLAYERS;
  roll(g);

  return g;
}

bool legal(const struct Game *g, size_t c, size_t f) {
  return c <= g->total_left &&
         (c > g->d1bid.c || (c == g->d1bid.c && f > g->d1bid.f));
}

void bid(struct Game *g, size_t c, size_t f) {
  UNREACHABLE(!legal(g, c, f));

  g->d2bid = g->d1bid;
  g->d1bid = (struct Bid){.p = g->p, .c = c, .f = f};
  do
    g->p = (g->p + 1) % NUM_PLAYERS;
  while (g->dice_left[g->p] == 0);
  g->turn++;
}

void challenge(struct Game *g) {
  UNREACHABLE(g->d1bid.c == 0 || g->d1bid.f == 0);

  size_t sum = g->total_dice[g->d1bid.f - 1];
  if (g->d1bid.f != 1)
    sum += g->total_dice[0];

  if (sum >= g->d1bid.c) {
    g->dice_left[g->p]--;
  } else {
    g->dice_left[g->d1bid.p]--;
    g->p = g->d1bid.p;
  }
  g->total_left--;

  roll(g);
}

void get_canonical(const struct Game *g, struct Tensor *t) {
  t->buf[0] = g->d2bid.c;
  t->buf[1] = g->d2bid.f;
  t->buf[2] = g->d1bid.c;
  t->buf[3] = g->d1bid.f;
  t->buf[4] = g->turn;
  for (size_t i = 0, j = 5; i < NUM_PLAYERS; i++)
    if (i == g->p)
      continue;
    else
      t->buf[j++] = g->dice_left[i];
  t->buf[8]  = g->dice[g->p][0];
  t->buf[9]  = g->dice[g->p][1];
  t->buf[10] = g->dice[g->p][2];
  t->buf[11] = g->dice[g->p][3];
  t->buf[12] = g->dice[g->p][4];
  t->buf[13] = g->dice[g->p][5];
}

void game_print(const struct Game *g) {
  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    printf("player %zu: ", i + 1);
    if (g->dice_left[i] != 0) {
      for (size_t j = 0; j < NUM_FACES; j++) {
        if (g->dice[i][j] != 0)
          printf("  %zu", g->dice[i][j]);
        else
          printf("   ");
      }
      printf("  | %2zu\n", g->dice_left[i]);
    } else {
      printf("\n");
    }
  }
  printf("  totals: ");
  for (size_t i = 0; i < NUM_FACES; i++) {
    if (g->total_dice[i] != 0)
      printf("  %zu", g->total_dice[i]);
    else
      printf("   ");
  }
  printf("  | %2zu\n", g->total_left);
}
