#include "game.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>

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
  return c > 0 && c <= g->total_left &&
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

bool challenge(struct Game *g) {
  UNREACHABLE(g->d1bid.c == 0 || g->d1bid.f == 0);
  bool good = false;

  size_t sum = g->total_dice[g->d1bid.f - 1];
  if (g->d1bid.f != 1)
    sum += g->total_dice[0];

  if (sum >= g->d1bid.c) {
    g->dice_left[g->p]--;
  } else {
    good = true;
    g->dice_left[g->d1bid.p]--;
    g->p = g->d1bid.p;
  }
  while (g->dice_left[g->p] == 0)
    g->p = (g->p + 1) % NUM_PLAYERS;
  g->total_left--;

  roll(g);

  return good;
}

void get_canonical(const struct Game *g, struct Tensor *t) {
  t->buf[0] = (float)g->d2bid.c / g->total_left;
  t->buf[1] = (float)g->d2bid.f / NUM_FACES;
  t->buf[2] = (float)g->d1bid.c / g->total_left;
  t->buf[3] = (float)g->d1bid.f / NUM_FACES;
  t->buf[4] = (float)g->turn / (NUM_PLAYERS * 5);
  for (size_t i = g->p, j = 5; j < 8; i++)
    if (i == g->p)
      continue;
    else
      t->buf[j++] = (float)g->dice_left[i % NUM_PLAYERS] / g->total_left;
  t->buf[8]  = (float)g->dice[g->p][0] / g->dice_left[g->p];
  t->buf[9]  = (float)g->dice[g->p][1] / g->dice_left[g->p];
  t->buf[10] = (float)g->dice[g->p][2] / g->dice_left[g->p];
  t->buf[11] = (float)g->dice[g->p][3] / g->dice_left[g->p];
  t->buf[12] = (float)g->dice[g->p][4] / g->dice_left[g->p];
  t->buf[13] = (float)g->dice[g->p][5] / g->dice_left[g->p];
  t->buf[14] = (float)g->dice_left[g->p] / g->total_left;
}

void game_print(const struct Game *g) {
  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    printf("player %zu: ", i + 1);
    for (size_t j = 0; j < NUM_FACES; j++) {
      if (g->dice[i][j] != 0)
        printf("  %zu", g->dice[i][j]);
      else
        printf("   ");
    }
    printf("  | %2zu\n", g->dice_left[i]);
  }
  for (size_t i = 0; i < 34; i++)
    printf("-");
  printf("\n  totals: ");
  for (size_t i = 0; i < NUM_FACES; i++) {
    if (g->total_dice[i] != 0)
      printf("  %zu", g->total_dice[i]);
    else
      printf("   ");
  }
  printf("  | %2zu\n", g->total_left);
}
