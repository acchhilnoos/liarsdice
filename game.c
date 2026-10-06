#include "game.h"
#include "config.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline size_t cftoidx(size_t c, size_t f)
{ return (c - 1) * NUM_FACES + (f - 1); }

static inline void advance(struct Game *g)
{
  do g->p = (g->p + 1) % NUM_PLAYERS;
  while (g->player_rem[g->p] == 0);
  g->turn++;
}

void roll(struct Game *g)
{
  memset(g->player_counts, 0, sizeof(g->player_counts));
  memset(g->game_counts, 0, sizeof(g->game_counts));

  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    for (size_t j = 0; j < g->player_rem[i]; j++) {
      int r                   = rand() % NUM_FACES;
      g->player_counts[i][r] += 1;
      g->game_counts[r]      += 1;
    }
  }

  g->turn   = 0;
  g->last.c = 0;
  g->last.f = 0;
  g->last.p = 0;
}

void game_restart(struct Game *g)
{
  *g = (struct Game){0};
  for (size_t i = 0; i < NUM_PLAYERS; i++)
    g->player_rem[i] = NUM_DICE_PER_PLAYER;
  roll(g);
}

struct Game *game_new(void)
{
  struct Game *g = malloc(sizeof(*g));
  if (!g) return NULL;

  game_restart(g);
  return g;
}

bool legal(const struct Game *g, size_t c, size_t f)
{
  size_t rem = 0;
  for (size_t i = 0; i < NUM_PLAYERS; i++) rem += g->player_rem[i];
  return c <= rem && ((g->last.c == 0 && g->last.f == 0) ||
                      (cftoidx(c, f) > cftoidx(g->last.c, g->last.f)));
}

void bid(struct Game *g, size_t c, size_t f)
{
  g->last.c = c;
  g->last.f = f;
  g->last.p = g->p;

  advance(g);
}

bool challenge(struct Game *g)
{
  bool good = false;

  size_t sum = g->game_counts[g->last.f - 1];
  if (g->last.f != 1) sum += g->game_counts[0];

  if (sum < g->last.c) {
    good = true;
    g->p = g->last.p;
  }

  g->player_rem[g->p]--;

  if (g->player_rem[g->p] == 0) advance(g);
  else g->turn++;

  roll(g);
  return good;
}

void get_canonical(const struct Game *g, struct Tensor *t)
{
  size_t idx = 0;
  size_t rem = 0;
  for (size_t i = 0; i < NUM_PLAYERS; i++) rem += g->player_rem[i];

  for (size_t i = 0; i < NUM_FACES; i++)
    t->buf[idx++] = (float)g->player_counts[g->p][i] / g->player_rem[g->p];

  t->buf[idx++] = (float)g->player_rem[g->p] / rem;

  for (size_t i = 0; i < NUM_PLAYERS; i++)
    t->buf[idx++] = (float)g->player_rem[(g->p + i) % NUM_PLAYERS] / rem;

  t->buf[idx++] = (float)rem / NUM_TOTAL_DICE;
}

void game_print(const struct Game *g, size_t p_human)
{
  size_t rem = 0;
  for (size_t i = 0; i < NUM_PLAYERS; i++) rem += g->player_rem[i];

  printf("            1  2  3  4  5  6\n");

  for (size_t i = 0; i < NUM_PLAYERS; i++) {
    printf("player %zu: ", i + 1);
    for (size_t j = 0; j < NUM_FACES; j++) {
      if (g->player_counts[i][j] != 0 &&
          (p_human >= NUM_PLAYERS || i == p_human))
        printf("%3zu", g->player_counts[i][j]);
      else printf("   ");
    }
    printf("  |%3zu\n", g->player_rem[i]);
  }

  for (size_t i = 0; i < 34; i++) printf("-");

  printf("\n  totals: ");
  for (size_t i = 0; i < NUM_FACES; i++) {
    if (g->game_counts[i] != 0)
      if (p_human >= NUM_PLAYERS) printf("%3zu", g->game_counts[i]);
      else if (g->player_counts[p_human][i] != 0)
        printf("%3zu", g->player_counts[p_human][i]);
      else printf("   ");
    else printf("   ");
  }
  printf("  |%3zu\n             ", rem);
  for (size_t i = 1; i < NUM_FACES; i++) {
    if (g->game_counts[i] + g->game_counts[1] != 0)
      if (p_human >= NUM_PLAYERS)
        printf("%3zu", g->game_counts[i] + g->game_counts[0]);
      else if (g->player_counts[p_human][i] + g->player_counts[p_human][0] != 0)
        printf("%3zu",
               g->player_counts[p_human][i] + g->player_counts[p_human][0]);
      else printf("   ");
    else printf("   ");
  }
  printf("\n");
}
