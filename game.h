#ifndef GAME_H
#define GAME_H

#include "config.h"
#include "tensor.h"
#include <stdbool.h>

struct Bid {
  size_t p, c, f;
};

struct Game {
  size_t bids[NUM_FACES];
  size_t player_counts[NUM_PLAYERS][NUM_FACES];
  size_t game_counts[NUM_FACES];
  size_t player_rem[NUM_PLAYERS];
  size_t game_rem;
  size_t p, turn;
  struct Bid last;
};

struct Game *game_new(void);
void game_restart(struct Game *g);

bool legal(const struct Game *g, size_t count, size_t face);
void bid(struct Game *g, size_t count, size_t face);
bool challenge(struct Game *g);

void get_canonical(const struct Game *g, struct Tensor *t);

void game_print(const struct Game *g);

#endif
