#ifndef GAME_H
#define GAME_H

#include "tensor.h"
#include <stdbool.h>

// WARN: MUST CHANGE WITH NUM_INPUTS,OUTPUTS IN network.h
#define NUM_FACES 6
#define NUM_PLAYERS 4

struct Bid {
  size_t p, c, f;
};

struct Game {
  size_t dice[NUM_PLAYERS][NUM_FACES];
  size_t dice_left[NUM_PLAYERS];
  size_t total_dice[NUM_FACES];
  size_t total_left;
  size_t p;
  size_t turn;
  struct Bid d1bid, d2bid;
};

struct Game *game_new(void);

bool legal(const struct Game *g, size_t count, size_t face);
void bid(struct Game *g, size_t count, size_t face);
bool challenge(struct Game *g);

void get_canonical(const struct Game *g, struct Tensor *t);

void game_print(const struct Game *g);

#endif
