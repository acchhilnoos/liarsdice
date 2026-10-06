#ifndef GAME_H
#define GAME_H

#include "config.h"
#include "tensor.h"
#include <stdbool.h>
#include <stddef.h>

struct Game {
  size_t player_counts[NUM_PLAYERS][NUM_FACES];
  size_t game_counts[NUM_FACES];
  size_t player_rem[NUM_PLAYERS];
  size_t p;
  size_t turn;
  struct {
    size_t c, f, p;
  } last;
};

struct Game *game_new(void);
void         game_restart(struct Game *g);

/** 1-indexed legal bet check */
bool legal(const struct Game *g, size_t count, size_t face);

void bid(struct Game *g, size_t count, size_t face);
bool challenge(struct Game *g);

void get_canonical(const struct Game *g, struct Tensor *t);

void game_print(const struct Game *g, size_t p_human);

#endif
