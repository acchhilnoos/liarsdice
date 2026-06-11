#include "game.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
  srand(time(NULL));

  struct Game *g = game_new();
  game_print(g);

  size_t alive;
  do {
    while (1) {
      size_t c, f;

      if (g->d1bid.c == g->total_left && g->d1bid.f == 6) {
        challenge(g);
        game_print(g);
        break;
      }

      do {
        c = rand() % (g->total_left - g->d1bid.c + 1) + g->d1bid.c;
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
