#ifndef CONFIG_H
#define CONFIG_H

#define UNREACHABLE(cond)                                                      \
  do {                                                                         \
    if (cond) {                                                                \
      fprintf(stderr, "unreachable at %s:%d (%s)!", __FILE_NAME__, __LINE__,   \
              __FUNCTION__);                                                   \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define NUM_FACES 6
#define NUM_PLAYERS 4

/*
 * last last count x1
 * last last face  x1
 * last count      x1
 * last face       x1
 * turn            x1
 * opp 1-3 left    x3
 * hand 1-6s       x6
 * hand left       x1
 */
#define NUM_INPUTS 15
/*
 * [count: 1-20] x [face: 1-6] x120
 * challenge                   x1
 */
#define NUM_OUTPUTS 121
#define CHALLENGE_IDX 120
#define NUM_LAYERS 5
#define POL_HEAD 3
#define VAL_HEAD 4
#define MAX_BATCH_SIZE 64

#endif
