#ifndef CONFIG_H
#define CONFIG_H

#define NUM_DICE_PER_PLAYER 5
#define NUM_FACES           6
#define NUM_PLAYERS         4
#define NUM_TOTAL_DICE      (NUM_DICE_PER_PLAYER * NUM_PLAYERS)
#define NUM_LEGAL_BIDS      (NUM_TOTAL_DICE * NUM_FACES)
#define CHALLENGE_IDX       NUM_LEGAL_BIDS

/**
 * highest bids  xF
 * last count    x1
 * last face     x1
 * player # dice xP
 * # dice        x1
 * hand          xF
 */
#define NUM_INPUTS   (NUM_PLAYERS + 2 * NUM_FACES + 3)
#define NUM_LAYERS   5
#define SIZE_HIDDEN  128
#define POL_HEAD_IDX (NUM_LAYERS - 2)
/**
 * action space x5P*F
 * challenge    x1
 */
#define SIZE_POL     (NUM_LEGAL_BIDS + 1)
#define VAL_HEAD_IDX (NUM_LAYERS - 1)

#define MAX_BATCH_SIZE 64

#endif
