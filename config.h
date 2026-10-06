#ifndef CONFIG_H
#define CONFIG_H

#define NUM_DICE_PER_PLAYER 5
#define NUM_FACES           6
#define NUM_PLAYERS         4
#define NUM_TOTAL_DICE      (NUM_DICE_PER_PLAYER * NUM_PLAYERS)

/* --- network --- */

#define NUM_MLP_LAYERS 3

#define GRU_WR_IDX NUM_MLP_LAYERS
#define GRU_UR_IDX (GRU_WR_IDX + 1)
#define GRU_WZ_IDX (GRU_UR_IDX + 1)
#define GRU_UZ_IDX (GRU_WZ_IDX + 1)
#define GRU_WH_IDX (GRU_UZ_IDX + 1)
#define GRU_UH_IDX (GRU_WH_IDX + 1)
#define GRU_BR_IDX GRU_WR_IDX
#define GRU_BZ_IDX GRU_WZ_IDX
#define GRU_BH_IDX GRU_WH_IDX
#define GRU_R_IDX  GRU_WR_IDX
#define GRU_Z_IDX  GRU_WZ_IDX
#define GRU_H_IDX  GRU_WH_IDX

#define POL_IDX (NUM_MLP_LAYERS + 6)
#define VAL_IDX (POL_IDX + 1)

#define NUM_TOTAL_LAYERS (VAL_IDX + 1)

/**
 * self  (% of hand)  NUM_FACES
 * self  (% of table) 1
 * opp   (% of table) NUM_PLAYERS
 * total (% of max)   1
 */
#define SIZE_INPUT  (NUM_FACES + 1 + NUM_PLAYERS + 1)
#define SIZE_HIDDEN 128
#define SIZE_GRU    128
/**
 * action space x5P*F
 * challenge    x1
 */
#define SIZE_POL      (NUM_TOTAL_DICE * NUM_FACES + 1)
#define CHALLENGE_IDX SIZE_POL

#define MAX_BATCH_SIZE 64

#endif
