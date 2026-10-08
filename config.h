#ifndef CONFIG_H
#define CONFIG_H

/* --- game --- */

#define NUM_DICE_PER_PLAYER 5
#define NUM_FACES           6
#define NUM_PLAYERS         4
#define NUM_TOTAL_DICE      (NUM_DICE_PER_PLAYER * NUM_PLAYERS)

/* --- reward shaping --- */

#define R_CHALLENGE_GOOD  0.5f
#define R_CHALLENGE_BAD   -0.5f
#define R_CHALLENGED_GOOD -0.5f
#define R_CHALLENGED_BAD  0.5f
#define R_WIN             1.0f
#define R_LOSS            -1.0f

/* --- network --- */

#define NUM_MLP_LAYERS 3

/**
 *    -1 0  1  2  3  4  5
 * hp       ur    uz    uh
 *    in wr    wz    wh
 *       br    bz    bh
 *       r     z     ht h
 */
#define GRU_WR_IDX NUM_MLP_LAYERS   // 0
#define GRU_UR_IDX (GRU_WR_IDX + 1) // 1
#define GRU_WZ_IDX (GRU_UR_IDX + 1) // 2
#define GRU_UZ_IDX (GRU_WZ_IDX + 1) // 3
#define GRU_WH_IDX (GRU_UZ_IDX + 1) // 4
#define GRU_UH_IDX (GRU_WH_IDX + 1) // 5

#define GRU_BR_IDX GRU_WR_IDX // 0
#define GRU_BZ_IDX GRU_WZ_IDX // 2
#define GRU_BH_IDX GRU_WH_IDX // 4

#define GRU_R_IDX  GRU_WR_IDX       // 0
#define GRU_Z_IDX  GRU_WZ_IDX       // 2
#define GRU_HT_IDX GRU_WH_IDX       // 4
#define GRU_H_IDX  (GRU_HT_IDX + 1) // 5

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
#define CHALLENGE_IDX (SIZE_POL - 1)

#define MAX_BATCH_SIZE 64

#endif
