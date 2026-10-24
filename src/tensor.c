#include "tensor.h"
#include <immintrin.h>
#include <arm_neon.h>
#include <stdlib.h>

#define _do_8(X)                                                               \
  X(0);                                                                        \
  X(1);                                                                        \
  X(2);                                                                        \
  X(3);                                                                        \
  X(4);                                                                        \
  X(5);                                                                        \
  X(6);                                                                        \
  X(7)

typedef struct Tensor T;

typedef const float *restrict read_only;
typedef float *restrict writeable;

int tensor_init(T *t, size_t y, size_t x)
{
  t->buf = calloc(y * x, sizeof(*t->buf));
  if (!t->buf) goto fail_b;
  t->grad = calloc(y * x, sizeof(*t->grad));
  if (!t->grad) goto fail_g;

  t->y = y;
  t->x = x;

  return 0;

fail_g:
  free(t->buf);
fail_b:
  return 1;
}

void tensor_free(T *t)
{
  free(t->buf);
  free(t->grad);
}

static void _matmul_add(read_only A, read_only B, writeable C, size_t b,
                        size_t m, size_t n)
{
#define stride      32
#define load_c(_x)  float32x4_t c_vec##_x = vld1q_f32(&C[i * n + k + _x * 4])
#define load_b(_x)  float32x4_t b_vec##_x = vld1q_f32(&B[j * n + k + _x * 4])
#define fma(_x)     c_vec##_x = vfmaq_f32(c_vec##_x, a_vec, b_vec##_x)
#define store_c(_x) vst1q_f32(&C[i * n + k + _x * 4], c_vec##_x)
  for (size_t i = 0; i < b; i++) {
    for (size_t k = 0; k + stride <= n; k += stride) {
      _do_8(load_c);

      for (size_t j = 0; j < m; j++) {
        float32x4_t a_vec = vdupq_n_f32(A[i * m + j]);

        _do_8(load_b);
        _do_8(fma);
      }

      _do_8(store_c);
    }

    if (n & (stride - 1))
      for (size_t j = 0; j < m; j++)
        for (size_t kk = (n & ~(stride - 1)); kk < n; kk++)
          C[i * n + kk] += A[i * m + j] * B[j * n + kk];
  }

#undef store_c
#undef fma
#undef load_b
#undef load_c
#undef stride
}

static void _matmul_add_bt(read_only A, read_only B, writeable C, size_t b,
                           size_t m, size_t n)
{
#define stride      32
#define load_s(_x)  float32x4_t vec_s##_x = vdupq_n_f32(0.0f)
#define load_og(_x) float32x4_t vec_a##_x = vld1q_f32(A + i * n + k + _x * 4)
#define load_b(_x)  float32x4_t vec_b##_x = vld1q_f32(B + j * n + k + _x * 4)
#define fma(_x)     vec_s##_x = vfmaq_f32(vec_s##_x, vec_a##_x, vec_b##_x)
  for (size_t i = 0; i < b; i++) {
    for (size_t j = 0; j < m; j++) {
      _do_8(load_s);

      for (size_t k = 0; k + stride <= n; k += stride) {
        _do_8(load_og);
        _do_8(load_b);
        _do_8(fma);
      }

      float32x4_t vec_s = vaddq_f32(
          vaddq_f32(vaddq_f32(vec_s0, vec_s1), vaddq_f32(vec_s2, vec_s3)),
          vaddq_f32(vaddq_f32(vec_s4, vec_s5), vaddq_f32(vec_s6, vec_s7)));
      float sum = vaddvq_f32(vec_s);

      if (n & (stride - 1))
        for (size_t k = n & ~(stride - 1); k < n; k++)
          sum += A[i * b + k] * B[j * n + k];

      C[i * m + j] += sum;
    }
  }

#undef fma
#undef load_b
#undef load_og
#undef load_s
#undef stride
}

void tensor_fc(const T *in, const T *w, const T *bias, T *out)
{
  if (bias)
    for (size_t i = 0; i < out->y; i++)
      memcpy(out->buf + i * out->x, bias->buf, bias->x * sizeof(*bias->buf));
  else tensor_zero(out);

  _matmul_add(in->buf, w->buf, out->buf, 1, w->y, w->x);
}
void tensor_fc_grad(T *in, T *w, T *bias, const T *out)
{
  _matmul_add_bt(out->grad, w->buf, in->grad, 1, w->y, w->x);
  _matmul_add(in->buf, out->grad, w->grad, in->x, 1, out->x);
  if (bias) tensor_add_grad(bias, out);
}

void tensor_gru(const T *in, const T *h_prev, const T *wr, const T *ur,
                const T *br, T *r, const T *wz, const T *uz, const T *bz, T *z,
                const T *wh, const T *uh, const T *bh, float *scratch,
                T *h_temp, T *h)
{
  size_t    size_h = tensor_size(h);
  writeable H = h->buf, HT = h_temp->buf, R = r->buf, Z = z->buf;
  read_only BH = bh->buf, BR = br->buf, BZ = bz->buf, HP = h_prev->buf,
            X = in->buf, UH = uh->buf, UR = ur->buf, UZ = uz->buf, WH = wh->buf,
            WR = wr->buf, WZ = wz->buf;

  for (size_t i = 0; i < size_h; i++) {
    R[i]  = BR[i];
    HT[i] = BH[i];
    Z[i]  = BZ[i];
  }

  // --- r ---
  // XWr
  _matmul_add(X, WR, R, 1, wr->y, wr->x);
  // HpUr
  _matmul_add(HP, UR, R, 1, ur->y, ur->x);
  // sigmoid
  for (size_t i = 0; i < size_h; i++) R[i] = 1.0f / (1.0f + expf(-R[i]));

  // --- z ---
  // XWz
  _matmul_add(X, WZ, Z, 1, wz->y, wz->x);
  // HpUZ
  _matmul_add(HP, UZ, Z, 1, uz->y, uz->x);
  // sigmoid
  for (size_t i = 0; i < size_h; i++) Z[i] = 1.0f / (1.0f + expf(-Z[i]));

  // --- Ht ---
  // (r @ Hp)
  for (size_t i = 0; i < size_h; i++) scratch[i] = R[i] * HP[i];
  // XWh
  _matmul_add(X, WH, HT, 1, wh->y, wh->x);
  // (r @ Hp)Uh
  _matmul_add(scratch, UH, HT, 1, uh->y, uh->x);
  // tanh
  for (size_t i = 0; i < size_h; i++) HT[i] = tanhf(HT[i]);

  // --- h ---
  // z @ Hp + (1 - z) @ Ht
  for (size_t i = 0; i < size_h; i++) H[i] = Z[i] * HP[i] + (1 - Z[i]) * HT[i];
}

void tensor_gru_grad(T *in, T *h_prev, T *wr, T *ur, T *br, T *r, T *wz, T *uz,
                     T *bz, T *z, T *wh, T *uh, T *bh, T *h_temp, const T *h)
{
  writeable dL_dBH = bh->grad, dL_dBR = br->grad, dL_dBZ = bz->grad,
            dL_dHP = h_prev->grad, dL_dHT = h_temp->grad, dL_dI = in->grad,
            dL_dR = r->grad, dL_dUH = uh->grad, dL_dUR = ur->grad,
            dL_dUZ = uz->grad, dL_dWH = wh->grad, dL_dWR = wr->grad,
            dL_dWZ = wz->grad, dL_dZ = z->grad;
  read_only dL_dH = h->grad, HP = h_prev->buf, HT = h_temp->buf, I = in->buf,
            R = r->buf, UH = uh->buf, UR = ur->buf, UZ = uz->buf, WH = wh->buf,
            WR = wr->buf, WZ = wz->buf, Z = z->buf;

  size_t size_h = tensor_size(h);
  size_t size_i = tensor_size(in);

  // dLdHt = dLdH  @ (1 - Z)
  for (size_t i = 0; i < size_h; i++) dL_dHT[i] = dL_dH[i] * (1 - Z[i]);

  // dLdBh = dLdHt @ (1 - Ht^2)
  for (size_t i = 0; i < size_h; i++)
    dL_dBH[i] += dL_dHT[i] * (1 - HT[i] * HT[i]);
  // dLdUh = (R @ Hp)^T(dLdBh)
  for (size_t i = 0; i < size_h; i++) {
    float hp_val = HP[i];
    float r_val  = R[i];
    for (size_t j = 0; j < size_h; j++)
      dL_dUH[i * size_h + j] += r_val * hp_val * dL_dBH[j];
  }
  // dLdWh = X^T(dLdBh)
  _matmul_add(I, dL_dBH, dL_dWH, size_i, 1, size_h);

  // dLdR = (dLdBh)Uh^T @ Hp
  tensor_zero_grad(r);
  _matmul_add(dL_dBH, UH, dL_dR, 1, size_h, size_h);
  for (size_t i = 0; i < size_h; i++) dL_dR[i] *= HP[i];

  // dLdBr = dLdR @ R @ (1 - R)
  for (size_t i = 0; i < size_h; i++) dL_dBR[i] += dL_dR[i] * R[i] * (1 - R[i]);
  // dLdUr = Hp^T(dLdBr)
  _matmul_add(HP, dL_dBR, dL_dUR, size_h, 1, size_h);
  // dLdWr = X^T(dLdBr)
  _matmul_add(I, dL_dBR, dL_dWR, size_i, 1, size_h);

  // dLdZ = dLdH @ (Hp - Ht)
  for (size_t i = 0; i < size_h; i++) dL_dZ[i] += dL_dH[i] * (HP[i] - HT[i]);

  // dLdBz = dLdZ @ Z @ (1 - Z)
  for (size_t i = 0; i < size_h; i++) dL_dBZ[i] += dL_dZ[i] * Z[i] * (1 - Z[i]);
  // dLdUz = Hp^T(dLdBz)
  _matmul_add(HP, dL_dBZ, dL_dUZ, size_h, 1, size_h);
  // dLdWz = X^T(dLdBz)
  _matmul_add(I, dL_dBZ, dL_dWZ, size_i, 1, size_h);

  // --- dLdX = (dLdHt)(dHtdX) + (dLdR)(dRdX) + (dLdZ)(dZdX) ---

  // (dLdHt)(dHtdX) = (dLdBh)Wh^T
  _matmul_add_bt(dL_dBH, WH, dL_dI, 1, size_i, size_h);
  // (dLdR)(dRdX) = (dLdBr)Wr^T
  _matmul_add_bt(dL_dBR, WR, dL_dI, 1, size_i, size_h);
  // (dLdZ)(dZdX) = (dLdBz)Wz^T
  _matmul_add_bt(dL_dBZ, WZ, dL_dI, 1, size_i, size_h);

  // --- dL/dHp =
  //       (dLdH)(dHdHp) + (dLdHt)(dHtdHp) + (dLdR)(dRdHp) + (dLdZ)(dZdHp) ---

  // (dLdHt)(dHtdHp) = (dLdBh)Uh^T @ R
  _matmul_add_bt(dL_dBH, UH, dL_dHP, 1, size_h, size_h);
  for (size_t i = 0; i < size_h; i++) dL_dHP[i] *= R[i];
  // (dLdH)(dHdHp) = dLdH @ Z
  for (size_t i = 0; i < size_h; i++) dL_dHP[i] += dL_dH[i] * Z[i];
  // (dLdR)(dRdHp) = (dLdBr)Ur^T
  _matmul_add_bt(dL_dBR, UR, dL_dHP, 1, size_h, size_h);
  // (dLdZ)(dZdHp) = (dLdBz)Uz^T
  _matmul_add_bt(dL_dBZ, UZ, dL_dHP, 1, size_h, size_h);
}
