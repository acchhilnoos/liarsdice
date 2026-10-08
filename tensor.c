#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct Tensor T;

typedef const float *const restrict read_only;
typedef float *const restrict writeable;

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

void tensor_fc(const T *in, const T *k, const T *bias, T *out)
{
  writeable o_ptr = out->buf;
  read_only i_ptr = in->buf, k_ptr = k->buf;

  if (bias)
    for (size_t ox = 0; ox < out->x; ox++) o_ptr[ox] = bias->buf[ox];
  else tensor_zero(out);

  for (size_t ix = 0; ix < in->x; ix++)
    for (size_t ox = 0; ox < out->x; ox++)
      o_ptr[ox] += i_ptr[ix] * k_ptr[ix * out->x + ox];
}

void tensor_fc_grad(T *in, T *k, T *bias, const T *out)
{
  writeable i_grd = in->grad, k_grd = k->grad;
  read_only i_ptr = in->buf, k_ptr = k->buf, o_grd = out->grad;

  for (size_t ix = 0; ix < in->x; ix++)
    for (size_t ox = 0; ox < out->x; ox++)
      i_grd[ix] += o_grd[ox] * k_ptr[ix * out->x + ox];

  for (size_t ix = 0; ix < in->x; ix++)
    for (size_t ox = 0; ox < out->x; ox++)
      k_grd[ix * out->x + ox] += o_grd[ox] * i_ptr[ix];

  if (bias) tensor_add_grad(bias, out);
}

void tensor_gru(const T *in, const T *h_prev, const T *wr, const T *ur,
                const T *br, T *r, const T *wz, const T *uz, const T *bz, T *z,
                const T *wh, const T *uh, const T *bh, T *h_temp, T *h)
{
  /**
   * r  =  sig(XWr + HpUr       + Br)
   * z  =  sig(XWz + HpUz       + Bz)
   * ht = tanh(XWh + (r @ Hp)Uh + Bh)
   * h  = z @ Hp + (1 - z) @ ht
   */
  writeable h_ptr = h->buf, ht_ptr = h_temp->buf, r_ptr = r->buf,
            z_ptr  = z->buf;
  read_only bh_ptr = bh->buf, br_ptr = br->buf, bz_ptr = bz->buf,
            hp_ptr = h_prev->buf, i_ptr = in->buf, uh_ptr = uh->buf,
            ur_ptr = ur->buf, uz_ptr = uz->buf, wh_ptr = wh->buf,
            wr_ptr = wr->buf, wz_ptr = wz->buf;

  size_t size_h = tensor_size(h);
  size_t size_i = tensor_size(in);

  // gates
  for (size_t ox = 0; ox < size_h; ox++) {
    r_ptr[ox]  = br_ptr[ox];
    z_ptr[ox]  = bz_ptr[ox];
    ht_ptr[ox] = bh_ptr[ox];
  }
  for (size_t ix = 0; ix < size_i; ix++) {
    const float i_val = i_ptr[ix];
    for (size_t ox = 0; ox < size_h; ox++) {
      r_ptr[ox]  += i_val * wr_ptr[ix * size_h + ox];
      z_ptr[ox]  += i_val * wz_ptr[ix * size_h + ox];
      ht_ptr[ox] += i_val * wh_ptr[ix * size_h + ox];
    }
  }
  for (size_t hpx = 0; hpx < size_h; hpx++) {
    const float hp_val = hp_ptr[hpx];
    for (size_t ox = 0; ox < size_h; ox++) {
      r_ptr[ox] += hp_val * ur_ptr[hpx * size_h + ox];
      z_ptr[ox] += hp_val * uz_ptr[hpx * size_h + ox];
    }
  }
  for (size_t ox = 0; ox < size_h; ox++) {
    r_ptr[ox] = 1.0f / (1.0f + expf(-r_ptr[ox]));
    z_ptr[ox] = 1.0f / (1.0f + expf(-z_ptr[ox]));
  }

  // candidate h
  for (size_t ox = 0; ox < size_h; ox++) {
    const float r_val = r_ptr[ox] * hp_ptr[ox];
    for (size_t htx = 0; htx < size_h; htx++)
      ht_ptr[htx] += r_val * uh_ptr[ox * size_h + htx];
  }
  for (size_t hx = 0; hx < size_h; hx++) ht_ptr[hx] = tanhf(ht_ptr[hx]);

  // h
  for (size_t hx = 0; hx < size_h; hx++)
    h_ptr[hx] = z_ptr[hx] * hp_ptr[hx] + (1 - z_ptr[hx]) * ht_ptr[hx];
}
void tensor_gru_grad(T *in, T *h_prev, T *wr, T *ur, T *br, T *r, T *wz, T *uz,
                     T *bz, T *z, T *wh, T *uh, T *bh, T *h_temp, const T *h)
{
  /**
   * dL/dH' = dL/dH @ (1 - Z)
   *
   * dL/dBh = (dL/dH')(dH'/dBh) = dL/dH' @ (1 - H'^2)
   * dL/dUh = (R @ Hp)^T(dL/dBh)
   * dL/dWh = X^T(dL/dBh)
   *
   * dL/dR  = (dL/dH')(dH'/dR) = (dL/dBh)Uh^T @ Hp
   * dL/dBr = (dL/dR)(dR/dBr)  = dL/dR @ R @ (1 - R)
   * dL/dUr = Hp^T(dL/dBr)
   * dL/dWr = X^T(dL/dBr)
   *
   * dL/dZ  = dL/dH @ (Hp - H')
   * dL/dBz = (dL/dZ)(dZ/dBz) = dL/dZ @ Z @ (1 - Z)
   * dL/dUz = Hp^T(dL/dBz)
   * dL/dWz = X^T(dL/dBz)
   *
   * dL/dX            = (dL/dH')(dH'/dX) + (dL/dR)(dR/dX) + (dL/dZ)(dZ/dX)
   * (dL/dH')(dH'/dX) = (dL/dBh)Wh^T
   * (dL/dR)(dR/dX)   = (dL/dBr)Wr^T
   * (dL/dZ)(dZ/dX)   = (dL/dBz)Wz^T
   *
   * dL/dHp = (dL/dH)(dH/Hp)
   *          + (dL/dH')(dH'/dHp)
   *          + (dL/dR)(dR/dHp)
   *          + (dL/dZ)(dZ/dHp)
   *        = dL/dH @ Z
   *          + (dL/dBh)Uh^T @ R
   *          + (dL/dBr)Ur^T
   *          + (dL/dBz)Uz^T
   */
  writeable bh_grd = bh->grad, br_grd = br->grad, bz_grd = bz->grad,
            hp_grd = h_prev->grad, ht_grd = h_temp->grad, i_grd = in->grad,
            r_grd = r->grad, uh_grd = uh->grad, ur_grd = ur->grad,
            uz_grd = uz->grad, wh_grd = wh->grad, wr_grd = wr->grad,
            wz_grd = wz->grad, z_grd = z->grad;
  read_only h_grd = h->grad, hp_ptr = h_prev->buf, ht_ptr = h_temp->buf,
            i_ptr = in->buf, r_ptr = r->buf, uh_ptr = uh->buf, ur_ptr = ur->buf,
            uz_ptr = uz->buf, wh_ptr = wh->buf, wr_ptr = wr->buf,
            wz_ptr = wz->buf, z_ptr = z->buf;

  size_t size_h = tensor_size(h);
  size_t size_i = tensor_size(in);

  // dL/dH'
  for (size_t i = 0; i < size_h; i++) ht_grd[i] = h_grd[i] * (1 - z_ptr[i]);
  // dL/dBh
  for (size_t i = 0; i < size_h; i++)
    bh_grd[i] = ht_grd[i] * (1 - ht_ptr[i] * ht_ptr[i]);
  // dL/dUh
  for (size_t i = 0; i < size_h; i++) {
    float hp_val = hp_ptr[i];
    float r_val  = r_ptr[i];
    for (size_t j = 0; j < size_h; j++)
      uh_grd[i * size_h + j] = r_val * hp_val * bh_grd[j];
  }
  // dL/dWh
  for (size_t i = 0; i < size_i; i++) {
    float i_val = i_ptr[i];
    for (size_t j = 0; j < size_h; j++)
      wh_grd[i * size_h + j] = i_val * bh_grd[j];
  }
  // (dL/dH')(dH'/dX)
  for (size_t i = 0; i < size_i; i++)
    for (size_t j = 0; j < size_h; j++)
      i_grd[i] += bh_grd[j] * wh_ptr[i * size_h + j];

  // dL/dR
  for (size_t i = 0; i < size_h; i++) {
    for (size_t j = 0; j < size_h; j++)
      r_grd[i] += bh_grd[j] * uh_ptr[i * size_h + j];
    r_grd[i] *= hp_ptr[i];
  }
  // dL/dBr
  for (size_t i = 0; i < size_h; i++)
    br_grd[i] = r_grd[i] * r_ptr[i] * (1 - r_ptr[i]);
  // dL/dUr
  for (size_t i = 0; i < size_h; i++) {
    float hp_val = hp_ptr[i];
    for (size_t j = 0; j < size_h; j++)
      ur_grd[i * size_h + j] = hp_val * br_grd[j];
  }
  // dL/dWr
  for (size_t i = 0; i < size_i; i++) {
    float i_val = i_ptr[i];
    for (size_t j = 0; j < size_h; j++)
      wr_grd[i * size_h + j] = i_val * br_grd[j];
  }
  // (dL/dR)(dR/dX)
  for (size_t i = 0; i < size_i; i++)
    for (size_t j = 0; j < size_h; j++)
      i_grd[i] += br_grd[j] * wr_ptr[i * size_h + j];

  // dL/dZ
  for (size_t i = 0; i < size_h; i++)
    z_grd[i] = h_grd[i] * (hp_ptr[i] - ht_ptr[i]);
  // dL/dBz
  for (size_t i = 0; i < size_h; i++)
    bz_grd[i] = z_grd[i] * z_ptr[i] * (1 - z_ptr[i]);
  // dL/dUz
  for (size_t i = 0; i < size_h; i++) {
    float hp_val = hp_ptr[i];
    for (size_t j = 0; j < size_h; j++)
      uz_grd[i * size_h + j] = hp_val * bz_grd[j];
  }
  // dL/dWz
  for (size_t i = 0; i < size_i; i++) {
    float i_val = i_ptr[i];
    for (size_t j = 0; j < size_h; j++)
      wz_grd[i * size_h + j] = i_val * bz_grd[j];
  }
  // (dL/dZ)(dZ/dX)
  for (size_t i = 0; i < size_i; i++)
    for (size_t j = 0; j < size_h; j++)
      i_grd[i] += bz_grd[j] * wz_ptr[i * size_h + j];

  // (dL/dH)(dH/dHp)
  for (size_t i = 0; i < size_h; i++) hp_grd[i] = h_grd[i] * z_ptr[i];
  // (dL/dH')(dH'/dHp)
  for (size_t i = 0; i < size_h; i++) {
    float sum = 0.0f;
    for (size_t j = 0; j < size_h; j++)
      sum += bh_grd[j] * uh_ptr[i * size_h + j];
    hp_grd[i] += sum * r_ptr[i];
  }
  // (dL/dR)(dR/dHp)
  for (size_t i = 0; i < size_h; i++)
    for (size_t j = 0; j < size_h; j++)
      hp_grd[i] += br_grd[j] * ur_ptr[i * size_h + j];
  // (dL/dZ)(dZ/dHp)
  for (size_t i = 0; i < size_h; i++)
    for (size_t j = 0; j < size_h; j++)
      hp_grd[i] += bz_grd[j] * uz_ptr[i * size_h + j];
}
