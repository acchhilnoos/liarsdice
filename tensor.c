#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void tensor_init(struct Tensor *t, size_t n, size_t y, size_t x, size_t c) {
  t->buf  = calloc(n * y * x * c, sizeof(*t->buf));
  t->grad = calloc(n * y * x * c, sizeof(*t->grad));

  t->n = n;
  t->y = y;
  t->x = x;
  t->c = c;
}

void tensor_reshape(struct Tensor *t, size_t n, size_t y, size_t x, size_t c) {
  t->n = n;
  t->y = y;
  t->x = x;
  t->c = c;
}

void tensor_zero(struct Tensor *t) {
  memset(t->buf, 0, tensor_size(t) * sizeof(*t->buf));
}
void tensor_zero_grad(struct Tensor *t) {
  memset(t->grad, 0, tensor_size(t) * sizeof(*t->grad));
}

void tensor_add(const struct Tensor *src, struct Tensor *dst) {
  size_t oc         = src->c;
  size_t plane_size = dst->y * dst->x;

  if (src->n == 1 && src->y == 1 && src->x == 1)
    for (size_t b = 0; b < dst->n; b++)
      for (size_t s = 0; s < dst->y * dst->x; s++)
        for (size_t c = 0; c < oc; c++)
          dst->buf[(b * plane_size + s) * oc + c] += src->buf[c];
  else
    for (size_t i = 0; i < tensor_size(dst); i++)
      dst->buf[i] += src->buf[i];
}

void tensor_add_grad(struct Tensor *src, const struct Tensor *dst) {
  size_t oc         = src->c;
  size_t plane_size = dst->y * dst->x;

  if (src->n == 1 && src->y == 1 && src->x == 1)
    for (size_t b = 0; b < dst->n; b++)
      for (size_t s = 0; s < plane_size; s++)
        for (size_t c = 0; c < oc; c++)
          src->grad[c] += dst->grad[(b * plane_size + s) * oc + c];
  else
    for (size_t i = 0; i < tensor_size(dst); i++)
      src->grad[i] += dst->grad[i];
}

void tensor_free(struct Tensor *t) {
  free(t->buf);
  free(t->grad);
}

void tensor_relu(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++) {
    if (t->buf[i] < 0)
      t->buf[i] = 0;
  }
}
void tensor_relu_grad(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++) {
    if (t->buf[i] <= 0.0f)
      t->grad[i] = 0.0f;
  }
}

void tensor_tanh(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++)
    t->buf[i] = tanhf(t->buf[i]);
}
void tensor_tanh_grad(struct Tensor *t) {
  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] *= (1.0f - t->buf[i] * t->buf[i]);
}

void tensor_softmax(struct Tensor *t) {
  size_t stride = t->y * t->x * t->c;
  for (size_t b = 0; b < t->n; b++) {
    float *buf = t->buf + b * stride;
    float  max = -FLT_MAX;
    for (size_t i = 0; i < stride; i++)
      if (buf[i] > max)
        max = buf[i];

    float sum = 0.0f;
    for (size_t i = 0; i < stride; i++) {
      buf[i] = expf(buf[i] - max);
      sum += buf[i];
    }

    for (size_t i = 0; i < stride; i++)
      buf[i] /= sum + 1e-8f;
  }
}

void tensor_softmax_grad(struct Tensor *t) {
  size_t stride = t->y * t->x * t->c;
  for (size_t b = 0; b < t->n; b++) {
    float *s  = t->buf  + b * stride;
    float *ds = t->grad + b * stride;

    float dot = 0.0f;
    for (size_t i = 0; i < stride; i++)
      dot += s[i] * ds[i];

    for (size_t i = 0; i < stride; i++)
      ds[i] = s[i] * (ds[i] - dot);
  }
}

void tensor_conv(const struct Tensor *in, const struct Tensor *k, const struct Tensor *bias,
                 struct Tensor *out, size_t pad, size_t dil) {
  /* in:    [b] [y] [x][ic]
   * k:    [ky][kx][ic][oc]
   * bias:  [1] [1] [1][oc]
   * out:   [b] [y] [x][oc]
   */
  // for (b) for (y) for (x)
  for (size_t b = 0; b < in->n; b++) {
    for (size_t y = 0; y < out->y; y++) {
      for (size_t x = 0; x < out->x; x++) {
        // init out[b][y][x] with bias
        float *out_vals = &tensor_at(out, b, y, x, 0);
        for (size_t oc = 0; oc < out->c; oc++)
          out_vals[oc] = bias ? tensor_at(bias, 0, 0, 0, oc) : 0.0f;

        // for (ky) for (kx) for (ic)
        for (size_t ky = 0; ky < k->n; ky++) {
          int in_y = (int)y + (int)ky * dil - (int)pad;
          if (in_y < 0 || in_y >= (int)in->y)
            continue;

          for (size_t kx = 0; kx < k->y; kx++) {
            int in_x = (int)x + (int)kx * dil - (int)pad;
            if (in_x < 0 || in_x >= (int)in->x)
              continue;

            float *in_ptr = &tensor_at(in, b, in_y, in_x, 0);
            float *k_ptr  = &tensor_at(k, ky, kx, 0, 0);

            for (size_t ic = 0; ic < in->c; ic++) {
              // in[b][y+ky][x+kx][ic]
              float in_val = in_ptr[ic];
              // for (oc)
              for (size_t oc = 0; oc < out->c; oc++) {
                // out[b][y][x][oc] +=
                //   in[b][y+ky][x+kx][ic] * k[ky][kx][ic][oc]
                out_vals[oc] += in_val * (*k_ptr++);
              }
            }
          }
        }
      }
    }
  }
}

void tensor_conv_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                      const struct Tensor *out, size_t pad, size_t dil) {
  /* in:    [b] [y] [x][ic]
   * k:    [ky][kx][ic][oc]
   * bias:  [1] [1] [1][oc]
   * out:   [b] [y] [x][oc]
   */
  // for (b) for (y) for (x)
  for (size_t b = 0; b < in->n; b++) {
    for (size_t y = 0; y < out->y; y++) {
      for (size_t x = 0; x < out->x; x++) {
        float *out_grad_ptr = &tensor_grad(out, b, y, x, 0);

        // for (ky) for (kx) for (ic) for (oc)
        for (size_t ky = 0; ky < k->n; ky++) {
          int in_y = (int)y + (int)ky * dil - (int)pad;
          if (in_y < 0 || in_y >= (int)in->y)
            continue;

          for (size_t kx = 0; kx < k->y; kx++) {
            int in_x = (int)x + (int)kx * dil - (int)pad;
            if (in_x < 0 || in_x >= (int)in->x)
              continue;

            float *in_grad_ptr = &tensor_grad(in, b, in_y, in_x, 0);
            float *k_ptr       = &tensor_at(k, ky, kx, 0, 0);

            for (size_t ic = 0; ic < in->c; ic++) {
              // in_grad[b][y][x][ic] +=
              //   out_grad[b][y-ky][x-kx][oc] * k[ky][kx][ic][oc]
              float sum = 0.0f;
              for (size_t oc = 0; oc < out->c; oc++) {
                sum += out_grad_ptr[oc] * (*k_ptr++);
              }
              in_grad_ptr[ic] += sum;
            }
          }
        }
      }
    }
  }

  // for (ky) for (kx) for (b) for (y) for (x) for (ic)
  for (size_t ky = 0; ky < k->n; ky++) {
    for (size_t kx = 0; kx < k->y; kx++) {
      for (size_t b = 0; b < in->n; b++) {
        for (size_t y = 0; y < out->y; y++) {
          int in_y = (int)y + (int)ky * dil - (int)pad;
          if (in_y < 0 || in_y >= (int)in->y)
            continue;

          for (size_t x = 0; x < out->x; x++) {
            int in_x = (int)x + (int)kx * dil - (int)pad;
            if (in_x < 0 || in_x >= (int)in->x)
              continue;

            float *in_ptr       = &tensor_at(in, b, in_y, in_x, 0);
            float *k_grad_ptr   = &tensor_grad(k, ky, kx, 0, 0);
            float *out_grad_ptr = &tensor_grad(out, b, y, x, 0);

            for (size_t ic = 0; ic < in->c; ic++) {
              // in[b][y+ky][x+kx][ic]
              float in_val = in_ptr[ic];
              // for (oc)
              for (size_t oc = 0; oc < out->c; oc++) {
                // k_grad[ky][kx][ic][oc] +=
                //   out_grad[b][y][x][oc] * in[b][y+ky][x+kx][ic]
                (*k_grad_ptr++) += out_grad_ptr[oc] * in_val;
              }
            }
          }
        }
      }
    }
  }

  if (bias) {
    tensor_add_grad(bias, out);
  }
}
