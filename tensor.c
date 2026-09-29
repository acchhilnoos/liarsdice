#include "tensor.h"
#include <float.h>
#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

extern inline size_t tensor_size(const struct Tensor *t);

int tensor_init(struct Tensor *t, size_t y, size_t x)
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

void tensor_reshape(struct Tensor *t, size_t y, size_t x)
{
  t->y = y;
  t->x = x;
}

void tensor_zero(struct Tensor *t)
{ memset(t->buf, 0, tensor_size(t) * sizeof(*t->buf)); }
void tensor_zero_grad(struct Tensor *t)
{ memset(t->grad, 0, tensor_size(t) * sizeof(*t->grad)); }

void tensor_add(const struct Tensor *src, struct Tensor *dst)
{
  for (size_t idx = 0; idx < tensor_size(dst); idx++)
    dst->buf[idx] += src->buf[idx];
}

void tensor_add_grad(struct Tensor *src, const struct Tensor *dst)
{
  for (size_t idx = 0; idx < tensor_size(dst); idx++)
    src->grad[idx] += dst->grad[idx];
}

void tensor_free(struct Tensor *t)
{
  free(t->buf);
  free(t->grad);
}

void tensor_relu(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++) {
    if (t->buf[i] < 0) t->buf[i] = 0;
  }
}
void tensor_relu_grad(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++) {
    if (t->buf[i] <= 0.0f) t->grad[i] = 0.0f;
  }
}

void tensor_tanh(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++) t->buf[i] = tanhf(t->buf[i]);
}
void tensor_tanh_grad(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] *= (1.0f - t->buf[i] * t->buf[i]);
}

void tensor_softmax(struct Tensor *t)
{
  float max = -FLT_MAX;
  for (size_t i = 0; i < tensor_size(t); i++)
    if (t->buf[i] > max) max = t->buf[i];

  float sum = 0.0f;
  for (size_t i = 0; i < tensor_size(t); i++) {
    t->buf[i]  = expf(t->buf[i] - max);
    sum       += t->buf[i];
  }

  for (size_t i = 0; i < tensor_size(t); i++) t->buf[i] /= sum;
}

void tensor_softmax_grad(struct Tensor *t)
{
  float dot = 0.0f;
  for (size_t i = 0; i < tensor_size(t); i++) dot += t->buf[i] * t->grad[i];

  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] = t->buf[i] * (t->grad[i] - dot);
}

void tensor_fc(const struct Tensor *in, const struct Tensor *k,
               const struct Tensor *bias, struct Tensor *out)
{
  float *const restrict o_ptr       = out->buf;
  const float *const restrict i_ptr = in->buf;
  const float *const restrict k_ptr = k->buf;

  for (size_t ox = 0; ox < out->x; ox++) o_ptr[ox] = bias->buf[ox];

  for (size_t ix = 0; ix < in->x; ix++) {
    size_t ox    = 0;
    __m256 i_vec = _mm256_set1_ps(i_ptr[ix]);
    for (; ox + 8 <= out->x; ox += 8) {
      __m256 o_vec = _mm256_loadu_ps(o_ptr + ox);
      __m256 k_vec = _mm256_loadu_ps(k_ptr + ix * out->x + ox);
      o_vec        = _mm256_fmadd_ps(i_vec, k_vec, o_vec);
      _mm256_storeu_ps(o_ptr + ox, o_vec);
    }
    for (; ox < out->x; ox++) o_ptr[ox] += i_ptr[ix] * k_ptr[ix * out->x + ox];
  }
}

void tensor_fc_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                    const struct Tensor *out)
{
  const float *const restrict o_grd = out->grad;
  const float *const restrict i_ptr = in->buf;
  float *const restrict i_grd       = in->grad;
  const float *const restrict k_ptr = k->buf;
  float *const restrict k_grd       = k->grad;

  for (size_t ix = 0; ix < in->x; ix++) {
    __m256 i_vec   = _mm256_set1_ps(i_ptr[ix]);
    __m256 sum_vec = _mm256_setzero_ps();
    size_t ox      = 0;
    for (; ox + 8 <= out->x; ox += 8) {
      __m256 k_vec  = _mm256_loadu_ps(k_ptr + ix * out->x + ox);
      __m256 kg_vec = _mm256_loadu_ps(k_grd + ix * out->x + ox);
      __m256 og_vec = _mm256_loadu_ps(o_grd + ox);

      sum_vec = _mm256_fmadd_ps(og_vec, k_vec, sum_vec);
      kg_vec  = _mm256_fmadd_ps(og_vec, i_vec, kg_vec);
      _mm256_storeu_ps(k_grd + ix * out->x + ox, kg_vec);
    }
    float sum = 0.0f;
    float sum_ptr[8];
    _mm256_storeu_ps(sum_ptr, sum_vec);
    for (size_t i = 0; i < 8; i++) sum += sum_ptr[i];

    for (; ox < out->x; ox++) {
      sum                     += o_grd[ox] * k_ptr[ix * out->x + ox];
      k_grd[ix * out->x + ox] += o_grd[ox] * i_ptr[ix];
    }
    i_grd[ix] += sum;
  }
  tensor_add_grad(bias, out);
}
