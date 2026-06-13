#include "tensor.h"
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

void tensor_init(struct Tensor *t, size_t y, size_t x) {
  t->buf  = calloc(y * x, sizeof(*t->buf));
  t->grad = calloc(y * x, sizeof(*t->grad));

  t->y = y;
  t->x = x;
}

void tensor_reshape(struct Tensor *t, size_t y, size_t x) {
  t->y = y;
  t->x = x;
}

void tensor_zero(struct Tensor *t) {
  memset(t->buf, 0, tensor_size(t) * sizeof(*t->buf));
}
void tensor_zero_grad(struct Tensor *t) {
  memset(t->grad, 0, tensor_size(t) * sizeof(*t->grad));
}

void tensor_add(const struct Tensor *src, struct Tensor *dst) {
  for (size_t idx = 0; idx < tensor_size(dst); idx++)
    dst->buf[idx] += src->buf[idx];
}

void tensor_add_grad(struct Tensor *src, const struct Tensor *dst) {
  for (size_t idx = 0; idx < tensor_size(dst); idx++)
    src->grad[idx] += dst->grad[idx];
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
  float max = -FLT_MAX;
  for (size_t i = 0; i < tensor_size(t); i++)
    if (t->buf[i] > max)
      max = t->buf[i];

  float sum = 0.0f;
  for (size_t i = 0; i < tensor_size(t); i++) {
    t->buf[i] = expf(t->buf[i] - max);
    sum += t->buf[i];
  }

  for (size_t i = 0; i < tensor_size(t); i++)
    t->buf[i] /= sum;
}

void tensor_softmax_grad(struct Tensor *t) {
  float dot = 0.0f;
  for (size_t i = 0; i < tensor_size(t); i++)
    dot += t->buf[i] * t->grad[i];

  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] = t->buf[i] * (t->grad[i] - dot);
}

void tensor_fc(const struct Tensor *in, const struct Tensor *k,
               const struct Tensor *bias, struct Tensor *out) {
  float *restrict out_ptr = out->buf;
  float *restrict in_ptr  = in->buf;
  float *restrict k_ptr   = k->buf;

  for (size_t ox = 0; ox < out->x; ox++)
    out_ptr[ox] = bias->buf[ox];

  for (size_t ix = 0; ix < in->x; ix++, in_ptr++)
    for (size_t ox = 0; ox < out->x; ox++)
      out_ptr[ox] += *in_ptr * *(k_ptr++);
}

void tensor_fc_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                    const struct Tensor *out) {
  float *restrict out_grad_ptr = out->grad;
  float *restrict in_grad_ptr  = in->grad;
  float *restrict k_ptr        = k->buf;

  for (size_t ix = 0; ix < in->x; ix++, in_grad_ptr++)
    for (size_t ox = 0; ox < out->x; ox++)
      *in_grad_ptr += out_grad_ptr[ox] * *(k_ptr++);

  float *restrict in_ptr     = in->buf;
  float *restrict k_grad_ptr = k->grad;

  for (size_t ix = 0; ix < in->x; ix++, in_ptr++)
    for (size_t ox = 0; ox < out->x; ox++)
      (*k_grad_ptr++) += out_grad_ptr[ox] * *in_ptr;

  if (bias) {
    tensor_add_grad(bias, out);
  }
}
