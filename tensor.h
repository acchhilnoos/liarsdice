#ifndef TENSOR_H
#define TENSOR_H

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <string.h>

struct Tensor {
  float *buf, *grad;
  size_t y, x;
};

static inline size_t tensor_size(const struct Tensor *t)
{ return (t->y * t->x); }
static inline void tensor_zero(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++) t->buf[i] = 0;
}
static inline void tensor_zero_grad(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++) t->grad[i] = 0;
}

int  tensor_init(struct Tensor *t, size_t y, size_t x);
void tensor_free(struct Tensor *t);

static inline void tensor_relu(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++)
    if (t->buf[i] < 0) t->buf[i] = 0;
}
static inline void tensor_relu_grad(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++)
    if (t->buf[i] <= 0.0f) t->grad[i] = 0.0f;
}

static inline void tensor_softmax(struct Tensor *t)
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
static inline void tensor_softmax_grad(struct Tensor *t)
{
  float dot = 0.0f;
  for (size_t i = 0; i < tensor_size(t); i++) dot += t->buf[i] * t->grad[i];
  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] = t->buf[i] * (t->grad[i] - dot);
}

static inline void tensor_tanh(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++) t->buf[i] = tanhf(t->buf[i]);
}
static inline void tensor_tanh_grad(struct Tensor *t)
{
  for (size_t i = 0; i < tensor_size(t); i++)
    t->grad[i] *= (1.0f - t->buf[i] * t->buf[i]);
}

static inline void tensor_add(const struct Tensor *src, struct Tensor *dst)
{
  for (size_t i = 0; i < tensor_size(dst); i++) dst->buf[i] += src->buf[i];
}
static inline void tensor_add_grad(struct Tensor *src, const struct Tensor *dst)
{
  for (size_t i = 0; i < tensor_size(dst); i++) src->grad[i] += dst->grad[i];
}

void tensor_fc(const struct Tensor *in, const struct Tensor *k,
               const struct Tensor *bias, struct Tensor *out);
void tensor_fc_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                    const struct Tensor *out);

void tensor_gru(const struct Tensor *in, const struct Tensor *h_prev,
                const struct Tensor *wr, const struct Tensor *ur,
                const struct Tensor *br, struct Tensor *r,
                const struct Tensor *wz, const struct Tensor *uz,
                const struct Tensor *bz, struct Tensor *z,
                const struct Tensor *wh, const struct Tensor *uh,
                const struct Tensor *bh, struct Tensor *h_temp,
                struct Tensor *h);
void tensor_gru_grad(struct Tensor *in, struct Tensor *h_prev,
                     struct Tensor *wr, struct Tensor *ur, struct Tensor *br,
                     struct Tensor *r, struct Tensor *wz, struct Tensor *uz,
                     struct Tensor *bz, struct Tensor *z, struct Tensor *wh,
                     struct Tensor *uh, struct Tensor *bh,
                     struct Tensor *h_temp, const struct Tensor *h);

#endif
