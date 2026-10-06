#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <string.h>

struct Tensor {
  float *buf, *grad;
  size_t y, x;
};

inline size_t tensor_size(const struct Tensor *t) { return (t->y * t->x); }
inline void   tensor_zero(struct Tensor *t)
{ memset(t->buf, 0, tensor_size(t) * sizeof(*t->buf)); }
inline void tensor_zero_grad(struct Tensor *t)
{ memset(t->grad, 0, tensor_size(t) * sizeof(*t->grad)); }

int  tensor_init(struct Tensor *t, size_t y, size_t x);
void tensor_free(struct Tensor *t);

void tensor_reshape(struct Tensor *t, size_t y, size_t x);

void tensor_relu(struct Tensor *t);
void tensor_relu_grad(struct Tensor *t);
void tensor_softmax(struct Tensor *t);
void tensor_softmax_grad(struct Tensor *t);
void tensor_tanh(struct Tensor *t);
void tensor_tanh_grad(struct Tensor *t);

void tensor_add(const struct Tensor *src, struct Tensor *dst);
void tensor_add_grad(struct Tensor *src, const struct Tensor *dst);

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
