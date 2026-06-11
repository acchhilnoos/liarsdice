#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define tensor_size(t) ((t)->n * (t)->c * (t)->y * (t)->x)
#define tensor_at(t, N, Y, X, C)                                               \
  (t)->buf[(((N) * (t)->y + (Y)) * (t)->x + (X)) * (t)->c + (C)]
#define tensor_grad(t, N, Y, X, C)                                             \
  (t)->grad[(((N) * (t)->y + (Y)) * (t)->x + (X)) * (t)->c + (C)]

struct Tensor {
  float *buf, *grad;
  size_t n, y, x, c;
};

void tensor_init(struct Tensor *t, size_t n, size_t y, size_t x, size_t c);
void tensor_free(struct Tensor *t);

void tensor_reshape(struct Tensor *t, size_t n, size_t y, size_t x, size_t c);

void tensor_zero(struct Tensor *t);
void tensor_zero_grad(struct Tensor *t);

void tensor_add(const struct Tensor *src, struct Tensor *dst);
void tensor_add_grad(struct Tensor *src, const struct Tensor *dst);

void tensor_relu(struct Tensor *t);
void tensor_relu_grad(struct Tensor *t);
void tensor_tanh(struct Tensor *t);
void tensor_tanh_grad(struct Tensor *t);
void tensor_softmax(struct Tensor *t);
void tensor_softmax_grad(struct Tensor *t);

void tensor_conv(const struct Tensor *in, const struct Tensor *k,
                 const struct Tensor *bias, struct Tensor *out, size_t pad,
                 size_t dil);
void tensor_conv_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                      const struct Tensor *out, size_t pad, size_t dil);

#endif
