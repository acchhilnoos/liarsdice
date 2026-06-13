#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>

#define tensor_size(t) ((t)->y * (t)->x)

struct Tensor {
  float *buf, *grad;
  size_t y, x;
};

void tensor_init(struct Tensor *t, size_t y, size_t x);
void tensor_free(struct Tensor *t);

void tensor_reshape(struct Tensor *t, size_t y, size_t x);

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

void tensor_fc(const struct Tensor *in, const struct Tensor *k,
               const struct Tensor *bias, struct Tensor *out);
void tensor_fc_grad(struct Tensor *in, struct Tensor *k, struct Tensor *bias,
                    const struct Tensor *out);

#endif
