#include "tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define EPS 1e-5

static int eq(float a, float b) { return fabsf(a - b) < EPS; }

static void check_buf(const char *name, const struct Tensor *t,
                      const float *expected, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (!eq(t->buf[i], expected[i])) {
      fprintf(stderr, "%s[%zu]: got %f, expected %f\n", name, i, t->buf[i],
              expected[i]);
      exit(1);
    }
  }
  printf("  %s: OK\n", name);
}

static void check_grad(const char *name, const struct Tensor *t,
                       const float *expected, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    if (!eq(t->grad[i], expected[i])) {
      fprintf(stderr, "%s grad[%zu]: got %f, expected %f\n", name, i,
              t->grad[i], expected[i]);
      exit(1);
    }
  }
  printf("  %s grad: OK\n", name);
}

static void test_init_free(void)
{
  printf("test_init_free...\n");
  struct Tensor t;
  tensor_init(&t, 2, 3);
  assert(t.y == 2 && t.x == 3);
  assert(t.buf && t.grad);
  for (size_t i = 0; i < 6; i++) assert(t.buf[i] == 0 && t.grad[i] == 0);
  tensor_free(&t);
  printf("  OK\n");
}

static void test_zero(void)
{
  printf("test_zero...\n");
  struct Tensor t;
  tensor_init(&t, 2, 3);
  t.buf[0] = 1;
  t.buf[5] = 2;
  tensor_zero(&t);
  for (size_t i = 0; i < 6; i++) assert(t.buf[i] == 0);
  tensor_free(&t);
  printf("  OK\n");
}

static void test_add(void)
{
  printf("test_add...\n");
  struct Tensor a, b;
  tensor_init(&a, 1, 3);
  tensor_init(&b, 1, 3);
  float av[3] = {1, 2, 3};
  float bv[3] = {4, 5, 6};
  for (int i = 0; i < 3; i++) a.buf[i] = av[i], b.buf[i] = bv[i];
  tensor_add(&a, &b);
  float exp[3] = {5, 7, 9};
  check_buf("add", &b, exp, 3);
  tensor_free(&a);
  tensor_free(&b);
}

static void test_relu(void)
{
  printf("test_relu...\n");
  struct Tensor t;
  tensor_init(&t, 1, 4);
  float v[4] = {-1, 0, 2, -3};
  for (int i = 0; i < 4; i++) t.buf[i] = v[i];
  tensor_relu(&t);
  float exp[4] = {0, 0, 2, 0};
  check_buf("relu", &t, exp, 4);

  for (int i = 0; i < 4; i++) t.grad[i] = 1;
  tensor_relu_grad(&t);
  float gexp[4] = {0, 0, 1, 0};
  check_grad("relu", &t, gexp, 4);
  tensor_free(&t);
}

static void test_tanh(void)
{
  printf("test_tanh...\n");
  struct Tensor t;
  tensor_init(&t, 1, 3);
  float v[3] = {0, 1, -1};
  for (int i = 0; i < 3; i++) t.buf[i] = v[i];
  tensor_tanh(&t);
  float exp[3] = {tanhf(0), tanhf(1), tanhf(-1)};
  check_buf("tanh", &t, exp, 3);

  for (int i = 0; i < 3; i++) t.grad[i] = 1;
  tensor_tanh_grad(&t);
  float gexp[3];
  for (int i = 0; i < 3; i++) gexp[i] = 1 - t.buf[i] * t.buf[i];
  check_grad("tanh", &t, gexp, 3);
  tensor_free(&t);
}

static void test_softmax(void)
{
  printf("test_softmax...\n");
  struct Tensor t;
  tensor_init(&t, 1, 3);
  float v[3] = {1, 2, 3};
  for (int i = 0; i < 3; i++) t.buf[i] = v[i];
  tensor_softmax(&t);
  float sum = 0;
  for (int i = 0; i < 3; i++) sum += t.buf[i];
  assert(eq(sum, 1.0f));
  float max = v[0];
  for (int i = 1; i < 3; i++)
    if (v[i] > max) max = v[i];
  float expv[3] = {expf(v[0] - max), expf(v[1] - max), expf(v[2] - max)};
  float esum    = expv[0] + expv[1] + expv[2];
  float exp[3]  = {expv[0] / esum, expv[1] / esum, expv[2] / esum};
  check_buf("softmax", &t, exp, 3);

  for (int i = 0; i < 3; i++) t.grad[i] = 1;
  tensor_softmax_grad(&t);
  float dot = 0;
  for (int i = 0; i < 3; i++) dot += t.buf[i] * 1;
  float gexp[3];
  for (int i = 0; i < 3; i++) gexp[i] = t.buf[i] * (1 - dot);
  check_grad("softmax", &t, gexp, 3);
  tensor_free(&t);
}

static void test_fc(void)
{
  printf("test_fc...\n");
  struct Tensor in, k, bias, out;
  tensor_init(&in, 1, 2);
  tensor_init(&k, 2, 3);
  tensor_init(&bias, 1, 3);
  tensor_init(&out, 1, 3);

  float inv[2]   = {1, 2};
  float kv[6]    = {1, 2, 3, 4, 5, 6};
  float biasv[3] = {0.1, 0.2, 0.3};
  for (int i = 0; i < 2; i++) in.buf[i] = inv[i];
  for (int i = 0; i < 6; i++) k.buf[i] = kv[i];
  for (int i = 0; i < 3; i++) bias.buf[i] = biasv[i];

  tensor_fc(&in, &k, &bias, &out);

  float exp[3] = {biasv[0] + inv[0] * kv[0] + inv[1] * kv[3],
                  biasv[1] + inv[0] * kv[1] + inv[1] * kv[4],
                  biasv[2] + inv[0] * kv[2] + inv[1] * kv[5]};
  check_buf("fc", &out, exp, 3);

  for (int i = 0; i < 3; i++) out.grad[i] = 1;
  tensor_fc_grad(&in, &k, &bias, &out);

  float in_grad_exp[2] = {kv[0] + kv[1] + kv[2], kv[3] + kv[4] + kv[5]};
  check_grad("fc in", &in, in_grad_exp, 2);

  float k_grad_exp[6] = {inv[0] * 1, inv[0] * 1, inv[0] * 1,
                         inv[1] * 1, inv[1] * 1, inv[1] * 1};
  check_grad("fc k", &k, k_grad_exp, 6);

  float bias_grad_exp[3] = {1, 1, 1};
  check_grad("fc bias", &bias, bias_grad_exp, 3);

  tensor_free(&in);
  tensor_free(&k);
  tensor_free(&bias);
  tensor_free(&out);
}

int main(void)
{
  test_init_free();
  test_zero();
  test_add();
  test_relu();
  test_tanh();
  test_softmax();
  test_fc();
  printf("\nAll tests passed.\n");
  return 0;
}
