#include "network.h"
#include <assert.h>
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

static void test_gru(void)
{
  printf("test_gru...\n");
  size_t size_h = 4, size_i = 3;

  struct Tensor in, h_prev;
  tensor_init(&in, 1, size_i);
  tensor_init(&h_prev, 1, size_h);

  struct Tensor wr, ur, br, wz, uz, bz, wh, uh, bh;
  tensor_init(&wr, size_i, size_h);
  tensor_init(&ur, size_h, size_h);
  tensor_init(&br, 1, size_h);
  tensor_init(&wz, size_i, size_h);
  tensor_init(&uz, size_h, size_h);
  tensor_init(&bz, 1, size_h);
  tensor_init(&wh, size_i, size_h);
  tensor_init(&uh, size_h, size_h);
  tensor_init(&bh, 1, size_h);

  struct Tensor r, z, h_temp, h;
  tensor_init(&r, 1, size_h);
  tensor_init(&z, 1, size_h);
  tensor_init(&h_temp, 1, size_h);
  tensor_init(&h, 1, size_h);

  // Simple deterministic values
  for (size_t i = 0; i < size_i; i++) in.buf[i] = (i + 1) * 0.1f;
  for (size_t i = 0; i < size_h; i++) h_prev.buf[i] = (i + 1) * 0.05f;

  for (size_t i = 0; i < size_i * size_h; i++) {
    wr.buf[i] = (i + 1) * 0.01f;
    wz.buf[i] = (i + 1) * 0.01f;
    wh.buf[i] = (i + 1) * 0.01f;
  }
  for (size_t i = 0; i < size_h * size_h; i++) {
    ur.buf[i] = (i + 1) * 0.01f;
    uz.buf[i] = (i + 1) * 0.01f;
    uh.buf[i] = (i + 1) * 0.01f;
  }
  for (size_t i = 0; i < size_h; i++) {
    br.buf[i] = 0.0f;
    bz.buf[i] = 0.0f;
    bh.buf[i] = 0.0f;
  }

  tensor_gru(&in, &h_prev, &wr, &ur, &br, &r, &wz, &uz, &bz, &z, &wh, &uh, &bh,
             &h_temp, &h);

  // Just verify outputs are in valid ranges
  for (size_t i = 0; i < size_h; i++) {
    assert(r.buf[i] >= 0 && r.buf[i] <= 1);
    assert(z.buf[i] >= 0 && z.buf[i] <= 1);
    assert(h_temp.buf[i] >= -1 && h_temp.buf[i] <= 1);
    assert(h.buf[i] >= -1 && h.buf[i] <= 1);
  }
  printf("  gru forward: OK\n");

  // Test backward: set all output grads to 1
  for (size_t i = 0; i < size_h; i++) { h.grad[i] = 1.0f; }
  tensor_gru_grad(&in, &h_prev, &wr, &ur, &br, &r, &wz, &uz, &bz, &z, &wh, &uh,
                  &bh, &h_temp, &h);

  // Verify gradients are non-zero (at least some)
  int has_grad = 0;
  for (size_t i = 0; i < size_i; i++)
    if (in.grad[i] != 0) has_grad = 1;
  for (size_t i = 0; i < size_h; i++)
    if (h_prev.grad[i] != 0) has_grad = 1;
  assert(has_grad);
  printf("  gru backward: OK\n");

  tensor_free(&in);
  tensor_free(&h_prev);
  tensor_free(&wr);
  tensor_free(&ur);
  tensor_free(&br);
  tensor_free(&wz);
  tensor_free(&uz);
  tensor_free(&bz);
  tensor_free(&wh);
  tensor_free(&uh);
  tensor_free(&bh);
  tensor_free(&r);
  tensor_free(&z);
  tensor_free(&h_temp);
  tensor_free(&h);
}

static void test_network(void)
{
  printf("test_network...\n");

  struct Network *n = network_new();
  assert(n != NULL);

  struct Game *g = game_new();
  assert(g != NULL);

  struct Tensor in, hp;
  tensor_init(&in, 1, SIZE_INPUT);
  tensor_init(&hp, 1, SIZE_GRU);

  // Set deterministic input
  for (size_t i = 0; i < SIZE_INPUT; i++) in.buf[i] = (i + 1) * 0.01f;
  for (size_t i = 0; i < SIZE_GRU; i++) hp.buf[i] = (i + 1) * 0.005f;

  struct Tensor activs[NUM_TOTAL_LAYERS];
  // MLP layers: SIZE_HIDDEN
  for (size_t i = 0; i < NUM_MLP_LAYERS; i++)
    tensor_init(&activs[i], 1, SIZE_HIDDEN);
  // GRU activations: SIZE_GRU
  for (size_t i = GRU_WR_IDX; i <= GRU_H_IDX; i++)
    tensor_init(&activs[i], 1, SIZE_GRU);
  // VAL: 1, POL: SIZE_POL
  tensor_init(&activs[VAL_IDX], 1, 1);
  tensor_init(&activs[POL_IDX], 1, SIZE_POL);

  network_forward(n, &in, &hp, g, activs);

  // Check policy is valid probability distribution (masked entries are ~0)
  float  sum   = 0;
  size_t valid = 0;
  for (size_t i = 0; i < SIZE_POL; i++) {
    assert(activs[POL_IDX].buf[i] >= 0);
    if (activs[POL_IDX].buf[i] > 1e-6f) {
      sum += activs[POL_IDX].buf[i];
      valid++;
    }
  }
  printf("  policy sum (valid=%zu): %f\n", valid, sum);
  assert(eq(sum, 1.0f));
  printf("  network forward: OK\n");

  // Test backward
  struct Tensor loss_p;
  tensor_init(&loss_p, 1, SIZE_POL);
  for (size_t i = 0; i < SIZE_POL; i++) loss_p.buf[i] = (i + 1) * 0.01f;
  float loss_v = 0.5f;

  network_backward(n, &in, &hp, activs, &loss_p, loss_v);

  // Check gradients exist via input/activations
  int has_grad = 0;
  for (size_t i = 0; i < SIZE_INPUT; i++)
    if (in.grad[i] != 0) has_grad = 1;
  for (size_t i = 0; i < SIZE_GRU; i++)
    if (hp.grad[i] != 0) has_grad = 1;
  assert(has_grad);
  printf("  network backward: OK\n");

  // Test SGD step
  network_sgd(n, 0.01f, 0.9f);
  printf("  network sgd: OK\n");

  for (size_t i = 0; i < NUM_MLP_LAYERS; i++) tensor_free(&activs[i]);
  for (size_t i = GRU_WR_IDX; i <= GRU_H_IDX; i++) tensor_free(&activs[i]);
  tensor_free(&activs[VAL_IDX]);
  tensor_free(&activs[POL_IDX]);
  tensor_free(&loss_p);
  tensor_free(&in);
  tensor_free(&hp);
  network_free(n);
  free(g);
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
  test_gru();
  test_network();
  printf("\nAll tests passed.\n");
  return 0;
}
