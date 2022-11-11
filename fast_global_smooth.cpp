#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

int LutAddress(int i, int j) { return (i << 8) + j; }

const std::array<float, 256 * 256> &GetExpLut(float sigma) {
  static std::array<float, 256 * 256> kLut;
  static bool kInited = false;

  if (kInited) {
    return kLut;
  }

  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 256; j++) {
      float value = std::exp(-std::abs(i - j) / sigma);
      kLut[LutAddress(i, j)] = value;
    }
  }

  kInited = true;

  return kLut;
}

void NeighborSimilarity(const uint8_t *g, const int n, float *w, float sigma) {
  // const auto &lut = GetExpLut(sigma);

  for (int x = 1; x < n; x++) {
    w[x] = std::exp(-std::abs(g[x] - g[x - 1]) / sigma);
    // w[x] = lut[LutAddress(g[x], g[x - 1])];
  }
}

void FastGlobalSmooth(const uint8_t *p_data, const int n, uint8_t *u,
                      float lambda, float sigma) {
  std::vector<float> w(n, 0), c_wave(n, 0), f_wave(n, 0);
  NeighborSimilarity(p_data, n, w.data(), sigma);

  const auto *g = p_data, *f = p_data;

  /**
   * compute c_wave and f_wave, formula(8)
   * factor = b-a*c_wave[x-1]
   * c_wave[x] = c/factor
   * f_wave[x] = (f[x]-a * f_wave[x-1]) / factor
   */

  // temp variables
  float a = 0, b = 0, c = 0;

  // special case x=0
  b = 1 + lambda * w[1], c = -lambda * w[1];
  c_wave[0] = c / b;
  f_wave[0] = f[0] / b;

  for (int x = 1; x < n; x++) {
    a = -lambda * w[x];
    b = 1 + lambda * (w[x] + w[x + 1]);
    c = -lambda * w[x + 1];

    const float factor = b - a * c_wave[x - 1];

    c_wave[x] = c / factor;
    f_wave[x] = (f[x] - a * f_wave[x - 1]) / factor;
  }

  // special case x=n-1
  a = -lambda * w[n - 1], b = 1 + lambda * w[n - 1];
  const float factor = b - a * c_wave[n - 2];
  c_wave[n - 1] = 0;
  f_wave[n - 1] = (f[n - 1] - a * f_wave[n - 2]) / factor;

  // then compute output, formula(9)
  u[n - 1] = f_wave[n - 1];
  for (int x = n - 2; x >= 0; x--) {
    u[x] = std::clamp<uint8_t>(f_wave[x] - c_wave[x] * u[x + 1], 0, 255);
  }
}