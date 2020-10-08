// Copyright 2020 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// TODO: Use fixed point & complex number classes
// Input: 3.16 signed fixed point
// Output: Number of iterations to escape, or ~0 for inside

long mandelbrot(long c_r, long c_i) {
  const long mand2 = long(2) << 16;
  const long mand1 = long(1) << 16;

  long z_r = 0, z_i = 0;

#pragma hls_unroll yes
  // TODO: Use templates
  for (long l = 0; l < 24; ++l) {
    // TODO: Saturation arithmetic
    const long z_r_p = ((z_r * z_r) >> 16) - ((z_i * z_i) >> 16) + c_r;
    const long z_i_p = 2 * ((z_r * z_i) >> 16) + c_i;
    z_r = z_r_p;
    z_i = z_i_p;
    if ((z_r < -mand2) || (z_r > mand1) || (z_i < -mand1) || (z_i > mand1))
      return l;
  }
  return ~long(0);
}

#if !NO_TESTBENCH
#include <cstdio>

int main() {
  const long h = 32;
  const long w = (h * 3) / 2;
#if YUV_OUT
  FILE *raw_out = fopen("/tmp/out.yuv", "wb");
#endif

  fprintf(stdout, " [");
  for (unsigned long y = 0; y < h; ++y) {
    fprintf(stdout, "[");
    for (unsigned long x = 0; x < w; ++x) {
      float xx = static_cast<float>(x) / w;
      float yy = static_cast<float>(y) / h;
      long l = mandelbrot((xx * 2.5 - 1.8) * (1 << 16),
                         (yy * 2.2 - 1.1) * (1 << 16));
#if YUV_OUT
      unsigned char c = (l == (~long(0))) ? 0 : 255 - (l * 5);
      fwrite(&c, 1, 1, raw_out);
#endif
      fprintf(stdout, "%u ", (unsigned)l);
      if (x < (w - 1)) fprintf(stdout, ",");
    }
    if (y < (h - 1))
      fprintf(stdout, "],");
    else
      fprintf(stdout, "]");
  }
  fprintf(stdout, "]");
#if YUV_OUT
  fclose(raw_out);
#endif
  return 0;
}
#endif
