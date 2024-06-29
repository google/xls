// Copyright 2021 The XLS Authors
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

#ifndef __MATH_H__
#define __MATH_H__

#include <cassert>

double floor(double x) {
  assert(!"Unimplemented");
  return 0.0;
}
float floorf(float x) {
  assert(!"Unimplemented");
  return 0.0f;
}
long double floorl(long double x) {
  assert(!"Unimplemented");
  return 0.0L;
}

double ceil(double x) {
  assert(!"Unimplemented");
  return 0.0;
}
float ceilf(float x) {
  assert(!"Unimplemented");
  return 0.0f;
}
long double ceill(long double x) {
  assert(!"Unimplemented");
  return 0.0L;
}

double frexp(double value, int *eptr) {
  assert(!"Unimplemented");
  return 0.0;
}

float frexpf(float x, int *pw2) {
  assert(!"Unimplemented");
  return 0.0;
}

#endif  //__MATH_H__
