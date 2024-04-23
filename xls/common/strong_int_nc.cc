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

// Negative compilation tests for StrongInt.

#include <cstdint>

#include "xls/common/strong_int.h"

XLS_DEFINE_STRONG_INT_TYPE(USD, int32_t);
XLS_DEFINE_STRONG_INT_TYPE(EUR, int32_t);

class Explicit {
 public:
  explicit Explicit(int32_t) {}
};

// Each case specified under each symbol must fail to compile.
static void MustNotCompile() {
  USD dollars(1);
  (void)dollars;  // Avoid unused variable warning.
  EUR euro(2);
  (void)euro;  // Avoid unused variable warning.

#if defined(TEST_CTOR)
  USD funny_money(euro);

#elif defined(TEST_ASSIGN)
  dollars = euro;

#elif defined(TEST_PLAIN_ASSIGN)
  dollars = 100;

#elif defined(TEST_IMPLICIT_CAST)
  int32_t raw = dollars;
  (void)raw;

#elif defined(TEST_IMPLICIT_DISALLOWED)
  Explicit x = dollars;
  (void)x;

#elif defined(TEST_ADD)
  dollars + euro;

#elif defined(TEST_SUBTRACT)
  dollars - euro;

#elif defined(TEST_BITWISE_AND)
  dollars & euro;

#elif defined(TEST_BITWISE_OR)
  dollars | euro;

#elif defined(TEST_BITWISE_XOR)
  dollars ^ euro;

#elif defined(TEST_ADD_ASSIGN)
  dollars += euro;

#elif defined(TEST_SUBTRACT_ASSIGN)
  dollars -= euro;

#elif defined(TEST_EQ)
  (dollars == euro);

#elif defined(TEST_NE)
  (dollars != euro);

#elif defined(TEST_LT)
  (dollars < euro);

#elif defined(TEST_LE)
  (dollars <= euro);

#elif defined(TEST_GT)
  (dollars > euro);

#elif defined(TEST_GE)
  (dollars >= euro);

#endif
}
