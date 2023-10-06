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

#include "gtest/gtest.h"
#include "xls/common/init_xls.h"

int main(int argc, char* argv[]) {
  testing::InitGoogleTest(&argc, argv);
  xls::InitXls(argv[0], argc, argv);

  // Rapidcheck parameters for deterministic quickchecks in unit tests.
  //
  // We use deterministic quickchecks to avoid "flakiness" in unit test targets.
  // Particular targets can be broken out from unit test files if we want longer
  // running entities.
  //
  // Note that, even in environments that use the same seed value, the
  // randomized values that result from the RNG sequence are not necessarily the
  // same as RNG device implementations may differ.
  //
  // Seed value was obtained arbitrarily via `random.randint(0, 2**64)`.
  setenv("RC_PARAMS",
         "seed=10066090388458203979 verbose_progress=1 max_success=300",
         /*replace=*/false);

  return RUN_ALL_TESTS();
}
