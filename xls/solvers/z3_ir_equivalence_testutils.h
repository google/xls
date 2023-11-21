// Copyright 2023 The XLS Authors
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

#ifndef XLS_SOLVERS_Z3_IR_EQUIVALENCE_TESTUTILS_H_
#define XLS_SOLVERS_Z3_IR_EQUIVALENCE_TESTUTILS_H_

#include <memory>

#include "absl/time/time.h"
#include "xls/common/source_location.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"

namespace xls::solvers::z3 {

class ScopedVerifyEquivalence {
 public:
  explicit ScopedVerifyEquivalence(
      Function* f, absl::Duration timeout = absl::InfiniteDuration(),
      xabsl::SourceLocation loc = xabsl::SourceLocation::current());

  ~ScopedVerifyEquivalence();

 private:
  Function* const f_;
  absl::Duration timeout_;
  const xabsl::SourceLocation loc_;

  std::unique_ptr<Package> clone_p_;
  Function* original_f_;
};

}  // namespace xls::solvers::z3

#endif  // XLS_SOLVERS_Z3_IR_EQUIVALENCE_TESTUTILS_H_
