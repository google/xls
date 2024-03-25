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

#include "xls/solvers/z3_ir_equivalence_testutils.h"

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/package.h"
#include "xls/solvers/z3_ir_equivalence.h"

namespace xls::solvers::z3 {

using status_testing::IsOkAndHolds;

ScopedVerifyEquivalence::ScopedVerifyEquivalence(Function* f,
                                                 absl::Duration timeout,
                                                 xabsl::SourceLocation loc)
    : f_(f), timeout_(timeout), loc_(loc) {
  clone_p_ = std::make_unique<Package>(
      absl::StrFormat("%s_original", f->package()->name()));
  absl::StatusOr<Function*> cloned =
      f_->Clone(absl::StrFormat("%s_original", f->name()), clone_p_.get());
  CHECK_OK(cloned.status());
  original_f_ = *std::move(cloned);
}

ScopedVerifyEquivalence::~ScopedVerifyEquivalence() {
  testing::ScopedTrace trace(
      loc_.file_name(), loc_.line(),
      absl::StrCat(
          "ScopedVerifyEquivalence failed to prove equivalence of function ",
          f_->name(), " before & after changes"));
  EXPECT_THAT(TryProveEquivalence(original_f_, f_, timeout_),
              IsOkAndHolds(true));
}

}  // namespace xls::solvers::z3
