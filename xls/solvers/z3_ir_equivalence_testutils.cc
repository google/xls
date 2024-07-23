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

#include <cstdint>
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
#include "xls/ir/proc.h"
#include "xls/ir/proc_testutils.h"
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls::solvers::z3 {

using status_testing::IsOkAndHolds;

using ::testing::_;
using ::testing::VariantWith;

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
              IsOkAndHolds(VariantWith<ProvenTrue>(_)));
  if (testing::Test::HasFailure()) {
    testing::Test::RecordProperty("original", original_f_->DumpIr());
    testing::Test::RecordProperty("final", f_->DumpIr());
  }
}

ScopedVerifyProcEquivalence::ScopedVerifyProcEquivalence(
    Proc* p, int64_t activation_count, bool include_state,
    absl::Duration timeout, xabsl::SourceLocation loc)
    : p_(p),
      activation_count_(activation_count),
      include_state_(include_state),
      timeout_(timeout),
      loc_(loc) {
  clone_package_ = std::make_unique<Package>(
      absl::StrFormat("%s_original", p->package()->name()));
  if (!p_->is_new_style_proc()) {
    for (auto* chan : p->package()->channels()) {
      CHECK_OK(clone_package_->CloneChannel(chan, chan->name()));
    }
  }
  absl::StatusOr<Proc*> cloned = p_->Clone(
      absl::StrFormat("%s_original", p_->name()), clone_package_.get());
  CHECK_OK(cloned.status());
  original_p_ = *std::move(cloned);
}

ScopedVerifyProcEquivalence::~ScopedVerifyProcEquivalence() {
  // XLS_ASSERT_OK_AND_ASSIGN doesn't like being used in destructors for some
  // reason?
  RunProcVerification();
}

void ScopedVerifyProcEquivalence::RunProcVerification() {
  testing::ScopedTrace trace(
      loc_.file_name(), loc_.line(),
      absl::StrCat("ScopedVerifyProcEquivalence failed to prove equivalence of "
                   "function ",
                   p_->name(), " before & after changes"));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * final_p_cloned,
                           p_->Clone(absl::StrFormat("%s_modified", p_->name()),
                                     clone_package_.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * original_f,
      UnrollProcToFunction(original_p_, activation_count_, include_state_));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      UnrollProcToFunction(final_p_cloned, activation_count_, include_state_));
  EXPECT_THAT(TryProveEquivalence(original_f, f, timeout_),
              IsOkAndHolds(IsProvenTrue()));
  if (testing::Test::HasFailure()) {
    testing::Test::RecordProperty("original", original_p_->DumpIr());
    testing::Test::RecordProperty("final", p_->DumpIr());
  }
}

}  // namespace xls::solvers::z3
