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
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/block_testutils.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_testutils.h"
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_translator.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls::solvers::z3 {

namespace {

static constexpr bool kHasMsan =
#if defined(ABSL_HAVE_MEMORY_SANITIZER)
    true;
#else
    false;
#endif
}  // namespace

using ::absl_testing::IsOkAndHolds;

using ::testing::_;
using ::testing::VariantWith;

ScopedVerifyEquivalence::ScopedVerifyEquivalence(Function* f,
                                                 bool ignore_asserts,
                                                 absl::Duration timeout,
                                                 xabsl::SourceLocation loc)
    : f_(f), ignore_asserts_(ignore_asserts), timeout_(timeout), loc_(loc) {
  clone_p_ = std::make_unique<Package>(
      absl::StrFormat("%s_original", f->package()->name()));
  absl::StatusOr<Function*> cloned =
      f_->Clone(absl::StrFormat("%s_original", f->name()), clone_p_.get());
  CHECK_OK(cloned.status());
  original_f_ = *std::move(cloned);
}

ScopedVerifyEquivalence::~ScopedVerifyEquivalence() {
  if constexpr (kHasMsan) {
    // Z3 is substantially slower in MSAN mode, and we already get proofs from
    // non-MSAN runs. No need to try the proof.
    LOG(INFO) << "Skipping Z3 proof, as we're built with MSAN enabled.";
    return;
  } else {
    testing::ScopedTrace trace(
        loc_.file_name(), loc_.line(),
        absl::StrCat(
            "ScopedVerifyEquivalence failed to prove equivalence of function ",
            f_->name(), " before & after changes"));
    absl::StatusOr<ProverResult> result =
        TryProveEquivalence(original_f_, f_, ignore_asserts_, timeout_);
    EXPECT_THAT(result, IsOkAndHolds(VariantWith<ProvenTrue>(_)));
    if (result.ok() && std::holds_alternative<ProvenFalse>(*result)) {
      testing::Test::RecordProperty("original",
                                    original_f_->DumpIr(CounterExampleAnnotator(
                                        std::get<ProvenFalse>(*result))));
      testing::Test::RecordProperty(
          "final",
          f_->DumpIr(CounterExampleAnnotator(std::get<ProvenFalse>(*result))));
    } else if (testing::Test::HasFailure()) {
      testing::Test::RecordProperty("original", original_f_->DumpIr());
      testing::Test::RecordProperty("final", f_->DumpIr());
    }
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
  std::optional<std::string> original_ir;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * original_f,
      UnrollProcToFunction(original_p_, activation_count_, include_state_));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      UnrollProcToFunction(final_p_cloned, activation_count_, include_state_));
  EXPECT_THAT(TryProveEquivalence(original_f, f, ignore_asserts_, timeout_),
              IsOkAndHolds(IsProvenTrue()));
  if (testing::Test::HasFailure()) {
    testing::Test::RecordProperty("original",
                                  original_ir.value_or(original_p_->DumpIr()));
    testing::Test::RecordProperty("final", p_->DumpIr());
  }
}

ScopedVerifyBlockEquivalence::ScopedVerifyBlockEquivalence(
    Block* b, int64_t tick_count, bool zero_invalid_channel_data,
    bool include_reg_state, absl::Duration timeout, xabsl::SourceLocation loc)
    : b_(b),
      tick_count_(tick_count),
      zero_invalid_channel_data_(zero_invalid_channel_data),
      include_reg_state_(include_reg_state),
      timeout_(timeout),
      loc_(loc) {
  clone_package_ = std::make_unique<Package>(
      absl::StrFormat("%s_original", b->package()->name()));
  absl::StatusOr<Block*> cloned = b_->Clone(
      absl::StrFormat("%s_original", b->name()), clone_package_.get());
  CHECK_OK(cloned.status());
  original_b_ = *std::move(cloned);
}

ScopedVerifyBlockEquivalence::~ScopedVerifyBlockEquivalence() {
  // XLS_ASSERT_OK_AND_ASSIGN doesn't like being used in destructors for some
  // reason?
  RunBlockVerification();
}

void ScopedVerifyBlockEquivalence::RunBlockVerification() {
  testing::ScopedTrace trace(
      loc_.file_name(), loc_.line(),
      absl::StrCat("ScopedVerifyBlockEquivalence failed to prove equivalence "
                   "of block ",
                   b_->name(), " before & after changes"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * final_b_cloned,
                           b_->Clone(absl::StrFormat("%s_modified", b_->name()),
                                     clone_package_.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * original_f,
      UnrollBlockToFunction(original_b_, tick_count_, include_reg_state_,
                            zero_invalid_channel_data_));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      UnrollBlockToFunction(final_b_cloned, tick_count_, include_reg_state_,
                            zero_invalid_channel_data_));
  auto equiv = TryProveEquivalence(original_f, f, ignore_asserts_, timeout_);
  EXPECT_THAT(equiv, IsOkAndHolds(IsProvenTrue()));
  if (testing::Test::HasFailure()) {
    testing::Test::RecordProperty("original", original_b_->DumpIr());
    testing::Test::RecordProperty("final", final_b_cloned->DumpIr());
    if (equiv.ok()) {
      testing::Test::RecordProperty("original_unrolled_annotated",
                                    original_f->DumpIr(CounterExampleAnnotator(
                                        std::get<ProvenFalse>(*equiv))));
      testing::Test::RecordProperty(
          "final_unrolled_annotated",
          f->DumpIr(CounterExampleAnnotator(std::get<ProvenFalse>(*equiv))));
    }
  }
}

}  // namespace xls::solvers::z3
