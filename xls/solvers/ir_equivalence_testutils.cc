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

#include "xls/solvers/ir_equivalence_testutils.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/strip.h"
#include "absl/time/time.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/block_testutils.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_testutils.h"
#include "xls/ir/value.h"
#include "xls/solvers/ir_equivalence.h"
#include "xls/solvers/prover_matchers.h"
#include "xls/solvers/solver.h"

namespace xls::solvers {

namespace {

static constexpr bool kHasMsan =
#if defined(ABSL_HAVE_MEMORY_SANITIZER)
    true;
#else
    false;
#endif

absl::flat_hash_map<ReceiveChannelRef, std::vector<Value>> ExtractProcInputs(
    Proc* p, Function* f, const ProvenFalse& fail) {
  absl::flat_hash_map<ReceiveChannelRef, std::vector<Value>> inputs;
  if (!fail.counterexample.ok()) {
    return inputs;
  }
  const auto& counterexample_map = *fail.counterexample;

  absl::flat_hash_map<std::string, ReceiveChannelRef> name_to_channel_ref;
  if (p->is_new_style_proc()) {
    for (ChannelInterface* interface : p->interface()) {
      if (interface->direction() == ChannelDirection::kReceive) {
        name_to_channel_ref[interface->name()] =
            static_cast<ReceiveChannelInterface*>(interface);
      }
    }
  } else {
    for (Channel* channel : p->package()->channels()) {
      name_to_channel_ref[std::string(channel->name())] = channel;
    }
  }

  for (Node* node : f->nodes()) {
    if (node->Is<Param>()) {
      Param* param = node->As<Param>();
      std::string_view param_name_view = param->name();
      if (absl::ConsumePrefix(&param_name_view, "available_")) {
        auto it = name_to_channel_ref.find(param_name_view);
        if (it == name_to_channel_ref.end()) {
          continue;
        }
        auto value_it = counterexample_map.find(param);
        if (value_it == counterexample_map.end()) {
          continue;
        }
        if (value_it->second.IsArray()) {
          auto array_elements = value_it->second.elements();
          inputs[it->second] =
              std::vector<Value>(array_elements.begin(), array_elements.end());
        }
      }
    }
  }
  return inputs;
}

absl::flat_hash_map<ReceiveChannelRef, std::vector<Value>> ExtractProcInputsSeq(
    Proc* p, Function* f, const ProvenFalse& fail) {
  absl::flat_hash_map<ReceiveChannelRef, std::vector<Value>> inputs;
  if (!fail.counterexample.ok()) {
    return inputs;
  }
  const auto& counterexample_map = *fail.counterexample;

  absl::flat_hash_map<std::string, ReceiveChannelRef> name_to_channel_ref;
  if (p->is_new_style_proc()) {
    for (ChannelInterface* interface : p->interface()) {
      if (interface->direction() == ChannelDirection::kReceive) {
        name_to_channel_ref[std::string(interface->name())] =
            static_cast<ReceiveChannelInterface*>(interface);
      }
    }
  } else {
    for (Channel* channel : p->package()->channels()) {
      name_to_channel_ref[std::string(channel->name())] = channel;
    }
  }

  // For each channel, what we received on each activation (or std::nullopt if
  // the channel was empty at that activation).
  absl::flat_hash_map<ReceiveChannelRef, std::vector<std::optional<Value>>>
      channel_values;

  for (Node* node : f->nodes()) {
    if (!node->Is<Param>()) {
      continue;
    }
    Param* param = node->As<Param>();
    std::string_view name = param->name();
    if (!absl::ConsumePrefix(&name, "recv_data_act")) {
      ADD_FAILURE() << "Unable to parse param name: " << param->ToString();
      continue;
    }
    size_t underscore_pos = name.find('_');
    if (underscore_pos == std::string_view::npos) {
      ADD_FAILURE() << "Unable to parse param name: " << param->ToString();
      continue;
    }
    std::string_view act_idx_str = name.substr(0, underscore_pos);
    std::string_view channel_name = name.substr(underscore_pos + 1);
    int64_t act_idx = 0;
    if (!absl::SimpleAtoi(act_idx_str, &act_idx)) {
      ADD_FAILURE() << "Unable to parse param name activation number: "
                    << param->ToString();
      continue;
    }
    auto it = name_to_channel_ref.find(channel_name);
    if (it != name_to_channel_ref.end()) {
      auto value_it = counterexample_map.find(param);
      if (value_it != counterexample_map.end()) {
        if (channel_values[it->second].size() <= act_idx) {
          channel_values[it->second].resize(act_idx + 1, std::nullopt);
        }
        channel_values[it->second][act_idx] = value_it->second;
      }
    }
  }

  for (const auto& [recv_ref, act_values] : channel_values) {
    std::vector<Value>& vec = inputs[recv_ref];
    for (const std::optional<Value>& val : act_values) {
      if (val) {
        vec.push_back(*val);
      }
    }
  }

  return inputs;
}

}  // namespace

using ::absl_testing::IsOkAndHolds;

using ::testing::_;
using ::testing::VariantWith;
using ::xls::solvers::IsProvenTrue;

ScopedVerifyEquivalence::ScopedVerifyEquivalence(Function* f,
                                                 bool ignore_asserts,
                                                 absl::Duration timeout,
                                                 xabsl::SourceLocation loc)
    : f_(f), ignore_asserts_(ignore_asserts), timeout_(timeout), loc_(loc) {
  if (timeout != absl::InfiniteDuration()) {
    limit_.timeout = timeout;
  }
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
        TryProveEquivalence(original_f_, f_, ignore_asserts_, kind_, limit_);
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
  if (timeout != absl::InfiniteDuration()) {
    limit_.timeout = timeout;
  }
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
  absl::StatusOr<ProverResult> equiv =
      TryProveEquivalence(original_f, f, ignore_asserts_, kind_, limit_);
  EXPECT_THAT(equiv, IsOkAndHolds(IsProvenTrue()));
  if (testing::Test::HasFailure()) {
    std::string original_dumped;
    std::string final_dumped;
    if (equiv.ok() && std::holds_alternative<ProvenFalse>(*equiv)) {
      const ProvenFalse& fail = std::get<ProvenFalse>(*equiv);
      auto original_inputs =
          ExtractProcInputsSeq(original_p_, original_f, fail);
      auto final_inputs = ExtractProcInputsSeq(final_p_cloned, f, fail);

      xls::ProcResultsAnnotatorOrNothing original_annotator(
          original_p_, activation_count_,
          /*output_value_count=*/activation_count_, original_inputs);
      original_dumped = original_p_->DumpIr(original_annotator);

      xls::ProcResultsAnnotatorOrNothing final_annotator(
          final_p_cloned, activation_count_,
          /*output_value_count=*/activation_count_, final_inputs);
      final_dumped = final_p_cloned->DumpIr(final_annotator);
    } else {
      original_dumped = original_p_->DumpIr();
      final_dumped = final_p_cloned->DumpIr();
    }
    testing::Test::RecordProperty("original", original_dumped);
    testing::Test::RecordProperty("final", final_dumped);
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
  if (timeout != absl::InfiniteDuration()) {
    limit_.timeout = timeout;
  }
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
  auto equiv =
      TryProveEquivalence(original_f, f, ignore_asserts_, kind_, limit_);
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

ScopedVerifyProcOutputEquivalence::ScopedVerifyProcOutputEquivalence(
    Proc* p, const Options& options, absl::Duration timeout,
    xabsl::SourceLocation loc)
    : p_(p), options_(options), timeout_(timeout), loc_(loc) {
  clone_package_ = std::make_unique<Package>(
      absl::StrFormat("%s_original", p->package()->name()));
  absl::StatusOr<Proc*> cloned = p_->Clone(
      absl::StrFormat("%s_original", p->name()), clone_package_.get());
  CHECK_OK(cloned.status());
  original_p_ = *std::move(cloned);
}

ScopedVerifyProcOutputEquivalence::~ScopedVerifyProcOutputEquivalence() {
  RunProcVerification();
}

void ScopedVerifyProcOutputEquivalence::RunProcVerification() {
  testing::ScopedTrace trace(
      loc_.file_name(), loc_.line(),
      absl::StrCat("ScopedVerifyProcOutputEquivalence failed to prove "
                   "equivalence of function ",
                   p_->name(), " before & after changes"));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * final_p_cloned,
                           p_->Clone(absl::StrFormat("%s_modified", p_->name()),
                                     clone_package_.get()));
  std::optional<std::string> original_ir;
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * original_f,
      UnrollProcToUntimedFunction(original_p_, options_.activation_count,
                                  options_.input_value_count,
                                  options_.output_value_count));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f,
      UnrollProcToUntimedFunction(
          final_p_cloned,
          options_.final_activation_count.value_or(options_.activation_count),
          options_.input_value_count, options_.output_value_count));
  auto equiv = TryProveEquivalence(original_f, f,
                                   /*ignore_asserts=*/true, kind_, limit_);
  EXPECT_THAT(equiv, IsOkAndHolds(IsProvenTrue()));
  if (testing::Test::HasFailure()) {
    std::string original_dumped;
    std::string final_dumped;
    if (equiv.ok() && std::holds_alternative<ProvenFalse>(*equiv)) {
      const ProvenFalse& fail = std::get<ProvenFalse>(*equiv);
      auto original_inputs = ExtractProcInputs(original_p_, original_f, fail);
      auto final_inputs = ExtractProcInputs(final_p_cloned, f, fail);

      xls::ProcResultsAnnotatorOrNothing original_annotator(
          original_p_, options_.activation_count, options_.output_value_count,
          original_inputs);
      original_dumped = original_p_->DumpIr(original_annotator);

      xls::ProcResultsAnnotatorOrNothing final_annotator(
          final_p_cloned,
          options_.final_activation_count.value_or(options_.activation_count),
          options_.output_value_count, final_inputs);
      final_dumped = final_p_cloned->DumpIr(final_annotator);
    } else {
      original_dumped = original_p_->DumpIr();
      final_dumped = final_p_cloned->DumpIr();
    }
    testing::Test::RecordProperty("original", original_dumped);
    testing::Test::RecordProperty("final", final_dumped);
  }
}

}  // namespace xls::solvers
