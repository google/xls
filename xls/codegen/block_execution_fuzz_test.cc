// Copyright 2024 The XLS Authors
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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "google/protobuf/text_format.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_type.pb.h"
#include "xls/jit/block_jit.h"
#include "xls/public/ir_parser.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls::verilog {
namespace {

// Common components for fuzzing a block against a oracle version.
//
// FuzzFiles contains the filenames for the test, source and codegen
// configurations.
//
// Subtypes of this class provide functions to generate the oracle from the
// source and configuration and support to create the appropriate runner.
//
// Once the test and oracle continuations are created they are ticked over using
// random input data and the output port values are compared. It is expected
// that at every tick the set of valid ports is the same and any valid ports
// have the same value set.
template <typename FuzzFiles>
class BaseExecutionFuzzer {
 public:
  constexpr static std::string_view kOptIrFile = FuzzFiles::kOptIrFile;
  constexpr static std::string_view kBlockIrFile = FuzzFiles::kBlockIrFile;
  constexpr static std::string_view kCodegenOptsFile =
      FuzzFiles::kCodegenOptsFile;
  constexpr static std::string_view kScheduleOptsFile =
      FuzzFiles::kScheduleOptsFile;
  constexpr static std::string_view kSigFile = FuzzFiles::kSigFile;

  explicit BaseExecutionFuzzer() {
    // Parse the oracle block
    oracle_package =
        ParsePackage(
            GetFileContents(GetXlsRunfilePath(kBlockIrFile).value()).value(),
            kBlockIrFile)
            .value();
    CHECK_EQ(oracle_package->blocks().size(), 1);

    // Generate the test block by loading the source opt_ir and modifying the
    // oracle's configuration.
    Block* oracle_block = oracle_package->blocks().front().get();
    test_package =
        ParsePackage(
            GetFileContents(GetXlsRunfilePath(kOptIrFile).value()).value(),
            kOptIrFile)
            .value();
    google::protobuf::TextFormat::ParseFromString(
        GetFileContents(GetXlsRunfilePath(kCodegenOptsFile).value()).value(),
        &original_codegen_flags);
    VLOG(2) << "Codegen opts (original): "
            << original_codegen_flags.DebugString();
    CodegenFlagsProto codegen_flags =
        FuzzFiles::ModifyOptions(original_codegen_flags);
    VLOG(2) << "Codegen opts (new): " << codegen_flags.DebugString();
    SchedulingOptionsFlagsProto schedule;
    google::protobuf::TextFormat::ParseFromString(
        GetFileContents(GetXlsRunfilePath(kScheduleOptsFile).value()).value(),
        &schedule);
    CHECK_EQ(test_package->blocks().size(), 0);
    CHECK_OK(
        ScheduleAndCodegen(test_package.get(), schedule, codegen_flags,
                           IsDelayModelSpecifiedViaFlag(schedule).value()));
    CHECK_EQ(test_package->blocks().size(), 1);
    Block* test_block = test_package->blocks().front().get();

    // Generate the runners (either JIT or interpreter) for the oracle and test
    // blocks.
    oracle_continuation = FuzzFiles::MakeContinuation(oracle_block);
    test_continuation = FuzzFiles::MakeContinuation(test_block);

    // Validate that test and oracle blocks have the same input and outputs.
    CHECK_NE(oracle_block->GetRegisters().size(),
             test_block->GetRegisters().size());

    auto filename = GetXlsRunfilePath(kSigFile);
    CHECK_OK(filename.status());
    CHECK_OK(ParseTextProtoFile(filename.value(), &msp));
    flat_input_size = absl::c_accumulate(
        msp.data_ports(), 0, [](int64_t v, const PortProto& port) {
          if (port.direction() != DirectionProto::DIRECTION_INPUT) {
            return v;
          }
          CHECK_EQ(port.type().type_enum(), TypeProto::BITS)
              << "Non bits inputs not yet supported.";
          return v + port.width();
        });

    CHECK_EQ(oracle_block->GetOutputPorts().size(),
             test_block->GetOutputPorts().size());
    for (int64_t i = 0; i < test_block->GetOutputPorts().size(); ++i) {
      CHECK_EQ(oracle_block->GetOutputPorts().at(i)->name(),
               test_block->GetOutputPorts().at(i)->name());
    }
    CHECK_EQ(oracle_block->GetInputPorts().size(),
             test_block->GetInputPorts().size());
    for (int64_t i = 0; i < test_block->GetInputPorts().size(); ++i) {
      CHECK_EQ(oracle_block->GetInputPorts().at(i)->name(),
               test_block->GetInputPorts().at(i)->name());
    }
  }

  // Do the actual execution for each input set.
  //
  // To simplify the creation of the test domains the input is just a set of
  // bits.
  void ExecuteRaw(const std::vector<std::vector<bool>>& bits) {
    auto to_values = [&](const std::vector<bool>& bits)
        -> absl::flat_hash_map<std::string, Value> {
      CHECK_EQ(bits.size(), flat_input_size);
      absl::flat_hash_map<std::string, Value> result;
      auto it = bits.cbegin();
      for (const auto& port : msp.data_ports()) {
        if (port.direction() == DirectionProto::DIRECTION_INPUT) {
          InlineBitmap ib(port.width());
          for (int64_t i = 0; i < port.width(); ++i) {
            CHECK(it != bits.cend());
            ib.Set(i, *it);
            ++it;
          }
          result[port.name()] = Value(Bits::FromBitmap(std::move(ib)));
        }
      }
      if (msp.has_reset()) {
        result[msp.reset().name()] = Value(UBits(*it ? 1 : 0, 1));
      }
      return result;
    };
    std::vector<absl::flat_hash_map<std::string, Value>> values;
    values.reserve(bits.size());
    absl::c_transform(bits, std::back_inserter(values), to_values);
    Execute(values);
  }

  // Run and validate the test against the oracle for every cycle.
  void Execute(
      const std::vector<absl::flat_hash_map<std::string, Value>>& values) {
    int64_t cycle = 0;
    for (const auto& args : values) {
      XLS_ASSERT_OK(oracle_continuation->RunOneCycle(args));
      XLS_ASSERT_OK(test_continuation->RunOneCycle(args));

      // Get the set of ports which are asserted valid and their values.
      auto valid_port_values =
          [&](const absl::flat_hash_map<std::string, Value>& values)
          -> absl::flat_hash_map<std::string, Value> {
        absl::flat_hash_map<std::string, Value> results;
        for (const auto& [name, val] : values) {
          if (name.ends_with(
                  original_codegen_flags.streaming_channel_valid_suffix())) {
            auto out_name = absl::StrReplaceAll(
                name,
                {{original_codegen_flags.streaming_channel_valid_suffix(),
                  original_codegen_flags.streaming_channel_data_suffix()}});
            if (val.bits().IsOne()) {
              results[out_name] = values.at(out_name);
            }
          }
        }
        return results;
      };
      // Get the set of ready ports
      auto ready_ports =
          [&](const absl::flat_hash_map<std::string, Value>& values)
          -> absl::flat_hash_map<std::string, Value> {
        absl::flat_hash_map<std::string, Value> results;
        for (const auto& [name, val] : values) {
          if (name.ends_with(
                  original_codegen_flags.streaming_channel_ready_suffix())) {
            results[name] = val;
          }
        }
        return results;
      };
      auto oracle_ports = oracle_continuation->output_ports();
      auto test_ports = test_continuation->output_ports();
      // Make sure the valid ports match
      ASSERT_EQ(valid_port_values(oracle_ports), valid_port_values(test_ports))
          << "cycle " << cycle;
      // Make sure the ready ports match
      ASSERT_EQ(ready_ports(oracle_ports), ready_ports(test_ports))
          << "cycle " << cycle;
      cycle++;
    }
  }

  static auto InputDomain(int64_t flat_bit_size, int64_t min_cycles,
                          int64_t max_cycles) {
    return fuzztest::VectorOf(fuzztest::VectorOf(fuzztest::Arbitrary<bool>())
                                  .WithSize(flat_bit_size))
        .WithMinSize(min_cycles)
        .WithMaxSize(max_cycles);
  }

 private:
  ModuleSignatureProto msp;
  int64_t flat_input_size = 0;
  CodegenFlagsProto original_codegen_flags;

  std::unique_ptr<Package> oracle_package;
  std::unique_ptr<BlockContinuation> oracle_continuation;
  std::unique_ptr<Package> test_package;
  std::unique_ptr<BlockContinuation> test_continuation;
  std::vector<std::string> port_names;
};

// Fuzzer for interpreter/jit on custom_schedule.x proc.
template <bool kInterpreter>
class CustomScheduleFuzzer
    : public BaseExecutionFuzzer<CustomScheduleFuzzer<kInterpreter>> {
 public:
  static constexpr std::string_view kOptIrFile =
      "xls/examples/custom_schedule.opt.ir";
  static constexpr std::string_view kBlockIrFile =
      "xls/examples/custom_schedule.block.ir";
  static constexpr std::string_view kCodegenOptsFile =
      "xls/examples/custom_schedule.codegen_options.textproto";
  constexpr static std::string_view kScheduleOptsFile =
      "xls/examples/custom_schedule.schedule_options.textproto";
  static constexpr std::string_view kSigFile =
      "xls/examples/custom_schedule.sig.textproto";

  // This test swaps register merge strategy.
  static CodegenFlagsProto ModifyOptions(const CodegenFlagsProto& opt) {
    CodegenFlagsProto res = opt;
    if (opt.register_merge_strategy() ==
        RegisterMergeStrategyProto::STRATEGY_DONT_MERGE) {
      res.set_register_merge_strategy(
          RegisterMergeStrategyProto::STRATEGY_IDENTITY_ONLY);
    } else {
      res.set_register_merge_strategy(
          RegisterMergeStrategyProto::STRATEGY_DONT_MERGE);
    }

    return res;
  }

  // Create a continuation with the default 'reset' register state.
  static std::unique_ptr<BlockContinuation> MakeContinuation(Block* b) {
    absl::flat_hash_map<std::string, Value> resets;
    for (Register* r : b->GetRegisters()) {
      resets[r->name()] =
          r->reset() ? r->reset()->reset_value : ZeroOfType(r->type());
    }
    if (kInterpreter) {
      return kInterpreterBlockEvaluator.NewContinuation(b, resets).value();
    }
    return kJitBlockEvaluator.NewContinuation(b, resets).value();
  }
};

using InterpreterCustomScheduleFuzzer = CustomScheduleFuzzer<true>;
using JitCustomScheduleFuzzer = CustomScheduleFuzzer<false>;

// Execute custom_schedule.x proc in interpreter for between 100 and 150 cycles.
FUZZ_TEST_F(InterpreterCustomScheduleFuzzer, ExecuteRaw)
    .WithDomains(
        InterpreterCustomScheduleFuzzer::InputDomain(13, /*min_cycles=*/100,
                                                     /*max_cycles=*/150));
// Execute custom_schedule.x proc in jit for between 100 and 5000 cycles.
FUZZ_TEST_F(JitCustomScheduleFuzzer, ExecuteRaw)
    .WithDomains(JitCustomScheduleFuzzer::InputDomain(13, /*min_cycles=*/100,
                                                      /*max_cycles=*/5000));

}  // namespace
}  // namespace xls::verilog
