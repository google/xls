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

#include "xls/fuzzer/sample_runner.h"

#include <cstdlib>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/fuzzer/cpp_sample_runner.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/simulation/check_simulator.h"
#include "xls/tools/eval_utils.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// An adder implementation using a proc in DSLX.
constexpr std::string_view kProcAdderDSLX = R"(
proc main {
  operand_0: chan<u32> in;
  operand_1: chan<u32> in;
  result: chan<u32> out;

  init { () }

  config(operand_0: chan<u32> in,
         operand_1: chan<u32> in,
         result: chan<u32> out
         ) {
    (operand_0, operand_1, result)
  }

  next(tok: token, state: ()) {
    let (tok_operand_0_val, operand_0_val) = recv(tok, operand_0);
    let (tok_operand_1_val, operand_1_val) = recv(tok, operand_1);
    let tok_recv = join(tok_operand_0_val, tok_operand_1_val);

    let result_val = operand_0_val + operand_1_val;
    let tok_send = send(tok_recv, result, result_val);
    ()
  }
}
)";

// A simple counter implementation using a proc in DSLX.
constexpr std::string_view kProcCounterDSLX = R"(
proc main {
  enable_counter: chan<bool> in;
  result: chan<u32> out;

  init { u32:42 }

  config(enable_counter: chan<bool> in,
         result: chan<u32> out
         ) {
    (enable_counter, result)
  }

  next(tok: token, counter_value: u32) {
    let (tok_enable_counter, enable_counter_val) = recv(tok, enable_counter);

    let result_val = if enable_counter_val == true {counter_value + u32:1}
      else {counter_value};
    let tok_send = send(tok_enable_counter, result, result_val);
    result_val
  }
}
)";

// An IR function with a loop which runs for many iterations.
constexpr std::string_view kLongLoopIR = R"(package long_loop

fn body(i: bits[64], accum: bits[64]) -> bits[64] {
  ret one: bits[64] = literal(value=1)
}

top fn main(x: bits[64]) -> bits[64] {
  zero: bits[64] = literal(value=0, id=1)
  ret result: bits[64] = counted_for(zero, trip_count=0xffff_ffff_ffff, stride=1, body=body, id=5)
}
)";

absl::StatusOr<dslx::InterpValue> InterpValueFromIrString(std::string_view s) {
  XLS_ASSIGN_OR_RETURN(Value v, Parser::ParseTypedValue(s));
  return dslx::ValueToInterpValue(v);
}

absl::StatusOr<dslx::InterpValue> SignedInterpValueFromIrString(
    std::string_view s) {
  XLS_ASSIGN_OR_RETURN(Value v, Parser::ParseTypedValue(s));
  XLS_ASSIGN_OR_RETURN(Bits bits, v.GetBitsWithStatus());
  return dslx::InterpValue::MakeSigned(bits);
}

absl::Status AppendToArgsBatchEntry(
    std::vector<dslx::InterpValue>& args_batch_entry, std::string_view s) {
  XLS_ASSIGN_OR_RETURN(dslx::InterpValue interp_value,
                       InterpValueFromIrString(s));
  args_batch_entry.push_back(interp_value);
  return absl::OkStatus();
}

absl::Status AddArgsBatchEntry(ArgsBatch& args_batch,
                               const std::vector<std::string_view>& values) {
  std::vector<dslx::InterpValue>& entry = args_batch.emplace_back();
  entry.reserve(values.size());
  for (std::string_view s : values) {
    XLS_RETURN_IF_ERROR(AppendToArgsBatchEntry(entry, s));
  }
  return absl::OkStatus();
}

absl::StatusOr<ArgsBatch> ToArgsBatch(
    const std::vector<std::vector<std::string_view>>& arg_values) {
  ArgsBatch args_batch;
  for (const std::vector<std::string_view>& values : arg_values) {
    XLS_RETURN_IF_ERROR(AddArgsBatchEntry(args_batch, values));
  }
  return args_batch;
}

class SampleRunnerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    XLS_ASSERT_OK_AND_ASSIGN(temp_dir_, TempDirectory::Create());
  }

  void TearDown() override {
    // Take ownership of the `temp_dir_` so it will be destroyed on return; this
    // lets us use early-exit control flow.
    TempDirectory temp_dir = *std::move(temp_dir_);
    temp_dir_.reset();

    // If the test failed, preserve the outputs in the undeclared outputs
    // directory (assuming one exists).
    if (!HasFailure()) {
      return;
    }

    const char* test_undeclared_outputs_dir =
        getenv("TEST_UNDECLARED_OUTPUTS_DIR");
    if (test_undeclared_outputs_dir == nullptr) {
      return;
    }
    std::filesystem::path undeclared_outputs_dir(test_undeclared_outputs_dir);

    const testing::TestInfo* test_info =
        testing::UnitTest::GetInstance()->current_test_info();
    if (test_info == nullptr) {
      return;
    }
    CHECK(test_info->name() != nullptr);

    std::filesystem::path test_outputs_path =
        undeclared_outputs_dir / test_info->name();
    if (test_info->type_param() != nullptr) {
      test_outputs_path /= test_info->type_param();
    }
    if (test_info->value_param() != nullptr) {
      test_outputs_path /= test_info->value_param();
    }
    CHECK(std::filesystem::create_directories(test_outputs_path));
    std::filesystem::copy(temp_dir.path(), test_outputs_path,
                          std::filesystem::copy_options::recursive);
  }

  std::filesystem::path GetTempPath() { return temp_dir_->path(); }

 private:
  std::optional<TempDirectory> temp_dir_;
};

TEST_F(SampleRunnerTest, InterpretDSLXSingleValue) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_convert_to_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           ToArgsBatch({{"bits[8]:42", "bits[8]:100"}}));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string results,
                           GetFileContents(GetTempPath() / "sample.x.results"));
  EXPECT_EQ(absl::StripAsciiWhitespace(results), "bits[8]:0x8e")
      << "args_batch = " << ArgsBatchToText(args_batch);
}

TEST_F(SampleRunnerTest, InterpretDSLXMultipleValues) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_convert_to_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           ToArgsBatch({{"bits[8]:42", "bits[8]:100"},
                                        {"bits[8]:222", "bits[8]:240"}}));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string results,
                           GetFileContents(GetTempPath() / "sample.x.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e", "bits[8]:0xce"));
}

TEST_F(SampleRunnerTest, InterpretInvalidDSLX) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "syntaxerror!!! fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_convert_to_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           ToArgsBatch({{"bits[8]:42", "bits[8]:100"}}));
  EXPECT_THAT(runner.Run(Sample(std::string(dslx_text), options, args_batch)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected start of top-level construct")));
  EXPECT_THAT(GetFileContents(GetTempPath() / "exception.txt"),
              IsOkAndHolds(HasSubstr("Expected start of top-level construct")));
}

TEST_F(SampleRunnerTest, DSLXToIR) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK(runner.Run(Sample(std::string(dslx_text), options, {})));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.ir"),
              IsOkAndHolds(HasSubstr("package sample")));
}

TEST_F(SampleRunnerTest, EvaluateIR) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch,
                           ToArgsBatch({{"bits[8]:42", "bits[8]:100"},
                                        {"bits[8]:222", "bits[8]:240"}}));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string dslx_results,
                           GetFileContents(GetTempPath() / "sample.x.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(dslx_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e", "bits[8]:0xce"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string ir_results,
      GetFileContents(GetTempPath() / "sample.ir.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(ir_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e", "bits[8]:0xce"));
}

TEST_F(SampleRunnerTest, EvaluateIRWide) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: bits[100], y: bits[100]) -> bits[100] { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(
      ArgsBatch args_batch,
      ToArgsBatch({
          {
              "bits[100]:0xc9f2c9cd04674edea40000000",  // 10**30
              "bits[100]:0xc9f2c9cd04674edea40000000",  // 10**30
          },
          {
              "bits[100]:0x100000000000000000000",  // 2**80
              "bits[100]:0x200000000000000000000",  // 2**81
          },
      }));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string dslx_results,
                           GetFileContents(GetTempPath() / "sample.x.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(dslx_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[100]:0x9_3e59_39a0_8ce9_dbd4_8000_0000",
                          "bits[100]:0x3_0000_0000_0000_0000_0000"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string ir_results,
      GetFileContents(GetTempPath() / "sample.ir.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(ir_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[100]:0x9_3e59_39a0_8ce9_dbd4_8000_0000",
                          "bits[100]:0x3_0000_0000_0000_0000_0000"));
}

TEST_F(SampleRunnerTest, InterpretMixedSignedness) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: s8) -> s8 { (x as s8) + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  ArgsBatch args_batch;
  std::vector<dslx::InterpValue>& args = args_batch.emplace_back();
  XLS_ASSERT_OK_AND_ASSIGN(dslx::InterpValue unsigned_value,
                           InterpValueFromIrString("bits[8]:42"));
  XLS_ASSERT_OK_AND_ASSIGN(dslx::InterpValue signed_value,
                           SignedInterpValueFromIrString("bits[8]:100"));
  args.push_back(unsigned_value);
  args.push_back(signed_value);
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string dslx_results,
                           GetFileContents(GetTempPath() / "sample.x.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(dslx_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string ir_results,
      GetFileContents(GetTempPath() / "sample.ir.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(ir_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e"));
}

TEST_F(SampleRunnerTest, InterpretMixedSignednessUnsignedInputs) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: s8) -> s8 { (x as s8) + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:0xb0",
                                                         "bits[8]:0x0a",
                                                     },
                                                 }));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string dslx_results,
                           GetFileContents(GetTempPath() / "sample.x.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(dslx_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0xba"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string ir_results,
      GetFileContents(GetTempPath() / "sample.ir.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(ir_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0xba"));
}

TEST_F(SampleRunnerTest, EvaluateIRMiscompareSingleResult) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_ir_main = [](const std::vector<std::string>&,
                          const std::filesystem::path&,
                          const SampleOptions&) -> absl::StatusOr<std::string> {
        return "bits[8]:1\n";
      }});
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));

  static constexpr std::string_view kExpectedError =
      R"(Result miscompare for sample 0:
args: bits[8]:0x2a; bits[8]:0x64
evaluated unopt IR (JIT), evaluated unopt IR (interpreter) =
   bits[8]:0x1
interpreted DSLX =
   bits[8]:0x8e)";
  EXPECT_THAT(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(kExpectedError)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string exception,
                           GetFileContents(GetTempPath() / "exception.txt"));
  EXPECT_THAT(exception, HasSubstr(kExpectedError));
}

TEST_F(SampleRunnerTest, EvaluateIRMiscompareMultipleResults) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_ir_main = [](const std::vector<std::string>&,
                          const std::filesystem::path&,
                          const SampleOptions&) -> absl::StatusOr<std::string> {
        return "bits[8]:100\n"
               "bits[8]:1\n";
      }});
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:40",
                                                         "bits[8]:60",
                                                     },
                                                     {
                                                         "bits[8]:2",
                                                         "bits[8]:1",
                                                     },
                                                 }));

  static constexpr std::string_view kExpectedError =
      R"(Result miscompare for sample 1:
args: bits[8]:0x2; bits[8]:0x1
evaluated unopt IR (JIT), evaluated unopt IR (interpreter) =
   bits[8]:0x1
interpreted DSLX =
   bits[8]:0x3)";
  EXPECT_THAT(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr(kExpectedError),
                     Not(HasSubstr("Result miscompare for sample 0:")))));
  XLS_ASSERT_OK_AND_ASSIGN(std::string exception,
                           GetFileContents(GetTempPath() / "exception.txt"));
  EXPECT_THAT(exception,
              AllOf(HasSubstr(kExpectedError),
                    Not(HasSubstr("Result miscompare for sample 0:"))));
}

TEST_F(SampleRunnerTest, EvaluateIRMiscompareNumberOfResults) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_ir_main = [](const std::vector<std::string>&,
                          const std::filesystem::path&,
                          const SampleOptions&) -> absl::StatusOr<std::string> {
        return "bits[8]:100\n"
               "bits[8]:100\n";
      }});
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:64",
                                                     },
                                                 }));

  EXPECT_THAT(runner.Run(Sample(std::string(dslx_text), options, args_batch)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Results for evaluated unopt IR (JIT) has 2 "
                                 "values, argument batch has 1")));
  XLS_ASSERT_OK_AND_ASSIGN(std::string exception,
                           GetFileContents(GetTempPath() / "exception.txt"));
  EXPECT_THAT(exception, HasSubstr("Results for evaluated unopt IR (JIT) has 2 "
                                   "values, argument batch has 1"));
}

TEST_F(SampleRunnerTest, InterpretOptIR) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string opt_ir,
                           GetFileContents(GetTempPath() / "sample.opt.ir"));
  EXPECT_THAT(opt_ir, HasSubstr("package sample"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string opt_ir_results,
      GetFileContents(GetTempPath() / "sample.opt.ir.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(opt_ir_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e"));
}

TEST_F(SampleRunnerTest, InterpretOptIRMiscompare) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_ir_main = [](const std::vector<std::string>& args,
                          const std::filesystem::path&, const SampleOptions&) {
        bool use_llvm_jit = absl::c_any_of(args, [](const std::string& arg) {
          return arg == "--use_llvm_jit";
        });
        bool is_opt_ir = absl::c_any_of(args, [](const std::string& arg) {
          return absl::EndsWith(arg, ".opt.ir");
        });

        if (use_llvm_jit && is_opt_ir) {
          return "bits[8]:0\n";  // incorrect result
        }
        return "bits[8]:100\n";  // correct result
      }});
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:40",
                                                         "bits[8]:60",
                                                     },
                                                 }));
  ASSERT_THAT(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("Result miscompare for sample 0"),
                     HasSubstr("\nevaluated opt IR (JIT) =\n   bits[8]:0x0"))));
}

TEST_F(SampleRunnerTest, CodegenCombinational) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_codegen(true);
  options.set_codegen_args({"--generator=combinational"});
  options.set_use_system_verilog(false);
  options.set_simulate(true);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GetFileContents(GetTempPath() / "sample.v"));
  EXPECT_THAT(verilog, HasSubstr("endmodule"));
  // A combinational block should not have a blocking assignment.
  EXPECT_THAT(verilog, Not(HasSubstr("<=")));

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog_results,
                           GetFileContents(GetTempPath() / "sample.v.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(verilog_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e"));
}

TEST_F(SampleRunnerTest, CodegenCombinationalWrongResults) {
  SampleRunner runner(
      GetTempPath(),
      {.simulate_module_main =
           [](const std::vector<std::string>&, const std::filesystem::path&,
              const SampleOptions&) -> absl::StatusOr<std::string> {
        return "bits[8]:1\n";
      }});
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_codegen(true);
  options.set_codegen_args({"--generator=combinational"});
  options.set_use_system_verilog(false);
  options.set_simulate(true);
  options.set_simulator("iverilog");
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));
  EXPECT_THAT(runner.Run(Sample(std::string(dslx_text), options, args_batch)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("Result miscompare for sample 0"))));
}

TEST_F(SampleRunnerTest, CodegenPipeline) {
  if (!DefaultSimulatorSupportsSystemVerilog()) {
    GTEST_SKIP() << "uses SystemVerilog, default simulator does not support";
  }
  SampleRunner runner(GetTempPath());
  constexpr std::string_view dslx_text =
      "fn main(x: u8, y: u8) -> u8 { x + y }";
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_codegen(true);
  options.set_codegen_args({
      "--generator=pipeline",
      "--pipeline_stages=2",
      "--reset_data_path=false",
  });
  options.set_use_system_verilog(true);
  options.set_simulate(true);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(dslx_text), options, args_batch)));

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GetFileContents(GetTempPath() / "sample.sv"));
  // A combinational block should have a blocking assignment.
  EXPECT_THAT(verilog, HasSubstr("<="));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog_results,
      GetFileContents(GetTempPath() / "sample.sv.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(verilog_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e"));
}

TEST_F(SampleRunnerTest, IRInput) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view ir_text = R"(
package foo

top fn foo(x: bits[8], y: bits[8]) -> bits[8] {
  ret add.1: bits[8] = add(x, y)
}
)";
  SampleOptions options;
  options.set_input_is_dslx(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));
  XLS_ASSERT_OK(runner.Run(Sample(std::string(ir_text), options, args_batch)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir,
                           GetFileContents(GetTempPath() / "sample.ir"));
  EXPECT_THAT(ir, HasSubstr("package foo"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string opt_ir,
                           GetFileContents(GetTempPath() / "sample.opt.ir"));
  EXPECT_THAT(opt_ir, HasSubstr("package foo"));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string opt_results,
      GetFileContents(GetTempPath() / "sample.opt.ir.results"));
  EXPECT_THAT(absl::StrSplit(absl::StripAsciiWhitespace(opt_results), "\n",
                             absl::SkipEmpty()),
              ElementsAre("bits[8]:0x8e"));
}

TEST_F(SampleRunnerTest, BadIRInput) {
  SampleRunner runner(GetTempPath());
  constexpr std::string_view ir_text = "bogus ir string";
  SampleOptions options;
  options.set_input_is_dslx(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[8]:42",
                                                         "bits[8]:100",
                                                     },
                                                 }));
  EXPECT_THAT(runner.Run(Sample(std::string(ir_text), options, args_batch)),
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("eval_ir_main"),
                             HasSubstr("returned non-zero exit status"))));
  EXPECT_THAT(GetFileContents(GetTempPath() / "eval_ir_main.stderr"),
              IsOkAndHolds(HasSubstr("Expected 'package' keyword")));
}

TEST_F(SampleRunnerTest, Timeout) {
  SampleRunner runner(GetTempPath());
  SampleOptions options;
  options.set_input_is_dslx(false);
  options.set_optimize_ir(false);
  options.set_use_jit(false);
  options.set_codegen(false);
  options.set_timeout_seconds(3);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[64]:42",
                                                     },
                                                 }));
  EXPECT_THAT(runner.Run(Sample(std::string(kLongLoopIR), options, args_batch)),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       AllOf(HasSubstr("timed out after 3 seconds"))));
}

TEST_F(SampleRunnerTest, CodegenCombinationalProcWithNoState) {
  SampleRunner runner(GetTempPath());
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_codegen(true);
  options.set_codegen_args({"--generator=combinational"});
  options.set_use_system_verilog(false);
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_simulate(true);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[32]:42",
                                                         "bits[32]:100",
                                                     },
                                                 }));
  XLS_ASSERT_OK(runner.Run(
      Sample(std::string(kProcAdderDSLX), options, args_batch,
             /*ir_channel_names=*/{"sample__operand_0", "sample__operand_1"})));

  constexpr std::string_view expected_result =
      "sample__result : {\n  bits[32]:0x8e\n}";
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.x.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.ir.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.opt.ir.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.v.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
}

TEST_F(SampleRunnerTest, CodegenPipelineProcWithState) {
  if (!DefaultSimulatorSupportsSystemVerilog()) {
    GTEST_SKIP() << "uses SystemVerilog, default simulator does not support";
  }
  SampleRunner runner(GetTempPath());
  SampleOptions options;
  options.set_input_is_dslx(true);
  options.set_ir_converter_args({"--top=main"});
  options.set_codegen(true);
  options.set_codegen_args({
      "--generator=pipeline",
      "--pipeline_stages=2",
      "--reset=rst",
      "--reset_data_path=false",
  });
  options.set_use_system_verilog(true);
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_simulate(true);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[1]:1",
                                                     },
                                                     {
                                                         "bits[1]:0",
                                                     },
                                                 }));
  XLS_ASSERT_OK(
      runner.Run(Sample(std::string(kProcCounterDSLX), options, args_batch,
                        /*ir_channel_names=*/{"sample__enable_counter"})));

  constexpr std::string_view expected_result =
      "sample__result : {\n  bits[32]:0x2b\n  bits[32]:0x2b\n}";
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.x.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.ir.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.opt.ir.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
  EXPECT_THAT(GetFileContents(GetTempPath() / "sample.sv.results"),
              IsOkAndHolds(HasSubstr(expected_result)));
}

TEST_F(SampleRunnerTest, MiscompareNumberOfChannels) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_proc_main =
           [](const std::vector<std::string>&, const std::filesystem::path&,
              const SampleOptions&) -> absl::StatusOr<std::string> {
        return ChannelValuesToString(
            {{"sample__result", {}}, {"extra_channel_name", {}}});
      }});
  SampleOptions options;
  options.set_ir_converter_args({"--top=main"});
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[1]:1",
                                                     },
                                                     {
                                                         "bits[1]:0",
                                                     },
                                                 }));
  EXPECT_THAT(
      runner.Run(Sample(std::string(kProcCounterDSLX), options, args_batch,
                        /*ir_channel_names=*/{"sample__enable_counter"})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("evaluated unopt IR (JIT) has 2 channel(s), "
                         "interpreted DSLX has 1 channel(s).")));
}

TEST_F(SampleRunnerTest, MiscompareChannelNames) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_proc_main =
           [](const std::vector<std::string>&, const std::filesystem::path&,
              const SampleOptions&) -> absl::StatusOr<std::string> {
        return ChannelValuesToString({{"sample__enable_counter", {}}});
      }});
  SampleOptions options;
  options.set_ir_converter_args({"--top=main"});
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[1]:1",
                                                     },
                                                     {
                                                         "bits[1]:0",
                                                     },
                                                 }));
  EXPECT_THAT(
      runner.Run(Sample(std::string(kProcCounterDSLX), options, args_batch,
                        /*ir_channel_names=*/{"sample__enable_counter"})),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "A channel named sample__enable_counter is present in evaluated "
              "unopt IR (JIT), but it is not present in interpreted DSLX.")));
}

TEST_F(SampleRunnerTest, MiscompareMissingChannel) {
  SampleRunner runner(
      GetTempPath(),
      {.eval_proc_main =
           [](const std::vector<std::string>&, const std::filesystem::path&,
              const SampleOptions&) -> absl::StatusOr<std::string> {
        return ChannelValuesToString({});
      }});
  SampleOptions options;
  options.set_ir_converter_args({"--top=main"});
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[1]:1",
                                                     },
                                                     {
                                                         "bits[1]:0",
                                                     },
                                                 }));
  EXPECT_THAT(
      runner.Run(Sample(std::string(kProcCounterDSLX), options, args_batch,
                        /*ir_channel_names=*/{"sample__enable_counter"})),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(
              HasSubstr("evaluated unopt IR (JIT) has 0 channel(s), "
                        "interpreted DSLX has 1 channel(s)."),
              HasSubstr("The IR channel names in evaluated unopt IR (JIT) are: "
                        "[]."),
              HasSubstr("The IR channel names in interpreted DSLX are: "
                        "['sample__result']."))));
}

TEST_F(SampleRunnerTest, MiscompareNumberOfChannelValues) {
  XLS_ASSERT_OK_AND_ASSIGN(Value value,
                           Parser::ParseTypedValue("bits[32]:0x2b"));
  SampleRunner runner(
      GetTempPath(),
      {.eval_proc_main =
           [value](const std::vector<std::string>&,
                   const std::filesystem::path&,
                   const SampleOptions&) -> absl::StatusOr<std::string> {
        return ChannelValuesToString({{"sample__result", {value}}});
      }});
  SampleOptions options;
  options.set_ir_converter_args({"--top=main"});
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[1]:1",
                                                     },
                                                     {
                                                         "bits[1]:0",
                                                     },
                                                 }));

  static constexpr std::string_view expected_error =
      "In evaluated unopt IR (JIT), channel 'sample__result' has 1 entries. "
      "However, in interpreted DSLX, channel 'sample__result' has 2 entries.";
  EXPECT_THAT(
      runner.Run(Sample(std::string(kProcCounterDSLX), options, args_batch,
                        /*ir_channel_names=*/{"sample__enable_counter"})),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(expected_error)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string exception,
                           GetFileContents(GetTempPath() / "exception.txt"));
  EXPECT_THAT(exception, HasSubstr(expected_error));
}

TEST_F(SampleRunnerTest, MiscompareChannelValues) {
  XLS_ASSERT_OK_AND_ASSIGN(Value correct_value,
                           Parser::ParseTypedValue("bits[32]:43"));
  XLS_ASSERT_OK_AND_ASSIGN(Value incorrect_value,
                           Parser::ParseTypedValue("bits[32]:42"));
  SampleRunner runner(
      GetTempPath(),
      {.eval_proc_main =
           [correct_value, incorrect_value](
               const std::vector<std::string>&, const std::filesystem::path&,
               const SampleOptions&) -> absl::StatusOr<std::string> {
        return ChannelValuesToString(
            {{"sample__result", {correct_value, incorrect_value}}});
      }});
  SampleOptions options;
  options.set_ir_converter_args({"--top=main"});
  options.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
  options.set_optimize_ir(false);
  XLS_ASSERT_OK_AND_ASSIGN(ArgsBatch args_batch, ToArgsBatch({
                                                     {
                                                         "bits[1]:1",
                                                     },
                                                     {
                                                         "bits[1]:0",
                                                     },
                                                 }));

  static constexpr std::string_view expected_error =
      "In evaluated unopt IR (JIT), at position 1 channel 'sample__result' has "
      "value u32:42. However, in interpreted DSLX, the value is u32:43.";
  EXPECT_THAT(
      runner.Run(Sample(std::string(kProcCounterDSLX), options, args_batch,
                        /*ir_channel_names=*/{"sample__enable_counter"})),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(expected_error)));
  XLS_ASSERT_OK_AND_ASSIGN(std::string exception,
                           GetFileContents(GetTempPath() / "exception.txt"));
  EXPECT_THAT(exception, HasSubstr(expected_error));
}

}  // namespace
}  // namespace xls
