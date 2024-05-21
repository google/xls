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

#include <algorithm>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/dslx/mangle.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass_pipeline.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

// Tests that DSLX constructs are properly optimized. This test is in C++
// because the complicated bit (matching of the IR) use a C++ API.
class DslxOptimizationTest : public IrTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<VerifiedPackage>> DslxToIr(
      std::string_view dslx) {
    XLS_ASSIGN_OR_RETURN(TempFile dslx_temp, TempFile::CreateWithContent(dslx));
    XLS_ASSIGN_OR_RETURN(
        std::filesystem::path ir_converter_main_path,
        GetXlsRunfilePath("xls/dslx/ir_convert/ir_converter_main"));
    std::pair<std::string, std::string> stdout_stderr;
    XLS_ASSIGN_OR_RETURN(
        stdout_stderr,
        SubprocessResultToStrings(SubprocessErrorAsStatus(InvokeSubprocess(
            {ir_converter_main_path.string(), dslx_temp.path().string()}))));
    return ParsePackage(stdout_stderr.first);
  }

  // Returns the number of operations with one of the given opcodes in the
  // function.
  int64_t OpCount(FunctionBase* function_base, absl::Span<const Op> ops) {
    int64_t count = 0;
    for (Node* node : function_base->nodes()) {
      if (std::find(ops.begin(), ops.end(), node->op()) != ops.end()) {
        ++count;
      }
    }
    return count;
  }

  // Returns true if the given IR function has a node with the given op.
  bool HasOp(FunctionBase* function_base, Op op) {
    return OpCount(function_base, {op}) != 0;
  }
};

TEST_F(DslxOptimizationTest, StdFindIndexOfLiteralArray) {
  std::string input = R"(import std;

const A = u32[3]:[10, 20, 30];

fn main(i: u32) -> (bool, u32) {
  std::find_index(A, i)
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package, DslxToIr(input));
  XLS_ASSERT_OK(RunOptimizationPassPipeline(package.get()).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string mangled_name,
      dslx::MangleDslxName(package->name(), "main",
                           dslx::CallingConvention::kTypical));
  XLS_ASSERT_OK(package->SetTopByName(mangled_name));
  Function* entry = package->GetTop().value()->AsFunctionOrDie();
  VLOG(1) << package->DumpIr();
  // Verify that no ArrayIndex or ArrayUpdate operations exist in the IR.
  // TODO(b/159035667): The optimized IR is much more complicated than it should
  // be. When optimizations have been added which simplify the IR, add stricter
  // tests here.
  EXPECT_FALSE(HasOp(entry, Op::kArrayIndex));
  EXPECT_FALSE(HasOp(entry, Op::kArrayUpdate));
  EXPECT_FALSE(HasOp(entry, Op::kReverse));
}

TEST_F(DslxOptimizationTest, AttributeNamePropagation) {
  std::string input = R"(
struct S {
  zub: u8,
  qux: u8,
}

fn get_zub(x: S) -> u8 {
  x.zub
}

fn get_qux(y: S) -> u8 {
  y.qux
}

fn pass_S(z: S) -> S {
  z
}

fn main(foo: S, foo_bar: S) -> u8 {
  get_zub(foo) + get_qux(pass_S(pass_S(foo_bar)))
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package, DslxToIr(input));
  XLS_ASSERT_OK(RunOptimizationPassPipeline(package.get()).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string mangled_name,
      dslx::MangleDslxName(package->name(), "main",
                           dslx::CallingConvention::kTypical));
  XLS_ASSERT_OK(package->SetTopByName(mangled_name));
  Function* entry = package->GetTop().value()->AsFunctionOrDie();
  EXPECT_THAT(entry->return_value(),
              m::Add(m::Name("foo_zub"), m::Name("foo_bar_qux")));
}

TEST_F(DslxOptimizationTest, UpdateSliceOfWideVector) {
  std::string input = R"(
pub fn make_mask
  <N : u32, B : u32, N_PLUS_1: u32 = {N + u32:1}, MAX_N_B:u32 = {if N > B { N } else { B }}>
  (num_ones : uN[B])
  -> uN[N] {
  let num_bits_clamped =
    if (num_ones as uN[MAX_N_B]) > (N as uN[MAX_N_B]) {
      (N as uN[MAX_N_B])
    } else {
      num_ones as uN[MAX_N_B]
    };
  let wider = (uN[N_PLUS_1]:1 << (num_bits_clamped as uN[N_PLUS_1]))
              - uN[N_PLUS_1]:1;
  wider as uN[N]
}

pub fn update_slice<N : u32, NUpdate : u32>(
  original : uN[N],
  offset : u32,
  length : u32,
  update : uN[NUpdate])
  -> uN[N] {
  let ms_shift = offset + length;
  let ms_mask = make_mask<N>(N - ms_shift) << (ms_shift as uN[N]);
  let msbits = original & ms_mask;

  let lsbits = original & make_mask<N>(offset);

  let update_clamped = (update as uN[N]) & make_mask<N>(length);
  let update_shifted = update_clamped << (offset as uN[N]);

  msbits | update_shifted | lsbits
}

// Update 32-bit wide slice of 320-bit vector. The slice is in one of 10 slots
// as indicated by idx.
fn main(idx: u4, update: u32, original: bits[320]) -> bits[320] {
  let offset: u32 = idx as u32 * u32:32;
  let length: u32 = u32: 32;
  update_slice(original, offset, length, update)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package, DslxToIr(input));
  XLS_ASSERT_OK(RunOptimizationPassPipeline(package.get()).status());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string mangled_name,
      dslx::MangleDslxName(package->name(), "main",
                           dslx::CallingConvention::kTypical));
  XLS_ASSERT_OK(package->SetTopByName(mangled_name));
  Function* entry = package->GetTop().value()->AsFunctionOrDie();

  // The optimized block could look like a concat of selects:
  //
  //   { idx == 9 ? update : original[288:319],
  //     idx == 8 ? update : original[256:277],
  //     ....
  //     idx == 0 ? update : original[0:31] }
  //
  // However, the optimizer is not there yet. Instead check that some of the ops
  // are eliminated.
  //
  // All compares should be eliminated.
  EXPECT_EQ(OpCount(entry, {Op::kUGt, Op::kUGe, Op::kULt, Op::kULe}), 0);

  // The original has five shifts. Verify that only four remain.
  EXPECT_EQ(OpCount(entry, {Op::kShll, Op::kDecode, Op::kShrl, Op::kShra}), 4);
}

TEST_F(DslxOptimizationTest, ReceiveZeroDefaultValue) {
  std::string input = R"(
proc main {
  in_ch: chan<u32> in;

  init { u32:42 }

  config(ch: chan<u32> in) {
    (ch, )
  }

  next (state: u32) {
    let (tok, data) = recv_if(join(), in_ch, state == u32:0, u32:0);
    data
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package, DslxToIr(input));
  Proc* main = package->procs().front().get();

  // A select is added to handle the default value. Verify it is removed.
  EXPECT_EQ(OpCount(main, {Op::kSel}), 1);
  XLS_ASSERT_OK(RunOptimizationPassPipeline(package.get()).status());
  // Selects with zero can be simplified to ANDs.
  EXPECT_EQ(OpCount(main, {Op::kSel, Op::kAnd}), 0);
}

TEST_F(DslxOptimizationTest, ReceiveZeroDefaultValueCompoundType) {
  std::string input = R"(
proc main {
  in_ch: chan<(u32, u16, u8)> in;

  init { (u32:0, u16:0, u8:0) }

  config(ch: chan<(u32, u16, u8)> in) {
    (ch, )
  }

  next (state: (u32, u16, u8)) {
    let (tok, data) = recv_if(join(), in_ch, state.0 == u32:0, (u32:0, u16:0, u8:0));
    data
  }
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package, DslxToIr(input));
  Proc* main = package->procs().front().get();

  // A select is added to handle the default value. Verify it is removed.
  EXPECT_EQ(OpCount(main, {Op::kSel}), 1);
  XLS_ASSERT_OK(RunOptimizationPassPipeline(package.get()).status());
  // Selects with zero can be simplified to ANDs.
  EXPECT_EQ(OpCount(main, {Op::kSel, Op::kAnd}), 0);
}

}  // namespace
}  // namespace xls
