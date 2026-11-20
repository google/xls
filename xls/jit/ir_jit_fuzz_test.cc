// Copyright 2026 The XLS Authors
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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/interpreter/observer.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_flattening.h"
#include "xls/ir/verifier.h"
#include "xls/jit/function_base_jit.h"
#include "xls/jit/function_jit.h"
#include "xls/jit/ir_builder_visitor.h"
#include "xls/jit/jit_buffer.h"
#include "xls/jit/jit_callbacks.h"
#include "xls/jit/jit_evaluator_options.h"
#include "xls/jit/llvm_compiler.h"
#include "xls/jit/observer.h"

namespace xls {
namespace {

using ::testing::UnorderedElementsAreArray;

constexpr int64_t kArgSetCount = 10;
constexpr int64_t kMinOptLevel = 0;
constexpr int64_t kMaxOptLevel = LlvmCompiler::kDefaultOptLevel;

auto OptLevelDomain() {
  return fuzztest::InRange<int64_t>(kMinOptLevel, kMaxOptLevel);
}

std::string FormatArgSet(absl::Span<const Value> arg_set) {
  return absl::StrCat("[",
                      absl::StrJoin(arg_set, "; ",
                                    [](std::string* out, const Value& value) {
                                      absl::StrAppend(out, value.ToString());
                                    }),
                      "]");
}

// Evaluates `function` through the packed-bitvector entrypoint of `jit`.
absl::StatusOr<InterpreterResult<Value>> RunPackedJit(
    Function* function, FunctionJit* jit, absl::Span<const Value> args) {
  const JittedFunctionBase& jitted_function_base = jit->jitted_function_base();
  XLS_RET_CHECK(jitted_function_base.HasPackedFunction());

  std::vector<std::vector<uint8_t>> packed_args;
  packed_args.reserve(args.size());
  std::vector<const uint8_t*> arg_ptrs;
  arg_ptrs.reserve(args.size());
  for (const Value& arg : args) {
    packed_args.push_back(FlattenValueToBits(arg).ToBytes());
    arg_ptrs.push_back(packed_args.back().data());
  }

  Type* return_type = function->return_type();
  int64_t return_bit_count = return_type->GetFlatBitCount();
  std::vector<uint8_t> packed_output(CeilOfRatio(return_bit_count, int64_t{8}),
                                     0);
  uint8_t* output_ptrs[1] = {packed_output.data()};

  JitTempBuffer temp_buffer = jitted_function_base.CreateTempBuffer();
  InterpreterEvents events;
  InstanceContext instance_context = InstanceContext::CreateForFunc();
  std::optional<int64_t> continuation =
      jitted_function_base.RunPackedJittedFunction(
          arg_ptrs.data(), output_ptrs, temp_buffer.get_base_pointer(), &events,
          &instance_context, jit->runtime(), /*continuation_point=*/0);
  XLS_RET_CHECK(continuation.has_value());
  XLS_RET_CHECK_EQ(*continuation, 0);

  Bits output_bits = Bits::FromBytes(packed_output, return_bit_count);
  XLS_ASSIGN_OR_RETURN(Value result,
                       UnflattenBitsToValue(output_bits, return_type));
  return InterpreterResult<Value>{std::move(result), std::move(events)};
}

// Rewrites every `smulp`/`umulp` node `m` so downstream consumers observe
// `(0, m.0 + m.1)` while `m` itself is still compiled and executed.
//
// `smulp` and `umulp` intentionally use distinct partial-product offsets in the
// interpreter vs. the JIT (`MulpOffsetForSimulation`), so only the sum of the
// two tuple elements is invariant across runtimes.
absl::Status CanonicalizeMulpOutputs(Package* package) {
  for (const std::unique_ptr<Function>& func : package->functions()) {
    std::vector<Node*> mulp_nodes;
    for (Node* node : func->nodes()) {
      if (node->OpIn({Op::kSMulp, Op::kUMulp})) {
        mulp_nodes.push_back(node);
      }
    }
    for (Node* mul : mulp_nodes) {
      int64_t element_width =
          mul->GetType()->AsTupleOrDie()->element_type(0)->GetFlatBitCount();
      XLS_ASSIGN_OR_RETURN(Node * lhs_part,
                           func->MakeNode<TupleIndex>(mul->loc(), mul, 0));
      XLS_ASSIGN_OR_RETURN(Node * rhs_part,
                           func->MakeNode<TupleIndex>(mul->loc(), mul, 1));
      XLS_ASSIGN_OR_RETURN(
          Node * product_sum,
          func->MakeNode<BinOp>(mul->loc(), lhs_part, rhs_part, Op::kAdd));
      XLS_ASSIGN_OR_RETURN(
          Node * zero,
          func->MakeNode<Literal>(mul->loc(), Value(UBits(0, element_width))));
      XLS_ASSIGN_OR_RETURN(
          Node * canonical_tuple,
          func->MakeNode<Tuple>(mul->loc(),
                                std::vector<Node*>{zero, product_sum}));
      XLS_RETURN_IF_ERROR(mul->ReplaceUsesWith(
          canonical_tuple,
          [&](Node* user) { return user != lhs_part && user != rhs_part; }));
    }
  }
  return VerifyPackage(package);
}

void CheckJitMatchesInterpreter(
    const FuzzPackageWithArgs& fuzz_package_with_args, int64_t opt_level) {
  Package* package = fuzz_package_with_args.fuzz_package.p.get();
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetTopAsFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionJit> jit,
      FunctionJit::Create(function, EvaluatorOptions(),
                          JitEvaluatorOptions().set_opt_level(opt_level)));

  for (size_t i = 0; i < fuzz_package_with_args.arg_sets.size(); ++i) {
    const std::vector<Value>& args = fuzz_package_with_args.arg_sets[i];
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> interp_result,
                             InterpretFunction(function, args));
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> jit_result,
                             jit->Run(args));
    EXPECT_EQ(jit_result.value, interp_result.value)
        << "Unpacked JIT mismatch at opt_level=" << opt_level << " for input #"
        << i << "\n"
        << "Args: " << FormatArgSet(args) << "\n"
        << "IR:\n"
        << package->DumpIr();
    EXPECT_EQ(jit_result.events, interp_result.events)
        << "Unpacked JIT events mismatch at opt_level=" << opt_level
        << " for input #" << i << "\n"
        << "Args: " << FormatArgSet(args) << "\n"
        << "IR:\n"
        << package->DumpIr();

    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> packed_result,
                             RunPackedJit(function, jit.get(), args));
    EXPECT_EQ(packed_result.value, interp_result.value)
        << "Packed JIT mismatch at opt_level=" << opt_level << " for input #"
        << i << "\n"
        << "Args: " << FormatArgSet(args) << "\n"
        << "IR:\n"
        << package->DumpIr();
    EXPECT_EQ(packed_result.events, interp_result.events)
        << "Packed JIT events mismatch at opt_level=" << opt_level
        << " for input #" << i << "\n"
        << "Args: " << FormatArgSet(args) << "\n"
        << "IR:\n"
        << package->DumpIr();
  }
}

void JitMatchesInterpreter(FuzzPackageWithArgs fuzz_package_with_args,
                           int64_t opt_level) {
  RecordFuzzInfo(fuzz_package_with_args);
  CheckJitMatchesInterpreter(fuzz_package_with_args, opt_level);
}
FUZZ_TEST(IrJitFuzzTest, JitMatchesInterpreter)
    .WithDomains(PackageWithArgsDomainBuilder(kArgSetCount)
                     .WithoutOperations({Op::kSMulp, Op::kUMulp})
                     .Build(),
                 OptLevelDomain());

void JitWithObserverMatchesInterpreter(
    FuzzPackageWithArgs fuzz_package_with_args, int64_t opt_level) {
  RecordFuzzInfo(fuzz_package_with_args);
  Package* package = fuzz_package_with_args.fuzz_package.p.get();
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetTopAsFunction());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionJit> jit,
      FunctionJit::Create(function, EvaluatorOptions(),
                          JitEvaluatorOptions()
                              .set_opt_level(opt_level)
                              .set_include_observer_callbacks(true)));

  for (size_t i = 0; i < fuzz_package_with_args.arg_sets.size(); ++i) {
    const std::vector<Value>& args = fuzz_package_with_args.arg_sets[i];
    CollectingEvaluationObserver interp_observer;
    XLS_ASSERT_OK_AND_ASSIGN(
        InterpreterResult<Value> interp_result,
        InterpretFunction(function, args, EvaluatorOptions(),
                          &interp_observer));

    CollectingEvaluationObserver jit_observer;
    RuntimeEvaluationObserverAdapter adapter(
        &jit_observer,
        [](int64_t node_ptr) {
          return absl::bit_cast<Node*>(static_cast<intptr_t>(node_ptr));
        },
        jit->runtime());
    XLS_ASSERT_OK(jit->SetRuntimeObserver(&adapter));
    XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> jit_result,
                             jit->Run(args));
    jit->ClearRuntimeObserver();

    EXPECT_EQ(jit_result.value, interp_result.value)
        << "JIT with observer mismatch at opt_level=" << opt_level
        << " for input #" << i << "\n"
        << "Args: " << FormatArgSet(args) << "\n"
        << "IR:\n"
        << package->DumpIr();
    EXPECT_EQ(jit_result.events, interp_result.events)
        << "JIT with observer events mismatch at opt_level=" << opt_level
        << " for input #" << i << "\n"
        << "Args: " << FormatArgSet(args) << "\n"
        << "IR:\n"
        << package->DumpIr();

    for (const auto& [node, jit_values] : jit_observer.values()) {
      ASSERT_TRUE(interp_observer.values().contains(node))
          << "JIT observed node not evaluated by interpreter at opt_level="
          << opt_level << ": " << node->ToString();
      EXPECT_THAT(jit_values,
                  UnorderedElementsAreArray(interp_observer.values().at(node)))
          << "Observed node value mismatch on " << node->ToString()
          << " at opt_level=" << opt_level << " for input #" << i << "\n"
          << "Args: " << FormatArgSet(args) << "\n"
          << "IR:\n"
          << package->DumpIr();
    }
    for (const auto& [node, interp_values] : interp_observer.values()) {
      Function* node_func = node->function_base()->AsFunctionOrDie();
      if (ShouldMaterializeAtUse(node) && node_func->return_value() != node) {
        continue;
      }
      EXPECT_TRUE(jit_observer.values().contains(node))
          << "Interpreter evaluated node missing from JIT observer at "
             "opt_level="
          << opt_level << ": " << node->ToString() << " (saw "
          << interp_values.size() << " evaluations)\nIR:\n"
          << package->DumpIr();
    }
  }
}
FUZZ_TEST(IrJitFuzzTest, JitWithObserverMatchesInterpreter)
    .WithDomains(PackageWithArgsDomainBuilder(kArgSetCount)
                     .WithoutOperations({Op::kSMulp, Op::kUMulp})
                     .Build(),
                 OptLevelDomain());

void JitMulpMatchesInterpreter(FuzzPackageWithArgs fuzz_package_with_args,
                               int64_t opt_level) {
  RecordFuzzInfo(fuzz_package_with_args);
  XLS_ASSERT_OK(
      CanonicalizeMulpOutputs(fuzz_package_with_args.fuzz_package.p.get()));
  CheckJitMatchesInterpreter(fuzz_package_with_args, opt_level);
}
FUZZ_TEST(IrJitFuzzTest, JitMulpMatchesInterpreter)
    .WithDomains(PackageWithArgsDomainBuilder(kArgSetCount)
                     .WithOperations({Op::kSMulp, Op::kUMulp, Op::kAdd,
                                      Op::kTupleIndex, Op::kTuple, Op::kArray,
                                      Op::kArrayIndex, Op::kSel})
                     .NoClz()
                     .NoCtz()
                     .Build(),
                 OptLevelDomain());

TEST(IrJitFuzzTest, CompoundTypesAndInvokeMatchInterpreter) {
  constexpr std::string_view kIr = R"xls_ir(package FuzzTest

fn combine(x: bits[13], y: bits[13]) -> bits[13] {
  neg_x: bits[13] = neg(x)
  bias: bits[13] = literal(value=5)
  sum: bits[13] = add(neg_x, y)
  ret result: bits[13] = xor(sum, bias)
}

top fn FuzzTest(p: (bits[13], (), bits[7][3])) -> (bits[13], (), bits[7][3]) {
  elem0: bits[13] = tuple_index(p, index=0)
  empty: () = tuple_index(p, index=1)
  arr: bits[7][3] = tuple_index(p, index=2)
  idx0: bits[2] = literal(value=0)
  idx2: bits[2] = literal(value=2)
  a0: bits[7] = array_index(arr, indices=[idx0])
  a2: bits[7] = array_index(arr, indices=[idx2])
  a0_ext: bits[13] = zero_ext(a0, new_bit_count=13)
  a2_ext: bits[13] = sign_ext(a2, new_bit_count=13)
  c0: bits[13] = invoke(elem0, a0_ext, to_apply=combine)
  c1: bits[13] = invoke(c0, a2_ext, to_apply=combine)
  ret out: (bits[13], (), bits[7][3]) = tuple(c1, empty, arr)
}
)xls_ir";
  const std::vector<std::vector<Value>> arg_sets = {
      {Value::TupleOwned({
          Value(UBits(0, 13)),
          Value::Tuple({}),
          Value::ArrayOwned(
              {Value(UBits(0, 7)), Value(UBits(1, 7)), Value(UBits(2, 7))}),
      })},
      {Value::TupleOwned({
          Value(UBits(0x1abc, 13)),
          Value::Tuple({}),
          Value::ArrayOwned({Value(UBits(0x7f, 7)), Value(UBits(0x2a, 7)),
                             Value(UBits(0x55, 7))}),
      })},
      {Value::TupleOwned({
          Value(Bits::AllOnes(13)),
          Value::Tuple({}),
          Value::ArrayOwned({Value(UBits(0x40, 7)), Value(UBits(0, 7)),
                             Value(Bits::AllOnes(7))}),
      })},
  };
  for (int64_t opt_level = kMinOptLevel; opt_level <= kMaxOptLevel;
       ++opt_level) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(kIr));
    FuzzPackageWithArgs fuzz_package_with_args{
        .fuzz_package = FuzzPackage{.p = std::move(package)},
        .arg_sets = arg_sets,
    };
    CheckJitMatchesInterpreter(fuzz_package_with_args, opt_level);
    JitWithObserverMatchesInterpreter(std::move(fuzz_package_with_args),
                                      opt_level);
  }
}

TEST(IrJitFuzzTest, PartialProductOpsMatchInterpreter) {
  constexpr std::string_view kIr = R"xls_ir(package FuzzTest

top fn FuzzTest(lhs: bits[9], rhs: bits[14]) -> ((bits[23], bits[23]), (bits[17], bits[17])) {
  prod_s: (bits[23], bits[23]) = smulp(lhs, rhs)
  prod_u: (bits[17], bits[17]) = umulp(lhs, rhs)
  ret out: ((bits[23], bits[23]), (bits[17], bits[17])) = tuple(prod_s, prod_u)
}
)xls_ir";
  const std::vector<std::vector<Value>> arg_sets = {
      {Value(UBits(0, 9)), Value(UBits(0, 14))},
      {Value(UBits(17, 9)), Value(UBits(1023, 14))},
      {Value(SBits(-123, 9)), Value(SBits(-4567, 14))},
      {Value(Bits::AllOnes(9)), Value(Bits::AllOnes(14))},
  };
  for (int64_t opt_level = kMinOptLevel; opt_level <= kMaxOptLevel;
       ++opt_level) {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                             Parser::ParsePackage(kIr));
    JitMulpMatchesInterpreter(
        FuzzPackageWithArgs{
            .fuzz_package = FuzzPackage{.p = std::move(package)},
            .arg_sets = arg_sets,
        },
        opt_level);
  }
}

}  // namespace
}  // namespace xls
