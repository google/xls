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

#include "xls/dslx/run_routines/run_routines.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <ctime>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/dslx/bytecode/bytecode.h"
#include "xls/dslx/bytecode/bytecode_cache.h"
#include "xls/dslx/bytecode/bytecode_emitter.h"
#include "xls/dslx/bytecode/bytecode_interpreter.h"
#include "xls/dslx/bytecode/bytecode_interpreter_options.h"
#include "xls/dslx/bytecode/proc_hierarchy_interpreter.h"
#include "xls/dslx/command_line_utils.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/error_printer.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_utils.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/ir_convert/convert_options.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/mangle.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/test_xml.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/interpreter/random_value.h"
#include "xls/ir/bits.h"
#include "xls/ir/events.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_translator.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {
// A few constants relating to the number of spaces to use in text formatting
// our test-runner output.
constexpr int kUnitSpaces = 7;
constexpr int kQuickcheckSpaces = 15;

// Helper routine for handling an error that occurs as the result of a test
// execution. Prints the error and that the test failed to stderr. Adds the test
// case to the accumulated test result data in `result`.
//
// Generally we expect that errors should be positional, but we will
// present it as an internal error if it is not instead of crashing because it
// results in better UX at this level of handling.
//
// Precondition: status must not be OK.
//
// Args:
// - result: the test result data to update
// - status: the status, e.g. a positional error that resulted from running
// - test_name: the name of the test (note that these are tests like a
// `--test_filter` flag would target)
// - start_pos: the position of the test construct
// - start: the time this particular test started
// - duration: the duration of the test
// - is_quickcheck: whether this is a quickcheck test
// - file_table: the file table to use for error reporting
// - vfs: the virtual file system to use for error reporting
void HandleError(TestResultData& result, const absl::Status& status,
                 std::string_view test_name, const Pos& start_pos,
                 const absl::Time& start, const absl::Duration& duration,
                 bool is_quickcheck, FileTable& file_table,
                 VirtualizableFilesystem& vfs) {
  CHECK(!status.ok()) << "HandleError called with status that is OK";
  VLOG(1) << "Handling error; status: " << status
          << " test_name: " << test_name;
  absl::StatusOr<PositionalErrorData> data =
      GetPositionalErrorData(status, std::nullopt, file_table);

  std::string one_liner;
  std::string suffix;
  if (data.ok()) {
    CHECK_OK(PrintPositionalError(data->span, data->GetMessageWithType(),
                                  std::cerr, PositionalErrorColor::kErrorColor,
                                  file_table, vfs));
    one_liner = data->GetMessageWithType();
  } else {
    // If we can't extract positional data we log the error and put the error
    // status into the "failed" prompted.
    LOG(ERROR) << "Internal error -- test " << test_name
               << " failed with a status that did not have DSLX position data: "
               << status;
    suffix = absl::StrCat(": internal error: ", status.ToString());
    one_liner = suffix;
  }

  // Add to test tracking data.
  result.AddTestCase(
      test_xml::TestCase{.name = std::string(test_name),
                         .file = std::string{start_pos.GetFilename(file_table)},
                         .line = start_pos.GetHumanLineno(),
                         .status = test_xml::RunStatus::kRun,
                         .result = test_xml::RunResult::kCompleted,
                         .time = duration,
                         .timestamp = start,
                         .failure = test_xml::Failure{.message = one_liner}});

  std::string spaces((is_quickcheck ? kQuickcheckSpaces : kUnitSpaces), ' ');
  std::cerr << absl::StreamFormat("[ %sFAILED ] %s%s", spaces, test_name,
                                  suffix)
            << "\n";
};

absl::Status RunDslxTestFunction(ImportData* import_data, TypeInfo* type_info,
                                 const Module* module, TestFunction* tf,
                                 const BytecodeInterpreterOptions& options) {
  auto cache = std::make_unique<BytecodeCache>();
  import_data->SetBytecodeCache(std::move(cache));
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<BytecodeFunction> bf,
      BytecodeEmitter::Emit(
          import_data, type_info, tf->fn(), std::nullopt,
          BytecodeEmitterOptions{.format_preference =
                                     options.format_preference()}));
  return BytecodeInterpreter::Interpret(import_data, bf.get(), /*args=*/{},
                                        /*channel_manager=*/std::nullopt,
                                        options)
      .status();
}

absl::Status RunDslxTestProc(ImportData* import_data, TypeInfo* type_info,
                             const Module* module, TestProc* tp,
                             const BytecodeInterpreterOptions& options) {
  auto cache = std::make_unique<BytecodeCache>();
  import_data->SetBytecodeCache(std::move(cache));

  XLS_ASSIGN_OR_RETURN(TypeInfo * ti,
                       type_info->GetTopLevelProcTypeInfo(tp->proc()));

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<ProcHierarchyInterpreter> hierarchy_interpreter,
      ProcHierarchyInterpreter::Create(import_data, ti, tp->proc(), options));

  // There should be a single top config argument: the terminator
  // channel. Determine the actual channel object.
  XLS_RET_CHECK_EQ(hierarchy_interpreter->InterfaceArgs().size(), 1);
  std::string terminal_channel_name =
      std::string{hierarchy_interpreter->GetInterfaceChannelName(0)};
  InterpValueChannel& terminal_channel =
      hierarchy_interpreter->GetInterfaceChannel(0);

  // Run until a single output appears in the terminal channel.
  absl::Status status =
      hierarchy_interpreter->TickUntilOutput({{terminal_channel_name, 1}})
          .status();
  std::optional<std::string> fail_label_opt =
      GetAssertionLabelFromError(status);
  if (!status.ok() && tp->expected_fail_label().has_value() &&
      fail_label_opt.has_value()) {
    if (*fail_label_opt == *tp->expected_fail_label()) {
      return absl::OkStatus();
    }
    return FailureErrorStatus(
        tp->proc()->span(),
        absl::StrFormat("Proc failed on '%s', but expected to fail on '%s'",
                        *fail_label_opt, *tp->expected_fail_label()),
        import_data->file_table());
  }
  XLS_RETURN_IF_ERROR(status);

  InterpValue ret_val = terminal_channel.Read();
  XLS_RET_CHECK(ret_val.IsBool());
  if (!ret_val.IsTrue()) {
    return FailureErrorStatus(tp->proc()->span(),
                              "Proc reported failure upon exit.",
                              import_data->file_table());
  }
  return absl::OkStatus();
}

// Pass that adds any asserts in a qc function to the goal and removes the
// assert nodes.
class QuickCheckProveAssertsNotFiredPass final
    : public OptimizationFunctionBasePass {
 public:
  QuickCheckProveAssertsNotFiredPass()
      : OptimizationFunctionBasePass("quickcheck-assert-suplement",
                                     "add asserts to quickcheck goal") {}
  ~QuickCheckProveAssertsNotFiredPass() final = default;

 protected:
  absl::StatusOr<bool> RunOnFunctionBaseInternal(
      FunctionBase* fb, const OptimizationPassOptions& options,
      PassResults* results, OptimizationContext& context) const override {
    XLS_RET_CHECK(fb->IsFunction()) << "not a qc function";
    xls::Function* f = fb->AsFunctionOrDie();
    XLS_RET_CHECK(f->return_value()->GetType()->IsBits())
        << "not a qc function";
    XLS_RET_CHECK_EQ(f->return_value()->GetType()->GetFlatBitCount(), 1)
        << "not a qc function";
    std::vector<Node*> ret = {f->return_value()};
    for (Node* n : context.ReverseTopoSort(f)) {
      if (n->Is<Assert>()) {
        Assert* a = n->As<Assert>();
        XLS_RETURN_IF_ERROR(a->ReplaceUsesWith(a->token()));
        ret.push_back(a->condition());
        XLS_RETURN_IF_ERROR(fb->RemoveNode(a));
      }
    }

    if (ret.size() == 1) {
      return false;
    }

    XLS_ASSIGN_OR_RETURN(
        Node * new_ret,
        fb->MakeNodeWithName<NaryOp>(f->return_value()->loc(), ret, Op::kAnd,
                                     "result_and_asserts_pass"));
    XLS_RETURN_IF_ERROR(f->set_return_value(new_ret));
    return true;
  }
};
}  // namespace

absl::StatusOr<std::unique_ptr<AbstractParsedTestRunner>>
DslxInterpreterTestRunner ::CreateTestRunner(ImportData* import_data,
                                             TypeInfo* type_info,
                                             Module* module) const {
  return std::make_unique<DslxInterpreterParsedTestRunner>(import_data,
                                                           type_info, module);
}
absl::StatusOr<RunResult> DslxInterpreterParsedTestRunner::RunTestFunction(
    std::string_view name, const BytecodeInterpreterOptions& options) {
  XLS_ASSIGN_OR_RETURN(TestFunction * tf, entry_module_->GetTest(name));
  return RunResult{.result = RunDslxTestFunction(import_data_, type_info_,
                                                 entry_module_, tf, options)};
}

absl::StatusOr<RunResult> DslxInterpreterParsedTestRunner::RunTestProc(
    std::string_view name, const BytecodeInterpreterOptions& options) {
  XLS_ASSIGN_OR_RETURN(TestProc * tp, entry_module_->GetTestProc(name));
  return RunResult{.result = RunDslxTestProc(import_data_, type_info_,
                                             entry_module_, tp, options)};
}

TestResultData::TestResultData(absl::Time start_time,
                               std::vector<test_xml::TestCase> test_cases)
    : start_time_(start_time), test_cases_(std::move(test_cases)) {}

int64_t TestResultData::GetFailedCount() const {
  return std::count_if(
      test_cases_.begin(), test_cases_.end(),
      [](const auto& test_case) { return test_case.failure.has_value(); });
}
int64_t TestResultData::GetSkippedCount() const {
  return std::count_if(
      test_cases_.begin(), test_cases_.end(), [](const auto& test_case) {
        return test_case.result == test_xml::RunResult::kFiltered;
      });
}

bool TestResultData::DidAnyFail() const {
  return std::any_of(
      test_cases_.begin(), test_cases_.end(),
      [](const auto& test_case) { return test_case.failure.has_value(); });
}

test_xml::TestSuites TestResultData::ToXmlSuites(
    std::string_view module_name) const {
  test_xml::TestCounts counts = {
      .tests = static_cast<int64_t>(test_cases_.size()),
      .failures = GetFailedCount(),
      .disabled = 0,
      .skipped = GetSkippedCount(),
      .errors = 0,
  };
  test_xml::TestSuites suites = {
      .counts = counts,
      .time = duration_,
      .timestamp = start_time_,
      .test_suites =
          {
              // We currently consider all the test cases inside of a single
              // file to be part of one suite.
              //
              // TODO(leary): 2024-02-08 We may want to break out quickcheck
              // tests vs
              // unit tests in the future.
              test_xml::TestSuite{
                  .name = absl::StrCat(module_name, " tests"),
                  .counts = counts,
                  .time = duration_,
                  .timestamp = start_time_,
                  .test_cases = test_cases_,
              },
          },
  };
  return suites;
}

static bool TestMatchesFilter(std::string_view test_name,
                              const RE2* test_filter) {
  if (test_filter == nullptr) {
    // All tests vacuously match the filter if there is no filter (i.e. we run
    // them all).
    return true;
  }
  return RE2::FullMatch(test_name, *test_filter);
}

// Populates a value from a given tuple type with a flat bit contents given by
// `i` -- this is useful for exhaustively iterating through a space to populate
// an aggregate value.
//
// Precondition: tuple_type->GetFlatBitCount() must be <= 64 so we have enough
// data in `i` to populate it.
static std::vector<Value> MakeFromUint64(xls::TupleType* tuple_type,
                                         uint64_t i) {
  // We turn the uint64_t contents into a bit vector of the flat bit count of
  // the tuple type and populate that tuple type from the bit string.
  InlineBitmap bitmap =
      InlineBitmap::FromWord(i, tuple_type->GetFlatBitCount());
  BitmapView view(bitmap);
  Value value = ZeroOfType(tuple_type);
  CHECK_OK(value.PopulateFrom(view));
  return value.GetElements().value();
};

static bool ValueIsValid(const Value& value, const Type& type) {
  if (type.IsEnum()) {
    const EnumType& enum_type = dynamic_cast<const EnumType&>(type);
    // Check the the bits value is contained within the enum's members.
    BitsType underlying_type(enum_type.is_signed(), enum_type.size());
    InterpValue interp_value =
        ValueToInterpValue(value, &underlying_type).value();
    return absl::c_contains(enum_type.AsEnum().members(), interp_value);
  }
  // All other values should be valid for their type by construction (i.e. we
  // would have generated them of the correct size and don't have any sparsity
  // in what is accepted).
  return true;
}

// Returns true if all the values are valid for the given type. Types like enums
// may not accept all valid values.
static bool ValuesAreValid(absl::Span<const Value> values,
                           absl::Span<const Type* const> types) {
  CHECK_EQ(values.size(), types.size());
  for (int64_t i = 0; i < values.size(); ++i) {
    const Value& value = values[i];
    const Type& type = *types[i];
    if (!ValueIsValid(value, type)) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<QuickCheckResults> DoQuickCheck(
    bool requires_implicit_token, dslx::FunctionType* dslx_fn_type,
    xls::Function* ir_function, std::string_view ir_name,
    AbstractRunComparator* run_comparator, int64_t seed,
    QuickCheckTestCases test_cases) {
  QuickCheckResults results;
  std::minstd_rand rng_engine(seed);
  xls::TupleType* ir_param_tuple = ir_function->package()->GetTupleType(
      ir_function->GetType()->parameters());

  int64_t num_tests;
  std::function<std::vector<Value>(int64_t)> make_arg_set;
  switch (test_cases.tag()) {
    case QuickCheckTestCasesTag::kExhaustive: {
      int64_t parameter_bit_count = ir_param_tuple->GetFlatBitCount();
      if (parameter_bit_count > 48) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Cannot run an exhaustive quickcheck for `%s` "
                            "because it has too large a parameter bit count; "
                            "got: %d",
                            ir_function->name(), parameter_bit_count));
      }
      num_tests = int64_t{1} << parameter_bit_count;
      make_arg_set = [&](int64_t i) {
        return MakeFromUint64(ir_param_tuple, i);
      };
      break;
    }
    case QuickCheckTestCasesTag::kCounted:
      num_tests =
          test_cases.count().value_or(QuickCheckTestCases::kDefaultTestCount);
      make_arg_set = [&](int64_t) {
        return RandomFunctionArguments(ir_function, rng_engine);
      };
      break;
  }

  std::unique_ptr<dslx::TokenType> token_type;
  std::unique_ptr<dslx::BitsType> bool_type;
  std::vector<const dslx::Type*> dslx_param_types;
  if (requires_implicit_token) {
    token_type = std::make_unique<dslx::TokenType>();
    bool_type = std::make_unique<dslx::BitsType>(false, 1);
    dslx_param_types.push_back(token_type.get());
    dslx_param_types.push_back(bool_type.get());
  }
  for (const std::unique_ptr<dslx::Type>& type : dslx_fn_type->params()) {
    dslx_param_types.push_back(type.get());
  }

  // Check that, after we've accounted for the potential implicit-token calling
  // convention, the number of IR function parameters and DSLX parameters line
  // up.
  XLS_RET_CHECK_EQ(ir_param_tuple->size(), dslx_param_types.size())
      << "IR param tuple size should match DSLX param types size";

  for (int i = 0; i < num_tests; i++) {
    {
      std::vector<Value> arg_set = make_arg_set(i);
      if (!ValuesAreValid(arg_set, dslx_param_types)) {
        // Note: if we reject an argument set, it counts as a test case -- this
        // makes sense for exhaustive mode but less sense for randomized mode,
        // we may want those to operate slightly differently.
        continue;
      }
      results.arg_sets.push_back(std::move(arg_set));
    }

    // TODO(https://github.com/google/xls/issues/506): 2021-10-15
    // Assertion failures should work out, but we should consciously decide
    // if/how we want to dump traces when running QuickChecks (always, for
    // failures, flag-controlled, ...).
    absl::Span<const Value> this_arg_set = results.arg_sets.back();
    XLS_ASSIGN_OR_RETURN(xls::Value result,
                         DropInterpreterEvents(run_comparator->RunIrFunction(
                             ir_name, ir_function, this_arg_set)));

    // In the case of an implicit token signature we get (token, bool) as the
    // result of the quickcheck'd function, so we unbox the boolean here.
    if (result.IsTuple()) {
      result = result.elements()[1];
      XLS_RET_CHECK(result.IsBits());
    }

    XLS_RET_CHECK(result.IsBits())
        << "quickcheck properties must return `bool`, should be validated by "
           "type checking; got: "
        << result;

    results.results.push_back(result);

    if (result.IsAllZeros()) {
      // We were able to falsify the xls_function (predicate), bail out early
      // and present this evidence.
      break;
    }
  }

  return results;
}

struct QuickcheckIrFn {
  std::string ir_name;
  xls::Function* ir_function;
  CallingConvention calling_convention;
};

static absl::StatusOr<QuickcheckIrFn> FindQuickcheckIrFn(Function* dslx_fn,
                                                         Package* ir_package) {
  // First we try to get the version of the function that doesn't need a token.
  XLS_ASSIGN_OR_RETURN(
      std::string ir_name,
      MangleDslxName(dslx_fn->owner()->name(), dslx_fn->identifier(),
                     CallingConvention::kTypical,
                     dslx_fn->GetFreeParametricKeySet()));
  std::optional<xls::Function*> maybe_ir_function =
      ir_package->TryGetFunction(ir_name);
  if (maybe_ir_function.has_value()) {
    return QuickcheckIrFn{ir_name, maybe_ir_function.value(),
                          CallingConvention::kTypical};
  }

  XLS_ASSIGN_OR_RETURN(
      ir_name, MangleDslxName(dslx_fn->owner()->name(), dslx_fn->identifier(),
                              CallingConvention::kImplicitToken,
                              dslx_fn->GetFreeParametricKeySet()));
  maybe_ir_function = ir_package->TryGetFunction(ir_name);
  if (maybe_ir_function.has_value()) {
    return QuickcheckIrFn{ir_name, maybe_ir_function.value(),
                          CallingConvention::kImplicitToken};
  }
  return absl::InternalError(
      absl::StrFormat("Could not find DSLX quickcheck function `%s` in IR "
                      "package `%s`; available IR functions: [%s]",
                      dslx_fn->identifier(), ir_package->name(),
                      absl::StrJoin(ir_package->GetFunctionNames(), ", ")));
}

static absl::Status RunQuickCheck(AbstractRunComparator* run_comparator,
                                  Package* ir_package, QuickCheck* quickcheck,
                                  TypeInfo* type_info, int64_t seed) {
  // Note: DSLX function.
  dslx::Function* dslx_fn = quickcheck->fn();

  XLS_ASSIGN_OR_RETURN(dslx::FunctionType * dslx_fn_type,
                       type_info->GetItemAs<dslx::FunctionType>(dslx_fn));

  // Validate the return type is a bool, we rely on that assumption here.
  const Type& return_type = dslx_fn_type->return_type();
  std::optional<BitsLikeProperties> bits_like_properties =
      GetBitsLike(return_type);
  XLS_RET_CHECK(bits_like_properties.has_value())
      << "quickcheck properties must return `bool`, should be validated by "
         "type checking";
  XLS_RET_CHECK(IsKnownU1(bits_like_properties.value()))
      << "quickcheck properties must return `bool`, should be validated by "
         "type checking";

  XLS_ASSIGN_OR_RETURN(QuickcheckIrFn qc_fn,
                       FindQuickcheckIrFn(dslx_fn, ir_package));

  XLS_ASSIGN_OR_RETURN(
      QuickCheckResults qc_results,
      DoQuickCheck(
          qc_fn.calling_convention == CallingConvention::kImplicitToken,
          dslx_fn_type, qc_fn.ir_function, qc_fn.ir_name, run_comparator, seed,
          quickcheck->test_cases()));

  // Extract the (inputs, outputs) from the results.
  const auto& [inputs, outputs] = qc_results;
  XLS_RET_CHECK(inputs.size() == outputs.size())
      << "inputs and outputs must have the same size";

  if (outputs.empty()) {
    // If we have a value like an empty enum we'll reject all samples, so we
    // want to make a reasonable error message for that case.
    return FailureErrorStatus(
        dslx_fn->span(),
        absl::StrFormat("quickcheck of `%s` rejected all input samples",
                        dslx_fn->identifier()),
        *dslx_fn->owner()->file_table());
  }

  XLS_ASSIGN_OR_RETURN(Bits last_result, outputs.back().GetBitsWithStatus());
  if (!last_result.IsZero()) {
    // Did not find a falsifying example.
    return absl::OkStatus();
  }

  const std::vector<Value>& last_ir_argset = inputs.back();
  const std::vector<std::unique_ptr<Type>>& dslx_params =
      dslx_fn_type->params();

  // If we're using the implicit-token calling convention, we have two extra
  // arguments at the start of the IR function: the token and activation
  // boolean.
  int64_t ir_args_start_index =
      qc_fn.calling_convention == CallingConvention::kImplicitToken ? 2 : 0;

  std::vector<InterpValue> dslx_argset;
  for (int64_t i = 0; i < dslx_params.size(); ++i) {
    const Type& arg_type = *dslx_params[i];
    const Value& value = last_ir_argset[ir_args_start_index + i];
    XLS_ASSIGN_OR_RETURN(InterpValue interp_value,
                         ValueToInterpValue(value, &arg_type));
    dslx_argset.push_back(interp_value);
  }

  std::string dslx_argset_str = absl::StrJoin(
      dslx_argset, ", ", [](std::string* out, const InterpValue& v) {
        absl::StrAppend(out, v.ToString());
      });
  return FailureErrorStatus(
      dslx_fn->span(),
      absl::StrFormat("Found falsifying example after %d tests: [%s]",
                      outputs.size(), dslx_argset_str),
      *dslx_fn->owner()->file_table());
}

static absl::Status RunQuickChecksIfJitEnabled(
    const RE2* test_filter, Module* entry_module, TypeInfo* type_info,
    AbstractRunComparator* run_comparator, Package* ir_package,
    std::optional<int64_t> seed, TestResultData& result,
    VirtualizableFilesystem& vfs) {
  if (run_comparator == nullptr) {
    // TODO(leary): 2024-02-08 Note that this skips /all/ the quickchecks so we
    // don't make an entry for it right now in the test XML.
    std::cerr << "[ SKIPPING QUICKCHECKS  ] (JIT is disabled)" << "\n";
    return absl::OkStatus();
  }
  if (!seed.has_value()) {
    // Note: we *want* to *provide* non-determinism by default. See
    // https://abseil.io/docs/cpp/guides/random#stability-of-generated-sequences
    // for rationale.
    seed = static_cast<int64_t>(getpid()) * static_cast<int64_t>(time(nullptr));
  }
  FileTable& file_table = *entry_module->file_table();
  bool any_quicktest_run = false;
  for (QuickCheck* quickcheck : entry_module->GetQuickChecks()) {
    const std::string& quickcheck_name = quickcheck->identifier();
    const Pos& start_pos = quickcheck->span().start();
    const absl::Time test_case_start = absl::Now();
    if (!TestMatchesFilter(quickcheck_name, test_filter)) {
      auto test_case_end = absl::Now();
      result.AddTestCase(test_xml::TestCase{
          .name = quickcheck_name,
          .file = std::string{start_pos.GetFilename(file_table)},
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kFiltered,
          .time = test_case_end - test_case_start,
          .timestamp = test_case_start});
      continue;
    }

    if (!any_quicktest_run) {
      // Only print the SEED if there is actually a test that is executed.
      std::cerr << absl::StreamFormat("[ SEED %*d ]\n", kQuickcheckSpaces + 1,
                                      *seed);
      any_quicktest_run = true;
    }
    std::cerr << "[ RUN QUICKCHECK        ] " << quickcheck_name
              << " cases: " << quickcheck->test_cases().ToString() << "\n";
    const absl::Status status =
        RunQuickCheck(run_comparator, ir_package, quickcheck, type_info, *seed);
    const absl::Duration duration = absl::Now() - test_case_start;
    if (!status.ok()) {
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  duration, /*is_quickcheck=*/true, file_table, vfs);
    } else {
      result.AddTestCase(test_xml::TestCase{
          .name = quickcheck_name,
          .file = std::string{start_pos.GetFilename(file_table)},
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kCompleted,
          .time = duration,
          .timestamp = test_case_start});
      std::cerr << "[                    OK ] " << quickcheck_name << "\n";
    }
  }
  std::cerr << absl::StreamFormat(
                   "[=======================] %d quickcheck(s) ran.",
                   entry_module->GetQuickChecks().size())
            << "\n";
  return absl::OkStatus();
}

absl::StatusOr<ParseAndProveResult> ParseAndProve(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndProveOptions& options) {
  const absl::Time parse_and_prove_start = absl::Now();
  TestResultData result(parse_and_prove_start, /*test_cases=*/{});

  std::unique_ptr<VirtualizableFilesystem> vfs;
  if (options.vfs_factory != nullptr) {
    vfs = options.vfs_factory();
  } else {
    vfs = std::make_unique<RealFilesystem>();
  }
  const ParseAndTypecheckOptions& parse_and_typecheck_options =
      options.parse_and_typecheck_options;

  auto import_data =
      CreateImportData(parse_and_typecheck_options.dslx_stdlib_path,
                       parse_and_typecheck_options.dslx_paths,
                       parse_and_typecheck_options.warnings, std::move(vfs));
  FileTable& file_table = import_data.file_table();
  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(program, filename, module_name, &import_data);
  if (!tm.ok()) {
    if (TryPrintError(tm.status(), file_table, import_data.vfs())) {
      result.Finish(TestResult::kParseOrTypecheckError,
                    absl::Now() - parse_and_prove_start);
      return ParseAndProveResult{.test_result_data = result};
    }
    return tm.status();
  }

  // If we're not executing, then we're just scanning for errors -- if warnings
  // are *not* errors, just elide printing them (or e.g. we'd show warnings for
  // files that had warnings suppressed at build time, which would gunk up build
  // logs unnecessarily.).
  if (parse_and_typecheck_options.warnings_as_errors) {
    PrintWarnings(tm->warnings, file_table, import_data.vfs());
  }

  if (parse_and_typecheck_options.warnings_as_errors &&
      !tm->warnings.warnings().empty()) {
    result.Finish(TestResult::kFailedWarnings,
                  absl::Now() - parse_and_prove_start);
    return ParseAndProveResult{.test_result_data = result};
  }

  Module* entry_module = tm->module;

  // We need to IR-convert the quickcheck property and then try to prove that
  // the return value is always true.
  absl::flat_hash_map<std::string, QuickCheck*> qcs =
      entry_module->GetQuickCheckByName();

  // Counter-examples map from failing test name -> counterexample values.
  absl::flat_hash_map<std::string, std::vector<Value>> counterexamples;

  for (const std::string& quickcheck_name :
       entry_module->GetQuickCheckNames()) {
    QuickCheck* quickcheck = qcs.at(quickcheck_name);
    const Pos& start_pos = quickcheck->span().start();
    Function* f = quickcheck->fn();
    VLOG(1) << "Found quickcheck function: " << f->identifier();

    auto test_case_start = absl::Now();

    if (!TestMatchesFilter(quickcheck_name, options.test_filter)) {
      auto test_case_end = absl::Now();
      result.AddTestCase(test_xml::TestCase{
          .name = quickcheck_name,
          .file = std::string{start_pos.GetFilename(file_table)},
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kFiltered,
          .time = test_case_end - test_case_start,
          .timestamp = test_case_start});
      continue;
    }
    std::cerr << "[ RUN QUICKCHECK        ] " << quickcheck_name << '\n';
    dslx::PackageConversionData conv{
        .package = std::make_unique<Package>(entry_module->name())};
    Package& package = *conv.package;

    // Helper that prevents us from mishandling various errors in the routine
    // via single use point.
    auto handle_if_error = [&](absl::Status status) {
      if (status.ok()) {
        return false;
      }
      HandleError(result, status, quickcheck_name, start_pos, test_case_start,
                  absl::Now() - test_case_start, /*is_quickcheck=*/true,
                  file_table, import_data.vfs());
      return true;
    };

    const absl::Status convert_status = ConvertOneFunctionIntoPackage(
        f, &import_data,
        /*parametric_env=*/nullptr, ConvertOptions{}, &conv);
    if (handle_if_error(convert_status)) {
      continue;
    }

    // Note: we need this to eliminate unoptimized IR constructs that are not
    // currently handled for translation; e.g. bounded-for-loops, asserts and
    // non-inlined function calls.
    auto pipeline = CreateOptimizationPassPipeline();

    // By the time this pass executes only the single top function is left.
    // Strip any remaining asserts from it and 'and' them to the quickcheck
    // goal.
    pipeline->Add<QuickCheckProveAssertsNotFiredPass>();
    pipeline->Add<DeadCodeEliminationPass>();
    PassResults results;
    OptimizationContext ctx;
    const absl::Status opt_status =
        pipeline->Run(conv.package.get(), {}, &results, ctx).status();
    if (handle_if_error(opt_status)) {
      continue;
    }

    const absl::StatusOr<std::string> ir_function_name = MangleDslxName(
        entry_module->name(), f->identifier(), CallingConvention::kTypical);
    if (handle_if_error(ir_function_name.status())) {
      continue;
    }

    const absl::StatusOr<xls::Function*> ir_function =
        package.GetFunction(*ir_function_name);
    if (handle_if_error(ir_function.status())) {
      continue;
    }

    VLOG(1) << "Found IR function: " << (*ir_function)->name();

    absl::StatusOr<solvers::z3::ProverResult> proven = solvers::z3::TryProve(
        *ir_function, (*ir_function)->return_value(),
        solvers::z3::Predicate::NotEqualToZero(), absl::InfiniteDuration());

    if (handle_if_error(proven.status())) {
      continue;
    }

    VLOG(1) << "Proven? "
            << (std::holds_alternative<solvers::z3::ProvenTrue>(*proven)
                    ? "true"
                    : "false");

    if (std::holds_alternative<solvers::z3::ProvenTrue>(*proven)) {
      absl::Time test_case_end = absl::Now();
      absl::Duration duration = test_case_end - test_case_start;
      result.AddTestCase(test_xml::TestCase{
          .name = std::string(quickcheck_name),
          .file = std::string{start_pos.GetFilename(file_table)},
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kCompleted,
          .time = duration,
          .timestamp = test_case_start,
      });
      std::cerr << "[                    OK ] " << quickcheck_name << "\n";
      continue;
    }

    const auto& proven_false = std::get<solvers::z3::ProvenFalse>(*proven);

    // Extract the counterexample, and collapse it back into sequential order.
    std::vector<Value> counterexample;
    using ParamValues = absl::flat_hash_map<const xls::Param*, Value>;
    XLS_ASSIGN_OR_RETURN(ParamValues counterexample_map,
                         proven_false.counterexample);
    for (const xls::Param* param : (*ir_function)->params()) {
      counterexample.push_back(counterexample_map[param]);
    }
    std::string one_liner =
        absl::StrCat("counterexample: ", absl::StrJoin(counterexample, ", "));
    const absl::Status proof_error_status =
        ProofErrorStatus(quickcheck->span(), one_liner, file_table);
    counterexamples[quickcheck_name] = std::move(counterexample);

    if (handle_if_error(proof_error_status)) {
      continue;
    }
  }

  result.Finish(TestResult::kSomeFailed, absl::Now() - parse_and_prove_start);
  std::cerr
      << absl::StreamFormat(
             "[=======================] %d test(s) ran; %d failed; %d skipped.",
             result.GetRanCount(), result.GetFailedCount(),
             result.GetSkippedCount())
      << '\n';

  result.Finish(
      result.DidAnyFail() ? TestResult::kSomeFailed : TestResult::kAllPassed,
      absl::Now() - parse_and_prove_start);
  return ParseAndProveResult{.test_result_data = std::move(result),
                             .counterexamples = std::move(counterexamples)};
}

absl::StatusOr<TestResultData> AbstractTestRunner::ParseAndTest(
    std::string_view program, std::string_view module_name,
    std::string_view filename, const ParseAndTestOptions& options) const {
  const absl::Time start = absl::Now();
  TestResultData result(start, /*test_cases=*/{});

  std::unique_ptr<VirtualizableFilesystem> vfs;
  if (options.vfs_factory != nullptr) {
    vfs = options.vfs_factory();
  } else {
    vfs = std::make_unique<RealFilesystem>();
  }
  const ParseAndTypecheckOptions& parse_and_typecheck_options =
      options.parse_and_typecheck_options;
  auto import_data =
      CreateImportData(parse_and_typecheck_options.dslx_stdlib_path,
                       parse_and_typecheck_options.dslx_paths,
                       parse_and_typecheck_options.warnings, std::move(vfs));
  FileTable& file_table = import_data.file_table();

  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(program, filename, module_name, &import_data, nullptr,
                        parse_and_typecheck_options.type_inference_version);
  if (!tm.ok()) {
    if (TryPrintError(tm.status(), import_data.file_table(),
                      import_data.vfs())) {
      result.Finish(TestResult::kParseOrTypecheckError, absl::Now() - start);
      return result;
    }
    return tm.status();
  }

  // If we're not executing, then we're just scanning for errors -- if warnings
  // are *not* errors, just elide printing them (or e.g. we'd show warnings for
  // files that had warnings suppressed at build time, which would gunk up build
  // logs unnecessarily.).
  if (options.execute || parse_and_typecheck_options.warnings_as_errors) {
    PrintWarnings(tm->warnings, import_data.file_table(), import_data.vfs());
  }

  if (parse_and_typecheck_options.warnings_as_errors &&
      !tm->warnings.warnings().empty()) {
    result.Finish(TestResult::kFailedWarnings, absl::Now() - start);
    return result;
  }

  // If not executing tests and quickchecks, then return vacuous success.
  if (!options.execute) {
    result.Finish(TestResult::kAllPassed, absl::Now() - start);
    return result;
  }

  Module* entry_module = tm->module;

  // If JIT comparisons are "on", we register a post-evaluation hook to compare
  // with the interpreter.
  std::unique_ptr<Package> ir_package;
  PostFnEvalHook post_fn_eval_hook;
  if (options.run_comparator != nullptr) {
    absl::StatusOr<dslx::PackageConversionData> ir_package_conversion_data =
        ConvertModuleToPackage(entry_module, &import_data,
                               options.convert_options);
    if (!ir_package_conversion_data.ok()) {
      if (TryPrintError(ir_package_conversion_data.status(),
                        import_data.file_table(), import_data.vfs())) {
        result.Finish(TestResult::kSomeFailed, absl::Now() - start);
        return result;
      }
      return xabsl::StatusBuilder(ir_package_conversion_data.status())
             << "Failed to convert input to IR for comparison. Consider "
                "turning off comparison with `--compare=none`: ";
    }
    ir_package = (*std::move(ir_package_conversion_data)).package;
    post_fn_eval_hook = [&ir_package, &import_data, &options](
                            const Function* f,
                            absl::Span<const InterpValue> args,
                            const ParametricEnv* parametric_env,
                            const InterpValue& got) -> absl::Status {
      XLS_RET_CHECK(f != nullptr);
      std::optional<bool> requires_implicit_token =
          import_data.GetRootTypeInfoForNode(f)
              .value()
              ->GetRequiresImplicitToken(*f);
      XLS_RET_CHECK(requires_implicit_token.has_value());
      return options.run_comparator->RunComparison(ir_package.get(),
                                                   *requires_implicit_token, f,
                                                   args, parametric_env, got);
    };
  }

  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<AbstractParsedTestRunner> runner,
      CreateTestRunner(&import_data, tm->type_info, entry_module));
  // Run unit tests.
  for (const std::string& test_name : entry_module->GetTestNames()) {
    auto test_case_start = absl::Now();
    ModuleMember* member = entry_module->FindMemberWithName(test_name).value();
    const Pos start_pos = GetPos(*member);

    if (!TestMatchesFilter(test_name, options.test_filter)) {
      auto test_case_end = absl::Now();
      result.AddTestCase(test_xml::TestCase{
          .name = test_name,
          .file = std::string{start_pos.GetFilename(file_table)},
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kFiltered,
          .time = test_case_end - test_case_start,
          .timestamp = test_case_start});
      continue;
    }

    std::cerr << "[ RUN UNITTEST  ] " << test_name << '\n';
    RunResult out;
    BytecodeInterpreterOptions interpreter_options;
    interpreter_options.post_fn_eval_hook(post_fn_eval_hook)
        .trace_hook(absl::bind_front(InfoLoggingTraceHook, file_table))
        .trace_channels(options.trace_channels)
        .trace_calls(options.trace_calls)
        .max_ticks(options.max_ticks)
        .format_preference(options.format_preference);
    if (std::holds_alternative<TestFunction*>(*member)) {
      XLS_ASSIGN_OR_RETURN(
          out, runner->RunTestFunction(test_name, interpreter_options));
    } else {
      XLS_ASSIGN_OR_RETURN(out,
                           runner->RunTestProc(test_name, interpreter_options));
    }
    auto test_case_end = absl::Now();

    if (out.result.ok()) {
      // Add to the tracking data.
      result.AddTestCase(test_xml::TestCase{
          .name = test_name,
          .file = std::string{start_pos.GetFilename(file_table)},
          .line = start_pos.GetHumanLineno(),
          .status = test_xml::RunStatus::kRun,
          .result = test_xml::RunResult::kCompleted,
          .time = test_case_end - test_case_start,
          .timestamp = test_case_start});
      std::cerr << "[            OK ]" << '\n';
    } else {
      HandleError(result, out.result, test_name, start_pos, test_case_start,
                  test_case_end - test_case_start,
                  /*is_quickcheck=*/false, file_table, import_data.vfs());
    }
  }

  std::cerr << absl::StreamFormat(
                   "[===============] %d test(s) ran; %d failed; %d skipped.",
                   result.GetRanCount(), result.GetFailedCount(),
                   result.GetSkippedCount())
            << '\n';

  // Run quickchecks, but only if the JIT is enabled.
  if (!entry_module->GetQuickChecks().empty()) {
    XLS_RETURN_IF_ERROR(RunQuickChecksIfJitEnabled(
        options.test_filter, entry_module, tm->type_info,
        options.run_comparator, ir_package.get(), options.seed, result,
        import_data.vfs()));
  }

  result.Finish(
      result.DidAnyFail() ? TestResult::kSomeFailed : TestResult::kAllPassed,
      absl::Now() - start);
  return result;
}

std::string_view TestResultToString(TestResult tr) {
  switch (tr) {
    case TestResult::kFailedWarnings:
      return "failed-warnings";
    case TestResult::kSomeFailed:
      return "some-failed";
    case TestResult::kAllPassed:
      return "all-passed";
    case TestResult::kParseOrTypecheckError:
      return "parse-or-typecheck-error";
  }
  LOG(FATAL) << "Invalid test result value: " << static_cast<int>(tr);
}

}  // namespace xls::dslx
