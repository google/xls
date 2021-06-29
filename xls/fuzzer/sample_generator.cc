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

#include "xls/fuzzer/sample_generator.h"

#include "absl/strings/match.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/ir/bits_ops.h"

namespace xls {

using dslx::ArrayType;
using dslx::AstGenerator;
using dslx::AstGeneratorOptions;
using dslx::BitsType;
using dslx::ConcreteType;
using dslx::FunctionType;
using dslx::ImportData;
using dslx::InterpValue;
using dslx::InterpValueTag;
using dslx::TupleType;
using dslx::TypecheckedModule;

double RngState::RandomDouble() {
  std::uniform_real_distribution<double> d(0.0, 1.0);
  return d(rng_);
}

int64_t RngState::RandRange(int64_t limit) {
  std::uniform_int_distribution<int64_t> d(0, limit - 1);
  return d(rng_);
}

int64_t RngState::RandRangeBiasedTowardsZero(int64_t limit) {
  XLS_CHECK_GT(limit, 0);
  if (limit == 1) {  // Only one possible value.
    return 0;
  }
  std::array<double, 3> i = {{0, 0, static_cast<double>(limit)}};
  std::array<double, 3> w = {{0, 1, 0}};
  std::piecewise_linear_distribution<double> d(i.begin(), i.end(), w.begin());
  double triangular = d(rng_);
  int64_t result = static_cast<int64_t>(std::ceil(triangular)) - 1;
  XLS_CHECK_GE(result, 0);
  XLS_CHECK_LT(result, limit);
  return result;
}

static absl::StatusOr<InterpValue> GenerateBitValue(int64_t bit_count,
                                                    RngState* rng,
                                                    bool is_signed) {
  AstGenerator g(AstGeneratorOptions(), &rng->rng());
  Bits bits = g.ChooseBitPattern(bit_count);
  auto tag = is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  return InterpValue::MakeBits(tag, std::move(bits));
}

// Note: "unbiased" here refers to the fact we don't use the history of
// previously generated values, but just sample arbitrarily something for the
// given bit count of the bits type. You'll see other routines taking "prior" as
// a history to help prevent repetition that could hide bugs.
static absl::StatusOr<InterpValue> GenerateUnbiasedArgument(
    const BitsType& bits_type, RngState* rng) {
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, bits_type.size().GetAsInt64());
  return GenerateBitValue(bit_count, rng, bits_type.is_signed());
}

// Generates an argument value of the same type as the concrete type.
static absl::StatusOr<InterpValue> GenerateArgument(
    const ConcreteType& arg_type, RngState* rng,
    absl::Span<const InterpValue> prior) {
  if (auto* tuple_type = dynamic_cast<const TupleType*>(&arg_type)) {
    std::vector<InterpValue> members;
    for (const std::unique_ptr<ConcreteType>& t : tuple_type->members()) {
      XLS_ASSIGN_OR_RETURN(InterpValue member,
                           GenerateArgument(*t, rng, prior));
      members.push_back(member);
    }
    return InterpValue::MakeTuple(members);
  }
  if (auto* array_type = dynamic_cast<const ArrayType*>(&arg_type)) {
    std::vector<InterpValue> elements;
    const ConcreteType& element_type = array_type->element_type();
    XLS_ASSIGN_OR_RETURN(int64_t array_size, array_type->size().GetAsInt64());
    for (int64_t i = 0; i < array_size; ++i) {
      XLS_ASSIGN_OR_RETURN(InterpValue element,
                           GenerateArgument(element_type, rng, prior));
      elements.push_back(element);
    }
    return InterpValue::MakeArray(std::move(elements));
  }
  auto* bits_type = dynamic_cast<const BitsType*>(&arg_type);
  XLS_RET_CHECK(bits_type != nullptr);
  if (prior.empty() || rng->RandomDouble() < 0.5) {
    return GenerateUnbiasedArgument(*bits_type, rng);
  }

  // Try to mutate a prior argument. If it happens to not be a bits type that we
  // look at, then just generate an unbiased argument.
  int64_t index = rng->RandRange(prior.size());
  if (!prior[index].IsBits()) {
    return GenerateUnbiasedArgument(*bits_type, rng);
  }

  Bits to_mutate = prior[index].GetBitsOrDie();

  XLS_ASSIGN_OR_RETURN(const int64_t target_bit_count,
                       bits_type->size().GetAsInt64());
  if (target_bit_count > to_mutate.bit_count()) {
    XLS_ASSIGN_OR_RETURN(
        InterpValue addendum,
        GenerateBitValue(target_bit_count - to_mutate.bit_count(), rng,
                         /*is_signed=*/false));
    to_mutate = bits_ops::Concat({to_mutate, addendum.GetBitsOrDie()});
  } else {
    to_mutate = to_mutate.Slice(0, target_bit_count);
  }

  InlineBitmap bitmap = to_mutate.bitmap();
  XLS_RET_CHECK_EQ(bitmap.bit_count(), target_bit_count);
  int64_t mutation_count = rng->RandRangeBiasedTowardsZero(target_bit_count);

  for (int64_t i = 0; i < mutation_count; ++i) {
    // Pick a random bit and flip it.
    int64_t bitno = rng->RandRange(target_bit_count);
    bitmap.Set(bitno, !bitmap.Get(bitno));
  }

  bool is_signed = bits_type->is_signed();
  auto tag = is_signed ? InterpValueTag::kSBits : InterpValueTag::kUBits;
  return InterpValue::MakeBits(tag, Bits::FromBitmap(std::move(bitmap)));
}

absl::StatusOr<std::vector<InterpValue>> GenerateArguments(
    absl::Span<const ConcreteType* const> arg_types, RngState* rng) {
  std::vector<InterpValue> args;
  for (const ConcreteType* arg_type : arg_types) {
    XLS_RET_CHECK(arg_type != nullptr);
    XLS_ASSIGN_OR_RETURN(InterpValue arg,
                         GenerateArgument(*arg_type, rng, args));
    args.push_back(std::move(arg));
  }
  return args;
}

// Returns randomly generated arguments for running codegen.
//
// These arguments are flags which are passed to codegen_main for generating
// Verilog. Randomly chooses either a purely combinational module or a
// feed-forward pipeline of a randome length.
//
// Args:
//   use_system_verilog: Whether to use SystemVerilog.
//   rng: Random number generator state.
static std::vector<std::string> GenerateCodegenArgs(bool use_system_verilog,
                                                    RngState* rng) {
  std::vector<std::string> args;
  if (use_system_verilog) {
    args.push_back("--use_system_verilog");
  } else {
    args.push_back("--nouse_system_verilog");
  }
  if (rng->RandomDouble() < 0.2) {
    args.push_back("--generator=combinational");
  } else {
    args.push_back("--generator=pipeline");
    args.push_back(absl::StrCat("--pipeline_stages=", rng->RandRange(10) + 1));
  }
  return args;
}

static absl::StatusOr<std::string> Generate(
    const AstGeneratorOptions& ast_options, RngState* rng) {
  AstGenerator g(ast_options, &rng->rng());
  XLS_ASSIGN_OR_RETURN(auto pair, g.GenerateFunctionInModule("main", "test"));
  return pair.second->ToString();
}

static absl::StatusOr<std::unique_ptr<FunctionType>> GetFunctionType(
    absl::string_view dslx_text, absl::string_view fn_name) {
  ImportData import_data;
  XLS_ASSIGN_OR_RETURN(TypecheckedModule tm,
                       ParseAndTypecheck(dslx_text, "get_function_type.x",
                                         "get_function_type", &import_data,
                                         /*additional_search_paths=*/{}));
  XLS_ASSIGN_OR_RETURN(dslx::Function * f,
                       tm.module->GetFunctionOrError(fn_name));
  XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                       tm.type_info->GetItemAs<FunctionType>(f));
  std::unique_ptr<ConcreteType> cloned = fn_type->CloneToUnique();
  return absl::WrapUnique(down_cast<FunctionType*>(cloned.release()));
}

absl::StatusOr<Sample> GenerateSample(const AstGeneratorOptions& options,
                                      int64_t calls_per_sample,
                                      const SampleOptions& default_options,
                                      RngState* rng) {
  // To workaround a bug in pipeline generator we modify the ast generator
  // options if pipeline generator is used.
  // TODO(https://github.com/google/xls/issues/346): 2021-03-19 Remove this
  // option when pipeline generator handles empty tuples properly.
  AstGeneratorOptions modified_options = options;

  // Generate the sample options which is how to *run* the generated
  // sample. AstGeneratorOptions 'options' is how to *generate* the sample.
  SampleOptions sample_options = default_options;
  // The generated sample is DSLX so input_is_dslx must be true.
  sample_options.set_input_is_dslx(true);
  XLS_RET_CHECK(!sample_options.codegen_args().has_value())
      << "Setting codegen arguments is not supported, they are randomly "
         "generated";
  if (sample_options.codegen()) {
    // Generate codegen args if codegen is given but no codegen args are
    // specified.
    sample_options.set_codegen_args(
        GenerateCodegenArgs(sample_options.use_system_verilog(), rng));
    for (const std::string& arg : sample_options.codegen_args().value()) {
      if (arg == "--generator=pipeline") {
        modified_options.generate_empty_tuples = false;
        break;
      }
    }
  }
  XLS_ASSIGN_OR_RETURN(std::string dslx_text, Generate(modified_options, rng));
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<FunctionType> fn_type,
                       GetFunctionType(dslx_text, "main"));
  std::vector<std::vector<InterpValue>> args_batch;
  for (int64_t i = 0; i < calls_per_sample; ++i) {
    std::vector<const ConcreteType*> params;
    for (const auto& param : fn_type->params()) {
      params.push_back(param.get());
    }
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                         GenerateArguments(params, rng));
    args_batch.push_back(std::move(args));
  }
  return Sample(std::move(dslx_text), std::move(sample_options),
                std::move(args_batch));
}

}  // namespace xls
