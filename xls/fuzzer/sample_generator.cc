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

#include <memory>
#include <string>
#include <string_view>
#include <variant>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/fuzzer/sample.h"

namespace xls {

using dslx::AstGenerator;
using dslx::AstGeneratorOptions;
using dslx::ConcreteType;
using dslx::FunctionType;
using dslx::ImportData;
using dslx::InterpValue;
using dslx::Module;
using dslx::ModuleMember;
using dslx::TypecheckedModule;

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
                                                    bool contains_registers,
                                                    ValueGenerator* value_gen) {
  std::vector<std::string> args;
  if (use_system_verilog) {
    args.push_back("--use_system_verilog");
  } else {
    args.push_back("--nouse_system_verilog");
  }
  if (value_gen->RandomDouble() < 0.2 && !contains_registers) {
    args.push_back("--generator=combinational");
  } else {
    args.push_back("--generator=pipeline");
    args.push_back(
        absl::StrCat("--pipeline_stages=", value_gen->RandRange(10) + 1));
  }
  if (contains_registers) {
    args.push_back("--reset=rst");
  }
  return args;
}

static absl::StatusOr<std::string> Generate(
    const AstGeneratorOptions& ast_options, ValueGenerator* value_gen) {
  AstGenerator g(ast_options, value_gen);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module,
                       g.Generate("main", "test"));
  return module->ToString();
}

// The function translates a list of ConcreteType unique_ptrs to a list of
// pointers to ConcreteType. The latter is used as a parameter to the
// GenerateArguments.
static std::vector<const ConcreteType*> TranslateConcreteTypeList(
    absl::Span<const std::unique_ptr<ConcreteType>> list) {
  std::vector<const ConcreteType*> translation(list.size());
  auto translation_iter = translation.begin();
  for (const auto& element : list) {
    *translation_iter = element.get();
    translation_iter++;
  }
  return translation;
}

// Returns the parameter types of a Function.
static absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>>
GetParamTypesOfFunction(dslx::Function* function, const TypecheckedModule& tm) {
  std::vector<std::unique_ptr<ConcreteType>> params;
  XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                       tm.type_info->GetItemAs<FunctionType>(function));
  for (const auto& param : fn_type->params()) {
    params.push_back(param->CloneToUnique());
  }
  return params;
}

// Returns the input channel payload types of a Proc.
static absl::StatusOr<std::vector<std::unique_ptr<ConcreteType>>>
GetInputChannelPayloadTypesOfProc(dslx::Proc* proc,
                                  const TypecheckedModule& tm) {
  std::vector<std::unique_ptr<ConcreteType>> input_channel_payload_types;
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * proc_type_info,
                       tm.type_info->GetTopLevelProcTypeInfo(proc));
  for (dslx::Param* member : proc->members()) {
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(member->type_annotation());
    if (channel_type == nullptr) {
      continue;
    }
    if (channel_type->direction() !=
        dslx::ChannelTypeAnnotation::Direction::kIn) {
      continue;
    }
    dslx::TypeAnnotation* payload_type = channel_type->payload();
    XLS_CHECK(proc_type_info->GetItem(payload_type).has_value());
    input_channel_payload_types.push_back(
        proc_type_info->GetItem(payload_type).value()->CloneToUnique());
  }
  return input_channel_payload_types;
}

// Returns the input channel names of the proc.
static std::vector<std::string> GetInputChannelNamesOfProc(dslx::Proc* proc) {
  std::vector<std::string> channel_names;
  for (dslx::Param* member : proc->members()) {
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(member->type_annotation());
    if (channel_type == nullptr) {
      continue;
    }
    if (channel_type->direction() !=
        dslx::ChannelTypeAnnotation::Direction::kIn) {
      continue;
    }
    channel_names.push_back(member->identifier());
  }
  return channel_names;
}

absl::StatusOr<Sample> GenerateFunctionSample(
    dslx::Function* function, const TypecheckedModule& tm,
    const SampleOptions& sample_options, ValueGenerator* value_gen,
    const std::string& dslx_text) {
  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<ConcreteType>> top_params,
                       GetParamTypesOfFunction(function, tm));
  std::vector<const ConcreteType*> params =
      TranslateConcreteTypeList(top_params);

  std::vector<std::vector<InterpValue>> args_batch;
  for (int64_t i = 0; i < sample_options.calls_per_sample(); ++i) {
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                         value_gen->GenerateInterpValues(params));
    args_batch.push_back(std::move(args));
  }

  return Sample(std::move(dslx_text), std::move(sample_options),
                std::move(args_batch));
}

absl::StatusOr<Sample> GenerateProcSample(dslx::Proc* proc,
                                          const TypecheckedModule& tm,
                                          const SampleOptions& sample_options,
                                          ValueGenerator* value_gen,
                                          const std::string& dslx_text) {
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<ConcreteType>> input_channel_payload_types,
      GetInputChannelPayloadTypesOfProc(proc, tm));
  std::vector<const ConcreteType*> input_channel_payload_types_ptr =
      TranslateConcreteTypeList(input_channel_payload_types);

  std::vector<std::vector<InterpValue>> channel_values_batch;
  for (int64_t i = 0; i < sample_options.proc_ticks().value(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        std::vector<InterpValue> channel_values,
        value_gen->GenerateInterpValues(input_channel_payload_types_ptr));
    channel_values_batch.push_back(std::move(channel_values));
  }

  std::vector<std::string> input_channel_names =
      GetInputChannelNamesOfProc(proc);
  std::vector<std::string> ir_channel_names(input_channel_names.size());
  for (int64_t index = 0; index < input_channel_names.size(); ++index) {
    ir_channel_names[index] =
        absl::StrCat(proc->owner()->name(), "__", input_channel_names[index]);
  }

  return Sample(std::move(dslx_text), std::move(sample_options),
                std::move(channel_values_batch), std::move(ir_channel_names));
}

absl::StatusOr<Sample> GenerateSample(
    const AstGeneratorOptions& generator_options,
    const SampleOptions& sample_options, ValueGenerator* value_gen) {
  constexpr std::string_view top_name = "main";
  if (generator_options.generate_proc) {
    XLS_CHECK_EQ(sample_options.calls_per_sample(), 0)
        << "calls per sample must be zero when generating a proc sample.";
    XLS_CHECK(sample_options.proc_ticks().has_value())
        << "proc ticks must have a value when generating a proc sample.";
  } else {
    XLS_CHECK(!sample_options.proc_ticks().has_value() ||
              sample_options.proc_ticks().value() == 0)
        << "proc ticks must not be set or have a zero value when generating a "
           "function sample.";
  }
  // Generate the sample options which is how to *run* the generated
  // sample. AstGeneratorOptions 'options' is how to *generate* the sample.
  SampleOptions sample_options_copy = sample_options;
  // The generated sample is DSLX so input_is_dslx must be true.
  sample_options_copy.set_input_is_dslx(true);
  XLS_RET_CHECK(!sample_options_copy.codegen_args().has_value())
      << "Setting codegen arguments is not supported, they are randomly "
         "generated";
  if (sample_options_copy.codegen()) {
    // Generate codegen args if codegen is given but no codegen args are
    // specified.
    sample_options_copy.set_codegen_args(
        GenerateCodegenArgs(sample_options_copy.use_system_verilog(),
                            generator_options.generate_proc &&
                                !generator_options.emit_stateless_proc,
                            value_gen));
  }
  XLS_ASSIGN_OR_RETURN(std::string dslx_text,
                       Generate(generator_options, value_gen));

  // Parse and type check the DSLX input to retrieve the top entity. The top
  // member must be a proc or a function.
  ImportData import_data(
      dslx::CreateImportData(/*stdlib_path=*/"",
                             /*additional_search_paths=*/{}));
  XLS_ASSIGN_OR_RETURN(
      TypecheckedModule tm,
      ParseAndTypecheck(dslx_text, "sample.x", "sample", &import_data));
  std::optional<ModuleMember*> module_member =
      tm.module->FindMemberWithName(top_name);
  XLS_CHECK(module_member.has_value());
  ModuleMember* member = module_member.value();

  if (generator_options.generate_proc) {
    XLS_CHECK(std::holds_alternative<dslx::Proc*>(*member));
    sample_options_copy.set_top_type(TopType::kProc);
    return GenerateProcSample(std::get<dslx::Proc*>(*member), tm,
                              sample_options_copy, value_gen, dslx_text);
  }
  XLS_CHECK(std::holds_alternative<dslx::Function*>(*member));
  sample_options_copy.set_top_type(TopType::kFunction);
  return GenerateFunctionSample(std::get<dslx::Function*>(*member), tm,
                                sample_options_copy, value_gen, dslx_text);
}

}  // namespace xls
