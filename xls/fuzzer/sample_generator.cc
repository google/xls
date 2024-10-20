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

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/bit_gen_ref.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/warning_kind.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"
#include "xls/fuzzer/value_generator.h"

namespace xls {
namespace {

using ::xls::dslx::AstGenerator;
using ::xls::dslx::AstGeneratorOptions;
using ::xls::dslx::AstNode;
using ::xls::dslx::AstNodeVisitorWithDefault;
using ::xls::dslx::FunctionType;
using ::xls::dslx::ImportData;
using ::xls::dslx::InterpValue;
using ::xls::dslx::Module;
using ::xls::dslx::ModuleMember;
using ::xls::dslx::ParseModule;
using ::xls::dslx::Type;
using ::xls::dslx::TypecheckedModule;

class HasNonBlockingRecvVisitor : public AstNodeVisitorWithDefault {
 public:
  bool GetHasNbRecv() const { return has_nb_recv_; }

  absl::Status HandleInvocation(const dslx::Invocation* n) override {
    absl::StatusOr<std::string> builtin_name =
        dslx::GetBuiltinName(n->callee());
    if (builtin_name.ok()) {
      has_nb_recv_ = (*builtin_name == "recv_non_blocking") ||
                     (*builtin_name == "recv_if_non_blocking");
    }
    return absl::OkStatus();
  }

 private:
  bool has_nb_recv_ = false;
};

class IsPotentiallyDelayingNodeVisitor : public AstNodeVisitorWithDefault {
 public:
  bool IsSend() const { return is_send_; }
  bool IsRecv() const { return is_recv_; }
  bool IsDelaying() const { return is_send_ || is_recv_; }

  absl::Status HandleInvocation(const dslx::Invocation* n) override {
    absl::StatusOr<std::string> builtin_name =
        dslx::GetBuiltinName(n->callee());
    if (builtin_name.ok()) {
      is_send_ = absl::StartsWith(*builtin_name, "send");
      is_recv_ = absl::StartsWith(*builtin_name, "recv");
    } else {
      is_send_ = false;
      is_recv_ = false;
    }
    return absl::OkStatus();
  }

 private:
  bool is_send_ = false;
  bool is_recv_ = false;
};

}  // namespace

static absl::StatusOr<bool> HasNonBlockingRecv(const std::string& dslx_text,
                                               dslx::FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> module,
      ParseModule(dslx_text, "sample.x", "sample", file_table));
  XLS_RET_CHECK(module != nullptr);

  HasNonBlockingRecvVisitor visitor;
  // Using std::deque to leverage the `insert` operator.
  std::deque<AstNode*> bfs_queue;
  bfs_queue.push_back(module.get());
  while (!bfs_queue.empty()) {
    AstNode* node = bfs_queue.front();
    bfs_queue.pop_front();
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    if (visitor.GetHasNbRecv()) {
      return true;
    }
    std::vector<AstNode*> children = node->GetChildren(false);
    bfs_queue.insert(bfs_queue.end(), children.begin(), children.end());
  }
  return visitor.GetHasNbRecv();
}

// Returns randomly generated arguments for running codegen.
//
// These arguments are flags which are passed to codegen_main for generating
// Verilog. Randomly chooses either a purely combinational module or a
// feed-forward pipeline of a random length (sufficient to support known
// delays).
//
// Args:
//   use_system_verilog: Whether to use SystemVerilog.
//   has_proc: Whether the IR semantics of the circuit has a proc.
//   has_registers: Whether the circuit has registers.
//   has_nb_recv: Whether the IR semantics of the circuit has a non-blocking
//     receive operation.
//   rng: Random number generator state.
static std::vector<std::string> GenerateCodegenArgs(
    bool use_system_verilog, bool has_proc, bool has_registers,
    int64_t min_stages, bool has_nb_recv, absl::BitGenRef bit_gen) {
  std::vector<std::string> args;
  if (use_system_verilog) {
    args.push_back("--use_system_verilog");
  } else {
    args.push_back("--nouse_system_verilog");
  }
  bool is_pipeline = has_registers || absl::Bernoulli(bit_gen, 0.8);
  if (is_pipeline) {
    args.push_back("--generator=pipeline");
    // Set the pipeline stage to one when fuzzing a proc with a non-blocking
    // receive to ensures proper validation of the verilog output with other
    // stages in the proc.
    // TODO(https://github.com/google/xls/issues/798) To enhance the coverage,
    // support pipeline stages greater than one.
    if (has_nb_recv) {
      args.push_back("--pipeline_stages=1");
    } else {
      int64_t stage_cnt =
          absl::Uniform<int64_t>(bit_gen, min_stages, min_stages + 10);
      args.push_back(absl::StrCat("--pipeline_stages=", stage_cnt));
      if (stage_cnt > 1) {
        int64_t worst_case_throughput =
            absl::Uniform<int64_t>(bit_gen, 0, stage_cnt);
        if (worst_case_throughput > 0) {
          args.push_back(absl::StrFormat("--worst_case_throughput=%d",
                                         worst_case_throughput));
        }
      }
    }
  } else {
    args.push_back("--generator=combinational");
  }
  if (has_registers || is_pipeline) {
    args.push_back("--reset=rst");
    // The simulation functions with an active low reset.
    args.push_back("--reset_active_low=false");
    // TODO(https://github.com/google/xls/issues/795) Test the
    // 'reset_asynchronous' flag in codegen in the fuzzer.
    args.push_back(
        absl::StrCat("--reset_asynchronous=",
                     absl::Bernoulli(bit_gen, 0.5) ? "true" : "false"));
  }
  if (is_pipeline && has_proc) {
    // For a pipelined proc, the data path may contain register driving control
    // (e.g. channel read). These registers need to have a defined value after
    // reset. As a result, the data path must be reset.
    args.push_back("--reset_data_path=true");
  } else {
    // A combinational circuit and circuit derived from a function do not
    // support resetting the data path.
    args.push_back("--reset_data_path=false");
  }
  // TODO(https://github.com/google/xls/issues/791).
  if (has_nb_recv) {
    args.push_back("--flop_inputs=true");
    args.push_back("--flop_inputs_kind=zerolatency");
  }
  args.push_back("--output_block_ir_path=sample.block.ir");
  return args;
}

// This function generates a module satisfying `ast_options`, and returns the
// DSLX for that module and the minimum number of stages it can be safely
// scheduled in.
static absl::StatusOr<std::pair<std::string, int64_t>> Generate(
    const AstGeneratorOptions& ast_options, absl::BitGenRef bit_gen,
    dslx::FileTable& file_table) {
  AstGenerator g(ast_options, bit_gen, file_table);
  XLS_ASSIGN_OR_RETURN(dslx::AnnotatedModule module,
                       g.Generate("main", "test"));
  return std::make_pair(module.module->ToString(), module.min_stages);
}

// The function translates a list of Type unique_ptrs to a list of
// pointers to Type. The latter is used as a parameter to the
// GenerateArguments.
static std::vector<const dslx::Type*> TranslateTypeList(
    absl::Span<const std::unique_ptr<dslx::Type>> list) {
  std::vector<const dslx::Type*> translation(list.size());
  auto translation_iter = translation.begin();
  for (const auto& element : list) {
    *translation_iter = element.get();
    translation_iter++;
  }
  return translation;
}

// Returns the parameter types of a Function.
static absl::StatusOr<std::vector<std::unique_ptr<dslx::Type>>>
GetParamTypesOfFunction(dslx::Function* function, const TypecheckedModule& tm) {
  std::vector<std::unique_ptr<dslx::Type>> params;
  XLS_ASSIGN_OR_RETURN(dslx::FunctionType * fn_type,
                       tm.type_info->GetItemAs<dslx::FunctionType>(function));
  for (const auto& param : fn_type->params()) {
    params.push_back(param->CloneToUnique());
  }
  return params;
}

// Returns the input channel payload types of a Proc.
static absl::StatusOr<std::vector<std::unique_ptr<dslx::Type>>>
GetInputChannelPayloadTypesOfProc(dslx::Proc* proc,
                                  const TypecheckedModule& tm) {
  std::vector<std::unique_ptr<dslx::Type>> input_channel_payload_types;
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * proc_type_info,
                       tm.type_info->GetTopLevelProcTypeInfo(proc));
  for (dslx::ProcMember* member : proc->members()) {
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(member->type_annotation());
    if (channel_type == nullptr) {
      continue;
    }
    if (channel_type->direction() != dslx::ChannelDirection::kIn) {
      continue;
    }
    dslx::TypeAnnotation* payload_type_annotation = channel_type->payload();
    std::optional<const dslx::Type*> maybe_payload_type =
        proc_type_info->GetItem(payload_type_annotation);
    XLS_RET_CHECK(maybe_payload_type.has_value());
    XLS_ASSIGN_OR_RETURN(const dslx::Type* payload_type,
                         UnwrapMetaType(*maybe_payload_type.value()));
    input_channel_payload_types.push_back(payload_type->CloneToUnique());
  }
  return input_channel_payload_types;
}

// Returns the input channel names of the proc.
static std::vector<std::string> GetInputChannelNamesOfProc(dslx::Proc* proc) {
  std::vector<std::string> channel_names;
  for (dslx::ProcMember* member : proc->members()) {
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(member->type_annotation());
    if (channel_type == nullptr) {
      continue;
    }
    if (channel_type->direction() != dslx::ChannelDirection::kIn) {
      continue;
    }
    channel_names.push_back(member->identifier());
  }
  return channel_names;
}

static absl::StatusOr<Sample> GenerateFunctionSample(
    dslx::Function* function, const TypecheckedModule& tm,
    const SampleOptions& sample_options, absl::BitGenRef bit_gen,
    const std::string& dslx_text) {
  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<dslx::Type>> top_params,
                       GetParamTypesOfFunction(function, tm));
  std::vector<const dslx::Type*> params = TranslateTypeList(top_params);

  std::vector<std::vector<InterpValue>> args_batch;
  for (int64_t i = 0; i < sample_options.calls_per_sample(); ++i) {
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                         GenerateInterpValues(bit_gen, params));
    args_batch.push_back(std::move(args));
  }

  return Sample(dslx_text, sample_options, std::move(args_batch));
}

static absl::StatusOr<Sample> GenerateProcSample(
    dslx::Proc* proc, const TypecheckedModule& tm,
    const SampleOptions& sample_options, absl::BitGenRef bit_gen,
    const std::string& dslx_text) {
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<dslx::Type>> input_channel_payload_types,
      GetInputChannelPayloadTypesOfProc(proc, tm));
  std::vector<const dslx::Type*> input_channel_payload_types_ptr =
      TranslateTypeList(input_channel_payload_types);

  std::vector<std::vector<InterpValue>> channel_values_batch;
  for (int64_t i = 0; i < sample_options.proc_ticks(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        std::vector<InterpValue> channel_values,
        GenerateInterpValues(bit_gen, input_channel_payload_types_ptr));
    channel_values_batch.push_back(std::move(channel_values));
  }

  std::vector<std::string> input_channel_names =
      GetInputChannelNamesOfProc(proc);
  std::vector<std::string> ir_channel_names(input_channel_names.size());
  for (int64_t index = 0; index < input_channel_names.size(); ++index) {
    ir_channel_names[index] =
        absl::StrCat(proc->owner()->name(), "__", input_channel_names[index]);
  }

  return Sample(dslx_text, sample_options, std::move(channel_values_batch),
                std::move(ir_channel_names));
}

absl::StatusOr<Sample> GenerateSample(
    const AstGeneratorOptions& generator_options,
    const SampleOptions& sample_options, absl::BitGenRef bit_gen,
    dslx::FileTable& file_table) {
  constexpr std::string_view top_name = "main";
  if (generator_options.generate_proc) {
    CHECK_EQ(sample_options.calls_per_sample(), 0)
        << "calls per sample must be zero when generating a proc sample.";
    CHECK_GT(sample_options.proc_ticks(), 0)
        << "proc ticks must have a value when generating a proc sample.";
  } else {
    CHECK_EQ(sample_options.proc_ticks(), 0)
        << "proc ticks must not have a zero value when generating a "
           "function sample.";
  }

  std::string dslx_text;
  bool has_nb_recv = false;
  int64_t min_stages = 1;
  do {
    XLS_ASSIGN_OR_RETURN(std::tie(dslx_text, min_stages),
                         Generate(generator_options, bit_gen, file_table));
    XLS_ASSIGN_OR_RETURN(has_nb_recv,
                         HasNonBlockingRecv(dslx_text, file_table));
    // If this sample is going through codegen, regenerate the sample until it's
    // legal; we currently can't verify latency-sensitive samples, which means
    // we can't have a non-blocking recv except with 1 pipeline stage.
  } while (sample_options.codegen() && has_nb_recv && min_stages > 1);

  // Generate the sample options which is how to *run* the generated
  // sample. AstGeneratorOptions 'options' is how to *generate* the sample.
  SampleOptions sample_options_copy = sample_options;
  // The generated sample is DSLX so input_is_dslx must be true.
  sample_options_copy.set_input_is_dslx(true);
  XLS_RET_CHECK(sample_options_copy.codegen_args().empty())
      << "Setting codegen arguments is not supported, they are randomly "
         "generated";
  if (sample_options_copy.codegen()) {
    // Generate codegen args if codegen is given but no codegen args are
    // specified.
    sample_options_copy.set_codegen_args(
        GenerateCodegenArgs(sample_options_copy.use_system_verilog(),
                            generator_options.generate_proc,
                            generator_options.generate_proc &&
                                !generator_options.emit_stateless_proc,
                            min_stages, has_nb_recv, bit_gen));
  }

  // Parse and type check the DSLX input to retrieve the top entity. The top
  // member must be a proc or a function.
  ImportData import_data(
      dslx::CreateImportData(/*stdlib_path=*/"",
                             /*additional_search_paths=*/{},
                             /*enabled_warnings=*/dslx::kAllWarningsSet,
                             std::make_unique<dslx::RealFilesystem>()));
  absl::StatusOr<TypecheckedModule> tm =
      ParseAndTypecheck(dslx_text, "sample.x", "sample", &import_data);
  if (!tm.ok()) {
    LOG(ERROR) << "Generated sample failed to parse.";
    XLS_LOG_LINES(ERROR, dslx_text);
    return tm.status();
  }

  std::optional<ModuleMember*> module_member =
      tm->module->FindMemberWithName(top_name);
  CHECK(module_member.has_value());
  ModuleMember* member = module_member.value();

  if (generator_options.generate_proc) {
    CHECK(std::holds_alternative<dslx::Proc*>(*member));
    sample_options_copy.set_sample_type(fuzzer::SAMPLE_TYPE_PROC);
    return GenerateProcSample(std::get<dslx::Proc*>(*member), *tm,
                              sample_options_copy, bit_gen, dslx_text);
  }
  CHECK(std::holds_alternative<dslx::Function*>(*member));
  sample_options_copy.set_sample_type(fuzzer::SAMPLE_TYPE_FUNCTION);
  return GenerateFunctionSample(std::get<dslx::Function*>(*member), *tm,
                                sample_options_copy, bit_gen, dslx_text);
}

}  // namespace xls
