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
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"
#include "xls/fuzzer/ast_generator.h"
#include "xls/fuzzer/sample.h"
#include "xls/fuzzer/sample.pb.h"
#include "xls/fuzzer/value_generator.h"
#include "xls/ir/format_preference.h"
#include "xls/tests/testvector.pb.h"

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

class HasMapBuiltinVisitor : public AstNodeVisitorWithDefault {
 public:
  bool GetHasMap() const { return has_map_; }

  absl::Status HandleInvocation(const dslx::Invocation* n) override {
    std::optional<std::string_view> builtin_name =
        dslx::GetBuiltinFnName(n->callee());
    if (builtin_name.has_value()) {
      has_map_ = has_map_ || builtin_name.value() == "map";
    }
    return absl::OkStatus();
  }

 private:
  bool has_map_ = false;
};

class HasNonBlockingRecvVisitor : public AstNodeVisitorWithDefault {
 public:
  bool GetHasNbRecv() const { return has_nb_recv_; }

  absl::Status HandleInvocation(const dslx::Invocation* n) override {
    std::optional<std::string_view> builtin_name =
        dslx::GetBuiltinFnName(n->callee());
    if (builtin_name.has_value()) {
      has_nb_recv_ = has_nb_recv_ ||
                     builtin_name.value() == "recv_non_blocking" ||
                     builtin_name.value() == "recv_if_non_blocking";
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
    std::optional<std::string_view> builtin_name =
        dslx::GetBuiltinFnName(n->callee());
    if (builtin_name.has_value()) {
      is_send_ = absl::StartsWith(*builtin_name, "send");
      is_recv_ = absl::StartsWith(*builtin_name, "recv");
    }
    return absl::OkStatus();
  }

 private:
  bool is_send_ = false;
  bool is_recv_ = false;
};

absl::StatusOr<bool> HasMapBuiltin(const std::string& dslx_text,
                                   dslx::FileTable& file_table) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<Module> module,
      ParseModule(dslx_text, "sample.x", "sample", file_table));
  XLS_RET_CHECK(module != nullptr);

  HasMapBuiltinVisitor visitor;
  // Using std::deque to leverage the `insert` operator.
  std::deque<AstNode*> bfs_queue;
  bfs_queue.push_back(module.get());
  while (!bfs_queue.empty()) {
    AstNode* node = bfs_queue.front();
    bfs_queue.pop_front();
    XLS_RETURN_IF_ERROR(node->Accept(&visitor));
    if (visitor.GetHasMap()) {
      return true;
    }
    std::vector<AstNode*> children = node->GetChildren(false);
    bfs_queue.insert(bfs_queue.end(), children.begin(), children.end());
  }
  return visitor.GetHasMap();
}
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
    bool use_system_verilog, bool has_proc, bool has_stateless_proc,
    int64_t min_stages, bool has_nb_recv, absl::BitGenRef bit_gen) {
  std::vector<std::string> args;
  if (use_system_verilog) {
    args.push_back("--use_system_verilog");
  } else {
    args.push_back("--nouse_system_verilog");
  }

  args.push_back("--output_block_ir_path=sample.block.ir");

  // For functions and stateless procs, randomly choose to generate either a
  // combinational block or pipelined block.
  bool generate_combinational_block =
      (!has_proc || has_stateless_proc) && absl::Bernoulli(bit_gen, 0.2);

  if (generate_combinational_block) {
    args.push_back("--generator=combinational");

    // A combinational circuit does not support resetting the data path.
    args.push_back("--reset_data_path=false");

    return args;
  }

  // Pipeline generator.
  args.push_back("--generator=pipeline");

  // Set the pipeline stage to one when fuzzing a proc with a non-blocking
  // receive to ensures proper validation of the verilog output with other
  // stages in the proc.

  // TODO(https://github.com/google/xls/issues/798) To enhance the coverage,
  // support pipeline stages greater than one.
  if (has_nb_recv) {
    args.push_back("--pipeline_stages=1");

    // TODO(https://github.com/google/xls/issues/791).
    args.push_back("--flop_inputs=true");
    args.push_back("--flop_inputs_kind=zerolatency");
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

  args.push_back("--reset=rst");

  // The simulation functions with an active low reset.
  args.push_back("--reset_active_low=false");

  // TODO(https://github.com/google/xls/issues/795) Test the
  // 'reset_asynchronous' flag in codegen in the fuzzer.
  bool async_reset = absl::Bernoulli(bit_gen, 0.5);
  args.push_back(
      absl::StrCat("--reset_asynchronous=", async_reset ? "true" : "false"));

  if (has_proc) {
    // For a pipelined proc, the data path may contain register driving control
    // (e.g. channel read). These registers need to have a defined value after
    // reset. As a result, the data path must be reset.
    args.push_back("--reset_data_path=true");
  } else {
    // Async reset without resetting the data-path tickles some bad behavior in
    // simulation causing spurious assert failures.
    bool reset_data_path = async_reset ? true : absl::Bernoulli(bit_gen, 0.5);
    args.push_back(
        absl::StrCat("--reset_data_path=", reset_data_path ? "true" : "false"));
  }
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

// Unwraps unique_ptr elements to a vector of pointers.
static std::vector<const dslx::Type*> UnwrapUniquePtrs(
    absl::Span<const std::unique_ptr<dslx::Type>> wrapped) {
  std::vector<const dslx::Type*> unwrapped;
  unwrapped.reserve(wrapped.size());
  for (const auto& unique : wrapped) {
    unwrapped.push_back(unique.get());
  }
  return unwrapped;
}

// Returns the parameter types of a Function.
static absl::StatusOr<std::vector<std::unique_ptr<dslx::Type>>>
GetParamTypesOfFunction(dslx::Function* function, const TypecheckedModule& tm) {
  std::vector<std::unique_ptr<dslx::Type>> params;
  XLS_ASSIGN_OR_RETURN(dslx::FunctionType * fn_type,
                       tm.type_info->GetItemAs<dslx::FunctionType>(function));
  for (const std::unique_ptr<dslx::Type>& param : fn_type->params()) {
    XLS_RET_CHECK(!param->IsMeta());
    XLS_RET_CHECK(dynamic_cast<const dslx::BitsConstructorType*>(param.get()) ==
                  nullptr)
        << "`BitsConstructorType`s are not valid parameter types.";

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
  for (dslx::Param* param : proc->config().params()) {
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(param->type_annotation());
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
  for (dslx::Param* param : proc->config().params()) {
    dslx::ChannelTypeAnnotation* channel_type =
        dynamic_cast<dslx::ChannelTypeAnnotation*>(param->type_annotation());
    if (channel_type == nullptr) {
      continue;
    }
    if (channel_type->direction() != dslx::ChannelDirection::kIn) {
      continue;
    }
    channel_names.push_back(param->identifier());
  }
  return channel_names;
}

// Convert InterpValue to value used in SampleInputsProto
std::string ToArgString(const InterpValue& v) {
  return v.ConvertToIr().value().ToString(FormatPreference::kHex);
}

// Converts a list of interpreter values to a string as needed in SampleInputs
std::string InterpValueListToString(
    const std::vector<InterpValue>& interpv_list) {
  return absl::StrJoin(interpv_list, "; ",
                       [](std::string* out, const InterpValue& v) {
                         absl::StrAppend(out, ToArgString(v));
                       });
}

static absl::StatusOr<Sample> GenerateFunctionSample(
    dslx::Function* function, const TypecheckedModule& tm,
    const SampleOptions& sample_options, absl::BitGenRef bit_gen,
    const std::string& dslx_text) {
  XLS_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<dslx::Type>> top_params,
                       GetParamTypesOfFunction(function, tm));
  std::vector<const dslx::Type*> params = UnwrapUniquePtrs(top_params);

  testvector::SampleInputsProto testvector;
  testvector::FunctionArgsProto* fun_args = testvector.mutable_function_args();

  for (int64_t i = 0; i < sample_options.calls_per_sample(); ++i) {
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                         GenerateInterpValues(bit_gen, params));
    fun_args->add_args(InterpValueListToString(args));
  }

  return Sample(dslx_text, sample_options, testvector);
}

static absl::StatusOr<Sample> GenerateProcSample(
    dslx::Proc* proc, const TypecheckedModule& tm,
    const SampleOptions& sample_options, absl::BitGenRef bit_gen,
    const std::string& dslx_text) {
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<dslx::Type>> input_channel_payload_types,
      GetInputChannelPayloadTypesOfProc(proc, tm));
  std::vector<const dslx::Type*> input_channel_payload_types_ptr =
      UnwrapUniquePtrs(input_channel_payload_types);

  // Create number of values needed for the proc_ticks.
  // Actual proc-tics the execution needs might be longer depending on
  // if it deals with holdoff values.
  std::vector<std::vector<InterpValue>> channel_values_batch;
  for (int64_t i = 0; i < sample_options.proc_ticks(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        std::vector<InterpValue> channel_values,
        GenerateInterpValues(bit_gen, input_channel_payload_types_ptr));
    channel_values_batch.push_back(std::move(channel_values));
  }

  const std::vector<std::string> input_channel_names =
      GetInputChannelNamesOfProc(proc);

  const int64_t holdoff_range = sample_options.proc_ticks() / 10;
  testvector::SampleInputsProto testvector;
  testvector::ChannelInputsProto* inputs_proto =
      testvector.mutable_channel_inputs();
  using testvector::ValidHoldoff;

  // Create data per channel. Each channel gets data with randomized
  // valid-holdoff prepended if requested.
  for (int64_t ch_idx = 0; ch_idx < input_channel_names.size(); ++ch_idx) {
    testvector::ChannelInputProto* channel_input = inputs_proto->add_inputs();
    channel_input->set_channel_name(
        absl::StrCat(proc->owner()->name(), "__", input_channel_names[ch_idx]));
    for (int64_t i = 0; i < sample_options.proc_ticks(); ++i) {
      if (sample_options.with_valid_holdoff()) {
        int64_t holdoff = absl::Uniform<int64_t>(bit_gen, 0, holdoff_range);
        ValidHoldoff* holdoff_data = channel_input->add_valid_holdoffs();
        holdoff_data->set_cycles(holdoff);
      }

      const std::vector<InterpValue>& args = channel_values_batch[i];
      channel_input->add_values(ToArgString(args[ch_idx]));
    }
  }

  return Sample(dslx_text, sample_options, testvector);
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
  } while (((sample_options.codegen() && min_stages > 1) ||
            generator_options.emit_proc_spawns) &&
           has_nb_recv);

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
    LOG(ERROR) << "Generated sample failed to parse-and-typecheck ("
               << dslx_text.size() << " bytes):";
    XLS_LOG_LINES(ERROR, dslx_text);
    return tm.status();
  }

  std::optional<ModuleMember*> module_member =
      tm->module->FindMemberWithName(top_name);
  CHECK(module_member.has_value());
  ModuleMember* member = module_member.value();

  // Also determine if the top is a stateless proc.
  bool is_proc = std::holds_alternative<dslx::Proc*>(*member);
  bool is_stateless_proc =
      is_proc && std::get<dslx::Proc*>(*member)->IsStateless();

  // Generate the sample options which is how to *run* the generated
  // sample. AstGeneratorOptions 'options' is how to *generate* the sample.
  SampleOptions sample_options_copy = sample_options;
  // The generated sample is DSLX so input_is_dslx must be true.
  sample_options_copy.set_input_is_dslx(true);

  // Disable unopt-ir interpreter if the dslx includes 'map'. This can be really
  // slow and we already have the dslx interpreter to provide a baseline.
  // TODO(https://github.com/google/xls/issues/2263): Ideally we should not need
  // to do this.
  XLS_ASSIGN_OR_RETURN(bool has_map, HasMapBuiltin(dslx_text, file_table));
  sample_options_copy.set_disable_unopt_interpreter(has_map);

  XLS_RET_CHECK(sample_options_copy.codegen_args().empty())
      << "Setting codegen arguments is not supported, they are randomly "
         "generated";
  if (sample_options_copy.codegen()) {
    // Generate codegen args if codegen is given but no codegen args are
    // specified.
    sample_options_copy.set_codegen_args(GenerateCodegenArgs(
        sample_options_copy.use_system_verilog(), is_proc, is_stateless_proc,
        min_stages, has_nb_recv, bit_gen));

    // Randomly also turn on Codegen NG pipeline.
    bool use_codegen_ng = absl::Bernoulli(bit_gen, 0.5);
    if (use_codegen_ng) {
      sample_options_copy.set_codegen_ng(true);
    }
  }

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
