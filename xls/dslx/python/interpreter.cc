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

#include <cstdint>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11_abseil/absl_casters.h"
#include "pybind11_abseil/statusor_caster.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/import_status_module.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/bytecode.h"
#include "xls/dslx/bytecode_emitter.h"
#include "xls/dslx/bytecode_interpreter.h"
#include "xls/dslx/concrete_type.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/import_routines.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/interp_value_helpers.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/python/errors.h"
#include "xls/dslx/symbolic_bindings.h"
#include "xls/ir/python/wrapper_types.h"

namespace py = pybind11;

namespace xls::dslx {

namespace {

absl::StatusOr<std::vector<std::vector<InterpValue>>> ConvertFunctionKwargs(
    const dslx::Function* f, const ImportData& import_data,
    const TypecheckedModule& tm,
    const std::vector<std::vector<InterpValue>>& args_batch) {
  XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                       tm.type_info->GetItemAs<FunctionType>(f));
  std::vector<std::vector<InterpValue>> converted_args;
  converted_args.reserve(args_batch.size());
  for (const std::vector<InterpValue>& unsigned_args : args_batch) {
    XLS_ASSIGN_OR_RETURN(std::vector<InterpValue> args,
                         SignConvertArgs(*fn_type, unsigned_args));
    converted_args.push_back(std::move(args));
  }
  return converted_args;
}

absl::StatusOr<std::vector<InterpValue>> RunFunctionBatched(
    const dslx::Function* f, ImportData& import_data,
    const TypecheckedModule& tm,
    const std::vector<std::vector<InterpValue>>& args_batch) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BytecodeFunction> bf,
                       BytecodeEmitter::Emit(&import_data, tm.type_info, f,
                                             /*caller_bindings=*/{}));
  XLS_ASSIGN_OR_RETURN(FunctionType * fn_type,
                       tm.type_info->GetItemAs<FunctionType>(f));
  std::vector<InterpValue> results;
  results.reserve(args_batch.size());
  for (const std::vector<InterpValue>& args : args_batch) {
    XLS_ASSIGN_OR_RETURN(InterpValue result, BytecodeInterpreter::Interpret(
                                                 &import_data, bf.get(), args));
    results.push_back(result);
  }
  return results;
}

absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>>
ConvertChannelValues(
    dslx::Proc* proc, const ImportData& import_data,
    const TypecheckedModule& tm,
    const std::vector<std::vector<InterpValue>>& input_channel_values) {
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * proc_type_info,
                       tm.type_info->GetTopLevelProcTypeInfo(proc));
  std::vector<ProcInstance> proc_instances;
  absl::flat_hash_map<std::string, InterpValue> converted_channel_values;
  // Positional indexes of the input channels in the config function.
  std::vector<int64_t> in_chan_indexes;

  std::string module_name = proc->owner()->name();
  for (int64_t index = 0; index < proc->config()->params().size(); ++index) {
    // Currently, only channels are supported as parameters to the config
    // function of a proc.
    Param* param = proc->config()->params().at(index);
    ChannelTypeAnnotation* channel_type =
        dynamic_cast<ChannelTypeAnnotation*>(param->type_annotation());
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }
    if (channel_type->direction() != ChannelTypeAnnotation::kIn) {
      continue;
    }
    converted_channel_values.insert(
        {param->identifier(), InterpValue::MakeChannel()});
    in_chan_indexes.push_back(index);
  }

  std::vector<std::unique_ptr<ConcreteType>> channel_payload_types(
      in_chan_indexes.size());
  for (int64_t index = 0; index < in_chan_indexes.size(); ++index) {
    XLS_ASSIGN_OR_RETURN(
        ConcreteType * type,
        proc_type_info->GetItemOrError(
            proc->config()->params().at(in_chan_indexes[index])));
    ChannelType* channel_type = dynamic_cast<ChannelType*>(type);
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }
    channel_payload_types[index] = channel_type->payload_type().CloneToUnique();
  }

  for (const std::vector<InterpValue>& values : input_channel_values) {
    XLS_CHECK_EQ(in_chan_indexes.size(), values.size())
        << "The input channel count should match the args count.";
    for (int64_t index = 0; index < values.size(); ++index) {
      Param* param = proc->config()->params().at(in_chan_indexes[index]);
      XLS_ASSIGN_OR_RETURN(
          InterpValue payload_value,
          SignConvertValue(*channel_payload_types[index], values[index]));
      converted_channel_values.at(param->identifier())
          .GetChannelOrDie()
          ->push_back(payload_value);
    }
  }
  return converted_channel_values;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<InterpValue>>>
RunProc(
    dslx::Proc* proc, ImportData& import_data, const TypecheckedModule& tm,
    const absl::flat_hash_map<std::string, InterpValue>& input_channel_values,
    int64_t proc_ticks) {
  XLS_ASSIGN_OR_RETURN(dslx::TypeInfo * proc_type_info,
                       tm.type_info->GetTopLevelProcTypeInfo(proc));
  std::vector<ProcInstance> proc_instances;
  std::vector<InterpValue> config_args;
  // Positional indexes of the output channels in the config function.
  std::vector<int64_t> out_chan_indexes;
  // The mapping of the channels in the output_channel_names follow the mapping
  // of out_chan_indexes. For example, out_channel_names[i] refers to same
  // channel at out_chan_indexes[i].
  std::vector<std::string> out_ir_channel_names;

  std::string module_name = proc->owner()->name();
  for (int64_t index = 0; index < proc->config()->params().size(); ++index) {
    Param* param = proc->config()->params().at(index);
    // Currently, only channels are supported as parameters to the config
    // function of a proc.
    ChannelTypeAnnotation* channel_type =
        dynamic_cast<ChannelTypeAnnotation*>(param->type_annotation());
    if (channel_type == nullptr) {
      return absl::InternalError(
          "Only channels are supported as parameters to the config function of "
          "a proc");
    }
    if (channel_type->direction() == ChannelTypeAnnotation::kIn) {
      config_args.push_back(input_channel_values.at(param->identifier()));
    } else if (channel_type->direction() == ChannelTypeAnnotation::kOut) {
      config_args.push_back(InterpValue::MakeChannel());
      out_chan_indexes.push_back(index);
      out_ir_channel_names.push_back(absl::StrCat(
          module_name, "__", proc->config()->params().at(index)->identifier()));
    }
  }

  XLS_RETURN_IF_ERROR(ProcConfigBytecodeInterpreter::EvalSpawn(
      &import_data, proc_type_info, absl::nullopt, absl::nullopt, proc,
      config_args, &proc_instances));
  // Currently a single proc is supported.
  XLS_CHECK_EQ(proc_instances.size(), 1);
  for (int i = 0; i < proc_ticks; i++) {
    XLS_RETURN_IF_ERROR(proc_instances[0].Run());
  }

  // TODO(vmirian): Ideally, the result should be a tuple containing two
  // tuples. The first entry is the result of the next function, the second is
  // the results of the output channels. Collect the result from the next
  // function.
  absl::flat_hash_map<std::string, std::vector<InterpValue>> all_channel_values;
  for (int64_t index = 0; index < out_chan_indexes.size(); ++index) {
    std::shared_ptr<InterpValue::Channel> channel =
        config_args[out_chan_indexes[index]].GetChannelOrDie();
    all_channel_values[out_ir_channel_names[index]] =
        std::vector<InterpValue>(channel->begin(), channel->end());
  }
  return all_channel_values;
}

}  // namespace

PYBIND11_MODULE(interpreter, m) {
  ImportStatusModule();

  m.def(
      "convert_function_kwargs",
      [](std::string_view top_name, const ImportData& import_data,
         const TypecheckedModule& tm,
         const std::vector<std::vector<InterpValue>>& args_batch)
          -> absl::StatusOr<std::vector<std::vector<InterpValue>>> {
        std::optional<ModuleMember*> module_member =
            tm.module->FindMemberWithName(top_name);
        XLS_CHECK(module_member.has_value());
        ModuleMember* member = module_member.value();
        XLS_CHECK(std::holds_alternative<dslx::Function*>(*member));
        dslx::Function* f = std::get<dslx::Function*>(*member);
        return ConvertFunctionKwargs(f, import_data, tm, args_batch);
      },
      py::arg("top_name"), py::arg("import_data"), py::arg("tm"),
      py::arg("args_batch"));

  m.def(
      "run_function_batched",
      [](std::string_view top_name, ImportData& import_data,
         const TypecheckedModule& tm,
         const std::vector<std::vector<InterpValue>>& args_batch)
          -> absl::StatusOr<std::vector<InterpValue>> {
        std::optional<ModuleMember*> module_member =
            tm.module->FindMemberWithName(top_name);
        XLS_CHECK(module_member.has_value());
        ModuleMember* member = module_member.value();
        XLS_CHECK(std::holds_alternative<dslx::Function*>(*member));
        dslx::Function* f = std::get<dslx::Function*>(*member);
        return RunFunctionBatched(f, import_data, tm, args_batch);
      },
      py::arg("top_name"), py::arg("import_data"), py::arg("tm"),
      py::arg("args_batch"));

  m.def(
      "convert_channel_values",
      [](std::string_view top_name, const ImportData& import_data,
         const TypecheckedModule& tm,
         const std::vector<std::vector<InterpValue>>& args_batch)
          -> absl::StatusOr<absl::flat_hash_map<std::string, InterpValue>> {
        std::optional<ModuleMember*> module_member =
            tm.module->FindMemberWithName(top_name);
        XLS_CHECK(module_member.has_value());
        ModuleMember* member = module_member.value();

        XLS_CHECK(std::holds_alternative<dslx::Proc*>(*member));
        dslx::Proc* proc = std::get<dslx::Proc*>(*member);
        return ConvertChannelValues(proc, import_data, tm, args_batch);
      },
      py::arg("top_name"), py::arg("import_data"), py::arg("tm"),
      py::arg("args_batch"));

  m.def(
      "run_proc",
      [](std::string_view top_name, ImportData& import_data,
         const TypecheckedModule& tm,
         const absl::flat_hash_map<std::string, InterpValue>&
             input_channel_values,
         int64_t proc_ticks)
          -> absl::StatusOr<
              absl::flat_hash_map<std::string, std::vector<InterpValue>>> {
        std::optional<ModuleMember*> module_member =
            tm.module->FindMemberWithName(top_name);
        XLS_CHECK(module_member.has_value());
        ModuleMember* member = module_member.value();

        XLS_CHECK(std::holds_alternative<dslx::Proc*>(*member));
        dslx::Proc* proc = std::get<dslx::Proc*>(*member);
        return RunProc(proc, import_data, tm, input_channel_values, proc_ticks);
      },
      py::arg("top_name"), py::arg("import_data"), py::arg("tm"),
      py::arg("input_channel_values"), py::arg("proc_ticks"));
}

}  // namespace xls::dslx
