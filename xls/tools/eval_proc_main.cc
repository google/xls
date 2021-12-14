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

// Tool to evaluate the behavior of a Proc network.

#include <cstdint>
#include <iostream>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_network_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/ir_parser.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/serial_proc_runtime.h"

constexpr const char* kUsage = R"(
Evaluates an IR file containing Procs. The Proc network will be ticked a
fixed number of times (specified on the command line) and the final state
value of each will be printed to the terminal upon completion.

Initial states are set according their declarations inside the IR itself.
)";

ABSL_FLAG(int64_t, ticks, -1, "Number of clock ticks to execute.");
ABSL_FLAG(std::string, backend, "serial_jit",
          "Backend to use for evaluation. Valid options are:\n"
          " - serial_jit : JIT-backed single-stepping runtime.\n"
          " - ir_interpreter     : Interpreter at the IR level.");
ABSL_FLAG(
    std::vector<std::string>, inputs_for_channels, {},
    "Comma separated list of channel=filename pairs, for example: ch_a=foo.ir. "
    "Files contain one XLS Value in human-readable form per line.");
ABSL_FLAG(
    std::vector<std::string>, expected_outputs_for_channels, {},
    "Comma separated list of channel=filename pairs, for example: ch_a=foo.ir. "
    "Files contain one XLS Value in human-readable form per line");

namespace xls {

absl::Status RunIrInterpreter(
    Package* package, int64_t ticks,
    absl::flat_hash_map<std::string, std::vector<Value>> inputs_for_channels,
    absl::flat_hash_map<std::string, std::vector<Value>>
        expected_outputs_for_channels) {
  XLS_ASSIGN_OR_RETURN(auto interpreter,
                       ProcNetworkInterpreter::Create(package, {}));

  ChannelQueueManager& queue_manager = interpreter->queue_manager();
  for (const auto& [channel_name, values] : inputs_for_channels) {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * in_queue,
                         queue_manager.GetQueueByName(channel_name));
    for (const Value& value : values) {
      XLS_RETURN_IF_ERROR(in_queue->Enqueue(value));
    }
  }

  XLS_CHECK_GT(ticks, 0);
  for (int i = 0; i < ticks; i++) {
    XLS_RETURN_IF_ERROR(interpreter->Tick());
  }

  // Sort the keys for stable print order.
  absl::flat_hash_map<Proc*, Value> states = interpreter->ResolveState();
  std::vector<Proc*> sorted_procs;
  for (const auto& [k, v] : states) {
    sorted_procs.push_back(k);
  }

  std::sort(sorted_procs.begin(), sorted_procs.end(),
            [](Proc* a, Proc* b) { return a->name() < b->name(); });
  for (const auto& proc : sorted_procs) {
    std::cout << "Proc " << proc->name() << " : " << states.at(proc)
              << std::endl;
  }

  for (const auto& [channel_name, values] : expected_outputs_for_channels) {
    XLS_ASSIGN_OR_RETURN(ChannelQueue * out_queue,
                         queue_manager.GetQueueByName(channel_name));
    for (const Value& value : values) {
      XLS_ASSIGN_OR_RETURN(Value out_val, out_queue->Dequeue());
      XLS_RET_CHECK_EQ(value, out_val);
    }
  }

  return absl::OkStatus();
}

absl::Status RunSerialJit(
    Package* package, int64_t ticks,
    absl::flat_hash_map<std::string, std::vector<Value>> inputs_for_channels,
    absl::flat_hash_map<std::string, std::vector<Value>>
        expected_outputs_for_channels) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> runtime,
                       SerialProcRuntime::Create(package));

  for (const auto& [channel_name, values] : inputs_for_channels) {
    XLS_ASSIGN_OR_RETURN(Channel * in_ch, package->GetChannel(channel_name));
    for (const Value& value : values) {
      XLS_RETURN_IF_ERROR(runtime->EnqueueValueToChannel(in_ch, value));
    }
  }

  XLS_CHECK_GT(ticks, 0);
  // If Tick() semantics change such that it returns once all Procs have run
  // _at_all_ (instead of only returning when all procs have fully completed),
  // then number-of-ticks-based timing won't work and we'll need to run based on
  // collecting some number of outputs.
  for (int64_t i = 0; i < ticks; i++) {
    XLS_RETURN_IF_ERROR(runtime->Tick());
  }

  for (int64_t i = 0; i < runtime->NumProcs(); i++) {
    XLS_ASSIGN_OR_RETURN(Proc * p, runtime->proc(i));
    XLS_ASSIGN_OR_RETURN(Value v, runtime->ProcState(i));
    std::cout << "Proc " << p->name() << " : " << v << std::endl;
  }

  bool checked_any_output = false;

  for (const auto& [channel_name, values] : expected_outputs_for_channels) {
    XLS_ASSIGN_OR_RETURN(Channel * out_ch, package->GetChannel(channel_name));
    for (const Value& value : values) {
      XLS_ASSIGN_OR_RETURN(Value out_val,
                           runtime->DequeueValueFromChannel(out_ch));
      XLS_RET_CHECK_EQ(value, out_val);
      checked_any_output = true;
    }
  }

  return checked_any_output
             ? absl::OkStatus()
             : absl::UnknownError(
                   "No output verified (empty expected values?)");
}

absl::StatusOr<std::vector<Value>> ParseValuesFile(std::string_view filename) {
  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(filename));
  std::vector<Value> ret;
  for (const auto& line :
       absl::StrSplit(contents, '\n', absl::SkipWhitespace())) {
    XLS_ASSIGN_OR_RETURN(Value expected_status, Parser::ParseTypedValue(line));
    ret.push_back(expected_status);
  }
  return ret;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::string>>
ParseChannelFilenames(std::vector<std::string> files_raw) {
  absl::flat_hash_map<std::string, std::string> ret;
  for (const std::string& file : files_raw) {
    std::vector<std::string> split = absl::StrSplit(file, '=');
    if (split.size() != 2) {
      return absl::InvalidArgumentError(
          "Format of argument should be channel=file");
    }

    ret[split[0]] = split[1];
  }
  return ret;
}

absl::Status RealMain(
    absl::string_view ir_file, absl::string_view backend, int64_t ticks,
    std::vector<std::string> inputs_for_channels_text,
    std::vector<std::string> expected_outputs_for_channels_text) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));

  absl::flat_hash_map<std::string, std::string> input_filenames;
  XLS_ASSIGN_OR_RETURN(input_filenames,
                       ParseChannelFilenames(inputs_for_channels_text));
  absl::flat_hash_map<std::string, std::vector<Value>> inputs_for_channels;
  for (const auto& [channel_name, filename] : input_filenames) {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> values, ParseValuesFile(filename));
    inputs_for_channels[channel_name] = values;
  }

  absl::flat_hash_map<std::string, std::string> expected_filenames;
  XLS_ASSIGN_OR_RETURN(
      expected_filenames,
      ParseChannelFilenames(expected_outputs_for_channels_text));
  absl::flat_hash_map<std::string, std::vector<Value>>
      expected_outputs_for_channels;
  for (const auto& [channel_name, filename] : expected_filenames) {
    XLS_ASSIGN_OR_RETURN(std::vector<Value> values, ParseValuesFile(filename));
    expected_outputs_for_channels[channel_name] = values;
  }

  if (backend == "serial_jit") {
    return RunSerialJit(package.get(), ticks, inputs_for_channels,
                        expected_outputs_for_channels);
  } else {
    return RunIrInterpreter(package.get(), ticks, inputs_for_channels,
                            expected_outputs_for_channels);
  }
}

}  // namespace xls

int main(int argc, char* argv[]) {
  std::vector<absl::string_view> positional_args =
      xls::InitXls(kUsage, argc, argv);
  if (positional_args.size() != 1) {
    XLS_LOG(QFATAL) << "One (and only one) IR file must be given.";
  }

  std::string backend = absl::GetFlag(FLAGS_backend);
  if (backend != "serial_jit" && backend != "ir_interpreter") {
    XLS_LOG(QFATAL) << "Unrecognized backend choice.";
  }

  int64_t ticks = absl::GetFlag(FLAGS_ticks);
  if (ticks <= 0) {
    XLS_LOG(QFATAL) << "--ticks must be specified (and > 0).";
  }

  XLS_QCHECK_OK(
      xls::RealMain(positional_args[0], backend, ticks,
                    absl::GetFlag(FLAGS_inputs_for_channels),
                    absl::GetFlag(FLAGS_expected_outputs_for_channels)));

  return 0;
}
