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

#include <iostream>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/proc_network_interpreter.h"
#include "xls/ir/ir_parser.h"
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

namespace xls {

absl::Status RunIrInterpreter(Package* package, int64_t ticks) {
  XLS_ASSIGN_OR_RETURN(auto interpreter,
                       ProcNetworkInterpreter::Create(package, {}));
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
  return absl::OkStatus();
}

absl::Status RunSerialJit(Package* package, int64_t ticks) {
  XLS_ASSIGN_OR_RETURN(auto runtime, SerialProcRuntime::Create(package));
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
  return absl::OkStatus();
}

absl::Status RealMain(absl::string_view ir_file, absl::string_view backend,
                      int64_t ticks) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_file));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));

  if (backend == "serial_jit") {
    return RunSerialJit(package.get(), ticks);
  } else {
    return RunIrInterpreter(package.get(), ticks);
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

  XLS_QCHECK_OK(xls::RealMain(positional_args[0], backend, ticks));

  return 0;
}
