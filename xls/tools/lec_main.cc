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

// Tool to prove or disprove logical equivalence of XLS IR and a netlist.

#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xls/common/exit_status.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/init_xls.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/netlist/cell_library.h"
#include "xls/netlist/function_extractor.h"
#include "xls/netlist/lib_parser.h"
#include "xls/netlist/netlist.h"
#include "xls/netlist/netlist.pb.h"
#include "xls/netlist/netlist_parser.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/solvers/z3_lec.h"
#include "../z3/src/api/z3_api.h"

ABSL_FLAG(std::string, cell_lib_path, "",
          "Path to the cell library. "
          "Either this or cell_proto_path should be set.");
ABSL_FLAG(std::string, cell_proto_path, "",
          "Path to the preprocessed cell library proto. "
          "This is a whole bunch faster than specifying an unprocessed "
          "cell library and should be favored.\n"
          "Either this or --cell_lib_path should be set.");
ABSL_FLAG(std::string, constraints_file, "",
          "Optional path to a DSLX file containing a input parameter "
          "constraint function. This function must have the same signature as "
          "the function being compared and must be called either \"main\" or "
          "have the same name as its containing package or else "
          "be the only function in the file. The "
          "function must return 1 for cases where all constraints are "
          "satisfied, and 0 otherwise.\n"
          "For example, a constraint function to restrict arg 0 to be == 5 "
          "and arg 1, tuple element 0 to be > 6 would be as follows:\n"
          "  fn constraints(a: bits[3], b: (bits[8], bits[16])) -> bits[1] {\n"
          "    let x0 = a == 5;\n"
          "    let x1 = b[0] > 6;\n"
          "    x0 & x1;\n"
          "  }");
ABSL_FLAG(std::string, entry_function_name, "",
          "Function (in the IR) to compare. If unset, the program will attempt "
          "to find and use an appropriate entry function.");
ABSL_FLAG(std::string, netlist_module_name, "",
          "Module name (in the netlist) to compare. If unset, the program will "
          "use the name of the entry function in the IR.");
ABSL_FLAG(std::string, ir_path, "", "Path to the XLS IR to compare against.");
ABSL_FLAG(std::string, netlist_path, "", "Path to the netlist.");
ABSL_FLAG(std::string, schedule_path, "",
          "Path to a PipelineSchedule textproto containing the schedule.\n"
          "!IMPORTANT! If the netlist spans multiple stages, a schedule MUST "
          "be specified. Otherwise, mapping IR nodes to netlist cells is "
          "impossible.");
ABSL_FLAG(int32_t, timeout_sec, -1,
          "Amount of time to allow for a LEC operation.");
ABSL_FLAG(bool, auto_stage, false,
          "If true, then the tool will determine on its own whether to perform "
          "staged or full LEC. This requires that a schedule be specified.");
ABSL_FLAG(int32_t, stage, -1,
          "Pipeline stage to evaluate. Requires --schedule.\n"
          "If \"schedule\" is set, but this is not, then the entire module "
          "will be evaluated.");

namespace xls {
namespace {

static constexpr std::string_view kIrConverterPath =
    "xls/dslx/ir_convert/ir_converter_main";

// Loads a cell library, either from a raw Liberty file or a preprocessed
// CellLibraryProto proto.
absl::StatusOr<netlist::CellLibrary> GetCellLibrary(
    std::string_view cell_lib_path, std::string_view cell_proto_path) {
  if (!cell_proto_path.empty()) {
    XLS_ASSIGN_OR_RETURN(std::string cell_proto_text,
                         GetFileContents(cell_proto_path));
    netlist::CellLibraryProto cell_proto;
    XLS_RET_CHECK(cell_proto.ParseFromString(cell_proto_text));
    return netlist::CellLibrary::FromProto(cell_proto);
  }
  XLS_ASSIGN_OR_RETURN(std::string lib_text, GetFileContents(cell_lib_path));
  XLS_ASSIGN_OR_RETURN(auto stream,
                       netlist::cell_lib::CharStream::FromText(lib_text));
  XLS_ASSIGN_OR_RETURN(netlist::CellLibraryProto proto,
                       netlist::function::ExtractFunctions(&stream));
  return netlist::CellLibrary::FromProto(proto);
}

// Loads and parses a netlist from a file.
absl::StatusOr<std::unique_ptr<netlist::rtl::Netlist>> GetNetlist(
    std::string_view netlist_path, netlist::CellLibrary* cell_library) {
  XLS_ASSIGN_OR_RETURN(std::string netlist_text, GetFileContents(netlist_path));
  netlist::rtl::Scanner scanner(netlist_text);
  return netlist::rtl::Parser::ParseNetlist(cell_library, &scanner);
}

// Returns true if the given function contains a large ( > 8 bit) multiply op.
bool IrContainsBigMul(Function* function) {
  for (const auto* node : function->nodes()) {
    if (node->Is<ArithOp>()) {
      const ArithOp* binop = node->As<ArithOp>();
      if (binop->OpIn({Op::kSMul, Op::kUMul})) {
        // Muls only work on Bits types.
        BitsType* type = binop->GetType()->AsBitsOrDie();
        if (type->GetFlatBitCount() > 8) {
          return true;
        }
      }
    }
  }
  return false;
}

// Alarm handling logic: to set timeouts, we need some means of interrupting the
// current procedure, for which we fall back to good ol' UNIX signals.
// Here's some static data we need for managing between thread contexts and for
// setting/clearing alarms themselves.
static absl::Mutex mutex;
static Z3_context active_context;
static bool z3_interrupted = false;
void AlarmHandler(int signum) {
  absl::MutexLock lock(&mutex);
  Z3_interrupt(active_context);
  z3_interrupted = true;
}

struct sigaction SetAlarm(int duration) {
  // We don't do anything with the old handler.
  struct sigaction new_action;
  memset(&new_action, 0, sizeof(new_action));
  new_action.sa_handler = AlarmHandler;
  new_action.sa_flags = 0;
  struct sigaction old_action;
  sigaction(SIGALRM, &new_action, &old_action);
  alarm(duration);
  return old_action;
}

void CancelAlarm(struct sigaction old_action) {
  alarm(0);
  struct sigaction dummy;
  sigaction(SIGALRM, &old_action, &dummy);
}

// This function applies heuristics to determine whether or not a full LEC can
// be performed or if we should break into stages. For now, these are simple:
// does the IR contain a greater-than-8-bit MUL?
absl::Status AutoStage(const solvers::z3::LecParams& lec_params,
                       const PipelineSchedule& schedule, int timeout_sec) {
  bool do_staged = false;

  // Other staged/full heuristics should go here.
  do_staged |= IrContainsBigMul(lec_params.ir_function);

  if (do_staged) {
    std::cout << "Performing staged LEC.\n";
    for (int i = 0; i < schedule.length(); i++) {
      std::cout << "Stage " << i << "...";
      XLS_ASSIGN_OR_RETURN(
          auto lec, solvers::z3::Lec::CreateForStage(lec_params, schedule, i));

      z3_interrupted = false;
      struct sigaction old_action = SetAlarm(timeout_sec);
      bool equal = lec->Run();
      CancelAlarm(old_action);
      absl::MutexLock lock(&mutex);
      if (z3_interrupted) {
        std::cout << "TIMED OUT!\n";
        continue;
      }

      if (equal) {
        std::cout << "PASSED!\n";
      } else {
        std::cout << "FAILED!\n";
        std::cout << '\n' << "IR/netlist value dump:" << '\n';
        lec->DumpIrTree();
      }
    }
  } else {
    std::cout << "Performing full LEC.\n";
    XLS_ASSIGN_OR_RETURN(auto lec, solvers::z3::Lec::Create(lec_params));
    bool equal = lec->Run();
    std::cout << lec->ResultToString() << '\n';
    if (!equal) {
      std::cout << '\n' << "IR/netlist value dump:" << '\n';
      lec->DumpIrTree();
    }
  }

  return absl::OkStatus();
}

}  // namespace

static absl::Status RealMain(
    std::string_view ir_path, std::string_view entry_function_name,
    std::string_view netlist_module_name, std::string_view cell_lib_path,
    std::string_view cell_proto_path, std::string_view netlist_path,
    std::string_view constraints_file, std::string_view schedule_path,
    int stage, bool auto_stage, int timeout_sec) {
  solvers::z3::LecParams lec_params;
  XLS_ASSIGN_OR_RETURN(std::string ir_text, GetFileContents(ir_path));
  XLS_ASSIGN_OR_RETURN(auto package, Parser::ParsePackage(ir_text));
  lec_params.ir_package = package.get();
  if (entry_function_name.empty()) {
    XLS_ASSIGN_OR_RETURN(lec_params.ir_function,
                         lec_params.ir_package->GetTopAsFunction());
  } else {
    XLS_ASSIGN_OR_RETURN(
        lec_params.ir_function,
        lec_params.ir_package->GetFunction(entry_function_name));
  }
  XLS_ASSIGN_OR_RETURN(auto cell_library,
                       GetCellLibrary(cell_lib_path, cell_proto_path));
  XLS_ASSIGN_OR_RETURN(auto netlist, GetNetlist(netlist_path, &cell_library));
  lec_params.netlist = netlist.get();
  lec_params.netlist_module_name = netlist_module_name;

  std::unique_ptr<solvers::z3::Lec> lec;
  if (!schedule_path.empty()) {
    XLS_ASSIGN_OR_RETURN(
        PackagePipelineSchedulesProto proto,
        ParseTextProtoFile<PackagePipelineSchedulesProto>(schedule_path));
    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        PipelineSchedule::FromProto(lec_params.ir_function, proto));
    if (auto_stage) {
      return AutoStage(lec_params, schedule, timeout_sec);
    }
    XLS_ASSIGN_OR_RETURN(
        lec, solvers::z3::Lec::CreateForStage(lec_params, schedule, stage));
  } else {
    XLS_ASSIGN_OR_RETURN(lec, solvers::z3::Lec::Create(lec_params));
  }

  std::unique_ptr<Package> constraints_pkg;
  if (!constraints_file.empty()) {
    XLS_ASSIGN_OR_RETURN(std::filesystem::path ir_converter_path,
                         GetXlsRunfilePath(kIrConverterPath));
    std::vector<std::string> args;
    args.push_back(ir_converter_path);
    args.push_back(std::string(constraints_file));
    XLS_ASSIGN_OR_RETURN(auto stdout_and_stderr,
                         SubprocessResultToStrings(
                             SubprocessErrorAsStatus(InvokeSubprocess(args))));

    XLS_ASSIGN_OR_RETURN(constraints_pkg,
                         Parser::ParsePackage(stdout_and_stderr.first));
    XLS_ASSIGN_OR_RETURN(Function * function,
                         constraints_pkg->GetTopAsFunction());
    XLS_RETURN_IF_ERROR(lec->AddConstraints(function));
  }

  struct sigaction old_action;
  if (timeout_sec != -1) {
    old_action = SetAlarm(timeout_sec);
  }
  bool equal = lec->Run();
  if (timeout_sec != -1) {
    CancelAlarm(old_action);
  }
  absl::MutexLock lock(&mutex);
  if (z3_interrupted) {
    return absl::DeadlineExceededError("LEC timed out.");
  }

  std::cout << lec->ResultToString() << '\n';
  if (!equal) {
    std::cout << '\n' << "IR/netlist value dump:" << '\n';
    lec->DumpIrTree();
  }

  return absl::OkStatus();
}

}  // namespace xls

int main(int argc, char* argv[]) {
  xls::InitXls(argv[0], argc, argv);

  std::string ir_path = absl::GetFlag(FLAGS_ir_path);
  QCHECK(!ir_path.empty()) << "--ir_path must be set.";

  std::string netlist_path = absl::GetFlag(FLAGS_netlist_path);
  QCHECK(!netlist_path.empty()) << "--netlist_path must be set.";

  std::string cell_lib_path = absl::GetFlag(FLAGS_cell_lib_path);
  std::string cell_proto_path = absl::GetFlag(FLAGS_cell_proto_path);
  QCHECK(cell_lib_path.empty() ^ cell_proto_path.empty())
      << "One (and only one) of --cell_lib_path and --cell_proto_path "
         "should be set.";

  std::string schedule_path = absl::GetFlag(FLAGS_schedule_path);
  int stage = absl::GetFlag(FLAGS_stage);
  QCHECK(stage == -1 || !schedule_path.empty())
      << "--schedule_path must be specified with --stage.";

  bool auto_stage = absl::GetFlag(FLAGS_auto_stage);
  QCHECK(!(auto_stage && stage != -1))
      << "Only one of --stage or --auto_stage may be specified.";

  QCHECK(!(auto_stage && schedule_path.empty()))
      << "--schedule_path must be specified with --auto_stage.";

  return xls::ExitStatus(xls::RealMain(
      ir_path, absl::GetFlag(FLAGS_entry_function_name),
      absl::GetFlag(FLAGS_netlist_module_name), cell_lib_path, cell_proto_path,
      netlist_path, absl::GetFlag(FLAGS_constraints_file), schedule_path, stage,
      auto_stage, absl::GetFlag(FLAGS_timeout_sec)));
}
