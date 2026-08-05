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

#include "xls/spin/spin_runner.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "google/protobuf/text_format.h"
#include "re2/re2.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/subprocess.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_kind.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/spin/promela_generator.h"
#include "xls/spin/trace_compare.h"

namespace xls::spin {

namespace {

constexpr std::string_view kDslxLocationPattern = R"(:(\d+:\d+))";
constexpr std::string_view kXlsAssertLabelPattern = R"(XLS_ASSERT:([^\n]+))";

// Output dir for one SPIN run: TEST_UNDECLARED_OUTPUTS_DIR/<subdir> if set,
// otherwise a temp directory deleted on destruction. Use operator/ for paths.
class OutputDir {
 public:
  static absl::StatusOr<OutputDir> Create(std::string_view subdir) {
    OutputDir result;
    if (const char* base = std::getenv("TEST_UNDECLARED_OUTPUTS_DIR");
        base != nullptr && *base != '\0') {
      result.path_ = std::filesystem::path(base) / subdir;
      XLS_RETURN_IF_ERROR(RecursivelyCreateDir(result.path_));
    } else {
      XLS_ASSIGN_OR_RETURN(result.tmp_, TempDirectory::Create(subdir));
      result.path_ = result.tmp_->path();
    }
    return result;
  }

  std::filesystem::path operator/(std::string_view name) const {
    return path_ / name;
  }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
  std::optional<TempDirectory> tmp_;
};

// Returns the #[test_proc] identifier to use as SPIN top, or "" if none.
// Uses test_filter to disambiguate multiple test procs; falls back to first.
std::string SelectTestProc(dslx::Module* module,
                           std::string_view entry_module_path,
                           const std::optional<std::string>& test_filter) {
  std::vector<dslx::TestProc*> test_procs = module->GetTestProcs();
  LOG(INFO) << "Found " << test_procs.size() << " test proc(s) in "
            << entry_module_path;
  if (test_procs.empty()) {
    return "";
  }
  if (test_procs.size() == 1) {
    return test_procs[0]->identifier();
  }

  if (test_filter.has_value()) {
    RE2 re(*test_filter);
    for (dslx::TestProc* test_proc : test_procs) {
      if (RE2::FullMatch(test_proc->identifier(), re)) {
        return test_proc->identifier();
      }
    }
  }
  std::string top = test_procs[0]->identifier();
  LOG(WARNING) << "multiple #[test_proc] entries; using " << top;
  return top;
}

// Generates Promela from `package` and writes it to `pml_path`.
absl::Status GenerateAndWritePml(Package* package,
                                 const PromelaGeneratorOptions& options,
                                 const std::filesystem::path& promela_path) {
  XLS_ASSIGN_OR_RETURN(std::string pml,
                       PromelaGenerator::Generate(package, options));
  LOG(INFO) << "Generated Promela model";
  XLS_RETURN_IF_ERROR(SetFileContents(promela_path, pml));
  LOG(INFO) << "Wrote Promela model to " << promela_path.string();
  return absl::OkStatus();
}

// Returns the "LINE:COL" source location of the Assert node whose label
// matches spin_label, or "" if not found.
std::string FindAssertLocation(Package* package, std::string_view spin_label) {
  for (FunctionBase* function_base : package->GetFunctionBases()) {
    for (Node* node : function_base->nodes()) {
      if (!node->Is<Assert>()) {
        continue;
      }
      const Assert* assert_op = node->As<Assert>();
      if (assert_op->label().has_value() && *assert_op->label() == spin_label) {
        std::string dslx_location;
        RE2::PartialMatch(assert_op->message(), kDslxLocationPattern,
                          &dslx_location);
        if (!dslx_location.empty()) {
          return dslx_location;
        }
      }
    }
  }
  return "";
}

// Returns the XLS_ASSERT label if spin_out contains "assertion violated",
// or nullopt if no violation. The label may be empty for non-XLS assertions.
std::optional<std::string> ExtractAssertionLabel(std::string_view spin_out) {
  if (!absl::StrContains(spin_out, "assertion violated")) {
    return std::nullopt;
  }
  std::string label;
  RE2::PartialMatch(spin_out, kXlsAssertLabelPattern, &label);
  return label;
}

// Cross-checks a SPIN assertion violation against the DSLX trace.
// Returns OK if DSLX fired the same assertion (consistent failure).
absl::Status CheckGuidedAssertion(Package* package, std::string_view spin_label,
                                  const EvaluatorResultsProto& proto) {
  LOG(WARNING) << "SPIN assertion violated: '" << spin_label << "'";
  const std::string assert_loc = FindAssertLocation(package, spin_label);

  bool had_assertion = false, same_location = false;
  for (const auto& eval_result : proto.results()) {
    for (const auto& assert_msg : eval_result.events().assert_msgs()) {
      had_assertion = true;
      std::string dslx_location;
      RE2::PartialMatch(assert_msg.message(), kDslxLocationPattern,
                        &dslx_location);
      if (!assert_loc.empty() && dslx_location == assert_loc) {
        same_location = true;
      }
    }
  }
  if (same_location) {
    LOG(INFO) << "SPIN assertion '" << spin_label
              << "' matches DSLX assertion; consistent failure";
    return absl::OkStatus();
  }
  if (had_assertion) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Promela assertion \"%s\" violated but does not match the "
        "assertion fired by the DSLX interpreter.",
        spin_label));
  }
  return absl::FailedPreconditionError(absl::StrFormat(
      "Promela assertion \"%s\" violated but the DSLX interpreter "
      "reported no assertion failures.",
      spin_label));
}

// Runs SPIN, logs exit status, writes stdout to dir/"spin_output.log".
absl::StatusOr<SubprocessResult> RunSpin(const std::vector<std::string>& argv,
                                         const OutputDir& dir) {
  XLS_ASSIGN_OR_RETURN(SubprocessResult result,
                       InvokeSubprocess(argv, dir.path()));
  LOG(INFO) << "SPIN complete (exit status " << result.exit_status << ")";
  XLS_RETURN_IF_ERROR(
      SetFileContents(dir / "spin_output.log", result.stdout_content));
  return result;
}

// Runs the DSLX interpreter on `spin_top` with channel tracing and returns the
// EvaluatorResultsProto. Used by RunSpinCheck when no proto is pre-supplied.
absl::StatusOr<EvaluatorResultsProto> CollectDslxTrace(
    std::string_view dslx_source, std::string_view entry_module_path,
    std::string_view module_name, std::string_view spin_top,
    const SpinRunOptions& options) {
  dslx::ParseAndTypecheckOptions typecheck_options;
  typecheck_options.dslx_stdlib_path = options.dslx_stdlib_path;
  typecheck_options.dslx_paths = options.dslx_paths;
  if (options.type_inference_v2) {
    typecheck_options.type_inference_version =
        dslx::TypeInferenceVersion::kVersion2;
  }
  RE2 top_proc_regex{std::string(spin_top)};
  EvaluatorResultsProto proto;
  dslx::ParseAndTestOptions test_options;
  test_options.parse_and_typecheck_options = typecheck_options;
  test_options.test_filter = &top_proc_regex;
  test_options.trace_channels = true;
  test_options.results_out = &proto;
  LOG(INFO) << "Running DSLX interpreter for '" << spin_top << "'";
  dslx::DslxInterpreterTestRunner runner;
  XLS_RETURN_IF_ERROR(runner
                          .ParseAndTest(dslx_source, module_name,
                                        entry_module_path, test_options)
                          .status());
  LOG(INFO) << "DSLX trace collected";
  return proto;
}

absl::StatusOr<std::filesystem::path> FindSpinBinary() {
  auto runfile = GetXlsRunfilePath("spin", "spin");
  if (runfile.ok() && std::filesystem::exists(*runfile)) {
    LOG(INFO) << "Using SPIN binary: " << runfile->string();
    return *runfile;
  }
  LOG(WARNING)
      << "SPIN binary not found in runfiles; falling back to PATH lookup";
  return std::filesystem::path("spin");
}

}  // namespace

absl::Status RunSpinCheck(std::string_view dslx_source,
                          std::string_view entry_module_path,
                          std::string_view module_name,
                          const SpinRunOptions& options) {
  std::string_view mode =
      options.exec_type == SpinExecutionType::kGuided ? "guided" : "exhaustive";
  LOG(INFO) << "Starting SPIN " << mode << " check for module '" << module_name
            << "' (" << entry_module_path << ")";

  dslx::ImportData import_data(dslx::CreateImportData(
      options.dslx_stdlib_path, options.dslx_paths, dslx::kAllWarningsSet,
      std::make_unique<dslx::RealFilesystem>()));
  XLS_ASSIGN_OR_RETURN(dslx::TypecheckedModule tm,
                       dslx::ParseAndTypecheck(dslx_source, entry_module_path,
                                               module_name, &import_data));

  std::string spin_top =
      SelectTestProc(tm.module, entry_module_path, options.test_filter);
  if (spin_top.empty()) {
    LOG(WARNING) << "no #[test_proc] found in " << entry_module_path
                 << "; skipping " << mode << " check";
    return absl::OkStatus();
  }

  LOG(INFO) << "Converting test proc '" << spin_top << "' to optimized IR";
  bool printed_error = false;
  dslx::ConvertOptions convert_options = {
      .emit_positions = true,
      .emit_assert = true,
      .verify_ir = true,
      .warnings_as_errors = false,
      .warnings = dslx::kAllWarningsSet,
      .convert_tests = true,
      .type_inference_v2 = options.type_inference_v2,
      .lower_to_proc_scoped_channels = true,
  };
  std::array<std::string_view, 1> module_path{entry_module_path};
  XLS_ASSIGN_OR_RETURN(
      dslx::PackageConversionData conversion_data,
      dslx::ConvertFilesToPackage(module_path, options.dslx_stdlib_path,
                                  options.dslx_paths, convert_options, spin_top,
                                  module_name, &printed_error));
  XLS_RETURN_IF_ERROR(
      RunOptimizationPassPipeline(conversion_data.package.get()).status());
  LOG(INFO) << "IR optimization complete; running Promela " << mode << " check";

  XLS_ASSIGN_OR_RETURN(std::filesystem::path spin_bin, FindSpinBinary());
  PromelaGeneratorOptions promela_options;
  promela_options.emit_source_hints = true;
  promela_options.emit_termination_hook = true;
  promela_options.emit_progress_labels = true;
  Package* package = conversion_data.package.get();

  if (options.exec_type == SpinExecutionType::kGuided) {
    EvaluatorResultsProto internal_proto;
    const EvaluatorResultsProto* proto_ptr = options.results_proto;
    if (proto_ptr == nullptr) {
      XLS_ASSIGN_OR_RETURN(internal_proto,
                           CollectDslxTrace(dslx_source, entry_module_path,
                                            module_name, spin_top, options));
      proto_ptr = &internal_proto;
    }
    std::string proto_text;
    if (!google::protobuf::TextFormat::PrintToString(*proto_ptr, &proto_text)) {
      return absl::InternalError("Failed to serialize EvaluatorResultsProto");
    }
    XLS_ASSIGN_OR_RETURN(OutputDir dir, OutputDir::Create("guided"));
    const std::filesystem::path promela_path = dir / "model.pml";
    const std::filesystem::path spin_trace_path = dir / "spin_trace.json";
    XLS_RETURN_IF_ERROR(
        GenerateAndWritePml(package, promela_options, promela_path));
    XLS_RETURN_IF_ERROR(
        SetFileContents(dir / "dslx_trace.textproto", proto_text));
    LOG(INFO) << "Running SPIN simulation on " << promela_path.string();
    XLS_ASSIGN_OR_RETURN(
        SubprocessResult spin_result,
        RunSpin({spin_bin.string(), "-c", "-Q", spin_trace_path.string(),
                 promela_path.string()},
                dir));
    if (spin_result.exit_status != 0) {
      if (auto label = ExtractAssertionLabel(spin_result.stdout_content);
          label.has_value()) {
        return CheckGuidedAssertion(package, *label, *proto_ptr);
      }
      return absl::InternalError(absl::StrFormat("spin failed (exit %d):\n%s",
                                                 spin_result.exit_status,
                                                 spin_result.stdout_content));
    }
    const bool timed_out =
        absl::StrContains(spin_result.stdout_content, "timeout");
    std::string_view terminator_channel =
        promela_options.emit_termination_hook ? "terminator" : "";
    XLS_ASSIGN_OR_RETURN(std::string spin_json,
                         GetFileContents(spin_trace_path));
    LOG(INFO) << "Parsing SPIN trace from " << spin_trace_path.string();
    XLS_ASSIGN_OR_RETURN(ProcInstPaths proc_paths, BuildProcInstPathsForSpin(package));
    DslxChannelNameMap channel_name_map = BuildDslxChannelNameMap(*tm.module);
    XLS_ASSIGN_OR_RETURN(TraceMap spin_events,
                         ParseSpinTrace(spin_json, proc_paths,
                                        terminator_channel));
    if (timed_out && !SpinTraceHasTerminator(spin_json, terminator_channel)) {
      return absl::DeadlineExceededError(absl::StrFormat(
          "SPIN simulation reached a deadlock before the terminator fired.\n"
          "The Promela model blocked with no enabled transitions.\n"
          "SPIN output:\n%s",
          spin_result.stdout_content));
    }
    XLS_ASSIGN_OR_RETURN(TraceMap dslx_events,
                         ParseDslxTrace(proto_text, terminator_channel,
                                        channel_name_map));
    LOG(INFO) << "Comparing SPIN and DSLX traces";
    XLS_RETURN_IF_ERROR(CompareTraces(spin_events, dslx_events));
    LOG(INFO) << "Trace comparison passed";
    return absl::OkStatus();
  } else {
    XLS_ASSIGN_OR_RETURN(OutputDir dir, OutputDir::Create("exhaustive"));
    const std::filesystem::path promela_path = dir / "model.pml";
    XLS_RETURN_IF_ERROR(
        GenerateAndWritePml(package, promela_options, promela_path));
    LOG(INFO) << "Running SPIN exhaustive search on " << promela_path.string();
    XLS_ASSIGN_OR_RETURN(
        SubprocessResult spin_result,
        RunSpin({spin_bin.string(), "-search", promela_path.string()}, dir));
    if (auto label = ExtractAssertionLabel(spin_result.stdout_content);
        label.has_value()) {
      return absl::FailedPreconditionError(absl::StrFormat(
          "SPIN exhaustive search found assertion violation: \"%s\".\n"
          "SPIN output:\n%s",
          *label, spin_result.stdout_content));
    }
    return absl::OkStatus();
  }
}

}  // namespace xls::spin
