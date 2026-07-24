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

#include "xls/spin/promela_generator.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/file/temp_directory.h"
#include "xls/common/status/matchers.h"
#include "xls/common/subprocess.h"
#include "xls/common/undeclared_outputs.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/ir_convert/ir_converter.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/dslx/run_routines/run_routines.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/optimization_pass_pipeline.h"
#include "xls/spin/spin_runner.h"
#include "xls/spin/trace_compare.h"

namespace xls::spin {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Not;

struct DslxFunctionParam {
  std::string name;  // e.g. "a"
  std::string type;  // e.g. "u32"
};

/* Helper functions */

// Returns the text from the opening '{' to the matching '}' for the first
// block whose header contains `header`, or "" if not found.
std::string_view BlockBody(std::string_view out, std::string_view header) {
  auto hpos = out.find(header);
  if (hpos == std::string_view::npos) {
    return "";
  }
  auto open = out.find('{', hpos);
  if (open == std::string_view::npos) {
    return "";
  }
  int64_t depth = 0;
  for (size_t i = open; i < out.size(); ++i) {
    if (out[i] == '{') {
      ++depth;
    } else if (out[i] == '}' && --depth == 0) {
      return out.substr(open, i - open + 1);
    }
  }
  return "";
}

std::string_view InitBody(std::string_view out) {
  return BlockBody(out, "\ninit {");
}

std::string_view ProctypeBody(std::string_view out, std::string_view name) {
  return BlockBody(out, absl::StrCat("proctype ", name, "("));
}

std::string_view InlineBody(std::string_view out, std::string_view name) {
  return BlockBody(out, absl::StrCat("inline ", name, "("));
}

// Writes content to TEST_UNDECLARED_OUTPUTS_DIR/<suite>.<test>.<filename>.
// No-op when TEST_UNDECLARED_OUTPUTS_DIR is not set.
void DumpArtifact(std::string_view filename, std::string_view content) {
  std::optional<std::filesystem::path> dir = GetUndeclaredOutputDirectory();
  if (!dir) {
    return;
  }
  const auto* info = testing::UnitTest::GetInstance()->current_test_info();
  std::string test_id =
      info ? absl::StrCat(info->test_suite_name(), ".", info->name())
           : "unknown";
  (void)SetFileContents(*dir / absl::StrCat(test_id, ".", filename), content);
}

// Generates Promela and dumps it as <package_name>.pml; returns the text.
absl::StatusOr<std::string> GenerateAndDump(
    Package* package, const PromelaGeneratorOptions& options = {}) {
  XLS_ASSIGN_OR_RETURN(std::string pml,
                       PromelaGenerator::Generate(package, options));
  DumpArtifact(absl::StrCat(package->name(), ".pml"), pml);
  return pml;
}

// Internal implementation: runs the full pipeline given a resolved source path
// and the directory to use for --dslx_path (so DSLX imports can be resolved).
testing::AssertionResult VerifyDslxPromelaTraces(
    const std::filesystem::path& src_path,
    const std::filesystem::path& dslx_path_dir, std::string_view dslx_top,
    bool emit_termination_hook) {
  auto spin_path = GetXlsRunfilePath("spin", "spin");
  if (!spin_path.ok()) {
    return testing::AssertionFailure() << "spin: " << spin_path.status();
  }

  auto impl = [&]() -> absl::Status {
    std::string module_name = src_path.stem().string();
    XLS_ASSIGN_OR_RETURN(std::string dslx_source, GetFileContents(src_path));

    // Step 1: DSLX interpreter API -> EvaluatorResultsProto.
    dslx::ParseAndTypecheckOptions ptc;
    ptc.dslx_paths = {dslx_path_dir};
    EvaluatorResultsProto proto;
    dslx::ParseAndTestOptions pts;
    pts.parse_and_typecheck_options = ptc;
    pts.trace_channels = true;
    pts.results_out = &proto;
    dslx::DslxInterpreterTestRunner runner;
    XLS_RETURN_IF_ERROR(
        runner.ParseAndTest(dslx_source, module_name, src_path.string(), pts)
            .status());
    std::string proto_text;
    if (!google::protobuf::TextFormat::PrintToString(proto, &proto_text)) {
      return absl::InternalError("proto serialization failed");
    }

    // Steps 2+3: DSLX -> optimised IR (API).
    dslx::ConvertOptions conv_opts;
    conv_opts.emit_positions = true;
    conv_opts.emit_assert = true;
    conv_opts.convert_tests = true;
    conv_opts.lower_to_proc_scoped_channels = true;
    bool printed_error = false;
    const std::string src_path_str = src_path.string();
    std::string_view module_paths[] = {src_path_str};
    XLS_ASSIGN_OR_RETURN(
        dslx::PackageConversionData conv,
        dslx::ConvertFilesToPackage(
            module_paths, /*dslx_stdlib_path=*/"", {dslx_path_dir}, conv_opts,
            std::string(dslx_top), module_name, &printed_error));
    XLS_RETURN_IF_ERROR(
        RunOptimizationPassPipeline(conv.package.get()).status());

    // Step 4: Optimised IR -> Promela (API).
    PromelaGeneratorOptions pml_opts;
    pml_opts.emit_termination_hook = emit_termination_hook;
    pml_opts.emit_source_hints = true;
    XLS_ASSIGN_OR_RETURN(std::string pml, PromelaGenerator::Generate(
                                              conv.package.get(), pml_opts));

    // Write Promela to a temp dir for SPIN.
    XLS_ASSIGN_OR_RETURN(auto tmp, TempDirectory::Create("dslx_promela_trace"));
    const std::filesystem::path dir = tmp.path();
    const std::filesystem::path pml_file = dir / "out.pml";
    const std::filesystem::path spin_trace = dir / "spin_trace.json";
    XLS_RETURN_IF_ERROR(SetFileContents(pml_file, pml));

    // Step 5: SPIN guided simulation -> JSON trace (still a subprocess).
    std::vector<std::string> argv = {spin_path->string(), "-c", "-Q",
                                     spin_trace.string(), pml_file.string()};
    XLS_ASSIGN_OR_RETURN(SubprocessResult r, InvokeSubprocess(argv, dir));
    if (r.exit_status != 0) {
      return absl::InternalError(absl::StrFormat(
          "spin failed (exit %d):\n%s", r.exit_status, r.stderr_content));
    }

    // Step 6: Parse both traces and compare.
    std::string_view term_chan = emit_termination_hook ? "terminator" : "";
    XLS_ASSIGN_OR_RETURN(std::string spin_json, GetFileContents(spin_trace));

    // Dump artifacts.
    DumpArtifact("harness.x", dslx_source);
    DumpArtifact("out.pml", pml);
    DumpArtifact("dslx_trace.textproto", proto_text);
    DumpArtifact("spin_trace.json", spin_json);
    XLS_ASSIGN_OR_RETURN(ProcInstPaths proc_paths,
                         BuildProcInstPathsForSpin(conv.package.get()));
    XLS_ASSIGN_OR_RETURN(TraceMap spin_events,
                         ParseSpinTrace(spin_json, proc_paths, term_chan));
    // Parse DSLX source to build the channel name map that rewrites variable
    // names (e.g. "req_s") to ChannelDecl strings (e.g. "req").
    dslx::ImportData import_data = dslx::CreateImportDataForTest();
    XLS_ASSIGN_OR_RETURN(
        dslx::TypecheckedModule tm,
        dslx::ParseAndTypecheck(dslx_source, src_path.string(),
                                module_name, &import_data));
    DslxChannelNameMap channel_name_map =
        spin::BuildDslxChannelNameMap(*tm.module);
    XLS_ASSIGN_OR_RETURN(TraceMap dslx_events,
                         ParseDslxTrace(proto_text, term_chan, channel_name_map));
    return CompareTraces(spin_events, dslx_events);
  };

  auto status = impl();
  if (!status.ok()) {
    return testing::AssertionFailure() << status.message();
  }
  return testing::AssertionSuccess();
}

// Generates a DSLX source file with the structure shown in the raw-string
// template at the bottom of this function.  The variable sections below are
// pre-built and substituted into the $PLACEHOLDER tokens.
std::string GenerateFunctionHarnessDslx(
    std::string_view fn_name, const std::vector<DslxFunctionParam>& params,
    std::string_view return_type, std::string_view fn_body,
    const std::vector<std::vector<std::string>>& test_cases,
    std::string_view fn_extras = "") {
  // Per-param string sections (each appended once per param in the loop below).
  std::string fn_params;   // "a: u32, b: u32" for the fn declaration
  std::string h_fields;    // "    chan_a: chan<u32> in;\n" x N
  std::string cfg_params;  // "chan_a: chan<u32> in, chan_b: chan<u32> in"
  std::string cfg_tuple;   // "chan_a, chan_b"
  std::string recvs;       // recv chain in next(state:())
  std::string call_args;   // "a, b"
  std::string t_fields;    // "    chan_a: chan<u32> out;\n" x N
  std::string chan_decls;  // "let (chan_a_s, chan_a_r) = chan<u32>("a");\n" x N
  std::string spawn_args;  // "chan_a_r, chan_b_r," (trailing comma consumed by
                           // $SPAWN_ARGS, key)
  std::string ret_tuple;   // "chan_a_s, chan_b_s," (trailing comma consumed by
                           // $RET_TUPLE, key)
  std::string loop_body;   // dispatch + send chain in next(i:u32)

  for (size_t i = 0; i < params.size(); ++i) {
    const auto& p = params[i];
    absl::StrAppendFormat(&h_fields, "    chan_%s: chan<%s> in;\n", p.name,
                          p.type);
    if (i > 0) {
      absl::StrAppend(&fn_params, ", ");
      absl::StrAppend(&cfg_params, ", ");
      absl::StrAppend(&cfg_tuple, ", ");
      absl::StrAppend(&call_args, ", ");
      absl::StrAppend(&spawn_args, ", ");
      absl::StrAppend(&ret_tuple, ", ");
    }
    absl::StrAppendFormat(&fn_params, "%s: %s", p.name, p.type);
    absl::StrAppendFormat(&cfg_params, "chan_%s: chan<%s> in", p.name, p.type);
    absl::StrAppendFormat(&cfg_tuple, "chan_%s", p.name);
    absl::StrAppendFormat(&recvs,
                          "        let (tok, %s) = recv(%s, chan_%s);\n",
                          p.name, i == 0 ? "join()" : "tok", p.name);
    absl::StrAppend(&call_args, p.name);
    absl::StrAppendFormat(&t_fields, "    chan_%s: chan<%s> out;\n", p.name,
                          p.type);
    absl::StrAppendFormat(
        &chan_decls,
        "        let (chan_%s_s, chan_%s_r) = chan<%s>(\"chan_%s\");\n", p.name,
        p.name, p.type, p.name);
    absl::StrAppendFormat(&spawn_args, "chan_%s_r", p.name);
    absl::StrAppendFormat(&ret_tuple, "chan_%s_s", p.name);
    absl::StrAppendFormat(&loop_body,
                          "        let tok = send(%s, chan_%s, v_%s);\n",
                          i == 0 ? "join()" : "tok", p.name, p.name);
  }

  // Build the full function declaration from components.
  std::string fn_decl;
  if (!fn_extras.empty()) {
    absl::StrAppendFormat(&fn_decl, "%s\n", fn_extras);
  }
  absl::StrAppendFormat(&fn_decl, "fn %s(%s) -> %s { %s }", fn_name, fn_params,
                        return_type, fn_body);

  // Build the test-case dispatch + send block.
  // One test case: simple bindings.  Multiple cases: match expression.
  std::string dispatch;
  if (test_cases.size() == 1) {
    for (size_t pi = 0; pi < params.size(); ++pi) {
      absl::StrAppendFormat(&dispatch, "        let v_%s = %s;\n",
                            params[pi].name, test_cases[0][pi]);
    }
  } else if (params.size() == 1) {
    absl::StrAppendFormat(&dispatch, "        let v_%s: %s = match i {\n",
                          params[0].name, params[0].type);
    for (size_t tc = 0; tc < test_cases.size(); ++tc) {
      if (tc + 1 < test_cases.size()) {
        absl::StrAppendFormat(&dispatch, "            u32:%zu => %s,\n", tc,
                              test_cases[tc][0]);
      } else {
        absl::StrAppendFormat(&dispatch, "            _ => %s,\n",
                              test_cases[tc][0]);
      }
    }
    absl::StrAppend(&dispatch, "        };\n");
  } else {
    absl::StrAppend(&dispatch, "        let (");
    for (size_t pi = 0; pi < params.size(); ++pi) {
      if (pi > 0) {
        absl::StrAppend(&dispatch, ", ");
      }
      absl::StrAppendFormat(&dispatch, "v_%s", params[pi].name);
    }
    absl::StrAppend(&dispatch, "): (");
    for (size_t pi = 0; pi < params.size(); ++pi) {
      if (pi > 0) {
        absl::StrAppend(&dispatch, ", ");
      }
      absl::StrAppend(&dispatch, params[pi].type);
    }
    absl::StrAppend(&dispatch, ") = match i {\n");
    for (size_t tc = 0; tc < test_cases.size(); ++tc) {
      if (tc + 1 < test_cases.size()) {
        absl::StrAppendFormat(&dispatch, "            u32:%zu => (", tc);
      } else {
        absl::StrAppend(&dispatch, "            _ => (");
      }
      for (size_t pi = 0; pi < params.size(); ++pi) {
        if (pi > 0) {
          absl::StrAppend(&dispatch, ", ");
        }
        absl::StrAppend(&dispatch, test_cases[tc][pi]);
      }
      absl::StrAppend(&dispatch, "),\n");
    }
    absl::StrAppend(&dispatch, "        };\n");
  }
  loop_body = dispatch + loop_body;

  std::string last_idx = absl::StrCat(test_cases.size() - 1);

  // The raw string below is the exact DSLX that gets generated.
  // $SPAWN_ARGS, and $RET_TUPLE, include the trailing comma so the template
  // reads as natural DSLX syntax.
  return absl::StrReplaceAll(
      R"(#![feature(type_inference_v2)]

$FN_BODY

proc $NAME_Harness {
$H_FIELDS    chan_result: chan<$RTYPE> out;

    config($CFG_PARAMS, chan_result: chan<$RTYPE> out) {
        ($CFG_TUPLE, chan_result)
    }

    init { () }

    next(state: ()) {
$RECVS        let r = $NAME($CALL_ARGS);
        send(tok, chan_result, r);
    }
}

#[test_proc]
proc $NAME_Test {
$T_FIELDS    chan_result: chan<$RTYPE> in;
    terminator: chan<bool> out;

    config(terminator: chan<bool> out) {
$CHAN_DECLS        let (chan_result_s, chan_result_r) = chan<$RTYPE>("chan_result");
        spawn $NAME_Harness($SPAWN_ARGS, chan_result_s);
        ($RET_TUPLE, chan_result_r, terminator)
    }

    init { u32:0 }

    next(i: u32) {
$LOOP_BODY        let (tok, _) = recv(tok, chan_result);
        send_if(tok, terminator, i == u32:$LAST_IDX, true);
        i + u32:1
    }
}
)",
      {
          {"$FN_BODY", fn_decl},
          {"$NAME", fn_name},
          {"$RTYPE", return_type},
          {"$H_FIELDS", h_fields},
          {"$CFG_PARAMS", cfg_params},
          {"$CFG_TUPLE", cfg_tuple},
          {"$RECVS", recvs},
          {"$CALL_ARGS", call_args},
          {"$T_FIELDS", t_fields},
          {"$CHAN_DECLS", chan_decls},
          {"$SPAWN_ARGS", spawn_args},
          {"$RET_TUPLE", ret_tuple},
          {"$LOOP_BODY", loop_body},
          {"$LAST_IDX", last_idx},
      });
}

/* Test fixture */

class PromelaGeneratorTest : public IrTestBase {
 protected:
  // Loads IR text from testdata/<filename>.
  absl::StatusOr<std::string> LoadTestIr(std::string_view filename);

  // Parses `ir_text` and runs the generator.
  absl::StatusOr<std::string> Generate(std::string_view ir_text);

  // Loads IR from testdata/<filename>, generates Promela, and returns it.
  absl::StatusOr<std::string> GenerateFromIrFile(std::string_view filename);

  // Generate Promela from `ir_text` and verify that `spin -a` accepts it.
  testing::AssertionResult GenerateAndSpinCheck(std::string_view ir_text);

  // Loads IR from the given Bazel runfile path, generates Promela, and verifies
  // that `spin -a` accepts it.
  testing::AssertionResult GenerateAndSpinCheckFromFile(
      std::string_view runfile_path);

  // Writes `promela` to a temp file and runs `spin -a` on it.
  testing::AssertionResult SpinSyntaxOk(std::string_view promela);

  // Runs the full pipeline for inline DSLX source text and compares traces.
  testing::AssertionResult DslxPromelaTracesMatch(
      std::string_view dslx_source, std::string_view dslx_top,
      bool emit_termination_hook = false);

  // Runs the full pipeline for a Bazel runfile path; uses the file's parent
  // directory as --dslx_path so imports resolve correctly.
  testing::AssertionResult DslxPromelaTracesMatchFile(
      std::string_view runfile_path, std::string_view dslx_top,
      bool emit_termination_hook = false);

  // Wraps a DSLX function in a harness + test proc, runs both DSLX interpreter
  // and SPIN simulation, and verifies that per-channel event sequences match.
  testing::AssertionResult DslxFunctionTracesMatch(
      std::string_view fn_name, const std::vector<DslxFunctionParam>& params,
      std::string_view return_type, std::string_view fn_body,
      const std::vector<std::vector<std::string>>& test_cases,
      std::string_view fn_extras = "");
};

absl::StatusOr<std::string> PromelaGeneratorTest::Generate(
    std::string_view ir_text) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> pkg,
                       IrTestBase::ParsePackageNoVerify(ir_text));
  return GenerateAndDump(pkg.get());
}

absl::StatusOr<std::string> PromelaGeneratorTest::LoadTestIr(
    std::string_view filename) {
  XLS_ASSIGN_OR_RETURN(
      std::filesystem::path path,
      GetXlsRunfilePath(absl::StrCat("xls/spin/testdata/", filename)));
  return GetFileContents(path);
}

absl::StatusOr<std::string> PromelaGeneratorTest::GenerateFromIrFile(
    std::string_view filename) {
  XLS_ASSIGN_OR_RETURN(std::string ir_text, LoadTestIr(filename));
  return Generate(ir_text);
}

testing::AssertionResult PromelaGeneratorTest::SpinSyntaxOk(
    std::string_view promela) {
  auto dir_or = TempDirectory::Create("spin_syntax");
  if (!dir_or.ok()) {
    return testing::AssertionFailure()
           << "TempDirectory::Create: " << dir_or.status();
  }
  const std::filesystem::path pml = dir_or->path() / "model.pml";
  if (auto s = SetFileContents(pml, promela); !s.ok()) {
    return testing::AssertionFailure() << "SetFileContents: " << s;
  }
  auto spin_or = GetXlsRunfilePath("spin", "spin");
  if (!spin_or.ok()) {
    return testing::AssertionFailure()
           << "GetXlsRunfilePath(spin): " << spin_or.status();
  }
  std::vector<std::string> argv = {spin_or->string(), "-a", pml.string()};
  auto result_or = InvokeSubprocess(argv, dir_or->path());
  if (!result_or.ok()) {
    return testing::AssertionFailure()
           << "InvokeSubprocess: " << result_or.status();
  }
  if (result_or->exit_status != 0) {
    return testing::AssertionFailure()
           << "spin -a exited " << result_or->exit_status << "\n"
           << "stdout:\n"
           << result_or->stdout_content << "stderr:\n"
           << result_or->stderr_content;
  }
  return testing::AssertionSuccess();
}

testing::AssertionResult PromelaGeneratorTest::GenerateAndSpinCheck(
    std::string_view ir_text) {
  auto pkg_or = IrTestBase::ParsePackageNoVerify(ir_text);
  if (!pkg_or.ok()) {
    return testing::AssertionFailure()
           << "ParsePackageNoVerify: " << pkg_or.status();
  }
  auto pml_or = GenerateAndDump(pkg_or->get());
  if (!pml_or.ok()) {
    return testing::AssertionFailure()
           << "PromelaGenerator::Generate: " << pml_or.status();
  }
  return SpinSyntaxOk(*pml_or);
}

testing::AssertionResult PromelaGeneratorTest::GenerateAndSpinCheckFromFile(
    std::string_view runfile_path) {
  auto ir_path_or = GetXlsRunfilePath(runfile_path);
  if (!ir_path_or.ok()) {
    return testing::AssertionFailure()
           << "GetXlsRunfilePath: " << ir_path_or.status();
  }
  auto ir_text_or = GetFileContents(*ir_path_or);
  if (!ir_text_or.ok()) {
    return testing::AssertionFailure()
           << "GetFileContents: " << ir_text_or.status();
  }
  return GenerateAndSpinCheck(*ir_text_or);
}

testing::AssertionResult PromelaGeneratorTest::DslxPromelaTracesMatch(
    std::string_view dslx_source, std::string_view dslx_top,
    bool emit_termination_hook) {
  auto tmp = TempDirectory::Create("dslx_src");
  if (!tmp.ok())
    return testing::AssertionFailure() << "TempDirectory: " << tmp.status();
  const std::filesystem::path src = tmp->path() / "src.x";
  if (auto s = SetFileContents(src, dslx_source); !s.ok())
    return testing::AssertionFailure() << "writing source: " << s;
  return VerifyDslxPromelaTraces(src, tmp->path(), dslx_top,
                                 emit_termination_hook);
}

testing::AssertionResult PromelaGeneratorTest::DslxPromelaTracesMatchFile(
    std::string_view runfile_path, std::string_view dslx_top,
    bool emit_termination_hook) {
  auto src = GetXlsRunfilePath(runfile_path);
  if (!src.ok())
    return testing::AssertionFailure() << "source file: " << src.status();
  return VerifyDslxPromelaTraces(*src, src->parent_path(), dslx_top,
                                 emit_termination_hook);
}

testing::AssertionResult PromelaGeneratorTest::DslxFunctionTracesMatch(
    std::string_view fn_name, const std::vector<DslxFunctionParam>& params,
    std::string_view return_type, std::string_view fn_body,
    const std::vector<std::vector<std::string>>& test_cases,
    std::string_view fn_extras) {
  std::string dslx_source = GenerateFunctionHarnessDslx(
      fn_name, params, return_type, fn_body, test_cases, fn_extras);
  std::string top = absl::StrCat(fn_name, "_Test");
  return DslxPromelaTracesMatch(dslx_source, top,
                                /*emit_termination_hook=*/true);
}

/* Init block and package header */

TEST_F(PromelaGeneratorTest, InitBlockRunsAllProcs) {
  // The init block spawns a proctype for every proc in the package.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, Generate(R"(
package multi_proc
top proc __multi_proc__ProcA_0_next<>(__state: bits[1], init={0}) {
  __state: bits[1] = state_read(state_element=__state, id=2)
  __token: token = literal(value=token, id=1)
  literal.3: bits[1] = literal(value=1, id=3)
  tuple.4: () = tuple(id=4)
  next_value.5: () = next_value(param=__state, value=__state, id=5)
}
proc __multi_proc__ProcB_0_next<>(__state: bits[1], init={0}) {
  __state: bits[1] = state_read(state_element=__state, id=7)
  __token: token = literal(value=token, id=6)
  literal.8: bits[1] = literal(value=1, id=8)
  tuple.9: () = tuple(id=9)
  next_value.10: () = next_value(param=__state, value=__state, id=10)
}
)"));
  std::string_view init = InitBody(out);
  ASSERT_THAT(init, Not(testing::IsEmpty()));
  EXPECT_THAT(init, HasSubstr("run __multi_proc__ProcA_0_next();"));
  EXPECT_THAT(init, HasSubstr("run __multi_proc__ProcB_0_next();"));
}

TEST_F(PromelaGeneratorTest, NoInitBlockForFunctionPackage) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("func.ir"));
  EXPECT_THAT(out, Not(HasSubstr("init {")));
}

TEST_F(PromelaGeneratorTest, HeaderContainsPackageName) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("func.ir"));
  EXPECT_THAT(out, HasSubstr("func"));
}

/* Function structure */

TEST_F(PromelaGeneratorTest, FuncStructure_Body) {
  // A function becomes a Promela inline macro with params and a _ret out-param.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("func.ir"));
  EXPECT_THAT(out, HasSubstr("inline fn___func__myfunc(a, b, _ret)"));
  EXPECT_THAT(out, Not(HasSubstr("proctype")));
  EXPECT_THAT(out, Not(HasSubstr("init {")));
}

TEST_F(PromelaGeneratorTest, FuncStructure_ArithmeticOp) {
  // Add emits as a typed variable assignment `int v_<name> = lhs + rhs;`.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("func.ir"));
  std::string_view fn = InlineBody(out, "fn___func__myfunc");
  ASSERT_THAT(fn, Not(testing::IsEmpty()));
  EXPECT_THAT(fn, HasSubstr("int v_add_3 = a + b;"));
}

TEST_F(PromelaGeneratorTest, FuncStructure_ResultViaRetParam) {
  // The return value is assigned to _ret; ordering: computation before assign.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("func.ir"));
  std::string_view fn = InlineBody(out, "fn___func__myfunc");
  ASSERT_THAT(fn, Not(testing::IsEmpty()));
  EXPECT_THAT(fn, HasSubstr("_ret = v_add_3;"));
  EXPECT_LT(fn.find("v_add_3 = a + b"), fn.find("_ret = v_add_3"));
}

/* Proc structure */

TEST_F(PromelaGeneratorTest, ProcStructure_Body) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("proc.ir"));
  EXPECT_THAT(
      out,
      HasSubstr("proctype __proc__MyProc_0_next(chan _req_r; chan _resp_s)"));
  std::string_view pt = ProctypeBody(out, "__proc__MyProc_0_next");
  ASSERT_THAT(pt, Not(testing::IsEmpty()));
  EXPECT_THAT(pt, HasSubstr("short s___state = 0;"));
  EXPECT_THAT(pt, HasSubstr("do\n  ::"));
  EXPECT_THAT(pt, HasSubstr("od"));
}

TEST_F(PromelaGeneratorTest, ProcStructure_StateInitialValue) {
  // Patch proc.ir to change initial state value from 0 to 7.
  XLS_ASSERT_OK_AND_ASSIGN(std::string ir_text, LoadTestIr("proc.ir"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string out,
      Generate(absl::StrReplaceAll(ir_text, {{"init={0}", "init={7}"}})));
  std::string_view pt = ProctypeBody(out, "__proc__MyProc_0_next");
  ASSERT_THAT(pt, Not(testing::IsEmpty()));
  EXPECT_THAT(pt, HasSubstr("short s___state = 7;"));
}

TEST_F(PromelaGeneratorTest, ProcStructure_StateUpdateAfterComputation) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("proc.ir"));
  std::string_view pt = ProctypeBody(out, "__proc__MyProc_0_next");
  ASSERT_THAT(pt, Not(testing::IsEmpty()));
  EXPECT_THAT(pt, HasSubstr("s___state = v_add_15;"));
  // State update must appear after the add that computes the new value.
  EXPECT_LT(pt.find("v_add_15 = (s___state +"),
            pt.find("s___state = v_add_15"));
}

TEST_F(PromelaGeneratorTest, ProcStructure_TokenStateNotEmitted) {
  // The token state element must not produce a Promela variable or assignment.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("proc.ir"));
  std::string_view pt = ProctypeBody(out, "__proc__MyProc_0_next");
  ASSERT_THAT(pt, Not(testing::IsEmpty()));
  EXPECT_THAT(pt, Not(HasSubstr("token")));
  EXPECT_THAT(pt, Not(HasSubstr("s___tkn")));
}

TEST_F(PromelaGeneratorTest, ProcStructure_XrForRecv) {
  // TODO: also check xs for send channels not polled via non-blocking receive.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("proc.ir"));
  std::string_view pt = ProctypeBody(out, "__proc__MyProc_0_next");
  ASSERT_THAT(pt, Not(testing::IsEmpty()));
  EXPECT_THAT(pt, HasSubstr("xr _req_r;"));
  EXPECT_THAT(pt, Not(HasSubstr("xs")));
  EXPECT_THAT(pt, Not(HasSubstr("xr _resp_s")));
}

TEST_F(PromelaGeneratorTest, ProcStructure_ChannelDeclarations) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("proc.ir"));
  std::string_view init = InitBody(out);
  ASSERT_THAT(init, Not(testing::IsEmpty()));
  EXPECT_THAT(init, HasSubstr("chan _req_r = [8] of { byte };"));
  EXPECT_THAT(init, HasSubstr("chan _resp_s = [8] of { int };"));
}

// I have to check this
TEST_F(PromelaGeneratorTest, ProcNoBodyOps) {
  // parent owns a channel and spawns child but has no sends/receives/state:
  // EmitProc returns early after emitting the channel declaration and run stmt.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc child<_ch: bits[8] in>(__tkn: token, __s: bits[8], init={token, 0}) {
  chan_interface _ch(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[8] = state_read(state_element=__s, id=2)
  receive.3: (token, bits[8]) = receive(__tkn, channel=_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next_value.5: () = next_value(param=__tkn, value=tuple_index.4, id=5)
  next_value.6: () = next_value(param=__s, value=__s, id=6)
}
top proc parent<>(__tkn: token, init={token}) {
  chan _ch(bits[8], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
  chan_interface _ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  chan_interface _ch(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  proc_instantiation child_inst(_ch, proc=child)
  __tkn: token = state_read(state_element=__tkn, id=7)
  next_value.8: () = next_value(param=__tkn, value=__tkn, id=8)
})"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get()));
  // parent emits channel decl and run, but no do/od loop.
  EXPECT_THAT(out, HasSubstr("chan _ch = ["));
  EXPECT_THAT(out, HasSubstr("run child(_ch);"));
  // Parent's proctype closes right after the run statement (no loop).
  EXPECT_THAT(out, HasSubstr("run child(_ch);\n\n}"));
}

/* Channel operations */

TEST_F(PromelaGeneratorTest, ProcReceive) {
  // Receive emits `chan ? data_var` inside a predicated if/fi block.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out,
                           GenerateFromIrFile("blocking_receive.ir"));
  EXPECT_THAT(out, HasSubstr("_in_r ? v_receive_6_data"));
}

TEST_F(PromelaGeneratorTest, ProcSend) {
  // Send emits `chan ! payload` inside a predicated if/fi; payload is the state
  // var.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out,
                           GenerateFromIrFile("unconditional_send.ir"));
  EXPECT_THAT(out, HasSubstr("_out_s ! s___state"));
}

TEST_F(PromelaGeneratorTest, ProcPredicatedSend) {
  // Conditional send wraps the `!` in `if :: (pred) -> chan ! val :: else ->
  // skip fi`.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out,
                           GenerateFromIrFile("predicated_send.ir"));
  EXPECT_THAT(out, HasSubstr("(v_and_7 != 0) -> _out_s ! s___state"));
  EXPECT_THAT(out, HasSubstr(":: else -> skip;"));
}

TEST_F(PromelaGeneratorTest, ProcNonBlockingReceive) {
  // Non-blocking receive must use an atomic poll+receive pair (?[]/?) so that
  // no other process can consume the message between the check and the read
  // (guide section 4).
  XLS_ASSERT_OK_AND_ASSIGN(std::string out,
                           GenerateFromIrFile("non_blocking_receive.ir"));
  EXPECT_THAT(out, HasSubstr("atomic {"));
  EXPECT_THAT(out, HasSubstr("_in_r?[v_receive_7_data]"));
  EXPECT_THAT(out, HasSubstr("_in_r ? v_receive_7_data"));
  EXPECT_THAT(out, HasSubstr("v_receive_7_data_valid = 1"));
  EXPECT_THAT(out, HasSubstr(":: else -> v_receive_7_data_valid = 0"));
}

TEST_F(PromelaGeneratorTest, ProcNonBlockingReceivePredicated) {
  // Predicated non-blocking receive: predicate is ANDed with the poll inside
  // the same atomic block.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, Generate(R"(package p
proc nb_cond_reader<in: bits[32] in>(__tkn: token, __s: bits[1], init={token, 0}) {
  chan_interface in(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[1] = state_read(state_element=__s, id=2)
  literal.3: bits[1] = literal(value=1, id=3)
  receive.4: (token, bits[32], bits[1]) = receive(__tkn, predicate=literal.3, blocking=false, channel=in, id=4)
  recv_tok: token = tuple_index(receive.4, index=0, id=5)
  next_value.6: () = next_value(param=__tkn, value=recv_tok, id=6)
  next_value.7: () = next_value(param=__s, value=__s, id=7)
})"));
  EXPECT_THAT(out, HasSubstr("atomic {"));
  EXPECT_THAT(out, HasSubstr("(v_literal_3 != 0) && in?[v_receive_4_data]"));
  EXPECT_THAT(out, Not(HasSubstr("nempty")));
}

TEST_F(PromelaGeneratorTest, ProcPredicatedReceive) {
  // Conditional receive wraps the `?` in `if :: (pred) -> chan ? var :: else ->
  // skip fi`.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out,
                           GenerateFromIrFile("predicated_receive.ir"));
  EXPECT_THAT(out, HasSubstr("(v_and_8 != 0) -> _in_r ? v_receive_9_data"));
  EXPECT_THAT(out, HasSubstr(":: else -> skip;"));
}
/* Operations */

TEST_F(PromelaGeneratorTest, TraceMatch_Add) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a + b",
      /*test_cases=*/
      {{"u32:1", "u32:2"}, {"u32:10", "u32:20"}, {"u32:100", "u32:200"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a - b",
      /*test_cases=*/
      {{"u32:10", "u32:3"}, {"u32:5", "u32:5"}, {"u32:100", "u32:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Neg) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}}, /*return_type=*/"s32",
      /*fn_body=*/"-a",
      /*test_cases=*/{{"s32:5"}, {"s32:0"}, {"s32:-3"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_UMul) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a * b",
      /*test_cases=*/
      {{"u32:3", "u32:4"}, {"u32:100", "u32:200"}, {"u32:0", "u32:5"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_SMul) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"b", "s32"}},
      /*return_type=*/"s32", /*fn_body=*/"a * b",
      /*test_cases=*/{{"s32:3", "s32:-4"}, {"s32:-2", "s32:-5"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_UDiv) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a / b",
      /*test_cases=*/{{"u32:10", "u32:3"}, {"u32:100", "u32:4"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_SDiv) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"b", "s32"}},
      /*return_type=*/"s32", /*fn_body=*/"a / b",
      /*test_cases=*/{{"s32:-10", "s32:3"}, {"s32:10", "s32:-2"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_UMod) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a % b",
      /*test_cases=*/{{"u32:10", "u32:3"}, {"u32:7", "u32:4"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_SMod) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"b", "s32"}},
      /*return_type=*/"s32", /*fn_body=*/"a % b",
      /*test_cases=*/{{"s32:10", "s32:3"}, {"s32:-10", "s32:3"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_And) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a & b",
      /*test_cases=*/{{"u32:255", "u32:15"}, {"u32:170", "u32:85"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Or) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a | b",
      /*test_cases=*/{{"u32:240", "u32:15"}, {"u32:0", "u32:0"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Xor) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a ^ b",
      /*test_cases=*/{{"u32:255", "u32:15"}, {"u32:170", "u32:170"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Not) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u8"}}, /*return_type=*/"u8",
      /*fn_body=*/"!a",
      /*test_cases=*/{{"u8:0"}, {"u8:255"}, {"u8:170"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_AndReduce) {
  // and_reduce requires unsigned input. Use !a to compute all-ones at runtime
  // (avoiding a large literal that would overflow SPIN's signed int).
  // !u32:0 = all_ones -> and_reduce = 1; !u32:1 = ...1110 -> 0.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}}, /*return_type=*/"u1",
      /*fn_body=*/"and_reduce(!a)",
      /*test_cases=*/
      {{"u32:0"},              // !0 = all_ones -> 1
       {"u32:1"},              // !1 = ...1110  -> 0
       {"u32:2147483647"}}));  // !0x7FFFFFFF = 0x80000000 -> 0
}

TEST_F(PromelaGeneratorTest, TraceMatch_OrReduce) {
  // or_reduce requires unsigned input; use values that fit in SPIN's int range.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}}, /*return_type=*/"u1",
      /*fn_body=*/"or_reduce(a)",
      /*test_cases=*/{{"u32:0"}, {"u32:1"}, {"u32:1073741824"}}));  // bit 30
                                                                    // set, fits
                                                                    // in signed
                                                                    // int
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shll) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"n", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a << n",
      /*test_cases=*/{{"u32:1", "u32:4"}, {"u32:3", "u32:8"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shrl) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"n", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a >> n",
      /*test_cases=*/{{"u32:256", "u32:4"}, {"u32:768", "u32:8"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shra) {
  // Arithmetic right shift on signed type: sign bit is replicated.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"n", "u32"}},
      /*return_type=*/"s32", /*fn_body=*/"a >> n",
      /*test_cases=*/{{"s32:-256", "u32:4"}, {"s32:256", "u32:4"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Eq) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u1", /*fn_body=*/"a == b",
      /*test_cases=*/{{"u32:5", "u32:5"}, {"u32:5", "u32:6"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Ne) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u1", /*fn_body=*/"a != b",
      /*test_cases=*/{{"u32:5", "u32:6"}, {"u32:5", "u32:5"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_ULt) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u1", /*fn_body=*/"a < b",
      /*test_cases=*/
      {{"u32:3", "u32:5"}, {"u32:5", "u32:5"}, {"u32:5", "u32:3"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_ULe) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u1", /*fn_body=*/"a <= b",
      /*test_cases=*/
      {{"u32:3", "u32:5"}, {"u32:5", "u32:5"}, {"u32:6", "u32:5"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_UGt) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u1", /*fn_body=*/"a > b",
      /*test_cases=*/
      {{"u32:5", "u32:3"}, {"u32:5", "u32:5"}, {"u32:3", "u32:5"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_UGe) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u1", /*fn_body=*/"a >= b",
      /*test_cases=*/
      {{"u32:5", "u32:3"}, {"u32:5", "u32:5"}, {"u32:3", "u32:5"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_SLt) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"b", "s32"}},
      /*return_type=*/"u1", /*fn_body=*/"a < b",
      /*test_cases=*/
      {{"s32:-1", "s32:0"}, {"s32:0", "s32:0"}, {"s32:1", "s32:-1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_SLe) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"b", "s32"}},
      /*return_type=*/"u1", /*fn_body=*/"a <= b",
      /*test_cases=*/
      {{"s32:-1", "s32:0"}, {"s32:0", "s32:0"}, {"s32:1", "s32:-1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_SGt) {
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "s32"}, {"b", "s32"}},
                                      /*return_type=*/"u1", /*fn_body=*/"a > b",
                                      /*test_cases=*/
                                      {{"s32:0", "s32:-1"},     // 0 > -1 -> 1
                                       {"s32:0", "s32:0"},      // 0 > 0  -> 0
                                       {"s32:-1", "s32:0"}}));  // -1 > 0 -> 0
}

TEST_F(PromelaGeneratorTest, TraceMatch_SGe) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s32"}, {"b", "s32"}},
      /*return_type=*/"u1", /*fn_body=*/"a >= b",
      /*test_cases=*/
      {{"s32:0", "s32:-1"}, {"s32:-1", "s32:0"}, {"s32:-1", "s32:-1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_BitSlice) {
  // a[4:8] extracts bits 4..7 (a 4-bit nibble from position 4).
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u16"}}, /*return_type=*/"u4",
      /*fn_body=*/"a[4:8]",
      /*test_cases=*/
      {{"u16:240"},      // 0x00F0 -> nibble[4:8] = 0xF = 15
       {"u16:15"},       // 0x000F -> nibble[4:8] = 0x0 = 0
       {"u16:4080"}}));  // 0x0FF0 -> nibble[4:8] = 0xF = 15
}

TEST_F(PromelaGeneratorTest, TraceMatch_DynamicBitSlice) {
  // a[start +: u4] extracts 4 bits starting at runtime index `start`.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u16"}, {"start", "u32"}},
      /*return_type=*/"u4", /*fn_body=*/"a[start +: u4]",
      /*test_cases=*/
      {{"u16:240", "u32:4"},    // 0x00F0, bits[4:8] = 0xF = 15
       {"u16:240", "u32:0"},    // 0x00F0, bits[0:4] = 0x0 = 0
       {"u16:15", "u32:0"}}));  // 0x000F, bits[0:4] = 0xF = 15
}

TEST_F(PromelaGeneratorTest, TraceMatch_SignExtend) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "s8"}}, /*return_type=*/"s32",
      /*fn_body=*/"a as s32",
      /*test_cases=*/
      {{"s8:127"},      // positive: no sign bit -> 127
       {"s8:-1"},       // 0xFF -> sign-extended to 0xFFFFFFFF = -1
       {"s8:-128"}}));  // 0x80 -> sign-extended to 0xFFFFFF80 = -128
}

TEST_F(PromelaGeneratorTest, TraceMatch_ZeroExtend) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u8"}}, /*return_type=*/"u32",
      /*fn_body=*/"a as u32",
      /*test_cases=*/{{"u8:0"}, {"u8:127"}, {"u8:255"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Concat) {
  // a ++ b: a in high byte, b in low byte of u16, then zero-extended to u32.
  // Single test case avoids the multi-case if/else chains that the optimizer
  // converts to sign-extend patterns, which would cause SPIN truncation
  // warnings when the resulting int is assigned to a byte variable.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u8"}, {"b", "u8"}},
      /*return_type=*/"u32", /*fn_body=*/"(a ++ b) as u32",
      /*test_cases=*/{{"u8:1", "u8:2"}}));  // 0x0102 = 258
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sel) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"s", "u1"}, {"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"if s == u1:0 { a } else { b }",
      /*test_cases=*/
      {{"u1:0", "u32:10", "u32:20"}, {"u1:1", "u32:10", "u32:20"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_OneHotSel) {
  // one_hot_sel: bit i of selector picks cases[i]; multiple bits OR the cases.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"s", "u2"}, {"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"one_hot_sel(s, [a, b])",
      /*test_cases=*/
      {{"u2:1", "u32:10", "u32:20"},     // bit 0 -> a = 10
       {"u2:2", "u32:10", "u32:20"}}));  // bit 1 -> b = 20
}

TEST_F(PromelaGeneratorTest, TraceMatch_Gate) {
  // gate!(cond, value): passes value through when cond is true, else 0.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"c", "u1"}, {"a", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"gate!(c as bool, a)",
      /*test_cases=*/{{"u1:1", "u32:42"}, {"u1:0", "u32:42"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Invoke) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}}, /*return_type=*/"u32",
      /*fn_body=*/"helper(a)",
      /*test_cases=*/{{"u32:5"}, {"u32:21"}, {"u32:0"}},
      /*fn_extras=*/"fn helper(x: u32) -> u32 { x + x }"));
}

TEST_F(PromelaGeneratorTest, TraceMatch_AddMul_U32) {
  // a + b * c: two operations chained, u32 (no overflow).
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}, {"c", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"a + b * c",
      /*test_cases=*/
      {{"u32:10", "u32:3", "u32:7"},       // 10 + 3*7   = 31
       {"u32:100", "u32:20", "u32:5"}}));  // 100 + 20*5 = 200
}

TEST_F(PromelaGeneratorTest, TraceMatch_AddMul_U8_Overflow) {
  // b*c = 10*30 = 300 -> u8:44;  a + 44 = 244+44 = 288 -> u8:32.
  // Both the intermediate multiply and the final add overflow u8.
  // Single case to avoid optimizer sign-extend patterns on sub-byte types.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u8"}, {"b", "u8"}, {"c", "u8"}},
      /*return_type=*/"u8", /*fn_body=*/"a + b * c",
      /*test_cases=*/{{"u8:244", "u8:10", "u8:30"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_CompareAfterOverflow_U8) {
  // a + b overflows u8: 200+100 = 300 -> u8:44.
  // The comparison then sees the wrapped value: u8:44 > u8:100 -> false ->
  // u8:0. Without intermediate masking the Promela would compare 300 > 100 ->
  // true -> u8:1 (wrong).
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u8"}, {"b", "u8"}, {"t", "u8"}},
      /*return_type=*/"u8", /*fn_body=*/"if a + b > t { u8:1 } else { u8:0 }",
      /*test_cases=*/{{"u8:200", "u8:100", "u8:100"}}));
}

// Mixed-width operands
// Shifts: result width = value width (amount width is independent).
// Concat: result width = sum of part widths.

TEST_F(PromelaGeneratorTest, TraceMatch_Shll_WideValue_NarrowAmount) {
  // u8 value, u4 shift amount -> result u8.  200 << 2 = 800, wraps to 32.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u8"}, {"n", "u4"}},
      /*return_type=*/"u8", /*fn_body=*/"a << n",
      /*test_cases=*/{{"u8:200", "u4:2"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shll_NarrowValue_WideAmount) {
  // u4 value, u8 shift amount -> result u4.  3 << 3 = 24, wraps to 8.
  // (Mirrors TraceMatch_Shll_Overflow_U4 but with a u8 shift amount.)
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u4"}, {"n", "u8"}},
      /*return_type=*/"u4", /*fn_body=*/"a << n",
      /*test_cases=*/{{"u4:3", "u8:3"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shrl_WideValue_NarrowAmount) {
  // u12 value, u4 shift amount -> result u12.  4095 >> 4 = 255.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u12"}, {"n", "u4"}},
      /*return_type=*/"u12", /*fn_body=*/"a >> n",
      /*test_cases=*/{{"u12:4095", "u4:4"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shll_WideValue_NarrowAmount_Overflow) {
  // u12 value, u4 shift amount -> result u12.  2048 << 1 = 4096, wraps to 0.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u12"}, {"n", "u4"}},
      /*return_type=*/"u12", /*fn_body=*/"a << n",
      /*test_cases=*/{{"u12:2048", "u4:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Concat_DifferentWidth) {
  // u4 ++ u8 -> u12.  0xA ++ 0xBC = 0xABC = 2748.
  // Result width = 4 + 8 = 12; value fits in short channel (2748 < 32768).
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u4"}, {"b", "u8"}},
      /*return_type=*/"u12", /*fn_body=*/"a ++ b",
      /*test_cases=*/{{"u4:10", "u8:188"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Concat_ThreeParts) {
  // u4 ++ u4 ++ u8 -> u16.  0x1 ++ 0x2 ++ 0x34 = 0x1234 = 4660.
  // Operand widths 4, 4, 8 all differ from result width 16.
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u4"}, {"b", "u4"}, {"c", "u8"}},
      /*return_type=*/"u16", /*fn_body=*/"a ++ b ++ c",
      /*test_cases=*/{{"u4:1", "u4:2", "u8:52"}}));
}

// Type-width-aware wrapping  (sub-int arithmetic must wrap at declared width)
// Each test uses a single input tuple so the optimizer emits constant
// assignments instead of if/else chains that would trigger sign-extend
// patterns.

// u4 (4-bit -> byte channel)
TEST_F(PromelaGeneratorTest, TraceMatch_Add_Overflow_U4) {
  // 15 + 1 = 16, wraps to 0 in u4
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u4"}, {"b", "u4"}},
                                      /*return_type=*/"u4", /*fn_body=*/"a + b",
                                      /*test_cases=*/{{"u4:15", "u4:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub_Underflow_U4) {
  // 0 - 1 = -1, wraps to 15 in u4
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u4"}, {"b", "u4"}},
                                      /*return_type=*/"u4", /*fn_body=*/"a - b",
                                      /*test_cases=*/{{"u4:0", "u4:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Mul_Overflow_U4) {
  // 5 * 4 = 20, wraps to 4 in u4 (20 & 0xF = 4)
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u4"}, {"b", "u4"}},
                                      /*return_type=*/"u4", /*fn_body=*/"a * b",
                                      /*test_cases=*/{{"u4:5", "u4:4"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Shll_Overflow_U4) {
  // 3 << 3 = 24, wraps to 8 in u4 (24 & 0xF = 8)
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u4"}, {"n", "u32"}},
      /*return_type=*/"u4", /*fn_body=*/"a << n",
      /*test_cases=*/{{"u4:3", "u32:3"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Not_U4) {
  // bitwise NOT: ~0101 = 1010, i.e. !u4:5 = u4:10
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u4"}}, /*return_type=*/"u4",
      /*fn_body=*/"!a", /*test_cases=*/{{"u4:5"}}));
}

// u6 (6-bit, non-power-of-2 -> byte channel)
TEST_F(PromelaGeneratorTest, TraceMatch_Add_Overflow_U6) {
  // 63 + 1 = 64, wraps to 0 in u6 (mask = 0x3F)
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u6"}, {"b", "u6"}},
                                      /*return_type=*/"u6", /*fn_body=*/"a + b",
                                      /*test_cases=*/{{"u6:63", "u6:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub_Underflow_U6) {
  // 0 - 1 = -1, wraps to 63 in u6
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u6"}, {"b", "u6"}},
                                      /*return_type=*/"u6", /*fn_body=*/"a - b",
                                      /*test_cases=*/{{"u6:0", "u6:1"}}));
}

// u8 (8-bit -> byte channel)
TEST_F(PromelaGeneratorTest, TraceMatch_Add_Overflow_U8) {
  // 255 + 1 = 256, wraps to 0 in u8
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u8"}, {"b", "u8"}},
                                      /*return_type=*/"u8", /*fn_body=*/"a + b",
                                      /*test_cases=*/{{"u8:255", "u8:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub_Underflow_U8) {
  // 0 - 1 = -1, wraps to 255 in u8
  EXPECT_TRUE(DslxFunctionTracesMatch(/*fn_name=*/"op",
                                      /*params=*/{{"a", "u8"}, {"b", "u8"}},
                                      /*return_type=*/"u8", /*fn_body=*/"a - b",
                                      /*test_cases=*/{{"u8:0", "u8:1"}}));
}

// u12 (12-bit, non-power-of-2 -> short channel; max value 4095 fits in short)
TEST_F(PromelaGeneratorTest, TraceMatch_Add_Overflow_U12) {
  // 4095 + 1 = 4096, wraps to 0 in u12 (mask = 0xFFF)
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u12"}, {"b", "u12"}},
      /*return_type=*/"u12", /*fn_body=*/"a + b",
      /*test_cases=*/{{"u12:4095", "u12:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub_Underflow_U12) {
  // 0 - 1 = -1, wraps to 4095 in u12
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u12"}, {"b", "u12"}},
      /*return_type=*/"u12", /*fn_body=*/"a - b",
      /*test_cases=*/{{"u12:0", "u12:1"}}));
}

// u20 (20-bit, non-power-of-2 -> int channel)
TEST_F(PromelaGeneratorTest, TraceMatch_Add_Overflow_U20) {
  // 1048575 + 1 = 1048576, wraps to 0 in u20 (mask = 0xFFFFF)
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u20"}, {"b", "u20"}},
      /*return_type=*/"u20", /*fn_body=*/"a + b",
      /*test_cases=*/{{"u20:1048575", "u20:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub_Underflow_U20) {
  // 0 - 1 = -1, wraps to 1048575 in u20
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u20"}, {"b", "u20"}},
      /*return_type=*/"u20", /*fn_body=*/"a - b",
      /*test_cases=*/{{"u20:0", "u20:1"}}));
}

// u31 (31-bit -> int channel; max value is 2^31-1 = INT_MAX)
TEST_F(PromelaGeneratorTest, TraceMatch_Add_Overflow_U31) {
  // 2147483647 + 1 overflows signed int to -2147483648; (-2147483648) &
  // 0x7FFFFFFF = 0
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u31"}, {"b", "u31"}},
      /*return_type=*/"u31", /*fn_body=*/"a + b",
      /*test_cases=*/{{"u31:2147483647", "u31:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Sub_Underflow_U31) {
  // 0 - 1 = -1, wraps to 2147483647 in u31
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u31"}, {"b", "u31"}},
      /*return_type=*/"u31", /*fn_body=*/"a - b",
      /*test_cases=*/{{"u31:0", "u31:1"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Nand) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"!(a & b)",
      /*test_cases=*/
      {{"u32:0", "u32:0"}, {"u32:5", "u32:3"}, {"u32:0xff", "u32:0xff"}}));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Nor) {
  EXPECT_TRUE(DslxFunctionTracesMatch(
      /*fn_name=*/"op", /*params=*/{{"a", "u32"}, {"b", "u32"}},
      /*return_type=*/"u32", /*fn_body=*/"!(a | b)",
      /*test_cases=*/
      {{"u32:0", "u32:0"}, {"u32:5", "u32:3"}, {"u32:0xff", "u32:0x00"}}));
}

TEST_F(PromelaGeneratorTest, PrioritySel_DirectIR) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, Generate(R"(package p
fn f(sel: bits[2], a: bits[32], b: bits[32], d: bits[32]) -> bits[32] {
  ret priority_sel.1: bits[32] = priority_sel(sel, cases=[a, b], default=d, id=1)
})"));
  // Default value is assigned first; each case overwrites if its bit is set.
  EXPECT_THAT(out, HasSubstr("v_priority_sel_1 = d"));
}

// No TraceMatch_PrioritySel: priority_sel lowers to array_index which is not
// yet supported by promela_converter_main.

/* Side-effect operations */

TEST_F(PromelaGeneratorTest, MinDelay_DirectIR) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, Generate(R"(package p
fn f() -> token {
  after_all.1: token = after_all(id=1)
  ret min_delay.2: token = min_delay(after_all.1, delay=3, id=2)
})"));
  // min_delay is token-only: no variable emitted, function still generates.
  EXPECT_THAT(out, HasSubstr("fn_f"));
}

TEST_F(PromelaGeneratorTest, Cover_DirectIR) {
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, Generate(R"(package p
fn f(cond: bits[1]) -> () {
  ret cover.1: () = cover(cond, label="my_label", id=1)
})"));
  // cover is unit-typed: no variable emitted, function still generates.
  EXPECT_THAT(out, HasSubstr("fn_f"));
}

TEST_F(PromelaGeneratorTest, ProcAssert) {
  // assert() emits a Promela assert() guarded by a conditional printf that
  // prints the label only when the condition is false.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("assertion.ir"));
  EXPECT_THAT(
      out, HasSubstr(R"(:: (v_or_9 == 0) -> printf("XLS_ASSERT:nonzero\n");)"));
  EXPECT_THAT(out, HasSubstr("assert(v_or_9 != 0)"));
}

/* Unimplemented operations */

TEST_F(PromelaGeneratorTest, Unimplemented_XorReduce) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[8]) -> bits[1] {
  ret xor_reduce.1: bits[1] = xor_reduce(a, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: xor_reduce")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_Reverse) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32]) -> bits[32] {
  ret reverse.1: bits[32] = reverse(a, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: reverse")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_OneHot) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[4]) -> bits[5] {
  ret one_hot.1: bits[5] = one_hot(a, lsb_prio=true, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: one_hot")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_Array) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32], b: bits[32]) -> bits[32][2] {
  ret array.1: bits[32][2] = array(a, b, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: array")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_ArrayConcat) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32][2], b: bits[32][2]) -> bits[32][4] {
  ret array_concat.1: bits[32][4] = array_concat(a, b, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: array_concat")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_ArrayIndex) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32][4], idx: bits[2]) -> bits[32] {
  ret array_index.1: bits[32] = array_index(a, indices=[idx], id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: array_index")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_ArraySlice) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32][4], idx: bits[2]) -> bits[32][2] {
  ret array_slice.1: bits[32][2] = array_slice(a, idx, width=2, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: array_slice")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_ArrayUpdate) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32][4], val: bits[32], idx: bits[2]) -> bits[32][4] {
  ret array_update.1: bits[32][4] = array_update(a, val, indices=[idx], id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: array_update")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_Map) {
  EXPECT_THAT(Generate(R"(package p
fn double(x: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, x, id=1)
}
fn f(a: bits[32][3]) -> bits[32][3] {
  ret map.2: bits[32][3] = map(a, to_apply=double, id=2)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: map")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_CountedFor) {
  EXPECT_THAT(Generate(R"(package p
fn body(i: bits[32], acc: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(i, acc, id=1)
}
fn f(x: bits[32]) -> bits[32] {
  ret counted_for.2: bits[32] = counted_for(x, trip_count=4, stride=1, body=body, id=2)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: counted_for")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_DynamicCountedFor) {
  EXPECT_THAT(Generate(R"(package p
fn body(i: bits[32], acc: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(i, acc, id=1)
}
fn f(x: bits[32], tc: bits[16], st: bits[16]) -> bits[32] {
  ret dynamic_counted_for.2: bits[32] = dynamic_counted_for(x, tc, st, body=body, id=2)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: dynamic_counted_for")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_BitSliceUpdate) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32], start: bits[5], val: bits[8]) -> bits[32] {
  ret bit_slice_update.1: bits[32] = bit_slice_update(a, start, val, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: bit_slice_update")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_Encode) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[16]) -> bits[4] {
  ret encode.1: bits[4] = encode(a, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: encode")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_Decode) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[3]) -> bits[8] {
  ret decode.1: bits[8] = decode(a, width=8, id=1)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: decode")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_SMulp) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32], b: bits[32]) -> bits[32] {
  smulp.1: (bits[32], bits[32]) = smulp(a, b, id=1)
  ret add.2: bits[32] = add(a, b, id=2)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: smulp")));
}

TEST_F(PromelaGeneratorTest, Unimplemented_UMulp) {
  EXPECT_THAT(Generate(R"(package p
fn f(a: bits[32], b: bits[32]) -> bits[32] {
  umulp.1: (bits[32], bits[32]) = umulp(a, b, id=1)
  ret add.2: bits[32] = add(a, b, id=2)
})"),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("unsupported op: umulp")));
}

/* Error cases */

TEST_F(PromelaGeneratorTest, ErrorOnOldStyleGlobalChannels) {
  // A package with procs that use package-level channels (not proc-scoped)
  // must be rejected with a clear INVALID_ARGUMENT error.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package old_style
chan step_in(bits[32], id=0, kind=streaming, ops=receive_only,
             flow_control=ready_valid, strictness=proven_mutually_exclusive)
chan count_out(bits[32], id=1, kind=streaming, ops=send_only,
               flow_control=ready_valid, strictness=proven_mutually_exclusive)
top proc counter(__count: bits[32], init={0}) {
  after_all.2: token = after_all(id=2)
  receive.3: (token, bits[32]) = receive(after_all.2, channel=step_in, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  step: bits[32] = tuple_index(receive.3, index=1, id=5)
  __count_r: bits[32] = state_read(state_element=__count, id=6)
  new_count: bits[32] = add(__count_r, step, id=7)
  send.8: token = send(tuple_index.4, new_count, channel=count_out, id=8)
  next_value.9: () = next_value(param=__count, value=new_count, id=9)
})"));
  EXPECT_THAT(GenerateAndDump(pkg.get()),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("proc-scoped channels")));
}

TEST_F(PromelaGeneratorTest, ErrorOnBitWidthOver32) {
  EXPECT_THAT(
      Generate(R"(package wide
fn f(x: bits[64]) -> bits[64] {
  ret identity.1: bits[64] = identity(x, id=1)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("bit width 64")));
}

TEST_F(PromelaGeneratorTest, FunctionOnlyPackageNotRejected) {
  // A package with only functions and no procs has no channels at all;
  // it must be accepted even though ChannelsAreProcScoped() returns false.
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, Generate(R"(package fns_only
fn add(a: bits[32], b: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(a, b, id=1)
})"));
  EXPECT_THAT(out, HasSubstr("inline fn_add"));
  EXPECT_THAT(out, Not(HasSubstr("init {")));
}

TEST_F(PromelaGeneratorTest, ErrorOnBitWidthOver32_ProcNode) {
  EXPECT_THAT(
      Generate(R"(package p
top proc my_proc<>(__tkn: token, init={token}) {
  __tkn: token = state_read(state_element=__tkn, id=1)
  lit: bits[64] = literal(value=0, id=2)
  next_value.3: () = next_value(param=__tkn, value=__tkn, id=3)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("bit width 64")));
}

TEST_F(PromelaGeneratorTest, ErrorOnBitWidthOver32_ProcStateElement) {
  // State element bits[64]: the corresponding Param node fires the check.
  EXPECT_THAT(
      Generate(R"(package p
top proc my_proc<>(__tkn: token, __state: bits[64], init={token, 0}) {
  __tkn: token = state_read(state_element=__tkn, id=1)
  next_value.2: () = next_value(param=__tkn, value=__tkn, id=2)
})"),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("bit width 64")));
}

TEST_F(PromelaGeneratorTest, ErrorOnBitWidthOver32_ProcInterfaceChannel) {
  // Interface channel bits[64] with no bits[64] nodes (no receive in body).
  EXPECT_THAT(Generate(R"(package p
top proc my_proc<_big: bits[64] in>(__tkn: token, init={token}) {
  chan_interface _big(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  next_value.2: () = next_value(param=__tkn, value=__tkn, id=2)
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("channel '_big'")));
}

TEST_F(PromelaGeneratorTest, ErrorOnBitWidthOver32_ProcOwnedChannel) {
  // Owned channel bits[64]: chan_interface entries carry the same type,
  // so the interface check fires and rejects the package.
  EXPECT_THAT(Generate(R"(package p
top proc my_proc<>(__tkn: token, init={token}) {
  chan _big(bits[64], id=0, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive)
  chan_interface _big(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  chan_interface _big(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=none, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  next_value.2: () = next_value(param=__tkn, value=__tkn, id=2)
})"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("channel '_big'")));
}

/* Generator options */

TEST_F(PromelaGeneratorTest, SourceLocationsDisabledByDefault) {
  // Without the option, no /* file:line:col */ comments should appear.
  // (The package header comment "/* Promela model ... */" is always present;
  // per-node location comments use the pattern "/* file:line:col */".)
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateFromIrFile("proc.ir"));
  EXPECT_THAT(out, Not(HasSubstr("/* file:")));
}

TEST_F(PromelaGeneratorTest, SourceLocationsEmitted) {
  // IR with an explicit source location on the add node.  The location is
  // encoded as (fileno, lineno, colno); fileno 1 maps to "counter.x".
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
file_number 1 "counter.x"
fn add(a: bits[32], b: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(a, b, id=1, pos=[(1,10,5)])
})"));
  PromelaGeneratorOptions opts;
  opts.emit_source_locations = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("/* counter.x:10:5 */"));
  EXPECT_THAT(out, HasSubstr("v_add_1 = a + b"));
}

TEST_F(PromelaGeneratorTest, SourceLocationsNodeWithoutLoc) {
  // Nodes with no recorded location must not produce a comment even when the
  // option is enabled.  (The package header is allowed; per-node location
  // comments use the pattern "/* file:line:col */".)
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
fn add(a: bits[32], b: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(a, b, id=1)
})"));
  PromelaGeneratorOptions opts;
  opts.emit_source_locations = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, Not(HasSubstr("/* file:")));
  EXPECT_THAT(out, HasSubstr("v_add_1 = a + b"));
}

TEST_F(PromelaGeneratorTest, OptionChannelDepth) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc sender<out_ch: bits[32] out>(__tkn: token, __s: bits[32], init={token, 0}) {
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[32] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=out_ch, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.channel_depth = 4;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("[4] of {"));
  EXPECT_THAT(out, Not(HasSubstr("[8] of {")));
}

TEST_F(PromelaGeneratorTest, OptionEmitSourceHints) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
fn add(a: bits[32], b: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(a, b, id=1)
})"));
  PromelaGeneratorOptions opts;
  opts.emit_source_hints = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("/* ir:"));
  EXPECT_THAT(out, HasSubstr("v_add_1 = a + b"));
}

TEST_F(PromelaGeneratorTest, OptionEmitTerminationHook) {
  // emit_termination_hook adds a __terminated guard to each proc loop and
  // a global flag that init sets after the terminator channel fires.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc sender<out_ch: bits[32] out>(__tkn: token, __s: bits[32], init={token, 0}) {
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[32] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=out_ch, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.emit_termination_hook = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("bit __terminated = 0;"));
  EXPECT_THAT(out, HasSubstr("(__terminated) -> break"));
}

TEST_F(PromelaGeneratorTest, OptionAssertSendOnFullChannel) {
  // assert_send_on_full_channel turns a blocked-on-full-channel situation into
  // an explicit SPIN assertion violation rather than a deadlock.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc sender<out_ch: bits[32] out>(__tkn: token, __s: bits[32], init={token, 0}) {
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[32] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=out_ch, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.assert_send_on_full_channel = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("assert(len(out_ch) < 8)"));
}

TEST_F(PromelaGeneratorTest, OptionWorstCaseThroughput) {
  // worst_case_throughput adds an idle-stall counter __thr so that SPIN
  // explores the case where the proc does real work at most once every N
  // iterations.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc sender<out_ch: bits[32] out>(__tkn: token, __s: bits[32], init={token, 0}) {
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[32] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=out_ch, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.worst_case_throughput["sender"] = 3;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("byte __thr = 0;"));
  EXPECT_THAT(out, HasSubstr(":: (__thr > 0) -> __thr--;"));
  EXPECT_THAT(out, HasSubstr(":: (__thr == 0) ->"));
  EXPECT_THAT(out, HasSubstr("__thr = 2;"));  // throughput - 1
}

TEST_F(PromelaGeneratorTest, OptionEmitProgressLabels) {
  // emit_progress_labels prefixes each channel op with a SPIN progress label
  // for livelock detection via spin -search -DNP.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc reader<in_ch: bits[32] in>(__tkn: token, __s: bits[1], init={token, 0}) {
  chan_interface in_ch(direction=receive, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[1] = state_read(state_element=__s, id=2)
  receive.3: (token, bits[32]) = receive(__tkn, channel=in_ch, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  next_value.5: () = next_value(param=__tkn, value=tuple_index.4, id=5)
  next_value.6: () = next_value(param=__s, value=__s, id=6)
})"));
  PromelaGeneratorOptions opts;
  opts.emit_progress_labels = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("progress_recv_in_ch: in_ch ?"));
}

TEST_F(PromelaGeneratorTest, OptionEmitProgressLabels_Send) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc writer<out_ch: bits[32] out>(__tkn: token, __s: bits[32], init={token, 0}) {
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[32] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=out_ch, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.emit_progress_labels = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("progress_send_out_ch: out_ch !"));
}

TEST_F(PromelaGeneratorTest, OptionWorstCaseThroughputLarge) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
proc sender<out_ch: bits[32] out>(__tkn: token, __s: bits[32], init={token, 0}) {
  chan_interface out_ch(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[32] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=out_ch, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.worst_case_throughput["sender"] = 257;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("short __thr = 0;"));
}

TEST_F(PromelaGeneratorTest, EmitInitTerminatorChannel) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
top proc my_proc<_terminator: bits[1] out>(__tkn: token, __s: bits[1], init={token, 0}) {
  chan_interface _terminator(direction=send, kind=streaming, strictness=proven_mutually_exclusive, flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  __s: bits[1] = state_read(state_element=__s, id=2)
  send.3: token = send(__tkn, __s, channel=_terminator, id=3)
  next_value.4: () = next_value(param=__tkn, value=send.3, id=4)
  next_value.5: () = next_value(param=__s, value=__s, id=5)
})"));
  PromelaGeneratorOptions opts;
  opts.emit_termination_hook = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("bit __term_val;"));
  EXPECT_THAT(out, HasSubstr("_terminator ? __term_val;"));
  EXPECT_THAT(out, HasSubstr("__terminated = 1;"));
}

TEST_F(PromelaGeneratorTest, SourceLocationsUnregisteredFile) {
  // fileno 99 is not registered; falls back to "file:99:line:col" format.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> pkg,
                           IrTestBase::ParsePackageNoVerify(R"(package p
fn add(a: bits[32], b: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(a, b, id=1, pos=[(99,10,5)])
})"));
  PromelaGeneratorOptions opts;
  opts.emit_source_locations = true;
  XLS_ASSERT_OK_AND_ASSIGN(std::string out, GenerateAndDump(pkg.get(), opts));
  EXPECT_THAT(out, HasSubstr("/* file:99:10:5 */"));
}

/* Spin syntax validation */

TEST_F(PromelaGeneratorTest, SpinSyntax_Proc) {
  EXPECT_TRUE(GenerateAndSpinCheckFromFile("xls/spin/testdata/proc.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_ProcReceive) {
  EXPECT_TRUE(
      GenerateAndSpinCheckFromFile("xls/spin/testdata/blocking_receive.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_ProcSend) {
  EXPECT_TRUE(
      GenerateAndSpinCheckFromFile("xls/spin/testdata/unconditional_send.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_PredicatedSend) {
  EXPECT_TRUE(
      GenerateAndSpinCheckFromFile("xls/spin/testdata/predicated_send.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_PredicatedReceive) {
  EXPECT_TRUE(
      GenerateAndSpinCheckFromFile("xls/spin/testdata/predicated_receive.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_ProcAssert) {
  EXPECT_TRUE(GenerateAndSpinCheckFromFile("xls/spin/testdata/assertion.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_MultiProc) {
  EXPECT_TRUE(GenerateAndSpinCheckFromFile("xls/spin/testdata/multi_proc.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_NonBlockingReceive) {
  EXPECT_TRUE(GenerateAndSpinCheckFromFile(
      "xls/spin/testdata/non_blocking_receive.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_XrHints) {
  // TODO: add xs coverage once the generator emits xs for safe send channels.
  EXPECT_TRUE(GenerateAndSpinCheckFromFile("xls/spin/testdata/xr_hints.ir"));
}

TEST_F(PromelaGeneratorTest, SpinSyntax_Deadlock) {
  // deadlock.x has a circular send/receive dependency between procs A and B.
  // The generated Promela must be syntactically valid even though exhaustive
  // search (spin -search) will find a deadlock at run time.
  EXPECT_TRUE(GenerateAndSpinCheckFromFile("xls/spin/testdata/deadlock.ir"));
}

/* End-to-end proc tests */

TEST_F(PromelaGeneratorTest, TraceMatch_Passthrough) {
  EXPECT_TRUE(DslxPromelaTracesMatchFile("xls/spin/testdata/passthrough.x",
                                         "PassthroughTest",
                                         /*emit_termination_hook=*/true));
}

TEST_F(PromelaGeneratorTest, TraceMatch_Counter) {
  EXPECT_TRUE(DslxPromelaTracesMatchFile("xls/spin/testdata/counter.x",
                                         "CounterTest",
                                         /*emit_termination_hook=*/true));
}

TEST_F(PromelaGeneratorTest, TraceMatch_ConditionalIo) {
  EXPECT_TRUE(DslxPromelaTracesMatchFile("xls/spin/testdata/conditional_io.x",
                                         "ConditionalIoTest",
                                         /*emit_termination_hook=*/true));
}

TEST_F(PromelaGeneratorTest, TraceMatch_ChannelNaming) {
  EXPECT_TRUE(DslxPromelaTracesMatchFile("xls/spin/testdata/channel_naming.x",
                                         "MultiplyBy4Test",
                                         /*emit_termination_hook=*/true));
}

}  // namespace
}  // namespace xls::spin
