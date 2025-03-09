// Copyright 2025 The XLS Authors
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

#include <ctime>

#include "absl/flags/flag.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/common/exit_status.h"
#include "xls/common/init_xls.h"
#include "xls/common/logging/log_lines.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/parse_and_typecheck.h"
#include "xls/fuzzer/ast_generator.h"

ABSL_FLAG(bool, emit_signed_types, true, "Emit signed types.");

ABSL_FLAG(int64_t, max_width_bits_types, 64,
          "Maximum width of generated bits types.");

ABSL_FLAG(int64_t, max_width_aggregate_types, 1024,
          "Maximum width of generated aggregate types (tuples, arrays, ..).");

ABSL_FLAG(bool, emit_loops, true, "Emit loops.");

ABSL_FLAG(bool, emit_gate, true, "Emit gates.");

ABSL_FLAG(bool, generate_proc, false, "Generate a proc.");

ABSL_FLAG(bool, emit_stateless_proc, false, "Emit only stateless procs.");

ABSL_FLAG(bool, emit_zero_width_bits_types, false,
          "Emit zero-wdith bits types.");

ABSL_FLAG(std::optional<int64_t>, seed, std::nullopt,
          "Seed value for generation. By default, a nondetermistic seed is "
          "used; if a seed is provided, it is used for determinism");

ABSL_FLAG(std::string, top_entity_name, "test_top",
          "Name of the generated top entity.");

ABSL_FLAG(std::string, module_name, "test_mod",
          "Name of the generated module.");

namespace xls {
namespace {

absl::Status RealMain() {
  std::optional<uint64_t> seed_flag = absl::GetFlag(FLAGS_seed);
  uint64_t rng_seed = seed_flag.has_value()
                          ? *seed_flag
                          : absl::Uniform<uint64_t>(absl::BitGen());
  std::mt19937_64 rng(rng_seed);

  xls::dslx::FileTable ft;
  xls::dslx::AstGeneratorOptions opts = {
      .emit_signed_types = absl::GetFlag(FLAGS_emit_signed_types),
      .max_width_bits_types = absl::GetFlag(FLAGS_max_width_bits_types),
      .max_width_aggregate_types =
          absl::GetFlag(FLAGS_max_width_aggregate_types),
      .emit_loops = absl::GetFlag(FLAGS_emit_loops),
      .emit_gate = absl::GetFlag(FLAGS_emit_gate),
      .generate_proc = absl::GetFlag(FLAGS_generate_proc),
      .emit_stateless_proc = absl::GetFlag(FLAGS_emit_stateless_proc),
      .emit_zero_width_bits_types =
          absl::GetFlag(FLAGS_emit_zero_width_bits_types),
  };

  std::string top_entity_name = absl::GetFlag(FLAGS_top_entity_name);
  std::string module_name = absl::GetFlag(FLAGS_module_name);

  xls::dslx::AstGenerator g(opts, rng, ft);
  XLS_ASSIGN_OR_RETURN(auto result, g.Generate(top_entity_name, module_name));

  auto dslx_text = result.module->ToString();

  // Parse and type check the DSLX program.
  dslx::ImportData import_data(
      dslx::CreateImportData(/*stdlib_path=*/"",
                             /*additional_search_paths=*/{},
                             /*enabled_warnings=*/dslx::kAllWarningsSet,
                             std::make_unique<dslx::RealFilesystem>()));
  absl::StatusOr<dslx::TypecheckedModule> tm =
      ParseAndTypecheck(dslx_text, "sample.x", "sample", &import_data);
  if (!tm.ok()) {
    LOG(ERROR) << "Generated sample failed to parse-and-typecheck ("
               << dslx_text.size() << " bytes):";
    XLS_LOG_LINES(ERROR, dslx_text);
    return tm.status();
  }

  std::cout << "// min_stages: " << result.min_stages << std::endl;
  std::cout << dslx_text << std::endl;

  return absl::OkStatus();
}

}  // namespace
}  // namespace xls

int main(int argc, char** argv) {
  std::vector<std::string_view> positional_arguments = xls::InitXls(
      absl::StrCat("Generate a random DSLX module; sample usage:\n\n\t",
                   argv[0]),
      argc, argv);

  if (!positional_arguments.empty()) {
    LOG(QFATAL) << "Unexpected positional arguments: "
                << absl::StrJoin(positional_arguments, ", ");
  }
  return xls::ExitStatus(xls::RealMain());
}
