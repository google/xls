// Copyright 2024 The XLS Authors
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

#include "xls/public/c_api.h"

#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/ir_parser.h"
#include "xls/public/runtime_build_actions.h"

namespace {

std::vector<std::filesystem::path> ToCpp(const char* additional_search_paths[],
                                         size_t additional_search_paths_count) {
  std::vector<std::filesystem::path> additional_search_paths_cpp;
  additional_search_paths_cpp.reserve(additional_search_paths_count);
  for (size_t i = 0; i < additional_search_paths_count; ++i) {
    const char* additional_search_path = additional_search_paths[i];
    CHECK(additional_search_path != nullptr);
    additional_search_paths_cpp.push_back(additional_search_path);
  }
  return additional_search_paths_cpp;
}

char* ToOwnedCString(const std::string& s) { return strdup(s.c_str()); }

// Helper function that we can use to adapt to the common C API pattern when
// we're returning an `absl::StatusOr<std::string>` value.
bool ReturnStringHelper(absl::StatusOr<std::string>& to_return,
                        char** error_out, char** value_out) {
  if (to_return.ok()) {
    *value_out = ToOwnedCString(to_return.value());
    *error_out = nullptr;
    return true;
  }

  *value_out = nullptr;
  *error_out = ToOwnedCString(to_return.status().ToString());
  return false;
}

}  // namespace

extern "C" {

bool xls_convert_dslx_to_ir(const char* dslx, const char* path,
                            const char* module_name,
                            const char* dslx_stdlib_path,
                            const char* additional_search_paths[],
                            size_t additional_search_paths_count,
                            char** error_out, char** ir_out) {
  CHECK(dslx != nullptr);
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxToIr(
      dslx, path, module_name, dslx_stdlib_path, additional_search_paths_cpp);
  return ReturnStringHelper(result, error_out, ir_out);
}

bool xls_convert_dslx_path_to_ir(const char* path, const char* dslx_stdlib_path,
                                 const char* additional_search_paths[],
                                 size_t additional_search_paths_count,
                                 char** error_out, char** ir_out) {
  CHECK(path != nullptr);
  CHECK(dslx_stdlib_path != nullptr);
  CHECK(error_out != nullptr);

  std::vector<std::filesystem::path> additional_search_paths_cpp =
      ToCpp(additional_search_paths, additional_search_paths_count);

  absl::StatusOr<std::string> result = xls::ConvertDslxPathToIr(
      path, dslx_stdlib_path, additional_search_paths_cpp);
  return ReturnStringHelper(result, error_out, ir_out);
}

bool xls_optimize_ir(const char* ir, const char* top, char** error_out,
                     char** ir_out) {
  CHECK(ir != nullptr);
  CHECK(top != nullptr);

  absl::StatusOr<std::string> result = xls::OptimizeIr(ir, top);
  return ReturnStringHelper(result, error_out, ir_out);
}

bool xls_mangle_dslx_name(const char* module_name, const char* function_name,
                          char** error_out, char** mangled_out) {
  CHECK(module_name != nullptr);
  CHECK(function_name != nullptr);

  absl::StatusOr<std::string> result =
      xls::MangleDslxName(module_name, function_name);
  return ReturnStringHelper(result, error_out, mangled_out);
}

bool xls_parse_typed_value(const char* input, char** error_out,
                           xls_value** xls_value_out) {
  CHECK(input != nullptr);
  CHECK(error_out != nullptr);
  CHECK(xls_value_out != nullptr);

  absl::StatusOr<xls::Value> value_or = xls::Parser::ParseTypedValue(input);
  if (value_or.ok()) {
    *xls_value_out = reinterpret_cast<xls_value*>(
        new xls::Value(std::move(value_or).value()));
    return true;
  }

  *xls_value_out = nullptr;
  *error_out = ToOwnedCString(value_or.status().ToString());
  return false;
}

void xls_value_free(xls_value* v) {
  if (v == nullptr) {
    return;
  }
  delete reinterpret_cast<xls::Value*>(v);
}

void xls_package_free(struct xls_package* p) {
  if (p == nullptr) {
    return;
  }
  delete reinterpret_cast<xls::Package*>(p);
}

bool xls_value_to_string(const struct xls_value* v, char** string_out) {
  CHECK(v != nullptr);
  CHECK(string_out != nullptr);
  std::string s = reinterpret_cast<const xls::Value*>(v)->ToString();
  *string_out = strdup(s.c_str());
  return *string_out != nullptr;
}

bool xls_value_eq(const struct xls_value* v, const struct xls_value* w) {
  CHECK(v != nullptr);
  CHECK(w != nullptr);

  const auto* lhs = reinterpret_cast<const xls::Value*>(v);
  const auto* rhs = reinterpret_cast<const xls::Value*>(w);
  return *lhs == *rhs;
}

bool xls_package_to_string(const struct xls_package* p, char** string_out) {
  CHECK(p != nullptr);
  CHECK(string_out != nullptr);
  std::string s = reinterpret_cast<const xls::Package*>(p)->DumpIr();
  *string_out = strdup(s.c_str());
  return *string_out != nullptr;
}

bool xls_parse_ir_package(const char* ir, const char* filename,
                          char** error_out,
                          struct xls_package** xls_package_out) {
  CHECK(ir != nullptr);
  CHECK(error_out != nullptr);
  CHECK(xls_package_out != nullptr);

  std::optional<std::string_view> cpp_filename;
  if (filename != nullptr) {
    cpp_filename.emplace(filename);
  }
  absl::StatusOr<std::unique_ptr<xls::Package>> package_or =
      xls::Parser::ParsePackage(ir, cpp_filename);
  if (package_or.ok()) {
    *xls_package_out =
        reinterpret_cast<xls_package*>(package_or.value().release());
    *error_out = nullptr;
    return true;
  }

  *xls_package_out = nullptr;
  *error_out = ToOwnedCString(package_or.status().ToString());
  return false;
}

bool xls_package_get_function(struct xls_package* package,
                              const char* function_name, char** error_out,
                              struct xls_function** result_out) {
  xls::Package* xls_package = reinterpret_cast<xls::Package*>(package);
  absl::StatusOr<xls::Function*> function_or =
      xls_package->GetFunction(function_name);
  if (function_or.ok()) {
    *result_out = reinterpret_cast<struct xls_function*>(function_or.value());
    return true;
  }

  *result_out = nullptr;
  *error_out = ToOwnedCString(function_or.status().ToString());
  return false;
}

bool xls_interpret_function(struct xls_function* function, size_t argc,
                            const struct xls_value** args, char** error_out,
                            struct xls_value** result_out) {
  CHECK(function != nullptr);
  CHECK(args != nullptr);

  xls::Function* xls_function = reinterpret_cast<xls::Function*>(function);

  std::vector<xls::Value> xls_args;
  xls_args.reserve(argc);
  for (size_t i = 0; i < argc; ++i) {
    CHECK(args[i] != nullptr);
    xls_args.push_back(*reinterpret_cast<const xls::Value*>(args[i]));
  }

  absl::StatusOr<xls::InterpreterResult<xls::Value>> result_or =
      xls::InterpretFunction(xls_function, xls_args);
  if (result_or.ok()) {
    // TODO(cdleary): 2024-05-30 We should pass back interpreter events through
    // this API instead of dropping them.
    xls::Value result_value =
        xls::DropInterpreterEvents(std::move(result_or)).value();
    *result_out = reinterpret_cast<struct xls_value*>(
        new xls::Value(std::move(result_value)));
    return true;
  }

  *result_out = nullptr;
  *error_out = ToOwnedCString(result_or.status().ToString());
  return false;
}

}  // extern "C"
