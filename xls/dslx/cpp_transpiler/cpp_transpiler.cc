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

#include "xls/dslx/cpp_transpiler/cpp_transpiler.h"

#include <filesystem>  // NOLINT
#include <memory>
#include <string>
#include <string_view>

#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/cpp_transpiler/cpp_type_generator.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

absl::StatusOr<CppSource> TranspileToCpp(Module* module,
                                         ImportData* import_data,
                                         std::string_view output_header_path,
                                         std::string_view namespaces) {
  constexpr std::string_view kHeaderTemplate =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#ifndef $0
#define $0
#include <array>
#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "xls/public/value.h"

$2$1$3

#endif  // $0
)";

  constexpr std::string_view kSourceTemplate =
      R"(// AUTOMATICALLY GENERATED FILE FROM `xls/dslx/cpp_transpiler`. DO NOT EDIT!
#include <array>
#include <string>
#include <string_view>
#include <vector>

#include "%s"
#include "absl/base/macros.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/public/status_macros.h"
#include "xls/public/value.h"

[[maybe_unused]] static bool FitsInNBitsSigned(int64_t value, int64_t n) {
  // All bits from [n - 1, 64) must be all zero or all ones.
  if (n >= 64) {
    return true;
  }
  // `mask` is 1111...00000 with n zeros.
  uint64_t mask = ~((uint64_t{1} << n) - 1);
  uint64_t value_as_unsigned = static_cast<uint64_t>(value);
  return (mask & value_as_unsigned) == 0 ||
       (mask & value_as_unsigned) == mask;
}

[[maybe_unused]] static bool FitsInNBitsUnsigned(uint64_t value, int64_t n) {
  if (n >= 64) {
    return true;
  }
  return value < (uint64_t{1} << n);
}

[[maybe_unused]] static std::string __indent(int64_t amount) {
  return std::string(amount * 2, ' ');
}

%s%s%s
)";
  XLS_ASSIGN_OR_RETURN(TypeInfo * type_info,
                       import_data->GetRootTypeInfo(module));
  std::vector<std::string> header;
  std::vector<std::string> source;

  // TODO(b/351861097): 2024-07-08 Convert dependent types in other imports so
  // that types defined in imported files can be used.
  for (const TypeDefinition& def : module->GetTypeDefinitions()) {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<CppTypeGenerator> generator,
                         CppTypeGenerator::Create(def, type_info, import_data));
    XLS_ASSIGN_OR_RETURN(CppSource result, generator->GetCppSource());
    header.push_back(result.header);
    source.push_back(result.source);
  }

  std::string header_guard;
  std::filesystem::path current_path = output_header_path;
  while (!current_path.empty() && current_path != current_path.root_path()) {
    std::string chunk =
        absl::AsciiStrToUpper(std::string{current_path.filename()});
    chunk = absl::StrReplaceAll(chunk, {{".", "_"}, {"-", "_"}});
    header_guard = absl::StrCat(chunk, "_", header_guard);
    current_path = current_path.parent_path();
  }

  std::string namespace_begin;
  std::string namespace_end;
  if (!namespaces.empty()) {
    namespace_begin = absl::StrCat("namespace ", namespaces, " {\n\n");
    namespace_end = absl::StrCat("\n\n}  // namespace ", namespaces);
  }

  return CppSource{
      absl::Substitute(kHeaderTemplate, header_guard,
                       absl::StrJoin(header, "\n\n"), namespace_begin,
                       namespace_end),
      absl::StrFormat(kSourceTemplate, output_header_path, namespace_begin,
                      absl::StrJoin(source, "\n\n"), namespace_end)};
}

}  // namespace xls::dslx
