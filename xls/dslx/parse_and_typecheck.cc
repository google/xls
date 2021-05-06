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

#include "xls/dslx/parse_and_typecheck.h"

#include "xls/dslx/parser.h"
#include "xls/dslx/scanner.h"
#include "xls/dslx/typecheck.h"

namespace xls::dslx {

absl::StatusOr<TypecheckedModule> ParseAndTypecheck(
    absl::string_view text, absl::string_view path,
    absl::string_view module_name, ImportData* import_data,
    absl::Span<const std::filesystem::path> additional_search_paths) {
  Scanner scanner{std::string{path}, std::string{text}};
  Parser parser(std::string{module_name}, &scanner);
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Module> module, parser.ParseModule());
  XLS_ASSIGN_OR_RETURN(
      TypeInfo * type_info,
      CheckModule(module.get(), import_data, additional_search_paths));
  TypecheckedModule result{module.get(), type_info};
  XLS_ASSIGN_OR_RETURN(ImportTokens subject,
                       ImportTokens::FromString(module_name));
  XLS_RETURN_IF_ERROR(
      import_data->Put(subject, ModuleInfo{std::move(module), type_info})
          .status());
  return result;
}

}  // namespace xls::dslx
