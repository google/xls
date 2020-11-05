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

#include "xls/dslx/import_routines.h"

#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/dslx/cpp_scanner.h"
#include "xls/dslx/parser.h"

namespace xls::dslx {

absl::StatusOr<const ModuleInfo*> DoImport(const TypecheckFn& ftypecheck,
                                           const ImportTokens& subject,
                                           ImportCache* cache) {
  if (cache->Contains(subject)) {
    return cache->Get(subject);
  }

  XLS_VLOG(3) << "DoImport (uncached) subject: " << subject.ToString();

  const std::vector<std::string>& pieces = subject.pieces();
  std::filesystem::path path;
  std::filesystem::path parent_path;
  if (pieces.size() == 1 && (pieces[0] == "std" || pieces[0] == "float32" ||
                             pieces[0] == "bfloat16")) {
    path = absl::StrFormat("xls/dslx/stdlib/%s.x", pieces[0]);
  } else {
    path = absl::StrJoin(pieces, "/") + ".x";
    parent_path = ".." / path;
  }

  std::string fully_qualified_name = absl::StrJoin(pieces, ".");
  std::string contents;

  if (FileExists(path).ok()) {
    XLS_ASSIGN_OR_RETURN(contents, GetFileContents(path));
  } else if (FileExists(parent_path).ok()) {
    // Genrules in-house execute inside a subdirectory, so we also search
    // starting from the parent directory for now.
    //
    // An alternative would be to explicitly note the DSLX_PATH when invoking
    // the tool in this special genrule context, but since we expect module
    // paths to be fully qualified at the moment, we opt for this kluge.
    path = parent_path;
    XLS_ASSIGN_OR_RETURN(contents, GetFileContents(path));
  } else {
    XLS_ASSIGN_OR_RETURN(path, GetXlsRunfilePath(path));
    XLS_ASSIGN_OR_RETURN(contents, GetFileContents(path));
  }

  XLS_VLOG(3) << "Parsing and typechecking " << fully_qualified_name
              << ": start";

  Scanner scanner(path, contents);
  Parser parser(/*module_name=*/fully_qualified_name, &scanner);
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<Module> module, parser.ParseModule());
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<TypeInfo> type_info, ftypecheck(module));
  return cache->Put(subject,
                    ModuleInfo{std::move(module), std::move(type_info)});
}

}  // namespace xls::dslx
