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

#ifndef XLS_DSLX_LSP_IMPORT_SENSITIVITY_H_
#define XLS_DSLX_LSP_IMPORT_SENSITIVITY_H_

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"

namespace xls::dslx {

// Tracks import attempts that are made by DSLX file URIs in order to determine
// which we should try to re-evaluate when there's a change.
class ImportSensitivity {
 public:
  void NoteImportAttempt(std::string_view importer_uri,
                         std::string_view imported_uri);

  std::vector<std::string> GatherAllSensitiveToChangeIn(std::string_view uri);

 private:
  // Note: we use a btree set for the values for stable ordering of results.
  absl::flat_hash_map<std::string, absl::btree_set<std::string>>
      imported_to_importers_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_LSP_IMPORT_SENSITIVITY_H_
