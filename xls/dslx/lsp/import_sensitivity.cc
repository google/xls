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

#include "xls/dslx/lsp/import_sensitivity.h"

#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_join.h"

namespace xls::dslx {

void ImportSensitivity::NoteImportAttempt(std::string_view importer_uri,
                                          std::string_view imported_uri) {
  imported_to_importers_[imported_uri].insert(std::string{importer_uri});
}

std::vector<std::string> ImportSensitivity::GatherAllSensitiveToChangeIn(
    std::string_view uri) {
  // We maintain a set for O(1) checking of whether a file has already been
  // placed in the result, but we keep a vector for stable dependency ordering
  // and avoiding non-deterministic behavior.
  absl::flat_hash_set<std::string> result_set = {std::string{uri}};
  std::vector<std::string> result = {std::string{uri}};

  auto add_result = [&](std::string_view to_add) {
    auto [it, inserted] = result_set.insert(std::string{to_add});
    if (inserted) {
      result.push_back(std::string{to_add});
    }
  };

  // We note which ones we've already checked and we keep a worklist of things
  // we need to check, seeded with the ones we know import `uri`.
  absl::flat_hash_set<std::string> checked;
  std::vector<std::string> to_check(imported_to_importers_[uri].begin(),
                                    imported_to_importers_[uri].end());

  for (const std::string& s : to_check) {
    add_result(s);
  }

  while (!to_check.empty()) {
    // Pluck off the `to_check` worklist.
    std::string uri = to_check.back();
    to_check.pop_back();

    auto [it, inserted] = checked.insert(uri);
    if (!inserted) {
      continue;
    }

    const absl::btree_set<std::string>& importers = imported_to_importers_[uri];
    for (const std::string& s : importers) {
      add_result(s);
    }
    to_check.insert(to_check.end(), importers.begin(), importers.end());
  }

  VLOG(5) << "GatherAllSensitiveToChangeIn; uri: " << uri
          << " result: " << absl::StrJoin(result, ", ");
  return result;
}

}  // namespace xls::dslx
