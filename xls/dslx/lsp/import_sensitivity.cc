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
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/dslx/lsp/lsp_uri.h"

namespace xls::dslx {

void ImportSensitivity::NoteImportAttempt(const LspUri& importer_uri,
                                          const LspUri& imported_uri) {
  imported_to_importers_[imported_uri].insert(importer_uri);
}

std::vector<LspUri> ImportSensitivity::GatherAllSensitiveToChangeIn(
    const LspUri& uri) {
  // We maintain a set for O(1) checking of whether a file has already been
  // placed in the result, but we keep a vector for stable dependency ordering
  // and avoiding non-deterministic behavior.
  absl::flat_hash_set<LspUri> result_set = {uri};
  std::vector<LspUri> result = {uri};

  auto add_result = [&](const LspUri& to_add) {
    auto [it, inserted] = result_set.insert(to_add);
    if (inserted) {
      result.push_back(to_add);
    }
  };

  // We note which ones we've already checked and we keep a worklist of things
  // we need to check, seeded with the ones we know import `uri`.
  absl::flat_hash_set<LspUri> checked;
  std::vector<LspUri> to_check(imported_to_importers_[uri].begin(),
                               imported_to_importers_[uri].end());

  for (const LspUri& s : to_check) {
    add_result(s);
  }

  while (!to_check.empty()) {
    // Pluck off the `to_check` worklist.
    const LspUri uri = to_check.back();
    to_check.pop_back();

    auto [it, inserted] = checked.insert(uri);
    if (!inserted) {
      continue;
    }

    const absl::btree_set<LspUri>& importers = imported_to_importers_[uri];
    for (const LspUri& s : importers) {
      add_result(s);
    }
    to_check.insert(to_check.end(), importers.begin(), importers.end());
  }

  VLOG(5) << "GatherAllSensitiveToChangeIn; uri: " << uri << " result: "
          << absl::StrJoin(result, ", ",
                           [](std::string* out, const LspUri& uri) {
                             absl::StrAppend(out, uri.GetStringView());
                           });
  return result;
}

}  // namespace xls::dslx
