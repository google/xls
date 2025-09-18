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

#include "xls/dslx/ir_convert/conversion_record.h"

#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/proc_id.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

std::string ConversionRecordsToString(
    absl::Span<const ConversionRecord> records) {
  return absl::StrCat(
      "[",
      absl::StrJoin(records, ",\n  ",
                    [](std::string* out, const ConversionRecord& record) {
                      absl::StrAppend(out, record.ToString());
                    }),
      "]");
}

// -- class ConversionRecord

/* static */ absl::Status ConversionRecord::ValidateParametrics(
    Function* f, const ParametricEnv& parametric_env) {
  absl::btree_set<std::string> symbolic_binding_keys =
      parametric_env.GetKeySet();

  auto set_to_string = [](const absl::btree_set<std::string>& s) {
    return absl::StrCat("{", absl::StrJoin(s, ", "), "}");
  };
  // TODO(leary): 2020-11-19 We use btrees in particular so this could use dual
  // iterators via the sorted property for O(n) superset comparison, but this
  // was easier to write and know it was correct on a first cut (couldn't find a
  // superset helper in absl's container algorithms at a first pass).
  auto is_superset = [](absl::btree_set<std::string> lhs,
                        const absl::btree_set<std::string>& rhs) {
    for (const auto& item : rhs) {
      lhs.erase(item);
    }
    return !lhs.empty();
  };

  if (is_superset(f->GetFreeParametricKeySet(), symbolic_binding_keys)) {
    return absl::InternalError(absl::StrFormat(
        "Not enough symbolic bindings to convert function: %s; need %s got %s",
        f->identifier(), set_to_string(f->GetFreeParametricKeySet()),
        set_to_string(symbolic_binding_keys)));
  }
  return absl::OkStatus();
}

/* static */ absl::StatusOr<ConversionRecord> ConversionRecord::Make(
    Function* f, const Invocation* invocation, Module* module,
    TypeInfo* type_info, ParametricEnv parametric_env,
    std::optional<ProcId> proc_id, bool is_top) {
  XLS_RETURN_IF_ERROR(ConversionRecord::ValidateParametrics(f, parametric_env));

  return ConversionRecord(f, invocation, module, type_info,
                          std::move(parametric_env), std::move(proc_id),
                          is_top);
}

std::string ConversionRecord::ToString() const {
  std::string proc_id = "<none>";
  if (proc_id_.has_value()) {
    proc_id = proc_id_.value().ToString();
  }
  return absl::StrFormat(
      "ConversionRecord{m=%s, f=%s, top=%s, pid=%s, parametric_env=%s}",
      module_->name(), f_->identifier(), is_top_ ? "true" : "false", proc_id,
      parametric_env_.ToString());
}

}  // namespace xls::dslx
