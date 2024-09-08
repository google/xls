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

#include "xls/dslx/type_system/format_type_mismatch.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/zip_types.h"

namespace xls::dslx {
namespace {

constexpr std::string_view kAnsiReset = "\33[0m";
constexpr std::string_view kAnsiRed = "\33[31m";
constexpr std::string_view kAnsiBoldOn = "\33[1m";
constexpr std::string_view kAnsiBoldOff = "\33[22m";

struct TupleMismatchData {
  const Type* different;
  int64_t index;
  const TupleType* parent;
};

struct MismatchData {
  std::vector<std::pair<const Type*, const Type*>> mismatches;
  std::vector<TupleMismatchData> tuple_missing;
  std::vector<TupleMismatchData> tuple_extra;
};

// Populates the ref given as `mismatches` with the mismatches.
//
// Note: we could have this use the auto-formatting pretty printer to get more
// readable line wrapping for very long types, but we hope that highlighting the
// subtype mismatches inside the broader type might suffice for now.
class Callbacks : public ZipTypesCallbacks {
 public:
  explicit Callbacks(MismatchData& mismatches) : mismatches_(mismatches) {}

  absl::Status NoteAggregateStart(const AggregatePair& aggregates) override {
    return absl::visit(
        Visitor{
            [&](std::pair<const TupleType*, const TupleType*>) {
              AddMatchedBoth("(");
              return absl::OkStatus();
            },
            [&](std::pair<const StructType*, const StructType*> p) {
              AddMatchedBoth(
                  absl::StrCat(p.first->nominal_type().identifier(), "{"));
              return absl::OkStatus();
            },
            [&](std::pair<const ArrayType*, const ArrayType*> p) {
              /* goes at the end */
              return absl::OkStatus();
            },
            [&](std::pair<const ChannelType*, const ChannelType*> p) {
              AddMatchedBoth("chan(");
              return absl::OkStatus();
            },
            [&](std::pair<const FunctionType*, const FunctionType*> p) {
              return absl::UnimplementedError(
                  "Cannot print diffs of function types.");
            },
            [&](std::pair<const MetaType*, const MetaType*> p) {
              AddMatchedBoth("typeof(");
              return absl::OkStatus();
            },
        },
        aggregates);
  }

  absl::Status NoteAggregateNext(const AggregatePair& aggregates) override {
    return absl::visit(
        Visitor{
            [&](auto p) { return absl::OkStatus(); },
            [&](std::pair<const TupleType*, const TupleType*>) {
              AddMatchedBoth(", ");
              return absl::OkStatus();
            },
            [&](std::pair<const StructType*, const StructType*> p) {
              AddMatchedBoth(", ");
              return absl::OkStatus();
            },
        },
        aggregates);
  }

  absl::Status NoteAggregateEnd(const AggregatePair& aggregates) override {
    return absl::visit(
        Visitor{
            [&](std::pair<const TupleType*, const TupleType*>) {
              AddMatchedBoth(")");
              return absl::OkStatus();
            },
            [&](std::pair<const StructType*, const StructType*>) {
              AddMatchedBoth("}");
              return absl::OkStatus();
            },
            [&](std::pair<const ArrayType*, const ArrayType*> p) {
              AddMatched(absl::StrCat("[", p.first->size().ToString(), "]"),
                         &colorized_lhs_);
              AddMatched(absl::StrCat("[", p.second->size().ToString(), "]"),
                         &colorized_rhs_);
              return absl::OkStatus();
            },
            [&](std::pair<const ChannelType*, const ChannelType*> p) {
              AddMatchedBoth(")");
              return absl::OkStatus();
            },
            [&](std::pair<const FunctionType*, const FunctionType*> p) {
              return absl::UnimplementedError(
                  "Cannot print diffs of function types.");
            },
            [&](std::pair<const MetaType*, const MetaType*> p) {
              AddMatchedBoth(")");
              return absl::OkStatus();
            },
        },
        aggregates);
  }

  absl::Status NoteMatchedLeafType(const Type& lhs, const Type* lhs_parent,
                                   const Type& rhs,
                                   const Type* rhs_parent) override {
    match_count_++;
    BeforeType(lhs, lhs_parent, rhs, rhs_parent);
    AddMatched(lhs.ToString(), &colorized_lhs_);
    AddMatched(rhs.ToString(), &colorized_rhs_);
    return absl::OkStatus();
  }

  absl::Status NoteTypeMismatch(const Type& lhs, const Type* lhs_parent,
                                const Type& rhs,
                                const Type* rhs_parent) override {
    if (auto* lhs_tuple = dynamic_cast<const TupleType*>(&lhs)) {
      if (auto* rhs_tuple = dynamic_cast<const TupleType*>(&rhs)) {
        XLS_RET_CHECK_NE(lhs_tuple->size(), rhs_tuple->size());
        for (int64_t i = std::min(lhs_tuple->size(), rhs_tuple->size());
             i < std::max(lhs_tuple->size(), rhs_tuple->size()); ++i) {
          const Type* elhs =
              i < lhs_tuple->size() ? &lhs_tuple->GetMemberType(i) : nullptr;
          const Type* erhs =
              i < rhs_tuple->size() ? &rhs_tuple->GetMemberType(i) : nullptr;
          if (elhs == nullptr) {
            mismatches_.tuple_extra.push_back(TupleMismatchData{
                .different = erhs, .index = i, .parent = rhs_tuple});
          }
          if (erhs == nullptr) {
            mismatches_.tuple_missing.push_back(TupleMismatchData{
                .different = elhs, .index = i, .parent = lhs_tuple});
          }
        }
      }
    }

    mismatches_.mismatches.push_back({&lhs, &rhs});
    BeforeType(lhs, lhs_parent, rhs, rhs_parent);
    AddMismatched(lhs.ToString(), rhs.ToString());
    return absl::OkStatus();
  }

  std::string_view colorized_lhs() const { return colorized_lhs_; }
  std::string_view colorized_rhs() const { return colorized_rhs_; }

  int64_t match_count() const { return match_count_; }

 private:
  // Adds a struct field before the RHS.
  void BeforeType(const Type& lhs, const Type* lhs_parent, const Type& rhs,
                  const Type* rhs_parent) {
    if (lhs_parent == nullptr) {
      return;
    }
    if (auto* parent_struct = dynamic_cast<const StructType*>(lhs_parent);
        parent_struct != nullptr) {
      int64_t index = parent_struct->IndexOf(lhs).value();
      AddMatchedBoth(absl::StrCat(parent_struct->GetMemberName(index), ": "));
    }
  }

  void AddMismatched(std::string_view lhs, std::string_view rhs) {
    absl::StrAppend(&colorized_lhs_, kAnsiRed, lhs, kAnsiReset);
    absl::StrAppend(&colorized_rhs_, kAnsiRed, rhs, kAnsiReset);
  }

  void AddMatched(std::string_view matched_text, std::string* out) {
    absl::StrAppend(out, matched_text);
  }
  // Helper that adds the matched text to both the LHS and RHS.
  void AddMatchedBoth(std::string_view matched_text) {
    AddMatched(matched_text, &colorized_lhs_);
    AddMatched(matched_text, &colorized_rhs_);
  }

  // We start the string off with an ANSI reset since we have our own coloring
  // we do inside.
  std::string colorized_lhs_;
  std::string colorized_rhs_;
  MismatchData& mismatches_;
  int64_t match_count_ = 0;
};

}  // namespace

absl::StatusOr<std::string> FormatTypeMismatch(const Type& lhs, const Type& rhs,
                                               const FileTable& file_table) {
  MismatchData data;

  Callbacks callbacks(data);

  XLS_RETURN_IF_ERROR(ZipTypes(lhs, rhs, callbacks));

  CHECK(!data.mismatches.empty())
      << "type mismatch info not constructed correctly for types "
      << lhs.GetDebugTypeName() << " vs. " << rhs.GetDebugTypeName();

  std::vector<std::string> lines;

  if (!data.tuple_missing.empty()) {
    lines.push_back("Tuple is missing elements:");
    for (const TupleMismatchData& tmd : data.tuple_missing) {
      lines.push_back(absl::StrFormat("   %s (index %d of %s)",
                                      tmd.different->ToString(), tmd.index,
                                      tmd.parent->ToString()));
    }
  }
  if (!data.tuple_extra.empty()) {
    lines.push_back("Tuple has extra elements:");
    for (const TupleMismatchData& tmd : data.tuple_extra) {
      lines.push_back(absl::StrFormat("   %s (index %d of %s)",
                                      tmd.different->ToString(), tmd.index,
                                      tmd.parent->ToString()));
    }
  }

  if (callbacks.match_count() == 0) {
    lines.push_back("Type mismatch:");
    std::string lhs_string = lhs.ToString();
    std::string rhs_string = rhs.ToString();
    // If the text of the LHS and RHS are identical (e.g. structs with the same
    // names that are defined in different modules are the cause of the
    // mismatch) we try to fully qualify type names in order to not give a
    // confusing error message.
    if (lhs_string == rhs_string) {
      lhs_string = lhs.ToStringFullyQualified(file_table);
      rhs_string = rhs.ToStringFullyQualified(file_table);
    }
    lines.push_back(absl::StrFormat("   %s", lhs_string));
    lines.push_back(absl::StrFormat("vs %s", rhs_string));
  } else {
    lines.push_back(absl::StrFormat("%sMismatched elements %swithin%s type:",
                                    kAnsiReset, kAnsiBoldOn, kAnsiBoldOff));
    for (auto [lhs_mismatch, rhs_mismatch] : data.mismatches) {
      lines.push_back(absl::StrFormat("   %s", lhs_mismatch->ToString()));
      lines.push_back(absl::StrFormat("vs %s", rhs_mismatch->ToString()));
    }
    lines.push_back(absl::StrFormat("%sOverall%s type mismatch:", kAnsiBoldOn,
                                    kAnsiBoldOff));
    lines.push_back(
        absl::StrFormat("%s   %s", kAnsiReset, callbacks.colorized_lhs()));
    lines.push_back(absl::StrFormat("vs %s", callbacks.colorized_rhs()));
  }
  return absl::StrJoin(lines, "\n");
}

}  // namespace xls::dslx
