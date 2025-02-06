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
#ifndef XLS_DSLX_EXHAUSTIVENESS_MATCH_EXHAUSTIVENESS_CHECKER_H_
#define XLS_DSLX_EXHAUSTIVENESS_MATCH_EXHAUSTIVENESS_CHECKER_H_

#include <optional>
#include <vector>

#include "xls/dslx/exhaustiveness/interp_value_interval.h"
#include "xls/dslx/exhaustiveness/nd_region.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"

namespace xls::dslx {

// Object that we can incrementally feed match arms/patterns to and ask whether
// we've reached a point where the patterns are exhaustive. This is useful for
// flagging a warning right when we've reached the point that the arms are
// exhaustive.
class MatchExhaustivenessChecker {
 public:
  MatchExhaustivenessChecker(const Span& matched_expr_span,
                             const ImportData& import_data,
                             const TypeInfo& type_info,
                             const Type& matched_type);

  // Returns whether we've reached a point of exhaustiveness after incorporating
  // the given `pattern`.
  bool AddPattern(const NameDefTree& pattern);

  // Returns whether, based on already-added patterns, we're exhaustive.
  bool IsExhaustive() const;

  // This method returns an optional "sample" value from the uncovered input
  // space. It picks (for now) the first uncovered ND region and for each
  // dimension, takes the lower bound. If there is only one dimension the value
  // is returned directly; otherwise the components are aggregated into a tuple.
  //
  std::optional<InterpValue> SampleSimplestUncoveredValue() const;

 private:
  const FileTable& file_table() const { return type_info_.file_table(); }

  const Span matched_expr_span_;

  const ImportData& import_data_;
  const TypeInfo& type_info_;
  const Type& matched_type_;

  // Flattened version of the pattern tuple, each element of this vector is a
  // dimension in the NdRegion below.
  std::vector<const Type*> leaf_types_;

  // The remaining region of the value space that we need to test.
  NdRegion remaining_;
};

// Returns the full interval range we use to represent the contents of an enum
// type -- exposed in the header for purposes of testing.
InterpValueInterval MakeFullIntervalForEnumType(const EnumType& enum_type);

// Returns the point interval range we use to represent the contents of an enum
// value -- exposed in the header for purposes of testing.
InterpValueInterval MakePointIntervalForEnumType(const EnumType& enum_type,
                                                 const InterpValue& value,
                                                 const ImportData& import_data);

}  // namespace xls::dslx

#endif  // XLS_DSLX_EXHAUSTIVENESS_MATCH_EXHAUSTIVENESS_CHECKER_H_
