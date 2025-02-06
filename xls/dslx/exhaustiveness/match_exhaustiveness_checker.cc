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

#include "xls/dslx/exhaustiveness/match_exhaustiveness_checker.h"

#include <memory>

#include "absl/log/log.h"
#include "xls/common/visitor.h"
#include "xls/dslx/import_data.h"

namespace xls::dslx {
namespace {

std::vector<const Type*> GetLeafTypesInternal(const Type& type) {
  if (type.IsTuple()) {
    std::vector<const Type*> result;
    for (const std::unique_ptr<Type>& member : type.AsTuple().members()) {
      std::vector<const Type*> member_leaf_types =
          GetLeafTypesInternal(*member);
      result.insert(result.end(), member_leaf_types.begin(),
                    member_leaf_types.end());
    }
    return result;
  }
  return {&type};
}

std::vector<const Type*> GetLeafTypes(const Type& type, const Span& span,
                                      const FileTable& file_table) {
  std::vector<const Type*> result = GetLeafTypesInternal(type);
  // Validate that all the matched-upon types are either bits or enums.
  for (const Type* leaf_type : result) {
    CHECK(GetBitsLike(*leaf_type).has_value() || leaf_type->IsEnum())
        << "Non-bits or non-enum type in matched-upon tuple: "
        << leaf_type->ToString() << " @ " << span.ToString(file_table);
  }
  return result;
}

// Sentinel type to indicate that some wildcard is present for a value. This
// lets us collapse out varieties of wildcards e.g. RestOfTuple and
// WildcardPattern and NameDef.
struct SomeWildcard {};

// NameDefTree::Leaf but where RestOfTuple has been resolved.
using PatternLeaf =
    std::variant<SomeWildcard, NameRef*, Range*, ColonRef*, Number*>;

InterpValueInterval MakeFullIntervalForType(const Type& type) {
  if (type.IsEnum()) {
    return MakeFullIntervalForEnumType(type.AsEnum());
  }
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  CHECK(bits_like.has_value())
      << "MakeFullIntervalForType; got non-bits type: " << type.ToString();
  int64_t bit_count = bits_like->size.GetAsInt64().value();
  bool is_signed = bits_like->is_signed.GetAsBool().value();
  InterpValue min = InterpValue::MakeMinValue(is_signed, bit_count);
  InterpValue max = InterpValue::MakeMaxValue(is_signed, bit_count);
  InterpValueInterval result(min, max);
  VLOG(5) << "MakeFullIntervalForType; type: `" << type.ToString()
          << "` result: " << result.ToString(/*show_types=*/false);
  return result;
}

// Returns the "full" intervals that can be used to represent the "no values
// have been exhausted" initial state.
std::vector<InterpValueInterval> GetFullIntervals(
    absl::Span<const Type* const> leaf_types) {
  std::vector<InterpValueInterval> result;
  for (const Type* leaf_type : leaf_types) {
    result.push_back(MakeFullIntervalForType(*leaf_type));
  }
  return result;
}

InterpValueInterval MakePointIntervalForType(const Type& type,
                                             const InterpValue& value,
                                             const ImportData& import_data) {
  VLOG(5) << "MakePointIntervalForType; type: `" << type.ToString()
          << "` value: `" << value.ToString() << "`";
  if (type.IsEnum()) {
    return MakePointIntervalForEnumType(type.AsEnum(), value, import_data);
  }
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  CHECK(bits_like.has_value())
      << "MakePointIntervalForType; got non-bits type: " << type.ToString();
  return InterpValueInterval(value, value);
}

InterpValueInterval MakeIntervalForType(const Type& type,
                                        const InterpValue& min,
                                        const InterpValue& max) {
  std::optional<BitsLikeProperties> bits_like = GetBitsLike(type);
  CHECK(bits_like.has_value())
      << "MakeIntervalForType; got non-bits type: " << type.ToString();
  return InterpValueInterval(min, max);
}

std::optional<InterpValueInterval> PatternToIntervalInternal(
    const PatternLeaf& leaf, const Type& leaf_type, const TypeInfo& type_info,
    const ImportData& import_data) {
  std::optional<InterpValueInterval> result = absl::visit(
      Visitor{
          [&](SomeWildcard /*unused*/) -> std::optional<InterpValueInterval> {
            return MakeFullIntervalForType(leaf_type);
          },
          [&](NameRef* name_ref) -> std::optional<InterpValueInterval> {
            std::optional<InterpValue> value =
                type_info.GetConstExprOption(name_ref);
            if (value.has_value()) {
              return MakePointIntervalForType(leaf_type, value.value(),
                                              import_data);
            }
            return MakeFullIntervalForType(leaf_type);
          },
          [&](Range* range) -> std::optional<InterpValueInterval> {
            std::optional<InterpValue> start =
                type_info.GetConstExprOption(range->start());
            std::optional<InterpValue> limit =
                type_info.GetConstExprOption(range->end());
            CHECK(start.has_value());
            CHECK(limit.has_value());
            if (start->Eq(limit.value())) {
              return std::nullopt;
            }
            std::optional<InterpValue> max = limit.value().Decrement();
            if (!max.has_value()) {
              // Underflow -- that means the range must be empty because the
              // limit is exclusive and is known to be representable in the
              // type.
              return std::nullopt;
            }
            if (max->Lt(start.value()).value().IsTrue()) {
              // max < start, so the range is empty.
              return std::nullopt;
            }
            return MakeIntervalForType(leaf_type, *start, max.value());
          },
          [&](ColonRef* colon_ref) -> std::optional<InterpValueInterval> {
            std::optional<InterpValue> value =
                type_info.GetConstExprOption(colon_ref);
            CHECK(value.has_value());
            VLOG(5) << "PatternToIntervalInternal; colon_ref: `"
                    << colon_ref->ToString() << "` value: `"
                    << value.value().ToString() << "`" << " leaf_type: `"
                    << leaf_type.ToString() << "`";
            return MakePointIntervalForType(leaf_type, value.value(),
                                            import_data);
          },
          [&](Number* number) -> std::optional<InterpValueInterval> {
            std::optional<InterpValue> value =
                type_info.GetConstExprOption(number);
            CHECK(value.has_value());
            return MakePointIntervalForType(leaf_type, value.value(),
                                            import_data);
          }},
      leaf);
  VLOG(5) << "PatternToIntervalInternal; leaf_type: `" << leaf_type.ToString()
          << "` result: "
          << (result.has_value() ? result->ToString(/*show_types=*/false)
                                 : "nullopt");
  return result;
}

PatternLeaf ToPatternLeaf(const NameDefTree::Leaf& leaf) {
  return absl::visit(
      Visitor{
          [&](NameDef* name_def) -> PatternLeaf { return SomeWildcard(); },
          [&](NameRef* name_ref) -> PatternLeaf { return name_ref; },
          [&](Range* range) -> PatternLeaf { return range; },
          [&](ColonRef* colon_ref) -> PatternLeaf { return colon_ref; },
          [&](WildcardPattern* wildcard_pattern) -> PatternLeaf {
            return SomeWildcard();
          },
          [&](Number* number) -> PatternLeaf { return number; },
          [&](RestOfTuple* rest_of_tuple) -> PatternLeaf {
            LOG(FATAL) << "RestOfTuple not valid for conversion to PatternLeaf";
          }},
      leaf);
}

std::vector<PatternLeaf> ExpandPatternLeaves(const NameDefTree& pattern,
                                             const Type& type,
                                             const FileTable& file_table) {
  VLOG(5) << "ExpandPatternLeaves; pattern: `" << pattern.ToString()
          << "` type: `" << type.ToString() << "`";
  // For an irrefutable pattern, simply return wildcards for every leaf.
  if (pattern.IsIrrefutable()) {
    std::vector<const Type*> leaf_types =
        GetLeafTypes(type, pattern.span(), file_table);
    return std::vector<PatternLeaf>(leaf_types.size(), SomeWildcard());
  }
  // If the type is not a tuple then we expect the pattern to be a single leaf.
  if (!type.IsTuple()) {
    std::vector<NameDefTree::Leaf> leaves = pattern.Flatten();
    CHECK_EQ(leaves.size(), 1)
        << "Expected a single leaf for non-tuple type, got " << leaves.size();
    return {ToPatternLeaf(leaves.front())};
  }
  // Walk through the pattern and expand any RestOfTuple markers into the
  // appropriate number of wildcards.
  //
  // In order to do this we have to recursively call to ExpandPatternLeaves for
  // any sub-tuples encountered.
  absl::Span<const std::unique_ptr<Type>> tuple_members =
      type.AsTuple().members();
  std::vector<std::variant<NameDefTree::Leaf, NameDefTree*>> flattened =
      pattern.Flatten1();

  // Note: there can be fewer flatten1'd nodes than tuple elements because of
  // RestOfTuple markers.
  //
  // We need the `+1` here because we can have RestOfTuple markers that map to
  // zero elements in the tuple (i.e. useless/redundant ones).
  CHECK_LE(flattened.size(), tuple_members.size() + 1);

  // The results correspond to leaf types.
  std::vector<PatternLeaf> result;

  // The tuple type index at *this level* of the tuple.
  // We bump this as we progress through -- note a single "flattened_index"
  // below can advance zero or more type indices.
  int64_t types_index = 0;

  for (int64_t flattened_index = 0; flattened_index < flattened.size();
       ++flattened_index) {
    VLOG(5) << "ExpandPatternLeaves; flattened_index: " << flattened_index
            << " flattened.size(): " << flattened.size()
            << " types_index: " << types_index
            << " tuple_members.size(): " << tuple_members.size();
    CHECK_LT(flattened_index, flattened.size())
        << "Flattened index out of bounds.";
    const auto& node = flattened[flattened_index];

    if (std::holds_alternative<NameDefTree*>(node)) {
      const NameDefTree* sub_pattern = std::get<NameDefTree*>(node);
      CHECK_LT(types_index, tuple_members.size());
      const Type& type_at_index = *tuple_members[types_index];

      std::vector<PatternLeaf> sub_pattern_leaves =
          ExpandPatternLeaves(*sub_pattern, type_at_index, file_table);

      result.insert(result.end(), sub_pattern_leaves.begin(),
                    sub_pattern_leaves.end());
      types_index += 1;
      continue;
    }
    const NameDefTree::Leaf& leaf = std::get<NameDefTree::Leaf>(node);
    absl::visit(
        Visitor{
            [&](const NameRef* n) {
              result.push_back(ToPatternLeaf(leaf));
              types_index += 1;
            },
            [&](const Range* r) {
              result.push_back(ToPatternLeaf(leaf));
              types_index += 1;
            },
            [&](const ColonRef* c) {
              result.push_back(ToPatternLeaf(leaf));
              types_index += 1;
            },
            [&](const Number* n) {
              result.push_back(ToPatternLeaf(leaf));
              types_index += 1;
            },
            [&](const RestOfTuple* /*unused*/) {
              // Instead of using flattened_index here, use types_index (the
              // number of tuple elements already matched) to figure out how
              // many items we need "in the rest".
              int64_t explicit_before = types_index;
              int64_t explicit_after = flattened.size() - flattened_index - 1;
              int64_t to_push =
                  tuple_members.size() - (explicit_before + explicit_after);
              VLOG(5) << "ExpandPatternLeaves; RestOfTuple at flattened_index: "
                      << flattened_index << " types_index: " << types_index
                      << " explicit_after: " << explicit_after
                      << " to_push: " << to_push;
              for (int64_t i = 0; i < to_push; ++i) {
                // We have to push wildcard data corresponding to the type.
                CHECK_LT(types_index, tuple_members.size());
                const Type& type_at_index = *tuple_members[types_index];
                for (int64_t i = 0;
                     i < GetLeafTypes(type_at_index, pattern.span(), file_table)
                             .size();
                     ++i) {
                  result.push_back(SomeWildcard());
                }
                types_index += 1;
              }
              VLOG(5) << "ExpandPatternLeaves; after RestOfTuple at "
                         "flattened_index: "
                      << flattened_index << " types_index: " << types_index
                      << " result.size(): " << result.size();
            },
            [&](const auto* irrefutable_leaf) {
              // Push back wildcards of the right size for the type.
              CHECK_LT(types_index, tuple_members.size());
              const Type& type_at_index = *tuple_members[types_index];
              for (int64_t i = 0;
                   i < GetLeafTypes(type_at_index, pattern.span(), file_table)
                           .size();
                   ++i) {
                result.push_back(SomeWildcard());
              }
              types_index += 1;
            }},
        leaf);
  }

  // Check that we got a consistent count between the razed tuple types and the
  // PatternLeaf vector.
  CHECK_EQ(result.size(), GetLeafTypes(type, pattern.span(), file_table).size())
      << "Sub-pattern leaves and tuple type must be the same size.";
  return result;
}

NdIntervalWithEmpty PatternToInterval(const NameDefTree& pattern,
                                      const Type& matched_type,
                                      absl::Span<const Type* const> leaf_types,
                                      const TypeInfo& type_info,
                                      const ImportData& import_data) {
  std::vector<PatternLeaf> pattern_leaves =
      ExpandPatternLeaves(pattern, matched_type, type_info.file_table());
  CHECK_EQ(pattern_leaves.size(), leaf_types.size())
      << "Pattern leaves and leaf types must be the same size.";

  // Each leaf describes some range in its dimension that it matches on --
  // together, they describe an n-dimensional interval.
  std::vector<std::optional<InterpValueInterval>> intervals;
  for (int64_t i = 0; i < pattern_leaves.size(); ++i) {
    intervals.push_back(PatternToIntervalInternal(
        pattern_leaves[i], *leaf_types[i], type_info, import_data));
  }
  NdIntervalWithEmpty result(intervals);
  VLOG(5) << "PatternToInterval; pattern: `" << pattern.ToString()
          << "` type: `" << matched_type.ToString()
          << "` result: " << result.ToString(/*show_types=*/false);
  return result;
}

NdRegion MakeFullNdRegion(absl::Span<const Type* const> leaf_types) {
  std::vector<InterpValueInterval> intervals = GetFullIntervals(leaf_types);
  std::vector<InterpValue> dim_extents;
  dim_extents.reserve(intervals.size());
  for (const InterpValueInterval& interval : intervals) {
    dim_extents.push_back(interval.max());
  }
  return NdRegion::MakeFromNdInterval(NdInterval(std::move(intervals)),
                                      std::move(dim_extents));
}

}  // namespace

// -- class MatchExhaustivenessChecker

MatchExhaustivenessChecker::MatchExhaustivenessChecker(
    const Span& matched_expr_span, const ImportData& import_data,
    const TypeInfo& type_info, const Type& matched_type)
    : matched_expr_span_(matched_expr_span),
      import_data_(import_data),
      type_info_(type_info),
      matched_type_(matched_type),
      leaf_types_(GetLeafTypes(matched_type, matched_expr_span, file_table())),
      remaining_(MakeFullNdRegion(leaf_types_)) {}

bool MatchExhaustivenessChecker::IsExhaustive() const {
  return remaining_.IsEmpty();
}

bool MatchExhaustivenessChecker::AddPattern(const NameDefTree& pattern) {
  VLOG(5) << "MatchExhaustivenessChecker::AddPattern: `" << pattern.ToString()
          << "` matched_type: `" << matched_type_.ToString() << "` @ "
          << pattern.span().ToString(file_table());

  NdIntervalWithEmpty this_pattern_interval = PatternToInterval(
      pattern, matched_type_, leaf_types_, type_info_, import_data_);
  remaining_ = remaining_.SubtractInterval(this_pattern_interval);
  return IsExhaustive();
}

std::optional<InterpValue>
MatchExhaustivenessChecker::SampleSimplestUncoveredValue() const {
  // If there are no uncovered regions, we are fully exhaustive.
  if (remaining_.IsEmpty()) {
    return std::nullopt;
  }

  // For now, just choose the first uncovered region.
  const NdInterval& nd_interval = remaining_.disjoint().front();
  std::vector<InterpValue> components;

  // For each dimension of the region, grab the lower bound (i.e. the simplest
  // value in that interval).
  for (int64_t i = 0; i < nd_interval.dims().size(); ++i) {
    const Type& type = *leaf_types_[i];
    const InterpValueInterval& interval = nd_interval.dims()[i];
    const InterpValue& min = interval.min();
    if (type.IsEnum()) {
      // We have to project back from dense space to enum name space.
      const EnumType& enum_type = type.AsEnum();
      const EnumDef& enum_def = enum_type.nominal_type();
      int64_t member_index = min.GetBitValueUnsigned().value();
      CHECK_LT(member_index, enum_def.values().size())
          << "Member index out of bounds: " << member_index
          << " for enum: " << enum_type.ToString();
      const EnumMember& member = enum_def.values()[member_index];
      InterpValue member_value =
          type_info_.GetConstExpr(member.name_def).value();
      VLOG(5) << "SampleSimplestUncoveredValue; enum_type: "
              << enum_type.ToString() << " member_index: " << member_index
              << " member: " << member.name_def->ToString()
              << " member_value: " << member_value.ToString();
      components.push_back(std::move(member_value));
    } else {
      components.push_back(min);
    }
  }

  // If we have a single component, return it directly; otherwise, return a
  // tuple.
  if (components.size() == 1) {
    return components[0];
  }
  return InterpValue::MakeTuple(components);
}

InterpValueInterval MakeFullIntervalForEnumType(const EnumType& enum_type) {
  int64_t bit_count = enum_type.size().GetAsInt64().value();
  const EnumDef& enum_def = enum_type.nominal_type();
  int64_t enum_value_count = enum_def.values().size();
  VLOG(5) << "MakeFullIntervalForEnumType; enum_type: " << enum_type.ToString()
          << " enum_value_count: " << enum_value_count;
  CHECK_GT(enum_value_count, 0)
      << "Cannot make full interval for enum type with no values: "
      << enum_type.ToString();
  // Note: regardless of the requested underlying type of the enum we use a
  // dense unsigned space to represent the values present in the enum namespace.
  InterpValue min = InterpValue::MakeUBits(bit_count, 0);
  InterpValue max = InterpValue::MakeUBits(bit_count, enum_value_count - 1);
  InterpValueInterval result(min, max);
  VLOG(5) << "MakeFullIntervalForEnumType; result: "
          << result.ToString(/*show_types=*/false);
  return result;
}

std::optional<int64_t> GetEnumMemberIndex(const EnumType& enum_type,
                                          const InterpValue& value,
                                          const ImportData& import_data) {
  const EnumDef& enum_def = enum_type.nominal_type();
  const TypeInfo& type_info =
      *import_data.GetRootTypeInfoForNode(&enum_def).value();
  for (int64_t i = 0; i < enum_def.values().size(); ++i) {
    const EnumMember& member = enum_def.values()[i];
    InterpValue member_val = type_info.GetConstExpr(member.name_def).value();
    if (member_val == value) {
      return i;
    }
  }
  return std::nullopt;
}

InterpValueInterval MakePointIntervalForEnumType(
    const EnumType& enum_type, const InterpValue& value,
    const ImportData& import_data) {
  CHECK(value.IsEnum())
      << "MakePointIntervalForEnumType; value is not an enum: "
      << value.ToString();
  int64_t bit_count = enum_type.size().GetAsInt64().value();
  // The `value` provided is the `i`th value in the dense enum space -- let's
  // determine that value `i`.
  int64_t member_index =
      GetEnumMemberIndex(enum_type, value, import_data).value();
  const InterpValue value_as_bits =
      InterpValue::MakeUBits(bit_count, member_index);
  VLOG(5) << "MakePointIntervalForEnumType; value_as_bits: "
          << value_as_bits.ToString() << " member_index: " << member_index;
  return InterpValueInterval(value_as_bits, value_as_bits);
}

}  // namespace xls::dslx
