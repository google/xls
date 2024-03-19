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

#include "xls/ir/value_builder.h"

#include <cstdint>
#include <iterator>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {

namespace {
absl::StatusOr<std::vector<Value>> BuildAll(
    absl::Span<ValueBuilder::MaybeValueBuilder const> src) {
  std::vector<Value> vals;
  absl::Status result = absl::OkStatus();
  absl::c_transform(src, std::back_inserter(vals),
                    [&](const ValueBuilder::MaybeValueBuilder& m) -> Value {
                      return absl::visit(
                          Visitor{
                              [&](const ValueBuilder& pv) -> Value {
                                auto v = pv.Build();
                                if (!v.ok()) {
                                  result.Update(v.status());
                                }
                                return v.value_or(Value());
                              },
                              [](const Value& v) -> Value { return v; },
                          },
                          m);
                    });

  XLS_RETURN_IF_ERROR(result);
  return vals;
}
}  // namespace

/* static */ ValueBuilder ValueBuilder::Array(
    absl::Span<ValueBuilder::MaybeValueBuilder const> elements) {
  return ValueBuilder(ArrayHolder{
      .v = std::vector<MaybeValueBuilder>(elements.begin(), elements.end())});
}

/* static */ ValueBuilder ValueBuilder::ArrayB(
    absl::Span<ValueBuilder const> elements) {
  return ValueBuilder(ArrayHolder{
      .v = std::vector<MaybeValueBuilder>(elements.begin(), elements.end())});
}

/* static */ ValueBuilder ValueBuilder::ArrayV(
    absl::Span<Value const> elements) {
  return ValueBuilder(ArrayHolder{
      .v = std::vector<MaybeValueBuilder>(elements.begin(), elements.end())});
}

/* static */ ValueBuilder ValueBuilder::Tuple(
    absl::Span<ValueBuilder::MaybeValueBuilder const> elements) {
  return ValueBuilder(TupleHolder{
      .v = std::vector<MaybeValueBuilder>(elements.begin(), elements.end())});
}

/* static */ ValueBuilder ValueBuilder::TupleB(
    absl::Span<ValueBuilder const> elements) {
  return ValueBuilder(TupleHolder{
      .v = std::vector<MaybeValueBuilder>(elements.begin(), elements.end())});
}

/* static */ ValueBuilder ValueBuilder::TupleV(
    absl::Span<Value const> elements) {
  return ValueBuilder(TupleHolder{
      .v = std::vector<MaybeValueBuilder>(elements.begin(), elements.end())});
}

/* static */ ValueBuilder ValueBuilder::UBitsArray(
    absl::Span<uint64_t const> elements, int64_t bit_count) {
  std::vector<Value> vs;
  vs.reserve(elements.size());
  absl::c_transform(elements, std::back_inserter(vs),
                    [&](uint64_t v) { return Value(UBits(v, bit_count)); });
  return ValueBuilder::ArrayV(vs);
}

/* static */ ValueBuilder ValueBuilder::UBits2DArray(
    absl::Span<const absl::Span<const uint64_t>> elements, int64_t bit_count) {
  std::vector<ValueBuilder> vs;
  vs.reserve(elements.size());
  absl::c_transform(elements, std::back_inserter(vs),
                    [&](absl::Span<uint64_t const> v) {
                      return ValueBuilder::UBitsArray(v, bit_count);
                    });
  return ValueBuilder::ArrayB(vs);
}

/* static */ ValueBuilder ValueBuilder::SBitsArray(
    absl::Span<const int64_t> elements, int64_t bit_count) {
  std::vector<Value> vs;
  vs.reserve(elements.size());
  absl::c_transform(elements, std::back_inserter(vs),
                    [&](int64_t v) { return Value(SBits(v, bit_count)); });
  return ValueBuilder::ArrayV(vs);
}

/* static */ ValueBuilder ValueBuilder::SBits2DArray(
    absl::Span<const absl::Span<const int64_t>> elements, int64_t bit_count) {
  std::vector<ValueBuilder> vs;
  vs.reserve(elements.size());
  absl::c_transform(elements, std::back_inserter(vs),
                    [&](absl::Span<int64_t const> v) {
                      return ValueBuilder::SBitsArray(v, bit_count);
                    });
  return ValueBuilder::ArrayB(vs);
}

absl::StatusOr<Value> ValueBuilder::Build() const {
  if (std::holds_alternative<Value>(my_value_)) {
    return std::get<Value>(my_value_);
  }
  if (IsArray()) {
    XLS_ASSIGN_OR_RETURN(auto vals,
                         BuildAll(std::get<ArrayHolder>(my_value_).v));
    return Value::Array(vals);
  }
  XLS_RET_CHECK(IsTuple());
  XLS_ASSIGN_OR_RETURN(auto vals, BuildAll(std::get<TupleHolder>(my_value_).v));
  return Value::Tuple(vals);
}

}  // namespace xls
