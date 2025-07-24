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

#include "xls/ir/value_test_util.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/fuzz_type_domain.h"
#include "xls/ir/type.h"
#include "xls/ir/type_manager.h"
#include "xls/ir/value.h"
#include "xls/ir/value_flattening.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

::testing::AssertionResult ValuesEqual(const Value& a, const Value& b) {
  if (a != b) {
    return testing::AssertionFailure()
           << a.ToString() << " != " << b.ToString();
  }
  return testing::AssertionSuccess();
}

fuzztest::Domain<Value> ArbitraryValue(fuzztest::Domain<TypeProto> type) {
  return fuzztest::FlatMap(
      [](TypeProto ty_proto) -> fuzztest::Domain<Value> {
        TypeManager man;
        auto ty_ptr = man.GetTypeFromProto(ty_proto);
        CHECK_OK(ty_ptr.status())
            << "Unable to parse type info from " << ty_proto.DebugString();
        Type* ty = ty_ptr.value();

        return fuzztest::Map(
            [](TypeProto ty_proto, Bits raw_data) -> Value {
              TypeManager man;
              auto ty_ptr = man.GetTypeFromProto(ty_proto);
              CHECK_OK(ty_ptr.status()) << "Unable to parse type info from "
                                        << ty_proto.DebugString();
              Type* ty = ty_ptr.value();
              auto res = UnflattenBitsToValue(raw_data, ty);
              CHECK_OK(res) << "Failed to unflatten "
                            << raw_data.ToDebugString() << " into " << ty;
              return res.value();
            },
            fuzztest::Just(ty_proto), ArbitraryBits(ty->GetFlatBitCount()));
      },
      type);
}

fuzztest::Domain<Value> ArbitraryValue(int64_t max_bit_count,
                                       int64_t max_elements,
                                       int64_t max_nesting_level) {
  return ArbitraryValue(
      TypeDomain(max_bit_count, max_elements, max_nesting_level));
}

fuzztest::Domain<Value> ArbitraryValue(TypeProto type) {
  return ArbitraryValue(fuzztest::Just(type));
}

}  // namespace xls
