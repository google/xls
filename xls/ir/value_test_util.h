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

#ifndef XLS_IR_VALUE_TEST_UTIL_H_
#define XLS_IR_VALUE_TEST_UTIL_H_

#include <cstdint>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/types/span.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/ir/xls_type.pb.h"

namespace xls {

using ElementOfProto = PackageInterfaceProto::FuzzTestDomain::ElementOf;

// Returns an assertion result indicating whether the given two values were
// equal. If equal the return value is AssertionSuccess, otherwise
// AssertionFailure. For large Values (arrays, tuples, and very wide bit widths)
// this method can give a more readable message than EXPECT_EQ(a, b) by
// highlighting the differences between the two Values rather than just dumping
// the string representation of the Values.
::testing::AssertionResult ValuesEqual(const Value& a, const Value& b);

// Create an arbitrary `Value` domain of the given type.
fuzztest::Domain<Value> ArbitraryValue(int64_t max_bit_count = 64,
                                       int64_t max_elements = 5,
                                       int64_t max_nesting_level = 5);

// Create a domain for an arbitrary value which is of the given type.
fuzztest::Domain<Value> ArbitraryValue(fuzztest::Domain<TypeProto> type);

// Create a domain for an arbitrary value which is of the given type.
fuzztest::Domain<Value> ArbitraryValue(TypeProto type);

// Parses the binary-format serialized proto of type ElementOfProto and returns
// the proto.
ElementOfProto ParseElementOfProto(absl::Span<const uint8_t> bytes);

// Create an element_of domain from a serialized ElementOf proto.
fuzztest::Domain<Value> ElementOfDomain(ElementOfProto proto);

}  // namespace xls

#endif  // XLS_IR_VALUE_TEST_UTIL_H_
