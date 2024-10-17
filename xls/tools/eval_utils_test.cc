// Copyright 2023 The XLS Authors
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

#include "xls/tools/eval_utils.h"

#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/btree_map.h"
#include "absl/status/status_matchers.h"
#include "google/protobuf/text_format.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"
#include "xls/tools/proc_channel_values.pb.h"

namespace xls {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::xls::proto_testing::EqualsProto;

TEST(EvalHelpersTest, ParseChannelValuesFromProto) {
  std::string_view proto = R"pb(
    channels {
      name: "foo",
      entry {
        bits: {
          bit_count: 24,
          data: "ABC",
        }
      }
      entry {
        bits: {
          bit_count: 64,
          data: "ABCDEFGH",
        }
      }
      entry {
        tuple: {
          elements: {
            bits: {
              bit_count: 64,
              data: "ABCDEFGH",
            }
          }
          elements: {
            bits: {
              bit_count: 8,
              data: "A",
            }
          }
        }
      }
    }
  )pb";
  ProcChannelValuesProto pcv;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto, &pcv));

  XLS_ASSERT_OK_AND_ASSIGN(auto result, ParseChannelValuesFromProto(pcv));
  // NB 0x41 is 'A'.
  // NB Little endian so early bytes appear at the end
  EXPECT_THAT(
      result,
      UnorderedElementsAre(Pair(
          "foo", ElementsAre(Value(UBits(0x434241, 24)),
                             Value(UBits(0x4847464544434241, 64)),
                             Value::Tuple({Value(UBits(0x4847464544434241, 64)),
                                           Value(UBits(0x41, 8))})))));
}

TEST(EvalHelpersTest, ChannelValuesToProto) {
  std::string_view proto = R"pb(
    channels {
      name: "foo",
      entry {
        bits: {
          bit_count: 24,
          data: "ABC",
        }
      }
      entry {
        bits: {
          bit_count: 64,
          data: "ABCDEFGH",
        }
      }
      entry {
        tuple: {
          elements: {
            bits: {
              bit_count: 64,
              data: "ABCDEFGH",
            }
          }
          elements: {
            bits: {
              bit_count: 8,
              data: "A",
            }
          }
        }
      }
    }
  )pb";
  ProcChannelValuesProto expected;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(proto, &expected));
  // NB 0x41 is 'A'.
  // NB Little endian so low bytes appear first.
  absl::btree_map<std::string, std::vector<Value>> input{
      {"foo",
       {Value(UBits(0x434241, 24)), Value(UBits(0x4847464544434241, 64)),
        Value::Tuple(
            {Value(UBits(0x4847464544434241, 64)), Value(UBits(0x41, 8))})}}};
  EXPECT_THAT(ChannelValuesToProto(input), IsOkAndHolds(EqualsProto(expected)));
}

}  // namespace
}  // namespace xls
