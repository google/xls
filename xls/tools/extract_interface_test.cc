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

#include "xls/tools/extract_interface.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/source_location.h"
#include "xls/ir/xls_ir_interface.pb.h"

namespace xls {
namespace {
using proto_testing::EqualsProto;
using proto_testing::IgnoringRepeatedFieldOrdering;
using proto_testing::Partially;

auto ProtoEquivalent(auto inner) {
  return Partially(IgnoringRepeatedFieldOrdering(EqualsProto(inner)));
}
class ExtractInterfaceTest : public IrTestBase {};

TEST_F(ExtractInterfaceTest, BasicFunction) {
  VerifiedPackage p("add_test");
  auto sl = p.AddSourceLocation("test_file", Lineno(0), Colno(0));
  {
    FunctionBuilder fb("add", &p);
    XLS_ASSERT_OK(fb.SetAsTop());
    fb.Add(fb.Param("l", p.GetBitsType(32)), fb.Param("r", p.GetBitsType(32)),
           SourceInfo(sl));
    XLS_ASSERT_OK(fb.Build().status());
  }
  {
    FunctionBuilder fb("cooler_add", &p);
    fb.Add(fb.Param("cool", p.GetBitsType(64)), fb.Literal(UBits(42, 64)));
    XLS_ASSERT_OK(fb.Build().status());
  }

  EXPECT_THAT(ExtractPackageInterface(&p), ProtoEquivalent(R"pb(
                name: "add_test"
                files: "test_file"
                functions {
                  base { top: true name: "add" }
                  parameters {
                    name: "l"
                    type { type_enum: BITS bit_count: 32 }
                  }
                  parameters {
                    name: "r"
                    type { type_enum: BITS bit_count: 32 }
                  }
                  result_type { type_enum: BITS bit_count: 32 }
                }
                functions {
                  base { top: false name: "cooler_add" }
                  parameters {
                    name: "cool"
                    type { type_enum: BITS bit_count: 64 }
                  }
                  result_type { type_enum: BITS bit_count: 64 }
                }
              )pb"));
}

TEST_F(ExtractInterfaceTest, BasicProc) {
  constexpr std::string_view kIr = R"(
package sample

file_number 0 "fake_file.x"

chan sample__operand_0(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan sample__operand_1(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan sample__result(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc add(__token: token, init={token}) {
  receive.4: (token, bits[32]) = receive(__token, channel=sample__operand_0, id=4)
  receive.7: (token, bits[32]) = receive(__token, channel=sample__operand_1, id=7)
  tok_operand_0_val: token = tuple_index(receive.4, index=0, id=5, pos=[(0,14,9)])
  tok_operand_1_val: token = tuple_index(receive.7, index=0, id=8, pos=[(0,15,9)])
  operand_0_val: bits[32] = tuple_index(receive.4, index=1, id=6, pos=[(0,14,28)])
  operand_1_val: bits[32] = tuple_index(receive.7, index=1, id=9, pos=[(0,15,28)])
  tok_recv: token = after_all(tok_operand_0_val, tok_operand_1_val, id=10)
  result_val: bits[32] = add(operand_0_val, operand_1_val, id=11, pos=[(0,18,35)])
  tok_send: token = send(tok_recv, result_val, channel=sample__result, id=12)
  after_all.14: token = after_all(__token, tok_operand_0_val, tok_operand_1_val, tok_recv, tok_send, id=14)
  next (after_all.14)
})";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(kIr));

  EXPECT_THAT(ExtractPackageInterface(p.get()),
              ProtoEquivalent(
                  R"pb(
                    name: "sample"
                    files: "fake_file.x"
                    channels {
                      name: "sample__operand_0"
                      type { type_enum: BITS bit_count: 32 }
                      direction: IN
                    }
                    channels {
                      name: "sample__operand_1"
                      type { type_enum: BITS bit_count: 32 }
                      direction: IN
                    }
                    channels {
                      name: "sample__result"
                      type { type_enum: BITS bit_count: 32 }
                      direction: OUT
                    }
                    procs {
                      base { top: true name: "add" }
                      state {
                        name: "__token"
                        type { type_enum: TOKEN }
                      }
                    }
                  )pb"));
}

}  // namespace
}  // namespace xls
