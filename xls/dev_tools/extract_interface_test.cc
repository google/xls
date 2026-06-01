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

#include "xls/dev_tools/extract_interface.h"

#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/attribute_data.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/source_location.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {
using ::xls::proto_testing::EqualsProto;
using ::xls::proto_testing::IgnoringRepeatedFieldOrdering;
using ::xls::proto_testing::Partially;

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

chan sample__operand_0(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan sample__operand_1(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan sample__result(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid)

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

TEST_F(ExtractInterfaceTest, BasicProcWithSchedule) {
  constexpr std::string_view kIr = R"(
package sample

file_number 0 "fake_file.x"

chan sample__operand_0(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan sample__operand_1(bits[32], id=1, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan sample__result(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid)

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

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto schedule,
      RunPipelineSchedule(
          p->GetTop().value(), delay_estimator,
          SchedulingOptions()
              .pipeline_stages(2)
              .worst_case_throughput(2)
              .add_constraint(
                  IOConstraint("sample__operand_0", IODirection::kReceive,
                               "sample__result", IODirection::kSend,
                               /*minimum_latency=*/1, /*maximum_latency=*/1))
              .add_constraint(
                  IOConstraint("sample__operand_1", IODirection::kReceive,
                               "sample__result", IODirection::kSend,
                               /*minimum_latency=*/1, /*maximum_latency=*/1))));
  XLS_ASSERT_OK_AND_ASSIGN(auto proto, schedule.ToProto(delay_estimator));
  PackageScheduleProto package_schedule_proto;
  package_schedule_proto.mutable_schedules()->insert({"add", proto});
  auto interface = ExtractPackageInterface(p.get(), package_schedule_proto);
  RecordProperty("interface", interface.DebugString());
  EXPECT_THAT(
      interface,
      ProtoEquivalent(
          R"pb(
            name: "sample"
            files: "fake_file.x"
            # Also includes channels and procs but those are tested by the
            # BasicProc test.
            scheduled_procs {
              proc {
                base { top: true name: "add" }
                state {
                  name: "__token"
                  type { type_enum: TOKEN }
                }
                state_values {
                  name {
                    name: "__token"
                    type { type_enum: TOKEN }
                  }
                  non_synthesizable: false
                }
              }
              pipeline_info { pipeline_length: 2 initiation_interval: 2 }
              sends {
                stage: 1
                channel_name: "sample__result"
                is_blocking: false
              }
              recvs {
                stage: 0
                channel_name: "sample__operand_0"
                is_blocking: true
              }
              recvs {
                stage: 0
                channel_name: "sample__operand_1"
                is_blocking: true
              }
              state_reads { name: "__token" stage: 0 }
              state_writes { name: "__token" stage: 1 }
            }
          )pb"));
}

TEST_F(ExtractInterfaceTest, BasicBlock) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.AddClockPort("clk"));
  bb.OutputPort(
      "baz", bb.InsertRegister("foo", bb.InputPort("bar", p->GetBitsType(32))));
  XLS_ASSERT_OK(bb.Build().status());

  EXPECT_THAT(ExtractPackageInterface(p.get()),
              ProtoEquivalent(
                  R"pb(
                    name: "BasicBlock"
                    blocks {
                      base { top: false name: "BasicBlock" }
                      registers {
                        name: "foo"
                        type { type_enum: BITS bit_count: 32 }
                      }
                      input_ports {
                        name: "bar"
                        type { type_enum: BITS bit_count: 32 }
                      }
                      output_ports {
                        name: "baz"
                        type { type_enum: BITS bit_count: 32 }
                      }
                    }
                  )pb"));
}

TEST_F(ExtractInterfaceTest, FuzzTestFunction) {
  constexpr std::string_view kIr = R"(
package test

#[fuzz_test(domains = `parameter_domains { range { min { bits { bit_count: 32 data: "\000" } } max { bits { bit_count: 32 data: "\012" } } } } parameter_domains { arbitrary: true }`)]
fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(kIr));

  PackageInterfaceProto proto = ExtractPackageInterface(p.get());

  ASSERT_EQ(proto.functions().size(), 1);
  const auto& func_proto = proto.functions(0);

  ASSERT_EQ(func_proto.parameter_domains().size(), 2);
  EXPECT_TRUE(func_proto.parameter_domains(1).arbitrary());
  EXPECT_TRUE(func_proto.parameter_domains(0).has_range());
}

TEST_F(ExtractInterfaceTest, FuzzTestFunctionNoDomains) {
  constexpr std::string_view kIr = R"(
package test

#[fuzz_test]
fn f(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(kIr));

  PackageInterfaceProto proto = ExtractPackageInterface(p.get());

  ASSERT_EQ(proto.functions().size(), 1);
  const auto& func_proto = proto.functions(0);

  EXPECT_THAT(func_proto.parameter_domains(), testing::IsEmpty());
}

TEST_F(ExtractInterfaceTest, FuzzTestFunctionInvalidArgKeyManual) {
  VerifiedPackage p("test_package");
  Function* f;
  {
    FunctionBuilder fb("f", &p);
    fb.Param("x", p.GetBitsType(32));
    XLS_ASSERT_OK_AND_ASSIGN(f, fb.Build());
  }

  std::vector<AttributeData::Argument> args;
  args.push_back(AttributeData::StringKeyValueArgument{
      .first = "invalid", .second = "value", .is_backticked = true});
  f->AddAttribute(AttributeData(AttributeKind::kFuzzTest, std::move(args)));

  EXPECT_DEATH(ExtractPackageInterface(&p),
               "kFuzzTest only supports 'domains' argument");
}

TEST_F(ExtractInterfaceTest, PackageWithMixOfFuzzAndNonFuzzFunctions) {
  constexpr std::string_view kIr = R"(
package test

#[fuzz_test(domains = `parameter_domains { arbitrary: true }`)]
fn fuzz_me(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}

fn dont_fuzz_me(x: bits[32]) -> bits[32] {
  ret x: bits[32] = param(name=x)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(auto p, ParsePackage(kIr));

  PackageInterfaceProto proto = ExtractPackageInterface(p.get());

  ASSERT_EQ(proto.functions().size(), 2);

  const PackageInterfaceProto::Function* fuzz_proto = nullptr;
  const PackageInterfaceProto::Function* non_fuzz_proto = nullptr;

  for (const auto& f : proto.functions()) {
    if (f.base().name() == "fuzz_me") {
      fuzz_proto = &f;
    } else if (f.base().name() == "dont_fuzz_me") {
      non_fuzz_proto = &f;
    }
  }

  ASSERT_NE(fuzz_proto, nullptr);
  ASSERT_NE(non_fuzz_proto, nullptr);

  EXPECT_EQ(fuzz_proto->parameter_domains().size(), 1);
  EXPECT_TRUE(fuzz_proto->parameter_domains(0).arbitrary());

  EXPECT_TRUE(non_fuzz_proto->parameter_domains().empty());
}

}  // namespace
}  // namespace xls
