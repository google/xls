// Copyright 2026 The XLS Authors
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

#include "xls/spin/trace_compare.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/ir_parser.h"

namespace xls::spin {
namespace {

using ::testing::ElementsAre;
constexpr std::string_view kTerminatorChannel = "terminator";

testing::AssertionResult CompareMaps(const TraceMap& spin,
                                     const TraceMap& dslx) {
  auto status = CompareTraces(spin, dslx);
  if (!status.ok()) {
    return testing::AssertionFailure() << status.message();
  }
  return testing::AssertionSuccess();
}

TEST(TraceCompareTest, ParseSpinTrace_FullPath) {
  // With proctype+pid in the JSON and a matching proc_paths entry, channel
  // keys are the full DSLX-style path.
  constexpr std::string_view kJson = R"(
{"channel_name":"_req","direction":"SEND","value":1,"proctype":"__foo__Parent_0_next","pid":2}
{"channel_name":"_resp","direction":"RECV","value":42,"proctype":"__foo__Parent_0_next","pid":2}
{"channel_name":"_req","direction":"SEND","value":2,"proctype":"__foo__Parent_0_next","pid":2}
)";
  ProcInstPaths proc_paths;
  proc_paths["__foo__Parent_0_next"] = {"Parent"};
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out, ParseSpinTrace(kJson, proc_paths));
  EXPECT_THAT((out[{"Parent::req", Direction::kSend}]), ElementsAre(1, 2));
  EXPECT_THAT((out[{"Parent::resp", Direction::kRecv}]), ElementsAre(42));
}

TEST(TraceCompareTest, ParseSpinTrace_MultiPid_MapsToInstances) {
  // Two events from different pids of the same proctype map to instance#0 and
  // instance#1 paths in the order the pids are first seen.
  constexpr std::string_view kJson = R"(
{"channel_name":"_ch","direction":"SEND","value":10,"proctype":"__foo__Worker_0_next","pid":5}
{"channel_name":"_ch","direction":"SEND","value":20,"proctype":"__foo__Worker_0_next","pid":7}
{"channel_name":"_ch","direction":"SEND","value":30,"proctype":"__foo__Worker_0_next","pid":5}
)";
  ProcInstPaths proc_paths;
  proc_paths["__foo__Worker_0_next"] = {"Top->Worker#0", "Top->Worker#1"};
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out, ParseSpinTrace(kJson, proc_paths));
  EXPECT_THAT((out[{"Top->Worker#0::ch", Direction::kSend}]), ElementsAre(10, 30));
  EXPECT_THAT((out[{"Top->Worker#1::ch", Direction::kSend}]), ElementsAre(20));
}

TEST(TraceCompareTest, ParseSpinTrace_FallbackToBareOnUnknownProctype) {
  // When proctype is absent or not in proc_paths, falls back to bare name.
  constexpr std::string_view kJson = R"(
{"channel_name":"_req","direction":"SEND","value":1}
{"channel_name":"_req","direction":"SEND","value":2,"proctype":"__unknown__P_0_next","pid":0}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out, ParseSpinTrace(kJson, {}));
  EXPECT_THAT((out[{"req", Direction::kSend}]), ElementsAre(1, 2));
}

TEST(TraceCompareTest, ParseSpinTrace_NegativeValueReinterpret) {
  // SPIN's 32-bit signed -1 must be reinterpreted as uint32 (4294967295).
  constexpr std::string_view kJson = R"(
{"channel_name":"_ch","direction":"SEND","value":-1,"proctype":"__foo__P_0_next","pid":0}
)";
  ProcInstPaths proc_paths;
  proc_paths["__foo__P_0_next"] = {"P"};
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out, ParseSpinTrace(kJson, proc_paths));
  EXPECT_THAT((out[{"P::ch", Direction::kSend}]), ElementsAre(4294967295LL));
}

TEST(TraceCompareTest, ParseSpinTrace_Terminator_Truncates) {
  constexpr std::string_view kJson = R"(
{"channel_name":"_data","direction":"SEND","value":1}
{"channel_name":"_terminator","direction":"SEND","value":1}
{"channel_name":"_data","direction":"SEND","value":2}
)";
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out,
                           ParseSpinTrace(kJson, {}, kTerminatorChannel));
  EXPECT_THAT((out[{"data", Direction::kSend}]), ElementsAre(1));
  EXPECT_EQ(out.count({"terminator", Direction::kSend}), 0);
}

TEST(TraceCompareTest, SpinTraceHasTerminator_Set) {
  constexpr std::string_view kJson = R"(
{"channel_name":"_terminator","direction":"SEND","value":1}
)";
  EXPECT_TRUE(SpinTraceHasTerminator(kJson, kTerminatorChannel));
}

TEST(TraceCompareTest, SpinTraceHasTerminator_NotSet) {
  constexpr std::string_view kJson = R"(
{"channel_name":"_data","direction":"SEND","value":1}
)";
  EXPECT_FALSE(SpinTraceHasTerminator(kJson, kTerminatorChannel));
}

TEST(TraceCompareTest, BuildProcInstPathsForSpin_Hierarchy) {
  constexpr std::string_view kIr = R"(package foo
proc __foo__Child_0_next<_ch: bits[32] in>(__tkn: token, init={token}) {
  chan_interface _ch(direction=receive, kind=streaming,
                     strictness=proven_mutually_exclusive,
                     flow_control=ready_valid, flop_kind=none)
  __tkn: token = state_read(state_element=__tkn, id=1)
  next_value.2: () = next_value(param=__tkn, value=__tkn, id=2)
}
top proc __foo__Parent_0_next<>(__tkn: token, init={token}) {
  chan _ch(bits[32], id=0, kind=streaming, ops=send_receive,
           flow_control=ready_valid,
           strictness=proven_mutually_exclusive)
  chan_interface _ch(direction=send, kind=streaming,
                     strictness=proven_mutually_exclusive,
                     flow_control=none, flop_kind=none)
  chan_interface _ch(direction=receive, kind=streaming,
                     strictness=proven_mutually_exclusive,
                     flow_control=none, flop_kind=none)
  proc_instantiation child_inst(_ch, proc=__foo__Child_0_next)
  __tkn: token = state_read(state_element=__tkn, id=3)
  next_value.4: () = next_value(param=__tkn, value=__tkn, id=4)
})";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kIr));
  XLS_ASSERT_OK_AND_ASSIGN(ProcInstPaths paths,
                           BuildProcInstPathsForSpin(package.get()));
  EXPECT_THAT(paths.at("__foo__Parent_0_next"), ElementsAre("Parent"));
  EXPECT_THAT(paths.at("__foo__Child_0_next"), ElementsAre("Parent->Child#0"));
}

TEST(TraceCompareTest, CompareMaps_Match) {
  TraceMap spin, dslx;
  spin[{"req", Direction::kSend}] = {1, 2, 3};
  dslx[{"req", Direction::kSend}] = {1, 2, 3};
  EXPECT_TRUE(CompareMaps(spin, dslx));
}

TEST(TraceCompareTest, CompareMaps_Mismatch) {
  TraceMap spin, dslx;
  spin[{"req", Direction::kSend}] = {1, 2, 3};
  dslx[{"req", Direction::kSend}] = {1, 2, 99};
  EXPECT_FALSE(CompareMaps(spin, dslx));
}

TEST(TraceCompareTest, CompareMaps_AsymmetricKeys) {
  TraceMap spin, dslx;
  spin[{"req", Direction::kSend}] = {1};
  dslx[{"resp", Direction::kRecv}] = {1};
  EXPECT_FALSE(CompareMaps(spin, dslx));
}

// Textproto helpers for ParseDslxTrace tests.
constexpr std::string_view kSingleEventProto = R"pb(
  results {
    events {
      trace_msgs { channel { channel_name: "FooTest->Sub#0::req_s" direction: SEND } }
    }
  }
)pb";

constexpr std::string_view kTerminatorProto = R"pb(
  results {
    events {
      trace_msgs { channel { channel_name: "FooTest::data" direction: SEND } }
      trace_msgs { channel { channel_name: "FooTest::terminator" direction: SEND } }
      trace_msgs { channel { channel_name: "FooTest::data" direction: SEND } }
    }
  }
)pb";

constexpr std::string_view kTwoInstancesProto = R"pb(
  results {
    events {
      trace_msgs { channel { channel_name: "ParentTest->Sub#0::req_r" direction: SEND } }
      trace_msgs { channel { channel_name: "ParentTest->Sub#1::req_r" direction: SEND } }
    }
  }
)pb";

TEST(TraceCompareTest, ParseDslxTrace_NoMap_StoresFullPath) {
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out, ParseDslxTrace(kSingleEventProto));
  EXPECT_THAT((out[{"FooTest->Sub#0::req_s", Direction::kSend}]), ElementsAre(0));
}

TEST(TraceCompareTest, ParseDslxTrace_MapRewrites_VarNameToChannelDecl) {
  DslxChannelNameMap map;
  map[std::make_tuple(std::string("Sub"), int64_t{0}, std::string("req_s"))] =
      "req";
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out,
                           ParseDslxTrace(kSingleEventProto, "", map));
  EXPECT_THAT((out[{"FooTest->Sub#0::req", Direction::kSend}]), ElementsAre(0));
  EXPECT_EQ(out.count({"FooTest->Sub#0::req_s", Direction::kSend}), 0);
}

TEST(TraceCompareTest, ParseDslxTrace_MapRewrites_PerInstance) {
  // Two instances of the same proc type with different channel bindings must
  // each map to their own ChannelDecl string.
  DslxChannelNameMap map;
  map[std::make_tuple(std::string("Sub"), int64_t{0}, std::string("req_r"))] =
      "ch_a";
  map[std::make_tuple(std::string("Sub"), int64_t{1}, std::string("req_r"))] =
      "ch_b";
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out,
                           ParseDslxTrace(kTwoInstancesProto, "", map));
  EXPECT_THAT((out[{"ParentTest->Sub#0::ch_a", Direction::kSend}]), ElementsAre(0));
  EXPECT_THAT((out[{"ParentTest->Sub#1::ch_b", Direction::kSend}]), ElementsAre(0));
}

TEST(TraceCompareTest, ParseDslxTrace_Terminator_TruncatesTrace) {
  // Events after the terminator SEND must not appear in the output.
  XLS_ASSERT_OK_AND_ASSIGN(TraceMap out,
                           ParseDslxTrace(kTerminatorProto, kTerminatorChannel));
  EXPECT_THAT((out[{"FooTest::data", Direction::kSend}]), ElementsAre(0));
  EXPECT_EQ(out.count({"FooTest::terminator", Direction::kSend}), 0);
}

}  // namespace
}  // namespace xls::spin
