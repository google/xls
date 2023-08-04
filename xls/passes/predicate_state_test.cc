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

#include "xls/passes/predicate_state.h"

#include "gtest/gtest.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"

namespace xls {

namespace {

TEST(PredicateState, Hash) {
  EXPECT_NE(absl::Hash<PredicateState>()(PredicateState()),
            absl::Hash<PredicateState>()(PredicateState(nullptr, 33)));
}

TEST(PredicateState, Stringify) {
  Package p("test");
  FunctionBuilder fb("test", &p);
  BValue sel = fb.Select(
      fb.Param("w", p.GetBitsType(2)),
      {fb.Param("x", p.GetBitsType(12)), fb.Param("y", p.GetBitsType(12))},
      fb.Param("z", p.GetBitsType(12)));
  EXPECT_EQ(
      absl::StrFormat("%v", PredicateState(sel.node()->As<Select>(), 0)),
      absl::StrFormat("PredicateState[sel.%d: arm: 0]", sel.node()->id()));
  EXPECT_EQ(absl::StrFormat("%v", PredicateState(sel.node()->As<Select>(),
                                                 PredicateState::kDefaultArm)),
            absl::StrFormat("PredicateState[sel.%d: arm: DEFAULT]",
                            sel.node()->id()));
}

}  // namespace
}  // namespace xls
