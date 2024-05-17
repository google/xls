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

#include "xls/dslx/interp_bindings.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

TEST(InterpBindingsTest, EmptyInstance) {
  InterpBindings bindings;
  (void)bindings;
}

TEST(InterpBindingsTest, AddResolveValue) {
  InterpBindings bindings;
  bindings.AddValue("t", InterpValue::MakeBool(true));
  bindings.AddValue("f", InterpValue::MakeBool(false));

  bindings.ResolveValueFromIdentifier("t").value().IsTrue();
  bindings.ResolveValueFromIdentifier("f").value().IsFalse();
}

TEST(InterpBindingsTest, ResolveValueViaParentLnk) {
  InterpBindings parent;
  parent.AddValue("t", InterpValue::MakeBool(true));

  InterpBindings child(&parent);
  child.ResolveValueFromIdentifier("t").value().IsTrue();
  EXPECT_THAT(child.ResolveModule("t"),
              status_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("identifier \"t\" was bound to a Value")));
}

TEST(InterpBindingsTest, GetKeysWithParentLink) {
  InterpBindings parent;
  parent.AddValue("p", InterpValue::MakeU32(42));

  InterpBindings child(&parent);
  child.AddValue("c", InterpValue::MakeU32(64));

  EXPECT_THAT(child.GetKeys(), testing::UnorderedElementsAre("c", "p"));
}

}  // namespace
}  // namespace xls::dslx
