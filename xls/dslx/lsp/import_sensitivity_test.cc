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

#include "xls/dslx/lsp/import_sensitivity.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

TEST(ImportSensitivityTest, SingleImport) {
  ImportSensitivity s;
  s.NoteImportAttempt("outer.x", "inner.x");

  auto sensitive_to_outer = s.GatherAllSensitiveToChangeIn("outer.x");
  ASSERT_EQ(sensitive_to_outer.size(), 1);
  EXPECT_THAT(sensitive_to_outer, testing::UnorderedElementsAre("outer.x"));

  auto sensitive_to_inner = s.GatherAllSensitiveToChangeIn("inner.x");
  EXPECT_EQ(sensitive_to_inner.size(), 2);
  EXPECT_THAT(sensitive_to_inner,
              testing::UnorderedElementsAre("outer.x", "inner.x"));
}

TEST(ImportSensitivityTest, TwoLevelLinear) {
  ImportSensitivity s;
  s.NoteImportAttempt("middle.x", "inner.x");
  s.NoteImportAttempt("outer.x", "middle.x");

  auto sensitive_to_inner = s.GatherAllSensitiveToChangeIn("inner.x");
  EXPECT_EQ(sensitive_to_inner.size(), 3);
  EXPECT_THAT(sensitive_to_inner,
              testing::UnorderedElementsAre("outer.x", "middle.x", "inner.x"));
}

TEST(ImportSensitivityTest, Diamond) {
  ImportSensitivity s;
  s.NoteImportAttempt("middle0.x", "leaf.x");
  s.NoteImportAttempt("middle1.x", "leaf.x");
  s.NoteImportAttempt("outer.x", "middle0.x");
  s.NoteImportAttempt("outer.x", "middle1.x");

  auto sensitive_to_inner = s.GatherAllSensitiveToChangeIn("leaf.x");
  EXPECT_EQ(sensitive_to_inner.size(), 4);
  EXPECT_THAT(sensitive_to_inner,
              testing::UnorderedElementsAre("outer.x", "middle0.x", "middle1.x",
                                            "leaf.x"));
}

}  // namespace
}  // namespace xls::dslx
