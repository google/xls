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
#include "xls/dslx/lsp/lsp_uri.h"

namespace xls::dslx {
namespace {

TEST(ImportSensitivityTest, SingleImport) {
  ImportSensitivity s;
  const auto outer_uri = LspUri("file://outer.x");
  const auto inner_uri = LspUri("file://inner.x");
  s.NoteImportAttempt(outer_uri, inner_uri);

  auto sensitive_to_outer = s.GatherAllSensitiveToChangeIn(outer_uri);
  ASSERT_EQ(sensitive_to_outer.size(), 1);
  EXPECT_THAT(sensitive_to_outer, testing::UnorderedElementsAre(outer_uri));

  auto sensitive_to_inner = s.GatherAllSensitiveToChangeIn(inner_uri);
  EXPECT_EQ(sensitive_to_inner.size(), 2);
  EXPECT_THAT(sensitive_to_inner,
              testing::UnorderedElementsAre(outer_uri, inner_uri));
}

TEST(ImportSensitivityTest, TwoLevelLinear) {
  ImportSensitivity s;
  const auto middle_uri = LspUri("file://middle.x");
  const auto inner_uri = LspUri("file://inner.x");
  const auto outer_uri = LspUri("file://outer.x");
  s.NoteImportAttempt(middle_uri, inner_uri);
  s.NoteImportAttempt(outer_uri, middle_uri);

  auto sensitive_to_inner = s.GatherAllSensitiveToChangeIn(inner_uri);
  EXPECT_EQ(sensitive_to_inner.size(), 3);
  EXPECT_THAT(sensitive_to_inner,
              testing::UnorderedElementsAre(outer_uri, middle_uri, inner_uri));
}

TEST(ImportSensitivityTest, Diamond) {
  ImportSensitivity s;
  const auto leaf_uri = LspUri("file://leaf.x");
  const auto middle0_uri = LspUri("file://middle0.x");
  const auto middle1_uri = LspUri("file://middle1.x");
  const auto outer_uri = LspUri("file://outer.x");
  s.NoteImportAttempt(middle0_uri, leaf_uri);
  s.NoteImportAttempt(middle1_uri, leaf_uri);
  s.NoteImportAttempt(outer_uri, middle0_uri);
  s.NoteImportAttempt(outer_uri, middle1_uri);

  auto sensitive_to_inner = s.GatherAllSensitiveToChangeIn(leaf_uri);
  EXPECT_EQ(sensitive_to_inner.size(), 4);
  EXPECT_THAT(sensitive_to_inner,
              testing::UnorderedElementsAre(outer_uri, middle0_uri, middle1_uri,
                                            leaf_uri));
}

}  // namespace
}  // namespace xls::dslx
