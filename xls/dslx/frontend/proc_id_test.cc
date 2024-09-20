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

#include "xls/dslx/frontend/proc_id.h"

#include <list>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/scanner.h"

namespace xls::dslx {
namespace {

class ProcIdTest : public ::testing::Test {
 protected:
  Proc* CreateEmptyProc(std::string_view name) {
    const std::string_view code_template = R"(proc %s {
    config() { () }
    init { () }
    next(state: ()) { () }
})";
    Scanner s{file_table_, Fileno(0), absl::StrFormat(code_template, name)};
    Parser parser{"test", &s};
    Bindings bindings;
    absl::StatusOr<Proc*> proc =
        parser.ParseProc(/*is_public=*/false, bindings);
    CHECK(proc.ok());
    modules_.emplace_back(std::move(parser.module()));
    return *proc;
  }

  void SetUp() override {
    foo_ = CreateEmptyProc("Foo");
    bar_ = CreateEmptyProc("Bar");
    baz_ = CreateEmptyProc("Baz");
  }

  FileTable file_table_;
  std::list<Module> modules_;
  Proc* foo_;
  Proc* bar_;
  Proc* baz_;
  ProcIdFactory factory_;
};

TEST_F(ProcIdTest, CreateProcIdNoParent) {
  const ProcId root = factory_.CreateProcId(/*parent=*/std::nullopt, foo_);
  EXPECT_EQ(root.ToString(), "Foo:0");
  EXPECT_FALSE(factory_.HasMultipleInstancesOfAnyProc());
}

TEST_F(ProcIdTest, CreateProcIdWithParent) {
  const ProcId root = factory_.CreateProcId(/*parent=*/std::nullopt, foo_);
  EXPECT_EQ(factory_.CreateProcId(root, bar_).ToString(), "Foo->Bar:0");
  EXPECT_FALSE(factory_.HasMultipleInstancesOfAnyProc());
}

TEST_F(ProcIdTest, MultiInstance) {
  const ProcId root = factory_.CreateProcId(/*parent=*/std::nullopt, foo_);
  EXPECT_EQ(factory_.CreateProcId(root, bar_).ToString(), "Foo->Bar:0");
  EXPECT_EQ(factory_.CreateProcId(root, bar_).ToString(), "Foo->Bar:1");
  EXPECT_TRUE(factory_.HasMultipleInstancesOfAnyProc());
}

TEST_F(ProcIdTest, MultiLevel) {
  const ProcId root = factory_.CreateProcId(/*parent=*/std::nullopt, foo_);
  const ProcId bar0 = factory_.CreateProcId(root, bar_);
  const ProcId bar1 = factory_.CreateProcId(root, bar_);
  EXPECT_EQ(factory_.CreateProcId(root, baz_).ToString(), "Foo->Baz:0");
  EXPECT_EQ(factory_.CreateProcId(bar0, baz_).ToString(), "Foo->Bar:0->Baz:0");
  EXPECT_EQ(factory_.CreateProcId(bar1, baz_).ToString(), "Foo->Bar:1->Baz:0");
  EXPECT_EQ(factory_.CreateProcId(bar1, baz_).ToString(), "Foo->Bar:1->Baz:1");
  EXPECT_EQ(factory_.CreateProcId(bar1, baz_).ToString(), "Foo->Bar:1->Baz:2");
  EXPECT_EQ(factory_.CreateProcId(root, baz_).ToString(), "Foo->Baz:1");
}

TEST_F(ProcIdTest, SharedInstance) {
  // This mimics the pattern of using the same instance number for next() as the
  // preceding config() for the same proc.
  const ProcId root = factory_.CreateProcId(/*parent=*/std::nullopt, foo_);
  EXPECT_EQ(factory_.CreateProcId(root, bar_, /*count_as_new_instance=*/false)
                .ToString(),
            "Foo->Bar:0");
  EXPECT_EQ(factory_.CreateProcId(root, bar_).ToString(), "Foo->Bar:0");
  EXPECT_FALSE(factory_.HasMultipleInstancesOfAnyProc());
  EXPECT_EQ(factory_.CreateProcId(root, bar_).ToString(), "Foo->Bar:1");
  EXPECT_TRUE(factory_.HasMultipleInstancesOfAnyProc());
}

}  // namespace
}  // namespace xls::dslx
