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
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/proc_test_utils.h"

namespace xls::dslx {
namespace {

class ProcIdTest : public ::testing::Test {
 protected:
  void SetUp() final {
    auto [foo_module, foo_proc] = CreateEmptyProc(file_table_, "Foo");
    auto [bar_module, bar_proc] = CreateEmptyProc(file_table_, "Bar");
    auto [baz_module, baz_proc] = CreateEmptyProc(file_table_, "Baz");
    foo_ = foo_proc;
    bar_ = bar_proc;
    baz_ = baz_proc;
    modules_.push_back(std::move(foo_module));
    modules_.push_back(std::move(bar_module));
    modules_.push_back(std::move(baz_module));
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
