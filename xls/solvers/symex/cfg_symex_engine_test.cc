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

#include "xls/solvers/symex/cfg_symex_engine.h"

#include "gtest/gtest.h"
#include "xls/solvers/symex/symex_engine.h"
#include "z3/src/api/z3.h"

namespace xls::solvers::symex {
namespace {

class CfgSymExEngineTest : public ::testing::Test {
 protected:
  void SetUp() override {
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
  }

  void TearDown() override {
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(CfgSymExEngineTest, InitializesWithDefaultOptions) {
  CfgSymExEngine engine(ctx_);
  EXPECT_EQ(engine.total_explored_paths(), 0);
  EXPECT_EQ(engine.feasible_paths(), 0);
}

TEST_F(CfgSymExEngineTest, RespectsCustomOptions) {
  SymExOptions options;
  options.max_paths = 42;
  options.max_depth = 128;
  options.check_feasibility = false;

  CfgSymExEngine engine(ctx_, options);
  EXPECT_EQ(engine.total_explored_paths(), 0);
  EXPECT_EQ(engine.feasible_paths(), 0);
}

}  // namespace
}  // namespace xls::solvers::symex
