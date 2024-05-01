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

#include "xls/data_structures/binary_decision_diagram.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

TEST(BinaryDecisionDiagramTest, BasicInvariants) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex var1 = bdd.NewVariable();
  BddNodeIndex var2 = bdd.NewVariable();

  EXPECT_EQ(bdd.GetNode(var1).variable, BddVariable(0));
  EXPECT_EQ(bdd.GetNode(var2).variable, BddVariable(1));

  EXPECT_THAT(bdd.Evaluate(var1, {{var1, true}}), IsOkAndHolds(true));
  EXPECT_THAT(bdd.Evaluate(var2, {{var2, false}}), IsOkAndHolds(false));

  BddNodeIndex not_var1 = bdd.Not(var1);
  EXPECT_NE(var1, not_var1);
  BddNodeIndex not_var2 = bdd.Not(var2);
  EXPECT_NE(var2, not_var2);
  EXPECT_NE(not_var1, not_var2);

  BddNodeIndex var1_or_var2 = bdd.Or(var1, var2);
  EXPECT_EQ(var1_or_var2, bdd.Or(var1, var2));

  BddNodeIndex var1_and_var2 = bdd.And(var1, var2);
  EXPECT_EQ(var1_and_var2, bdd.And(var1, var2));

  EXPECT_NE(var1_or_var2, var1_and_var2);

  EXPECT_EQ(bdd.path_count(var1), 2);
  EXPECT_EQ(bdd.path_count(var2), 2);
  EXPECT_EQ(bdd.path_count(not_var1), 2);
  EXPECT_EQ(bdd.path_count(var1_or_var2), 3);
  EXPECT_EQ(bdd.path_count(var1_and_var2), 3);
}

TEST(BinaryDecisionDiagramTest, BddSize) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex var1 = bdd.NewVariable();
  BddNodeIndex var2 = bdd.NewVariable();

  {
    // Add new expressions should increase the number of nodes.
    int64_t before_size = bdd.size();
    bdd.Or(var1, var2);
    bdd.And(var1, var2);
    EXPECT_GT(bdd.size(), before_size);
  }

  {
    // Constructing expressions that have already be created should not add any
    // additional nodes to the BDD.
    int64_t before_size = bdd.size();
    bdd.Or(var1, var1);
    bdd.Or(var1, var2);
    bdd.Or(var2, var1);
    bdd.And(var1, var2);
    bdd.And(var2, var1);
    EXPECT_EQ(bdd.size(), before_size);
  }
}

TEST(BinaryDecisionDiagramTest, ToString) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex x0 = bdd.NewVariable();
  BddNodeIndex x1 = bdd.NewVariable();
  BddNodeIndex x2 = bdd.NewVariable();
  BddNodeIndex x3 = bdd.NewVariable();

  EXPECT_EQ(bdd.ToStringDnf(bdd.zero()), "0");
  EXPECT_EQ(bdd.ToStringDnf(bdd.one()), "1");
  EXPECT_EQ(bdd.ToStringDnf(x0), "x0");
  EXPECT_EQ(bdd.ToStringDnf(x1), "x1");
  EXPECT_EQ(bdd.ToStringDnf(bdd.And(x0, x1)), "x0.x1");
  EXPECT_EQ(bdd.ToStringDnf(bdd.Or(x1, x2)), "x1 + !x1.x2");
  EXPECT_EQ(bdd.ToStringDnf(bdd.And(bdd.Or(x0, x1), bdd.Or(x2, x3))),
            "x0.x2 + x0.!x2.x3 + !x0.x1.x2 + !x0.x1.!x2.x3");
  EXPECT_EQ(bdd.ToStringDnf(bdd.And(bdd.Or(x0, x1), bdd.Or(x2, x3)),
                            /*minterm_limit=*/2),
            "x0.x2 + x0.!x2.x3 + ...");
  EXPECT_EQ(bdd.ToStringDnf(bdd.And(bdd.Or(x0, x1), bdd.Or(x2, x3)),
                            /*minterm_limit=*/3),
            "x0.x2 + x0.!x2.x3 + !x0.x1.x2 + ...");
  EXPECT_EQ(bdd.ToStringDnf(bdd.And(bdd.Or(x0, x1), bdd.Or(x2, x3)),
                            /*minterm_limit=*/4),
            "x0.x2 + x0.!x2.x3 + !x0.x1.x2 + !x0.x1.!x2.x3");
}

TEST(BinaryDecisionDiagramTest, TrivialEvaluations) {
  BinaryDecisionDiagram bdd;
  EXPECT_THAT(bdd.Evaluate(bdd.zero(), {}), IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(bdd.one(), {}), IsOkAndHolds(true));

  BddNodeIndex x = bdd.NewVariable();

  EXPECT_THAT(bdd.Evaluate(bdd.zero(), {{x, false}}), IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(bdd.zero(), {{x, true}}), IsOkAndHolds(false));

  EXPECT_THAT(bdd.Evaluate(bdd.one(), {{x, false}}), IsOkAndHolds(true));
  EXPECT_THAT(bdd.Evaluate(bdd.one(), {{x, true}}), IsOkAndHolds(true));
}

TEST(BinaryDecisionDiagramTest, AndOrIdentities) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex var1 = bdd.NewVariable();
  BddNodeIndex var2 = bdd.NewVariable();
  BddNodeIndex var3 = bdd.NewVariable();

  EXPECT_EQ(var1, bdd.Or(var1, var1));
  EXPECT_EQ(var1, bdd.And(var1, var1));

  // Commutativity.
  EXPECT_EQ(bdd.Or(var2, var1), bdd.Or(var1, var2));
  EXPECT_EQ(bdd.And(var2, var1), bdd.And(var1, var2));

  // Associativity.
  EXPECT_EQ(bdd.Or(bdd.Or(var1, var2), var3), bdd.Or(var1, bdd.Or(var2, var3)));
  EXPECT_EQ(bdd.And(bdd.And(var1, var2), var3),
            bdd.And(var1, bdd.And(var2, var3)));

  EXPECT_EQ(bdd.one(), bdd.Or(var1, bdd.Not(var1)));
  EXPECT_EQ(bdd.zero(), bdd.And(var1, bdd.Not(var1)));

  // De Morgan's law.
  EXPECT_EQ(bdd.Or(var1, var2), bdd.Not(bdd.And(bdd.Not(var1), bdd.Not(var2))));
  EXPECT_EQ(bdd.And(var1, var2), bdd.Not(bdd.Or(bdd.Not(var1), bdd.Not(var2))));
}

TEST(BinaryDecisionDiagramTest, TwoWayOr) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex var1 = bdd.NewVariable();
  BddNodeIndex var2 = bdd.NewVariable();

  BddNodeIndex var1_or_var2 = bdd.Or(var1, var2);
  EXPECT_THAT(bdd.Evaluate(var1_or_var2, {{var1, false}, {var2, false}}),
              IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(var1_or_var2, {{var1, false}, {var2, true}}),
              IsOkAndHolds(true));
  EXPECT_THAT(bdd.Evaluate(var1_or_var2, {{var1, true}, {var2, false}}),
              IsOkAndHolds(true));
  EXPECT_THAT(bdd.Evaluate(var1_or_var2, {{var1, true}, {var2, true}}),
              IsOkAndHolds(true));
}

TEST(BinaryDecisionDiagramTest, TwoWayAnd) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex var1 = bdd.NewVariable();
  BddNodeIndex var2 = bdd.NewVariable();

  BddNodeIndex var1_and_var2 = bdd.And(var1, var2);
  EXPECT_THAT(bdd.Evaluate(var1_and_var2, {{var1, false}, {var2, false}}),
              IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(var1_and_var2, {{var1, false}, {var2, true}}),
              IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(var1_and_var2, {{var1, true}, {var2, false}}),
              IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(var1_and_var2, {{var1, true}, {var2, true}}),
              IsOkAndHolds(true));
}

TEST(BinaryDecisionDiagramTest, MultiwayAndOr) {
  BinaryDecisionDiagram bdd;
  BddNodeIndex var1 = bdd.NewVariable();
  BddNodeIndex var2 = bdd.NewVariable();
  BddNodeIndex var3 = bdd.NewVariable();
  BddNodeIndex var4 = bdd.NewVariable();

  BddNodeIndex or_reduction = bdd.Or(var1, bdd.Or(var2, bdd.Or(var3, var4)));
  BddNodeIndex and_reduction =
      bdd.And(var1, bdd.And(var2, bdd.And(var3, var4)));
  for (bool v1 : {false, true}) {
    for (bool v2 : {false, true}) {
      for (bool v3 : {false, true}) {
        for (bool v4 : {false, true}) {
          EXPECT_THAT(
              bdd.Evaluate(or_reduction,
                           {{var1, v1}, {var2, v2}, {var3, v3}, {var4, v4}}),
              IsOkAndHolds(v1 || v2 || v3 || v4));
          EXPECT_THAT(
              bdd.Evaluate(and_reduction,
                           {{var1, v1}, {var2, v2}, {var3, v3}, {var4, v4}}),
              IsOkAndHolds(v1 && v2 && v3 && v4));
        }
      }
    }
  }

  EXPECT_EQ(bdd.path_count(or_reduction), 5);
  EXPECT_EQ(bdd.path_count(and_reduction), 5);
}

TEST(BinaryDecisionDiagramTest, Parity) {
  // Construct and test a 64-bit even parity expression.
  BinaryDecisionDiagram bdd;
  std::vector<BddNodeIndex> variables;
  for (int64_t i = 0; i < 64; ++i) {
    variables.push_back(bdd.NewVariable());
  }

  BddNodeIndex parity = bdd.zero();
  for (int64_t i = 0; i < 64; ++i) {
    parity = bdd.Or(bdd.And(parity, bdd.Not(variables[i])),
                    bdd.And(bdd.Not(parity), variables[i]));

    if (i < 30) {
      EXPECT_EQ(bdd.path_count(parity), 1LL << (i + 1));
    } else {
      EXPECT_EQ(bdd.path_count(parity), std::numeric_limits<int32_t>::max());
    }
  }

  auto uint64_to_bools = [&](uint64_t value) {
    absl::flat_hash_map<BddNodeIndex, bool> values;
    for (int64_t i = 0; i < 64; ++i) {
      values[variables[i]] = (((value >> i) & 1) != 0u);
    }
    return values;
  };
  EXPECT_THAT(bdd.Evaluate(parity, uint64_to_bools(0)), IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(parity, uint64_to_bools(1)), IsOkAndHolds(true));
  EXPECT_THAT(bdd.Evaluate(parity, uint64_to_bools(0b110011001100)),
              IsOkAndHolds(false));
  EXPECT_THAT(bdd.Evaluate(parity, uint64_to_bools(0b1100110011001)),
              IsOkAndHolds(true));
  EXPECT_THAT(bdd.Evaluate(parity, uint64_to_bools(0xffffffffffffffffULL)),
              IsOkAndHolds(false));
}

TEST(BinaryDecisionDiagramTest, ThreeVariableExhaustive) {
  // Generate all three-variable boolean functions and test each with all
  // possible inputs.
  BinaryDecisionDiagram bdd;
  std::vector<BddNodeIndex> vars;
  for (int64_t i = 0; i < 3; ++i) {
    vars.push_back(bdd.NewVariable());
  }

  // Generate all minterms.
  std::vector<BddNodeIndex> minterms;
  for (BddNodeIndex v2 : {bdd.Not(vars[2]), vars[2]}) {
    for (BddNodeIndex v1 : {bdd.Not(vars[1]), vars[1]}) {
      for (BddNodeIndex v0 : {bdd.Not(vars[0]), vars[0]}) {
        minterms.push_back(bdd.And(v2, bdd.And(v1, v0)));
      }
    }
  }

  // There are 256 different three variable functions.
  for (int64_t truth_table = 0; truth_table < 256; ++truth_table) {
    // Encode the function in the BDD as a sum of minterms.
    BddNodeIndex func = bdd.zero();
    for (int64_t j = 0; j < 8; ++j) {
      if ((truth_table >> j) & 1) {
        func = bdd.Or(func, minterms[j]);
      }
    }

    auto to_binary = [](int64_t value) {
      std::string result;
      for (int64_t i = 7; i >= 0; --i) {
        result += (value >> i) & 1 ? "1" : "0";
      }
      return result;
    };
    VLOG(1) << "truth_table = " << to_binary(truth_table);
    VLOG(1) << "func = " << bdd.ToStringDnf(func);

    // Now evaluate the function for all inputs
    for (bool x2 : {false, true}) {
      for (bool x1 : {false, true}) {
        for (bool x0 : {false, true}) {
          bool expected = ((truth_table >>
                            (4 * static_cast<int>(x2) +
                             2 * static_cast<int>(x1) + static_cast<int>(x0))) &
                           1) != 0;
          EXPECT_THAT(
              bdd.Evaluate(func, {{vars[0], x0}, {vars[1], x1}, {vars[2], x2}}),
              IsOkAndHolds(expected))
              << bdd.ToStringDnf(func) << " evaluated with x0=" << x0
              << ", x1=" << x1 << ", x2=" << x2;
        }
      }
    }
  }
}

}  // namespace
}  // namespace xls
