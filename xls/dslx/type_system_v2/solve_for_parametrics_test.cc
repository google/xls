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

#include "xls/dslx/type_system_v2/solve_for_parametrics.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/parser.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/interp_value.h"
#include "xls/ir/bits.h"

namespace xls::dslx {
namespace {

using absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class SolveForParametricsTest : public ::testing::Test {
 public:
  absl::StatusOr<std::unique_ptr<Module>> Parse(std::string_view program) {
    scanner_.emplace(file_table_, Fileno(0), std::string(program));
    parser_.emplace("test", &*scanner_);
    return parser_->ParseModule();
  }

  absl::StatusOr<InterpValue> EvaluateLiteral(const AstNode* node,
                                              bool is_signed, int bit_count) {
    if (const Number* literal = dynamic_cast<const Number*>(node)) {
      XLS_ASSIGN_OR_RETURN(Bits bits, literal->GetBits(bit_count, file_table_));
      return InterpValue::MakeBits(is_signed, bits);
    }
    return absl::InvalidArgumentError(
        absl::Substitute("Not a literal: $0", node->ToString()));
  }

  FileTable file_table_;
  std::optional<Scanner> scanner_;
  std::optional<Parser> parser_;
};

TEST_F(SolveForParametricsTest, SolveWithNoSolution) {
  // In this example, we can't solve for `N` using the type annotation of `a`.
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: u32, b: uN[N]) -> uN[N] { b }
const BAR: uN[4] = uN[4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, IsEmpty());
}

TEST_F(SolveForParametricsTest, SolveForUnWithUn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const BAR: uN[4] = uN[4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(4))));
}

TEST_F(SolveForParametricsTest, SolveForUnWithConstant) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
const X = u32:1;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const BAR: uN[X] = uN[X]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return InterpValue::MakeU32(5);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(5))));
}

TEST_F(SolveForParametricsTest, SolveForUnWithUnWithExpr) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
const X = u32:1;
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const BAR: uN[4 + X] = uN[4 + X]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return InterpValue::MakeU32(5);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(5))));
}

TEST_F(SolveForParametricsTest, SolveForUnWithBuiltin) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
const BAR: u4 = u4:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(4))));
}

TEST_F(SolveForParametricsTest, SolveForSnWithSn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: s32>(a: sN[N]) -> sN[N] { a }
const BAR: sN[4] = sN[4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForUnWithXn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: s32>(a: uN[N]) -> uN[N] { a }
const BAR: xN[false][4] = xN[false][4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForUnWithBits) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: s32>(a: uN[N]) -> uN[N] { a }
const BAR: bits[4] = bits[4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForBitsWithUn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: s32>(a: bits[N]) -> bits[N] { a }
const BAR: uN[4] = uN[4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForXnWithXn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<S: bool, N: s32>(a: xN[S][N]) -> xN[S][N] { a }
const BAR: xN[true][4] = xN[true][4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* s = foo->parametric_bindings()[0];
  const ParametricBinding* n = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{s, n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(s, InterpValue::MakeU32(1)),
                                           Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForNOnlyInXn) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<S: bool, N: s32>(a: xN[S][N]) -> xN[S][N] { a }
const BAR: xN[true][4] = xN[true][4]:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForXnWithBuiltin) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<S: bool, N: s32>(a: xN[S][N]) -> xN[S][N] { a }
const BAR: s4 = s4:1;
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* s = foo->parametric_bindings()[0];
  const ParametricBinding* n = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{s, n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(s, InterpValue::MakeBool(1)),
                                           Pair(n, InterpValue::MakeS32(4))));
}

TEST_F(SolveForParametricsTest, SolveForBuiltinArray) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: u32[N]) -> u32[N] { a }
const BAR: u32[3] = [u32:0, u32:1, u32:3];
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(3))));
}

TEST_F(SolveForParametricsTest, SolveForArrayOfGenericType) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<T: type, N: u32>(a: T[N]) -> T[N] { a }
const BAR: u32[3] = [u32:0, u32:1, u32:3];
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* t = foo->parametric_bindings()[0];
  const ParametricBinding* n = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{t, n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_EQ(values.size(), 2);
  EXPECT_TRUE(values.contains(t));
  EXPECT_TRUE(values.contains(n));
  EXPECT_EQ(ToString(values.at(t)), "u32");
  EXPECT_EQ(values.at(n), InterpValueOrTypeAnnotation(InterpValue::MakeU32(3)));
}

TEST_F(SolveForParametricsTest, SolveForArrayOfTupleGenericType) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<T: type, N: u32>(a: T[N]) -> T[N] { a }
const BAR: (s16, u32)[3] = [(0, 1), (2, 3), (4, 5)];
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* t = foo->parametric_bindings()[0];
  const ParametricBinding* n = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{t, n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_EQ(values.size(), 2);
  EXPECT_TRUE(values.contains(t));
  EXPECT_TRUE(values.contains(n));
  EXPECT_EQ(ToString(values.at(t)), "(s16, u32)");
  EXPECT_EQ(values.at(n), InterpValueOrTypeAnnotation(InterpValue::MakeU32(3)));
}

TEST_F(SolveForParametricsTest, SolveForGenericTypeInTuple) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<T: type>(a: (T, u32)) -> (T, u32) { a }
const BAR: (s16, u32) = (s16:5, u32:2);
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* t = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{t},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_EQ(values.size(), 1);
  EXPECT_TRUE(values.contains(t));
  EXPECT_EQ(ToString(values.at(t)), "s16");
}

TEST_F(SolveForParametricsTest, SolveFor2dBuiltinArray) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<M: u32, N: u32>(a: u32[M][N]) -> u32[M][N] { a }
const BAR: u32[33][34] = zero!<u32[33][34]>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* m = foo->parametric_bindings()[0];
  const ParametricBinding* n = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{m, n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(m, InterpValue::MakeU32(33)),
                                           Pair(n, InterpValue::MakeU32(34))));
}

TEST_F(SolveForParametricsTest, SolveFor2dBuiltinArrayWithConflictFails) {
  // In this case we are naming both of the dimensions N but trying to use 2
  // different values for them.
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: u32[N][N]) -> u32[N][N] { a }
const BAR: u32[33][34] = zero!<u32[33][34]>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  EXPECT_THAT(
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("u32:33 vs. u32:34")));
}

TEST_F(SolveForParametricsTest, SolveFor2dXnArray) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<S: u32, W: u32, M: u32, N: u32>(a: xN[S][W][M][N]) -> xN[S][W][M][N] {
  a
}
const BAR: xN[false][24][33][34] = zero!<xN[false][24][33][34]>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* s = foo->parametric_bindings()[0];
  const ParametricBinding* w = foo->parametric_bindings()[1];
  const ParametricBinding* m = foo->parametric_bindings()[2];
  const ParametricBinding* n = foo->parametric_bindings()[3];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values, SolveForParametrics(
                  bar->type_annotation(), a->type_annotation(),
                  absl::flat_hash_set<const ParametricBinding*>{m, n, s, w},
                  [&](const TypeAnnotation*, const Expr* expr) {
                    return EvaluateLiteral(expr, false, 32);
                  }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(s, InterpValue::MakeU32(false)),
                                           Pair(w, InterpValue::MakeU32(24)),
                                           Pair(m, InterpValue::MakeU32(33)),
                                           Pair(n, InterpValue::MakeU32(34))));
}

TEST_F(SolveForParametricsTest, SolveForArrayOfTuples) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32, X: u32>(a: (uN[N], (s4, sN[N]))[X]) -> u32 { 0 }
const BAR: (u10, (s4, sN[10]))[20] = zero!<(u10, (s4, sN[10]))[20]>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const ParametricBinding* x = foo->parametric_bindings()[1];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n, x},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(10)),
                                           Pair(x, InterpValue::MakeU32(20))));
}

TEST_F(SolveForParametricsTest, SolveWithStructuralMismatchFails) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: u32[N]) -> u32 { 0 }
const BAR: (uN[5], bool) = zero!<(uN[5], bool)>();
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  XLS_ASSERT_OK_AND_ASSIGN(const ConstantDef* bar,
                           module->GetMemberOrError<ConstantDef>("BAR"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  EXPECT_THAT(
      SolveForParametrics(bar->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*, const Expr* expr) {
                            return EvaluateLiteral(expr, false, 32);
                          }),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Mismatch: (uN[5], bool) vs. u32[N]")));
}

TEST_F(SolveForParametricsTest, SolveForReentrantParametricInvocation) {
  XLS_ASSERT_OK_AND_ASSIGN(auto module, Parse(R"(
fn foo<N: u32>(a: uN[N]) -> uN[N] { a }
)"));
  XLS_ASSERT_OK_AND_ASSIGN(const Function* foo,
                           module->GetMemberOrError<Function>("foo"));
  const ParametricBinding* n = foo->parametric_bindings()[0];
  const Param* a = foo->params()[0];
  absl::flat_hash_map<const ParametricBinding*, InterpValueOrTypeAnnotation>
      values;
  XLS_ASSERT_OK_AND_ASSIGN(
      values,
      SolveForParametrics(a->type_annotation(), a->type_annotation(),
                          absl::flat_hash_set<const ParametricBinding*>{n},
                          [&](const TypeAnnotation*,
                              const Expr* expr) -> absl::StatusOr<InterpValue> {
                            // Simulate having an "N" value from a lower-frame
                            // `foo` call.
                            EXPECT_EQ(expr->ToString(), "N");
                            return InterpValue::MakeU32(4);
                          }));
  EXPECT_THAT(values, UnorderedElementsAre(Pair(n, InterpValue::MakeU32(4))));
}

}  // namespace
}  // namespace xls::dslx
