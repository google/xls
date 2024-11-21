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
#include "xls/solvers/z3_utils.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/solvers/z3_op_translator.h"
#include "z3/src/api/z3_api.h"

namespace xls::solvers::z3 {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class Z3UtilsTest : public testing::Test {
 public:
  Z3UtilsTest() {
    config_ = Z3_mk_config();
    Z3_set_param_value(config_, "proof", "true");
    ctx_ = Z3_mk_context(config_);
    solver_ = CreateSolver(ctx_, /*num_threads=*/1);
  }

  ~Z3UtilsTest() override {
    Z3_solver_dec_ref(ctx_, solver_);
    Z3_del_context(ctx_);
    Z3_del_config(config_);
  }

 protected:
  Z3_config config_;
  Z3_context ctx_;
  Z3_solver solver_;
};

// Verifies that z3 boolean values are hexified correctly.
TEST_F(Z3UtilsTest, Hexifies) {
  std::string input_text =
      "This is some fun text. It has a boolean string, #b1010010110100101 .";
  std::string output_text = HexifyOutput(input_text);
  EXPECT_EQ(output_text,
            "This is some fun text. It has a boolean string, #xa5a5 .");
}

TEST_F(Z3UtilsTest, NodeValueBits) {
  Package p("test_pkg");

  Z3OpTranslator t(ctx_);
  Z3_ast param = t.MakeBvParam(1, "p");
  Z3_ast extended_param = t.ZextBy1b(param);
  Z3_ast is_all_z = t.EqZero(extended_param);
  Z3_ast all_one = t.Fill(true, /*bit_count=*/2);
  Z3_ast is_all_o = t.Eq(extended_param, all_one);
  Z3_ast all_z_or_o = t.Or(is_all_o, is_all_z);
  Z3_ast all_z_or_o_bool = t.EqZeroBool(all_z_or_o);

  // We assert that all_z_or_o is false.
  EXPECT_EQ(t.GetSortName(all_z_or_o_bool), "Bool");
  Z3_solver_assert(ctx_, solver_, all_z_or_o_bool);

  // We find an example where the zero-extension of a single bit value is /not/
  // all-zeros or all-ones.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  ASSERT_EQ(satisfiable, Z3_L_TRUE);

  Z3_model model = Z3_solver_get_model(ctx_, solver_);
  XLS_ASSERT_OK_AND_ASSIGN(Value param_value,
                           NodeValue(ctx_, model, param, p.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(Bits param_bits, param_value.GetBitsWithStatus());
  EXPECT_EQ(param_bits, UBits(1, 1));
}

TEST_F(Z3UtilsTest, NodeValueBitsBigger) {
  Package p("test_pkg");

  Z3OpTranslator t(ctx_);
  Z3_ast param = t.MakeBvParam(256, "p");
  Z3_ast extended_param = t.ZextBy1b(param);
  Z3_ast is_all_z = t.EqZero(extended_param);
  Z3_ast all_one = t.Fill(true, /*bit_count=*/257);
  Z3_ast is_all_o = t.Eq(extended_param, all_one);
  Z3_ast all_z_or_o = t.Or(is_all_o, is_all_z);
  Z3_ast all_z_or_o_bool = t.EqZeroBool(all_z_or_o);

  // We assert that all_z_or_o is false.
  EXPECT_EQ(t.GetSortName(all_z_or_o_bool), "Bool");
  Z3_solver_assert(ctx_, solver_, all_z_or_o_bool);

  // We find an example where the zero-extension of a value is /not/ all-zeros
  // or all-ones.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  ASSERT_EQ(satisfiable, Z3_L_TRUE);

  Z3_model model = Z3_solver_get_model(ctx_, solver_);
  XLS_ASSERT_OK_AND_ASSIGN(Value param_value,
                           NodeValue(ctx_, model, param, p.GetBitsType(256)));
  XLS_ASSERT_OK_AND_ASSIGN(Bits param_bits, param_value.GetBitsWithStatus());
  EXPECT_EQ(param_bits.bit_count(), 256);
  EXPECT_NE(param_bits, UBits(0, 256));
}

TEST_F(Z3UtilsTest, NodeValueArray) {
  Package p("test_pkg");

  Z3OpTranslator t(ctx_);
  // Make two params: an array of 2 1-bit vectors, and two indices into it.
  Z3_sort element_sort = Z3_mk_bv_sort(ctx_, 1);
  Z3_sort index_sort = Z3_mk_bv_sort(ctx_, 1);
  Z3_sort array_sort = Z3_mk_array_sort(ctx_, index_sort, element_sort);

  Z3_ast array = Z3_mk_const(ctx_, Z3_mk_string_symbol(ctx_, "p"), array_sort);
  Z3_ast index1 = t.MakeBvParam(1, "i");
  Z3_ast index2 = t.MakeBvParam(1, "j");

  Z3_ast selected1 = Z3_mk_select(ctx_, array, index1);
  Z3_ast selected2 = Z3_mk_select(ctx_, array, index2);

  // Assert that the values selected are not equal.
  Z3_ast are_different = t.NeBool(selected1, selected2);

  // We assert that are_different is true.
  EXPECT_EQ(t.GetSortName(are_different), "Bool");
  Z3_solver_assert(ctx_, solver_, are_different);

  // We find an example where not all elements are equal.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  ASSERT_EQ(satisfiable, Z3_L_TRUE);

  Z3_model model = Z3_solver_get_model(ctx_, solver_);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_value,
      NodeValue(ctx_, model, array, p.GetArrayType(2, p.GetBitsType(1))));
  ASSERT_TRUE(array_value.IsArray());
  EXPECT_THAT(array_value.GetElements(),
              IsOkAndHolds(UnorderedElementsAre(Value(UBits(0, 1)),
                                                Value(UBits(1, 1)))));
}

TEST_F(Z3UtilsTest, NodeValueArrayBigger) {
  Package p("test_pkg");

  Z3OpTranslator t(ctx_);
  // Make two params: an array of 16 8-bit vectors, and two indices into it.
  Z3_sort element_sort = Z3_mk_bv_sort(ctx_, 8);
  Z3_sort index_sort = Z3_mk_bv_sort(ctx_, 4);
  Z3_sort array_sort = Z3_mk_array_sort(ctx_, index_sort, element_sort);

  Z3_ast array = Z3_mk_const(ctx_, Z3_mk_string_symbol(ctx_, "p"), array_sort);
  Z3_ast index1 = t.MakeBvParam(4, "i");
  Z3_ast index2 = t.MakeBvParam(4, "j");

  Z3_ast selected1 = Z3_mk_select(ctx_, array, index1);
  Z3_ast selected2 = Z3_mk_select(ctx_, array, index2);

  // Assert that the values selected are not equal.
  Z3_ast are_different = t.NeBool(selected1, selected2);

  // We assert that are_different is true.
  EXPECT_EQ(t.GetSortName(are_different), "Bool");
  Z3_solver_assert(ctx_, solver_, are_different);

  // We find an example where not all elements are equal.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  ASSERT_EQ(satisfiable, Z3_L_TRUE);

  Z3_model model = Z3_solver_get_model(ctx_, solver_);
  XLS_ASSERT_OK_AND_ASSIGN(
      Value array_value,
      NodeValue(ctx_, model, array, p.GetArrayType(16, p.GetBitsType(8))));
  ASSERT_TRUE(array_value.IsArray());

  XLS_ASSERT_OK_AND_ASSIGN(Value index1_value,
                           NodeValue(ctx_, model, index1, p.GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(Value index2_value,
                           NodeValue(ctx_, model, index2, p.GetBitsType(4)));
  ASSERT_TRUE(index1_value.IsBits());
  ASSERT_TRUE(index2_value.IsBits());
  EXPECT_NE(index1_value, index2_value);

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> elements,
                           array_value.GetElements());
  absl::flat_hash_set<Bits> elements_set;
  for (const Value& element : elements) {
    XLS_ASSERT_OK_AND_ASSIGN(Bits element_bits, element.GetBitsWithStatus());
    elements_set.insert(element_bits);
  }
  EXPECT_GT(elements_set.size(), 1)
      << "Only one unique element: " << array_value.ToString();
}

TEST_F(Z3UtilsTest, NodeValueTuple) {
  Package p("test_pkg");

  Z3OpTranslator t(ctx_);
  // Make one param: a tuple of two bits-typed values.
  Type* tuple_type = p.GetTupleType({p.GetBitsType(1), p.GetBitsType(1)});
  Z3_sort tuple_sort = TypeToSort(ctx_, *tuple_type);
  Z3_ast tuple = Z3_mk_const(ctx_, Z3_mk_string_symbol(ctx_, "p"), tuple_sort);

  Z3_func_decl proj_fn1 = Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, 0);
  Z3_ast element1 = Z3_mk_app(ctx_, proj_fn1, 1, &tuple);

  Z3_func_decl proj_fn2 = Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, 1);
  Z3_ast element2 = Z3_mk_app(ctx_, proj_fn2, 1, &tuple);

  // Assert that neither element is zero.
  Z3_ast not_zero = t.AndBool(t.NeZeroBool(element1), t.NeZeroBool(element2));

  // We assert that not_zero is true.
  EXPECT_EQ(t.GetSortName(not_zero), "Bool");
  Z3_solver_assert(ctx_, solver_, not_zero);

  // We find an example where neither element is zero.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  ASSERT_EQ(satisfiable, Z3_L_TRUE);

  Z3_model model = Z3_solver_get_model(ctx_, solver_);
  XLS_ASSERT_OK_AND_ASSIGN(Value tuple_value,
                           NodeValue(ctx_, model, tuple, tuple_type));
  ASSERT_TRUE(tuple_value.IsTuple());
  EXPECT_THAT(
      tuple_value.GetElements(),
      IsOkAndHolds(ElementsAre(Value(UBits(1, 1)), Value(UBits(1, 1)))));
}

TEST_F(Z3UtilsTest, NodeValueTupleBigger) {
  Package p("test_pkg");

  Z3OpTranslator t(ctx_);
  // Make one param: a tuple of two bits-typed values.
  Type* tuple_type = p.GetTupleType({p.GetBitsType(8), p.GetBitsType(16)});
  Z3_sort tuple_sort = TypeToSort(ctx_, *tuple_type);
  Z3_ast tuple = Z3_mk_const(ctx_, Z3_mk_string_symbol(ctx_, "p"), tuple_sort);

  Z3_func_decl proj_fn1 = Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, 0);
  Z3_ast element1 = Z3_mk_app(ctx_, proj_fn1, 1, &tuple);

  Z3_func_decl proj_fn2 = Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, 1);
  Z3_ast element2 = Z3_mk_app(ctx_, proj_fn2, 1, &tuple);

  // Assert that neither element is zero.
  Z3_ast not_zero = t.AndBool(t.NeZeroBool(element1), t.NeZeroBool(element2));

  // We assert that not_zero is true.
  EXPECT_EQ(t.GetSortName(not_zero), "Bool");
  Z3_solver_assert(ctx_, solver_, not_zero);

  // We find an example where neither element is zero.
  Z3_lbool satisfiable = Z3_solver_check(ctx_, solver_);
  ASSERT_EQ(satisfiable, Z3_L_TRUE);

  Z3_model model = Z3_solver_get_model(ctx_, solver_);
  XLS_ASSERT_OK_AND_ASSIGN(Value tuple_value,
                           NodeValue(ctx_, model, tuple, tuple_type));
  ASSERT_TRUE(tuple_value.IsTuple());
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<Value> elements,
                           tuple_value.GetElements());
  ASSERT_THAT(elements, SizeIs(2));
  EXPECT_EQ(elements[0].bits().bit_count(), 8);
  EXPECT_FALSE(elements[0].bits().IsZero());
  EXPECT_EQ(elements[1].bits().bit_count(), 16);
  EXPECT_FALSE(elements[1].bits().IsZero());
}

}  // namespace
}  // namespace xls::solvers::z3
