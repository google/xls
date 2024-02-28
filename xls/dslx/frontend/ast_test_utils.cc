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

#include "xls/dslx/frontend/ast_test_utils.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

std::pair<Module, Binop*> MakeCastWithinLtComparison() {
  Module m("test", /*fs_path=*/std::nullopt);
  const Span fake_span;
  BuiltinNameDef* x_def = m.GetOrCreateBuiltinNameDef("x");
  NameRef* x_ref = m.Make<NameRef>(fake_span, "x", x_def);

  BuiltinTypeAnnotation* builtin_u32 = m.Make<BuiltinTypeAnnotation>(
      fake_span, BuiltinType::kU32, m.GetOrCreateBuiltinNameDef("u32"));

  // type t = u32;
  NameDef* t_def = m.Make<NameDef>(fake_span, "t", /*definer=*/nullptr);
  TypeAlias* type_alias =
      m.Make<TypeAlias>(fake_span, t_def, builtin_u32, /*is_public=*/false);
  t_def->set_definer(type_alias);

  TypeRef* type_ref = m.Make<TypeRef>(fake_span, type_alias);

  auto* type_ref_type_annotation = m.Make<TypeRefTypeAnnotation>(
      fake_span, type_ref, /*parametrics=*/std::vector<ExprOrType>{});

  // x as t < x
  Cast* cast = m.Make<Cast>(fake_span, x_ref, type_ref_type_annotation);
  Binop* lt = m.Make<Binop>(fake_span, BinopKind::kLt, cast, x_ref);
  return std::make_pair(std::move(m), lt);
}

std::pair<Module, Index*> MakeCastWithinIndexExpression() {
  Module m("test", /*fs_path=*/std::nullopt);
  const Span fake_span;
  BuiltinNameDef* x_def = m.GetOrCreateBuiltinNameDef("x");
  NameRef* x_ref = m.Make<NameRef>(fake_span, "x", x_def);

  BuiltinNameDef* i_def = m.GetOrCreateBuiltinNameDef("i");
  NameRef* i_ref = m.Make<NameRef>(fake_span, "i", i_def);

  BuiltinTypeAnnotation* builtin_u32 = m.Make<BuiltinTypeAnnotation>(
      fake_span, BuiltinType::kU32, m.GetOrCreateBuiltinNameDef("u32"));
  ArrayTypeAnnotation* u32_4 = m.Make<ArrayTypeAnnotation>(
      fake_span, builtin_u32,
      m.Make<Number>(fake_span, "42", NumberKind::kOther, /*type=*/nullptr));

  // (x as u32[4])[i]
  Cast* cast = m.Make<Cast>(fake_span, x_ref, u32_4);
  Index* index = m.Make<Index>(fake_span, cast, i_ref);
  return std::make_pair(std::move(m), index);
}

std::pair<Module, TupleIndex*> MakeIndexWithinTupleIndexExpression() {
  Module m("test", /*fs_path=*/std::nullopt);
  const Span fake_span;
  BuiltinNameDef* x_def = m.GetOrCreateBuiltinNameDef("x");
  NameRef* x_ref = m.Make<NameRef>(fake_span, "x", x_def);

  BuiltinNameDef* i_def = m.GetOrCreateBuiltinNameDef("i");
  NameRef* i_ref = m.Make<NameRef>(fake_span, "i", i_def);

  Index* index = m.Make<Index>(fake_span, x_ref, i_ref);
  Number* two =
      m.Make<Number>(fake_span, "2", NumberKind::kOther, /*type=*/nullptr);
  TupleIndex* tuple_index = m.Make<TupleIndex>(fake_span, index, two);
  return std::make_pair(std::move(m), tuple_index);
}

std::pair<Module, XlsTuple*> MakeNElementTupleExpression(
    int64_t n, bool has_trailing_comma) {
  CHECK(n != 1 || has_trailing_comma);  // n==1 -> has_trailing_comme
  Module m("test", /*fs_path=*/std::nullopt);
  const Span fake_span;
  std::vector<Expr*> elements;
  elements.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    std::string name = absl::StrFormat("x%d", i);
    BuiltinNameDef* x_def = m.GetOrCreateBuiltinNameDef(name);
    NameRef* x_ref = m.Make<NameRef>(fake_span, name, x_def);
    elements.push_back(x_ref);
  }

  XlsTuple* tuple = m.Make<XlsTuple>(fake_span, elements,
                                     /*has_trailing_comma=*/has_trailing_comma

  );
  return std::make_pair(std::move(m), tuple);
}

std::pair<Module, Unop*> MakeCastWithinNegateExpression() {
  Module m("test", /*fs_path=*/std::nullopt);
  const Span fake_span;
  BuiltinNameDef* x_def = m.GetOrCreateBuiltinNameDef("x");
  NameRef* x_ref = m.Make<NameRef>(fake_span, "x", x_def);

  BuiltinTypeAnnotation* builtin_u32 = m.Make<BuiltinTypeAnnotation>(
      fake_span, BuiltinType::kU32, m.GetOrCreateBuiltinNameDef("u32"));

  // x as u32
  Cast* cast = m.Make<Cast>(fake_span, x_ref, builtin_u32);
  Unop* unop = m.Make<Unop>(fake_span, UnopKind::kNegate, cast);
  return std::make_pair(std::move(m), unop);
}

std::pair<Module, Attr*> MakeArithWithinAttrExpression() {
  Module m("test", /*fs_path=*/std::nullopt);
  const Span fake_span;
  BuiltinNameDef* x_def = m.GetOrCreateBuiltinNameDef("x");
  NameRef* x_ref = m.Make<NameRef>(fake_span, "x", x_def);
  BuiltinNameDef* y_def = m.GetOrCreateBuiltinNameDef("y");
  NameRef* y_ref = m.Make<NameRef>(fake_span, "y", y_def);

  Binop* binop = m.Make<Binop>(fake_span, BinopKind::kMul, x_ref, y_ref);
  Attr* attr = m.Make<Attr>(fake_span, binop, "my_attr");
  return std::make_pair(std::move(m), attr);
}

}  // namespace xls::dslx
