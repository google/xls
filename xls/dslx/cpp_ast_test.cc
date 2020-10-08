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

#include "xls/dslx/cpp_ast.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

TEST(CppAst, ModuleWithConstant) {
  Module m("test");
  const Span fake_span;
  NameDef* name_def = m.Make<NameDef>(fake_span, std::string("MOL"));
  Number* number = m.Make<Number>(fake_span, std::string("42"),
                                  NumberKind::kOther, /*type=*/nullptr);
  ConstantDef* constant_def = m.Make<ConstantDef>(fake_span, name_def, number);
  m.AddTop(constant_def);

  EXPECT_EQ(m.ToString(), "const MOL = 42;");
}

}  // namespace
}  // namespace xls::dslx
