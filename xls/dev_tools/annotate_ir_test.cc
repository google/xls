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

#include "xls/dev_tools/annotate_ir.h"

#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/verifier.h"

namespace xls {
namespace {

using ::testing::HasSubstr;

// A trivial custom IrAnnotator for testing that AnnotateIr emits parameter
// annotations as comment lines above the function and node annotations with
// suffix `// annotation[...]`, and that the resulting IR is valid and
// parseable by the XLS frontend.
class TrivialCustomAnnotator : public IrAnnotator {
 public:
  Annotation NodeAnnotation(Node* node) const override {
    return Annotation{
        .suffix = absl::StrFormat("annotation[%s]", node->GetName()),
    };
  }
};

class AnnotateIrTest : public IrTestBase {};

TEST_F(AnnotateIrTest, CustomAnnotatorEmitsValidParseableIr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue sum = fb.Add(x, y, SourceInfo(), "sum");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sum));
  XLS_ASSERT_OK(p->SetTop(f));

  TrivialCustomAnnotator custom_annotator;
  std::string annotated_ir = AnnotateIr(p.get(), custom_annotator);

  EXPECT_EQ(
      annotated_ir,
      absl::StrFormat("package %s\n\n"
                      "// x: bits[32] = param(name=x, id=1) // annotation[x]\n"
                      "// y: bits[32] = param(name=y, id=2) // annotation[y]\n"
                      "top fn %s(x: bits[32] id=1, y: bits[32] id=2) -> "
                      "bits[32] {\n"
                      "  ret sum: bits[32] = add(x, y, id=3) // "
                      "annotation[sum]\n"
                      "}\n",
                      p->name(), TestName()));

  // Verify that the annotated IR is valid and parseable by the XLS IR parser.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> reparsed,
                           Parser::ParsePackage(annotated_ir));
  XLS_EXPECT_OK(VerifyPackage(reparsed.get()));
}

TEST_F(AnnotateIrTest, ConfiguredVisibilityAnnotatorEmitsValidParseableIr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(1));
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue add = fb.Add(x, y, SourceInfo(), "sum");
  BValue sub = fb.Subtract(x, y, SourceInfo(), "diff");
  BValue result = fb.Select(sel, add, sub, SourceInfo(), "res");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));
  XLS_ASSERT_OK(p->SetTop(f));

  XLS_ASSERT_OK_AND_ASSIGN(AnnotatorKind kind,
                           ParseAnnotatorKind("visibility"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string annotated_ir, AnnotateIr(p.get(), kind));

  EXPECT_THAT(annotated_ir,
              HasSubstr("// x: bits[32] = param(name=x, id=2) // visible[1]"));
  EXPECT_THAT(annotated_ir, HasSubstr("sum: bits[32] = add(x, y, id=4) // "
                                      "visible[x0]"));
  EXPECT_THAT(annotated_ir, HasSubstr("diff: bits[32] = sub(x, y, id=5) // "
                                      "visible[!x0]"));

  // Verify the output IR can be parsed back by the XLS IR parser.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> reparsed,
                           Parser::ParsePackage(annotated_ir));
  XLS_EXPECT_OK(VerifyPackage(reparsed.get()));
}

TEST_F(AnnotateIrTest, ConfiguredDelayAnnotatorEmitsValidParseableIr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(8));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue sum = fb.Add(x, y, SourceInfo(), "sum");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sum));
  XLS_ASSERT_OK(p->SetTop(f));

  XLS_ASSERT_OK_AND_ASSIGN(AnnotatorKind kind, ParseAnnotatorKind("delay"));
  XLS_ASSERT_OK_AND_ASSIGN(std::string annotated_ir,
                           AnnotateIr(p.get(), kind, "unit"));

  EXPECT_THAT(annotated_ir,
              HasSubstr("// x: bits[8] = param(name=x, id=1) // [0ps (+0ps)]"));
  EXPECT_THAT(annotated_ir,
              HasSubstr("ret sum: bits[8] = add(x, y, id=3) // [1ps (+1ps)]"));

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> reparsed,
                           Parser::ParsePackage(annotated_ir));
  XLS_EXPECT_OK(VerifyPackage(reparsed.get()));
}

}  // namespace
}  // namespace xls
