// Copyright 2025 The XLS Authors
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

#include "xls/dslx/type_system_v2/type_system_tracer.h"

#include <memory>
#include <optional>

#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/dslx/warning_collector.h"
#include "xls/dslx/warning_kind.h"

namespace xls::dslx {
namespace {

class TypeSystemTracerTest : public ::testing::Test {
 public:
  void SetUp() override {
    tracer_ = TypeSystemTracer::Create(/*active=*/true);
    import_data_.emplace(CreateImportDataForTest());
    warning_collector_.emplace(kAllWarningsSet);
    module_ =
        std::make_unique<Module>("test", /*fs_path=*/std::nullopt, file_table_);
  }

  std::unique_ptr<TypeSystemTracer> tracer_;
  FileTable file_table_;
  std::optional<ImportData> import_data_;
  std::optional<WarningCollector> warning_collector_;
  std::unique_ptr<Module> module_;
};

TEST_F(TypeSystemTracerTest, InactiveTracer) {
  auto tracer = TypeSystemTracer::Create(/*active=*/false);
  Number* one =
      module_->Make<Number>(Span::Fake(), "1", NumberKind::kOther, nullptr);
  tracer->TraceUnify(one);
  EXPECT_EQ(tracer->ConvertTracesToString(), "");
  EXPECT_EQ(tracer->ConvertStatsToString(file_table_), "");
}

TEST_F(TypeSystemTracerTest, ConvertTracesToStringWithNoTraces) {
  EXPECT_EQ(tracer_->ConvertTracesToString(), "");
}

TEST_F(TypeSystemTracerTest, ConvertTracesToString) {
  Number* one =
      module_->Make<Number>(Span::Fake(), "1", NumberKind::kOther, nullptr);
  Number* two =
      module_->Make<Number>(Span::Fake(), "2", NumberKind::kOther, nullptr);
  TypeAnnotation* u32 = nullptr;
  {
    TypeSystemTrace convert_node = tracer_->TraceConvertNode(one);
    {
      TypeSystemTrace unify = tracer_->TraceUnify(one);
      TypeSystemTrace evaluate = tracer_->TraceEvaluate(std::nullopt, one);
    }
    u32 = CreateU32Annotation(*module_, Span::Fake());
    TypeSystemTrace concretize = tracer_->TraceConcretize(u32);
  }
  TypeSystemTrace convert_node2 = tracer_->TraceConvertNode(two);

  EXPECT_EQ(absl::StrCat("\n", tracer_->ConvertTracesToString()),
            absl::StrFormat(R"(
ConvertNode (%p, node: 1, kind: number)
   Unify (%p, node: 1, kind: number)
      Evaluate (%p, node: 1, kind: number)
   Concretize (%p, annotation: u32)
ConvertNode (%p, node: 2, kind: number)
)",
                            one, one, one, u32, two));
}

}  // namespace
}  // namespace xls::dslx
