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

#include "xls/dslx/type_system_v2/fast_concretizer.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/matchers.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"

namespace xls::dslx {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;

class FastConcretizerTest : public ::testing::Test {
 public:
  void SetUp() final {
    module_ =
        std::make_unique<Module>("test", /*fs_path=*/std::nullopt, file_table_);
    concretizer_ = FastConcretizer::Create(file_table_);
  }

 protected:
  FileTable file_table_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<FastConcretizer> concretizer_;
};

TEST_F(FastConcretizerTest, Bool) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      concretizer_->Concretize(CreateBoolAnnotation(*module_, Span::Fake())));
  EXPECT_TRUE(type->CompatibleWith(BitsType(false, 1)));
}

TEST_F(FastConcretizerTest, U32) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      concretizer_->Concretize(CreateU32Annotation(*module_, Span::Fake())));
  EXPECT_TRUE(type->CompatibleWith(BitsType(false, 32)));
}

TEST_F(FastConcretizerTest, UN) {
  EXPECT_THAT(concretizer_->Concretize(
                  CreateUnOrSnElementAnnotation(*module_, Span::Fake(), false)),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(FastConcretizerTest, S32) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      concretizer_->Concretize(CreateS32Annotation(*module_, Span::Fake())));
  EXPECT_TRUE(type->CompatibleWith(BitsType(true, 32)));
}

TEST_F(FastConcretizerTest, SN) {
  EXPECT_THAT(concretizer_->Concretize(
                  CreateUnOrSnElementAnnotation(*module_, Span::Fake(), true)),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(FastConcretizerTest, UN32) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> type,
                           concretizer_->Concretize(CreateUnOrSnAnnotation(
                               *module_, Span::Fake(), false, 32)));
  EXPECT_TRUE(type->CompatibleWith(BitsType(false, 32)));
}

TEST_F(FastConcretizerTest, UNWith33BitDim) {
  EXPECT_THAT(concretizer_->Concretize(CreateUnOrSnAnnotation(
                  *module_, Span::Fake(), false, 0x100000000)),
              Not(IsOk()));
}

TEST_F(FastConcretizerTest, UNWithNegativeDim) {
  EXPECT_THAT(concretizer_->Concretize(
                  CreateUnOrSnAnnotation(*module_, Span::Fake(), false, -1)),
              Not(IsOk()));
}

TEST_F(FastConcretizerTest, SN32) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Type> type,
                           concretizer_->Concretize(CreateUnOrSnAnnotation(
                               *module_, Span::Fake(), true, 32)));
  EXPECT_TRUE(type->CompatibleWith(BitsType(true, 32)));
}

TEST_F(FastConcretizerTest, ArrayOfBuiltin) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      concretizer_->Concretize(module_->Make<ArrayTypeAnnotation>(
          Span::Fake(), CreateU32Annotation(*module_, Span::Fake()),
          module_->Make<Number>(Span::Fake(), "5", NumberKind::kOther,
                                nullptr))));
  EXPECT_TRUE(
      type->CompatibleWith(ArrayType(std::make_unique<BitsType>(false, 32),
                                     TypeDim(InterpValue::MakeU32(5)))));
}

TEST_F(FastConcretizerTest, TupleOfBuiltin) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      concretizer_->Concretize(module_->Make<TupleTypeAnnotation>(
          Span::Fake(),
          std::vector<TypeAnnotation*>{
              CreateU32Annotation(*module_, Span::Fake()),
              CreateUnOrSnAnnotation(*module_, Span::Fake(), true, 8)})));
  std::vector<std::unique_ptr<Type>> members;
  members.push_back(std::make_unique<BitsType>(false, 32));
  members.push_back(std::make_unique<BitsType>(true, 8));
  EXPECT_TRUE(type->CompatibleWith(TupleType(std::move(members))));
}

TEST_F(FastConcretizerTest, FunctionOfBuiltin) {
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Type> type,
      concretizer_->Concretize(module_->Make<FunctionTypeAnnotation>(
          std::vector<const TypeAnnotation*>{
              CreateU32Annotation(*module_, Span::Fake()),
              CreateUnOrSnAnnotation(*module_, Span::Fake(), true, 8)},
          CreateU32Annotation(*module_, Span::Fake()))));
  std::vector<std::unique_ptr<Type>> param_types;
  param_types.push_back(std::make_unique<BitsType>(false, 32));
  param_types.push_back(std::make_unique<BitsType>(true, 8));
  EXPECT_TRUE(type->CompatibleWith(FunctionType(
      std::move(param_types), std::make_unique<BitsType>(false, 32))));
}

}  // namespace
}  // namespace xls::dslx
