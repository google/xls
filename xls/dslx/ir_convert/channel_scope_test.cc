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

#include "xls/dslx/ir_convert/channel_scope.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/create_import_data.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/ir_convert/conversion_info.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/package.h"

namespace xls::dslx {
namespace {

using status_testing::StatusIs;
using ::testing::IsEmpty;

constexpr std::string_view kPackageName = "the_package";

class ChannelScopeTest : public ::testing::Test {
 public:
  void SetUp() override {
    conv_.package = std::make_unique<Package>(kPackageName);
    import_data_ = std::make_unique<ImportData>(CreateImportDataForTest());
    module_ = std::make_unique<Module>("test", /*fs_path=*/std::nullopt,
                                       import_data_->file_table());
    XLS_ASSERT_OK_AND_ASSIGN(type_info_, type_info_owner_.New(module_.get()));
    scope_ = std::make_unique<ChannelScope>(&conv_, import_data_.get());
    scope_->EnterFunctionContext(type_info_, bindings_);
  }

 protected:
  TypeAnnotation* GetU32TypeAnnotation() {
    BuiltinType builtin_type = BuiltinTypeFromString("u32").value();
    TypeAnnotation* type_annot = module_->Make<BuiltinTypeAnnotation>(
        Span::Fake(), builtin_type,
        module_->GetOrCreateBuiltinNameDef(builtin_type));
    type_info_->SetItem(type_annot, MetaType(BitsType::MakeU32()));
    return type_annot;
  }

  ChannelDecl* MakeU32ChannelDecl(
      std::string_view name,
      const std::optional<std::vector<Expr*>>& dims = std::nullopt) {
    TypeAnnotation* data_type_annot = GetU32TypeAnnotation();
    type_info_->SetItem(data_type_annot, MetaType(BitsType::MakeU32()));
    String* name_expr = module_->Make<String>(Span::Fake(), name);
    return module_->Make<ChannelDecl>(Span::Fake(), data_type_annot, dims,
                                      /*fifo_depth=*/std::nullopt, *name_expr);
  }

  Number* MakeU32(std::string_view value) {
    return module_->Make<Number>(Span::Fake(), std::string(value),
                                 NumberKind::kOther, GetU32TypeAnnotation());
  }

  Index* CreateIndexOp(ChannelDecl* decl,
                       const std::vector<std::string_view>& indices) {
    NameDef* fake_array = module_->Make<NameDef>(Span::Fake(), "arr", nullptr);
    absl::StatusOr<ChannelOrArray> channel_or_array =
        scope_->AssociateWithExistingChannelOrArray(fake_array, decl);
    XLS_EXPECT_OK(channel_or_array);
    NameRef* fake_array_ref =
        module_->Make<NameRef>(Span::Fake(), "arr", fake_array);
    Index* index =
        module_->Make<Index>(Span::Fake(), fake_array_ref, MakeU32(indices[0]));
    for (int i = 1; i < indices.size(); i++) {
      index = module_->Make<Index>(Span::Fake(), index, MakeU32(indices[i]));
    }
    return index;
  }

  std::unique_ptr<ImportData> import_data_;
  ParametricEnv bindings_;
  PackageConversionData conv_;
  std::unique_ptr<Module> module_;
  TypeInfoOwner type_info_owner_;
  TypeInfo* type_info_;
  std::unique_ptr<ChannelScope> scope_;
};

TEST_F(ChannelScopeTest, DefineChannel) {
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel");
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<Channel*>(result));
  Channel* channel = std::get<Channel*>(result);
  EXPECT_EQ(channel->name(), "the_package__the_channel");
  EXPECT_EQ(channel->supported_ops(), ChannelOps::kSendReceive);
  EXPECT_TRUE(channel->type()->IsBits());
  EXPECT_THAT(channel->initial_values(), IsEmpty());
}

TEST_F(ChannelScopeTest, DefineChannelArray) {
  std::vector<Expr*> dims = {MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
}

TEST_F(ChannelScopeTest, AssociateWithExistingChannelDecl) {
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel");
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<Channel*>(result));
  NameDef* name_def = module_->Make<NameDef>(Span::Fake(), "ch", nullptr);
  XLS_EXPECT_OK_AND_EQ(
      scope_->AssociateWithExistingChannelOrArray(name_def, decl),
      std::get<Channel*>(result));
}

TEST_F(ChannelScopeTest, AssociateWithExistingChannelArrayDecl) {
  std::vector<Expr*> dims = {MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  NameDef* name_def = module_->Make<NameDef>(Span::Fake(), "ch", nullptr);
  XLS_EXPECT_OK_AND_EQ(
      scope_->AssociateWithExistingChannelOrArray(name_def, decl),
      std::get<ChannelArray*>(result));
}

TEST_F(ChannelScopeTest, AssociateWithExistingChannelOrArrayNonexistent) {
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel");
  NameDef* name_def = module_->Make<NameDef>(Span::Fake(), "ch", nullptr);
  EXPECT_THAT(scope_->AssociateWithExistingChannelOrArray(name_def, decl),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ChannelScopeTest, AssociateWithExistingChannel) {
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel");
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<Channel*>(result));
  NameDef* name_def = module_->Make<NameDef>(Span::Fake(), "ch", nullptr);
  XLS_EXPECT_OK(scope_->AssociateWithExistingChannelOrArray(name_def, result));
}

TEST_F(ChannelScopeTest, AssociateWithExistingChannelArray) {
  std::vector<Expr*> dims = {MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  NameDef* name_def = module_->Make<NameDef>(Span::Fake(), "ch", nullptr);
  XLS_EXPECT_OK(scope_->AssociateWithExistingChannelOrArray(name_def, result));
}

TEST_F(ChannelScopeTest, HandleChannelIndex1DValid) {
  std::vector<Expr*> dims = {MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * channel, scope_->GetChannelForArrayIndex(
                                                  CreateIndexOp(decl, {"2"})));
  EXPECT_EQ(channel->name(), "the_package__the_channel__2");
}

TEST_F(ChannelScopeTest, HandleChannelIndex2DValid) {
  std::vector<Expr*> dims = {MakeU32("2"), MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel,
      scope_->GetChannelForArrayIndex(CreateIndexOp(decl, {"4", "1"})));
  EXPECT_EQ(channel->name(), "the_package__the_channel__4_1");
}

TEST_F(ChannelScopeTest, HandleChannelIndexWithNonArray) {
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel");
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<Channel*>(result));
  EXPECT_THAT(scope_->GetChannelForArrayIndex(CreateIndexOp(decl, {"4"})),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ChannelScopeTest, HandleChannelIndexWithTooManyIndices) {
  std::vector<Expr*> dims = {MakeU32("2"), MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  EXPECT_THAT(
      scope_->GetChannelForArrayIndex(CreateIndexOp(decl, {"4", "1", "1"})),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ChannelScopeTest, HandleChannelIndexWithInsufficientIndices) {
  std::vector<Expr*> dims = {MakeU32("2"), MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  EXPECT_THAT(scope_->GetChannelForArrayIndex(CreateIndexOp(decl, {"4"})),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(ChannelScopeTest, HandleChannelIndexWithOutOfRangeIndices) {
  std::vector<Expr*> dims = {MakeU32("2"), MakeU32("5")};
  ChannelDecl* decl = MakeU32ChannelDecl("the_channel", dims);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelOrArray result,
                           scope_->DefineChannelOrArray(decl));
  EXPECT_TRUE(std::holds_alternative<ChannelArray*>(result));
  EXPECT_THAT(scope_->GetChannelForArrayIndex(CreateIndexOp(decl, {"5", "0"})),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace xls::dslx
