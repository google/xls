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

#include "xls/dslx/type_system_v2/inference_table.h"

#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/type_system/type_info_to_proto.h"
#include "xls/dslx/type_system_v2/inference_table_to_type_info.h"

namespace xls::dslx {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ContainsRegex;
using ::xls::proto_testing::EqualsProto;

class InferenceTableTest : public ::testing::Test {
 public:
  void SetUp() override {
    module_ =
        std::make_unique<Module>("test", /*fs_path=*/std::nullopt, file_table_);
    table_ = InferenceTable::Create(*module_, file_table_);
  }

  absl::StatusOr<TypeInfoProto> ConvertTableToProto() {
    XLS_ASSIGN_OR_RETURN(
        TypeInfo * ti, InferenceTableToTypeInfo(*table_, *module_,
                                                type_info_owner_, file_table_));
    return TypeInfoToProto(*ti);
  }

  FileTable file_table_;
  TypeInfoOwner type_info_owner_;
  std::unique_ptr<Module> module_;
  std::unique_ptr<InferenceTable> table_;
};

TEST_F(InferenceTableTest, TypeInfoForEmptyTable) {
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto proto, ConvertTableToProto());
  EXPECT_THAT(proto, EqualsProto(""));
}

TEST_F(InferenceTableTest, TypeInfoForOneSimpleAnnotation) {
  // Just one def for x:u32.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  TypeAnnotation* annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x, annotation));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto proto, ConvertTableToProto());
  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        nodes {
          kind: AST_NODE_KIND_NAME_DEF
          span {
            start { filename: "<no-file>" lineno: 0 colno: 0 }
            limit { filename: "<no-file>" lineno: 0 colno: 0 }
          }
          type {
            bits_type {
              is_signed: false
              dim {
                interp_value {
                  bits { is_signed: false bit_count: 32 data: "\000\000\000 " }
                }
              }
            }
          }
        }
      )pb"));
}

TEST_F(InferenceTableTest, SetTypeVariableToNonInferenceVariable) {
  // Set type for `x` to a `NameRef` to `N` which is not an inference variable.
  // It should fail because it needs to be an inference variable.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* n = module_->Make<NameDef>(Span::Fake(), "N", /*definer=*/nullptr);
  NameRef* n_ref = module_->Make<NameRef>(Span::Fake(), "N", n);
  EXPECT_THAT(table_->SetTypeVariable(x, n_ref),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(InferenceTableTest, SetTypeVariableToNonType) {
  // Set type for `x` to `N` which is an integer-kind inference variable. It
  // should fail because it needs to be type-kind.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* n = module_->Make<NameDef>(Span::Fake(), "N", /*definer=*/nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(
      NameRef * n_var,
      table_->DefineInternalVariable(InferenceVariableKind::kInteger, n, "N"));
  EXPECT_THAT(table_->SetTypeVariable(x, n_var),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(InferenceTableTest, SignednessMismatch) {
  // Apply the same type variable to the LHS and RHS of an addition, then claim
  // that the LHS (x) is annotated as u32 and the RHS (y) is annotated as s32.
  // Upon associating s32 with y, we should get a signedness mismatch.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* y = module_->Make<NameDef>(Span::Fake(), "y", /*definer=*/nullptr);
  NameRef* x_ref = module_->Make<NameRef>(Span::Fake(), "x", x);
  NameRef* y_ref = module_->Make<NameRef>(Span::Fake(), "y", y);
  AstNode* add_node =
      module_->Make<Binop>(Span::Fake(), BinopKind::kAdd, x_ref, y_ref);

  TypeAnnotation* u32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));
  TypeAnnotation* s32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kS32,
      module_->GetOrCreateBuiltinNameDef("s32"));

  XLS_ASSERT_OK_AND_ASSIGN(
      NameRef * t0, table_->DefineInternalVariable(InferenceVariableKind::kType,
                                                   add_node, "T0"));
  XLS_EXPECT_OK(table_->SetTypeVariable(x_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, u32_annotation));

  EXPECT_THAT(
      table_->SetTypeAnnotation(y_ref, s32_annotation),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ContainsRegex("signed vs. unsigned mismatch.*s32.*vs. u32")));
}

TEST_F(InferenceTableTest, SignednessAgreement) {
  // Apply the same type variable to the LHS and RHS of an addition, then claim
  // that the LHS (x) and the RHS (y) are both annotated as u32. This should not
  // error.
  NameDef* x = module_->Make<NameDef>(Span::Fake(), "x", /*definer=*/nullptr);
  NameDef* y = module_->Make<NameDef>(Span::Fake(), "y", /*definer=*/nullptr);
  NameRef* x_ref = module_->Make<NameRef>(Span::Fake(), "x", x);
  NameRef* y_ref = module_->Make<NameRef>(Span::Fake(), "y", y);
  AstNode* add_node =
      module_->Make<Binop>(Span::Fake(), BinopKind::kAdd, x_ref, y_ref);

  TypeAnnotation* u32_annotation = module_->Make<BuiltinTypeAnnotation>(
      Span::Fake(), BuiltinType::kU32,
      module_->GetOrCreateBuiltinNameDef("u32"));

  XLS_ASSERT_OK_AND_ASSIGN(
      NameRef * t0, table_->DefineInternalVariable(InferenceVariableKind::kType,
                                                   add_node, "T0"));
  XLS_EXPECT_OK(table_->SetTypeVariable(x_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeVariable(y_ref, t0));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(x_ref, u32_annotation));
  XLS_EXPECT_OK(table_->SetTypeAnnotation(y_ref, u32_annotation));
  XLS_ASSERT_OK_AND_ASSIGN(TypeInfoProto proto, ConvertTableToProto());
  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        nodes {
          kind: AST_NODE_KIND_NAME_REF
          span {
            start { filename: "<no-file>" lineno: 0 colno: 0 }
            limit { filename: "<no-file>" lineno: 0 colno: 0 }
          }
          type {
            bits_type {
              is_signed: false
              dim {
                interp_value {
                  bits { is_signed: false bit_count: 32 data: "\000\000\000 " }
                }
              }
            }
          }
        }
        nodes {
          kind: AST_NODE_KIND_NAME_REF
          span {
            start { filename: "<no-file>" lineno: 0 colno: 0 }
            limit { filename: "<no-file>" lineno: 0 colno: 0 }
          }
          type {
            bits_type {
              is_signed: false
              dim {
                interp_value {
                  bits { is_signed: false bit_count: 32 data: "\000\000\000 " }
                }
              }
            }
          }
        }
      )pb"));
}

}  // namespace
}  // namespace xls::dslx
