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

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node_visitor_with_default.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system_v2/type_annotation_utils.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"

namespace xls::dslx {
namespace {

class FastConcretizerImpl : public FastConcretizer,
                            public AstNodeVisitorWithDefault {
 public:
  explicit FastConcretizerImpl(const FileTable& file_table)
      : file_table_(file_table) {}

  absl::StatusOr<std::unique_ptr<Type>> Concretize(
      const TypeAnnotation* annotation) final {
    absl::StatusOr<SignednessAndBitCountResult> signedness_and_bit_count =
        GetSignednessAndBitCount(annotation);
    if (signedness_and_bit_count.ok()) {
      XLS_ASSIGN_OR_RETURN(bool is_signed,
                           GetBool(signedness_and_bit_count->signedness));
      XLS_ASSIGN_OR_RETURN(uint32_t bit_count,
                           GetU32(signedness_and_bit_count->bit_count));
      return std::make_unique<BitsType>(is_signed, bit_count);
    }

    result_ = nullptr;
    XLS_RETURN_IF_ERROR(annotation->Accept(this));
    XLS_RET_CHECK(result_ != nullptr);
    return std::move(result_);
  }

  absl::Status HandleArrayTypeAnnotation(
      const ArrayTypeAnnotation* array_type) final {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> element_type,
                         Concretize(array_type->element_type()));
    XLS_ASSIGN_OR_RETURN(uint32_t dim, GetU32(array_type->dim()));
    result_ = std::make_unique<ArrayType>(std::move(element_type),
                                          TypeDim(InterpValue::MakeU32(dim)));
    return absl::OkStatus();
  }

  absl::Status HandleTupleTypeAnnotation(
      const TupleTypeAnnotation* tuple_type) final {
    std::vector<std::unique_ptr<Type>> member_types;
    member_types.reserve(tuple_type->members().size());
    for (const TypeAnnotation* member : tuple_type->members()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> member_type,
                           Concretize(member));
      member_types.push_back(std::move(member_type));
    }
    result_ = std::make_unique<TupleType>(std::move(member_types));
    return absl::OkStatus();
  }

  absl::Status HandleFunctionTypeAnnotation(
      const FunctionTypeAnnotation* function_type) final {
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> return_type,
                         Concretize(function_type->return_type()));
    std::vector<std::unique_ptr<Type>> param_types;
    param_types.reserve(function_type->param_types().size());
    for (const TypeAnnotation* param : function_type->param_types()) {
      XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> param_type, Concretize(param));
      param_types.push_back(std::move(param_type));
    }
    result_ = std::make_unique<FunctionType>(std::move(param_types),
                                             std::move(return_type));
    return absl::OkStatus();
  }

  absl::Status DefaultHandler(const AstNode*) final {
    return absl::UnimplementedError("Node is not handled by fast concretizer.");
  }

 private:
  absl::StatusOr<bool> GetBool(std::variant<bool, const Expr*> value) {
    if (std::holds_alternative<bool>(value)) {
      return std::get<bool>(value);
    }
    if (std::get<const Expr*>(value)->kind() == AstNodeKind::kNumber) {
      const auto* literal =
          down_cast<const Number*>(std::get<const Expr*>(value));
      XLS_ASSIGN_OR_RETURN(Bits bits, literal->GetBits(32, file_table_));
      if (bits_ops::SGreaterThanOrEqual(bits, 0) &&
          bits_ops::SLessThanOrEqual(bits, 1)) {
        return bits.Get(0);
      }
    }
    return absl::UnimplementedError("Unsupported value for fast concretizer.");
  }

  absl::StatusOr<uint32_t> GetU32(std::variant<int64_t, const Expr*> value) {
    constexpr int64_t kMaxUint32AsInt64 = 0x0ffffffff;
    if (std::holds_alternative<int64_t>(value)) {
      const int64_t result = std::get<int64_t>(value);
      if (result >= 0 && result <= kMaxUint32AsInt64) {
        return result;
      }
    }
    if (std::get<const Expr*>(value)->kind() == AstNodeKind::kNumber) {
      const auto* literal =
          down_cast<const Number*>(std::get<const Expr*>(value));
      XLS_ASSIGN_OR_RETURN(const Bits bits, literal->GetBits(32, file_table_));
      if (bits_ops::SGreaterThanOrEqual(bits, 0) &&
          bits_ops::SLessThanOrEqual(bits, kMaxUint32AsInt64)) {
        XLS_ASSIGN_OR_RETURN(uint64_t result,
                             literal->GetAsUint64(file_table_));
        return result;
      }
    }
    return absl::UnimplementedError("Unsupported value for fast concretizer.");
  }

  const FileTable& file_table_;
  std::unique_ptr<Type> result_;
};

}  // namespace

std::unique_ptr<FastConcretizer> FastConcretizer::Create(
    const FileTable& file_table) {
  return std::make_unique<FastConcretizerImpl>(file_table);
}

}  // namespace xls::dslx
