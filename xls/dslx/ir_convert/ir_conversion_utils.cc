// Copyright 2021 The XLS Authors
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
#include "xls/dslx/ir_convert/ir_conversion_utils.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/parametric_env.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"

namespace xls::dslx {

absl::StatusOr<int64_t> ResolveDimToInt(const TypeDim& dim,
                                        const ParametricEnv& bindings) {
  return dim.value().GetBitValueViaSign();
}

absl::StatusOr<xls::Type*> TypeToIr(Package* package, const Type& type,
                                    const ParametricEnv& bindings) {
  VLOG(5) << "Converting concrete type to IR: " << type;

  struct Visitor : public TypeVisitor {
   public:
    Visitor(const ParametricEnv& bindings, Package* package)
        : bindings_(bindings), package_(package) {}

    absl::Status HandleArray(const ArrayType& t) override {
      XLS_ASSIGN_OR_RETURN(int64_t element_count,
                           ResolveDimToInt(t.size(), bindings_));

      if (dynamic_cast<const BitsConstructorType*>(&t.element_type())) {
        retval_ = package_->GetBitsType(element_count);
        return absl::OkStatus();
      }

      XLS_ASSIGN_OR_RETURN(xls::Type * element_type,
                           TypeToIr(package_, t.element_type(), bindings_));
      xls::Type* result = package_->GetArrayType(element_count, element_type);
      VLOG(5) << "Converted type to IR; concrete type: " << t
              << " ir: " << result->ToString()
              << " element_count: " << element_count;
      retval_ = result;
      return absl::OkStatus();
    }
    absl::Status HandleBits(const BitsType& t) override {
      XLS_ASSIGN_OR_RETURN(int64_t bit_count,
                           ResolveDimToInt(t.size(), bindings_));
      retval_ = package_->GetBitsType(bit_count);
      return absl::OkStatus();
    }
    absl::Status HandleEnum(const EnumType& t) override {
      XLS_ASSIGN_OR_RETURN(int64_t bit_count, t.size().GetAsInt64());
      retval_ = package_->GetBitsType(bit_count);
      return absl::OkStatus();
    }
    absl::Status HandleToken(const TokenType& t) override {
      retval_ = package_->GetTokenType();
      return absl::OkStatus();
    }
    absl::Status HandleStruct(const StructType& t) override {
      std::vector<xls::Type*> members;
      members.reserve(t.members().size());
      for (const std::unique_ptr<Type>& m : t.members()) {
        XLS_ASSIGN_OR_RETURN(xls::Type * type,
                             TypeToIr(package_, *m, bindings_));
        members.push_back(type);
      }
      retval_ = package_->GetTupleType(members);
      return absl::OkStatus();
    }
    absl::Status HandleProc(const ProcType& t) override {
      // TODO: https://github.com/google/xls/issues/836 - Support this.
      return absl::UnimplementedError(absl::StrCat(
          "IR lowering for impl-style procs is not yet supported: ",
          t.ToString()));
    }
    absl::Status HandleTuple(const TupleType& t) override {
      std::vector<xls::Type*> members;
      members.reserve(t.members().size());
      for (const std::unique_ptr<Type>& m : t.members()) {
        XLS_ASSIGN_OR_RETURN(xls::Type * type,
                             TypeToIr(package_, *m, bindings_));
        members.push_back(type);
      }
      retval_ = package_->GetTupleType(members);
      return absl::OkStatus();
    }
    absl::Status HandleFunction(const FunctionType& t) override {
      return absl::UnimplementedError(absl::StrCat(
          "Cannot convert function type to XLS IR type: ", t.ToString()));
    }
    absl::Status HandleChannel(const ChannelType& t) override {
      return absl::UnimplementedError(absl::StrCat(
          "Cannot convert channel type to XLS IR type: ", t.ToString()));
    }
    absl::Status HandleBitsConstructor(const BitsConstructorType& t) override {
      return absl::UnimplementedError(
          absl::StrCat("Cannot convert bits-constructor type to XLS IR type: ",
                       t.ToString()));
    }
    // Note: this is a bit of a kluge, we just turn metatypes into their
    // corresponding (unwrapped) IR type.
    absl::Status HandleMeta(const MetaType& t) override {
      XLS_ASSIGN_OR_RETURN(retval_,
                           TypeToIr(package_, *t.wrapped(), bindings_));
      return absl::OkStatus();
    }
    absl::Status HandleModule(const ModuleType& t) override {
      return absl::UnimplementedError(absl::StrCat(
          "Cannot convert module type to XLS IR type: ", t.ToString()));
    }

    xls::Type* retval() const { return retval_; }

   private:
    const ParametricEnv& bindings_;
    Package* package_;
    xls::Type* retval_ = nullptr;
  };

  Visitor v(bindings, package);
  XLS_RETURN_IF_ERROR(type.Accept(v));
  return v.retval();
}

std::optional<Function*> GetProcNextFunction(const ProcDef* proc) {
  for (ImplMember member : (*proc->impl())->members()) {
    if (!std::holds_alternative<Function*>(member)) {
      continue;
    }

    Function* fn = std::get<Function*>(member);
    if (fn->identifier() == "next") {
      return fn;
    }
  }

  return std::nullopt;
}

absl::StatusOr<std::vector<Function*>> GetProcConstructors(const ProcDef* p,
                                                           const TypeInfo* ti) {
  XLS_RET_CHECK(p->impl().has_value());
  const Impl* impl = *p->impl();
  std::vector<Function*> result;
  for (ImplMember member : impl->members()) {
    if (std::holds_alternative<Function*>(member)) {
      Function* function = std::get<Function*>(member);
      XLS_ASSIGN_OR_RETURN(const Type* fn_type, ti->GetItemOrError(function));
      XLS_RET_CHECK(fn_type->IsFunction());

      if (!fn_type->AsFunction().params().empty()) {
        const Type& first_param_type = *fn_type->AsFunction().params().front();
        if (first_param_type.IsProc() &&
            &first_param_type.AsProc().struct_def_base() == p) {
          continue;
        }
      }

      // It's only a constructor if it returns effectively `Self`.
      const Type& return_type = fn_type->AsFunction().return_type();
      if (return_type.IsProc() &&
          &return_type.AsProc().struct_def_base() == p) {
        result.push_back(function);
      }
    }
  }
  return result;
}

absl::StatusOr<Function*> GetTopProcConstructor(const ProcDef* proc,
                                                const TypeInfo* ti) {
  XLS_ASSIGN_OR_RETURN(std::vector<Function*> constructors,
                       GetProcConstructors(proc, ti));
  if (constructors.empty()) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Proc '$0' does not have a constructor, i.e. a static function "
        "returning Self, so it cannot be used as a top proc.",
        proc->identifier()));
  }

  if (constructors.size() > 1) {
    return absl::InvalidArgumentError(absl::Substitute(
        "Proc '$0' has $1 possible constructors, i.e. static functions "
        "returning Self. In order to be used as a top proc, there must only be "
        "one constructor.",
        proc->identifier(), constructors.size()));
  }

  return constructors.front();
}

}  // namespace xls::dslx
