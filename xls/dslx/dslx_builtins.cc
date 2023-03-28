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

#include "xls/dslx/dslx_builtins.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/type_system/concrete_type.h"

namespace xls::dslx {
namespace {

using ArgTypes = const std::vector<const ConcreteType*>&;
using ParametricBindings = std::optional<std::vector<ParametricBinding*>>;
using ConstexprEvalFn = std::function<absl::Status(int64_t argno)>;

// Fluent API for checking argument type properties (and raising errors).
//
// Note: we use guards at the start of the check methods and assignment to
// status_ instead of status_.Update() so that things like Len(2) failing
// automatically short circuit attempts to access particular indices, instead of
// causing faults...
//
// So you can safely write `checker.Len(2).IsUN(0).IsUN(1)` without worrying
// "what if the len wasn't two?!"
class Checker {
 public:
  Checker(ArgTypes arg_types, std::string_view name, const Span& span,
          DeduceCtx& deduce_ctx)
      : arg_types_(arg_types),
        name_(name),
        span_(span),
        deduce_ctx_(deduce_ctx) {}

  Checker& Len(int64_t target) {
    if (!status_.ok()) {
      return *this;
    }
    if (arg_types_.size() != target) {
      status_ = ArgCountMismatchErrorStatus(
          span_,
          absl::StrFormat("Invalid number of arguments passed to '%s'", name_));
    }
    return *this;
  }
  Checker& Eq(const ConcreteType& lhs, const ConcreteType& rhs,
              const std::function<std::string()>& make_message) {
    if (!status_.ok()) {
      return *this;
    }
    if (lhs != rhs) {
      status_ = deduce_ctx_.TypeMismatchError(span_, nullptr, lhs, nullptr, rhs,
                                              make_message());
    }
    return *this;
  }
  Checker& IsFn(int64_t argno, int64_t argc,
                const FunctionType** out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const ConcreteType& t = *arg_types_[argno];
    if (auto* f = dynamic_cast<const FunctionType*>(&t)) {
      if (f->GetParamCount() == argc) {
        if (out != nullptr) {
          *out = f;
        }
      } else {
        status_ = TypeInferenceErrorStatus(
            span_, &t,
            absl::StrFormat("Want argument %d to '%s' to be a function with %d "
                            "parameters; got %d",
                            argno, name_, argc, f->GetParamCount()));
      }
    } else {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to '%s' to be a function; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& IsArray(int64_t argno, const ArrayType** out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const ConcreteType& t = *arg_types_[argno];
    if (auto* a = dynamic_cast<const ArrayType*>(&t)) {
      if (out != nullptr) {
        *out = a;
      }
    } else {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to '%s' to be an array; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& IsBits(int64_t argno, const BitsType** out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const ConcreteType& t = *arg_types_[argno];
    if (auto* a = dynamic_cast<const BitsType*>(&t)) {
      if (out != nullptr) {
        *out = a;
      }
    } else {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to '%s' to be bits typed; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& IsUN(int64_t argno) {
    if (!status_.ok()) {
      return *this;
    }
    const ConcreteType& t = *arg_types_[argno];
    if (auto* a = dynamic_cast<const BitsType*>(&t);
        a == nullptr || a->is_signed()) {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to be unsigned bits; got %s", argno,
                          t.ToString()));
    }
    return *this;
  }
  Checker& CheckIsBits(const ConcreteType& t,
                       const std::function<std::string()> make_msg) {
    if (!status_.ok()) {
      return *this;
    }
    if (auto* bits = dynamic_cast<const BitsType*>(&t); bits == nullptr) {
      status_ = TypeInferenceErrorStatus(span_, &t, make_msg());
    }
    return *this;
  }
  Checker& CheckIsLen(const ArrayType& t, int64_t target,
                      const std::function<std::string()> make_msg) {
    if (!status_.ok()) {
      return *this;
    }
    if (t.size() != ConcreteTypeDim::CreateU32(target)) {
      status_ = TypeInferenceErrorStatus(span_, &t, make_msg());
    }
    return *this;
  }
  Checker& IsU1(int64_t argno) {
    if (!status_.ok()) {
      return *this;
    }
    const ConcreteType& t = *arg_types_[argno];
    if (auto* bits = dynamic_cast<const BitsType*>(&t);
        bits == nullptr || bits->size() != ConcreteTypeDim::CreateU32(1)) {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Expected argument %d to '%s' to be a u1; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& TypesAreSame(const ConcreteType& t, const ConcreteType& u,
                        const std::function<std::string()> make_msg) {
    if (!status_.ok()) {
      return *this;
    }
    if (t != u) {
      status_ = deduce_ctx_.TypeMismatchError(span_, nullptr, t, nullptr, u,
                                              make_msg());
    }
    return *this;
  }
  Checker& ArgsSameType(int64_t argno0, int64_t argno1) {
    if (!status_.ok()) {
      return *this;
    }
    const ConcreteType& lhs = *arg_types_[argno0];
    const ConcreteType& rhs = *arg_types_[argno1];
    return TypesAreSame(lhs, rhs, [&] {
      return absl::StrFormat(
          "Want arguments %d and %d to '%s' to be of the same type; got %s vs "
          "%s",
          argno0, argno1, name_, lhs.ToString(), rhs.ToString());
    });
  }

  const absl::Status& status() const { return status_; }

 private:
  ArgTypes arg_types_;
  std::string_view name_;
  const Span& span_;
  absl::Status status_;
  DeduceCtx& deduce_ctx_;
};

}  // namespace

// TODO(leary): 2019-12-12 These *could* be automatically made by interpreting
// the signature string, but just typing in the limited set we use is easier for
// now.
static void PopulateSignatureToLambdaMap(
    absl::flat_hash_map<std::string, SignatureFn>* map_ptr) {
  auto& map = *map_ptr;
  map["(T) -> T"] = [](const SignatureData& data,
                       DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(
        Checker(data.arg_types, data.name, data.span, *ctx).Len(1).status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(uN[T], uN[T]) -> (u1, uN[T])"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .ArgsSameType(0, 1)
                            .status());
    std::vector<std::unique_ptr<ConcreteType>> elements;
    elements.push_back(BitsType::MakeU1());
    elements.push_back(data.arg_types[0]->CloneToUnique());
    auto return_type = std::make_unique<TupleType>(std::move(elements));
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(T[M], uN[N], T[P]) -> T[P]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* a0;
    const ArrayType* a2;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(3)
                       .IsArray(0, &a0)
                       .IsUN(1)
                       .IsArray(2, &a2);
    XLS_RETURN_IF_ERROR(checker.status());
    checker.Eq(a0->element_type(), a2->element_type(), [&] {
      return absl::StrFormat(
          "Element type of argument 0 %s should match element type of argument "
          "2 %s",
          a0->ToString(), a2->ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[2]->CloneToUnique())};
  };
  map["(xN[N], xN[M][N]) -> xN[M]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* a;
    const BitsType* b;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsBits(0, &b)
                       .IsArray(1, &a);
    XLS_RETURN_IF_ERROR(checker.status());
    const ConcreteType& return_type = a->element_type();
    checker.CheckIsBits(return_type, [&] {
      return absl::StrFormat("Want arg 1 element type to be bits; got %s",
                             return_type.ToString());
    });
    XLS_ASSIGN_OR_RETURN(
        int64_t target,
        std::get<InterpValue>(b->size().value()).GetBitValueUint64());
    checker.CheckIsLen(*a, target, [&] {
      return absl::StrFormat("Bit width %d must match %s array size %s", target,
                             a->ToString(), a->size().ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), return_type.CloneToUnique())};
  };
  map["(T, T) -> T"] = [](const SignatureData& data,
                          DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .ArgsSameType(0, 1)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(T, T) -> ()"] = [](const SignatureData& data,
                           DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .ArgsSameType(0, 1)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), ConcreteType::MakeUnit())};
  };
  map["(const uN[N], const uN[N]) -> uN[N][R]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .ArgsSameType(0, 1)
                            .status());
    XLS_ASSIGN_OR_RETURN(InterpValue start, data.constexpr_eval(0));
    XLS_ASSIGN_OR_RETURN(InterpValue limit, data.constexpr_eval(1));
    XLS_ASSIGN_OR_RETURN(int64_t start_int, start.GetBitValueUint64());
    XLS_ASSIGN_OR_RETURN(int64_t limit_int, limit.GetBitValueUint64());
    int64_t length = limit_int - start_int;
    if (length < 0) {
      return TypeInferenceErrorStatus(
          data.span, nullptr,
          absl::StrFormat("Need limit to '%s' to be >= than start value; "
                          "start: %s, limit: %s",
                          data.name, start.ToString(), limit.ToString()));
    }
    auto return_type = std::make_unique<ArrayType>(
        data.arg_types[0]->CloneToUnique(), ConcreteTypeDim::CreateU32(length));
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(T[N], uN[M], T) -> T[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* a;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(3)
                       .IsArray(0, &a)
                       .IsUN(1);
    XLS_RETURN_IF_ERROR(checker.status());
    checker.TypesAreSame(a->element_type(), *data.arg_types[2], [&] {
      return absl::StrFormat(
          "Want argument 0 element type %s to match argument 2 type %s",
          a->element_type().ToString(), data.arg_types[2]->ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(xN[M], xN[N]) -> xN[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsBits(0)
                            .IsBits(1)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["(uN[M], uN[N]) -> ()"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .IsUN(1)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), ConcreteType::MakeUnit())};
  };
  map["(uN[M], uN[N]) -> uN[M+N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .IsUN(1)
                            .status());
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim m,
                         data.arg_types[0]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim n,
                         data.arg_types[1]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim sum, m.Add(n));
    auto return_type =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/sum);
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(uN[N], uN[U], uN[V]) -> uN[V]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(3)
                            .IsUN(0)
                            .IsUN(1)
                            .IsUN(2)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[2]->CloneToUnique())};
  };
  map["(uN[N], uN[U], uN[V]) -> uN[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(3)
                            .IsUN(0)
                            .IsUN(1)
                            .IsUN(2)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(uN[N]) -> uN[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsUN(0)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(T[N]) -> T[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsArray(0)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(uN[N]) -> u1"] = [](const SignatureData& data,
                            DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsUN(0)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), BitsType::MakeU1())};
  };
  map["(u1, T) -> T"] = [](const SignatureData& data,
                           DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsU1(0)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["(u1, T, T) -> T"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(3)
                            .IsU1(0)
                            .ArgsSameType(1, 2)
                            .status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["(uN[N], u1) -> uN[N+1]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .IsU1(1)
                            .status());
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim n,
                         data.arg_types[0]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim np1,
                         n.Add(ConcreteTypeDim::CreateU32(1)));
    auto return_type =
        std::make_unique<BitsType>(/*signed=*/false, /*size=*/np1);
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(T[N]) -> (u32, T)[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* a;
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsArray(0, &a)
                            .status());
    const ConcreteType& t = a->element_type();
    std::vector<std::unique_ptr<ConcreteType>> element_types;
    element_types.push_back(BitsType::MakeU32());
    element_types.push_back(t.CloneToUnique());
    auto e = std::make_unique<TupleType>(std::move(element_types));
    auto return_type = std::make_unique<ArrayType>(std::move(e), a->size());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  // Note: for map's signature we instantiate the (possibly parametric) function
  // argument.
  map["(T[N], (T) -> U) -> U[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* a = nullptr;
    const FunctionType* f = nullptr;
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsArray(0, &a)
                            .IsFn(1, /*argc=*/1, &f)
                            .status());

    const ConcreteType& t = a->element_type();
    std::vector<InstantiateArg> mapped_fn_args;
    mapped_fn_args.push_back(
        InstantiateArg{t.CloneToUnique(), data.arg_spans[0]});

    absl::Span<const ParametricConstraint> mapped_parametric_bindings;
    if (data.parametric_bindings.has_value()) {
      mapped_parametric_bindings = data.parametric_bindings.value();
    }

    // Note that InstantiateFunction will check that the mapped function type
    // lines up with the array (we're providing it the argument types it's being
    // invoked with).
    XLS_ASSIGN_OR_RETURN(
        TypeAndBindings tab,
        InstantiateFunction(
            data.span, *f, mapped_fn_args, ctx,
            /*parametric_constraints=*/mapped_parametric_bindings,
            /*explicit_bindings=*/{}));
    auto return_type =
        std::make_unique<ArrayType>(std::move(tab.type), a->size());
    return TypeAndBindings{
        std::make_unique<FunctionType>(CloneToUnique(data.arg_types),
                                       std::move(return_type)),
        tab.parametric_env};
  };
  map["(u8[N], u1) -> ()"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* array_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsArray(0, &array_type)
                       .IsU1(1);
    checker.Eq(array_type->element_type(), BitsType(false, 8), [&] {
      return absl::StrFormat("Element type of argument 0 %s should be a u8.",
                             array_type->ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), ConcreteType::MakeUnit())};
  };
  map["(u8[N], T) -> T"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const ArrayType* array_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsArray(0, &array_type);
    checker.Eq(array_type->element_type(), BitsType(false, 8), [&] {
      return absl::StrFormat("Element type of argument 0 %s should be a u8.",
                             array_type->ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["(uN[N], uN[N]) -> (uN[N], uN[N])"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const BitsType* lhs_type;
    const BitsType* rhs_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsBits(0, &lhs_type)
                       .IsBits(1, &rhs_type);
    XLS_CHECK(!lhs_type->is_signed());
    XLS_CHECK(!rhs_type->is_signed());
    XLS_RETURN_IF_ERROR(checker.status());

    checker.Eq(*data.arg_types[0], *data.arg_types[1], [&] {
      return absl::StrCat("Elements should have same type, got ",
                          data.arg_types[0]->ToString(), " and ",
                          data.arg_types[1]->ToString());
    });
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim n,
                         data.arg_types[0]->GetTotalBitCount());

    std::vector<std::unique_ptr<ConcreteType>> return_type_elems(2);
    return_type_elems[0] =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/n);
    return_type_elems[1] =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/n);

    auto return_type =
        std::make_unique<TupleType>(std::move(return_type_elems));
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(sN[N], sN[N]) -> (sN[N], sN[N])"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndBindings> {
    const BitsType* lhs_type;
    const BitsType* rhs_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsBits(0, &lhs_type)
                       .IsBits(1, &rhs_type);
    XLS_CHECK(lhs_type->is_signed());
    XLS_CHECK(rhs_type->is_signed());
    XLS_RETURN_IF_ERROR(checker.status());

    checker.Eq(*data.arg_types[0], *data.arg_types[1], [&] {
      return absl::StrCat("Elements should have same type, got ",
                          data.arg_types[0]->ToString(), " and ",
                          data.arg_types[1]->ToString());
    });
    XLS_ASSIGN_OR_RETURN(ConcreteTypeDim n,
                         data.arg_types[0]->GetTotalBitCount());

    std::vector<std::unique_ptr<ConcreteType>> return_type_elems(2);
    return_type_elems[0] =
        std::make_unique<BitsType>(/*is_signed=*/true, /*size=*/n);
    return_type_elems[1] =
        std::make_unique<BitsType>(/*is_signed=*/true, /*size=*/n);

    auto return_type =
        std::make_unique<TupleType>(std::move(return_type_elems));
    return TypeAndBindings{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
}

static const absl::flat_hash_map<std::string, SignatureFn>&
GetSignatureToLambdaMap() {
  static auto* map = ([] {
    auto* map = new absl::flat_hash_map<std::string, SignatureFn>();
    PopulateSignatureToLambdaMap(map);
    return map;
  })();
  return *map;
}

const absl::flat_hash_set<std::string>& GetUnaryParametricBuiltinNames() {
  // Set of unary builtins appropriate as functions - that transform values.
  // TODO(b/144724970): Add enumerate here (and maybe move to ir_converter.py).
  static auto* set = new absl::flat_hash_set<std::string>{"clz", "ctz"};
  return *set;
}

absl::StatusOr<SignatureFn> GetParametricBuiltinSignature(
    std::string_view builtin_name) {
  const absl::flat_hash_map<std::string, BuiltinsData>& parametric_builtins =
      GetParametricBuiltins();
  auto it = parametric_builtins.find(builtin_name);
  bool name_is_builtin_parametric =
      (it == parametric_builtins.end()) ? false : !it->second.is_ast_node;
  if (!name_is_builtin_parametric) {
    return absl::InvalidArgumentError(
        absl::StrFormat("'%s' is not a parametric builtin", builtin_name));
  }
  const std::string& signature = it->second.signature;
  XLS_VLOG(5) << builtin_name << " => " << signature;
  const auto& lambda_map = GetSignatureToLambdaMap();
  auto it2 = lambda_map.find(signature);
  XLS_CHECK(it2 != lambda_map.end()) << signature;
  return it2->second;
}

}  // namespace xls::dslx
