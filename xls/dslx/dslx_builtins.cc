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

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/errors.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_node.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/deduce.h"
#include "xls/dslx/type_system/deduce_ctx.h"
#include "xls/dslx/type_system/parametric_constraint.h"
#include "xls/dslx/type_system/parametric_instantiator.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_and_parametric_env.h"
#include "xls/dslx/type_system/unwrap_meta_type.h"

namespace xls::dslx {
namespace {

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
  // TODO(cdleary): 2023-06-02 Get the spans of all the individual argument
  // expressions for more precise error pinpointing in the resulting messages.
  Checker(absl::Span<const Type* const> arg_types, std::string_view name,
          const Span& span, DeduceCtx& deduce_ctx)
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
  Checker& Eq(const Type& lhs, const Type& rhs,
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
    const Type& t = GetArgType(argno);
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
  Checker& IsToken(int64_t argno) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
    if (auto* tok = dynamic_cast<const TokenType*>(&t); tok != nullptr) {
      return *this;
    }
    status_ = TypeInferenceErrorStatus(
        span_, &t,
        absl::StrFormat("Want argument %d to '%s' to be a token; got %s", argno,
                        name_, t.ToString()));
    return *this;
  }
  Checker& IsRecvChan(int64_t argno, const ChannelType** out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
    if (auto* c = dynamic_cast<const ChannelType*>(&t); c != nullptr) {
      if (c->direction() != ChannelDirection::kIn) {
        status_ = TypeInferenceErrorStatus(
            span_, &t,
            absl::StrFormat(
                "Want argument %d to '%s' to be an 'in' (recv) channel; got %s",
                argno, name_, t.ToString()));
        return *this;
      }
      if (out != nullptr) {
        *out = c;
      }
    } else {
      // Not a channel type.
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to '%s' to be a channel; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& IsSendChan(int64_t argno, const ChannelType** out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
    if (auto* c = dynamic_cast<const ChannelType*>(&t)) {
      if (c->direction() != ChannelDirection::kOut) {
        status_ = TypeInferenceErrorStatus(
            span_, &t,
            absl::StrFormat("Want argument %d to '%s' to be an 'out' (send) "
                            "channel; got %s",
                            argno, name_, t.ToString()));
        return *this;
      }
      if (out != nullptr) {
        *out = c;
      }
    } else {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to '%s' to be a channel; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& IsArray(int64_t argno, const ArrayType** out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
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
    const Type& t = GetArgType(argno);
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
    const Type& t = GetArgType(argno);
    if (auto* a = dynamic_cast<const BitsType*>(&t);
        a == nullptr || a->is_signed()) {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to be unsigned bits; got %s", argno,
                          t.ToString()));
    }
    return *this;
  }
  Checker& IsSN(int64_t argno, const BitsType** type_out = nullptr) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
    auto* a = dynamic_cast<const BitsType*>(&t);
    if (a == nullptr || !a->is_signed()) {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to be signed bits; got %s", argno,
                          t.ToString()));
    }
    if (type_out != nullptr) {
      *type_out = a;
    }
    return *this;
  }
  Checker& IsBool(int64_t argno) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
    if (t != *BitsType::MakeU1()) {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Want argument %d to be a bool; got %s", argno,
                          t.ToString()));
    }
    return *this;
  }
  Checker& CheckIsBits(const Type& t,
                       const std::function<std::string()>& make_msg) {
    if (!status_.ok()) {
      return *this;
    }
    if (auto* bits = dynamic_cast<const BitsType*>(&t); bits == nullptr) {
      status_ = TypeInferenceErrorStatus(span_, &t, make_msg());
    }
    return *this;
  }
  Checker& CheckIsLen(const ArrayType& t, int64_t target,
                      const std::function<std::string()>& make_msg) {
    uint32_t target_u32 = static_cast<uint32_t>(target);
    CHECK_EQ(target_u32, target);
    if (!status_.ok()) {
      return *this;
    }
    if (t.size() != TypeDim::CreateU32(target_u32)) {
      status_ = TypeInferenceErrorStatus(span_, &t, make_msg());
    }
    return *this;
  }
  Checker& IsU1(int64_t argno) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& t = GetArgType(argno);
    if (auto* bits = dynamic_cast<const BitsType*>(&t);
        bits == nullptr || bits->size() != TypeDim::CreateU32(1)) {
      status_ = TypeInferenceErrorStatus(
          span_, &t,
          absl::StrFormat("Expected argument %d to '%s' to be a u1; got %s",
                          argno, name_, t.ToString()));
    }
    return *this;
  }
  Checker& TypesAreSame(const Type& t, const Type& u,
                        const std::function<std::string()>& make_msg) {
    if (!status_.ok()) {
      return *this;
    }
    if (t != u) {
      status_ = deduce_ctx_.TypeMismatchError(span_, nullptr, t, nullptr, u,
                                              make_msg());
    }
    return *this;
  }
  Checker& ArgSameType(int64_t argno, const Type& want) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& got = GetArgType(argno);
    return TypesAreSame(got, want, [&] {
      return absl::StrFormat("Want argument %d to '%s' to have type %s; got %s",
                             argno, name_, want.ToString(), got.ToString());
    });
  }
  Checker& ArgsSameType(int64_t argno0, int64_t argno1) {
    if (!status_.ok()) {
      return *this;
    }
    const Type& lhs = GetArgType(argno0);
    const Type& rhs = GetArgType(argno1);
    return TypesAreSame(lhs, rhs, [&] {
      return absl::StrFormat(
          "Want arguments %d and %d to '%s' to be of the same type; got %s vs "
          "%s",
          argno0, argno1, name_, lhs.ToString(), rhs.ToString());
    });
  }

  const absl::Status& status() const { return status_; }

 private:
  const Type& GetArgType(int64_t argno) const {
    const Type* t = arg_types_.at(argno);
    CHECK(t != nullptr);
    return *t;
  }

  absl::Span<const Type* const> arg_types_;
  std::string_view name_;
  const Span& span_;
  absl::Status status_;
  DeduceCtx& deduce_ctx_;
};

absl::StatusOr<TypeAndParametricEnv> CheckRecvSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  const ChannelType* chan_type = nullptr;
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                     .Len(2)
                     .IsToken(0)
                     .IsRecvChan(1, &chan_type);

  // Note: we can't access chan_type if the checking failed, so we have to
  // check for an error status / early return here.
  XLS_RETURN_IF_ERROR(checker.status());

  auto return_type = TupleType::Create2(
      std::make_unique<TokenType>(), chan_type->payload_type().CloneToUnique());
  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::move(return_type))};
}

absl::StatusOr<TypeAndParametricEnv> CheckRecvNonBlockingSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  const ChannelType* chan_type = nullptr;
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                     .Len(3)
                     .IsToken(0)
                     .IsRecvChan(1, &chan_type);

  // Note: we can't access chan_type if the checking failed, so we have to
  // check for an error status / early return here.
  XLS_RETURN_IF_ERROR(checker.status());

  checker.ArgSameType(2, chan_type->payload_type());
  XLS_RETURN_IF_ERROR(checker.status());

  auto return_type = TupleType::Create3(
      std::make_unique<TokenType>(), chan_type->payload_type().CloneToUnique(),
      BitsType::MakeU1());
  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::move(return_type))};
}

absl::StatusOr<TypeAndParametricEnv> CheckRecvIfSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  const ChannelType* chan_type = nullptr;
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                     .Len(4)
                     .IsToken(0)
                     .IsRecvChan(1, &chan_type)
                     .IsBool(2);

  // Note: we can't access chan_type if the checking failed, so we have to
  // check for an error status / early return here.
  XLS_RETURN_IF_ERROR(checker.status());

  checker.ArgSameType(3, chan_type->payload_type());
  XLS_RETURN_IF_ERROR(checker.status());

  auto return_type = TupleType::Create2(
      std::make_unique<TokenType>(), chan_type->payload_type().CloneToUnique());
  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::move(return_type))};
}

absl::StatusOr<TypeAndParametricEnv> CheckRecvIfNonBlockingSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  const ChannelType* chan_type = nullptr;
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                     .Len(4)
                     .IsToken(0)
                     .IsRecvChan(1, &chan_type)
                     .IsBool(2);

  // Note: we can't access chan_type if the checking failed, so we have to
  // check for an error status / early return here.
  XLS_RETURN_IF_ERROR(checker.status());

  checker.ArgSameType(3, chan_type->payload_type());
  XLS_RETURN_IF_ERROR(checker.status());

  auto return_type = TupleType::Create3(
      std::make_unique<TokenType>(), chan_type->payload_type().CloneToUnique(),
      BitsType::MakeU1());
  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::move(return_type))};
}

absl::StatusOr<TypeAndParametricEnv> CheckSendSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  const ChannelType* chan_type = nullptr;
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                     .Len(3)
                     .IsToken(0)
                     .IsSendChan(1, &chan_type);

  // Note: we can't access chan_type if the checking failed, so we have to
  // check for an error status / early return here.
  XLS_RETURN_IF_ERROR(checker.status());

  checker.ArgSameType(2, chan_type->payload_type());
  XLS_RETURN_IF_ERROR(checker.status());

  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::make_unique<TokenType>())};
}

absl::StatusOr<TypeAndParametricEnv> CheckSendIfSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  const ChannelType* chan_type = nullptr;
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                     .Len(4)
                     .IsToken(0)
                     .IsSendChan(1, &chan_type)
                     .IsBool(2);

  // Note: we can't access chan_type if the checking failed, so we have to
  // check for an error status / early return here.
  XLS_RETURN_IF_ERROR(checker.status());

  checker.ArgSameType(3, chan_type->payload_type());
  XLS_RETURN_IF_ERROR(checker.status());

  auto return_type = std::make_unique<TokenType>();
  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::move(return_type))};
}

absl::StatusOr<TypeAndParametricEnv> CheckJoinSignature(
    const SignatureData& data, DeduceCtx* ctx) {
  auto checker = Checker(data.arg_types, data.name, data.span, *ctx);
  for (size_t i = 0; i < data.arg_types.size(); ++i) {
    checker.IsToken(i);
  }
  XLS_RETURN_IF_ERROR(checker.status());
  auto return_type = std::make_unique<TokenType>();
  return TypeAndParametricEnv{std::make_unique<FunctionType>(
      CloneToUnique(data.arg_types), std::move(return_type))};
}

static void AddUnaryArbitraryTypeIdentitySignature(
    absl::flat_hash_map<std::string, SignatureFn>& map) {
  map["(T) -> T"] = [](const SignatureData& data,
                       DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(
        Checker(data.arg_types, data.name, data.span, *ctx).Len(1).status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
}

static void AddBinaryArbitraryTypeSignature(
    absl::flat_hash_map<std::string, SignatureFn>& map) {
  map["(T, T) -> T"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .ArgsSameType(0, 1)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
}

static void AddBinaryArbitrarySignToUnitSignature(
    absl::flat_hash_map<std::string, SignatureFn>& map) {
  map["(xN[N], xN[N]) -> ()"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsBits(0)
                       .IsBits(1)
                       .ArgsSameType(0, 1);
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), Type::MakeUnit())};
  };
}

static void AddByteArrayAndTProducesTSignature(
    absl::flat_hash_map<std::string, SignatureFn>& map) {
  map["(u8[N], T) -> T"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const ArrayType* array_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsArray(0, &array_type);
    // Need to ensure we didn't get an error before we use the array type.
    XLS_RETURN_IF_ERROR(checker.status());
    checker.Eq(array_type->element_type(), BitsType(false, 8), [&] {
      return absl::StrFormat("Element type of argument 0 %s should be a u8.",
                             array_type->ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
}

static void AddZipLikeSignature(
    absl::flat_hash_map<std::string, SignatureFn>& map) {
  map["(T[N], U[N]) -> (T, U)[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const ArrayType* t_array;
    const ArrayType* u_array;
    Checker checker(data.arg_types, data.name, data.span, *ctx);
    XLS_RETURN_IF_ERROR(
        checker.Len(2).IsArray(0, &t_array).IsArray(1, &u_array).status());
    const Type& t = t_array->element_type();
    const Type& u = u_array->element_type();

    XLS_ASSIGN_OR_RETURN(
        int64_t size,
        std::get<InterpValue>(t_array->size().value()).GetBitValueViaSign());
    XLS_RETURN_IF_ERROR(
        checker
            .CheckIsLen(
                *u_array, size,
                [&] {
                  return absl::StrFormat(
                      "Array size of %s must match array size of %s (%d)",
                      t_array->ToString(), u_array->ToString(), size);
                })
            .status());

    std::vector<std::unique_ptr<Type>> element_types;
    element_types.push_back(t.CloneToUnique());
    element_types.push_back(u.CloneToUnique());
    auto e = std::make_unique<TupleType>(std::move(element_types));
    auto return_type =
        std::make_unique<ArrayType>(std::move(e), t_array->size());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
}

static absl::flat_hash_map<std::string, SignatureFn>
PopulateSignatureToLambdaMap() {
  absl::flat_hash_map<std::string, SignatureFn> map;

  // Note: we start to break some of these out to helper functions as they run
  // afoul of over-long function lint. (It at least gives us the opportunity to
  // put some slightly more descriptive names for the signature in the C++
  // function name.)
  AddBinaryArbitrarySignToUnitSignature(map);
  AddUnaryArbitraryTypeIdentitySignature(map);
  AddBinaryArbitraryTypeSignature(map);
  AddByteArrayAndTProducesTSignature(map);
  AddZipLikeSignature(map);

  map["(uN[T], uN[T]) -> (u1, uN[T])"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .ArgsSameType(0, 1)
                            .status());
    std::vector<std::unique_ptr<Type>> elements;
    elements.push_back(BitsType::MakeU1());
    elements.push_back(data.arg_types[0]->CloneToUnique());
    auto return_type = std::make_unique<TupleType>(std::move(elements));
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(T[M], uN[N], T[P]) -> T[P]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
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
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[2]->CloneToUnique())};
  };
  map["(xN[N], xN[M][N]) -> xN[M]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const ArrayType* a;
    const BitsType* b;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsBits(0, &b)
                       .IsArray(1, &a);
    XLS_RETURN_IF_ERROR(checker.status());
    const Type& return_type = a->element_type();
    checker.CheckIsBits(return_type, [&] {
      return absl::StrFormat("Want arg 1 element type to be bits; got %s",
                             return_type.ToString());
    });
    XLS_ASSIGN_OR_RETURN(
        int64_t target,
        std::get<InterpValue>(b->size().value()).GetBitValueViaSign());
    checker.CheckIsLen(*a, target, [&] {
      return absl::StrFormat("Bit width %d must match %s array size %s", target,
                             a->ToString(), a->size().ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), return_type.CloneToUnique())};
  };
  map["(T, T) -> ()"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .ArgsSameType(0, 1)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), Type::MakeUnit())};
  };
  map["<U>(T) -> U"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(
        Checker(data.arg_types, data.name, data.span, *ctx).Len(1).status());

    if (data.arg_explicit_parametrics.size() != 1) {
      return ArgCountMismatchErrorStatus(
          data.span,
          absl::StrFormat("Invalid number of parametrics passed to '%s', "
                          "expected 1, got %d",
                          data.name, data.arg_explicit_parametrics.size()));
    }

    ExprOrType param_type = data.arg_explicit_parametrics.at(0);
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> return_type,
                         DeduceAndResolve(ToAstNode(param_type), ctx));
    XLS_ASSIGN_OR_RETURN(return_type, UnwrapMetaType(std::move(return_type),
                                                     data.span, data.name));

    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(const uN[N], const uN[N]) -> uN[N][R]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .ArgsSameType(0, 1)
                            .status());
    XLS_ASSIGN_OR_RETURN(InterpValue start, data.constexpr_eval(0));
    XLS_ASSIGN_OR_RETURN(InterpValue limit, data.constexpr_eval(1));
    XLS_ASSIGN_OR_RETURN(int64_t start_int, start.GetBitValueViaSign());
    XLS_ASSIGN_OR_RETURN(int64_t limit_int, limit.GetBitValueViaSign());
    int64_t length = limit_int - start_int;
    if (length < 0) {
      return TypeInferenceErrorStatus(
          data.span, nullptr,
          absl::StrFormat("Need limit to '%s' to be >= than start value; "
                          "start: %s, limit: %s",
                          data.name, start.ToString(), limit.ToString()));
    }
    XLS_RET_CHECK_EQ(static_cast<uint32_t>(length), length);
    auto return_type = std::make_unique<ArrayType>(
        data.arg_types[0]->CloneToUnique(),
        TypeDim::CreateU32(static_cast<uint32_t>(length)));
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(T[N]) -> u32"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    auto checker =
        Checker(data.arg_types, data.name, data.span, *ctx).Len(1).IsArray(0);
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), BitsType::MakeU32())};
  };
  map["(T[N], uN[M], T) -> T[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
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
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(xN[M], xN[N]) -> xN[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsBits(0)
                            .IsBits(1)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["(uN[M], uN[N]) -> ()"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .IsUN(1)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), Type::MakeUnit())};
  };
  map["(uN[M], uN[N]) -> uN[M+N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .IsUN(1)
                            .status());
    XLS_ASSIGN_OR_RETURN(TypeDim m, data.arg_types[0]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(TypeDim n, data.arg_types[1]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(TypeDim sum, m.Add(n));
    auto return_type =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/sum);
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(uN[N], uN[U], uN[V]) -> uN[V]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(3)
                            .IsUN(0)
                            .IsUN(1)
                            .IsUN(2)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[2]->CloneToUnique())};
  };
  map["(uN[N], uN[U], uN[V]) -> uN[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(3)
                            .IsUN(0)
                            .IsUN(1)
                            .IsUN(2)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(uN[N]) -> uN[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsUN(0)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(T[N]) -> T[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsArray(0)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[0]->CloneToUnique())};
  };
  map["(uN[N]) -> u1"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsUN(0)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), BitsType::MakeU1())};
  };
  map["(u1, T) -> T"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsU1(0)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["(u1, T, T) -> T"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(3)
                            .IsU1(0)
                            .ArgsSameType(1, 2)
                            .status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), data.arg_types[1]->CloneToUnique())};
  };
  map["<uN[M]>(uN[N]) -> uN[M]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsUN(0)
                            .status());

    if (data.arg_explicit_parametrics.size() != 1) {
      return ArgCountMismatchErrorStatus(
          data.span,
          absl::StrFormat("Invalid number of parametrics passed to '%s', "
                          "expected 1, got %d",
                          data.name, data.arg_explicit_parametrics.size()));
    }
    AstNode* param_type = ToAstNode(data.arg_explicit_parametrics.at(0));
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<Type> return_type,
                         DeduceAndResolve(param_type, ctx));
    XLS_ASSIGN_OR_RETURN(return_type, UnwrapMetaType(std::move(return_type),
                                                     data.span, data.name));
    if (auto* a = dynamic_cast<const BitsType*>(return_type.get());
        a == nullptr || a->is_signed()) {
      return TypeInferenceErrorStatus(
          param_type->GetSpan().value_or(FakeSpan()), return_type.get(),
          absl::StrFormat("Want return type to be unsigned bits; got %s",
                          return_type->ToString()));
    }

    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(uN[N]) -> uN[ceil(log2(N))]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsUN(0)
                            .status());
    XLS_ASSIGN_OR_RETURN(TypeDim n, data.arg_types[0]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(TypeDim log2_n, n.CeilOfLog2());
    auto return_type =
        std::make_unique<BitsType>(/*signed=*/false, /*size=*/log2_n);
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(uN[N], u1) -> uN[N+1]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsUN(0)
                            .IsU1(1)
                            .status());
    XLS_ASSIGN_OR_RETURN(TypeDim n, data.arg_types[0]->GetTotalBitCount());
    XLS_ASSIGN_OR_RETURN(TypeDim np1, n.Add(TypeDim::CreateU32(1)));
    auto return_type =
        std::make_unique<BitsType>(/*signed=*/false, /*size=*/np1);
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(T[N]) -> (u32, T)[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const ArrayType* a;
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(1)
                            .IsArray(0, &a)
                            .status());
    const Type& t = a->element_type();
    std::vector<std::unique_ptr<Type>> element_types;
    element_types.push_back(BitsType::MakeU32());
    element_types.push_back(t.CloneToUnique());
    auto e = std::make_unique<TupleType>(std::move(element_types));
    auto return_type = std::make_unique<ArrayType>(std::move(e), a->size());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  // Note: for map's signature we instantiate the (possibly parametric) function
  // argument.
  map["(T[N], (T) -> U) -> U[N]"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const ArrayType* a = nullptr;
    const FunctionType* f_type = nullptr;
    XLS_RETURN_IF_ERROR(Checker(data.arg_types, data.name, data.span, *ctx)
                            .Len(2)
                            .IsArray(0, &a)
                            .IsFn(1, /*argc=*/1, &f_type)
                            .status());

    const Type& t = a->element_type();
    std::vector<InstantiateArg> mapped_fn_args;
    mapped_fn_args.push_back(
        InstantiateArg{t.CloneToUnique(), data.arg_spans[0]});

    absl::Span<const ParametricConstraint> mapped_parametric_bindings;
    if (data.parametric_bindings.has_value()) {
      mapped_parametric_bindings = data.parametric_bindings.value();
    }

    Expr* fn_expr = data.args.at(1);
    NameRef* fn_name = dynamic_cast<NameRef*>(fn_expr);
    XLS_RET_CHECK(fn_name != nullptr);
    AstNode* fn_ast_node = fn_name->GetDefiner();
    auto* fn = dynamic_cast<Function*>(fn_ast_node);
    XLS_RET_CHECK(fn != nullptr);

    // Note that InstantiateFunction will check that the mapped function type
    // lines up with the array (we're providing it the argument types it's being
    // invoked with).
    XLS_ASSIGN_OR_RETURN(
        TypeAndParametricEnv tab,
        InstantiateFunction(
            data.span, *fn, *f_type, mapped_fn_args, ctx,
            /*parametric_constraints=*/mapped_parametric_bindings,
            /*explicit_bindings=*/{}));
    auto return_type =
        std::make_unique<ArrayType>(std::move(tab.type), a->size());
    return TypeAndParametricEnv{
        std::make_unique<FunctionType>(CloneToUnique(data.arg_types),
                                       std::move(return_type)),
        tab.parametric_env};
  };
  map["(u8[N], u1) -> ()"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const ArrayType* array_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsArray(0, &array_type)
                       .IsU1(1);
    // Need to ensure we didn't get an error before we use the array type.
    XLS_RETURN_IF_ERROR(checker.status());
    checker.Eq(array_type->element_type(), BitsType(false, 8), [&] {
      return absl::StrFormat("Element type of argument 0 %s should be a u8.",
                             array_type->ToString());
    });
    XLS_RETURN_IF_ERROR(checker.status());
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), Type::MakeUnit())};
  };
  map["(uN[N], uN[N]) -> (uN[N], uN[N])"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const BitsType* lhs_type;
    const BitsType* rhs_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsBits(0, &lhs_type)
                       .IsBits(1, &rhs_type);
    XLS_RETURN_IF_ERROR(checker.status());

    checker.Eq(*data.arg_types[0], *data.arg_types[1], [&] {
      return absl::StrCat("Elements should have same type, got ",
                          data.arg_types[0]->ToString(), " and ",
                          data.arg_types[1]->ToString());
    });
    XLS_ASSIGN_OR_RETURN(TypeDim n, data.arg_types[0]->GetTotalBitCount());

    std::vector<std::unique_ptr<Type>> return_type_elems(2);
    return_type_elems[0] =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/n);
    return_type_elems[1] =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/n);

    auto return_type =
        std::make_unique<TupleType>(std::move(return_type_elems));
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  map["(sN[N], sN[N]) -> (uN[N], uN[N])"] =
      [](const SignatureData& data,
         DeduceCtx* ctx) -> absl::StatusOr<TypeAndParametricEnv> {
    const BitsType* lhs_type;
    const BitsType* rhs_type;
    auto checker = Checker(data.arg_types, data.name, data.span, *ctx)
                       .Len(2)
                       .IsSN(0, &lhs_type)
                       .IsSN(1, &rhs_type);
    XLS_RETURN_IF_ERROR(checker.status());

    checker.Eq(*data.arg_types[0], *data.arg_types[1], [&] {
      return absl::StrCat("Elements should have same type, got ",
                          data.arg_types[0]->ToString(), " and ",
                          data.arg_types[1]->ToString());
    });
    XLS_ASSIGN_OR_RETURN(TypeDim n, data.arg_types[0]->GetTotalBitCount());

    std::vector<std::unique_ptr<Type>> return_type_elems(2);
    return_type_elems[0] =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/n);
    return_type_elems[1] =
        std::make_unique<BitsType>(/*is_signed=*/false, /*size=*/n);

    auto return_type =
        std::make_unique<TupleType>(std::move(return_type_elems));
    return TypeAndParametricEnv{std::make_unique<FunctionType>(
        CloneToUnique(data.arg_types), std::move(return_type))};
  };
  // recv
  map["(token, recv_chan<T>) -> (token, T)"] = CheckRecvSignature;
  // recv_non_blocking
  map["(token, recv_chan<T>, T) -> (token, T, bool)"] =
      CheckRecvNonBlockingSignature;
  // recv_if
  map["(token, recv_chan<T>, bool, T) -> (token, T)"] = CheckRecvIfSignature;
  // recv_if_non_blocking
  map["(token, recv_chan<T>, bool, T) -> (token, T, bool)"] =
      CheckRecvIfNonBlockingSignature;
  // send
  map["(token, send_chan<T>, T) -> token"] = CheckSendSignature;
  // send_if
  map["(token, send_chan<T>, bool, T) -> token"] = CheckSendIfSignature;
  // join
  map["(token...) -> token"] = CheckJoinSignature;
  return map;
}

const absl::flat_hash_map<std::string, SignatureFn>& GetSignatureToLambdaMap() {
  static const absl::NoDestructor<absl::flat_hash_map<std::string, SignatureFn>>
      map(PopulateSignatureToLambdaMap());
  return *map;
}

}  // namespace

// TODO(leary): 2019-12-12 These *could* be automatically made by interpreting
// the signature string, but just typing in the limited set we use is easier for
// now.
const absl::flat_hash_set<std::string>& GetUnaryParametricBuiltinNames() {
  // Set of unary builtins appropriate as functions - that transform values.
  // TODO(b/144724970): Add enumerate here (and maybe move to ir_converter.py).
  static const absl::NoDestructor<absl::flat_hash_set<std::string>> set(
      {"clz", "ctz"});
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
  CHECK(it2 != lambda_map.end()) << signature;
  return it2->second;
}

}  // namespace xls::dslx
