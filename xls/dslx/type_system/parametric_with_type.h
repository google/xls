// Copyright 2023 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_WITH_TYPE_H_
#define XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_WITH_TYPE_H_

#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Describes an argument being presented for instantiation (of a parametric
// function or struct) -- these argument expressions have types and come from
// some originating span, which is used for error reporting.
//
// Note that both *function* instantiation and *struct* instantiation
// conceptually have "argument" values given, with the 'actual' types, that are
// filling in the (possibly parametric) slots of the formal (declared) types --
// the formal types may be parametric.
class InstantiateArg {
 public:
  InstantiateArg(std::unique_ptr<Type> type, Span span)
      : type_(std::move(type)), span_(std::move(span)) {
    // This should be the type of the expression argument so it should not be a
    // metatype.
    CHECK(!type_->IsMeta());
  }

  std::unique_ptr<Type>& type() { return type_; }
  const std::unique_ptr<Type>& type() const { return type_; }
  const Span& span() const { return span_; }

 private:
  std::unique_ptr<Type> type_;
  const Span span_;
};

// Decorates a parametric binding with its (deduced) Type.
//
// These are provided as inputs to parametric instantiation functions in
// `xls/dslx/type_system/parametric_instantiator.h`.
class ParametricWithType {
 public:
  // Decorates the given "binding" with the provided type information.
  ParametricWithType(const ParametricBinding& binding,
                     std::unique_ptr<Type> type);

  std::string_view identifier() const { return binding_.identifier(); }
  const Type& type() const { return *type_; }

  // Expr on the parametric binding -- note that this can be null when the
  // parametric has no default expression.
  Expr* ABSL_NULLABLE expr() const { return binding_.expr(); }

  std::string ToString() const;

 private:
  const ParametricBinding& binding_;
  std::unique_ptr<Type> type_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_WITH_TYPE_H_
