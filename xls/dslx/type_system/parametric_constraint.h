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

#ifndef XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_CONSTRAINT_H_
#define XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_CONSTRAINT_H_

#include <memory>
#include <string>

#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/type_system/concrete_type.h"

namespace xls::dslx {

// Describes an argument being presented for instantiation (of a parametric
// function or struct) -- these argument expressions have types and come from
// some originating span, which is used for error reporting.
//
// Note that both *function* instantiation and *struct* instantiation
// conceptually have "argument" values given, with the 'actual' types, that are
// filling in the (possibly parametric) slots of the formal (declared) types --
// the formal types may be parametric.
struct InstantiateArg {
  std::unique_ptr<ConcreteType> type;
  const Span span;
};

// Decorates a parametric binding with its (deduced) ConcreteType.
//
// These are provided as inputs to parametric instantiation functions in
// `xls/dslx/type_system/parametric_instantiator.h`.
class ParametricConstraint {
 public:
  // Decorates the given "binding" with the provided type information.
  ParametricConstraint(const ParametricBinding& binding,
                       std::unique_ptr<ConcreteType> type);

  // Decorates the given "binding" with the type information as above, but
  // exposes the (replacement) expression "expr".
  ParametricConstraint(const ParametricBinding& binding,
                       std::unique_ptr<ConcreteType> type, Expr* expr);

  const std::string& identifier() const { return binding_->identifier(); }
  const ConcreteType& type() const { return *type_; }
  Expr* expr() const { return expr_; }

  std::string ToString() const;

 private:
  const ParametricBinding* binding_;
  std::unique_ptr<ConcreteType> type_;

  // Expression that the parametric value should take on (e.g. when there are
  // "derived parametrics" that are computed from other parametric values). Note
  // that this may be null.
  Expr* expr_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_PARAMETRIC_CONSTRAINT_H_
