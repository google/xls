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

// Sub-AST used for representing parametric type expressions.

#ifndef XLS_DSLX_PARAMETRIC_EXPRESSION_H_
#define XLS_DSLX_PARAMETRIC_EXPRESSION_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/pos.h"

namespace xls::dslx {

// Abstract base class for a parametric dimension expression.
//
// Parametric dimension expressions can be evaluated with an environment. For
// example, with the parametric type:
//
//     bits[M+N]
//
// If we evalute the parametric expression M+N with an environment {M: 3, N; 7}
// we get:
//
//     bits[10]
class ParametricExpression {
 public:
  using Evaluated =
      absl::variant<InterpValue, std::unique_ptr<ParametricExpression>>;
  using EnvValue = absl::variant<const ParametricExpression*, InterpValue>;
  using Env = absl::flat_hash_map<std::string, EnvValue>;

  virtual ~ParametricExpression() = default;

  virtual std::string ToRepr() const = 0;
  virtual std::string ToString() const = 0;
  virtual absl::flat_hash_set<std::string> GetFreeVariables() const = 0;
  virtual Evaluated Evaluate(const Env& env) const = 0;
  virtual bool operator==(const ParametricExpression& other) const = 0;
  bool operator!=(const ParametricExpression& other) const {
    return !(*this == other);
  }
  virtual std::unique_ptr<ParametricExpression> Clone() const = 0;

  // Adds together two parametric expression environment values.
  static std::unique_ptr<ParametricExpression> Add(const EnvValue& lhs,
                                                   const EnvValue& rhs);

  // Multiplies together two parametric expression environment values.
  static std::unique_ptr<ParametricExpression> Mul(const EnvValue& lhs,
                                                   const EnvValue& rhs);

  // Converts the environment value to an owned parametric expression.
  static std::unique_ptr<ParametricExpression> ToOwned(const EnvValue& operand);

  // Converts the environment value to an evaluated value.
  static Evaluated ToEvaluated(const EnvValue& value);

  // Converts the evaluated value into an environment value (unowned).
  //
  // Note that the unique_ptr within Evaluated should outlive the returned
  // environment variable reference (borrowed pointer).
  static EnvValue ToEnvValue(const Evaluated& v);

  // If e is a constant, unwraps it into an integral value, otherwise returns
  // the parametric expression as-is.
  static Evaluated TryUnwrapConstant(std::unique_ptr<ParametricExpression> e);
};

// Represents a constant value in a parametric dimension expression.
//
// For example, when you do:
//
//    bits[1]:0 ++ bits[N]:x
//
// It produces a parametric expression of type:
//
//    bits[1+N]
//
// Where the '1' is a parametric constant.
class ParametricConstant : public ParametricExpression {
 public:
  explicit ParametricConstant(InterpValue value) : value_(value) {}

  std::string ToString() const override { return value_.ToString(); }
  std::string ToRepr() const override {
    return absl::StrFormat("ParametricConstant(%s)", value_.ToString());
  }
  bool operator==(const ParametricExpression& other) const override {
    if (auto* o = dynamic_cast<const ParametricConstant*>(&other)) {
      return value_ == o->value_;
    }
    return false;
  }

  Evaluated Evaluate(const Env& env) const override { return value_; }
  absl::flat_hash_set<std::string> GetFreeVariables() const override {
    return {};
  }

  std::unique_ptr<ParametricExpression> Clone() const override {
    return absl::make_unique<ParametricConstant>(value_);
  }

  InterpValue value() const { return value_; }

 private:
  InterpValue value_;
};

// Represents an add in a parametric dimension expression.
class ParametricAdd : public ParametricExpression {
 public:
  ParametricAdd(std::unique_ptr<ParametricExpression> lhs,
                std::unique_ptr<ParametricExpression> rhs)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  std::string ToString() const override {
    return absl::StrFormat("(%s+%s)", lhs_->ToString(), rhs_->ToString());
  }
  std::string ToRepr() const override {
    return absl::StrFormat("ParametricAdd(%s, %s)", lhs_->ToRepr(),
                           rhs_->ToRepr());
  }

  Evaluated Evaluate(const Env& env) const override {
    Evaluated lhs = lhs_->Evaluate(env);
    Evaluated rhs = rhs_->Evaluate(env);
    return TryUnwrapConstant(
        ParametricExpression::Add(ToEnvValue(lhs), ToEnvValue(rhs)));
  }

  absl::flat_hash_set<std::string> GetFreeVariables() const override {
    absl::flat_hash_set<std::string> result = lhs_->GetFreeVariables();
    for (std::string fv : rhs_->GetFreeVariables()) {
      result.insert(std::move(fv));
    }
    return result;
  }

  bool operator==(const ParametricExpression& other) const override {
    if (auto* o = dynamic_cast<const ParametricAdd*>(&other)) {
      return *lhs_ == *o->lhs_ && *rhs_ == *o->rhs_;
    }
    return false;
  }

  std::unique_ptr<ParametricExpression> Clone() const override {
    return absl::make_unique<ParametricAdd>(lhs_->Clone(), rhs_->Clone());
  }

 private:
  std::unique_ptr<ParametricExpression> lhs_;
  std::unique_ptr<ParametricExpression> rhs_;
};

// Represents a multiplication in a parametric dimension expression.
class ParametricMul : public ParametricExpression {
 public:
  ParametricMul(std::unique_ptr<ParametricExpression> lhs,
                std::unique_ptr<ParametricExpression> rhs)
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  Evaluated Evaluate(const Env& env) const override {
    Evaluated lhs = lhs_->Evaluate(env);
    Evaluated rhs = rhs_->Evaluate(env);
    return TryUnwrapConstant(
        ParametricExpression::Mul(ToEnvValue(lhs), ToEnvValue(rhs)));
  }

  absl::flat_hash_set<std::string> GetFreeVariables() const override {
    absl::flat_hash_set<std::string> result = lhs_->GetFreeVariables();
    for (std::string fv : rhs_->GetFreeVariables()) {
      result.insert(std::move(fv));
    }
    return result;
  }

  std::unique_ptr<ParametricExpression> Clone() const override {
    return absl::make_unique<ParametricMul>(lhs_->Clone(), rhs_->Clone());
  }

  bool operator==(const ParametricExpression& other) const override {
    if (auto* o = dynamic_cast<const ParametricMul*>(&other)) {
      return *lhs_ == *o->lhs_ && *rhs_ == *o->rhs_;
    }
    return false;
  }

  std::string ToString() const override {
    return absl::StrFormat("(%s*%s)", lhs_->ToString(), rhs_->ToString());
  }
  std::string ToRepr() const override {
    return absl::StrFormat("ParametricMul(%s, %s)", lhs_->ToRepr(),
                           rhs_->ToRepr());
  }

 private:
  std::unique_ptr<ParametricExpression> lhs_;
  std::unique_ptr<ParametricExpression> rhs_;
};

// Represents a symbol in a parametric dimension expression.
//
// For example, in the expression:
//
//     bits[M+N+1]
//
// Both M and N are parametric symbols.
class ParametricSymbol : public ParametricExpression {
 public:
  ParametricSymbol(std::string identifier, Span span)
      : identifier_(std::move(identifier)), span_(std::move(span)) {}

  std::string ToString() const override { return identifier_; }
  std::string ToRepr() const override {
    return absl::StrFormat("ParametricSymbol(\"%s\")", identifier_);
  }
  bool operator==(const ParametricExpression& other) const override {
    if (auto* o = dynamic_cast<const ParametricSymbol*>(&other)) {
      return identifier_ == o->identifier_;
    }
    return false;
  }

  Evaluated Evaluate(const Env& env) const override {
    auto it = env.find(identifier_);
    if (it == env.end()) {
      return Clone();
    }
    return ToEvaluated(it->second);
  }

  absl::flat_hash_set<std::string> GetFreeVariables() const override {
    return {identifier_};
  }

  std::unique_ptr<ParametricExpression> Clone() const override {
    return absl::make_unique<ParametricSymbol>(identifier_, span_);
  }

  const std::string& identifier() const { return identifier_; }

 private:
  std::string identifier_;  // Text identifier for the parametric symbol.
  Span span_;  // Span in the source text where this parametric symbol resides.
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_PARAMETRIC_EXPRESSION_H_
