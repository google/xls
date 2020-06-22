// Copyright 2020 Google LLC
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

#ifndef XLS_PASSES_TERNARY_LOGIC_H_
#define XLS_PASSES_TERNARY_LOGIC_H_

#include <iosfwd>
#include <vector>

#include "absl/strings/string_view.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/bits.h"
#include "xls/ir/ternary.h"

namespace xls {

class TernaryEvaluator : public AbstractEvaluator<TernaryValue> {
 public:
  TernaryValue One() const override { return TernaryValue::kKnownOne; }

  TernaryValue Zero() const override { return TernaryValue::kKnownZero; }

  TernaryValue Not(const TernaryValue& input) const override {
    switch (input) {
      case (TernaryValue::kKnownZero):
        return TernaryValue::kKnownOne;
      case (TernaryValue::kKnownOne):
        return TernaryValue::kKnownZero;
      case (TernaryValue::kUnknown):
        return TernaryValue::kUnknown;
    }
  }

  TernaryValue And(const TernaryValue& a,
                   const TernaryValue& b) const override {
    return ternary_ops::And(a, b);
  }

  TernaryValue Or(const TernaryValue& a, const TernaryValue& b) const override {
    return ternary_ops::Or(a, b);
  }
};

}  // namespace xls

#endif  // XLS_PASSES_TERNARY_LOGIC_H_
