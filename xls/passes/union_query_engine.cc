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

#include "xls/passes/union_query_engine.h"

#include "xls/ir/bits_ops.h"

namespace xls {

absl::StatusOr<std::unique_ptr<UnionQueryEngine>> UnionQueryEngine::Run(
    std::vector<std::unique_ptr<QueryEngine>> engines) {
  UnionQueryEngine result;
  result.engines_ = std::move(engines);
  return std::make_unique<UnionQueryEngine>(std::move(result));
}

bool UnionQueryEngine::IsTracked(Node* node) const {
  for (const auto& engine : engines_) {
    if (engine->IsTracked(node)) {
      return true;
    }
  }
  return false;
}

const Bits& UnionQueryEngine::GetKnownBits(Node* node) const {
  if (known_bits_.contains(node)) {
    return known_bits_.at(node);
  }

  Bits known(node->GetType()->GetFlatBitCount());
  for (const auto& engine : engines_) {
    if (engine->IsTracked(node)) {
      known = bits_ops::Or(known, engine->GetKnownBits(node));
    }
  }

  const_cast<UnionQueryEngine*>(this)->known_bits_[node] = known;

  return known_bits_.at(node);
}

const Bits& UnionQueryEngine::GetKnownBitsValues(Node* node) const {
  if (known_bit_values_.contains(node)) {
    return known_bit_values_.at(node);
  }

  Bits known_values(node->GetType()->GetFlatBitCount());
  for (const auto& engine : engines_) {
    // TODO(taktoa): check for inconsistencies between engines
    if (engine->IsTracked(node)) {
      known_values = bits_ops::Or(
          known_values, bits_ops::And(engine->GetKnownBits(node),
                                      engine->GetKnownBitsValues(node)));
    }
  }

  const_cast<UnionQueryEngine*>(this)->known_bit_values_[node] = known_values;

  return known_bit_values_.at(node);
}

bool UnionQueryEngine::AtMostOneTrue(absl::Span<BitLocation const> bits) const {
  for (const auto& engine : engines_) {
    if (engine->AtMostOneTrue(bits)) {
      return true;
    }
  }
  return false;
}

bool UnionQueryEngine::AtLeastOneTrue(
    absl::Span<BitLocation const> bits) const {
  for (const auto& engine : engines_) {
    if (engine->AtLeastOneTrue(bits)) {
      return true;
    }
  }
  return false;
}

bool UnionQueryEngine::KnownEquals(const BitLocation& a,
                                   const BitLocation& b) const {
  for (const auto& engine : engines_) {
    if (engine->KnownEquals(a, b)) {
      return true;
    }
  }
  return false;
}

bool UnionQueryEngine::KnownNotEquals(const BitLocation& a,
                                      const BitLocation& b) const {
  for (const auto& engine : engines_) {
    if (engine->KnownNotEquals(a, b)) {
      return true;
    }
  }
  return false;
}

bool UnionQueryEngine::Implies(const BitLocation& a,
                               const BitLocation& b) const {
  for (const auto& engine : engines_) {
    if (engine->Implies(a, b)) {
      return true;
    }
  }
  return false;
}

absl::optional<Bits> UnionQueryEngine::ImpliedNodeValue(
    absl::Span<const std::pair<BitLocation, bool>> predicate_bit_values,
    Node* node) const {
  for (const auto& engine : engines_) {
    if (auto i = engine->ImpliedNodeValue(predicate_bit_values, node)) {
      return i;
    }
  }
  return absl::nullopt;
}

}  // namespace xls
