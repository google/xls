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

#include "xls/passes/bdd_query_engine.h"

#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"

namespace xls {

/* static */
xabsl::StatusOr<std::unique_ptr<BddQueryEngine>> BddQueryEngine::Run(
    Function* f, int64 minterm_limit) {
  auto query_engine = absl::WrapUnique(new BddQueryEngine(minterm_limit));
  XLS_ASSIGN_OR_RETURN(query_engine->bdd_function_,
                       BddFunction::Run(f, minterm_limit));
  // Construct the Bits objects indication which bit values are statically known
  // for each node and what those values are (0 or 1) if known.
  BinaryDecisionDiagram& bdd = query_engine->bdd();
  for (Node* node : f->nodes()) {
    if (node->GetType()->IsBits()) {
      absl::InlinedVector<bool, 1> known_bits;
      absl::InlinedVector<bool, 1> bits_values;
      for (int64 i = 0; i < node->BitCountOrDie(); ++i) {
        if (query_engine->GetBddNode(BitLocation(node, i)) == bdd.zero()) {
          known_bits.push_back(true);
          bits_values.push_back(false);
        } else if (query_engine->GetBddNode(BitLocation(node, i)) ==
                   bdd.one()) {
          known_bits.push_back(true);
          bits_values.push_back(true);
        } else {
          known_bits.push_back(false);
          bits_values.push_back(false);
        }
      }
      query_engine->known_bits_[node] = Bits(known_bits);
      query_engine->bits_values_[node] = Bits(bits_values);
    }
  }
  return std::move(query_engine);
}

bool BddQueryEngine::AtMostOneTrue(absl::Span<BitLocation const> bits) const {
  BddNodeIndex result = bdd().zero();
  for (int64 i = 0; i < bits.size(); ++i) {
    if (!IsTracked(bits[i].node)) {
      return false;
    }
  }
  // Compute the OR-reduction of a pairwise AND of all bits. If this value is
  // zero then no two bits can be simultaneously true. Equivalently: at most one
  // bit is true.
  for (int64 i = 0; i < bits.size(); ++i) {
    for (int64 j = i + 1; j < bits.size(); ++j) {
      result =
          bdd().Or(result, bdd().And(GetBddNode(bits[i]), GetBddNode(bits[j])));
      if (ExceedsMintermLimit(result)) {
        XLS_VLOG(3) << "AtMostOneTrue exceeded minterm limit of "
                    << minterm_limit_;
        return false;
      }
    }
  }
  return result == bdd().zero();
}

bool BddQueryEngine::AtLeastOneTrue(absl::Span<BitLocation const> bits) const {
  BddNodeIndex result = bdd().zero();
  // At least one bit is true is equivalent to an OR-reduction of all the bits.
  for (const BitLocation& location : bits) {
    if (!IsTracked(location.node)) {
      return false;
    }
    result = bdd().Or(result, GetBddNode(location));
    if (ExceedsMintermLimit(result)) {
      XLS_VLOG(3) << "AtLeastOneTrue exceeded minterm limit of "
                  << minterm_limit_;
      return false;
    }
  }
  return result == bdd().one();
}

bool BddQueryEngine::Implies(const BitLocation& a, const BitLocation& b) const {
  if (!IsTracked(a.node) || !IsTracked(b.node)) {
    return false;
  }
  // A implies B  <=>  !(A && !B)
  return bdd().And(GetBddNode(a), bdd().Not(GetBddNode(b))) == bdd().zero();
}

bool BddQueryEngine::KnownEquals(const BitLocation& a,
                                 const BitLocation& b) const {
  if (!IsTracked(a.node) || !IsTracked(b.node)) {
    return false;
  }
  return GetBddNode(a) == GetBddNode(b);
}

bool BddQueryEngine::KnownNotEquals(const BitLocation& a,
                                    const BitLocation& b) const {
  if (!IsTracked(a.node) || !IsTracked(b.node)) {
    return false;
  }
  return GetBddNode(a) == bdd().Not(GetBddNode(b));
}

}  // namespace xls
