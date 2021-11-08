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

#include "xls/passes/query_engine.h"

#include "xls/common/logging/logging.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/ternary.h"

namespace xls {
namespace {

// Converts the bits of the given node into a vector of BitLocations.
std::vector<TreeBitLocation> ToTreeBitLocations(Node* node) {
  XLS_CHECK(node->GetType()->IsBits());
  std::vector<TreeBitLocation> locations;
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    locations.emplace_back(TreeBitLocation(node, i));
  }
  return locations;
}

// Converts the single-bit Nodes in preds into a vector of BitLocations. Each
// element in preds must be a single-bit bits-typed Node.
std::vector<TreeBitLocation> ToTreeBitLocations(absl::Span<Node* const> preds) {
  std::vector<TreeBitLocation> locations;
  for (Node* pred : preds) {
    XLS_CHECK(pred->GetType()->IsBits());
    XLS_CHECK_EQ(pred->BitCountOrDie(), 1);
    locations.emplace_back(TreeBitLocation(pred, 0));
  }
  return locations;
}

}  // namespace

bool QueryEngine::AtMostOneNodeTrue(absl::Span<Node* const> preds) const {
  return AtMostOneTrue(ToTreeBitLocations(preds));
}

bool QueryEngine::AtMostOneBitTrue(Node* node) const {
  return AtMostOneTrue(ToTreeBitLocations(node));
}

bool QueryEngine::AtLeastOneNodeTrue(absl::Span<Node* const> preds) const {
  return AtLeastOneTrue(ToTreeBitLocations(preds));
}

bool QueryEngine::AtLeastOneBitTrue(Node* node) const {
  return AtLeastOneTrue(ToTreeBitLocations(node));
}

bool QueryEngine::IsKnown(const TreeBitLocation& bit) const {
  if (!IsTracked(bit.node())) {
    return false;
  }
  return GetTernary(bit.node()).Get(bit.tree_index())[bit.bit_index()] !=
         TernaryValue::kUnknown;
}

bool QueryEngine::IsMsbKnown(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  return ternary_ops::ToKnownBits(ternary).msb();
}

bool QueryEngine::IsOne(const TreeBitLocation& bit) const {
  if (!IsKnown(bit)) {
    return false;
  }
  TernaryVector ternary = GetTernary(bit.node()).Get(bit.tree_index());
  return ternary_ops::ToKnownBitsValues(ternary).Get(bit.bit_index());
}

bool QueryEngine::IsZero(const TreeBitLocation& bit) const {
  if (!IsKnown(bit)) {
    return false;
  }
  TernaryVector ternary = GetTernary(bit.node()).Get(bit.tree_index());
  return !ternary_ops::ToKnownBitsValues(ternary).Get(bit.bit_index());
}

bool QueryEngine::GetKnownMsb(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  XLS_CHECK(IsMsbKnown(node));
  TernaryVector ternary = GetTernary(node).Get({});
  return ternary_ops::ToKnownBitsValues(ternary).msb();
}

bool QueryEngine::IsAllZeros(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  for (TernaryValue t : ternary) {
    if (t != TernaryValue::kKnownZero) {
      return false;
    }
  }
  return true;
}

bool QueryEngine::IsAllOnes(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  for (TernaryValue t : ternary) {
    if (t != TernaryValue::kKnownOne) {
      return false;
    }
  }
  return true;
}

bool QueryEngine::AllBitsKnown(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  if (!IsTracked(node)) {
    return false;
  }
  TernaryVector ternary = GetTernary(node).Get({});
  for (TernaryValue t : ternary) {
    if (t == TernaryValue::kUnknown) {
      return false;
    }
  }
  return true;
}

Bits QueryEngine::MaxUnsignedValue(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  absl::InlinedVector<bool, 1> bits(node->BitCountOrDie());
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    bits[i] = IsZero(TreeBitLocation(node, i)) ? false : true;
  }
  return Bits(bits);
}

Bits QueryEngine::MinUnsignedValue(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  absl::InlinedVector<bool, 16> bits(node->BitCountOrDie());
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    bits[i] = IsOne(TreeBitLocation(node, i));
  }
  return Bits(bits);
}

bool QueryEngine::NodesKnownUnsignedNotEquals(Node* a, Node* b) const {
  XLS_CHECK(a->GetType()->IsBits());
  XLS_CHECK(b->GetType()->IsBits());
  int64_t max_width = std::max(a->BitCountOrDie(), b->BitCountOrDie());
  auto get_known_bit = [this](Node* n, int64_t index) {
    if (index >= n->BitCountOrDie()) {
      return TernaryValue::kKnownZero;
    }
    TreeBitLocation location(n, index);
    if (IsZero(location)) {
      return TernaryValue::kKnownZero;
    }
    if (IsOne(location)) {
      return TernaryValue::kKnownOne;
    }
    return TernaryValue::kUnknown;
  };

  for (int64_t i = 0; i < max_width; ++i) {
    TernaryValue a_bit = get_known_bit(a, i);
    TernaryValue b_bit = get_known_bit(b, i);
    if (a_bit != b_bit && a_bit != TernaryValue::kUnknown &&
        b_bit != TernaryValue::kUnknown) {
      return true;
    }
  }
  return false;
}

bool QueryEngine::NodesKnownUnsignedEquals(Node* a, Node* b) const {
  XLS_CHECK(a->GetType()->IsBits());
  XLS_CHECK(b->GetType()->IsBits());
  TernaryVector a_ternary = GetTernary(a).Get({});
  TernaryVector b_ternary = GetTernary(b).Get({});
  return a == b ||
         (AllBitsKnown(a) && AllBitsKnown(b) &&
          bits_ops::UEqual(ternary_ops::ToKnownBitsValues(a_ternary),
                           ternary_ops::ToKnownBitsValues(b_ternary)));
}

std::string QueryEngine::ToString(Node* node) const {
  XLS_CHECK(node->GetType()->IsBits());
  XLS_CHECK(IsTracked(node));
  return xls::ToString(GetTernary(node).Get({}));
}

}  // namespace xls
