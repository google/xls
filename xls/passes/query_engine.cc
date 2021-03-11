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
std::vector<BitLocation> ToBitLocations(Node* node) {
  XLS_CHECK(node->GetType()->IsBits());
  std::vector<BitLocation> locations;
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    locations.push_back({node, i});
  }
  return locations;
}

// Converts the single-bit Nodes in preds into a vector of BitLocations. Each
// element in preds must be a single-bit bits-typed Node.
std::vector<BitLocation> ToBitLocations(absl::Span<Node* const> preds) {
  std::vector<BitLocation> locations;
  for (Node* pred : preds) {
    XLS_CHECK(pred->GetType()->IsBits());
    XLS_CHECK_EQ(pred->BitCountOrDie(), 1);
    locations.emplace_back(pred, 0);
  }
  return locations;
}

}  // namespace

bool QueryEngine::AtMostOneNodeTrue(absl::Span<Node* const> preds) const {
  return AtMostOneTrue(ToBitLocations(preds));
}

bool QueryEngine::AtMostOneBitTrue(Node* node) const {
  return AtMostOneTrue(ToBitLocations(node));
}

bool QueryEngine::AtLeastOneNodeTrue(absl::Span<Node* const> preds) const {
  return AtLeastOneTrue(ToBitLocations(preds));
}

bool QueryEngine::AtLeastOneBitTrue(Node* node) const {
  return AtLeastOneTrue(ToBitLocations(node));
}

bool QueryEngine::IsKnown(const BitLocation& bit) const {
  if (!IsTracked(bit.node)) {
    return false;
  }
  return GetKnownBits(bit.node).Get(bit.bit_index);
}

bool QueryEngine::IsMsbKnown(Node* node) const {
  return IsTracked(node) && GetKnownBits(node).msb();
}

bool QueryEngine::IsOne(const BitLocation& bit) const {
  return IsKnown(bit) && GetKnownBitsValues(bit.node).Get(bit.bit_index);
}

bool QueryEngine::IsZero(const BitLocation& bit) const {
  return IsKnown(bit) && !GetKnownBitsValues(bit.node).Get(bit.bit_index);
}

bool QueryEngine::GetKnownMsb(Node* node) const {
  XLS_CHECK(IsMsbKnown(node));
  return GetKnownBitsValues(node).msb();
}

bool QueryEngine::IsAllZeros(Node* node) const {
  return IsTracked(node) && GetKnownBits(node).IsAllOnes() &&
         GetKnownBitsValues(node).IsZero();
}

bool QueryEngine::IsAllOnes(Node* node) const {
  return IsTracked(node) && GetKnownBits(node).IsAllOnes() &&
         GetKnownBitsValues(node).IsAllOnes();
}

bool QueryEngine::AllBitsKnown(Node* node) const {
  return IsTracked(node) && GetKnownBits(node).IsAllOnes();
}

Bits QueryEngine::MaxUnsignedValue(Node* node) const {
  absl::InlinedVector<bool, 1> bits(node->BitCountOrDie());
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    bits[i] = IsZero(BitLocation{node, i}) ? false : true;
  }
  return Bits(bits);
}

Bits QueryEngine::MinUnsignedValue(Node* node) const {
  absl::InlinedVector<bool, 16> bits(node->BitCountOrDie());
  for (int64_t i = 0; i < node->BitCountOrDie(); ++i) {
    bits[i] = IsOne(BitLocation{node, i});
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
    BitLocation location(n, index);
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
  return a == b ||
         (AllBitsKnown(a) && AllBitsKnown(b) &&
          bits_ops::UEqual(GetKnownBitsValues(a), GetKnownBitsValues(b)));
}

std::string QueryEngine::ToString(Node* node) const {
  XLS_CHECK(IsTracked(node));
  std::string ret = "0b";
  for (int64_t i = GetKnownBits(node).bit_count() - 1; i >= 0; --i) {
    std::string c = "X";
    if (IsKnown(BitLocation(node, i))) {
      c = IsOne(BitLocation(node, i)) ? "1" : "0";
    }
    absl::StrAppend(&ret, c);
    if ((i % 4) == 0 && i != 0) {
      absl::StrAppend(&ret, "_");
    }
  }
  return ret;
}

}  // namespace xls
