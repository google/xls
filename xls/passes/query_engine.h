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

#ifndef XLS_PASSES_QUERY_ENGINE_H_
#define XLS_PASSES_QUERY_ENGINE_H_

#include "absl/types/variant.h"
#include "xls/common/status/statusor.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"

namespace xls {

// Abstraction representing a particular bit of a particular XLS Node.
struct BitLocation {
  BitLocation() : node(nullptr), bit_index(0) {}
  BitLocation(Node* n, int64 i) : node(n), bit_index(i) {}

  Node* node;
  int64 bit_index;
};

// An abstract base class providing an interface for answering queries about the
// values of and relationship between bits in an XLS function. Information
// provided include statically known bit values and implications between bits in
// the graph.
//
// Generally query methods returning a boolean value return true if the
// condition is known to be true, and false if the condition cannot be
// determined.  This means a false return value does *not* mean that the
// condition is necessarily false. For example, KnownEqual(a, b) returning false
// does not mean that 'a' is necessarily not equal 'b'. Rather, the false return
// value indicates that analysis could not determine whether 'a' and 'b' are
// necessarily equal.
// TODO(meheff): Support types other than bits type.
class QueryEngine {
 public:
  virtual ~QueryEngine() = default;

  // Returns whether any information is available for this node.
  virtual bool IsTracked(Node* node) const = 0;

  // Returns a Bits object indicating which bits have known values for the given
  // node. 'node' must be a Bits type. The Bits object matches the width of the
  // respective Node. A one in a bit position means that the bit has a
  // statically known value (0 or 1).
  virtual const Bits& GetKnownBits(Node* node) const = 0;

  // Returns a Bits object indicating the values (0 or 1) of bits in the given
  // node for bits with known values. If a value at a bit position is not known,
  // the respective value is zero.
  virtual const Bits& GetKnownBitsValues(Node* node) const = 0;

  // Returns true if at most one of the given bits can be true.
  virtual bool AtMostOneTrue(absl::Span<BitLocation const> bits) const = 0;

  // Returns true if at least one of the given bits is true.
  virtual bool AtLeastOneTrue(absl::Span<BitLocation const> bits) const = 0;

  // Returns true if 'a' implies 'b'.
  virtual bool Implies(const BitLocation& a, const BitLocation& b) const = 0;

  // If a particular value of 'node' (true or false for all bits)
  // is implied when the bits in 'predicate_bit_values' have the given values,
  // the implied value of 'node' is returned.
  virtual absl::optional<Bits> ImpliedNodeValue(
      absl::Span<const std::pair<BitLocation, bool>> predicate_bit_values,
      Node* node) const = 0;

  // Returns true if 'a' equals 'b'
  virtual bool KnownEquals(const BitLocation& a,
                           const BitLocation& b) const = 0;

  // Returns true if 'a' is the inverse of 'b'
  virtual bool KnownNotEquals(const BitLocation& a,
                              const BitLocation& b) const = 0;

  // Returns true if at most/least one of the values in 'preds' is true. Each
  // value in 'preds' must be a single-bit bits-typed value.
  bool AtMostOneNodeTrue(absl::Span<Node* const> preds) const;
  bool AtLeastOneNodeTrue(absl::Span<Node* const> preds) const;

  // Returns true if at most/least one of the bits in 'node' is true. 'node'
  // must be bits-typed.
  bool AtMostOneBitTrue(Node* node) const;
  bool AtLeastOneBitTrue(Node* node) const;

  // Returns whether the value of the output bit of the given node at the given
  // index is known (definitely zero or one).
  bool IsKnown(const BitLocation& bit) const;

  // Returns if the most-significant bit is known of 'node'.
  bool IsMsbKnown(Node* node) const;

  // Returns the value of the most-significant bit of 'node'. Precondition: the
  // most-significan bit must be known (IsMsbKnown returns true),
  bool GetKnownMsb(Node* node) const;

  // Returns whether the value of the output bit of the given node at the given
  // index is definitely one (or zero).
  bool IsOne(const BitLocation& bit) const;
  bool IsZero(const BitLocation& bit) const;

  // Returns whether every bit in the output of the given node is definitely one
  // (or zero).
  bool IsAllZeros(Node* node) const;
  bool IsAllOnes(Node* node) const;

  // Returns whether *all* the bits are known for "node".
  bool AllBitsKnown(Node* node) const;

  // Returns the known bits information of the given node as a string of ternary
  // symbols (0, 1, or X) with a '0b' prefix. For example: 0b1XX0.
  std::string ToString(Node* node) const;
};

}  // namespace xls

#endif  // XLS_PASSES_QUERY_ENGINE_H_
