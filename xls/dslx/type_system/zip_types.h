// Copyright 2024 The XLS Authors
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

#ifndef XLS_DSLX_TYPE_SYSTEM_ZIP_TYPES_H_
#define XLS_DSLX_TYPE_SYSTEM_ZIP_TYPES_H_

#include <utility>
#include <variant>

#include "absl/status/status.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {

// Datatype representing aggregates that have the same type on the LHS and RHS.
using AggregatePair =
    std::variant<std::pair<const TupleType*, const TupleType*>,
                 std::pair<const StructType*, const StructType*>,
                 std::pair<const ArrayType*, const ArrayType*>,
                 std::pair<const ChannelType*, const ChannelType*>,
                 std::pair<const FunctionType*, const FunctionType*>,
                 std::pair<const MetaType*, const MetaType*> >;

// This is a bit similar to "SAX parser" style callbacks, if you're familiar
// with that pattern -- it helps us note points in the traversal (e.g. so we can
// format things) and where mismatches have occurred (so we can call them out
// separately or mark them in red, or whatnot).
class ZipTypesCallbacks {
 public:
  virtual ~ZipTypesCallbacks() = default;

  // These are called if the same aggregate type is present on the lhs and rhs,
  // to note we're entering/leaving it.
  virtual absl::Status NoteAggregateStart(const AggregatePair& aggregates) = 0;
  virtual absl::Status NoteAggregateEnd(const AggregatePair& aggregates) = 0;

  // Called when there is a leaf type (non aggregate) where the types are
  // type-compatible.
  virtual absl::Status NoteMatchedLeafType(const Type& lhs,
                                           const Type* lhs_parent,
                                           const Type& rhs,
                                           const Type* rhs_parent) = 0;

  // Called when there is a type (could be leaf or aggregate) where the types
  // are not type-compatible -- we do not recurse into these as they likely do
  // not have a common internal structure given that they mismatch.
  virtual absl::Status NoteTypeMismatch(const Type& lhs, const Type* lhs_parent,
                                        const Type& rhs,
                                        const Type* rhs_parent) = 0;
};

// Zips the /common structure/ of "lhs" type and "rhs" type, invoking "f" at all
// of the matching nodes up to the frontier of mismatch.
//
// When structurally mismatched, the lhs and rhs types are not recursed into (as
// there is no way to have a common structure to the recursion with disparate
// types).
//
// This is generally useful for error message printing where we need to flag the
// places in which types have diverged in a larger nested type tree.
absl::Status ZipTypes(const Type& lhs, const Type& rhs,
                      ZipTypesCallbacks& callbacks);

}  // namespace xls::dslx

#endif  // XLS_DSLX_TYPE_SYSTEM_ZIP_TYPES_H_
