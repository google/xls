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

#include "xls/dslx/type_system/zip_types.h"

#include <cstdint>
#include <utility>

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/type_system/type.h"

namespace xls::dslx {
namespace {

// Forward decl.
absl::Status ZipTypesWithParents(const Type& lhs, const Type& rhs,
                                 const Type* lhs_parent, const Type* rhs_parent,
                                 ZipTypesCallbacks& callbacks);

// This is an implementation detail in traversing types and then recursively
// calling ZipTypes -- we inherit TypeVisitor because we need to learn the
// actual type of the generic `Type` on the left hand side and then compare that
// to what we see on the right hand side at each step.
class ZipTypeVisitor : public TypeVisitor {
 public:
  explicit ZipTypeVisitor(const Type& rhs, const Type* lhs_parent,
                          const Type* rhs_parent, ZipTypesCallbacks& callbacks)
      : rhs_(rhs),
        lhs_parent_(lhs_parent),
        rhs_parent_(rhs_parent),
        callbacks_(callbacks) {}

  ~ZipTypeVisitor() override = default;

  // -- various non-aggregate types

  absl::Status HandleEnum(const EnumType& lhs) override {
    return HandleNonAggregate(lhs);
  }
  absl::Status HandleBits(const BitsType& lhs) override {
    return HandleNonAggregate(lhs);
  }
  absl::Status HandleBitsConstructor(const BitsConstructorType& lhs) override {
    return HandleNonAggregate(lhs);
  }
  absl::Status HandleToken(const TokenType& lhs) override {
    return HandleNonAggregate(lhs);
  }

  // -- types that contain other types

  absl::Status HandleTuple(const TupleType& lhs) override {
    if (auto* rhs = dynamic_cast<const TupleType*>(&rhs_)) {
      return HandleTupleLike(lhs, *rhs);
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }
  absl::Status HandleStruct(const StructType& lhs) override {
    return HandleStructTypeBase<StructType>(lhs);
  }
  absl::Status HandleProc(const ProcType& lhs) override {
    return HandleStructTypeBase<ProcType>(lhs);
  }
  absl::Status HandleArray(const ArrayType& lhs) override {
    if (auto* rhs = dynamic_cast<const ArrayType*>(&rhs_)) {
      if (lhs.size() != rhs->size()) {
        return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
      }
      AggregatePair aggregates = std::make_pair(&lhs, rhs);
      XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateStart(aggregates));
      const Type& lhs_elem = lhs.element_type();
      const Type& rhs_elem = rhs->element_type();
      XLS_RETURN_IF_ERROR(ZipTypes(lhs_elem, rhs_elem, callbacks_));
      return callbacks_.NoteAggregateEnd(aggregates);
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }
  absl::Status HandleChannel(const ChannelType& lhs) override {
    if (auto* rhs = dynamic_cast<const ChannelType*>(&rhs_)) {
      // If channel directions don't match, capture the full channel strings.
      if (lhs.direction() != rhs->direction()) {
        return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
      }

      AggregatePair aggregates = std::make_pair(&lhs, rhs);
      XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateStart(aggregates));
      XLS_RETURN_IF_ERROR(
          ZipTypes(lhs.payload_type(), rhs->payload_type(), callbacks_));
      return callbacks_.NoteAggregateEnd(aggregates);
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }
  absl::Status HandleFunction(const FunctionType& lhs) override {
    if (auto* rhs = dynamic_cast<const FunctionType*>(&rhs_)) {
      AggregatePair aggregates = std::make_pair(&lhs, rhs);
      XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateStart(aggregates));
      for (int64_t i = 0; i < lhs.GetParamCount(); ++i) {
        XLS_RETURN_IF_ERROR(
            ZipTypes(*lhs.GetParams()[i], *rhs->GetParams()[i], callbacks_));
      }
      XLS_RETURN_IF_ERROR(
          ZipTypes(lhs.return_type(), rhs->return_type(), callbacks_));
      return callbacks_.NoteAggregateEnd(aggregates);
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }
  absl::Status HandleMeta(const MetaType& lhs) override {
    if (auto* rhs = dynamic_cast<const MetaType*>(&rhs_)) {
      AggregatePair aggregates = std::make_pair(&lhs, rhs);
      XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateStart(aggregates));
      XLS_RETURN_IF_ERROR(
          ZipTypes(*lhs.wrapped(), *rhs->wrapped(), callbacks_));
      return callbacks_.NoteAggregateEnd(aggregates);
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }

 private:
  template <typename T>
  absl::Status HandleStructTypeBase(const T& lhs) {
    if (auto* rhs = dynamic_cast<const T*>(&rhs_)) {
      if (&lhs.nominal_type() == &rhs->nominal_type()) {
        return HandleTupleLike(lhs, *rhs);
      }
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }

  // Handles tuples and structs which are quite similar.
  template <typename T>
  absl::Status HandleTupleLike(const T& lhs, const T& rhs) {
    bool structurally_compatible = lhs.size() == rhs.size();
    if (!structurally_compatible) {
      return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs, rhs_parent_);
    }
    AggregatePair aggregates = std::make_pair(&lhs, &rhs);
    XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateStart(aggregates));
    for (int64_t i = 0; i < lhs.size(); ++i) {
      const Type& lhs_elem = lhs.GetMemberType(i);
      const Type& rhs_elem = rhs.GetMemberType(i);
      XLS_RETURN_IF_ERROR(
          ZipTypesWithParents(lhs_elem, rhs_elem, &lhs, &rhs, callbacks_));
      if (i + 1 != lhs.size()) {
        XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateNext(aggregates));
      }
    }
    XLS_RETURN_IF_ERROR(callbacks_.NoteAggregateEnd(aggregates));
    return absl::OkStatus();
  }

  absl::Status HandleNonAggregate(const Type& lhs) {
    if (lhs.CompatibleWith(rhs_)) {
      return callbacks_.NoteMatchedLeafType(lhs, lhs_parent_, rhs_,
                                            rhs_parent_);
    }
    return callbacks_.NoteTypeMismatch(lhs, lhs_parent_, rhs_, rhs_parent_);
  }

  const Type& rhs_;
  const Type* lhs_parent_;
  const Type* rhs_parent_;
  ZipTypesCallbacks& callbacks_;
};

absl::Status ZipTypesWithParents(const Type& lhs, const Type& rhs,
                                 const Type* lhs_parent, const Type* rhs_parent,
                                 ZipTypesCallbacks& callbacks) {
  ZipTypeVisitor visitor(rhs, lhs_parent, rhs_parent, callbacks);
  return lhs.Accept(visitor);
}

}  // namespace

absl::Status ZipTypes(const Type& lhs, const Type& rhs,
                      ZipTypesCallbacks& callbacks) {
  return ZipTypesWithParents(lhs, rhs, nullptr, nullptr, callbacks);
}

}  // namespace xls::dslx
