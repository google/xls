// Copyright 2025 The XLS Authors
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

#ifndef XLS_CONTRIB_XLSCC_TRANSLATOR_TYPES_H_
#define XLS_CONTRIB_XLSCC_TRANSLATOR_TYPES_H_

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/ComputeDependence.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/TypeBase.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/node_manipulation.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xlscc {

class TranslatorTypeInterface;
struct GeneratedFunctionSlice;

class ConstValue;
// Base class for immutable objects representing XLS[cc] value types
// These are not 1:1 with clang::Types, and do not represent qualifiers
//  such as const and reference.
class CType {
 public:
  virtual ~CType() = 0;
  virtual bool operator==(const CType& o) const = 0;
  bool operator!=(const CType& o) const;

  virtual absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const;
  virtual int GetBitWidth() const;
  virtual explicit operator std::string() const;
  virtual xls::Type* GetXLSType(xls::Package* package) const;
  virtual bool StoredAsXLSBits() const;
  virtual absl::Status GetMetadata(
      TranslatorTypeInterface& translator, xlscc_metadata::Type* output,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const;
  virtual absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                        ConstValue const_value,
                                        xlscc_metadata::Value* output) const;

  template <typename Derived>
  const Derived* As() const {
    CHECK(Is<Derived>());
    return dynamic_cast<const Derived*>(this);
  }

  template <typename Derived>
  bool Is() const {
    return dynamic_cast<const Derived*>(this) != nullptr;
  }

  inline std::string debug_string() const { return std::string(*this); }
};

// C/C++ void
class CVoidType : public CType {
 public:
  ~CVoidType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  bool operator==(const CType& o) const override;
};

// __xls_bits special built-in type
class CBitsType : public CType {
 public:
  explicit CBitsType(int width);
  ~CBitsType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  bool operator==(const CType& o) const override;
  bool StoredAsXLSBits() const override;

 private:
  int width_;
};

// Any native integral type: char, short, int, long, etc
class CIntType : public CType {
 public:
  ~CIntType() override;
  CIntType(int width, bool is_signed, bool is_declared_as_char = false);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  bool operator==(const CType& o) const override;

  xls::Type* GetXLSType(xls::Package* package) const override;
  bool StoredAsXLSBits() const override;

  inline int width() const { return width_; }
  inline bool is_signed() const { return is_signed_; }
  inline bool is_declared_as_char() const { return is_declared_as_char_; }

 protected:
  const int width_;
  const bool is_signed_;
  // We use this field to tell "char" declarations from explicitly-qualified
  // "signed char" or "unsigned char" declarations, as in C++ "char" is neither
  // signed nor unsigned, while the explicitly-qualified declarations have
  // signedness.  The field is set to true for "char" declarations and false for
  // every other integer type.  The field is strictly for generating metadata;
  // it is not IR generation.
  const bool is_declared_as_char_;
};

// Any native decimal type: float, double, etc
class CFloatType : public CType {
 public:
  ~CFloatType() override;
  explicit CFloatType(bool double_precision);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  bool operator==(const CType& o) const override;

  xls::Type* GetXLSType(xls::Package* package) const override;
  bool StoredAsXLSBits() const override;

  inline bool double_precision() const { return double_precision_; }

 protected:
  const bool double_precision_;
};

// C++ enum or enum class
class CEnumType : public CIntType {
 public:
  ~CEnumType() override;
  CEnumType(std::string name, int width, bool is_signed,
            absl::btree_map<std::string, int64_t> variants_by_name);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  bool operator==(const CType& o) const override;

  xls::Type* GetXLSType(xls::Package* package) const override;
  bool StoredAsXLSBits() const override;

  inline int width() const { return width_; }
  inline bool is_signed() const { return is_signed_; }

 private:
  std::string name_;
  absl::btree_map<std::string, int64_t> variants_by_name_;
  absl::btree_map<int64_t, std::vector<std::string>> variants_by_value_;
};

// C++ bool
class CBoolType : public CType {
 public:
  ~CBoolType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  bool operator==(const CType& o) const override;
  bool StoredAsXLSBits() const override;
};

// C/C++ struct field
class CField {
 public:
  CField(const clang::NamedDecl* name, int index, std::shared_ptr<CType> type);

  const clang::NamedDecl* name() const;
  int index() const;
  std::shared_ptr<CType> type() const;
  absl::Status GetMetadata(
      TranslatorTypeInterface& translator, xlscc_metadata::StructField* output,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const;
  absl::Status GetMetadataValue(TranslatorTypeInterface* t,
                                ConstValue const_value,
                                xlscc_metadata::StructFieldValue* output) const;

 private:
  const clang::NamedDecl* name_;
  int index_;
  std::shared_ptr<CType> type_;
};

// C/C++ struct
class CStructType : public CType {
 public:
  CStructType(std::vector<std::shared_ptr<CField>> fields, bool no_tuple_flag,
              bool synthetic_int_flag);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  bool operator==(const CType& o) const override;
  absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const override;

  // Returns true if the #pragma hls_notuple or hls_synthetic_int directive was
  // given for the struct
  bool no_tuple_flag() const;
  bool synthetic_int_flag() const;
  const std::vector<std::shared_ptr<CField>>& fields() const;
  const absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>&
  fields_by_name() const;
  // Get the full CField struct by name.
  // returns nullptr if the field is not found
  std::shared_ptr<CField> get_field(const clang::NamedDecl* name) const;
  absl::StatusOr<int64_t> count_lvalue_compounds(
      TranslatorTypeInterface& translator) const;

 private:
  bool no_tuple_flag_;
  bool synthetic_int_flag_;
  std::vector<std::shared_ptr<CField>> fields_;
  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>
      fields_by_name_;
};

class CInternalTuple : public CType {
 public:
  explicit CInternalTuple(std::vector<std::shared_ptr<CType>> fields);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  bool operator==(const CType& o) const override;

  const std::vector<std::shared_ptr<CType>>& fields() const;

 private:
  std::vector<std::shared_ptr<CType>> fields_;
};

// An alias for a type that can be instantiated. Typically this is reduced to
//  another CType via Translator::ResolveTypeInstance()
//
// The reason for this to exist is that two types may have exactly
//  the same content and structure, but still be considered different in C/C++
//  because of their different typenames.
// For example, structs A and B still do not have the same type:
// struct A { int x; int y; };
// struct B { int x; int y; };
//
// They may also have different template parameters. CInstantiableTypeAliases
//  for Foo<true> and Foo<false> are not equal.
// template<bool Tp>
// struct Foo { int bar; };
class CInstantiableTypeAlias : public CType {
 public:
  explicit CInstantiableTypeAlias(const clang::NamedDecl* base);

  const clang::NamedDecl* base() const;

  bool operator==(const CType& o) const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  explicit operator std::string() const override;
  int GetBitWidth() const override;
  absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const override;

 private:
  const clang::NamedDecl* base_;
};

// C/C++ native array
class CArrayType : public CType {
 public:
  CArrayType(std::shared_ptr<CType> element, int size, bool use_tuple);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const override;

  int GetSize() const;
  std::shared_ptr<CType> GetElementType() const;
  bool GetUseTuple() const;

 private:
  std::shared_ptr<CType> element_;
  int size_;
  bool use_tuple_;
};

// Pointer in C/C++
class CPointerType : public CType {
 public:
  explicit CPointerType(std::shared_ptr<CType> pointee_type);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const override;

  std::shared_ptr<CType> GetPointeeType() const;

 private:
  std::shared_ptr<CType> pointee_type_;
};

// Reference in C/C++
class CReferenceType : public CType {
 public:
  explicit CReferenceType(std::shared_ptr<CType> pointee_type);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const override;

  std::shared_ptr<CType> GetPointeeType() const;

 private:
  std::shared_ptr<CType> pointee_type_;
};

enum class OpType {
  kNull = 0,
  kSend,
  kRecv,
  kSendRecv,
  kRead,
  kWrite,
  kTrace,
  kLoopBegin,
  kLoopEndJump,
};
enum class InterfaceType { kNull = 0, kDirect, kFIFO, kMemory, kTrace };
enum class TraceType { kNull = 0, kAssert, kTrace };
// TODO(seanhaskell): Remove with old FSM
enum class IOSchedulingOption { kNone = 0, kASAPBefore = 1, kASAPAfter = 2 };

// __xls_channel in C/C++
class CChannelType : public CType {
 public:
  CChannelType(std::shared_ptr<CType> item_type, int64_t memory_size);
  CChannelType(std::shared_ptr<CType> item_type, int64_t memory_size,
               OpType op_type);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(TranslatorTypeInterface& translator,
                           xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(TranslatorTypeInterface& translator,
                                ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  std::shared_ptr<CType> GetItemType() const;
  OpType GetOpType() const;
  int64_t GetMemorySize() const;
  int64_t GetMemoryAddressWidth() const;
  std::optional<int64_t> GetMemoryMaskWidth() const;
  static std::shared_ptr<CType> MemoryAddressType(int64_t memory_size);
  static int64_t MemoryAddressWidth(int64_t memory_size);

  absl::StatusOr<xls::Type*> GetReadRequestType(xls::Package* package,
                                                xls::Type* item_type) const;
  absl::StatusOr<xls::Type*> GetReadResponseType(xls::Package* package,
                                                 xls::Type* item_type) const;
  absl::StatusOr<xls::Type*> GetWriteRequestType(xls::Package* package,
                                                 xls::Type* item_type) const;
  absl::StatusOr<xls::Type*> GetWriteResponseType(xls::Package* package,
                                                  xls::Type* item_type) const;

  absl::StatusOr<bool> ContainsLValues(
      TranslatorTypeInterface& translator) const override;

 private:
  std::shared_ptr<CType> item_type_;
  OpType op_type_;
  int64_t memory_size_;
};

struct IOChannel;

// In general, the original clang::Expr used to assign the pointer is saved
// as an lvalue. However, it's necessary to create synthetic selects for
// conditional assignments to pointers. Ternaries are also generated this way
// for consistency.
class LValue {
 public:
  LValue() : is_null_(true) {}
  explicit LValue(const clang::Expr* leaf) : leaf_(leaf) {
    CHECK_NE(leaf_, nullptr);
    // Should use select constructor
    CHECK(clang::dyn_cast<clang::ConditionalOperator>(leaf_) == nullptr);
    // Should be removed by CreateReferenceValue()
    CHECK(clang::dyn_cast<clang::ParenExpr>(leaf_) == nullptr);
  }
  explicit LValue(IOChannel* channel_leaf) : channel_leaf_(channel_leaf) {}
  LValue(TrackedBValue cond, std::shared_ptr<LValue> lvalue_true,
         std::shared_ptr<LValue> lvalue_false)
      : cond_(cond), lvalue_true_(lvalue_true), lvalue_false_(lvalue_false) {
    // Allow for null condition
    if (cond_.valid()) {
      CHECK(cond_.GetType()->IsBits());
      CHECK_EQ(cond_.BitCountOrDie(), 1);
    }
    CHECK_NE(lvalue_true_.get(), nullptr);
    CHECK_NE(lvalue_false_.get(), nullptr);
    is_select_ = true;
  }
  explicit LValue(const absl::flat_hash_map<int64_t, std::shared_ptr<LValue>>&
                      compound_by_index)
      : compound_by_index_(compound_by_index) {
    absl::flat_hash_set<int64_t> to_erase;
    for (const auto& [idx, lval] : compound_by_index_) {
      if (lval == nullptr) {
        to_erase.insert(idx);
      }
    }
    for (int64_t idx : to_erase) {
      compound_by_index_.erase(idx);
    }
  }

  bool is_select() const { return is_select_; }
  const TrackedBValue& cond() const { return cond_; }
  std::shared_ptr<LValue> lvalue_true() const { return lvalue_true_; }
  std::shared_ptr<LValue> lvalue_false() const { return lvalue_false_; }
  const absl::flat_hash_map<int64_t, std::shared_ptr<LValue>>& get_compounds()
      const {
    return compound_by_index_;
  }
  std::shared_ptr<LValue> get_compound_or_null(int64_t idx) const {
    if (compound_by_index_.contains(idx)) {
      return compound_by_index_.at(idx);
    }
    return nullptr;
  }

  const clang::Expr* leaf() const { return leaf_; }
  IOChannel* channel_leaf() const { return channel_leaf_; }
  bool is_null() const { return is_null_; }

  std::string debug_string() const {
    if (is_null()) {
      return "null";
    }
    if (leaf() != nullptr) {
      return absl::StrFormat("leaf(%p)", leaf());
    }
    if (channel_leaf() != nullptr) {
      return absl::StrFormat("channel(%p)", channel_leaf());
    }
    if (is_select()) {
      return absl::StrFormat("%s ? %s : %s", cond().ToString(),
                             lvalue_true()->debug_string(),
                             lvalue_false()->debug_string());
    }
    std::string ret = "(";
    for (const auto& [idx, lval] : get_compounds()) {
      ret += absl::StrFormat("[%i]: %s ", idx,
                             lval ? lval->debug_string() : "(null)");
    }
    return ret + ")";
  }

 private:
  bool is_null_ = false;
  const clang::Expr* leaf_ = nullptr;
  IOChannel* channel_leaf_ = nullptr;

  bool is_select_ = false;
  TrackedBValue cond_;
  std::shared_ptr<LValue> lvalue_true_ = nullptr;
  std::shared_ptr<LValue> lvalue_false_ = nullptr;

  absl::flat_hash_map<int64_t, std::shared_ptr<LValue>> compound_by_index_;
};

// Immutable object representing an XLS[cc] value. The class associates an
//  XLS IR value expression with an XLS[cc] type (derived from C/C++).
// This class is necessary because an XLS "bits[16]" might be a native short
//  native unsigned short, __xls_bits, or a class containing a single __xls_bits
//  And each of these types implies different translation behaviors.
// For types other than pointers, only rvalue is used. For pointers, only lvalue
//  is used.
class CValue {
 public:
  CValue() = default;
  CValue(TrackedBValue rvalue, std::shared_ptr<CType> type,
         bool disable_type_check = false,
         std::shared_ptr<LValue> lvalue = nullptr)
      : rvalue_(rvalue), lvalue_(lvalue), type_(std::move(type)) {
    CHECK_NE(type_.get(), nullptr);
    if (!disable_type_check) {
      CHECK(!type_->StoredAsXLSBits() ||
            rvalue.BitCountOrDie() == type_->GetBitWidth());
      // Structs (and their aliases) can have compound lvalues and also rvalues
      CHECK(!(lvalue && !dynamic_cast<CReferenceType*>(type_.get()) &&
              !dynamic_cast<CPointerType*>(type_.get()) &&
              !dynamic_cast<CChannelType*>(type_.get()) &&
              !dynamic_cast<CStructType*>(type_.get()) &&
              !dynamic_cast<CInstantiableTypeAlias*>(type_.get())));
      // Pointers are stored as empty tuples in structs
      const bool rvalue_empty =
          !rvalue.valid() || (rvalue.GetType()->IsTuple() &&
                              rvalue.GetType()->AsTupleOrDie()->size() == 0);
      CHECK(!(lvalue && rvalue.valid()) ||
            dynamic_cast<CStructType*>(type_.get()) ||
            dynamic_cast<CInstantiableTypeAlias*>(type_.get()));
      // Only supporting lvalues of the form &... for pointers
      if (dynamic_cast<CPointerType*>(type_.get()) != nullptr ||
          dynamic_cast<CReferenceType*>(type_.get()) != nullptr ||
          dynamic_cast<CChannelType*>(type_.get()) != nullptr) {
        // Always get rvalue from lvalue, otherwise changes to the original
        // won't
        //  be reflected when the pointer is dereferenced.
        CHECK(rvalue_empty);
      } else if (dynamic_cast<CChannelType*>(type_.get()) != nullptr) {
        CHECK(rvalue_empty && lvalue != nullptr);
      } else if (dynamic_cast<CVoidType*>(type_.get()) != nullptr) {
        CHECK(rvalue_empty);
      } else {
        CHECK(rvalue.valid());
      }
    }
  }
  CValue(std::shared_ptr<LValue> lvalue, std::shared_ptr<CType> type,
         bool disable_type_check = false)
      : CValue(TrackedBValue(), type, disable_type_check, lvalue) {}

  const TrackedBValue& rvalue() const { return rvalue_; }
  std::shared_ptr<CType> type() const { return type_; }
  std::shared_ptr<LValue> lvalue() const { return lvalue_; }
  std::string debug_string() const {
    return absl::StrFormat(
        "(rval=%s, type=%s, lval=%s)", rvalue_.ToString(),
        (type_ != nullptr) ? std::string(*type_) : "(null)",
        lvalue_ ? lvalue_->debug_string().c_str() : "(null)");
  }
  bool operator==(const CValue& o) const {
    if (!NodesEquivalentWithContinuations(rvalue_.node(), o.rvalue_.node())) {
      return false;
    }
    if (lvalue_ != o.lvalue_) {
      return false;
    }
    if (*type_ != *o.type_) {
      return false;
    }
    return true;
  }
  bool operator!=(const CValue& o) const { return !(*this == o); }
  bool valid() const {
    const bool is_valid = rvalue_.valid() || (lvalue_ != nullptr);
    CHECK(!is_valid || type_ != nullptr);
    return is_valid;
  }

 private:
  TrackedBValue rvalue_;
  std::shared_ptr<LValue> lvalue_;
  std::shared_ptr<CType> type_;
};

// Similar to CValue, but contains an xls::Value for a constant expression
class ConstValue {
 public:
  ConstValue() = default;
  ConstValue(xls::Value value, std::shared_ptr<CType> type,
             bool disable_type_check = false)
      : value_(value), type_(std::move(type)) {
    CHECK(disable_type_check || !type_->StoredAsXLSBits() ||
          value.GetFlatBitCount() == type_->GetBitWidth());
  }

  friend bool operator==(const ConstValue& lhs, const ConstValue& rhs) {
    return lhs.value_ == rhs.value_ && (*lhs.type_) == (*rhs.type_);
  }

  xls::Value rvalue() const { return value_; }
  std::shared_ptr<CType> type() const { return type_; }

  std::string debug_string() const {
    return absl::StrFormat("(value=%s, type=%s)", value_.ToString(),
                           (type_ != nullptr) ? std::string(*type_) : "(null)");
  }

 private:
  xls::Value value_;
  std::shared_ptr<CType> type_;
};

struct ChannelBundle {
  xls::Channel* regular = nullptr;

  xls::Channel* read_request = nullptr;
  xls::Channel* read_response = nullptr;
  xls::Channel* write_request = nullptr;
  xls::Channel* write_response = nullptr;

  inline bool operator==(const ChannelBundle& o) const {
    return regular == o.regular && read_request == o.read_request &&
           read_response == o.read_response &&
           write_request == o.write_request &&
           write_response == o.write_response;
  }

  inline bool operator!=(const ChannelBundle& o) const { return !(*this == o); }

  inline bool operator<(const ChannelBundle& o) const {
    if (regular != o.regular) {
      return regular < o.regular;
    }
    if (read_request != o.read_request) {
      return read_request < o.read_request;
    }
    if (read_response != o.read_response) {
      return read_response < o.read_response;
    }
    if (write_request != o.write_request) {
      return write_request < o.write_request;
    }
    return write_response < o.write_response;
  }
};

struct IOOp;

// This is an interface to the Translator's type translation facilities,
// enabling code to use them without linking in the entire Translator.
// It should serve as a start to eventually break this functionality out
// from the monolithic Translator class.
class TranslatorTypeInterface {
 public:
  virtual ~TranslatorTypeInterface() = default;

  virtual absl::StatusOr<xlscc_metadata::IntType> GenerateSyntheticInt(
      std::shared_ptr<CType> ctype) = 0;

  virtual xls::SourceInfo GetLoc(const clang::Stmt& stmt) = 0;
  virtual xls::SourceInfo GetLoc(const clang::Decl& decl) = 0;

  virtual absl::StatusOr<std::shared_ptr<CType>> ResolveTypeInstance(
      std::shared_ptr<CType> t) = 0;

  virtual TrackedBValue GetStructFieldXLS(TrackedBValue val, int64_t index,
                                          const CStructType& type,
                                          const xls::SourceInfo& loc) = 0;
  virtual absl::StatusOr<std::shared_ptr<CType>> TranslateTypeFromClang(
      clang::QualType t, const xls::SourceInfo& loc,
      bool array_as_tuple = false) = 0;

  virtual std::shared_ptr<CType> GetCTypeForAlias(
      const std::shared_ptr<CInstantiableTypeAlias>& alias) = 0;

  // This version is public because it needs to be accessed by CStructType
  static absl::StatusOr<xls::Value> GetStructFieldXLS(xls::Value val, int index,
                                                      const CStructType& type);

  virtual void AppendMessageTraces(std::string* message,
                                   const xls::SourceInfo& loc) = 0;
};

struct GenerateIOReturn {
  TrackedBValue token_out;
  // May be invalid if the op doesn't receive anything (send, trace, etc)
  TrackedBValue received_value;
  TrackedBValue io_condition;
};

class TranslatorIOInterface {
 public:
  virtual ~TranslatorIOInterface() = default;

  virtual std::optional<ChannelBundle> GetChannelBundleForOp(
      const IOOp& op, const xls::SourceInfo& loc) = 0;

  virtual absl::StatusOr<TrackedBValue> GetOpCondition(
      const IOOp& op, TrackedBValue op_out_value, xls::ProcBuilder& pb) = 0;

  // Returns new token
  virtual absl::StatusOr<TrackedBValue> GenerateTrace(
      TrackedBValue trace_out_value, TrackedBValue before_token,
      TrackedBValue condition, const IOOp& op, xls::ProcBuilder& pb) = 0;

  virtual absl::StatusOr<GenerateIOReturn> GenerateIO(
      const IOOp& op, TrackedBValue before_token, TrackedBValue op_out_value,
      xls::ProcBuilder& pb,
      const std::optional<ChannelBundle>& optional_channel_bundle,
      std::optional<TrackedBValue> extra_condition = std::nullopt) = 0;

  virtual absl::StatusOr<TrackedBValue> GetIOOpRetValueFromSlice(
      TrackedBValue slice_ret_val, const GeneratedFunctionSlice& slice,
      const xls::SourceInfo& loc) = 0;
};

// This base class provides common functionality from the Translator class,
// such as error handling, so that new functionality may be implemented
// separately from the Translator monolith. For example, its methods can use
// XLSCC_CHECK_*.
class GeneratorBase {
 public:
  explicit GeneratorBase(TranslatorTypeInterface& translator_types)
      : translator_types_(translator_types) {}
  virtual ~GeneratorBase() = default;

  template <typename... Args>
  std::string ErrorMessage(const xls::SourceInfo& loc,
                           const absl::FormatSpec<Args...>& format,
                           const Args&... args) {
    std::string result = absl::StrFormat(format, args...);

    translator_types().AppendMessageTraces(&result, loc);

    return result;
  }

 protected:
  TranslatorTypeInterface& translator_types() { return translator_types_; }

 private:
  TranslatorTypeInterface& translator_types_;
};

void GetAllBValuesForCValue(const CValue& cval,
                            std::vector<const TrackedBValue*>& out);
void GetAllBValuesForLValue(std::shared_ptr<LValue> lval,
                            std::vector<const TrackedBValue*>& out);

class OrderCValuesFunc {
 public:
  std::vector<const CValue*> operator()(
      const absl::flat_hash_set<const CValue*>& cvals_unordered);
};

template <typename K>
using CValueMap =
    DeterministicMapBase<K, CValue, absl::flat_hash_map<K, CValue>,
                         OrderCValuesFunc>;

class OrderLValuesFunc {
 public:
  std::vector<const std::shared_ptr<LValue>*> operator()(
      const absl::flat_hash_set<const std::shared_ptr<LValue>*>&
          lvals_unordered);
};

template <typename K>
using LValueMap =
    DeterministicMapBase<K, std::shared_ptr<LValue>,
                         absl::flat_hash_map<K, std::shared_ptr<LValue>>,
                         OrderLValuesFunc>;

// Tracks information about an __xls_channel parameter to a function
struct IOChannel {
  // Unique within the function
  std::string unique_name;
  // Type of item the channel transfers
  std::shared_ptr<CType> item_type;
  // Memory size (if applicable)
  int64_t memory_size = -1;
  // The total number of IO ops on the channel within the function
  // (IO ops are conditional, so this is the maximum in a real invocation)
  int total_ops = 0;
  // If not nullptr, the channel isn't explicitly present in the source
  // For example, the channels used for pipelined for loops
  // (can be nullptr if loop body isn't generated)
  // (the record must be passed up)
  std::optional<xls::Channel*> generated = std::nullopt;
  // Declared inside of a function, so that the record must be passed up
  bool internal_to_function = false;
};

// Tracks information about an IO op on an __xls_channel parameter to a function
struct IOOp {
  // --- Preserved across calls ---
  OpType op = OpType::kNull;

  bool is_blocking = true;

  // Source location for messages
  xls::SourceInfo op_location;

  // For OpType::kTrace
  // Assert just puts condition in ret_val. This is not the assertion condition
  //  but the condition for the assertion to fire (!assert condition)
  // Trace puts (condition, ... args ...) in ret_val
  TraceType trace_type = TraceType::kNull;

  std::string trace_message_string;
  std::string label_string;

  // If None is specified, then actions happen in parallel, except
  // as sequenced by after_ops. If ASAP is specified, then after_ops
  // is unused.
  // TODO(seanhaskell): Remove with old FSM
  IOSchedulingOption scheduling_option = IOSchedulingOption::kNone;

  // Jump has a pointer to the begin, begin has a pointer to the jump
  const IOOp* loop_op_paired = nullptr;

  // --- Not preserved across calls ---

  std::string final_param_name;

  // Must be sequenced after these ops via tokens
  // This is translated across calls
  // Ops must be in GeneratedFunction::io_ops respecting this order
  // TODO(seanhaskell): Remove with old FSM
  std::vector<const IOOp*> after_ops;

  IOChannel* channel = nullptr;

  // For calls to subroutines with IO inside
  const IOOp* sub_op = nullptr;

  // Input __xls_channel parameters take tuple types containing a value for
  //  each read() op. This is the index of this op in the tuple.
  int channel_op_index;

  // Output value from function for IO op
  TrackedBValue ret_value;

  // For reads: input value from function parameter for Recv op
  CValue input_value;
};

enum class SideEffectingParameterType { kNull = 0, kIOOp, kStatic };

// Describes a generated parameter from IO, statics, etc
struct SideEffectingParameter {
  SideEffectingParameterType type = SideEffectingParameterType::kNull;
  std::string param_name;
  IOOp* io_op = nullptr;
  const clang::NamedDecl* static_value = nullptr;
  xls::Type* xls_io_param_type = nullptr;
};

struct GeneratedFunction;

// Encapsulates values needed to generate procs for a pipelined loop body
// TODO(seanhaskell): Remove with old FSM
struct PipelinedLoopSubProc {
  std::string name_prefix;

  xls::SourceInfo loc;

  GeneratedFunction* enclosing_func = nullptr;
  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CType>>
      outer_variable_types;

  // These reference the enclosing GeneratedFunction
  IOChannel* context_out_channel;
  IOChannel* context_in_channel;

  std::shared_ptr<CStructType> context_cvars_struct_ctype;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t> context_field_indices;

  std::shared_ptr<CStructType> context_in_cvars_struct_ctype;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t>
      context_in_field_indices;

  std::shared_ptr<CStructType> context_out_cvars_struct_ctype;
  std::shared_ptr<CInternalTuple> context_out_lval_conds_ctype;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t>
      context_out_field_indices;

  uint64_t extra_return_count;
  // Can't copy, since pointers are kept
  std::unique_ptr<GeneratedFunction> generated_func;

  std::vector<const clang::NamedDecl*> variable_fields_order;
  std::vector<const clang::NamedDecl*> vars_changed_in_body;
  // (Decl, access count)
  std::vector<std::pair<const clang::NamedDecl*, int64_t>>
      vars_accessed_in_body;
  std::vector<const clang::NamedDecl*> vars_to_save_between_iters;
};

// A value outputted from a function slice for potential later use.
// One of these is generated per node that is referred to by a TrackedBValue
// at the time the function gets "sliced" during generation.
//
// Since only one is generated per node, it may be referenced by multiple
// ContinuationInputs (& TrackedBValues) in later slices.
struct ContinuationValue {
  xls::Node* output_node = nullptr;

  // name, decls are for test/debug only
  std::string name;
  absl::flat_hash_set<const clang::NamedDecl*> decls;

  // Precomputed literal, used for unrolling, IO pruning, etc
  std::optional<xls::Value> literal = std::nullopt;

  // If this is true, then the value doesn't need to be stored in a state
  // element, as it is assumed to be always available externally.
  bool direct_in = false;

  // Ordered list of the BValues for the Node that feeds this output.
  // A ContinuationValue corresponds to exactly one Node, but several
  // BValues can point to this Node.
  std::vector<TrackedBValue*> created_from;
};

// A value inputted to a function slice, continuing a TrackedBValue from
// a previous slice.
//
// There can be multiple of these for a given parameter to the function,
// in which case there is a "phi". Which value is fed into the parameter
// is determined by control flow, typically the last one produced.
struct ContinuationInput {
  ContinuationValue* continuation_out = nullptr;
  xls::Param* input_node = nullptr;

  // name, decls are for test/debug only
  std::string name;
  absl::flat_hash_set<const clang::NamedDecl*> decls;
};

// One "slice" of a C++ function. When there are side-effecting operations
// in a C++ function, N+1 XLS IR functions will be generated, where N is the
// number of side-effecting operations. Continuation values are then used
// to pass data between the slices, usually from earlier to later.
struct GeneratedFunctionSlice {
  xls::Function* function = nullptr;
  const IOOp* after_op = nullptr;
  bool is_slice_before = false;
  std::vector<const clang::NamedDecl*> static_values;
  std::list<ContinuationValue> continuations_out;
  std::list<ContinuationInput> continuations_in;

  // Saved parameter order
  std::list<SideEffectingParameter> side_effecting_parameters;

  // Temporaries used during slice generation only, not optimization
  absl::flat_hash_map<const clang::NamedDecl*, ContinuationValue*>
      continuation_outputs_by_decl_top_context;
  absl::flat_hash_map<const clang::NamedDecl*, ContinuationInput*>
      continuation_inputs_by_decl_top_context;
};

// Encapsulates values produced when generating IR for a function
struct GeneratedFunction {
  const clang::FunctionDecl* clang_decl = nullptr;

  // TODO(seanhaskell): Remove when switching to new FSM
  xls::Function* xls_func = nullptr;

  std::list<GeneratedFunctionSlice> slices;

  int64_t declaration_count = 0;

  int64_t return_value_count = 0;

  bool in_synthetic_int = false;

  bool uses_on_reset = false;

  std::shared_ptr<LValue> this_lvalue;
  std::shared_ptr<LValue> return_lvalue;

  absl::flat_hash_map<const clang::NamedDecl*, uint64_t>
      declaration_order_by_name_;

  // Ordered for determinism
  // Use Translator::AddChannel() to add while checking uniqueness
  std::list<IOChannel> io_channels;

  // Sub procs that must be generated to use the function
  std::list<PipelinedLoopSubProc> sub_procs;

  // Sub procs may be in subroutines (owned by Translator)
  absl::flat_hash_map<const IOChannel*, const PipelinedLoopSubProc*>
      pipeline_loops_by_internal_channel;

  // ParamDecls and FieldDecls (for block as class)
  // Used for top channel injections
  LValueMap<const clang::NamedDecl*> lvalues_by_param;

  // All the IO Ops occurring within the function. Order matters here,
  //  as it is assumed that write() ops will depend only on values produced
  //  by read() ops occurring *before* them in the list.
  // Also, The XLS token will be threaded through the IO ops (Send, Receive)
  //  in the order specified in this list.
  // Use list for safe pointers to values
  std::list<IOOp> io_ops;

  // Number of non-channel ops added
  int no_channel_op_count = 0;

  // Saved parameter order
  // TODO(seanhaskell): Remove with old FSM
  std::list<SideEffectingParameter> side_effecting_parameters;

  // Global values built with this FunctionBuilder
  CValueMap<const clang::NamedDecl*> global_values;

  // Static declarations with initializers
  absl::flat_hash_map<const clang::NamedDecl*, ConstValue> static_values;

  // This must be remembered from call to call so generated channels are not
  // duplicated
  absl::flat_hash_map<IOChannel*, IOChannel*>
      generated_caller_channels_by_callee;

  // Enables state element re-use by tracking call by reference parameters
  // This is a multimap in order to detect cases with multiple calls
  absl::flat_hash_map<const clang::NamedDecl*,
                      absl::flat_hash_set<const clang::NamedDecl*>>
      caller_decls_by_callee_param;

  // XLS channel orders for all calls. Vector has an element for each
  // channel parameter.
  std::map<std::vector<ChannelBundle>, int64_t>
      state_index_by_called_channel_order;

  // For sub-blocks
  xls::Channel* direct_ins_channel = nullptr;

  // Temporary: should be empty after function is completed.
  // Note that these may be across multiple function slices.
  std::vector<xls::Param*> masked_op_params_to_remove;

  template <typename ValueType>
  std::vector<const clang::NamedDecl*> DeterministicKeyNames(
      const absl::flat_hash_map<const clang::NamedDecl*, ValueType>& map)
      const {
    std::vector<const clang::NamedDecl*> ret;
    for (const auto& [name, _] : map) {
      ret.push_back(name);
    }
    SortNamesDeterministically(ret);
    return ret;
  }
  void SortNamesDeterministically(
      std::vector<const clang::NamedDecl*>& names) const;
  std::vector<const clang::NamedDecl*> GetDeterministicallyOrderedStaticValues()
      const;
};

enum DebugIrTraceFlags {
  DebugIrTraceFlags_None = 0,
  DebugIrTraceFlags_LoopContext = 1,
  DebugIrTraceFlags_LoopControl = 2,
  DebugIrTraceFlags_FSMStates = 4,
  DebugIrTraceFlags_PrevStateIOReferences = 8,
  DebugIrTraceFlags_OptimizationWarnings = 16,
};

struct NextStateValue {
  // When the condition is true for multiple next state values,
  // the one with the lower priority is taken.
  // Whenever more than one next value is specified,
  // a priority must be specified, and all conditions must be valid.
  int64_t priority = -1L;
  std::string extra_label = "";
  TrackedBValue value;
  // condition being invalid indicates unconditional update (literal 1)
  TrackedBValue condition;
};

struct GenerateFSMInvocationReturn {
  TrackedBValue return_value;
  TrackedBValue returns_this_activation;
  absl::btree_multimap<const xls::StateElement*, NextStateValue>
      extra_next_state_values;
};

int Debug_CountNodes(const xls::Node* node,
                     std::set<const xls::Node*>& visited);
std::string Debug_NodeToInfix(TrackedBValue bval);
std::string Debug_NodeToInfix(const xls::Node* node, int64_t& n_printed);
std::string Debug_OpName(const IOOp& op);

void LogContinuations(const xlscc::GeneratedFunction& func);

// Simplifies `bval` (or any dependencies) where AND or OR ops can be
// short-circuited.
//
// Performs constexpr evaluation to perform:
// 1) OR(0, a, b) => OR(a, b)
// 2) OR(1, a, b) => 1
// 3) AND(0, a, b) => 0
// 4) AND(1, a, b) => AND(a, b)
absl::Status ShortCircuitBVal(TrackedBValue& bval, const xls::SourceInfo& loc);

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_TRANSLATOR_TYPES_H_
