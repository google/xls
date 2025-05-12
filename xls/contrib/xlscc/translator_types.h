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

#include "absl/base/attributes.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/Attr.h"
#include "clang/include/clang/AST/Attrs.inc"
#include "clang/include/clang/AST/ComputeDependence.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/node_manipulation.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/ir/bits.h"
#include "xls/ir/caret.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xlscc {

class Translator;

class ConstValue;
// Base class for immutable objects representing XLS[cc] value types
// These are not 1:1 with clang::Types, and do not represent qualifiers
//  such as const and reference.
class CType {
 public:
  virtual ~CType() = 0;
  virtual bool operator==(const CType& o) const = 0;
  bool operator!=(const CType& o) const;

  virtual absl::StatusOr<bool> ContainsLValues(Translator& translator) const;
  virtual int GetBitWidth() const;
  virtual explicit operator std::string() const;
  virtual xls::Type* GetXLSType(xls::Package* package) const;
  virtual bool StoredAsXLSBits() const;
  virtual absl::Status GetMetadata(
      Translator& translator, xlscc_metadata::Type* output,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const;
  virtual absl::Status GetMetadataValue(Translator& translator,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
  // We use this field to tell "char" declarations from explcitly-qualified
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
      Translator& translator, xlscc_metadata::StructField* output,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const;
  absl::Status GetMetadataValue(Translator* t, ConstValue const_value,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  bool operator==(const CType& o) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

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
  absl::StatusOr<int64_t> count_lvalue_compounds(Translator& translator) const;

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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  explicit operator std::string() const override;
  int GetBitWidth() const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

  std::shared_ptr<CType> GetPointeeType() const;

 private:
  std::shared_ptr<CType> pointee_type_;
};

enum class OpType { kNull = 0, kSend, kRecv, kSendRecv, kRead, kWrite, kTrace };
enum class InterfaceType { kNull = 0, kDirect, kFIFO, kMemory, kTrace };
enum class TraceType { kNull = 0, kAssert, kTrace };
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
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator, ConstValue const_value,
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

  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

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

// Similiar to CValue, but contains an xls::Value for a constant expression
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
}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_TRANSLATOR_TYPES_H_
