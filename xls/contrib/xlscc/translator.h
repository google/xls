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

#ifndef XLS_CONTRIB_XLSCC_TRANSLATOR_H_
#define XLS_CONTRIB_XLSCC_TRANSLATOR_H_

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "xls/common/logging/logging.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/hls_block.pb.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/ir/bits.h"
#include "xls/ir/caret.h"
#include "xls/ir/channel.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/solvers/z3_ir_translator.h"
#include "../z3/src/api/z3_api.h"

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
                                        const ConstValue const_value,
                                        xlscc_metadata::Value* output) const;

  template <typename Derived>
  const Derived* As() const {
    XLS_CHECK(Is<Derived>());
    return dynamic_cast<const Derived*>(this);
  }

  template <typename Derived>
  bool Is() const {
    return typeid(*this) == typeid(Derived);
  }
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
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
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
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
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
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  bool operator==(const CType& o) const override;

  xls::Type* GetXLSType(xls::Package* package) const override;
  bool StoredAsXLSBits() const override;

  inline int width() const { return width_; }
  inline bool is_signed() const { return is_signed_; }
  inline bool is_declared_as_char() const { return is_declared_as_char_; }

 private:
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

// C++ bool
class CBoolType : public CType {
 public:
  ~CBoolType() override;

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
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
  absl::Status GetMetadataValue(Translator* t, const ConstValue const_value,
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
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
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

 private:
  bool no_tuple_flag_;
  bool synthetic_int_flag_;
  std::vector<std::shared_ptr<CField>> fields_;
  absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>
      fields_by_name_;
};

class CInternalTuple : public CType {
 public:
  CInternalTuple(std::vector<std::shared_ptr<CType>> fields);

  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
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
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
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
  CArrayType(std::shared_ptr<CType> element, int size);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

  int GetSize() const;
  std::shared_ptr<CType> GetElementType() const;

 private:
  std::shared_ptr<CType> element_;
  int size_;
};

// Pointer in C/C++
class CPointerType : public CType {
 public:
  CPointerType(std::shared_ptr<CType> pointee_type);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

  std::shared_ptr<CType> GetPointeeType() const;

 private:
  std::shared_ptr<CType> pointee_type_;
};

// Reference in C/C++
class CReferenceType : public CType {
 public:
  CReferenceType(std::shared_ptr<CType> pointee_type);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
                                xlscc_metadata::Value* output) const override;
  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

  std::shared_ptr<CType> GetPointeeType() const;

 private:
  std::shared_ptr<CType> pointee_type_;
};

enum class OpType { kNull = 0, kSend, kRecv };
enum class InterfaceType { kNull = 0, kDirect, kFIFO };

// __xls_channel in C/C++
class CChannelType : public CType {
 public:
  CChannelType(std::shared_ptr<CType> item_type, OpType op_type);
  bool operator==(const CType& o) const override;
  int GetBitWidth() const override;
  explicit operator std::string() const override;
  absl::Status GetMetadata(Translator& translator, xlscc_metadata::Type* output,
                           absl::flat_hash_set<const clang::NamedDecl*>&
                               aliases_used) const override;
  absl::Status GetMetadataValue(Translator& translator,
                                const ConstValue const_value,
                                xlscc_metadata::Value* output) const override;

  std::shared_ptr<CType> GetItemType() const;
  OpType GetOpType() const;

  absl::StatusOr<bool> ContainsLValues(Translator& translator) const override;

 private:
  std::shared_ptr<CType> item_type_;
  OpType op_type_;
};

struct IOChannel;

// In general, the original clang::Expr used to assign the pointer is saved
// as an lvalue. However, it's necessary to create synthetic selects for
// conditional assignments to pointers. Ternaries are also generated this way
// for consistency.
class LValue {
 public:
  LValue() : is_null_(true) {}
  LValue(const clang::Expr* leaf) : leaf_(leaf) {
    XLS_CHECK_NE(leaf_, nullptr);
  }
  LValue(IOChannel* channel_leaf) : channel_leaf_(channel_leaf) {}
  LValue(xls::BValue cond, std::shared_ptr<LValue> lvalue_true,
         std::shared_ptr<LValue> lvalue_false)
      : cond_(cond), lvalue_true_(lvalue_true), lvalue_false_(lvalue_false) {
    XLS_CHECK(cond_.valid());
    XLS_CHECK(cond_.GetType()->IsBits());
    XLS_CHECK(cond_.BitCountOrDie() == 1);
    XLS_CHECK_NE(lvalue_true_.get(), nullptr);
    XLS_CHECK_NE(lvalue_false_.get(), nullptr);
  }
  LValue(const absl::flat_hash_map<int64_t, std::shared_ptr<LValue>>&
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

  bool is_select() const { return cond_.valid(); }
  xls::BValue cond() const { return cond_; }
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
    std::string ret;
    for (const auto& [idx, lval] : get_compounds()) {
      ret += absl::StrFormat("[%i]: %s ", idx,
                             lval ? lval->debug_string() : "(null)");
    }
    return ret;
  }

 private:
  bool is_null_ = false;
  const clang::Expr* leaf_ = nullptr;
  IOChannel* channel_leaf_ = nullptr;

  xls::BValue cond_;
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
  CValue() {}
  CValue(xls::BValue rvalue, std::shared_ptr<CType> type,
         bool disable_type_check = false,
         std::shared_ptr<LValue> lvalue = nullptr)
      : rvalue_(rvalue), lvalue_(lvalue), type_(std::move(type)) {
    if (!disable_type_check) {
      XLS_CHECK(!type_->StoredAsXLSBits() ||
                rvalue.BitCountOrDie() == type_->GetBitWidth());
      // Structs (and their aliases) can have compound lvalues and also rvalues
      XLS_CHECK(!(lvalue && !dynamic_cast<CReferenceType*>(type_.get()) &&
                  !dynamic_cast<CPointerType*>(type_.get()) &&
                  !dynamic_cast<CChannelType*>(type_.get()) &&
                  !dynamic_cast<CStructType*>(type_.get()) &&
                  !dynamic_cast<CInstantiableTypeAlias*>(type_.get())));
      // Pointers are stored as empty tuples in structs
      const bool rvalue_empty =
          !rvalue.valid() || (rvalue.GetType()->IsTuple() &&
                              rvalue.GetType()->AsTupleOrDie()->size() == 0);
      XLS_CHECK(!(lvalue && rvalue.valid()) ||
                dynamic_cast<CStructType*>(type_.get()) ||
                dynamic_cast<CInstantiableTypeAlias*>(type_.get()));
      // Only supporting lvalues of the form &... for pointers
      if (dynamic_cast<CPointerType*>(type_.get()) != nullptr ||
          dynamic_cast<CReferenceType*>(type_.get()) != nullptr ||
          dynamic_cast<CChannelType*>(type_.get()) != nullptr) {
        // Always get rvalue from lvalue, otherwise changes to the original
        // won't
        //  be reflected when the pointer is dereferenced.
        XLS_CHECK(rvalue_empty);
      } else if (dynamic_cast<CChannelType*>(type_.get()) != nullptr) {
        XLS_CHECK(rvalue_empty && lvalue != nullptr);
      } else if (dynamic_cast<CVoidType*>(type_.get()) != nullptr) {
        XLS_CHECK(rvalue_empty);
      } else {
        XLS_CHECK(rvalue.valid());
      }
    }
  }

  xls::BValue rvalue() const { return rvalue_; }
  std::shared_ptr<CType> type() const { return type_; }
  std::shared_ptr<LValue> lvalue() const { return lvalue_; }
  std::string debug_string() const {
    return absl::StrFormat("(rval=%s, type=%s, lval=%p)", rvalue_.ToString(),
                           (type_ != nullptr) ? std::string(*type_) : "(null)",
                           lvalue_.get());
  }
  bool operator==(const CValue& o) const {
    if (rvalue_.node() != o.rvalue_.node()) {
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
    XLS_CHECK(!is_valid || type_ != nullptr);
    return is_valid;
  }

 private:
  xls::BValue rvalue_;
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
    XLS_CHECK(disable_type_check || !type_->StoredAsXLSBits() ||
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

// Tracks information about an __xls_channel parameter to a function
struct IOChannel {
  // Unique within the function
  std::string unique_name;
  // Type of item the channel transfers
  std::shared_ptr<CType> item_type;
  // Direction of the port (in/out)
  OpType channel_op_type = OpType::kNull;
  // The total number of IO ops on the channel within the function
  // (IO ops are conditional, so this is the maximum in a real invocation)
  int total_ops = 0;
  // If not nullptr, the channel isn't explicitly present in the source
  // For example, the channels used for pipelined for loops
  xls::Channel* generated = nullptr;
};

// Tracks information about an IO op on an __xls_channel parameter to a function
struct IOOp {
  OpType op;

  IOChannel* channel = nullptr;

  // For calls to subroutines with IO inside
  const IOOp* sub_op;

  // Input __xls_channel parameters take tuple types containing a value for
  //  each read() op. This is the index of this op in the tuple.
  int channel_op_index;

  // Output value from function for IO op
  xls::BValue ret_value;

  // For reads: input value from function parameter for Recv op
  CValue input_value;

  bool is_blocking = true;

  // Source location for messages
  xls::SourceInfo op_location;

  // Must be sequenced after these ops via tokens
  // Ops be in GeneratedFunction::io_ops respecting this order
  std::vector<const IOOp*> after_ops;
};

enum class SideEffectingParameterType { kNull = 0, kIOOp, kStatic };

// Describes a generated parameter from IO, statics, etc
struct SideEffectingParameter {
  SideEffectingParameterType type = SideEffectingParameterType::kNull;
  std::string param_name;
  IOOp* io_op = nullptr;
  const clang::NamedDecl* static_value = nullptr;
};

// Encapsulates values produced when generating IR for a function
struct GeneratedFunction {
  const clang::FunctionDecl* clang_decl = nullptr;
  xls::Function* xls_func = nullptr;

  int64_t declaration_count = 0;

  int64_t return_value_count = 0;

  bool in_synthetic_int = false;

  std::shared_ptr<LValue> return_lvalue;

  std::shared_ptr<LValue> this_lvalue;

  absl::flat_hash_map<const clang::NamedDecl*, uint64_t>
      declaration_order_by_name_;

  std::list<IOChannel> io_channels;

  // Used for translating IO ops from callee to caller. Needs to be here
  // to map ops to the same channel across multiple calls to the same callee.
  absl::flat_hash_map<IOChannel*, IOChannel*> caller_channels_by_callee_channel;

  // Not all IO channels will be in these maps
  absl::flat_hash_map<const clang::NamedDecl*, IOChannel*> io_channels_by_decl;
  absl::flat_hash_map<IOChannel*, const clang::NamedDecl*> decls_by_io_channel;

  // All the IO Ops occurring within the function. Order matters here,
  //  as it is assumed that write() ops will depend only on values produced
  //  by read() ops occurring *before* them in the list.
  // Also, The XLS token will be threaded through the IO ops (Send, Receive)
  //  in the order specified in this list.
  // Use list for safe pointers to values
  std::list<IOOp> io_ops;

  // Saved parameter order
  std::list<SideEffectingParameter> side_effecting_parameters;

  // Global values built with this FunctionBuilder
  absl::flat_hash_map<const clang::NamedDecl*, CValue> global_values;

  // Static declarations with initializers
  absl::flat_hash_map<const clang::NamedDecl*, ConstValue> static_values;

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

int Debug_CountNodes(const xls::Node* node,
                     std::set<const xls::Node*>& visited);
std::string Debug_NodeToInfix(xls::BValue bval);
std::string Debug_NodeToInfix(const xls::Node* node, int64_t& n_printed);

// Encapsulates a context for translating Clang AST to XLS IR.
// This is roughly equivalent to a "scope" in C++. There will typically
//  be at least one context pushed into the context stack for each C++ scope.
// The Translator::PopContext() function will propagate certain values, such
//  as new CValues for assignments to variables declared outside the scope,
//  up to the next context / outer scope.
struct TranslationContext {
  xls::BValue not_full_condition_bval(const xls::SourceInfo& loc) const {
    if (!full_condition.valid()) {
      return fb->Literal(xls::UBits(0, 1), loc);
    }
    return fb->Not(full_condition, loc);
  }

  xls::BValue full_condition_bval(const xls::SourceInfo& loc) const {
    if (!full_condition.valid()) {
      return fb->Literal(xls::UBits(1, 1), loc);
    }
    return full_condition;
  }

  xls::BValue not_relative_condition_bval(const xls::SourceInfo& loc) const {
    if (!relative_condition.valid()) {
      return fb->Literal(xls::UBits(0, 1), loc);
    }
    return fb->Not(relative_condition, loc);
  }

  xls::BValue relative_condition_bval(const xls::SourceInfo& loc) const {
    if (!relative_condition.valid()) {
      return fb->Literal(xls::UBits(1, 1), loc);
    }
    return relative_condition;
  }

  void and_condition_util(xls::BValue and_condition, xls::BValue& mod_condition,
                          const xls::SourceInfo& loc) const {
    if (!mod_condition.valid()) {
      mod_condition = and_condition;
    } else {
      mod_condition = fb->And(mod_condition, and_condition, loc);
    }
  }

  void or_condition_util(xls::BValue or_condition, xls::BValue& mod_condition,
                         const xls::SourceInfo& loc) const {
    if (!mod_condition.valid()) {
      mod_condition = or_condition;
    } else {
      mod_condition = fb->Or(mod_condition, or_condition, loc);
    }
  }

  void print_vars() const {
    std::cerr << "Context {" << std::endl;
    std::cerr << "  vars:" << std::endl;
    for (const auto& var : variables) {
      std::cerr << "  -- " << var.first->getNameAsString() << ": "
                << var.second.rvalue().ToString() << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  void print_vars_infix() const {
    std::cerr << "Context {" << std::endl;
    std::cerr << "  vars:" << std::endl;
    for (const auto& var : variables) {
      std::cerr << "  -- " << var.first->getNameAsString() << ": "
                << Debug_NodeToInfix(var.second.rvalue()) << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  std::shared_ptr<CType> return_type;
  xls::BuilderBase* fb = nullptr;

  // Information being gathered about function currently being processed
  GeneratedFunction* sf = nullptr;

  // "this" uses the key of the clang::NamedDecl* of the method
  absl::flat_hash_map<const clang::NamedDecl*, CValue> variables;

  xls::BValue return_val;
  xls::BValue last_return_condition;
  // For "control flow": assignments after a return are conditional on this
  xls::BValue have_returned_condition;

  // Condition for assignments
  xls::BValue full_condition;
  xls::BValue full_condition_on_enter_block;
  xls::BValue relative_condition;

  // These flags control the behavior of break and continue statements
  bool in_for_body = false;
  bool in_switch_body = false;

  // Used in translating for loops
  // invalid indicates no break/continue
  xls::BValue relative_break_condition;
  xls::BValue relative_continue_condition;

  // Switch stuff
  // hit_break is set when a break is encountered inside of a switch body.
  // This signals from GenerateIR_Stmt() to GenerateIR_Switch().
  bool hit_break = false;
  // For checking for conditional breaks. If a break occurs in a context
  //  with a condition that's not equal to the enclosing "switch condition",
  //  ie that specified by the enclosing case or default, then a conditional
  //  break is detected, which is unsupported and an error.
  xls::BValue full_switch_cond;

  // Ignore pointer qualifiers in type translation. Normally pointers
  //  cause an unsupported error, but when this flag is true,
  //  "Foo*" is translated as "Foo".
  // This mode was created to handle the "this" pointer, which is mandatory
  //  to handle classes.
  bool ignore_pointers = false;

  // Assume for loops without pragmas are unrolled
  bool for_loops_default_unroll = false;

  // Flag set in pipelined for body
  // TODO(seanhaskell): Remove once all features are supported
  bool in_pipelined_for_body = false;
  int64_t outer_pipelined_loop_init_interval = 0;

  // When set to true, the expression is evaluated as an lvalue, for pointer
  // assignments
  bool lvalue_mode = false;

  // These flags control the behavior when the context / scope is exited
  bool propagate_up = true;
  bool propagate_break_up = true;
  bool propagate_continue_up = true;

  bool mask_assignments = false;

  // Don't create side-effects when exploring.
  bool mask_side_effects = false;
  bool any_side_effects_requested = false;
};

class Translator {
  void debug_prints(const TranslationContext& context);

 public:
  // Make unrolling configurable from main
  explicit Translator(
      bool error_on_init_interval = false, int64_t max_unroll_iters = 1000,
      int64_t warn_unroll_iters = 100,
      std::unique_ptr<CCParser> existing_parser = std::unique_ptr<CCParser>());
  ~Translator();

  // This function uses Clang to parse a source file and then walks its
  //  AST to discover global constructs. It will also scan the file
  //  and includes, recursively, for #pragma statements.
  //
  // Among these are functions, which can be used as entry points
  //  for translation to IR.
  //
  // source_filename must be .cc
  // Retains references to the TU until ~Translator()
  absl::Status ScanFile(std::string_view source_filename,
                        absl::Span<std::string_view> command_line_args);

  // Call after ScanFile, as the top function may be specified by #pragma
  // If none was found, an error is returned
  absl::StatusOr<std::string> GetEntryFunctionName() const;

  absl::Status SelectTop(std::string_view top_function_name);

  // Generates IR as an XLS function, that is, a pure function without
  //  IO / state / side effects.
  // If top_function is 0 or "" then top must be specified via pragma
  // force_static=true Means the function is not generated with a "this"
  //  parameter & output. It is generated as if static was specified in the
  //  method prototype.
  absl::StatusOr<GeneratedFunction*> GenerateIR_Top_Function(
      xls::Package* package, bool force_static = false,
      bool member_references_become_channels = false,
      int default_init_interval = 0);

  // Generates IR as an HLS block / XLS proc.
  absl::StatusOr<xls::Proc*> GenerateIR_Block(xls::Package* package,
                                              const HLSBlock& block,
                                              int top_level_init_interval = 0);

  // Generates IR as an HLS block / XLS proc.
  // Top is a method, block specification is extracted from the class.
  absl::StatusOr<xls::Proc*> GenerateIR_BlockFromClass(
      xls::Package* package, HLSBlock* block_spec_out,
      int top_level_init_interval = 0);

  // Ideally, this would be done using the opt_main tool, but for now
  //  codegen is done by XLS[cc] for combinational blocks.
  absl::Status InlineAllInvokes(xls::Package* package);

  // Generate some useful metadata after either GenerateIR_Top_Function() or
  //  GenerateIR_Block() has run.
  absl::StatusOr<xlscc_metadata::MetadataOutput> GenerateMetadata();
  absl::Status GenerateFunctionMetadata(
      const clang::FunctionDecl* func,
      xlscc_metadata::FunctionPrototype* output,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used);
  void AddSourceInfoToPackage(xls::Package& package);

  inline void SetIOTestMode() { io_test_mode_ = true; }

 private:
  friend class CInstantiableTypeAlias;
  friend class CStructType;
  friend class CInternalTuple;

  std::function<std::optional<std::string>(xls::Fileno)> LookUpInPackage();
  template <typename... Args>
  std::string ErrorMessage(const xls::SourceInfo& loc,
                           const absl::FormatSpec<Args...>& format,
                           const Args&... args) {
    std::string result = absl::StrFormat(format, args...);
    for (const xls::SourceLocation& location : loc.locations) {
      absl::StrAppend(&result, "\n", PrintCaret(LookUpInPackage(), location));
    }
    return result;
  }

  // This object is used to push a new translation context onto the stack
  //  and then to pop it via RAII. This guard provides options for which bits of
  //  context to propagate up when popping it from the stack.
  struct PushContextGuard {
    PushContextGuard(Translator& translator, const xls::SourceInfo& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
    }
    PushContextGuard(Translator& translator, xls::BValue and_condition,
                     const xls::SourceInfo& loc)
        : translator(translator), loc(loc) {
      translator.PushContext();
      absl::Status status = translator.and_condition(and_condition, loc);
      if (!status.ok()) {
        XLS_LOG(ERROR) << status.message();
      }
    }
    ~PushContextGuard() {
      absl::Status status = translator.PopContext(loc);
      if (!status.ok()) {
        XLS_LOG(ERROR) << status.message();
      }
    }

    Translator& translator;
    xls::SourceInfo loc;
  };

  // This guard makes pointers translate, instead of generating errors, for a
  //  period determined by RAII.
  struct IgnorePointersGuard {
    explicit IgnorePointersGuard(Translator& translator)
        : translator(translator), enabled(false) {}
    ~IgnorePointersGuard() {
      if (enabled) {
        translator.context().ignore_pointers = prev_val;
      }
    }

    void enable() {
      enabled = true;
      prev_val = translator.context().ignore_pointers;
      translator.context().ignore_pointers = true;
    }

    Translator& translator;
    bool enabled;
    bool prev_val;
  };

  // This guard makes assignments no-ops, for a period determined by RAII.
  struct MaskAssignmentsGuard {
    explicit MaskAssignmentsGuard(Translator& translator)
        : translator(translator),
          prev_val(translator.context().mask_side_effects) {
      translator.context().mask_assignments = true;
    }
    ~MaskAssignmentsGuard() {
      translator.context().mask_assignments = prev_val;
    }

    Translator& translator;
    bool prev_val;
  };

  // This guard evaluates pointer expressions as lvalues, for a period
  // determined by RAII.
  struct LValueModeGuard {
    explicit LValueModeGuard(Translator& translator)
        : translator(translator), prev_val(translator.context().lvalue_mode) {
      translator.context().lvalue_mode = true;
    }
    ~LValueModeGuard() { translator.context().lvalue_mode = prev_val; }

    Translator& translator;
    bool prev_val;
  };

  // The maximum number of iterations before loop unrolling fails.
  const int64_t max_unroll_iters_;
  // The maximum number of iterations before loop unrolling prints a warning.
  const int64_t warn_unroll_iters_;

  // Generate an error when an init interval > supported is requested?
  const bool error_on_init_interval_;

  // Makes translation of external channel parameters optional,
  // so that IO operations can be generated without calling GenerateIR_Block()
  bool io_test_mode_ = false;

  struct InstTypeHash {
    size_t operator()(
        const std::shared_ptr<CInstantiableTypeAlias>& value) const {
      const std::size_t hash =
          std::hash<const clang::NamedDecl*>()(value->base());
      return size_t(hash);
    }
  };

  struct InstTypeEq {
    bool operator()(const std::shared_ptr<CInstantiableTypeAlias>& x,
                    const std::shared_ptr<CInstantiableTypeAlias>& y) const {
      return *x == *y;
    }
  };

  absl::flat_hash_map<std::shared_ptr<CInstantiableTypeAlias>,
                      std::shared_ptr<CType>, InstTypeHash, InstTypeEq>
      inst_types_;
  absl::flat_hash_map<const clang::NamedDecl*,
                      std::unique_ptr<GeneratedFunction>>
      inst_functions_;

  void print_types() {
    std::cerr << "Types {" << std::endl;
    for (const auto& var : inst_types_) {
      std::cerr << "  -- " << std::string(*var.first) << ": "
                << std::string(*var.second) << std::endl;
    }
    std::cerr << "}" << std::endl;
  }

  // The translator assumes NamedDecls are unique. This set is used to
  //  generate an error if that assumption is violated.
  absl::flat_hash_set<const clang::NamedDecl*> unique_decl_ids_;

  // Scans for top-level function candidates
  absl::Status VisitFunction(const clang::FunctionDecl* funcdecl);
  absl::Status ScanFileForPragmas(std::string filename);

  absl::flat_hash_map<const clang::FunctionDecl*, std::string>
      xls_names_for_functions_generated_;

  int next_asm_number_ = 1;
  int next_for_number_ = 1;

  mutable std::unique_ptr<clang::MangleContext> mangler_;

  TranslationContext& PushContext();
  absl::Status PopContext(const xls::SourceInfo& loc);
  absl::Status PropagateVariables(const TranslationContext& from,
                                  TranslationContext& to,
                                  const xls::SourceInfo& loc);

  xls::Package* package_ = nullptr;
  int default_init_interval_ = 0;

  // Initially contains keys for the parameters of the top function,
  // then subroutine parameters are added as they are translated.
  absl::flat_hash_map<const clang::NamedDecl*, xls::Channel*>
      external_channels_by_decl_;

  // Used as a stack, but need to peek 2nd to top
  std::list<TranslationContext> context_stack_;

  TranslationContext& context();

  absl::Status and_condition(xls::BValue and_condition,
                             const xls::SourceInfo& loc);

  absl::StatusOr<CValue> Generate_UnaryOp(const clang::UnaryOperator* uop,
                                          const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_Synthetic_ByOne(
      xls::Op xls_op, bool is_pre, CValue sub_value,
      const clang::Expr* sub_expr,  // For assignment
      const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_BinaryOp(
      clang::BinaryOperator::Opcode clang_op, bool is_assignment,
      std::shared_ptr<CType> result_type, const clang::Expr* lhs,
      const clang::Expr* rhs, const xls::SourceInfo& loc);
  absl::Status MinSizeArraySlices(CValue& true_cv, CValue& false_cv,
                                  std::shared_ptr<CType>& result_type,
                                  const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_TernaryOp(xls::BValue cond, CValue true_cv,
                                            CValue false_cv,
                                            std::shared_ptr<CType> result_type,
                                            const xls::SourceInfo& loc);
  absl::StatusOr<CValue> Generate_TernaryOp(std::shared_ptr<CType> result_type,
                                            const clang::Expr* cond_expr,
                                            const clang::Expr* true_expr,
                                            const clang::Expr* false_expr,
                                            const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_Expr(const clang::Expr* expr,
                                         const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_Expr(std::shared_ptr<LValue> expr,
                                         const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_MemberExpr(const clang::MemberExpr* expr,
                                               const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GenerateIR_Call(const clang::CallExpr* call,
                                         const xls::SourceInfo& loc);

  absl::StatusOr<CValue> GenerateIR_Call(
      const clang::FunctionDecl* funcdecl,
      std::vector<const clang::Expr*> expr_args, xls::BValue* this_inout,
      std::shared_ptr<LValue>* this_lval, const xls::SourceInfo& loc);

  absl::Status FailIfTypeHasDtors(const clang::CXXRecordDecl* cxx_record);
  bool LValueContainsOnlyChannels(std::shared_ptr<LValue> lvalue);

  absl::Status TranslateAddCallerChannelsByCalleeChannel(
      std::shared_ptr<LValue> caller_lval, std::shared_ptr<LValue> callee_lval,
      const xls::SourceInfo& loc);

  absl::Status ValidateLValue(std::shared_ptr<LValue> lval,
                              const xls::SourceInfo& loc);

  absl::StatusOr<std::shared_ptr<LValue>> TranslateLValueChannels(
      std::shared_ptr<LValue> outer_lval,
      const absl::flat_hash_map<IOChannel*, IOChannel*>&
          inner_channels_by_outer_channel,
      const xls::SourceInfo& loc);

  // This is a work-around for non-const operator [] needing to return
  //  a reference to the object being modified.
  absl::StatusOr<bool> ApplyArrayAssignHack(
      const clang::CXXOperatorCallExpr* op_call, const xls::SourceInfo& loc,
      CValue* output);

  struct PreparedBlock {
    GeneratedFunction* xls_func;
    std::vector<xls::BValue> args;
    // Not used for direct-ins
    absl::flat_hash_map<IOChannel*, xls::Channel*>
        xls_channel_by_function_channel;
    absl::flat_hash_map<const IOOp*, int> arg_index_for_op;
    absl::flat_hash_map<const IOOp*, int> return_index_for_op;
    absl::flat_hash_map<const clang::NamedDecl*, int64_t>
        return_index_for_static;
    absl::flat_hash_map<const clang::NamedDecl*, int64_t>
        state_index_for_static;
    int64_t state_init_count = 0;
    xls::BValue token;
  };

  struct ExternalChannelInfo {
    const clang::NamedDecl* decl;
    std::shared_ptr<CChannelType> channel_type;
    InterfaceType interface_type;
    bool extra_return = false;
  };

  absl::StatusOr<xls::Proc*> GenerateIR_Block(
      xls::Package* package, const HLSBlock& block,
      std::shared_ptr<CStructType> this_type,
      const clang::CXXRecordDecl* this_decl,
      const std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& body_loc, int top_level_init_interval,
      bool force_static, bool member_references_become_channels);

  // Verifies the function prototype in the Clang AST and HLSBlock are sound.
  absl::Status GenerateIRBlockCheck(
      const HLSBlock& block, const std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& body_loc);

  // Creates xls::Channels in the package
  absl::Status GenerateExternalChannels(
      const HLSBlock& block, const std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& loc);

  absl::StatusOr<xls::Value> GenerateTopClassInitValue(
      std::shared_ptr<CStructType> this_type,
      // Can be nullptr
      const clang::CXXRecordDecl* this_decl, const xls::SourceInfo& body_loc);

  // Prepares IO channels for generating XLS Proc
  // definition can be null, and then channels_by_name can also be null. They
  // are only used for direct-ins
  absl::Status GenerateIRBlockPrepare(
      PreparedBlock& prepared, xls::ProcBuilder& pb, int64_t next_return_index,
      int64_t next_state_index, std::shared_ptr<CStructType> this_type,
      // Can be nullptr
      const clang::CXXRecordDecl* this_decl,
      const std::list<ExternalChannelInfo>& top_decls,
      const xls::SourceInfo& body_loc);

  // Returns last invoke's return value
  absl::StatusOr<xls::BValue> GenerateIOInvokes(
      PreparedBlock& prepared, xls::ProcBuilder& pb,
      const xls::SourceInfo& body_loc);

  struct IOOpReturn {
    bool generate_expr;
    CValue value;
  };
  // Checks if an expression is an IO op, and if so, generates the value
  //  to replace it in IR generation.
  absl::StatusOr<IOOpReturn> InterceptIOOp(const clang::Expr* expr,
                                           const xls::SourceInfo& loc);

  // IOOp must have io_call, and op members filled in
  // Returns permanent IOOp pointer
  absl::StatusOr<IOOp*> AddOpToChannel(IOOp& op, IOChannel* channel_param,
                                       const xls::SourceInfo& loc,
                                       bool mask = false);
  absl::Status CreateChannelParam(
      const clang::NamedDecl* channel_name,
      const std::shared_ptr<CChannelType>& channel_type,
      const xls::SourceInfo& loc);
  IOChannel* AddChannel(const clang::NamedDecl* decl, IOChannel new_channel);
  absl::StatusOr<std::shared_ptr<CChannelType>> GetChannelType(
      const clang::QualType& channel_type, clang::ASTContext& ctx,
      const xls::SourceInfo& loc);

  absl::StatusOr<bool> ExprIsChannel(const clang::Expr* object,
                                     const xls::SourceInfo& loc);
  absl::StatusOr<bool> TypeIsChannel(clang::QualType param,
                                     const xls::SourceInfo& loc);
  // Returns nullptr if the parameter isn't a channel
  absl::StatusOr<IOChannel*> GetChannelForExprOrNull(const clang::Expr* object);
  // Returns nullptr if the parameter isn't a channel
  absl::StatusOr<const clang::NamedDecl*> GetChannelParamForExprOrNull(
      const clang::Expr* object);

  absl::Status GenerateIR_Compound(const clang::Stmt* body,
                                   clang::ASTContext& ctx);
  absl::Status GenerateIR_Stmt(const clang::Stmt* stmt, clang::ASTContext& ctx);
  absl::Status GenerateIR_StaticDecl(const clang::VarDecl* vard,
                                     const clang::NamedDecl* namedecl,
                                     const xls::SourceInfo& loc);

  absl::Status CheckInitIntervalValidity(int initiation_interval_arg,
                                         const xls::SourceInfo& loc);

  // init, cond, and inc can be nullptr
  absl::Status GenerateIR_Loop(bool always_first_iter, const clang::Stmt* init,
                               const clang::Expr* cond_expr,
                               const clang::Stmt* inc, const clang::Stmt* body,
                               const clang::PresumedLoc& presumed_loc,
                               const xls::SourceInfo& loc,
                               clang::ASTContext& ctx);

  // init, cond, and inc can be nullptr
  absl::Status GenerateIR_UnrolledLoop(bool always_first_iter,
                                       const clang::Stmt* init,
                                       const clang::Expr* cond_expr,
                                       const clang::Stmt* inc,
                                       const clang::Stmt* body,
                                       clang::ASTContext& ctx,
                                       const xls::SourceInfo& loc);
  // init, cond, and inc can be nullptr
  absl::Status GenerateIR_PipelinedLoop(
      bool always_first_iter, const clang::Stmt* init,
      const clang::Expr* cond_expr, const clang::Stmt* inc,
      const clang::Stmt* body, int64_t initiation_interval_arg,
      clang::ASTContext& ctx, const xls::SourceInfo& loc);
  absl::Status GenerateIR_PipelinedLoopBody(
      const clang::Expr* cond_expr, const clang::Stmt* inc,
      const clang::Stmt* body, int64_t init_interval, clang::ASTContext& ctx,
      std::string_view name_prefix, IOChannel* context_out_channel,
      IOChannel* context_in_channel, xls::Type* context_xls_type,
      std::shared_ptr<CStructType> context_ctype,
      absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<LValue>>*
          lvalues_out,
      const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
          variable_field_indices,
      const std::vector<const clang::NamedDecl*>& variable_fields_order,
      std::vector<const clang::NamedDecl*>& vars_changed_in_body,
      const xls::SourceInfo& loc);
  absl::Status GenerateIR_Switch(const clang::SwitchStmt* switchst,
                                 clang::ASTContext& ctx,
                                 const xls::SourceInfo& loc);

  struct ResolvedInheritance {
    std::shared_ptr<CField> base_field;
    std::shared_ptr<const CStructType> resolved_struct;
    const clang::NamedDecl* base_field_name;
  };

  absl::StatusOr<ResolvedInheritance> ResolveInheritance(
      std::shared_ptr<CType> sub_type, std::shared_ptr<CType> to_type);

  absl::StatusOr<xls::BValue> GenTypeConvert(CValue const& in,
                                             std::shared_ptr<CType> out_type,
                                             const xls::SourceInfo& loc);
  absl::StatusOr<xls::BValue> GenBoolConvert(CValue const& in,
                                             const xls::SourceInfo& loc);

  absl::StatusOr<xls::Value> CreateDefaultRawValue(std::shared_ptr<CType> t,
                                                   const xls::SourceInfo& loc);
  absl::StatusOr<xls::BValue> CreateDefaultValue(std::shared_ptr<CType> t,
                                                 const xls::SourceInfo& loc);
  absl::StatusOr<CValue> CreateInitListValue(
      const std::shared_ptr<CType>& t, const clang::InitListExpr* init_list,
      const xls::SourceInfo& loc);
  absl::StatusOr<CValue> CreateInitValue(const std::shared_ptr<CType>& ctype,
                                         const clang::Expr* initializer,
                                         const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<LValue>> CreateReferenceValue(
      const clang::Expr* initializer, const xls::SourceInfo& loc);
  absl::StatusOr<CValue> GetOnReset(const xls::SourceInfo& loc);
  absl::StatusOr<bool> DeclIsOnReset(const clang::NamedDecl* decl);
  absl::StatusOr<CValue> GetIdentifier(const clang::NamedDecl* decl,
                                       const xls::SourceInfo& loc);

  absl::StatusOr<CValue> TranslateVarDecl(const clang::VarDecl* decl,
                                          const xls::SourceInfo& loc);
  absl::StatusOr<CValue> TranslateEnumConstantDecl(
      const clang::EnumConstantDecl* decl, const xls::SourceInfo& loc);
  absl::Status Assign(const clang::NamedDecl* lvalue, const CValue& rvalue,
                      const xls::SourceInfo& loc);
  absl::Status Assign(const clang::Expr* lvalue, const CValue& rvalue,
                      const xls::SourceInfo& loc);
  absl::Status Assign(std::shared_ptr<LValue> lvalue, const CValue& rvalue,
                      const xls::SourceInfo& loc);
  absl::StatusOr<const clang::NamedDecl*> GetThisDecl(
      const xls::SourceInfo& loc, bool for_declaration = false);
  absl::StatusOr<CValue> PrepareRValueWithSelect(
      const CValue& lvalue, const CValue& rvalue,
      const xls::BValue& relative_condition, const xls::SourceInfo& loc);

  absl::Status DeclareVariable(const clang::NamedDecl* lvalue,
                               const CValue& rvalue, const xls::SourceInfo& loc,
                               bool check_unique_ids = true);

  absl::Status DeclareStatic(const clang::NamedDecl* lvalue,
                             const ConstValue& init, const xls::SourceInfo& loc,
                             bool check_unique_ids = true);

  // If the decl given is a forward declaration, the definition with a body will
  // be returned. This is done in multiple places because the ParamVarDecls vary
  // in each declaration.
  absl::StatusOr<const clang::Stmt*> GetFunctionBody(
      const clang::FunctionDecl*& funcdecl);
  absl::StatusOr<GeneratedFunction*> GenerateIR_Function(
      const clang::FunctionDecl* funcdecl, std::string_view name_override = "",
      bool force_static = false,
      bool member_references_become_channels = false);

  struct StrippedType {
    StrippedType(clang::QualType base, bool is_ref)
        : base(base), is_ref(is_ref) {}
    clang::QualType base;
    bool is_ref;
  };

  absl::StatusOr<StrippedType> StripTypeQualifiers(clang::QualType t);
  absl::Status ScanStruct(const clang::RecordDecl* sd);

  absl::StatusOr<std::shared_ptr<CType>> InterceptBuiltInStruct(
      const clang::RecordDecl* sd);

  absl::StatusOr<std::shared_ptr<CType>> TranslateTypeFromClang(
      clang::QualType t, const xls::SourceInfo& loc);
  absl::StatusOr<xls::Type*> TranslateTypeToXLS(std::shared_ptr<CType> t,
                                                const xls::SourceInfo& loc);
  absl::StatusOr<std::shared_ptr<CType>> ResolveTypeInstance(
      std::shared_ptr<CType> t);
  absl::StatusOr<std::shared_ptr<CType>> ResolveTypeInstanceDeeply(
      std::shared_ptr<CType> t);
  absl::StatusOr<GeneratedFunction*> TranslateFunctionToXLS(
      const clang::FunctionDecl* decl);
  absl::StatusOr<bool> FunctionIsInSyntheticInt(
      const clang::FunctionDecl* decl);

  absl::StatusOr<int64_t> EvaluateInt64(const clang::Expr& expr,
                                        const class clang::ASTContext& ctx,
                                        const xls::SourceInfo& loc);
  absl::StatusOr<bool> EvaluateBool(const clang::Expr& expr,
                                    const class clang::ASTContext& ctx,
                                    const xls::SourceInfo& loc);
  absl::StatusOr<xls::Value> EvaluateNode(xls::Node* node,
                                          const xls::SourceInfo& loc,
                                          bool do_check = true);

  absl::Status ShortCircuitNode(xls::Node* node, xls::BValue& top_bval,
                                xls::Node* parent,
                                absl::flat_hash_set<xls::Node*>& visited,
                                const xls::SourceInfo& loc);
  absl::Status ShortCircuitBVal(xls::BValue& bval, const xls::SourceInfo& loc);
  absl::StatusOr<xls::Value> EvaluateBVal(xls::BValue bval,
                                          const xls::SourceInfo& loc,
                                          bool do_check = true);
  absl::StatusOr<int64_t> EvaluateBValInt64(xls::BValue bval,
                                            const xls::SourceInfo& loc,
                                            bool do_check = true);
  absl::StatusOr<Z3_lbool> IsBitSatisfiable(
      xls::Node* node, Z3_solver& solver,
      xls::solvers::z3::IrTranslator& z3_translator);

  absl::StatusOr<ConstValue> TranslateBValToConstVal(const CValue& bvalue,
                                                     const xls::SourceInfo& loc,
                                                     bool do_check = true);

  absl::StatusOr<xls::Op> XLSOpcodeFromClang(clang::BinaryOperatorKind clang_op,
                                             const CType& left_type,
                                             const CType& result_type,
                                             const xls::SourceInfo& loc);
  std::string XLSNameMangle(clang::GlobalDecl decl) const;

  xls::BValue MakeFunctionReturn(const std::vector<xls::BValue>& bvals,
                                 const xls::SourceInfo& loc);
  xls::BValue GetFunctionReturn(xls::BValue val, int index,
                                int expected_returns,
                                const clang::FunctionDecl* func,
                                const xls::SourceInfo& loc);

  absl::Status GenerateMetadataCPPName(const clang::NamedDecl* decl_in,
                                       xlscc_metadata::CPPName* name_out);

  absl::Status GenerateMetadataType(
      const clang::QualType& type_in, xlscc_metadata::Type* type_out,
      absl::flat_hash_set<const clang::NamedDecl*>& aliases_used);

  absl::StatusOr<xlscc_metadata::IntType> GenerateSyntheticInt(
      std::shared_ptr<CType> ctype);

  // StructUpdate builds and returns a new CValue for a struct with the
  // value of one field changed. The other fields, if any, take their values
  // from struct_before, and the new value for the field named by field_name is
  // set to rvalue. The type of structure built is specified by type.
  absl::StatusOr<CValue> StructUpdate(CValue struct_before, CValue rvalue,
                                      const clang::NamedDecl* field_name,
                                      const CStructType& type,
                                      const xls::SourceInfo& loc);
  // Creates an BValue for a struct of type stype from field BValues given in
  //  order within bvals.
  xls::BValue MakeStructXLS(const std::vector<xls::BValue>& bvals,
                            const CStructType& stype,
                            const xls::SourceInfo& loc);
  // Creates a Value for a struct of type stype from field BValues given in
  //  order within bvals.
  xls::Value MakeStructXLS(const std::vector<xls::Value>& vals,
                           const CStructType& stype);
  // Returns the BValue for the field with index "index" from a BValue for a
  //  struct of type "type"
  // This version cannot be static because it needs access to the
  //  FunctionBuilder from the context
  xls::BValue GetStructFieldXLS(xls::BValue val, int index,
                                const CStructType& type,
                                const xls::SourceInfo& loc);

  // Returns a BValue for a copy of array_to_update with slice_to_write replaced
  // at start_index.
  absl::StatusOr<xls::BValue> UpdateArraySlice(xls::BValue array_to_update,
                                               xls::BValue start_index,
                                               xls::BValue slice_to_write,
                                               const xls::SourceInfo& loc);
  int64_t ArrayBValueWidth(xls::BValue array_bval);

 public:
  // This version is public because it needs to be accessed by CStructType
  static absl::StatusOr<xls::Value> GetStructFieldXLS(xls::Value val, int index,
                                                      const CStructType& type);

 private:
  // Gets the appropriate XLS type for a struct. For example, it might be an
  //  xls::Tuple, or if #pragma hls_notuple was specified, it might be
  //  the single field's type
  absl::StatusOr<xls::Type*> GetStructXLSType(
      const std::vector<xls::Type*>& members, const CStructType& type,
      const xls::SourceInfo& loc);

  // "Flexible Tuple" functions
  // These functions will create a tuple if there is more than one
  //  field, or they will pass through the value if there
  //  is exactly 1 value. 0 values is unsupported.
  // This makes the generated IR less cluttered, as extra single-item tuples
  //  aren't generated.

  // Wraps bvals in a "flexible tuple"
  xls::BValue MakeFlexTuple(const std::vector<xls::BValue>& bvals,
                            const xls::SourceInfo& loc);
  // Gets the XLS type for a "flexible tuple" made from these elements
  xls::Type* GetFlexTupleType(const std::vector<xls::Type*>& members);
  // Gets the value of a field in a "flexible tuple"
  // val is the "flexible tuple" value
  // index is the index of the field
  // n_fields is the total number of fields
  // op_name is passed to the FunctionBuilder
  xls::BValue GetFlexTupleField(xls::BValue val, int index, int n_fields,
                                const xls::SourceInfo& loc,
                                std::string_view op_name = "");
  // Changes the value of a field in a "flexible tuple"
  // tuple_val is the "flexible tuple" value
  // new_val is the value to set the field to
  // index is the index of the field
  // n_fields is the total number of fields
  xls::BValue UpdateFlexTupleField(xls::BValue tuple_val, xls::BValue new_val,
                                   int index, int n_fields,
                                   const xls::SourceInfo& loc);

  void FillLocationProto(const clang::SourceLocation& location,
                         xlscc_metadata::SourceLocation* location_out);
  void FillLocationRangeProto(const clang::SourceRange& range,
                              xlscc_metadata::SourceLocationRange* range_out);

  std::unique_ptr<CCParser> parser_;

  // Convenience calls to CCParser
  absl::StatusOr<Pragma> FindPragmaForLoc(const clang::SourceLocation& loc);
  absl::StatusOr<Pragma> FindPragmaForLoc(const clang::PresumedLoc& ploc);
  std::string LocString(const xls::SourceInfo& loc);
  xls::SourceInfo GetLoc(const clang::Stmt& stmt);
  xls::SourceInfo GetLoc(const clang::Decl& decl);
  inline std::string LocString(const clang::Decl& decl) {
    return LocString(GetLoc(decl));
  }
  clang::PresumedLoc GetPresumedLoc(const clang::Stmt& stmt);
  clang::PresumedLoc GetPresumedLoc(const clang::Decl& decl);
};

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_TRANSLATOR_H_
