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

#include "xls/contrib/xlscc/translator.h"

#include <cstdint>
#include <memory>
#include <regex>  // NOLINT
#include <sstream>
#include <typeinfo>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "clang/include/clang/AST/APValue.h"
#include "clang/include/clang/AST/AST.h"
#include "clang/include/clang/AST/ASTConsumer.h"
#include "clang/include/clang/AST/ASTContext.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/DeclTemplate.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/ExprCXX.h"
#include "clang/include/clang/AST/Mangle.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/AST/RecursiveASTVisitor.h"
#include "clang/include/clang/AST/Stmt.h"
#include "clang/include/clang/AST/TemplateBase.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/ABI.h"
#include "clang/include/clang/Basic/FileManager.h"
#include "clang/include/clang/Basic/OperatorKinds.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Basic/SourceManager.h"
#include "clang/include/clang/Basic/Specifiers.h"
#include "clang/include/clang/Basic/TypeTraits.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/VirtualFileSystem.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/cc_parser.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/caret.h"
#include "xls/ir/channel.h"
#include "xls/ir/fileno.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/dfe_pass.h"
#include "xls/passes/identity_removal_pass.h"
#include "xls/passes/inlining_pass.h"
#include "xls/passes/standard_pipeline.h"
#include "xls/passes/tuple_simplification_pass.h"
#include "xls/passes/verifier_checker.h"
#include "xls/solvers/z3_ir_translator.h"
#include "../z3/src/api/z3_api.h"
#include "re2/re2.h"

using std::list;
using std::ostringstream;
using std::shared_ptr;
using std::string;
using std::vector;

namespace {

// Clang has a complex multiple inheritance hierarchy, but it's not polymorphic,
// so we can't use down_cast which uses dynamic_cast.
template <typename To, typename From>  // use like this: down_cast<T*>(foo);
inline To clang_down_cast(From* f) {   // so we only accept pointers
  static_assert((std::is_base_of<From, std::remove_pointer_t<To>>::value),
                "target type not derived from source type");

  return static_cast<To>(f);
}

// Returns monotonically increasing time in seconds
double doubletime() {
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  return tv.tv_sec + static_cast<double>(tv.tv_usec) / 1000000.0;
}

}  // namespace

namespace xlscc {

CType::~CType() {}

bool CType::operator!=(const CType& o) const { return !(*this == o); }

int CType::GetBitWidth() const { return 0; }

CType::operator std::string() const { return "CType"; }

xls::Type* CType::GetXLSType(xls::Package* /*package*/) const {
  XLS_LOG(FATAL) << "GetXLSType() unsupported in CType base class";
  return nullptr;
}

bool CType::StoredAsXLSBits() const { return false; }

absl::Status CType::GetMetadata(Translator& translator,
                                xlscc_metadata::Type* output) const {
  return absl::UnimplementedError(
      "GetMetadata unsupported in CType base class");
}

absl::Status CType::GetMetadataValue(Translator& translator,
                                     const ConstValue const_value,
                                     xlscc_metadata::Value* output) const {
  return absl::UnimplementedError(
      "GetMetadataValue unsupported in CType base class");
}

CVoidType::~CVoidType() {}

int CVoidType::GetBitWidth() const {
  XLS_CHECK(false);
  return 0;
}

absl::Status CVoidType::GetMetadata(Translator& translator,
                                    xlscc_metadata::Type* output) const {
  (void)output->mutable_as_void();
  return absl::OkStatus();
}

absl::Status CVoidType::GetMetadataValue(Translator& translator,
                                         const ConstValue const_value,
                                         xlscc_metadata::Value* output) const {
  return absl::OkStatus();
}

CVoidType::operator std::string() const { return "void"; }

bool CVoidType::operator==(const CType& o) const { return o.Is<CVoidType>(); }

CBitsType::CBitsType(int width) : width_(width) {}

CBitsType::~CBitsType() {}

int CBitsType::GetBitWidth() const { return width_; }

CBitsType::operator std::string() const {
  return absl::StrFormat("bits[%d]", width_);
}

absl::Status CBitsType::GetMetadata(Translator& translator,
                                    xlscc_metadata::Type* output) const {
  output->mutable_as_bits()->set_width(width_);
  return absl::OkStatus();
}

absl::Status CBitsType::GetMetadataValue(Translator& translator,
                                         const ConstValue const_value,
                                         xlscc_metadata::Value* output) const {
  vector<uint8_t> bytes = const_value.rvalue().bits().ToBytes();
  output->set_as_bits(bytes.data(), bytes.size());
  return absl::OkStatus();
}

bool CBitsType::StoredAsXLSBits() const { return true; }

bool CBitsType::operator==(const CType& o) const {
  if (!o.Is<CBitsType>()) return false;
  const auto* o_derived = o.As<CBitsType>();
  return width_ == o_derived->width_;
}

CIntType::~CIntType() {}

CIntType::CIntType(int width, bool is_signed, bool is_declared_as_char)
    : width_(width),
      is_signed_(is_signed),
      is_declared_as_char_(is_declared_as_char) {}

xls::Type* CIntType::GetXLSType(xls::Package* package) const {
  return package->GetBitsType(width_);
}

bool CIntType::operator==(const CType& o) const {
  if (!o.Is<CIntType>()) return false;
  const auto* o_derived = o.As<CIntType>();
  if (width_ != o_derived->width_) return false;
  return is_signed_ == o_derived->is_signed_;
}

int CIntType::GetBitWidth() const { return width_; }

bool CIntType::StoredAsXLSBits() const { return true; }

CIntType::operator std::string() const {
  const std::string pre = is_signed_ ? "" : "unsigned_";
  if (width_ == 32) {
    return pre + "int";
  } else if (width_ == 1) {
    return pre + "pseudobool";
  } else if (width_ == 64) {
    return pre + "int64_t";
  } else if (width_ == 16) {
    return pre + "short";
  } else if (width_ == 8) {
    return pre + (is_declared_as_char() ? "char" : "int8_t");
  } else {
    XLS_CHECK(0);
    return "Unsupported";
  }
}

absl::Status CIntType::GetMetadata(Translator& translator,
                                   xlscc_metadata::Type* output) const {
  output->mutable_as_int()->set_width(width_);
  output->mutable_as_int()->set_is_signed(is_signed_);
  if (width_ == 8) {
    output->mutable_as_int()->set_is_declared_as_char(is_declared_as_char_);
  }
  return absl::OkStatus();
}

absl::Status CIntType::GetMetadataValue(Translator& translator,
                                        const ConstValue const_value,
                                        xlscc_metadata::Value* output) const {
  auto value = const_value.rvalue();
  XLS_CHECK(value.IsBits());
  if (is_signed()) {
    XLS_ASSIGN_OR_RETURN(int64_t signed_value, value.bits().ToInt64());
    output->mutable_as_int()->set_signed_value(signed_value);
  } else {
    XLS_ASSIGN_OR_RETURN(uint64_t unsigned_value, value.bits().ToUint64());
    output->mutable_as_int()->set_unsigned_value(unsigned_value);
  }
  return absl::OkStatus();
}

CBoolType::~CBoolType() {}

int CBoolType::GetBitWidth() const { return 1; }

CBoolType::operator std::string() const { return "bool"; }

absl::Status CBoolType::GetMetadata(Translator& translator,
                                    xlscc_metadata::Type* output) const {
  (void)output->mutable_as_bool();
  return absl::OkStatus();
}

absl::Status CBoolType::GetMetadataValue(Translator& translator,
                                         const ConstValue const_value,
                                         xlscc_metadata::Value* output) const {
  auto value = const_value.rvalue();
  XLS_CHECK(value.IsBits());
  XLS_ASSIGN_OR_RETURN(uint64_t bool_value, value.bits().ToUint64());
  output->set_as_bool(bool_value == 1);
  return absl::OkStatus();
}

bool CBoolType::operator==(const CType& o) const { return o.Is<CBoolType>(); }

bool CBoolType::StoredAsXLSBits() const { return true; }

CInstantiableTypeAlias::CInstantiableTypeAlias(const clang::NamedDecl* base)
    : base_(base) {}

const clang::NamedDecl* CInstantiableTypeAlias::base() const { return base_; }

CInstantiableTypeAlias::operator std::string() const {
  return absl::StrFormat("{%s}", base_->getNameAsString());
}

absl::Status CInstantiableTypeAlias::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output) const {
  output->mutable_as_inst()->mutable_name()->set_name(base_->getNameAsString());
  output->mutable_as_inst()->mutable_name()->set_fully_qualified_name(
      base_->getQualifiedNameAsString());
  output->mutable_as_inst()->mutable_name()->set_id(
      reinterpret_cast<uint64_t>(base_));

  if (base_->getKind() == clang::Decl::ClassTemplateSpecialization) {
    auto special =
        clang_down_cast<const clang::ClassTemplateSpecializationDecl*>(base_);
    const clang::TemplateArgumentList& arguments = special->getTemplateArgs();
    for (int argi = 0; argi < arguments.size(); ++argi) {
      const clang::TemplateArgument& arg = arguments.get(argi);
      xlscc_metadata::TemplateArgument* proto_arg =
          output->mutable_as_inst()->add_args();
      switch (arg.getKind()) {
        case clang::TemplateArgument::ArgKind::Integral:
          proto_arg->set_as_integral(arg.getAsIntegral().getExtValue());
          break;
        case clang::TemplateArgument::ArgKind::Type: {
          xls::SourceInfo loc = translator.GetLoc(*base_);
          XLS_ASSIGN_OR_RETURN(
              std::shared_ptr<CType> arg_ctype,
              translator.TranslateTypeFromClang(arg.getAsType(), loc));
          XLS_RETURN_IF_ERROR(
              arg_ctype->GetMetadata(translator, proto_arg->mutable_as_type()));
          break;
        }
        default:
          return absl::UnimplementedError(
              absl::StrFormat("Unimplemented template argument kind %i",
                              static_cast<int>(arg.getKind())));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status CInstantiableTypeAlias::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  std::shared_ptr<CInstantiableTypeAlias> inst(
      new CInstantiableTypeAlias(base_));
  auto found = translator.inst_types_.find(inst);
  XLS_CHECK(found != translator.inst_types_.end());
  auto struct_type =
      std::dynamic_pointer_cast<const CStructType>(found->second);

  // Handle __xls_bits
  if (struct_type == nullptr) {
    XLS_CHECK_EQ(base_->getNameAsString(), "__xls_bits");

    vector<uint8_t> bytes = const_value.rvalue().bits().ToBytes();
    output->set_as_bits(bytes.data(), bytes.size());

    return absl::OkStatus();
  }

  XLS_RETURN_IF_ERROR(
      struct_type->GetMetadataValue(translator, const_value, output));
  return absl::OkStatus();
}

int CInstantiableTypeAlias::GetBitWidth() const {
  XLS_LOG(FATAL) << "GetBitWidth() unsupported for CInstantiableTypeAlias";
  return 0;
}

bool CInstantiableTypeAlias::operator==(const CType& o) const {
  if (!o.Is<CInstantiableTypeAlias>()) return false;
  const auto* o_derived = o.As<CInstantiableTypeAlias>();
  return base_ == o_derived->base_;
}

CStructType::CStructType(std::vector<std::shared_ptr<CField>> fields,
                         bool no_tuple_flag)
    : no_tuple_flag_(no_tuple_flag), fields_(fields) {
  for (const std::shared_ptr<CField>& pf : fields) {
    XLS_CHECK(!fields_by_name_.contains(pf->name()));
    fields_by_name_[pf->name()] = pf;
  }
}

absl::Status CStructType::GetMetadata(Translator& translator,
                                      xlscc_metadata::Type* output) const {
  output->mutable_as_struct()->set_no_tuple(no_tuple_flag_);

  absl::flat_hash_map<
      int, std::pair<const clang::NamedDecl*, std::shared_ptr<CField>>>
      fields_by_index;

  auto size = fields_by_name_.size();
  for (std::pair<const clang::NamedDecl*, std::shared_ptr<CField>> field :
       fields_by_name_) {
    fields_by_index[size - 1 - field.second->index()] = field;
  }

  for (int i = 0; i < fields_by_name_.size(); ++i) {
    std::pair<const clang::NamedDecl*, std::shared_ptr<CField>> field =
        fields_by_index[i];
    XLS_RETURN_IF_ERROR(field.second->GetMetadata(
        translator, output->mutable_as_struct()->add_fields()));
  }

  return absl::OkStatus();
}

absl::Status CStructType::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  output->mutable_as_struct()->set_no_tuple(no_tuple_flag_);

  absl::flat_hash_map<
      int, std::pair<const clang::NamedDecl*, std::shared_ptr<CField>>>
      fields_by_index;

  for (auto field : fields_by_name_) {
    fields_by_index[field.second->index()] = field;
  }
  for (int i = 0; i < fields_by_name_.size(); ++i) {
    auto field = fields_by_index[i];
    auto struct_field_value = output->mutable_as_struct()->add_fields();
    auto name_out = struct_field_value->mutable_name();
    name_out->set_fully_qualified_name(field.first->getNameAsString());
    name_out->set_name(field.first->getNameAsString());
    name_out->set_id(reinterpret_cast<uint64_t>(field.first));
    XLS_ASSIGN_OR_RETURN(
        xls::Value elem_value,
        Translator::GetStructFieldXLS(const_value.rvalue(), i, *this));
    XLS_RETURN_IF_ERROR(field.second->type()->GetMetadataValue(
        translator, ConstValue(elem_value, field.second->type()),
        struct_field_value->mutable_value()));
  }
  return absl::OkStatus();
}

bool CStructType::no_tuple_flag() const { return no_tuple_flag_; }

int CStructType::GetBitWidth() const {
  int ret = 0;
  for (const std::shared_ptr<CField>& field : fields_) {
    ret += field->type()->GetBitWidth();
  }
  return ret;
}

CStructType::operator std::string() const {
  std::ostringstream ostr;
  ostr << "{";
  if (no_tuple_flag_) {
    ostr << " notuple ";
  }
  for (const std::shared_ptr<CField>& it : fields_) {
    ostr << "[" << it->index() << "] "
         << (it->name() ? it->name()->getNameAsString()
                        : std::string("nullptr"))
         << ": " << string(*it->type());
  }
  ostr << "}";
  return ostr.str();
}

bool CStructType::operator==(const CType& o) const {
  XLS_LOG(FATAL) << "operator== unsupported on structs";
  return false;
}

const std::vector<std::shared_ptr<CField>>& CStructType::fields() const {
  return fields_;
}

const absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>&
CStructType::fields_by_name() const {
  return fields_by_name_;
}

std::shared_ptr<CField> CStructType::get_field(
    const clang::NamedDecl* name) const {
  auto found = fields_by_name_.find(name);
  if (found == fields_by_name_.end()) {
    return std::shared_ptr<CField>();
  }
  return found->second;
}

CField::CField(const clang::NamedDecl* name, int index,
               std::shared_ptr<CType> type)
    : name_(name), index_(index), type_(type) {}

int CField::index() const { return index_; }

const clang::NamedDecl* CField::name() const { return name_; }

std::shared_ptr<CType> CField::type() const { return type_; }

CArrayType::CArrayType(std::shared_ptr<CType> element, int size)
    : element_(element), size_(size) {
  XLS_CHECK(size_ > 0);
}

bool CArrayType::operator==(const CType& o) const {
  if (!o.Is<CArrayType>()) return false;
  const auto* o_derived = o.As<CArrayType>();
  return *element_ == *o_derived->element_ && size_ == o_derived->size_;
}

int CArrayType::GetBitWidth() const { return size_ * element_->GetBitWidth(); }

absl::Status CField::GetMetadata(Translator& translator,
                                 xlscc_metadata::StructField* output) const {
  output->set_name(name_->getNameAsString());
  XLS_RETURN_IF_ERROR(type_->GetMetadata(translator, output->mutable_type()));
  return absl::OkStatus();
}

int CArrayType::GetSize() const { return size_; }

std::shared_ptr<CType> CArrayType::GetElementType() const { return element_; }

CArrayType::operator std::string() const {
  return absl::StrFormat("%s[%i]", string(*element_), size_);
}

absl::Status CArrayType::GetMetadata(Translator& translator,
                                     xlscc_metadata::Type* output) const {
  output->mutable_as_array()->set_size(size_);
  XLS_RETURN_IF_ERROR(element_->GetMetadata(
      translator, output->mutable_as_array()->mutable_element_type()));
  return absl::OkStatus();
}

absl::Status CArrayType::GetMetadataValue(Translator& translator,
                                          const ConstValue const_value,
                                          xlscc_metadata::Value* output) const {
  vector<xls::Value> values = const_value.rvalue().GetElements().value();
  for (auto& val : values) {
    XLS_RETURN_IF_ERROR(element_->GetMetadataValue(
        translator, ConstValue(val, element_),
        output->mutable_as_array()->add_element_values()));
  }
  return absl::OkStatus();
}

CPointerType::CPointerType(std::shared_ptr<CType> pointee_type)
    : pointee_type_(pointee_type) {}

bool CPointerType::operator==(const CType& o) const {
  if (!o.Is<CPointerType>()) return false;
  const auto* o_derived = o.As<CPointerType>();
  return *pointee_type_ == *o_derived->pointee_type_;
}

int CPointerType::GetBitWidth() const { return pointee_type_->GetBitWidth(); }

std::shared_ptr<CType> CPointerType::GetPointeeType() const {
  return pointee_type_;
}

CPointerType::operator std::string() const {
  return absl::StrFormat("%s*", string(*pointee_type_));
}

absl::Status CPointerType::GetMetadata(Translator& translator,
                                       xlscc_metadata::Type* output) const {
  XLS_CHECK(false) << "TODO: Metadata for pointers";
  return absl::OkStatus();
}

absl::Status CPointerType::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  XLS_CHECK(false) << "TODO: Metadata for pointers";
  return absl::OkStatus();
}

Translator::Translator(bool error_on_init_interval, int64_t max_unroll_iters,
                       int64_t warn_unroll_iters,
                       std::unique_ptr<CCParser> existing_parser)
    : max_unroll_iters_(max_unroll_iters),
      warn_unroll_iters_(warn_unroll_iters),
      error_on_init_interval_(error_on_init_interval) {
  context_stack_.push_front(TranslationContext());
  if (existing_parser != nullptr) {
    parser_ = std::move(existing_parser);
  } else {
    parser_ = std::make_unique<CCParser>();
  }
}

Translator::~Translator() {}

TranslationContext& Translator::PushContext() {
  auto ocond = context().full_condition;
  context_stack_.push_front(context());
  context().full_condition_on_enter_block = ocond;
  context().relative_condition = xls::BValue();
  context().propagate_up = true;
  context().propagate_break_up = true;
  context().propagate_continue_up = true;
  return context();
}

absl::Status Translator::PopContext(const xls::SourceInfo& loc) {
  const bool propagate_up = context().propagate_up;
  const bool propagate_break_up = context().propagate_break_up;
  const bool propagate_continue_up = context().propagate_continue_up;

  // Copy updated variables
  TranslationContext popped = context();
  context_stack_.pop_front();

  XLS_CHECK(!context_stack_.empty());
  if (!propagate_up) {
    return absl::OkStatus();
  }

  XLS_RETURN_IF_ERROR(
      PropagateVariables(/*from=*/popped, /*to=*/context(), loc));

  context().return_val = popped.return_val;
  context().last_return_condition = popped.last_return_condition;
  context().have_returned_condition = popped.have_returned_condition;

  context().any_side_effects_requested =
      context().any_side_effects_requested || popped.any_side_effects_requested;

  if (popped.have_returned_condition.valid()) {
    XLS_RETURN_IF_ERROR(and_condition(
        context().fb->Not(popped.have_returned_condition, loc), loc));
  }

  if (propagate_break_up && popped.relative_break_condition.valid()) {
    xls::BValue saved_popped_relative_break_condition =
        popped.relative_break_condition;

    if (popped.relative_break_condition.valid() &&
        context().relative_condition.valid()) {
      context().and_condition_util(context().relative_condition,
                                   saved_popped_relative_break_condition, loc);
    }

    XLS_RETURN_IF_ERROR(
        and_condition(context().fb->Not(popped.relative_break_condition), loc));

    context().or_condition_util(saved_popped_relative_break_condition,
                                context().relative_break_condition, loc);
  }

  if (propagate_continue_up && popped.relative_continue_condition.valid()) {
    xls::BValue saved_popped_relative_continue_condition =
        popped.relative_continue_condition;

    if (popped.relative_continue_condition.valid() &&
        context().relative_condition.valid()) {
      context().and_condition_util(context().relative_condition,
                                   saved_popped_relative_continue_condition,
                                   loc);
    }

    XLS_RETURN_IF_ERROR(and_condition(
        context().fb->Not(popped.relative_continue_condition), loc));

    context().or_condition_util(saved_popped_relative_continue_condition,
                                context().relative_continue_condition, loc);
  }
  return absl::OkStatus();
}

absl::Status Translator::PropagateVariables(const TranslationContext& from,
                                            TranslationContext& to,
                                            const xls::SourceInfo& loc) {
  XLS_CHECK_NE(from.sf, nullptr);
  for (const clang::NamedDecl* name :
       from.sf->DeterministicKeyNames(from.variables)) {
    if (to.variables.contains(name) &&
        (to.variables.at(name) != from.variables.at(name))) {
      XLS_ASSIGN_OR_RETURN(CValue prepared,
                           PrepareRValueWithSelect(
                               to.variables.at(name), from.variables.at(name),
                               from.relative_condition, loc));

      // Don't use Assign(), it uses context()
      to.variables.at(name) = prepared;
    } else if (to.sf->static_values.contains(name)) {
      to.variables[name] = from.variables.at(name);
    }
  }

  return absl::OkStatus();
}

std::function<std::optional<std::string>(xls::Fileno)>
Translator::LookUpInPackage() {
  return [=](xls::Fileno file_number) {
    AddSourceInfoToPackage(*package_);
    return package_->GetFilename(file_number);
  };
}

template <typename... Args>
std::string Translator::ErrorMessage(const xls::SourceInfo& loc,
                                     const absl::FormatSpec<Args...>& format,
                                     const Args&... args) {
  std::string result = absl::StrFormat(format, args...);
  for (const xls::SourceLocation& location : loc.locations) {
    absl::StrAppend(&result, "\n", PrintCaret(LookUpInPackage(), location));
  }
  return result;
}

void Translator::AddSourceInfoToPackage(xls::Package& package) {
  parser_->AddSourceInfoToPackage(package);
}

TranslationContext& Translator::context() { return context_stack_.front(); }

absl::Status Translator::and_condition(xls::BValue and_condition,
                                       const xls::SourceInfo& loc) {
  // Whenever the condition changes,
  // selects need to be generated in the enclosing context (if any)
  if (context().propagate_up && (context_stack_.size() > 2)) {
    auto iter = context_stack_.begin();
    TranslationContext& top = *iter;
    XLS_CHECK_EQ(&top, &context());

    ++iter;

    TranslationContext& second_to_top = *iter;
    XLS_CHECK_NE(&second_to_top, &context());

    XLS_CHECK_NE(second_to_top.sf, nullptr);
    XLS_CHECK_NE(top.sf, nullptr);
    XLS_CHECK_NE(&second_to_top, &top);
    XLS_CHECK_EQ(top.fb, second_to_top.fb);
    XLS_CHECK_EQ(top.sf, second_to_top.sf);
    XLS_RETURN_IF_ERROR(PropagateVariables(/*from=*/top,
                                           /*to=*/second_to_top, loc));
  }

  context().and_condition_util(and_condition, context().relative_condition,
                               loc);
  context().and_condition_util(and_condition, context().full_condition, loc);
  return absl::OkStatus();
}

absl::StatusOr<const clang::NamedDecl*> Translator::GetThisDecl(
    const xls::SourceInfo& loc, bool for_declaration) {
  XLS_CHECK_NE(context().sf->clang_decl, nullptr);
  if (!for_declaration &&
      !context().variables.contains(context().sf->clang_decl)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Tried to access 'this' in a context without any enclosing class "
        "(top level methods are not supported) at %s",
        LocString(loc)));
  }
  return context().sf->clang_decl;
}

absl::StatusOr<xls::BValue> Translator::StructUpdate(
    xls::BValue struct_before, CValue rvalue,
    const clang::NamedDecl* field_name, const CStructType& stype,
    const xls::SourceInfo& loc) {
  const absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>&
      fields_by_name = stype.fields_by_name();
  auto found_field = fields_by_name.find(field_name);
  if (found_field == fields_by_name.end()) {
    return absl::NotFoundError(
        ErrorMessage(loc, "Assignment to unknown field %s in type %s",
                     field_name->getNameAsString(), string(stype)));
  }
  const CField& cfield = *found_field->second;

  if (*cfield.type() != *rvalue.type()) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc,
                     "Cannot assign rvalue of type %s to struct "
                     "field of type %s",
                     string(*rvalue.type()), string(*cfield.type())));
  }

  // No tuple update, so we need to rebuild the tuple
  std::vector<xls::BValue> bvals;
  for (auto it = stype.fields().begin(); it != stype.fields().end(); it++) {
    std::shared_ptr<CField> fp = *it;
    xls::BValue bval;
    if (fp->index() != cfield.index()) {
      bval = GetStructFieldXLS(struct_before, fp->index(), stype, loc);
    } else {
      bval = rvalue.rvalue();
    }
    bvals.push_back(bval);
  }

  xls::BValue new_tuple = MakeStructXLS(bvals, stype, loc);

  return new_tuple;
}

xls::BValue Translator::MakeStructXLS(
    const std::vector<xls::BValue>& bvals_reverse, const CStructType& stype,
    const xls::SourceInfo& loc) {
  std::vector<xls::BValue> bvals = bvals_reverse;
  std::reverse(bvals.begin(), bvals.end());
  XLS_CHECK_EQ(bvals.size(), stype.fields().size());
  xls::BValue ret =
      stype.no_tuple_flag() ? bvals[0] : context().fb->Tuple(bvals, loc);
  return ret;
}

xls::Value Translator::MakeStructXLS(
    const std::vector<xls::Value>& vals_reverse, const CStructType& stype) {
  std::vector<xls::Value> vals = vals_reverse;
  std::reverse(vals.begin(), vals.end());
  XLS_CHECK_EQ(vals.size(), stype.fields().size());
  xls::Value ret = stype.no_tuple_flag() ? vals[0] : xls::Value::Tuple(vals);
  return ret;
}

xls::BValue Translator::GetStructFieldXLS(xls::BValue val, int index,
                                          const CStructType& type,
                                          const xls::SourceInfo& loc) {
  XLS_CHECK_LT(index, type.fields().size());
  return type.no_tuple_flag() ? val
                              : context().fb->TupleIndex(
                                    val, type.fields().size() - 1 - index, loc);
}

absl::StatusOr<xls::Value> Translator::GetStructFieldXLS(
    xls::Value val, int index, const CStructType& type) {
  XLS_CHECK_LT(index, type.fields().size());
  if (type.no_tuple_flag()) {
    return val;
  }
  XLS_ASSIGN_OR_RETURN(std::vector<xls::Value> values, val.GetElements());
  return values.at(type.fields().size() - 1 - index);
}

absl::StatusOr<xls::Type*> Translator::GetStructXLSType(
    const std::vector<xls::Type*>& members, const CStructType& type,
    const xls::SourceInfo& loc) {
  if (type.no_tuple_flag() && members.size() != 1) {
    return absl::FailedPreconditionError(
        ErrorMessage(loc,
                     "Pragma hls_no_tuple must be used on structs with only "
                     "1 field, but %s has %i\n",
                     string(type), members.size()));
  }
  return type.no_tuple_flag() ? members[0] : package_->GetTupleType(members);
}

xls::BValue Translator::MakeFlexTuple(const std::vector<xls::BValue>& bvals,
                                      const xls::SourceInfo& loc) {
  XLS_CHECK(!bvals.empty());
  return (bvals.size() == 1) ? bvals[0] : context().fb->Tuple(bvals, loc);
}

xls::BValue Translator::GetFlexTupleField(xls::BValue val, int index,
                                          int n_fields,
                                          const xls::SourceInfo& loc,
                                          std::string_view op_name) {
  XLS_CHECK_GT(n_fields, 0);
  return (n_fields == 1) ? val
                         : context().fb->TupleIndex(val, index, loc, op_name);
}

xls::Type* Translator::GetFlexTupleType(
    const std::vector<xls::Type*>& members) {
  XLS_CHECK(!members.empty());
  return (members.size() == 1) ? members[0] : package_->GetTupleType(members);
}

xls::BValue Translator::MakeFunctionReturn(
    const std::vector<xls::BValue>& bvals, const xls::SourceInfo& loc) {
  return MakeFlexTuple(bvals, loc);
}

xls::BValue Translator::UpdateFlexTupleField(xls::BValue tuple_val,
                                             xls::BValue new_val, int index,
                                             int n_fields,
                                             const xls::SourceInfo& loc) {
  if (n_fields == 1) {
    return new_val;
  }

  std::vector<xls::BValue> new_args;
  new_args.reserve(n_fields);
  for (int i = 0; i < n_fields; ++i) {
    new_args.push_back(
        (i == index) ? new_val : context().fb->TupleIndex(tuple_val, i, loc));
  }
  return context().fb->Tuple(new_args, loc);
}

xls::BValue Translator::GetFunctionReturn(xls::BValue val, int index,
                                          int expected_returns,
                                          const clang::FunctionDecl* /*func*/,
                                          const xls::SourceInfo& loc) {
  return GetFlexTupleField(val, index, expected_returns, loc);
}

std::string Translator::XLSNameMangle(clang::GlobalDecl decl) const {
  std::string res;
  llvm::raw_string_ostream os(res);
  if (!mangler_)
    mangler_.reset(decl.getDecl()->getASTContext().createMangleContext());
  mangler_->mangleCXXName(decl, os);
  return res;
}

absl::StatusOr<IOOp*> Translator::AddOpToChannel(IOOp& op, IOChannel* channel,
                                                 const xls::SourceInfo& loc,
                                                 bool mask) {
  context().any_side_effects_requested = true;

  if (context().mask_side_effects || mask) {
    IOOpReturn ret;
    ret.generate_expr = false;
    XLS_ASSIGN_OR_RETURN(xls::BValue default_bval,
                         CreateDefaultValue(channel->item_type, loc));
    op.input_value = CValue(default_bval, channel->item_type);
    return &op;
  }
  XLS_CHECK_NE(channel, nullptr);
  XLS_CHECK_EQ(op.channel, nullptr);
  op.channel_op_index = channel->total_ops++;
  op.channel = channel;
  op.op_location = loc;

  // Operation type is added late, as it's only known from the read()/write()
  // call(s)
  if (channel->channel_op_type == OpType::kNull) {
    channel->channel_op_type = op.op;
  } else {
    if (channel->channel_op_type != op.op) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Channels should be either input or output"));
    }
  }

  // Channel must be inserted first by AddOpToChannel
  if (op.op == OpType::kRecv) {
    XLS_ASSIGN_OR_RETURN(xls::Type * xls_item_type,
                         TranslateTypeToXLS(channel->item_type, loc));

    const int64_t channel_op_index = op.channel_op_index;

    std::string safe_param_name =
        absl::StrFormat("%s_op%i", op.channel->unique_name, channel_op_index);

    xls::BValue pbval =
        context().fb->Param(safe_param_name, xls_item_type, loc);

    // Check for duplicate params
    XLS_CHECK(pbval.valid());

    op.input_value = CValue(pbval, channel->item_type);
  }

  context().sf->io_ops.push_back(op);

  if (op.op == OpType::kRecv) {
    SideEffectingParameter side_effecting_param;
    side_effecting_param.type = SideEffectingParameterType::kIOOp;
    side_effecting_param.param_name =
        op.input_value.rvalue().node()->As<xls::Param>()->GetName();
    side_effecting_param.io_op = &context().sf->io_ops.back();
    context().sf->side_effecting_parameters.push_back(side_effecting_param);
  }

  return &context().sf->io_ops.back();
}

absl::StatusOr<std::shared_ptr<CType>> Translator::GetChannelType(
    const clang::ParmVarDecl* channel_param, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                       StripTypeQualifiers(channel_param->getType()));

  if (stripped.base->getTypeClass() !=
      clang::Type::TypeClass::TemplateSpecialization) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel type should be a template specialization"));
  }

  auto template_spec =
      clang_down_cast<const clang::TemplateSpecializationType*>(
          stripped.base.getTypePtr());

  if (template_spec->getNumArgs() != 1) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel should have 1 template args"));
  }

  const clang::TemplateArgument& arg = template_spec->getArg(0);

  return TranslateTypeFromClang(arg.getAsType(), loc);
}

absl::Status Translator::CreateChannelParam(
    const clang::ParmVarDecl* channel_param, const xls::SourceInfo& loc) {
  XLS_CHECK(!context().sf->io_channels_by_param.contains(channel_param));

  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                       GetChannelType(channel_param, loc));

  IOChannel new_channel;

  new_channel.item_type = ctype;
  new_channel.unique_name = channel_param->getNameAsString();

  context().sf->io_channels.push_back(new_channel);
  context().sf->io_channels_by_param[channel_param] =
      &context().sf->io_channels.back();
  context().sf->params_by_io_channel[&context().sf->io_channels.back()] =
      channel_param;

  context().channel_params.insert(channel_param);

  return absl::OkStatus();
}

absl::StatusOr<Translator::IOOpReturn> Translator::InterceptIOOp(
    const clang::Expr* expr, const xls::SourceInfo& loc) {
  if (expr->getStmtClass() == clang::Stmt::CXXMemberCallExprClass) {
    auto member_call = clang_down_cast<const clang::CXXMemberCallExpr*>(expr);
    const clang::Expr* object = member_call->getImplicitObjectArgument();

    XLS_ASSIGN_OR_RETURN(bool is_channel, ExprIsChannel(object, loc));
    if (is_channel) {
      // Duplicated code in GenerateIR_Call()?
      if (object->getStmtClass() != clang::Stmt::DeclRefExprClass) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "IO ops should be on direct DeclRefs"));
      }
      auto object_ref = clang_down_cast<const clang::DeclRefExpr*>(object);
      if (object_ref->getDecl()->getKind() != clang::Decl::ParmVar) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "IO ops should be on channel parameters"));
      }

      const clang::FunctionDecl* funcdecl = member_call->getDirectCallee();
      const std::string op_name = funcdecl->getNameAsString();
      auto channel_param =
          clang_down_cast<const clang::ParmVarDecl*>(object_ref->getDecl());

      IOChannel* channel = context().sf->io_channels_by_param.at(channel_param);
      xls::BValue op_condition = context().full_condition_bval(loc);
      XLS_CHECK(op_condition.valid());

      // Short circuit the op condition if possible
      XLS_RETURN_IF_ERROR(ShortCircuitBVal(op_condition, loc));

      // Ignore IO ops that are definitely condition = 0
      // XLS opt also does this down-stream, but we try to do it here
      // for cases like "if(constexpr) {ch.read();} else {ch.write();}
      // which otherwise confuse XLS[cc] itself.
      bool do_default = false;

      absl::StatusOr<xls::Value> eval_result = EvaluateBVal(op_condition, loc);
      if (eval_result.ok()) {
        if (eval_result.value().IsAllZeros()) {
          do_default = true;
        }
      }

      auto call = clang_down_cast<const clang::CallExpr*>(expr);

      IOOpReturn ret;
      ret.generate_expr = false;

      IOOp op;
      const clang::Expr* assign_ret_value_to = nullptr;

      if (op_name == "read") {
        if (call->getNumArgs() == 1) {
          assign_ret_value_to = call->getArg(0);
        } else if (call->getNumArgs() != 0) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "IO read() should have one or zero argument(s)"));
        }
        op.op = OpType::kRecv;
        op.ret_value = op_condition;
      } else if (op_name == "write") {
        if (call->getNumArgs() != 1) {
          return absl::UnimplementedError(
              ErrorMessage(loc, "IO write() should have one argument"));
        }

        XLS_ASSIGN_OR_RETURN(CValue out_val,
                             GenerateIR_Expr(call->getArg(0), loc));
        std::vector<xls::BValue> sp = {out_val.rvalue(), op_condition};
        op.ret_value = context().fb->Tuple(sp, loc);
        op.op = OpType::kSend;

      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unsupported IO op: %s", op_name));
      }

      XLS_ASSIGN_OR_RETURN(
          IOOp * op_ptr, AddOpToChannel(op, channel, loc, /*mask=*/do_default));
      (void)op_ptr;

      ret.value = op.input_value;
      if (assign_ret_value_to != nullptr) {
        XLS_RETURN_IF_ERROR(Assign(assign_ret_value_to, ret.value, loc));
      }

      return ret;
    }
  }

  IOOpReturn ret;
  ret.generate_expr = true;
  return ret;
}

absl::StatusOr<GeneratedFunction*> Translator::GenerateIR_Function(
    const clang::FunctionDecl* funcdecl, absl::string_view name_override,
    bool force_static) {
  XLS_ASSIGN_OR_RETURN(const clang::Stmt* body, GetFunctionBody(funcdecl));

  std::string xls_name;

  if (!name_override.empty()) {
    xls_name = name_override;
  } else {
    clang::GlobalDecl global_decl;
    if (funcdecl->getKind() == clang::Decl::CXXConstructor) {
      global_decl = clang::GlobalDecl(
          clang_down_cast<const clang::CXXConstructorDecl*>(funcdecl),
          clang::Ctor_Complete);
    } else {
      global_decl = clang::GlobalDecl(funcdecl);
    }
    xls_name = XLSNameMangle(global_decl);
  }

  XLS_CHECK(!xls_names_for_functions_generated_.contains(funcdecl));

  xls_names_for_functions_generated_[funcdecl] = xls_name;

  xls::FunctionBuilder builder(xls_name, package_);

  PushContextGuard context_guard(*this, GetLoc(*funcdecl));

  auto signature = absl::implicit_cast<const clang::NamedDecl*>(funcdecl);

  inst_functions_[signature] = std::make_unique<GeneratedFunction>();
  GeneratedFunction& sf = *inst_functions_[signature];

  sf.clang_decl = funcdecl;

  // Functions need a clean context
  context() = TranslationContext();
  context().propagate_up = false;

  context().fb = absl::implicit_cast<xls::BuilderBase*>(&builder);
  context().sf = &sf;

  // Unroll for loops in default function bodies without pragma
  context().for_loops_default_unroll = funcdecl->isDefaulted();
  context().outer_pipelined_loop_init_interval = default_init_interval_;
  XLS_ASSIGN_OR_RETURN(
      context().return_type,
      TranslateTypeFromClang(funcdecl->getReturnType(), GetLoc(*funcdecl),
                             /*allow_references=*/funcdecl->isDefaulted()));

  // If add_this_return is true, then a return value is added for the
  //  "this" object, pointed to be the "this" pointer in methods
  bool add_this_return = false;
  vector<const clang::NamedDecl*> ref_returns;

  xls::SourceInfo body_loc = GetLoc(*funcdecl);

  // "this" input for methods
  if ((funcdecl->getKind() == clang::FunctionDecl::Kind::CXXMethod) ||
      (funcdecl->getKind() == clang::FunctionDecl::Kind::CXXConversion) ||
      (funcdecl->getKind() == clang::FunctionDecl::Kind::CXXDestructor) ||
      (funcdecl->getKind() == clang::FunctionDecl::Kind::CXXConstructor)) {
    auto method = clang_down_cast<const clang::CXXMethodDecl*>(funcdecl);
    if (!method->isStatic() && !force_static) {
      // "This" is a PointerType, ignore and treat as reference
      const clang::QualType& thisQual = method->getThisType();
      XLS_CHECK(thisQual->isPointerType());

      add_this_return = !thisQual->getPointeeType().isConstQualified();

      const clang::QualType& q = thisQual->getPointeeOrArrayElementType()
                                     ->getCanonicalTypeUnqualified();

      XLS_ASSIGN_OR_RETURN(
          auto thisctype,
          TranslateTypeFromClang(q, body_loc,
                                 /*allow_references=*/funcdecl->isDefaulted()));
      XLS_ASSIGN_OR_RETURN(xls::Type * xls_type,
                           TranslateTypeToXLS(thisctype, body_loc));

      CValue this_val =
          CValue(context().fb->Param("this", xls_type, body_loc), thisctype);
      XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                           GetThisDecl(body_loc, /*for_declaration=*/true));
      XLS_RETURN_IF_ERROR(DeclareVariable(this_decl, this_val, body_loc));
    }
  }

  absl::flat_hash_set<std::string> used_parameter_names;

  for (const clang::ParmVarDecl* p : funcdecl->parameters()) {
    auto namedecl = absl::implicit_cast<const clang::NamedDecl*>(p);

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_ASSIGN_OR_RETURN(
        std::shared_ptr<CType> obj_type,
        TranslateTypeFromClang(stripped.base, GetLoc(*p),
                               /*allow_references=*/funcdecl->isDefaulted()));

    xls::Type* xls_type = nullptr;

    XLS_ASSIGN_OR_RETURN(bool is_channel,
                         TypeIsChannel(p->getType(), GetLoc(*p)));
    if (is_channel) {
      XLS_RETURN_IF_ERROR(CreateChannelParam(p, GetLoc(*p)));
      continue;
    }

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      ref_returns.push_back(namedecl);
    }

    if (xls_type == nullptr) {
      XLS_ASSIGN_OR_RETURN(xls_type, TranslateTypeToXLS(obj_type, GetLoc(*p)));
    }

    std::string safe_param_name = namedecl->getNameAsString();
    if (safe_param_name.empty()) safe_param_name = "implicit";

    for (int iter = 0; used_parameter_names.contains(safe_param_name); ++iter) {
      safe_param_name += absl::StrFormat("%i", used_parameter_names.size());

      if (iter > 10) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Couldn't find a safe parameter name at %s",
                            LocString(GetLoc(*p))));
      }
    }
    XLS_CHECK(!used_parameter_names.contains(safe_param_name));
    used_parameter_names.insert(safe_param_name);

    xls::BValue pbval =
        context().fb->Param(safe_param_name, xls_type, body_loc);

    // Create CValue without type check
    XLS_RETURN_IF_ERROR(
        DeclareVariable(namedecl, CValue(pbval, obj_type, true), body_loc));
  }

  // Generate constructor initializers
  if (funcdecl->getKind() == clang::FunctionDecl::Kind::CXXConstructor) {
    auto constructor =
        clang_down_cast<const clang::CXXConstructorDecl*>(funcdecl);
    XLS_CHECK(add_this_return);
    XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                         GetThisDecl(GetLoc(*constructor)));
    XLS_ASSIGN_OR_RETURN(CValue this_val,
                         GetIdentifier(this_decl, GetLoc(*constructor)));
    XLS_ASSIGN_OR_RETURN(auto resolved_type,
                         ResolveTypeInstance(this_val.type()));
    auto struct_type = std::dynamic_pointer_cast<CStructType>(resolved_type);
    XLS_CHECK(struct_type);

    const auto& fields_by_name = struct_type->fields_by_name();
    absl::flat_hash_map<int, xls::BValue> indices_to_update;

    for (const clang::CXXCtorInitializer* init : constructor->inits()) {
      XLS_ASSIGN_OR_RETURN(
          CValue rvalue,
          GenerateIR_Expr(init->getInit(), GetLoc(*constructor)));

      // Base class constructors don't have member names
      const clang::NamedDecl* member_name = nullptr;
      if (init->getMember()) {
        member_name =
            absl::implicit_cast<const clang::NamedDecl*>(init->getMember());
      } else {
        member_name = absl::implicit_cast<const clang::NamedDecl*>(
            init->getInit()->getType()->getAsRecordDecl());
        XLS_CHECK(member_name);
      }

      auto found = fields_by_name.find(member_name);
      XLS_CHECK(found != fields_by_name.end());
      XLS_CHECK(found->second->name() == member_name);
      XLS_CHECK(indices_to_update.find(found->second->index()) ==
                indices_to_update.end());
      XLS_CHECK(*found->second->type() == *rvalue.type());
      indices_to_update[found->second->index()] = rvalue.rvalue();
    }

    std::vector<xls::BValue> bvals;
    for (auto it = struct_type->fields().begin();
         it != struct_type->fields().end(); it++) {
      std::shared_ptr<CField> field = *it;
      auto found = indices_to_update.find(field->index());
      xls::BValue bval;
      if (found != indices_to_update.end()) {
        bval = found->second;
      } else {
        bval = GetStructFieldXLS(this_val.rvalue(), field->index(),
                                 *struct_type, GetLoc(*constructor));
      }
      bvals.push_back(bval);
    }

    CValue new_this_val =
        CValue(MakeStructXLS(bvals, *struct_type, GetLoc(*constructor)),
               this_val.type());
    XLS_RETURN_IF_ERROR(Assign(this_decl, new_this_val, body_loc));
  }

  // Extra context layer to generate selects
  {
    PushContextGuard top_select_guard(*this, GetLoc(*funcdecl));
    context().propagate_up = true;

    if (body != nullptr) {
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, funcdecl->getASTContext()));
    }
  }

  vector<xls::BValue> return_bvals;

  // First static returns
  for (const clang::NamedDecl* decl :
       context().sf->GetDeterministicallyOrderedStaticValues()) {
    XLS_ASSIGN_OR_RETURN(CValue value, GetIdentifier(decl, body_loc));
    return_bvals.push_back(value.rvalue());
  }

  // Then this return
  if (add_this_return) {
    XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                         GetThisDecl(body_loc));
    XLS_ASSIGN_OR_RETURN(CValue this_val, GetIdentifier(this_decl, body_loc));
    return_bvals.push_back(this_val.rvalue());
  }

  // Then explicit return
  if (!funcdecl->getReturnType()->isVoidType()) {
    return_bvals.emplace_back(context().return_val);
  }

  // Then reference parameter returns
  for (const clang::NamedDecl* ret_ident : ref_returns) {
    XLS_ASSIGN_OR_RETURN(CValue found, GetIdentifier(ret_ident, body_loc));
    return_bvals.emplace_back(found.rvalue());
  }

  // IO returns
  for (const IOOp& op : sf.io_ops) {
    XLS_CHECK(op.ret_value.valid());
    return_bvals.push_back(op.ret_value);
  }

  sf.return_value_count = return_bvals.size();

  if (return_bvals.empty()) {
    return &sf;
  }

  if (return_bvals.size() == 1) {
    // XLS functions return the last value added to the FunctionBuilder
    // So this makes sure the correct value is last.
    context().return_val = return_bvals[0];
  } else {
    context().return_val = MakeFunctionReturn(return_bvals, body_loc);
  }

  if (!sf.io_ops.empty() && funcdecl->isOverloadedOperator()) {
    return absl::UnimplementedError(
        ErrorMessage(body_loc, "IO ops in operator calls are not supported"));
  }

  XLS_ASSIGN_OR_RETURN(sf.xls_func,
                       builder.BuildWithReturnValue(context().return_val));

  return &sf;
}

absl::StatusOr<std::shared_ptr<CType>> Translator::InterceptBuiltInStruct(
    const clang::RecordDecl* sd) {
  // "__xls_bits" is a special built-in type: CBitsType
  // The number of bits, or width, is specified as a single integer-type
  //  template parameter.
  if (sd->getNameAsString() == "__xls_bits") {
    // __xls_bits must always have a template parameter
    if (sd->getDeclKind() != clang::Decl::Kind::ClassTemplateSpecialization) {
      return absl::UnimplementedError(absl::StrFormat(
          "__xls_bits should be used in template specialization %s",
          LocString(GetLoc(*sd))));
    }
    auto temp_spec =
        clang_down_cast<const clang::ClassTemplateSpecializationDecl*>(sd);
    const clang::TemplateArgumentList& temp_args = temp_spec->getTemplateArgs();
    if ((temp_args.size() != 1) ||
        (temp_args.get(0).getKind() !=
         clang::TemplateArgument::ArgKind::Integral)) {
      return absl::UnimplementedError(absl::StrFormat(
          "__xls_bits should have on integral template argument (width) %s",
          LocString(GetLoc(*sd))));
    }

    llvm::APSInt width_aps = temp_args.get(0).getAsIntegral();
    return std::shared_ptr<CType>(new CBitsType(width_aps.getExtValue()));
  }
  return nullptr;
}

absl::Status Translator::ScanStruct(const clang::RecordDecl* sd) {
  std::shared_ptr<CInstantiableTypeAlias> signature(new CInstantiableTypeAlias(
      absl::implicit_cast<const clang::NamedDecl*>(sd)));

  std::shared_ptr<CType> new_type;

  // Check for built-in XLS[cc] types
  XLS_ASSIGN_OR_RETURN(new_type, InterceptBuiltInStruct(sd));

  // If no built-in type was found, interpret as a normal C++ struct
  if (new_type == nullptr) {
    std::vector<std::shared_ptr<CField>> fields;

    const clang::CXXRecordDecl* cxx_record = nullptr;

    // Clang may express a concrete class/struct type in two different ways:
    // A CXXRecord if it's not templatized, or a ClassTemplateSpecialization
    //  if it is.
    if (absl::implicit_cast<const clang::Decl*>(sd)->getKind() ==
        clang::Decl::Kind::CXXRecord) {
      cxx_record = clang_down_cast<const clang::CXXRecordDecl*>(sd);
    } else if (absl::implicit_cast<const clang::Decl*>(sd)->getKind() ==
               clang::Decl::Kind::ClassTemplateSpecialization) {
      auto specialization =
          clang_down_cast<const clang::ClassTemplateSpecializationDecl*>(sd);

      cxx_record = clang_down_cast<const clang::CXXRecordDecl*>(
          specialization->getDefinition());
    }

    if (cxx_record == nullptr) {
      return absl::UnavailableError(ErrorMessage(
          GetLoc(*sd),
          "Definition for CXXRecord '%s' isn't available from Clang. A "
          "possible work-around is to declare an instance of this class.",
          signature->base()->getNameAsString()));
    }

    // Interpret forward declarations as empty structs
    if (cxx_record->hasDefinition()) {
      for (auto base : cxx_record->bases()) {
        const clang::RecordDecl* base_struct =
            base.getType()->getAsRecordDecl();
        XLS_ASSIGN_OR_RETURN(
            auto field_type,
            TranslateTypeFromClang(
                base_struct->getTypeForDecl()->getCanonicalTypeInternal(),
                GetLoc(*base_struct)));

        fields.push_back(std::shared_ptr<CField>(new CField(
            absl::implicit_cast<const clang::NamedDecl*>(base_struct),
            fields.size(), field_type)));
      }

      for (const clang::FieldDecl* it : sd->fields()) {
        XLS_ASSIGN_OR_RETURN(
            std::shared_ptr<CType> field_type,
            TranslateTypeFromClang(it->getType(),
                                   GetLoc(*it->getCanonicalDecl())));

        // Up cast FieldDecl to NamedDecl because NamedDecl pointers are used to
        //  track identifiers by XLS[cc], no matter the type of object being
        //  identified
        fields.push_back(std::shared_ptr<CField>(
            new CField(absl::implicit_cast<const clang::NamedDecl*>(it),
                       fields.size(), field_type)));
      }
    } else {
      XLS_LOG(WARNING) << ErrorMessage(
          GetLoc(*cxx_record),
          "Warning: interpreting definition-less struct '%s' as empty",
          signature->base()->getNameAsString());
    }

    XLS_ASSIGN_OR_RETURN(Pragma pragma, FindPragmaForLoc(GetPresumedLoc(*sd)));
    new_type.reset(new CStructType(fields, pragma.type() == Pragma_NoTuples));
  }

  inst_types_[signature] = new_type;

  return absl::OkStatus();
}

absl::StatusOr<xls::Op> Translator::XLSOpcodeFromClang(
    clang::BinaryOperatorKind clang_op, const CType& left_type,
    const CType& result_type, const xls::SourceInfo& loc) {
  if (clang_op == clang::BinaryOperatorKind::BO_Comma) {
    return xls::Op::kIdentity;
  }
  if (result_type.Is<CIntType>()) {
    auto result_int_type = result_type.As<CIntType>();
    switch (clang_op) {
      case clang::BinaryOperatorKind::BO_Assign:
        return xls::Op::kIdentity;
      case clang::BinaryOperatorKind::BO_Add:
      case clang::BinaryOperatorKind::BO_AddAssign:
        return xls::Op::kAdd;
      case clang::BinaryOperatorKind::BO_Sub:
      case clang::BinaryOperatorKind::BO_SubAssign:
        return xls::Op::kSub;
      case clang::BinaryOperatorKind::BO_Mul:
      case clang::BinaryOperatorKind::BO_MulAssign:
        return result_int_type->is_signed() ? xls::Op::kSMul : xls::Op::kUMul;
      case clang::BinaryOperatorKind::BO_Div:
      case clang::BinaryOperatorKind::BO_DivAssign:
        return result_int_type->is_signed() ? xls::Op::kSDiv : xls::Op::kUDiv;
      case clang::BinaryOperatorKind::BO_Rem:
      case clang::BinaryOperatorKind::BO_RemAssign:
        return result_int_type->is_signed() ? xls::Op::kSMod : xls::Op::kUMod;
      case clang::BinaryOperatorKind::BO_Shl:
      case clang::BinaryOperatorKind::BO_ShlAssign:
        return xls::Op::kShll;
      case clang::BinaryOperatorKind::BO_Shr:
      case clang::BinaryOperatorKind::BO_ShrAssign:
        return result_int_type->is_signed() ? xls::Op::kShra : xls::Op::kShrl;
      case clang::BinaryOperatorKind::BO_And:
      case clang::BinaryOperatorKind::BO_AndAssign:
        return xls::Op::kAnd;
      case clang::BinaryOperatorKind::BO_Or:
      case clang::BinaryOperatorKind::BO_OrAssign:
        return xls::Op::kOr;
      case clang::BinaryOperatorKind::BO_Xor:
      case clang::BinaryOperatorKind::BO_XorAssign:
        return xls::Op::kXor;
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented binary operator %i for result %s",
                         clang_op, std::string(result_type)));
    }
  }
  if (result_type.Is<CBoolType>()) {
    if (left_type.Is<CIntType>()) {
      auto input_int_type = left_type.As<CIntType>();
      switch (clang_op) {
        case clang::BinaryOperatorKind::BO_GT:
          return input_int_type->is_signed() ? xls::Op::kSGt : xls::Op::kUGt;
        case clang::BinaryOperatorKind::BO_LT:
          return input_int_type->is_signed() ? xls::Op::kSLt : xls::Op::kULt;
        case clang::BinaryOperatorKind::BO_GE:
          return input_int_type->is_signed() ? xls::Op::kSGe : xls::Op::kUGe;
        case clang::BinaryOperatorKind::BO_LE:
          return input_int_type->is_signed() ? xls::Op::kSLe : xls::Op::kULe;
        default:
          break;
      }
    }
    switch (clang_op) {
      case clang::BinaryOperatorKind::BO_Assign:
        return xls::Op::kIdentity;
      case clang::BinaryOperatorKind::BO_EQ:
        return xls::Op::kEq;
      case clang::BinaryOperatorKind::BO_NE:
        return xls::Op::kNe;
      // Clang generates an ImplicitCast to bool for the parameters to
      //  logical expressions (eg && ||), so logical ops (eg & |) are sufficient
      case clang::BinaryOperatorKind::BO_LAnd:
      case clang::BinaryOperatorKind::BO_AndAssign:
        return xls::Op::kAnd;
      case clang::BinaryOperatorKind::BO_LOr:
      case clang::BinaryOperatorKind::BO_OrAssign:
        return xls::Op::kOr;
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented binary operator %i for result %s with input %s at "
            "%s",
            clang_op, std::string(result_type), std::string(left_type),
            LocString(loc)));
    }
  }

  if (result_type.Is<CPointerType>()) {
    if (clang_op == clang::BinaryOperatorKind::BO_Assign) {
      return xls::Op::kIdentity;
    }
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented binary operator %i for pointer", clang_op));
  }

  return absl::UnimplementedError(
      ErrorMessage(loc, "Binary operators unimplemented for type %s",
                   std::string(result_type)));
}

absl::StatusOr<CValue> Translator::TranslateVarDecl(
    const clang::VarDecl* decl, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                       TranslateTypeFromClang(decl->getType(), loc));

  const clang::Expr* initializer = decl->getAnyInitializer();
  std::shared_ptr<LValue> lvalue = nullptr;
  xls::BValue init_val;
  if (initializer) {
    LValueModeGuard lvalue_mode(*this);
    XLS_ASSIGN_OR_RETURN(CValue cv, GenerateIR_Expr(initializer, loc));
    XLS_ASSIGN_OR_RETURN(init_val, GenTypeConvert(cv, ctype, loc));
    lvalue = cv.lvalue();
    if (ctype->Is<CPointerType>() && !lvalue) {
      return absl::UnimplementedError(
          ErrorMessage(loc,
                       "Initializer for pointer has no lvalue (unsupported "
                       "construct such as ternary?)"));
    }
  } else {
    XLS_ASSIGN_OR_RETURN(init_val, CreateDefaultValue(ctype, loc));
  }
  return CValue(init_val, ctype, /*disable_type_check=*/false, lvalue);
}

absl::StatusOr<CValue> Translator::TranslateEnumConstantDecl(
    const clang::EnumConstantDecl* decl, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                       TranslateTypeFromClang(decl->getType(), loc));
  auto val = xls::Value(
      xls::UBits(decl->getInitVal().getExtValue(), ctype->GetBitWidth()));
  xls::BValue init_val = context().fb->Literal(val, loc);
  return CValue(init_val, ctype, /*disable_type_check=*/false);
}

absl::StatusOr<CValue> Translator::GetOnReset(const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(const clang::VarDecl* on_reset_var_decl,
                       parser_->GetXlsccOnReset());
  auto on_reset_decl = static_cast<const clang::NamedDecl*>(on_reset_var_decl);

  if (!context().variables.contains(on_reset_decl)) {
    XLS_CHECK(!context().sf->static_values.contains(on_reset_decl));
    ConstValue init_val(xls::Value(xls::UBits(1, 1)),
                        std::make_shared<CBoolType>());
    XLS_RETURN_IF_ERROR(DeclareStatic(on_reset_decl, init_val, loc));
  }

  return context().variables.at(on_reset_decl);
}

absl::StatusOr<bool> Translator::DeclIsOnReset(const clang::NamedDecl* decl) {
  XLS_ASSIGN_OR_RETURN(const clang::VarDecl* on_reset_var_decl,
                       parser_->GetXlsccOnReset());
  return decl == static_cast<const clang::NamedDecl*>(on_reset_var_decl);
}

absl::StatusOr<CValue> Translator::GetIdentifier(const clang::NamedDecl* decl,
                                                 const xls::SourceInfo& loc) {
  if (context().channel_params.contains(decl)) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel parameter reference unsupported %s",
                     decl->getNameAsString()));
  }

  XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(decl));
  if (is_on_reset) {
    return GetOnReset(loc);
  }

  // CValue on file in the context?
  auto found = context().variables.find(decl);
  if (found != context().variables.end()) {
    return found->second;
  }

  // Is it an enum?
  auto enum_decl = dynamic_cast<const clang::EnumConstantDecl*>(decl);
  // Is this static/global?
  auto var_decl = dynamic_cast<const clang::VarDecl*>(decl);

  if (var_decl != nullptr && var_decl->isStaticDataMember() &&
      (!var_decl->getType().isConstQualified())) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Mutable static data members not implemented %s",
                     decl->getNameAsString()));
  }

  XLS_CHECK(var_decl == nullptr || !var_decl->isStaticLocal() ||
            var_decl->getType().isConstQualified());

  XLS_CHECK(!(enum_decl && var_decl));

  if (var_decl == nullptr && enum_decl == nullptr) {
    return absl::NotFoundError(
        ErrorMessage(loc, "Undeclared identifier %s", decl->getNameAsString()));
  }

  const clang::NamedDecl* name =
      (var_decl != nullptr)
          ? absl::implicit_cast<const clang::NamedDecl*>(var_decl)
          : absl::implicit_cast<const clang::NamedDecl*>(enum_decl);

  // Don't re-build the global value for each reference
  // They need to be built once for each Function[Builder]
  auto found_global = context().sf->global_values.find(name);
  if (found_global != context().sf->global_values.end()) {
    return found_global->second;
  }

  const xls::SourceInfo global_loc = GetLoc(*name);

  CValue value;

  if (enum_decl != nullptr) {
    const llvm::APSInt& aps = enum_decl->getInitVal();

    std::shared_ptr<CType> type(std::make_shared<CIntType>(32, false));
    xls::BValue bval =
        context().fb->Literal(xls::UBits(aps.getExtValue(), 32), global_loc);

    value = CValue(bval, type);
  } else {
    XLS_CHECK(var_decl->hasGlobalStorage());

    XLS_CHECK(context().fb);

    if (var_decl->getInit() != nullptr) {
      XLS_ASSIGN_OR_RETURN(value,
                           GenerateIR_Expr(var_decl->getInit(), global_loc));
      if (var_decl->isStaticLocal() || var_decl->isStaticDataMember()) {
        // Statics must have constant initialization
        if (!EvaluateBVal(value.rvalue(), global_loc).ok()) {
          return absl::InvalidArgumentError(
              ErrorMessage(loc, "Statics must have constant initializers"));
        }
      }
    } else {
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> type,
          TranslateTypeFromClang(var_decl->getType(), global_loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue bval,
                           CreateDefaultValue(type, global_loc));
      value = CValue(bval, type);
    }
  }
  context().sf->global_values[name] = value;
  return value;
}

absl::StatusOr<CValue> Translator::PrepareRValueWithSelect(
    const CValue& lvalue, const CValue& rvalue,
    const xls::BValue& relative_condition, const xls::SourceInfo& loc) {
  CValue rvalue_to_use = rvalue;
  // Avoid generating unnecessary selects
  absl::StatusOr<xls::Value> const_var_cond = xls::Value(xls::UBits(1, 1));
  if (relative_condition.valid()) {
    const_var_cond = EvaluateBVal(relative_condition, loc);
  }
  if (const_var_cond.ok() && const_var_cond.value().IsAllOnes()) {
    return rvalue_to_use;
  }

  // Typical rvalue case
  if (!lvalue.type()->Is<CPointerType>()) {
    XLS_CHECK(rvalue.rvalue().valid());
    XLS_CHECK(lvalue.rvalue().valid());
    XLS_CHECK_EQ(rvalue.rvalue().GetType()->kind(),
                 lvalue.rvalue().GetType()->kind());
    XLS_CHECK_EQ(rvalue_to_use.lvalue(), nullptr);
    auto cond_sel = context().fb->Select(relative_condition, rvalue.rvalue(),
                                         lvalue.rvalue(), loc);
    rvalue_to_use = CValue(cond_sel, rvalue_to_use.type());
  } else {
    // LValue (pointer) case
    XLS_CHECK_NE(rvalue_to_use.lvalue(), nullptr);
    XLS_CHECK_NE(lvalue.lvalue(), nullptr);

    auto select_lvalue = std::make_shared<LValue>(
        relative_condition, rvalue_to_use.lvalue(), lvalue.lvalue());

    rvalue_to_use = CValue(xls::BValue(), rvalue_to_use.type(),
                           /*disable_type_check=*/false, select_lvalue);
  }
  return rvalue_to_use;
}

absl::Status Translator::Assign(const clang::NamedDecl* lvalue,
                                const CValue& rvalue,
                                const xls::SourceInfo& loc) {
  context().any_side_effects_requested = true;
  if (context().mask_side_effects || context().mask_assignments) {
    return absl::OkStatus();
  }

  // Don't allow assignment to globals. This doesn't work because
  //  each function has a different FunctionBuilder.
  if (auto var_decl = dynamic_cast<const clang::VarDecl*>(lvalue);
      var_decl != nullptr) {
    if (var_decl->hasGlobalStorage() && (!var_decl->isStaticLocal()) &&
        (!var_decl->isStaticDataMember())) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Assignments to global variables not supported for %s",
          lvalue->getNameAsString()));
    }
  }

  XLS_ASSIGN_OR_RETURN(CValue found, GetIdentifier(lvalue, loc));

  if (found.type()->Is<CPointerType>()) {
    // If re-assigning the pointer, then treat it as usual
    if (rvalue.type()->Is<CPointerType>()) {
      if (rvalue.lvalue() == nullptr) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Initializer for pointer has no lvalue "
                         "(unsupported construct such as ternary?)"));
      }
    } else {
      XLS_CHECK(rvalue.rvalue().valid());

      // Otherwise, assign to the lvalue pointed to. This is necessary for
      // array-slicing, for example: void foo(int arr[4])
      // ...
      // foo(&arr[2]); // Modifies indices 2-6
      if (found.lvalue() == nullptr) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Pointer is uninitialized (no lvalue associated)"));
      }
      return Assign(found.lvalue(), rvalue, loc);
    }
  } else {
    // Not a pointer assignment
    XLS_CHECK(rvalue.rvalue().valid());
  }

  if (*found.type() != *rvalue.type()) {
    lvalue->dump();
    return absl::InvalidArgumentError(ErrorMessage(
        loc, "Cannot assign rvalue of type %s to lvalue of type %s",
        std::string(*rvalue.type()), std::string(*found.type())));
  }

  XLS_CHECK(context().variables.contains(lvalue));

  context().variables.at(lvalue) = rvalue;
  return absl::OkStatus();
}

int64_t Translator::ArrayBValueWidth(xls::BValue array_bval) {
  xls::Type* type = array_bval.node()->GetType();
  XLS_CHECK(type->IsArray());
  return type->AsArrayOrDie()->size();
}

absl::StatusOr<xls::BValue> Translator::UpdateArraySlice(
    xls::BValue array_to_update, xls::BValue start_index,
    xls::BValue slice_to_write, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(xls::Value start_index_val,
                       EvaluateBVal(start_index, loc));
  XLS_CHECK(start_index_val.IsBits());
  XLS_ASSIGN_OR_RETURN(int64_t const_index, start_index_val.bits().ToInt64());

  const int64_t total_width = ArrayBValueWidth(array_to_update);
  const int64_t slice_width = ArrayBValueWidth(slice_to_write);

  if (total_width < (const_index + slice_width)) {
    return absl::OutOfRangeError(
        ErrorMessage(loc, "Array slice out of bounds"));
  }

  int64_t remaining_width = total_width;

  std::vector<xls::BValue> parts;

  if (const_index > 0) {
    parts.push_back(context().fb->ArraySlice(
        array_to_update, context().fb->Literal(xls::SBits(0, 32), loc),
        const_index, loc));
    remaining_width -= const_index;
  }

  parts.push_back(slice_to_write);
  remaining_width -= slice_width;

  if (remaining_width > 0) {
    parts.push_back(context().fb->ArraySlice(
        array_to_update,
        context().fb->Literal(xls::SBits(total_width - remaining_width, 32),
                              loc),
        remaining_width, loc));
  }

  xls::BValue updated_array = context().fb->ArrayConcat(parts, loc);
  const int64_t updated_array_width = ArrayBValueWidth(updated_array);

  XLS_CHECK_EQ(total_width, updated_array_width);

  return updated_array;
}

absl::Status Translator::Assign(std::shared_ptr<LValue> lvalue,
                                const CValue& rvalue,
                                const xls::SourceInfo& loc) {
  if (!lvalue->is_select()) {
    return Assign(lvalue->leaf(), rvalue, loc);
  }

  // Apply the select condition to assign to true expression
  {
    PushContextGuard condition_guard(*this, loc);
    XLS_RETURN_IF_ERROR(and_condition(lvalue->cond(), loc));
    XLS_RETURN_IF_ERROR(Assign(lvalue->lvalue_true(), rvalue, loc));
  }
  // Apply ! the select condition to assign to false expression
  {
    xls::BValue sel_cond = context().fb->Not(lvalue->cond(), loc);
    PushContextGuard condition_guard(*this, loc);
    XLS_RETURN_IF_ERROR(and_condition(sel_cond, loc));
    XLS_RETURN_IF_ERROR(Assign(lvalue->lvalue_false(), rvalue, loc));
  }

  return absl::OkStatus();
}

absl::Status Translator::Assign(const clang::Expr* lvalue, const CValue& rvalue,
                                const xls::SourceInfo& loc) {
  switch (lvalue->getStmtClass()) {
    // Assign to a variable using the identifier it was declared with
    // foo = rvalue
    case clang::Stmt::DeclRefExprClass: {
      auto cast = clang_down_cast<const clang::DeclRefExpr*>(lvalue);
      const clang::NamedDecl* named = cast->getFoundDecl();
      return Assign(named, rvalue, loc);
    }
    // Assignment to a parenthetical expression
    // (...) = rvalue
    case clang::Stmt::ParenExprClass:
      return Assign(
          clang_down_cast<const clang::ParenExpr*>(lvalue)->getSubExpr(),
          rvalue, loc);
    // cast<type>(...) = rvalue
    case clang::Stmt::CXXStaticCastExprClass:
    case clang::Stmt::ImplicitCastExprClass: {
      auto cast = clang_down_cast<const clang::CastExpr*>(lvalue);

      // Don't generate pointer errors for C++ "this" keyword
      IgnorePointersGuard ignore_pointers(*this);
      if (cast->getSubExpr()->getStmtClass() == clang::Stmt::CXXThisExprClass) {
        ignore_pointers.enable();
      }

      XLS_ASSIGN_OR_RETURN(CValue sub,
                           GenerateIR_Expr(cast->getSubExpr(), loc));

      auto from_arr_type = std::dynamic_pointer_cast<CArrayType>(sub.type());

      // Inheritance
      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> to_type,
                           TranslateTypeFromClang(cast->getType(), loc));

      XLS_ASSIGN_OR_RETURN(ResolvedInheritance inheritance,
                           ResolveInheritance(sub.type(), to_type));

      CValue adjusted_rvalue = rvalue;

      // Are we casting to a derived class?
      if (inheritance.base_field != nullptr) {
        XLS_CHECK(inheritance.resolved_struct != nullptr);
        XLS_CHECK((*rvalue.type()) == *inheritance.base_field->type());
        XLS_ASSIGN_OR_RETURN(
            xls::BValue updated_derived,
            StructUpdate(sub.rvalue(), rvalue, inheritance.base_field_name,
                         *inheritance.resolved_struct, loc));
        adjusted_rvalue = CValue(updated_derived, sub.type());
      }

      return Assign(cast->getSubExpr(), adjusted_rvalue, loc);
    }
    // This happens when copy constructors with non-const reference inputs are
    // invoked. class Temporary { Temporary() { } }; class Blah { Blah(Temporary
    // &in) { } }; Blah x(Temporary());
    case clang::Stmt::MaterializeTemporaryExprClass: {
      // Ignore assignment to temporaries.
      return absl::OkStatus();
    }

    // Assign to an array element
    // (...)[index] = rvalue
    case clang::Stmt::ArraySubscriptExprClass: {
      auto* cast = clang_down_cast<const clang::ArraySubscriptExpr*>(lvalue);
      XLS_ASSIGN_OR_RETURN(CValue arr_val,
                           GenerateIR_Expr(cast->getBase(), loc));
      if (!arr_val.type()->Is<CArrayType>()) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Only array subscript assignments directly to "
                         "arrays supported (not pointers, yet)"));
      }
      auto arr_type = arr_val.type()->As<CArrayType>();
      XLS_ASSIGN_OR_RETURN(CValue idx_val,
                           GenerateIR_Expr(cast->getIdx(), loc));
      if (*rvalue.type() != *arr_type->GetElementType()) {
        return absl::InvalidArgumentError(ErrorMessage(
            loc,
            "Cannot assign rvalue of type %s to element of array of type %s",
            std::string(*rvalue.type()),
            std::string(*arr_type->GetElementType())));
      }
      auto arr_rvalue =
          CValue(context().fb->ArrayUpdate(arr_val.rvalue(), rvalue.rvalue(),
                                           {idx_val.rvalue()}, loc),
                 arr_val.type());
      return Assign(cast->getBase(), arr_rvalue, loc);
    }
    // Assign to a struct element
    // (...).member = rvalue
    case clang::Stmt::MemberExprClass: {
      const clang::MemberExpr* cast =
          clang_down_cast<const clang::MemberExpr*>(lvalue);
      clang::ValueDecl* member = cast->getMemberDecl();

      if (member->getKind() != clang::ValueDecl::Kind::Field) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Unimplemented assignment to lvalue member kind %s",
            member->getDeclKindName()));
      }

      CValue struct_prev_val;

      XLS_ASSIGN_OR_RETURN(struct_prev_val,
                           GenerateIR_Expr(cast->getBase(), loc));

      XLS_ASSIGN_OR_RETURN(auto resolved_type,
                           ResolveTypeInstance(struct_prev_val.type()));

      if (auto sitype = std::dynamic_pointer_cast<CStructType>(resolved_type)) {
        auto field = clang_down_cast<clang::FieldDecl*>(member);

        XLS_ASSIGN_OR_RETURN(
            xls::BValue new_tuple,
            StructUpdate(struct_prev_val.rvalue(), rvalue,
                         // Up cast to NamedDecl because NamedDecl pointers
                         //  are used to track identifiers
                         absl::implicit_cast<const clang::NamedDecl*>(field),
                         *sitype, loc));

        auto newval = CValue(new_tuple, struct_prev_val.type());
        return Assign(cast->getBase(), newval, loc);
      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Unimplemented fielddecl assignment to "
                         "non-struct typed lvalue of type %s",
                         string(*struct_prev_val.type())));
      }
    }
    case clang::Stmt::UnaryOperatorClass: {
      auto uop = clang_down_cast<const clang::UnaryOperator*>(lvalue);
      if (uop->getOpcode() == clang::UnaryOperatorKind::UO_AddrOf) {
        if (uop->getSubExpr()->getStmtClass() !=
            clang::Stmt::ArraySubscriptExprClass) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "Only assignment to array slices supported via pointers"));
        }
        const clang::ArraySubscriptExpr* subscript_sub_expr =
            clang_down_cast<const clang::ArraySubscriptExpr*>(
                uop->getSubExpr());

        XLS_ASSIGN_OR_RETURN(
            CValue base_cv,
            GenerateIR_Expr(subscript_sub_expr->getBase(), loc));
        XLS_CHECK(base_cv.type()->Is<CArrayType>());
        XLS_ASSIGN_OR_RETURN(
            CValue index_cv,
            GenerateIR_Expr(subscript_sub_expr->getIdx(), loc));
        XLS_CHECK(index_cv.type()->Is<CIntType>());

        XLS_ASSIGN_OR_RETURN(
            xls::BValue updated_array,
            UpdateArraySlice(base_cv.rvalue(), index_cv.rvalue(),
                             rvalue.rvalue(), loc));

        return Assign(subscript_sub_expr->getBase(),
                      CValue(updated_array, base_cv.type()), loc);
      }
      if (uop->getOpcode() != clang::UnaryOperatorKind::UO_Deref) {
        return absl::UnimplementedError(
            ErrorMessage(loc,
                         "Unimplemented assignment to unary operator lvalue "
                         "with opcode %i",
                         uop->getOpcode()));
      }

      // Deref is the pointer dereferencing operator: *ptr
      // We simply ignore this for "this", so *this just evaluates to the
      //  "this" BValue from the TranslationContext
      if (uop->getSubExpr()->getStmtClass() == clang::Stmt::CXXThisExprClass) {
        XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl,
                             GetThisDecl(loc));
        return Assign(this_decl, rvalue, loc);
      }

      return absl::UnimplementedError(absl::StrFormat(
          "Unsupported assignment to dereference of statement of class %i at "
          "%s",
          (int)uop->getSubExpr()->getStmtClass(), LocString(loc)));
    }
    case clang::Stmt::CXXThisExprClass: {
      XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl, GetThisDecl(loc));
      return Assign(this_decl, rvalue, loc);
    }
    case clang::Stmt::ConditionalOperatorClass: {
      auto cond = clang_down_cast<const clang::ConditionalOperator*>(lvalue);
      XLS_ASSIGN_OR_RETURN(
          shared_ptr<CType> result_type,
          TranslateTypeFromClang(cond->getType().getCanonicalType(), loc));
      if (!result_type->Is<CPointerType>()) {
        return absl::UnimplementedError(ErrorMessage(
            loc,
            "Ternaries in lvalues only supported for pointers, type used is %s",
            (std::string)*result_type));
      }
      XLS_ASSIGN_OR_RETURN(
          CValue lcv,
          Generate_TernaryOp(result_type, cond->getCond(), cond->getTrueExpr(),
                             cond->getFalseExpr(), loc));
      XLS_CHECK_NE(lcv.lvalue().get(), nullptr);
      return Assign(lcv.lvalue(), rvalue, loc);
    }
    default: {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Unimplemented assignment to lvalue of type %s",
                       lvalue->getStmtClassName()));
    }
  }
}

absl::Status Translator::DeclareVariable(const clang::NamedDecl* lvalue,
                                         const CValue& rvalue,
                                         const xls::SourceInfo& loc,
                                         bool check_unique_ids) {
  if (context().variables.contains(lvalue)) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Declaration '%s' duplicated\n", lvalue->getNameAsString()));
  }

  if (check_unique_ids) {
    if (check_unique_ids_.contains(lvalue)) {
      return absl::InternalError(
          ErrorMessage(loc, "Code assumes NamedDecls are unique, but %s isn't",
                       lvalue->getNameAsString()));
    }
  }
  check_unique_ids_.insert(lvalue);

  context().sf->declaration_order_by_name_[lvalue] =
      ++context().sf->declaration_count;
  context().variables[lvalue] = rvalue;
  return absl::OkStatus();
}

absl::Status Translator::DeclareStatic(const clang::NamedDecl* lvalue,
                                       const ConstValue& init,
                                       const xls::SourceInfo& loc,
                                       bool check_unique_ids) {
  XLS_CHECK(!context().sf->static_values.contains(lvalue) ||
            context().sf->static_values.at(lvalue) == init);

  context().sf->declaration_order_by_name_[lvalue] =
      ++context().sf->declaration_count;

  context().sf->static_values[lvalue] = init;

  // Mangle the name since statics with the same name may occur in different
  // contexts
  std::string xls_name = XLSNameMangle(clang::GlobalDecl(lvalue));

  XLS_ASSIGN_OR_RETURN(xls::Type * xls_type,
                       TranslateTypeToXLS(init.type(), loc));

  const xls::BValue bval = context().fb->Param(xls_name, xls_type, loc);

  XLS_CHECK(bval.valid());

  SideEffectingParameter side_effecting_param;
  side_effecting_param.type = SideEffectingParameterType::kStatic;
  side_effecting_param.param_name = bval.node()->As<xls::Param>()->GetName();
  side_effecting_param.static_value = lvalue;
  context().sf->side_effecting_parameters.push_back(side_effecting_param);

  return DeclareVariable(lvalue, CValue(bval, init.type()), loc,
                         check_unique_ids);
}

absl::StatusOr<CValue> Translator::Generate_Synthetic_ByOne(
    xls::Op xls_op, bool is_pre, CValue sub_value, const clang::Expr* sub_expr,
    const xls::SourceInfo& loc) {
  auto sub_type = std::dynamic_pointer_cast<CIntType>(sub_value.type());
  const int width = sub_type->width();
  xls::BValue literal_one = context().fb->Literal(
      sub_type->is_signed() ? xls::SBits(1, width) : xls::UBits(1, width));
  xls::BValue result_val =
      context().fb->AddBinOp(xls_op, sub_value.rvalue(), literal_one, loc);
  // No extra bits because this is only for built-ins
  std::shared_ptr<CType> result_type = sub_value.type();

  XLS_RETURN_IF_ERROR(Assign(sub_expr, CValue(result_val, result_type), loc));

  xls::BValue return_val = is_pre ? result_val : sub_value.rvalue();
  return CValue(return_val, result_type);
}

absl::StatusOr<CValue> Translator::Generate_UnaryOp(
    const clang::UnaryOperator* uop, const xls::SourceInfo& loc) {
  auto clang_op = uop->getOpcode();

  XLS_ASSIGN_OR_RETURN(CValue lhs_cv, GenerateIR_Expr(uop->getSubExpr(), loc));
  XLS_ASSIGN_OR_RETURN(
      shared_ptr<CType> result_type,
      TranslateTypeFromClang(uop->getType().getCanonicalType(), loc));

  if (clang_op == clang::UnaryOperatorKind::UO_AddrOf) {
    auto result_pointer_type =
        std::dynamic_pointer_cast<CPointerType>(result_type);
    XLS_CHECK_NE(result_pointer_type.get(), nullptr);

    if (context().lvalue_mode) {
      // Include & in the lvalue expression, so that Assign()
      // can just look for that
      return CValue(xls::BValue(), result_pointer_type,
                    /*disable_type_check=*/false,
                    std::make_shared<LValue>(uop));
    }

    const clang::Expr* sub_expr = uop->getSubExpr();
    if (sub_expr->getStmtClass() != clang::Stmt::ArraySubscriptExprClass) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Address of sub expression of class %i",
                       static_cast<int>(sub_expr->getStmtClass())));
    }
    auto array_subscript_expr =
        clang_down_cast<const clang::ArraySubscriptExpr*>(sub_expr);

    const clang::Expr* base_expr = array_subscript_expr->getBase();
    XLS_ASSIGN_OR_RETURN(CValue base_cv, GenerateIR_Expr(base_expr, loc));

    if (!base_cv.type()->Is<CArrayType>()) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc,
                       "Address-of (&) operator "
                       "only supported on arrays, for array slicing"));
    }

    XLS_ASSIGN_OR_RETURN(CValue array_idx_cv,
                         GenerateIR_Expr(array_subscript_expr->getIdx(), loc));
    auto array_int_type =
        std::dynamic_pointer_cast<CIntType>(array_idx_cv.type());
    if (array_int_type == nullptr) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc, "Array index must be an integer"));
    }

    std::shared_ptr<CType> pointee_type = result_pointer_type->GetPointeeType();

    XLS_CHECK(*lhs_cv.type() == *pointee_type);

    xls::BValue array_slice_in = base_cv.rvalue();

    XLS_CHECK(array_slice_in.GetType()->IsArray());

    // Out of bounds slices wrap around
    const int64_t array_slice_in_size =
        array_slice_in.GetType()->AsArrayOrDie()->size();

    xls::BValue sliced_array = context().fb->ArraySlice(
        array_slice_in, array_idx_cv.rvalue(), array_slice_in_size, loc);

    XLS_CHECK(sliced_array.GetType()->IsArray());

    return CValue(sliced_array, std::make_shared<CArrayType>(
                                    pointee_type, array_slice_in_size));
  }

  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> resolved_type,
                       ResolveTypeInstance(result_type));

  XLS_ASSIGN_OR_RETURN(xls::BValue lhs_cvc,
                       GenTypeConvert(lhs_cv, result_type, loc));
  CValue lhs_cvcv(lhs_cvc, lhs_cv.type());

  if (auto result_int_type = std::dynamic_pointer_cast<CIntType>(result_type)) {
    switch (clang_op) {
      case clang::UnaryOperatorKind::UO_Minus:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNeg, lhs_cvcv.rvalue(), loc),
            result_type);
      case clang::UnaryOperatorKind::UO_Plus:
        return lhs_cvcv;
      case clang::UnaryOperatorKind::UO_LNot:
      case clang::UnaryOperatorKind::UO_Not:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNot, lhs_cvcv.rvalue(), loc),
            result_type);
      case clang::UnaryOperatorKind::UO_PreInc:
        return Generate_Synthetic_ByOne(xls::Op::kAdd, true, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      case clang::UnaryOperatorKind::UO_PreDec:
        return Generate_Synthetic_ByOne(xls::Op::kSub, true, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      case clang::UnaryOperatorKind::UO_PostInc:
        return Generate_Synthetic_ByOne(xls::Op::kAdd, false, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      case clang::UnaryOperatorKind::UO_PostDec:
        return Generate_Synthetic_ByOne(xls::Op::kSub, false, lhs_cvcv,
                                        uop->getSubExpr(), loc);
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented unary operator %i", clang_op));
    }
  } else if (clang_op == clang::UnaryOperatorKind::UO_Deref &&
             (std::dynamic_pointer_cast<CStructType>(resolved_type) ||
              std::dynamic_pointer_cast<CBitsType>(resolved_type))) {
    // We don't support pointers so we don't care about this.
    // It's needed for *this
    XLS_CHECK(uop->getSubExpr()->getStmtClass() ==
              clang::Stmt::CXXThisExprClass);
    return lhs_cvcv;
  } else if (auto result_int_type =
                 std::dynamic_pointer_cast<CBoolType>(result_type)) {
    switch (clang_op) {
      case clang::UnaryOperatorKind::UO_LNot:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNot, lhs_cvcv.rvalue(), loc),
            result_type);
      default:
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented unary operator %i", clang_op));
    }
  } else {
    return absl::UnimplementedError(
        ErrorMessage(loc,
                     "Unary operators on types other than builtin-int "
                     "unimplemented for type %s",
                     std::string(*result_type)));
  }
}

absl::StatusOr<CValue> Translator::Generate_BinaryOp(
    clang::BinaryOperatorKind clang_op, bool is_assignment,
    std::shared_ptr<CType> result_type, const clang::Expr* lhs,
    const clang::Expr* rhs, const xls::SourceInfo& loc) {
  CValue result;
  {
    // Don't reduce operands to logical boolean operators to 1 bit.
    std::shared_ptr<CType> input_type = result_type;
    if (input_type->Is<CBoolType>()) {
      XLS_ASSIGN_OR_RETURN(input_type,
                           TranslateTypeFromClang(lhs->getType(), loc));
    }

    if (clang_op == clang::BinaryOperatorKind::BO_Comma) {
      CValue lhs_cv;
      XLS_ASSIGN_OR_RETURN(lhs_cv, GenerateIR_Expr(lhs, loc));
    }
    XLS_ASSIGN_OR_RETURN(
        xls::Op xls_op,
        XLSOpcodeFromClang(clang_op, *input_type, *result_type, loc));

    CValue rhs_cv;
    {
      // For pointer assignments
      LValueModeGuard lvalue_mode(*this);
      XLS_ASSIGN_OR_RETURN(rhs_cv, GenerateIR_Expr(rhs, loc));
    }

    XLS_ASSIGN_OR_RETURN(xls::BValue rhs_cvc,
                         GenTypeConvert(rhs_cv, input_type, loc));
    CValue rhs_cvcv(rhs_cvc, input_type, /*disable_type_check=*/false,
                    rhs_cv.lvalue());

    result = rhs_cvcv;

    if (xls_op != xls::Op::kIdentity) {
      XLS_ASSIGN_OR_RETURN(CValue lhs_cv, GenerateIR_Expr(lhs, loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue lhs_cvc,
                           GenTypeConvert(lhs_cv, input_type, loc));
      CValue lhs_cvcv(lhs_cvc, lhs_cv.type());

      if (xls::IsOpClass<xls::CompareOp>(xls_op)) {
        result = CValue(context().fb->AddCompareOp(xls_op, lhs_cvcv.rvalue(),
                                                   rhs_cvcv.rvalue(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::ArithOp>(xls_op)) {
        result = CValue(context().fb->AddArithOp(
                            xls_op, lhs_cvcv.rvalue(), rhs_cvcv.rvalue(),
                            result_type->GetBitWidth(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::BinOp>(xls_op)) {
        result = CValue(context().fb->AddBinOp(xls_op, lhs_cvcv.rvalue(),
                                               rhs_cvcv.rvalue(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::NaryOp>(xls_op)) {
        result = CValue(
            context().fb->AddNaryOp(
                xls_op,
                std::vector<xls::BValue>{lhs_cvcv.rvalue(), rhs_cvcv.rvalue()},
                loc),
            result_type);

      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Internal error: unknown XLS op class"));
      }
    }
  }

  if (is_assignment) {
    XLS_RETURN_IF_ERROR(Assign(lhs, result, loc));
  }

  return result;
}

absl::StatusOr<CValue> Translator::Generate_TernaryOp(
    std::shared_ptr<CType> result_type, const clang::Expr* cond_expr,
    const clang::Expr* true_expr, const clang::Expr* false_expr,
    const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(CValue sel_val, GenerateIR_Expr(cond_expr, loc));
  XLS_CHECK(sel_val.type()->Is<CBoolType>());

  XLS_ASSIGN_OR_RETURN(CValue true_cv, GenerateIR_Expr(true_expr, loc));
  XLS_ASSIGN_OR_RETURN(CValue false_cv, GenerateIR_Expr(false_expr, loc));

  if (result_type->Is<CPointerType>()) {
    if (context().lvalue_mode) {
      XLS_CHECK_NE(true_cv.lvalue(), nullptr);
      XLS_CHECK_NE(false_cv.lvalue(), nullptr);
      auto select_lvalue = std::make_shared<LValue>(
          sel_val.rvalue(), true_cv.lvalue(), false_cv.lvalue());
      return CValue(xls::BValue(), result_type, /*disable_type_check=*/false,
                    select_lvalue);
    }
    // RValue mode
    XLS_RETURN_IF_ERROR(
        MinSizeArraySlices(true_cv, false_cv, result_type, loc));
  }

  return Generate_TernaryOp(sel_val.rvalue(), true_cv, false_cv, result_type,
                            loc);
}

absl::StatusOr<CValue> Translator::Generate_TernaryOp(
    xls::BValue cond, CValue true_cv, CValue false_cv,
    std::shared_ptr<CType> result_type, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(xls::BValue true_val,
                       GenTypeConvert(true_cv, result_type, loc));
  XLS_ASSIGN_OR_RETURN(xls::BValue false_val,
                       GenTypeConvert(false_cv, result_type, loc));
  xls::BValue ret_val = context().fb->Select(cond, true_val, false_val, loc);
  return CValue(ret_val, result_type);
}

absl::StatusOr<std::shared_ptr<CType>> Translator::ResolveTypeInstance(
    std::shared_ptr<CType> t) {
  auto inst = std::dynamic_pointer_cast<CInstantiableTypeAlias>(t);

  // Check if it's a concrete type or an alias
  if (inst == nullptr) {
    return t;
  }

  // Check if it's already been scanned
  {
    auto found = inst_types_.find(inst);

    if (found != inst_types_.end()) {
      return found->second;
    }
  }

  // Needs to be scanned from AST
  XLS_RETURN_IF_ERROR(
      ScanStruct(clang_down_cast<const clang::RecordDecl*>(inst->base())));

  auto found = inst_types_.find(inst);
  XLS_CHECK(found != inst_types_.end());

  return found->second;
}

absl::StatusOr<std::shared_ptr<CType>> Translator::ResolveTypeInstanceDeeply(
    std::shared_ptr<CType> t) {
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ret, ResolveTypeInstance(t));

  // Handle structs
  {
    auto ret_struct = std::dynamic_pointer_cast<const CStructType>(ret);

    if (ret_struct != nullptr) {
      std::vector<std::shared_ptr<CField>> fields;
      for (const std::shared_ptr<CField>& field : ret_struct->fields()) {
        XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> field_type,
                             ResolveTypeInstanceDeeply(field->type()));
        fields.push_back(std::make_shared<CField>(field->name(), field->index(),
                                                  field_type));
      }
      return std::make_shared<CStructType>(fields, ret_struct->no_tuple_flag());
    }
  }

  // Handle arrays
  {
    auto ret_array = std::dynamic_pointer_cast<const CArrayType>(ret);

    if (ret_array != nullptr) {
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> elem_type,
          ResolveTypeInstanceDeeply(ret_array->GetElementType()));
      return std::make_shared<CArrayType>(elem_type, ret_array->GetSize());
    }
  }

  return ret;
}

absl::StatusOr<GeneratedFunction*> Translator::TranslateFunctionToXLS(
    const clang::FunctionDecl* decl) {
  auto found =
      inst_functions_.find(absl::implicit_cast<const clang::NamedDecl*>(decl));
  if (found != inst_functions_.end()) {
    return found->second.get();
  } else {
    return GenerateIR_Function(decl);
  }
}

absl::StatusOr<CValue> Translator::GenerateIR_Call(const clang::CallExpr* call,
                                                   const xls::SourceInfo& loc) {
  const clang::FunctionDecl* funcdecl = call->getDirectCallee();

  if (funcdecl->getNameAsString() == "__xlscc_unimplemented") {
    return absl::UnimplementedError(ErrorMessage(loc, "Unimplemented marker"));
  }

  CValue this_value_orig;

  const clang::Expr* this_expr = nullptr;
  xls::BValue thisval;
  xls::BValue* pthisval = nullptr;

  int skip_args = 0;

  // If true, an extra return value is expected for the modified "this" object
  //  in a method call. This mechanism is similar to that used for mutable
  //  reference parameters. A "this" pointer cannot be used in the usual way,
  //  since BValues are immutable, and pointers are unsupported.
  bool add_this_return = false;

  // This variable is set to true for specific calls, such as assignment
  //  operators, which are considered "sequencing safe", ie unsequenced
  //  assignment error checking isn't needed for them.
  bool sequencing_safe = false;

  // Evaluate if "this" argument is necessary (eg for method calls)
  if (call->getStmtClass() == clang::Stmt::StmtClass::CXXMemberCallExprClass) {
    auto member_call = clang_down_cast<const clang::CXXMemberCallExpr*>(call);
    this_expr = member_call->getImplicitObjectArgument();
    pthisval = &thisval;

    const clang::QualType& thisQual =
        member_call->getMethodDecl()->getThisType();
    XLS_CHECK(thisQual->isPointerType());
    add_this_return = !thisQual->getPointeeType().isConstQualified();
  } else if (call->getStmtClass() ==
             clang::Stmt::StmtClass::CXXOperatorCallExprClass) {
    auto op_call = clang_down_cast<const clang::CXXOperatorCallExpr*>(call);

    if (op_call->isAssignmentOp()) sequencing_safe = true;

    const clang::FunctionDecl* callee = op_call->getDirectCallee();
    if (callee->getKind() == clang::Decl::CXXMethod) {
      CValue ret;

      // There is a special case here for a certain expression form
      XLS_ASSIGN_OR_RETURN(bool applied,
                           ApplyArrayAssignHack(op_call, loc, &ret));
      if (applied) {
        return ret;
      }

      // this comes as first argument for operators
      this_expr = call->getArg(0);

      const clang::QualType& thisQual =
          clang_down_cast<const clang::CXXMethodDecl*>(
              op_call->getDirectCallee())
              ->getThisType();
      XLS_CHECK(thisQual->isPointerType());
      add_this_return = !thisQual->getPointeeType().isConstQualified();
      ++skip_args;
    }
  }

  if (this_expr) {
    {
      // The Assign() statement below will take care of any assignments
      //  in the expression for "this". Don't do these twice, as it can cause
      //  issues like double-increment https://github.com/google/xls/issues/389
      MaskAssignmentsGuard mask(*this);
      XLS_ASSIGN_OR_RETURN(this_value_orig, GenerateIR_Expr(this_expr, loc));
    }

    thisval = this_value_orig.rvalue();
    pthisval = &thisval;
  }

  std::vector<const clang::Expr*> args;
  for (int pi = skip_args; pi < call->getNumArgs(); ++pi) {
    args.push_back(call->getArg(pi));
  }
  XLS_ASSIGN_OR_RETURN(CValue call_res,
                       GenerateIR_Call(funcdecl, args, pthisval, loc));

  if (add_this_return) {
    XLS_CHECK(pthisval);
    XLS_RETURN_IF_ERROR(
        Assign(this_expr, CValue(thisval, this_value_orig.type()), loc));
  }

  return call_res;
}

absl::StatusOr<bool> Translator::ApplyArrayAssignHack(
    const clang::CXXOperatorCallExpr* op_call, const xls::SourceInfo& loc,
    CValue* output) {
  // Hack to avoid returning reference object.
  //  xls_int[n] = val
  //  CXXOperatorCallExpr '=' {
  //    MaterializeTemporaryExpr {
  //      CXXOperatorCallExpr '[]' {
  //      }
  //    }
  //  }
  if (!op_call->isAssignmentOp()) {
    return false;
  }
  if (op_call->getArg(0)->getStmtClass() !=
      clang::Stmt::MaterializeTemporaryExprClass) {
    return false;
  }
  auto materialize = clang_down_cast<const clang::MaterializeTemporaryExpr*>(
      op_call->getArg(0));
  if (materialize->getSubExpr()->getStmtClass() !=
      clang::Stmt::CXXOperatorCallExprClass) {
    return false;
  }
  auto sub_op_call = clang_down_cast<const clang::CXXOperatorCallExpr*>(
      materialize->getSubExpr());
  if (sub_op_call->getOperator() !=
      clang::OverloadedOperatorKind::OO_Subscript) {
    return false;
  }
  const clang::Expr* ivalue = sub_op_call->getArg(1);
  const clang::Expr* rvalue = op_call->getArg(1);
  const clang::Expr* lvalue = sub_op_call->getArg(0);

  const clang::CXXRecordDecl* stype = lvalue->getType()->getAsCXXRecordDecl();
  if (stype == nullptr) {
    return false;
  }
  for (auto method : stype->methods()) {
    if (method->getNameAsString() == "set_element") {
      auto to_call = dynamic_cast<const clang::FunctionDecl*>(method);

      XLS_CHECK(to_call != nullptr);

      XLS_ASSIGN_OR_RETURN(CValue lvalue_initial, GenerateIR_Expr(lvalue, loc));

      xls::BValue this_inout = lvalue_initial.rvalue();
      XLS_ASSIGN_OR_RETURN(
          CValue f_return,
          GenerateIR_Call(to_call, {ivalue, rvalue}, &this_inout, loc));
      XLS_RETURN_IF_ERROR(
          Assign(lvalue, CValue(this_inout, lvalue_initial.type()), loc));
      *output = f_return;
      return true;
    }
  }
  // Recognized the pattern, but no set_element() method to use
  return false;
}

absl::StatusOr<const clang::Stmt*> Translator::GetFunctionBody(
    const clang::FunctionDecl*& funcdecl) {
  const bool trivial = funcdecl->hasTrivialBody() || funcdecl->isTrivial();

  if (!trivial && funcdecl->getBody() == nullptr) {
    return absl::NotFoundError(ErrorMessage(GetLoc(*funcdecl),
                                            "Function %s used but has no body",
                                            funcdecl->getNameAsString()));
  }

  // funcdecl parameters may be different for forward declarations
  const clang::FunctionDecl* definition = nullptr;
  const clang::Stmt* body = funcdecl->getBody(definition);
  if (definition == nullptr) {
    if (!trivial) {
      return absl::NotFoundError(ErrorMessage(
          GetLoc(*funcdecl), "Function %s has no body or definition",
          funcdecl->getNameAsString()));
    }
  } else {
    funcdecl = definition;
  }
  XLS_CHECK(body != nullptr || trivial);
  return body;
}

// this_inout can be nullptr for non-members
absl::StatusOr<CValue> Translator::GenerateIR_Call(
    const clang::FunctionDecl* funcdecl,
    std::vector<const clang::Expr*> expr_args, xls::BValue* this_inout,
    const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(const clang::Stmt* body, GetFunctionBody(funcdecl));
  (void)body;

  // Translate external channels
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* callee_param = funcdecl->getParamDecl(pi);

    XLS_ASSIGN_OR_RETURN(bool is_channel, TypeIsChannel(callee_param->getType(),
                                                        GetLoc(*callee_param)));
    if (!is_channel) {
      continue;
    }

    const clang::Expr* call_arg = expr_args[pi];
    if (call_arg->getStmtClass() != clang::Stmt::DeclRefExprClass) {
      return absl::UnimplementedError(ErrorMessage(
          GetLoc(*callee_param), "IO operations should be DeclRefs"));
    }
    auto call_decl_ref_arg =
        clang_down_cast<const clang::DeclRefExpr*>(call_arg);
    if (call_decl_ref_arg->getDecl()->getKind() != clang::Decl::ParmVar) {
      return absl::UnimplementedError(ErrorMessage(
          GetLoc(*callee_param), "IO operations should be on parameters"));
    }

    auto caller_channel_param = clang_down_cast<const clang::ParmVarDecl*>(
        call_decl_ref_arg->getDecl());

    if (!external_channels_by_param_.contains(caller_channel_param)) {
      XLS_CHECK(io_test_mode_)
          << "Caller channel param " << caller_channel_param->getNameAsString()
          << " not in external_channels_by_param_ map";
    }

    if (external_channels_by_param_.contains(callee_param) &&
        (external_channels_by_param_.at(callee_param) !=
         external_channels_by_param_.at(caller_channel_param))) {
      return absl::UnimplementedError(
          ErrorMessage(GetLoc(*callee_param),
                       "IO ops in pipelined loops in subroutines called "
                       "with multiple different channel arguments"));
    }

    if (external_channels_by_param_.contains(caller_channel_param)) {
      external_channels_by_param_[callee_param] =
          external_channels_by_param_.at(caller_channel_param);
    }
  }

  // Make sure subroutine is generated
  XLS_ASSIGN_OR_RETURN(GeneratedFunction * func,
                       TranslateFunctionToXLS(funcdecl));

  XLS_CHECK_NE(func, nullptr);

  // Function with no outputs
  if (func->xls_func == nullptr) {
    return CValue();
  }

  std::vector<xls::BValue> args;
  int expected_returns = 0;

  // Add this if needed
  bool add_this_return = false;
  if (this_inout) {
    args.push_back(*this_inout);

    // "This" is a PointerType, ignore and treat as reference
    auto method = clang_down_cast<const clang::CXXMethodDecl*>(funcdecl);
    const clang::QualType& thisQual = method->getThisType();
    XLS_CHECK(thisQual->isPointerType());

    add_this_return = !thisQual->getPointeeType().isConstQualified();
  }

  // Number of return values expected. If >1, the return will be a tuple.
  // (See MakeFunctionReturn()).
  if (add_this_return) ++expected_returns;
  if (!funcdecl->getReturnType()->isVoidType()) ++expected_returns;

  if (expr_args.size() != funcdecl->getNumParams()) {
    return absl::UnimplementedError(ErrorMessage(
        loc,
        "Parameter count mismatch: %i params in FunctionDecl, %i arguments to "
        "call",
        funcdecl->getNumParams(), static_cast<int>(expr_args.size())));
  }

  // Add other parameters
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

    // Map callee IO ops last
    if (func->io_channels_by_param.contains(p)) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      ++expected_returns;
    }

    XLS_ASSIGN_OR_RETURN(CValue argv, GenerateIR_Expr(expr_args[pi], loc));
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> argt,
                         TranslateTypeFromClang(stripped.base, loc));

    xls::BValue pass_bval = argv.rvalue();
    std::shared_ptr<CType> pass_type = argv.type();

    if (argv.type()->Is<CPointerType>() || argv.type()->Is<CArrayType>()) {
      auto arg_arr_type = argt->As<CArrayType>();

      // Pointer to array
      if (argv.type()->Is<CPointerType>()) {
        if (argv.lvalue() == nullptr) {
          return absl::UnimplementedError(
              ErrorMessage(loc,
                           "Pointer value has no lvalue (unsupported "
                           "construct such as ternary?)"));
        }
        XLS_ASSIGN_OR_RETURN(CValue pass_rval,
                             GenerateIR_Expr(argv.lvalue(), loc));
        pass_bval = pass_rval.rvalue();
      }

      XLS_CHECK(pass_bval.valid());

      int64_t pass_bval_arr_size = ArrayBValueWidth(pass_bval);

      if (pass_bval_arr_size < arg_arr_type->GetSize()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Array slice out of bounds"));
      }

      if (pass_bval_arr_size != arg_arr_type->GetSize()) {
        pass_bval = context().fb->ArraySlice(
            pass_bval, context().fb->Literal(xls::SBits(0, 32), loc),
            arg_arr_type->GetSize(), loc);
      }

      std::shared_ptr<CType> element_type;

      if (argv.type()->Is<CPointerType>()) {
        auto argv_pointer_type = argv.type()->As<CPointerType>();
        element_type = argv_pointer_type->GetPointeeType();
      } else if (argv.type()->Is<CArrayType>()) {
        auto argv_array_type = argv.type()->As<CArrayType>();
        element_type = argv_array_type->GetElementType();
      } else {
        XLS_CHECK_EQ("Internal consistency failure", nullptr);
      }

      pass_type =
          std::make_shared<CArrayType>(element_type, arg_arr_type->GetSize());
    }

    if (*pass_type != *argt) {
      return absl::InternalError(ErrorMessage(
          loc,
          "Internal error: expression type %s doesn't match "
          "parameter type %s in function %s",
          string(*argv.type()), string(*argt), funcdecl->getNameAsString()));
    }

    args.push_back(pass_bval);
  }

  // Map callee IO ops
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* callee_channel = funcdecl->getParamDecl(pi);

    if (!func->io_channels_by_param.contains(callee_channel)) {
      continue;
    }

    const clang::Expr* call_arg = expr_args[pi];

    // Duplicated code in InterceptIOOp()?
    if (call_arg->getStmtClass() != clang::Stmt::DeclRefExprClass) {
      return absl::UnimplementedError(
          absl::StrFormat("IO operations should be DeclRefs"));
    }
    auto call_decl_ref_arg =
        clang_down_cast<const clang::DeclRefExpr*>(call_arg);
    if (call_decl_ref_arg->getDecl()->getKind() != clang::Decl::ParmVar) {
      return absl::UnimplementedError(
          absl::StrFormat("IO operations should be on parameters"));
    }

    auto caller_channel_param = clang_down_cast<const clang::ParmVarDecl*>(
        call_decl_ref_arg->getDecl());

    IOChannel* caller_channel =
        context().sf->io_channels_by_param.at(caller_channel_param);

    for (IOOp& callee_op : func->io_ops) {
      // Skip internal channels
      if (!func->params_by_io_channel.contains(callee_op.channel)) {
        continue;
      }

      const clang::ParmVarDecl* callee_op_channel =
          func->params_by_io_channel.at(callee_op.channel);
      if (callee_op_channel != callee_channel) {
        continue;
      }
      if (!context().sf->caller_channels_by_callee_op.contains(&callee_op)) {
        context().sf->caller_channels_by_callee_op[&callee_op] = caller_channel;
      } else {
        XLS_CHECK_EQ(context().sf->caller_channels_by_callee_op.at(&callee_op),
                     caller_channel);
      }
    }
  }

  // Translate generated channels
  for (IOOp& callee_op : func->io_ops) {
    if (callee_op.channel->generated == nullptr) {
      continue;
    }
    if (context().sf->caller_channels_by_callee_op.contains(&callee_op)) {
      continue;
    }
    IOChannel* callee_generated_channel = callee_op.channel;
    context().sf->io_channels.push_back(*callee_generated_channel);
    IOChannel* caller_generated_channel = &context().sf->io_channels.back();
    caller_generated_channel->total_ops = 0;
    context().sf->caller_channels_by_callee_op[&callee_op] =
        caller_generated_channel;
  }

  // Map callee ops. There can be multiple for one channel
  std::multimap<IOOp*, IOOp*> caller_ops_by_callee_op;

  for (IOOp& callee_op : func->io_ops) {
    IOChannel* caller_channel =
        context().sf->caller_channels_by_callee_op.at(&callee_op);

    // Add super op
    IOOp caller_op;

    caller_op.op = callee_op.op;
    caller_op.sub_op = &callee_op;

    XLS_ASSIGN_OR_RETURN(
        IOOp * caller_op_ptr,
        AddOpToChannel(caller_op, caller_channel, callee_op.op_location));
    caller_ops_by_callee_op.insert(
        std::pair<IOOp*, IOOp*>(&callee_op, caller_op_ptr));

    // Count expected IO returns
    ++expected_returns;
  }

  // Pass side effecting parameters to call in the order they are declared
  for (const SideEffectingParameter& side_effecting_param :
       func->side_effecting_parameters) {
    switch (side_effecting_param.type) {
      case SideEffectingParameterType::kIOOp: {
        IOOp* callee_op = side_effecting_param.io_op;
        XLS_CHECK(callee_op->op == OpType::kRecv);
        auto range = caller_ops_by_callee_op.equal_range(callee_op);
        for (auto caller_it = range.first; caller_it != range.second;
             ++caller_it) {
          IOOp* caller_op = caller_it->second;
          XLS_CHECK(caller_op->op == OpType::kRecv);
          XLS_CHECK(caller_op->input_value.rvalue().valid());
          args.push_back(caller_op->input_value.rvalue());
          // Expected return already expected in above loop
        }
        break;
      }
      case SideEffectingParameterType::kStatic: {
        // May already be declared if there are multiple calls to the same
        // static-containing function
        if (!context().variables.contains(side_effecting_param.static_value)) {
          XLS_RETURN_IF_ERROR(DeclareStatic(
              side_effecting_param.static_value,
              func->static_values.at(side_effecting_param.static_value), loc,
              /* check_unique_ids= */ false));
        }
        XLS_ASSIGN_OR_RETURN(
            CValue value,
            GetIdentifier(side_effecting_param.static_value, loc));
        XLS_CHECK(value.rvalue().valid());
        args.push_back(value.rvalue());
        // Count expected static returns
        ++expected_returns;
        break;
      }
      default: {
        return absl::InternalError(
            ErrorMessage(loc, "Unknown type of SideEffectingParameter"));
      }
    }
  }
  xls::BValue raw_return = context().fb->Invoke(args, func->xls_func, loc);
  XLS_CHECK(expected_returns == 0 || raw_return.valid());

  list<xls::BValue> unpacked_returns;
  if (expected_returns == 1) {
    unpacked_returns.emplace_back(raw_return);
  } else {
    for (int r = 0; r < expected_returns; ++r) {
      unpacked_returns.emplace_back(
          GetFunctionReturn(raw_return, r, expected_returns, funcdecl, loc));
    }
  }

  CValue retval;

  // First static outputs from callee
  for (const clang::NamedDecl* namedecl :
       func->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = func->static_values.at(namedecl);

    XLS_CHECK(!unpacked_returns.empty());
    xls::BValue static_output = unpacked_returns.front();
    unpacked_returns.pop_front();

    // Skip assignment to on reset static, as assignment to globals is an error
    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      continue;
    }
    XLS_RETURN_IF_ERROR(
        Assign(namedecl, CValue(static_output, initval.type()), loc));
  }

  // Then this return
  if (add_this_return) {
    XLS_CHECK(!unpacked_returns.empty());
    *this_inout = unpacked_returns.front();
    unpacked_returns.pop_front();
  }

  // Then explicit return
  if (funcdecl->getReturnType()->isVoidType()) {
    retval = CValue(xls::BValue(), shared_ptr<CType>(new CVoidType()));
  } else {
    XLS_ASSIGN_OR_RETURN(
        auto ctype,
        TranslateTypeFromClang(funcdecl->getReturnType(), loc,
                               /*allow_references=*/funcdecl->isDefaulted()));
    XLS_CHECK(!unpacked_returns.empty());
    retval = CValue(unpacked_returns.front(), ctype);
    unpacked_returns.pop_front();
  }

  // Then reference parameter returns
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

    // IO returns are later
    if (func->io_channels_by_param.contains(p)) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_ASSIGN_OR_RETURN(
        shared_ptr<CType> ctype,
        TranslateTypeFromClang(stripped.base, GetLoc(*p),
                               /*allow_references=*/funcdecl->isDefaulted()));

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      XLS_CHECK(!unpacked_returns.empty());
      LValueModeGuard lvalue_guard(*this);
      XLS_RETURN_IF_ERROR(
          Assign(expr_args[pi], CValue(unpacked_returns.front(), ctype), loc));
      unpacked_returns.pop_front();
    }
  }

  // Callee IO returns
  for (IOOp& callee_op : func->io_ops) {
    auto range = caller_ops_by_callee_op.equal_range(&callee_op);
    for (auto caller_it = range.first; caller_it != range.second; ++caller_it) {
      IOOp* caller_op = caller_it->second;

      XLS_CHECK(!unpacked_returns.empty());
      caller_op->ret_value = unpacked_returns.front();
      unpacked_returns.pop_front();
    }
  }

  XLS_CHECK(unpacked_returns.empty());
  return retval;
}

absl::StatusOr<Translator::ResolvedInheritance> Translator::ResolveInheritance(
    std::shared_ptr<CType> sub_type, std::shared_ptr<CType> to_type) {
  auto sub_struct =
      std::dynamic_pointer_cast<const CInstantiableTypeAlias>(sub_type);
  auto to_struct =
      std::dynamic_pointer_cast<const CInstantiableTypeAlias>(to_type);

  if (sub_struct && to_struct) {
    XLS_ASSIGN_OR_RETURN(auto sub_struct_res, ResolveTypeInstance(sub_type));
    auto resolved_struct =
        std::dynamic_pointer_cast<const CStructType>(sub_struct_res);
    if (resolved_struct) {
      std::shared_ptr<CField> base_field =
          resolved_struct->get_field(to_struct->base());

      // Derived to Base
      if (base_field) {
        ResolvedInheritance ret;
        ret.base_field = base_field;
        ret.resolved_struct = resolved_struct;
        ret.base_field_name = to_struct->base();
        return ret;
      }
    }
  }
  return ResolvedInheritance();
}

absl::StatusOr<CValue> Translator::GenerateIR_Expr(const clang::Expr* expr,
                                                   const xls::SourceInfo& loc) {
  switch (expr->getStmtClass()) {
    case clang::Stmt::UnaryOperatorClass: {
      auto uop = clang_down_cast<const clang::UnaryOperator*>(expr);
      return Generate_UnaryOp(uop, loc);
    }
    // Compound assignment is like a += b
    case clang::Stmt::CompoundAssignOperatorClass:
    case clang::Stmt::BinaryOperatorClass: {
      auto bop = clang_down_cast<const clang::BinaryOperator*>(expr);
      auto clang_op = bop->getOpcode();
      XLS_ASSIGN_OR_RETURN(
          shared_ptr<CType> result_type,
          TranslateTypeFromClang(bop->getType().getCanonicalType(), loc));
      return Generate_BinaryOp(clang_op, bop->isAssignmentOp(), result_type,
                               bop->getLHS(), bop->getRHS(), loc);
    }
    // Ternary: a ? b : c
    case clang::Stmt::ConditionalOperatorClass: {
      auto cond = clang_down_cast<const clang::ConditionalOperator*>(expr);
      XLS_ASSIGN_OR_RETURN(
          shared_ptr<CType> result_type,
          TranslateTypeFromClang(cond->getType().getCanonicalType(), loc));
      return Generate_TernaryOp(result_type, cond->getCond(),
                                cond->getTrueExpr(), cond->getFalseExpr(), loc);
    }
    case clang::Stmt::CXXMemberCallExprClass:
    case clang::Stmt::CXXOperatorCallExprClass:
    case clang::Stmt::CallExprClass: {
      auto call = clang_down_cast<const clang::CallExpr*>(expr);

      XLS_ASSIGN_OR_RETURN(IOOpReturn ret, InterceptIOOp(call, GetLoc(*call)));

      // If this call is an IO op, then return the IO value, rather than
      //  generating the call.
      if (!ret.generate_expr) {
        return ret.value;
      }

      return GenerateIR_Call(call, loc);
    }
    case clang::Stmt::IntegerLiteralClass: {
      auto ilit = clang_down_cast<const clang::IntegerLiteral*>(expr);
      llvm::APInt api = ilit->getValue();
      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                           TranslateTypeFromClang(ilit->getType(), loc));
      // Raw data is in little endian format
      auto api_raw = reinterpret_cast<const uint8_t*>(api.getRawData());
      vector<uint8_t> truncated;
      const int truncated_n = ((ctype->GetBitWidth() + 7) / 8);
      truncated.reserve(truncated_n);
      for (int i = 0; i < truncated_n; ++i) {
        truncated.emplace_back(api_raw[i]);
      }
      // Convert to big endian
      std::reverse(truncated.begin(), truncated.end());
      // FromBytes() accepts big endian format
      auto lbits = xls::Bits::FromBytes(truncated, ctype->GetBitWidth());
      return CValue(context().fb->Literal(lbits, loc), ctype);
    }
    case clang::Stmt::CharacterLiteralClass: {
      auto charlit = clang_down_cast<const clang::CharacterLiteral*>(expr);
      if (charlit->getKind() != clang::CharacterLiteral::CharacterKind::Ascii) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented character literaly type %i",
                         static_cast<int>(charlit->getKind())));
      }
      shared_ptr<CType> ctype(new CIntType(8, true, true));
      return CValue(
          context().fb->Literal(xls::UBits(charlit->getValue(), 8), loc),
          ctype);
    }
    case clang::Stmt::CXXBoolLiteralExprClass: {
      auto bl = clang_down_cast<const clang::CXXBoolLiteralExpr*>(expr);
      xls::BValue val =
          context().fb->Literal(xls::UBits(bl->getValue() ? 1 : 0, 1), loc);
      return CValue(val, shared_ptr<CType>(new CBoolType()));
    }
    // This is just a marker Clang places in the AST to show that a template
    //  parameter was substituted. It wraps the substituted value, like:
    // SubstNonTypeTemplateParmExprClass { replacement = IntegerLiteral }
    case clang::Stmt::SubstNonTypeTemplateParmExprClass: {
      auto subst =
          clang_down_cast<const clang::SubstNonTypeTemplateParmExpr*>(expr);
      return GenerateIR_Expr(subst->getReplacement(), loc);
    }
    // Similar behavior for all cast styles. Clang already enforced the C++
    //  static type-checking rules by this point.
    case clang::Stmt::CXXFunctionalCastExprClass:
    case clang::Stmt::CStyleCastExprClass:
    case clang::Stmt::CXXStaticCastExprClass:
    case clang::Stmt::ImplicitCastExprClass: {
      // For converting this pointer from base to derived
      auto cast = clang_down_cast<const clang::CastExpr*>(expr);
      // Don't generate pointer errors for C++ "this" keyword
      IgnorePointersGuard ignore_pointers(*this);
      if (cast->getSubExpr()->getStmtClass() == clang::Stmt::CXXThisExprClass) {
        ignore_pointers.enable();
      }

      XLS_ASSIGN_OR_RETURN(CValue sub,
                           GenerateIR_Expr(cast->getSubExpr(), loc));

      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> to_type,
                           TranslateTypeFromClang(cast->getType(), loc));

      if (to_type->Is<CVoidType>()) {
        return CValue(xls::BValue(), to_type);
      }

      // Sometimes array types are converted to pointer types via ImplicitCast,
      // even nested as in mutable array -> mutable pointer -> const pointer.
      // Since we don't generally support pointers (except for array slicing),
      // this case is short-circuited, and the nested expression is evaluated
      // directly, ignoring the casts.
      {
        // Ignore nested ImplicitCastExprs. This case breaks the logic below.
        auto nested_implicit = cast;

        while (nested_implicit->getSubExpr()->getStmtClass() ==
               clang::Stmt::ImplicitCastExprClass) {
          nested_implicit = clang_down_cast<const clang::CastExpr*>(
              nested_implicit->getSubExpr());
        }

        auto from_arr_type = std::dynamic_pointer_cast<CArrayType>(sub.type());

        // Avoid decay of array to pointer, pointers are unsupported
        if (from_arr_type && nested_implicit->getType()->isPointerType()) {
          XLS_ASSIGN_OR_RETURN(
              CValue sub, GenerateIR_Expr(nested_implicit->getSubExpr(), loc));

          return sub;
        }
      }

      XLS_ASSIGN_OR_RETURN(ResolvedInheritance inheritance,
                           ResolveInheritance(sub.type(), to_type));

      // Are we casting to a derived class?
      if (inheritance.base_field != nullptr) {
        XLS_CHECK(inheritance.resolved_struct != nullptr);

        xls::BValue val =
            GetStructFieldXLS(sub.rvalue(), inheritance.base_field->index(),
                              *inheritance.resolved_struct, loc);

        return CValue(val, to_type);
      }

      // Pointer conversions
      if (sub.type()->Is<CPointerType>()) {
        if (to_type->Is<CPointerType>()) {
          return sub;
        }
        if (to_type->Is<CArrayType>()) {
          return GenerateIR_Expr(sub.lvalue(), loc);
        }
        return absl::UnimplementedError(
            ErrorMessage(loc, "Don't know how to convert %s to pointer type",
                         std::string(*sub.type())));
      }

      XLS_ASSIGN_OR_RETURN(xls::BValue subc, GenTypeConvert(sub, to_type, loc));

      return CValue(subc, to_type, /*disable_type_check=*/true, sub.lvalue());
    }
    case clang::Stmt::CXXThisExprClass: {
      XLS_ASSIGN_OR_RETURN(const clang::NamedDecl* this_decl, GetThisDecl(loc));
      XLS_ASSIGN_OR_RETURN(CValue this_val, GetIdentifier(this_decl, loc));
      return this_val;
    }
    // ExprWithCleanups preserves some metadata from Clang's parsing process,
    //  which I think is meant to be used for IDE editing tools. It is
    //  irrelevant to XLS[cc].
    case clang::Stmt::ExprWithCleanupsClass: {
      auto cast = clang_down_cast<const clang::ExprWithCleanups*>(expr);
      return GenerateIR_Expr(cast->getSubExpr(), loc);
    }
    // MaterializeTemporaryExpr wraps instantiation of temporary objects
    // We don't support memory management, so this is irrelevant to us.
    case clang::Stmt::MaterializeTemporaryExprClass: {
      auto cast = clang_down_cast<const clang::MaterializeTemporaryExpr*>(expr);
      return GenerateIR_Expr(cast->getSubExpr(), loc);
    }
    // Occurs in the AST for explicit constructor calls. "Foo a = Foo();"
    case clang::Stmt::CXXTemporaryObjectExprClass:
    // Constructor call
    case clang::Stmt::CXXConstructExprClass: {
      auto cast = clang_down_cast<const clang::CXXConstructExpr*>(expr);
      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> octype,
                           TranslateTypeFromClang(cast->getType(), loc));

      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                           ResolveTypeInstance(octype));

      // A struct/class is being constructed
      if (ctype->Is<CStructType>()) {
        XLS_ASSIGN_OR_RETURN(xls::BValue dv, CreateDefaultValue(octype, loc));
        std::vector<const clang::Expr*> args;
        args.reserve(cast->getNumArgs());
        for (int pi = 0; pi < cast->getNumArgs(); ++pi) {
          args.push_back(cast->getArg(pi));
        }
        XLS_ASSIGN_OR_RETURN(CValue ret, GenerateIR_Call(cast->getConstructor(),
                                                         args, &dv, loc));
        XLS_CHECK(ret.type()->Is<CVoidType>());
        return CValue(dv, octype);
      }

      // A built-in type is being constructed. Create default value if there's
      //  no constructor parameter
      if (cast->getNumArgs() == 0) {
        XLS_ASSIGN_OR_RETURN(xls::BValue dv, CreateDefaultValue(octype, loc));
        return CValue(dv, octype);
      } else if (cast->getNumArgs() == 1) {
        XLS_ASSIGN_OR_RETURN(CValue pv, GenerateIR_Expr(cast->getArg(0), loc));
        XLS_ASSIGN_OR_RETURN(xls::BValue converted,
                             GenTypeConvert(pv, octype, loc));
        return CValue(converted, octype);
      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unsupported constructor argument count %i",
                         cast->getNumArgs()));
      }
    }
    case clang::Stmt::ArraySubscriptExprClass: {
      auto* cast = clang_down_cast<const clang::ArraySubscriptExpr*>(expr);
      XLS_ASSIGN_OR_RETURN(CValue arr_val,
                           GenerateIR_Expr(cast->getBase(), loc));
      // Implicit dereference
      if (arr_val.type()->Is<CPointerType>()) {
        XLS_CHECK_NE(arr_val.lvalue(), nullptr);
        XLS_ASSIGN_OR_RETURN(arr_val, GenerateIR_Expr(arr_val.lvalue(), loc));
      }
      auto arr_type = arr_val.type()->As<CArrayType>();
      XLS_ASSIGN_OR_RETURN(CValue idx_val,
                           GenerateIR_Expr(cast->getIdx(), loc));
      return CValue(
          context().fb->ArrayIndex(arr_val.rvalue(), {idx_val.rvalue()}, loc),
          arr_type->GetElementType());
    }
    // Access to a struct member, for example: x.foo
    case clang::Stmt::MemberExprClass: {
      auto* cast = clang_down_cast<const clang::MemberExpr*>(expr);
      return GenerateIR_MemberExpr(cast, loc);
    }
    // Wraps another expression in parenthesis: (sub_expr).
    // This is irrelevant to XLS[cc], as the parenthesis already affected
    //  Clang's AST ordering.
    case clang::Stmt::ParenExprClass: {
      auto* cast = clang_down_cast<const clang::ParenExpr*>(expr);
      return GenerateIR_Expr(cast->getSubExpr(), loc);
    }
    // A reference to a declaration using its identifier
    case clang::Stmt::DeclRefExprClass: {
      const auto* cast = clang_down_cast<const clang::DeclRefExpr*>(expr);
      const clang::NamedDecl* named = cast->getFoundDecl();
      XLS_ASSIGN_OR_RETURN(CValue cval, GetIdentifier(named, loc));
      return cval;
    }
    // Wraps the value of an argument default
    case clang::Stmt::CXXDefaultArgExprClass: {
      auto* arg_expr = clang_down_cast<const clang::CXXDefaultArgExpr*>(expr);
      return GenerateIR_Expr(arg_expr->getExpr(), loc);
    }
    // Wraps certain expressions evaluatable in a constant context
    // I am not sure when exactly Clang chooses to do this.
    case clang::Stmt::ConstantExprClass: {
      auto* const_expr = clang_down_cast<const clang::ConstantExpr*>(expr);
      return GenerateIR_Expr(const_expr->getSubExpr(), loc);
    }
    // This occurs inside of an ArrayInitLoopExpr, and wraps a value
    //  that is created by implication, rather than explicitly.
    case clang::Stmt::OpaqueValueExprClass: {
      auto* const_expr = clang_down_cast<const clang::OpaqueValueExpr*>(expr);
      return GenerateIR_Expr(const_expr->getSourceExpr(), loc);
    }
    // The case in which I've seen Clang generate this is when a struct is
    //  initialized with an array inside.
    // struct ts { tss vv[4]; };
    case clang::Stmt::ArrayInitLoopExprClass: {
      auto* loop_expr = clang_down_cast<const clang::ArrayInitLoopExpr*>(expr);

      XLS_ASSIGN_OR_RETURN(CValue expr,
                           GenerateIR_Expr(loop_expr->getCommonExpr(), loc));

      auto arr_type = std::dynamic_pointer_cast<CArrayType>(expr.type());
      XLS_CHECK(arr_type && (arr_type->GetSize() ==
                             loop_expr->getArraySize().getLimitedValue()));

      return expr;
    }
    // An expression "T()" which creates a value-initialized rvalue of type T,
    // which is a non-class type. For example: return int();
    case clang::Stmt::CXXScalarValueInitExprClass: {
      auto* scalar_init_expr =
          clang_down_cast<const clang::CXXScalarValueInitExpr*>(expr);
      XLS_ASSIGN_OR_RETURN(
          shared_ptr<CType> ctype,
          TranslateTypeFromClang(scalar_init_expr->getType(), loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue def, CreateDefaultValue(ctype, loc));
      return CValue(def, ctype);
    }
    // Implicitly generated value, as in an incomplete initializer list
    case clang::Stmt::ImplicitValueInitExprClass: {
      auto* implicit_value_init_expr =
          clang_down_cast<const clang::ImplicitValueInitExpr*>(expr);
      XLS_ASSIGN_OR_RETURN(
          shared_ptr<CType> ctype,
          TranslateTypeFromClang(implicit_value_init_expr->getType(), loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue def, CreateDefaultValue(ctype, loc));
      return CValue(def, ctype);
    }
    case clang::Stmt::CXXDefaultInitExprClass: {
      auto* default_init_expr =
          clang_down_cast<const clang::CXXDefaultInitExpr*>(expr);
      return GenerateIR_Expr(default_init_expr->getExpr(), loc);
    }
    case clang::Stmt::StringLiteralClass: {
      auto* string_literal_expr =
          clang_down_cast<const clang::StringLiteral*>(expr);
      if (!(string_literal_expr->isOrdinary() ||
            string_literal_expr->isUTF8())) {
        return absl::UnimplementedError(
            "Only 8 bit character strings supported");
      }
      llvm::StringRef strref = string_literal_expr->getString();
      std::string str = strref.str();

      std::shared_ptr<CType> element_type(new CIntType(8, true, true));
      std::shared_ptr<CType> type(new CArrayType(element_type, str.size()));

      std::vector<xls::Value> elements;

      for (char c : str) {
        elements.push_back(xls::Value(xls::SBits(c, 8)));
      }

      XLS_ASSIGN_OR_RETURN(xls::Value arrval, xls::Value::Array(elements));

      return CValue(context().fb->Literal(arrval, loc), type);
    }
    case clang::Stmt::InitListExprClass: {
      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                           TranslateTypeFromClang(expr->getType(), loc));
      XLS_ASSIGN_OR_RETURN(
          xls::BValue init_val,
          CreateInitListValue(
              ctype, clang_down_cast<const clang::InitListExpr*>(expr), loc));
      return CValue(init_val, ctype);
    }
    case clang::Stmt::UnaryExprOrTypeTraitExprClass: {
      auto* unary_or_type_expr =
          clang_down_cast<const clang::UnaryExprOrTypeTraitExpr*>(expr);
      if (unary_or_type_expr->getKind() != clang::UETT_SizeOf) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented UnaryExprOrTypeTraitExpr kind %i",
                         unary_or_type_expr->getKind()));
      }

      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> ret_ctype,
          TranslateTypeFromClang(unary_or_type_expr->getType(), loc));
      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> arg_ctype,
          TranslateTypeFromClang(unary_or_type_expr->getTypeOfArgument(), loc));
      // Remove CInstantiableTypeAliases since CType::BitWidth() cannot resolve
      // them
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> resolved_arg_ctype,
                           ResolveTypeInstanceDeeply(arg_ctype));

      XLS_LOG(WARNING) << ErrorMessage(
          loc, "Warning: sizeof evaluating to size in BITS");

      const int64_t ret_width = ret_ctype->GetBitWidth();
      return CValue(
          context().fb->Literal(
              xls::SBits(resolved_arg_ctype->GetBitWidth(), ret_width), loc),
          std::make_shared<CIntType>(ret_width, true));
    }
    default: {
      expr->dump();
      return absl::UnimplementedError(ErrorMessage(
          loc, "Unimplemented expression %s", expr->getStmtClassName()));
    }
  }
}

absl::Status Translator::MinSizeArraySlices(CValue& true_cv, CValue& false_cv,
                                            std::shared_ptr<CType>& result_type,
                                            const xls::SourceInfo& loc) {
  // Array slices are the size of source arrays, and indices just wrap around.
  // Take the smaller size
  if (true_cv.type()->Is<CArrayType>() && false_cv.type()->Is<CArrayType>() &&
      (*true_cv.type()->As<CArrayType>()->GetElementType() ==
       *false_cv.type()->As<CArrayType>()->GetElementType())) {
    int64_t min_size = std::min(true_cv.type()->As<CArrayType>()->GetSize(),
                                false_cv.type()->As<CArrayType>()->GetSize());
    result_type = std::make_shared<CArrayType>(
        true_cv.type()->As<CArrayType>()->GetElementType(), min_size);
    XLS_CHECK(true_cv.rvalue().valid());
    XLS_CHECK(false_cv.rvalue().valid());
    xls::BValue bval_0 = context().fb->Literal(xls::UBits(0, 32), loc);
    true_cv = CValue(
        context().fb->ArraySlice(true_cv.rvalue(), bval_0, min_size, loc),
        result_type);
    false_cv = CValue(
        context().fb->ArraySlice(false_cv.rvalue(), bval_0, min_size, loc),
        result_type);
  } else {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Select on different lvalue types %s vs %s",
        std::string(*true_cv.type()), std::string(*false_cv.type())));
  }
  return absl::OkStatus();
}

absl::StatusOr<CValue> Translator::GenerateIR_Expr(std::shared_ptr<LValue> expr,
                                                   const xls::SourceInfo& loc) {
  if (!expr->is_select()) {
    return GenerateIR_Expr(expr->leaf(), loc);
  }

  XLS_ASSIGN_OR_RETURN(CValue true_cv,
                       GenerateIR_Expr(expr->lvalue_true(), loc));
  XLS_ASSIGN_OR_RETURN(CValue false_cv,
                       GenerateIR_Expr(expr->lvalue_false(), loc));

  std::shared_ptr<CType> result_type = true_cv.type();

  if (*true_cv.type() != *false_cv.type()) {
    XLS_RETURN_IF_ERROR(
        MinSizeArraySlices(true_cv, false_cv, result_type, loc));
  }

  return Generate_TernaryOp(expr->cond(), true_cv, false_cv, result_type, loc);
}

absl::StatusOr<CValue> Translator::GenerateIR_MemberExpr(
    const clang::MemberExpr* expr, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(CValue leftval, GenerateIR_Expr(expr->getBase(), loc));
  XLS_ASSIGN_OR_RETURN(auto itype, ResolveTypeInstance(leftval.type()));

  auto sitype = std::dynamic_pointer_cast<CStructType>(itype);

  if (sitype == nullptr) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented member access on type %s", string(*itype)));
  }

  // Get the value referred to
  clang::ValueDecl* member = expr->getMemberDecl();

  // VarDecl for static values
  if (member->getKind() == clang::ValueDecl::Kind::Var) {
    XLS_ASSIGN_OR_RETURN(
        CValue val,
        TranslateVarDecl(clang_down_cast<const clang::VarDecl*>(member), loc));
    return val;
  }
  if (member->getKind() == clang::ValueDecl::Kind::EnumConstant) {
    XLS_ASSIGN_OR_RETURN(
        CValue val,
        TranslateEnumConstantDecl(
            clang_down_cast<const clang::EnumConstantDecl*>(member), loc));
    return val;
  }
  if (member->getKind() != clang::ValueDecl::Kind::Field) {
    // Otherwise only FieldDecl is supported. This is the non-static "foo.bar"
    // form.
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unimplemented member expression %s", member->getDeclKindName()));
  }

  auto field = clang_down_cast<clang::FieldDecl*>(member);
  const auto& fields_by_name = sitype->fields_by_name();
  auto found_field =
      // Upcast to NamedDecl because we track unique identifiers
      //  with NamedDecl pointers
      fields_by_name.find(absl::implicit_cast<const clang::NamedDecl*>(field));
  if (found_field == fields_by_name.end()) {
    return absl::NotFoundError(
        ErrorMessage(loc, "Member access on unknown field %s in type %s",
                     field->getNameAsString(), string(*itype)));
  }
  const CField& cfield = *found_field->second;
  // Upcast to NamedDecl because we track unique identifiers
  //  with NamedDecl pointers
  XLS_CHECK_EQ(cfield.name(),
               absl::implicit_cast<const clang::NamedDecl*>(field));
  xls::BValue bval =
      GetStructFieldXLS(leftval.rvalue(), cfield.index(), *sitype, loc);
  return CValue(bval, cfield.type());
}

absl::StatusOr<xls::BValue> Translator::CreateDefaultValue(
    std::shared_ptr<CType> t, const xls::SourceInfo& loc) {
  if (t->Is<CPointerType>()) {
    return xls::BValue();
  }

  XLS_ASSIGN_OR_RETURN(xls::Value value, CreateDefaultRawValue(t, loc));
  return context().fb->Literal(value, loc);
}

absl::StatusOr<xls::Value> Translator::CreateDefaultRawValue(
    std::shared_ptr<CType> t, const xls::SourceInfo& loc) {
  if (t->Is<CIntType>()) {
    return xls::Value(xls::UBits(0, t->As<CIntType>()->width()));
  } else if (t->Is<CBitsType>()) {
    auto it = t->As<CBitsType>();
    return xls::Value(xls::UBits(0, it->GetBitWidth()));
  } else if (t->Is<CBoolType>()) {
    return xls::Value(xls::UBits(0, 1));
  } else if (t->Is<CArrayType>()) {
    auto it = t->As<CArrayType>();
    std::vector<xls::Value> element_vals;
    XLS_ASSIGN_OR_RETURN(xls::Value default_elem_val,
                         CreateDefaultRawValue(it->GetElementType(), loc));
    element_vals.resize(it->GetSize(), default_elem_val);
    return xls::Value::ArrayOrDie(element_vals);
  } else if (t->Is<CStructType>()) {
    auto it = t->As<CStructType>();
    vector<xls::Value> args;
    for (const std::shared_ptr<CField>& field : it->fields()) {
      XLS_ASSIGN_OR_RETURN(xls::Value fval,
                           CreateDefaultRawValue(field->type(), loc));
      args.push_back(fval);
    }
    return MakeStructXLS(args, *it);
  } else if (t->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(auto resolved, ResolveTypeInstance(t));
    return CreateDefaultRawValue(resolved, loc);
  } else if (t->Is<CPointerType>()) {
    const CPointerType* pointer_type = t->As<CPointerType>();
    return CreateDefaultRawValue(pointer_type->GetPointeeType(), loc);
  } else {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Don't know how to make default for type %s", std::string(*t)));
  }
}

absl::StatusOr<xls::BValue> Translator::CreateInitListValue(
    const std::shared_ptr<CType>& t, const clang::InitListExpr* init_list,
    const xls::SourceInfo& loc) {
  if (t->Is<CArrayType>()) {
    auto array_t = t->As<CArrayType>();
    if (array_t->GetSize() < init_list->getNumInits()) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Wrong number of initializers"));
    }
    XLS_ASSIGN_OR_RETURN(Pragma pragma,
                         FindPragmaForLoc(init_list->getBeginLoc()));
    if (pragma.type() != Pragma_ArrayAllowDefaultPad &&
        array_t->GetSize() != init_list->getNumInits() &&
        init_list->getNumInits() != 1) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Wrong number of initializers"));
    }
    std::vector<xls::BValue> element_vals;
    for (int i = 0; i < array_t->GetSize(); ++i) {
      const clang::Expr* this_init;
      if (i < init_list->getNumInits()) {
        this_init = init_list->getInit(i);
      } else {
        this_init = init_list->getArrayFiller();
      }
      xls::BValue this_val;
      if (this_init->getStmtClass() == clang::Stmt::InitListExprClass) {
        XLS_ASSIGN_OR_RETURN(
            this_val,
            CreateInitListValue(
                array_t->GetElementType(),
                clang_down_cast<const clang::InitListExpr*>(this_init), loc));
      } else {
        XLS_ASSIGN_OR_RETURN(CValue expr_val, GenerateIR_Expr(this_init, loc));
        if (*expr_val.type() != *array_t->GetElementType()) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "Wrong initializer type %s", string(*expr_val.type())));
        }
        this_val = expr_val.rvalue();
      }
      if (init_list->getNumInits() == 1 &&
          array_t->GetSize() != init_list->getNumInits() &&
          !EvaluateBVal(this_val, loc)->IsAllZeros()) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Non-zero initializers must have exact element count"));
      }
      element_vals.push_back(this_val);
    }
    XLS_ASSIGN_OR_RETURN(xls::Type * xls_elem_type,
                         TranslateTypeToXLS(array_t->GetElementType(), loc));
    return context().fb->Array(element_vals, xls_elem_type, loc);
  }
  if (t->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> struct_type,
                         ResolveTypeInstance(t));
    auto struct_type_ptr = struct_type->As<CStructType>();
    XLS_CHECK_NE(nullptr, struct_type_ptr);
    XLS_CHECK_EQ(struct_type_ptr->fields().size(), init_list->getNumInits());
    std::vector<xls::BValue> field_vals;
    for (uint64_t i = 0; i < init_list->getNumInits(); ++i) {
      XLS_ASSIGN_OR_RETURN(CValue value,
                           GenerateIR_Expr(init_list->getInit(i), loc));
      XLS_CHECK(*value.type() == *struct_type_ptr->fields().at(i)->type());
      field_vals.push_back(value.rvalue());
    }
    return MakeStructXLS(field_vals, *struct_type_ptr, loc);
  } else {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Don't know how to interpret initializer list for type %s",
        string(*t)));
  }
}

absl::StatusOr<xls::Value> Translator::EvaluateNode(
    xls::Node* node, const xls::SourceInfo& loc) {
  xls::IrInterpreter visitor({});
  absl::Status status = node->Accept(&visitor);
  if (!status.ok()) {
    return absl::UnavailableError(
        ErrorMessage(loc,
                     "Couldn't evaluate node as a constant. Error from IR "
                     "interpreter was: %s",
                     status.message()));
  }
  xls::Value result = visitor.ResolveAsValue(node);
  return result;
}

absl::Status Translator::ShortCircuitBVal(xls::BValue& bval,
                                          const xls::SourceInfo& loc) {
  absl::flat_hash_set<xls::Node*> visited;
  return ShortCircuitNode(bval.node(), bval, nullptr, visited, loc);
}

absl::Status Translator::ShortCircuitNode(
    xls::Node* node, xls::BValue& top_bval, xls::Node* parent,
    absl::flat_hash_set<xls::Node*>& visited, const xls::SourceInfo& loc) {
  if (visited.contains(node)) {
    return absl::OkStatus();
  }

  visited.insert(node);

  // Depth-first to allow multi-step short circuits
  // Index based to avoid modify while iterating
  for (int oi = 0; oi < node->operand_count(); ++oi) {
    xls::Node* op = node->operand(oi);
    XLS_RETURN_IF_ERROR(ShortCircuitNode(op, top_bval, node, visited, loc));
  }

  // Don't duplicate literals
  if (node->Is<xls::Literal>()) {
    return absl::OkStatus();
  }

  absl::StatusOr<xls::Value> const_result = EvaluateNode(node, loc);

  // Try to replace the node with a literal
  if (const_result.ok()) {
    xls::BValue literal_bval =
        context().fb->Literal(const_result.value(), node->loc());

    if (parent != nullptr) {
      XLS_CHECK(parent->ReplaceOperand(node, literal_bval.node()));
    } else {
      top_bval = literal_bval;
    }
    return absl::OkStatus();
  }

  if (!((node->op() == xls::Op::kAnd) || (node->op() == xls::Op::kOr))) {
    return absl::OkStatus();
  }

  for (xls::Node* op : node->operands()) {
    // Operands that can be evaluated will already have been turned into
    // literals by the above depth-first literalization
    if (!op->Is<xls::Literal>()) {
      continue;
    }
    xls::Literal* literal_node = op->As<xls::Literal>();

    const xls::Value& const_value = literal_node->value();

    if ((node->op() == xls::Op::kAnd) && (!const_value.IsAllZeros())) {
      continue;
    }
    if ((node->op() == xls::Op::kOr) && (!const_value.IsAllOnes())) {
      continue;
    }

    // Replace the node with its literal operand
    if (parent != nullptr) {
      XLS_CHECK(parent->ReplaceOperand(node, op));
    } else {
      top_bval = xls::BValue(op, context().fb);
    }

    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::StatusOr<xls::Value> Translator::EvaluateBVal(
    xls::BValue bval, const xls::SourceInfo& loc) {
  return EvaluateNode(bval.node(), loc);
}

absl::StatusOr<ConstValue> Translator::TranslateBValToConstVal(
    const CValue& bvalue, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(xls::Value const_value,
                       EvaluateBVal(bvalue.rvalue(), loc));
  return ConstValue(const_value, bvalue.type());
}

absl::Status Translator::GenerateIR_Compound(const clang::Stmt* body,
                                             clang::ASTContext& ctx) {
  if (body == nullptr) {
    // Empty block, nothing to do
    return absl::OkStatus();
  }

  if (body->getStmtClass() == clang::Stmt::CompoundStmtClass) {
    for (const clang::Stmt* body_st : body->children()) {
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(body_st, ctx));
    }
  } else {
    // For single-line bodies
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(body, ctx));
  }

  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_StaticDecl(const clang::VarDecl* vard,
                                               const clang::NamedDecl* namedecl,
                                               const xls::SourceInfo& loc) {
  bool use_on_reset = false;
  bool any_side_effects = false;
  ConstValue init;
  CValue translated_without_side_effects;

  {
    PushContextGuard guard(*this, loc);
    context().mask_side_effects = true;
    context().any_side_effects_requested = false;

    // Check for side-effects
    XLS_ASSIGN_OR_RETURN(translated_without_side_effects,
                         TranslateVarDecl(vard, loc));

    if (context().any_side_effects_requested) {
      use_on_reset = true;
      any_side_effects = true;
    } else {
      // Check for const-evaluatability
      absl::StatusOr<ConstValue> translate_result =
          TranslateBValToConstVal(translated_without_side_effects, loc);
      if (!translate_result.ok()) {
        use_on_reset = true;
      } else {
        init = translate_result.value();
      }
    }

    if (use_on_reset) {
      XLS_ASSIGN_OR_RETURN(
          xls::Value default_val,
          CreateDefaultRawValue(translated_without_side_effects.type(), loc));
      init = ConstValue(default_val, translated_without_side_effects.type());
    }
  }

  // If there are no side effects and it's const-qualified,
  // then state isn't needed. It can just be a literal.
  if (!use_on_reset && vard->getType().isConstQualified()) {
    XLS_RETURN_IF_ERROR(
        DeclareVariable(namedecl, translated_without_side_effects, loc));
    return absl::OkStatus();
  }

  XLS_RETURN_IF_ERROR(DeclareStatic(namedecl, init, loc));

  if (!use_on_reset) {
    return absl::OkStatus();
  }

  // Select using __xlscc_on_reset
  CValue translated_with_side_effects = translated_without_side_effects;

  // First, if there are side-effects, retranslate with side-effects enabled,
  // conditional on __xlscc_on_reset
  XLS_ASSIGN_OR_RETURN(CValue on_reset_val, GetOnReset(loc));
  XLS_CHECK(on_reset_val.rvalue().valid());
  XLS_CHECK_EQ(on_reset_val.rvalue().BitCountOrDie(), 1);

  PushContextGuard guard(*this, on_reset_val.rvalue(), loc);
  XLS_ASSIGN_OR_RETURN(translated_with_side_effects,
                       TranslateVarDecl(vard, loc));
  XLS_CHECK(translated_with_side_effects.rvalue().valid());

  // This assignment will generate a select on __xlscc_on_reset
  XLS_RETURN_IF_ERROR(Assign(namedecl, translated_with_side_effects, loc));

  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_Stmt(const clang::Stmt* stmt,
                                         clang::ASTContext& ctx) {
  const xls::SourceInfo loc = GetLoc(*stmt);

  // TODO(seanhaskell): A cleaner way to check if it's any kind of Expr?
  if (absl::StrContains(stmt->getStmtClassName(), "Literal") ||
      absl::StrContains(stmt->getStmtClassName(), "Expr") ||
      absl::StrContains(stmt->getStmtClassName(), "Operator")) {
    XLS_ASSIGN_OR_RETURN(
        absl::StatusOr<CValue> rv,
        GenerateIR_Expr(clang_down_cast<const clang::Expr*>(stmt), loc));
    return rv.status();
  }
  switch (stmt->getStmtClass()) {
    case clang::Stmt::ReturnStmtClass: {
      if (context().in_pipelined_for_body) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Returns in pipelined loop body unimplemented"));
      }
      auto rts = clang_down_cast<const clang::ReturnStmt*>(stmt);
      const clang::Expr* rvalue = rts->getRetValue();
      if (rvalue != nullptr) {
        XLS_ASSIGN_OR_RETURN(CValue rv, GenerateIR_Expr(rvalue, loc));
        XLS_ASSIGN_OR_RETURN(xls::BValue crv,
                             GenTypeConvert(rv, context().return_type, loc));

        if (!context().return_val.valid()) {
          context().return_val = crv;
        } else {
          // This is the normal case, where the last return was conditional
          if (context().last_return_condition.valid()) {
            // If there are multiple returns with the same condition, this will
            // take the first one
            xls::BValue this_cond =
                context().full_condition.valid()
                    ? context().full_condition
                    : context().fb->Literal(xls::UBits(1, 1));

            // Select the previous return instead of this one if
            //  the last return condition is true or this one is false
            // Scenario A (sel_prev_ret_cond = true):
            //  if(something true) return last;  // take this return value
            //  if(something true) return this;
            // Scenario B (sel_prev_ret_cond = false):
            //  if(something false) return last;
            //  if(something true) return this;  // take this return value
            // Scenario C  (sel_prev_ret_cond = true):
            //  if(something true) return last;  // take this return value
            //  if(something false) return this;
            // Scenario D  (sel_prev_ret_cond = false):
            //  if(something false) return last;
            //  if(something false) return this;
            //  return later_on;                 // take this return value
            // Scenario E  (sel_prev_ret_cond = true):
            //  return earnier_on;               // take this return value
            //  if(something true) return last;
            xls::BValue sel_prev_ret_cond = context().fb->Or(
                context().last_return_condition, context().fb->Not(this_cond));
            context().return_val = context().fb->Select(
                sel_prev_ret_cond, context().return_val, crv, loc);
          } else {
            // In the case of multiple unconditional returns, take the first one
            // (no-op)
          }
        }

        if (context().full_condition.valid()) {
          context().last_return_condition = context().full_condition;
        } else {
          context().last_return_condition =
              context().fb->Literal(xls::UBits(1, 1));
        }
      }

      xls::BValue reach_here_cond = context().full_condition_bval(loc);

      if (!context().have_returned_condition.valid()) {
        context().have_returned_condition = reach_here_cond;
      } else {
        context().have_returned_condition = context().fb->Or(
            reach_here_cond, context().have_returned_condition);
      }
      XLS_RETURN_IF_ERROR(and_condition(
          context().fb->Not(context().have_returned_condition, loc), loc));
      break;
    }
    case clang::Stmt::DeclStmtClass: {
      auto declstmt = clang_down_cast<const clang::DeclStmt*>(stmt);

      for (auto decl : declstmt->decls()) {
        if (decl->getKind() == clang::Decl::Kind::Typedef) break;
        if (decl->getKind() == clang::Decl::Kind::StaticAssert) break;
        if (decl->getKind() == clang::Decl::Kind::Enum) break;

        if (decl->getKind() != clang::Decl::Kind::Var) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "DeclStmt other than Var (%s)", decl->getDeclKindName()));
        }

        auto namedecl = clang_down_cast<const clang::NamedDecl*>(decl);
        auto vard = clang_down_cast<const clang::VarDecl*>(decl);

        if (vard->isStaticLocal() || vard->isStaticDataMember()) {
          XLS_RETURN_IF_ERROR(GenerateIR_StaticDecl(vard, namedecl, loc));
        } else {
          XLS_ASSIGN_OR_RETURN(CValue translated, TranslateVarDecl(vard, loc));
          XLS_RETURN_IF_ERROR(DeclareVariable(namedecl, translated, loc));
        }
      }
      break;
    }
    case clang::Stmt::GCCAsmStmtClass: {
      const auto* pasm = clang_down_cast<const clang::GCCAsmStmt*>(stmt);
      std::string sasm = pasm->getAsmString()->getString().str();

      vector<xls::BValue> args;

      for (int i = 0; i < pasm->getNumInputs(); ++i) {
        const clang::Expr* expr = pasm->getInputExpr(i);
        if (expr->isIntegerConstantExpr(ctx)) {
          const std::string name = pasm->getInputConstraint(i).str();
          XLS_ASSIGN_OR_RETURN(auto val, EvaluateInt64(*expr, ctx, loc));
          sasm = std::regex_replace(
              sasm, std::regex(absl::StrFormat(R"(\b%s\b)", name)),
              absl::StrCat(val));
        } else {
          XLS_ASSIGN_OR_RETURN(CValue ret, GenerateIR_Expr(expr, loc));
          args.emplace_back(ret.rvalue());
        }
      }

      // Unique function name
      RE2::GlobalReplace(&sasm, "\\(fid\\)",
                         absl::StrFormat("fid%i", next_asm_number_++));
      // Unique IR instruction name
      RE2::GlobalReplace(&sasm, "\\(aid\\)",
                         absl::StrFormat("aid%i", next_asm_number_++));
      // File location
      RE2::GlobalReplace(&sasm, "\\(loc\\)", loc.ToString());

      if (pasm->getNumOutputs() != 1) {
        return absl::UnimplementedError(
            absl::StrFormat("asm must have exactly 1 output"));
      }

      XLS_ASSIGN_OR_RETURN(CValue out_val,
                           GenerateIR_Expr(pasm->getOutputExpr(0), loc));

      // verify_function_only because external channels are defined up-front,
      //  which generates "No receive/send node" errors
      XLS_ASSIGN_OR_RETURN(
          xls::Function * af,
          xls::Parser::ParseFunction(sasm, package_,
                                     /*verify_function_only=*/true));

      // No type conversion in or out: inline IR can do whatever it wants.
      // If you use inline IR, you should know exactly what you're doing.
      xls::BValue fret = context().fb->Invoke(args, af, loc);

      XLS_RETURN_IF_ERROR(
          Assign(pasm->getOutputExpr(0), CValue(fret, out_val.type()), loc));

      break;
    }
    case clang::Stmt::IfStmtClass: {
      const auto* ifst = clang_down_cast<const clang::IfStmt*>(stmt);
      XLS_ASSIGN_OR_RETURN(CValue cond, GenerateIR_Expr(ifst->getCond(), loc));
      XLS_CHECK(cond.type()->Is<CBoolType>());
      if (ifst->getInit()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unimplemented C++17 if initializers"));
      }
      if (ifst->getThen()) {
        PushContextGuard context_guard(*this, cond.rvalue(), loc);
        XLS_RETURN_IF_ERROR(GenerateIR_Compound(ifst->getThen(), ctx));
      }
      if (ifst->getElse()) {
        PushContextGuard context_guard(*this, context().fb->Not(cond.rvalue()),
                                       loc);
        XLS_RETURN_IF_ERROR(GenerateIR_Compound(ifst->getElse(), ctx));
      }

      break;
    }
    case clang::Stmt::ForStmtClass: {
      auto forst = clang_down_cast<const clang::ForStmt*>(stmt);
      XLS_RETURN_IF_ERROR(GenerateIR_Loop(
          /*always_first_iter=*/false, forst->getInit(), forst->getCond(),
          forst->getInc(), forst->getBody(), GetPresumedLoc(*forst), loc, ctx));
      break;
    }
    case clang::Stmt::WhileStmtClass: {
      auto forst = clang_down_cast<const clang::WhileStmt*>(stmt);
      XLS_RETURN_IF_ERROR(GenerateIR_Loop(/*always_first_iter=*/false,
                                          /*init=*/nullptr, forst->getCond(),
                                          /*inc=*/nullptr, forst->getBody(),
                                          GetPresumedLoc(*forst), loc, ctx));
      break;
    }
    case clang::Stmt::DoStmtClass: {
      auto dost = clang_down_cast<const clang::DoStmt*>(stmt);
      XLS_RETURN_IF_ERROR(GenerateIR_Loop(/*always_first_iter=*/true,
                                          /*init=*/nullptr, dost->getCond(),
                                          /*inc=*/nullptr, dost->getBody(),
                                          GetPresumedLoc(*dost), loc, ctx));
      break;
    }
    case clang::Stmt::SwitchStmtClass: {
      auto switchst = clang_down_cast<const clang::SwitchStmt*>(stmt);
      return GenerateIR_Switch(switchst, ctx, loc);
    }
    case clang::Stmt::ContinueStmtClass: {
      // Continue should be used inside of for loop bodies only
      XLS_CHECK(context().in_for_body);
      context().relative_continue_condition =
          context().relative_condition_bval(loc);
      // Make the rest of the block no-op
      XLS_RETURN_IF_ERROR(
          and_condition(context().fb->Literal(xls::UBits(0, 1), loc), loc));
      break;
    }
    case clang::Stmt::BreakStmtClass: {
      if (context().in_for_body) {
        // We are in a for body
        XLS_CHECK(!context().in_switch_body);
        context().relative_break_condition =
            context().relative_condition_bval(loc);

        // Make the rest of the block no-op
        XLS_RETURN_IF_ERROR(
            and_condition(context().fb->Literal(xls::UBits(0, 1), loc), loc));
      } else {
        // We are in a switch body
        XLS_CHECK(context().in_switch_body);
        // We use the original condition because we only care about
        //  enclosing conditions, such as if(...) { break; }
        //  Not if(...) {return;} break;
        if (context().full_condition_on_enter_block.node() !=
            context().full_switch_cond.node()) {
          return absl::UnimplementedError(
              ErrorMessage(loc, "Conditional breaks are not supported"));
        }
        context().hit_break = true;
      }
      break;
    }
    case clang::Stmt::CompoundStmtClass: {
      PushContextGuard context_guard(*this, loc);
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(stmt, ctx));
      break;
    }
    // Empty line (just ;)
    case clang::Stmt::NullStmtClass: {
      break;
    }
    // Just ignore labels for now
    case clang::Stmt::LabelStmtClass: {
      auto label_stmt = clang_down_cast<const clang::LabelStmt*>(stmt);
      return GenerateIR_Stmt(label_stmt->getSubStmt(), ctx);
    }
    default:
      stmt->dump();
      return absl::UnimplementedError(ErrorMessage(
          loc, "Unimplemented construct %s", stmt->getStmtClassName()));
  }
  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_Loop(
    bool always_first_iter, const clang::Stmt* init,
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, const clang::PresumedLoc& presumed_loc,
    const xls::SourceInfo& loc, clang::ASTContext& ctx) {
  if (cond_expr != nullptr && cond_expr->isIntegerConstantExpr(ctx)) {
    // special case for "for (;0;) {}" (essentially no op)
    XLS_ASSIGN_OR_RETURN(auto constVal, EvaluateInt64(*cond_expr, ctx, loc));
    if (constVal == 0) {
      return absl::OkStatus();
    }
  }
  XLS_ASSIGN_OR_RETURN(Pragma pragma, FindPragmaForLoc(presumed_loc));
  if (pragma.type() == Pragma_Unroll || context().for_loops_default_unroll) {
    return GenerateIR_UnrolledLoop(always_first_iter, init, cond_expr, inc,
                                   body, ctx, loc);
  }
  // Pipelined loops can inherit their initiation interval from enclosing
  // loops, so they can be allowed not to have a #pragma.
  int init_interval = pragma.int_argument();
  // Pragma might not be null, because labels get searched backwards
  if (pragma.type() != Pragma_InitInterval) {
    XLS_CHECK(!context().in_pipelined_for_body ||
              (context().outer_pipelined_loop_init_interval > 0));
    init_interval = context().outer_pipelined_loop_init_interval;
  }
  if (init_interval <= 0) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "For loop missing #pragma"));
  }

  // Pipelined do-while
  return GenerateIR_PipelinedLoop(always_first_iter, init, cond_expr, inc, body,
                                  init_interval, ctx, loc);
}

int Debug_CountNodes(const xls::Node* node,
                     std::set<const xls::Node*>& visited) {
  if (visited.find(node) != visited.end()) {
    return 0;
  }
  visited.insert(node);

  int ret = 1;
  for (const xls::Node* child : node->operands()) {
    ret += Debug_CountNodes(child, visited);
  }
  return ret;
}

std::string Debug_NodeToInfix(xls::BValue bval) {
  if (bval.node() == nullptr) {
    return "[null]";
  }
  int64_t n_printed = 0;
  return Debug_NodeToInfix(bval.node(), n_printed);
}

std::string Debug_NodeToInfix(const xls::Node* node, int64_t& n_printed) {
  ++n_printed;
  if (n_printed > 100) {
    return "[...]";
  }

  if (node->Is<xls::Literal>()) {
    const xls::Literal* literal = node->As<xls::Literal>();
    if (literal->value().kind() == xls::ValueKind::kBits) {
      return absl::StrFormat("%li", literal->value().bits().ToInt64().value());
    }
  }
  if (node->Is<xls::Param>()) {
    const xls::Param* param = node->As<xls::Param>();
    return param->GetName();
  }
  if (node->Is<xls::UnOp>()) {
    const xls::UnOp* op = node->As<xls::UnOp>();
    if (op->op() == xls::Op::kNot) {
      return absl::StrFormat("!%s",
                             Debug_NodeToInfix(op->operand(0), n_printed));
    }
  }
  if (node->op() == xls::Op::kSGt) {
    return absl::StrFormat("(%s > %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kSLt) {
    return absl::StrFormat("(%s < %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kSLe) {
    return absl::StrFormat("(%s <= %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kEq) {
    return absl::StrFormat("(%s == %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kAnd) {
    return absl::StrFormat("(%s & %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kOr) {
    return absl::StrFormat("(%s | %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kAdd) {
    return absl::StrFormat("(%s + %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }
  if (node->op() == xls::Op::kSignExt) {
    return absl::StrFormat("%s",
                           Debug_NodeToInfix(node->operand(0), n_printed));
  }
  if (node->op() == xls::Op::kSel) {
    return absl::StrFormat("(%s ? %s : %s)",
                           Debug_NodeToInfix(node->operand(0), n_printed),
                           Debug_NodeToInfix(node->operand(2), n_printed),
                           Debug_NodeToInfix(node->operand(1), n_printed));
  }

  return absl::StrFormat("[unsupported %s / %s]", node->GetName(),
                         typeid(*node).name());
}

absl::StatusOr<Z3_lbool> Translator::IsBitSatisfiable(
    xls::Node* node, Z3_solver& solver,
    xls::solvers::z3::IrTranslator& z3_translator) {
  XLS_CHECK_EQ(node->BitCountOrDie(), 1);

  Z3_context ctx = z3_translator.ctx();
  xls::solvers::z3::ScopedErrorHandler seh(ctx);

  Z3_ast z3_node = z3_translator.GetTranslation(node);

  Z3_ast asserted = xls::solvers::z3::BitVectorToBoolean(ctx, z3_node);
  Z3_lbool satisfiable = Z3_solver_check_assumptions(ctx, solver, 1, &asserted);

  if (seh.status().ok()) {
    return satisfiable;
  }
  return seh.status();
}

absl::Status Translator::GenerateIR_UnrolledLoop(bool always_first_iter,
                                                 const clang::Stmt* init,
                                                 const clang::Expr* cond_expr,
                                                 const clang::Stmt* inc,
                                                 const clang::Stmt* body,
                                                 clang::ASTContext& ctx,
                                                 const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<xls::solvers::z3::IrTranslator> z3_translator_parent,
      xls::solvers::z3::IrTranslator::CreateAndTranslate(
          /*source=*/nullptr,
          /*allow_unsupported=*/true));

  Z3_solver solver =
      xls::solvers::z3::CreateSolver(z3_translator_parent->ctx(), 1);

  class SolverDeref {
   public:
    SolverDeref(Z3_context ctx, Z3_solver solver)
        : ctx_(ctx), solver_(solver) {}
    ~SolverDeref() { Z3_solver_dec_ref(ctx_, solver_); }

   private:
    Z3_context ctx_;
    Z3_solver solver_;
  };

  // Generate the declaration within a private context
  PushContextGuard for_init_guard(*this, loc);
  context().propagate_break_up = false;
  context().propagate_continue_up = false;
  context().in_for_body = true;
  context().in_switch_body = false;

  if (init != nullptr) {
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(init, ctx));
  }

  // Loop unrolling causes duplicate NamedDecls which fail the soundness
  // check. Reset the known set before each iteration.
  auto saved_check_ids = check_unique_ids_;

  double slowest_iter = 0;

  for (int64_t nIters = 0;; ++nIters) {
    const bool first_iter = nIters == 0;
    const bool always_this_iter = always_first_iter && first_iter;

    const double iter_start = doubletime();

    check_unique_ids_ = saved_check_ids;

    if (nIters >= max_unroll_iters_) {
      return absl::ResourceExhaustedError(
          ErrorMessage(loc, "Loop unrolling broke at maximum %i iterations",
                       max_unroll_iters_));
    }
    if (nIters == warn_unroll_iters_) {
      XLS_LOG(WARNING) << ErrorMessage(
          loc, "Loop unrolling has reached %i iterations", warn_unroll_iters_);
    }

    // Generate condition.
    //
    // Outside of body context guard so it applies to increment
    // Also, if this is inside the body context guard then the break condition
    // feeds back on itself in an explosion of complexity
    // via assignments to any variables used in the condition.
    if (!always_this_iter && cond_expr != nullptr) {
      XLS_ASSIGN_OR_RETURN(CValue cond_expr_cval,
                           GenerateIR_Expr(cond_expr, loc));
      XLS_CHECK(cond_expr_cval.type()->Is<CBoolType>());
      context().or_condition_util(
          context().fb->Not(cond_expr_cval.rvalue(), loc),
          context().relative_break_condition, loc);
      XLS_RETURN_IF_ERROR(and_condition(cond_expr_cval.rvalue(), loc));
    }

    // Generate body
    {
      PushContextGuard for_body_guard(*this, loc);
      context().propagate_break_up = true;
      context().propagate_continue_up = false;

      // Check condition first
      if (context().relative_break_condition.valid() && !always_this_iter) {
        // Simplify break logic in easy ways;
        // Z3 fails to solve some cases without this.
        XLS_RETURN_IF_ERROR(
            ShortCircuitBVal(context().relative_break_condition, loc));

        // Use Z3 to check if another loop iteration is possible.
        xls::BValue not_break =
            context().fb->Not(context().relative_break_condition);

        XLS_ASSIGN_OR_RETURN(
            std::unique_ptr<xls::solvers::z3::IrTranslator> z3_translator,
            xls::solvers::z3::IrTranslator::CreateAndTranslate(
                /*ctx=*/z3_translator_parent->ctx(),
                /*source=*/not_break.node(),
                /*allow_unsupported=*/true));

        XLS_ASSIGN_OR_RETURN(
            Z3_lbool result,
            IsBitSatisfiable(not_break.node(), solver, *z3_translator));

        // No combination of variables can satisfy !break condition.
        if (result == Z3_L_FALSE) {
          break;
        }
      }

      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, ctx));
    }

    // Generate increment
    // Outside of body guard because continue would skip.
    if (inc != nullptr) {
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(inc, ctx));
    }
    // Print slow unrolling warning
    const double iter_end = doubletime();
    const double iter_seconds = iter_end - iter_start;

    if (iter_seconds > 0.1 && iter_seconds > slowest_iter) {
      XLS_LOG(WARNING) << ErrorMessage(
          loc, "Slow loop unrolling iteration %i: %fms", nIters, iter_seconds);
      slowest_iter = iter_seconds;
    }
  }

  return absl::OkStatus();
}

void GeneratedFunction::SortNamesDeterministically(
    std::vector<const clang::NamedDecl*>& names) const {
  std::sort(names.begin(), names.end(),
            [this](const clang::NamedDecl* a, const clang::NamedDecl* b) {
              return declaration_order_by_name_.at(a) <
                     declaration_order_by_name_.at(b);
            });
}

std::vector<const clang::NamedDecl*>
GeneratedFunction::GetDeterministicallyOrderedStaticValues() const {
  std::vector<const clang::NamedDecl*> ret;
  for (const auto& [decl, _] : static_values) {
    ret.push_back(decl);
  }
  SortNamesDeterministically(ret);
  return ret;
}

absl::Status Translator::CheckInitIntervalValidity(int initiation_interval_arg,
                                                   const xls::SourceInfo& loc) {
  if (initiation_interval_arg != 1) {
    std::string message = ErrorMessage(
        loc,
        "Only initiation interval 1 supported, %i requested, defaulting to 1",
        initiation_interval_arg);
    if (error_on_init_interval_) {
      return absl::UnimplementedError(message);
    }
    XLS_LOG(WARNING) << message;
  }
  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_PipelinedLoop(
    bool always_first_iter, const clang::Stmt* init,
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, int64_t initiation_interval_arg,
    clang::ASTContext& ctx, const xls::SourceInfo& loc) {
  XLS_RETURN_IF_ERROR(CheckInitIntervalValidity(initiation_interval_arg, loc));

  // Generate the loop counter declaration within a private context
  // By doing this here, it automatically gets rolled into proc state
  // This causes it to be automatically reset on break
  PushContextGuard for_init_guard(*this, loc);

  if (init != nullptr) {
    XLS_RETURN_IF_ERROR(GenerateIR_Stmt(init, ctx));
  }

  // Condition must be checked at the start
  if (!always_first_iter && cond_expr != nullptr) {
    XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_expr, loc));
    XLS_CHECK(cond_cval.type()->Is<CBoolType>());

    XLS_RETURN_IF_ERROR(and_condition(cond_cval.rvalue(), loc));
  }

  // Pack context tuple
  CValue context_tuple_out;
  std::shared_ptr<CStructType> context_struct_type;
  absl::flat_hash_map<const clang::NamedDecl*, uint64_t> variable_field_indices;
  std::vector<const clang::NamedDecl*> variable_fields_order;
  {
    std::vector<std::shared_ptr<CField>> fields;
    std::vector<xls::BValue> tuple_values;

    // Create a deterministic field order
    for (const auto& [decl, _] : context().variables) {
      XLS_CHECK(context().sf->declaration_order_by_name_.contains(decl));
      variable_fields_order.push_back(decl);
    }

    context().sf->SortNamesDeterministically(variable_fields_order);

    for (const clang::NamedDecl* decl : variable_fields_order) {
      const CValue& cvalue = context().variables.at(decl);
      XLS_CHECK(cvalue.rvalue().valid());
      const uint64_t field_idx = tuple_values.size();
      variable_field_indices[decl] = field_idx;
      tuple_values.push_back(cvalue.rvalue());
      auto field_ptr = std::make_shared<CField>(decl, field_idx, cvalue.type());
      fields.push_back(field_ptr);
    }

    context_struct_type = std::make_shared<CStructType>(fields, false);
    context_tuple_out =
        CValue(MakeStructXLS(tuple_values, *context_struct_type, loc),
               context_struct_type);
  }

  // Create synthetic channels and IO ops
  xls::Type* context_xls_type = context_tuple_out.rvalue().GetType();

  const std::string name_prefix =
      absl::StrFormat("__for_%i", next_for_number_++);

  IOChannel* context_out_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_out", name_prefix);
    XLS_ASSIGN_OR_RETURN(
        xls::Channel * xls_channel,
        package_->CreateStreamingChannel(
            ch_name, xls::ChannelOps::kSendReceive, context_xls_type,
            /*initial_values=*/{}, /*fifo_depth=*/0,
            xls::FlowControl::kReadyValid));
    IOChannel new_channel;
    new_channel.item_type = context_tuple_out.type();
    new_channel.unique_name = ch_name;
    new_channel.channel_op_type = OpType::kSend;
    new_channel.generated = xls_channel;
    context().sf->io_channels.push_back(new_channel);
    context_out_channel = &context().sf->io_channels.back();
  }
  IOChannel* context_in_channel = nullptr;
  {
    std::string ch_name = absl::StrFormat("%s_ctx_in", name_prefix);
    XLS_ASSIGN_OR_RETURN(
        xls::Channel * xls_channel,
        package_->CreateStreamingChannel(
            ch_name, xls::ChannelOps::kSendReceive, context_xls_type,
            /*initial_values=*/{}, /*fifo_depth=*/0,
            xls::FlowControl::kReadyValid));
    IOChannel new_channel;
    new_channel.item_type = context_tuple_out.type();
    new_channel.unique_name = ch_name;
    new_channel.channel_op_type = OpType::kRecv;
    new_channel.generated = xls_channel;
    context().sf->io_channels.push_back(new_channel);
    context_in_channel = &context().sf->io_channels.back();
  }

  {
    IOOp op;
    op.op = OpType::kSend;
    std::vector<xls::BValue> sp = {context_tuple_out.rvalue(),
                                   context().full_condition_bval(loc)};
    op.ret_value = context().fb->Tuple(sp, loc);
    XLS_RETURN_IF_ERROR(AddOpToChannel(op, context_out_channel, loc).status());
  }

  IOOp* ctx_in_op_ptr;
  {
    IOOp op;
    op.op = OpType::kRecv;
    op.ret_value = context().full_condition_bval(loc);
    XLS_ASSIGN_OR_RETURN(ctx_in_op_ptr,
                         AddOpToChannel(op, context_in_channel, loc));
  }

  // Create loop body proc
  std::vector<const clang::NamedDecl*> vars_changed_in_body;
  XLS_RETURN_IF_ERROR(GenerateIR_PipelinedLoopBody(
      cond_expr, inc, body, initiation_interval_arg, ctx, name_prefix,
      context_out_channel, context_in_channel, context_xls_type,
      context_struct_type, variable_field_indices, variable_fields_order,
      vars_changed_in_body, loc));

  // Unpack context tuple
  xls::BValue context_tuple_recvd = ctx_in_op_ptr->input_value.rvalue();
  {
    // Don't assign to variables that aren't changed in the loop body,
    // as this creates extra state
    for (const clang::NamedDecl* decl : vars_changed_in_body) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      const CValue cval(GetStructFieldXLS(context_tuple_recvd, field_idx,
                                          *context_struct_type, loc),
                        context().variables.at(decl).type());
      XLS_RETURN_IF_ERROR(Assign(decl, cval, loc));
    }
  }

  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_PipelinedLoopBody(
    const clang::Expr* cond_expr, const clang::Stmt* inc,
    const clang::Stmt* body, int64_t init_interval, clang::ASTContext& ctx,
    std::string_view name_prefix, IOChannel* context_out_channel,
    IOChannel* context_in_channel, xls::Type* context_xls_type,
    std::shared_ptr<CStructType> context_ctype,
    const absl::flat_hash_map<const clang::NamedDecl*, uint64_t>&
        variable_field_indices,
    const std::vector<const clang::NamedDecl*>& variable_fields_order,
    std::vector<const clang::NamedDecl*>& vars_changed_in_body,
    const xls::SourceInfo& loc) {
  const uint64_t total_context_values = context_ctype->fields().size();

  std::vector<const clang::NamedDecl*> vars_to_save_between_iters;

  // Generate body function
  GeneratedFunction generated_func;
  XLS_CHECK_NE(context().sf, nullptr);
  XLS_CHECK_NE(context().sf->clang_decl, nullptr);
  generated_func.clang_decl = context().sf->clang_decl;
  uint64_t extra_return_count = 0;
  {
    // Inherit explicit channels
    GeneratedFunction& enclosing_func = *context().sf;
    for (IOChannel& enclosing_channel : enclosing_func.io_channels) {
      if (enclosing_channel.generated != nullptr) {
        continue;
      }
      const clang::ParmVarDecl* channel_param =
          enclosing_func.params_by_io_channel.at(&enclosing_channel);

      generated_func.io_channels.push_back(enclosing_channel);
      IOChannel* inner_channel = &generated_func.io_channels.back();
      generated_func.io_channels_by_param[channel_param] = inner_channel;
      generated_func.params_by_io_channel[inner_channel] = channel_param;

      inner_channel->total_ops = 0;
    }

    // Set up IR generation
    xls::FunctionBuilder body_builder(absl::StrFormat("%s_func", name_prefix),
                                      package_);

    xls::BValue context_param = body_builder.Param(
        absl::StrFormat("%s_context", name_prefix), context_xls_type, loc);

    TranslationContext& prev_context = context();
    PushContextGuard context_guard(*this, loc);

    context() = TranslationContext();
    context().propagate_up = false;

    context().fb = absl::implicit_cast<xls::BuilderBase*>(&body_builder);
    context().sf = &generated_func;
    context().in_pipelined_for_body = true;
    context().outer_pipelined_loop_init_interval = init_interval;

    // Context in
    absl::flat_hash_map<const clang::NamedDecl*, xls::BValue> prev_vars;

    for (const clang::NamedDecl* decl : variable_fields_order) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      const CValue& outer_value = prev_context.variables.at(decl);
      xls::BValue param_bval =
          GetStructFieldXLS(context_param, field_idx, *context_ctype, loc);

      XLS_RETURN_IF_ERROR(
          DeclareVariable(decl, CValue(param_bval, outer_value.type()), loc,
                          /*check_unique_ids=*/false));

      prev_vars[decl] = param_bval;
    }

    xls::BValue do_break = context().fb->Literal(xls::UBits(0, 1));

    // Generate body
    // Don't apply continue conditions to increment
    // This context pop will top generate selects
    {
      PushContextGuard context_guard(*this, loc);
      context().propagate_break_up = false;
      context().propagate_continue_up = false;
      context().in_for_body = true;

      XLS_CHECK_GT(context().outer_pipelined_loop_init_interval, 0);

      XLS_CHECK_NE(body, nullptr);
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, ctx));

      // break_condition is the assignment condition
      if (context().relative_break_condition.valid()) {
        xls::BValue break_cond = context().relative_break_condition;
        do_break = context().fb->Or(do_break, break_cond, loc);
      }
    }

    // Increment
    // Break condition skips increment
    if (inc != nullptr) {
      // This context pop will top generate selects
      PushContextGuard context_guard(*this, loc);
      XLS_RETURN_IF_ERROR(and_condition(context().fb->Not(do_break, loc), loc));
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(inc, ctx));
    }

    // Check condition
    if (cond_expr != nullptr) {
      // This context pop will top generate selects
      PushContextGuard context_guard(*this, loc);

      XLS_ASSIGN_OR_RETURN(CValue cond_cval, GenerateIR_Expr(cond_expr, loc));
      XLS_CHECK(cond_cval.type()->Is<CBoolType>());
      xls::BValue break_on_cond_val = context().fb->Not(cond_cval.rvalue());

      do_break = context().fb->Or(do_break, break_on_cond_val, loc);
    }

    // Context out
    std::vector<xls::BValue> tuple_values;
    tuple_values.resize(total_context_values);
    for (const clang::NamedDecl* decl : variable_fields_order) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      tuple_values[field_idx] = context().variables.at(decl).rvalue();
    }

    xls::BValue ret_ctx = MakeStructXLS(tuple_values, *context_ctype, loc);
    std::vector<xls::BValue> return_bvals = {ret_ctx, do_break};

    // For GenerateIRBlock_Prepare() / GenerateIOInvokes()
    extra_return_count += return_bvals.size();

    // First static returns
    for (const clang::NamedDecl* decl :
         generated_func.GetDeterministicallyOrderedStaticValues()) {
      XLS_ASSIGN_OR_RETURN(CValue value, GetIdentifier(decl, loc));
      return_bvals.push_back(value.rvalue());
    }

    // IO returns
    for (IOOp& op : generated_func.io_ops) {
      XLS_CHECK(op.ret_value.valid());
      return_bvals.push_back(op.ret_value);
    }

    xls::BValue ret_val = MakeFlexTuple(return_bvals, loc);
    generated_func.return_value_count = return_bvals.size();
    XLS_ASSIGN_OR_RETURN(generated_func.xls_func,
                         body_builder.BuildWithReturnValue(ret_val));

    // Analyze context variables changed
    for (const clang::NamedDecl* decl : variable_fields_order) {
      const xls::BValue& prev_bval = prev_vars.at(decl);
      const xls::BValue& curr_bval = context().variables.at(decl).rvalue();
      if (prev_bval.node() != curr_bval.node()) {
        vars_changed_in_body.push_back(decl);
      }
    }

    context().sf->SortNamesDeterministically(vars_changed_in_body);

    // All variables now are saved in state, because a streaming channel
    // is used for the context
    vars_to_save_between_iters = variable_fields_order;
  }

  // Generate body proc

  // Construct initial state
  std::vector<xls::Value> init_values = {// First tick is the first iteration
                                         xls::Value(xls::UBits(1, 1))};
  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    const CValue& prev_value = context().variables.at(decl);
    XLS_ASSIGN_OR_RETURN(xls::Value def, CreateDefaultRawValue(
                                             prev_value.type(), GetLoc(*decl)));
    init_values.push_back(def);
  }

  const int64_t extra_state_count = init_values.size();

  for (const clang::NamedDecl* namedecl :
       generated_func.GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = generated_func.static_values.at(namedecl);
    init_values.push_back(initval.rvalue());
  }

  xls::ProcBuilder pb(absl::StrFormat("%s_proc", name_prefix),
                      /*token_name=*/"tkn", package_);
  pb.StateElement("st", xls::Value::Tuple(init_values));

  // For utility functions like MakeStructXls()
  PushContextGuard pb_guard(*this, loc);
  context().fb = absl::implicit_cast<xls::BuilderBase*>(&pb);

  xls::BValue token = pb.GetTokenParam();

  xls::BValue first_iter_state_in = pb.TupleIndex(pb.GetStateParam(0), 0, loc);

  xls::BValue recv_condition = first_iter_state_in;
  XLS_CHECK_EQ(recv_condition.GetType()->GetFlatBitCount(), 1);

  xls::BValue receive =
      pb.ReceiveIf(context_out_channel->generated, token, recv_condition, loc);
  xls::BValue token_ctx = pb.TupleIndex(receive, 0);
  xls::BValue received_context = pb.TupleIndex(receive, 1);

  token = token_ctx;

  // Add selects for changed context variables
  xls::BValue selected_context;
  {
    std::vector<xls::BValue> context_values;
    for (uint64_t fi = 0; fi < total_context_values; ++fi) {
      context_values.push_back(
          GetStructFieldXLS(received_context, fi, *context_ctype, loc));
    }

    // After first flag
    uint64_t state_tup_idx = 1;
    for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
      const uint64_t field_idx = variable_field_indices.at(decl);
      XLS_CHECK_LT(field_idx, context_values.size());
      xls::BValue context_val =
          GetStructFieldXLS(received_context, field_idx, *context_ctype, loc);
      xls::BValue prev_state_val =
          pb.TupleIndex(pb.GetStateParam(0), state_tup_idx++, loc);
      context_values[field_idx] =
          pb.Select(first_iter_state_in, context_val, prev_state_val, loc);
    }
    selected_context = MakeStructXLS(context_values, *context_ctype, loc);
  }

  for (const IOOp& op : generated_func.io_ops) {
    if (op.channel->generated != nullptr) {
      continue;
    }

    const clang::ParmVarDecl* param =
        generated_func.params_by_io_channel.at(op.channel);

    XLS_CHECK(io_test_mode_ || external_channels_by_param_.contains(param));
  }

  // Invoke loop over IOs
  PreparedBlock prepared;
  prepared.xls_func = &generated_func;
  prepared.args.push_back(selected_context);
  prepared.token = token;

  XLS_RETURN_IF_ERROR(GenerateIRBlockPrepare(
      prepared, pb,
      /*next_return_index=*/extra_return_count,
      /*next_state_index=*/extra_state_count,
      /*definition =*/nullptr, /*channels_by_name=*/nullptr, loc));

  XLS_ASSIGN_OR_RETURN(xls::BValue ret_tup,
                       GenerateIOInvokes(prepared, pb, loc));

  token = prepared.token;

  xls::BValue updated_context = pb.TupleIndex(ret_tup, 0, loc);
  xls::BValue do_break = pb.TupleIndex(ret_tup, 1, loc);

  // Send back context on break
  token = pb.SendIf(context_in_channel->generated, token, do_break,
                    updated_context, loc);

  // Construct next state
  std::vector<xls::BValue> next_state_values = {// First iteration next tick?
                                                do_break};
  for (const clang::NamedDecl* decl : vars_to_save_between_iters) {
    const uint64_t field_idx = variable_field_indices.at(decl);
    xls::BValue val =
        GetStructFieldXLS(updated_context, field_idx, *context_ctype, loc);
    next_state_values.push_back(val);
  }
  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    XLS_CHECK(context().fb == &pb);

    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "__xlscc_on_reset unsupported in pipelined loops"));
    }

    next_state_values.push_back(pb.TupleIndex(
        ret_tup, prepared.return_index_for_static.at(namedecl), loc));
  }

  xls::BValue next_state = pb.Tuple(next_state_values);
  XLS_RETURN_IF_ERROR(pb.Build(token, {next_state}).status());

  return absl::OkStatus();
}

// First, flatten the statements in the switch
// It follows a strange pattern where
// case X: foo(); bar(); break;
// Has the form:
// case X: { foo(); } bar(); break;
// And even:
// case X: case Y: bar(); break;
// Has the form:
// case X: { case Y: } bar(); break;
static void FlattenCaseOrDefault(
    const clang::Stmt* stmt, clang::ASTContext& ctx,
    std::vector<const clang::Stmt*>& flat_statements) {
  flat_statements.push_back(stmt);
  if (stmt->getStmtClass() == clang::Stmt::CaseStmtClass) {
    auto case_it = clang_down_cast<const clang::CaseStmt*>(stmt);
    FlattenCaseOrDefault(case_it->getSubStmt(), ctx, flat_statements);
  } else if (stmt->getStmtClass() == clang::Stmt::DefaultStmtClass) {
    auto default_it = clang_down_cast<const clang::DefaultStmt*>(stmt);
    FlattenCaseOrDefault(default_it->getSubStmt(), ctx, flat_statements);
  }
}

absl::Status Translator::GenerateIR_Switch(const clang::SwitchStmt* switchst,
                                           clang::ASTContext& ctx,
                                           const xls::SourceInfo& loc) {
  PushContextGuard switch_guard(*this, loc);
  context().in_switch_body = true;
  context().in_for_body = false;

  if (switchst->getInit()) {
    return absl::UnimplementedError(ErrorMessage(loc, "Switch init"));
  }
  XLS_ASSIGN_OR_RETURN(CValue switch_val,
                       GenerateIR_Expr(switchst->getCond(), loc));
  if (!switch_val.type()->Is<CIntType>()) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Switch on non-integers"));
  }

  // (See comment for FlattenCaseOrDefault())
  std::vector<const clang::Stmt*> flat_statements;
  auto body = switchst->getBody();
  for (const clang::Stmt* child : body->children()) {
    FlattenCaseOrDefault(child, ctx, flat_statements);
  }

  // Scan all cases to create default condition
  std::vector<xls::BValue> case_conds;
  for (const clang::Stmt* stmt : flat_statements) {
    if (stmt->getStmtClass() == clang::Stmt::CaseStmtClass) {
      auto case_it = clang_down_cast<const clang::CaseStmt*>(stmt);
      const xls::SourceInfo loc = GetLoc(*case_it);
      XLS_ASSIGN_OR_RETURN(CValue case_val,
                           GenerateIR_Expr(case_it->getLHS(), loc));
      auto case_int_type = std::dynamic_pointer_cast<CIntType>(case_val.type());
      XLS_CHECK(case_int_type);
      if (*switch_val.type() != *case_int_type) {
        return absl::UnimplementedError(ErrorMessage(loc, ""));
      }
      if (case_it->getRHS()) {
        return absl::UnimplementedError(ErrorMessage(loc, "Case RHS"));
      }
      xls::BValue case_condition =
          context().fb->Eq(switch_val.rvalue(), case_val.rvalue(), loc);
      case_conds.emplace_back(case_condition);
    }
  }

  xls::BValue accum_cond;

  // for indexing into case_conds
  int case_idx = 0;
  for (const clang::Stmt* stmt : flat_statements) {
    const xls::SourceInfo loc = GetLoc(*stmt);
    xls::BValue cond;

    if (stmt->getStmtClass() == clang::Stmt::CaseStmtClass) {
      cond = case_conds[case_idx++];
    } else if (stmt->getStmtClass() == clang::Stmt::DefaultStmtClass) {
      cond = (case_conds.empty())
                 ? context().fb->Literal(xls::UBits(1, 1), loc)
                 : context().fb->Not(context().fb->Or(case_conds, loc), loc);
    } else {
      // For anything other than a case or break, translate it through
      //  the default path. case and break  can still occur inside of
      //  CompoundStmts, and will be processed in GenerateIR_Stmt().

      // No condition = false
      xls::BValue and_accum = accum_cond.valid()
                                  ? accum_cond
                                  : context().fb->Literal(xls::UBits(0, 1));
      // Break goes through this path, sets hit_break
      auto ocond = context().full_condition;
      PushContextGuard stmt_guard(*this, and_accum, loc);
      context().hit_break = false;
      context().full_switch_cond = ocond;
      XLS_RETURN_IF_ERROR(GenerateIR_Compound(stmt, ctx));

      if (context().hit_break) {
        accum_cond = xls::BValue();
      }
      continue;
    }

    // This was a case or default
    if (accum_cond.valid()) {
      accum_cond = context().fb->Or(cond, accum_cond, loc);
    } else {
      accum_cond = cond;
    }
  }

  XLS_CHECK(case_idx == case_conds.size());

  return absl::OkStatus();
}

absl::StatusOr<int64_t> Translator::EvaluateInt64(
    const clang::Expr& expr, const class clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  clang::Expr::EvalResult result;
  if (!expr.EvaluateAsInt(result, ctx, clang::Expr::SE_NoSideEffects,
                          /*InConstantContext=*/false)) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc, "Failed to evaluate expression as int"));
  }
  const clang::APValue& val = result.Val;
  const llvm::APSInt& aps = val.getInt();

  return aps.getExtValue();
}

absl::StatusOr<bool> Translator::EvaluateBool(
    const clang::Expr& expr, const class clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  bool result;
  if (!expr.EvaluateAsBooleanCondition(result, ctx,
                                       /*InConstantContext=*/false)) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc, "Failed to evaluate expression as bool"));
  }
  return result;
}

absl::StatusOr<std::shared_ptr<CType>> Translator::TranslateTypeFromClang(
    const clang::QualType& t, const xls::SourceInfo& loc,
    bool allow_references) {
  const clang::Type* type = t.getTypePtr();

  if (type->getTypeClass() == clang::Type::TypeClass::Builtin) {
    auto builtin = clang_down_cast<const clang::BuiltinType*>(type);
    if (builtin->isVoidType()) {
      return shared_ptr<CType>(new CVoidType());
    }
    if (!builtin->isInteger()) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "BuiltIn type other than integer"));
    }
    switch (builtin->getKind()) {
      case clang::BuiltinType::Kind::Short:
        return shared_ptr<CType>(new CIntType(16, true));
      case clang::BuiltinType::Kind::UShort:
        return shared_ptr<CType>(new CIntType(16, false));
      case clang::BuiltinType::Kind::Int:
        return shared_ptr<CType>(new CIntType(32, true));
      case clang::BuiltinType::Kind::Long:
      case clang::BuiltinType::Kind::LongLong:
        return shared_ptr<CType>(new CIntType(64, true));
      case clang::BuiltinType::Kind::UInt:
        return shared_ptr<CType>(new CIntType(32, false));
      case clang::BuiltinType::Kind::ULong:
      case clang::BuiltinType::Kind::ULongLong:
        return shared_ptr<CType>(new CIntType(64, false));
      case clang::BuiltinType::Kind::Bool:
        return shared_ptr<CType>(new CBoolType());
      case clang::BuiltinType::Kind::SChar:
        return shared_ptr<CType>(new CIntType(8, true, false));
      case clang::BuiltinType::Kind::Char_S:  // These depend on the target
        return shared_ptr<CType>(new CIntType(8, true, true));
      case clang::BuiltinType::Kind::UChar:
        return shared_ptr<CType>(new CIntType(8, false, false));
      case clang::BuiltinType::Kind::Char_U:
        return shared_ptr<CType>(new CIntType(8, false, true));
      default:
        return absl::UnimplementedError(
            absl::StrFormat("Unsupported BuiltIn type %i", builtin->getKind()));
    }
  } else if (type->getTypeClass() == clang::Type::TypeClass::Enum) {
    return shared_ptr<CType>(new CIntType(64, true));
  } else if (type->getTypeClass() ==
             clang::Type::TypeClass::TemplateSpecialization) {
    // Up-cast to avoid multiple inheritance of getAsRecordDecl()
    std::shared_ptr<CInstantiableTypeAlias> ret(new CInstantiableTypeAlias(
        absl::implicit_cast<const clang::NamedDecl*>(type->getAsRecordDecl())));
    return ret;
  } else if (type->getTypeClass() == clang::Type::TypeClass::Record) {
    auto record = clang_down_cast<const clang::RecordType*>(type);
    clang::RecordDecl* decl = record->getDecl();

    switch (decl->getDeclKind()) {
      case clang::Decl::Kind::CXXRecord: {
        return std::shared_ptr<CType>(new CInstantiableTypeAlias(
            absl::implicit_cast<const clang::NamedDecl*>(
                decl->getTypeForDecl()->getAsRecordDecl())));
      }
      case clang::Decl::Kind::ClassTemplateSpecialization: {
        return std::shared_ptr<CType>(new CInstantiableTypeAlias(
            absl::implicit_cast<const clang::NamedDecl*>(
                decl->getTypeForDecl()->getAsRecordDecl())));
      }
      default:
        return absl::UnimplementedError(ErrorMessage(
            loc, "Unsupported recordtype kind %s in translate: %s",
            // getDeclKindName() is inherited from multiple base classes, so
            //  it is necessary to up-cast before calling it to avoid an error.
            absl::implicit_cast<const clang::Decl*>(decl)->getDeclKindName(),
            decl->getNameAsString()));
    }

  } else if (type->getTypeClass() == clang::Type::TypeClass::ConstantArray) {
    auto array = clang_down_cast<const clang::ConstantArrayType*>(type);
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> type,
                         TranslateTypeFromClang(array->getElementType(), loc));
    return std::shared_ptr<CType>(
        new CArrayType(type, array->getSize().getZExtValue()));
  } else if (type->getTypeClass() == clang::Type::TypeClass::Typedef) {
    auto td = clang_down_cast<const clang::TypedefType*>(type);
    return TranslateTypeFromClang(td->getDecl()->getUnderlyingType(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Elaborated) {
    auto elab = clang_down_cast<const clang::ElaboratedType*>(type);
    return TranslateTypeFromClang(elab->getCanonicalTypeInternal(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Auto) {
    auto aut = clang_down_cast<const clang::AutoType*>(type);
    return TranslateTypeFromClang(
        aut->getContainedDeducedType()->getDeducedType(), loc);
  } else if (type->getTypeClass() ==
             clang::Type::TypeClass::SubstTemplateTypeParm) {
    auto subst = clang_down_cast<const clang::SubstTemplateTypeParmType*>(type);
    return TranslateTypeFromClang(subst->getReplacementType(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Decayed) {
    // No pointer support
    auto dec = clang_down_cast<const clang::DecayedType*>(type);
    return TranslateTypeFromClang(dec->getOriginalType(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::LValueReference) {
    if (!allow_references && !t.isConstQualified()) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "References not supported in this context"));
    }
    // No pointer support
    auto lval = clang_down_cast<const clang::LValueReferenceType*>(type);
    return TranslateTypeFromClang(lval->getPointeeType(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Paren) {
    auto lval = clang_down_cast<const clang::ParenType*>(type);
    return TranslateTypeFromClang(lval->desugar(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Pointer) {
    auto lval = clang_down_cast<const clang::PointerType*>(type);
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> pointee_type,
                         TranslateTypeFromClang(lval->getPointeeType(), loc));
    if (context().ignore_pointers) {
      return pointee_type;
    }
    return std::shared_ptr<CType>(new CPointerType(pointee_type));
  } else {
    type->dump();
    return absl::UnimplementedError(
        ErrorMessage(loc, "Unsupported type class in translate: %s",
                     type->getTypeClassName()));
  }
}

absl::StatusOr<xls::Type*> Translator::TranslateTypeToXLS(
    std::shared_ptr<CType> t, const xls::SourceInfo& loc) {
  if (t->Is<CIntType>()) {
    auto it = t->As<CIntType>();
    return package_->GetBitsType(it->width());
  } else if (t->Is<CBitsType>()) {
    auto it = t->As<CBitsType>();
    return package_->GetBitsType(it->GetBitWidth());
  } else if (t->Is<CBoolType>()) {
    return package_->GetBitsType(1);
  } else if (t->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(auto ctype, ResolveTypeInstance(t));
    return TranslateTypeToXLS(ctype, loc);
  } else if (t->Is<CStructType>()) {
    auto it = t->As<CStructType>();
    std::vector<xls::Type*> members;
    for (auto it2 = it->fields().rbegin(); it2 != it->fields().rend(); it2++) {
      std::shared_ptr<CField> field = *it2;
      XLS_ASSIGN_OR_RETURN(xls::Type * ft,
                           TranslateTypeToXLS(field->type(), loc));
      members.push_back(ft);
    }
    return GetStructXLSType(members, *it, loc);
  } else if (t->Is<CArrayType>()) {
    auto it = t->As<CArrayType>();
    XLS_ASSIGN_OR_RETURN(auto xls_elem_type,
                         TranslateTypeToXLS(it->GetElementType(), loc));
    return package_->GetArrayType(it->GetSize(), xls_elem_type);
  } else {
    auto& r = *t;
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unsupported CType subclass in TranslateTypeToXLS: %s",
        typeid(r).name()));
  }
}

absl::StatusOr<Translator::StrippedType> Translator::StripTypeQualifiers(
    const clang::QualType& t) {
  StrippedType ret = StrippedType(t, false);

  // TODO(seanhaskell): Idiomatize this b/240570750
  {
    const clang::Type* type = ret.base.getTypePtr();
    if ((type->getTypeClass() == clang::Type::TypeClass::RValueReference) ||
        (type->getTypeClass() == clang::Type::TypeClass::LValueReference)) {
      ret = StrippedType(type->getPointeeType(), true);
    } else if (type->getTypeClass() == clang::Type::TypeClass::Decayed) {
      auto dec = clang_down_cast<const clang::DecayedType*>(type);
      ret = StrippedType(dec->getOriginalType(), true);
    }
  }

  const bool wasConst = ret.base.isConstQualified();

  const clang::Type* type = ret.base.getTypePtr();
  if (type->getTypeClass() == clang::Type::TypeClass::Elaborated) {
    auto dec = clang_down_cast<const clang::ElaboratedType*>(type);
    ret = StrippedType(dec->desugar(), ret.is_ref);
  }

  if (wasConst) {
    ret.base.addConst();
  }

  return ret;
}

absl::Status Translator::ScanFile(
    absl::string_view source_filename,
    absl::Span<absl::string_view> command_line_args) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->ScanFile(source_filename, command_line_args);
}

absl::StatusOr<std::string> Translator::GetEntryFunctionName() const {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->GetEntryFunctionName();
}

absl::Status Translator::SelectTop(absl::string_view top_function_name) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->SelectTop(top_function_name);
}

absl::StatusOr<GeneratedFunction*> Translator::GenerateIR_Top_Function(
    xls::Package* package, bool force_static, int default_init_interval) {
  const clang::FunctionDecl* top_function = nullptr;

  XLS_CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  package_ = package;
  default_init_interval_ = default_init_interval;

  XLS_ASSIGN_OR_RETURN(
      GeneratedFunction * ret,
      GenerateIR_Function(top_function, top_function->getNameAsString(),
                          force_static));

  if (ret->xls_func == nullptr) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top function %s has no outputs at %s", top_function->getNameAsString(),
        LocString(*top_function)));
  }

  return ret;
}

absl::Status Translator::GenerateExternalChannels(
    const absl::flat_hash_map<std::string, HLSChannel>& channels_by_name,
    const HLSBlock& block, const clang::FunctionDecl* definition,
    const xls::SourceInfo& loc) {
  for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
    const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

    xls::Channel* new_channel = nullptr;

    const HLSChannel& hls_channel =
        channels_by_name.at(param->getNameAsString());
    if (hls_channel.type() == ChannelType::FIFO) {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                           GetChannelType(param, loc));
      XLS_ASSIGN_OR_RETURN(xls::Type * data_type,
                           TranslateTypeToXLS(ctype, loc));

      XLS_ASSIGN_OR_RETURN(
          new_channel,
          package_->CreateStreamingChannel(
              hls_channel.name(),
              hls_channel.is_input() ? xls::ChannelOps::kReceiveOnly
                                     : xls::ChannelOps::kSendOnly,
              data_type, /*initial_values=*/{}, /*fifo_depth=*/absl::nullopt,
              xls::FlowControl::kReadyValid));
    } else {
      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                           TranslateTypeFromClang(stripped.base, loc));

      XLS_ASSIGN_OR_RETURN(xls::Type * data_type,
                           TranslateTypeToXLS(ctype, loc));
      XLS_ASSIGN_OR_RETURN(new_channel, package_->CreateSingleValueChannel(
                                            hls_channel.name(),
                                            hls_channel.is_input()
                                                ? xls::ChannelOps::kReceiveOnly
                                                : xls::ChannelOps::kSendOnly,
                                            data_type));
    }
    XLS_CHECK(!external_channels_by_param_.contains(param));
    external_channels_by_param_[param] = new_channel;
  }
  return absl::OkStatus();
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(
    xls::Package* package, const HLSBlock& block, int top_level_init_interval) {
  // Create external channels
  const clang::FunctionDecl* top_function = nullptr;

  XLS_CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(top_function, parser_->GetTopFunction());

  const clang::FunctionDecl* definition = nullptr;
  top_function->getBody(definition);
  xls::SourceInfo body_loc = GetLoc(*definition);
  package_ = package;

  XLS_RETURN_IF_ERROR(
      CheckInitIntervalValidity(top_level_init_interval, body_loc));

  absl::flat_hash_map<std::string, HLSChannel> channels_by_name;

  XLS_RETURN_IF_ERROR(
      GenerateIRBlockCheck(channels_by_name, block, definition, body_loc));
  XLS_RETURN_IF_ERROR(
      GenerateExternalChannels(channels_by_name, block, definition, body_loc));

  // Generate function without FIFO channel parameters
  // Force top function in block to be static.
  PreparedBlock prepared;
  XLS_ASSIGN_OR_RETURN(
      prepared.xls_func,
      GenerateIR_Top_Function(package, true, top_level_init_interval));

  std::vector<xls::Value> static_init_values;
  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = prepared.xls_func->static_values.at(namedecl);
    static_init_values.push_back(initval.rvalue());
  }

  xls::ProcBuilder pb(block.name() + "_proc", /*token_name=*/"tkn", package);
  pb.StateElement("st", xls::Value::Tuple(static_init_values));

  prepared.token = pb.GetTokenParam();

  XLS_RETURN_IF_ERROR(GenerateIRBlockPrepare(prepared, pb,
                                             /*next_return_index=*/0,
                                             /*next_state_index=*/0, definition,
                                             &channels_by_name, body_loc));

  XLS_ASSIGN_OR_RETURN(xls::BValue last_ret_val,
                       GenerateIOInvokes(prepared, pb, body_loc));

  // Create next state value
  std::vector<xls::BValue> static_next_values;
  XLS_CHECK_GE(prepared.xls_func->return_value_count,
               prepared.xls_func->static_values.size());
  for (const clang::NamedDecl* namedecl :
       prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
    XLS_CHECK(context().fb == &pb);
    xls::BValue next_val = GetFlexTupleField(
        last_ret_val, prepared.return_index_for_static[namedecl],
        prepared.xls_func->return_value_count, body_loc);

    XLS_ASSIGN_OR_RETURN(bool is_on_reset, DeclIsOnReset(namedecl));
    if (is_on_reset) {
      next_val = pb.Literal(xls::Value(xls::UBits(0, 1)), body_loc);
    }
    static_next_values.push_back(next_val);
  }
  const xls::BValue next_state = pb.Tuple(static_next_values);

  return pb.Build(prepared.token, {next_state});
}

absl::StatusOr<xls::BValue> Translator::GenerateIOInvokes(
    PreparedBlock& prepared, xls::ProcBuilder& pb,
    const xls::SourceInfo& body_loc) {
  XLS_CHECK(&pb == context().fb);

  XLS_CHECK_GE(prepared.xls_func->return_value_count,
               prepared.xls_func->io_ops.size());

  std::vector<xls::BValue> fan_out_tokens;

  // The function is first invoked with defaults for any
  //  read() IO Ops.
  // If there are any read() IO Ops, then it will be invoked again
  //  for each read Op below.
  // Statics don't need to generate additional invokes, since they need not
  //  exchange any data with the outside world between iterations.
  xls::BValue last_ret_val =
      pb.Invoke(prepared.args, prepared.xls_func->xls_func, body_loc);
  for (const IOOp& op : prepared.xls_func->io_ops) {
    xls::Channel* xls_channel =
        prepared.xls_channel_by_function_channel.at(op.channel);
    const int return_index = prepared.return_index_for_op.at(&op);

    xls::SourceInfo op_loc = op.op_location;

    if (op.op == OpType::kRecv) {
      const int arg_index = prepared.arg_index_for_op.at(&op);
      XLS_CHECK(arg_index >= 0 && arg_index < prepared.args.size());

      xls::BValue condition = GetFlexTupleField(
          last_ret_val, return_index, prepared.xls_func->return_value_count,
          op_loc, absl::StrFormat("%s_pred", xls_channel->name()));

      XLS_CHECK_EQ(condition.GetType()->GetFlatBitCount(), 1);
      xls::BValue receive =
          pb.ReceiveIf(xls_channel, prepared.token, condition, op_loc);
      const xls::BValue new_token = pb.TupleIndex(receive, 0);
      fan_out_tokens.push_back(new_token);

      xls::BValue in_val = pb.TupleIndex(receive, 1);

      prepared.args[arg_index] = in_val;

      // The function is invoked again with the value received from the channel
      //  for each read() Op. The final invocation will produce all complete
      //  outputs.
      last_ret_val =
          pb.Invoke(prepared.args, prepared.xls_func->xls_func, op_loc);
    } else if (op.op == OpType::kSend) {
      xls::BValue send_tup =
          GetFlexTupleField(last_ret_val, return_index,
                            prepared.xls_func->return_value_count, op_loc);
      xls::BValue val = pb.TupleIndex(send_tup, 0, op_loc);
      xls::BValue condition = pb.TupleIndex(
          send_tup, 1, op_loc, absl::StrFormat("%s_pred", xls_channel->name()));

      const xls::BValue new_token =
          pb.SendIf(xls_channel, prepared.token, condition, {val}, op_loc);

      fan_out_tokens.push_back(new_token);
    }
  }

  if (!fan_out_tokens.empty()) {
    prepared.token = pb.AfterAll(fan_out_tokens, body_loc);
  }

  return last_ret_val;
}

absl::Status Translator::GenerateIRBlockCheck(
    absl::flat_hash_map<std::string, HLSChannel>& channels_by_name,
    const HLSBlock& block, const clang::FunctionDecl* definition,
    const xls::SourceInfo& body_loc) {
  if (!block.has_name()) {
    return absl::InvalidArgumentError(absl::StrFormat("HLSBlock has no name"));
  }

  absl::flat_hash_set<string> channel_names_in_block;
  for (const HLSChannel& channel : block.channels()) {
    if (!channel.has_name() || !channel.has_is_input() || !channel.has_type()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel is incomplete in proto"));
    }

    if (channels_by_name.contains(channel.name())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate channel name %s", channel.name()));
    }

    channels_by_name[channel.name()] = channel;
    channel_names_in_block.insert(channel.name());
  }

  if (definition->parameters().size() != block.channels_size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Top function has %i parameters, but block proto defines %i channels",
        definition->parameters().size(), block.channels_size()));
  }

  for (const clang::ParmVarDecl* param : definition->parameters()) {
    if (!channel_names_in_block.contains(param->getNameAsString())) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block proto does not contain channels '%s' in function prototype",
          param->getNameAsString()));
    }
    channel_names_in_block.erase(param->getNameAsString());
  }

  if (!channel_names_in_block.empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Block proto contains %i channels not in function prototype",
        channel_names_in_block.size()));
  }

  return absl::OkStatus();
}

absl::Status Translator::GenerateIRBlockPrepare(
    PreparedBlock& prepared, xls::ProcBuilder& pb, int64_t next_return_index,
    int64_t next_state_index, const clang::FunctionDecl* definition,
    const absl::flat_hash_map<std::string, HLSChannel>* channels_by_name,
    const xls::SourceInfo& body_loc) {
  // For defaults, updates, invokes
  context().fb = dynamic_cast<xls::BuilderBase*>(&pb);

  // Add returns for static locals
  {
    for (const clang::NamedDecl* namedecl :
         prepared.xls_func->GetDeterministicallyOrderedStaticValues()) {
      prepared.return_index_for_static[namedecl] = next_return_index++;
      prepared.state_index_for_static[namedecl] = next_state_index++;
    }
  }

  // Prepare direct-ins
  if (definition != nullptr) {
    XLS_CHECK_NE(channels_by_name, nullptr);

    for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
      const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

      const HLSChannel& hls_channel =
          channels_by_name->at(param->getNameAsString());
      if (hls_channel.type() != ChannelType::DIRECT_IN) {
        continue;
      }

      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));

      xls::Channel* xls_channel = external_channels_by_param_.at(param);

      xls::BValue receive = pb.Receive(xls_channel, prepared.token);
      prepared.token = pb.TupleIndex(receive, 0);
      xls::BValue direct_in_value = pb.TupleIndex(receive, 1);

      prepared.args.push_back(direct_in_value);

      // If it's const or not a reference, then there's no return
      if (stripped.is_ref && !stripped.base.isConstQualified()) {
        ++next_return_index;
      }
    }
  }

  // Initialize parameters to defaults, handle direct-ins, create channels
  // Add channels in order of function prototype
  // Find return indices for ops
  for (const IOOp& op : prepared.xls_func->io_ops) {
    prepared.return_index_for_op[&op] = next_return_index++;

    if (op.channel->generated != nullptr) {
      prepared.xls_channel_by_function_channel[op.channel] =
          op.channel->generated;
      continue;
    }

    const clang::ParmVarDecl* param =
        prepared.xls_func->params_by_io_channel.at(op.channel);

    if (!prepared.xls_channel_by_function_channel.contains(op.channel)) {
      xls::Channel* xls_channel = external_channels_by_param_.at(param);
      prepared.xls_channel_by_function_channel[op.channel] = xls_channel;
    }
  }

  // Params
  for (const xlscc::SideEffectingParameter& param :
       prepared.xls_func->side_effecting_parameters) {
    switch (param.type) {
      case xlscc::SideEffectingParameterType::kIOOp: {
        const IOOp& op = *param.io_op;
        if (op.channel->channel_op_type == OpType::kRecv) {
          XLS_ASSIGN_OR_RETURN(
              xls::BValue val,
              CreateDefaultValue(op.channel->item_type, body_loc));

          prepared.arg_index_for_op[&op] = prepared.args.size();
          prepared.args.push_back(val);
        }
        break;
      }
      case xlscc::SideEffectingParameterType::kStatic: {
        const uint64_t static_idx =
            prepared.state_index_for_static.at(param.static_value);
        prepared.args.push_back(
            pb.TupleIndex(pb.GetStateParam(0), static_idx, body_loc));
        break;
      }
      default: {
        return absl::InternalError(
            ErrorMessage(body_loc, "Unknown type of SideEffectingParameter"));
        break;
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> Translator::ExprIsChannel(const clang::Expr* object,
                                               const xls::SourceInfo& loc) {
  // Avoid "this", as it's a pointer
  if (object->getStmtClass() == clang::Expr::ImplicitCastExprClass &&
      clang_down_cast<const clang::CastExpr*>(object)
              ->getSubExpr()
              ->getStmtClass() == clang::Expr::CXXThisExprClass) {
    return false;
  }
  if (object->getStmtClass() == clang::Expr::CXXThisExprClass) {
    return false;
  }
  return TypeIsChannel(object->getType(), loc);
}

absl::StatusOr<bool> Translator::TypeIsChannel(const clang::QualType& param,
                                               const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(StrippedType stripped, StripTypeQualifiers(param));
  absl::StatusOr<std::shared_ptr<CType>> obj_type_ret =
      TranslateTypeFromClang(stripped.base, loc);

  // Ignore un-translatable types like pointers
  if (!obj_type_ret.ok()) {
    return false;
  }

  std::shared_ptr<CType> obj_type = obj_type_ret.value();

  if (auto obj_inst_type =
          std::dynamic_pointer_cast<const CInstantiableTypeAlias>(obj_type)) {
    if (obj_inst_type->base()->getNameAsString() == "__xls_channel") {
      return true;
    }
  }

  return false;
}

absl::Status Translator::InlineAllInvokes(xls::Package* package) {
  std::unique_ptr<xls::CompoundPass> pipeline =
      xls::CreateStandardPassPipeline();
  xls::PassOptions options;
  xls::PassResults results;

  // This pass wants a delay estimator
  options.skip_passes = {"bdd_cse"};

  XLS_RETURN_IF_ERROR(pipeline->Run(package, options, &results).status());
  return absl::OkStatus();
}

absl::Status Translator::GenerateFunctionMetadata(
    const clang::FunctionDecl* func,
    xlscc_metadata::FunctionPrototype* output) {
  output->mutable_name()->set_name(func->getNameAsString());
  XLS_CHECK(xls_names_for_functions_generated_.contains(func));
  output->mutable_name()->set_xls_name(
      xls_names_for_functions_generated_.at(func));
  output->mutable_name()->set_fully_qualified_name(
      func->getQualifiedNameAsString());
  output->mutable_name()->set_id(
      reinterpret_cast<uint64_t>(dynamic_cast<const clang::NamedDecl*>(func)));

  FillLocationRangeProto(func->getReturnTypeSourceRange(),
                         output->mutable_return_location());
  FillLocationRangeProto(func->getParametersSourceRange(),
                         output->mutable_parameters_location());
  FillLocationRangeProto(func->getSourceRange(),
                         output->mutable_whole_declaration_location());

  XLS_RETURN_IF_ERROR(GenerateMetadataType(func->getReturnType(),
                                           output->mutable_return_type()));

  for (int64_t pi = 0; pi < func->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = func->getParamDecl(pi);
    xlscc_metadata::FunctionParameter* proto_param = output->add_params();
    proto_param->set_name(p->getNameAsString());

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_RETURN_IF_ERROR(
        GenerateMetadataType(stripped.base, proto_param->mutable_type()));

    proto_param->set_is_reference(stripped.is_ref);
    proto_param->set_is_const(stripped.base.isConstQualified());
  }

  auto found = inst_functions_.find(func);
  if (found == inst_functions_.end()) {
    return absl::NotFoundError(
        "GenerateFunctionMetadata called for FuncDecl that has not been "
        "processed for IR generation.");
  }
  for (const clang::NamedDecl* namedecl :
       found->second->GetDeterministicallyOrderedStaticValues()) {
    const ConstValue& initval = found->second->static_values.at(namedecl);
    auto p = clang_down_cast<const clang::ValueDecl*>(namedecl);
    xlscc_metadata::FunctionValue* proto_static_value =
        output->add_static_values();
    XLS_RETURN_IF_ERROR(
        GenerateMetadataCPPName(namedecl, proto_static_value->mutable_name()));
    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));
    XLS_RETURN_IF_ERROR(GenerateMetadataType(
        stripped.base, proto_static_value->mutable_type()));
    XLS_ASSIGN_OR_RETURN(
        std::shared_ptr<CType> ctype,
        TranslateTypeFromClang(stripped.base, xls::SourceInfo()));
    XLS_RETURN_IF_ERROR(ctype->GetMetadataValue(
        *this, initval, proto_static_value->mutable_value()));
  }

  return absl::OkStatus();
}

absl::StatusOr<xlscc_metadata::MetadataOutput> Translator::GenerateMetadata() {
  XLS_CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(const clang::FunctionDecl* top_function,
                       parser_->GetTopFunction());

  xlscc_metadata::MetadataOutput ret;

  parser_->AddSourceInfoToMetadata(ret);

  // Top function proto
  XLS_RETURN_IF_ERROR(
      GenerateFunctionMetadata(top_function, ret.mutable_top_func_proto()));

  for (auto const& [decl, xls_name] : xls_names_for_functions_generated_) {
    XLS_RETURN_IF_ERROR(
        GenerateFunctionMetadata(decl, ret.add_all_func_protos()));
  }

  // Struct types
  for (std::pair<std::shared_ptr<CInstantiableTypeAlias>,
                 std::shared_ptr<CType>>
           type : inst_types_) {
    auto ctype_as_struct =
        clang_down_cast<const CStructType*>(type.second.get());
    if (ctype_as_struct == nullptr) {
      continue;
    }

    xlscc_metadata::Type* struct_out = ret.add_structs();
    XLS_RETURN_IF_ERROR(type.first->GetMetadata(
        *this, struct_out->mutable_as_struct()->mutable_name()));
    XLS_RETURN_IF_ERROR(ctype_as_struct->GetMetadata(*this, struct_out));
  }

  return ret;
}

void Translator::FillLocationProto(
    const clang::SourceLocation& location,
    xlscc_metadata::SourceLocation* location_out) {
  // Check that the location exists
  // this may be invalid if the function has no parameters
  if (location.isInvalid()) {
    return;
  }
  const clang::PresumedLoc& presumed = parser_->sm_->getPresumedLoc(location);
  location_out->set_filename(presumed.getFilename());
  location_out->set_line(presumed.getLine());
  location_out->set_column(presumed.getColumn());
}

void Translator::FillLocationRangeProto(
    const clang::SourceRange& range,
    xlscc_metadata::SourceLocationRange* range_out) {
  FillLocationProto(range.getBegin(), range_out->mutable_begin());
  FillLocationProto(range.getEnd(), range_out->mutable_end());
}

absl::Status Translator::GenerateMetadataCPPName(
    const clang::NamedDecl* decl_in, xlscc_metadata::CPPName* name_out) {
  name_out->set_fully_qualified_name(decl_in->getQualifiedNameAsString());
  name_out->set_name(decl_in->getNameAsString());
  name_out->set_id(reinterpret_cast<uint64_t>(decl_in));
  return absl::OkStatus();
}

absl::Status Translator::GenerateMetadataType(const clang::QualType& type_in,
                                              xlscc_metadata::Type* type_out) {
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                       TranslateTypeFromClang(type_in, xls::SourceInfo(),
                                              /*allow_references=*/true));
  return ctype->GetMetadata(*this, type_out);
}

absl::StatusOr<xls::BValue> Translator::GenTypeConvert(
    CValue const& in, std::shared_ptr<CType> out_type,
    const xls::SourceInfo& loc) {
  XLS_CHECK_NE(in.type().get(), nullptr);
  XLS_CHECK_NE(out_type.get(), nullptr);
  if (*in.type() == *out_type) {
    return in.rvalue();
  }
  if (out_type->Is<CStructType>()) {
    return in.rvalue();
  } else if (out_type->Is<CVoidType>()) {
    return xls::BValue();
  } else if (out_type->Is<CArrayType>()) {
    return in.rvalue();
  } else if (out_type->Is<CBitsType>()) {
    auto out_bits_type = out_type->As<CBitsType>();
    XLS_ASSIGN_OR_RETURN(auto conv_type, ResolveTypeInstance(in.type()));
    if (!conv_type->Is<CBitsType>()) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Cannot convert type %s to bits", std::string(*in.type())));
    }
    if (conv_type->GetBitWidth() != out_bits_type->GetBitWidth()) {
      return absl::UnimplementedError(absl::StrFormat(
          "No implicit bit width conversions for __xls_bits: from %s to %s at "
          "%s",
          std::string(*in.type()), std::string(*out_type), LocString(loc)));
    }
    return in.rvalue();
  } else if (out_type->Is<CBoolType>()) {
    return GenBoolConvert(in, loc);
  } else if (out_type->Is<CIntType>()) {
    if (!(in.type()->Is<CBoolType>() || in.type()->Is<CIntType>())) {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Cannot convert type %s to int", std::string(*in.type())));
    }

    const int expr_width = in.type()->GetBitWidth();
    if (expr_width == out_type->GetBitWidth()) {
      return in.rvalue();
    } else if (expr_width < out_type->GetBitWidth()) {
      auto p_in_int = std::dynamic_pointer_cast<const CIntType>(in.type());
      if ((!in.type()->Is<CBoolType>()) &&
          (p_in_int != nullptr && p_in_int->is_signed())) {
        return context().fb->SignExtend(in.rvalue(), out_type->GetBitWidth(),
                                        loc);
      } else {
        return context().fb->ZeroExtend(in.rvalue(), out_type->GetBitWidth(),
                                        loc);
      }
    } else {
      return context().fb->BitSlice(in.rvalue(), 0, out_type->GetBitWidth(),
                                    loc);
    }
  } else if (out_type->Is<CInstantiableTypeAlias>()) {
    XLS_ASSIGN_OR_RETURN(auto t, ResolveTypeInstance(out_type));
    return GenTypeConvert(in, t, loc);
  }
  return absl::UnimplementedError(
      ErrorMessage(loc, "Don't know how to convert %s to type %s",
                   std::string(*in.type()), std::string(*out_type)));
}

absl::StatusOr<xls::BValue> Translator::GenBoolConvert(
    CValue const& in, const xls::SourceInfo& loc) {
  if (!(in.type()->Is<CBoolType>() || in.type()->Is<CIntType>())) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Cannot convert type %s to bool", std::string(*in.type())));
  }
  XLS_CHECK_GT(in.type()->GetBitWidth(), 0);
  if (in.type()->GetBitWidth() == 1) {
    return in.rvalue();
  }
  xls::BValue const0 =
      context().fb->Literal(xls::UBits(0, in.type()->GetBitWidth()), loc);
  return context().fb->Ne(in.rvalue(), const0, loc);
}

std::string Translator::LocString(const xls::SourceInfo& loc) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->LocString(loc);
}

xls::SourceInfo Translator::GetLoc(const clang::Stmt& stmt) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->GetLoc(stmt);
}

xls::SourceInfo Translator::GetLoc(const clang::Decl& decl) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->GetLoc(decl);
}

clang::PresumedLoc Translator::GetPresumedLoc(const clang::Stmt& stmt) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->GetPresumedLoc(stmt);
}

clang::PresumedLoc Translator::GetPresumedLoc(const clang::Decl& decl) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->GetPresumedLoc(decl);
}

absl::StatusOr<Pragma> Translator::FindPragmaForLoc(
    const clang::SourceLocation& loc) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->FindPragmaForLoc(loc);
}

absl::StatusOr<Pragma> Translator::FindPragmaForLoc(
    const clang::PresumedLoc& ploc) {
  XLS_CHECK_NE(parser_.get(), nullptr);
  return parser_->FindPragmaForLoc(ploc);
}

}  // namespace xlscc
