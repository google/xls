// Copyright 2022 The XLS Authors
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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"

using std::shared_ptr;
using std::string;
using std::vector;

namespace xlscc {

CType::~CType() {}

bool CType::operator!=(const CType& o) const { return !(*this == o); }

int CType::GetBitWidth() const { return 0; }

absl::StatusOr<bool> CType::ContainsLValues(Translator& translator) const {
  return false;
}

CType::operator std::string() const { return "CType"; }

xls::Type* CType::GetXLSType(xls::Package* /*package*/) const {
  XLS_LOG(FATAL) << "GetXLSType() unsupported in CType base class";
  return nullptr;
}

bool CType::StoredAsXLSBits() const { return false; }

absl::Status CType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
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

absl::Status CVoidType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
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

absl::Status CBitsType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  output->mutable_as_bits()->set_width(width_);
  return absl::OkStatus();
}

absl::Status CBitsType::GetMetadataValue(Translator& translator,
                                         const ConstValue const_value,
                                         xlscc_metadata::Value* output) const {
  vector<uint8_t> bytes = const_value.rvalue().bits().ToBytes();
  // Bits::ToBytes is little-endian, data is stored in the proto as
  // big-endian.
  std::reverse(bytes.begin(), bytes.end());
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
  }
  if (width_ == 1) {
    return pre + "pseudobool";
  }
  if (width_ == 64) {
    return pre + "int64_t";
  }
  if (width_ == 16) {
    return pre + "short";
  }
  if (width_ == 8) {
    return pre + (is_declared_as_char() ? "char" : "int8_t");
  }
  XLS_CHECK(0);
  return "Unsupported";
}

absl::Status CIntType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  output->mutable_as_int()->set_width(width_);
  output->mutable_as_int()->set_is_signed(is_signed_);
  if (width_ == 8) {
    output->mutable_as_int()->set_is_declared_as_char(is_declared_as_char_);
  }
  output->mutable_as_int()->set_is_synthetic(false);
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

absl::Status CBoolType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
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
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  auto temp_alias = std::make_shared<CInstantiableTypeAlias>(base_);
  absl::StatusOr<xlscc_metadata::IntType> is_synthetic_int =
      translator.GenerateSyntheticInt(temp_alias);

  if (is_synthetic_int.ok()) {
    // hls_synthetic_int case
    *output->mutable_as_int() = is_synthetic_int.value();
    return absl::OkStatus();
  }

  // Recurse for aliases used in referenced struct
  {
    XLS_ASSIGN_OR_RETURN(auto resolved,
                         translator.ResolveTypeInstance(temp_alias));
    xlscc_metadata::Type dummy_type;
    XLS_RETURN_IF_ERROR(
        resolved->GetMetadata(translator, &dummy_type, aliases_used));
  }

  aliases_used.insert(base_);

  output->mutable_as_inst()->mutable_name()->set_name(base_->getNameAsString());
  output->mutable_as_inst()->mutable_name()->set_fully_qualified_name(
      base_->getQualifiedNameAsString());
  output->mutable_as_inst()->mutable_name()->set_id(
      reinterpret_cast<uint64_t>(base_));

  if (auto special =
          clang::dyn_cast<const clang::ClassTemplateSpecializationDecl>(
              base_)) {
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
          xlscc_metadata::Type* as_type = proto_arg->mutable_as_type();
          XLS_RETURN_IF_ERROR(
              arg_ctype->GetMetadata(translator, as_type, aliases_used));
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
  bool write_as_bits = false;
  std::shared_ptr<CInstantiableTypeAlias> inst(
      new CInstantiableTypeAlias(base_));
  if (translator.GenerateSyntheticInt(inst).ok()) {
    write_as_bits = true;
  }

  auto found = translator.inst_types_.find(inst);
  XLS_CHECK(found != translator.inst_types_.end());
  auto struct_type =
      std::dynamic_pointer_cast<const CStructType>(found->second);

  // Handle __xls_bits
  if (struct_type == nullptr) {
    XLS_CHECK_EQ(base_->getNameAsString(), "__xls_bits");
    write_as_bits = true;
  }

  if (write_as_bits) {
    XLS_CHECK(const_value.rvalue().IsBits());
    vector<uint8_t> bytes = const_value.rvalue().bits().ToBytes();
    // Bits::ToBytes is little-endian, data is stored in the proto as
    // big-endian.
    std::reverse(bytes.begin(), bytes.end());
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

absl::StatusOr<bool> CInstantiableTypeAlias::ContainsLValues(
    Translator& translator) const {
  auto temp_alias = std::make_shared<CInstantiableTypeAlias>(base_);
  XLS_ASSIGN_OR_RETURN(auto resolved,
                       translator.ResolveTypeInstance(temp_alias));
  XLS_ASSIGN_OR_RETURN(bool ret, resolved->ContainsLValues(translator));
  return ret;
}

CStructType::CStructType(std::vector<std::shared_ptr<CField>> fields,
                         bool no_tuple_flag, bool synthetic_int_flag)
    : no_tuple_flag_(no_tuple_flag),
      synthetic_int_flag_(synthetic_int_flag),
      fields_(fields) {
  for (const std::shared_ptr<CField>& pf : fields) {
    XLS_CHECK(!fields_by_name_.contains(pf->name()));
    fields_by_name_[pf->name()] = pf;
  }
}

absl::Status CStructType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  // Synthetic int classes / structs should never be emitted
  XLS_CHECK(!synthetic_int_flag_);

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
        translator, output->mutable_as_struct()->add_fields(), aliases_used));
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

bool CStructType::synthetic_int_flag() const { return synthetic_int_flag_; }

int CStructType::GetBitWidth() const {
  int ret = 0;
  for (const std::shared_ptr<CField>& field : fields_) {
    ret += field->type()->GetBitWidth();
  }
  return ret;
}

absl::StatusOr<bool> CStructType::ContainsLValues(
    Translator& translator) const {
  for (const std::shared_ptr<CField>& field : fields_) {
    XLS_ASSIGN_OR_RETURN(bool ret, field->type()->ContainsLValues(translator));

    if (ret) {
      return true;
    }
  }
  return false;
}

CStructType::operator std::string() const {
  std::ostringstream ostr;
  ostr << "{";
  if (no_tuple_flag_) {
    ostr << " notuple ";
  }
  if (synthetic_int_flag_) {
    ostr << " synthetic_int ";
  }
  for (const std::shared_ptr<CField>& it : fields_) {
    ostr << "[" << it->index() << "] "
         << (it->name() != nullptr ? it->name()->getNameAsString()
                                   : std::string("nullptr"))
         << ": " << string(*it->type()) << " ";
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

CInternalTuple::CInternalTuple(std::vector<std::shared_ptr<CType>> fields)
    : fields_(fields) {}

absl::Status CInternalTuple::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  for (int i = 0; i < fields_.size(); ++i) {
    std::shared_ptr<CType> field = fields_[i];
    XLS_RETURN_IF_ERROR(field->GetMetadata(
        translator, output->mutable_as_tuple()->add_fields(), aliases_used));
  }
  return absl::OkStatus();
}

absl::Status CInternalTuple::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  for (int i = 0; i < fields_.size(); ++i) {
    auto field = fields_[i];
    auto struct_field_value = output->mutable_as_tuple()->add_fields();
    XLS_ASSIGN_OR_RETURN(std::vector<xls::Value> values,
                         const_value.rvalue().GetElements());
    xls::Value elem_value = values.at(this->fields().size() - 1 - i);
    XLS_RETURN_IF_ERROR(
        field->GetMetadataValue(translator, ConstValue(elem_value, field),
                                struct_field_value->mutable_value()));
  }
  return absl::OkStatus();
}

int CInternalTuple::GetBitWidth() const {
  int ret = 0;
  for (const std::shared_ptr<CType>& field : fields_) {
    ret += field->GetBitWidth();
  }
  return ret;
}

CInternalTuple::operator std::string() const {
  std::ostringstream ostr;
  ostr << "InternalTuple{";
  for (int i = 0; i < fields_.size(); ++i) {
    const std::shared_ptr<CType>& field = fields_[i];
    if (i > 0) {
      ostr << ", ";
    }
    ostr << "[" << i << "] "
         << ": " << string(*field);
  }
  ostr << "}";
  return ostr.str();
}

bool CInternalTuple::operator==(const CType& o) const {
  if (!o.Is<CInternalTuple>()) {
    return false;
  }
  auto obj = o.As<CInternalTuple>();
  for (int i = 0; i < fields_.size(); ++i) {
    if (obj->fields_[i] != fields_[i]) {
      return false;
    }
  }
  return true;
}

const std::vector<std::shared_ptr<CType>>& CInternalTuple::fields() const {
  return fields_;
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

absl::StatusOr<bool> CArrayType::ContainsLValues(Translator& translator) const {
  XLS_ASSIGN_OR_RETURN(bool ret, element_->ContainsLValues(translator));
  return ret;
}

absl::Status CField::GetMetadata(
    Translator& translator, xlscc_metadata::StructField* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  output->set_name(name_->getNameAsString());
  XLS_RETURN_IF_ERROR(
      type_->GetMetadata(translator, output->mutable_type(), aliases_used));
  return absl::OkStatus();
}

int CArrayType::GetSize() const { return size_; }

std::shared_ptr<CType> CArrayType::GetElementType() const { return element_; }

CArrayType::operator std::string() const {
  return absl::StrFormat("%s[%i]", string(*element_), size_);
}

absl::Status CArrayType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  output->mutable_as_array()->set_size(size_);
  XLS_RETURN_IF_ERROR(element_->GetMetadata(
      translator, output->mutable_as_array()->mutable_element_type(),
      aliases_used));
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

int CPointerType::GetBitWidth() const { return 0; }

absl::StatusOr<bool> CPointerType::ContainsLValues(
    Translator& translator) const {
  return true;
}

std::shared_ptr<CType> CPointerType::GetPointeeType() const {
  return pointee_type_;
}

CPointerType::operator std::string() const {
  return absl::StrFormat("%s*", string(*pointee_type_));
}

absl::Status CPointerType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  return absl::UnavailableError(
      "Can't generate externally useful metadata for pointers");
}

absl::Status CPointerType::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  return absl::UnavailableError(
      "Can't generate externally useful metadata for pointers");
}

CReferenceType::CReferenceType(std::shared_ptr<CType> pointee_type)
    : pointee_type_(pointee_type) {}

bool CReferenceType::operator==(const CType& o) const {
  if (!o.Is<CReferenceType>()) return false;
  const auto* o_derived = o.As<CReferenceType>();
  return *pointee_type_ == *o_derived->pointee_type_;
}

int CReferenceType::GetBitWidth() const { return pointee_type_->GetBitWidth(); }

absl::StatusOr<bool> CReferenceType::ContainsLValues(
    Translator& translator) const {
  return true;
}

std::shared_ptr<CType> CReferenceType::GetPointeeType() const {
  return pointee_type_;
}

CReferenceType::operator std::string() const {
  return absl::StrFormat("%s&", string(*pointee_type_));
}

absl::Status CReferenceType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  return absl::UnavailableError(
      "Can't generate externally useful metadata for references");
}

absl::Status CReferenceType::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  return absl::UnavailableError(
      "Can't generate externally useful metadata for references");
}

CChannelType::CChannelType(std::shared_ptr<CType> item_type, OpType op_type)
    : item_type_(item_type), op_type_(op_type) {}

bool CChannelType::operator==(const CType& o) const {
  if (!o.Is<CChannelType>()) return false;
  const auto* o_derived = o.As<CChannelType>();
  return *item_type_ == *o_derived->item_type_ &&
         op_type_ == o_derived->op_type_;
}

int CChannelType::GetBitWidth() const { return item_type_->GetBitWidth(); }

std::shared_ptr<CType> CChannelType::GetItemType() const { return item_type_; }

CChannelType::operator std::string() const {
  return absl::StrFormat("channel<%s,%s>", string(*item_type_),
                         (op_type_ == OpType::kRecv)
                             ? "recv"
                             : ((op_type_ == OpType::kSend) ? "send" : "null"));
}

absl::Status CChannelType::GetMetadata(
    Translator& translator, xlscc_metadata::Type* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) const {
  XLS_RETURN_IF_ERROR(item_type_->GetMetadata(
      translator, output->mutable_as_channel()->mutable_item_type(),
      aliases_used));

  return absl::OkStatus();
}

absl::Status CChannelType::GetMetadataValue(
    Translator& translator, const ConstValue const_value,
    xlscc_metadata::Value* output) const {
  return absl::OkStatus();
}

absl::StatusOr<bool> CChannelType::ContainsLValues(
    Translator& translator) const {
  return true;
}

OpType CChannelType::GetOpType() const { return op_type_; }

}  //  namespace xlscc
