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

#include <cstdint>
#include <list>
#include <memory>

#include "absl/container/flat_hash_set.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "clang/include/clang/AST/Decl.h"
#include "clang/include/clang/AST/DeclCXX.h"
#include "clang/include/clang/AST/DeclTemplate.h"
#include "clang/include/clang/AST/TemplateBase.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/SourceLocation.h"
#include "clang/include/clang/Basic/SourceManager.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/ir/source_location.h"

namespace xlscc {

absl::Status Translator::GenerateMetadataCPPName(
    const clang::NamedDecl* decl_in, xlscc_metadata::CPPName* name_out) {
  name_out->set_fully_qualified_name(decl_in->getQualifiedNameAsString());
  name_out->set_name(decl_in->getNameAsString());
  name_out->set_id(reinterpret_cast<uint64_t>(decl_in));
  return absl::OkStatus();
}

absl::Status Translator::GenerateMetadataType(
    const clang::QualType& type_in, xlscc_metadata::Type* type_out,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) {
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                       TranslateTypeFromClang(type_in, xls::SourceInfo()));
  const clang::RecordDecl* decl = type_in->getAsRecordDecl();
  if (decl != nullptr) {
    FillLocationRangeProto(decl->getSourceRange(),
                           type_out->mutable_declaration_location());
  }
  XLS_RETURN_IF_ERROR(ctype->GetMetadata(*this, type_out, aliases_used));
  if (type_out->has_as_int() && type_out->as_int().is_synthetic()) {
    type_out->clear_declaration_location();
  }
  return absl::OkStatus();
}

absl::StatusOr<xlscc_metadata::IntType> Translator::GenerateSyntheticInt(
    std::shared_ptr<CType> ctype) {
  const CStructType* struct_type = nullptr;
  auto inst_type = dynamic_cast<const CInstantiableTypeAlias*>(ctype.get());
  if (inst_type != nullptr) {
    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> resolved,
                         ResolveTypeInstance(ctype));
    struct_type = dynamic_cast<const CStructType*>(resolved.get());
    if (struct_type != nullptr && struct_type->synthetic_int_flag()) {
      CHECK(clang::isa<clang::ClassTemplateSpecializationDecl>(
          inst_type->base()));
      auto special = clang::cast<clang::ClassTemplateSpecializationDecl>(
          inst_type->base());
      const clang::TemplateArgumentList& arguments = special->getTemplateArgs();
      CHECK_EQ(arguments.size(), 2);

      const clang::TemplateArgument& width_arg = arguments.get(0);
      CHECK_EQ(width_arg.getKind(), clang::TemplateArgument::ArgKind::Integral);
      const clang::TemplateArgument& signed_arg = arguments.get(1);
      CHECK_EQ(signed_arg.getKind(),
               clang::TemplateArgument::ArgKind::Integral);

      xlscc_metadata::IntType ret;
      ret.set_is_synthetic(true);
      ret.set_width(width_arg.getAsIntegral().getExtValue());
      ret.set_is_signed(signed_arg.getAsIntegral().getExtValue() == 1);
      return ret;
    }
  }
  return absl::NotFoundError("not synthetic int");
}

absl::Status Translator::GenerateFunctionMetadata(
    const clang::FunctionDecl* func, xlscc_metadata::FunctionPrototype* output,
    absl::flat_hash_set<const clang::NamedDecl*>& aliases_used) {
  output->mutable_name()->set_name(func->getNameAsString());
  CHECK(xls_names_for_functions_generated_.contains(func));
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

  // Return metadata
  {
    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(func->getReturnType()));

    XLS_RETURN_IF_ERROR(GenerateMetadataType(
        stripped.base, output->mutable_return_type(), aliases_used));

    if (stripped.is_ref) {
      output->set_returns_reference(stripped.is_ref);
    }
  }

  if (clang::isa<clang::CXXMethodDecl>(func)) {
    auto method = clang::dyn_cast<clang::CXXMethodDecl>(func);
    output->set_is_method(true);

    if (!method->isStatic()) {
      const clang::QualType& this_ptr_type = method->getThisType();
      const clang::QualType& this_type = this_ptr_type->getPointeeType();
      output->set_is_const(this_type.isConstQualified());

      XLS_RETURN_IF_ERROR(GenerateMetadataType(
          this_type, output->mutable_this_type(), aliases_used));
    }
  }

  for (int64_t pi = 0; pi < func->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = func->getParamDecl(pi);
    xlscc_metadata::FunctionParameter* proto_param = output->add_params();
    proto_param->set_name(p->getNameAsString());

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_RETURN_IF_ERROR(GenerateMetadataType(
        stripped.base, proto_param->mutable_type(), aliases_used));

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
    auto p = clang::cast<const clang::ValueDecl>(namedecl);
    xlscc_metadata::FunctionValue* proto_static_value =
        output->add_static_values();
    XLS_RETURN_IF_ERROR(
        GenerateMetadataCPPName(namedecl, proto_static_value->mutable_name()));
    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));
    if (stripped.is_ref) {
      return absl::UnimplementedError(
          ErrorMessage(GetLoc(*p), "Metadata for static reference"));
    }
    XLS_RETURN_IF_ERROR(GenerateMetadataType(
        stripped.base, proto_static_value->mutable_type(), aliases_used));
    XLS_ASSIGN_OR_RETURN(
        std::shared_ptr<CType> ctype,
        TranslateTypeFromClang(stripped.base, xls::SourceInfo()));
    XLS_RETURN_IF_ERROR(ctype->GetMetadataValue(
        *this, initval, proto_static_value->mutable_value()));
    FillLocationRangeProto(namedecl->getSourceRange(),
                           proto_static_value->mutable_declaration_location());
  }
  // TODO: Add lvalues if found->second->this_lvalue != null
  return absl::OkStatus();
}

absl::StatusOr<xlscc_metadata::MetadataOutput> Translator::GenerateMetadata() {
  CHECK_NE(parser_.get(), nullptr);
  XLS_ASSIGN_OR_RETURN(const clang::FunctionDecl* top_function,
                       parser_->GetTopFunction());

  PushContextGuard temporary_this_context(*this, GetLoc(*top_function));

  // Don't allow "this" to be propagated up: it's only temporary for use
  // within the initializer list
  context().propagate_up = false;
  context().override_this_decl_ = top_function;
  context().ast_context = &top_function->getASTContext();

  xlscc_metadata::MetadataOutput ret;

  parser_->AddSourceInfoToMetadata(ret);

  absl::flat_hash_set<const clang::NamedDecl*> aliases_used_unordered;

  // Top function proto
  XLS_RETURN_IF_ERROR(GenerateFunctionMetadata(
      top_function, ret.mutable_top_func_proto(), aliases_used_unordered));

  for (auto const& [decl, xls_name] : xls_names_for_functions_generated_) {
    if (auto method_decl = clang::dyn_cast<clang::CXXMethodDecl>(decl);
        method_decl != nullptr) {
      auto type_alias = std::make_shared<CInstantiableTypeAlias>(
          static_cast<const clang::NamedDecl*>(method_decl->getParent()));
      auto found = inst_types_.find(type_alias);
      if (found == inst_types_.end()) {
        continue;
      }
      CHECK(found != inst_types_.end());
      auto struct_type = dynamic_cast<const CStructType*>(found->second.get());
      CHECK_NE(struct_type, nullptr);
      if (struct_type->synthetic_int_flag()) {
        continue;
      }
    }
    XLS_RETURN_IF_ERROR(GenerateFunctionMetadata(
        decl, ret.add_all_func_protos(), aliases_used_unordered));
  }

  std::list<const clang::NamedDecl*> aliases_used(
      aliases_used_unordered.begin(), aliases_used_unordered.end());

  aliases_used.sort(
      [](const clang::NamedDecl* a, const clang::NamedDecl* b) -> bool {
        return a->getQualifiedNameAsString() < b->getQualifiedNameAsString();
      });

  // Recurses
  absl::flat_hash_set<const clang::NamedDecl*> aliases_dummy;

  while (!aliases_used.empty()) {
    const clang::NamedDecl* alias = aliases_used.front();
    aliases_used.pop_front();

    auto temp_alias = std::make_shared<CInstantiableTypeAlias>(alias);
    // Ignore __xls_channel
    if (inst_types_.contains(temp_alias)) {
      std::shared_ptr<CType> type = inst_types_.at(temp_alias);
      auto ctype_as_struct = dynamic_cast<const CStructType*>(type.get());
      if (ctype_as_struct == nullptr) {
        continue;
      }
      xlscc_metadata::Type* struct_out = ret.add_structs();
      XLS_RETURN_IF_ERROR(temp_alias->GetMetadata(
          *this, struct_out->mutable_as_struct()->mutable_name(),
          aliases_dummy));
      XLS_RETURN_IF_ERROR(
          ctype_as_struct->GetMetadata(*this, struct_out, aliases_dummy));
    }
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

}  // namespace xlscc
