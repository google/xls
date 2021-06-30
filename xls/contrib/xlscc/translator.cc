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

#include <fstream>
#include <iostream>
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
#include "absl/synchronization/blocking_counter.h"
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
#include "clang/include/clang/Frontend/CompilerInstance.h"
#include "clang/include/clang/Frontend/FrontendActions.h"
#include "clang/include/clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/include/clang/Tooling/CommonOptionsParser.h"
#include "clang/include/clang/Tooling/Tooling.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/VirtualFileSystem.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/metadata_output.pb.h"
#include "xls/interpreter/ir_interpreter.h"
#include "xls/ir/bits.h"
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

absl::Status CType::GetMetadata(xlscc_metadata::Type* output) const {
  return absl::UnimplementedError(
      "GetMetadata unsupported in CType base class");
}

CVoidType::~CVoidType() {}

int CVoidType::GetBitWidth() const {
  XLS_CHECK(false);
  return 0;
}

absl::Status CVoidType::GetMetadata(xlscc_metadata::Type* output) const {
  (void)output->mutable_as_void();
  return absl::OkStatus();
}

CVoidType::operator std::string() const { return "void"; }

bool CVoidType::operator==(const CType& o) const {
  const auto* o_derived = dynamic_cast<const CVoidType*>(&o);
  return o_derived != nullptr;
}

CBitsType::CBitsType(int width) : width_(width) {}

CBitsType::~CBitsType() {}

int CBitsType::GetBitWidth() const { return width_; }

CBitsType::operator std::string() const {
  return absl::StrFormat("bits[%d]", width_);
}

absl::Status CBitsType::GetMetadata(xlscc_metadata::Type* output) const {
  output->mutable_as_bits()->set_width(width_);
  return absl::OkStatus();
}

bool CBitsType::StoredAsXLSBits() const { return true; }

bool CBitsType::operator==(const CType& o) const {
  const auto* o_derived = dynamic_cast<const CBitsType*>(&o);
  if (o_derived == nullptr) return false;
  return width_ == o_derived->width_;
}

CIntType::~CIntType() {}

CIntType::CIntType(int width, bool is_signed)
    : width_(width), is_signed_(is_signed) {}

xls::Type* CIntType::GetXLSType(xls::Package* package) const {
  return package->GetBitsType(width_);
}

bool CIntType::operator==(const CType& o) const {
  const auto* o_derived = dynamic_cast<const CIntType*>(&o);
  if (o_derived == nullptr) return false;
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
    return pre + "psuedobool";
  } else if (width_ == 64) {
    return pre + "int64_t";
  } else if (width_ == 16) {
    return pre + "short";
  } else if (width_ == 8) {
    return pre + "char";
  } else {
    XLS_CHECK(0);
    return "Unsupported";
  }
}

absl::Status CIntType::GetMetadata(xlscc_metadata::Type* output) const {
  output->mutable_as_int()->set_width(width_);
  output->mutable_as_int()->set_is_signed(is_signed_);
  return absl::OkStatus();
}

CBoolType::~CBoolType() {}

int CBoolType::GetBitWidth() const { return 1; }

CBoolType::operator std::string() const { return "bool"; }

absl::Status CBoolType::GetMetadata(xlscc_metadata::Type* output) const {
  (void)output->mutable_as_bool();
  return absl::OkStatus();
}

bool CBoolType::operator==(const CType& o) const {
  const auto* o_derived = dynamic_cast<const CBoolType*>(&o);
  return o_derived != nullptr;
}

bool CBoolType::StoredAsXLSBits() const { return true; }

CInstantiableTypeAlias::CInstantiableTypeAlias(const clang::NamedDecl* base)
    : base_(base) {}
const clang::NamedDecl* CInstantiableTypeAlias::base() const { return base_; }

CInstantiableTypeAlias::operator std::string() const {
  return absl::StrFormat("{%s}", base_->getNameAsString());
}

absl::Status CInstantiableTypeAlias::GetMetadata(
    xlscc_metadata::Type* output) const {
  output->mutable_as_inst()->mutable_name()->set_name(base_->getNameAsString());
  output->mutable_as_inst()->mutable_name()->set_fully_qualified_name(
      base_->getQualifiedNameAsString());
  output->mutable_as_inst()->mutable_name()->set_id(
      reinterpret_cast<uint64_t>(base_));
  return absl::OkStatus();
}

int CInstantiableTypeAlias::GetBitWidth() const {
  XLS_CHECK(false);
  return 0;
}

bool CInstantiableTypeAlias::operator==(const CType& o) const {
  const auto* o_derived = dynamic_cast<const CInstantiableTypeAlias*>(&o);
  // Different type
  if (o_derived == nullptr) {
    return false;
  }
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

absl::Status CStructType::GetMetadata(xlscc_metadata::Type* output) const {
  output->mutable_as_struct()->set_no_tuple(no_tuple_flag_);

  absl::flat_hash_map<
      int, std::pair<const clang::NamedDecl*, std::shared_ptr<CField>>>
      fields_by_index;

  for (std::pair<const clang::NamedDecl*, std::shared_ptr<CField>> field :
       fields_by_name_) {
    fields_by_index[field.second->index()] = field;
  }

  for (int i = 0; i < fields_by_name_.size(); ++i) {
    std::pair<const clang::NamedDecl*, std::shared_ptr<CField>> field =
        fields_by_index[i];
    XLS_RETURN_IF_ERROR(
        field.second->GetMetadata(output->mutable_as_struct()->add_fields()));
  }

  return absl::OkStatus();
}

bool CStructType::no_tuple_flag() const { return no_tuple_flag_; }

int CStructType::GetBitWidth() const {
  XLS_LOG(FATAL) << "GetBitWidth() unsupported on structs";
  return 0;
}

CStructType::operator std::string() const {
  std::ostringstream ostr;
  ostr << "{";
  if (no_tuple_flag_) {
    ostr << " notuple ";
  }
  for (const std::shared_ptr<CField>& it : fields_) {
    ostr << "[" << it->index() << "] " << it->name() << ": "
         << string(*it->type());
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
    : element_(element), size_(size) {}

bool CArrayType::operator==(const CType& o) const {
  const auto* o_derived = dynamic_cast<const CArrayType*>(&o);
  if (o_derived == nullptr) return false;
  return *element_ == *o_derived->element_ && size_ == o_derived->size_;
}

int CArrayType::GetBitWidth() const {
  XLS_LOG(FATAL) << "GetBitWidth() unsupported on native arrays";
  return 0;
}

absl::Status CField::GetMetadata(xlscc_metadata::StructField* output) const {
  output->set_name(name_->getNameAsString());
  XLS_RETURN_IF_ERROR(type_->GetMetadata(output->mutable_type()));
  return absl::OkStatus();
}

int CArrayType::GetSize() const { return size_; }

std::shared_ptr<CType> CArrayType::GetElementType() const { return element_; }

CArrayType::operator std::string() const {
  return absl::StrFormat("%s[%i]", string(*element_), size_);
}

absl::Status CArrayType::GetMetadata(xlscc_metadata::Type* output) const {
  output->mutable_as_array()->set_size(size_);
  XLS_RETURN_IF_ERROR(element_->GetMetadata(
      output->mutable_as_array()->mutable_element_type()));
  return absl::OkStatus();
}

Translator::Translator(int max_unroll_iters)
    : max_unroll_iters_(max_unroll_iters) {
  context_stack_.push(TranslationContext());
}

Translator::~Translator() {
  // Allow parser and its thread to be destroyed
  libtool_wait_for_destruct_->DecrementCount();
  // Wait for it to be gone
  libtool_thread_->Join();
}

TranslationContext& Translator::PushContext() {
  auto ocond = context().condition;
  context_stack_.push(context());
  context().original_condition = ocond;
  return context();
}

void Translator::PopContext(bool propagate_up, bool propagate_break_up,
                            bool propagate_continue_up,
                            const xls::SourceLocation& loc) {
  // Copy updated variables
  TranslationContext popped = context();
  context_stack_.pop();

  XLS_CHECK(!context_stack_.empty());

  if (propagate_up) {
    auto& cur_vars = context().variables;
    for (const auto& var_it : popped.variables) {
      const clang::NamedDecl* name = var_it.first;
      if (cur_vars.contains(name)) {
        cur_vars[name] = var_it.second;
      }
    }

    context().return_val = popped.return_val;
    context().last_return_condition = popped.last_return_condition;
    context().have_returned_condition = popped.have_returned_condition;
    context().mask_assignments = popped.mask_assignments;
    context().this_val = popped.this_val;

    if (context().in_fork) {
      context().forbidden_rvalues = popped.forbidden_rvalues;
    }

    if (popped.have_returned_condition.valid()) {
      context().and_condition(
          context().fb->Not(popped.have_returned_condition, loc), loc);
    }

    if (propagate_break_up && popped.break_condition.valid()) {
      context().and_condition(popped.break_condition, loc);
      context().break_condition = popped.break_condition;
    }
    if (propagate_continue_up && popped.continue_condition.valid()) {
      context().and_condition(popped.continue_condition, loc);
      context().continue_condition = popped.continue_condition;
    }
  }
}

TranslationContext& Translator::context() { return context_stack_.top(); }

bool Translator::LibToolVisitFunction(clang::FunctionDecl* func) {
  if (libtool_visit_status_.ok()) libtool_visit_status_ = VisitFunction(func);
  return libtool_visit_status_.ok();
}

Translator::LibToolThread::LibToolThread(
    absl::string_view source_filename,
    absl::Span<absl::string_view> command_line_args, Translator& translator)
    : source_filename_(source_filename),
      command_line_args_(command_line_args),
      translator_(translator) {}

void Translator::LibToolThread::Start() {
  thread_.emplace([this] { Run(); });
}

void Translator::LibToolThread::Join() { thread_->Join(); }

void Translator::LibToolThread::Run() {
  std::vector<std::string> argv;
  argv.emplace_back("binary");
  argv.emplace_back(source_filename_);
  for (const auto& view : command_line_args_) {
    argv.emplace_back(view);
  }
  argv.emplace_back("-fsyntax-only");
  argv.emplace_back("-std=c++17");
  argv.emplace_back("-nostdinc");

  llvm::IntrusiveRefCntPtr<clang::FileManager> libtool_files;

  std::unique_ptr<LibToolFrontendAction> libtool_action(
      new LibToolFrontendAction(translator_));

  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> mem_fs(
      new llvm::vfs::InMemoryFileSystem);
  mem_fs->addFile("/xls_builtin.h", 0,
                  llvm::MemoryBuffer::getMemBuffer(
                      R"(
#ifndef __XLS_BUILTIN_H
#define __XLS_BUILTIN_H
template<int N>
struct __xls_bits { };

template<typename T>
class __xls_channel {
 public:
  T read() {
    return T();
  }
  T write(T val) {
    return val;
  }
};

#endif//__XLS_BUILTIN_H
          )"));

  llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlay_fs(
      new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));

  overlay_fs->pushOverlay(mem_fs);

  libtool_files =
      new clang::FileManager(clang::FileSystemOptions(), overlay_fs);

  std::unique_ptr<clang::tooling::ToolInvocation> libtool_inv(
      new clang::tooling::ToolInvocation(argv, std::move(libtool_action),
                                         libtool_files.get()));

  llvm::IntrusiveRefCntPtr<clang::DiagnosticOptions> diag_opts =
      new clang::DiagnosticOptions();
  Translator::DiagnosticInterceptor diag_print(translator_, llvm::errs(),
                                               &*diag_opts);
  libtool_inv->setDiagnosticConsumer(&diag_print);

  // Errors are extracted via DiagnosticInterceptor,
  //  since we block in run() until ~Translator()
  (void)libtool_inv->run();
}

void Translator::LibToolFrontendAction::EndSourceFileAction() {
  // ToolInvocation::run() from returning until ~Translator()
  translator_.libtool_wait_for_parse_->DecrementCount();
  translator_.libtool_wait_for_destruct_->Wait();
}

absl::Status Translator::ScanFile(
    absl::string_view source_filename,
    absl::Span<absl::string_view> command_line_args) {
  // The AST is destroyed after ToolInvocation::run() returns
  //
  // However, we want to preserve it to access it across multiple passes and
  //  various subsequent calls, such as GenerateIR().
  //
  // Therefore, ToolInvocation::Run() is executed on another thread,
  //  and the ASTFrontendAction::EndSourceFileAction() blocks it
  //  until ~Translator(), preserving the AST.
  libtool_thread_ = absl::WrapUnique(
      new LibToolThread(source_filename, command_line_args, *this));

  libtool_wait_for_parse_ = absl::make_unique<absl::BlockingCounter>(1);
  libtool_wait_for_destruct_ = absl::make_unique<absl::BlockingCounter>(1);
  libtool_visit_status_ = absl::OkStatus();
  libtool_thread_->Start();
  libtool_wait_for_parse_->Wait();
  return libtool_visit_status_;
}

absl::StatusOr<std::string> Translator::GetEntryFunctionName() const {
  if (!top_function_) {
    return absl::NotFoundError("No top function found");
  }
  return top_function_->getNameAsString();
}

static size_t match_pragma(std::string pragma_string, std::string name) {
  size_t at = pragma_string.find(name);
  if (at == std::string::npos) return std::string::npos;
  size_t lcs = pragma_string.find("//");
  if (lcs != std::string::npos) {
    if (lcs < at) return std::string::npos;
  }
  size_t bcs = pragma_string.find("*/");
  if (bcs != std::string::npos) {
    if (bcs >= (at + name.length())) return std::string::npos;
  }
  return at;
}

absl::Status Translator::ScanFileForPragmas(std::string filename) {
  std::ifstream fin(filename);
  if (!fin.good()) {
    if (filename != "/xls_builtin.h") {
      return absl::NotFoundError(absl::StrFormat(
          "Unable to open file to scan for pragmas: %s\n", filename));
    }
  }
  int lineno = 1;
  for (std::string line; std::getline(fin, line); ++lineno) {
    size_t at;
    if ((at = line.find("#pragma")) != std::string::npos) {
      PragmaLoc location(filename, lineno);

      if ((at = match_pragma(line, "#pragma hls_no_tuple")) !=
          std::string::npos) {
        hls_pragmas_[location] = Pragma_NoTuples;
      } else if ((at = match_pragma(line, "#pragma hls_unroll yes")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma_Unroll;
      } else if ((at = match_pragma(line, "#pragma hls_top")) !=
                 std::string::npos) {
        hls_pragmas_[location] = Pragma_Top;
      }
    }
  }

  files_scanned_for_pragmas_.insert(filename);
  return absl::OkStatus();
}

absl::StatusOr<Translator::Pragma> Translator::FindPragmaForLoc(
    const clang::PresumedLoc& ploc) {
  if (!files_scanned_for_pragmas_.contains(ploc.getFilename())) {
    XLS_RETURN_IF_ERROR(ScanFileForPragmas(ploc.getFilename()));
  }
  // Look on the line before
  auto found =
      hls_pragmas_.find(PragmaLoc(ploc.getFilename(), ploc.getLine() - 1));
  if (found == hls_pragmas_.end()) return Pragma_Null;
  return found->second;
}

absl::Status Translator::AssignThis(const CValue& rvalue,
                                    const xls::SourceLocation& loc) {
  XLS_CHECK(context().this_val.value().valid());
  if (*context().this_val.type() != *rvalue.type()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Assignment to this with type %s which is not the class type %s at %s",
        string(*context().this_val.type()), string(*rvalue.type()),
        LocString(loc)));
  }
  XLS_ASSIGN_OR_RETURN(context().this_val,
                       PrepareRValueForAssignment(context().this_val, rvalue));

  return absl::OkStatus();
}

absl::StatusOr<xls::BValue> Translator::StructUpdate(
    xls::BValue struct_before, CValue rvalue,
    const clang::NamedDecl* field_name, const CStructType& stype,
    const xls::SourceLocation& loc) {
  const absl::flat_hash_map<const clang::NamedDecl*, std::shared_ptr<CField>>&
      fields_by_name = stype.fields_by_name();
  auto found_field = fields_by_name.find(field_name);
  if (found_field == fields_by_name.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Assignment to unknown field %s in type %s at %s",
        field_name->getNameAsString(), string(stype), LocString(loc)));
  }
  const CField& cfield = *found_field->second;

  if (*cfield.type() != *rvalue.type()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot assign rvalue of type %s to struct "
        "field of type %s at %s",
        string(*rvalue.type()), string(*cfield.type()), LocString(loc)));
  }

  // No tuple update, so we need to rebuild the tuple
  std::vector<xls::BValue> bvals;
  for (auto it = stype.fields().rbegin(); it != stype.fields().rend(); it++) {
    std::shared_ptr<CField> fp = *it;
    xls::BValue bval;
    if (fp->index() != cfield.index()) {
      bval = GetStructFieldXLS(struct_before, fp->index(), stype, loc);
    } else {
      bval = rvalue.value();
    }
    bvals.push_back(bval);
  }

  xls::BValue new_tuple = MakeStructXLS(bvals, stype, loc);

  return new_tuple;
}

xls::BValue Translator::MakeStructXLS(const std::vector<xls::BValue>& bvals,
                                      const CStructType& stype,
                                      const xls::SourceLocation& loc) {
  XLS_CHECK_EQ(bvals.size(), stype.fields().size());
  xls::BValue ret =
      stype.no_tuple_flag() ? bvals[0] : context().fb->Tuple(bvals, loc);
  return ret;
}
xls::BValue Translator::GetStructFieldXLS(xls::BValue val, int index,
                                          const CStructType& type,
                                          const xls::SourceLocation& loc) {
  XLS_CHECK_LT(index, type.fields().size());
  return type.no_tuple_flag() ? val
                              : context().fb->TupleIndex(
                                    val, type.fields().size() - 1 - index, loc);
}

absl::StatusOr<xls::Type*> Translator::GetStructXLSType(
    const std::vector<xls::Type*>& members, const CStructType& type,
    const xls::SourceLocation& loc) {
  if (type.no_tuple_flag() && members.size() != 1) {
    return absl::FailedPreconditionError(
        absl::StrFormat("Pragma hls_no_tuple must be used on structs with only "
                        "1 field, but %s has %i at %s\n",
                        string(type), members.size(), LocString(loc)));
  }
  return type.no_tuple_flag() ? members[0]
                              : context().package->GetTupleType(members);
}

xls::BValue Translator::MakeFlexTuple(const std::vector<xls::BValue>& bvals,
                                      const xls::SourceLocation& loc) {
  XLS_CHECK(!bvals.empty());
  return (bvals.size() == 1) ? bvals[0] : context().fb->Tuple(bvals, loc);
}

xls::BValue Translator::GetFlexTupleField(xls::BValue val, int index,
                                          int n_fields,
                                          const xls::SourceLocation& loc) {
  XLS_CHECK_GT(n_fields, 0);
  return (n_fields == 1) ? val : context().fb->TupleIndex(val, index, loc);
}

xls::Type* Translator::GetFlexTupleType(
    const std::vector<xls::Type*>& members) {
  XLS_CHECK(!members.empty());
  return (members.size() == 1) ? members[0]
                               : context().package->GetTupleType(members);
}

xls::BValue Translator::MakeFunctionReturn(
    const std::vector<xls::BValue>& bvals, const xls::SourceLocation& loc) {
  return MakeFlexTuple(bvals, loc);
}

xls::BValue Translator::UpdateFlexTupleField(xls::BValue tuple_val,
                                             xls::BValue new_val, int index,
                                             int n_fields,
                                             const xls::SourceLocation& loc) {
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
                                          const xls::SourceLocation& loc) {
  return GetFlexTupleField(val, index, expected_returns, loc);
}

clang::PresumedLoc Translator::GetPresumedLoc(clang::SourceManager& sm,
                                              const clang::Stmt& stmt) {
  return sm.getPresumedLoc(stmt.getSourceRange().getBegin());
}

xls::SourceLocation Translator::GetLoc(clang::SourceManager& sm,
                                       const clang::Stmt& stmt) {
  return GetLoc(GetPresumedLoc(sm, stmt));
}

clang::PresumedLoc Translator::GetPresumedLoc(const clang::Decl& decl) {
  clang::SourceManager& sm = decl.getASTContext().getSourceManager();
  return sm.getPresumedLoc(decl.getSourceRange().getBegin());
}

xls::SourceLocation Translator::GetLoc(const clang::Decl& decl) {
  return GetLoc(GetPresumedLoc(decl));
}

xls::SourceLocation Translator::GetLoc(clang::SourceManager& sm,
                                       const clang::Expr& expr) {
  return GetLoc(sm.getPresumedLoc(expr.getExprLoc()));
}

xls::SourceLocation Translator::GetLoc(const clang::PresumedLoc& loc) {
  if (!loc.isValid()) {
    return xls::SourceLocation(xls::Fileno(-1), xls::Lineno(-1),
                               xls::Colno(-1));
  }

  auto found = file_numbers_.find(loc.getFilename());

  int id = 0;

  if (found == file_numbers_.end()) {
    id = next_file_number_++;
    file_numbers_[loc.getFilename()] = id;
  } else {
    id = found->second;
  }

  return xls::SourceLocation(xls::Fileno(id), xls::Lineno(loc.getLine()),
                             xls::Colno(loc.getColumn()));
}

std::string Translator::LocString(const xls::SourceLocation& loc) {
  std::string found_str = "Unknown";
  for (const auto& it : file_numbers_) {
    if (it.second == static_cast<int>(loc.fileno())) {
      found_str = it.first;
      break;
    }
  }
  return absl::StrFormat("%s:%i:%i", found_str, static_cast<int>(loc.lineno()),
                         static_cast<int>(loc.fileno()));
}

// Scans for top-level function candidates
absl::Status Translator::VisitFunction(const clang::FunctionDecl* funcdecl) {
  const std::string fname = funcdecl->getNameAsString();

  // Top can't be a template function
  if (funcdecl->getTemplatedKind() !=
      clang::FunctionDecl::TemplatedKind::TK_NonTemplate) {
    return absl::OkStatus();
  }

  // Top can't be a forward declarations
  if (!funcdecl->doesThisDeclarationHaveABody()) {
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(Pragma pragma,
                       FindPragmaForLoc(GetPresumedLoc(*funcdecl)));

  if (pragma == Pragma_Top || fname == top_function_name_) {
    if (top_function_) {
      return absl::AlreadyExistsError(absl::StrFormat(
          "Two top functions defined, at %s, previously at %s",
          LocString(GetLoc(*top_function_)), LocString(GetLoc(*funcdecl))));
    }
    top_function_ = funcdecl;
  }

  return absl::OkStatus();
}

std::string Translator::XLSNameMangle(clang::GlobalDecl decl) const {
  std::string res;
  llvm::raw_string_ostream os(res);
  if (!mangler_)
    mangler_.reset(decl.getDecl()->getASTContext().createMangleContext());
  mangler_->mangleCXXName(decl, os);
  return res;
}

absl::Status Translator::AddOpToChannel(IOOp op,
                                        const xls::SourceLocation& loc) {
  // Scan channel
  if (!context().sf->io_channels.contains(op.channel)) {
    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(op.channel->getType()));

    if (stripped.base->getTypeClass() !=
        clang::Type::TypeClass::TemplateSpecialization) {
      return absl::UnimplementedError(absl::StrFormat(
          "Channel type should be a template specialization at %s",
          LocString(loc)));
    }

    auto template_spec =
        clang_down_cast<const clang::TemplateSpecializationType*>(
            stripped.base.getTypePtr());

    if (template_spec->getNumArgs() != 1) {
      return absl::UnimplementedError(absl::StrFormat(
          "Channel should have 1 template args at %s", LocString(loc)));
    }

    const clang::TemplateArgument& arg = template_spec->getArg(0);

    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                         TranslateTypeFromClang(arg.getAsType(), loc));

    context().sf->io_channels[op.channel].item_type = ctype;
    context().sf->io_channels[op.channel].total_ops = 0;
    context().sf->io_channels[op.channel].channel_op_type = op.op;
  }

  op.channel_tuple_index = context().sf->io_channels[op.channel].total_ops++;

  if (context().sf->io_channels[op.channel].channel_op_type != op.op) {
    return absl::UnimplementedError(absl::StrFormat(
        "Channels should be either input or output at %s", LocString(loc)));
  }

  context().sf->io_ops.push_back(op);
  context().sf->op_by_call[op.io_call].ops.push_back(
      &context().sf->io_ops.back());
  context().sf->op_by_call[op.io_call].generated_ops = 0;

  return absl::OkStatus();
}

absl::StatusOr<CValue> Translator::GetChannelInputValue(
    const IOOp& op, const xls::SourceLocation& loc) {
  auto ch_id = absl::implicit_cast<const clang::NamedDecl*>(op.channel);

  XLS_ASSIGN_OR_RETURN(CValue param_val, GetIdentifier(ch_id, loc));

  const int ch_idx = op.channel_tuple_index;

  XLS_CHECK(context().sf->io_channels.contains(op.channel));

  const IOChannel& channel_info = context().sf->io_channels[op.channel];

  xls::BValue val =
      GetFlexTupleField(param_val.value(), ch_idx, channel_info.total_ops, loc);
  return CValue(val, param_val.type());
}

absl::StatusOr<Translator::IOOpReturn> Translator::InterceptIOOp(
    const clang::Expr* expr, const xls::SourceLocation& loc) {
  if (expr->getStmtClass() == clang::Stmt::CXXMemberCallExprClass) {
    auto member_call = clang_down_cast<const clang::CXXMemberCallExpr*>(expr);
    const clang::Expr* object = member_call->getImplicitObjectArgument();

    XLS_ASSIGN_OR_RETURN(bool is_channel, ExprIsChannel(object, loc));
    if (is_channel) {
      if (object->getStmtClass() != clang::Stmt::DeclRefExprClass) {
        return absl::UnimplementedError(absl::StrFormat(
            "IO ops should be on direct DeclRefs at %s", LocString(loc)));
      }
      auto object_ref = clang_down_cast<const clang::DeclRefExpr*>(object);
      if (object_ref->getDecl()->getKind() != clang::Decl::ParmVar) {
        return absl::UnimplementedError(absl::StrFormat(
            "IO ops should be on channel parameters at %s", LocString(loc)));
      }

      const clang::FunctionDecl* funcdecl = member_call->getDirectCallee();
      const std::string op_name = funcdecl->getNameAsString();
      auto channel =
          clang_down_cast<const clang::ParmVarDecl*>(object_ref->getDecl());
      xls::BValue op_condition = context().condition_bval(loc);
      XLS_CHECK(op_condition.valid());

      auto call = clang_down_cast<const clang::CallExpr*>(expr);
      XLS_CHECK(context().sf->op_by_call.contains(call));
      IOCall& io_call = context().sf->op_by_call[call];

      IOOpReturn ret;
      ret.generate_expr = false;

      // Don't double-increment in unsequenced assignment detection
      if (context().mask_assignments) {
        XLS_CHECK(context().sf->io_channels.contains(channel));
        const IOChannel& io_channel = context().sf->io_channels[channel];
        XLS_ASSIGN_OR_RETURN(xls::BValue val,
                             CreateDefaultValue(io_channel.item_type, loc));
        ret.value = CValue(val, io_channel.item_type);
        return ret;
      }

      XLS_CHECK(io_call.generated_ops >= 0 &&
                io_call.generated_ops < io_call.ops.size());

      IOOp& op = *io_call.ops[io_call.generated_ops];

      ++io_call.generated_ops;

      XLS_CHECK(op_condition.valid());

      XLS_CHECK(channel == op.channel);

      if (op_name == "read") {
        XLS_ASSIGN_OR_RETURN(ret.value, GetChannelInputValue(op, loc));

        // Unsequenced error detection will regenerate, so just take the first
        // value
        if (!op.ret_value.valid()) {
          op.ret_value = op_condition;
        }
      } else if (op_name == "write") {
        if (call->getNumArgs() != 1) {
          return absl::UnimplementedError(absl::StrFormat(
              "IO write() should have one argument at %s", LocString(loc)));
        }

        // Unsequenced error detection will regenerate, so just take the first
        // value
        if (!op.ret_value.valid()) {
          XLS_ASSIGN_OR_RETURN(CValue out_val,
                               GenerateIR_Expr(call->getArg(0), loc));
          std::vector<xls::BValue> sp = {out_val.value(), op_condition};
          op.ret_value = context().fb->Tuple(sp);
        }
      } else {
        return absl::UnimplementedError(
            absl::StrFormat("Unsupported IO op: %s", op_name));
      }

      return ret;
    }
  }

  IOOpReturn ret;
  ret.generate_expr = true;
  return ret;
}

absl::Status Translator::TranslateIOOps(const clang::CallExpr* call,
                                        const clang::FunctionDecl* callee,
                                        const std::list<IOOp>& callee_ops,
                                        clang::SourceManager& sm) {
  for (const IOOp& callee_op : callee_ops) {
    int found_channel_index = -1;
    for (int param = 0; param < callee->getNumParams(); ++param) {
      if (callee_op.channel == callee->getParamDecl(param)) {
        found_channel_index = param;
        break;
      }
    }
    XLS_CHECK(found_channel_index >= 0);
    XLS_CHECK(found_channel_index < call->getNumArgs());
    const clang::Expr* call_arg = call->getArg(found_channel_index);
    if (call_arg->getStmtClass() != clang::Stmt::DeclRefExprClass) {
      return absl::UnimplementedError(
          absl::StrFormat("IO operations should be DeclRefs"));
    }
    auto decl_ref_arg = clang_down_cast<const clang::DeclRefExpr*>(call_arg);
    if (decl_ref_arg->getDecl()->getKind() != clang::Decl::ParmVar) {
      return absl::UnimplementedError(
          absl::StrFormat("IO operations should be on parameters"));
    }
    IOOp caller_op;

    caller_op.channel =
        clang_down_cast<const clang::ParmVarDecl*>(decl_ref_arg->getDecl());
    caller_op.io_call = call;
    caller_op.op = callee_op.op;
    caller_op.sub_op = &callee_op;

    XLS_RETURN_IF_ERROR(AddOpToChannel(caller_op, GetLoc(sm, *call)));
  }
  return absl::OkStatus();
}

absl::Status Translator::DeepScanForIO(const clang::Stmt* body,
                                       clang::ASTContext& ctx) {
  if (!body) {
    // Empty block, nothing to do
    return absl::OkStatus();
  }

  if (body->getStmtClass() == clang::Stmt::CompoundStmtClass) {
    for (auto it = body->children().begin(); it != body->children().end();
         ++it) {
      const clang::Stmt* body_st = *it;
      XLS_RETURN_IF_ERROR(DeepScanForIO(body_st, ctx));
    }
  } else {
    xls::SourceLocation body_loc = GetLoc(ctx.getSourceManager(), *body);

    // For single-line bodies
    // Check if this is an IO op
    if (body->getStmtClass() == clang::Stmt::CXXMemberCallExprClass) {
      auto member_call = clang_down_cast<const clang::CXXMemberCallExpr*>(body);
      const clang::Expr* object = member_call->getImplicitObjectArgument();

      XLS_ASSIGN_OR_RETURN(bool is_channel, ExprIsChannel(object, body_loc));

      if (is_channel) {
        if (object->getStmtClass() != clang::Stmt::DeclRefExprClass) {
          return absl::UnimplementedError(
              "IO ops should be on direct DeclRefs");
        }
        auto object_ref = clang_down_cast<const clang::DeclRefExpr*>(object);
        if (object_ref->getDecl()->getKind() != clang::Decl::ParmVar) {
          return absl::UnimplementedError(
              "IO ops should be on channel parameters");
        }

        const clang::FunctionDecl* funcdecl = member_call->getDirectCallee();
        const std::string op_name = funcdecl->getNameAsString();
        IOOp op;
        op.io_call = clang_down_cast<const clang::CallExpr*>(body);
        op.channel =
            clang_down_cast<const clang::ParmVarDecl*>(object_ref->getDecl());
        op.sub_op = nullptr;

        if (op_name == "read") {
          op.op = OpType::kRecv;
        } else if (op_name == "write") {
          op.op = OpType::kSend;
        } else {
          return absl::UnimplementedError(
              absl::StrFormat("Unsupported IO op: %s", op_name));
        }

        XLS_RETURN_IF_ERROR(AddOpToChannel(op, body_loc));
      }
    }

    // Add IO ops from functions / methods called
    if ((body->getStmtClass() == clang::Stmt::CXXMemberCallExprClass) ||
        (body->getStmtClass() == clang::Stmt::CallExprClass)) {
      auto call = clang_down_cast<const clang::CallExpr*>(body);

      const clang::FunctionDecl* definition = call->getDirectCallee();

      XLS_ASSIGN_OR_RETURN(GeneratedFunction * func,
                           TranslateFunctionToXLS(definition));

      // Duplication per-call is intended
      if (!func->io_ops.empty()) {
        XLS_RETURN_IF_ERROR(TranslateIOOps(call, definition, func->io_ops,
                                           ctx.getSourceManager()));
      }
    }

    if (body->getStmtClass() == clang::Stmt::ForStmtClass) {
      auto forst = clang_down_cast<const clang::ForStmt*>(body);

      // Don't generate double declared error when actually generating IR
      auto saved_check_unique_ids = check_unique_ids_;

      PushContextGuard context_guard(*this, body_loc);
      context_guard.propagate_up = false;

      XLS_RETURN_IF_ERROR(GenerateIR_UnrolledFor(forst, ctx, body_loc, true));

      check_unique_ids_ = saved_check_unique_ids;
    } else {
      // Recurse into sub-statements, sub-expressions
      for (const clang::Stmt* child : body->children()) {
        XLS_RETURN_IF_ERROR(DeepScanForIO(child, ctx));
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<GeneratedFunction*> Translator::GenerateIR_Function(
    const clang::FunctionDecl* funcdecl, absl::string_view name_override,
    bool force_static) {
  const bool trivial = funcdecl->hasTrivialBody() || funcdecl->isTrivial();

  if (!trivial && !funcdecl->getBody()) {
    return absl::UnimplementedError(
        absl::StrFormat("Function %s used but has no body at %s",
                        funcdecl->getNameAsString(), LocString(*funcdecl)));
  }

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

  xls::FunctionBuilder builder(xls_name, context().package);

  TranslationContext& prev_context = context();
  PushContextGuard context_guard(*this, GetLoc(*funcdecl));
  context_guard.propagate_up = false;

  auto signature = absl::implicit_cast<const clang::NamedDecl*>(funcdecl);

  inst_functions_[signature] = absl::make_unique<GeneratedFunction>();
  GeneratedFunction& sf = *inst_functions_[signature];

  // Functions need a clean context
  context() = TranslationContext();
  context().package = prev_context.package;
  context().fb = absl::implicit_cast<xls::BuilderBase*>(&builder);
  context().sf = &sf;

  XLS_ASSIGN_OR_RETURN(
      context().return_type,
      TranslateTypeFromClang(funcdecl->getReturnType(), GetLoc(*funcdecl)));

  // If add_this_return is true, then a return value is added for the
  //  "this" object, pointed to be the "this" pointer in methods
  bool add_this_return = false;
  vector<const clang::NamedDecl*> ref_returns;

  // funcdecl parameters may be different for forward declarations
  const clang::FunctionDecl* definition = nullptr;
  const clang::Stmt* body = funcdecl->getBody(definition);
  if (definition == nullptr) {
    XLS_CHECK(trivial);
    definition = funcdecl;
  }
  xls::SourceLocation body_loc = GetLoc(*definition);

  // Scan for IO in order to add extra params for IO reads
  // The output is added to *context().sf
  XLS_RETURN_IF_ERROR(DeepScanForIO(body, definition->getASTContext()));

  if (!sf.io_ops.empty() && funcdecl->isOverloadedOperator()) {
    return absl::UnimplementedError(
        absl::StrFormat("IO ops in operator calls are not supported at %s",
                        LocString(body_loc)));
  }

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

      XLS_ASSIGN_OR_RETURN(auto thisctype, TranslateTypeFromClang(q, body_loc));
      XLS_ASSIGN_OR_RETURN(auto xls_type,
                           TranslateTypeToXLS(thisctype, body_loc));

      // Don't use AssignThis(), since this is the "declaration"
      context().this_val = CValue(
          context().fb->Param("this", xls_type, body_loc),
          thisctype);
    }
  }

  for (const clang::ParmVarDecl* p : definition->parameters()) {
    auto namedecl = absl::implicit_cast<const clang::NamedDecl*>(p);

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> obj_type,
                         TranslateTypeFromClang(stripped.base, GetLoc(*p)));

    xls::Type* xls_type = nullptr;

    XLS_ASSIGN_OR_RETURN(bool is_channel,
                         TypeIsChannel(p->getType(), GetLoc(*p)));
    if (is_channel) {
      std::vector<xls::Type*> member_types;

      for (const IOOp& op : sf.io_ops) {
        if (op.channel == p && op.op == OpType::kRecv) {
          XLS_CHECK(sf.io_channels.contains(op.channel));
          const IOChannel& scanned_ch = sf.io_channels[op.channel];
          XLS_ASSIGN_OR_RETURN(
              xls::Type * xls_item_type,
              TranslateTypeToXLS(scanned_ch.item_type, GetLoc(*op.channel)));

          member_types.push_back(xls_item_type);
        }
      }

      if (member_types.empty()) {
        // Don't add a parameter if there are no inputs for the channel
        continue;
      }

      if (context().sf->io_channels.contains(p)) {
        const IOChannel& io_ch = context().sf->io_channels[p];

        xls_type = GetFlexTupleType(member_types);
        obj_type = io_ch.item_type;
      }
    } else {
      // Const references don't need a return
      if (stripped.is_ref && (!stripped.base.isConstQualified())) {
        ref_returns.push_back(namedecl);
      }
    }

    if (xls_type == nullptr) {
      XLS_ASSIGN_OR_RETURN(xls_type, TranslateTypeToXLS(obj_type, GetLoc(*p)));
    }

    std::string safe_param_name = namedecl->getNameAsString();
    if (safe_param_name.empty()) safe_param_name = "implicit";

    xls::BValue pbval = context().fb->Param(
        safe_param_name, xls_type, body_loc);

    // Create CValue without type check
    XLS_RETURN_IF_ERROR(
        DeclareVariable(namedecl, CValue(pbval, obj_type, true), body_loc));
  }

  // Generate constructor initializers
  if (funcdecl->getKind() == clang::FunctionDecl::Kind::CXXConstructor) {
    auto constructor =
        clang_down_cast<const clang::CXXConstructorDecl*>(funcdecl);
    XLS_CHECK(add_this_return);
    XLS_ASSIGN_OR_RETURN(auto resolved_type,
                         ResolveTypeInstance(context().this_val.type()));
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
      indices_to_update[found->second->index()] = rvalue.value();
    }

    std::vector<xls::BValue> bvals;
    for (auto it = struct_type->fields().rbegin();
         it != struct_type->fields().rend(); it++) {
      std::shared_ptr<CField> field = *it;
      auto found = indices_to_update.find(field->index());
      xls::BValue bval;
      if (found != indices_to_update.end()) {
        bval = found->second;
      } else {
        bval = GetStructFieldXLS(context().this_val.value(), field->index(),
                                 *struct_type, GetLoc(*constructor));
      }
      bvals.push_back(bval);
    }

    // Don't use AssignThis(), since this is the "declaration"
    context().this_val =
        CValue(MakeStructXLS(bvals, *struct_type, GetLoc(*constructor)),
               context().this_val.type());
  }

  if (body != nullptr) {
    XLS_RETURN_IF_ERROR(GenerateIR_Compound(body, funcdecl->getASTContext()));
  } else {
    XLS_CHECK(trivial);
  }

  vector<xls::BValue> return_bvals;

  // First this return
  if (add_this_return) {
    return_bvals.push_back(context().this_val.value());
  }

  // Then explicit return
  if (!funcdecl->getReturnType()->isVoidType()) {
    return_bvals.emplace_back(context().return_val);
  }

  // Then reference parameter returns
  for (const clang::NamedDecl* ret_ident : ref_returns) {
    XLS_ASSIGN_OR_RETURN(CValue found, GetIdentifier(ret_ident, body_loc));
    return_bvals.emplace_back(found.value());
  }

  // IO returns
  for (const IOOp& op : sf.io_ops) {
    XLS_CHECK(op.ret_value.valid());
    return_bvals.push_back(op.ret_value);
  }

  if (return_bvals.empty()) {
    return absl::UnimplementedError(
        absl::StrFormat("Function %s has no outputs at %s",
                        funcdecl->getNameAsString(), LocString(*funcdecl)));
  }

  if (return_bvals.size() == 1) {
    // XLS functions return the last value added to the FunctionBuilder
    // So this makes sure the correct value is last.
    context().return_val = return_bvals[0];
  } else {
    context().return_val = MakeFunctionReturn(return_bvals, body_loc);
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

    XLS_CHECK(cxx_record != nullptr);

    for (auto base : cxx_record->bases()) {
      const clang::RecordDecl* base_struct = base.getType()->getAsRecordDecl();
      XLS_ASSIGN_OR_RETURN(
          auto field_type,
          TranslateTypeFromClang(
              base_struct->getTypeForDecl()->getCanonicalTypeInternal(),
              GetLoc(*base_struct)));

      fields.push_back(std::shared_ptr<CField>(
          new CField(absl::implicit_cast<const clang::NamedDecl*>(base_struct),
                     fields.size(), field_type)));
    }

    for (const clang::FieldDecl* it : sd->fields()) {
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> field_type,
                           TranslateTypeFromClang(
                               it->getType(), GetLoc(*it->getCanonicalDecl())));

      // Up cast FieldDecl to NamedDecl because NamedDecl pointers are used to
      //  track identifiers by XLS[cc], no matter the type of object being
      //  identified
      fields.push_back(std::shared_ptr<CField>(
          new CField(absl::implicit_cast<const clang::NamedDecl*>(it),
                     fields.size(), field_type)));
    }

    XLS_ASSIGN_OR_RETURN(Pragma pragma, FindPragmaForLoc(GetPresumedLoc(*sd)));
    new_type.reset(new CStructType(fields, pragma == Pragma_NoTuples));
  }

  inst_types_[signature] = new_type;

  return absl::OkStatus();
}

absl::StatusOr<xls::Op> Translator::XLSOpcodeFromClang(
    clang::BinaryOperatorKind clang_op, const CType& left_type,
    const CType& result_type, const xls::SourceLocation& loc) {
  if (auto result_int_type = dynamic_cast<const CIntType*>(&result_type)) {
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
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented binary operator %i for result %s at %s", clang_op,
            std::string(result_type), LocString(loc)));
    }
  }
  if (dynamic_cast<const CBoolType*>(&result_type) != nullptr) {
    if (auto input_int_type = dynamic_cast<const CIntType*>(&left_type)) {
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
        return xls::Op::kAnd;
      case clang::BinaryOperatorKind::BO_LOr:
        return xls::Op::kOr;
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented binary operator %i for result %s with input %s at "
            "%s",
            clang_op, std::string(result_type), std::string(left_type),
            LocString(loc)));
    }
  } else {
    return absl::UnimplementedError(
        absl::StrFormat("Binary operators on types other than builtin-int "
                        "unimplemented for type %s at %s",
                        std::string(result_type), LocString(loc)));
  }
}

absl::StatusOr<CValue> Translator::TranslateVarDecl(
    const clang::VarDecl* decl, const xls::SourceLocation& loc) {
  const clang::Expr* initializer = decl->getAnyInitializer();

  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                       TranslateTypeFromClang(decl->getType(), loc));
  xls::BValue init_val;
  if (initializer) {
    // Sometimes the InitListExpr is hidden by being inside of an
    // ExprWithCleanups
    if (initializer->getStmtClass() == clang::Stmt::ExprWithCleanupsClass) {
      initializer = clang_down_cast<const clang::ExprWithCleanups*>(initializer)
                        ->getSubExpr();
    }

    if (initializer->getStmtClass() == clang::Stmt::InitListExprClass) {
      XLS_ASSIGN_OR_RETURN(
          init_val,
          CreateInitValue(
              *ctype, clang_down_cast<const clang::InitListExpr*>(initializer),
              loc));
    } else {
      XLS_ASSIGN_OR_RETURN(CValue cv, GenerateIR_Expr(initializer, loc));
      XLS_ASSIGN_OR_RETURN(init_val, GenTypeConvert(cv, ctype, loc));
    }
  } else {
    XLS_ASSIGN_OR_RETURN(init_val, CreateDefaultValue(ctype, loc));
  }

  return CValue(init_val, ctype);
}

absl::StatusOr<CValue> Translator::GetIdentifier(const clang::NamedDecl* decl,
                                                 const xls::SourceLocation& loc,
                                                 bool for_lvalue) {
  auto found = context().variables.find(decl);
  if (found == context().variables.end()) {
    // Is it an enum?
    auto enum_decl = dynamic_cast<const clang::EnumConstantDecl*>(decl);
    // Is this static/global?
    auto var_decl = dynamic_cast<const clang::VarDecl*>(decl);

    XLS_CHECK(!(enum_decl && var_decl));

    if (var_decl == nullptr && enum_decl == nullptr) {
      return absl::NotFoundError(
          absl::StrFormat("Undeclared identifier %s at %s",
                          decl->getNameAsString(), LocString(loc)));
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

    const xls::SourceLocation global_loc = GetLoc(*name);

    CValue value;

    if (enum_decl != nullptr) {
      const llvm::APSInt& aps = enum_decl->getInitVal();

      std::shared_ptr<CType> type(std::make_shared<CIntType>(32, true));
      xls::BValue bval =
          context().fb->Literal(xls::SBits(aps.getExtValue(), 32), global_loc);

      value = CValue(bval, type);
    } else {
      XLS_CHECK(var_decl->hasGlobalStorage());

      XLS_CHECK(context().fb);

      if (var_decl->getInit() != nullptr) {
        XLS_ASSIGN_OR_RETURN(value,
                             GenerateIR_Expr(var_decl->getInit(), global_loc));
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
  if (!for_lvalue) {
    if (context().forbidden_rvalues.contains(decl)) {
      return absl::UnimplementedError(
          absl::StrFormat("Declref to variable %s assigned in same context "
                          "(unsequenced in an expression?) at %s",
                          decl->getNameAsString(), LocString(loc)));
    }
  }
  return found->second;
}

absl::StatusOr<CValue> Translator::PrepareRValueForAssignment(
    const CValue& lvalue, const CValue& rvalue) {
  CValue rvalue_to_use = lvalue;
  if (!context().mask_assignments) {
    rvalue_to_use = rvalue;

    if (context().condition.valid() && (!context().unconditional_assignment)) {
      auto cond_sel = context().fb->Select(context().condition, rvalue.value(),
                                           lvalue.value());

      rvalue_to_use = CValue(cond_sel, rvalue_to_use.type());
    }
  }
  return rvalue_to_use;
}

absl::Status Translator::Assign(const clang::NamedDecl* lvalue,
                                const CValue& rvalue,
                                const xls::SourceLocation& loc) {
  // Don't allow assignment to globals. This doesn't work because
  //  each function has a different FunctionBuilder.
  if (auto var_decl = dynamic_cast<const clang::VarDecl*>(lvalue);
      var_decl != nullptr && var_decl->hasGlobalStorage()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Assignments to global variables not supported for %s at %s",
        lvalue->getNameAsString(), LocString(loc)));
  }

  XLS_ASSIGN_OR_RETURN(CValue found, GetIdentifier(lvalue, loc, true));
  if (*found.type() != *rvalue.type()) {
    lvalue->dump();
    return absl::InvalidArgumentError(absl::StrFormat(
        "Cannot assign rvalue of type %s to lvalue of type %s at %s",
        std::string(*rvalue.type()), std::string(*found.type()),
        LocString(loc)));
  }
  if (context().in_fork) {
    context().forbidden_rvalues.insert(lvalue);
  }

  if (context().forbidden_lvalues.contains(lvalue)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Assignment to lvalue %s is forbidden in this context at %s",
        lvalue->getNameAsString(), LocString(loc)));
  }

  XLS_ASSIGN_OR_RETURN(context().variables[lvalue],
                       PrepareRValueForAssignment(found, rvalue));

  return absl::OkStatus();
}

absl::Status Translator::Assign(const clang::Expr* lvalue, const CValue& rvalue,
                                const xls::SourceLocation& loc) {
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

      // Avoid decay of array to pointer, pointers are unsupported
      if (from_arr_type && cast->getType()->isPointerType()) {
        return Assign(cast->getSubExpr(), rvalue, loc);
      }

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
            StructUpdate(sub.value(), rvalue, inheritance.base_field_name,
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
      PushContextGuard fork_guard(*this, loc);
      context().in_fork = true;

      auto* cast = clang_down_cast<const clang::ArraySubscriptExpr*>(lvalue);
      XLS_ASSIGN_OR_RETURN(CValue arr_val,
                           GenerateIR_Expr(cast->getLHS(), loc));
      auto arr_type = dynamic_cast<CArrayType*>(arr_val.type().get());
      XLS_CHECK(arr_type != nullptr);
      XLS_ASSIGN_OR_RETURN(CValue idx_val,
                           GenerateIR_Expr(cast->getRHS(), loc));
      if (*rvalue.type() != *arr_type->GetElementType()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Cannot assign rvalue of type %s to element of array of type %s at "
            "%s",
            std::string(*rvalue.type()),
            std::string(*arr_type->GetElementType()), LocString(loc)));
      }
      auto arr_rvalue =
          CValue(context().fb->ArrayUpdate(arr_val.value(), rvalue.value(),
                                           {idx_val.value()}, loc),
                 arr_val.type());
      return Assign(cast->getLHS(), arr_rvalue, loc);
    }
    // Assign to a struct element
    // (...).member = rvalue
    case clang::Stmt::MemberExprClass: {
      const clang::MemberExpr* cast =
          clang_down_cast<const clang::MemberExpr*>(lvalue);
      clang::ValueDecl* member = cast->getMemberDecl();

      if (member->getKind() != clang::ValueDecl::Kind::Field) {
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented assignment to lvalue member kind %s at %s",
            member->getDeclKindName(), LocString(loc)));
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
            StructUpdate(struct_prev_val.value(), rvalue,
                         // Up cast to NamedDecl because NamedDecl pointers
                         //  are used to track identifiers
                         absl::implicit_cast<const clang::NamedDecl*>(field),
                         *sitype, loc));

        auto newval = CValue(new_tuple, struct_prev_val.type());
        return Assign(cast->getBase(), newval, loc);
      } else {
        return absl::UnimplementedError(
            absl::StrFormat("Unimplemented fielddecl assignment to "
                            "non-struct typed lvalue of type %s at %s",
                            string(*struct_prev_val.type()), LocString(loc)));
      }
    }
    case clang::Stmt::UnaryOperatorClass: {
      auto uop = clang_down_cast<const clang::UnaryOperator*>(lvalue);
      // Deref is the pointer dereferencing operator: *ptr
      // We simply ignore this for "this", so *this just evaluates to the
      //  "this" BValue from the TranslationContext
      if ((uop->getOpcode() == clang::UnaryOperatorKind::UO_Deref) &&
          (uop->getSubExpr()->getStmtClass() ==
           clang::Stmt::CXXThisExprClass)) {
        return AssignThis(rvalue, loc);

      } else {
        return absl::UnimplementedError(
            absl::StrFormat("Unimplemented assignment to unary operator lvalue "
                            "with opcode %i at %s",
                            uop->getOpcode(), LocString(loc)));
      }
    }
    case clang::Stmt::CXXThisExprClass: {
      return AssignThis(rvalue, loc);
    }
    default: {
      lvalue->dump();
      return absl::UnimplementedError(
          absl::StrFormat("Unimplemented assignment to lvalue of type %s at %s",
                          lvalue->getStmtClassName(), LocString(loc)));
    }
  }
}

absl::Status Translator::DeclareVariable(const clang::NamedDecl* lvalue,
                                         const CValue& rvalue,
                                         const xls::SourceLocation& loc) {
  if (context().variables.contains(lvalue)) {
    return absl::UnimplementedError(
        absl::StrFormat("Declaration '%s' duplicated at %s\n",
                        lvalue->getNameAsString(), LocString(loc)));
  }
  if (check_unique_ids_.contains(lvalue)) {
    return absl::InternalError(
        absl::StrFormat("Code assumes NamedDecls are unique, but %s isn't",
                        lvalue->getNameAsString()));
  }
  check_unique_ids_.insert(lvalue);
  context().variables[lvalue] = rvalue;
  return absl::OkStatus();
}

absl::StatusOr<CValue> Translator::Generate_Synthetic_ByOne(
    xls::Op xls_op, bool is_pre, CValue sub_value, const clang::Expr* sub_expr,
    const xls::SourceLocation& loc) {
  auto sub_type = std::dynamic_pointer_cast<CIntType>(sub_value.type());
  const int width = sub_type->width();
  xls::BValue literal_one = context().fb->Literal(
      sub_type->is_signed() ? xls::SBits(1, width) : xls::UBits(1, width));
  xls::BValue result_val =
      context().fb->AddBinOp(xls_op, sub_value.value(), literal_one, loc);
  // No extra bits because this is only for built-ins
  std::shared_ptr<CType> result_type = sub_value.type();

  XLS_RETURN_IF_ERROR(Assign(sub_expr, CValue(result_val, result_type), loc));

  xls::BValue return_val = is_pre ? result_val : sub_value.value();
  return CValue(return_val, result_type);
}

absl::StatusOr<CValue> Translator::Generate_UnaryOp(
    const clang::UnaryOperator* uop, const xls::SourceLocation& loc) {
  auto clang_op = uop->getOpcode();

  XLS_ASSIGN_OR_RETURN(
      shared_ptr<CType> result_type,
      TranslateTypeFromClang(uop->getType().getCanonicalType(), loc));
  XLS_ASSIGN_OR_RETURN(shared_ptr<CType> resolved_type,
                       ResolveTypeInstance(result_type));

  XLS_ASSIGN_OR_RETURN(CValue lhs_cv, GenerateIR_Expr(uop->getSubExpr(), loc));
  XLS_ASSIGN_OR_RETURN(xls::BValue lhs_cvc,
                       GenTypeConvert(lhs_cv, result_type, loc));
  CValue lhs_cvcv(lhs_cvc, lhs_cv.type());

  if (auto result_int_type = std::dynamic_pointer_cast<CIntType>(result_type)) {
    switch (clang_op) {
      case clang::UnaryOperatorKind::UO_Minus:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNeg, lhs_cvcv.value(), loc),
            result_type);
      case clang::UnaryOperatorKind::UO_LNot:
      case clang::UnaryOperatorKind::UO_Not:
        return CValue(
            context().fb->AddUnOp(xls::Op::kNot, lhs_cvcv.value(), loc),
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
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented unary operator %i at %s", clang_op, LocString(loc)));
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
            context().fb->AddUnOp(xls::Op::kNot, lhs_cvcv.value(), loc),
            result_type);
      default:
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented unary operator %i at %s", clang_op, LocString(loc)));
    }
  } else {
    return absl::UnimplementedError(
        absl::StrFormat("Unary operators on types other than builtin-int "
                        "unimplemented for type %s at %s",
                        std::string(*result_type), LocString(loc)));
  }
}

absl::StatusOr<CValue> Translator::Generate_BinaryOp(
    clang::BinaryOperatorKind clang_op, bool is_assignment,
    std::shared_ptr<CType> result_type, const clang::Expr* lhs,
    const clang::Expr* rhs, const xls::SourceLocation& loc) {
  CValue result;
  {
    // Allow assignment to compound LHS which is modified in RHS,
    //  since the evaluation order is always safe.
    DisallowAssignmentGuard allow_assignment_guard(*this);

    PushContextGuard fork_guard(*this, loc);
    context().in_fork = true;

    // Generate the other way around to detect unsequenced errors,
    //  and to get the lhs type
    // unsequenced_gen_backwards_ should only be turned off for debugging
    if (unsequenced_gen_backwards_) {
      MaskAssignmentsGuard no_assignments(*this);
      XLS_ASSIGN_OR_RETURN(CValue ignore, GenerateIR_Expr(lhs, loc));
    }

    // Don't reduce operands to logical boolean operators to 1 bit.
    std::shared_ptr<CType> input_type = result_type;
    if (dynamic_cast<const CBoolType*>(input_type.get())) {
      XLS_ASSIGN_OR_RETURN(input_type,
                           TranslateTypeFromClang(lhs->getType(), loc));
    }

    XLS_ASSIGN_OR_RETURN(
        xls::Op xls_op,
        XLSOpcodeFromClang(clang_op, *input_type, *result_type, loc));

    XLS_ASSIGN_OR_RETURN(CValue rhs_cv, GenerateIR_Expr(rhs, loc));

    XLS_ASSIGN_OR_RETURN(xls::BValue rhs_cvc,
                         GenTypeConvert(rhs_cv, input_type, loc));
    CValue rhs_cvcv(rhs_cvc, input_type);

    result = rhs_cvcv;

    if (xls_op != xls::Op::kIdentity) {
      XLS_ASSIGN_OR_RETURN(CValue lhs_cv, GenerateIR_Expr(lhs, loc));
      XLS_ASSIGN_OR_RETURN(xls::BValue lhs_cvc,
                           GenTypeConvert(lhs_cv, input_type, loc));
      CValue lhs_cvcv(lhs_cvc, lhs_cv.type());

      if (xls::IsOpClass<xls::CompareOp>(xls_op)) {
        result = CValue(context().fb->AddCompareOp(xls_op, lhs_cvcv.value(),
                                                   rhs_cvcv.value(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::ArithOp>(xls_op)) {
        result = CValue(
            context().fb->AddArithOp(xls_op, lhs_cvcv.value(), rhs_cvcv.value(),
                                     result_type->GetBitWidth(), loc),
            result_type);
      } else if (xls::IsOpClass<xls::BinOp>(xls_op)) {
        result = CValue(context().fb->AddBinOp(xls_op, lhs_cvcv.value(),
                                               rhs_cvcv.value(), loc),
                        result_type);
      } else if (xls::IsOpClass<xls::NaryOp>(xls_op)) {
        result = CValue(
            context().fb->AddNaryOp(
                xls_op,
                std::vector<xls::BValue>{lhs_cvcv.value(), rhs_cvcv.value()},
                loc),
            result_type);

      } else {
        return absl::UnimplementedError(absl::StrFormat(
            "Internal error: unknown XLS op class at %s", LocString(loc)));
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
    const xls::SourceLocation& loc) {
  PushContextGuard fork_guard(*this, loc);
  context().in_fork = true;

  XLS_ASSIGN_OR_RETURN(CValue sel_val, GenerateIR_Expr(cond_expr, loc));
  XLS_CHECK(dynamic_cast<CBoolType*>(sel_val.type().get()));
  XLS_ASSIGN_OR_RETURN(CValue true_cv, GenerateIR_Expr(true_expr, loc));
  XLS_ASSIGN_OR_RETURN(CValue false_cv, GenerateIR_Expr(false_expr, loc));
  XLS_ASSIGN_OR_RETURN(xls::BValue true_val,
                       GenTypeConvert(true_cv, result_type, loc));
  XLS_ASSIGN_OR_RETURN(xls::BValue false_val,
                       GenTypeConvert(false_cv, result_type, loc));
  // Regenerate in opposite order to detect unsequenced errors
  if (unsequenced_gen_backwards_) {
    XLS_RETURN_IF_ERROR(GenTypeConvert(true_cv, result_type, loc).status());
    XLS_RETURN_IF_ERROR(GenerateIR_Expr(cond_expr, loc).status());
  }
  xls::BValue ret_val =
      context().fb->Select(sel_val.value(), true_val, false_val, loc);
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

absl::StatusOr<CValue> Translator::GenerateIR_Call(
    const clang::CallExpr* call, const xls::SourceLocation& loc) {
  const clang::FunctionDecl* funcdecl = call->getDirectCallee();

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

  PushContextGuard fork_guard(*this, loc);
  if (this_expr) {
    // If there is a "this" and any arguments, then this is a fork
    // A fork is when an expression has more than one sub-expression.
    // In C++ the order of evaluation of these is undefined, and different
    //  compilers produce different behaviors.
    // If there are assignments inside these sub-expressions, their order is
    //  undefined, and XLS[cc] is strict about references to variables assigned
    //  inside a "fork", generating an unsequenced assignment error instead of
    //  arbitrary / undefined behavior.
    if (!sequencing_safe)
      context().in_fork = context().in_fork || call->getNumArgs() > 0;

    {
      // The Assign() statement below will take care of any assignments
      //  in the expression for "this". Don't do these twice, as it can cause
      //  issues like double-increment https://github.com/google/xls/issues/389
      MaskAssignmentsGuard mask(*this);
      XLS_ASSIGN_OR_RETURN(this_value_orig, GenerateIR_Expr(this_expr, loc));
    }

    thisval = this_value_orig.value();
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
    const clang::CXXOperatorCallExpr* op_call, const xls::SourceLocation& loc,
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

      xls::BValue this_inout = lvalue_initial.value();
      XLS_ASSIGN_OR_RETURN(
          CValue f_return,
          GenerateIR_Call(to_call, {ivalue, rvalue}, &this_inout, loc,
                          // Avoid unsequenced error
                          /*force_no_fork=*/true));
      XLS_RETURN_IF_ERROR(
          Assign(lvalue, CValue(this_inout, lvalue_initial.type()), loc));
      *output = f_return;
      return true;
    }
  }
  // Recognized the pattern, but no set_element() method to use
  return false;
}

// this_inout can be nullptr for non-members
absl::StatusOr<CValue> Translator::GenerateIR_Call(
    const clang::FunctionDecl* funcdecl,
    std::vector<const clang::Expr*> expr_args, xls::BValue* this_inout,
    const xls::SourceLocation& loc, bool force_no_fork) {
  XLS_ASSIGN_OR_RETURN(GeneratedFunction * func,
                       TranslateFunctionToXLS(funcdecl));

  vector<xls::BValue> args;

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
  int expected_returns = 0;
  if (add_this_return) ++expected_returns;
  if (!funcdecl->getReturnType()->isVoidType()) ++expected_returns;

  if (expr_args.size() != funcdecl->getNumParams()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Parameter count mismatch: %i params in FunctionDecl, %i arguments to "
        "call at %s",
        funcdecl->getNumParams(), static_cast<int>(expr_args.size()),
        LocString(loc)));
  }

  PushContextGuard fork_guard(*this, loc);
  if (!force_no_fork) {
    context().in_fork = context().in_fork || (funcdecl->getNumParams() > 1);
  }

  // Add other parameters
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

    if (func->io_channels.contains(p)) {
      const IOChannel& channel_info = func->io_channels[p];
      if (channel_info.channel_op_type == OpType::kSend) {
        // No need for an input parameter for output channels
        continue;
      }

      XLS_CHECK(channel_info.channel_op_type == OpType::kRecv);

      std::vector<xls::BValue> insp;

      for (const IOOp& op : context().sf->io_ops) {
        if (op.sub_op && p == op.sub_op->channel) {
          XLS_ASSIGN_OR_RETURN(CValue inval, GetChannelInputValue(op, loc));

          insp.push_back(inval.value());
        }
      }

      xls::BValue in_val = MakeFlexTuple(insp, GetLoc(*funcdecl));
      args.push_back(in_val);
      continue;
    }

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      ++expected_returns;
    }

    XLS_ASSIGN_OR_RETURN(CValue argv, GenerateIR_Expr(expr_args[pi], loc));
    XLS_ASSIGN_OR_RETURN(auto argt, TranslateTypeFromClang(stripped.base, loc));

    if (*argv.type() != *argt) {
      return absl::InternalError(
          absl::StrFormat("Internal error: expression type %s doesn't match "
                          "parameter type %s in function %s at %s",
                          string(*argv.type()), string(*argt),
                          funcdecl->getNameAsString(), LocString(loc)));
    }

    args.push_back(argv.value());
  }

  // Count expected IO returns
  expected_returns += func->io_ops.size();

  // Generate the other way around to detect unsequenced errors
  if (unsequenced_gen_backwards_) {
    MaskAssignmentsGuard no_assignments(*this);
    for (int pi = (funcdecl->getNumParams() - 1); pi >= 0; --pi) {
      const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

      if (func->io_channels.contains(p)) {
        const IOChannel& channel_info = func->io_channels[p];
        // No need for an input parameter for output chanels
        if (channel_info.channel_op_type == OpType::kSend) {
          continue;
        }
      }
      XLS_RETURN_IF_ERROR(GenerateIR_Expr(expr_args[pi], loc).status());
    }
  }

  xls::BValue raw_return = context().fb->Invoke(args, func->xls_func, loc);

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

  // First this return
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
        auto ctype, TranslateTypeFromClang(funcdecl->getReturnType(), loc));
    XLS_CHECK(!unpacked_returns.empty());
    retval = CValue(unpacked_returns.front(), ctype);
    unpacked_returns.pop_front();
  }

  // Then reference parameter returns
  for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

    // IO returns are later
    if (func->io_channels.contains(p)) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> ctype,
                         TranslateTypeFromClang(stripped.base, GetLoc(*p)));

    // Const references don't need a return
    if (stripped.is_ref && (!stripped.base.isConstQualified())) {
      XLS_CHECK(!unpacked_returns.empty());
      XLS_RETURN_IF_ERROR(
          Assign(expr_args[pi], CValue(unpacked_returns.front(), ctype), loc));
      unpacked_returns.pop_front();
    }
  }

  // IO returns
  for (IOOp& op : context().sf->io_ops) {
    for (int pi = 0; pi < funcdecl->getNumParams(); ++pi) {
      const clang::ParmVarDecl* p = funcdecl->getParamDecl(pi);

      if (op.sub_op && op.sub_op->channel == p) {
        XLS_CHECK(!unpacked_returns.empty());
        op.ret_value = unpacked_returns.front();
        unpacked_returns.pop_front();
        break;
      }
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

absl::StatusOr<CValue> Translator::GenerateIR_Expr(
    const clang::Expr* expr, const xls::SourceLocation& loc) {
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

      XLS_ASSIGN_OR_RETURN(IOOpReturn ret, InterceptIOOp(call, loc));

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
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented character literaly type %i at %s",
            static_cast<int>(charlit->getKind()), LocString(loc)));
      }
      shared_ptr<CType> ctype(new CIntType(8, true));
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

      XLS_ASSIGN_OR_RETURN(
          std::shared_ptr<CType> sub_type,
          TranslateTypeFromClang(cast->getSubExpr()->getType(), loc));

      auto from_arr_type = std::dynamic_pointer_cast<CArrayType>(sub_type);

      // Avoid decay of array to pointer, pointers are unsupported
      if (from_arr_type && cast->getType()->isPointerType()) {
        XLS_ASSIGN_OR_RETURN(CValue sub,
                             GenerateIR_Expr(cast->getSubExpr(), loc));

        return sub;
      }

      XLS_ASSIGN_OR_RETURN(CValue sub,
                           GenerateIR_Expr(cast->getSubExpr(), loc));

      XLS_ASSIGN_OR_RETURN(shared_ptr<CType> to_type,
                           TranslateTypeFromClang(cast->getType(), loc));

      XLS_ASSIGN_OR_RETURN(ResolvedInheritance inheritance,
                           ResolveInheritance(sub.type(), to_type));

      // Are we casting to a derived class?
      if (inheritance.base_field != nullptr) {
        XLS_CHECK(inheritance.resolved_struct != nullptr);

        xls::BValue val =
            GetStructFieldXLS(sub.value(), inheritance.base_field->index(),
                              *inheritance.resolved_struct, loc);
        return CValue(val, to_type);
      }

      XLS_ASSIGN_OR_RETURN(xls::BValue subc, GenTypeConvert(sub, to_type, loc));
      return CValue(subc, to_type);
    }
    case clang::Stmt::CXXThisExprClass: {
      return context().this_val;
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
      if (dynamic_cast<const CStructType*>(ctype.get()) != nullptr) {
        XLS_ASSIGN_OR_RETURN(xls::BValue dv, CreateDefaultValue(octype, loc));
        std::vector<const clang::Expr*> args;
        args.reserve(cast->getNumArgs());
        for (int pi = 0; pi < cast->getNumArgs(); ++pi) {
          args.push_back(cast->getArg(pi));
        }
        XLS_ASSIGN_OR_RETURN(CValue ret, GenerateIR_Call(cast->getConstructor(),
                                                         args, &dv, loc));
        XLS_CHECK(dynamic_cast<const CVoidType*>(ret.type().get()));
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
            absl::StrFormat("Unsupported constructor argument count %i at %s",
                            cast->getNumArgs(), LocString(loc)));
      }
    }
    case clang::Stmt::ArraySubscriptExprClass: {
      PushContextGuard fork_guard(*this, loc);
      context().in_fork = true;
      auto* cast = clang_down_cast<const clang::ArraySubscriptExpr*>(expr);
      XLS_ASSIGN_OR_RETURN(CValue arr_val,
                           GenerateIR_Expr(cast->getLHS(), loc));
      auto arr_type = dynamic_cast<CArrayType*>(arr_val.type().get());
      XLS_CHECK(arr_type != nullptr);
      XLS_ASSIGN_OR_RETURN(CValue idx_val,
                           GenerateIR_Expr(cast->getRHS(), loc));

      // Generate the other way around to detect unsequenced errors,
      //  and to get the lhs type
      // unsequenced_gen_backwards_ should only be turned off for debugging
      if (unsequenced_gen_backwards_) {
        MaskAssignmentsGuard no_assignments(*this);
        XLS_ASSIGN_OR_RETURN(CValue ignore,
                             GenerateIR_Expr(cast->getLHS(), loc));
      }

      return CValue(
          context().fb->ArrayIndex(arr_val.value(), {idx_val.value()}, loc),
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
    case clang::Stmt::StringLiteralClass: {
      auto* string_literal_expr =
          clang_down_cast<const clang::StringLiteral*>(expr);
      if (!(string_literal_expr->isAscii() || string_literal_expr->isUTF8())) {
        return absl::UnimplementedError(
            "Only 8 bit character strings supported");
      }
      llvm::StringRef strref = string_literal_expr->getString();
      std::string str = strref.str();

      std::shared_ptr<CType> element_type(new CIntType(8, true));
      std::shared_ptr<CType> type(new CArrayType(element_type, str.size()));

      std::vector<xls::Value> elements;

      for (char c : str) {
        elements.push_back(xls::Value(xls::SBits(c, 8)));
      }

      XLS_ASSIGN_OR_RETURN(xls::Value arrval, xls::Value::Array(elements));

      return CValue(context().fb->Literal(arrval, loc), type);
    }
    default: {
      expr->dump();
      return absl::UnimplementedError(
          absl::StrFormat("Unimplemented expression %s at %s",
                          expr->getStmtClassName(), LocString(loc)));
    }
  }
}

absl::StatusOr<CValue> Translator::GenerateIR_MemberExpr(
    const clang::MemberExpr* expr, const xls::SourceLocation& loc) {
  XLS_ASSIGN_OR_RETURN(CValue leftval, GenerateIR_Expr(expr->getBase(), loc));
  XLS_ASSIGN_OR_RETURN(auto itype, ResolveTypeInstance(leftval.type()));

  auto sitype = std::dynamic_pointer_cast<CStructType>(itype);

  if (sitype == nullptr) {
    return absl::UnimplementedError(
        absl::StrFormat("Unimplemented member access on type %s at %s",
                        string(*itype), LocString(loc)));
  }

  // Get the value referred to
  clang::ValueDecl* member = expr->getMemberDecl();

  // VarDecl for static values
  if (member->getKind() == clang::ValueDecl::Kind::Var) {
    XLS_ASSIGN_OR_RETURN(
        CValue val,
        TranslateVarDecl(clang_down_cast<const clang::VarDecl*>(member), loc));
    return val;
  } else if (member->getKind() != clang::ValueDecl::Kind::Field) {
    // Otherwise only FieldDecl is supported. This is the non-static "foo.bar"
    // form.
    return absl::UnimplementedError(
        absl::StrFormat("Unimplemented member expression %s at %s",
                        member->getDeclKindName(), LocString(loc)));
  }

  auto field = clang_down_cast<clang::FieldDecl*>(member);
  const auto& fields_by_name = sitype->fields_by_name();
  auto found_field =
      // Upcast to NamedDecl because we track unique identifiers
      //  with NamedDecl pointers
      fields_by_name.find(absl::implicit_cast<const clang::NamedDecl*>(field));
  if (found_field == fields_by_name.end()) {
    return absl::NotFoundError(absl::StrFormat(
        "Member access on unknown field %s in type %s at %s",
        field->getNameAsString(), string(*itype), LocString(loc)));
  }
  const CField& cfield = *found_field->second;
  // Upcast to NamedDecl because we track unique identifiers
  //  with NamedDecl pointers
  XLS_CHECK_EQ(cfield.name(),
               absl::implicit_cast<const clang::NamedDecl*>(field));
  xls::BValue bval =
      GetStructFieldXLS(leftval.value(), cfield.index(), *sitype, loc);
  return CValue(bval, cfield.type());
}

absl::StatusOr<xls::BValue> Translator::CreateDefaultValue(
    std::shared_ptr<CType> t, const xls::SourceLocation& loc) {
  if (auto it = dynamic_cast<const CIntType*>(t.get())) {
    return context().fb->Literal(xls::UBits(0, it->width()), loc);
  } else if (auto it = dynamic_cast<const CBitsType*>(t.get())) {
    return context().fb->Literal(xls::UBits(0, it->GetBitWidth()), loc);
  } else if (dynamic_cast<const CBoolType*>(t.get()) != nullptr) {
    return context().fb->Literal(xls::UBits(0, 1), loc);
  } else if (auto it = dynamic_cast<const CStructType*>(t.get())) {
    vector<xls::BValue> args;
    for (auto it2 = it->fields().rbegin(); it2 != it->fields().rend(); it2++) {
      std::shared_ptr<CField> field = *it2;
      XLS_ASSIGN_OR_RETURN(xls::BValue fval,
                           CreateDefaultValue(field->type(), loc));
      args.push_back(fval);
    }
    return MakeStructXLS(args, *it, loc);
  } else if (auto it = dynamic_cast<const CArrayType*>(t.get())) {
    std::vector<xls::BValue> element_vals;
    XLS_ASSIGN_OR_RETURN(xls::BValue default_elem_val,
                         CreateDefaultValue(it->GetElementType(), loc));
    element_vals.resize(it->GetSize(), default_elem_val);
    XLS_ASSIGN_OR_RETURN(xls::Type * xls_elem_type,
                         TranslateTypeToXLS(it->GetElementType(), loc));
    return context().fb->Array(element_vals, xls_elem_type, loc);
  } else if (dynamic_cast<const CInstantiableTypeAlias*>(t.get()) != nullptr) {
    XLS_ASSIGN_OR_RETURN(auto resolved, ResolveTypeInstance(t));
    return CreateDefaultValue(resolved, loc);
  } else {
    return absl::UnimplementedError(
        absl::StrFormat("Don't know how to make default for type %s at %s",
                        std::string(*t), LocString(loc)));
  }
}

absl::StatusOr<xls::BValue> Translator::CreateInitValue(
    const CType& t, const clang::InitListExpr* init_list,
    const xls::SourceLocation& loc) {
  if (auto array_t = dynamic_cast<const CArrayType*>(&t)) {
    if (array_t->GetSize() != init_list->getNumInits()) {
      return absl::UnimplementedError(absl::StrFormat(
          "Wrong number of initializers at %s", LocString(loc)));
    }

    PushContextGuard fork_guard(*this, loc);
    context().in_fork = context().in_fork || (array_t->GetSize() > 1);

    std::vector<xls::BValue> element_vals;
    for (int i = 0; i < array_t->GetSize(); ++i) {
      const clang::Expr* this_init = init_list->getInit(i);
      xls::BValue this_val;
      if (this_init->getStmtClass() == clang::Stmt::InitListExprClass) {
        XLS_ASSIGN_OR_RETURN(
            this_val,
            CreateInitValue(
                *array_t->GetElementType(),
                clang_down_cast<const clang::InitListExpr*>(this_init), loc));
      } else {
        XLS_ASSIGN_OR_RETURN(CValue expr_val, GenerateIR_Expr(this_init, loc));
        if (*expr_val.type() != *array_t->GetElementType()) {
          return absl::UnimplementedError(
              absl::StrFormat("Wrong initializer type %s at %s",
                              string(*expr_val.type()), LocString(loc)));
        }
        this_val = expr_val.value();
      }
      element_vals.push_back(this_val);
    }
    XLS_ASSIGN_OR_RETURN(xls::Type * xls_elem_type,
                         TranslateTypeToXLS(array_t->GetElementType(), loc));
    return context().fb->Array(element_vals, xls_elem_type, loc);
  } else {
    return absl::UnimplementedError(absl::StrFormat(
        "Don't know how to interpret initializer list for type %s at %s",
        string(t), LocString(loc)));
  }
}

absl::StatusOr<xls::Value> Translator::EvaluateBVal(xls::BValue bval) {
  xls::IrInterpreter visitor({});
  XLS_RETURN_IF_ERROR(bval.node()->Accept(&visitor));
  xls::Value result = visitor.ResolveAsValue(bval.node());
  return result;
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

absl::Status Translator::GenerateIR_Stmt(const clang::Stmt* stmt,
                                         clang::ASTContext& ctx) {
  clang::SourceManager& sm = ctx.getSourceManager();
  const xls::SourceLocation loc = GetLoc(sm, *stmt);
  DisallowAssignmentGuard unseq_guard(*this);

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
                context().condition.valid()
                    ? context().condition
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
                sel_prev_ret_cond, context().return_val, crv);
          } else {
            // In the case of multiple unconditional returns, take the first one
            // (no-op)
          }
        }

        if (context().condition.valid()) {
          context().last_return_condition = context().condition;
        } else {
          context().last_return_condition =
              context().fb->Literal(xls::UBits(1, 1));
        }
      }

      xls::BValue reach_here_cond =
          context().condition.valid()
              ? context().condition
              : context().fb->Literal(xls::UBits(1, 1), loc);

      if (!context().have_returned_condition.valid()) {
        context().have_returned_condition = reach_here_cond;
      } else {
        context().have_returned_condition = context().fb->Or(
            reach_here_cond, context().have_returned_condition);
      }
      context().and_condition(
          context().fb->Not(context().have_returned_condition, loc), loc);
      break;
    }
    case clang::Stmt::DeclStmtClass: {
      auto declstmt = clang_down_cast<const clang::DeclStmt*>(stmt);

      for (auto decl : declstmt->decls()) {
        if (decl->getKind() == clang::Decl::Kind::Typedef) break;
        if (decl->getKind() == clang::Decl::Kind::StaticAssert) break;
        if (decl->getKind() != clang::Decl::Kind::Var) {
          return absl::UnimplementedError(
              absl::StrFormat("DeclStmt other than Var (%s) at %s",
                              decl->getDeclKindName(), LocString(loc)));
        }
        auto namedecl = clang_down_cast<const clang::NamedDecl*>(decl);
        auto vard = clang_down_cast<const clang::VarDecl*>(decl);

        XLS_ASSIGN_OR_RETURN(CValue translated, TranslateVarDecl(vard, loc));

        XLS_RETURN_IF_ERROR(DeclareVariable(namedecl, translated, loc));
      }
      break;
    }
    case clang::Stmt::GCCAsmStmtClass: {
      const auto* pasm = clang_down_cast<const clang::GCCAsmStmt*>(stmt);
      std::string sasm = pasm->getAsmString()->getString().str();

      vector<xls::BValue> args;

      PushContextGuard fork_guard(*this, loc);
      context().in_fork = true;

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
          args.emplace_back(ret.value());
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

      XLS_ASSIGN_OR_RETURN(xls::Function * af,
                           xls::Parser::ParseFunction(sasm, context().package));

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
      XLS_CHECK(dynamic_cast<CBoolType*>(cond.type().get()));
      if (ifst->getInit()) {
        return absl::UnimplementedError(absl::StrFormat(
            "Unimplemented C++17 if initializers at %s", LocString(loc)));
      }
      if (ifst->getThen()) {
        PushContextGuard context_guard(*this, cond.value(), loc);
        XLS_RETURN_IF_ERROR(GenerateIR_Compound(ifst->getThen(), ctx));
      }
      if (ifst->getElse()) {
        PushContextGuard context_guard(*this, context().fb->Not(cond.value()),
                                       loc);
        XLS_RETURN_IF_ERROR(GenerateIR_Compound(ifst->getElse(), ctx));
      }

      break;
    }
    case clang::Stmt::ForStmtClass: {
      auto forst = clang_down_cast<const clang::ForStmt*>(stmt);

      XLS_ASSIGN_OR_RETURN(Pragma pragma,
                           FindPragmaForLoc(GetPresumedLoc(sm, *forst)));
      if (pragma != Pragma_Unroll) {
        return absl::UnimplementedError(
            absl::StrFormat("Only unrolled for loops currently supported at %s",
                            LocString(loc)));
      }

      XLS_RETURN_IF_ERROR(GenerateIR_UnrolledFor(forst, ctx, loc, false));
      break;
    }
    case clang::Stmt::SwitchStmtClass: {
      auto switchst = clang_down_cast<const clang::SwitchStmt*>(stmt);
      return GenerateIR_Switch(switchst, ctx, loc);
    }
    case clang::Stmt::ContinueStmtClass: {
      // Continue should be used inside of for loop bodies only
      XLS_CHECK(context().in_for_body);
      xls::BValue not_cond = context().not_condition(loc);
      context().and_condition(not_cond, loc);
      context().continue_condition = not_cond;
      break;
    }
    case clang::Stmt::BreakStmtClass: {
      if (context().in_for_body) {
        // We are in a for body
        xls::BValue not_cond = context().not_condition(loc);
        XLS_CHECK(!context().in_switch_body);
        context().and_condition(not_cond, loc);
        context().break_condition = not_cond;
      } else {
        // We are in a switch body
        XLS_CHECK(context().in_switch_body);
        // We use the original condition because we only care about
        //  enclosing conditions, such as if(...) { break; }
        //  Not if(...) {return;} break;
        if (context().original_condition.node() !=
            context().switch_cond.node()) {
          return absl::UnimplementedError(absl::StrFormat(
              "Conditional breaks are not supported at %s", LocString(loc)));
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
    default:
      stmt->dump();
      return absl::UnimplementedError(
          absl::StrFormat("Unimplemented construct %s at %s",
                          stmt->getStmtClassName(), LocString(loc)));
  }
  return absl::OkStatus();
}

absl::Status Translator::GenerateIR_UnrolledFor(const clang::ForStmt* stmt,
                                                clang::ASTContext& ctx,
                                                const xls::SourceLocation& loc,
                                                bool deep_scan) {
  if (stmt->getInit() == nullptr) {
    return absl::UnimplementedError(
        absl::StrFormat("Unrolled for must have an initializer"));
  }

  if (stmt->getCond() == nullptr) {
    return absl::UnimplementedError(
        absl::StrFormat("Unrolled for must have a condition"));
  }

  if (stmt->getInc() == nullptr) {
    return absl::UnimplementedError(
        absl::StrFormat("Unrolled for must have an increment"));
  }

  if (stmt->getInit()->getStmtClass() != clang::Stmt::DeclStmtClass) {
    return absl::UnimplementedError(absl::StrFormat(
        "Unrolled for initializer must be a DeclStmt, but is a %s at %s",
        stmt->getInit()->getStmtClassName(), LocString(loc)));
  }

  auto declstmt = clang_down_cast<const clang::DeclStmt*>(stmt->getInit());
  if (!declstmt->isSingleDecl()) {
    return absl::UnimplementedError(
        absl::StrFormat("Decl groups at %s", LocString(loc)));
  }

  // Generate the declaration within a private context
  PushContextGuard for_init_guard(*this, loc);
  for_init_guard.propagate_break_up = false;
  for_init_guard.propagate_continue_up = false;
  context().in_for_body = true;

  XLS_RETURN_IF_ERROR(GenerateIR_Stmt(stmt->getInit(), ctx));

  const clang::Decl* decl = declstmt->getSingleDecl();
  auto counter_namedecl = clang_down_cast<const clang::NamedDecl*>(decl);

  XLS_RETURN_IF_ERROR(GetIdentifier(counter_namedecl, loc).status());

  // Loop unrolling causes duplicate NamedDecls which fail the soundness check.
  // Reset the known set before each iteration.
  auto saved_check_ids = check_unique_ids_;

  for (int nIters = 0;; ++nIters) {
    check_unique_ids_ = saved_check_ids;

    if (nIters >= max_unroll_iters_) {
      return absl::UnimplementedError(
          absl::StrFormat("Loop unrolling broke at maximum %i iterations at %s",
                          max_unroll_iters_, LocString(loc)));
    }
    XLS_ASSIGN_OR_RETURN(CValue cond_expr,
                         GenerateIR_Expr(stmt->getCond(), loc));
    XLS_CHECK(dynamic_cast<CBoolType*>(cond_expr.type().get()));
    XLS_ASSIGN_OR_RETURN(xls::Value cond_val, EvaluateBVal(cond_expr.value()));
    if (cond_val.IsAllZeros()) break;

    // Generate body
    {
      PushContextGuard for_body_guard(*this, loc);
      for_body_guard.propagate_break_up = true;
      for_body_guard.propagate_continue_up = false;

      // Don't allow assignment to the loop var in the body
      DisallowAssignmentGuard assign_guard(*this);
      context().forbidden_lvalues.insert(
          clang_down_cast<const clang::NamedDecl*>(decl));

      if (!deep_scan) {
        XLS_RETURN_IF_ERROR(GenerateIR_Compound(stmt->getBody(), ctx));
      } else {
        XLS_RETURN_IF_ERROR(DeepScanForIO(stmt->getBody(), ctx));
      }
    }

    // Counter increment
    {
      // Unconditionally assign to loop unrolling variable(s)
      UnconditionalAssignmentGuard assign_guard(*this);
      XLS_RETURN_IF_ERROR(GenerateIR_Stmt(stmt->getInc(), ctx));
    }

    // Flatten the counter for efficiency
    XLS_ASSIGN_OR_RETURN(CValue counter_new_bval,
                         GetIdentifier(counter_namedecl, loc));
    XLS_ASSIGN_OR_RETURN(xls::Value counter_new_val,
                         EvaluateBVal(counter_new_bval.value()));

    CValue flat_counter_val(context().fb->Literal(counter_new_val),
                            counter_new_bval.type());
    {
      // Unconditionally assign to loop unrolling variable(s)
      UnconditionalAssignmentGuard assign_guard(*this);
      XLS_RETURN_IF_ERROR(Assign(counter_namedecl, flat_counter_val, loc));
    }
  }

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
                                           const xls::SourceLocation& loc) {
  clang::SourceManager& sm = ctx.getSourceManager();

  PushContextGuard switch_guard(*this, loc);
  context().in_switch_body = true;
  context().in_for_body = false;

  if (switchst->getInit()) {
    return absl::UnimplementedError(
        absl::StrFormat("Switch init at %s", LocString(loc)));
  }
  XLS_ASSIGN_OR_RETURN(CValue switch_val,
                       GenerateIR_Expr(switchst->getCond(), loc));
  if (!dynamic_cast<CIntType*>(switch_val.type().get())) {
    return absl::UnimplementedError(
        absl::StrFormat("Switch on non-integers at %s", LocString(loc)));
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
      const xls::SourceLocation loc = GetLoc(sm, *case_it);
      XLS_ASSIGN_OR_RETURN(CValue case_val,
                           GenerateIR_Expr(case_it->getLHS(), loc));
      auto case_int_type = std::dynamic_pointer_cast<CIntType>(case_val.type());
      XLS_CHECK(case_int_type);
      if (*switch_val.type() != *case_int_type) {
        return absl::UnimplementedError(absl::StrFormat(" %s", LocString(loc)));
      }
      if (case_it->getRHS()) {
        return absl::UnimplementedError(
            absl::StrFormat("Case RHS at %s", LocString(loc)));
      }
      xls::BValue case_condition =
          context().fb->Eq(switch_val.value(), case_val.value(), loc);
      case_conds.emplace_back(case_condition);
    }
  }

  xls::BValue accum_cond;

  // for indexing into case_conds
  int case_idx = 0;
  for (const clang::Stmt* stmt : flat_statements) {
    const xls::SourceLocation loc = GetLoc(sm, *stmt);
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
      auto ocond = context().condition;
      PushContextGuard stmt_guard(*this, and_accum, loc);
      context().hit_break = false;
      context().switch_cond = ocond;
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
    const xls::SourceLocation& loc) {
  clang::Expr::EvalResult result;
  if (!expr.EvaluateAsInt(result, ctx, clang::Expr::SE_NoSideEffects,
                          /*InConstantContext=*/false)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to evaluate expression as int at %s", LocString(loc)));
  }
  const clang::APValue& val = result.Val;
  const llvm::APSInt& aps = val.getInt();

  return aps.getExtValue();
}

absl::StatusOr<bool> Translator::EvaluateBool(
    const clang::Expr& expr, const class clang::ASTContext& ctx,
    const xls::SourceLocation& loc) {
  bool result;
  if (!expr.EvaluateAsBooleanCondition(result, ctx,
                                       /*InConstantContext=*/false)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to evaluate expression as bool at %s", LocString(loc)));
  }
  return result;
}

absl::StatusOr<std::shared_ptr<CType>> Translator::TranslateTypeFromClang(
    const clang::QualType& t, const xls::SourceLocation& loc) {
  const clang::Type* type = t.getTypePtr();

  if (type->getTypeClass() == clang::Type::TypeClass::Builtin) {
    auto builtin = clang_down_cast<const clang::BuiltinType*>(type);
    if (builtin->isVoidType()) {
      return shared_ptr<CType>(new CVoidType());
    }
    if (!builtin->isInteger()) {
      return absl::UnimplementedError(absl::StrFormat(
          "BuiltIn type other than integer at %s", LocString(loc)));
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
      case clang::BuiltinType::Kind::Char_S:  // These depend on the target
      case clang::BuiltinType::Kind::Char_U:
        return shared_ptr<CType>(new CIntType(8, true));
      case clang::BuiltinType::Kind::UChar:
        return shared_ptr<CType>(new CIntType(8, false));
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
        return absl::UnimplementedError(absl::StrFormat(
            "Unsupported recordtype kind %s in translate: %s at %s",
            // getDeclKindName() is inherited from multiple base classes, so
            //  it is necessary to up-cast before calling it to avoid an error.
            absl::implicit_cast<const clang::Decl*>(decl)->getDeclKindName(),
            decl->getNameAsString(), LocString(loc)));
    }

  } else if (type->getTypeClass() == clang::Type::TypeClass::ConstantArray) {
    auto array = clang_down_cast<const clang::ConstantArrayType*>(type);
    XLS_ASSIGN_OR_RETURN(shared_ptr<CType> type,
                         TranslateTypeFromClang(array->getElementType(), loc));
    return shared_ptr<CType>(
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
    // No pointer support
    auto lval = clang_down_cast<const clang::LValueReferenceType*>(type);
    return TranslateTypeFromClang(lval->getPointeeType(), loc);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Pointer) {
    auto lval = clang_down_cast<const clang::PointerType*>(type);
    if (context().ignore_pointers) {
      return TranslateTypeFromClang(lval->getPointeeType(), loc);
    }
    lval->dump();
    return absl::UnimplementedError(
        absl::StrFormat("Pointers are unsupported at %s", LocString(loc)));
  } else {
    type->dump();
    return absl::UnimplementedError(
        absl::StrFormat("Unsupported type class in translate: %s at %s",
                        type->getTypeClassName(), LocString(loc)));
  }
}

absl::StatusOr<xls::Type*> Translator::TranslateTypeToXLS(
    std::shared_ptr<CType> t, const xls::SourceLocation& loc) {
  if (auto it = dynamic_cast<const CIntType*>(t.get())) {
    return context().package->GetBitsType(it->width());
  } else if (auto it = dynamic_cast<const CBitsType*>(t.get())) {
    return context().package->GetBitsType(it->GetBitWidth());
  } else if (dynamic_cast<const CBoolType*>(t.get()) != nullptr) {
    return context().package->GetBitsType(1);
  } else if (dynamic_cast<const CInstantiableTypeAlias*>(t.get()) != nullptr) {
    XLS_ASSIGN_OR_RETURN(auto ctype, ResolveTypeInstance(t));
    return TranslateTypeToXLS(ctype, loc);
  } else if (auto it = dynamic_cast<const CStructType*>(t.get())) {
    std::vector<xls::Type*> members;
    for (auto it2 = it->fields().rbegin();
         it2 != it->fields().rend(); it2++) {
      std::shared_ptr<CField> field = *it2;
      XLS_ASSIGN_OR_RETURN(xls::Type * ft,
                           TranslateTypeToXLS(field->type(), loc));
      members.push_back(ft);
    }
    return GetStructXLSType(members, *it, loc);
  } else if (auto it = dynamic_cast<const CArrayType*>(t.get())) {
    XLS_ASSIGN_OR_RETURN(auto xls_elem_type,
                         TranslateTypeToXLS(it->GetElementType(), loc));
    return context().package->GetArrayType(it->GetSize(), xls_elem_type);
  } else {
    return absl::UnimplementedError(
        absl::StrFormat("Unsupported CType subclass in TranslateTypeToXLS: %s",
                        typeid(*t).name()));
  }
}

absl::StatusOr<Translator::StrippedType> Translator::StripTypeQualifiers(
    const clang::QualType& t) {
  const clang::Type* type = t.getTypePtr();

  if ((type->getTypeClass() == clang::Type::TypeClass::RValueReference) ||
      (type->getTypeClass() == clang::Type::TypeClass::LValueReference)) {
    return StrippedType(type->getPointeeType(), true);
  } else if (type->getTypeClass() == clang::Type::TypeClass::Decayed) {
    auto dec = clang_down_cast<const clang::DecayedType*>(type);
    return StrippedType(dec->getOriginalType(), true);
  } else {
    return StrippedType(t, false);
  }
}

absl::Status Translator::SelectTop(absl::string_view top_function_name) {
  top_function_name_ = top_function_name;
  return absl::OkStatus();
}

absl::StatusOr<GeneratedFunction*> Translator::GenerateIR_Top_Function(
    xls::Package* package, bool force_static) {
  if (top_function_ == nullptr) {
    return absl::NotFoundError("No top function found");
  }

  context().package = package;

  XLS_ASSIGN_OR_RETURN(
      GeneratedFunction * ret,
      GenerateIR_Function(top_function_, top_function_->getNameAsString(),
                          force_static));

  return ret;
}

absl::StatusOr<xls::Channel*> Translator::CreateChannel(
    const HLSChannel& hls_channel, std::shared_ptr<CType> ctype,
    xls::Package* package, int port_order, const xls::SourceLocation& loc) {
  XLS_ASSIGN_OR_RETURN(xls::Type * data_type, TranslateTypeToXLS(ctype, loc));

  if (hls_channel.type() == ChannelType::FIFO) {
    return package->CreateStreamingChannel(
        hls_channel.name(),
        hls_channel.is_input() ? xls::ChannelOps::kReceiveOnly
                               : xls::ChannelOps::kSendOnly,
        data_type, /*initial_values=*/{}, xls::FlowControl::kReadyValid);
  }
  XLS_RET_CHECK_EQ(hls_channel.type(), ChannelType::DIRECT_IN);
  return package->CreateSingleValueChannel(hls_channel.name(),
                                           hls_channel.is_input()
                                               ? xls::ChannelOps::kReceiveOnly
                                               : xls::ChannelOps::kSendOnly,
                                           data_type);
}

absl::StatusOr<xls::Proc*> Translator::GenerateIR_Block(xls::Package* package,
                                                        const HLSBlock& block) {
  if (!top_function_) {
    return absl::UnimplementedError("Not top function found");
  }

  const clang::FunctionDecl* definition = nullptr;
  top_function_->getBody(definition);
  xls::SourceLocation body_loc = GetLoc(*definition);

  xls::ProcBuilder pb(block.name() + "_proc",
                      /*init_value=*/xls::Value::Tuple({}),
                      /*token_name=*/"tkn", /*state_name=*/"st", package);

  PreparedBlock prepared;

  // Generate function without FIFO channel parameters
  context().package = package;

  // Force top function in block to be static.
  // TODO(seanhaskell): Handle block class members
  XLS_ASSIGN_OR_RETURN(prepared.xls_func,
                       GenerateIR_Top_Function(package, true));

  prepared.token = pb.GetTokenParam();

  XLS_RETURN_IF_ERROR(
      GenerateIRBlockCheck(prepared, block, definition, body_loc));

  XLS_RETURN_IF_ERROR(GenerateIRBlockPrepare(prepared, pb, block, definition,
                                             package, body_loc));

  // The function is first invoked with defaults for any
  //  read() IO Ops.
  // If there are any read() IO Ops, then it will be invoked again
  //  for each read Op below.
  xls::BValue last_ret_val =
      pb.Invoke(prepared.args, prepared.xls_func->xls_func, body_loc);

  for (const IOOp& op : prepared.xls_func->io_ops) {
    XLS_CHECK(prepared.fifo_by_param.contains(op.channel));
    xls::Channel* xls_channel = prepared.fifo_by_param[op.channel];
    XLS_CHECK(prepared.return_index_for_op.contains(&op));
    const int return_index = prepared.return_index_for_op[&op];

    if (op.op == OpType::kRecv) {
      XLS_CHECK(prepared.arg_index_for_op.contains(&op));
      const int arg_index = prepared.arg_index_for_op[&op];
      XLS_CHECK(arg_index >= 0 && arg_index < prepared.args.size());

      xls::BValue condition =
          pb.TupleIndex(last_ret_val, return_index, body_loc,
                        absl::StrFormat("%s_pred", xls_channel->name()));

      xls::BValue receive =
          pb.ReceiveIf(xls_channel, prepared.token, condition, body_loc);
      prepared.token = pb.TupleIndex(receive, 0);
      xls::BValue in_val = pb.TupleIndex(receive, 1);

      XLS_CHECK(prepared.xls_func->io_channels.contains(op.channel));
      const IOChannel& io_channel = prepared.xls_func->io_channels[op.channel];
      prepared.args[arg_index] = UpdateFlexTupleField(
          prepared.args[arg_index], in_val, op.channel_tuple_index,
          io_channel.total_ops, body_loc);

      // The function is invoked again with the value received from the channel
      //  for each read() Op. The final invocation will produce all complete
      //  outputs.
      last_ret_val =
          pb.Invoke(prepared.args, prepared.xls_func->xls_func, body_loc);
    } else if (op.op == OpType::kSend) {
      xls::BValue send_tup =
          pb.TupleIndex(last_ret_val, return_index, body_loc);
      xls::BValue val = pb.TupleIndex(send_tup, 0, body_loc);
      xls::BValue condition =
          pb.TupleIndex(send_tup, 1, body_loc,
                        absl::StrFormat("%s_pred", xls_channel->name()));

      prepared.token =
          pb.SendIf(xls_channel, prepared.token, condition, {val}, body_loc);
    }
  }

  return pb.Build(prepared.token, pb.GetStateParam());
}

absl::Status Translator::GenerateIRBlockCheck(
    PreparedBlock& prepared, const HLSBlock& block,
    const clang::FunctionDecl* definition,
    const xls::SourceLocation& body_loc) {
  if (!block.has_name()) {
    return absl::InvalidArgumentError(absl::StrFormat("HLSBlock has no name"));
  }

  absl::flat_hash_set<string> channel_names_in_block;
  for (const HLSChannel& channel : block.channels()) {
    if (!channel.has_name() || !channel.has_is_input() || !channel.has_type()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel is incomplete in proto"));
    }

    if (prepared.channels_by_name.contains(channel.name())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate channel name %s", channel.name()));
    }

    prepared.channels_by_name[channel.name()] = channel;
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
    PreparedBlock& prepared, xls::ProcBuilder& pb, const HLSBlock& block,
    const clang::FunctionDecl* definition, xls::Package* package,
    const xls::SourceLocation& body_loc) {
  int next_return_index = 0;

  // For defaults, updates, invokes
  context().fb = dynamic_cast<xls::BuilderBase*>(&pb);

  // Initialize parameters to defaults, handle direct-ins, create channels
  // Add channels in order of function prototype
  for (int pidx = 0; pidx < definition->getNumParams(); ++pidx) {
    const clang::ParmVarDecl* param = definition->getParamDecl(pidx);

    if (!prepared.channels_by_name.contains(param->getNameAsString())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Function refers to channel %s not in block proto",
                          param->getNameAsString()));
    }

    HLSChannel& hls_channel =
        prepared.channels_by_name[param->getNameAsString()];

    if (hls_channel.type() == ChannelType::DIRECT_IN) {
      XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                           StripTypeQualifiers(param->getType()));
      XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                           TranslateTypeFromClang(stripped.base, body_loc));

      XLS_ASSIGN_OR_RETURN(
          xls::Channel * channel,
          CreateChannel(hls_channel, ctype, package, pidx, body_loc));

      xls::BValue receive = pb.Receive(channel, prepared.token);
      prepared.token = pb.TupleIndex(receive, 0);
      xls::BValue direct_in_value = pb.TupleIndex(receive, 1);

      prepared.args.push_back(direct_in_value);

      // If it's const or not a reference, then there's no return
      if (stripped.is_ref && !param->getType().isConstQualified()) {
        ++next_return_index;
      }
      continue;
    }

    XLS_CHECK(hls_channel.type() == ChannelType::FIFO);

    XLS_CHECK(prepared.xls_func->io_channels.contains(param));
    const IOChannel& io_channel = prepared.xls_func->io_channels[param];

    XLS_ASSIGN_OR_RETURN(xls::Channel * channel,
                         CreateChannel(hls_channel, io_channel.item_type,
                                       package, pidx, body_loc));
    prepared.fifo_by_param[param] = channel;

    if (io_channel.channel_op_type == OpType::kRecv) {
      std::vector<xls::BValue> op_vals;

      for (const IOOp& op : prepared.xls_func->io_ops) {
        if (op.channel != param) {
          continue;
        }

        XLS_ASSIGN_OR_RETURN(
            xls::BValue val,
            CreateDefaultValue(io_channel.item_type, body_loc));
        op_vals.push_back(val);
        prepared.arg_index_for_op[&op] = prepared.args.size();
      }

      prepared.args.push_back(MakeFlexTuple(op_vals, body_loc));
      XLS_CHECK(op_vals.size() == io_channel.total_ops);
    }
  }

  // Find return indices for ops
  for (const IOOp& op : prepared.xls_func->io_ops) {
    XLS_CHECK(
        prepared.channels_by_name.contains(op.channel->getNameAsString()));
    HLSChannel& hls_channel =
        prepared.channels_by_name[op.channel->getNameAsString()];
    if (hls_channel.type() != ChannelType::FIFO) continue;

    prepared.return_index_for_op[&op] = next_return_index++;
  }

  return absl::OkStatus();
}

absl::StatusOr<bool> Translator::ExprIsChannel(const clang::Expr* object,
                                               const xls::SourceLocation& loc) {
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
                                               const xls::SourceLocation& loc) {
  XLS_ASSIGN_OR_RETURN(StrippedType stripped, StripTypeQualifiers(param));
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> obj_type,
                       TranslateTypeFromClang(stripped.base, loc));

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

absl::StatusOr<xlscc_metadata::MetadataOutput> Translator::GenerateMetadata() {
  if (!top_function_) {
    return absl::NotFoundError("No top function found");
  }

  xlscc_metadata::MetadataOutput ret;

  // Top function proto
  ret.mutable_top_func_proto()->mutable_name()->set_name(
      top_function_->getNameAsString());
  ret.mutable_top_func_proto()->mutable_name()->set_fully_qualified_name(
      top_function_->getQualifiedNameAsString());
  ret.mutable_top_func_proto()->mutable_name()->set_id(
      reinterpret_cast<uint64_t>(
          dynamic_cast<const clang::NamedDecl*>(top_function_)));

  XLS_RETURN_IF_ERROR(GenerateMetadataType(
      top_function_->getReturnType(),
      ret.mutable_top_func_proto()->mutable_return_type()));

  for (int pi = 0; pi < top_function_->getNumParams(); ++pi) {
    const clang::ParmVarDecl* p = top_function_->getParamDecl(pi);
    xlscc_metadata::FunctionParameter* proto_param =
        ret.mutable_top_func_proto()->add_params();
    proto_param->set_name(p->getNameAsString());

    XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                         StripTypeQualifiers(p->getType()));

    XLS_RETURN_IF_ERROR(
        GenerateMetadataType(stripped.base, proto_param->mutable_type()));

    proto_param->set_is_reference(stripped.is_ref);
    proto_param->set_is_const(stripped.base.isConstQualified());
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
        struct_out->mutable_as_struct()->mutable_name()));
    XLS_RETURN_IF_ERROR(ctype_as_struct->GetMetadata(struct_out));
  }

  return ret;
}

absl::Status Translator::GenerateMetadataType(const clang::QualType& type_in,
                                              xlscc_metadata::Type* type_out) {
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                       TranslateTypeFromClang(type_in, xls::SourceLocation()));
  return ctype->GetMetadata(type_out);
}

absl::StatusOr<xls::BValue> Translator::GenTypeConvert(
    CValue const& in, std::shared_ptr<CType> out_type,
    const xls::SourceLocation& loc) {
  if (dynamic_cast<const CStructType*>(out_type.get())) {
    return in.value();
  } else if (dynamic_cast<const CVoidType*>(out_type.get())) {
    return xls::BValue();
  } else if (dynamic_cast<const CArrayType*>(out_type.get())) {
    return in.value();
  } else if (auto out_bits_type =
                 dynamic_cast<const CBitsType*>(out_type.get())) {
    XLS_ASSIGN_OR_RETURN(auto conv_type, ResolveTypeInstance(in.type()));
    auto in_bits_type = dynamic_cast<const CBitsType*>(conv_type.get());
    if (!in_bits_type) {
      return absl::UnimplementedError(
          absl::StrFormat("Cannot convert type %s to bits at %s",
                          std::string(*in.type()), LocString(loc)));
    }
    if (conv_type->GetBitWidth() != out_bits_type->GetBitWidth()) {
      return absl::UnimplementedError(absl::StrFormat(
          "No implicit bit width conversions for __xls_bits: from %s to %s at "
          "%s",
          std::string(*in.type()), std::string(*out_type), LocString(loc)));
    }
    return in.value();
  } else if (dynamic_cast<const CBoolType*>(out_type.get())) {
    return GenBoolConvert(in, loc);
  } else if (dynamic_cast<const CIntType*>(out_type.get()) != nullptr) {
    if (!(dynamic_cast<const CBoolType*>(in.type().get()) != nullptr ||
          dynamic_cast<const CIntType*>(in.type().get()) != nullptr)) {
      return absl::UnimplementedError(
          absl::StrFormat("Cannot convert type %s to int at %s",
                          std::string(*in.type()), LocString(loc)));
    }

    const int expr_width = in.type()->GetBitWidth();
    if (expr_width == out_type->GetBitWidth()) {
      return in.value();
    } else if (expr_width < out_type->GetBitWidth()) {
      auto p_in_int = std::dynamic_pointer_cast<const CIntType>(in.type());
      if ((dynamic_cast<const CBoolType*>(in.type().get()) == nullptr) &&
          (p_in_int != nullptr && p_in_int->is_signed())) {
        return context().fb->SignExtend(in.value(), out_type->GetBitWidth(),
                                        loc);
      } else {
        return context().fb->ZeroExtend(in.value(), out_type->GetBitWidth(),
                                        loc);
      }
    } else {
      return context().fb->BitSlice(in.value(), 0, out_type->GetBitWidth(),
                                    loc);
    }
  } else if (dynamic_cast<const CInstantiableTypeAlias*>(out_type.get()) !=
             nullptr) {
    XLS_ASSIGN_OR_RETURN(auto t, ResolveTypeInstance(out_type));
    return GenTypeConvert(in, t, loc);
  } else {
    return absl::UnimplementedError(absl::StrFormat(
        "Don't know how to convert %s to type %s at %s",
        std::string(*in.type()), std::string(*out_type), LocString(loc)));
  }
}

absl::StatusOr<xls::BValue> Translator::GenBoolConvert(
    CValue const& in, const xls::SourceLocation& loc) {
  if (!((dynamic_cast<const CBoolType*>(in.type().get()) != nullptr) ||
        (dynamic_cast<CIntType*>(in.type().get()) != nullptr))) {
    return absl::UnimplementedError(
        absl::StrFormat("Cannot convert type %s to bool at %s",
                        std::string(*in.type()), LocString(loc)));
  }
  XLS_CHECK_GT(in.type()->GetBitWidth(), 0);
  if (in.type()->GetBitWidth() == 1) {
    return in.value();
  } else {
    xls::BValue const0 =
        context().fb->Literal(xls::UBits(0, in.type()->GetBitWidth()), loc);
    return context().fb->Ne(in.value(), const0, loc);
  }
}

}  // namespace xlscc
