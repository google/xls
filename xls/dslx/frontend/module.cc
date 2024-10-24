// Copyright 2023 The XLS Authors
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

#include "xls/dslx/frontend/module.h"

#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/visitor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {

// -- class Module

Module::Module(std::string name, std::optional<std::filesystem::path> fs_path,
               FileTable& file_table)
    : AstNode(this),
      name_(std::move(name)),
      fs_path_(std::move(fs_path)),
      file_table_(&file_table) {
  VLOG(3) << "Created module \"" << name_ << "\" @ " << this;
}

Module::~Module() {
  VLOG(3) << "Destroying module \"" << name_ << "\" @ " << this;
}

std::string Module::ToString() const {
  // Don't print Proc functions, as they'll be printed as part of the procs
  // themselves.
  std::vector<ModuleMember> print_top;
  for (const auto& member : top_) {
    if (std::holds_alternative<Function*>(member) &&
        std::get<Function*>(member)->proc().has_value()) {
      continue;
    }
    print_top.push_back(member);
  }

  std::string body = absl::StrJoin(
      print_top, "\n", [](std::string* out, const ModuleMember& member) {
        absl::StrAppend(out, ToAstNode(member)->ToString());
      });

  if (!annotations().empty()) {
    // Make an annotation block above the module contents if there are
    // annotations.
    std::string header = absl::StrJoin(
        annotations(), "\n", [](std::string* out, ModuleAnnotation annotation) {
          switch (annotation) {
            case ModuleAnnotation::kAllowNonstandardConstantNaming:
              absl::StrAppend(out, "#![allow(nonstandard_constant_naming)]");
              break;
            case ModuleAnnotation::kAllowNonstandardMemberNaming:
              absl::StrAppend(out, "#![allow(nonstandard_member_naming)]");
              break;
          }
        });
    return absl::StrCat(header, "\n\n", body);
  }

  return body;
}

const AstNode* Module::FindNode(AstNodeKind kind, const Span& target) const {
  for (const auto& node : nodes_) {
    if (node->kind() == kind && node->GetSpan().has_value() &&
        node->GetSpan().value() == target) {
      return node.get();
    }
  }
  return nullptr;
}

std::vector<const AstNode*> Module::FindIntercepting(const Pos& target) const {
  std::vector<const AstNode*> found;
  for (const auto& node : nodes_) {
    if (node->GetSpan().has_value() && node->GetSpan()->Contains(target)) {
      found.push_back(node.get());
    }
  }
  return found;
}

std::vector<const AstNode*> Module::FindContained(const Span& target) const {
  std::vector<const AstNode*> found;
  for (const auto& node : nodes_) {
    if (std::optional<Span> node_span = node->GetSpan();
        node_span.has_value() && target.Contains(node_span.value())) {
      found.push_back(node.get());
    }
  }
  return found;
}

std::optional<Function*> Module::GetFunction(std::string_view target_name) {
  for (ModuleMember& member : top_) {
    if (std::holds_alternative<Function*>(member)) {
      Function* f = std::get<Function*>(member);
      if (f->identifier() == target_name) {
        return f;
      }
    }
  }
  return std::nullopt;
}

absl::StatusOr<TestFunction*> Module::GetTest(std::string_view target_name) {
  for (ModuleMember& member : top_) {
    if (std::holds_alternative<TestFunction*>(member)) {
      TestFunction* t = std::get<TestFunction*>(member);
      if (t->identifier() == target_name) {
        return t;
      }
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No test in module %s with name \"%s\"", name_, target_name));
}

absl::StatusOr<TestProc*> Module::GetTestProc(std::string_view target_name) {
  for (ModuleMember& member : top_) {
    if (std::holds_alternative<TestProc*>(member)) {
      auto* t = std::get<TestProc*>(member);
      if (t->proc()->identifier() == target_name) {
        return t;
      }
    }
  }
  return absl::NotFoundError(absl::StrFormat(
      "No test proc in module %s with name \"%s\"", name_, target_name));
}

std::vector<std::string> Module::GetTestNames() const {
  std::vector<std::string> result;
  for (auto& member : top_) {
    if (std::holds_alternative<TestFunction*>(member)) {
      TestFunction* t = std::get<TestFunction*>(member);
      result.push_back(t->identifier());
    } else if (std::holds_alternative<TestProc*>(member)) {
      TestProc* tp = std::get<TestProc*>(member);
      result.push_back(tp->proc()->identifier());
    }
  }
  return result;
}

std::vector<std::string> Module::GetQuickCheckNames() const {
  std::vector<std::string> result;
  for (auto& member : top_) {
    if (std::holds_alternative<QuickCheck*>(member)) {
      QuickCheck* q = std::get<QuickCheck*>(member);
      result.push_back(q->identifier());
    }
  }
  return result;
}

std::vector<std::string> Module::GetFunctionNames() const {
  std::vector<std::string> result;
  for (auto& member : top_) {
    if (std::holds_alternative<Function*>(member)) {
      result.push_back(std::get<Function*>(member)->identifier());
    }
  }
  return result;
}

const StructDef* Module::FindStructDef(const Span& span) const {
  return down_cast<const StructDef*>(FindNode(AstNodeKind::kStructDef, span));
}

const EnumDef* Module::FindEnumDef(const Span& span) const {
  return down_cast<const EnumDef*>(FindNode(AstNodeKind::kEnumDef, span));
}

bool Module::IsPublicMember(const AstNode& node) const {
  for (const ModuleMember& member : top_) {
    const AstNode* member_node = ToAstNode(member);
    if (member_node == &node) {
      return IsPublic(member);
    }
  }
  return false;
}

std::optional<ModuleMember*> Module::FindMemberWithName(
    std::string_view target) {
  for (ModuleMember& member : top_) {
    if (std::holds_alternative<Function*>(member)) {
      if (std::get<Function*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<Proc*>(member)) {
      if (std::get<Proc*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<TestFunction*>(member)) {
      if (std::get<TestFunction*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<TestProc*>(member)) {
      if (std::get<TestProc*>(member)->proc()->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<QuickCheck*>(member)) {
      if (std::get<QuickCheck*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<TypeAlias*>(member)) {
      if (std::get<TypeAlias*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<StructDef*>(member)) {
      if (std::get<StructDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<ProcDef*>(member)) {
      if (std::get<ProcDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<ConstantDef*>(member)) {
      if (std::get<ConstantDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<EnumDef*>(member)) {
      if (std::get<EnumDef*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<Import*>(member)) {
      if (std::get<Import*>(member)->identifier() == target) {
        return &member;
      }
    } else if (std::holds_alternative<ConstAssert*>(member)) {
      continue;  // These have no name / binding.
    } else if (std::holds_alternative<Impl*>(member)) {
      continue;  // These have no name / binding.
    } else {
      LOG(FATAL) << "Unhandled module member variant: "
                 << ToAstNode(member)->GetNodeTypeName();
    }
  }
  return std::nullopt;
}

absl::StatusOr<ConstantDef*> Module::GetConstantDef(std::string_view target) {
  std::optional<ModuleMember*> member = FindMemberWithName(target);
  if (!member.has_value()) {
    return absl::NotFoundError(
        absl::StrFormat("Could not find member named '%s' in module.", target));
  }
  if (!std::holds_alternative<ConstantDef*>(*member.value())) {
    return absl::NotFoundError(absl::StrFormat(
        "Member named '%s' in module was not a constant.", target));
  }
  return std::get<ConstantDef*>(*member.value());
}

absl::flat_hash_map<std::string, TypeDefinition>
Module::GetTypeDefinitionByName() const {
  absl::flat_hash_map<std::string, TypeDefinition> result;
  for (auto& member : top_) {
    if (std::holds_alternative<TypeAlias*>(member)) {
      TypeAlias* td = std::get<TypeAlias*>(member);
      result[td->identifier()] = td;
    } else if (std::holds_alternative<EnumDef*>(member)) {
      EnumDef* enum_def = std::get<EnumDef*>(member);
      result[enum_def->identifier()] = enum_def;
    } else if (std::holds_alternative<StructDef*>(member)) {
      StructDef* struct_def = std::get<StructDef*>(member);
      result[struct_def->identifier()] = struct_def;
    } else if (std::holds_alternative<ProcDef*>(member)) {
      ProcDef* proc_def = std::get<ProcDef*>(member);
      result[proc_def->identifier()] = proc_def;
    }
  }
  return result;
}

std::vector<TypeDefinition> Module::GetTypeDefinitions() const {
  std::vector<TypeDefinition> results;
  for (auto& member : top_) {
    if (std::holds_alternative<TypeAlias*>(member)) {
      TypeAlias* td = std::get<TypeAlias*>(member);
      results.push_back(td);
    } else if (std::holds_alternative<EnumDef*>(member)) {
      EnumDef* enum_def = std::get<EnumDef*>(member);
      results.push_back(enum_def);
    } else if (std::holds_alternative<StructDef*>(member)) {
      StructDef* struct_def = std::get<StructDef*>(member);
      results.push_back(struct_def);
    } else if (std::holds_alternative<ProcDef*>(member)) {
      ProcDef* proc_def = std::get<ProcDef*>(member);
      results.push_back(proc_def);
    }
  }
  return results;
}

std::vector<AstNode*> Module::GetChildren(bool want_types) const {
  std::vector<AstNode*> results;
  results.reserve(top_.size());
  for (ModuleMember member : top_) {
    results.push_back(ToAstNode(member));
  }
  return results;
}

absl::StatusOr<TypeDefinition> Module::GetTypeDefinition(
    std::string_view name) const {
  absl::flat_hash_map<std::string, TypeDefinition> map =
      GetTypeDefinitionByName();
  auto it = map.find(name);
  if (it == map.end()) {
    return absl::NotFoundError(
        absl::StrCat("Could not find type definition for name: ", name));
  }
  return it->second;
}

absl::Status Module::AddTop(ModuleMember member,
                            const MakeCollisionError& make_collision_error) {
  // Get name
  std::optional<std::string> member_name = absl::visit(
      Visitor{
          [](Function* f) { return std::make_optional(f->identifier()); },
          [](Proc* p) { return std::make_optional(p->identifier()); },
          [](TestFunction* tf) { return std::make_optional(tf->identifier()); },
          [](TestProc* tp) {
            return std::make_optional(tp->proc()->identifier());
          },
          [](QuickCheck* qc) { return std::make_optional(qc->identifier()); },
          [](TypeAlias* td) { return std::make_optional(td->identifier()); },
          [](StructDef* sd) { return std::make_optional(sd->identifier()); },
          [](ProcDef* pd) { return std::make_optional(pd->identifier()); },
          [](Impl* id) -> std::optional<std::string> { return std::nullopt; },
          [](ConstantDef* cd) { return std::make_optional(cd->identifier()); },
          [](EnumDef* ed) { return std::make_optional(ed->identifier()); },
          [](Import* i) { return std::make_optional(i->identifier()); },
          [](VerbatimNode*) -> std::optional<std::string> {
            return std::nullopt;
          },
          [](ConstAssert* n) -> std::optional<std::string> {
            return std::nullopt;
          },
      },
      member);

  if (member_name.has_value() && top_by_name_.contains(member_name.value())) {
    const AstNode* node = ToAstNode(top_by_name_.at(member_name.value()));
    const Span existing_span = node->GetSpan().value();
    const AstNode* new_node = ToAstNode(member);
    const Span new_span = new_node->GetSpan().value();
    if (make_collision_error != nullptr) {
      return make_collision_error(name_, member_name.value(), existing_span,
                                  node, new_span, new_node);
    }
    return absl::InvalidArgumentError(absl::StrFormat(
        "Module %s already contains a member named %s @ %s: %s", name_,
        member_name.value(), existing_span.ToString(*file_table_),
        node->ToString()));
  }

  top_.push_back(member);
  if (member_name.has_value()) {
    top_by_name_.insert({member_name.value(), member});
  }
  return absl::OkStatus();
}

std::string_view GetModuleMemberTypeName(const ModuleMember& module_member) {
  return absl::visit(Visitor{
                         [](Function*) { return "function"; },
                         [](Proc*) { return "proc"; },
                         [](TestFunction*) { return "test-function"; },
                         [](TestProc*) { return "test-proc"; },
                         [](QuickCheck*) { return "quick-check"; },
                         [](TypeAlias*) { return "type-alias"; },
                         [](StructDef*) { return "struct-definition"; },
                         [](ProcDef*) { return "proc-definition"; },
                         [](Impl*) { return "impl"; },
                         [](ConstantDef*) { return "constant-definition"; },
                         [](EnumDef*) { return "enum-definition"; },
                         [](Import*) { return "import"; },
                         [](VerbatimNode*) { return "verbatim"; },
                         [](ConstAssert*) { return "const-assert"; },
                     },
                     module_member);
}

bool IsPublic(const ModuleMember& member) {
  return absl::visit(Visitor{
                         [](const Function* m) { return m->is_public(); },
                         [](const Proc* m) { return m->is_public(); },
                         [](const TypeAlias* m) { return m->is_public(); },
                         [](const StructDef* m) { return m->is_public(); },
                         [](const ProcDef* m) { return m->is_public(); },
                         [](const Impl* m) { return m->is_public(); },
                         [](const ConstantDef* m) { return m->is_public(); },
                         [](const EnumDef* m) { return m->is_public(); },
                         [](const TestFunction* m) { return false; },
                         [](const TestProc* m) { return false; },
                         [](const QuickCheck* m) { return false; },
                         [](const Import* m) { return false; },
                         [](const ConstAssert* m) { return false; },
                         [](const VerbatimNode*) { return false; },
                     },
                     member);
}

Pos GetPos(const ModuleMember& module_member) {
  const AstNode* n = ToAstNode(module_member);
  std::optional<Span> span = n->GetSpan();
  CHECK(span.has_value());
  return span->start();
}

NameDef* ModuleMemberGetNameDef(const ModuleMember& mm) {
  return absl::visit(
      Visitor{
          [](Function* n) -> NameDef* { return n->name_def(); },
          [](Proc* n) -> NameDef* { return n->name_def(); },
          [](TestFunction* n) -> NameDef* { return n->name_def(); },
          [](TestProc* n) -> NameDef* { return n->name_def(); },
          [](QuickCheck* n) -> NameDef* { return n->name_def(); },
          [](TypeAlias* n) -> NameDef* { return &n->name_def(); },
          [](StructDef* n) -> NameDef* { return n->name_def(); },
          [](ProcDef* n) -> NameDef* { return n->name_def(); },
          [](Impl* n) -> NameDef* { return nullptr; },
          [](ConstantDef* n) -> NameDef* { return n->name_def(); },
          [](EnumDef* n) -> NameDef* { return n->name_def(); },
          [](Import* n) -> NameDef* { return &n->name_def(); },
          [](ConstAssert* n) -> NameDef* { return nullptr; },
          [](VerbatimNode*) -> NameDef* { return nullptr; },
      },
      mm);
}

Conditional* MakeTernary(Module* module, const Span& span, Expr* test,
                         Expr* consequent, Expr* alternate) {
  return module->Make<Conditional>(
      span, test,
      module->Make<StatementBlock>(
          consequent->span(),
          std::vector<Statement*>{module->Make<Statement>(consequent)}, false),
      module->Make<StatementBlock>(
          alternate->span(),
          std::vector<Statement*>{module->Make<Statement>(alternate)}, false));
}

}  // namespace xls::dslx
