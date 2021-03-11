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

#include "xls/dslx/extract_conversion_order.h"

#include "xls/common/status/ret_check.h"
#include "xls/dslx/dslx_builtins.h"

namespace xls::dslx {

Callee::Callee(Function* f, Module* m, TypeInfo* type_info,
               SymbolicBindings sym_bindings)
    : f_(f),
      m_(m),
      type_info_(type_info),
      sym_bindings_(std::move(sym_bindings)) {
  XLS_CHECK_EQ(f->owner(), m);
  XLS_CHECK_EQ(type_info->module(), m)
      << type_info->module()->name() << " vs " << m->name();
}

static std::string CalleesToString(absl::Span<const Callee> callees) {
  return absl::StrCat("[",
                      absl::StrJoin(callees, ", ",
                                    [](std::string* out, const Callee& callee) {
                                      absl::StrAppend(out, callee.ToString());
                                    }),
                      "]");
}

static std::string ConversionRecordsToString(
    absl::Span<const ConversionRecord> records) {
  return absl::StrCat(
      "[",
      absl::StrJoin(records, ",\n  ",
                    [](std::string* out, const ConversionRecord& record) {
                      absl::StrAppend(out, record.ToString());
                    }),
      "]");
}

std::string Callee::ToString() const {
  return absl::StrFormat("Callee{m=%s, f=%s, bindings=%s}", m_->name(),
                         f_->identifier(), sym_bindings_.ToString());
}

std::string ConversionRecord::ToString() const {
  return absl::StrFormat(
      "ConversionRecord{m=%s, f=%s, bindings=%s, callees=%s}", m->name(),
      f->identifier(), bindings.ToString(), CalleesToString(callees));
}

class InvocationVisitor : public AstNodeVisitorWithDefault {
 public:
  InvocationVisitor(Module* module, TypeInfo* type_info,
                    const SymbolicBindings& bindings)
      : module_(module), type_info_(type_info), bindings_(bindings) {
    XLS_CHECK_EQ(type_info_->module(), module_);
  }

  ~InvocationVisitor() override = default;

  absl::Status HandleInvocation(Invocation* node) {
    Module* this_m = nullptr;
    Function* f = nullptr;
    TypeInfo* invocation_type_info = type_info_;
    if (auto* colon_ref = dynamic_cast<ColonRef*>(node->callee())) {
      absl::optional<Import*> import = colon_ref->ResolveImportSubject();
      XLS_RET_CHECK(import.has_value());
      absl::optional<const ImportedInfo*> info =
          type_info_->GetImported(*import);
      XLS_RET_CHECK(info.has_value());
      this_m = (*info)->module;
      invocation_type_info = (*info)->type_info;
      f = this_m->GetFunction(colon_ref->attr()).value();
    } else if (auto* name_ref = dynamic_cast<NameRef*>(node->callee())) {
      this_m = module_;
      // TODO(leary): 2020-01-16 change to detect builtinnamedef map, identifier
      // is fragile due to shadowing.
      std::string fn_identifier = name_ref->identifier();
      if (fn_identifier == "map") {
        // We need to make sure we convert the mapped function!
        XLS_RET_CHECK_EQ(node->args().size(), 2);
        Expr* fn_node = node->args()[1];
        XLS_VLOG(5) << "map() invoking: " << fn_node->ToString();
        if (auto* mapped_colon_ref = dynamic_cast<ColonRef*>(fn_node)) {
          XLS_VLOG(5) << "map() invoking ColonRef";
          fn_identifier = mapped_colon_ref->attr();
          absl::optional<Import*> import =
              mapped_colon_ref->ResolveImportSubject();
          XLS_RET_CHECK(import.has_value());
          absl::optional<const ImportedInfo*> info =
              type_info_->GetImported(*import);
          XLS_RET_CHECK(info.has_value());
          this_m = (*info)->module;
        } else {
          XLS_VLOG(5) << "map() invoking NameRef";
          auto* mapped_name_ref = dynamic_cast<NameRef*>(fn_node);
          XLS_RET_CHECK(mapped_name_ref != nullptr);
          fn_identifier = mapped_name_ref->identifier();
        }
      }

      absl::optional<Function*> maybe_f = this_m->GetFunction(fn_identifier);
      if (maybe_f.has_value()) {
        f = maybe_f.value();
      } else {
        if (GetParametricBuiltins().contains(name_ref->identifier())) {
          return absl::OkStatus();
        }
        return absl::InternalError("Could not resolve invoked function: " +
                                   fn_identifier);
      }
    } else {
      return absl::UnimplementedError(
          "Only calls to named functions are currently supported "
          "for IR conversion; callee: " +
          node->callee()->ToString());
    }

    XLS_VLOG(5) << "Getting callee bindings for invocation: "
                << node->ToString() << " @ " << node->span()
                << " caller bindings: " << bindings_.ToString();
    absl::optional<const SymbolicBindings*> callee_bindings =
        type_info_->GetInvocationSymbolicBindings(node, bindings_);
    if (callee_bindings.has_value() && !(*callee_bindings)->empty()) {
      XLS_RET_CHECK(*callee_bindings != nullptr);
      XLS_VLOG(5) << "Found callee bindings: " << **callee_bindings;
      invocation_type_info =
          type_info_->GetInstantiation(node, **callee_bindings).value();
    }
    callees_.push_back(
        Callee{f, this_m, invocation_type_info,
               callee_bindings ? **callee_bindings : SymbolicBindings()});
    return absl::OkStatus();
  }

  absl::Status HandleColonRef(ColonRef* node) override {
    // Aside from direct function invocations (handled above), ColonRefs may
    // themselves be defined by arbitrary expressions, which may themselves
    // contain invocations, e.g.,
    // pub const MY_EXPORTED_CONSTANT = foo(u32:5);
    // so we need to traverse them.
    absl::optional<Import*> import = node->ResolveImportSubject();
    if (!import.has_value()) {
      return absl::OkStatus();
    }

    absl::optional<const ImportedInfo*> info = type_info_->GetImported(*import);
    XLS_RET_CHECK(info.has_value());
    Module* import_mod = (*info)->module;
    TypeInfo* import_type_info = (*info)->type_info;
    auto member_or = import_mod->FindMemberWithName(node->attr());
    XLS_RET_CHECK(member_or.has_value());
    ModuleMember* mm = member_or.value();

    // Constants or enum values could be defined in terms of constant
    // invocations, so we need to traverse values each sort.
    // Other ModuleMembers (Function, TestFunction, QuickCheck, StructDef,
    // Import) can't be defined in terms of constants, so they can be skipped.
    // TODO(rspringer): 2021-03-10 Currently, type aliases of the form:
    //   pub const MY_CONST = u32:1024;
    //   pub type MyType = u32[std::clog2(MY_CONST)];
    // can't be deduced. Once they can, then TypeDefs will need to be added
    // here. Check StructDefs then, as well (e.g.,
    //   struct Foo {
    //     a: bits[std::clog2(MY_CONST)],
    //   }
    if (absl::holds_alternative<ConstantDef*>(*mm)) {
      AstNode* member = absl::get<ConstantDef*>(*mm);
      InvocationVisitor sub_visitor(import_mod, import_type_info, bindings_);
      XLS_RETURN_IF_ERROR(
          WalkPostOrder(member, &sub_visitor, /*want_types=*/true));
      callees_.insert(callees_.end(), sub_visitor.callees().begin(),
                      sub_visitor.callees().end());
    } else if (absl::holds_alternative<EnumDef*>(*mm)) {
      AstNode* member = absl::get<EnumDef*>(*mm);
      InvocationVisitor sub_visitor(import_mod, import_type_info, bindings_);
      XLS_RETURN_IF_ERROR(
          WalkPostOrder(member, &sub_visitor, /*want_types=*/true));
      callees_.insert(callees_.end(), sub_visitor.callees().begin(),
                      sub_visitor.callees().end());
    }

    return absl::OkStatus();
  }

  absl::Status HandleConstRef(ConstRef* node) override {
    XLS_RET_CHECK(node->name_def() != nullptr);
    AstNode* definer = node->name_def()->definer();
    if (definer == nullptr) {
      XLS_VLOG(3) << "NULL ConstRef definer: " << node->ToString();
      return absl::OkStatus();
    }

    InvocationVisitor sub_visitor(module_, type_info_, bindings_);
    XLS_RETURN_IF_ERROR(
        WalkPostOrder(definer, &sub_visitor, /*want_types=*/true));
    callees_.insert(callees_.end(), sub_visitor.callees().begin(),
                    sub_visitor.callees().end());
    return absl::OkStatus();
  }

  std::vector<Callee>& callees() { return callees_; }

 private:
  Module* module_;
  TypeInfo* type_info_;
  const SymbolicBindings& bindings_;
  std::vector<Callee> callees_;
};

// Traverses the definition of a node to find callees.
//
// Args:
//   node: AST construct to inspect for calls.
//   m: Module that "node" resides in.
//   type_info: Node to type mapping that should be used with "node".
//   imports: Mapping of modules imported by m.
//   bindings: Bindings used in instantiation of "node".
//
// Returns:
//   Callee functions invoked by "node", and the parametric bindings used in
//   each of those invocations.
static absl::StatusOr<std::vector<Callee>> GetCallees(
    AstNode* node, Module* m, TypeInfo* type_info,
    const SymbolicBindings& bindings) {
  XLS_CHECK_EQ(type_info->module(), m);
  InvocationVisitor visitor(m, type_info, bindings);
  XLS_RETURN_IF_ERROR(
      WalkPostOrder(ToAstNode(node), &visitor, /*want_types=*/true));
  return std::move(visitor.callees());
}

static bool IsReady(absl::variant<Function*, TestFunction*> f, Module* m,
                    const SymbolicBindings& bindings,
                    const std::vector<ConversionRecord>* ready) {
  // Test functions are always the root and non-parametric, so they're always
  // ready.
  if (absl::holds_alternative<TestFunction*>(f)) {
    return true;
  }
  for (const ConversionRecord& cr : *ready) {
    if (cr.f == absl::get<Function*>(f) && cr.m == m &&
        cr.bindings == bindings) {
      return true;
    }
  }
  return false;
}

// Forward decl.
static absl::Status AddToReady(absl::variant<Function*, TestFunction*> f,
                               Module* m, TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready);

static absl::Status ProcessCallees(absl::Span<const Callee> orig_callees,
                                   std::vector<ConversionRecord>* ready) {
  // Knock out all callees that are already in the (ready) order.
  std::vector<Callee> non_ready;
  {
    for (const Callee& callee : orig_callees) {
      if (!IsReady(callee.f(), callee.m(), callee.sym_bindings(), ready)) {
        non_ready.push_back(callee);
      }
    }
  }

  // For all the remaining callees (that were not ready), add them to the list
  // before us, since we depend upon them.
  for (const Callee& callee : non_ready) {
    XLS_RETURN_IF_ERROR(AddToReady(
        absl::variant<Function*, TestFunction*>(callee.f()), callee.m(),
        callee.type_info(), callee.sym_bindings(), ready));
  }

  return absl::OkStatus();
}

// Adds (f, bindings) to conversion order after deps have been added.
static absl::Status AddToReady(absl::variant<Function*, TestFunction*> f,
                               Module* m, TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready) {
  XLS_CHECK_EQ(type_info->module(), m);
  if (IsReady(f, m, bindings, ready)) {
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(const std::vector<Callee> orig_callees,
                       GetCallees(ToAstNode(f), m, type_info, bindings));
  XLS_VLOG(5) << "Original callees of " << absl::get<Function*>(f)->identifier()
              << ": " << CalleesToString(orig_callees);
  XLS_RETURN_IF_ERROR(ProcessCallees(orig_callees, ready));

  XLS_RET_CHECK(!IsReady(f, m, bindings, ready));

  // We don't convert the bodies of test constructs of IR.
  if (!absl::holds_alternative<TestFunction*>(f)) {
    auto* fn = absl::get<Function*>(f);
    XLS_VLOG(3) << "Adding to ready sequence: " << fn->identifier();
    ready->push_back(
        ConversionRecord{fn, m, type_info, bindings, orig_callees});
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<ConversionRecord>> GetOrder(Module* module,
                                                       TypeInfo* type_info,
                                                       bool traverse_tests) {
  XLS_CHECK_EQ(type_info->module(), module);
  std::vector<ConversionRecord> ready;

  for (ModuleMember member : module->top()) {
    if (absl::holds_alternative<QuickCheck*>(member)) {
      auto* quickcheck = absl::get<QuickCheck*>(member);
      Function* function = quickcheck->f();
      XLS_RET_CHECK(!function->IsParametric()) << function->ToString();

      XLS_RETURN_IF_ERROR(
          AddToReady(function, module, type_info, SymbolicBindings(), &ready));
    } else if (absl::holds_alternative<Function*>(member)) {
      auto* function = absl::get<Function*>(member);
      if (function->IsParametric()) {
        continue;
      }

      XLS_RETURN_IF_ERROR(
          AddToReady(function, module, type_info, SymbolicBindings(), &ready));
    } else if (absl::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = absl::get<ConstantDef*>(member);
      XLS_ASSIGN_OR_RETURN(
          const std::vector<Callee> callees,
          GetCallees(constant_def, module, type_info, SymbolicBindings()));
      XLS_RETURN_IF_ERROR(ProcessCallees(callees, &ready));
    }
  }

  if (traverse_tests) {
    for (TestFunction* test : module->GetTests()) {
      XLS_RETURN_IF_ERROR(
          AddToReady(test, module, type_info, SymbolicBindings(), &ready));
    }
  }

  XLS_VLOG(5) << "Ready list: " << ConversionRecordsToString(ready);

  return ready;
}

}  // namespace xls::dslx
