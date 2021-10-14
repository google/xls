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

#include "absl/status/status.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/symbolized_stacktrace.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/builtins_metadata.h"
#include "xls/dslx/symbolic_bindings.h"

namespace xls::dslx {

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

// -- class Callee

/* static */ absl::StatusOr<Callee> Callee::Make(
    Function* f, Invocation* invocation, Module* module, TypeInfo* type_info,
    SymbolicBindings sym_bindings, absl::optional<ProcId> proc_id) {
  XLS_RETURN_IF_ERROR(ConversionRecord::ValidateParametrics(f, sym_bindings));
  return Callee(f, invocation, module, type_info, std::move(sym_bindings),
                proc_id);
}

Callee::Callee(Function* f, Invocation* invocation, Module* m,
               TypeInfo* type_info, SymbolicBindings sym_bindings,
               absl::optional<ProcId> proc_id)
    : f_(f),
      invocation_(invocation),
      m_(m),
      type_info_(type_info),
      sym_bindings_(std::move(sym_bindings)),
      proc_id_(proc_id) {
  XLS_CHECK_EQ(type_info->module(), m)
      << "type_info module: " << type_info->module()->name()
      << " vs module: " << m->name();
}

std::string Callee::ToString() const {
  std::string proc_id = "<none>";
  if (proc_id_.has_value()) {
    proc_id = proc_id_.value().ToString();
  }
  return absl::StrFormat(
      "Callee{m=%s, f=%s, i=%s, pid=%s, bindings=%s}", m_->name(),
      f_->identifier(),
      invocation_ == nullptr ? "<top level>" : invocation_->ToString(), proc_id,
      sym_bindings_.ToString());
}

// -- class ConversionRecord

/* static */ absl::Status ConversionRecord::ValidateParametrics(
    Function* f, const SymbolicBindings& symbolic_bindings) {
  absl::btree_set<std::string> symbolic_binding_keys =
      symbolic_bindings.GetKeySet();

  auto set_to_string = [](const absl::btree_set<std::string>& s) {
    return absl::StrCat("{", absl::StrJoin(s, ", "), "}");
  };
  // TODO(leary): 2020-11-19 We use btrees in particular so this could use dual
  // iterators via the sorted property for O(n) superset comparison, but this
  // was easier to write and know it was correct on a first cut (couldn't find a
  // superset helper in absl's container algorithms at a first pass).
  auto is_superset = [](absl::btree_set<std::string> lhs,
                        const absl::btree_set<std::string>& rhs) {
    for (const auto& item : rhs) {
      lhs.erase(item);
    }
    return !lhs.empty();
  };

  if (is_superset(f->GetFreeParametricKeySet(), symbolic_binding_keys)) {
    return absl::InternalError(absl::StrFormat(
        "Not enough symbolic bindings to convert function: %s; need %s got %s",
        f->identifier(), set_to_string(f->GetFreeParametricKeySet()),
        set_to_string(symbolic_binding_keys)));
  }
  return absl::OkStatus();
}

/* static */ absl::StatusOr<ConversionRecord> ConversionRecord::Make(
    Function* f, Invocation* invocation, Module* module, TypeInfo* type_info,
    SymbolicBindings symbolic_bindings, std::vector<Callee> callees,
    absl::optional<ProcId> proc_id) {
  XLS_RETURN_IF_ERROR(
      ConversionRecord::ValidateParametrics(f, symbolic_bindings));

  return ConversionRecord(f, invocation, module, type_info,
                          std::move(symbolic_bindings), std::move(callees),
                          proc_id);
}

std::string ConversionRecord::ToString() const {
  std::string proc_id = "<none>";
  if (proc_id_.has_value()) {
    proc_id = proc_id_.value().ToString();
  }
  return absl::StrFormat(
      "ConversionRecord{m=%s, f=%s, pid=%s, symbolic_bindings=%s, "
      "callees=%s}",
      module_->name(), f_->identifier(), proc_id, symbolic_bindings_.ToString(),
      CalleesToString(callees_));
}

class InvocationVisitor : public AstNodeVisitorWithDefault {
 public:
  InvocationVisitor(Module* module, TypeInfo* type_info,
                    const SymbolicBindings& bindings,
                    absl::optional<ProcId> proc_id)
      : module_(module),
        type_info_(type_info),
        bindings_(bindings),
        proc_id_(proc_id) {
    XLS_CHECK_EQ(type_info_->module(), module_);
  }

  ~InvocationVisitor() override = default;

  // Helper type used to hold callee information for different forms of
  // invocations.
  struct CalleeInfo {
    Module* module = nullptr;
    Function* callee = nullptr;
    TypeInfo* type_info = nullptr;
  };

  absl::Status HandleInvocation(Invocation* node) override {
    absl::optional<CalleeInfo> callee_info;
    if (auto* colon_ref = dynamic_cast<ColonRef*>(node->callee())) {
      XLS_ASSIGN_OR_RETURN(callee_info, HandleColonRefInvocation(colon_ref));
    } else if (auto* name_ref = dynamic_cast<NameRef*>(node->callee())) {
      XLS_ASSIGN_OR_RETURN(callee_info,
                           HandleNameRefInvocation(name_ref, node));
    } else {
      return absl::UnimplementedError(
          "Only calls to named functions are currently supported "
          "for IR conversion; callee: " +
          node->callee()->ToString());
    }

    if (!callee_info.has_value()) {
      // Happens for example when we're invoking a builtin, there's nothing to
      // convert.
      return absl::OkStatus();
    }

    // We only add to proc_stack if this is a new _Proc_, not a Function, since
    // Functions can't spawn Procs.
    absl::optional<ProcId> proc_id;
    auto maybe_proc = callee_info->callee->proc();
    if (maybe_proc.has_value()) {
      XLS_RET_CHECK(proc_id_.has_value())
          << "Functions cannot spawning procs - this should be caught earlier.";

      std::vector<Proc*> proc_stack = proc_id_.value().proc_stack;
      proc_stack.push_back(maybe_proc.value());
      proc_id = ProcId{proc_stack, proc_instances_[proc_stack]};
      // Only increment on next so that Config & Next have the same ID.
      // This assumes that we call a Proc's config & next calls in that order,
      // which is indeed the case.
      if (callee_info->callee->tag() == Function::Tag::kProcNext) {
        proc_instances_[proc_stack]++;
      }
    }

    // See if there are parametric bindings to use in the callee for this
    // invocation.
    XLS_VLOG(5) << "Getting callee bindings for invocation: "
                << node->ToString() << " @ " << node->span()
                << " caller bindings: " << bindings_.ToString();
    absl::optional<const SymbolicBindings*> callee_bindings =
        type_info_->GetInstantiationCalleeBindings(node, bindings_);
    if (callee_bindings.has_value()) {
      XLS_RET_CHECK(*callee_bindings != nullptr);
      XLS_VLOG(5) << "Found callee bindings: " << **callee_bindings
                  << " for node: " << node->ToString();
      absl::optional<TypeInfo*> instantiation_type_info =
          type_info_->GetInstantiationTypeInfo(node, **callee_bindings);
      XLS_RET_CHECK(instantiation_type_info.has_value())
          << "Could not find instantiation for `" << node->ToString() << "`"
          << " via bindings: " << **callee_bindings;
      // Note: when mapping a function that is non-parametric, the instantiated
      // type info can be nullptr (no associated type info, as the callee didn't
      // have to be instantiated).
      if (*instantiation_type_info != nullptr) {
        callee_info->type_info = *instantiation_type_info;
      }
    }

    XLS_ASSIGN_OR_RETURN(
        auto callee,
        Callee::Make(callee_info->callee, node, callee_info->module,
                     callee_info->type_info,
                     callee_bindings ? **callee_bindings : SymbolicBindings(),
                     proc_id));
    callees_.push_back(std::move(callee));
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
      ConstantDef* member = absl::get<ConstantDef*>(*mm);
      XLS_VLOG(5) << absl::StreamFormat("ColonRef %s @ %s is to ConstantDef",
                                        node->ToString(),
                                        node->span().ToString());
      // Note that constant definitions may be local -- we /only/ pass our
      // bindings if it is a local constant definition. Otherwise it's at the
      // top level and we use fresh bindings.
      InvocationVisitor sub_visitor(import_mod, import_type_info, bindings_,
                                    proc_id_);
      XLS_RETURN_IF_ERROR(
          WalkPostOrder(member, &sub_visitor, /*want_types=*/true));
      callees_.insert(callees_.end(), sub_visitor.callees().begin(),
                      sub_visitor.callees().end());
    } else if (absl::holds_alternative<EnumDef*>(*mm)) {
      EnumDef* member = absl::get<EnumDef*>(*mm);
      SymbolicBindings empty_bindings;
      InvocationVisitor sub_visitor(import_mod, import_type_info,
                                    empty_bindings, proc_id_);
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

    ConstantDef* const_def = down_cast<ConstantDef*>(definer);
    if (const_def->is_local()) {
      // We've already walked any local constant definitions.
      return absl::OkStatus();
    }

    SymbolicBindings empty_bindings;
    InvocationVisitor sub_visitor(module_, type_info_, empty_bindings,
                                  proc_id_);
    XLS_RETURN_IF_ERROR(
        WalkPostOrder(const_def, &sub_visitor, /*want_types=*/true));
    callees_.insert(callees_.end(), sub_visitor.callees().begin(),
                    sub_visitor.callees().end());
    return absl::OkStatus();
  }

  std::vector<Callee>& callees() { return callees_; }

 private:
  // Helper for invocations of ColonRef callees.
  absl::StatusOr<CalleeInfo> HandleColonRefInvocation(ColonRef* colon_ref) {
    absl::optional<Import*> import = colon_ref->ResolveImportSubject();
    XLS_RET_CHECK(import.has_value());
    absl::optional<const ImportedInfo*> info = type_info_->GetImported(*import);
    XLS_RET_CHECK(info.has_value());
    Module* module = (*info)->module;
    XLS_ASSIGN_OR_RETURN(Function * f,
                         module->GetFunctionOrError(colon_ref->attr()));
    return CalleeInfo{module, f, (*info)->type_info};
  }

  // Helper for invocations of NameRef callees.
  absl::StatusOr<absl::optional<CalleeInfo>> HandleNameRefInvocation(
      NameRef* name_ref, Invocation* invocation) {
    Module* this_m = module_;
    TypeInfo* callee_type_info = type_info_;
    // TODO(leary): 2020-01-16 change to detect builtinnamedef map, identifier
    // is fragile due to shadowing.
    std::string identifier = name_ref->identifier();
    if (identifier == "map") {
      // We need to make sure we convert the mapped function!
      XLS_RET_CHECK_EQ(invocation->args().size(), 2);
      Expr* fn_node = invocation->args()[1];
      XLS_VLOG(5) << "map() invoking: " << fn_node->ToString();
      if (auto* mapped_colon_ref = dynamic_cast<ColonRef*>(fn_node)) {
        XLS_VLOG(5) << "map() invoking ColonRef: "
                    << mapped_colon_ref->ToString();
        identifier = mapped_colon_ref->attr();
        absl::optional<Import*> import =
            mapped_colon_ref->ResolveImportSubject();
        XLS_RET_CHECK(import.has_value());
        absl::optional<const ImportedInfo*> info =
            type_info_->GetImported(*import);
        XLS_RET_CHECK(info.has_value());
        this_m = (*info)->module;
        callee_type_info = (*info)->type_info;
        XLS_VLOG(5) << "Module for callee: " << this_m->name();
      } else {
        XLS_VLOG(5) << "map() invoking NameRef";
        auto* mapped_name_ref = dynamic_cast<NameRef*>(fn_node);
        XLS_RET_CHECK(mapped_name_ref != nullptr);
        identifier = mapped_name_ref->identifier();
      }
    }

    Function* f = nullptr;
    absl::StatusOr<Function*> maybe_f = this_m->GetFunctionOrError(identifier);
    if (maybe_f.ok()) {
      f = maybe_f.value();
    } else {
      if (IsNameParametricBuiltin(identifier)) {
        return absl::nullopt;
      }
      return absl::InternalError("Could not resolve invoked function: " +
                                 identifier);
    }

    return CalleeInfo{this_m, f, callee_type_info};
  }

  Module* module_;
  TypeInfo* type_info_;
  const SymbolicBindings& bindings_;
  absl::optional<ProcId> proc_id_;
  absl::flat_hash_map<std::vector<Proc*>, int> proc_instances_;

  // Built up list of callee records discovered during traversal.
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
    const SymbolicBindings& bindings, absl::optional<ProcId> proc_id) {
  XLS_VLOG(5) << "Getting callees of " << node->ToString();
  XLS_CHECK_EQ(type_info->module(), m);
  InvocationVisitor visitor(m, type_info, bindings, proc_id);
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

  auto matches_fn = [&](const ConversionRecord& cr, Function* f) {
    return cr.f() == f && cr.module() == m &&
           cr.symbolic_bindings() == bindings && !cr.proc_id().has_value();
  };

  for (const ConversionRecord& cr : *ready) {
    if (matches_fn(cr, absl::get<Function*>(f))) {
      return true;
    }
  }
  return false;
}

// Forward decl.
static absl::Status AddToReady(absl::variant<Function*, TestFunction*> f,
                               Invocation* invocation, Module* m,
                               TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready,
                               absl::optional<ProcId> proc_id);

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
    XLS_RETURN_IF_ERROR(AddToReady(callee.f(), callee.invocation(), callee.m(),
                                   callee.type_info(), callee.sym_bindings(),
                                   ready, callee.proc_id()));
  }

  return absl::OkStatus();
}

// Adds (f, bindings) to conversion order after deps have been added.
static absl::Status AddToReady(absl::variant<Function*, TestFunction*> f,
                               Invocation* invocation, Module* m,
                               TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready,
                               const absl::optional<ProcId> proc_id) {
  XLS_CHECK_EQ(type_info->module(), m);
  if (IsReady(f, m, bindings, ready)) {
    return absl::OkStatus();
  }

  XLS_ASSIGN_OR_RETURN(
      const std::vector<Callee> orig_callees,
      GetCallees(ToAstNode(f), m, type_info, bindings, proc_id));
  XLS_VLOG(5) << "Original callees of " << absl::get<Function*>(f)->identifier()
              << ": " << CalleesToString(orig_callees);
  XLS_RETURN_IF_ERROR(ProcessCallees(orig_callees, ready));

  XLS_RET_CHECK(!IsReady(f, m, bindings, ready));

  // We don't convert the bodies of test constructs of IR.
  if (absl::holds_alternative<TestFunction*>(f)) {
    return absl::OkStatus();
  }

  auto* fn = absl::get<Function*>(f);
  XLS_VLOG(3) << "Adding to ready sequence: " << fn->identifier();
  XLS_ASSIGN_OR_RETURN(ConversionRecord cr,
                       ConversionRecord::Make(fn, invocation, m, type_info,
                                              bindings, orig_callees, proc_id));
  ready->push_back(std::move(cr));
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

      XLS_RETURN_IF_ERROR(AddToReady(function, /*invocation=*/nullptr, module,
                                     type_info, SymbolicBindings(), &ready,
                                     {}));
    } else if (absl::holds_alternative<Function*>(member)) {
      // Proc creation is driven by Spawn instantiations - the required constant
      // args are only specified there, so we can't convert Procs as encountered
      // at top level.
      Function* f = absl::get<Function*>(member);
      if (f->IsParametric() || f->proc().has_value()) {
        continue;
      }

      XLS_RETURN_IF_ERROR(AddToReady(f, /*invocation=*/nullptr, module,
                                     type_info, SymbolicBindings(), &ready,
                                     {}));
    } else if (absl::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = absl::get<ConstantDef*>(member);
      XLS_ASSIGN_OR_RETURN(
          const std::vector<Callee> callees,
          GetCallees(constant_def, module, type_info, SymbolicBindings(), {}));
      XLS_RETURN_IF_ERROR(ProcessCallees(callees, &ready));
    }
  }

  if (traverse_tests) {
    for (TestFunction* test : module->GetTests()) {
      XLS_RETURN_IF_ERROR(AddToReady(test, /*invocation=*/nullptr, module,
                                     type_info, SymbolicBindings(), &ready,
                                     {}));
    }
  }

  XLS_VLOG(5) << "Ready list: " << ConversionRecordsToString(ready);

  return ready;
}

absl::StatusOr<std::vector<ConversionRecord>> GetOrderForEntry(
    absl::variant<Function*, Proc*> entry, TypeInfo* type_info) {
  std::vector<ConversionRecord> ready;
  if (absl::holds_alternative<Function*>(entry)) {
    Function* f = absl::get<Function*>(entry);
    XLS_RETURN_IF_ERROR(AddToReady(f,
                                   /*invocation=*/nullptr, f->owner(),
                                   type_info, SymbolicBindings(), &ready, {}));
    return ready;
  }

  Proc* p = absl::get<Proc*>(entry);
  XLS_RETURN_IF_ERROR(AddToReady(p->next(),
                                 /*invocation=*/nullptr, p->owner(), type_info,
                                 SymbolicBindings(), &ready, ProcId{{p}, 0}));
  XLS_RETURN_IF_ERROR(AddToReady(p->config(),
                                 /*invocation=*/nullptr, p->owner(), type_info,
                                 SymbolicBindings(), &ready, ProcId{{p}, 0}));

  // Constants and "member" vars are assigned and defined in Procs' "config'"
  // functions, so we need to execute those before their "next" functions.
  // We ALSO need to evaluate them in reverse order, since proc N might be
  // defined in terms of its parent proc N-1.
  std::vector<ConversionRecord> final_order;
  std::vector<ConversionRecord> config_fns;
  std::vector<ConversionRecord> next_fns;
  for (const auto& record : ready) {
    if (record.f()->tag() == Function::Tag::kProcConfig) {
      config_fns.push_back(record);
    } else if (record.f()->tag() == Function::Tag::kProcNext) {
      next_fns.push_back(record);
    } else {
      // Regular functions can go wherever.
      final_order.push_back(record);
    }
  }
  // Need to reverse so constants can prop in config
  std::reverse(config_fns.begin(), config_fns.end());
  final_order.insert(final_order.end(), config_fns.begin(), config_fns.end());
  final_order.insert(final_order.end(), next_fns.begin(), next_fns.end());
  return final_order;
}

}  // namespace xls::dslx
