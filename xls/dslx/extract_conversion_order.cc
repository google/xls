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

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/status/ret_check.h"
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
    Function* f, const Invocation* invocation, Module* module,
    TypeInfo* type_info, SymbolicBindings sym_bindings,
    std::optional<ProcId> proc_id) {
  XLS_RETURN_IF_ERROR(ConversionRecord::ValidateParametrics(f, sym_bindings));
  return Callee(f, invocation, module, type_info, std::move(sym_bindings),
                proc_id);
}

Callee::Callee(Function* f, const Invocation* invocation, Module* m,
               TypeInfo* type_info, SymbolicBindings sym_bindings,
               std::optional<ProcId> proc_id)
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
    Function* f, const Invocation* invocation, Module* module,
    TypeInfo* type_info, SymbolicBindings symbolic_bindings,
    std::vector<Callee> callees, std::optional<ProcId> proc_id, bool is_top) {
  XLS_RETURN_IF_ERROR(
      ConversionRecord::ValidateParametrics(f, symbolic_bindings));

  return ConversionRecord(f, invocation, module, type_info,
                          std::move(symbolic_bindings), std::move(callees),
                          proc_id, is_top);
}

std::string ConversionRecord::ToString() const {
  std::string proc_id = "<none>";
  if (proc_id_.has_value()) {
    proc_id = proc_id_.value().ToString();
  }
  return absl::StrFormat(
      "ConversionRecord{m=%s, f=%s, top=%s, pid=%s, symbolic_bindings=%s, "
      "callees=%s}",
      module_->name(), f_->identifier(), is_top_ ? "true" : "false", proc_id,
      symbolic_bindings_.ToString(), CalleesToString(callees_));
}

// TODO(vmirian) 2022-02-11 Consider collapsing this visitor
// (CalleeCollectorVisitor) with the InvocationVisitor.
// Collects callee information from Invocation node into a list.
class CalleeCollectorVisitor : public AstNodeVisitorWithDefault {
 public:
  CalleeCollectorVisitor(Module* module, TypeInfo* type_info)
      : module_(module), type_info_(type_info) {
    XLS_CHECK_EQ(type_info_->module(), module_);
  }
  absl::Status HandleInvocation(const Invocation* node) override {
    std::optional<CalleeInfo> callee_info;
    if (auto* colon_ref = dynamic_cast<const ColonRef*>(node->callee())) {
      XLS_ASSIGN_OR_RETURN(callee_info, HandleColonRefInvocation(colon_ref));
    } else if (auto* name_ref = dynamic_cast<const NameRef*>(node->callee())) {
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

    callees_.emplace_back(std::move(callee_info.value()));
    return absl::OkStatus();
  }

  ~CalleeCollectorVisitor() override = default;

  // Helper type used to hold callee information for different forms of
  // invocations.
  struct CalleeInfo {
    Module* module = nullptr;
    Function* function = nullptr;

    template <typename H>
    friend H AbslHashValue(H h, const CalleeInfo& c);

    friend bool operator==(const CalleeInfo& lhs, const CalleeInfo& rhs);
  };

  absl::Span<const CalleeInfo> GetCallees() { return callees_; }

 private:
  // Helper for invocations of ColonRef callees.
  absl::StatusOr<CalleeInfo> HandleColonRefInvocation(
      const ColonRef* colon_ref) {
    std::optional<Import*> import = colon_ref->ResolveImportSubject();
    XLS_RET_CHECK(import.has_value());
    std::optional<const ImportedInfo*> info = type_info_->GetImported(*import);
    XLS_RET_CHECK(info.has_value());
    Module* module = (*info)->module;
    XLS_ASSIGN_OR_RETURN(Function * f,
                         module->GetMemberOrError<Function>(colon_ref->attr()));
    return CalleeInfo{module, f};
  }

  // Helper for invocations of NameRef callees.
  absl::StatusOr<std::optional<CalleeInfo>> HandleNameRefInvocation(
      const NameRef* name_ref, const Invocation* invocation) {
    Module* this_m = module_;
    // TODO(leary): 2020-01-16 change to detect builtinnamedef map, identifier
    // is fragile due to shadowing. It is also appears in the InvocationVisitor.
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
        std::optional<Import*> import =
            mapped_colon_ref->ResolveImportSubject();
        XLS_RET_CHECK(import.has_value());
        std::optional<const ImportedInfo*> info =
            type_info_->GetImported(*import);
        XLS_RET_CHECK(info.has_value());
        this_m = (*info)->module;
        XLS_VLOG(5) << "Module for callee: " << this_m->name();
      } else {
        XLS_VLOG(5) << "map() invoking NameRef";
        auto* mapped_name_ref = dynamic_cast<NameRef*>(fn_node);
        XLS_RET_CHECK(mapped_name_ref != nullptr);
        identifier = mapped_name_ref->identifier();
      }
    }

    Function* f = nullptr;
    absl::StatusOr<Function*> maybe_f =
        this_m->GetMemberOrError<Function>(identifier);
    if (maybe_f.ok()) {
      f = maybe_f.value();
    } else {
      if (IsNameParametricBuiltin(identifier)) {
        return absl::nullopt;
      }
      return absl::InternalError("Could not resolve invoked function: " +
                                 identifier);
    }

    return CalleeInfo{this_m, f};
  }
  Module* module_;
  TypeInfo* type_info_;
  std::vector<CalleeInfo> callees_;
};

template <typename H>
H AbslHashValue(H h, const CalleeCollectorVisitor::CalleeInfo& c) {
  return H::combine(std::move(h), c.module, c.function);
}

bool operator==(const CalleeCollectorVisitor::CalleeInfo& lhs,
                const CalleeCollectorVisitor::CalleeInfo& rhs) {
  return lhs.module == rhs.module && lhs.function == rhs.function;
}

// Collects all Invocation nodes below the visited node.
class InvocationVisitor : public ExprVisitor {
 public:
  InvocationVisitor(Module* module, TypeInfo* type_info,
                    const SymbolicBindings& bindings,
                    std::optional<ProcId> proc_id)
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

  absl::Status HandleArray(const Array* expr) override {
    for (const Expr* element : expr->members()) {
      XLS_RETURN_IF_ERROR(element->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleAttr(const Attr* expr) override {
    return expr->lhs()->AcceptExpr(this);
  }

  absl::Status HandleBinop(const Binop* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    return expr->rhs()->AcceptExpr(this);
  }

  absl::Status HandleBlock(const Block* expr) override {
    return expr->body()->AcceptExpr(this);
  }

  absl::Status HandleCast(const Cast* expr) override {
    return expr->expr()->AcceptExpr(this);
  }

  absl::Status HandleConstantArray(const ConstantArray* expr) override {
    return HandleArray(expr);
  }

  absl::Status HandleFor(const For* expr) override {
    XLS_RETURN_IF_ERROR(expr->init()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->iterable()->AcceptExpr(this));
    return expr->body()->AcceptExpr(this);
  }

  absl::Status HandleFormatMacro(const FormatMacro* expr) override {
    for (const Expr* arg : expr->args()) {
      XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleIndex(const Index* expr) override {
    XLS_RETURN_IF_ERROR(expr->lhs()->AcceptExpr(this));
    // WidthSlice and Slice alternatives only contain constant values and so
    // don't need to be traversed.
    if (std::holds_alternative<Expr*>(expr->rhs())) {
      XLS_RETURN_IF_ERROR(std::get<Expr*>(expr->rhs())->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleInvocation(const Invocation* node) override {
    std::optional<CalleeInfo> callee_info;
    for (const Expr* arg : node->args()) {
      XLS_RETURN_IF_ERROR(arg->AcceptExpr(this));
    }

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
    std::optional<ProcId> proc_id;
    auto maybe_proc = callee_info->callee->proc();
    if (maybe_proc.has_value()) {
      XLS_RET_CHECK(proc_id_.has_value()) << "Functions cannot spawn procs.";

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
    std::optional<const SymbolicBindings*> callee_bindings =
        type_info_->GetInvocationCalleeBindings(node, bindings_);
    if (callee_bindings.has_value()) {
      XLS_RET_CHECK(*callee_bindings != nullptr);
      XLS_VLOG(5) << "Found callee bindings: " << **callee_bindings
                  << " for node: " << node->ToString();
      std::optional<TypeInfo*> instantiation_type_info =
          type_info_->GetInvocationTypeInfo(node, **callee_bindings);
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

  absl::Status HandleLet(const Let* expr) override {
    XLS_RETURN_IF_ERROR(expr->rhs()->AcceptExpr(this));
    return expr->body()->AcceptExpr(this);
  }

  absl::Status HandleMatch(const Match* expr) override {
    XLS_RETURN_IF_ERROR(expr->matched()->AcceptExpr(this));
    for (const MatchArm* arm : expr->arms()) {
      XLS_RETURN_IF_ERROR(arm->expr()->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleRange(const Range* expr) override {
    XLS_RETURN_IF_ERROR(expr->start()->AcceptExpr(this));
    return expr->end()->AcceptExpr(this);
  }

  absl::Status HandleRecvIf(const RecvIf* expr) override {
    return expr->condition()->AcceptExpr(this);
  }

  absl::Status HandleRecvIfNonBlocking(const RecvIfNonBlocking* expr) override {
    return expr->condition()->AcceptExpr(this);
  }

  absl::Status HandleSend(const Send* expr) override {
    return expr->payload()->AcceptExpr(this);
  }

  absl::Status HandleSendIf(const SendIf* expr) override {
    XLS_RETURN_IF_ERROR(expr->condition()->AcceptExpr(this));
    return expr->payload()->AcceptExpr(this);
  }

  absl::Status HandleSpawn(const Spawn* expr) override {
    XLS_RETURN_IF_ERROR(expr->callee()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->config()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->next()->AcceptExpr(this));
    return expr->body()->AcceptExpr(this);
  }

  absl::Status HandleSplatStructInstance(
      const SplatStructInstance* expr) override {
    XLS_RETURN_IF_ERROR(expr->splatted()->AcceptExpr(this));
    for (const auto& member : expr->members()) {
      XLS_RETURN_IF_ERROR(member.second->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleStructInstance(const StructInstance* expr) override {
    for (const auto& member : expr->GetUnorderedMembers()) {
      XLS_RETURN_IF_ERROR(member.second->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  absl::Status HandleTernary(const Ternary* expr) override {
    XLS_RETURN_IF_ERROR(expr->test()->AcceptExpr(this));
    XLS_RETURN_IF_ERROR(expr->consequent()->AcceptExpr(this));
    return expr->alternate()->AcceptExpr(this);
  }

  absl::Status HandleTupleIndex(const TupleIndex* expr) override {
    return expr->lhs()->AcceptExpr(this);
  }

  absl::Status HandleUnop(const Unop* expr) override {
    return expr->operand()->AcceptExpr(this);
  }

  absl::Status HandleXlsTuple(const XlsTuple* expr) override {
    for (const Expr* member : expr->members()) {
      XLS_RETURN_IF_ERROR(member->AcceptExpr(this));
    }
    return absl::OkStatus();
  }

  // Null handlers (we're not using ExprVisitorWithDefault to guard against
  // accidental omissions).
#define DEFAULT_HANDLE(TYPE) \
  absl::Status Handle##TYPE(const TYPE*) override { return absl::OkStatus(); }

  // No ColonRef handling (ColonRef invocations are handled in HandleInvocation;
  // all other ColonRefs are to constant values, which don't need their deriving
  // exprs converted to IR.
  // No ConstRef handling: constants don't need their defiving exprs converted
  // to IR.
  DEFAULT_HANDLE(ChannelDecl)
  DEFAULT_HANDLE(ColonRef)
  DEFAULT_HANDLE(ConstRef)
  DEFAULT_HANDLE(Join)
  DEFAULT_HANDLE(NameRef)
  DEFAULT_HANDLE(Number)
  DEFAULT_HANDLE(Recv)
  DEFAULT_HANDLE(RecvNonBlocking)
  DEFAULT_HANDLE(String)
  DEFAULT_HANDLE(UnrollFor)
#undef DEFAULT_HANDLE

  std::vector<Callee>& callees() { return callees_; }

 private:
  // Helper for invocations of ColonRef callees.
  absl::StatusOr<CalleeInfo> HandleColonRefInvocation(
      const ColonRef* colon_ref) {
    std::optional<Import*> import = colon_ref->ResolveImportSubject();
    XLS_RET_CHECK(import.has_value());
    std::optional<const ImportedInfo*> info = type_info_->GetImported(*import);
    XLS_RET_CHECK(info.has_value());
    Module* module = (*info)->module;
    XLS_ASSIGN_OR_RETURN(Function * f,
                         module->GetMemberOrError<Function>(colon_ref->attr()));
    return CalleeInfo{module, f, (*info)->type_info};
  }

  // Helper for invocations of NameRef callees.
  absl::StatusOr<std::optional<CalleeInfo>> HandleNameRefInvocation(
      const NameRef* name_ref, const Invocation* invocation) {
    Module* this_m = module_;
    TypeInfo* callee_type_info = type_info_;
    // TODO(leary): 2020-01-16 change to detect builtinnamedef map, identifier
    // is fragile due to shadowing. It is also appears in the
    // CalleeCollectorVisitor.
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
        std::optional<Import*> import =
            mapped_colon_ref->ResolveImportSubject();
        XLS_RET_CHECK(import.has_value());
        std::optional<const ImportedInfo*> info =
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
    absl::StatusOr<Function*> maybe_f =
        this_m->GetMemberOrError<Function>(identifier);
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
  std::optional<ProcId> proc_id_;
  absl::flat_hash_map<std::vector<Proc*>, int> proc_instances_;

  // Built up list of callee records discovered during traversal.
  std::vector<Callee> callees_;
};

// Top level procs are procs where their config or next function is not invoked
// within the module.
// For example, for a given module with four procs: ProcA, ProcB, ProcC and
// ProcD. The procs have the following invocation scheme: ProcA invokes ProcC
// and ProcD, and ProcB does not invoke any procs. Given that there is no
// invocation of ProcA and ProcB, they (ProcA and ProcB) are the top level
// procs.
static absl::StatusOr<std::vector<Proc*>> GetTopLevelProcs(
    Module* module, TypeInfo* type_info) {
  CalleeCollectorVisitor visitor(module, type_info);
  std::vector<Proc*> procs;
  XLS_RETURN_IF_ERROR(WalkPostOrder(module, &visitor, /*want_types=*/true));
  absl::Span<const CalleeCollectorVisitor::CalleeInfo> callees =
      visitor.GetCallees();
  absl::flat_hash_set<CalleeCollectorVisitor::CalleeInfo> callee_set;
  for (auto& callee : callees) {
    callee_set.insert(callee);
  }
  for (Proc* proc : module->GetProcs()) {
    // Checking the presence of the config function suffices since the config
    // is instantiated prior to the next function.
    if (callee_set.find({proc->owner(), proc->config()}) != callee_set.end()) {
      continue;
    }
    procs.emplace_back(proc);
  }
  return procs;
}

// This function removes duplicate conversion records containing a non-derived
// DSLX functions from a list. A non-derived non-parametric function is not an
// inferred function (e.g. derived from a proc spawn/invocation or a parametric
// function). The 'ready' input list is modified and cannot be nullptr.
static void RemoveFunctionDuplicates(std::vector<ConversionRecord>* ready) {
  for (auto iter_func = ready->begin(); iter_func != ready->end();
       iter_func++) {
    const ConversionRecord& function_cr = *iter_func;
    for (auto iter_subject = iter_func + 1; iter_subject != ready->end();) {
      const ConversionRecord& subject_cr = *iter_subject;

      bool same_fns = function_cr.f() == subject_cr.f();
      bool either_is_proc_instance_fn =
          function_cr.f()->tag() == Function::Tag::kProcConfig ||
          function_cr.f()->tag() == Function::Tag::kProcNext ||
          subject_cr.f()->tag() == Function::Tag::kProcConfig ||
          subject_cr.f()->tag() == Function::Tag::kProcNext;
      bool either_is_parametric =
          function_cr.f()->IsParametric() || subject_cr.f()->IsParametric();

      if (same_fns && !either_is_proc_instance_fn && !either_is_parametric) {
        iter_subject = ready->erase(iter_subject);
        continue;
      }
      iter_subject++;
    }
  }
}

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
    Expr* node, Module* m, TypeInfo* type_info,
    const SymbolicBindings& bindings, std::optional<ProcId> proc_id) {
  XLS_VLOG(5) << "Getting callees of " << node->ToString();
  XLS_CHECK_EQ(type_info->module(), m);
  InvocationVisitor visitor(m, type_info, bindings, proc_id);
  XLS_RETURN_IF_ERROR(node->AcceptExpr(&visitor));
  return std::move(visitor.callees());
}

static bool IsReady(std::variant<Function*, TestFunction*> f, Module* m,
                    const SymbolicBindings& bindings,
                    const std::vector<ConversionRecord>* ready) {
  // Test functions are always the root and non-parametric, so they're always
  // ready.
  if (std::holds_alternative<TestFunction*>(f)) {
    return true;
  }

  auto matches_fn = [&](const ConversionRecord& cr, Function* f) {
    return cr.f() == f && cr.module() == m &&
           cr.symbolic_bindings() == bindings && !cr.proc_id().has_value();
  };

  for (const ConversionRecord& cr : *ready) {
    if (matches_fn(cr, std::get<Function*>(f))) {
      return true;
    }
  }
  return false;
}

// Forward decl.
static absl::Status AddToReady(std::variant<Function*, TestFunction*> f,
                               const Invocation* invocation, Module* m,
                               TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready,
                               std::optional<ProcId> proc_id,
                               bool is_top = false);

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
static absl::Status AddToReady(std::variant<Function*, TestFunction*> f,
                               const Invocation* invocation, Module* m,
                               TypeInfo* type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready,
                               const std::optional<ProcId> proc_id,
                               bool is_top) {
  XLS_CHECK_EQ(type_info->module(), m);
  if (IsReady(f, m, bindings, ready)) {
    return absl::OkStatus();
  }

  Expr* body;
  if (std::holds_alternative<Function*>(f)) {
    body = std::get<Function*>(f)->body();
  } else {
    body = std::get<TestFunction*>(f)->body();
  }
  XLS_ASSIGN_OR_RETURN(const std::vector<Callee> orig_callees,
                       GetCallees(body, m, type_info, bindings, proc_id));
  XLS_VLOG(5) << "Original callees of " << std::get<Function*>(f)->identifier()
              << ": " << CalleesToString(orig_callees);
  XLS_RETURN_IF_ERROR(ProcessCallees(orig_callees, ready));

  XLS_RET_CHECK(!IsReady(f, m, bindings, ready));

  // We don't convert the bodies of test constructs of IR.
  if (std::holds_alternative<TestFunction*>(f)) {
    return absl::OkStatus();
  }

  auto* fn = std::get<Function*>(f);
  XLS_VLOG(3) << "Adding to ready sequence: " << fn->identifier();
  XLS_ASSIGN_OR_RETURN(
      ConversionRecord cr,
      ConversionRecord::Make(fn, invocation, m, type_info, bindings,
                             orig_callees, proc_id, is_top));
  ready->push_back(std::move(cr));
  return absl::OkStatus();
}

static absl::StatusOr<std::vector<ConversionRecord>> GetOrderForProc(
    std::variant<Proc*, TestProc*> entry, TypeInfo* type_info, bool is_top) {
  std::vector<ConversionRecord> ready;
  Proc* p;
  if (std::holds_alternative<TestProc*>(entry)) {
    p = std::get<TestProc*>(entry)->proc();
  } else {
    p = std::get<Proc*>(entry);
  }

  // The next function of a proc is the entry function when converting a proc to
  // IR.
  XLS_RETURN_IF_ERROR(AddToReady(p->next(),
                                 /*invocation=*/nullptr, p->owner(), type_info,
                                 SymbolicBindings(), &ready, ProcId{{p}, 0},
                                 is_top));
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

absl::StatusOr<std::vector<ConversionRecord>> GetOrder(Module* module,
                                                       TypeInfo* type_info,
                                                       bool traverse_tests) {
  XLS_CHECK_EQ(type_info->module(), module);
  std::vector<ConversionRecord> ready;

  for (ModuleMember member : module->top()) {
    if (std::holds_alternative<QuickCheck*>(member)) {
      auto* quickcheck = std::get<QuickCheck*>(member);
      Function* function = quickcheck->f();
      XLS_RET_CHECK(!function->IsParametric()) << function->ToString();

      XLS_RETURN_IF_ERROR(AddToReady(function, /*invocation=*/nullptr, module,
                                     type_info, SymbolicBindings(), &ready,
                                     {}));
    } else if (std::holds_alternative<Function*>(member)) {
      // Proc creation is driven by Spawn instantiations - the required constant
      // args are only specified there, so we can't convert Procs as encountered
      // at top level.
      Function* f = std::get<Function*>(member);
      if (f->IsParametric() || f->proc().has_value()) {
        continue;
      }

      XLS_RETURN_IF_ERROR(AddToReady(f, /*invocation=*/nullptr, module,
                                     type_info, SymbolicBindings(), &ready,
                                     {}));
    } else if (std::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = std::get<ConstantDef*>(member);
      XLS_ASSIGN_OR_RETURN(const std::vector<Callee> callees,
                           GetCallees(constant_def->value(), module, type_info,
                                      SymbolicBindings(), {}));
      XLS_RETURN_IF_ERROR(ProcessCallees(callees, &ready));
    }
  }

  // Collect the top level procs.
  XLS_ASSIGN_OR_RETURN(std::vector<Proc*> top_level_procs,
                       GetTopLevelProcs(module, type_info));
  // Get the order for each top level proc.
  for (Proc* proc : top_level_procs) {
    XLS_ASSIGN_OR_RETURN(TypeInfo * proc_ti,
                         type_info->GetTopLevelProcTypeInfo(proc));

    XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> proc_ready,
                         GetOrderForProc(proc, proc_ti, /*is_top=*/false));
    ready.insert(ready.end(), proc_ready.begin(), proc_ready.end());
  }

  // Collect tests.
  if (traverse_tests) {
    for (TestFunction* test : module->GetFunctionTests()) {
      XLS_RETURN_IF_ERROR(AddToReady(test, /*invocation=*/nullptr, module,
                                     type_info, SymbolicBindings(), &ready,
                                     {}));
    }
    for (TestProc* test : module->GetProcTests()) {
      XLS_ASSIGN_OR_RETURN(TypeInfo * proc_ti,
                           type_info->GetTopLevelProcTypeInfo(test->proc()));
      XLS_ASSIGN_OR_RETURN(std::vector<ConversionRecord> proc_ready,
                           GetOrderForProc(test, proc_ti, /*is_top=*/false));
      ready.insert(ready.end(), proc_ready.begin(), proc_ready.end());
    }
  }

  // Remove duplicated functions. When performing a complete module conversion,
  // the functions and the proc are converted in that order. However, procs may
  // call functions resulting in functions being accounted for twice. There must
  // be a single instance of the function to convert.
  RemoveFunctionDuplicates(&ready);

  XLS_VLOG(5) << "Ready list: " << ConversionRecordsToString(ready);

  return ready;
}

absl::StatusOr<std::vector<ConversionRecord>> GetOrderForEntry(
    std::variant<Function*, Proc*> entry, TypeInfo* type_info) {
  std::vector<ConversionRecord> ready;
  if (std::holds_alternative<Function*>(entry)) {
    Function* f = std::get<Function*>(entry);
    if (f->proc().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          type_info, type_info->GetTopLevelProcTypeInfo(f->proc().value()));
    }
    XLS_RETURN_IF_ERROR(AddToReady(f,
                                   /*invocation=*/nullptr, f->owner(),
                                   type_info, SymbolicBindings(), &ready, {},
                                   /*is_top=*/true));
    RemoveFunctionDuplicates(&ready);
    return ready;
  }

  Proc* p = std::get<Proc*>(entry);
  XLS_ASSIGN_OR_RETURN(TypeInfo * new_ti,
                       type_info->GetTopLevelProcTypeInfo(p));
  XLS_ASSIGN_OR_RETURN(ready, GetOrderForProc(p, new_ti, /*is_top=*/true));
  RemoveFunctionDuplicates(&ready);
  return ready;
}

}  // namespace xls::dslx
