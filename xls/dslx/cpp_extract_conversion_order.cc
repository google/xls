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

#include "xls/dslx/cpp_extract_conversion_order.h"

#include "xls/common/status/ret_check.h"
#include "xls/dslx/dslx_builtins.h"

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

std::string Callee::ToString() const {
  return absl::StrFormat("Callee{m=%s, f=%s, bindings=%s}", m->name(),
                         f->identifier(), sym_bindings.ToString());
}

std::string ConversionRecord::ToString() const {
  return absl::StrFormat(
      "ConversionRecord{m=%s, f=%s, bindings=%s, callees=%s}", m->name(),
      f->identifier(), bindings.ToString(), CalleesToString(callees));
}

class InvocationVisitor : public AstNodeVisitorWithDefault {
 public:
  InvocationVisitor(Module* module, const std::shared_ptr<TypeInfo>& type_info,
                    const SymbolicBindings& bindings)
      : module_(module), type_info_(type_info), bindings_(bindings) {}

  ~InvocationVisitor() override = default;

  absl::Status HandleInvocation(Invocation* node) {
    Module* this_m = nullptr;
    Function* f = nullptr;
    if (auto* colon_ref = dynamic_cast<ColonRef*>(node->callee())) {
      absl::optional<Import*> import = colon_ref->ResolveImportSubject();
      XLS_RET_CHECK(import.has_value());
      absl::optional<const ImportedInfo*> info =
          type_info_->GetImported(*import);
      XLS_RET_CHECK(info.has_value());
      this_m = (*info)->module.get();
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
          this_m = (*info)->module.get();
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
          "Only calls to named functionsa re currently supported, got "
          "callee: " +
          node->callee()->ToString());
    }

    absl::optional<std::shared_ptr<TypeInfo>> invocation_type_info;
    XLS_VLOG(5) << "Getting callee bindings for invocation: "
                << node->ToString()
                << " caller bindings: " << bindings_.ToString();
    absl::optional<const SymbolicBindings*> callee_bindings =
        type_info_->GetInvocationSymbolicBindings(node, bindings_);
    if (callee_bindings.has_value()) {
      XLS_RET_CHECK(*callee_bindings != nullptr);
      invocation_type_info =
          type_info_->GetInstantiation(node, **callee_bindings);
    }
    // If we don't have any special type info to use for the invocation
    if (!invocation_type_info.has_value()) {
      invocation_type_info.emplace(type_info_);
    }
    callees_.push_back(
        Callee{f, this_m, invocation_type_info.value(),
               callee_bindings ? **callee_bindings : SymbolicBindings()});
    return absl::OkStatus();
  }

  std::vector<Callee>& callees() { return callees_; }

 private:
  Module* module_;
  const std::shared_ptr<TypeInfo>& type_info_;
  const SymbolicBindings& bindings_;
  std::vector<Callee> callees_;
};

// Traverses the definition of f to find callees.
//
// Args:
//   func: Function/test construct to inspect for calls.
//   m: Module that f resides in.
//   type_info: Node to type mapping that should be used with f.
//   imports: Mapping of modules imported by m.
//   bindings: Bindings used in instantiation of f.
//
// Returns:
//   Callee functions invoked by f, and the parametric bindings used in each of
//   those invocations.
static absl::StatusOr<std::vector<Callee>> GetCallees(
    absl::variant<Function*, TestFunction*> func, Module* m,
    const std::shared_ptr<TypeInfo>& type_info,
    const SymbolicBindings& bindings) {
  InvocationVisitor visitor(m, type_info, bindings);
  XLS_RETURN_IF_ERROR(
      WalkPostOrder(ToAstNode(func), &visitor, /*want_types=*/true));
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

// Adds (f, bindings) to conversion order after deps have been added.
static absl::Status AddToReady(absl::variant<Function*, TestFunction*> f,
                               Module* m,
                               const std::shared_ptr<TypeInfo>& type_info,
                               const SymbolicBindings& bindings,
                               std::vector<ConversionRecord>* ready) {
  if (IsReady(f, m, bindings, ready)) {
    return absl::OkStatus();
  }

  // Knock out all callees that are already in the (ready) order.
  std::vector<Callee> non_ready;
  XLS_ASSIGN_OR_RETURN(const std::vector<Callee> orig_callees,
                       GetCallees(f, m, type_info, bindings));

  XLS_VLOG(5) << "Original callees of " << absl::get<Function*>(f)->identifier()
              << ": " << CalleesToString(orig_callees);

  {
    for (const Callee& callee : std::vector<Callee>(orig_callees)) {
      if (!IsReady(callee.f, callee.m, callee.sym_bindings, ready)) {
        non_ready.push_back(callee);
      }
    }
  }

  // For all the remaining callees (that were not ready), add them to the list
  // before us, since we depend upon them.
  for (const Callee& callee : non_ready) {
    XLS_RETURN_IF_ERROR(
        AddToReady(absl::variant<Function*, TestFunction*>(callee.f), callee.m,
                   callee.type_info, callee.sym_bindings, ready));
  }

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

absl::StatusOr<std::vector<ConversionRecord>> GetOrder(
    Module* module, const std::shared_ptr<TypeInfo>& type_info,
    bool traverse_tests) {
  std::vector<ConversionRecord> ready;

  // Functions in the module should become ready in dependency order (they
  // referred to each other's names).
  for (QuickCheck* quickcheck : module->GetQuickChecks()) {
    Function* function = quickcheck->f();
    XLS_RET_CHECK(!function->IsParametric()) << function->ToString();

    XLS_RETURN_IF_ERROR(
        AddToReady(function, module, type_info, SymbolicBindings(), &ready));
  }

  for (Function* function : module->GetFunctions()) {
    if (function->IsParametric()) {
      continue;
    }

    XLS_RETURN_IF_ERROR(
        AddToReady(function, module, type_info, SymbolicBindings(), &ready));
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
