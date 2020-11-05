// Copyright 2020 The XLS Authors
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

#include "xls/dslx/cpp_evaluate.h"

#include "xls/common/status/ret_check.h"
#include "xls/dslx/type_info.h"

namespace xls::dslx {
namespace {

using Value = InterpValue;
using Tag = InterpValueTag;

}  // namespace

absl::StatusOr<InterpValue> EvaluateIndexBitslice(TypeInfo* type_info,
                                                  Index* expr,
                                                  InterpBindings* bindings,
                                                  const Bits& bits) {
  IndexRhs index = expr->rhs();
  XLS_RET_CHECK(absl::holds_alternative<Slice*>(index));
  auto index_slice = absl::get<Slice*>(index);

  const SymbolicBindings& sym_bindings = bindings->fn_ctx()->sym_bindings;

  absl::optional<SliceData::StartWidth> maybe_saw =
      type_info->GetSliceStartWidth(index_slice, sym_bindings);
  XLS_RET_CHECK(maybe_saw.has_value());
  const auto& saw = maybe_saw.value();
  return Value::MakeBits(Tag::kUBits, bits.Slice(saw.start, saw.width));
}

absl::StatusOr<InterpValue> EvaluateNameRef(NameRef* expr,
                                            InterpBindings* bindings,
                                            ConcreteType* type_context) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpValue> EvaluateConstRef(ConstRef* expr,
                                             InterpBindings* bindings,
                                             ConcreteType* type_context) {
  return bindings->ResolveValue(expr);
}

absl::StatusOr<InterpBindings> MakeTopLevelBindings(
    const std::shared_ptr<Module>& module, const TypecheckFn& typecheck,
    const EvaluateFn& eval, const IsWipFn& is_wip, const NoteWipFn& note_wip,
    ImportCache* cache) {
  XLS_VLOG(3) << "Making top level bindings for module: " << module->name();
  InterpBindings b(/*parent=*/nullptr);

  // Add all the builtin functions.
  for (Builtin builtin : kAllBuiltins) {
    b.AddFn(BuiltinToString(builtin), InterpValue::MakeFunction(builtin));
  }

  // Add all the functions in the top level scope for the module.
  for (Function* f : module->GetFunctions()) {
    b.AddFn(f->identifier(),
            InterpValue::MakeFunction(InterpValue::UserFnData{module, f}));
  }

  // Add all the type definitions in the top level scope for the module to the
  // bindings.
  for (TypeDefinition td : module->GetTypeDefinitions()) {
    if (absl::holds_alternative<TypeDef*>(td)) {
      auto* type_def = absl::get<TypeDef*>(td);
      b.AddTypeDef(type_def->identifier(), type_def);
    } else if (absl::holds_alternative<StructDef*>(td)) {
      auto* struct_def = absl::get<StructDef*>(td);
      b.AddStructDef(struct_def->identifier(), struct_def);
    } else {
      auto* enum_def = absl::get<EnumDef*>(td);
      b.AddEnumDef(enum_def->identifier(), enum_def);
    }
  }

  // Add constants/imports present at the top level to the bindings.
  for (ModuleMember member : module->top()) {
    XLS_VLOG(3) << "Evaluating module member: "
                << ToAstNode(member)->ToString();
    if (absl::holds_alternative<ConstantDef*>(member)) {
      auto* constant_def = absl::get<ConstantDef*>(member);
      if (is_wip(module, constant_def)) {
        XLS_VLOG(3) << "Saw WIP constant definition; breaking early! "
                    << constant_def->ToString();
        break;
      }
      XLS_VLOG(3) << "MakeTopLevelBindings evaluating: "
                  << constant_def->ToString();
      absl::optional<InterpValue> precomputed =
          note_wip(module, constant_def, absl::nullopt);
      absl::optional<InterpValue> result;
      if (precomputed.has_value()) {  // If we already computed it, use that.
        result = precomputed.value();
      } else {  // Otherwise, evaluate it and make a note.
        XLS_ASSIGN_OR_RETURN(result, eval(module, constant_def->value(), &b));
        note_wip(module, constant_def, *result);
      }
      XLS_CHECK(result.has_value());
      b.AddValue(constant_def->identifier(), *result);
      XLS_VLOG(3) << "MakeTopLevelBindings evaluated: "
                  << constant_def->ToString() << " to " << result->ToString();
      continue;
    }
    if (absl::holds_alternative<Import*>(member)) {
      auto* import = absl::get<Import*>(member);
      XLS_VLOG(3) << "MakeTopLevelBindings importing: " << import->ToString();
      XLS_ASSIGN_OR_RETURN(
          const ModuleInfo* imported,
          DoImport(typecheck, ImportTokens(import->subject()), cache));
      XLS_VLOG(3) << "MakeTopLevelBindings adding import " << import->ToString()
                  << " as \"" << import->identifier() << "\"";
      b.AddModule(import->identifier(), imported->module.get());
      continue;
    }
  }

  // Add a helpful value to the binding keys just to indicate what module these
  // top level bindings were created for, helpful for debugging.
  b.AddValue(absl::StrCat("__top_level_bindings_", module->name()),
             InterpValue::MakeNil());

  return b;
}

}  // namespace xls::dslx
