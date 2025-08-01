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

#include "xls/dslx/frontend/parser.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_builder.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/dslx/channel_direction.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/ast_cloner.h"
#include "xls/dslx/frontend/ast_utils.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/builtins_metadata.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/proc.h"
#include "xls/dslx/frontend/scanner_keywords.inc"
#include "xls/dslx/frontend/token.h"
#include "xls/ir/channel.h"
#include "xls/ir/code_template.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/name_uniquer.h"

namespace xls::dslx {
namespace {

constexpr std::string_view kConstAssertIdentifier = "const_assert!";

absl::StatusOr<std::vector<ExprOrType>> CloneParametrics(
    absl::Span<const ExprOrType> eots) {
  std::vector<ExprOrType> results;
  results.reserve(eots.size());
  for (const ExprOrType& eot : eots) {
    XLS_RETURN_IF_ERROR(
        absl::visit(Visitor{
                        [&](Expr* n) -> absl::Status {
                          XLS_ASSIGN_OR_RETURN(auto* cloned, CloneNode(n));
                          results.push_back(cloned);
                          return absl::OkStatus();
                        },
                        [&](TypeAnnotation* n) -> absl::Status {
                          XLS_ASSIGN_OR_RETURN(auto* cloned, CloneNode(n));
                          results.push_back(cloned);
                          return absl::OkStatus();
                        },
                    },
                    eot));
  }
  return results;
}

absl::Status MakeModuleTopCollisionError(const FileTable& file_table,
                                         std::string_view module_name,
                                         std::string_view member_name,
                                         const Span& existing_span,
                                         const AstNode* existing_node,
                                         const Span& new_span,
                                         const AstNode* new_node) {
  return ParseErrorStatus(
      new_span,
      absl::StrFormat("Module `%s` already contains a member named `%s` @ %s",
                      module_name, member_name,
                      existing_span.ToString(file_table)),
      file_table);
}

ColonRef::Subject CloneSubject(Module* module,
                               const ColonRef::Subject subject) {
  if (std::holds_alternative<NameRef*>(subject)) {
    NameRef* name_ref = std::get<NameRef*>(subject);
    return module->Make<NameRef>(name_ref->span(), name_ref->identifier(),
                                 name_ref->name_def());
  }

  ColonRef* colon_ref = std::get<ColonRef*>(subject);
  ColonRef::Subject clone_subject = CloneSubject(module, colon_ref->subject());
  return module->Make<ColonRef>(colon_ref->span(), clone_subject,
                                colon_ref->attr());
}

bool HasChannelElement(const TypeAnnotation* type) {
  if (const auto* array_type = dynamic_cast<const ArrayTypeAnnotation*>(type);
      array_type != nullptr) {
    return HasChannelElement(array_type->element_type());
  }
  if (const auto* channel_type =
          dynamic_cast<const ChannelTypeAnnotation*>(type);
      channel_type != nullptr) {
    return true;
  }
  if (const auto* tuple_type = dynamic_cast<const TupleTypeAnnotation*>(type);
      tuple_type != nullptr) {
    for (const auto* sub_type : tuple_type->members()) {
      if (HasChannelElement(sub_type)) {
        return true;
      }
    }
  }

  return false;
}

template <typename K, typename V>
std::vector<K> MapKeysSorted(const absl::flat_hash_map<K, V>& m) {
  std::vector<K> keys;
  keys.reserve(m.size());
  for (const auto& [k, v] : m) {
    keys.push_back(k);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

absl::StatusOr<TypeDefinition> BoundNodeToTypeDefinition(BoundNode bn) {
  // clang-format off
  if (auto* e = TryGet<TypeAlias*>(bn)) { return TypeDefinition(e); }
  if (auto* e = TryGet<StructDef*>(bn)) { return TypeDefinition(e); }
  if (auto* e = TryGet<ProcDef*>(bn)) { return TypeDefinition(e); }
  if (auto* e = TryGet<EnumDef*>(bn)) { return TypeDefinition(e); }
  if (auto* e = TryGet<UseTreeEntry*>(bn)) { return TypeDefinition(e); }
  // clang-format on

  return absl::InvalidArgumentError("Could not convert to type definition: " +
                                    ToAstNode(bn)->ToString());
}

bool IsComparisonBinopKind(const Expr* e) {
  auto* binop = dynamic_cast<const Binop*>(e);
  if (binop == nullptr) {
    return false;
  }
  return GetBinopComparisonKinds().contains(binop->binop_kind());
}

ExprRestrictions MakeRestrictions(
    const std::vector<ExprRestriction>& restrictions) {
  uint64_t value = 0;
  for (ExprRestriction restriction : restrictions) {
    value |= static_cast<uint64_t>(restriction);
  }
  return ExprRestrictions(value);
}

bool IsExprRestrictionEnabled(ExprRestrictions restrictions,
                              ExprRestriction target) {
  uint64_t target_u64 = static_cast<uint64_t>(target);
  CHECK_NE(target_u64, 0);

  // All restriction values should be a pow2.
  CHECK_EQ(target_u64 & (target_u64 - 1), 0);

  return (static_cast<uint64_t>(restrictions) & target_u64) != 0;
}

std::string ExprRestrictionsToString(ExprRestrictions restrictions) {
  if (restrictions == kNoRestrictions) {
    return "none";
  }
  // Note: right now we only have one flag that can be in the set.
  CHECK_EQ(static_cast<uint64_t>(restrictions),
           static_cast<uint64_t>(ExprRestriction::kNoStructLiteral));
  return "{no-struct-literal}";
}

}  // namespace

absl::StatusOr<BuiltinType> Parser::TokenToBuiltinType(const Token& tok) {
  return BuiltinTypeFromString(*tok.GetValue());
}

absl::Status Parser::ParseErrorStatus(const Span& span,
                                      std::string_view message) const {
  return xls::dslx::ParseErrorStatus(span, message, file_table());
}

absl::StatusOr<Function*> Parser::ParseImplFunction(
    const Pos& start_pos, bool is_public, Bindings& bindings,
    TypeAnnotation* struct_ref) {
  return ParseFunctionInternal(start_pos, is_public, /*is_test_utility=*/false,
                               bindings, struct_ref);
}

absl::StatusOr<Function*> Parser::ParseFunction(
    const Pos& start_pos, bool is_public, bool is_test_utility,
    Bindings& bindings,
    absl::flat_hash_map<std::string, Function*>* name_to_fn) {
  XLS_ASSIGN_OR_RETURN(
      Function * f,
      ParseFunctionInternal(start_pos, is_public, is_test_utility, bindings));
  if (name_to_fn == nullptr) {
    return f;
  }
  auto [item, inserted] = name_to_fn->insert({f->identifier(), f});
  if (!inserted) {
    return ParseErrorStatus(
        f->name_def()->span(),
        absl::StrFormat("Function '%s' is defined in this module multiple "
                        "times; previously @ %s'",
                        f->identifier(),
                        item->second->span().ToString(file_table())));
  }
  XLS_RETURN_IF_ERROR(VerifyParentage(f));
  return f;
}

// Lambda syntax: | <PARAM>[: <TYPE>], ... | [-> <RETURN_TYPE>] { <BODY> }
absl::StatusOr<Lambda*> Parser::ParseLambda(Bindings& bindings) {
  Pos start_pos = GetPos();
  VLOG(5) << "ParseLambda @ " << start_pos;
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  std::vector<Param*> params;
  if (peek->kind() == TokenKind::kBar) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kBar));
    auto parse_param = [&] { return ParseParam(bindings); };
    XLS_ASSIGN_OR_RETURN(params,
                         ParseCommaSeq<Param*>(parse_param, TokenKind::kBar));
  } else {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDoubleBar));
  }

  XLS_ASSIGN_OR_RETURN(bool dropped_arrow, TryDropToken(TokenKind::kArrow));
  TypeAnnotation* return_type = nullptr;
  if (dropped_arrow) {
    XLS_ASSIGN_OR_RETURN(return_type, ParseTypeAnnotation(bindings));
  }

  XLS_ASSIGN_OR_RETURN(StatementBlock * body, ParseBlockExpression(bindings));
  return module_->Make<Lambda>(Span(start_pos, GetPos()), params, return_type,
                               body);
}

absl::Status Parser::ParseModuleAttribute() {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrack));
  Span attribute_span;
  XLS_ASSIGN_OR_RETURN(std::string attribute,
                       PopIdentifierOrError(&attribute_span));
  if (attribute == "feature") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
    XLS_ASSIGN_OR_RETURN(std::string feature, PopIdentifierOrError());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    if (feature == "use_syntax") {
      module_->AddAttribute(ModuleAttribute::kAllowUseSyntax, attribute_span);
    } else if (feature == "type_inference_v1") {
      module_->AddAttribute(ModuleAttribute::kTypeInferenceVersion1,
                            attribute_span);
    } else if (feature == "type_inference_v2") {
      module_->AddAttribute(ModuleAttribute::kTypeInferenceVersion2,
                            attribute_span);
    } else if (feature == "channel_attributes") {
      module_->AddAttribute(ModuleAttribute::kChannelAttributes,
                            attribute_span);
    } else {
      return ParseErrorStatus(
          attribute_span,
          absl::StrFormat("Unsupported feature: `%s`", feature));
    }

    if (module_->attributes().contains(
            ModuleAttribute::kTypeInferenceVersion1) &&
        module_->attributes().contains(
            ModuleAttribute::kTypeInferenceVersion2)) {
      return ParseErrorStatus(attribute_span,
                              "Module cannot have both `type_inference_v1` and "
                              "`type_inference_v2` attributes.");
    }
    return absl::OkStatus();
  }
  if (attribute != "allow") {
    return ParseErrorStatus(
        attribute_span,
        absl::StrCat("Unsupported module-level attribute: ", attribute));
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  XLS_ASSIGN_OR_RETURN(std::string to_allow, PopIdentifierOrError());
  if (to_allow == "nonstandard_constant_naming") {
    module_->AddAttribute(ModuleAttribute::kAllowNonstandardConstantNaming);
  }
  if (to_allow == "nonstandard_member_naming") {
    module_->AddAttribute(ModuleAttribute::kAllowNonstandardMemberNaming);
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Module>> Parser::ParseModule(
    Bindings* bindings) {
  const Pos module_start_pos = GetPos();

  std::optional<Bindings> stack_bindings;
  if (bindings == nullptr) {
    stack_bindings.emplace();
    bindings = &*stack_bindings;
  }
  XLS_RET_CHECK(bindings != nullptr);

  for (auto const& it : GetParametricBuiltins()) {
    std::string name(it.first);
    bindings->Add(name, module_->GetOrCreateBuiltinNameDef(name));
  }

#define ADD_SIZED_TYPE_KEYWORD(__enum, __caps, __str) \
  bindings->Add(__str, module_->GetOrCreateBuiltinNameDef(__str));
  XLS_DSLX_SIZED_TYPE_KEYWORDS(ADD_SIZED_TYPE_KEYWORD);
#undef ADD_SIZED_TYPE_KEYWORD

  absl::flat_hash_map<std::string, Function*> name_to_fn;

  auto make_collision_error =
      absl::bind_front(&MakeModuleTopCollisionError, file_table());

  while (!AtEof()) {
    XLS_ASSIGN_OR_RETURN(bool peek_is_eof, PeekTokenIs(TokenKind::kEof));
    if (peek_is_eof) {
      VLOG(3) << "Parser saw EOF token for module " << module_->name()
              << ", stopping.";
      break;
    }

    const Pos start_pos = GetPos();
    XLS_ASSIGN_OR_RETURN(bool dropped_pub, TryDropKeyword(Keyword::kPub));
    if (dropped_pub) {
      XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
      if (peek->IsKeyword(Keyword::kFn)) {
        XLS_ASSIGN_OR_RETURN(
            Function * fn,
            ParseFunction(start_pos,
                          /*is_public=*/true, /*is_test_utility=*/false,
                          *bindings, &name_to_fn));
        XLS_RETURN_IF_ERROR(module_->AddTop(fn, make_collision_error));
        continue;
      }

      if (peek->IsKeyword(Keyword::kProc)) {
        XLS_ASSIGN_OR_RETURN(ModuleMember proc,
                             ParseProc(start_pos,
                                       /*is_public=*/true,
                                       /*is_test_utility=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(proc, make_collision_error));
        continue;
      }

      if (peek->IsKeyword(Keyword::kStruct)) {
        XLS_ASSIGN_OR_RETURN(
            StructDef * struct_def,
            ParseStruct(start_pos, /*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(struct_def, make_collision_error));
        continue;
      }

      if (peek->IsKeyword(Keyword::kImpl)) {
        XLS_ASSIGN_OR_RETURN(
            Impl * impl, ParseImpl(start_pos, /*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(impl, make_collision_error));
        continue;
      }

      if (peek->IsKeyword(Keyword::kEnum)) {
        XLS_ASSIGN_OR_RETURN(
            EnumDef * enum_def,
            ParseEnumDef(start_pos, /*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(enum_def, make_collision_error));
        continue;
      }

      if (peek->IsKeyword(Keyword::kConst)) {
        XLS_ASSIGN_OR_RETURN(
            ConstantDef * constant_def,
            ParseConstantDef(start_pos, /*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(constant_def, make_collision_error));
        continue;
      }

      if (peek->IsKeyword(Keyword::kType)) {
        XLS_RET_CHECK(bindings != nullptr);
        XLS_ASSIGN_OR_RETURN(
            TypeAlias * type_alias,
            ParseTypeAlias(start_pos, /*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(type_alias, make_collision_error));
        continue;
      }

      return ParseErrorStatus(peek->span(),
                              "Expect a function, proc, struct, enum, or type "
                              "after 'pub' keyword.");
    }

    XLS_ASSIGN_OR_RETURN(std::optional<Token> hash,
                         TryPopToken(TokenKind::kHash));
    if (hash.has_value()) {
      XLS_ASSIGN_OR_RETURN(bool dropped_bang, TryDropToken(TokenKind::kBang));
      if (dropped_bang) {
        XLS_RETURN_IF_ERROR(ParseModuleAttribute());
        continue;
      }

      XLS_ASSIGN_OR_RETURN(
          auto attribute,
          ParseAttribute(&name_to_fn, *bindings, hash->span().start()));
      XLS_RETURN_IF_ERROR(absl::visit(
          Visitor{
              [&](auto* t) { return module_->AddTop(t, make_collision_error); },
              [&](TypeDefinition def) {
                return absl::visit(
                    Visitor{
                        [&](ColonRef* c) {
                          return absl::FailedPreconditionError(
                              "ColonRef cannot be annotated with an attribute");
                        },
                        [&](auto* t) {
                          return module_->AddTop(t, make_collision_error);
                        },
                        [&](UseTreeEntry* n) {
                          return absl::FailedPreconditionError(
                              "UseTreeEntry cannot be annotated with an "
                              "attribute");
                        }},
                    def);
              },
              [&](std::nullptr_t) { return absl::OkStatus(); },
          },
          attribute));
      continue;
    }

    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());

    if (peek->IsIdentifier(kConstAssertIdentifier)) {
      XLS_ASSIGN_OR_RETURN(ConstAssert * const_assert,
                           ParseConstAssert(*bindings));
      // Note: const_assert! doesn't make a binding so we don't need to provide
      // the error lambda.
      XLS_RETURN_IF_ERROR(
          module_->AddTop(const_assert, /*make_collision_error=*/nullptr));
      continue;
    }

    auto top_level_error = [peek, this] {
      return ParseErrorStatus(
          peek->span(),
          absl::StrFormat("Expected start of top-level construct; got: '%s'",
                          peek->ToString()));
    };
    if (peek->kind() != TokenKind::kKeyword) {
      return top_level_error();
    }

    XLS_RET_CHECK(bindings != nullptr);
    switch (peek->GetKeyword()) {
      case Keyword::kFn: {
        XLS_ASSIGN_OR_RETURN(
            Function * fn,
            ParseFunction(GetPos(),
                          /*is_public=*/false, /*is_test_utility=*/false,
                          *bindings, &name_to_fn));
        XLS_RETURN_IF_ERROR(module_->AddTop(fn, make_collision_error));
        break;
      }
      case Keyword::kProc: {
        XLS_ASSIGN_OR_RETURN(ModuleMember proc,
                             ParseProc(GetPos(),
                                       /*is_public=*/false,
                                       /*is_test_utility=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(proc, make_collision_error));
        break;
      }
      case Keyword::kImport: {
        if (module_->attributes().contains(ModuleAttribute::kAllowUseSyntax)) {
          return ParseErrorStatus(
              peek->span(),
              "`import` syntax is disabled for this module via "
              "`#![feature(use_syntax)]` at module scope; use `use` instead");
        }
        XLS_ASSIGN_OR_RETURN(Import * import, ParseImport(*bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(import, make_collision_error));
        break;
      }
      case Keyword::kUse: {
        if (!module_->attributes().contains(ModuleAttribute::kAllowUseSyntax)) {
          return ParseErrorStatus(
              peek->span(),
              "`use` syntax is not enabled for this module; enable with "
              "`#![feature(use_syntax)]` at module scope");
        }
        XLS_ASSIGN_OR_RETURN(Use * use, ParseUse(*bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(use, make_collision_error));
        break;
      }
      case Keyword::kType: {
        XLS_RET_CHECK(bindings != nullptr);
        XLS_ASSIGN_OR_RETURN(
            TypeAlias * type_alias,
            ParseTypeAlias(GetPos(), /*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(type_alias, make_collision_error));
        break;
      }
      case Keyword::kStruct: {
        XLS_ASSIGN_OR_RETURN(
            StructDef * struct_def,
            ParseStruct(start_pos, /*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(struct_def, make_collision_error));
        break;
      }
      case Keyword::kEnum: {
        XLS_ASSIGN_OR_RETURN(
            EnumDef * enum_def,
            ParseEnumDef(start_pos, /*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(enum_def, make_collision_error));
        break;
      }
      case Keyword::kConst: {
        XLS_ASSIGN_OR_RETURN(
            ConstantDef * const_def,
            ParseConstantDef(start_pos, /*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(const_def, make_collision_error));
        break;
      }
      case Keyword::kImpl: {
        XLS_ASSIGN_OR_RETURN(
            Impl * impl, ParseImpl(start_pos, /*is_public=*/false, *bindings));

        XLS_RETURN_IF_ERROR(module_->AddTop(impl, make_collision_error));
        break;
      }

      default:
        return top_level_error();
    }
  }

  // Ensure we've consumed all tokens when we're done parsing, as a
  // post-condition.
  XLS_RET_CHECK(AtEof());

  XLS_RETURN_IF_ERROR(VerifyParentage(module_.get()));

  module_->set_span(Span(module_start_pos, GetPos()));

  auto result = std::move(module_);
  module_ = nullptr;
  return result;
}

absl::StatusOr<ChannelConfig> Parser::ParseExprAttribute(Bindings& bindings,
                                                         const Pos& hash_pos) {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrack));
  XLS_ASSIGN_OR_RETURN(
      Token attribute_tok,
      PopTokenOrError(TokenKind::kIdentifier, /*start=*/nullptr,
                      "Expected attribute identifier"));
  const std::string& attribute_name = attribute_tok.GetStringValue();
  if (attribute_name == "channel") {
    // TODO: google/xls#1023 - consider moving or allowing the channel attribute
    // to the outside of the let binding, e.g.
    // #[channel()] let _ = chan<...>("...");
    ChannelKind kind = ChannelKind::kStreaming;
    std::optional<int64_t> depth;
    std::optional<bool> bypass;
    std::optional<bool> register_push_outputs;
    std::optional<bool> register_pop_outputs;
    std::optional<FlopKind> input_flop_kind;
    std::optional<FlopKind> output_flop_kind;
    auto parse_kv_pair = [&]() -> absl::StatusOr<std::monostate> {
      XLS_ASSIGN_OR_RETURN(Token key_tok,
                           PopTokenOrError(TokenKind::kIdentifier));
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
      std::string_view key = key_tok.GetStringValue();
      if (key == "kind") {
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kIdentifier));
        XLS_ASSIGN_OR_RETURN(kind,
                             StringToChannelKind(value_tok.GetStringValue()));
      } else if (key == "depth") {
        depth.emplace();
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kNumber));
        XLS_ASSIGN_OR_RETURN(*depth, value_tok.GetValueAsInt64());
      } else if (key == "register_push_outputs") {
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kKeyword));
        switch (value_tok.GetKeyword()) {
          case xls::dslx::Keyword::kTrue:
            register_push_outputs.emplace(true);
            break;
          case xls::dslx::Keyword::kFalse:
            register_push_outputs.emplace(false);
            break;
          default:
            return ParseErrorStatus(
                value_tok.span(),
                absl::StrFormat("Expected boolean for register_push_outputs, "
                                "got %s.",
                                value_tok.ToErrorString()));
        }
      } else if (key == "register_pop_outputs") {
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kKeyword));
        switch (value_tok.GetKeyword()) {
          case xls::dslx::Keyword::kTrue:
            register_pop_outputs.emplace(true);
            break;
          case xls::dslx::Keyword::kFalse:
            register_pop_outputs.emplace(false);
            break;
          default:
            return ParseErrorStatus(
                value_tok.span(),
                absl::StrFormat("Expected boolean for register_pop_outputs, "
                                "got %s.",
                                value_tok.ToErrorString()));
        }
      } else if (key == "bypass") {
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kKeyword));
        switch (value_tok.GetKeyword()) {
          case xls::dslx::Keyword::kTrue:
            bypass.emplace(true);
            break;
          case xls::dslx::Keyword::kFalse:
            bypass.emplace(false);
            break;
          default:
            return ParseErrorStatus(
                value_tok.span(),
                absl::StrFormat("Expected boolean for bypass, got %s.",
                                value_tok.ToErrorString()));
        }
      } else if (key == "input_flop_kind") {
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kIdentifier));
        std::string_view value = value_tok.GetStringValue();
        absl::StatusOr<FlopKind> flop_kind = StringToFlopKind(value);
        if (!flop_kind.ok()) {
          return ParseErrorStatus(
              value_tok.span(),
              absl::StrFormat("Expected flop kind for output_flop_kind, got "
                              "error %s for value %s.",
                              flop_kind.status().message(), value));
        }
        input_flop_kind.emplace(*flop_kind);
      } else if (key == "output_flop_kind") {
        XLS_ASSIGN_OR_RETURN(Token value_tok,
                             PopTokenOrError(TokenKind::kIdentifier));
        std::string_view value = value_tok.GetStringValue();
        absl::StatusOr<FlopKind> flop_kind = StringToFlopKind(value);
        if (!flop_kind.ok()) {
          return ParseErrorStatus(
              value_tok.span(),
              absl::StrFormat("Expected flop kind for output_flop_kind, got "
                              "error %s for value %s.",
                              flop_kind.status().message(), value));
        }
        output_flop_kind.emplace(*flop_kind);
      } else {
        return ParseErrorStatus(
            key_tok.span(),
            absl::StrFormat("Unknown key %s for channel attribute.", key));
      }
      return std::monostate();
    };
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
    XLS_RETURN_IF_ERROR(
        ParseCommaSeq<std::monostate>(parse_kv_pair, TokenKind::kCParen)
            .status());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    std::optional<FifoConfig> fifo_config;
    if (depth.has_value() || bypass.has_value() ||
        register_push_outputs.has_value() || register_pop_outputs.has_value()) {
      depth = depth.value_or(1);
      bypass = bypass.value_or(*depth == 0 ? true : false);
      register_push_outputs = register_push_outputs.value_or(false);
      register_pop_outputs = register_pop_outputs.value_or(false);
      fifo_config.emplace(*depth, *bypass, *register_push_outputs,
                          *register_pop_outputs);
    }
    ChannelConfig channel_config;
    if (fifo_config.has_value() || input_flop_kind.has_value() ||
        output_flop_kind.has_value()) {
      channel_config = channel_config.WithFifoConfig(fifo_config);
      channel_config = channel_config.WithInputFlopKind(input_flop_kind);
      channel_config = channel_config.WithOutputFlopKind(output_flop_kind);
    }
    return channel_config;
  }
  return ParseErrorStatus(
      attribute_tok.span(),
      absl::StrFormat("Unknown attribute: '%s'", attribute_name));
}

absl::StatusOr<std::variant<TestFunction*, Function*, TestProc*, Proc*,
                            QuickCheck*, TypeDefinition, std::nullptr_t>>
Parser::ParseAttribute(absl::flat_hash_map<std::string, Function*>* name_to_fn,
                       Bindings& bindings, const Pos& hash_pos) {
  // Ignore the Rust "bang" in Attribute declarations, i.e. we don't yet have
  // a use for inner vs. outer attributes, but that day will likely come.
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrack));
  XLS_ASSIGN_OR_RETURN(
      Token attribute_tok,
      PopTokenOrError(TokenKind::kIdentifier, /*start=*/nullptr,
                      "Expected attribute identifier"));
  const std::string& attribute_name = attribute_tok.GetStringValue();

  if (attribute_name == "dslx_format_disable") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    XLS_ASSIGN_OR_RETURN(bool is_public, TryDropKeyword(Keyword::kPub));
    XLS_ASSIGN_OR_RETURN(Function * fn, ParseFunction(hash_pos, is_public,
                                                      /*is_test_utility=*/false,
                                                      bindings, name_to_fn));
    fn->set_disable_format(true);
    return fn;
  }
  if (attribute_name == "cfg") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
    XLS_ASSIGN_OR_RETURN(Token parameter_name,
                         PopTokenOrError(TokenKind::kIdentifier));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    if (parameter_name.GetStringValue() != "test") {
      return ParseErrorStatus(
          attribute_tok.span(),
          absl::StrFormat(
              "Unknown parameter name in the #[cfg()] attribute: '%s'",
              parameter_name.ToString()));
    }

    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    XLS_ASSIGN_OR_RETURN(bool is_public, TryDropKeyword(Keyword::kPub));

    XLS_ASSIGN_OR_RETURN(const Token* t, PeekToken());
    if (t->IsKeyword(Keyword::kFn)) {
      XLS_ASSIGN_OR_RETURN(
          Function * fn,
          ParseFunction(hash_pos, is_public, /*is_test_utility=*/true, bindings,
                        name_to_fn));
      return fn;
    } else if (t->IsKeyword(Keyword::kProc)) {
      XLS_ASSIGN_OR_RETURN(ModuleMember m,
                           ParseProc(GetPos(), /*is_public=*/false,
                                     /*is_test_utility=*/true, bindings));
      Proc* p = std::get<Proc*>(m);
      return p;
    } else {
      return ParseErrorStatus(
          attribute_tok.span(),
          "#[cfg()] attribute should only be used for functions and procs");
    }
  }
  if (attribute_name == "extern_verilog") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
    Pos template_start = GetPos();
    Pos template_limit;
    XLS_ASSIGN_OR_RETURN(
        Token ffi_annotation_token,
        PopTokenOrError(TokenKind::kString, /*start=*/nullptr,
                        "extern_verilog template", &template_limit));
    std::string ffi_annotation = *ffi_annotation_token.GetValue();
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    XLS_ASSIGN_OR_RETURN(bool dropped_pub, TryDropKeyword(Keyword::kPub));
    XLS_ASSIGN_OR_RETURN(
        Function * f,
        ParseFunction(template_start, dropped_pub,
                      /*is_test_utility=*/false, bindings, name_to_fn));
    absl::StatusOr<ForeignFunctionData> parsed_ffi_annotation =
        ForeignFunctionDataCreateFromTemplate(ffi_annotation);
    if (!parsed_ffi_annotation.ok()) {
      const int64_t error_at =
          CodeTemplate::ExtractErrorColumn(parsed_ffi_annotation.status());
      Pos error_pos{template_start.fileno(), template_start.lineno(),
                    template_start.colno() + error_at};
      dslx::Span error_span(error_pos, error_pos);
      return ParseErrorStatus(error_span,
                              parsed_ffi_annotation.status().message());
    }
    f->set_extern_verilog_module(*parsed_ffi_annotation);
    return f;
  }
  if (attribute_name == "quickcheck") {
    return ParseQuickCheck(name_to_fn, bindings, hash_pos);
  }
  if (attribute_name == "sv_type") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
    Pos ident_limit;
    XLS_ASSIGN_OR_RETURN(
        Token sv_type_id,
        PopTokenOrError(TokenKind::kString, /*start=*/&attribute_tok,
                        "sv_type identifier", &ident_limit));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    XLS_ASSIGN_OR_RETURN(const Token* t, PeekToken());
    bool is_pub = false;
    const Pos start_pos = GetPos();
    if (t->IsKeyword(Keyword::kPub)) {
      is_pub = true;
      XLS_RETURN_IF_ERROR(PopToken().status());
      XLS_ASSIGN_OR_RETURN(t, PeekToken());
    }
    if (t->IsKeyword(Keyword::kType)) {
      XLS_ASSIGN_OR_RETURN(TypeAlias * type_alias,
                           ParseTypeAlias(start_pos, is_pub, bindings));
      type_alias->set_extern_type_name(sv_type_id.GetStringValue());
      return type_alias;
    }
    if (t->IsKeyword(Keyword::kStruct)) {
      XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                           ParseStruct(start_pos, is_pub, bindings));
      struct_def->set_extern_type_name(sv_type_id.GetStringValue());
      return struct_def;
    }
    if (t->IsKeyword(Keyword::kEnum)) {
      XLS_ASSIGN_OR_RETURN(EnumDef * enum_def,
                           ParseEnumDef(start_pos, is_pub, bindings));
      enum_def->set_extern_type_name(sv_type_id.GetStringValue());
      return enum_def;
    }
    return ParseErrorStatus(attribute_tok.span(),
                            "#[sv_type(\"name\")] is only valid on type-alias, "
                            "struct or enum definitions");
  }
  if (attribute_name == "test") {
    XLS_ASSIGN_OR_RETURN(Token cbrack, PopTokenOrError(TokenKind::kCBrack));
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(Keyword::kFn)) {
      return ParseTestFunction(bindings, Span(hash_pos, cbrack.span().limit()));
    }

    return ParseErrorStatus(
        peek->span(), absl::StrCat("Invalid test type: ", peek->ToString()));
  }
  if (attribute_name == "test_proc") {
    std::optional<std::string> expected_fail_label = std::nullopt;
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->kind() == TokenKind::kOParen) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
      XLS_ASSIGN_OR_RETURN(Token parameter_name,
                           PopTokenOrError(TokenKind::kIdentifier));

      if (parameter_name.GetStringValue() == "expected_fail_label") {
        XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
        XLS_ASSIGN_OR_RETURN(Token parameter_value,
                             PopTokenOrError(TokenKind::kString));
        expected_fail_label = parameter_value.GetStringValue();
      } else {
        return ParseErrorStatus(
            attribute_tok.span(),
            absl::StrFormat(
                "Unknown parameter name in the #[test_proc] attribute: '%s'",
                parameter_name.ToString()));
      }
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    return ParseTestProc(bindings, expected_fail_label);
  }
  return ParseErrorStatus(
      attribute_tok.span(),
      absl::StrFormat("Unknown attribute: '%s'", attribute_name));
}

absl::StatusOr<Expr*> Parser::ParseExpression(Bindings& bindings,
                                              ExprRestrictions restrictions) {
  XLS_ASSIGN_OR_RETURN(std::optional<Token> hash,
                       TryPopToken(TokenKind::kHash));
  std::optional<ChannelConfig> channel_config;
  if (hash.has_value()) {
    absl::flat_hash_map<std::string, std::string> attributes;

    XLS_ASSIGN_OR_RETURN(channel_config,
                         ParseExprAttribute(bindings, hash->span().start()));
    if (channel_config.has_value() &&
        !module_->attributes().contains(ModuleAttribute::kChannelAttributes)) {
      return ParseErrorStatus(
          hash->span(),
          "#[channel()] syntax is not enabled for this module; enable with "
          "`#![feature(channel_attributes)]` at module scope");
    }
  }

  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (channel_config.has_value() && !peek->IsKeyword(Keyword::kChan)) {
    return ParseErrorStatus(
        peek->span(), absl::StrFormat("Channel config must be specified before "
                                      "a channel declaration (got %s).",
                                      peek->ToString()));
  }

  XLS_ASSIGN_OR_RETURN(ExpressionDepthGuard expr_depth, BumpExpressionDepth());

  VLOG(5) << "ParseExpression @ " << GetPos() << " peek: `" << peek->ToString()
          << "`";
  if (peek->IsKeyword(Keyword::kFor)) {
    return ParseFor(bindings);
  }
  if (peek->IsKeyword(Keyword::kUnrollFor)) {
    return ParseUnrollFor(bindings);
  }
  if (peek->IsKeyword(Keyword::kChan)) {
    return ParseChannelDecl(bindings, channel_config);
  }
  if (peek->IsKeyword(Keyword::kSpawn)) {
    return ParseSpawn(bindings);
  }
  if (peek->kind() == TokenKind::kOBrace) {
    return ParseBlockExpression(bindings);
  }
  if (peek->kind() == TokenKind::kBar ||
      peek->kind() == TokenKind::kDoubleBar) {
    return ParseLambda(bindings);
  }
  return ParseRangeExpression(bindings, restrictions);
}

absl::StatusOr<Expr*> Parser::ParseRangeExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  VLOG(5) << "ParseRangeExpression @ " << GetPos();
  XLS_ASSIGN_OR_RETURN(Expr * result,
                       ParseLogicalOrExpression(bindings, restrictions));
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (peek->kind() == TokenKind::kDoubleDot ||
      peek->kind() == TokenKind::kDoubleDotEquals) {
    bool inclusive_end = peek->kind() == TokenKind::kDoubleDotEquals;
    DropTokenOrDie();
    XLS_ASSIGN_OR_RETURN(Expr * rhs,
                         ParseLogicalOrExpression(bindings, restrictions));
    result =
        module_->Make<Range>(Span(result->span().start(), rhs->span().limit()),
                             result, inclusive_end, rhs);
  }
  return result;
}

absl::StatusOr<Conditional*> Parser::ParseConditionalNode(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_ASSIGN_OR_RETURN(Token if_kw, PopKeywordOrError(Keyword::kIf));
  XLS_ASSIGN_OR_RETURN(
      Expr * test,
      ParseExpression(bindings,
                      MakeRestrictions({ExprRestriction::kNoStructLiteral})));
  XLS_ASSIGN_OR_RETURN(StatementBlock * consequent,
                       ParseBlockExpression(bindings));

  XLS_ASSIGN_OR_RETURN(bool has_else, PeekTokenIs(Keyword::kElse));

  std::variant<StatementBlock*, Conditional*> alternate;
  if (has_else) {
    XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kElse));
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(Keyword::kIf)) {  // else if
      XLS_ASSIGN_OR_RETURN(alternate,
                           ParseConditionalNode(bindings, kNoRestrictions));
    } else {  // normal else
      XLS_ASSIGN_OR_RETURN(alternate, ParseBlockExpression(bindings));
    }
  } else {  // if without else, add artificial empty block
    alternate = module_->Make<StatementBlock>(Span(GetPos(), GetPos()),
                                              std::vector<Statement*>(),
                                              /*trailing_semi=*/true);
  }

  auto* outer_conditional = module_->Make<Conditional>(
      Span(if_kw.span().start(), GetPos()), test, consequent, alternate,
      /*in_parens=*/false, has_else);
  for (StatementBlock* block : outer_conditional->GatherBlocks()) {
    block->SetEnclosing(outer_conditional);
  }
  return outer_conditional;
}

absl::StatusOr<ConstAssert*> Parser::ParseConstAssert(
    Bindings& bindings, const Token* ABSL_NULLABLE identifier) {
  Pos start;
  if (identifier == nullptr) {
    Span identifier_span;
    XLS_ASSIGN_OR_RETURN(std::string identifier_value,
                         PopIdentifierOrError(&identifier_span));
    start = identifier_span.start();
    XLS_RET_CHECK_EQ(identifier_value, "const_assert!");
  } else {
    start = identifier->span().start();
  }

  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kOParen, /*start=*/nullptr,
                       "Expected a '(' after const_assert! macro"));
  XLS_ASSIGN_OR_RETURN(Expr * arg, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kCParen, /*start=*/nullptr,
                       "Expected a ')' after const_assert! argument"));
  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kSemi, /*start=*/nullptr,
                       "Expected a ';' after const_assert! statement"));
  Pos limit = GetPos();
  const Span span(start, limit);
  return module_->Make<ConstAssert>(span, arg);
}

absl::StatusOr<TypeAlias*> Parser::ParseTypeAlias(const Pos& start_pos,
                                                  bool is_public,
                                                  Bindings& bindings) {
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kType));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDefNoBind());
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemi));
  Span span(start_pos, GetPos());
  auto* type_alias =
      module_->Make<TypeAlias>(span, *name_def, *type, is_public);
  name_def->set_definer(type_alias);
  bindings.Add(name_def->identifier(), type_alias);
  return type_alias;
}

absl::StatusOr<Number*> Parser::TokenToNumber(const Token& tok) {
  NumberKind kind;
  switch (tok.kind()) {
    case TokenKind::kCharacter:
      kind = NumberKind::kCharacter;
      break;
    case TokenKind::kKeyword:
      kind = NumberKind::kBool;
      break;
    default:
      kind = NumberKind::kOther;
      break;
  }
  return module_->Make<Number>(tok.span(), *tok.GetValue(), kind,
                               /*type=*/nullptr);
}

absl::StatusOr<TypeRefOrAnnotation> Parser::ParseTypeRef(Bindings& bindings,
                                                         const Token& tok) {
  if (tok.kind() != TokenKind::kIdentifier) {
    return ParseErrorStatus(tok.span(), absl::StrFormat("Expected type; got %s",
                                                        tok.ToErrorString()));
  }
  VLOG(5) << "ParseTypeRef token " << tok.ToString();

  XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  if (peek_is_double_colon) {
    return ParseModTypeRef(bindings, tok);
  }
  XLS_ASSIGN_OR_RETURN(
      BoundNode type_def,
      bindings.ResolveNodeOrError(*tok.GetValue(), tok.span(), file_table()));
  if (std::holds_alternative<NameDef*>(type_def)) {
    NameDef* name_def = std::get<NameDef*>(type_def);
    if (auto gta = dynamic_cast<GenericTypeAnnotation*>(name_def->definer())) {
      VLOG(5) << "ParseTypeRef ResolveNode to GenericTypeAnnotation: "
              << name_def->definer()->ToString();

      return module_->Make<TypeVariableTypeAnnotation>(
          module_->Make<NameRef>(tok.span(), name_def->identifier(), name_def));
    }
  }
  if (!IsOneOf<TypeAlias, EnumDef, StructDef, ProcDef, UseTreeEntry>(
          ToAstNode(type_def))) {
    return ParseErrorStatus(
        tok.span(),
        absl::StrFormat(
            "Expected a type, but identifier '%s' doesn't resolve to "
            "a type, it resolved to a %s",
            *tok.GetValue(), BoundNodeGetTypeString(type_def)));
  }

  XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                       BoundNodeToTypeDefinition(type_def));
  return module_->Make<TypeRef>(tok.span(), type_definition);
}

absl::StatusOr<TypeAnnotation*> Parser::ParseTypeAnnotation(
    Bindings& bindings, std::optional<Token> first, bool allow_generic_type) {
  XLS_ASSIGN_OR_RETURN(ExpressionDepthGuard expr_depth, BumpExpressionDepth());

  VLOG(5) << "ParseTypeAnnotation @ " << GetPos();
  if (!first.has_value()) {
    XLS_ASSIGN_OR_RETURN(first, PopToken());
  }
  const Token& tok = first.value();
  VLOG(5) << "ParseTypeAnnotation; popped: " << tok.ToString()
          << " is type keyword? " << tok.IsTypeKeyword();

  if (tok.IsTypeKeyword()) {  // Builtin types.
    Pos start_pos = tok.span().start();
    if (tok.GetKeyword() == Keyword::kSelfType) {
      return ParseErrorStatus(tok.span(),
                              "Parameter with type `Self` must be named `self` "
                              "and can only be used in `impl` functions");
    }
    if (allow_generic_type && tok.GetKeyword() == Keyword::kType) {
      return module_->Make<GenericTypeAnnotation>(tok.span());
    }
    if (tok.GetKeyword() == Keyword::kChan) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOAngle));
      XLS_ASSIGN_OR_RETURN(TypeAnnotation * payload,
                           ParseTypeAnnotation(bindings));
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCAngle));

      XLS_ASSIGN_OR_RETURN(bool peek_is_obrack,
                           PeekTokenIs(TokenKind::kOBrack));
      std::optional<std::vector<Expr*>> dims;
      if (peek_is_obrack) {
        XLS_ASSIGN_OR_RETURN(dims, ParseDims(bindings));
      }

      XLS_ASSIGN_OR_RETURN(Token tok, PopToken());
      if (tok.kind() != TokenKind::kKeyword) {
        return ParseErrorStatus(
            tok.span(),
            absl::StrFormat(
                "Expected channel direction (\"in\" or \"out\"); got %s.",
                tok.ToString()));
      }
      if (tok.GetKeyword() != Keyword::kIn &&
          tok.GetKeyword() != Keyword::kOut) {
        return ParseErrorStatus(
            tok.span(),
            absl::StrFormat(
                "Expected channel direction (\"in\" or \"out\"); got %s.",
                tok.ToString()));
      }
      ChannelDirection direction = tok.GetKeyword() == Keyword::kIn
                                       ? ChannelDirection::kIn
                                       : ChannelDirection::kOut;

      Span span(start_pos, GetPos());
      return module_->Make<ChannelTypeAnnotation>(span, direction, payload,
                                                  dims);
    }

    Pos limit_pos = tok.span().limit();

    std::vector<Expr*> dims;
    XLS_ASSIGN_OR_RETURN(bool peek_is_obrack, PeekTokenIs(TokenKind::kOBrack));
    if (peek_is_obrack) {
      XLS_ASSIGN_OR_RETURN(dims, ParseDims(bindings, &limit_pos));
    }
    return MakeBuiltinTypeAnnotation(Span(start_pos, limit_pos), tok, dims);
  }

  if (tok.kind() == TokenKind::kOParen) {  // Tuple of types.
    auto parse_type_annotation = [this, &bindings] {
      return ParseTypeAnnotation(bindings);
    };
    XLS_ASSIGN_OR_RETURN(std::vector<TypeAnnotation*> types,
                         ParseCommaSeq<TypeAnnotation*>(parse_type_annotation,
                                                        TokenKind::kCParen));
    VLOG(5) << "ParseTypeAnnotation; got " << types.size() << " tuple members";

    Span span(tok.span().start(), GetPos());
    TypeAnnotation* type =
        module_->Make<TupleTypeAnnotation>(span, std::move(types));
    VLOG(5) << "ParseTypeAnnotation; result type: " << type->ToString();

    // Enable array of tuple type annotation.
    XLS_ASSIGN_OR_RETURN(bool peek_is_obrack, PeekTokenIs(TokenKind::kOBrack));
    if (peek_is_obrack) {
      XLS_ASSIGN_OR_RETURN(std::vector<Expr*> dims, ParseDims(bindings));
      for (Expr* dim : dims) {
        type = module_->Make<ArrayTypeAnnotation>(span, type, dim);
      }
    }
    return type;
  }

  // If the leader is not builtin and not a tuple, it's some form of type
  // reference.
  XLS_ASSIGN_OR_RETURN(TypeRefOrAnnotation type_ref,
                       ParseTypeRef(bindings, tok));
  return ParseTypeRefParametricsAndDims(bindings, tok.span(), type_ref);
}

absl::StatusOr<TypeAnnotation*> Parser::ParseTypeRefParametricsAndDims(
    Bindings& bindings, const Span& span, TypeRefOrAnnotation type_ref) {
  VLOG(5) << "ParseTypeRefParametricsAndDims @ " << GetPos();
  std::vector<ExprOrType> parametrics;
  XLS_ASSIGN_OR_RETURN(bool peek_is_oangle, PeekTokenIs(TokenKind::kOAngle));
  if (peek_is_oangle) {
    XLS_ASSIGN_OR_RETURN(parametrics, ParseParametrics(bindings));
  }

  std::vector<Expr*> dims;
  XLS_ASSIGN_OR_RETURN(bool peek_is_obrack, PeekTokenIs(TokenKind::kOBrack));
  if (peek_is_obrack) {  // Array type annotation.
    XLS_ASSIGN_OR_RETURN(dims, ParseDims(bindings));
  }

  return MakeTypeRefTypeAnnotation(Span(span.start(), GetPos()), type_ref, dims,
                                   std::move(parametrics));
}

absl::StatusOr<NameRef*> Parser::ParseNameRef(Bindings& bindings,
                                              const Token* tok) {
  std::optional<Token> popped;
  if (tok == nullptr) {
    XLS_ASSIGN_OR_RETURN(
        popped, PopTokenOrError(TokenKind::kIdentifier, /*start=*/nullptr,
                                "Expected name reference identiifer"));
    tok = &popped.value();
  }

  XLS_RET_CHECK(tok->IsKindIn({TokenKind::kIdentifier, TokenKind::kKeyword}));

  if (tok->GetValue() == "_") {
    return ParseErrorStatus(
        tok->span(), "Wildcard pattern `_` cannot be used as a reference");
  }

  XLS_ASSIGN_OR_RETURN(
      BoundNode bn,
      bindings.ResolveNodeOrError(*tok->GetValue(), tok->span(), file_table()));
  AnyNameDef name_def = BoundNodeToAnyNameDef(bn);
  return module_->Make<NameRef>(tok->span(), *tok->GetValue(), name_def);
}

absl::StatusOr<ColonRef*> Parser::ParseColonRef(Bindings& bindings,
                                                ColonRef::Subject subject,
                                                const Span& subject_span) {
  Pos start = subject_span.start();
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDoubleColon));
  while (true) {
    XLS_ASSIGN_OR_RETURN(
        Token value_tok,
        PopTokenOrError(TokenKind::kIdentifier, /*start=*/nullptr,
                        "Expected colon-reference identifier"));
    Span span(start, GetPos());
    subject = module_->Make<ColonRef>(span, subject, *value_tok.GetValue());
    start = GetPos();
    XLS_ASSIGN_OR_RETURN(bool dropped_colon,
                         TryDropToken(TokenKind::kDoubleColon));
    if (dropped_colon) {
      continue;
    }
    return std::get<ColonRef*>(subject);
  }
}

absl::StatusOr<Expr*> Parser::ParseCastOrEnumRefOrStructInstanceOrToken(
    Bindings& bindings) {
  VLOG(5) << "ParseCastOrEnumRefOrStructInstanceOrToken @ " << GetPos()
          << " peek: " << PeekToken().value()->ToString();

  Token tok = PopTokenOrDie();
  XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  if (peek_is_double_colon) {
    XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &tok));
    // TODO: google/xls#1030 - This assumes it is calling an imported function,
    // (e.g., `lib::function();`) not instantiating an imported struct (e.g.,
    // `lib::my_struct{};`). Instantiating both locally-defined and imported
    // structs really should be unified in this method.
    return ParseColonRef(bindings, subject, subject->span());
  }

  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type,
                       ParseTypeAnnotation(bindings, tok));

  // After parsing the type, check again for `peek_is_double_colon` to catch
  // accessing impl members of parametric structs (e.g.,
  // `MyStruct<u32:5>::SOME_CONSTANT`).
  XLS_ASSIGN_OR_RETURN(peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  auto* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(type);
  if (type_ref != nullptr && peek_is_double_colon) {
    return ParseColonRef(bindings, type_ref, type->span());
  }
  XLS_ASSIGN_OR_RETURN(bool peek_is_obrace, PeekTokenIs(TokenKind::kOBrace));
  if (peek_is_obrace) {
    return ParseStructInstance(bindings, type);
  }
  XLS_ASSIGN_OR_RETURN(bool peek_is_oparen, PeekTokenIs(TokenKind::kOParen));
  if (peek_is_oparen) {
    // Looks like a built-in function call with name overlapping with a type;
    // probably `token()`.
    return ParseNameRef(bindings, &tok);
  }
  return ParseCast(bindings, type);
}

absl::StatusOr<Expr*> Parser::ParseStructInstance(Bindings& bindings,
                                                  TypeAnnotation* type) {
  VLOG(5) << "ParseStructInstance @ " << GetPos();

  // Note: any explicit parametrics will be codified in the type we parse ahead
  // of the struct members.
  if (type == nullptr) {
    XLS_ASSIGN_OR_RETURN(type, ParseTypeAnnotation(bindings));
  }

  const Pos start_pos = GetPos();

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace, /*start=*/nullptr,
                                       "Opening brace for struct instance."));

  using StructInstanceMember = std::pair<std::string, Expr*>;
  auto parse_struct_member =
      [this, &bindings]() -> absl::StatusOr<StructInstanceMember> {
    XLS_ASSIGN_OR_RETURN(
        Token tok, PopTokenOrError(TokenKind::kIdentifier, /*start=*/nullptr,
                                   "Expected struct instance's member name"));
    XLS_ASSIGN_OR_RETURN(bool dropped_colon, TryDropToken(TokenKind::kColon));
    if (dropped_colon) {
      XLS_ASSIGN_OR_RETURN(Expr * e, ParseExpression(bindings));
      return std::make_pair(*tok.GetValue(), e);
    }

    XLS_ASSIGN_OR_RETURN(NameRef * name_ref, ParseNameRef(bindings, &tok));
    return std::make_pair(*tok.GetValue(), name_ref);
  };

  std::vector<StructInstanceMember> members;
  bool must_end = false;

  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_cbrace, TryDropToken(TokenKind::kCBrace));
    if (dropped_cbrace) {
      break;
    }
    if (must_end) {
      XLS_RETURN_IF_ERROR(
          DropTokenOrError(TokenKind::kCBrace, /*start=*/nullptr,
                           "Closing brace for struct instance."));
      break;
    }
    XLS_ASSIGN_OR_RETURN(bool dropped_double_dot,
                         TryDropToken(TokenKind::kDoubleDot));
    if (dropped_double_dot) {
      XLS_ASSIGN_OR_RETURN(Expr * splatted, ParseExpression(bindings));
      XLS_RETURN_IF_ERROR(DropTokenOrError(
          TokenKind::kCBrace, /*start=*/nullptr,
          "Closing brace after struct instance \"splat\" (..) expression."));
      Span span(start_pos, GetPos());
      return module_->Make<SplatStructInstance>(span, type, std::move(members),
                                                splatted);
    }

    XLS_ASSIGN_OR_RETURN(StructInstanceMember member, parse_struct_member());
    members.push_back(std::move(member));
    XLS_ASSIGN_OR_RETURN(bool dropped_comma, TryDropToken(TokenKind::kComma));
    must_end = !dropped_comma;
  }
  Span span(start_pos, GetPos());
  return module_->Make<StructInstance>(span, type, std::move(members));
}

absl::StatusOr<std::variant<NameRef*, ColonRef*>> Parser::ParseNameOrColonRef(
    Bindings& bindings, std::string_view context) {
  VLOG(5) << "ParseNameOrColonRef  @ " << GetPos() << " context: " << context;

  XLS_ASSIGN_OR_RETURN(Token tok, PopSelfOrIdentifier(context));
  XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  if (peek_is_double_colon) {
    XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &tok));
    return ParseColonRef(bindings, subject, subject->span());
  }
  return ParseNameRef(bindings, &tok);
}

absl::StatusOr<Token> Parser::PopSelfOrIdentifier(std::string_view context) {
  XLS_ASSIGN_OR_RETURN(bool is_self, PeekTokenIs(Keyword::kSelf));
  if (is_self) {
    return PopTokenOrError(TokenKind::kKeyword,
                           /*start=*/nullptr, context);
  }
  if (parse_fn_stubs_) {
    // Allow "token" as an identifier, for parsing builtin stubs.
    XLS_ASSIGN_OR_RETURN(bool is_token, PeekTokenIs(Keyword::kToken));
    if (is_token) {
      return PopTokenOrError(TokenKind::kKeyword, /*start=*/nullptr, context);
    }
  }
  return PopTokenOrError(TokenKind::kIdentifier,
                         /*start=*/nullptr, context);
}

absl::StatusOr<NameDef*> Parser::ParseNameDefNoBind() {
  XLS_ASSIGN_OR_RETURN(Token tok,
                       PopSelfOrIdentifier("Expected name (definition)"));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, TokenToNameDef(tok));
  return name_def;
}

absl::StatusOr<NameDef*> Parser::ParseNameDef(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDefNoBind());
  bindings.Add(name_def->identifier(), name_def);
  return name_def;
}

absl::StatusOr<NameDefTree*> Parser::ParseNameDefTree(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token start, PopTokenOrError(TokenKind::kOParen));

  auto parse_name_def_or_tree = [&bindings,
                                 this]() -> absl::StatusOr<NameDefTree*> {
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->kind() == TokenKind::kOParen) {
      XLS_ASSIGN_OR_RETURN(ExpressionDepthGuard expr_depth,
                           BumpExpressionDepth());
      return ParseNameDefTree(bindings);
    }
    XLS_ASSIGN_OR_RETURN(auto name_def, ParseNameDefOrWildcard(bindings));
    auto tree_leaf = WidenVariantTo<NameDefTree::Leaf>(name_def);
    return module_->Make<NameDefTree>(GetSpan(name_def), tree_leaf);
  };

  XLS_ASSIGN_OR_RETURN(
      std::vector<NameDefTree*> branches,
      ParseCommaSeq<NameDefTree*>(parse_name_def_or_tree, TokenKind::kCParen));
  NameDefTree* ndt = module_->Make<NameDefTree>(
      Span(start.span().start(), GetPos()), std::move(branches));

  // Check that the name definitions are unique -- can't bind the same name
  // multiple times in one destructuring assignment.
  std::vector<NameDef*> name_defs = ndt->GetNameDefs();
  absl::flat_hash_map<std::string_view, NameDef*> seen;
  for (NameDef* name_def : name_defs) {
    if (!seen.insert({name_def->identifier(), name_def}).second) {
      return ParseErrorStatus(
          name_def->span(),
          absl::StrFormat(
              "Name '%s' is defined twice in this pattern; previously @ %s",
              name_def->identifier(),
              seen[name_def->identifier()]->span().ToString(file_table())));
    }
  }
  return ndt;
}

absl::StatusOr<Array*> Parser::ParseArray(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token start_tok, PopTokenOrError(TokenKind::kOBrack));

  struct EllipsisSentinel {
    Span span;
  };

  using ExprOrEllipsis = std::variant<Expr*, EllipsisSentinel>;
  auto parse_ellipsis_or_expression =
      [this, &bindings]() -> absl::StatusOr<ExprOrEllipsis> {
    XLS_ASSIGN_OR_RETURN(bool peek_is_ellipsis,
                         PeekTokenIs(TokenKind::kEllipsis));
    if (peek_is_ellipsis) {
      Token tok = PopTokenOrDie();
      return EllipsisSentinel{tok.span()};
    }
    return ParseExpression(bindings);
  };
  auto get_span = [](const ExprOrEllipsis& e) {
    if (std::holds_alternative<Expr*>(e)) {
      return std::get<Expr*>(e)->span();
    }
    return std::get<EllipsisSentinel>(e).span;
  };

  Pos cbrack_pos;
  XLS_ASSIGN_OR_RETURN(
      std::vector<ExprOrEllipsis> members,
      ParseCommaSeq<ExprOrEllipsis>(parse_ellipsis_or_expression,
                                    {TokenKind::kCBrack}, &cbrack_pos));
  std::vector<Expr*> exprs;
  bool has_trailing_ellipsis = false;
  for (int64_t i = 0; i < members.size(); ++i) {
    const ExprOrEllipsis& member = members[i];
    if (std::holds_alternative<EllipsisSentinel>(member)) {
      if (i + 1 == members.size()) {
        has_trailing_ellipsis = true;
        members.pop_back();
      } else {
        return ParseErrorStatus(get_span(member),
                                "Ellipsis may only be in trailing position.");
      }
    } else {
      exprs.push_back(std::get<Expr*>(member));
    }
  }

  Span span(start_tok.span().start(), cbrack_pos);
  return module_->Make<Array>(span, std::move(exprs), has_trailing_ellipsis);
}

absl::StatusOr<ExprOrType> Parser::MaybeParseCast(Bindings& bindings,
                                                  TypeAnnotation* type) {
  XLS_ASSIGN_OR_RETURN(const bool peek_is_colon,
                       PeekTokenIs(TokenKind::kColon));
  if (peek_is_colon) {
    return ParseCast(bindings, type);
  }
  return type;
}

absl::StatusOr<Expr*> Parser::ParseCast(Bindings& bindings,
                                        TypeAnnotation* type) {
  VLOG(5) << "ParseCast @ " << GetPos()
          << " type: " << (type == nullptr ? "<null>" : type->ToString());

  if (type == nullptr) {
    absl::StatusOr<TypeAnnotation*> type_status = ParseTypeAnnotation(bindings);
    if (type_status.status().ok()) {
      type = type_status.value();
    } else {
      PositionalErrorData data =
          GetPositionalErrorData(type_status.status(), std::nullopt,
                                 file_table())
              .value();
      return ParseErrorStatus(
          data.span,
          absl::StrFormat("Expected a type as part of a cast expression: %s",
                          data.message));
    }
  }

  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                       "Expect colon after type annotation in cast"));
  XLS_ASSIGN_OR_RETURN(Expr * term, ParseTerm(bindings, kNoRestrictions));
  if (IsOneOf<Number, Array>(term)) {
    if (auto* n = dynamic_cast<Number*>(term)) {
      n->SetTypeAnnotation(type);
      // We just added a type annotation - a new child node - to `n`.
      n->SetParentage();
    } else {
      auto* a = dynamic_cast<Array*>(term);
      a->set_type_annotation(type);
      a->SetParentage();
    }
    return term;
  }

  // This handles the case where we're trying to make a tuple constant that has
  // an explicitly annotated type; e.g.
  //
  // ```dslx
  // const MY_TUPLE = (u32, u64):(u32:32, u64:64);
  // ```
  if (auto* tuple = dynamic_cast<XlsTuple*>(term);
      tuple != nullptr && std::all_of(tuple->members().begin(),
                                      tuple->members().end(), IsConstant)) {
    return term;
  }

  return ParseErrorStatus(
      type->span(),
      "Old-style cast only permitted for constant arrays/tuples "
      "and literal numbers.");
}

absl::StatusOr<Expr*> Parser::ParseBinopChain(
    const std::function<absl::StatusOr<Expr*>()>& sub_production,
    std::variant<absl::Span<TokenKind const>, absl::Span<Keyword const>>
        target_tokens) {
  XLS_ASSIGN_OR_RETURN(Expr * lhs, sub_production());
  while (true) {
    bool peek_in_targets;
    if (std::holds_alternative<absl::Span<TokenKind const>>(target_tokens)) {
      XLS_ASSIGN_OR_RETURN(
          peek_in_targets,
          PeekTokenIn(std::get<absl::Span<TokenKind const>>(target_tokens)));
    } else {
      XLS_ASSIGN_OR_RETURN(
          peek_in_targets,
          PeekKeywordIn(std::get<absl::Span<Keyword const>>(target_tokens)));
    }
    if (peek_in_targets) {
      Token op = PopTokenOrDie();
      XLS_ASSIGN_OR_RETURN(Expr * rhs, sub_production());
      XLS_ASSIGN_OR_RETURN(BinopKind kind,
                           BinopKindFromString(TokenKindToString(op.kind())));
      Span span(lhs->span().start(), rhs->span().limit());
      lhs = module_->Make<Binop>(span, kind, lhs, rhs, op.span());
    } else {
      break;
    }
  }
  return lhs;
}

absl::StatusOr<Expr*> Parser::ParseLogicalAndExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  VLOG(5) << "ParseLogicalAndExpression @ " << GetPos();
  std::initializer_list<TokenKind> kinds = {TokenKind::kDoubleAmpersand};
  return ParseBinopChain(
      [this, &bindings, restrictions] {
        return ParseComparisonExpression(bindings, restrictions);
      },
      kinds);
}

absl::StatusOr<Expr*> Parser::ParseLogicalOrExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  VLOG(5) << "ParseLogicalOrExpression @ " << GetPos();
  static const std::initializer_list<TokenKind> kinds = {TokenKind::kDoubleBar};
  return ParseBinopChain(
      [this, &bindings, restrictions] {
        return ParseLogicalAndExpression(bindings, restrictions);
      },
      kinds);
}

absl::StatusOr<Expr*> Parser::ParseStrongArithmeticExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  auto sub_production = [&]() -> absl::StatusOr<Expr*> {
    XLS_ASSIGN_OR_RETURN(bool peek_is_if, PeekTokenIs(Keyword::kIf));
    XLS_ASSIGN_OR_RETURN(
        Expr * lhs, peek_is_if ? ParseConditionalNode(bindings, restrictions)
                               : ParseTerm(bindings, restrictions));
    return TryParseCastAsAndRhs(lhs, bindings, restrictions);
  };
  return ParseBinopChain(sub_production, kStrongArithmeticKinds);
}

absl::StatusOr<Expr*> Parser::ParseWeakArithmeticExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  auto sub_production = [&] {
    return ParseStrongArithmeticExpression(bindings, restrictions);
  };
  return ParseBinopChain(sub_production, kWeakArithmeticKinds);
}

absl::StatusOr<Expr*> Parser::ParseBitwiseExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  auto sub_production = [&] {
    return ParseWeakArithmeticExpression(bindings, restrictions);
  };
  return ParseBinopChain(sub_production, kBitwiseKinds);
}

absl::StatusOr<Expr*> Parser::ParseAndExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  static const std::initializer_list<TokenKind> amp = {TokenKind::kAmpersand};
  auto sub_production = [&] {
    return ParseBitwiseExpression(bindings, restrictions);
  };
  return ParseBinopChain(sub_production, amp);
}

absl::StatusOr<Expr*> Parser::ParseXorExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  static const std::initializer_list<TokenKind> hat = {TokenKind::kHat};
  return ParseBinopChain(
      [this, &bindings, restrictions] {
        return ParseAndExpression(bindings, restrictions);
      },
      hat);
}

absl::StatusOr<Expr*> Parser::ParseOrExpression(Bindings& bindings,
                                                ExprRestrictions restrictions) {
  std::initializer_list<TokenKind> bar = {TokenKind::kBar};
  return ParseBinopChain(
      [this, &bindings, restrictions] {
        return ParseXorExpression(bindings, restrictions);
      },
      bar);
}

absl::StatusOr<Expr*> Parser::ParseComparisonExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  VLOG(5) << "ParseComparisonExpression @ " << GetPos() << " peek: `"
          << PeekToken().value()->ToString()
          << "` restrictions: " << ExprRestrictionsToString(restrictions);
  XLS_ASSIGN_OR_RETURN(Expr * lhs, ParseOrExpression(bindings, restrictions));
  while (true) {
    VLOG(5) << "ParseComparisonExpression; lhs: `" << lhs->ToString()
            << "` peek: `" << PeekToken().value()->ToString() << "`";
    XLS_ASSIGN_OR_RETURN(bool peek_in_targets, PeekTokenIn(kComparisonKinds));
    if (!peek_in_targets) {
      VLOG(5) << "Peek is not in comparison kinds.";
      break;
    }

    Token op = PopTokenOrDie();
    XLS_ASSIGN_OR_RETURN(BinopKind kind,
                         BinopKindFromString(TokenKindToString(op.kind())));
    XLS_ASSIGN_OR_RETURN(Expr * rhs, ParseOrExpression(bindings, restrictions));
    if (!lhs->in_parens() && IsComparisonBinopKind(lhs) &&
        GetBinopComparisonKinds().contains(kind)) {
      return ParseErrorStatus(op.span(),
                              "comparison operators cannot be chained");
    }
    Span span(lhs->span().start(), rhs->span().limit());
    lhs = module_->Make<Binop>(span, kind, lhs, rhs, op.span());
  }
  VLOG(5) << "ParseComparisonExpression; result: `" << lhs->ToString() << "`";
  return lhs;
}

absl::StatusOr<NameDefTree*> Parser::ParsePattern(Bindings& bindings,
                                                  bool within_tuple_pattern) {
  XLS_ASSIGN_OR_RETURN(ExpressionDepthGuard depth_guard, BumpExpressionDepth());

  XLS_ASSIGN_OR_RETURN(std::optional<Token> oparen,
                       TryPopToken(TokenKind::kOParen));
  if (oparen.has_value()) {
    return ParseTuplePattern(oparen->span().start(), bindings);
  }

  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (peek->kind() == TokenKind::kIdentifier) {
    XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
    if (*tok.GetValue() == "_") {
      return module_->Make<NameDefTree>(
          tok.span(), module_->Make<WildcardPattern>(tok.span()));
    }
    // TODO: https://github.com/google/xls/issues/1459 - Handle rest-of-tuple.
    XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                         PeekTokenIs(TokenKind::kDoubleColon));
    if (peek_is_double_colon) {  // Mod or enum ref.
      XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &tok));
      XLS_ASSIGN_OR_RETURN(ColonRef * colon_ref,
                           ParseColonRef(bindings, subject, subject->span()));
      Span span(tok.span().start(), colon_ref->span().limit());
      return module_->Make<NameDefTree>(span, colon_ref);
    }

    std::string identifier = tok.GetValue().value();
    if (std::optional<BoundNode> resolved = bindings.ResolveNode(identifier);
        resolved.has_value()) {
      AnyNameDef any_name_def =
          bindings.ResolveNameOrNullopt(identifier).value();
      NameRef* ref =
          module_->Make<NameRef>(tok.span(), identifier, any_name_def);
      return module_->Make<NameDefTree>(tok.span(), ref);
    }

    // If the name is not bound, this pattern is creating a binding.
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, TokenToNameDef(tok));
    bindings.Add(name_def->identifier(), name_def);
    Span span(tok.span().start(), GetPos());
    auto* result = module_->Make<NameDefTree>(span, name_def);
    name_def->set_definer(result);
    return result;
  }

  if (peek->kind() == TokenKind::kDoubleDot) {
    if (!within_tuple_pattern) {
      return ParseErrorStatus(
          peek->span(),
          "`..` patterns are not allowed outside of a tuple pattern");
    }
    XLS_ASSIGN_OR_RETURN(Token rest, PopTokenOrError(TokenKind::kDoubleDot));
    return module_->Make<NameDefTree>(rest.span(),
                                      module_->Make<RestOfTuple>(rest.span()));
  }

  if (peek->IsKindIn({TokenKind::kNumber, TokenKind::kCharacter, Keyword::kTrue,
                      Keyword::kFalse}) ||
      peek->IsTypeKeyword()) {
    XLS_ASSIGN_OR_RETURN(Number * number, ParseNumber(bindings));
    XLS_ASSIGN_OR_RETURN(bool peek_is_double_dot,
                         PeekTokenIs(TokenKind::kDoubleDot));
    XLS_ASSIGN_OR_RETURN(bool peek_is_double_dot_equals,
                         PeekTokenIs(TokenKind::kDoubleDotEquals));
    if (peek_is_double_dot || peek_is_double_dot_equals) {
      XLS_RETURN_IF_ERROR(DropToken());
      XLS_ASSIGN_OR_RETURN(Number * limit, ParseNumber(bindings));
      auto* range = module_->Make<Range>(
          Span(number->span().start(), limit->span().limit()), number,
          peek_is_double_dot_equals, limit, /*in_parens=*/false,
          /*pattern_semantics=*/true);
      return module_->Make<NameDefTree>(range->span(), range);
    }
    return module_->Make<NameDefTree>(number->span(), number);
  }

  return ParseErrorStatus(
      peek->span(),
      absl::StrFormat("Expected pattern; got %s", peek->ToErrorString()));
}

absl::StatusOr<Match*> Parser::ParseMatch(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token match, PopKeywordOrError(Keyword::kMatch));
  XLS_ASSIGN_OR_RETURN(Expr * matched, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));

  std::vector<MatchArm*> arms;
  bool must_end = false;
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_cbrace, TryDropToken(TokenKind::kCBrace));
    if (dropped_cbrace) {
      break;
    }
    if (must_end) {
      XLS_RETURN_IF_ERROR(
          DropTokenOrError(TokenKind::kCBrace, /*start=*/nullptr,
                           "Expected '}' because no ',' was seen to indicate "
                           "an additional match case."));
      break;
    }
    Bindings arm_bindings(&bindings);
    XLS_ASSIGN_OR_RETURN(
        NameDefTree * first_pattern,
        ParsePattern(arm_bindings, /*within_tuple_pattern=*/false));
    std::vector<NameDefTree*> patterns = {first_pattern};
    while (true) {
      XLS_ASSIGN_OR_RETURN(bool dropped_bar, TryDropToken(TokenKind::kBar));
      if (!dropped_bar) {
        break;
      }
      if (arm_bindings.HasLocalBindings()) {
        // TODO(leary): 2020-09-12 Loosen this restriction? They just have to
        // bind the same exact set of names.
        std::vector<std::string> locals =
            MapKeysSorted(arm_bindings.local_bindings());
        return ParseErrorStatus(
            first_pattern->span(),
            absl::StrFormat("Cannot have multiple patterns that bind names; "
                            "previously bound: %s",
                            absl::StrJoin(locals, ", ")));
      }
      XLS_ASSIGN_OR_RETURN(
          NameDefTree * pattern,
          ParsePattern(arm_bindings, /*within_tuple_pattern=*/false));
      patterns.push_back(pattern);
    }
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kFatArrow));
    XLS_ASSIGN_OR_RETURN(Expr * rhs, ParseExpression(arm_bindings));

    // The span of the match arm is from the start of the pattern to the end of
    // the RHS expression.
    Span span(patterns[0]->span().start(), rhs->span().limit());

    arms.push_back(module_->Make<MatchArm>(span, std::move(patterns), rhs));
    XLS_ASSIGN_OR_RETURN(bool dropped_comma, TryDropToken(TokenKind::kComma));
    must_end = !dropped_comma;
  }
  Span span(match.span().start(), GetPos());
  return module_->Make<Match>(span, matched, std::move(arms));
}

absl::StatusOr<UseTreeEntry*> Parser::ParseUseTreeEntry(Bindings& bindings) {
  // Get the identifier for this level of the tree.
  XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
  std::string identifier = *tok.GetValue();

  // The next level of the tree can be peer subtrees, or a single child.
  // If the next level is present it is indicated by a `::`.
  XLS_ASSIGN_OR_RETURN(bool saw_double_colon,
                       TryDropToken(TokenKind::kDoubleColon));
  if (!saw_double_colon) {
    // Must be a leaf, as we see no subsequent `::` to indicate there is a
    // subsequent level.
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, TokenToNameDef(tok));
    auto* use_tree_entry = module_->Make<UseTreeEntry>(name_def, tok.span());
    bindings.Add(name_def->identifier(), use_tree_entry);
    name_def->set_definer(use_tree_entry);
    return use_tree_entry;
  }

  // If we've gotten here we know there's a next level, we're just looking to
  // see if it's peer subtrees or a single subtree.
  XLS_ASSIGN_OR_RETURN(bool saw_obrace, TryDropToken(TokenKind::kOBrace));
  if (saw_obrace) {
    // Multiple peer subtrees -- present as children (subtrees) to make this
    // level.
    std::vector<UseTreeEntry*> children;
    while (true) {
      XLS_ASSIGN_OR_RETURN(UseTreeEntry * child, ParseUseTreeEntry(bindings));
      children.push_back(child);
      XLS_ASSIGN_OR_RETURN(bool saw_cbrace, TryDropToken(TokenKind::kCBrace));
      if (saw_cbrace) {
        break;
      }
      XLS_RETURN_IF_ERROR(DropTokenOrError(
          TokenKind::kComma, /*start=*/nullptr,
          "Expect a ',' to separate multiple entries in a `use` statement"));
    }
    return module_->Make<UseTreeEntry>(
        UseInteriorEntry{identifier, std::move(children)}, tok.span());
  }

  // Must be a single child in the subsequent level -- we recurse here to look
  // for additional levels after it.
  XLS_ASSIGN_OR_RETURN(UseTreeEntry * child, ParseUseTreeEntry(bindings));
  std::vector<UseTreeEntry*> children;
  children.reserve(1);
  children.push_back(std::move(child));
  return module_->Make<UseTreeEntry>(
      UseInteriorEntry{identifier, std::move(children)}, tok.span());
}

absl::StatusOr<Use*> Parser::ParseUse(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token kw, PopKeywordOrError(Keyword::kUse));

  XLS_ASSIGN_OR_RETURN(bool peek_is_obrace, PeekTokenIs(TokenKind::kOBrace));
  if (peek_is_obrace) {
    // `use { ... }`
    return ParseErrorStatus(
        Span(kw.span().start(), GetPos()),
        "Cannot `use` multiple modules in one statement; e.g. `use {foo, bar}` "
        "is not allowed -- please break into multiple statements");
  }

  XLS_ASSIGN_OR_RETURN(UseTreeEntry * root, ParseUseTreeEntry(bindings));
  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kSemi, /*start=*/&kw,
                       "Expect a ';' at end of `use` statement"));
  Span span(kw.span().start(), GetPos());
  return module_->Make<Use>(span, *root);
}

absl::StatusOr<Import*> Parser::ParseImport(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token kw, PopKeywordOrError(Keyword::kImport));
  XLS_ASSIGN_OR_RETURN(Token first_tok,
                       PopTokenOrError(TokenKind::kIdentifier));
  std::vector<Token> toks = {first_tok};
  std::vector<std::string> subject = {*first_tok.GetValue()};
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_dot, TryDropToken(TokenKind::kDot));
    if (!dropped_dot) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
    toks.push_back(tok);
    subject.push_back(*tok.GetValue());
  }

  XLS_ASSIGN_OR_RETURN(bool dropped_as, TryDropKeyword(Keyword::kAs));
  NameDef* name_def;
  std::optional<std::string> alias;
  if (dropped_as) {
    XLS_ASSIGN_OR_RETURN(name_def, ParseNameDef(bindings));
    alias = name_def->identifier();
  } else {
    XLS_ASSIGN_OR_RETURN(name_def, TokenToNameDef(toks.back()));
  }

  if (std::optional<ModuleMember*> existing_member =
          module_->FindMemberWithName(name_def->identifier())) {
    const std::optional<Span> maybe_span =
        ToAstNode(*existing_member.value())->GetSpan();
    std::string span_str = maybe_span.has_value()
                               ? " at " + maybe_span->ToString(file_table())
                               : "";
    return ParseErrorStatus(
        name_def->span(),
        absl::StrFormat("Import of `%s` is shadowing an existing definition%s",
                        name_def->identifier(), span_str));
  }

  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kSemi, /*start=*/&kw,
                       /*context=*/"Expect an ';' at end of import statement"));
  Span span(kw.span().start(), GetPos());
  auto* import = module_->Make<Import>(span, subject, *name_def, alias);
  name_def->set_definer(import);
  bindings.Add(name_def->identifier(), import);
  return import;
}

absl::StatusOr<Function*> Parser::ParseFunctionInternal(
    const Pos& start_pos, bool is_public, bool is_test_utility,
    Bindings& outer_bindings, TypeAnnotation* struct_ref) {
  XLS_ASSIGN_OR_RETURN(Token fn_tok, PopKeywordOrError(Keyword::kFn));

  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(outer_bindings));

  Bindings bindings(&outer_bindings);
  bindings.NoteFunctionScoped();
  bindings.Add(name_def->identifier(), name_def);

  XLS_ASSIGN_OR_RETURN(bool dropped_oangle, TryDropToken(TokenKind::kOAngle));
  std::vector<ParametricBinding*> parametric_bindings;
  if (dropped_oangle) {  // Parametric.
    XLS_ASSIGN_OR_RETURN(parametric_bindings,
                         ParseParametricBindings(bindings));
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Param*> params,
                       ParseParams(bindings, struct_ref));

  XLS_ASSIGN_OR_RETURN(bool dropped_arrow, TryDropToken(TokenKind::kArrow));
  TypeAnnotation* return_type = nullptr;
  if (dropped_arrow) {
    XLS_ASSIGN_OR_RETURN(bool is_self, PeekTokenIs(Keyword::kSelfType));
    if (is_self) {
      CHECK(struct_ref != nullptr);
      XLS_ASSIGN_OR_RETURN(Token self_tok,
                           PopKeywordOrError(Keyword::kSelfType));
      return_type = module_->Make<SelfTypeAnnotation>(
          self_tok.span(), /*explicit_type=*/true, struct_ref);
    } else {
      XLS_ASSIGN_OR_RETURN(return_type, ParseTypeAnnotation(bindings));
    }
  }

  StatementBlock* body = nullptr;
  if (parse_fn_stubs_) {
    // Must have a semicolon
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemi));
    body = module_->Make<StatementBlock>(Span(start_pos, GetPos()),
                                         std::vector<Statement*>{},
                                         /*trailing_semi=*/true);
  } else {
    XLS_ASSIGN_OR_RETURN(body, ParseBlockExpression(bindings));
  }
  Function* f = module_->Make<Function>(Span(start_pos, GetPos()), name_def,
                                        std::move(parametric_bindings), params,
                                        return_type, body, FunctionTag::kNormal,
                                        is_public, is_test_utility);
  name_def->set_definer(f);
  return f;
}

absl::StatusOr<QuickCheckTestCases> Parser::ParseQuickCheckConfig() {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
  if (tok.GetValue() == "exhaustive") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    return QuickCheckTestCases::Exhaustive();
  }
  if (tok.GetValue() == "test_count") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
    XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kNumber));
    XLS_ASSIGN_OR_RETURN(int64_t count, tok.GetValueAsInt64());
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    return QuickCheckTestCases::Counted(count);
  }
  return ParseErrorStatus(tok.span(),
                          "Expected 'exhaustive' or 'test_count' in "
                          "quickcheck attribute");
}

absl::StatusOr<QuickCheck*> Parser::ParseQuickCheck(
    absl::flat_hash_map<std::string, Function*>* name_to_fn, Bindings& bindings,
    const Pos& hash_pos) {
  std::optional<QuickCheckTestCases> test_cases;
  XLS_ASSIGN_OR_RETURN(bool peek_is_oparen, PeekTokenIs(TokenKind::kOParen));
  if (peek_is_oparen) {  // Config is specified.
    XLS_ASSIGN_OR_RETURN(test_cases, ParseQuickCheckConfig());
  } else {
    test_cases = QuickCheckTestCases::Counted(std::nullopt);
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
  XLS_ASSIGN_OR_RETURN(
      Function * fn,
      ParseFunction(GetPos(), /*is_public=*/false,
                    /*is_test_utility=*/false, bindings, name_to_fn));
  const Span quickcheck_span(hash_pos, fn->span().limit());
  return module_->Make<QuickCheck>(quickcheck_span, fn, test_cases.value());
}

absl::StatusOr<XlsTuple*> Parser::ParseTupleRemainder(const Pos& start_pos,
                                                      Expr* first,
                                                      Bindings& bindings) {
  VLOG(5) << "ParseTupleRemainder @ " << GetPos();
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kComma));
  auto parse_expression = [this, &bindings]() -> absl::StatusOr<Expr*> {
    return ParseExpression(bindings);
  };
  bool saw_trailing_comma = false;
  XLS_ASSIGN_OR_RETURN(
      std::vector<Expr*> es,
      ParseCommaSeq<Expr*>(parse_expression, TokenKind::kCParen, nullptr,
                           &saw_trailing_comma));
  if (es.empty()) {
    // If ParseCommaSeq ends up not parsing anything, we need to remember that
    // there was a comma before ParseCommaSeq was called.
    saw_trailing_comma = true;
  }
  es.insert(es.begin(), first);
  Span span(start_pos, GetPos());
  return module_->Make<XlsTuple>(span, std::move(es), saw_trailing_comma);
}

absl::StatusOr<Expr*> Parser::ParseTermLhsParenthesized(
    Bindings& outer_bindings, const Pos& start_pos) {
  Expr* lhs = nullptr;
  Token oparen = PopTokenOrDie();
  XLS_ASSIGN_OR_RETURN(bool next_is_cparen, PeekTokenIs(TokenKind::kCParen));
  if (next_is_cparen) {  // Empty tuple.
    XLS_ASSIGN_OR_RETURN(Token tok, PopToken());
    Span span(start_pos, GetPos());
    lhs = module_->Make<XlsTuple>(span, std::vector<Expr*>{},
                                  /*has_trailing_comma=*/false);
  } else {
    XLS_ASSIGN_OR_RETURN(lhs, ParseExpression(outer_bindings));
    XLS_ASSIGN_OR_RETURN(bool peek_is_comma, PeekTokenIs(TokenKind::kComma));
    if (peek_is_comma) {  // Singleton tuple.
      XLS_ASSIGN_OR_RETURN(
          lhs, ParseTupleRemainder(oparen.span().start(), lhs, outer_bindings));
    } else {
      XLS_RETURN_IF_ERROR(
          DropTokenOrError(TokenKind::kCParen, /*start=*/&oparen,
                           "Expected ')' at end of parenthesized expression"));
      // Make a note the expression was
      // wrapped in parens. This helps us disambiguate when people chain
      // comparison operators on purpose vs accidentally e.g.
      //    x == y == z    // error
      //    (x == y) == z  // ok
      lhs->set_in_parens(true);
    }
  }
  CHECK(lhs != nullptr);
  return lhs;
}

absl::StatusOr<Expr*> Parser::ParseTermLhs(Bindings& outer_bindings,
                                           ExprRestrictions restrictions) {
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());

  XLS_ASSIGN_OR_RETURN(ExpressionDepthGuard expr_depth, BumpExpressionDepth());

  const Pos start_pos = peek->span().start();
  VLOG(5) << "ParseTerm @ " << start_pos << " peek: `" << peek->ToString()
          << "` restrictions: " << ExprRestrictionsToString(restrictions);

  bool peek_is_kw_in = peek->IsKeyword(Keyword::kIn);
  bool peek_is_kw_out = peek->IsKeyword(Keyword::kOut);
  bool peek_is_kw_self = peek->IsKeyword(Keyword::kSelf);

  Expr* lhs = nullptr;
  if (peek->IsKindIn({TokenKind::kNumber, TokenKind::kCharacter}) ||
      peek->IsKeywordIn({Keyword::kTrue, Keyword::kFalse})) {
    XLS_ASSIGN_OR_RETURN(lhs, ParseNumber(outer_bindings));
  } else if (peek->kind() == TokenKind::kString) {
    Token tok = PopTokenOrDie();
    // TODO(https://github.com/google/xls/issues/1105): 2021-05-20 Add
    // zero-length string support akin to zero-length array support.
    if (tok.GetValue()->empty()) {
      return ParseErrorStatus(tok.span(),
                              "Zero-length strings are not supported.");
    }
    return module_->Make<String>(tok.span(), *tok.GetValue());
  } else if (peek->IsKindIn({TokenKind::kBang, TokenKind::kMinus})) {
    Token tok = PopTokenOrDie();
    XLS_ASSIGN_OR_RETURN(Expr * arg, ParseTerm(outer_bindings, restrictions));
    UnopKind unop_kind;
    switch (tok.kind()) {
        // clang-format off
      case TokenKind::kBang: unop_kind = UnopKind::kInvert; break;
      case TokenKind::kMinus: unop_kind = UnopKind::kNegate; break;
      // clang-format on
      default:
        LOG(FATAL) << "Inconsistent unary operation token kind.";
    }
    Span span(tok.span().start(), arg->span().limit());
    lhs = module_->Make<Unop>(span, unop_kind, arg, tok.span());
  } else if (peek->IsTypeKeyword() ||
             (peek->kind() == TokenKind::kIdentifier &&
              outer_bindings.ResolveNodeIsTypeDefinition(*peek->GetValue()))) {
    VLOG(5) << "ParseTerm, kind is identifier AND it it a known type";
    // The "local" struct case goes into here because it recognized the
    // my_struct as a type
    XLS_ASSIGN_OR_RETURN(
        lhs, ParseCastOrEnumRefOrStructInstanceOrToken(outer_bindings));
  } else if (peek->kind() == TokenKind::kIdentifier || peek_is_kw_in ||
             peek_is_kw_out || peek_is_kw_self) {
    VLOG(5) << "ParseTerm, kind is identifier but not a known type";
    XLS_ASSIGN_OR_RETURN(auto name_or_colon_ref,
                         ParseNameOrColonRef(outer_bindings));
    if (std::holds_alternative<ColonRef*>(name_or_colon_ref) &&
        !IsExprRestrictionEnabled(restrictions,
                                  ExprRestriction::kNoStructLiteral)) {
      ColonRef* colon_ref = std::get<ColonRef*>(name_or_colon_ref);
      TypeAnnotation* type = nullptr;
      TypeRef* type_ref = nullptr;

      Transaction parse_oangle_txn(this, &outer_bindings);

      // If we do an assign-or-return (status error propagation) we need to
      // nullify the transaction.
      absl::Cleanup rollback_by_default([&parse_oangle_txn] {
        if (!parse_oangle_txn.completed()) {
          parse_oangle_txn.Rollback();
        }
      });

      XLS_ASSIGN_OR_RETURN(bool found_oangle, PeekTokenIs(TokenKind::kOAngle));
      if (found_oangle) {
        VLOG(5) << "ParseTerm, kind is ColonRef then oAngle; trying parametric";
        type_ref = module_->Make<TypeRef>(colon_ref->span(), colon_ref);

        // We're looking either at an imported parametric struct, an imported
        // parametric enum comparison, or an imported parametric function call.
        // This block only deals with parametric struct instantiation; the other
        // possibilities are handled in different places, so we will rollback
        // if it's not actually an imported parametric struct instantiation.
        auto type_and_dims = ParseTypeRefParametricsAndDims(
            outer_bindings, colon_ref->span(), type_ref);
        if (!type_and_dims.ok()) {
          // Probably a comparison, so the oAngle wasn't part of a parametric
          // reference, and we won't be doing a struct instantiation.
          VLOG(5)
              << "ParseTerm, kind is ColonRef then oAngle, but not parametric";
          type = nullptr;
          type_ref = nullptr;
        } else {
          type = *type_and_dims;
          VLOG(5) << "ParseTerm, kind is ColonRef then oAngle, and parametric "
                     "type *and dims* "
                  << type->ToString();
        }
      }

      XLS_ASSIGN_OR_RETURN(bool found_obrace, PeekTokenIs(TokenKind::kOBrace));
      if (found_obrace) {
        VLOG(5) << "ParseTerm, kind is ColonRef then oBrace";
        parse_oangle_txn.Commit();

        if (!found_oangle) {
          type_ref = module_->Make<TypeRef>(colon_ref->span(), colon_ref);
          // Just a regular imported struct instantiation. Make the type
          // annotation now.
          XLS_ASSIGN_OR_RETURN(
              type, MakeTypeRefTypeAnnotation(colon_ref->span(), type_ref,
                                              /*dims=*/{}, /*parametrics=*/{}));
        }
        return ParseStructInstance(outer_bindings, type);
      }

      XLS_ASSIGN_OR_RETURN(bool found_colon, PeekTokenIs(TokenKind::kColon));
      if (type != nullptr && found_colon) {
        VLOG(5) << "ParseTerm, kind is ColonRef then oColon";
        // Probably a literal array. Commit the parsing of the LHS so far, and
        // continue processing as if it's a cast.
        parse_oangle_txn.Commit();
        XLS_ASSIGN_OR_RETURN(lhs, ParseCast(outer_bindings, type));
      } else {
        VLOG(5)
            << "ParseTerm, kind is ColonRef, but not a struct instantiation ";

        // Roll back the transaction, so we can continue as if we
        // never processed the kOAngle.
        parse_oangle_txn.Rollback();
      }
    }
    if (lhs == nullptr) {
      lhs = ToExprNode(name_or_colon_ref);
    }
  } else if (peek->kind() == TokenKind::kOParen) {
    XLS_ASSIGN_OR_RETURN(
        lhs, ParseParentheticalOrCastLhs(outer_bindings, start_pos));
  } else if (peek->IsKeyword(Keyword::kMatch)) {  // Match expression.
    XLS_ASSIGN_OR_RETURN(lhs, ParseMatch(outer_bindings));
  } else if (peek->kind() == TokenKind::kOBrack) {  // Array expression.
    XLS_ASSIGN_OR_RETURN(lhs, ParseArray(outer_bindings));
  } else if (peek->IsKeyword(Keyword::kIf)) {  // Conditional expression.
    XLS_ASSIGN_OR_RETURN(lhs,
                         ParseRangeExpression(outer_bindings, kNoRestrictions));
  } else {
    return ParseErrorStatus(
        peek->span(),
        absl::StrFormat("Expected start of an expression; got: %s",
                        peek->ToErrorString()));
  }
  CHECK(lhs != nullptr);
  return lhs;
}

absl::StatusOr<Expr*> Parser::ParseParentheticalOrCastLhs(
    Bindings& outer_bindings, const Pos& start_pos) {
  // This is potentially a cast to tuple type, e.g. `(u8, u16):(u8:5, u16:1)`.
  // Try to parse as a parenthesized expression first and fall back to tuple
  // cast.  In the case that both fail, return the status for the parenthetical
  // since it's likely the more common usage.
  //
  // While it isn't explicitly enforced, we don't expect a case where both would
  // succeed.
  Transaction parse_txn(this, &outer_bindings);
  absl::StatusOr<Expr*> parenth =
      ParseTermLhsParenthesized(*parse_txn.bindings(), start_pos);
  if (parenth.ok()) {
    parse_txn.Commit();
    return parenth;
  }
  parse_txn.Rollback();
  absl::StatusOr<Expr*> cast =
      ParseCastOrEnumRefOrStructInstanceOrToken(outer_bindings);
  if (cast.ok()) {
    return cast;
  }
  return parenth;
}

absl::StatusOr<Expr*> Parser::ParseTermRhs(Expr* lhs, Bindings& outer_bindings,
                                           ExprRestrictions restrictions) {
  const Pos new_pos = GetPos();
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  switch (peek->kind()) {
    case TokenKind::kColon: {  // Possibly a Number of ColonRef type.
      // The only valid construct here would be declaring a number via
      // ColonRef-colon-Number, e.g., "module::type:7".
      if (dynamic_cast<ColonRef*>(lhs) == nullptr) {
        goto done;
      }

      Span span(new_pos, GetPos());
      auto* type_ref =
          module_->Make<TypeRef>(span, ToTypeDefinition(lhs).value());
      auto* type_annot = module_->Make<TypeRefTypeAnnotation>(
          span, type_ref, /*parametrics=*/std::vector<ExprOrType>{});
      XLS_ASSIGN_OR_RETURN(lhs, ParseCast(outer_bindings, type_annot));
      break;
    }
    case TokenKind::kOParen: {  // Invocation.
      DropTokenOrDie();
      XLS_ASSIGN_OR_RETURN(std::vector<Expr*> args,
                           ParseCommaSeq<Expr*>(
                               [&outer_bindings, this] {
                                 return ParseExpression(outer_bindings);
                               },
                               TokenKind::kCParen));
      XLS_ASSIGN_OR_RETURN(
          lhs, BuildMacroOrInvocation(Span(new_pos, GetPos()), outer_bindings,
                                      lhs, std::move(args)));
      break;
    }
    case TokenKind::kDot: {  // Attribute or tuple index access.
      DropTokenOrDie();
      XLS_ASSIGN_OR_RETURN(Token tok, PopToken());
      const Span span(new_pos, GetPos());
      if (tok.kind() == TokenKind::kIdentifier) {
        lhs = module_->Make<Attr>(span, lhs, *tok.GetValue());
      } else if (tok.kind() == TokenKind::kNumber) {
        XLS_ASSIGN_OR_RETURN(Number * number, TokenToNumber(tok));
        lhs = module_->Make<TupleIndex>(span, lhs, number);
      } else {
        return ParseErrorStatus(
            span,
            absl::StrFormat(
                "Unknown dot ('.') reference: expected number or identifier; "
                "got %s",
                tok.ToString()));
      }
      break;
    }
    case TokenKind::kOBrack: {  // Indexing.
      DropTokenOrDie();

      XLS_ASSIGN_OR_RETURN(bool dropped_colon, TryDropToken(TokenKind::kColon));
      if (dropped_colon) {  // Slice-from-beginning.
        XLS_ASSIGN_OR_RETURN(lhs, ParseBitSlice(new_pos, lhs, outer_bindings,
                                                /*start=*/nullptr));
        break;
      }

      // Index may be followed by a `:` for a slice, or a `+:` for a
      // type-sized slice.
      XLS_ASSIGN_OR_RETURN(Expr * index, ParseExpression(outer_bindings));
      XLS_ASSIGN_OR_RETURN(peek, PeekToken());
      switch (peek->kind()) {
        case TokenKind::kPlusColon: {  // Explicit width slice.
          DropTokenOrDie();
          Expr* start = index;
          XLS_ASSIGN_OR_RETURN(TypeAnnotation * width,
                               ParseTypeAnnotation(outer_bindings));
          Span span(new_pos, GetPos());
          auto* width_slice = module_->Make<WidthSlice>(span, start, width);
          lhs = module_->Make<Index>(span, lhs, width_slice);
          XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
          break;
        }
        case TokenKind::kColon: {  // Slice to end.
          DropTokenOrDie();
          XLS_ASSIGN_OR_RETURN(
              lhs, ParseBitSlice(new_pos, lhs, outer_bindings, index));
          break;
        }
        default: {
          XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));

          // It's either an Index if the LHS is a NameRef or
          // ColonRef-to-ConstantDef, or an Array if lhs is a
          // ColonRef-to-type.
          const Span span(new_pos, GetPos());
          XLS_ASSIGN_OR_RETURN(peek, PeekToken());
          // Array type before literal, e.g. Foo[2]:[...]
          //                  this colon ----------^
          if (peek->kind() == TokenKind::kColon) {
            DropTokenOrDie();
            absl::StatusOr<TypeDefinition> type_definition =
                ToTypeDefinition(lhs);
            if (!type_definition.ok()) {
              const Span error_span(lhs->span().start(), index->span().limit());
              return ParseErrorStatus(
                  error_span,
                  absl::StrFormat(
                      "Type before ':' for presumed array literal was not a "
                      "type definition; got `%s` (kind: %v)",
                      lhs->ToString(), lhs->kind()));
            }
            // TODO(rspringer): We can't currently support parameterized
            // ColonRef-to-types with this function structure.
            auto* type_ref = module_->Make<TypeRef>(span, *type_definition);
            auto* type_ref_type = module_->Make<TypeRefTypeAnnotation>(
                span, type_ref, /*parametrics=*/std::vector<ExprOrType>());
            auto* array_type =
                module_->Make<ArrayTypeAnnotation>(span, type_ref_type, index);
            XLS_ASSIGN_OR_RETURN(Array * array, ParseArray(outer_bindings));
            array->set_type_annotation(array_type);
            lhs = array;
            lhs->SetParentage();
          } else {
            lhs = module_->Make<Index>(span, lhs, index);
          }
        }
      }
      break;
    }
    case TokenKind::kOAngle: {
      // Comparison op, parametric function invocation, or parametric function
      // reference (e.g. for a `map` call).
      Transaction sub_txn(this, &outer_bindings);
      absl::Cleanup sub_cleanup = [&sub_txn]() { sub_txn.Rollback(); };

      absl::StatusOr<std::vector<ExprOrType>> parametrics =
          ParseParametrics(*sub_txn.bindings());
      if (!parametrics.ok()) {
        VLOG(5) << "ParseParametrics gave error: " << parametrics.status();
        goto done;
      }

      XLS_ASSIGN_OR_RETURN(bool has_open_paren,
                           PeekTokenIs(TokenKind::kOParen));
      if (!has_open_paren) {
        sub_txn.CommitAndCancelCleanup(&sub_cleanup);
        return module_->Make<FunctionRef>(lhs->span(), lhs, *parametrics);
      }
      CHECK_OK(PopToken().status());
      Bindings* b = sub_txn.bindings();
      XLS_ASSIGN_OR_RETURN(
          std::vector<Expr*> args,
          ParseCommaSeq<Expr*>([b, this] { return ParseExpression(*b); },
                               TokenKind::kCParen));
      XLS_ASSIGN_OR_RETURN(
          lhs, BuildMacroOrInvocation(Span(new_pos, GetPos()),
                                      *sub_txn.bindings(), lhs, std::move(args),
                                      std::move(*parametrics)));
      sub_txn.CommitAndCancelCleanup(&sub_cleanup);
      break;
    }
    case TokenKind::kArrow:
      // If we're a term followed by an arrow...then we followed the wrong
      // production, as arrows are only allowed after fn decls. Rewind.
      // Should this be something else, like a "wrong production" error?
      return ParseErrorStatus(
          lhs->span(), "Parenthesized expression cannot precede an arrow.");
    default:
      goto done;
  }
done:
  return lhs;
}

ExpressionDepthGuard::~ExpressionDepthGuard() {
  if (parser_ != nullptr) {
    parser_->approximate_expression_depth_--;
    CHECK_GE(parser_->approximate_expression_depth_, 0);
    parser_ = nullptr;
  }
}

absl::StatusOr<ExpressionDepthGuard> Parser::BumpExpressionDepth() {
  if (++approximate_expression_depth_ >= kApproximateExpressionDepthLimit) {
    return ParseErrorStatus(Span(GetPos(), GetPos()),
                            "Extremely deep nesting detected -- please break "
                            "into multiple statements");
  }
  return ExpressionDepthGuard(this);
}

absl::StatusOr<Expr*> Parser::ParseTerm(Bindings& outer_bindings,
                                        ExprRestrictions restrictions) {
  XLS_ASSIGN_OR_RETURN(ExpressionDepthGuard expr_depth, BumpExpressionDepth());

  XLS_ASSIGN_OR_RETURN(Expr * lhs, ParseTermLhs(outer_bindings, restrictions));

  while (true) {
    XLS_ASSIGN_OR_RETURN(Expr * new_lhs,
                         ParseTermRhs(lhs, outer_bindings, restrictions));
    if (new_lhs == lhs) {
      return lhs;
    }
    lhs = new_lhs;
  }
}

absl::StatusOr<Expr*> Parser::BuildFormatMacroWithVerbosityArgument(
    const Span& span, std::string_view name, std::vector<Expr*> args,
    const std::vector<ExprOrType>& parametrics) {
  if (args.size() < 2) {
    return ParseErrorStatus(
        span,
        absl::Substitute("$0 macro must have at least 2 arguments.", name));
  }
  // Extract the verbosity argument and pass on the remainder of the arguments.
  Expr* verbosity = args[0];
  args.erase(args.begin());
  return BuildFormatMacro(span, name, std::move(args), parametrics, verbosity);
}

absl::StatusOr<Expr*> Parser::BuildFormatMacro(
    const Span& span, std::string_view name, std::vector<Expr*> args,
    const std::vector<ExprOrType>& parametrics,
    std::optional<Expr*> verbosity) {
  if (!parametrics.empty()) {
    return ParseErrorStatus(
        span, absl::Substitute("$0 macro does not expect parametric arguments.",
                               name));
  }

  if (args.empty()) {
    return ParseErrorStatus(
        span,
        absl::Substitute("$0 macro must have at least 1 argument.", name));
  }

  Expr* format_arg = args[0];
  String* format_string = dynamic_cast<String*>(format_arg);
  if (!format_string) {
    return ParseErrorStatus(
        span, absl::Substitute("Expected a literal format string; got `$0`",
                               format_arg->ToString()));
  }

  const std::string& format_text = format_string->text();
  absl::StatusOr<std::vector<FormatStep>> format_result =
      ParseFormatString(format_text);
  if (!format_result.ok()) {
    return ParseErrorStatus(format_string->span(),
                            format_result.status().message());
  }

  // Remove the format string argument before building the macro call.
  args.erase(args.begin());
  return module_->Make<FormatMacro>(span, std::string(name), *format_result,
                                    args, verbosity);
}

absl::StatusOr<Expr*> Parser::BuildMacroOrInvocation(
    const Span& span, Bindings& bindings, Expr* callee, std::vector<Expr*> args,
    std::vector<ExprOrType> parametrics) {
  if (auto* name_ref = dynamic_cast<NameRef*>(callee)) {
    if (auto* builtin = TryGet<BuiltinNameDef*>(name_ref->name_def())) {
      std::string name = builtin->identifier();
      if (name == "trace_fmt!") {
        return BuildFormatMacro(span, name, args, parametrics);
      }

      if (name == "vtrace_fmt!") {
        return BuildFormatMacroWithVerbosityArgument(span, name, args,
                                                     parametrics);
      }

      if (name == "zero!") {
        if (parametrics.size() != 1) {
          return ParseErrorStatus(
              span,
              absl::StrFormat("%s macro takes a single parametric argument "
                              "(the type to create a zero value for, e.g. "
                              "`zero!<T>()`; got %d parametric arguments",
                              name, parametrics.size()));
        }
        if (!args.empty()) {
          return ParseErrorStatus(
              span,
              absl::StrFormat("%s macro does not take any arguments", name));
        }

        return module_->Make<ZeroMacro>(span, parametrics.at(0));
      }

      if (name == "all_ones!") {
        if (parametrics.size() != 1) {
          return ParseErrorStatus(
              span,
              absl::StrFormat("%s macro takes a single parametric argument "
                              "(the type to create an all-ones value for, e.g. "
                              "`all_ones!<T>()`; got %d parametric arguments",
                              name, parametrics.size()));
        }
        if (!args.empty()) {
          return ParseErrorStatus(
              span,
              absl::StrFormat("%s macro does not take any arguments", name));
        }

        return module_->Make<AllOnesMacro>(span, parametrics.at(0));
      }

      if (name == "fail!") {
        if (args.size() != 2) {
          return ParseErrorStatus(
              span, "fail!() requires two arguments: a label and a condition.");
        }

        String* label = dynamic_cast<String*>(args.at(0));
        if (label == nullptr) {
          return ParseErrorStatus(
              span, "The first argument to fail!() must be a label string.");
        }

        if (!NameUniquer::IsValidIdentifier(label->text())) {
          return ParseErrorStatus(label->span(),
                                  "The label parameter to fail!() must be a "
                                  "valid Verilog identifier.");
        }
        XLS_RETURN_IF_ERROR(
            bindings.AddFailLabel(label->text(), label->span(), file_table()));
      }
    }
  }
  return module_->Make<Invocation>(span, callee, std::move(args),
                                   std::move(parametrics));
}

absl::StatusOr<Spawn*> Parser::ParseSpawn(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token spawn, PopKeywordOrError(Keyword::kSpawn));
  XLS_ASSIGN_OR_RETURN(auto name_or_colon_ref, ParseNameOrColonRef(bindings));

  std::vector<ExprOrType> parametrics;
  XLS_ASSIGN_OR_RETURN(bool peek_is_oangle, PeekTokenIs(TokenKind::kOAngle));
  if (peek_is_oangle) {
    XLS_ASSIGN_OR_RETURN(parametrics, ParseParametrics(bindings));
  }

  Expr* spawnee;
  Expr* config_ref;
  Expr* next_ref;
  Expr* init_ref;
  if (std::holds_alternative<NameRef*>(name_or_colon_ref)) {
    NameRef* name_ref = std::get<NameRef*>(name_or_colon_ref);
    spawnee = name_ref;
    // We avoid name collisions b/w existing functions and Proc config/next fns
    // by using a "." as the separator, which is invalid for function
    // specifications.
    std::string config_name = absl::StrCat(name_ref->identifier(), ".config");
    std::string next_name = absl::StrCat(name_ref->identifier(), ".next");
    std::string init_name = absl::StrCat(name_ref->identifier(), ".init");
    XLS_ASSIGN_OR_RETURN(AnyNameDef config_def,
                         bindings.ResolveNameOrError(
                             config_name, spawnee->span(), file_table()));
    if (!std::holds_alternative<const NameDef*>(config_def)) {
      return absl::InternalError("Proc config should be named \".config\"");
    }
    config_ref =
        module_->Make<NameRef>(name_ref->span(), config_name, config_def);

    XLS_ASSIGN_OR_RETURN(
        AnyNameDef next_def,
        bindings.ResolveNameOrError(next_name, spawnee->span(), file_table()));
    if (!std::holds_alternative<const NameDef*>(next_def)) {
      return absl::InternalError("Proc next should be named \".next\"");
    }
    next_ref = module_->Make<NameRef>(name_ref->span(), next_name, next_def);

    XLS_ASSIGN_OR_RETURN(
        AnyNameDef init_def,
        bindings.ResolveNameOrError(init_name, spawnee->span(), file_table()));
    if (!std::holds_alternative<const NameDef*>(init_def)) {
      return absl::InternalError("Proc init should be named \".init\"");
    }
    init_ref = module_->Make<NameRef>(name_ref->span(), init_name, init_def);
  } else {
    ColonRef* colon_ref = std::get<ColonRef*>(name_or_colon_ref);
    spawnee = colon_ref;

    // Problem: If naively assigned, the colon_ref subject would end up being a
    // child of both `config_ref` and `next_ref`, which is forbidden. To avoid
    // this, just clone the subject (references are easily clonable).
    config_ref =
        module_->Make<ColonRef>(colon_ref->span(), colon_ref->subject(),
                                absl::StrCat(colon_ref->attr(), ".config"));

    ColonRef::Subject clone_subject =
        CloneSubject(module_.get(), colon_ref->subject());
    next_ref =
        module_->Make<ColonRef>(colon_ref->span(), clone_subject,
                                absl::StrCat(colon_ref->attr(), ".next"));

    clone_subject = CloneSubject(module_.get(), colon_ref->subject());
    init_ref =
        module_->Make<ColonRef>(colon_ref->span(), clone_subject,
                                absl::StrCat(colon_ref->attr(), ".init"));
  }

  auto parse_args = [this, &bindings] { return ParseExpression(bindings); };
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  Pos config_start = GetPos();
  XLS_ASSIGN_OR_RETURN(std::vector<Expr*> config_args,
                       ParseCommaSeq<Expr*>(parse_args, TokenKind::kCParen));
  Pos config_limit = GetPos();

  // We don't have the Proc definition here - it could be a ColonRef, for
  // example - so we can't know for sure if the spawnee has a state param or
  // not. Thus, we unconditionally create an invocation of the `init` function
  // and set that as the state.
  Pos next_start = GetPos();
  XLS_ASSIGN_OR_RETURN(auto init_parametrics, CloneParametrics(parametrics));
  auto* init_invocation = module_->Make<Invocation>(
      init_ref->span(), init_ref, std::vector<Expr*>(),
      std::move(init_parametrics));
  Pos next_limit = GetPos();

  XLS_ASSIGN_OR_RETURN(auto config_parametrics, CloneParametrics(parametrics));
  auto* config_invoc =
      module_->Make<Invocation>(Span(config_start, config_limit), config_ref,
                                config_args, std::move(config_parametrics));

  XLS_ASSIGN_OR_RETURN(auto next_parametrics, CloneParametrics(parametrics));
  auto* next_invoc = module_->Make<Invocation>(
      Span(next_start, next_limit), next_ref,
      std::vector<Expr*>({init_invocation}), std::move(next_parametrics));

  return module_->Make<Spawn>(Span(spawn.span().start(), next_limit), spawnee,
                              config_invoc, next_invoc, std::move(parametrics));
}

absl::StatusOr<Index*> Parser::ParseBitSlice(const Pos& start_pos, Expr* lhs,
                                             Bindings& bindings, Expr* start) {
  Expr* limit_expr = nullptr;
  XLS_ASSIGN_OR_RETURN(bool peek_is_cbrack, PeekTokenIs(TokenKind::kCBrack));
  if (!peek_is_cbrack) {
    XLS_ASSIGN_OR_RETURN(limit_expr, ParseExpression(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack, /*start=*/nullptr,
                                       "at end of bit slice"));

  // Type deduction will verify that start & limit are constexpr.
  Slice* index =
      module_->Make<Slice>(Span(start_pos, GetPos()), start, limit_expr);
  return module_->Make<Index>(Span(start_pos, GetPos()), lhs, index);
}

absl::StatusOr<Expr*> Parser::TryParseCastAsAndRhs(
    Expr* lhs, Bindings& bindings, ExprRestrictions restrictions) {
  VLOG(5) << "TryParseCastAsAndRhs @ " << GetPos()
          << " restrictions: " << ExprRestrictionsToString(restrictions);
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_as, TryDropKeyword(Keyword::kAs));
    if (!dropped_as) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
    Span span(lhs->span().start(), type->span().limit());
    lhs = module_->Make<Cast>(span, lhs, type);
  }
  return lhs;
}

absl::StatusOr<ConstantDef*> Parser::ParseConstantDef(const Pos& start_pos,
                                                      bool is_public,
                                                      Bindings& bindings) {
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kConst));
  Bindings new_bindings(/*parent=*/&bindings);
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(new_bindings));
  if (bindings.HasName(name_def->identifier())) {
    Span span = BoundNodeGetSpan(
        bindings.ResolveNode(name_def->identifier()).value(), file_table());
    return ParseErrorStatus(
        name_def->span(),
        absl::StrFormat(
            "Constant definition is shadowing an existing definition from %s",
            span.ToString(file_table())));
  }

  XLS_ASSIGN_OR_RETURN(bool dropped_colon, TryDropToken(TokenKind::kColon));
  TypeAnnotation* annotated_type = nullptr;
  if (dropped_colon) {
    XLS_ASSIGN_OR_RETURN(annotated_type, ParseTypeAnnotation(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
  XLS_ASSIGN_OR_RETURN(Expr * expr, ParseExpression(bindings));
  Pos limit_pos;
  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kSemi, /*start=*/nullptr,
                       absl::StrFormat("after constant definition of `%s`",
                                       name_def->identifier()),
                       &limit_pos));
  Span span(start_pos, limit_pos);
  auto* result = module_->Make<ConstantDef>(span, name_def, annotated_type,
                                            expr, is_public);
  name_def->set_definer(result);
  bindings.Add(name_def->identifier(), result);
  return result;
}

absl::StatusOr<Function*> Parser::ParseProcConfig(
    Bindings& outer_bindings,
    std::vector<ParametricBinding*> parametric_bindings,
    const std::vector<ProcMember*>& proc_members, std::string_view proc_name,
    bool is_public, bool is_test_utility) {
  Bindings bindings(&outer_bindings);
  XLS_ASSIGN_OR_RETURN(Token config_tok,
                       PopTokenOrError(TokenKind::kIdentifier));
  if (!config_tok.IsIdentifier("config")) {
    return ParseErrorStatus(
        config_tok.span(),
        absl::StrCat("Expected 'config', got ", config_tok.ToString()));
  }

  XLS_ASSIGN_OR_RETURN(std::vector<Param*> config_params,
                       ParseParams(bindings));

  XLS_ASSIGN_OR_RETURN(StatementBlock * block, ParseBlockExpression(bindings));

  if (block->empty() || block->trailing_semi()) {
    // Implicitly nil tuple as a result.
  } else {
    Expr* final_expr = std::get<Expr*>(block->statements().back()->wrapped());

    if (dynamic_cast<XlsTuple*>(final_expr) == nullptr) {
      Span final_stmt_span =
          ToAstNode(block->statements().back()->wrapped())->GetSpan().value();
      return ParseErrorStatus(
          final_stmt_span,
          "The final expression in a Proc config must be a tuple with one "
          "element for each Proc data member.");
    }

    VLOG(5) << "ParseProcConfig; final expr: `" << final_expr->ToString()
            << "`";
  }

  NameDef* name_def = module_->Make<NameDef>(
      config_tok.span(), absl::StrCat(proc_name, ".config"), nullptr);

  std::vector<TypeAnnotation*> return_elements;
  return_elements.reserve(proc_members.size());
  for (const ProcMember* member : proc_members) {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * member_type_clone,
        CloneNode(member->type_annotation(), &PreserveTypeDefinitionsReplacer));
    return_elements.push_back(member_type_clone);
  }
  TypeAnnotation* return_type =
      module_->Make<TupleTypeAnnotation>(config_tok.span(), return_elements);
  Function* config = module_->Make<Function>(
      block->span(), name_def, std::move(parametric_bindings),
      std::move(config_params), return_type, block, FunctionTag::kProcConfig,
      is_public, is_test_utility);
  name_def->set_definer(config);

  return config;
}

absl::StatusOr<Function*> Parser::ParseProcNext(
    Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
    std::string_view proc_name, bool is_public, bool is_test_utility) {
  Bindings inner_bindings(&bindings);
  inner_bindings.NoteFunctionScoped();

  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (!peek->IsIdentifier("next")) {
    return ParseErrorStatus(
        peek->span(), absl::StrCat("Expected 'next', got ", peek->ToString()));
  }
  XLS_RETURN_IF_ERROR(DropToken());
  XLS_ASSIGN_OR_RETURN(Token oparen, PopTokenOrError(TokenKind::kOParen));

  auto parse_param = [this, &inner_bindings]() -> absl::StatusOr<Param*> {
    return ParseParam(inner_bindings);
  };

  XLS_ASSIGN_OR_RETURN(std::vector<Param*> next_params,
                       ParseCommaSeq<Param*>(parse_param, TokenKind::kCParen));

  if (next_params.size() != 1) {
    std::string next_params_str =
        absl::StrJoin(next_params, ", ", [](std::string* out, const Param* p) {
          absl::StrAppend(out, p->identifier());
        });
    return ParseErrorStatus(
        Span(GetPos(), GetPos()),
        absl::StrFormat("A Proc next function takes one argument (a recurrent "
                        "state element); got %d parameters: [%s]",
                        next_params.size(), next_params_str));
  }

  Param* state_param = next_params.back();
  if (HasChannelElement(state_param->type_annotation())) {
    return ParseErrorStatus(state_param->span(),
                            "Channels cannot be Proc next params.");
  }

  XLS_ASSIGN_OR_RETURN(TypeAnnotation * return_type,
                       CloneNode(state_param->type_annotation(),
                                 &PreserveTypeDefinitionsReplacer));
  XLS_ASSIGN_OR_RETURN(StatementBlock * body,
                       ParseBlockExpression(inner_bindings));
  Span span(oparen.span().start(), GetPos());
  NameDef* name_def =
      module_->Make<NameDef>(span, absl::StrCat(proc_name, ".next"), nullptr);
  Function* next = module_->Make<Function>(
      span, name_def, std::move(parametric_bindings),
      std::vector<Param*>({state_param}), return_type, body,
      FunctionTag::kProcNext, is_public, is_test_utility);
  name_def->set_definer(next);

  return next;
}

// Implementation note: this is basically ParseFunction(), except with no return
// type.
absl::StatusOr<Function*> Parser::ParseProcInit(
    Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
    std::string_view proc_name, bool is_public, bool is_test_utility) {
  Bindings inner_bindings(&bindings);
  XLS_ASSIGN_OR_RETURN(Token init_identifier, PopToken());
  if (!init_identifier.IsIdentifier("init")) {
    return ParseErrorStatus(init_identifier.span(),
                            absl::StrCat("Expected \"init\", got ",
                                         init_identifier.ToString(), "\"."));
  }

  NameDef* name_def = module_->Make<NameDef>(
      init_identifier.span(), absl::StrCat(proc_name, ".init"), nullptr);

  XLS_ASSIGN_OR_RETURN(StatementBlock * body,
                       ParseBlockExpression(inner_bindings));
  Span span(init_identifier.span().start(), GetPos());
  Function* init = module_->Make<Function>(
      span, name_def, std::move(parametric_bindings), std::vector<Param*>(),
      /*return_type=*/nullptr, body, FunctionTag::kProcInit, is_public,
      is_test_utility);
  name_def->set_definer(init);
  return init;
}

std::vector<StructMemberNode*> ConvertProcMembersToStructMembers(
    Module* module, const std::vector<ProcMember*>& proc_members) {
  std::vector<StructMemberNode*> struct_members;
  struct_members.reserve(proc_members.size());
  for (ProcMember* proc_member : proc_members) {
    // Best estimate of the colon span is between the name and the type.
    Span colon_span(proc_member->name_def()->span().limit(),
                    proc_member->type_annotation()->span().start());
    struct_members.push_back(module->Make<StructMemberNode>(
        proc_member->span(), proc_member->name_def(), colon_span,
        proc_member->type_annotation()));
  }
  return struct_members;
}

template <typename T>
absl::StatusOr<ModuleMember> Parser::ParseProcLike(const Pos& start_pos,
                                                   bool is_public,
                                                   bool is_test_utility,
                                                   Bindings& outer_bindings,
                                                   Keyword keyword) {
  XLS_ASSIGN_OR_RETURN(Token leading_token, PopKeywordOrError(keyword));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(outer_bindings));

  // Bindings for "within the proc" scope.
  //
  // These are filled with e.g. the proc's name, the parametric bindings for
  // this proc.
  Bindings proc_bindings(&outer_bindings);
  proc_bindings.Add(name_def->identifier(), name_def);

  // These are the proc bindings with the addition of proc members -- we want to
  // exclude members for the `config()` function evaluation.
  //
  // Note that anything we add to the `proc_bindings` shows up in the
  // `member_bindings`.
  Bindings member_bindings(&proc_bindings);

  // Parse any parametrics for the proc.
  XLS_ASSIGN_OR_RETURN(bool dropped_oangle, TryDropToken(TokenKind::kOAngle));
  std::vector<ParametricBinding*> parametric_bindings;
  if (dropped_oangle) {  // Parametric.
    XLS_ASSIGN_OR_RETURN(parametric_bindings,
                         ParseParametricBindings(proc_bindings));
  }

  Pos obrace_pos;
  XLS_RETURN_IF_ERROR(DropTokenOrError(
      TokenKind::kOBrace, nullptr, "'{' at start of proc-like", &obrace_pos));

  // Helper, if "f" is already non-null we give back an appropriate error
  // message given the "peek" token that names the function.
  auto check_not_yet_specified =
      [this, name_def](Function* f, const Token* peek) -> absl::Status {
    if (f != nullptr) {
      return ParseErrorStatus(
          peek->span(),
          absl::StrFormat("proc `%s` %s function was already specified @ %s",
                          name_def->identifier(), *peek->GetValue(),
                          peek->span().ToString(file_table())));
    }
    return absl::OkStatus();
  };

  auto make_collision_error =
      absl::bind_front(&MakeModuleTopCollisionError, file_table());

  ProcLikeBody proc_like_body = {
      .config = nullptr,
      .next = nullptr,
      .init = nullptr,
  };
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  std::optional<const Token*> first_semi_separator;
  std::optional<const Token*> first_comma_separator;
  while (peek->kind() != TokenKind::kCBrace) {
    if (peek->IsKeyword(Keyword::kType)) {
      XLS_ASSIGN_OR_RETURN(
          TypeAlias * type_alias,
          ParseTypeAlias(GetPos(), /*is_public=*/false, proc_bindings));
      proc_like_body.stmts.push_back(type_alias);
    } else if (peek->IsKeyword(Keyword::kConst)) {
      XLS_ASSIGN_OR_RETURN(
          ConstantDef * constant,
          ParseConstantDef(GetPos(), /*is_public=*/false, proc_bindings));
      proc_like_body.stmts.push_back(constant);
    } else if (peek->IsIdentifier("config")) {
      XLS_RETURN_IF_ERROR(check_not_yet_specified(proc_like_body.config, peek));

      // We make a more specific/helpful error message when you try to refer to
      // a name that the config function is supposed to define from within the
      // config function.
      auto specialize_config_name_error =
          [proc_like_body](const absl::Status& status) {
            xabsl::StatusBuilder builder(status);
            std::optional<std::string_view> bad_name =
                MaybeExtractParseNameError(status);
            if (bad_name.has_value() &&
                HasMemberNamed(proc_like_body, bad_name.value())) {
              builder << absl::StreamFormat(
                  "\"%s\" is a proc member, but those cannot be referenced "
                  "from within a proc config function.",
                  bad_name.value());
            }
            return builder;
          };

      // Note: the config function does not have access to the proc members,
      // because that's what it is defining. It does, however, have access to
      // type aliases and similar. As a result, we use `proc_bindings` instead
      // of `member_bindings` as the base bindings here.
      Bindings this_bindings(&proc_bindings);
      XLS_ASSIGN_OR_RETURN(
          Function * config,
          ParseProcConfig(this_bindings, parametric_bindings,
                          proc_like_body.members, name_def->identifier(),
                          is_public, is_test_utility),
          _.With(specialize_config_name_error));

      proc_like_body.config = config;

      // TODO(https://github.com/google/xls/issues/1029): 2024-03-25 this is a
      // bit of a kluge -- we add a function identifier that uses a reserved
      // character to the outer (e.g. module-level) bindings to avoid
      // collisions.
      outer_bindings.Add(config->name_def()->identifier(), config->name_def());

      XLS_RETURN_IF_ERROR(module_->AddTop(config, make_collision_error));
      proc_like_body.stmts.push_back(config);
    } else if (peek->IsIdentifier("next")) {
      XLS_RETURN_IF_ERROR(check_not_yet_specified(proc_like_body.next, peek));

      // Note: parsing of the `next()` function does have access to members,
      // unlike `config()`.
      XLS_ASSIGN_OR_RETURN(
          Function * next,
          ParseProcNext(member_bindings, parametric_bindings,
                        name_def->identifier(), is_public, is_test_utility));
      proc_like_body.next = next;
      XLS_RETURN_IF_ERROR(module_->AddTop(next, make_collision_error));

      // TODO(https://github.com/google/xls/issues/1029): 2024-03-25 this is a
      // bit of a kluge -- we add a function identifier that uses a reserved
      // character to the outer (e.g. module-level) bindings to avoid
      // collisions.
      outer_bindings.Add(next->name_def()->identifier(), next->name_def());
      proc_like_body.stmts.push_back(next);
    } else if (peek->IsIdentifier("init")) {
      XLS_RETURN_IF_ERROR(check_not_yet_specified(proc_like_body.init, peek));

      XLS_ASSIGN_OR_RETURN(
          Function * init,
          ParseProcInit(member_bindings, parametric_bindings,
                        name_def->identifier(), is_public, is_test_utility));
      proc_like_body.init = init;
      XLS_RETURN_IF_ERROR(module_->AddTop(init, make_collision_error));

      // TODO(https://github.com/google/xls/issues/1029): 2024-03-25 this is a
      // bit of a kluge -- we add a function identifier that uses a reserved
      // character to the outer (e.g. module-level) bindings to avoid
      // collisions.
      outer_bindings.Add(init->name_def()->identifier(), init->name_def());
      proc_like_body.stmts.push_back(init);
    } else if (peek->kind() == TokenKind::kIdentifier) {
      XLS_ASSIGN_OR_RETURN(Token identifier_tok, PopToken());
      XLS_ASSIGN_OR_RETURN(bool peek_is_colon, PeekTokenIs(TokenKind::kColon));
      // If there's a colon after the identifier, we're parsing a proc member.
      if (peek_is_colon) {
        // Note: to parse a member, we use the memberless bindings (e.g. to
        // capture type aliases and similar) and them collapse the new member
        // binding into the proc-level bindings.
        XLS_ASSIGN_OR_RETURN(ProcMember * member,
                             ParseProcMember(member_bindings, identifier_tok));
        XLS_ASSIGN_OR_RETURN(const Token* separator, PeekToken());
        if (!separator->IsKindIn(
                {TokenKind::kSemi, TokenKind::kComma, TokenKind::kCBrace})) {
          return ParseErrorStatus(
              separator->span(),
              absl::StrCat("Expected a ';' or ',' after proc member, got: ",
                           separator->ToString()));
        }
        if (!first_comma_separator.has_value() &&
            separator->kind() == TokenKind::kComma) {
          first_comma_separator = separator;
        }
        if (!first_semi_separator.has_value() &&
            separator->kind() == TokenKind::kSemi) {
          first_semi_separator = separator;
        }
        if (separator->kind() != TokenKind::kCBrace) {
          CHECK_OK(DropToken());
        }
        proc_like_body.members.push_back(member);
        proc_like_body.stmts.push_back(member);
      } else if (identifier_tok.IsIdentifier(kConstAssertIdentifier)) {
        XLS_ASSIGN_OR_RETURN(ConstAssert * const_assert,
                             ParseConstAssert(proc_bindings, &identifier_tok));
        proc_like_body.stmts.push_back(const_assert);
      } else {
        return ParseErrorStatus(
            peek->span(),
            absl::StrFormat(
                "Expected either a proc member, type alias, or `%s` at proc "
                "scope; got identifier: `%s`",
                kConstAssertIdentifier, *peek->GetValue()));
      }
    } else {
      return ParseErrorStatus(
          peek->span(),
          absl::StrFormat("Unexpected token in proc body: %s; want one of "
                          "'config', 'next', or 'init'",
                          peek->ToErrorString()));
    }

    XLS_ASSIGN_OR_RETURN(peek, PeekToken());
  }

  const bool has_any_functions = proc_like_body.config != nullptr ||
                                 proc_like_body.next != nullptr ||
                                 proc_like_body.init != nullptr;
  std::vector<std::string_view> missing_functions;
  if (proc_like_body.init == nullptr) {
    missing_functions.push_back("`init`");
  }
  if (proc_like_body.config == nullptr) {
    missing_functions.push_back("`config`");
  }
  if (proc_like_body.next == nullptr) {
    missing_functions.push_back("`next`");
  }
  if (has_any_functions && !missing_functions.empty()) {
    return ParseErrorStatus(
        Span(leading_token.span().start(), GetPos()),
        absl::StrFormat("Procs must define `init`, `config` and `next` "
                        "functions; missing: %s.",
                        absl::StrJoin(missing_functions, ", ")));
  }

  XLS_ASSIGN_OR_RETURN(Token cbrace, PopTokenOrError(TokenKind::kCBrace));
  const Span span(start_pos, cbrace.span().limit());
  const Span body_span(obrace_pos, cbrace.span().limit());

  if (!has_any_functions) {
    if (first_semi_separator.has_value()) {
      return ParseErrorStatus(
          (*first_semi_separator)->span(),
          "Impl-style procs must use commas to separate members.");
    }
    // Assume this is an impl-style proc and return a `ProcDef` for it.
    ProcDef* proc_def =
        module_->Make<ProcDef>(span, name_def, std::move(parametric_bindings),
                               ConvertProcMembersToStructMembers(
                                   module_.get(), proc_like_body.members),
                               is_public);
    outer_bindings.Add(name_def->identifier(), proc_def);
    return proc_def;
  }

  if (first_comma_separator.has_value()) {
    return ParseErrorStatus(
        (*first_comma_separator)->span(),
        "Non-impl-style procs must use semicolons to separate members.");
  }

  // Just as with proc member decls, we need the init fn to have its own return
  // type, to avoid parent/child relationship violations.
  XLS_ASSIGN_OR_RETURN(auto* init_return_type,
                       CloneNode(proc_like_body.next->return_type(),
                                 &PreserveTypeDefinitionsReplacer));
  init_return_type->SetParentage();
  proc_like_body.init->set_return_type(
      down_cast<TypeAnnotation*>(init_return_type));
  proc_like_body.init->SetParentage();

  auto* proc_like = module_->Make<T>(span, body_span, name_def,
                                     std::move(parametric_bindings),
                                     proc_like_body, is_public);

  // Now that the proc is defined we can set a bunch of links to point at it.
  proc_like_body.config->set_proc(proc_like);
  proc_like_body.next->set_proc(proc_like);
  proc_like_body.init->set_proc(proc_like);
  name_def->set_definer(proc_like);

  XLS_RETURN_IF_ERROR(VerifyParentage(proc_like));
  return proc_like;
}

absl::StatusOr<ModuleMember> Parser::ParseProc(const Pos& start_pos,
                                               bool is_public,
                                               bool is_test_utility,
                                               Bindings& outer_bindings) {
  return ParseProcLike<Proc>(start_pos, is_public, is_test_utility,
                             outer_bindings, Keyword::kProc);
}

absl::StatusOr<ChannelDecl*> Parser::ParseChannelDecl(
    Bindings& bindings, const std::optional<ChannelConfig>& channel_config) {
  XLS_ASSIGN_OR_RETURN(Token channel, PopKeywordOrError(Keyword::kChan));

  std::optional<std::vector<Expr*>> dims = std::nullopt;
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOAngle));

  // For parametric instantiation we allow a form like:
  //
  //  chan<MyStruct<Y>>
  //
  // Which can require us to interpret the >> as two close-angle tokens instead
  // of a single '>>' token.
  DisableDoubleCAngle();
  absl::Cleanup re_enable_double_cangle = [this] { EnableDoubleCAngle(); };
  XLS_ASSIGN_OR_RETURN(auto* type, ParseTypeAnnotation(bindings));
  Pos limit_pos = type->span().limit();

  XLS_ASSIGN_OR_RETURN(bool dropped_comma, TryDropToken(TokenKind::kComma));
  std::optional<ExprOrType> fifo_depth_parametric;
  if (dropped_comma) {
    XLS_ASSIGN_OR_RETURN(fifo_depth_parametric, ParseParametricArg(bindings));
    limit_pos = ExprOrTypeSpan(fifo_depth_parametric.value()).limit();
  }
  XLS_ASSIGN_OR_RETURN(Token cangle_tok, PopTokenOrError(TokenKind::kCAngle));

  ChannelDeclMetadata channel_metadata = std::monostate{};
  if (channel_config.has_value()) {
    channel_metadata = *channel_config;
  }

  if (fifo_depth_parametric.has_value()) {
    if (module_->attributes().contains(ModuleAttribute::kChannelAttributes)) {
      return ParseErrorStatus(
          ExprOrTypeSpan(*fifo_depth_parametric),
          "Cannot specify fifo depth when new-style channel attributes are "
          "enabled, please use the new syntax.");
    }
    if (channel_config.has_value()) {
      // Should not happen because the previous check for
      // #[feature(channel_attributes)] should prevent this from happening, but
      // let's double check.
      return ParseErrorStatus(
          ExprOrTypeSpan(*fifo_depth_parametric),
          "Cannot specify both fifo depth and channel config.");
    }
    if (std::holds_alternative<Expr*>(*fifo_depth_parametric)) {
      channel_metadata = std::get<Expr*>(*fifo_depth_parametric);
    } else {
      return ParseErrorStatus(
          ExprOrTypeSpan(*fifo_depth_parametric),
          "Expected fifo depth to be expression, got type.");
    }
  }

  XLS_ASSIGN_OR_RETURN(bool peek_is_obrack, PeekTokenIs(TokenKind::kOBrack));
  if (peek_is_obrack) {
    XLS_ASSIGN_OR_RETURN(dims, ParseDims(bindings, &limit_pos));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen, nullptr,
                                       "expect '(' for channel declaration"));
  XLS_ASSIGN_OR_RETURN(Expr * channel_name_expr, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(
      TokenKind::kCParen, nullptr,
      "expect ')' at end of channel declaration (after name)"));

  XLS_RET_CHECK_NE(channel_name_expr, nullptr);
  return module_->Make<ChannelDecl>(Span(channel.span().start(), limit_pos),
                                    type, dims, channel_metadata,
                                    *channel_name_expr);
}

absl::StatusOr<std::vector<Expr*>> Parser::ParseDims(Bindings& bindings,
                                                     Pos* limit_pos) {
  XLS_ASSIGN_OR_RETURN(Token obrack, PopTokenOrError(TokenKind::kOBrack));

  XLS_ASSIGN_OR_RETURN(bool peek_is_cbrack, PeekTokenIs(TokenKind::kCBrack));
  if (peek_is_cbrack) {
    return ParseErrorStatus(
        obrack.span(),
        "Unsized arrays are not supported, a (constant) size is required.");
  }

  XLS_ASSIGN_OR_RETURN(Expr * first_dim,
                       ParseRangeExpression(bindings, kNoRestrictions));

  std::vector<Expr*> dims = {first_dim};
  const char* const kContext = "at end of type dimensions";
  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kCBrack, &obrack, kContext, limit_pos));
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_obrack,
                         TryDropToken(TokenKind::kOBrack, limit_pos));
    if (!dropped_obrack) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(Expr * dim,
                         ParseRangeExpression(bindings, kNoRestrictions));
    dims.push_back(dim);
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack, /*start=*/&obrack,
                                         /*context=*/kContext,
                                         /*limit_pos=*/limit_pos));
  }
  return dims;
}

absl::StatusOr<TypeRef*> Parser::ParseModTypeRef(Bindings& bindings,
                                                 const Token& start_tok) {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDoubleColon));
  XLS_ASSIGN_OR_RETURN(
      BoundNode bn, bindings.ResolveNodeOrError(
                        *start_tok.GetValue(), start_tok.span(), file_table()));
  if (!std::holds_alternative<Import*>(bn) &&
      !std::holds_alternative<UseTreeEntry*>(bn)) {
    return ParseErrorStatus(
        start_tok.span(),
        absl::StrFormat("Expected module for module-reference; got %s",
                        ToAstNode(bn)->ToString()));
  }
  XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &start_tok));
  XLS_ASSIGN_OR_RETURN(Token type_name,
                       PopTokenOrError(TokenKind::kIdentifier, &start_tok,
                                       "module type-reference"));
  const Span span(start_tok.span().start(), type_name.span().limit());
  ColonRef* mod_ref =
      module_->Make<ColonRef>(span, subject, *type_name.GetValue());
  return module_->Make<TypeRef>(span, mod_ref);
}

absl::StatusOr<Let*> Parser::ParseLet(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token start_tok, PopToken());
  bool const_;
  if (start_tok.IsKeyword(Keyword::kLet)) {
    const_ = false;
  } else if (start_tok.IsKeyword(Keyword::kConst)) {
    const_ = true;
  } else {
    return ParseErrorStatus(
        start_tok.span(),
        absl::StrFormat("Expected 'let' or 'const'; got %s @ %s",
                        start_tok.ToErrorString(),
                        start_tok.span().ToString(file_table())));
  }

  XLS_ASSIGN_OR_RETURN(bool peek_is_mut, PeekTokenIs(Keyword::kMut));
  if (peek_is_mut) {
    return ParseErrorStatus(
        start_tok.span(),
        "`mut` and mutable bindings are not supported in DSLX");
  }

  Bindings new_bindings(&bindings);
  NameDef* name_def = nullptr;
  NameDefTree* name_def_tree;
  XLS_ASSIGN_OR_RETURN(bool peek_is_oparen, PeekTokenIs(TokenKind::kOParen));
  if (peek_is_oparen) {  // Destructuring binding.
    XLS_ASSIGN_OR_RETURN(name_def_tree, ParseNameDefTree(new_bindings));
  } else {
    XLS_ASSIGN_OR_RETURN(name_def, ParseNameDef(new_bindings));
    if (name_def->identifier() == "_") {
      name_def_tree = module_->Make<NameDefTree>(
          name_def->span(), module_->Make<WildcardPattern>(name_def->span()));
    } else {
      name_def_tree = module_->Make<NameDefTree>(name_def->span(), name_def);
    }
  }

  if (const_) {
    // Mark this NDT as const. Also disallow destructuring assignment for
    // constants.
    const_ndts_.insert(name_def_tree);
    if (name_def_tree->Flatten().size() != 1) {
      return ParseErrorStatus(
          name_def_tree->span(),
          absl::StrFormat(
              "Constant definitions can not use destructuring assignment: %s",
              name_def_tree->ToString()));
    }
  }

  XLS_ASSIGN_OR_RETURN(bool dropped_colon, TryDropToken(TokenKind::kColon));
  TypeAnnotation* annotated_type = nullptr;
  if (dropped_colon) {
    XLS_ASSIGN_OR_RETURN(annotated_type, ParseTypeAnnotation(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
  XLS_ASSIGN_OR_RETURN(Expr * rhs, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemi));

  Span span(start_tok.span().start(), GetPos());
  Let* let =
      module_->Make<Let>(span, name_def_tree, annotated_type, rhs, const_);
  if (const_ && name_def != nullptr) {
    name_def->set_definer(let);
  } else if (name_def != nullptr) {
    name_def->set_definer(rhs);
  }

  // Now that we're dong parsing the RHS expression using the old bindings,
  // update the old bindings with the bindings made by this let statement.
  bindings.ConsumeChild(&new_bindings);
  return let;
}

absl::StatusOr<For*> Parser::ParseFor(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token for_kw, PopKeywordOrError(Keyword::kFor));

  Bindings for_bindings(&bindings);
  XLS_ASSIGN_OR_RETURN(NameDefTree * names, ParseNameDefTree(for_bindings));
  XLS_ASSIGN_OR_RETURN(bool peek_is_colon, PeekTokenIs(TokenKind::kColon));
  TypeAnnotation* type = nullptr;
  if (peek_is_colon) {
    XLS_RETURN_IF_ERROR(
        DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                         "Expect type annotation on for-loop values."));
    XLS_ASSIGN_OR_RETURN(type, ParseTypeAnnotation(for_bindings));
  }
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kIn));
  XLS_ASSIGN_OR_RETURN(
      Expr * iterable,
      ParseExpression(bindings,
                      MakeRestrictions({ExprRestriction::kNoStructLiteral})));
  XLS_ASSIGN_OR_RETURN(StatementBlock * body,
                       ParseBlockExpression(for_bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(
      TokenKind::kOParen, &for_kw,
      "Need an initial accumulator value to start the for loop."));

  // We must be sure to use the outer bindings when parsing the init
  // expression, since the for loop bindings haven't happened yet (no loop
  // trips have iterated when the init value is evaluated).
  XLS_ASSIGN_OR_RETURN(Expr * init, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
  return module_->Make<For>(Span(for_kw.span().start(), GetPos()), names, type,
                            iterable, body, init);
}

absl::StatusOr<UnrollFor*> Parser::ParseUnrollFor(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token unroll_for,
                       PopKeywordOrError(Keyword::kUnrollFor));

  Bindings for_bindings(&bindings);
  XLS_ASSIGN_OR_RETURN(NameDefTree * names, ParseNameDefTree(for_bindings));
  XLS_ASSIGN_OR_RETURN(bool peek_is_colon, PeekTokenIs(TokenKind::kColon));
  TypeAnnotation* types = nullptr;
  if (peek_is_colon) {
    XLS_RETURN_IF_ERROR(
        DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                         "Expect type annotation on for-loop values."));
    XLS_ASSIGN_OR_RETURN(types, ParseTypeAnnotation(for_bindings));
  }

  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kIn));
  XLS_ASSIGN_OR_RETURN(Expr * iterable, ParseExpression(bindings));
  XLS_ASSIGN_OR_RETURN(StatementBlock * body,
                       ParseBlockExpression(for_bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  XLS_ASSIGN_OR_RETURN(Expr * init, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));

  return module_->Make<UnrollFor>(Span(unroll_for.span().start(), GetPos()),
                                  names, types, iterable, body, init);
}

absl::StatusOr<EnumDef*> Parser::ParseEnumDef(const Pos& start_pos,
                                              bool is_public,
                                              Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token enum_tok, PopKeywordOrError(Keyword::kEnum));

  // We don't bind the enum's name until the end to prevent recursive references
  // to itself in its body.
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDefNoBind());

  XLS_ASSIGN_OR_RETURN(bool saw_colon, TryDropToken(TokenKind::kColon));
  TypeAnnotation* type_annotation = nullptr;
  if (saw_colon) {
    XLS_ASSIGN_OR_RETURN(type_annotation, ParseTypeAnnotation(bindings));
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));
  Bindings enum_bindings(&bindings);

  auto parse_enum_entry = [this,
                           &enum_bindings]() -> absl::StatusOr<EnumMember> {
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(enum_bindings));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
    XLS_ASSIGN_OR_RETURN(Expr * expr, ParseExpression(enum_bindings));
    return EnumMember{.name_def = name_def, .value = expr};
  };

  XLS_ASSIGN_OR_RETURN(
      std::vector<EnumMember> entries,
      ParseCommaSeq<EnumMember>(parse_enum_entry, TokenKind::kCBrace));
  auto* enum_def = module_->Make<EnumDef>(Span(start_pos, GetPos()), name_def,
                                          type_annotation, entries, is_public);
  bindings.Add(name_def->identifier(), enum_def);
  name_def->set_definer(enum_def);
  return enum_def;
}

absl::StatusOr<TypeAnnotation*> Parser::MakeBuiltinTypeAnnotation(
    const Span& span, const Token& tok, absl::Span<Expr* const> dims) {
  XLS_ASSIGN_OR_RETURN(BuiltinType builtin_type, TokenToBuiltinType(tok));
  auto* builtin_name_def = module_->GetOrCreateBuiltinNameDef(*tok.GetValue());
  TypeAnnotation* elem_type = module_->Make<BuiltinTypeAnnotation>(
      tok.span(), builtin_type, builtin_name_def);
  for (Expr* dim : dims) {
    elem_type = module_->Make<ArrayTypeAnnotation>(span, elem_type, dim);
  }
  return elem_type;
}

absl::StatusOr<TypeAnnotation*> Parser::MakeTypeRefTypeAnnotation(
    const Span& span, TypeRefOrAnnotation type_ref,
    absl::Span<Expr* const> dims, std::vector<ExprOrType> parametrics) {
  TypeAnnotation* elem_type;
  if (std::holds_alternative<TypeAnnotation*>(type_ref)) {
    elem_type = std::get<TypeAnnotation*>(type_ref);
  } else {
    elem_type = module_->Make<TypeRefTypeAnnotation>(
        span, std::get<TypeRef*>(type_ref), std::move(parametrics));
  }
  for (Expr* dim : dims) {
    elem_type = module_->Make<ArrayTypeAnnotation>(span, elem_type, dim);
  }
  return elem_type;
}

absl::StatusOr<std::variant<NameDef*, WildcardPattern*, RestOfTuple*>>
Parser::ParseNameDefOrWildcard(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(std::optional<Token> tok, TryPopIdentifierToken("_"));
  if (tok) {
    return module_->Make<WildcardPattern>(tok->span());
  }
  XLS_ASSIGN_OR_RETURN(std::optional<Token> rest,
                       TryPopToken(TokenKind::kDoubleDot));
  if (rest) {
    return module_->Make<RestOfTuple>(rest->span());
  }
  return ParseNameDef(bindings);
}

absl::StatusOr<Param*> Parser::ParseParam(Bindings& bindings,
                                          TypeAnnotation* struct_ref) {
  TypeAnnotation* type;
  XLS_ASSIGN_OR_RETURN(NameDef * name, ParseNameDef(bindings));
  XLS_ASSIGN_OR_RETURN(bool peek_is_colon, PeekTokenIs(TokenKind::kColon));
  if (name->identifier() == KeywordToString(Keyword::kSelf)) {
    CHECK(struct_ref != nullptr);
    bool explicit_type = peek_is_colon;
    type = module_->Make<SelfTypeAnnotation>(name->span(), explicit_type,
                                             struct_ref);
    if (peek_is_colon) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kColon));
      XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kSelfType));
    }
  } else {
    XLS_RETURN_IF_ERROR(
        DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                         "Expect type annotation on parameters"));
    XLS_ASSIGN_OR_RETURN(type, ParseTypeAnnotation(bindings));
  }
  if (dynamic_cast<SelfTypeAnnotation*>(type) &&
      name->identifier() != KeywordToString(Keyword::kSelf)) {
    return ParseErrorStatus(name->span(),
                            "Parameter with type `Self` must be named `self`");
  }
  auto* param = module_->Make<Param>(name, type);
  name->set_definer(param);
  return param;
}

absl::StatusOr<ProcMember*> Parser::ParseProcMember(
    Bindings& bindings, const Token& identifier_tok) {
  XLS_ASSIGN_OR_RETURN(NameDef * name, TokenToNameDef(identifier_tok));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                                       "Expect type annotation on parameters"));
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
  auto* member = module_->Make<ProcMember>(name, type);
  name->set_definer(member);
  bindings.Add(name->identifier(), name);
  return member;
}

absl::StatusOr<std::vector<Param*>> Parser::ParseParams(
    Bindings& bindings, TypeAnnotation* struct_ref) {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  auto sub_production = [&] { return ParseParam(bindings, struct_ref); };
  XLS_ASSIGN_OR_RETURN(
      std::vector<Param*> params,
      ParseCommaSeq<Param*>(sub_production, TokenKind::kCParen));
  for (int i = 1; i < params.size(); i++) {
    Param* p = params.at(i);
    if (dynamic_cast<SelfTypeAnnotation*>(p->type_annotation())) {
      return ParseErrorStatus(p->span(), "`self` must be the first parameter");
    }
  }
  return params;
}

absl::StatusOr<Number*> Parser::ParseNumber(Bindings& bindings) {
  // Token pointers are not guaranteed to persist through Peek/Pop calls, so we
  // need to make a copy for logging below.
  XLS_ASSIGN_OR_RETURN(const Token* peek_tmp, PeekToken());
  Token peek = *peek_tmp;

  if (peek.kind() == TokenKind::kNumber ||
      peek.kind() == TokenKind::kCharacter ||
      peek.IsKeywordIn({Keyword::kTrue, Keyword::kFalse})) {
    return TokenToNumber(PopTokenOrDie());
  }

  // Numbers can also be given as u32:4 -- last ditch effort to parse one of
  // those.
  absl::StatusOr<Expr*> cast = ParseCast(bindings);
  if (cast.ok() && dynamic_cast<Number*>(cast.value()) != nullptr) {
    return dynamic_cast<Number*>(cast.value());
  }

  return ParseErrorStatus(peek.span(),
                          absl::StrFormat("Expected number; got %s @ %s",
                                          TokenKindToString(peek.kind()),
                                          peek.span().ToString(file_table())));
}

absl::StatusOr<StructDef*> Parser::ParseStruct(const Pos& start_pos,
                                               bool is_public,
                                               Bindings& bindings) {
  VLOG(5) << "ParseStruct @ " << GetPos();
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kStruct));

  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(bindings));

  XLS_ASSIGN_OR_RETURN(bool dropped_oangle, TryDropToken(TokenKind::kOAngle));
  std::vector<ParametricBinding*> parametric_bindings;
  if (dropped_oangle) {
    XLS_ASSIGN_OR_RETURN(parametric_bindings,
                         ParseParametricBindings(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));

  auto parse_struct_member =
      [this, &bindings]() -> absl::StatusOr<StructMemberNode*> {
    Pos node_start_pos = GetPos();
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDefNoBind());
    Pos colon_start_pos = GetPos();
    XLS_RETURN_IF_ERROR(
        DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                         "Expect type annotation on struct field"));
    Pos colon_end_pos = GetPos();
    Span colon_span(colon_start_pos, colon_end_pos);
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
    Pos node_end_pos = GetPos();
    return module_->Make<StructMemberNode>(Span(node_start_pos, node_end_pos),
                                           name_def, colon_span, type);
  };

  XLS_ASSIGN_OR_RETURN(std::vector<StructMemberNode*> members,
                       ParseCommaSeq<StructMemberNode*>(parse_struct_member,
                                                        TokenKind::kCBrace));

  Span span(start_pos, GetPos());
  auto* struct_def =
      module_->Make<StructDef>(span, name_def, std::move(parametric_bindings),
                               std::move(members), is_public);

  name_def->set_definer(struct_def);

  bindings.Add(name_def->identifier(), struct_def);
  return struct_def;
}

absl::StatusOr<Impl*> Parser::ParseImpl(const Pos& start_pos, bool is_public,
                                        Bindings& bindings) {
  VLOG(5) << "ParseImpl @ " << GetPos();

  Bindings impl_bindings(&bindings);
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kImpl));
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type,
                       ParseTypeAnnotation(impl_bindings));

  absl::Status wrong_type_error = ParseErrorStatus(
      type->span(), "'impl' can only be defined for a 'struct'");

  TypeRefTypeAnnotation* type_ref = dynamic_cast<TypeRefTypeAnnotation*>(type);
  if (type_ref == nullptr) {
    return wrong_type_error;
  }
  std::optional<StructDefBase*> struct_def =
      TypeDefinitionToStructDefBase(type_ref->type_ref()->type_definition());
  if (!struct_def.has_value()) {
    return wrong_type_error;
  }
  if ((*struct_def)->impl().has_value()) {
    return ParseErrorStatus(
        type->span(), "'impl' can only be defined once for a given 'struct'");
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace, /*start=*/nullptr,
                                       "Opening brace for impl."));

  std::vector<ImplMember> members;
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool found_cbrace, TryDropToken(TokenKind::kCBrace));
    if (found_cbrace) {
      break;
    }
    Pos member_start_pos = GetPos();
    XLS_ASSIGN_OR_RETURN(bool next_is_public, TryDropKeyword(Keyword::kPub));
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(Keyword::kConst)) {
      XLS_ASSIGN_OR_RETURN(
          ConstantDef * constant,
          ParseConstantDef(member_start_pos, next_is_public, impl_bindings));
      members.push_back(constant);
    } else if (peek->IsKeyword(Keyword::kFn)) {
      XLS_ASSIGN_OR_RETURN(Function * function,
                           ParseImplFunction(member_start_pos, next_is_public,
                                             impl_bindings, type));
      members.push_back(function);
    } else {
      return ParseErrorStatus(
          peek->span(), "Only constants or functions are supported in impl");
    }
  }
  Span span(start_pos, GetPos());
  auto* impl = module_->Make<Impl>(span, type, std::move(members), is_public);
  (*struct_def)->set_impl(impl);
  for (Function* f : impl->GetFunctions()) {
    f->set_impl(impl);
  }
  return impl;
}

absl::StatusOr<NameDefTree*> Parser::ParseTuplePattern(const Pos& start_pos,
                                                       Bindings& bindings) {
  std::vector<NameDefTree*> members;
  bool must_end = false;
  bool rest_of_tuple_seen = false;
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_cparen, TryDropToken(TokenKind::kCParen));
    if (dropped_cparen) {
      break;
    }
    if (must_end) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
      break;
    }
    XLS_ASSIGN_OR_RETURN(NameDefTree * pattern,
                         ParsePattern(bindings, /*within_tuple_pattern=*/true));
    if (pattern->IsRestOfTupleLeaf()) {
      if (rest_of_tuple_seen) {
        return ParseErrorStatus(
            pattern->span(),
            "Rest-of-tuple (`..`) can only be used once per tuple pattern.");
      }
      rest_of_tuple_seen = true;
    }
    members.push_back(pattern);
    XLS_ASSIGN_OR_RETURN(bool dropped_comma, TryDropToken(TokenKind::kComma));
    must_end = !dropped_comma;
  }
  Span span(start_pos, GetPos());
  return module_->Make<NameDefTree>(span, std::move(members));
}

absl::StatusOr<StatementBlock*> Parser::ParseBlockExpression(
    Bindings& bindings) {
  Bindings block_bindings(&bindings);
  Pos start_pos = GetPos();
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));
  // For empty block we consider that it had a trailing semi and return unit
  // from it.
  bool last_expr_had_trailing_semi = true;
  std::vector<Statement*> stmts;
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_cbrace, TryDropToken(TokenKind::kCBrace));
    if (dropped_cbrace) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(Keyword::kType)) {
      XLS_ASSIGN_OR_RETURN(
          TypeAlias * alias,
          ParseTypeAlias(GetPos(), /*is_public=*/false, block_bindings));
      stmts.push_back(module_->Make<Statement>(alias));
      last_expr_had_trailing_semi = true;
    } else if (peek->IsKeyword(Keyword::kLet) ||
               peek->IsKeyword(Keyword::kConst)) {
      XLS_ASSIGN_OR_RETURN(Let * let, ParseLet(block_bindings));
      stmts.push_back(module_->Make<Statement>(let));
      last_expr_had_trailing_semi = true;
    } else if (peek->IsIdentifier(kConstAssertIdentifier)) {
      XLS_ASSIGN_OR_RETURN(ConstAssert * const_assert,
                           ParseConstAssert(block_bindings));
      stmts.push_back(module_->Make<Statement>(const_assert));
      last_expr_had_trailing_semi = true;
    } else {
      VLOG(5) << "ParseBlockExpression; parsing expression with bindings: ["
              << absl::StrJoin(block_bindings.GetLocalBindings(), ", ") << "]";
      XLS_ASSIGN_OR_RETURN(Expr * e, ParseExpression(block_bindings));
      stmts.push_back(module_->Make<Statement>(e));

      XLS_ASSIGN_OR_RETURN(bool dropped_semi, TryDropToken(TokenKind::kSemi));
      last_expr_had_trailing_semi = dropped_semi;
      if (!dropped_semi) {
        XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrace));
        break;
      }
    }
  }
  return module_->Make<StatementBlock>(Span(start_pos, GetPos()), stmts,
                                       last_expr_had_trailing_semi);
}

absl::StatusOr<std::vector<ParametricBinding*>> Parser::ParseParametricBindings(
    Bindings& bindings) {
  auto parse_parametric_binding =
      [this, &bindings]() -> absl::StatusOr<ParametricBinding*> {
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(bindings));
    XLS_RETURN_IF_ERROR(
        DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                         "Expect type annotation on parametric"));
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * type,
                         ParseTypeAnnotation(bindings,
                                             /*first=*/std::nullopt,
                                             /*allow_generic_type=*/true));
    if (GenericTypeAnnotation* gta =
            dynamic_cast<GenericTypeAnnotation*>(type)) {
      name_def->set_definer(gta);
    }
    XLS_ASSIGN_OR_RETURN(bool dropped_equals, TryDropToken(TokenKind::kEquals));
    Expr* expr = nullptr;
    if (dropped_equals) {
      XLS_RETURN_IF_ERROR(
          DropTokenOrError(TokenKind::kOBrace, /*start=*/nullptr,
                           "expected '{' because parametric expressions must "
                           "be enclosed in braces"));
      XLS_ASSIGN_OR_RETURN(expr, ParseExpression(bindings));
      XLS_RETURN_IF_ERROR(
          DropTokenOrError(TokenKind::kCBrace, /*start=*/nullptr,
                           "expected '}' because parametric expressions must "
                           "be enclosed in braces"));
    }
    return module_->Make<ParametricBinding>(name_def, type, expr);
  };
  XLS_ASSIGN_OR_RETURN(std::vector<ParametricBinding*> parametric_bindings,
                       ParseCommaSeq<ParametricBinding*>(
                           parse_parametric_binding, TokenKind::kCAngle));
  absl::flat_hash_map<std::string, ParametricBinding*> seen;
  for (ParametricBinding* binding : parametric_bindings) {
    if (seen.contains(binding->name_def()->identifier())) {
      return ParseErrorStatus(
          binding->name_def()->span(),
          absl::StrFormat("Duplicate parametric binding: `%s`",
                          binding->name_def()->identifier()));
    }
    seen[binding->name_def()->identifier()] = binding;
  }
  return parametric_bindings;
}

absl::StatusOr<ExprOrType> Parser::ParseParametricArg(Bindings& bindings) {
  VLOG(5) << "ParseParametricArg @ " << GetPos();
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (peek->kind() == TokenKind::kOBrace) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));
    // Conditional expressions are the first below the let/for/while set.
    XLS_ASSIGN_OR_RETURN(Expr * expr,
                         ParseRangeExpression(bindings, kNoRestrictions));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrace));

    return expr;
  }

  // We permit bare numbers as parametrics for convenience.
  if (peek->kind() == TokenKind::kNumber || peek->IsKeyword(Keyword::kTrue) ||
      peek->IsKeyword(Keyword::kFalse)) {
    return TokenToNumber(PopTokenOrDie());
  }

  if (peek->kind() == TokenKind::kIdentifier) {
    // In general, an identifier may refer to either a non-builtin type or
    // value. If it's a type, get on the track that `ParseTypeAnnotation` takes
    // with type refs.
    XLS_ASSIGN_OR_RETURN(auto nocr, ParseNameOrColonRef(bindings));
    absl::StatusOr<BoundNode> def = bindings.ResolveNodeOrError(
        ToAstNode(nocr)->ToString(), peek->span(), file_table());
    if (def.ok() && IsOneOf<TypeAlias, EnumDef, StructDef>(ToAstNode(*def))) {
      XLS_ASSIGN_OR_RETURN(TypeDefinition type_definition,
                           BoundNodeToTypeDefinition(*def));
      XLS_ASSIGN_OR_RETURN(
          TypeAnnotation * type_annotation,
          ParseTypeRefParametricsAndDims(
              bindings, peek->span(),
              module_->Make<TypeRef>(peek->span(), type_definition)));
      return MaybeParseCast(bindings, type_annotation);
    }
    if (std::holds_alternative<ColonRef*>(nocr)) {
      Span identifier_span = peek->span();
      XLS_ASSIGN_OR_RETURN(peek, PeekToken());
      // It may be an imported type followed by parametrics and dims, in which
      // case we preserve the ColonRef in the type definition without
      // resolution. Note: we may eventually need more elaborate deferral of the
      // type vs. value decision, if we want to support e.g. ColonRef + dims to
      // refer to a constant value here.
      if (peek->IsKindIn({TokenKind::kOAngle, TokenKind::kOBrack})) {
        XLS_ASSIGN_OR_RETURN(
            TypeAnnotation * type_annotation,
            ParseTypeRefParametricsAndDims(
                bindings, identifier_span,
                module_->Make<TypeRef>(identifier_span,
                                       std::get<ColonRef*>(nocr))));
        return MaybeParseCast(bindings, type_annotation);
      }
      // A ColonRef followed by a colon should be a cast. In all other cases, we
      // don't know if a ColonRef is a value or type, so we don't want to just
      // make a TypeAnnotation and do a MaybeCast here.
      XLS_ASSIGN_OR_RETURN(const bool peek_is_colon,
                           PeekTokenIs(TokenKind::kColon));
      if (peek_is_colon) {
        XLS_ASSIGN_OR_RETURN(
            TypeAnnotation * type_annotation,
            MakeTypeRefTypeAnnotation(
                identifier_span,
                module_->Make<TypeRef>(identifier_span,
                                       std::get<ColonRef*>(nocr)),
                /*dims=*/{}, /*parametrics=*/{}));
        return ParseCast(bindings, type_annotation);
      }
    }
    // Otherwise, it's a value or an unadorned imported type.
    return ToExprNode(nocr);
  }

  // If it's not an identifier, it must be a built-in type or cast to one.
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type_annotation,
                       ParseTypeAnnotation(bindings));
  return MaybeParseCast(bindings, type_annotation);
}

absl::StatusOr<std::vector<ExprOrType>> Parser::ParseParametrics(
    Bindings& bindings) {
  VLOG(5) << "ParseParametrics @ " << GetPos();
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOAngle));

  // For parametric instantiation we allow a form like:
  //
  //  chan<MyStruct<Y>>
  //
  // Which can require us to interpret the >> as two close-angle tokens instead
  // of a single '>>' token.
  DisableDoubleCAngle();
  absl::Cleanup re_enable_double_cangle = [this] { EnableDoubleCAngle(); };

  return ParseCommaSeq<ExprOrType>(
      [this, &bindings]() { return ParseParametricArg(bindings); },
      TokenKind::kCAngle);
}

absl::StatusOr<TestFunction*> Parser::ParseTestFunction(
    Bindings& bindings, const Span& attribute_span) {
  XLS_ASSIGN_OR_RETURN(
      Function * f, ParseFunctionInternal(GetPos(), /*is_public=*/false,
                                          /*is_test_utility=*/false, bindings));
  XLS_RET_CHECK(f != nullptr);
  if (std::optional<ModuleMember*> member =
          module_->FindMemberWithName(f->identifier())) {
    return ParseErrorStatus(
        f->name_def()->span(),
        absl::StrFormat(
            "Test function '%s' has same name as module member @ %s",
            f->identifier(),
            ToAstNode(**member)->GetSpan()->ToString(file_table())));
  }
  Span tf_span(attribute_span.start(), f->span().limit());
  TestFunction* tf = module_->Make<TestFunction>(tf_span, *f);
  tf->SetParentage();  // Ensure the function has its parent marked.
  return tf;
}

absl::StatusOr<TestProc*> Parser::ParseTestProc(
    Bindings& bindings, std::optional<std::string> expected_fail_label) {
  XLS_ASSIGN_OR_RETURN(ModuleMember m,
                       ParseProc(GetPos(), /*is_public=*/false,
                                 /*is_test_utility=*/false, bindings));
  if (!std::holds_alternative<Proc*>(m)) {
    // TODO: https://github.com/google/xls/issues/836 - Support `ProcDef` here.
    ProcDef* proc_def = std::get<ProcDef*>(m);
    return ParseErrorStatus(
        proc_def->span(),
        absl::StrCat("Test proc with impl is not yet supported at ",
                     proc_def->GetSpan()->ToString(file_table())));
  }
  Proc* p = std::get<Proc*>(m);
  if (std::optional<ModuleMember*> member =
          module_->FindMemberWithName(p->identifier())) {
    return ParseErrorStatus(
        p->span(), absl::StrFormat(
                       "Test proc '%s' has same name as module member @ %s",
                       p->identifier(),
                       ToAstNode(**member)->GetSpan()->ToString(file_table())));
  }

  // Verify no state or config args
  return module_->Make<TestProc>(p, expected_fail_label);
}

const Span& GetSpan(
    const std::variant<NameDef*, WildcardPattern*, RestOfTuple*>& v) {
  if (std::holds_alternative<NameDef*>(v)) {
    return std::get<NameDef*>(v)->span();
  }
  if (std::holds_alternative<RestOfTuple*>(v)) {
    return std::get<RestOfTuple*>(v)->span();
  }
  return std::get<WildcardPattern*>(v)->span();
}

}  // namespace xls::dslx
