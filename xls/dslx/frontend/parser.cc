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

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
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
#include "xls/ir/code_template.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/name_uniquer.h"

namespace xls::dslx {
namespace {

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

absl::Status MakeModuleTopCollisionError(std::string_view module_name,
                                         std::string_view member_name,
                                         const Span& existing_span,
                                         const AstNode* existing_node,
                                         const Span& new_span,
                                         const AstNode* new_node) {
  return ParseErrorStatus(
      new_span,
      absl::StrFormat("Module `%s` already contains a member named `%s` @ %s",
                      module_name, member_name, existing_span.ToString()));
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

bool TypeIsToken(TypeAnnotation* type) {
  auto* builtin_type = dynamic_cast<BuiltinTypeAnnotation*>(type);
  if (builtin_type == nullptr) {
    return false;
  }

  return builtin_type->builtin_type() == BuiltinType::kToken;
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
  if (auto* e = TryGet<EnumDef*>(bn)) { return TypeDefinition(e); }
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

absl::StatusOr<Function*> Parser::ParseFunction(
    bool is_public, Bindings& bindings,
    absl::flat_hash_map<std::string, Function*>* name_to_fn) {
  XLS_ASSIGN_OR_RETURN(Function * f,
                       ParseFunctionInternal(is_public, bindings));
  if (name_to_fn == nullptr) {
    return f;
  }
  auto [item, inserted] = name_to_fn->insert({f->identifier(), f});
  if (!inserted) {
    return ParseErrorStatus(
        f->name_def()->span(),
        absl::StrFormat("Function '%s' is defined in this module multiple "
                        "times; previously @ %s'",
                        f->identifier(), item->second->span().ToString()));
  }
  XLS_RETURN_IF_ERROR(VerifyParentage(f));
  return f;
}

absl::Status Parser::ParseModuleAttribute() {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrack));
  Span identifier_span;
  XLS_ASSIGN_OR_RETURN(std::string identifier,
                       PopIdentifierOrError(&identifier_span));
  if (identifier != "allow") {
    return ParseErrorStatus(
        identifier_span,
        "Only 'allow' is supported as a module-level attribute");
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  XLS_ASSIGN_OR_RETURN(std::string to_allow, PopIdentifierOrError());
  if (to_allow == "nonstandard_constant_naming") {
    module_->AddAnnotation(ModuleAnnotation::kAllowNonstandardConstantNaming);
  }
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Module>> Parser::ParseModule(
    Bindings* bindings) {
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

  while (!AtEof()) {
    XLS_ASSIGN_OR_RETURN(bool peek_is_eof, PeekTokenIs(TokenKind::kEof));
    if (peek_is_eof) {
      XLS_VLOG(3) << "Parser saw EOF token for module " << module_->name()
                  << ", stopping.";
      break;
    }

    XLS_ASSIGN_OR_RETURN(bool dropped_pub, TryDropKeyword(Keyword::kPub));
    if (dropped_pub) {
      XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
      if (peek->IsKeyword(Keyword::kFn)) {
        XLS_ASSIGN_OR_RETURN(Function * fn,
                             ParseFunction(
                                 /*is_public=*/true, *bindings, &name_to_fn));
        XLS_RETURN_IF_ERROR(module_->AddTop(fn, MakeModuleTopCollisionError));
        continue;
      }

      if (peek->IsKeyword(Keyword::kProc)) {
        XLS_ASSIGN_OR_RETURN(Proc * proc, ParseProc(
                                              /*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(proc, MakeModuleTopCollisionError));
        continue;
      }

      if (peek->IsKeyword(Keyword::kStruct)) {
        XLS_ASSIGN_OR_RETURN(StructDef * struct_def,
                             ParseStruct(/*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(struct_def, MakeModuleTopCollisionError));
        continue;
      }

      if (peek->IsKeyword(Keyword::kEnum)) {
        XLS_ASSIGN_OR_RETURN(EnumDef * enum_def,
                             ParseEnumDef(/*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(enum_def, MakeModuleTopCollisionError));
        continue;
      }

      if (peek->IsKeyword(Keyword::kConst)) {
        XLS_ASSIGN_OR_RETURN(ConstantDef * def,
                             ParseConstantDef(/*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(def, MakeModuleTopCollisionError));
        continue;
      }

      if (peek->IsKeyword(Keyword::kType)) {
        XLS_RET_CHECK(bindings != nullptr);
        XLS_ASSIGN_OR_RETURN(TypeAlias * type_alias,
                             ParseTypeAlias(/*is_public=*/true, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(type_alias, MakeModuleTopCollisionError));
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
              [&](TestFunction* t) {
                return module_->AddTop(t, MakeModuleTopCollisionError);
              },
              [&](Function* f) {
                return module_->AddTop(f, MakeModuleTopCollisionError);
              },
              [&](TestProc* tp) {
                return module_->AddTop(tp, MakeModuleTopCollisionError);
              },
              [&](QuickCheck* qc) {
                return module_->AddTop(qc, MakeModuleTopCollisionError);
              },
              [&](std::nullptr_t) { return absl::OkStatus(); },
          },
          attribute));
      continue;
    }

    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());

    if (peek->IsIdentifier("const_assert!")) {
      XLS_ASSIGN_OR_RETURN(ConstAssert * const_assert,
                           ParseConstAssert(*bindings));
      // Note: const_assert! doesn't make a binding so we don't need to provide
      // the error lambda.
      XLS_RETURN_IF_ERROR(
          module_->AddTop(const_assert, /*make_collision_error=*/nullptr));
      continue;
    }

    auto top_level_error = [peek] {
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
        XLS_ASSIGN_OR_RETURN(Function * fn,
                             ParseFunction(
                                 /*is_public=*/false, *bindings, &name_to_fn));
        XLS_RETURN_IF_ERROR(module_->AddTop(fn, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kProc: {
        XLS_ASSIGN_OR_RETURN(Proc * proc, ParseProc(
                                              /*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(module_->AddTop(proc, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kImport: {
        XLS_ASSIGN_OR_RETURN(Import * import, ParseImport(*bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(import, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kType: {
        XLS_RET_CHECK(bindings != nullptr);
        XLS_ASSIGN_OR_RETURN(TypeAlias * type_alias,
                             ParseTypeAlias(/*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(type_alias, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kStruct: {
        XLS_ASSIGN_OR_RETURN(StructDef * struct_,
                             ParseStruct(/*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(struct_, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kEnum: {
        XLS_ASSIGN_OR_RETURN(EnumDef * enum_,
                             ParseEnumDef(/*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(enum_, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kConst: {
        XLS_ASSIGN_OR_RETURN(ConstantDef * const_def,
                             ParseConstantDef(/*is_public=*/false, *bindings));
        XLS_RETURN_IF_ERROR(
            module_->AddTop(const_def, MakeModuleTopCollisionError));
        break;
      }
      case Keyword::kImpl: {
        return ParseErrorStatus(
            peek->span(),
            "`impl` is not yet implemented in DSLX, please use stand-alone "
            "functions that take the struct as a value instead");
      }
      default:
        return top_level_error();
    }
  }

  // Ensure we've consumed all tokens when we're done parsing, as a
  // post-condition.
  XLS_RET_CHECK(AtEof());

  XLS_RETURN_IF_ERROR(VerifyParentage(module_.get()));
  auto result = std::move(module_);
  module_ = nullptr;
  return result;
}

absl::StatusOr<std::variant<TestFunction*, Function*, TestProc*, QuickCheck*,
                            std::nullptr_t>>
Parser::ParseAttribute(absl::flat_hash_map<std::string, Function*>* name_to_fn,
                       Bindings& bindings, const Pos& hash_pos) {
  // Ignore the Rust "bang" in Attribute declarations, i.e. we don't yet have
  // a use for inner vs. outer attributes, but that day will likely come.
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrack));
  XLS_ASSIGN_OR_RETURN(Token directive_tok,
                       PopTokenOrError(TokenKind::kIdentifier));
  const std::string& directive_name = directive_tok.GetStringValue();

  if (directive_name == "test") {
    XLS_ASSIGN_OR_RETURN(Token cbrack, PopTokenOrError(TokenKind::kCBrack));
    XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
    if (peek->IsKeyword(Keyword::kFn)) {
      return ParseTestFunction(bindings, Span(hash_pos, cbrack.span().limit()));
    }

    return ParseErrorStatus(
        peek->span(), absl::StrCat("Invalid test type: ", peek->ToString()));
  }
  if (directive_name == "extern_verilog") {
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
    XLS_ASSIGN_OR_RETURN(Function * f,
                         ParseFunction(dropped_pub, bindings, name_to_fn));
    absl::StatusOr<ForeignFunctionData> parsed_ffi_annotation =
        ForeignFunctionDataCreateFromTemplate(ffi_annotation);
    if (!parsed_ffi_annotation.ok()) {
      const int64_t error_at =
          CodeTemplate::ExtractErrorColumn(parsed_ffi_annotation.status());
      Pos error_pos{template_start.filename(), template_start.lineno(),
                    template_start.colno() + error_at};
      dslx::Span error_span(error_pos, error_pos);
      return ParseErrorStatus(error_span,
                              parsed_ffi_annotation.status().message());
    }
    f->set_extern_verilog_module(*parsed_ffi_annotation);
    return f;
  }
  if (directive_name == "test_proc") {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
    return ParseTestProc(bindings);
  }
  if (directive_name == "quickcheck") {
    XLS_ASSIGN_OR_RETURN(QuickCheck * n,
                         ParseQuickCheck(name_to_fn, bindings, hash_pos));
    return n;
  }
  return ParseErrorStatus(
      directive_tok.span(),
      absl::StrFormat("Unknown directive: '%s'", directive_name));
}

absl::StatusOr<Expr*> Parser::ParseExpression(Bindings& bindings,
                                              ExprRestrictions restrictions) {
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());

  if (++approximate_expression_depth_ >= kApproximateExpressionDepthLimit) {
    return ParseErrorStatus(peek->span(),
                            "Expression is too deeply nested, please break "
                            "into multiple statements");
  }
  auto bump_down = absl::Cleanup([this] { approximate_expression_depth_--; });

  XLS_VLOG(5) << "ParseExpression @ " << GetPos() << " peek: `"
              << peek->ToString() << "`";
  if (peek->IsKeyword(Keyword::kFor)) {
    return ParseFor(bindings);
  }
  if (peek->IsKeyword(Keyword::kUnrollFor)) {
    return ParseUnrollFor(bindings);
  }
  if (peek->IsKeyword(Keyword::kChannel)) {
    return ParseChannelDecl(bindings);
  }
  if (peek->IsKeyword(Keyword::kSpawn)) {
    return ParseSpawn(bindings);
  }
  if (peek->kind() == TokenKind::kOBrace) {
    return ParseBlockExpression(bindings);
  }
  return ParseConditionalExpression(bindings, restrictions);
}

absl::StatusOr<Expr*> Parser::ParseRangeExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_VLOG(5) << "ParseRangeExpression @ " << GetPos();
  XLS_ASSIGN_OR_RETURN(Expr * result,
                       ParseLogicalOrExpression(bindings, restrictions));
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (peek->kind() == TokenKind::kDoubleDot) {
    DropTokenOrDie();
    XLS_ASSIGN_OR_RETURN(Expr * rhs,
                         ParseLogicalOrExpression(bindings, restrictions));
    result = module_->Make<Range>(
        Span(result->span().start(), rhs->span().limit()), result, rhs);
  }
  return result;
}

absl::StatusOr<Conditional*> Parser::ParseConditionalNode(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_ASSIGN_OR_RETURN(Token if_, PopKeywordOrError(Keyword::kIf));
  XLS_ASSIGN_OR_RETURN(
      Expr * test,
      ParseExpression(bindings,
                      MakeRestrictions({ExprRestriction::kNoStructLiteral})));
  XLS_ASSIGN_OR_RETURN(Block * consequent, ParseBlockExpression(bindings));
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kElse));

  std::variant<Block*, Conditional*> alternate;

  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (peek->IsKeyword(Keyword::kIf)) {  // Conditional expression.
    XLS_ASSIGN_OR_RETURN(alternate,
                         ParseConditionalNode(bindings, kNoRestrictions));
  } else {
    XLS_ASSIGN_OR_RETURN(alternate, ParseBlockExpression(bindings));
  }

  return module_->Make<Conditional>(Span(if_.span().start(), GetPos()), test,
                                    consequent, alternate);
}

absl::StatusOr<Expr*> Parser::ParseConditionalExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_VLOG(5) << "ParseConditionalExpression @ " << GetPos()
              << " restrictions: " << ExprRestrictionsToString(restrictions);
  XLS_ASSIGN_OR_RETURN(bool peek_is_if, PeekTokenIs(Keyword::kIf));
  if (peek_is_if) {
    return ParseConditionalNode(bindings, restrictions);
  }

  // No leading 'if' keyword -- we fall back to the RangeExpression production.
  return ParseRangeExpression(bindings, restrictions);
}

absl::StatusOr<ConstAssert*> Parser::ParseConstAssert(Bindings& bindings) {
  Pos start = GetPos();
  XLS_ASSIGN_OR_RETURN(std::string identifier, PopIdentifierOrError());
  XLS_RET_CHECK_EQ(identifier, "const_assert!");
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

absl::StatusOr<TypeAlias*> Parser::ParseTypeAlias(bool is_public,
                                                  Bindings& bindings) {
  const Pos start_pos = GetPos();
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kType));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemi));
  Span span(start_pos, GetPos());
  auto* type_alias = module_->Make<TypeAlias>(span, name_def, type, is_public);
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

absl::StatusOr<TypeRef*> Parser::ParseTypeRef(Bindings& bindings,
                                              const Token& tok) {
  if (tok.kind() != TokenKind::kIdentifier) {
    return ParseErrorStatus(tok.span(), absl::StrFormat("Expected type; got %s",
                                                        tok.ToErrorString()));
  }

  XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  if (peek_is_double_colon) {
    return ParseModTypeRef(bindings, tok);
  }
  XLS_ASSIGN_OR_RETURN(BoundNode type_def, bindings.ResolveNodeOrError(
                                               *tok.GetValue(), tok.span()));
  if (!IsOneOf<TypeAlias, EnumDef, StructDef>(ToAstNode(type_def))) {
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
    Bindings& bindings, std::optional<Token> first) {
  XLS_VLOG(5) << "ParseTypeAnnotation @ " << GetPos();
  if (!first.has_value()) {
    XLS_ASSIGN_OR_RETURN(first, PopToken());
  }
  const Token& tok = first.value();
  XLS_VLOG(5) << "ParseTypeAnnotation; popped: " << tok.ToString();

  if (tok.IsTypeKeyword()) {  // Builtin types.
    Pos start_pos = tok.span().start();
    if (tok.GetKeyword() == Keyword::kChannel) {
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
      TypeAnnotation* type =
          module_->Make<ChannelTypeAnnotation>(span, direction, payload, dims);

      return type;
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
    XLS_VLOG(5) << "ParseTypeAnnotation; got " << types.size()
                << " tuple members";

    Span span(tok.span().start(), GetPos());
    TypeAnnotation* type =
        module_->Make<TupleTypeAnnotation>(span, std::move(types));
    XLS_VLOG(5) << "ParseTypeAnnotation; result type: " << type->ToString();

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
  XLS_ASSIGN_OR_RETURN(TypeRef * type_ref, ParseTypeRef(bindings, tok));

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

  Span span(tok.span().start(), GetPos());
  return MakeTypeRefTypeAnnotation(span, type_ref, dims,
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

  // If we failed to parse this ref, then put it back on the queue, in case
  // we try another production.
  XLS_ASSIGN_OR_RETURN(
      BoundNode bn, bindings.ResolveNodeOrError(*tok->GetValue(), tok->span()));
  AnyNameDef name_def = BoundNodeToAnyNameDef(bn);
  if (std::holds_alternative<ConstantDef*>(bn)) {
    return module_->Make<ConstRef>(tok->span(), *tok->GetValue(), name_def);
  }

  if (std::holds_alternative<NameDef*>(bn)) {
    // As opposed to the AnyNameDef above.
    const AstNode* node = std::get<NameDef*>(bn);
    while (node->parent() != nullptr &&
           node->parent()->kind() == AstNodeKind::kNameDefTree) {
      node = node->parent();
    }

    // Since Let construction is deferred, we can't look up the definer of this
    // NameDef[Tree]; it doesn't exist yet. Instead, we just note that a given
    // NDT is const (or not, by omission in the set).
    if (dynamic_cast<const NameDefTree*>(node) != nullptr &&
        const_ndts_.contains(dynamic_cast<const NameDefTree*>(node))) {
      return module_->Make<ConstRef>(tok->span(), *tok->GetValue(), name_def);
    }
  }
  return module_->Make<NameRef>(tok->span(), *tok->GetValue(), name_def);
}

absl::StatusOr<ColonRef*> Parser::ParseColonRef(Bindings& bindings,
                                                ColonRef::Subject subject) {
  Pos start = GetPos();
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kDoubleColon));
  while (true) {
    XLS_ASSIGN_OR_RETURN(Token value_tok,
                         PopTokenOrError(TokenKind::kIdentifier));
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

absl::StatusOr<Expr*> Parser::ParseCastOrEnumRefOrStructInstance(
    Bindings& bindings) {
  XLS_VLOG(5) << "ParseCastOrEnumRefOrStructInstance @ " << GetPos()
              << " peek: `" << PeekToken().value()->ToString() << "`";

  Token tok = PopTokenOrDie();
  XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  if (peek_is_double_colon) {
    XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &tok));
    XLS_ASSIGN_OR_RETURN(ColonRef * ref, ParseColonRef(bindings, subject));
    return ref;
  }

  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type,
                       ParseTypeAnnotation(bindings, tok));
  XLS_ASSIGN_OR_RETURN(bool peek_is_obrace, PeekTokenIs(TokenKind::kOBrace));
  Expr* expr;
  if (peek_is_obrace) {
    XLS_ASSIGN_OR_RETURN(expr, ParseStructInstance(bindings, type));
  } else {
    XLS_ASSIGN_OR_RETURN(expr, ParseCast(bindings, type));
  }
  return expr;
}

absl::StatusOr<Expr*> Parser::ParseStructInstance(Bindings& bindings,
                                                  TypeAnnotation* type) {
  XLS_VLOG(5) << "ParseStructInstance @ " << GetPos();

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
  XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier,
                                                  /*start=*/nullptr, context));
  XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                       PeekTokenIs(TokenKind::kDoubleColon));
  if (peek_is_double_colon) {
    XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &tok));
    return ParseColonRef(bindings, subject);
  }
  return ParseNameRef(bindings, &tok);
}

absl::StatusOr<NameDef*> Parser::ParseNameDef(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token tok, PopTokenOrError(TokenKind::kIdentifier));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, TokenToNameDef(tok));
  bindings.Add(name_def->identifier(), name_def);
  return name_def;
}

absl::StatusOr<NameDefTree*> Parser::ParseNameDefTree(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token start, PopTokenOrError(TokenKind::kOParen));

  auto parse_name_def_or_tree = [&bindings,
                                 this]() -> absl::StatusOr<NameDefTree*> {
    XLS_ASSIGN_OR_RETURN(bool peek_is_oparen, PeekTokenIs(TokenKind::kOParen));
    if (peek_is_oparen) {
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
              seen[name_def->identifier()]->span().ToString()));
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
  if (std::all_of(exprs.begin(), exprs.end(), IsConstant)) {
    return module_->Make<ConstantArray>(span, std::move(exprs),
                                        has_trailing_ellipsis);
  }
  return module_->Make<Array>(span, std::move(exprs), has_trailing_ellipsis);
}

absl::StatusOr<Expr*> Parser::ParseCast(Bindings& bindings,
                                        TypeAnnotation* type) {
  XLS_VLOG(5) << "ParseCast @ " << GetPos()
              << " type: " << (type == nullptr ? "<null>" : type->ToString());

  if (type == nullptr) {
    absl::StatusOr<TypeAnnotation*> type_status = ParseTypeAnnotation(bindings);
    if (type_status.status().ok()) {
      type = type_status.value();
    } else {
      PositionalErrorData data =
          GetPositionalErrorData(type_status.status()).value();
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
      lhs = module_->Make<Binop>(op.span(), kind, lhs, rhs);
    } else {
      break;
    }
  }
  return lhs;
}

absl::StatusOr<Expr*> Parser::ParseLogicalAndExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_VLOG(5) << "ParseLogicalAndExpression @ " << GetPos();
  std::initializer_list<TokenKind> kinds = {TokenKind::kDoubleAmpersand};
  return ParseBinopChain(
      [this, &bindings, restrictions] {
        return ParseComparisonExpression(bindings, restrictions);
      },
      kinds);
}

absl::StatusOr<Expr*> Parser::ParseLogicalOrExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_VLOG(5) << "ParseLogicalOrExpression @ " << GetPos();
  static const std::initializer_list<TokenKind> kinds = {TokenKind::kDoubleBar};
  return ParseBinopChain(
      [this, &bindings, restrictions] {
        return ParseLogicalAndExpression(bindings, restrictions);
      },
      kinds);
}

absl::StatusOr<Expr*> Parser::ParseStrongArithmeticExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  auto sub_production = [&] {
    return ParseCastAsExpression(bindings, restrictions);
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
  XLS_VLOG(5) << "ParseComparisonExpression @ " << GetPos() << " peek: `"
              << PeekToken().value()->ToString()
              << "` restrictions: " << ExprRestrictionsToString(restrictions);
  XLS_ASSIGN_OR_RETURN(Expr * lhs, ParseOrExpression(bindings, restrictions));
  while (true) {
    XLS_VLOG(5) << "ParseComparisonExpression; lhs: `" << lhs->ToString()
                << "` peek: `" << PeekToken().value()->ToString() << "`";
    XLS_ASSIGN_OR_RETURN(bool peek_in_targets, PeekTokenIn(kComparisonKinds));
    if (!peek_in_targets) {
      XLS_VLOG(5) << "Peek is not in comparison kinds.";
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
    lhs = module_->Make<Binop>(op.span(), kind, lhs, rhs);
  }
  XLS_VLOG(5) << "ParseComparisonExpression; result: `" << lhs->ToString()
              << "`";
  return lhs;
}

absl::StatusOr<NameDefTree*> Parser::ParsePattern(Bindings& bindings) {
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
    XLS_ASSIGN_OR_RETURN(bool peek_is_double_colon,
                         PeekTokenIs(TokenKind::kDoubleColon));
    if (peek_is_double_colon) {  // Mod or enum ref.
      XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &tok));
      XLS_ASSIGN_OR_RETURN(ColonRef * colon_ref,
                           ParseColonRef(bindings, subject));
      return module_->Make<NameDefTree>(tok.span(), colon_ref);
    }

    std::optional<BoundNode> resolved = bindings.ResolveNode(*tok.GetValue());
    NameRef* ref;
    if (resolved) {
      AnyNameDef name_def =
          bindings.ResolveNameOrNullopt(*tok.GetValue()).value();
      if (std::holds_alternative<ConstantDef*>(*resolved)) {
        ref = module_->Make<ConstRef>(tok.span(), *tok.GetValue(), name_def);
      } else {
        ref = module_->Make<NameRef>(tok.span(), *tok.GetValue(), name_def);
      }
      return module_->Make<NameDefTree>(tok.span(), ref);
    }

    // If the name is not bound, this pattern is creating a binding.
    XLS_ASSIGN_OR_RETURN(NameDef * name_def, TokenToNameDef(tok));
    bindings.Add(name_def->identifier(), name_def);
    auto* result = module_->Make<NameDefTree>(tok.span(), name_def);
    name_def->set_definer(result);
    return result;
  }

  if (peek->IsKindIn({TokenKind::kNumber, TokenKind::kCharacter, Keyword::kTrue,
                      Keyword::kFalse}) ||
      peek->IsTypeKeyword()) {
    XLS_ASSIGN_OR_RETURN(Number * number, ParseNumber(bindings));
    XLS_ASSIGN_OR_RETURN(bool peek_is_double_dot,
                         PeekTokenIs(TokenKind::kDoubleDot));
    if (peek_is_double_dot) {
      XLS_RETURN_IF_ERROR(DropToken());
      XLS_ASSIGN_OR_RETURN(Number * limit, ParseNumber(bindings));
      auto* range = module_->Make<Range>(
          Span(number->span().start(), limit->span().limit()), number, limit);
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
    XLS_ASSIGN_OR_RETURN(NameDefTree * first_pattern,
                         ParsePattern(arm_bindings));
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
      XLS_ASSIGN_OR_RETURN(NameDefTree * pattern, ParsePattern(arm_bindings));
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
    std::string span_str =
        maybe_span.has_value() ? " at " + maybe_span->ToString() : "";
    return ParseErrorStatus(
        name_def->span(),
        absl::StrFormat("Import of `%s` is shadowing an existing definition%s",
                        name_def->identifier(), span_str));
  }

  XLS_RETURN_IF_ERROR(
      DropTokenOrError(TokenKind::kSemi, /*start=*/&kw,
                       /*context=*/"Expect an ';' at end of import statement"));

  auto* import = module_->Make<Import>(kw.span(), subject, *name_def, alias);
  name_def->set_definer(import);
  bindings.Add(name_def->identifier(), import);
  return import;
}

absl::StatusOr<Function*> Parser::ParseFunctionInternal(
    bool is_public, Bindings& outer_bindings) {
  XLS_ASSIGN_OR_RETURN(Token fn_tok, PopKeywordOrError(Keyword::kFn));
  const Pos start_pos = fn_tok.span().start();

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

  XLS_ASSIGN_OR_RETURN(std::vector<Param*> params, ParseParams(bindings));

  XLS_ASSIGN_OR_RETURN(bool dropped_arrow, TryDropToken(TokenKind::kArrow));
  TypeAnnotation* return_type = nullptr;
  if (dropped_arrow) {
    XLS_ASSIGN_OR_RETURN(return_type, ParseTypeAnnotation(bindings));
  }

  XLS_ASSIGN_OR_RETURN(Block * body, ParseBlockExpression(bindings));
  Function* f = module_->Make<Function>(
      Span(start_pos, GetPos()), name_def, std::move(parametric_bindings),
      params, return_type, body, FunctionTag::kNormal, is_public);
  name_def->set_definer(f);
  return f;
}

absl::StatusOr<QuickCheck*> Parser::ParseQuickCheck(
    absl::flat_hash_map<std::string, Function*>* name_to_fn, Bindings& bindings,
    const Pos& hash_pos) {
  std::optional<int64_t> test_count;
  XLS_ASSIGN_OR_RETURN(bool peek_is_paren, PeekTokenIs(TokenKind::kOParen));
  if (peek_is_paren) {  // Config is specified.
    DropTokenOrDie();
    Span config_name_span;
    XLS_ASSIGN_OR_RETURN(std::string config_name,
                         PopIdentifierOrError(&config_name_span));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
    if (config_name == "test_count") {
      XLS_ASSIGN_OR_RETURN(Token count_token,
                           PopTokenOrError(TokenKind::kNumber));
      XLS_ASSIGN_OR_RETURN(test_count, count_token.GetValueAsInt64());
      if (test_count <= 0) {
        return ParseErrorStatus(
            count_token.span(),
            absl::StrFormat("Number of tests should be > 0, got %d",
                            *test_count));
      }
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
    } else {
      return ParseErrorStatus(
          config_name_span,
          absl::StrFormat("Unknown configuration key in directive: '%s'",
                          config_name));
    }
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrack));
  XLS_ASSIGN_OR_RETURN(
      Function * fn, ParseFunction(/*is_public=*/false, bindings, name_to_fn));
  const Span quickcheck_span(hash_pos, fn->span().limit());
  return module_->Make<QuickCheck>(quickcheck_span, fn, test_count);
}

absl::StatusOr<XlsTuple*> Parser::ParseTupleRemainder(const Pos& start_pos,
                                                      Expr* first,
                                                      Bindings& bindings) {
  XLS_VLOG(5) << "ParseTupleRemainder @ " << GetPos();
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
  const Pos start_pos = peek->span().start();
  XLS_VLOG(5) << "ParseTerm @ " << start_pos << " peek: `" << peek->ToString()
              << "` restrictions: " << ExprRestrictionsToString(restrictions);

  bool peek_is_kw_in = peek->IsKeyword(Keyword::kIn);
  bool peek_is_kw_out = peek->IsKeyword(Keyword::kOut);

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
        XLS_LOG(FATAL) << "Inconsistent unary operation token kind.";
    }
    Span span(start_pos, GetPos());
    lhs = module_->Make<Unop>(span, unop_kind, arg);
  } else if (peek->IsTypeKeyword() ||
             (peek->kind() == TokenKind::kIdentifier &&
              outer_bindings.ResolveNodeIsTypeDefinition(*peek->GetValue()))) {
    XLS_ASSIGN_OR_RETURN(lhs,
                         ParseCastOrEnumRefOrStructInstance(outer_bindings));
  } else if (peek->kind() == TokenKind::kIdentifier || peek_is_kw_in ||
             peek_is_kw_out) {
    XLS_ASSIGN_OR_RETURN(auto nocr, ParseNameOrColonRef(outer_bindings));
    if (std::holds_alternative<ColonRef*>(nocr) &&
        !IsExprRestrictionEnabled(restrictions,
                                  ExprRestriction::kNoStructLiteral)) {
      XLS_ASSIGN_OR_RETURN(bool peek_is_obrace,
                           PeekTokenIs(TokenKind::kOBrace));
      if (peek_is_obrace) {
        ColonRef* colon_ref = std::get<ColonRef*>(nocr);
        TypeRef* type_ref =
            module_->Make<TypeRef>(colon_ref->span(), colon_ref);
        XLS_ASSIGN_OR_RETURN(
            TypeAnnotation * type,
            MakeTypeRefTypeAnnotation(colon_ref->span(), type_ref, {}, {}));
        return ParseStructInstance(outer_bindings, type);
      }
    }
    lhs = ToExprNode(nocr);
  } else if (peek->kind() == TokenKind::kOParen) {  // Parenthesized expression.
    XLS_ASSIGN_OR_RETURN(lhs,
                         ParseTermLhsParenthesized(outer_bindings, start_pos));
  } else if (peek->IsKeyword(Keyword::kMatch)) {  // Match expression.
    XLS_ASSIGN_OR_RETURN(lhs, ParseMatch(outer_bindings));
  } else if (peek->kind() == TokenKind::kOBrack) {  // Array expression.
    XLS_ASSIGN_OR_RETURN(lhs, ParseArray(outer_bindings));
  } else if (peek->IsKeyword(Keyword::kIf)) {  // Conditional expression.
    XLS_ASSIGN_OR_RETURN(
        lhs, ParseConditionalExpression(outer_bindings, kNoRestrictions));
  } else {
    return ParseErrorStatus(
        peek->span(),
        absl::StrFormat("Expected start of an expression; got: %s",
                        peek->ToErrorString()));
  }
  CHECK(lhs != nullptr);
  return lhs;
}

absl::StatusOr<Expr*> Parser::ParseTermRhs(Expr* lhs, Bindings& outer_bindings,
                                           ExprRestrictions restrictions) {
  const Pos new_pos = GetPos();
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  switch (peek->kind()) {
    case TokenKind::kColon: {  // Possibly a Number of ColonRef type.
      Span span(new_pos, GetPos());
      // The only valid construct here would be declaring a number via
      // ColonRef-colon-Number, e.g., "module::type:7".
      if (dynamic_cast<ColonRef*>(lhs) == nullptr) {
        goto done;
      }

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
            absl::StatusOr<TypeDefinition> type_definition_or =
                ToTypeDefinition(lhs);
            if (!type_definition_or.ok()) {
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
            auto* type_ref =
                module_->Make<TypeRef>(span, type_definition_or.value());
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
      // Comparison op or parametric function invocation.
      // TODO(rspringer): Or parameterization on ColonRef-to-type.
      Transaction sub_txn(this, &outer_bindings);
      absl::Cleanup sub_cleanup = [&sub_txn]() { sub_txn.Rollback(); };

      auto parametrics_or = ParseParametrics(*sub_txn.bindings());
      if (!parametrics_or.ok()) {
        XLS_VLOG(5) << "ParseParametrics gave error: "
                    << parametrics_or.status();
        goto done;
      }

      XLS_ASSIGN_OR_RETURN(
          Token tok,
          PopTokenOrError(
              TokenKind::kOParen, /*start=*/nullptr,
              "Expected a '(' after parametrics for function invocation."));
      Bindings* b = sub_txn.bindings();
      XLS_ASSIGN_OR_RETURN(
          std::vector<Expr*> args,
          ParseCommaSeq<Expr*>([b, this] { return ParseExpression(*b); },
                               TokenKind::kCParen));
      XLS_ASSIGN_OR_RETURN(
          lhs, BuildMacroOrInvocation(Span(new_pos, GetPos()),
                                      *sub_txn.bindings(), lhs, std::move(args),
                                      std::move(parametrics_or).value()));
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

absl::StatusOr<Expr*> Parser::ParseTerm(Bindings& outer_bindings,
                                        ExprRestrictions restrictions) {
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

absl::StatusOr<Expr*> Parser::BuildMacroOrInvocation(
    const Span& span, Bindings& bindings, Expr* callee, std::vector<Expr*> args,
    std::vector<ExprOrType> parametrics) {
  if (auto* name_ref = dynamic_cast<NameRef*>(callee)) {
    if (auto* builtin = TryGet<BuiltinNameDef*>(name_ref->name_def())) {
      std::string name = builtin->identifier();
      if (name == "trace_fmt!") {
        if (!parametrics.empty()) {
          return ParseErrorStatus(
              span, absl::StrFormat(
                        "%s macro does not take parametric arguments", name));
        }
        if (args.empty()) {
          return ParseErrorStatus(
              span, absl::StrFormat("%s macro must have at least one argument",
                                    name));
        }

        Expr* format_arg = args[0];
        if (String* format_string = dynamic_cast<String*>(format_arg)) {
          const std::string& format_text = format_string->text();
          absl::StatusOr<std::vector<FormatStep>> format_result =
              ParseFormatString(format_text);
          if (!format_result.ok()) {
            return ParseErrorStatus(format_string->span(),
                                    format_result.status().message());
          }
          // Remove the format string argument before building the macro call.
          args.erase(args.begin());
          return module_->Make<FormatMacro>(span, name, format_result.value(),
                                            args);
        }

        return ParseErrorStatus(
            span, absl::StrFormat("The first argument of the %s macro must "
                                  "be a literal string.",
                                  name));
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
            bindings.AddFailLabel(label->text(), label->span()));
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
    // We avoid name collisions b/w exiesting functions and Proc config/next fns
    // by using a "." as the separator, which is invalid for function
    // specifications.
    std::string config_name = absl::StrCat(name_ref->identifier(), ".config");
    std::string next_name = absl::StrCat(name_ref->identifier(), ".next");
    std::string init_name = absl::StrCat(name_ref->identifier(), ".init");
    XLS_ASSIGN_OR_RETURN(
        AnyNameDef config_def,
        bindings.ResolveNameOrError(config_name, spawnee->span()));
    if (!std::holds_alternative<const NameDef*>(config_def)) {
      return absl::InternalError("Proc config should be named \".config\"");
    }
    config_ref =
        module_->Make<NameRef>(name_ref->span(), config_name, config_def);

    XLS_ASSIGN_OR_RETURN(AnyNameDef next_def, bindings.ResolveNameOrError(
                                                  next_name, spawnee->span()));
    if (!std::holds_alternative<const NameDef*>(next_def)) {
      return absl::InternalError("Proc next should be named \".next\"");
    }
    next_ref = module_->Make<NameRef>(name_ref->span(), next_name, next_def);

    XLS_ASSIGN_OR_RETURN(AnyNameDef init_def, bindings.ResolveNameOrError(
                                                  init_name, spawnee->span()));
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

absl::StatusOr<Expr*> Parser::ParseCastAsExpression(
    Bindings& bindings, ExprRestrictions restrictions) {
  XLS_VLOG(5) << "ParseCastAsExpression @ " << GetPos()
              << " restrictions: " << ExprRestrictionsToString(restrictions);
  XLS_ASSIGN_OR_RETURN(Expr * lhs, ParseTerm(bindings, restrictions));
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

absl::StatusOr<ConstantDef*> Parser::ParseConstantDef(bool is_public,
                                                      Bindings& bindings) {
  const Pos start_pos = GetPos();
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kConst));
  Bindings new_bindings(/*parent=*/&bindings);
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(new_bindings));
  if (bindings.HasName(name_def->identifier())) {
    Span span =
        BoundNodeGetSpan(bindings.ResolveNode(name_def->identifier()).value());
    return ParseErrorStatus(
        name_def->span(),
        absl::StrFormat(
            "Constant definition is shadowing an existing definition from %s",
            span.ToString()));
  }

  XLS_ASSIGN_OR_RETURN(bool dropped_colon, TryDropToken(TokenKind::kColon));
  TypeAnnotation* annotated_type = nullptr;
  if (dropped_colon) {
    XLS_ASSIGN_OR_RETURN(annotated_type, ParseTypeAnnotation(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kEquals));
  XLS_ASSIGN_OR_RETURN(Expr * expr, ParseExpression(bindings));
  Pos limit_pos;
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemi, /*start_pos=*/nullptr,
                                       "after constant definition",
                                       &limit_pos));
  Span span(start_pos, limit_pos);
  auto* result = module_->Make<ConstantDef>(span, name_def, annotated_type,
                                            expr, is_public);
  name_def->set_definer(result);
  bindings.Add(name_def->identifier(), result);
  return result;
}

absl::StatusOr<std::vector<ProcMember*>> Parser::CollectProcMembers(
    Bindings& bindings) {
  std::vector<ProcMember*> members;

  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  while (!peek->IsIdentifier("config") && !peek->IsIdentifier("next") &&
         !peek->IsIdentifier("init")) {
    XLS_ASSIGN_OR_RETURN(ProcMember * member, ParseProcMember(bindings));
    members.push_back(member);
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kSemi));
    XLS_ASSIGN_OR_RETURN(peek, PeekToken());
  }

  for (const ProcMember* member : members) {
    bindings.Add(member->identifier(), member->name_def());
  }

  return members;
}

absl::StatusOr<Function*> Parser::ParseProcConfig(
    Bindings& outer_bindings,
    std::vector<ParametricBinding*> parametric_bindings,
    const std::vector<ProcMember*>& proc_members, std::string_view proc_name,
    bool is_public) {
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

  XLS_ASSIGN_OR_RETURN(Block * block, ParseBlockExpression(bindings));

  Expr* final_expr;
  if (block->empty() || block->trailing_semi()) {
    // TODO(https://github.com/google/xls/issues/1124): 2023-08-31 If the block
    // is empty we make a fake tuple expression for the return value.
    final_expr = module_->Make<XlsTuple>(block->span(), std::vector<Expr*>{},
                                         /*has_trailing_comma=*/false);
    block->AddStatement(module_->Make<Statement>(final_expr));
    block->DisableTrailingSemi();
  } else {
    final_expr = std::get<Expr*>(block->statements().back()->wrapped());
  }

  XLS_VLOG(5) << "ParseProcConfig; final expr: `" << final_expr->ToString()
              << "`";

  if (dynamic_cast<XlsTuple*>(final_expr) == nullptr) {
    Span final_stmt_span =
        ToAstNode(block->statements().back()->wrapped())->GetSpan().value();
    return ParseErrorStatus(
        final_stmt_span,
        "The final expression in a Proc config must be a tuple with one "
        "element for each Proc data member.");
  }

  NameDef* name_def = module_->Make<NameDef>(
      config_tok.span(), absl::StrCat(proc_name, ".config"), nullptr);

  std::vector<TypeAnnotation*> return_elements;
  return_elements.reserve(proc_members.size());
  for (const ProcMember* member : proc_members) {
    XLS_ASSIGN_OR_RETURN(
        TypeAnnotation * member_type_clone,
        CloneNodeSansTypeDefinitions(member->type_annotation()));
    return_elements.push_back(member_type_clone);
  }
  TypeAnnotation* return_type =
      module_->Make<TupleTypeAnnotation>(config_tok.span(), return_elements);
  Function* config = module_->Make<Function>(
      block->span(), name_def, std::move(parametric_bindings),
      std::move(config_params), return_type, block, FunctionTag::kProcConfig,
      is_public);
  name_def->set_definer(config);

  return config;
}

absl::StatusOr<Function*> Parser::ParseProcNext(
    Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
    std::string_view proc_name, bool is_public) {
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

  Transaction txn(this, &inner_bindings);
  auto cleanup = absl::Cleanup([&txn]() { txn.Rollback(); });
  XLS_ASSIGN_OR_RETURN(std::vector<Param*> next_params_for_return_type,
                       ParseCommaSeq<Param*>(parse_param, TokenKind::kCParen));
  std::move(cleanup).Invoke();

  XLS_ASSIGN_OR_RETURN(std::vector<Param*> next_params,
                       ParseCommaSeq<Param*>(parse_param, TokenKind::kCParen));
  if (next_params.empty() || !TypeIsToken(next_params[0]->type_annotation())) {
    return ParseErrorStatus(
        Span(GetPos(), GetPos()),
        "The first parameter in a Proc next function must be a token.");
  }

  if (next_params.size() != 2) {
    return ParseErrorStatus(Span(GetPos(), GetPos()),
                            "A Proc next function takes two arguments: "
                            "a token and a recurrent state element.");
  }

  TypeAnnotation* return_type;
  if (next_params.size() == 2) {
    Param* state = next_params_for_return_type.at(1);
    if (HasChannelElement(state->type_annotation())) {
      return ParseErrorStatus(state->span(),
                              "Channels cannot be Proc next params.");
    }

    if (TypeIsToken(state->type_annotation())) {
      return ParseErrorStatus(
          state->span(),
          "Only the first parameter in a Proc next function may be a token.");
    }

    return_type = state->type_annotation();
  } else {
    return_type = module_->Make<TupleTypeAnnotation>(
        next_params_for_return_type[0]->span(), std::vector<TypeAnnotation*>());
  }

  XLS_ASSIGN_OR_RETURN(Block * body, ParseBlockExpression(inner_bindings));
  Span span(oparen.span().start(), GetPos());
  NameDef* name_def =
      module_->Make<NameDef>(span, absl::StrCat(proc_name, ".next"), nullptr);
  Function* next = module_->Make<Function>(
      span, name_def, std::move(parametric_bindings), next_params, return_type,
      body, FunctionTag::kProcNext, is_public);
  name_def->set_definer(next);

  return next;
}

// Implementation note: this is basically ParseFunction(), except with no return
// type.
absl::StatusOr<Function*> Parser::ParseProcInit(
    Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
    std::string_view proc_name) {
  Bindings inner_bindings(&bindings);
  XLS_ASSIGN_OR_RETURN(Token init_identifier, PopToken());
  if (!init_identifier.IsIdentifier("init")) {
    return ParseErrorStatus(init_identifier.span(),
                            absl::StrCat("Expected \"init\", got ",
                                         init_identifier.ToString(), "\"."));
  }

  NameDef* name_def = module_->Make<NameDef>(
      init_identifier.span(), absl::StrCat(proc_name, ".init"), nullptr);

  XLS_ASSIGN_OR_RETURN(Block * body, ParseBlockExpression(inner_bindings));
  Span span(init_identifier.span().start(), GetPos());
  Function* init = module_->Make<Function>(
      span, name_def, std::move(parametric_bindings), std::vector<Param*>(),
      /*return_type=*/nullptr, body, FunctionTag::kProcInit,
      /*is_public=*/false);
  name_def->set_definer(init);
  return init;
}

absl::StatusOr<Proc*> Parser::ParseProc(bool is_public,
                                        Bindings& outer_bindings) {
  XLS_ASSIGN_OR_RETURN(Token proc_token, PopKeywordOrError(Keyword::kProc));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(outer_bindings));

  Bindings bindings(&outer_bindings);
  bindings.Add(name_def->identifier(), name_def);

  XLS_ASSIGN_OR_RETURN(bool dropped_oangle, TryDropToken(TokenKind::kOAngle));
  std::vector<ParametricBinding*> parametric_bindings;
  if (dropped_oangle) {  // Parametric.
    XLS_ASSIGN_OR_RETURN(parametric_bindings,
                         ParseParametricBindings(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));

  // Create a separate Bindings object to see what's added during member
  // collection, so we can report better errors below. Members will be added to
  // "bindings" in the next call.
  Bindings memberless_bindings = bindings.Clone();
  XLS_ASSIGN_OR_RETURN(std::vector<ProcMember*> proc_members,
                       CollectProcMembers(bindings));

  Function* config = nullptr;
  Function* next = nullptr;
  Function* init = nullptr;

  auto check_not_yet_specified = [name_def](Function* f,
                                            const Token* peek) -> absl::Status {
    if (f != nullptr) {
      return ParseErrorStatus(
          peek->span(),
          absl::StrFormat("proc `%s` %s function was already specified @ %s",
                          name_def->identifier(), *peek->GetValue(),
                          peek->span().ToString()));
    }
    return absl::OkStatus();
  };

  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  while (peek->kind() != TokenKind::kCBrace) {
    if (peek->IsIdentifier("config")) {
      XLS_RETURN_IF_ERROR(check_not_yet_specified(config, peek));

      Bindings this_bindings = memberless_bindings.Clone();
      auto config_or =
          ParseProcConfig(this_bindings, parametric_bindings, proc_members,
                          name_def->identifier(), is_public);
      if (std::optional<std::string_view> bad_name =
              MaybeExtractParseNameError(config_or.status());
          bad_name.has_value() && bindings.HasName(*bad_name) &&
          !memberless_bindings.HasName(*bad_name)) {
        xabsl::StatusBuilder builder(config_or.status());
        builder << absl::StreamFormat(
            "\"%s\" is a proc member, "
            "but those cannot be referenced "
            "from within a proc config function.",
            bad_name.value());
        return builder;
      }
      XLS_RETURN_IF_ERROR(config_or.status());
      config = config_or.value();
      outer_bindings.Add(config->name_def()->identifier(), config->name_def());
      XLS_RETURN_IF_ERROR(module_->AddTop(config, MakeModuleTopCollisionError));
    } else if (peek->IsIdentifier("next")) {
      XLS_RETURN_IF_ERROR(check_not_yet_specified(next, peek));

      XLS_ASSIGN_OR_RETURN(next,
                           ParseProcNext(bindings, parametric_bindings,
                                         name_def->identifier(), is_public));
      XLS_RETURN_IF_ERROR(module_->AddTop(next, MakeModuleTopCollisionError));
      outer_bindings.Add(next->name_def()->identifier(), next->name_def());
    } else if (peek->IsIdentifier("init")) {
      XLS_RETURN_IF_ERROR(check_not_yet_specified(init, peek));

      XLS_ASSIGN_OR_RETURN(init, ParseProcInit(bindings, parametric_bindings,
                                               name_def->identifier()));
      XLS_RETURN_IF_ERROR(module_->AddTop(init, MakeModuleTopCollisionError));
      outer_bindings.Add(init->name_def()->identifier(), init->name_def());
    } else {
      return ParseErrorStatus(
          peek->span(),
          absl::StrFormat("Unexpected token in proc body: %s; want one of "
                          "'config', 'next', or 'init'",
                          peek->ToErrorString()));
    }

    XLS_ASSIGN_OR_RETURN(peek, PeekToken());
  }

  if (config == nullptr || next == nullptr || init == nullptr) {
    return ParseErrorStatus(
        Span(proc_token.span().start(), GetPos()),
        "Procs must define \"init\", \"config\" and \"next\" functions.");
  }

  // Just as with proc member decls, we need the init fn to have its own return
  // type, to avoid parent/child relationship violations.
  XLS_ASSIGN_OR_RETURN(auto* init_return_type,
                       CloneNodeSansTypeDefinitions(next->return_type()));
  init_return_type->SetParentage();
  init->set_return_type(down_cast<TypeAnnotation*>(init_return_type));
  init->SetParentage();

  XLS_ASSIGN_OR_RETURN(Token cbrace, PopTokenOrError(TokenKind::kCBrace));
  const Span span(proc_token.span().start(), cbrace.span().limit());
  ProcBody body = {
      .stmts = {},  // TODO(leary): 2024-02-09 Populate statements.
      .config = config,
      .next = next,
      .init = init,
      .members = proc_members,
  };
  auto proc =
      module_->Make<Proc>(span, name_def, std::move(parametric_bindings),
                          std::move(body), is_public);
  name_def->set_definer(proc);
  config->set_proc(proc);
  next->set_proc(proc);
  init->set_proc(proc);
  XLS_RETURN_IF_ERROR(VerifyParentage(proc));
  return proc;
}

absl::StatusOr<ChannelDecl*> Parser::ParseChannelDecl(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token channel, PopKeywordOrError(Keyword::kChannel));

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

  std::optional<Expr*> fifo_depth;
  if (fifo_depth_parametric.has_value()) {
    if (std::holds_alternative<Expr*>(fifo_depth_parametric.value())) {
      fifo_depth = std::get<Expr*>(fifo_depth_parametric.value());
    } else {
      return ParseErrorStatus(
          ExprOrTypeSpan(fifo_depth_parametric.value()),
          "Expected fifo depth to be expression, got type.");
    }
  }

  XLS_ASSIGN_OR_RETURN(bool peek_is_obrack, PeekTokenIs(TokenKind::kOBrack));
  if (peek_is_obrack) {
    XLS_ASSIGN_OR_RETURN(dims, ParseDims(bindings, &limit_pos));
  }
  return module_->Make<ChannelDecl>(Span(channel.span().start(), limit_pos),
                                    type, dims, fifo_depth);
}

absl::StatusOr<std::vector<Expr*>> Parser::ParseDims(Bindings& bindings,
                                                     Pos* limit_pos) {
  XLS_ASSIGN_OR_RETURN(Token obrack, PopTokenOrError(TokenKind::kOBrack));
  XLS_ASSIGN_OR_RETURN(Expr * first_dim,
                       ParseConditionalExpression(bindings, kNoRestrictions));
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
                         ParseConditionalExpression(bindings, kNoRestrictions));
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
      BoundNode bn,
      bindings.ResolveNodeOrError(*start_tok.GetValue(), start_tok.span()));
  if (!std::holds_alternative<Import*>(bn)) {
    return ParseErrorStatus(
        start_tok.span(),
        absl::StrFormat("Expected module for module-reference; got %s",
                        ToAstNode(bn)->ToString()));
  }
  XLS_ASSIGN_OR_RETURN(NameRef * subject, ParseNameRef(bindings, &start_tok));
  XLS_ASSIGN_OR_RETURN(Token type_name,
                       PopTokenOrError(TokenKind::kIdentifier));
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
                        start_tok.span().ToString()));
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
    // Mark this NDT as const to determine if refs to it should be ConstRefs
    // or NameRefs. Also disallow destructuring assignment for constants.
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
  if (const_) {
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
  XLS_ASSIGN_OR_RETURN(Token for_, PopKeywordOrError(Keyword::kFor));

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
  XLS_ASSIGN_OR_RETURN(Block * body, ParseBlockExpression(for_bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(
      TokenKind::kOParen, &for_,
      "Need an initial accumulator value to start the for loop."));

  // We must be sure to use the outer bindings when parsing the init
  // expression, since the for loop bindings haven't happened yet (no loop
  // trips have iterated when the init value is evaluated).
  XLS_ASSIGN_OR_RETURN(Expr * init, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
  return module_->Make<For>(Span(for_.span().limit(), GetPos()), names, type,
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
  XLS_ASSIGN_OR_RETURN(Block * body, ParseBlockExpression(for_bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  XLS_ASSIGN_OR_RETURN(Expr * init, ParseExpression(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));

  return module_->Make<UnrollFor>(Span(unroll_for.span().limit(), GetPos()),
                                  names, types, iterable, body, init);
}

absl::StatusOr<EnumDef*> Parser::ParseEnumDef(bool is_public,
                                              Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Token enum_tok, PopKeywordOrError(Keyword::kEnum));
  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(bindings));
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
    return EnumMember{name_def, expr};
  };

  XLS_ASSIGN_OR_RETURN(
      std::vector<EnumMember> entries,
      ParseCommaSeq<EnumMember>(parse_enum_entry, TokenKind::kCBrace));
  auto* enum_def =
      module_->Make<EnumDef>(Span(enum_tok.span().start(), GetPos()), name_def,
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
    const Span& span, TypeRef* type_ref, absl::Span<Expr* const> dims,
    std::vector<ExprOrType> parametrics) {
  TypeAnnotation* elem_type = module_->Make<TypeRefTypeAnnotation>(
      span, type_ref, std::move(parametrics));
  for (Expr* dim : dims) {
    elem_type = module_->Make<ArrayTypeAnnotation>(span, elem_type, dim);
  }
  return elem_type;
}

absl::StatusOr<std::variant<NameDef*, WildcardPattern*>>
Parser::ParseNameDefOrWildcard(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(std::optional<Token> tok, TryPopIdentifierToken("_"));
  if (tok) {
    return module_->Make<WildcardPattern>(tok->span());
  }
  return ParseNameDef(bindings);
}

absl::StatusOr<Param*> Parser::ParseParam(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(NameDef * name, ParseNameDef(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                                       "Expect type annotation on parameters"));
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
  auto* param = module_->Make<Param>(name, type);
  name->set_definer(param);
  return param;
}

absl::StatusOr<ProcMember*> Parser::ParseProcMember(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(NameDef * name, ParseNameDef(bindings));
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                                       "Expect type annotation on parameters"));
  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
  auto* member = module_->Make<ProcMember>(name, type);
  name->set_definer(member);
  return member;
}

absl::StatusOr<std::vector<Param*>> Parser::ParseParams(Bindings& bindings) {
  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
  auto sub_production = [&] { return ParseParam(bindings); };
  return ParseCommaSeq<Param*>(sub_production, TokenKind::kCParen);
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

  return ParseErrorStatus(
      peek.span(),
      absl::StrFormat("Expected number; got %s @ %s",
                      TokenKindToString(peek.kind()), peek.span().ToString()));
}

absl::StatusOr<StructDef*> Parser::ParseStruct(bool is_public,
                                               Bindings& bindings) {
  const Pos start_pos = GetPos();
  XLS_RETURN_IF_ERROR(DropKeywordOrError(Keyword::kStruct));

  XLS_ASSIGN_OR_RETURN(NameDef * name_def, ParseNameDef(bindings));

  XLS_ASSIGN_OR_RETURN(bool dropped_oangle, TryDropToken(TokenKind::kOAngle));
  std::vector<ParametricBinding*> parametric_bindings;
  if (dropped_oangle) {
    XLS_ASSIGN_OR_RETURN(parametric_bindings,
                         ParseParametricBindings(bindings));
  }

  XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));

  auto parse_struct_member = [this,
                              &bindings]() -> absl::StatusOr<StructMember> {
    Span name_span;
    XLS_ASSIGN_OR_RETURN(std::string name, PopIdentifierOrError(&name_span));
    XLS_RETURN_IF_ERROR(
        DropTokenOrError(TokenKind::kColon, /*start=*/nullptr,
                         "Expect type annotation on struct field"));
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
    return StructMember{name_span, name, type};
  };

  XLS_ASSIGN_OR_RETURN(
      std::vector<StructMember> members,
      ParseCommaSeq<StructMember>(parse_struct_member, TokenKind::kCBrace));

  Span span(start_pos, GetPos());
  auto* struct_def =
      module_->Make<StructDef>(span, name_def, std::move(parametric_bindings),
                               std::move(members), is_public);
  bindings.Add(name_def->identifier(), struct_def);
  return struct_def;
}

absl::StatusOr<NameDefTree*> Parser::ParseTuplePattern(const Pos& start_pos,
                                                       Bindings& bindings) {
  std::vector<NameDefTree*> members;
  bool must_end = false;
  while (true) {
    XLS_ASSIGN_OR_RETURN(bool dropped_cparen, TryDropToken(TokenKind::kCParen));
    if (dropped_cparen) {
      break;
    }
    if (must_end) {
      XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCParen));
      break;
    }
    XLS_ASSIGN_OR_RETURN(NameDefTree * pattern, ParsePattern(bindings));
    members.push_back(pattern);
    XLS_ASSIGN_OR_RETURN(bool dropped_comma, TryDropToken(TokenKind::kComma));
    must_end = !dropped_comma;
  }
  Span span(start_pos, GetPos());
  return module_->Make<NameDefTree>(span, std::move(members));
}

absl::StatusOr<Block*> Parser::ParseBlockExpression(Bindings& bindings) {
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
      XLS_ASSIGN_OR_RETURN(TypeAlias * alias,
                           ParseTypeAlias(/*is_public=*/false, block_bindings));
      stmts.push_back(module_->Make<Statement>(alias));
      last_expr_had_trailing_semi = true;
    } else if (peek->IsKeyword(Keyword::kLet) ||
               peek->IsKeyword(Keyword::kConst)) {
      XLS_ASSIGN_OR_RETURN(Let * let, ParseLet(block_bindings));
      stmts.push_back(module_->Make<Statement>(let));
      last_expr_had_trailing_semi = true;
    } else if (peek->IsIdentifier("const_assert!")) {
      XLS_ASSIGN_OR_RETURN(ConstAssert * const_assert,
                           ParseConstAssert(block_bindings));
      stmts.push_back(module_->Make<Statement>(const_assert));
      last_expr_had_trailing_semi = true;
    } else {
      XLS_VLOG(5) << "ParseBlockExpression; parsing expression with bindings: ["
                  << absl::StrJoin(block_bindings.GetLocalBindings(), ", ")
                  << "]";
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
  return module_->Make<Block>(Span(start_pos, GetPos()), stmts,
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
    XLS_ASSIGN_OR_RETURN(TypeAnnotation * type, ParseTypeAnnotation(bindings));
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
  return ParseCommaSeq<ParametricBinding*>(parse_parametric_binding,
                                           TokenKind::kCAngle);
}

absl::StatusOr<ExprOrType> Parser::ParseParametricArg(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(const Token* peek, PeekToken());
  if (peek->kind() == TokenKind::kOBrace) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOBrace));
    // Conditional expressions are the first below the let/for/while set.
    XLS_ASSIGN_OR_RETURN(
        Expr * expr, ParseConditionalExpression(bindings, kNoRestrictions));
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kCBrace));

    return expr;
  }

  // We permit bare numbers as parametrics for convenience.
  if (peek->kind() == TokenKind::kNumber || peek->IsKeyword(Keyword::kTrue) ||
      peek->IsKeyword(Keyword::kFalse)) {
    return TokenToNumber(PopTokenOrDie());
  }

  if (peek->kind() == TokenKind::kIdentifier) {
    XLS_ASSIGN_OR_RETURN(auto nocr, ParseNameOrColonRef(bindings));
    return ToExprNode(nocr);
  }

  XLS_ASSIGN_OR_RETURN(TypeAnnotation * type_annotation,
                       ParseTypeAnnotation(bindings));

  {
    XLS_ASSIGN_OR_RETURN(const Token* peek2, PeekToken());
    if (peek2->kind() == TokenKind::kColon) {
      return ParseCast(bindings, type_annotation);
    }
  }

  return type_annotation;
}

absl::StatusOr<std::vector<ExprOrType>> Parser::ParseParametrics(
    Bindings& bindings) {
  XLS_VLOG(5) << "ParseParametrics @ " << GetPos();
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
    Bindings& bindings, const Span& directive_span) {
  XLS_ASSIGN_OR_RETURN(Function * f,
                       ParseFunctionInternal(/*is_public=*/false, bindings));
  XLS_RET_CHECK(f != nullptr);
  if (std::optional<ModuleMember*> member =
          module_->FindMemberWithName(f->identifier())) {
    return ParseErrorStatus(
        f->name_def()->span(),
        absl::StrFormat(
            "Test function '%s' has same name as module member @ %s",
            f->identifier(), ToAstNode(**member)->GetSpan()->ToString()));
  }
  Span tf_span(directive_span.start(), f->span().limit());
  TestFunction* tf = module_->Make<TestFunction>(tf_span, *f);
  tf->SetParentage();  // Ensure the function has its parent marked.
  return tf;
}

absl::StatusOr<TestProc*> Parser::ParseTestProc(Bindings& bindings) {
  XLS_ASSIGN_OR_RETURN(Proc * p, ParseProc(/*is_public=*/false, bindings));
  if (std::optional<ModuleMember*> member =
          module_->FindMemberWithName(p->identifier())) {
    return ParseErrorStatus(
        p->span(),
        absl::StrFormat("Test proc '%s' has same name as module member @ %s",
                        p->identifier(),
                        ToAstNode(**member)->GetSpan()->ToString()));
  }

  // Verify no state or config args
  return module_->Make<TestProc>(p);
}

const Span& GetSpan(const std::variant<NameDef*, WildcardPattern*>& v) {
  if (std::holds_alternative<NameDef*>(v)) {
    return std::get<NameDef*>(v)->span();
  }
  return std::get<WildcardPattern*>(v)->span();
}

}  // namespace xls::dslx
