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

#ifndef XLS_DSLX_PARSER_H_
#define XLS_DSLX_PARSER_H_

#include "absl/status/statusor.h"
#include "xls/dslx/ast.h"
#include "xls/dslx/bindings.h"
#include "xls/dslx/token_parser.h"

namespace xls::dslx {

template <typename T, typename... Types>
inline T TryGet(const std::variant<Types...>& v) {
  if (std::holds_alternative<T>(v)) {
    return std::get<T>(v);
  }
  return nullptr;
}

class Parser : public TokenParser {
 public:
  Parser(std::string module_name, Scanner* scanner)
      : TokenParser(scanner), module_(new Module(std::move(module_name))) {}

  absl::StatusOr<Function*> ParseFunction(
      bool is_public, Bindings* bindings,
      absl::flat_hash_map<std::string, Function*>* name_to_fn = nullptr);

  absl::StatusOr<Proc*> ParseProc(bool is_public, Bindings* bindings);

  absl::StatusOr<std::unique_ptr<Module>> ParseModule(
      Bindings* bindings = nullptr);

  // Parses an expression out of the token stream.
  absl::StatusOr<Expr*> ParseExpression(Bindings* bindings);

  // TOOD(leary): 2020-09-11 Would be better to rename this to "type alias".
  absl::StatusOr<TypeDef*> ParseTypeDefinition(bool is_public,
                                               Bindings* bindings);

 private:
  friend class ParserTest;

  // Simple helper class to wrap the operations necessary to evaluate [parser]
  // productions as transactions - with "Commit" or "Rollback" operations.
  class Transaction {
   public:
    Transaction(Parser* parser, Bindings* bindings)
        : parser_(parser),
          checkpoint_(parser->SaveScannerCheckpoint()),
          parent_bindings_(bindings),
          child_bindings_(bindings),
          completed_(false) {}
    ~Transaction() {
      XLS_CHECK(completed_) << "Uncompleted state transaction!";
    }

    // Call on successful production: saves the changes from this
    // transaction to the parent bindings.
    void Commit() {
      XLS_CHECK(!completed_) << "Doubly-completed transaction!";
      if (parent_bindings_ != nullptr) {
        parent_bindings_->ConsumeChild(&child_bindings_);
      }
      completed_ = true;
    }

    // Convenience call to cancel a cleanup (usually one that invokes
    // Rollback()) on commit.
    template <typename CleanupT>
    void CommitAndCancelCleanup(CleanupT* cleanup) {
      Commit();
      if (cleanup) {
        std::move(*cleanup).Cancel();
      }
    }

    // Call on failed production: un-does changes to bindings and scanner state.
    void Rollback() {
      XLS_CHECK(!completed_) << "Doubly-completed transaction!";
      parser_->RestoreScannerCheckpoint(checkpoint_);
      child_bindings_ = Bindings(parent_bindings_);
      completed_ = true;
    }

    Bindings* bindings() { return &child_bindings_; }

   private:
    Parser* parser_;
    ScannerCheckpoint checkpoint_;
    Bindings* parent_bindings_;
    Bindings child_bindings_;
    // True if this transaction has been either committed or rolled back.
    bool completed_;
  };

  // Helper that parses a comma-delimited sequence of grammatical productions.
  //
  // Expects the caller to have popped the "initiator" token; however, this
  // (callee) pops the terminator token so the caller does not need to.
  //
  // Permits a trailing comma.
  //
  // Args:
  //  fparse: Parses the grammatical production (i.e. the thing after each
  //    comma).
  //  terminator: Token that terminates the sequence; e.g. ')' or ']' or similar
  //    (may be a keyword).
  template <typename T>
  absl::StatusOr<std::vector<T>> ParseCommaSeq(
      const std::function<absl::StatusOr<T>()>& fparse,
      std::variant<TokenKind, Keyword> terminator,
      Pos* terminator_limit = nullptr) {
    auto try_pop_terminator = [&]() -> absl::StatusOr<bool> {
      if (std::holds_alternative<TokenKind>(terminator)) {
        return TryDropToken(std::get<TokenKind>(terminator), terminator_limit);
      }
      return TryDropKeyword(std::get<Keyword>(terminator), terminator_limit);
    };
    auto drop_terminator_or_error = [&]() -> absl::Status {
      if (std::holds_alternative<TokenKind>(terminator)) {
        return DropTokenOrError(std::get<TokenKind>(terminator),
                                /*start=*/nullptr, /*context=*/"",
                                /*limit_pos=*/terminator_limit);
      }
      return DropKeywordOrError(std::get<Keyword>(terminator),
                                terminator_limit);
    };

    std::vector<T> parsed;
    bool must_end = false;
    while (true) {
      XLS_ASSIGN_OR_RETURN(bool popped_terminator, try_pop_terminator());
      if (popped_terminator) {
        break;
      }
      if (must_end) {
        XLS_RETURN_IF_ERROR(drop_terminator_or_error());
        break;
      }
      XLS_ASSIGN_OR_RETURN(T elem, fparse());
      parsed.push_back(elem);
      XLS_ASSIGN_OR_RETURN(bool dropped_comma, TryDropToken(TokenKind::kComma));
      must_end = !dropped_comma;
    }
    return parsed;
  }

  absl::StatusOr<Expr*> ParseDim(Bindings* bindings);

  // Parses dimension on a type; e.g. `u32[3]` => `(3,)`; `uN[2][3]` => `(3,
  // 2)`.
  absl::StatusOr<std::vector<Expr*>> ParseDims(Bindings* bindings,
                                               Pos* limit_pos = nullptr);

  absl::StatusOr<TypeRef*> ParseModTypeRef(Bindings* bindings,
                                           const Token& start_tok);

  absl::StatusOr<StructRef> ResolveStruct(Bindings* bindings,
                                          TypeAnnotation* type);

  // Parses an AST construct that refers to a type; e.g. a name or a colon-ref.
  absl::StatusOr<TypeRef*> ParseTypeRef(Bindings* bindings, const Token& tok);

  absl::StatusOr<TypeAnnotation*> ParseTypeAnnotation(Bindings* bindings);

  absl::StatusOr<NameRef*> ParseNameRef(Bindings* bindings,
                                        const Token* tok = nullptr);

  // Precondition: token cursor should be over a double colon '::' token.
  absl::StatusOr<ColonRef*> ParseColonRef(Bindings* bindings,
                                          ColonRef::Subject subject);

  absl::StatusOr<Expr*> ParseCastOrEnumRefOrStructInstance(Bindings* bindings);

  absl::StatusOr<Expr*> ParseStructInstance(Bindings* bindings,
                                            TypeAnnotation* type = nullptr);

  absl::StatusOr<Expr*> ParseCastOrStructInstance(Bindings* bindings);

  absl::StatusOr<std::variant<NameRef*, ColonRef*>> ParseNameOrColonRef(
      Bindings* bindings, std::string_view context = "");

  absl::StatusOr<NameDef*> ParseNameDef(Bindings* bindings);

  absl::StatusOr<std::variant<NameDef*, WildcardPattern*>>
  ParseNameDefOrWildcard(Bindings* bindings);

  // TryOrRollback wraps straighforward transactions: call a function, and
  // rollback or commit depending on the result. Just a shorthand to reduce
  // transaction code footprint.
  template <typename ReturnT>
  absl::StatusOr<ReturnT> TryOrRollback(
      Bindings* bindings,
      std::function<absl::StatusOr<ReturnT>(Bindings* bindings)> fn) {
    Transaction txn(this, bindings);
    auto status_or_result = fn(bindings);
    if (status_or_result.ok()) {
      txn.Commit();
    } else {
      txn.Rollback();
    }
    return status_or_result;
  }

  // Parses tree of name defs and returns it.
  //
  // For example, the left hand side of:
  //
  //  let (a, (b, (c)), d) = ...
  //
  // This is used for tuple-like (sometimes known as "destructing") let binding.
  absl::StatusOr<NameDefTree*> ParseNameDefTree(Bindings* bindings);

  absl::StatusOr<Number*> TokenToNumber(const Token& tok);
  absl::StatusOr<NameDef*> TokenToNameDef(const Token& tok) {
    return module_->Make<NameDef>(tok.span(), *tok.GetValue(), nullptr);
  }
  absl::StatusOr<BuiltinType> TokenToBuiltinType(const Token& tok);
  absl::StatusOr<TypeAnnotation*> MakeBuiltinTypeAnnotation(
      const Span& span, const Token& tok, absl::Span<Expr* const> dims);
  absl::StatusOr<TypeAnnotation*> MakeTypeRefTypeAnnotation(
      const Span& span, TypeRef* type_ref, std::vector<Expr*> dims,
      std::vector<Expr*> parametrics);

  // Returns a parsed number (literal number) expression.
  absl::StatusOr<Number*> ParseNumber(Bindings* bindings);

  absl::StatusOr<std::variant<Number*, NameRef*>> ParseNumOrConstRef(
      Bindings* bindings);

  absl::StatusOr<Let*> ParseLet(Bindings* bindings);

  // Parses the remainder of a tuple expression.
  //
  // We can't tell until we've parsed the first expression whether we're parsing
  // a parenthesized expression; e.g. `(x)` or a tuple expression `(x, y)` -- as
  // a result we use this helper routine once we discover we're parsing a tuple
  // instead of a parenthesized expression, which is why "first" is passed from
  // the caller.
  //
  // Args:
  //  start_pos: The position of the '(' token that started this tuple.
  //  first: The parse expression in the tuple as already parsed by the caller.
  //  bindings: Bindings to use in the parsing of the tuple expression.
  absl::StatusOr<XlsTuple*> ParseTupleRemainder(const Pos& start_pos,
                                                Expr* first,
                                                Bindings* bindings);

  absl::StatusOr<Array*> ParseArray(Bindings* bindings);

  absl::StatusOr<Expr*> ParseCast(Bindings* bindings,
                                  TypeAnnotation* type = nullptr);

  // Parses a term as a component of an expression and returns it.
  //
  // Terms are more atomic than arithmetic expressions.
  absl::StatusOr<Expr*> ParseTerm(Bindings* bindings);

  // Parses a slicing index expression.
  absl::StatusOr<Index*> ParseBitSlice(const Pos& start_pos, Expr* lhs,
                                       Bindings* bindings,
                                       Expr* start = nullptr);

  // Parses a chain of binary operations at a given precedence level.
  //
  // For example, a sequence like "x + y + z" is left associative, so we form a
  // left-leaning AST like:
  //
  //    add(add(x, y), z)
  //
  // Generally a grammar production will join together two stronger production
  // rules; e.g.
  //
  //    WEAK_ARITHMETIC_EXPR ::=
  //      STRONG_ARITHMETIC_EXPR [+-] STRONG_ARITHMETIC_EXPR
  //
  // So that expressions like `a*b + c*d` work as expected, so the
  // sub_production gives the more tightly binding production for this to call.
  // After we call it for the "left hand side" we see if the token is in the
  // target_token set (e.g. '+' or '-' in the example above), and if so, parse
  // the "right hand side" to create a binary operation. If not, we simply
  // return the result of the "left hand side" production (since we don't see
  // the target token that indicates the kind of expression we're interested
  // in).
  absl::StatusOr<Expr*> ParseBinopChain(
      const std::function<absl::StatusOr<Expr*>()>& sub_production,
      std::variant<absl::Span<TokenKind const>, absl::Span<Keyword const>>
          target_tokens);

  absl::StatusOr<Expr*> ParseCastAsExpression(Bindings* bindings);

  template <typename T>
  std::function<absl::StatusOr<T>()> BindFront(
      absl::StatusOr<T> (Parser::*f)(Bindings*), Bindings* bindings) {
    return [this, bindings, f] { return (this->*f)(bindings); };
  }

  static constexpr std::initializer_list<TokenKind> kStrongArithmeticKinds = {
      TokenKind::kStar, TokenKind::kSlash, TokenKind::kPercent};
  static constexpr std::initializer_list<TokenKind> kWeakArithmeticKinds = {
      TokenKind::kPlus, TokenKind::kDoublePlus, TokenKind::kMinus};
  static constexpr std::initializer_list<TokenKind> kBitwiseKinds = {
      TokenKind::kDoubleOAngle,
      TokenKind::kDoubleCAngle,
  };
  static constexpr std::initializer_list<TokenKind> kComparisonKinds = {
      TokenKind::kDoubleEquals, TokenKind::kBangEquals,
      TokenKind::kCAngle,       TokenKind::kCAngleEquals,
      TokenKind::kOAngle,       TokenKind::kOAngleEquals};

  absl::StatusOr<Expr*> ParseStrongArithmeticExpression(Bindings* bindings) {
    return ParseBinopChain(BindFront(&Parser::ParseCastAsExpression, bindings),
                           kStrongArithmeticKinds);
  }

  absl::StatusOr<Expr*> ParseWeakArithmeticExpression(Bindings* bindings) {
    return ParseBinopChain(
        BindFront(&Parser::ParseStrongArithmeticExpression, bindings),
        kWeakArithmeticKinds);
  }

  absl::StatusOr<Expr*> ParseBitwiseExpression(Bindings* bindings) {
    return ParseBinopChain(
        BindFront(&Parser::ParseWeakArithmeticExpression, bindings),
        kBitwiseKinds);
  }

  absl::StatusOr<Expr*> ParseAndExpression(Bindings* bindings) {
    std::initializer_list<TokenKind> amp = {TokenKind::kAmpersand};
    return ParseBinopChain(BindFront(&Parser::ParseBitwiseExpression, bindings),
                           amp);
  }

  absl::StatusOr<Expr*> ParseXorExpression(Bindings* bindings) {
    std::initializer_list<TokenKind> hat = {TokenKind::kHat};
    return ParseBinopChain(BindFront(&Parser::ParseAndExpression, bindings),
                           hat);
  }

  absl::StatusOr<Expr*> ParseOrExpression(Bindings* bindings) {
    std::initializer_list<TokenKind> bar = {TokenKind::kBar};
    return ParseBinopChain(BindFront(&Parser::ParseXorExpression, bindings),
                           bar);
  }

  absl::StatusOr<Expr*> ParseComparisonExpression(Bindings* bindings);

  absl::StatusOr<Expr*> ParseLogicalAndExpression(Bindings* bindings) {
    std::initializer_list<TokenKind> kinds = {TokenKind::kDoubleAmpersand};
    return ParseBinopChain(
        BindFront(&Parser::ParseComparisonExpression, bindings), kinds);
  }

  absl::StatusOr<Expr*> ParseLogicalOrExpression(Bindings* bindings) {
    std::initializer_list<TokenKind> kinds = {TokenKind::kDoubleBar};
    return ParseBinopChain(
        BindFront(&Parser::ParseLogicalAndExpression, bindings), kinds);
  }

  absl::StatusOr<Expr*> ParseRangeExpression(Bindings* bindings);

  // Parses a ternary expression or expression of higher precedence.
  //
  // Example:
  //
  //    foo if bar else baz
  //
  // TODO(leary): 2020-09-12 Switch to Rust-style block expressions.
  absl::StatusOr<Expr*> ParseTernaryExpression(Bindings* bindings);

  absl::StatusOr<Param*> ParseParam(Bindings* bindings);

  // Parses a sequence of parameters, starting with cursor over '(', returns
  // after ')' is consumed.
  //
  // Permits trailing commas.
  absl::StatusOr<std::vector<Param*>> ParseParams(Bindings* bindings) {
    XLS_RETURN_IF_ERROR(DropTokenOrError(TokenKind::kOParen));
    return ParseCommaSeq<Param*>(BindFront(&Parser::ParseParam, bindings),
                                 TokenKind::kCParen);
  }

  absl::StatusOr<NameDefTree*> ParseTuplePattern(const Pos& start_pos,
                                                 Bindings* bindings);

  // Returns a parsed pattern; e.g. one that would guard a match arm.
  //
  //  Pattern ::= TuplePattern
  //            | WildcardPattern
  //            | ColonRef
  //            | NameDef
  //            | NameRef
  //            | ConstRef
  //            | Number
  absl::StatusOr<NameDefTree*> ParsePattern(Bindings* bindings);

  // Parses a match expression.
  absl::StatusOr<Match*> ParseMatch(Bindings* bindings);

  // Parses a channel declaration.
  absl::StatusOr<ChannelDecl*> ParseChannelDecl(Bindings* bindings);

  // Parses a for loop construct; e.g.
  //
  //  for (i, accum) in range(3) {
  //    accum + i
  //  }(0)
  //
  // The init value is passed to the loop and the body updates the value;
  // ultimately the loop terminates and the final accum value is returned.
  absl::StatusOr<For*> ParseFor(Bindings* bindings);

  // Parses an "unroll for" macro-like construct, e.g.
  //
  // unroll_for! i in range(u32:, u32:4) {
  //   spawn my_proc(...)(...);
  // }
  absl::StatusOr<UnrollFor*> ParseUnrollFor(Bindings* bindings);

  // Parses an enum definition; e.g.
  //
  //  enum Foo : u2 {
  //    A = 0,
  //    B = 1,
  //  }
  absl::StatusOr<EnumDef*> ParseEnumDef(bool is_public, Bindings* bindings);

  absl::StatusOr<StructDef*> ParseStruct(bool is_public, Bindings* bindings);

  absl::StatusOr<Block*> ParseBlockExpression(Bindings* bindings);

  absl::StatusOr<Expr*> ParseParenthesizedExpr(Bindings* bindings);

  // Parses parametric bindings that lead a function.
  //
  // For example:
  //
  //  fn [X: u32, Y: u32 = X+X] f(x: bits[X]) { ... }
  //      ^------------------^
  //
  // Note that some bindings have expressions and other do not, because they
  // take on a value presented by the type of a formal parameter.
  absl::StatusOr<std::vector<ParametricBinding*>> ParseParametricBindings(
      Bindings* bindings);

  // Parses parametric dims that follow a struct type annotation.
  //
  // For example:
  //
  //    x: ParametricStruct<u32:4, N as u64>
  //                       ^---------------^
  absl::StatusOr<std::vector<Expr*>> ParseParametrics(Bindings* bindings);

  // Parses a function out of the token stream.
  absl::StatusOr<Function*> ParseFunctionInternal(bool is_public,
                                                  Bindings* outer_bindings);

  // Parses an import statement into an Import AST node.
  absl::StatusOr<Import*> ParseImport(Bindings* bindings);

  // Returns TestFunction AST node by parsing new-style unit test construct.
  absl::StatusOr<TestFunction*> ParseTestFunction(
      Bindings* bindings, const Span& directive_span);

  absl::StatusOr<TestProc*> ParseTestProc(Bindings* bindings);

  // Parses a constant definition (e.g. at the top level of a module). Token
  // cursor should be over the `const` keyword.
  absl::StatusOr<ConstantDef*> ParseConstantDef(bool is_public,
                                                Bindings* bindings);

  absl::StatusOr<QuickCheck*> ParseQuickCheck(
      absl::flat_hash_map<std::string, Function*>* name_to_fn,
      Bindings* bindings, const Span& directive_span);

  // Parses DSLX attributes, analogous to Rust's attributes.
  absl::StatusOr<
      std::variant<TestFunction*, TestProc*, QuickCheck*, std::nullptr_t>>
  ParseAttribute(absl::flat_hash_map<std::string, Function*>* name_to_fn,
                 Bindings* bindings);

  // Parses a "spawn" statement, which creates & initializes a proc.
  absl::StatusOr<Spawn*> ParseSpawn(Bindings* bindings);

  // Helper function that chooses between building a FormatMacro or an
  // Invocation based on the callee.
  absl::StatusOr<Expr*> BuildMacroOrInvocation(
      Span span, Bindings* bindings, Expr* callee, std::vector<Expr*> args,
      std::vector<Expr*> parametrics = std::vector<Expr*>({}));

  // Traverses a Proc declaration to collect all the member data elements
  // present therein - in other words, it collects everything but the "config"
  // and "next" elements.
  absl::StatusOr<std::vector<Param*>> CollectProcMembers(Bindings* bindings);

  // Parses Proc config, next, and init functions, respectively.
  absl::StatusOr<Function*> ParseProcConfig(
      Bindings* bindings,
      const std::vector<ParametricBinding*>& parametric_bindings,
      const std::vector<Param*>& proc_members, std::string_view proc_name,
      bool is_public);
  absl::StatusOr<Function*> ParseProcNext(
      Bindings* bindings,
      const std::vector<ParametricBinding*>& parametric_bindings,
      std::string_view proc_name, bool is_public);
  absl::StatusOr<Function*> ParseProcInit(
      Bindings* bindings,
      const std::vector<ParametricBinding*>& parametric_bindings,
      std::string_view proc_name);

  // We use this function instead of CloneAst because we don't want to clone the
  // underlying StructDefs, EnumDefs, etc., or else ConcreteType comparisons
  // will fail (StructType::operator==).
  // TODO(rspringer): Should we change that operator to use structural instead
  // of pointer equality?
  absl::StatusOr<TypeAnnotation*> CloneReturnType(TypeAnnotation* input);

  std::unique_ptr<Module> module_;

  // `Let` nodes are created _after_ those that use their namedefs (due to the
  // chaining of the `body` member variable. We need to know, though, if a
  // reference to such an NDT (or element thereof) is to a constant or not, so
  // we can emit either a ConstRef or NameRef. This set holds those NDTs known
  // to be constant for that purpose.
  absl::flat_hash_set<NameDefTree*> const_ndts_;
};

const Span& GetSpan(const std::variant<NameDef*, WildcardPattern*>& v);

}  // namespace xls::dslx

#endif  // XLS_DSLX_PARSER_H_
