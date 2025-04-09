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

#ifndef XLS_DSLX_FRONTEND_PARSER_H_
#define XLS_DSLX_FRONTEND_PARSER_H_

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

#include "absl/base/attributes.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strong_int.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/bindings.h"
#include "xls/dslx/frontend/module.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/frontend/scanner.h"
#include "xls/dslx/frontend/token.h"
#include "xls/dslx/frontend/token_parser.h"

namespace xls::dslx {

template <typename T, typename... Types>
inline T TryGet(const std::variant<Types...>& v) {
  if (std::holds_alternative<T>(v)) {
    return std::get<T>(v);
  }
  return nullptr;
}

// As a convenient way to reuse grammar rules with modifications, we allow
// "restriction" flags to be passed to expression productions.
//
// Implementation note: these must each be a power of two so they can be used in
// an integer flag set.
//
// This is useful, for example, when we want to parse a conditional test
// expression, but we don't want to allow struct literals in that position
// because it would make the grammar ambiguous (because a '{' can belong to
// either a conditional body or a struct instance); we can call
// ParseExpression() with the "no struct literal" restriction to help us wrangle
// this sort of case.
enum class ExprRestriction : uint8_t {
  kNone = 0,
  kNoStructLiteral = 1,
};

// Flag set of ExprRestriction values.
XLS_DEFINE_STRONG_INT_TYPE(ExprRestrictions, uint32_t);

constexpr ExprRestrictions kNoRestrictions = ExprRestrictions(0);

// forward declaration
class Parser;

// RAII guard used to ensure the expression nesting depth does not get
// unreasonably deep (which can cause stack overflows and "erroneously" flag
// fuzzing issues in that dimension).
class ABSL_MUST_USE_RESULT ExpressionDepthGuard final {
 public:
  explicit ExpressionDepthGuard(Parser* parser) : parser_(parser) {}
  ~ExpressionDepthGuard();

  // move-only type, and we use the parser pointer to track which instance is
  // performing the side effect in the destructor if the original is moved
  ExpressionDepthGuard(ExpressionDepthGuard&& other) : parser_(other.parser_) {
    other.parser_ = nullptr;
  }
  ExpressionDepthGuard& operator=(ExpressionDepthGuard&& other) {
    parser_ = other.parser_;
    other.parser_ = nullptr;
    return *this;
  }

  ExpressionDepthGuard(const ExpressionDepthGuard&) = delete;
  ExpressionDepthGuard& operator=(const ExpressionDepthGuard&) = delete;

 private:
  Parser* parser_;
};

using TypeRefOrAnnotation = std::variant<TypeRef*, TypeAnnotation*>;

class Parser : public TokenParser {
 public:
  Parser(std::string module_name, Scanner* scanner, bool parse_fn_stubs = false)
      : TokenParser(scanner),
        module_(new Module(std::move(module_name), scanner->filename(),
                           scanner->file_table())),
        parse_fn_stubs_(parse_fn_stubs) {}

  const FileTable& file_table() const { return scanner().file_table(); }
  FileTable& file_table() { return scanner().file_table(); }

  absl::StatusOr<Function*> ParseFunction(
      const Pos& start_pos, bool is_public, Bindings& bindings,
      absl::flat_hash_map<std::string, Function*>* name_to_fn = nullptr);

  absl::StatusOr<Lambda*> ParseLambda(Bindings& bindings);

  absl::StatusOr<ModuleMember> ParseProc(const Pos& start_pos, bool is_public,
                                         Bindings& bindings);

  absl::StatusOr<std::unique_ptr<Module>> ParseModule(
      Bindings* bindings = nullptr);

  // Parses an expression out of the token stream.
  absl::StatusOr<Expr*> ParseExpression(
      Bindings& bindings, ExprRestrictions restrictions = kNoRestrictions);

  // Parses a block expression of the form:
  //
  //    "{" seq(Stmt ";") [;] "}"
  //
  // The optional trailing semicolon presence is noted on the Block AST node
  // returned.
  //
  // Takes an optional `prologue` of statements to prepend to the block.
  absl::StatusOr<StatementBlock*> ParseBlockExpression(Bindings& bindings);

  absl::StatusOr<TypeAlias*> ParseTypeAlias(const Pos& start_pos,
                                            bool is_public, Bindings& bindings);

  // Args:
  //  bindings: bindings to be used in parsing the assert expression.
  //  identifier: [optional] token that contains the `const_assert!` identifier
  //    -- if this is not given it is assumed that it should be popped from the
  //    token steam.
  absl::StatusOr<ConstAssert*> ParseConstAssert(
      Bindings& bindings, const Token* ABSL_NULLABLE identifier = nullptr);

  Module& module() { return *module_; }

  absl::Status ParseErrorStatus(const Span& span,
                                std::string_view message) const;

 private:
  friend class ParserTest;
  friend class ExpressionDepthGuard;

  // Simple helper class to wrap the operations necessary to evaluate [parser]
  // productions as transactions - with "Commit" or "Rollback" operations.
  class Transaction {
   public:
    Transaction(Parser* parser, Bindings* bindings)
        : parser_(parser),
          checkpoint_(parser->SaveScannerCheckpoint()),
          parent_bindings_(bindings),
          child_bindings_(bindings) {}

    ~Transaction() { CHECK(completed_) << "Uncompleted state transaction!"; }

    // Call on successful production: saves the changes from this
    // transaction to the parent bindings.
    void Commit() {
      CHECK(!completed_) << "Doubly-completed transaction!";
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
      CHECK(!completed_) << "Doubly-completed transaction!";
      parser_->RestoreScannerCheckpoint(checkpoint_);
      child_bindings_ = Bindings(parent_bindings_);
      completed_ = true;
    }

    Bindings* bindings() { return &child_bindings_; }
    bool completed() const { return completed_; }

   private:
    Parser* parser_;
    ScannerCheckpoint checkpoint_;
    Bindings* parent_bindings_;
    Bindings child_bindings_;
    // True if this transaction has been either committed or rolled back.
    bool completed_ = false;
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
      Pos* terminator_limit = nullptr, bool* saw_trailing_comma = nullptr) {
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
    bool saw_trailing_comma_local = false;
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
      saw_trailing_comma_local = dropped_comma;
    }
    if (saw_trailing_comma != nullptr) {  // Populate outparam.
      *saw_trailing_comma = saw_trailing_comma_local;
    }
    return parsed;
  }

  // Parses dimension on a type; e.g. `u32[3]` => `(3,)`; `uN[2][3]` => `(3,
  // 2)`.
  absl::StatusOr<std::vector<Expr*>> ParseDims(Bindings& bindings,
                                               Pos* limit_pos = nullptr);

  absl::StatusOr<TypeRef*> ParseModTypeRef(Bindings& bindings,
                                           const Token& start_tok);

  // Parses an AST construct that refers to a type; e.g. a name or a colon-ref.
  absl::StatusOr<TypeRefOrAnnotation> ParseTypeRef(Bindings& bindings,
                                                   const Token& tok);

  // allow_generic_type indicates `T: type` is allowed.
  absl::StatusOr<TypeAnnotation*> ParseTypeAnnotation(
      Bindings& bindings, std::optional<Token> first = std::nullopt,
      bool allow_generic_type = false);

  // Parses the parametrics and dims after a `TypeRef` that the caller has
  // already parsed, producing a `TypeAnnotation` for the whole thing.
  absl::StatusOr<TypeAnnotation*> ParseTypeRefParametricsAndDims(
      Bindings& bindings, const Span& span, TypeRefOrAnnotation type_ref);

  absl::StatusOr<NameRef*> ParseNameRef(Bindings& bindings,
                                        const Token* tok = nullptr);

  // Precondition: token cursor should be over a double colon '::' token.
  absl::StatusOr<ColonRef*> ParseColonRef(Bindings& bindings,
                                          ColonRef::Subject subject,
                                          const Span& subject_span);

  absl::StatusOr<Expr*> ParseCastOrEnumRefOrStructInstanceOrToken(
      Bindings& bindings);

  absl::StatusOr<Expr*> ParseStructInstance(Bindings& bindings,
                                            TypeAnnotation* type = nullptr);

  absl::StatusOr<std::variant<NameRef*, ColonRef*>> ParseNameOrColonRef(
      Bindings& bindings, std::string_view context = "");

  // As above, but does not add the parsed identifier to any set of bindings.
  absl::StatusOr<NameDef*> ParseNameDefNoBind();

  // Parses a name definition and adds it to the given set of bindings.
  //
  // Wrapper around `ParseNameDefNoBind` above.
  absl::StatusOr<NameDef*> ParseNameDef(Bindings& bindings);

  absl::StatusOr<Token> PopSelfOrIdentifier(std::string_view context);

  absl::StatusOr<std::variant<NameDef*, WildcardPattern*, RestOfTuple*>>
  ParseNameDefOrWildcard(Bindings& bindings);

  // Parses tree of name defs and returns it.
  //
  // For example, the left hand side of:
  //
  //  let (a, (b, (c)), d) = ...
  //
  // This is used for tuple-like (sometimes known as "destructing") let binding.
  absl::StatusOr<NameDefTree*> ParseNameDefTree(Bindings& bindings);

  absl::StatusOr<Number*> TokenToNumber(const Token& tok);
  absl::StatusOr<NameDef*> TokenToNameDef(const Token& tok) {
    return module_->Make<NameDef>(tok.span(), *tok.GetValue(), nullptr);
  }
  absl::StatusOr<BuiltinType> TokenToBuiltinType(const Token& tok);
  absl::StatusOr<TypeAnnotation*> MakeBuiltinTypeAnnotation(
      const Span& span, const Token& tok, absl::Span<Expr* const> dims);
  absl::StatusOr<TypeAnnotation*> MakeTypeRefTypeAnnotation(
      const Span& span, TypeRefOrAnnotation type_ref,
      absl::Span<Expr* const> dims, std::vector<ExprOrType> parametrics);

  // Returns a parsed number (literal number) expression.
  absl::StatusOr<Number*> ParseNumber(Bindings& bindings);

  absl::StatusOr<Let*> ParseLet(Bindings& bindings);

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
                                                Bindings& bindings);

  absl::StatusOr<Array*> ParseArray(Bindings& bindings);

  // If the next token is a colon, then parses a cast to `type`; otherwise just
  // returns `type`.
  absl::StatusOr<ExprOrType> MaybeParseCast(Bindings& bindings,
                                            TypeAnnotation* type);

  absl::StatusOr<Expr*> ParseCast(Bindings& bindings,
                                  TypeAnnotation* type = nullptr);

  // Parses a term as a component of an expression and returns it.
  //
  // Terms are more atomic than arithmetic expressions.
  absl::StatusOr<Expr*> ParseTerm(Bindings& bindings,
                                  ExprRestrictions restrictions);

  // Parses the "left hand side" of a term expression -- since term is the
  // tightest-binding expression production, we look for tokens following the
  // left-hand side like '[' or '(' in ParseTerm() above.
  absl::StatusOr<Expr*> ParseTermLhs(Bindings& bindings,
                                     ExprRestrictions restrictions);

  // Attempts to parse a parenthetical and falls back to parsing as cast on
  // failure.
  absl::StatusOr<Expr*> ParseParentheticalOrCastLhs(Bindings& outer_bindings,
                                                    const Pos& start_pos);

  absl::StatusOr<Expr*> ParseTermLhsParenthesized(Bindings& bindings,
                                                  const Pos& start_pos);

  // When there is no "chained" right hand side observed, returns the original
  // "lhs".
  absl::StatusOr<Expr*> ParseTermRhs(Expr* lhs, Bindings& bindings,
                                     ExprRestrictions restrictions);

  // Parses a slicing index expression.
  absl::StatusOr<Index*> ParseBitSlice(const Pos& start_pos, Expr* lhs,
                                       Bindings& bindings,
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

  absl::StatusOr<Expr*> ParseCastAsExpression(Bindings& bindings,
                                              ExprRestrictions restrictions);

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

  absl::StatusOr<Expr*> ParseStrongArithmeticExpression(
      Bindings& bindings, ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseWeakArithmeticExpression(
      Bindings& bindings, ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseBitwiseExpression(Bindings& bindings,
                                               ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseAndExpression(Bindings& bindings,
                                           ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseXorExpression(Bindings& bindings,
                                           ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseOrExpression(Bindings& bindings,
                                          ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseComparisonExpression(
      Bindings& bindings, ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseLogicalAndExpression(
      Bindings& bindings, ExprRestrictions restrictions);

  absl::StatusOr<Expr*> ParseLogicalOrExpression(Bindings& bindings,
                                                 ExprRestrictions restrictions);

  // RangeExpression ::=
  //   LogicalOrExpression [".." LogicalOrExpression]
  absl::StatusOr<Expr*> ParseRangeExpression(Bindings& bindings,
                                             ExprRestrictions restrictions);

  // Parses a conditional expression (or expression that binds more tightly).
  //
  // Example:
  //
  //    if { bar } else { baz }
  //
  // ConditionalExpression ::=
  //    ConditionalNode
  //  | RangeExpression
  absl::StatusOr<Expr*> ParseConditionalExpression(
      Bindings& bindings, ExprRestrictions restrictions);

  // Parses a conditional construct and returns it -- the token cursor should be
  // hovering over the "if" keyword on invocation for this to be successful.
  absl::StatusOr<Conditional*> ParseConditionalNode(
      Bindings& bindings, ExprRestrictions restrictions);

  absl::StatusOr<Param*> ParseParam(Bindings& bindings);

  // Parses a member declaration in the body of a `proc` definition.
  absl::StatusOr<ProcMember*> ParseProcMember(Bindings& bindings,
                                              const Token& identifier_tok);

  // Parses a sequence of parameters, starting with cursor over '(', returns
  // after ')' is consumed.
  //
  // Permits trailing commas.
  absl::StatusOr<std::vector<Param*>> ParseParams(Bindings& bindings);

  absl::StatusOr<NameDefTree*> ParseTuplePattern(const Pos& start_pos,
                                                 Bindings& bindings);

  // Returns a parsed pattern; e.g. one that would guard a match arm.
  //
  //  Pattern ::= TuplePattern
  //            | WildcardPattern
  //            | ColonRef
  //            | NameDef
  //            | NameRef
  //            | Number
  absl::StatusOr<NameDefTree*> ParsePattern(Bindings& bindings,
                                            bool within_tuple_pattern);

  // Parses a match expression.
  absl::StatusOr<Match*> ParseMatch(Bindings& bindings);

  // Parses a channel declaration.
  absl::StatusOr<ChannelDecl*> ParseChannelDecl(Bindings& bindings);

  // Parses a for loop construct; e.g.
  //
  //  for (i, accum) in range(3) {
  //    accum + i
  //  }(0)
  //
  // The init value is passed to the loop and the body updates the value;
  // ultimately the loop terminates and the final accum value is returned.
  absl::StatusOr<For*> ParseFor(Bindings& bindings);

  // Parses an "unroll for" macro-like construct, e.g.
  //
  // unroll_for! i in range(u32:, u32:4) {
  //   spawn my_proc(...)(...);
  // }
  absl::StatusOr<UnrollFor*> ParseUnrollFor(Bindings& bindings);

  // Parses an enum definition; e.g.
  //
  //  enum Foo : u2 {
  //    A = 0,
  //    B = 1,
  //  }
  absl::StatusOr<EnumDef*> ParseEnumDef(const Pos& start_pos, bool is_public,
                                        Bindings& bindings);

  absl::StatusOr<StructDef*> ParseStruct(const Pos& start_pos, bool is_public,
                                         Bindings& bindings);

  absl::StatusOr<Impl*> ParseImpl(const Pos& start_pos, bool is_public,
                                  Bindings& bindings);

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
      Bindings& bindings);

  // Parses parametric dims that follow a struct type annotation.
  //
  // For example:
  //
  //    x: ParametricStruct<u32:4, N as u64>
  //                       ^---------------^
  absl::StatusOr<std::vector<ExprOrType>> ParseParametrics(Bindings& bindings);

  // Parses a single parametric arg.
  //
  // For example:
  //
  //    x: ParametricStruct<u32:4, N as u64>
  //                        ^---^
  absl::StatusOr<ExprOrType> ParseParametricArg(Bindings& bindings);

  // Parses a function out of the token stream.
  absl::StatusOr<Function*> ParseFunctionInternal(const Pos& start_pos,
                                                  bool is_public,
                                                  Bindings& outer_bindings);

  // Parses an import statement into an `Import` AST node.
  absl::StatusOr<Import*> ParseImport(Bindings& bindings);

  // Parses a single entry in a `use` tree -- this can be a leaf or an interior
  // entry.
  absl::StatusOr<UseTreeEntry*> ParseUseTreeEntry(Bindings& bindings);

  // Parses a use statement into a `Use` AST node.
  absl::StatusOr<Use*> ParseUse(Bindings& bindings);

  // Returns TestFunction AST node by parsing new-style unit test construct.
  absl::StatusOr<TestFunction*> ParseTestFunction(Bindings& bindings,
                                                  const Span& attribute_span);

  absl::StatusOr<TestProc*> ParseTestProc(Bindings& bindings);

  // Parses a constant definition (e.g. at the top level of a module). Token
  // cursor should be over the `const` keyword.
  absl::StatusOr<ConstantDef*> ParseConstantDef(const Pos& start_pos,
                                                bool is_public,
                                                Bindings& bindings);

  absl::StatusOr<QuickCheck*> ParseQuickCheck(
      absl::flat_hash_map<std::string, Function*>* name_to_fn,
      Bindings& bindings, const Pos& hash_pos);

  // Parses the test count configuration for a quickcheck attribute.
  absl::StatusOr<QuickCheckTestCases> ParseQuickCheckConfig();

  // Parses a module-level attribute -- cursor should be over the open bracket.
  //
  // Side-effect: module_ is tagged with the parsed attribute on success.
  absl::Status ParseModuleAttribute();

  // Parses DSLX attributes, analogous to Rust's attributes.
  //
  // This accepts the following:
  // #[test] Expects a 'fn', returns TestFunction*
  // #[extern_verilog(...)] Expects a fn, returns Function*
  // #[test_proc] Expects a proc, returns TestProc*
  // #[quickcheck(...)] Expects a fn, returns QuickCheck*
  // #[sv_type(...)] Expects a TypeDefinition, returns TypeDefinition
  absl::StatusOr<std::variant<TestFunction*, Function*, TestProc*, QuickCheck*,
                              TypeDefinition, std::nullptr_t>>
  ParseAttribute(absl::flat_hash_map<std::string, Function*>* name_to_fn,
                 Bindings& bindings, const Pos& hash_pos);

  // Parses a "spawn" statement, which creates & initializes a proc.
  absl::StatusOr<Spawn*> ParseSpawn(Bindings& bindings);

  // Helper function that chooses between building a FormatMacro or an
  // Invocation based on the callee.
  absl::StatusOr<Expr*> BuildMacroOrInvocation(
      const Span& span, Bindings& bindings, Expr* callee,
      std::vector<Expr*> args,
      std::vector<ExprOrType> parametrics = std::vector<ExprOrType>{});

  // Helper function that builds a `FormatMacro` corresponding to a DSLX
  // invocation with verbosity, like vtrace_fmt!(...). The verbosity should be
  // the first element of `args`.
  absl::StatusOr<Expr*> BuildFormatMacroWithVerbosityArgument(
      const Span& span, std::string_view name, std::vector<Expr*> args,
      const std::vector<ExprOrType>& parametrics);

  // Helper function that builds a `FormatMacro` corresponding to a DSLX
  // invocation like trace_fmt!(...) or vtrace_fmt!(...), after the caller has
  // extracted and removed any verbosity argument from `args`.
  absl::StatusOr<Expr*> BuildFormatMacro(
      const Span& span, std::string_view name, std::vector<Expr*> args,
      const std::vector<ExprOrType>& parametrics,
      std::optional<Expr*> verbosity = std::nullopt);

  // Parses a proc config function.
  //
  // Args:
  //  parametric_bindings: Parametric bindings created at the proc level.
  //  proc_members: Member declarations at the proc scope.
  absl::StatusOr<Function*> ParseProcConfig(
      Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
      const std::vector<ProcMember*>& proc_members, std::string_view proc_name,
      bool is_public);

  absl::StatusOr<Function*> ParseProcNext(
      Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
      std::string_view proc_name, bool is_public);

  absl::StatusOr<Function*> ParseProcInit(
      Bindings& bindings, std::vector<ParametricBinding*> parametric_bindings,
      std::string_view proc_name, bool is_public);

  // Parses a proc-like entity (i.e. either a Proc or a Block). This will yield
  // a node of type `T` unless the entity parsed is actually an impl-style
  // `ProcDef`.
  template <typename T>
  absl::StatusOr<ModuleMember> ParseProcLike(const Pos& start_pos,
                                             bool is_public,
                                             Bindings& outer_bindings,
                                             Keyword keyword);

  // Bumps the internally-tracked expression depth so we can provide a useful
  // error if we over-recurse on expression depth past the point a user would
  // reasonably be expected to do. This (e.g.) helps avoid stack overflows
  // during fuzzing.
  absl::StatusOr<ExpressionDepthGuard> BumpExpressionDepth();

  std::unique_ptr<Module> module_;

  // `Let` nodes are created _after_ those that use their namedefs (due to the
  // chaining of the `body` member variable. We need to know, though, if a
  // reference to such an NDT (or element thereof) is to a constant or not, so
  // we can emit a NameRef. This set holds those NDTs known
  // to be constant for that purpose.
  absl::flat_hash_set<NameDefTree*> const_ndts_;

  // To avoid over-recursion (and ensuing stack overflows) we keep track of the
  // approximate expression depth, and bail when expressions are unreasonably
  // deeply nested.
  static constexpr int64_t kApproximateExpressionDepthLimit = 64;
  int64_t approximate_expression_depth_ = 0;

  // When true, we expect no function bodies, and a semicolon after
  // the return type of a function.
  bool parse_fn_stubs_;
};

const Span& GetSpan(
    const std::variant<NameDef*, WildcardPattern*, RestOfTuple*>& v);

}  // namespace xls::dslx

#endif  // XLS_DSLX_FRONTEND_PARSER_H_
