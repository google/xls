// Copyright 2020 Google LLC
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

// Subset-of-verilog AST, suitable for combining as datastructures before
// emission.

#ifndef XLS_CODEGEN_VAST_H_
#define XLS_CODEGEN_VAST_H_

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/types/optional.h"
#include "absl/types/variant.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"

namespace xls {
namespace verilog {

// Forward declarations.
class VerilogFile;
class Expression;
class IndexableExpression;
class LogicRef;
class Literal;
class Unary;

// Returns a sanitized identifier string based on the given name. Invalid
// characters are replaced with '_'.
std::string SanitizeIdentifier(absl::string_view name);

// Base type for a VAST node.
//
// Ownership is generally held by File in user code.
class VastNode {
 public:
  virtual ~VastNode() = default;
};

// Trait used for named entities.
class NamedTrait : public VastNode {
 public:
  ~NamedTrait() override = default;

  // Returns a name that can be used to refer to the object in generated Verilog
  // code; e.g. for a macro this would return "`THING", for a wire it would
  // return "wire_name".
  virtual std::string GetName() const = 0;
};

// Represents a behavioral statement.
class Statement : public VastNode {
 public:
  ~Statement() override = default;

  virtual std::string Emit() = 0;
};

// Defines a named reg/wire of a given width.
class Def : public Statement {
 public:
  Def(absl::string_view name, Expression* width, bool is_signed = false)
      : name_(name), width_(width), is_signed_(is_signed) {}

  std::string GetName() const { return name_; }

  std::string EmitNoSemi() {
    std::string result = Emit();
    return std::string(absl::StripSuffix(result, ";"));
  }

  const std::string& name() const { return name_; }

  Expression* width() const {
    return width_;
  }

  // Returns true if this is a Def of an array.
  virtual bool IsArrayDef() const { return false; }

  bool is_signed() const { return is_signed_; }

 private:
  std::string name_;
  Expression* width_;
  bool is_signed_;
};

// Represents an uninitialized register definition.
struct UninitializedSentinel {};

// Register initialization value.
using RegInit = absl::variant<Expression*, UninitializedSentinel>;

std::string ToString(const RegInit& init);

class WireDef : public Def {
 public:
  explicit WireDef(absl::string_view name, Expression* width,
                   bool is_signed = false)
      : Def(name, width, is_signed) {}

  std::string Emit() override;
};

// Register definition.
class RegDef : public Def {
 public:
  RegDef(absl::string_view name, Expression* width,
         RegInit init = UninitializedSentinel(), bool is_signed = false)
      : Def(name, width, is_signed), init_(init) {}

  std::string Emit() override;

 protected:
  RegInit init_;
};

// Unpacked arrays can be declared using sizes (SystemVerilog only) or ranges
// (SystemVerilog or Verilog) for the bounds. For example, the following two
// declarations are equivalent:
//
//   reg [7:0] foo[42][123]
//   reg [7:0] foo[0:41][0:122]
//
// UnpackedArrayBounds is a sum type which holds either form.
using UnpackedArrayBound =
    absl::variant<Expression*, std::pair<Expression*, Expression*>>;

// Unpacked array register definition.
// TODO(meheff): This probably should be merged into RegDef where RegDef takes
// an optional set of unpacked array bounds.
class UnpackedArrayRegDef : public RegDef {
 public:
  UnpackedArrayRegDef(absl::string_view name, Expression* element_width,
                      absl::Span<const UnpackedArrayBound> bounds,
                      RegInit init = UninitializedSentinel())
      : RegDef(name, element_width, init),
        bounds_(bounds.begin(), bounds.end()) {}

  std::string Emit() override;

  absl::Span<const UnpackedArrayBound> bounds() const { return bounds_; }

  bool IsArrayDef() const override { return true; }

 private:
  std::vector<UnpackedArrayBound> bounds_;
  VerilogFile* file_;
};

// Unpacked array wire definition.
// TODO(meheff): This probably should be merged into WireDef where WireDef takes
// an optional set of unpacked array bounds.
class UnpackedArrayWireDef : public WireDef {
 public:
  UnpackedArrayWireDef(absl::string_view name, Expression* element_width,
                       absl::Span<const UnpackedArrayBound> bounds)
      : WireDef(name, element_width), bounds_(bounds.begin(), bounds.end()) {}

  std::string Emit() override;

  absl::Span<const UnpackedArrayBound> bounds() const { return bounds_; }

  bool IsArrayDef() const override { return true; }

 private:
  std::vector<UnpackedArrayBound> bounds_;
  VerilogFile* file_;
};

// Represents a #${delay} statement.
class DelayStatement : public Statement {
 public:
  // If delay_statement is non-null then this represents a delayed statement:
  //
  //   #${delay} ${delayed_statement};
  //
  // otherwise this is a solitary delay statement:
  //
  //   #${delay};
  explicit DelayStatement(Expression* delay,
                          Statement* delayed_statement = nullptr)
      : delay_(delay), delayed_statement_(delayed_statement) {}

  std::string Emit() override;

 private:
  Expression* delay_;
  Statement* delayed_statement_;
};

// Represents a wait statement.
class WaitStatement : public Statement {
 public:
  explicit WaitStatement(Expression* event) : event_(event) {}

  std::string Emit() override;

 private:
  Expression* event_;
};

// Represents a forever construct which runs a statement continuously.
class Forever : public Statement {
 public:
  explicit Forever(Statement* statement) : statement_(statement) {}

  std::string Emit() override;

 private:
  Statement* statement_;
};

// Represents a blocking assignment ("lhs = rhs;")
class BlockingAssignment : public Statement {
 public:
  BlockingAssignment(Expression* lhs, Expression* rhs)
      : lhs_(XLS_DIE_IF_NULL(lhs)), rhs_(rhs) {}

  std::string Emit() override;

 private:
  Expression* lhs_;
  Expression* rhs_;
};

// Represents a nonblocking assignment  ("lhs <= rhs;").
class NonblockingAssignment : public Statement {
 public:
  NonblockingAssignment(Expression* lhs, Expression* rhs)
      : lhs_(XLS_DIE_IF_NULL(lhs)), rhs_(rhs) {}

  std::string Emit() override;

 private:
  Expression* lhs_;
  Expression* rhs_;
};

// An abstraction representing a sequence of statements within a structured
// procedure (e.g., an "always" statement).
class StatementBlock : public VastNode {
 public:
  explicit StatementBlock(VerilogFile* f) : parent_(f) {}

  // Constructs and adds a statement to the block. Ownership is maintained by
  // the parent VerilogFile. Example:
  //   Case* c = Add<Case>(subject);
  template <typename T, typename... Args>
  inline T* Add(Args&&... args);

  std::string Emit();
  VerilogFile* parent() const { return parent_; }

 private:
  VerilogFile* parent_;
  std::vector<Statement*> statements_;
};

// Represents a 'default' case arm label.
struct DefaultSentinel {};

// Represents a label within a case statement.
using CaseLabel = absl::variant<Expression*, DefaultSentinel>;

// Represents an arm of a case statement.
class CaseArm : public VastNode {
 public:
  CaseArm(VerilogFile* f, CaseLabel label);

  std::string GetLabelString();
  StatementBlock* statements() { return statements_; }

 private:
  CaseLabel label_;
  StatementBlock* statements_;
};

// Represents a case statement.
class Case : public Statement {
 public:
  explicit Case(VerilogFile* f, Expression* subject)
      : parent_(f), subject_(subject) {}

  StatementBlock* AddCaseArm(CaseLabel label);

  std::string Emit() override;

 private:
  VerilogFile* parent_;
  Expression* subject_;
  std::vector<CaseArm*> arms_;
};

// Represents an if statement with optional "else if" and "else" blocks.
class Conditional : public Statement {
 public:
  Conditional(VerilogFile* f, Expression* condition);

  // Returns a pointer to the statement block of the consequent.
  StatementBlock* consequent() { return consequent_; }

  // Adds an alternate clause ("else if" or "else") and returns a pointer to the
  // consequent. The alternate is final (an "else") if condition is null. Dies
  // if a final alternate ("else") clause has been previously added.
  StatementBlock* AddAlternate(Expression* condition = nullptr);

  std::string Emit() override;

 private:
  VerilogFile* parent_;
  Expression* condition_;
  StatementBlock* consequent_;

  // The alternate clauses ("else if's" and "else"). If the Expression* is null
  // then the alternate is unconditional ("else"). This can only appear as the
  // final alternate.
  std::vector<std::pair<Expression*, StatementBlock*>> alternates_;
};

// Represents a while loop construct.
class WhileStatement : public Statement {
 public:
  WhileStatement(VerilogFile* f, Expression* condition);

  std::string Emit() override;

  StatementBlock* statements() { return statements_; }

 private:
  Expression* condition_;
  StatementBlock* statements_;
};

// Represents a repeat construct.
class RepeatStatement : public Statement {
 public:
  RepeatStatement(Expression* repeat_count, Statement* statement)
      : repeat_count_(repeat_count), statement_(statement) {}

  std::string Emit() override;

 private:
  Expression* repeat_count_;
  Statement* statement_;
};

// Represents an event control statement. This is represented as "@(...);" where
// "..." is the event expression..
class EventControl : public Statement {
 public:
  explicit EventControl(Expression* event_expression)
      : event_expression_(event_expression) {}

  std::string Emit() override;

 private:
  Expression* event_expression_;
};

// Specifies input/output direction (e.g. for a port).
enum class Direction {
  kInput,
  kOutput,
};

std::string ToString(Direction direction);
inline std::ostream& operator<<(std::ostream& os, Direction d) {
  os << ToString(d);
  return os;
}

// Represents a Verilog expression.
class Expression : public VastNode {
 public:
  ~Expression() override = default;

  virtual bool IsLiteral() const { return false; }
  virtual bool IsLiteralWithValue(int64 target) const { return false; }
  Literal* AsLiteralOrDie();

  virtual bool IsLogicRef() const { return false; }
  LogicRef* AsLogicRefOrDie();

  virtual bool IsIndexableExpression() const { return false; }
  IndexableExpression* AsIndexableExpressionOrDie();

  virtual bool IsUnary() const { return false; }
  Unary* AsUnaryOrDie();

  // Returns the precedence of the expression. This is used when emitting the
  // Expression to determine if parentheses are necessary. Expressions which are
  // leaves such as Literal or LogicRef don't strictly have a precedence, but
  // for the purposses of emission we consider them to have maximum precedence
  // so they are never wrapped in parentheses. Operator (derived from
  // Expression) are the only types with non-max precedence.
  //
  // Precedence of operators in Verilog operator precendence (from LRM):
  //   Highest:  (12)  + - ! ~ (unary) & | ^ (reductions)
  //             (11)  **
  //             (10)  * / %
  //              (9)  + - (binary)
  //              (8)  << >> <<< >>>
  //              (7)  < <= > >=
  //              (6)  == != === !==
  //              (5)  & ~&
  //              (4)  ^ ^~ ~^
  //              (3)  | ~|
  //              (2)  &&
  //              (1)  ||
  //   Lowest:    (0)  ?: (conditional operator)
  static constexpr int64 kMaxPrecedence = 13;
  static constexpr int64 kMinPrecedence = -1;
  virtual int64 precedence() const { return kMaxPrecedence; }

  virtual std::string Emit() = 0;
};

// Represents an X value.
class XSentinel : public Expression {
 public:
  explicit XSentinel(int64 width) : width_(width) {}

  std::string Emit() override;

 private:
  int64 width_;
};

// Represents an operation (unary, binary, etc) with a particular precedence.
class Operator : public Expression {
 public:
  explicit Operator(int64 precedence) : precedence_(precedence) {}
  int64 precedence() const override { return precedence_; }

 private:
  int64 precedence_;
};

// A posedge edge identifier expression.
class PosEdge : public Expression {
 public:
  explicit PosEdge(Expression* expression) : expression_(expression) {}

  std::string Emit() override;

 private:
  Expression* expression_;
};

// A negedge edge identifier expression.
class NegEdge : public Expression {
 public:
  explicit NegEdge(Expression* expression) : expression_(expression) {}

  std::string Emit() override;

 private:
  Expression* expression_;
};

// Represents a connection of either a module parameter or a port to its
// surrounding environment.
struct Connection {
  std::string port_name;
  Expression* expression;
};

// Represents a module instantiation.
class Instantiation : public VastNode {
 public:
  Instantiation(absl::string_view module_name, absl::string_view instance_name,
                absl::Span<const Connection> parameters,
                absl::Span<const Connection> connections)
      : module_name_(module_name),
        instance_name_(instance_name),
        parameters_(parameters.begin(), parameters.end()),
        connections_(connections.begin(), connections.end()) {}

  std::string Emit();

 private:
  std::string module_name_;
  std::string instance_name_;
  std::vector<Connection> parameters_;
  std::vector<Connection> connections_;
};

// Represents a reference to an already-defined macro.
class MacroRef : public Expression {
 public:
  explicit MacroRef(std::string name) : name_(name) {}

  std::string Emit() override;

 private:
  std::string name_;
};

// Defines a module parameter.
class Parameter : public NamedTrait {
 public:
  explicit Parameter(absl::string_view name, Expression* rhs)
      : name_(name), rhs_(rhs) {}

  std::string Emit();
  std::string GetName() const override { return name_; }

 private:
  std::string name_;
  Expression* rhs_;
};

// Defines an item in a localparam.
class LocalParamItem : public NamedTrait {
 public:
  explicit LocalParamItem(absl::string_view name, Expression* rhs)
      : name_(name), rhs_(rhs) {}

  std::string GetName() const override { return name_; }

  std::string Emit();

 private:
  std::string name_;
  Expression* rhs_;
};

// Refers to an item in a localparam for use in expressions.
class LocalParamItemRef : public Expression {
 public:
  explicit LocalParamItemRef(LocalParamItem* item) : item_(item) {}

  std::string Emit() override { return item_->GetName(); }

 private:
  LocalParamItem* item_;
};

// Defines a localparam.
class LocalParam : public VastNode {
 public:
  explicit LocalParam(VerilogFile* f) : parent_(f) {}
  LocalParamItemRef* AddItem(absl::string_view name, Expression* value);

  std::string Emit();

 private:
  VerilogFile* parent_;
  std::vector<LocalParamItem*> items_;
};

// Refers to a Parameter's definition for use in an expression.
class ParameterRef : public Expression {
 public:
  explicit ParameterRef(Parameter* parameter) : parameter_(parameter) {}

  std::string Emit() override { return parameter_->GetName(); }

 private:
  Parameter* parameter_;
};

// An indexable expression that can be bit-sliced or indexed.
class IndexableExpression : public Expression {
 public:
  bool IsIndexableExpression() const override { return true; }

  // Returns whether this is a scalar register reference that should be referred
  // to by name because indexing a scalar is invalid (System)Verilog.
  virtual bool IsScalarReg() const { return false; }
};

// Forward declaration.
template <int64 N>
class LogicRefN;

// Refers to a WireDef's or RegDef's definition.
class LogicRef : public IndexableExpression {
 public:
  explicit LogicRef(Def* def) : def_(XLS_DIE_IF_NULL(def)) {}

  bool IsLogicRef() const override { return true; }

  std::string Emit() override { return def_->name(); }

  bool IsScalarReg() const override {
    return def_->width()->IsLiteralWithValue(1) && !def_->IsArrayDef();
  }

  // Performs a checked-conversion of this LogicRef into a logic ref of the
  // given width N.
  //
  // The width of this logic ref must be a literal value equal to N.
  template <int64 N>
  LogicRefN<N>* AsLogicRefNOrDie() {
    using CastedT = LogicRefN<N>;
    auto* result = static_cast<CastedT*>(this);
    result->CheckInvariants();
    return result;
  }

  // Returns the width of this logic signal.
  Expression* width() {
    auto* result = def_->width();
    XLS_CHECK(result != nullptr);
    return result;
  }

  // Returns the Def that this LogicRef refers to.
  Def* def() const { return def_; }

  // Returns the name of the underlying Def this object refers to.
  std::string GetName() const { return def()->GetName(); }

 private:
  // Logic signal definition.
  Def* def_;
};

// Templated subtype for representing fixed-width logic signals that have the
// width imbued in the type. This is helpful in scenarios where only one width
// of logic is acceptable; e.g. for a clock signal a "LogicRef1*" is more
// precise.
template <int64 N>
class LogicRefN : public LogicRef {
 public:
  explicit LogicRefN(Def* def) : LogicRef(def) { CheckInvariants(); }

  void CheckInvariants() {
    XLS_CHECK(width()->IsLiteralWithValue(N))
        << "Expected logic of width: " << N << " found: " << width()->Emit();
  }
};

// Some helper definitions for convenience.
using LogicRef1 = LogicRefN<1>;
using LogicRef8 = LogicRefN<8>;

// Represents a Verilog unary expression.
class Unary : public Operator {
 public:
  Unary(absl::string_view op, Expression* arg, int64 precedence)
      : Operator(precedence), op_(op), arg_(arg) {}

  bool IsUnary() const override { return true; }

  std::string Emit() override;

 private:
  std::string op_;
  Expression* arg_;
};

// Abstraction describing a reset signal.
struct Reset {
  LogicRef1* signal;
  bool asynchronous;
  bool active_low;
};

// Defines an always_ff-equivalent block.
// TODO(meheff): Replace uses of AlwaysFlop with Always or AlwaysFf. AlwaysFlop
// has a higher level of abstraction which is now better handled by
// ModuleBuilder.
class AlwaysFlop : public VastNode {
 public:
  AlwaysFlop(VerilogFile* file, LogicRef* clk,
             absl::optional<Reset> rst = absl::nullopt);

  // Add a register controlled by this AlwaysFlop. 'reset_value' can only be
  // non-null if the AlwaysFlop has a reset signal.
  void AddRegister(LogicRef* reg, Expression* reg_next,
                   Expression* reset_value = nullptr);

  std::string Emit();

 private:
  VerilogFile* file_;
  LogicRef* clk_;
  absl::optional<Reset> rst_;
  // The top-level block inside the always statement.
  StatementBlock* top_block_;

  // The block containing the assignments active when resetting. This is nullptr
  // if the AlwaysFlop has no reset signal.
  StatementBlock* reset_block_;

  // The block containing the non-reset assignments.
  StatementBlock* assignment_block_;
};

class StructuredProcedure : public VastNode {
 public:
  explicit StructuredProcedure(VerilogFile* f);
  virtual std::string Emit() = 0;

  StatementBlock* statements() { return statements_; }

 protected:
  StatementBlock* statements_;
};

// Represents the '*' which can occur in an always sensitivity list.
struct ImplicitEventExpression {};

// Elements which can appear in a sensitivity list for an always or always_ff
// block.
using SensitivityListElement =
    absl::variant<ImplicitEventExpression, PosEdge*, NegEdge*>;

// Base class for 'always' style blocks with a sensitivity list.
class AlwaysBase : public StructuredProcedure {
 public:
  AlwaysBase(VerilogFile* f,
             absl::Span<const SensitivityListElement> sensitivity_list)
      : StructuredProcedure(f),
        sensitivity_list_(sensitivity_list.begin(), sensitivity_list.end()) {}
  std::string Emit() override;

 protected:
  virtual std::string name() const = 0;

  std::vector<SensitivityListElement> sensitivity_list_;
};

// Defines an always block.
class Always : public AlwaysBase {
 public:
  Always(VerilogFile* f,
         absl::Span<const SensitivityListElement> sensitivity_list)
      : AlwaysBase(f, sensitivity_list) {}

 protected:
  std::string name() const override { return "always"; }
};

// Defines an always_comb block.
class AlwaysComb : public AlwaysBase {
 public:
  explicit AlwaysComb(VerilogFile* f) : AlwaysBase(f, {}) {}
  std::string Emit() override;

 protected:
  std::string name() const override { return "always_comb"; }
};

// Defines an always_ff block.
class AlwaysFf : public AlwaysBase {
 public:
  AlwaysFf(VerilogFile* f,
           absl::Span<const SensitivityListElement> sensitivity_list)
      : AlwaysBase(f, sensitivity_list) {}

 protected:
  std::string name() const override { return "always_ff"; }
};

// Defines an 'initial' block.
class Initial : public StructuredProcedure {
 public:
  explicit Initial(VerilogFile* f) : StructuredProcedure(f) {}
  std::string Emit() override;
};

class Concat : public Expression {
 public:
  explicit Concat(absl::Span<Expression* const> args)
      : args_(args.begin(), args.end()), replication_(absl::nullopt) {}

  // Defines a concatenation with replication. Example: {3{1'b101}}
  Concat(Expression* replication, absl::Span<Expression* const> args)
      : args_(args.begin(), args.end()), replication_(replication) {}

  std::string Emit() override;

 private:
  std::vector<Expression*> args_;
  absl::optional<Expression*> replication_;
};

// An array assignment pattern such as: "'{foo, bar, baz}"
class ArrayAssignmentPattern : public IndexableExpression {
 public:
  explicit ArrayAssignmentPattern(absl::Span<Expression* const> args)
      : args_(args.begin(), args.end()) {}

  std::string Emit() override;

 private:
  std::vector<Expression*> args_;
};

class BinaryInfix : public Operator {
 public:
  BinaryInfix(Expression* lhs, absl::string_view op, Expression* rhs,
              int64 precedence)
      : Operator(precedence),
        op_(op),
        lhs_(XLS_DIE_IF_NULL(lhs)),
        rhs_(XLS_DIE_IF_NULL(rhs)) {}

  std::string Emit() override;

 private:
  std::string op_;
  Expression* lhs_;
  Expression* rhs_;
};

// Defines a literal value (width and value).
class Literal : public Expression {
 public:
  explicit Literal(Bits bits, FormatPreference format,
                   bool emit_bit_count = true)
      : bits_(bits), format_(format), emit_bit_count_(emit_bit_count) {
    XLS_CHECK(emit_bit_count_ || bits.bit_count() == 32);
  }

  std::string Emit() override;

  const Bits& bits() const { return bits_; }

  bool IsLiteral() const override { return true; }
  bool IsLiteralWithValue(int64 target) const override;

 private:
  Bits bits_;
  FormatPreference format_;
  // Whether to emit the bit count when emitting the number. This can only be
  // false if the width of bits_ is 32 as the width of an undecorated number
  // literal in Verilog is 32.
  bool emit_bit_count_;
};

// Represents a quoted literal string.
class QuotedString : public Expression {
 public:
  explicit QuotedString(absl::string_view str) : str_(str) {}

  std::string Emit() override;

 private:
  std::string str_;
};

class XLiteral : public Expression {
 public:
  std::string Emit() override { return "'X"; }
};

// Represents a Verilog slice expression; e.g.
//
//    subject[hi:lo]
class Slice : public Expression {
 public:
  Slice(IndexableExpression* subject, Expression* hi, Expression* lo)
      : subject_(subject), hi_(hi), lo_(lo) {}

  std::string Emit() override;

 private:
  IndexableExpression* subject_;
  Expression* hi_;
  Expression* lo_;
};

// Represents a Verilog indexed part-select expression; e.g.
//
//    subject[start +: width]
class DynamicSlice : public Expression {
 public:
  DynamicSlice(IndexableExpression* subject, Expression* start,
               Expression* width)
      : subject_(subject), start_(start), width_(width) {}

  std::string Emit() override;

 private:
  IndexableExpression* subject_;
  Expression* start_;
  Expression* width_;
};

// Represents a Verilog indexing operation; e.g.
//
//    subject[index]
class Index : public IndexableExpression {
 public:
  Index(IndexableExpression* subject, Expression* index)
      : subject_(subject), index_(index) {}

  std::string Emit() override;

 private:
  IndexableExpression* subject_;
  Expression* index_;
};

// Represents a Verilog ternary operator; e.g.
//
//    test ? consequent : alternate
class Ternary : public Expression {
 public:
  Ternary(Expression* test, Expression* consequent, Expression* alternate)
      : test_(test), consequent_(consequent), alternate_(alternate) {}

  std::string Emit() override;
  int64 precedence() const override { return 0; }

 private:
  Expression* test_;
  Expression* consequent_;
  Expression* alternate_;
};

// Represents a continuous assignment statement (e.g. at module scope).
//
// Note that the LHS of a continuous assignment can also be some kinds of
// expressions; e.g.
//
//    assign {x, y, z} = {a, b, c};
class ContinuousAssignment : public VastNode {
 public:
  ContinuousAssignment(Expression* lhs, Expression* rhs)
      : lhs_(lhs), rhs_(rhs) {}

  std::string Emit();

 private:
  Expression* lhs_;
  Expression* rhs_;
};

class BlankLine : public Statement {
 public:
  std::string Emit() override { return ""; }
};

// Places a comment in statement position (we can think of comments as
// meaningless expression statements that do nothing).
class Comment : public Statement {
 public:
  explicit Comment(absl::string_view text) : text_(text) {}

  std::string Emit() override;

 private:
  std::string text_;
};

// Represents call of a system task such as $display.
class SystemTaskCall : public Statement {
 public:
  // An argumentless invocation of a system task such as: $finish;
  explicit SystemTaskCall(absl::string_view name) : name_(name) {}

  // An invocation of a system task with arguments.
  SystemTaskCall(absl::string_view name, absl::Span<Expression* const> args)
      : name_(name) {
    args_ = std::vector<Expression*>(args.begin(), args.end());
  }

  std::string Emit() override;

 private:
  std::string name_;
  absl::optional<std::vector<Expression*>> args_;
};

// Represents statement function call expression such as $time.
class SystemFunctionCall : public Expression {
 public:
  // An argumentless invocation of a system function such as: $time;
  explicit SystemFunctionCall(absl::string_view name) : name_(name) {}

  // An invocation of a system function with arguments.
  SystemFunctionCall(absl::string_view name, absl::Span<Expression* const> args)
      : name_(name) {
    args_ = std::vector<Expression*>(args.begin(), args.end());
  }

  std::string Emit() override;

 private:
  std::string name_;
  absl::optional<std::vector<Expression*>> args_;
};

// Represents a $display function call.
class Display : public SystemTaskCall {
 public:
  explicit Display(absl::Span<Expression* const> args)
      : SystemTaskCall("display", args) {}
};

// Represents a $strobe function call.
class Strobe : public SystemTaskCall {
 public:
  explicit Strobe(absl::Span<Expression* const> args)
      : SystemTaskCall("strobe", args) {}
};

// Represents a $monitor function call.
class Monitor : public SystemTaskCall {
 public:
  explicit Monitor(absl::Span<Expression* const> args)
      : SystemTaskCall("monitor", args) {}
};

// Represents a $finish function call.
class Finish : public SystemTaskCall {
 public:
  Finish() : SystemTaskCall("finish") {}
};

// Represents a $signed function call which casts its argument to signed.
class SignedCast : public SystemFunctionCall {
 public:
  explicit SignedCast(Expression* value)
      : SystemFunctionCall("signed", {value}) {}
};

// Represents a $unsigned function call which casts its argument to unsigned.
class UnsignedCast : public SystemFunctionCall {
 public:
  explicit UnsignedCast(Expression* value)
      : SystemFunctionCall("unsigned", {value}) {}
};

// Represents the definition of a Verilog function.
class VerilogFunction : public VastNode {
 public:
  VerilogFunction(absl::string_view name, int64 result_width,
                  VerilogFile* file);

  // Adds an argument to the function and returns a reference to its value which
  // can be used in the body of the function.
  LogicRef* AddArgument(absl::string_view name, int64 width);

  // Adds a RegDef to the function and returns a LogicRef to it. This should be
  // used for adding RegDefs to the function instead of AddStatement because
  // the RegDefs need to appear outside the statement block (begin/end block).
  template <typename... Args>
  inline LogicRef* AddRegDef(Args&&... args);

  // Constructs and adds a statement to the block. Ownership is maintained by
  // the parent VerilogFile. Example:
  //   Case* c = Add<Case>(subject);
  template <typename T, typename... Args>
  inline T* AddStatement(Args&&... args);

  // Returns a reference to the variable representing the return value of the
  // function. Assigning to this reference sets the return value of the
  // function.
  LogicRef* return_value_ref();

  // Returns the name of the function.
  std::string name() { return name_; }

  std::string Emit();

 private:
  std::string name_;
  int64 result_width_;
  RegDef* return_value_def_;

  // The block containing all of the statements of the function. SystemVerilog
  // allows multiple statements in a function without encapsulating them in a
  // begin/end block (StatementBlock), but Verilog does not so we choose the
  // least common denominator.
  StatementBlock* statement_block_;

  VerilogFile* file_;

  std::vector<RegDef*> argument_defs_;

  // The RegDefs of reg's defined in the function. These are emitted before the
  // statement block.
  std::vector<RegDef*> block_reg_defs_;
};

// Represents a call to a VerilogFunction.
class VerilogFunctionCall : public Expression {
 public:
  VerilogFunctionCall(VerilogFunction* func, absl::Span<Expression* const> args)
      : func_(func), args_(args.begin(), args.end()) {}

  std::string Emit() override;

 private:
  VerilogFunction* func_;
  std::vector<Expression*> args_;
};

class ModuleSection;

// Represents a member of a module.
using ModuleMember =
    absl::variant<Def*,                   // Logic definition.
                  LocalParam*,            // Module-local parameter.
                  Parameter*,             // Module parameter.
                  Instantiation*,         // module instantiaion.
                  ContinuousAssignment*,  // Continuous assignment.
                  StructuredProcedure*,   // Initial or always comb block.
                  AlwaysFlop*,            // "Flip-Flop" block.
                  Comment*,               // Comment text.
                  BlankLine*,             // Blank line.
                  VerilogFunction*,       // Function definition
                  ModuleSection*>;

// A ModuleSection is a container of ModuleMembers used to organize the contents
// of a module. A Module contains a single top-level ModuleSection which may
// contain other ModuleSections. ModuleSections enables modules to be
// constructed a non-linear, random-access fashion by appending members to
// different sections rather than just appending to the end of module.
// TODO(meheff): Move Module methods AddReg*, AddWire*, etc to ModuleSection.
class ModuleSection : public VastNode {
 public:
  explicit ModuleSection(VerilogFile* file) : file_(file) {}

  std::string Emit() const;

  // Constructs and adds a module member of type T to the section. Ownership is
  // maintained by the parent VerilogFile. Templatized on T in order to return a
  // pointer to the derived type.
  template <typename T, typename... Args>
  inline T* Add(Args&&... args);

  template <typename T>
  T AddModuleMember(T member) {
    members_.push_back(member);
    return member;
  }

  // Recursively gathers and returns all ModuleMembers contained in this section
  // or any section contained in this section.
  std::vector<ModuleMember> GatherMembers() const;

  VerilogFile* file() const { return file_; }

 private:
  VerilogFile* file_;
  std::vector<ModuleMember> members_;
};

// Represents a module port.
struct Port {
  static Port FromProto(const PortProto& proto, VerilogFile* f);

  absl::StatusOr<PortProto> ToProto() const;
  const std::string& name() const { return wire->name(); }
  std::string ToString() const;

  Direction direction;
  Def* wire;
};

// Helper for converting a sequence of ports to a string; e.g. for debugging /
// logging.
std::string PortsToString(absl::Span<const Port> ports);

// Returns the flattened number of input/output bits required to represent the
// port set.
absl::StatusOr<int64> GetInputBits(absl::Span<const Port> ports);
absl::StatusOr<int64> GetOutputBits(absl::Span<const Port> ports);

// Represents a module definition.
class Module : public VastNode {
 public:
  Module(absl::string_view name, VerilogFile* parent)
      : parent_(parent), name_(name), top_(parent) {}

  // Constructs and adds a node to the module. Ownership is maintained by the
  // parent VerilogFile. Most constructs should be added to the module. The
  // exceptions are the AddFoo convenience methods defined below which return a
  // Ref to the object created rather than the object itself.
  template <typename T, typename... Args>
  inline T* Add(Args&&... args);

  // Adds a (wire) port to this module with the given direction/name/width and
  // returns a reference to that wire.
  //
  // Note that width is permitted to be nullptr for default width (of a single
  // bit).
  LogicRef* AddPort(Direction direction, absl::string_view name, int64 width);
  LogicRef* AddPortAsExpression(Direction direction, absl::string_view name,
                                Expression* width);

  // Convenience wrappers around AddPort().
  LogicRef1* AddInput(absl::string_view name);
  LogicRef1* AddOutput(absl::string_view name);

  // Adds a reg/wire Def to the module with the given width and initialized with
  // the given value. Returns a reference to the reg/wire.
  LogicRef* AddReg(absl::string_view name, int64 width,
                   absl::optional<int64> init = absl::nullopt,
                   ModuleSection* section = nullptr);
  LogicRef* AddRegAsExpression(absl::string_view name, Expression* width,
                               RegInit init = UninitializedSentinel(),
                               ModuleSection* section = nullptr);

  // Adds a unpacked array register.
  LogicRef* AddUnpackedArrayReg(
      absl::string_view name, Expression* element_width,
      absl::Span<const UnpackedArrayBound> array_bounds,
      RegInit init = UninitializedSentinel(), ModuleSection* section = nullptr);

  LogicRef* AddWire(absl::string_view name, int64 width,
                    ModuleSection* section = nullptr);
  LogicRef* AddWireAsExpression(absl::string_view name, Expression* width,
                                ModuleSection* section = nullptr);

  // Wrappers which construct a single bit Def.
  LogicRef1* AddReg1(absl::string_view name,
                     absl::optional<int64> init = absl::nullopt,
                     ModuleSection* section = nullptr);
  LogicRef1* AddWire1(absl::string_view name, ModuleSection* section = nullptr);

  // Wrappers which construct an 8-bit Def.
  LogicRef8* AddReg8(absl::string_view name,
                     absl::optional<int64> init = absl::nullopt,
                     ModuleSection* section = nullptr);
  LogicRef8* AddWire8(absl::string_view name, ModuleSection* section = nullptr);

  ParameterRef* AddParameter(absl::string_view name, Expression* rhs);

  std::string Emit();

  VerilogFile* parent() const { return parent_; }

  // Adds a previously constructed VAST construct to the module.
  template <typename T>
  T AddModuleMember(T member) {
    top_.AddModuleMember(member);
    return member;
  }

  ModuleSection* top() { return &top_; }

  absl::Span<const Port> ports() const { return ports_; }
  const std::string& name() const { return name_; }

 private:
  std::string Emit(ModuleMember* member);

  VerilogFile* parent_;
  std::string name_;
  std::vector<Port> ports_;

  ModuleSection top_;
};

// Represents a file-level inclusion directive.
class Include : public VastNode {
 public:
  explicit Include(absl::string_view path) : path_(path) {}

  std::string Emit();

 private:
  std::string path_;
};

using FileMember = absl::variant<Module*, Include*>;

// Represents a file (as a Verilog translation-unit equivalent).
class VerilogFile {
 public:
  VerilogFile() {}

  Module* AddModule(absl::string_view name) {
    return Add(Make<Module>(name, this));
  }
  void AddInclude(absl::string_view path) { Add(Make<Include>(path)); }

  template <typename T>
  T* Add(T* member) {
    members_.push_back(member);
    return member;
  }

  template <typename T, typename... Args>
  T* Make(Args&&... args) {
    std::unique_ptr<T> value =
        absl::make_unique<T>(std::forward<Args>(args)...);
    T* ptr = value.get();
    nodes_.push_back(std::move(value));
    return ptr;
  }

  std::string Emit();

  verilog::Slice* Slice(IndexableExpression* subject, Expression* hi,
                        Expression* lo) {
    return Make<verilog::Slice>(subject, hi, lo);
  }
  verilog::Slice* Slice(IndexableExpression* subject, int64 hi, int64 lo) {
    XLS_CHECK_GE(hi, 0);
    XLS_CHECK_GE(lo, 0);
    return Make<verilog::Slice>(subject, MaybePlainLiteral(hi),
                                MaybePlainLiteral(lo));
  }

  verilog::DynamicSlice* DynamicSlice(IndexableExpression* subject,
                                      Expression* start, Expression* width) {
    return Make<verilog::DynamicSlice>(subject, start, width);
  }
  verilog::DynamicSlice* DynamicSlice(IndexableExpression* subject,
                                      Expression* start, int64 width) {
    XLS_CHECK_GT(width, 0);
    return Make<verilog::DynamicSlice>(subject, start,
                                       MaybePlainLiteral(width));
  }

  verilog::Index* Index(IndexableExpression* subject, Expression* index) {
    return Make<verilog::Index>(subject, index);
  }
  verilog::Index* Index(IndexableExpression* subject, int64 index) {
    XLS_CHECK_GE(index, 0);
    return Make<verilog::Index>(subject, MaybePlainLiteral(index));
  }

  Unary* Negate(Expression* expression) {
    return Make<Unary>("-", expression, /*precedence=*/12);
  }
  Unary* BitwiseNot(Expression* expression) {
    return Make<Unary>("~", expression, /*precedence=*/12);
  }
  Unary* LogicalNot(Expression* expression) {
    return Make<Unary>("!", expression, /*precedence=*/12);
  }
  Unary* AndReduce(Expression* expression) {
    return Make<Unary>("&", expression, /*precedence=*/12);
  }
  Unary* OrReduce(Expression* expression) {
    return Make<Unary>("|", expression, /*precedence=*/12);
  }
  Unary* XorReduce(Expression* expression) {
    return Make<Unary>("^", expression, /*precedence=*/12);
  }

  xls::verilog::Concat* Concat(absl::Span<Expression* const> args) {
    return Make<xls::verilog::Concat>(args);
  }
  xls::verilog::Concat* Concat(Expression* replication,
                               absl::Span<Expression* const> args) {
    return Make<xls::verilog::Concat>(replication, args);
  }
  xls::verilog::Concat* Concat(int64 replication,
                               absl::Span<Expression* const> args) {
    return Make<xls::verilog::Concat>(MaybePlainLiteral(replication), args);
  }

  xls::verilog::ArrayAssignmentPattern* ArrayAssignmentPattern(
      absl::Span<Expression* const> args) {
    return Make<xls::verilog::ArrayAssignmentPattern>(args);
  }

  BinaryInfix* Add(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "+", rhs, /*precedence=*/9);
  }
  BinaryInfix* LogicalAnd(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "&&", rhs, /*precedence=*/2);
  }
  BinaryInfix* BitwiseAnd(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "&", rhs, /*precedence=*/5);
  }
  BinaryInfix* NotEquals(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "!=", rhs, /*precedence=*/6);
  }
  BinaryInfix* Equals(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "==", rhs, /*precedence=*/6);
  }
  BinaryInfix* GreaterThanEquals(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, ">=", rhs, /*precedence=*/7);
  }
  BinaryInfix* GreaterThan(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, ">", rhs, /*precedence=*/7);
  }
  BinaryInfix* LessThanEquals(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "<=", rhs, /*precedence=*/7);
  }
  BinaryInfix* LessThan(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "<", rhs, /*precedence=*/7);
  }
  BinaryInfix* Div(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "/", rhs, /*precedence=*/10);
  }
  BinaryInfix* Mul(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "*", rhs, /*precedence=*/10);
  }
  BinaryInfix* BitwiseOr(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "|", rhs, /*precedence=*/3);
  }
  BinaryInfix* LogicalOr(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "||", rhs, /*precedence=*/1);
  }
  BinaryInfix* BitwiseXor(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "^", rhs, /*precedence=*/4);
  }
  BinaryInfix* Shll(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "<<", rhs, /*precedence=*/8);
  }
  BinaryInfix* Shra(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, ">>>", rhs, /*precedence=*/8);
  }
  BinaryInfix* Shrl(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, ">>", rhs, /*precedence=*/8);
  }
  BinaryInfix* Sub(Expression* lhs, Expression* rhs) {
    return Make<BinaryInfix>(lhs, "-", rhs, /*precedence=*/9);
  }

  // Only for use in testing.
  BinaryInfix* NotEqualsX(Expression* lhs) {
    return Make<BinaryInfix>(lhs, "!==", XLiteral(), /*precedence=*/6);
  }
  BinaryInfix* EqualsX(Expression* lhs) {
    return Make<BinaryInfix>(lhs, "===", XLiteral(), /*precedence=*/6);
  }

  verilog::Ternary* Ternary(Expression* cond, Expression* consequent,
                            Expression* alternate) {
    return Make<verilog::Ternary>(cond, consequent, alternate);
  }

  verilog::XLiteral* XLiteral() { return Make<verilog::XLiteral>(); }

  // Creates an literal with the given value and bit_count.
  verilog::Literal* Literal(uint64 value, int64 bit_count,
                            FormatPreference format = FormatPreference::kHex) {
    return Make<verilog::Literal>(UBits(value, bit_count), format);
  }

  // Creates an literal whose value and width is given by a Bits object.
  verilog::Literal* Literal(const Bits& bits,
                            FormatPreference format = FormatPreference::kHex) {
    return Make<verilog::Literal>(bits, format);
  }

  // Creates a one-bit literal.
  verilog::Literal* Literal1(int64 value) {
    // Avoid taking a bool argument as many types implicitly and undesirably
    // convert to bool
    XLS_CHECK((value == 0) || (value == 1));
    return Make<verilog::Literal>(UBits(value, 1), FormatPreference::kHex);
  }

  // Creates a decimal literal representing a plain decimal number without a bit
  // count prefix (e.g., "42"). Use for clarity when bit width does not matter,
  // for example, as bit-slice indices.
  verilog::Literal* PlainLiteral(int32 value) {
    return Make<verilog::Literal>(SBits(value, 32), FormatPreference::kDefault,
                                  /*emit_bit_count=*/false);
  }

 private:
  // Same as PlainLiteral if value fits in an int32. Otherwise creates a 64-bit
  // literal to hold the value.
  verilog::Literal* MaybePlainLiteral(int64 value) {
    if (value >= std::numeric_limits<int32>::min() &&
        value <= std::numeric_limits<int32>::max()) {
      return PlainLiteral(value);
    } else {
      return Literal(SBits(value, 64));
    }
  }

  std::vector<FileMember> members_;
  std::vector<std::unique_ptr<VastNode>> nodes_;
};

template <typename T, typename... Args>
inline T* StatementBlock::Add(Args&&... args) {
  T* ptr = parent_->Make<T>(std::forward<Args>(args)...);
  statements_.push_back(ptr);
  return ptr;
}

template <typename T, typename... Args>
inline T* VerilogFunction::AddStatement(Args&&... args) {
  return statement_block_->Add<T>(std::forward<Args>(args)...);
}

template <typename... Args>
inline LogicRef* VerilogFunction::AddRegDef(Args&&... args) {
  RegDef* ptr = file_->Make<RegDef>(std::forward<Args>(args)...);
  block_reg_defs_.push_back(ptr);
  return file_->Make<LogicRef>(ptr);
}

template <typename T, typename... Args>
inline T* Module::Add(Args&&... args) {
  T* ptr = parent_->Make<T>(std::forward<Args>(args)...);
  AddModuleMember(ptr);
  return ptr;
}

template <typename T, typename... Args>
inline T* ModuleSection::Add(Args&&... args) {
  T* ptr = file_->Make<T>(std::forward<Args>(args)...);
  AddModuleMember(ptr);
  return ptr;
}

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_VAST_H_
