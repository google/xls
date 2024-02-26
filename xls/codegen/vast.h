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

// Subset-of-verilog AST, suitable for combining as datastructures before
// emission.

#ifndef XLS_CODEGEN_VAST_H_
#define XLS_CODEGEN_VAST_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {

enum class FileType {
  kVerilog,
  kSystemVerilog,
};

// Forward declarations.
class VastNode;
class VerilogFile;
class Expression;
class IndexableExpression;
class LogicRef;
class Literal;
class Unary;

class LineSpan {
 public:
  LineSpan(int64_t start_line, int64_t end_line)
      : start_line_(start_line), end_line_(end_line) {}

  int64_t StartLine() const { return start_line_; }

  int64_t EndLine() const { return end_line_; }

  std::string ToString() const {
    return absl::StrCat("(", StartLine(), ", ", EndLine(), ")");
  }

  friend bool operator==(const LineSpan& lhs, const LineSpan& rhs) {
    return lhs.start_line_ == rhs.start_line_ && lhs.end_line_ == rhs.end_line_;
  }

 private:
  int64_t start_line_;
  int64_t end_line_;
};

// A data structure that keeps track of a set of line spans, along with a
// "hanging" span that hasn't been ended yet. When `LineInfo::Start` is called,
// `hanging_start_line` is populated with the current line number, and then when
// `LineInfo::End` is subsequently called, `hanging_start_line` and the current
// line number are used to add a `LineSpan` to `completed_spans` and the
// `hanging_start_line` is set back to `std::nullopt`.
struct PartialLineSpans {
  // A set of nonoverlapping spans.
  std::vector<LineSpan> completed_spans;
  // The start line number of a span that hasn't been ended yet.
  std::optional<int64_t> hanging_start_line;

  // Print this as a string.
  // When there is no `hanging_start_line`, it looks like `[(1, 2), (5, 9)]`.
  // When there is, it looks like `[(1, 2), (5, 9); 12]`
  std::string ToString() const;
};

// A data structure for keeping track of the line numbers that a given VastNode
// generates into.
//
// NOTE: if you don't care about that data, every function in this module that
// accepts a `LineInfo*` can safely accept a `nullptr`.
class LineInfo {
 public:
  LineInfo() = default;

  // Start recording a region in which the given node is active.
  // CHECK fails if called multiple times with no intervening `End` calls.
  void Start(const VastNode* node);

  // Stop recording a region in which the given node is active.
  // CHECK fails if `Start` has not been called an odd number of times on this
  // node previous to this call.
  // CHECK fails if called multiple times with no intervening `Start` calls.
  void End(const VastNode* node);

  // Increase the current line number by the given value.
  // You may pass in a negative number, but only if that sequence of calls
  // could equivalently be achieved through only nonnegative numbers.
  //
  // For example,
  // `Start(x), End(x), Increase(4), Increase(-2), Start(y), End(y)`
  // is a valid sequence of calls, since it is equivalent to
  // `Start(x), End(x), Increase(2), Start(y), End(y)`.
  //
  // On the other hand,
  // `Start(x), End(x), Increase(-2), Start(y), End(y)`
  // is an invalid sequence of calls, since there is no semantically equivalent
  // sequence of calls that does not include negative numbers.
  void Increase(int64_t delta);

  // Returns the underlying relation between nodes and spans.
  const absl::flat_hash_map<const VastNode*, PartialLineSpans>& Spans() const {
    return spans_;
  }

  // Returns the line spans that a given node covers.
  // Returns std::nullopt if the given node has a hanging span (Start w/o End).
  // Returns std::nullopt if the given node was never recorded.
  std::optional<std::vector<LineSpan>> LookupNode(const VastNode* node) const;

 private:
  int64_t current_line_number_ = 0;
  absl::flat_hash_map<const VastNode*, PartialLineSpans> spans_;
};

// Returns a sanitized identifier string based on the given name. Invalid
// characters are replaced with '_'.
std::string SanitizeIdentifier(std::string_view name);

// Base type for a VAST node. All nodes are owned by a VerilogFile.
class VastNode {
 public:
  explicit VastNode(VerilogFile* file, const SourceInfo& loc)
      : file_(file), loc_(loc) {}
  virtual ~VastNode() = default;

  // The file which owns this node.
  VerilogFile* file() const { return file_; }

  const SourceInfo& loc() const { return loc_; }

  virtual std::string Emit(LineInfo* line_info) const = 0;

 private:
  VerilogFile* file_;
  SourceInfo loc_;
};

// Trait used for named entities.
class NamedTrait : public VastNode {
 public:
  using VastNode::VastNode;

  // Returns a name that can be used to refer to the object in generated Verilog
  // code; e.g. for a macro this would return "`THING", for a wire it would
  // return "wire_name".
  virtual std::string GetName() const = 0;
};

// Represents a behavioral statement.
class Statement : public VastNode {
 public:
  using VastNode::VastNode;
};

// Represents the data type of a net/variable/argument/etc in Verilog.
class DataType : public VastNode {
 public:
  DataType(VerilogFile* file, const SourceInfo& loc) : VastNode(file, loc) {}

  // Returns whether this is a scalar signal type (for example, "wire foo").
  virtual bool IsScalar() const { return false; }

  // Returns the width of the def (not counting packed or unpacked dimensions)
  // as an int64_t. Returns an error if this is not possible because the width
  // is not a literal. For example, the width of the following def is 8:
  //   wire [7:0][3:0][42:0] foo;
  virtual absl::StatusOr<int64_t> WidthAsInt64() const = 0;

  // Return flattened bit count of the def (total width of the def including
  // packed dimensions). For example, the following has a flat bit count of
  // 8 * 4 * 5 = 160:
  //   wire [7:0][3:0][4:0] foo;
  // Returns an error if this computation is not possible because the width or a
  // packed dimension is not a literal.
  virtual absl::StatusOr<int64_t> FlatBitCountAsInt64() const = 0;

  // Returns the width expression for the type. Scalars (e.g.,
  // "wire foo;") and integers return std::nullopt.
  virtual std::optional<Expression*> width() const = 0;

  virtual bool is_signed() const { return false; }

  std::string Emit(LineInfo* line_info) const override {
    XLS_LOG(FATAL) << "EmitWithIdentifier should be called rather than emit";
  }

  // Returns a string which denotes this type along with an identifier for use
  // in definitions, arguments, etc. Example output if identifier is 'foo':
  //
  //  [7:0] foo
  //  signed [123:0] foo
  //  [1:0][33:0][44:0] foo [32][111]
  //
  // This method is required rather than simply Emit because an identifier
  // string is nested within the string describing the type.
  virtual std::string EmitWithIdentifier(LineInfo* line_info,
                                         std::string_view identifier) const = 0;
};

// Represents a scalar type. Example:
//   wire foo;
class ScalarType : public DataType {
 public:
  ScalarType(VerilogFile* file, const SourceInfo& loc) : DataType(file, loc) {}

  bool IsScalar() const override { return true; }
  absl::StatusOr<int64_t> WidthAsInt64() const override { return 1; }
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const override { return 1; }
  std::optional<Expression*> width() const override { return std::nullopt; }
  std::string EmitWithIdentifier(LineInfo* line_info,
                                 std::string_view identifier) const override;
};

// Represents an integer type. Example:
//   integer foo;
class IntegerType : public DataType {
 public:
  IntegerType(VerilogFile* file, const SourceInfo& loc) : DataType(file, loc) {}

  bool IsScalar() const override { return false; }
  absl::StatusOr<int64_t> WidthAsInt64() const override {
    return absl::InvalidArgumentError("Cannot get width of integer types");
  }
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const override {
    return absl::InvalidArgumentError(
        "Cannot get flat bit count of integer types");
  }
  std::optional<Expression*> width() const override { return std::nullopt; }
  std::string EmitWithIdentifier(LineInfo* line_info,
                                 std::string_view identifier) const override;
};

// Represents a bit-vector type. Example:
//   reg[7:0] foo;
class BitVectorType : public DataType {
 public:
  BitVectorType(Expression* width, bool is_signed, VerilogFile* file,
                const SourceInfo& loc)
      : DataType(file, loc), width_(width), is_signed_(is_signed) {}
  BitVectorType(int64_t width, bool is_signed, VerilogFile* file,
                const SourceInfo& loc);

  bool IsScalar() const override { return false; }
  absl::StatusOr<int64_t> WidthAsInt64() const override;
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const override;
  std::optional<Expression*> width() const override { return width_; }
  bool is_signed() const override { return is_signed_; }
  std::string EmitWithIdentifier(LineInfo* line_info,
                                 std::string_view identifier) const override;

 private:
  Expression* width_;
  bool is_signed_;
};

// Represents a packed array of bit-vectors type. Example:
//   wire [7:0][42:0][3:0] foo;
class PackedArrayType : public DataType {
 public:
  PackedArrayType(Expression* width, absl::Span<Expression* const> packed_dims,
                  bool is_signed, VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc),
        width_(width),
        is_signed_(is_signed),
        packed_dims_(packed_dims.begin(), packed_dims.end()) {
    XLS_CHECK(!packed_dims.empty());
  }
  PackedArrayType(int64_t width, absl::Span<const int64_t> packed_dims,
                  bool is_signed, VerilogFile* file, const SourceInfo& loc);

  bool IsScalar() const override { return false; }
  absl::StatusOr<int64_t> WidthAsInt64() const override;
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const override;
  std::optional<Expression*> width() const override { return width_; }
  bool is_signed() const override { return is_signed_; }
  std::string EmitWithIdentifier(LineInfo* line_info,
                                 std::string_view identifier) const override;

  // Returns the packed dimensions for the type. For example, the net type for
  // "wire [7:0][42:0][3:0] foo;" has {43, 4} as the packed dimensions. The
  // inner-most dimension is not included as that is considered for type
  // purposes as the underlying array element type ("wire [7:0]"). The ordering
  // of indices matches the order in which they are emitted in the emitted
  // Verilog.
  absl::Span<Expression* const> packed_dims() const { return packed_dims_; }

 private:
  Expression* width_;
  bool is_signed_;
  std::vector<Expression*> packed_dims_;
};

// Represents an unpacked array of bit-vectors or packed array types. Example:
//   wire [7:0][42:0][3:0] foo [1:0];
// Where [1:0] is the unpacked array dimensions.
class UnpackedArrayType : public DataType {
 public:
  UnpackedArrayType(DataType* element_type,
                    absl::Span<Expression* const> unpacked_dims,
                    VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc),
        element_type_(element_type),
        unpacked_dims_(unpacked_dims.begin(), unpacked_dims.end()) {
    XLS_CHECK(!unpacked_dims.empty());
    XLS_CHECK(dynamic_cast<UnpackedArrayType*>(element_type) == nullptr);
  }
  UnpackedArrayType(DataType* element_type,
                    absl::Span<const int64_t> packed_dims, VerilogFile* file,
                    const SourceInfo& loc);

  bool IsScalar() const override { return false; }
  absl::StatusOr<int64_t> WidthAsInt64() const override;
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const override;
  std::optional<Expression*> width() const override {
    return element_type_->width();
  }
  bool is_signed() const override { return element_type_->is_signed(); }
  std::string EmitWithIdentifier(LineInfo* line_info,
                                 std::string_view identifier) const override;

  // Returns the unpacked dimensions for the type. For example, the net type for
  // "wire [7:0][42:0] foo [123][7];" has {123, 7} as the unpacked
  // dimensions.
  absl::Span<Expression* const> unpacked_dims() const { return unpacked_dims_; }

 private:
  DataType* element_type_;
  std::vector<Expression*> unpacked_dims_;
};

// The kind of a net/variable. kReg, kWire, kLogic can be arbitrarily
// typed. kInteger definitions can only be of IntegerType.
enum class DataKind : int8_t { kReg, kWire, kLogic, kInteger };

// Represents the definition of a variable or net.
class Def : public Statement {
 public:
  // Constructor for a single-bit signal without a range (width) specification.
  // Examples:
  //   wire foo;
  //   reg bar;
  Def(std::string_view name, DataKind data_kind, DataType* data_type,
      VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        name_(name),
        data_kind_(data_kind),
        data_type_(std::move(data_type)) {}

  std::string Emit(LineInfo* line_info) const override;

  // Emit the definition without the trailing semicolon.
  std::string EmitNoSemi(LineInfo* line_info) const;

  const std::string& GetName() const { return name_; }
  DataKind data_kind() const { return data_kind_; }
  DataType* data_type() const { return data_type_; }

 private:
  std::string name_;
  DataKind data_kind_;
  DataType* data_type_;
};

// A wire definition. Example:
//   wire [41:0] foo;
class WireDef : public Def {
 public:
  WireDef(std::string_view name, DataType* data_type, VerilogFile* file,
          const SourceInfo& loc)
      : Def(name, DataKind::kWire, data_type, file, loc), init_(nullptr) {}
  WireDef(std::string_view name, DataType* data_type, Expression* init,
          VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kWire, data_type, file, loc), init_(init) {}
  std::string Emit(LineInfo* line_info) const override;

 protected:
  Expression* init_;
};

// Register variable definition.Example:
//   reg [41:0] foo;
class RegDef : public Def {
 public:
  RegDef(std::string_view name, DataType* data_type, VerilogFile* file,
         const SourceInfo& loc)
      : Def(name, DataKind::kReg, data_type, file, loc), init_(nullptr) {}
  RegDef(std::string_view name, DataType* data_type, Expression* init,
         VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kReg, data_type, file, loc), init_(init) {}

  std::string Emit(LineInfo* line_info) const override;

 protected:
  Expression* init_;
};

// Logic variable definition.Example:
//   logic [41:0] foo;
class LogicDef : public Def {
 public:
  LogicDef(std::string_view name, DataType* data_type, VerilogFile* file,
           const SourceInfo& loc)
      : Def(name, DataKind::kLogic, data_type, file, loc), init_(nullptr) {}
  LogicDef(std::string_view name, DataType* data_type, Expression* init,
           VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kLogic, data_type, file, loc), init_(init) {}

  std::string Emit(LineInfo* line_info) const override;

 protected:
  Expression* init_;
};

// Integer variable definition.Example:
//   integer foo;
class IntegerDef : public Def {
 public:
  IntegerDef(std::string_view name, VerilogFile* file, const SourceInfo& loc);
  IntegerDef(std::string_view name, DataType* data_type, Expression* init,
             VerilogFile* file, const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const override;

 protected:
  Expression* init_;
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
  DelayStatement(Expression* delay, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), delay_(delay), delayed_statement_(nullptr) {}
  DelayStatement(Expression* delay, Statement* delayed_statement,
                 VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        delay_(delay),
        delayed_statement_(delayed_statement) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* delay_;
  Statement* delayed_statement_;
};

// Represents a wait statement.
class WaitStatement : public Statement {
 public:
  WaitStatement(Expression* event, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), event_(event) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* event_;
};

// Represents a forever construct which runs a statement continuously.
class Forever : public Statement {
 public:
  Forever(Statement* statement, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), statement_(statement) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Statement* statement_;
};

// Represents a blocking assignment ("lhs = rhs;")
class BlockingAssignment : public Statement {
 public:
  BlockingAssignment(Expression* lhs, Expression* rhs, VerilogFile* file,
                     const SourceInfo& loc)
      : Statement(file, loc), lhs_(ABSL_DIE_IF_NULL(lhs)), rhs_(rhs) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* lhs_;
  Expression* rhs_;
};

// Represents a nonblocking assignment  ("lhs <= rhs;").
class NonblockingAssignment : public Statement {
 public:
  NonblockingAssignment(Expression* lhs, Expression* rhs, VerilogFile* file,
                        const SourceInfo& loc)
      : Statement(file, loc), lhs_(ABSL_DIE_IF_NULL(lhs)), rhs_(rhs) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* lhs_;
  Expression* rhs_;
};

// An abstraction representing a sequence of statements within a structured
// procedure (e.g., an "always" statement).
class StatementBlock : public VastNode {
 public:
  using VastNode::VastNode;

  // Constructs and adds a statement to the block. Ownership is maintained by
  // the parent VerilogFile. Example:
  //   Case* c = Add<Case>(subject);
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::vector<Statement*> statements_;
};

// Represents a 'default' case arm label.
struct DefaultSentinel {};

// Represents a label within a case statement.
using CaseLabel = std::variant<Expression*, DefaultSentinel>;

// Represents an arm of a case statement.
class CaseArm : public VastNode {
 public:
  CaseArm(CaseLabel label, VerilogFile* file, const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const override;
  StatementBlock* statements() { return statements_; }

 private:
  CaseLabel label_;
  StatementBlock* statements_;
};

enum class CaseKeyword {
  kCase,
  kCasez,
};

enum class CaseModifier {
  kUnique,
};

struct CaseType {
  CaseKeyword keyword;
  std::optional<CaseModifier> modifier;

  explicit CaseType(CaseKeyword keyword, std::optional<CaseModifier> modifier)
      : keyword(keyword), modifier(modifier) {}
  explicit CaseType(CaseKeyword keyword)
      : keyword(keyword), modifier(std::nullopt) {}
};

// Represents a case statement.
class Case : public Statement {
 public:
  Case(Expression* subject, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        subject_(subject),
        case_type_(CaseType(CaseKeyword::kCase)) {}

  Case(Expression* subject, CaseType case_type, VerilogFile* file,
       const SourceInfo& loc)
      : Statement(file, loc), subject_(subject), case_type_(case_type) {}

  StatementBlock* AddCaseArm(CaseLabel label);

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* subject_;
  std::vector<CaseArm*> arms_;
  CaseType case_type_;
};

// Represents an if statement with optional "else if" and "else" blocks.
class Conditional : public Statement {
 public:
  Conditional(Expression* condition, VerilogFile* file, const SourceInfo& loc);

  // Returns a pointer to the statement block of the consequent.
  StatementBlock* consequent() const { return consequent_; }

  // Adds an alternate clause ("else if" or "else") and returns a pointer to the
  // consequent. The alternate is final (an "else") if condition is null. Dies
  // if a final alternate ("else") clause has been previously added.
  StatementBlock* AddAlternate(Expression* condition = nullptr);

  std::string Emit(LineInfo* line_info) const override;

 private:
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
  WhileStatement(Expression* condition, VerilogFile* file,
                 const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const override;

  StatementBlock* statements() const { return statements_; }

 private:
  Expression* condition_;
  StatementBlock* statements_;
};

// Represents a repeat construct.
class RepeatStatement : public Statement {
 public:
  RepeatStatement(Expression* repeat_count, VerilogFile* file,
                  const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const override;

  StatementBlock* statements() const { return statements_; }

 private:
  Expression* repeat_count_;
  StatementBlock* statements_;
};

// Represents an event control statement. This is represented as "@(...);" where
// "..." is the event expression..
class EventControl : public Statement {
 public:
  EventControl(Expression* event_expression, VerilogFile* file,
               const SourceInfo& loc)
      : Statement(file, loc), event_expression_(event_expression) {}

  std::string Emit(LineInfo* line_info) const override;

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
  using VastNode::VastNode;

  virtual bool IsLiteral() const { return false; }

  // Returns true if the node is a literal with the given unsigned value.
  virtual bool IsLiteralWithValue(int64_t target) const { return false; }

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
  static constexpr int64_t kMaxPrecedence = 13;
  static constexpr int64_t kMinPrecedence = -1;
  virtual int64_t precedence() const { return kMaxPrecedence; }
};

// Represents an X value.
class XSentinel : public Expression {
 public:
  XSentinel(int64_t width, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), width_(width) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  int64_t width_;
};

// Specifies the four possible values a bit can have under SystemVerilog's
// 4-valued logic.
enum class FourValueBit {
  kZero,
  kOne,
  kUnknown,
  kHighZ,
};

// Represents a four-valued binary literal, e.g. 4'b0110, 4'b1???, or 4'b01?X.
class FourValueBinaryLiteral : public Expression {
 public:
  FourValueBinaryLiteral(absl::Span<FourValueBit const> value,
                         VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), bits_(value.begin(), value.end()) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::vector<FourValueBit> bits_;
};

// Represents an operation (unary, binary, etc) with a particular precedence.
class Operator : public Expression {
 public:
  Operator(int64_t precedence, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), precedence_(precedence) {}
  int64_t precedence() const override { return precedence_; }

 private:
  int64_t precedence_;
};

// A posedge edge identifier expression.
class PosEdge : public Expression {
 public:
  PosEdge(Expression* expression, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), expression_(expression) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* expression_;
};

// A negedge edge identifier expression.
class NegEdge : public Expression {
 public:
  NegEdge(Expression* expression, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), expression_(expression) {}

  std::string Emit(LineInfo* line_info) const override;

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
  Instantiation(std::string_view module_name, std::string_view instance_name,
                absl::Span<const Connection> parameters,
                absl::Span<const Connection> connections, VerilogFile* file,
                const SourceInfo& loc)
      : VastNode(file, loc),
        module_name_(module_name),
        instance_name_(instance_name),
        parameters_(parameters.begin(), parameters.end()),
        connections_(connections.begin(), connections.end()) {}

  std::string Emit(LineInfo* line_info) const override;

 protected:
  std::string module_name_;
  std::string instance_name_;
  std::vector<Connection> parameters_;
  std::vector<Connection> connections_;
};

// Template instantiation for FFI.
// TODO(hzeller) 2023-06-12 Longer term, this should not be needed.
class TemplateInstantiation : public Instantiation {
 public:
  TemplateInstantiation(std::string_view instance_name,
                        std::string_view code_template,
                        absl::Span<const Connection> connections,
                        VerilogFile* file, const SourceInfo& loc)
      : Instantiation("", instance_name,
                      /*parameters=*/{}, connections, file, loc),
        template_text_(code_template) {
    // Not using module name as this is provided in user-template
    // Not using parameters, they are indistinguishable from other connections.
  }

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string template_text_;
};

// Represents a reference to an already-defined macro. For example: `MY_MACRO.
class MacroRef : public Expression {
 public:
  MacroRef(std::string_view name, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), name_(name) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string name_;
};

// Defines a module parameter.
class Parameter : public NamedTrait {
 public:
  Parameter(std::string_view name, Expression* rhs, VerilogFile* file,
            const SourceInfo& loc)
      : NamedTrait(file, loc), name_(name), rhs_(rhs) {}

  std::string Emit(LineInfo* line_info) const override;
  std::string GetName() const override { return name_; }

 private:
  std::string name_;
  Expression* rhs_;
};

// Defines an item in a localparam.
class LocalParamItem : public NamedTrait {
 public:
  LocalParamItem(std::string_view name, Expression* rhs, VerilogFile* file,
                 const SourceInfo& loc)
      : NamedTrait(file, loc), name_(name), rhs_(rhs) {}

  std::string GetName() const override { return name_; }

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string name_;
  Expression* rhs_;
};

// Refers to an item in a localparam for use in expressions.
class LocalParamItemRef : public Expression {
 public:
  LocalParamItemRef(LocalParamItem* item, VerilogFile* file,
                    const SourceInfo& loc)
      : Expression(file, loc), item_(item) {}

  std::string Emit(LineInfo* line_info) const override {
    return item_->GetName();
  }

 private:
  LocalParamItem* item_;
};

// Defines a localparam.
class LocalParam : public VastNode {
 public:
  using VastNode::VastNode;
  LocalParamItemRef* AddItem(std::string_view name, Expression* value,
                             const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::vector<LocalParamItem*> items_;
};

// Refers to a Parameter's definition for use in an expression.
class ParameterRef : public Expression {
 public:
  ParameterRef(Parameter* parameter, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), parameter_(parameter) {}

  std::string Emit(LineInfo* line_info) const override {
    return parameter_->GetName();
  }

 private:
  Parameter* parameter_;
};

// An indexable expression that can be bit-sliced or indexed.
class IndexableExpression : public Expression {
 public:
  using Expression::Expression;

  bool IsIndexableExpression() const override { return true; }
};

// Reference to the definition of a WireDef, RegDef, or LogicDef.
class LogicRef : public IndexableExpression {
 public:
  LogicRef(Def* def, VerilogFile* file, const SourceInfo& loc)
      : IndexableExpression(file, loc), def_(ABSL_DIE_IF_NULL(def)) {}

  bool IsLogicRef() const override { return true; }

  std::string Emit(LineInfo* line_info) const override {
    return def_->GetName();
  }

  // Returns the Def that this LogicRef refers to.
  Def* def() const { return def_; }

  // Returns the name of the underlying Def this object refers to.
  std::string GetName() const { return def()->GetName(); }

 private:
  // Logic signal definition.
  Def* def_;
};

// Represents a Verilog unary expression.
class Unary : public Operator {
 public:
  Unary(std::string_view op, Expression* arg, int64_t precedence,
        VerilogFile* file, const SourceInfo& loc)
      : Operator(precedence, file, loc), op_(op), arg_(arg) {}

  bool IsUnary() const override { return true; }

  // Returns true if this is reduction operation (OR, NOR, AND, NAND, XOR or
  // XNOR).
  bool IsReduction() const {
    return op_ == "|" || op_ == "~|" || op_ == "&" || op_ == "~&" ||
           op_ == "^" || op_ == "~^";
  }
  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string op_;
  Expression* arg_;
};

// Abstraction describing a reset signal.
// TODO(https://github.com/google/xls/issues/317): This belongs at a higher
// level of abstraction.
struct Reset {
  LogicRef* signal;
  bool asynchronous;
  bool active_low;
};

// Defines an always_ff-equivalent block.
// TODO(https://github.com/google/xls/issues/317): Replace uses of AlwaysFlop
// with Always or AlwaysFf. AlwaysFlop has a higher level of abstraction which
// is now better handled by ModuleBuilder.
class AlwaysFlop : public VastNode {
 public:
  AlwaysFlop(LogicRef* clk, VerilogFile* file, const SourceInfo& loc);
  AlwaysFlop(LogicRef* clk, Reset rst, VerilogFile* file,
             const SourceInfo& loc);

  // Add a register controlled by this AlwaysFlop. 'reset_value' can only have a
  // value if the AlwaysFlop has a reset signal.
  void AddRegister(LogicRef* reg, Expression* reg_next, const SourceInfo& loc,
                   Expression* reset_value = nullptr);

  std::string Emit(LineInfo* line_info) const override;

 private:
  LogicRef* clk_;
  std::optional<Reset> rst_;
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
  explicit StructuredProcedure(VerilogFile* file, const SourceInfo& loc);

  StatementBlock* statements() { return statements_; }

 protected:
  StatementBlock* statements_;
};

// Represents the '*' which can occur in an always sensitivity list.
struct ImplicitEventExpression {};

// Elements which can appear in a sensitivity list for an always or always_ff
// block.
using SensitivityListElement =
    std::variant<ImplicitEventExpression, PosEdge*, NegEdge*, LogicRef*>;

// Base class for 'always' style blocks with a sensitivity list.
class AlwaysBase : public StructuredProcedure {
 public:
  AlwaysBase(absl::Span<const SensitivityListElement> sensitivity_list,
             VerilogFile* file, const SourceInfo& loc)
      : StructuredProcedure(file, loc),
        sensitivity_list_(sensitivity_list.begin(), sensitivity_list.end()) {}
  std::string Emit(LineInfo* line_info) const override;

 protected:
  virtual std::string name() const = 0;

  std::vector<SensitivityListElement> sensitivity_list_;
};

// Defines an always block.
class Always : public AlwaysBase {
 public:
  Always(absl::Span<const SensitivityListElement> sensitivity_list,
         VerilogFile* file, const SourceInfo& loc)
      : AlwaysBase(sensitivity_list, file, loc) {}

 protected:
  std::string name() const override { return "always"; }
};

// Defines an always_comb block.
class AlwaysComb : public AlwaysBase {
 public:
  explicit AlwaysComb(VerilogFile* file, const SourceInfo& loc)
      : AlwaysBase({}, file, loc) {}
  std::string Emit(LineInfo* line_info) const override;

 protected:
  std::string name() const override { return "always_comb"; }
};

// Defines an always_ff block.
class AlwaysFf : public AlwaysBase {
 public:
  using AlwaysBase::AlwaysBase;

 protected:
  std::string name() const override { return "always_ff"; }
};

// Defines an 'initial' block.
class Initial : public StructuredProcedure {
 public:
  using StructuredProcedure::StructuredProcedure;

  std::string Emit(LineInfo* line_info) const override;
};

class Concat : public Expression {
 public:
  Concat(absl::Span<Expression* const> args, VerilogFile* file,
         const SourceInfo& loc)
      : Expression(file, loc),
        args_(args.begin(), args.end()),
        replication_(nullptr) {}

  // Defines a concatenation with replication. Example: {3{1'b101}}
  Concat(Expression* replication, absl::Span<Expression* const> args,
         VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        args_(args.begin(), args.end()),
        replication_(replication) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::vector<Expression*> args_;
  Expression* replication_;
};

// An array assignment pattern such as: "'{foo, bar, baz}"
class ArrayAssignmentPattern : public IndexableExpression {
 public:
  ArrayAssignmentPattern(absl::Span<Expression* const> args, VerilogFile* file,
                         const SourceInfo& loc)
      : IndexableExpression(file, loc), args_(args.begin(), args.end()) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::vector<Expression*> args_;
};

class BinaryInfix : public Operator {
 public:
  BinaryInfix(Expression* lhs, std::string_view op, Expression* rhs,
              int64_t precedence, VerilogFile* file, const SourceInfo& loc)
      : Operator(precedence, file, loc),
        op_(op),
        lhs_(ABSL_DIE_IF_NULL(lhs)),
        rhs_(ABSL_DIE_IF_NULL(rhs)) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string op_;
  Expression* lhs_;
  Expression* rhs_;
};

// Defines a literal value (width and value).
class Literal : public Expression {
 public:
  Literal(Bits bits, FormatPreference format, VerilogFile* file,
          const SourceInfo& loc)
      : Expression(file, loc),
        bits_(bits),
        format_(format),
        emit_bit_count_(true) {}

  Literal(Bits bits, FormatPreference format, bool emit_bit_count,
          VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        bits_(bits),
        format_(format),
        emit_bit_count_(emit_bit_count) {
    XLS_CHECK(emit_bit_count_ || bits.bit_count() == 32);
  }

  std::string Emit(LineInfo* line_info) const override;

  const Bits& bits() const { return bits_; }

  bool IsLiteral() const override { return true; }
  bool IsLiteralWithValue(int64_t target) const override;

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
  QuotedString(std::string_view str, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), str_(str) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string str_;
};

class XLiteral : public Expression {
 public:
  using Expression::Expression;

  std::string Emit(LineInfo* line_info) const override { return "'X"; }
};

// Represents a Verilog slice expression; e.g.
//
//    subject[hi:lo]
class Slice : public Expression {
 public:
  Slice(IndexableExpression* subject, Expression* hi, Expression* lo,
        VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), subject_(subject), hi_(hi), lo_(lo) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  IndexableExpression* subject_;
  Expression* hi_;
  Expression* lo_;
};

// Represents a Verilog indexed part-select expression; e.g.
//
//    subject[start +: width]
class PartSelect : public Expression {
 public:
  PartSelect(IndexableExpression* subject, Expression* start, Expression* width,
             VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        subject_(subject),
        start_(start),
        width_(width) {}

  std::string Emit(LineInfo* line_info) const override;

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
  Index(IndexableExpression* subject, Expression* index, VerilogFile* file,
        const SourceInfo& loc)
      : IndexableExpression(file, loc), subject_(subject), index_(index) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  IndexableExpression* subject_;
  Expression* index_;
};

// Represents a Verilog ternary operator; e.g.
//
//    test ? consequent : alternate
class Ternary : public Expression {
 public:
  Ternary(Expression* test, Expression* consequent, Expression* alternate,
          VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        test_(test),
        consequent_(consequent),
        alternate_(alternate) {}

  std::string Emit(LineInfo* line_info) const override;
  int64_t precedence() const override { return 0; }

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
  ContinuousAssignment(Expression* lhs, Expression* rhs, VerilogFile* file,
                       const SourceInfo& loc)
      : VastNode(file, loc), lhs_(lhs), rhs_(rhs) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* lhs_;
  Expression* rhs_;
};

class BlankLine : public Statement {
 public:
  BlankLine(VerilogFile* file, const SourceInfo& loc) : Statement(file, loc) {}
  using Statement::Statement;

  std::string Emit(LineInfo* line_info) const override { return ""; }
};

// Represents a SystemVerilog concurrent assert statement of the following
// form:
//
//   [LABEL:] assert property (
//       @(CLOCKING_EVENT)
//       [disable iff DISABLE_IFF] CONDITION)
//     else $error(0, message);
class ConcurrentAssertion : public Statement {
 public:
  ConcurrentAssertion(Expression* condition, Expression* clocking_event,
                      std::optional<Expression*> disable_iff,
                      std::string_view label, std::string_view error_message,
                      VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        condition_(condition),
        clocking_event_(clocking_event),
        disable_iff_(disable_iff),
        label_(label),
        error_message_(error_message) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  Expression* condition_;
  Expression* clocking_event_;
  std::optional<Expression*> disable_iff_;
  std::string label_;
  std::string error_message_;
};

// Represents a SystemVerilog cover properly statement, such as
//
// ```
//   cover property <name> (clk, <condition>)
// ```
//
// Such a statement will cause the simulator to count the number of times that
// the given condition is true, and associate that value with <name>.
class Cover : public Statement {
 public:
  Cover(LogicRef* clk, Expression* condition, std::string_view label,
        VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), clk_(clk), condition_(condition), label_(label) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  LogicRef* clk_;
  Expression* condition_;
  std::string label_;
};

// Places a comment in statement position (we can think of comments as
// meaningless expression statements that do nothing).
class Comment : public Statement {
 public:
  Comment(std::string_view text, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), text_(text) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string text_;
};

// A string which is emitted verbatim in the position of a statement.
class InlineVerilogStatement : public Statement {
 public:
  InlineVerilogStatement(std::string_view text, VerilogFile* file,
                         const SourceInfo& loc)
      : Statement(file, loc), text_(text) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string text_;
};

// A reference to a definition defined within a InlineVerilogStatement. Can be
// used where a VAST node Ref is needed to refer to a value (wire, reg, etc)
// defined in a InlineVerilogStatement.
class InlineVerilogRef : public IndexableExpression {
 public:
  InlineVerilogRef(std::string_view name, InlineVerilogStatement* raw_statement,
                   VerilogFile* file, const SourceInfo& loc)
      : IndexableExpression(file, loc), name_(name) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string name_;
};

// Represents call of a system task such as $display.
class SystemTaskCall : public Statement {
 public:
  // An argumentless invocation of a system task such as: $finish;
  SystemTaskCall(std::string_view name, VerilogFile* file,
                 const SourceInfo& loc)
      : Statement(file, loc), name_(name) {}

  // An invocation of a system task with arguments.
  SystemTaskCall(std::string_view name, absl::Span<Expression* const> args,
                 VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        name_(name),
        args_(std::vector<Expression*>(args.begin(), args.end())) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string name_;
  std::optional<std::vector<Expression*>> args_;
};

// Represents statement function call expression such as $time.
class SystemFunctionCall : public Expression {
 public:
  // An argumentless invocation of a system function such as: $time;
  SystemFunctionCall(std::string_view name, VerilogFile* file,
                     const SourceInfo& loc)
      : Expression(file, loc), name_(name) {}

  // An invocation of a system function with arguments.
  SystemFunctionCall(std::string_view name, absl::Span<Expression* const> args,
                     VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        name_(name),
        args_(std::vector<Expression*>(args.begin(), args.end())) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string name_;
  std::optional<std::vector<Expression*>> args_;
};

// Represents a $display function call.
class Display : public SystemTaskCall {
 public:
  Display(absl::Span<Expression* const> args, VerilogFile* file,
          const SourceInfo& loc)
      : SystemTaskCall("display", args, file, loc) {}
};

// Represents a $strobe function call.
class Strobe : public SystemTaskCall {
 public:
  Strobe(absl::Span<Expression* const> args, VerilogFile* file,
         const SourceInfo& loc)
      : SystemTaskCall("strobe", args, file, loc) {}
};

// Represents a $monitor function call.
class Monitor : public SystemTaskCall {
 public:
  Monitor(absl::Span<Expression* const> args, VerilogFile* file,
          const SourceInfo& loc)
      : SystemTaskCall("monitor", args, file, loc) {}
};

// Represents a $finish function call.
class Finish : public SystemTaskCall {
 public:
  Finish(VerilogFile* file, const SourceInfo& loc)
      : SystemTaskCall("finish", file, loc) {}
};

// Represents a $signed function call which casts its argument to signed.
class SignedCast : public SystemFunctionCall {
 public:
  SignedCast(Expression* value, VerilogFile* file, const SourceInfo& loc)
      : SystemFunctionCall("signed", {value}, file, loc) {}
};

// Represents a $unsigned function call which casts its argument to unsigned.
class UnsignedCast : public SystemFunctionCall {
 public:
  UnsignedCast(Expression* value, VerilogFile* file, const SourceInfo& loc)
      : SystemFunctionCall("unsigned", {value}, file, loc) {}
};

// Represents the definition of a Verilog function.
class VerilogFunction : public VastNode {
 public:
  VerilogFunction(std::string_view name, DataType* result_type,
                  VerilogFile* file, const SourceInfo& loc);

  // Adds an argument to the function and returns a reference to its value which
  // can be used in the body of the function.
  LogicRef* AddArgument(std::string_view name, DataType* type,
                        const SourceInfo& loc);

  // Adds a RegDef to the function and returns a LogicRef to it. This should be
  // used for adding RegDefs to the function instead of AddStatement because
  // the RegDefs need to appear outside the statement block (begin/end block).
  template <typename... Args>
  inline LogicRef* AddRegDef(const SourceInfo& loc, Args&&... args);

  // Constructs and adds a statement to the block. Ownership is maintained by
  // the parent VerilogFile. Example:
  //   Case* c = Add<Case>(subject);
  template <typename T, typename... Args>
  inline T* AddStatement(const SourceInfo& loc, Args&&... args);

  // Creates and returns a reference to the variable representing the return
  // value of the function. Assigning to this reference sets the return value of
  // the function.
  LogicRef* return_value_ref();

  // Returns the name of the function.
  std::string name() const { return name_; }

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string name_;
  RegDef* return_value_def_;

  // The block containing all of the statements of the function. SystemVerilog
  // allows multiple statements in a function without encapsulating them in a
  // begin/end block (StatementBlock), but Verilog does not so we choose the
  // least common denominator.
  StatementBlock* statement_block_;

  std::vector<RegDef*> argument_defs_;

  // The RegDefs of reg's defined in the function. These are emitted before the
  // statement block.
  std::vector<RegDef*> block_reg_defs_;
};

// Represents a call to a VerilogFunction.
class VerilogFunctionCall : public Expression {
 public:
  VerilogFunctionCall(VerilogFunction* func, absl::Span<Expression* const> args,
                      VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), func_(func), args_(args.begin(), args.end()) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  VerilogFunction* func_;
  std::vector<Expression*> args_;
};

class ModuleSection;

// Represents a member of a module.
using ModuleMember =
    std::variant<Def*,                     // Logic definition.
                 LocalParam*,              // Module-local parameter.
                 Parameter*,               // Module parameter.
                 Instantiation*,           // module instantiaion.
                 ContinuousAssignment*,    // Continuous assignment.
                 StructuredProcedure*,     // Initial or always comb block.
                 AlwaysComb*,              // An always_comb block.
                 AlwaysFf*,                // An always_ff block.
                 AlwaysFlop*,              // "Flip-Flop" block.
                 Comment*,                 // Comment text.
                 BlankLine*,               // Blank line.
                 InlineVerilogStatement*,  // InlineVerilog string statement.
                 VerilogFunction*,         // Function definition
                 Cover*, ConcurrentAssertion*, ModuleSection*>;

// A ModuleSection is a container of ModuleMembers used to organize the contents
// of a module. A Module contains a single top-level ModuleSection which may
// contain other ModuleSections. ModuleSections enables modules to be
// constructed a non-linear, random-access fashion by appending members to
// different sections rather than just appending to the end of module.
// TODO(meheff): Move Module methods AddReg*, AddWire*, etc to ModuleSection.
class ModuleSection : public VastNode {
 public:
  using VastNode::VastNode;

  // Constructs and adds a module member of type T to the section. Ownership is
  // maintained by the parent VerilogFile. Templatized on T in order to return a
  // pointer to the derived type.
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  template <typename T>
  T AddModuleMember(T member) {
    members_.push_back(member);
    return member;
  }

  // Recursively gathers and returns all ModuleMembers contained in this section
  // or any section contained in this section.
  std::vector<ModuleMember> GatherMembers() const;

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::vector<ModuleMember> members_;
};

// Represents a module port.
// TODO(meheff): 2021/04/26 Sink this data type into Module as a nested type and
// remove superfluous proto conversion (or even remove it).
struct Port {
  static Port FromProto(const PortProto& proto, VerilogFile* file);

  absl::StatusOr<PortProto> ToProto() const;
  const std::string& name() const { return wire->GetName(); }
  std::string ToString() const;

  Direction direction;
  Def* wire;
};

// Represents a module definition.
class Module : public VastNode {
 public:
  Module(std::string_view name, VerilogFile* file, const SourceInfo& loc)
      : VastNode(file, loc), name_(name), top_(file, loc) {}

  // Constructs and adds a node to the module. Ownership is maintained by the
  // parent VerilogFile. Most constructs should be added to the module. The
  // exceptions are the AddFoo convenience methods defined below which return a
  // Ref to the object created rather than the object itself.
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  // Adds a (wire) port to this module with the given name and type. Returns a
  // reference to that wire.
  LogicRef* AddInput(std::string_view name, DataType* type,
                     const SourceInfo& loc);
  LogicRef* AddOutput(std::string_view name, DataType* type,
                      const SourceInfo& loc);

  // Adds a reg/wire definition to the module with the given type and, for regs,
  // initialized with the given value. Returns a reference to the definition.
  LogicRef* AddReg(std::string_view name, DataType* type,
                   const SourceInfo& loc, Expression* init = nullptr,
                   ModuleSection* section = nullptr);
  LogicRef* AddWire(std::string_view name, DataType* type,
                    const SourceInfo& loc, ModuleSection* section = nullptr);
  LogicRef* AddWire(std::string_view name, DataType* type, Expression* init,
                    const SourceInfo& loc, ModuleSection* section = nullptr);
  LogicRef* AddInteger(std::string_view name, const SourceInfo& loc,
                       ModuleSection* section = nullptr);

  ParameterRef* AddParameter(std::string_view name, Expression* rhs,
                             const SourceInfo& loc);

  // Adds a previously constructed VAST construct to the module.
  template <typename T>
  T AddModuleMember(T member) {
    top_.AddModuleMember(member);
    return member;
  }

  ModuleSection* top() { return &top_; }

  absl::Span<const Port> ports() const { return ports_; }
  const std::string& name() const { return name_; }

  std::string Emit(LineInfo* line_info) const override;

 private:
  // Add the given Def as a port on the module.
  LogicRef* AddPortDef(Direction direction, Def* def, const SourceInfo& loc);

  std::string EmitMember(LineInfo* line_info, ModuleMember* member);

  std::string name_;
  std::vector<Port> ports_;

  ModuleSection top_;
};

// Represents a file-level inclusion directive.
class Include : public VastNode {
 public:
  Include(std::string_view path, VerilogFile* file, const SourceInfo& loc)
      : VastNode(file, loc), path_(path) {}

  std::string Emit(LineInfo* line_info) const override;

 private:
  std::string path_;
};

using FileMember = std::variant<Module*, Include*, BlankLine*, Comment*>;

// Represents a file (as a Verilog translation-unit equivalent).
class VerilogFile {
 public:
  explicit VerilogFile(FileType file_type) : file_type_(file_type) {}

  Module* AddModule(std::string_view name, const SourceInfo& loc) {
    return Add(Make<Module>(loc, name));
  }
  void AddInclude(std::string_view path, const SourceInfo& loc) {
    Add(Make<Include>(loc, path));
  }

  template <typename T>
  T* Add(T* member) {
    members_.push_back(member);
    return member;
  }

  template <typename T, typename... Args>
  T* Make(const SourceInfo& loc, Args&&... args) {
    std::unique_ptr<T> value =
        std::make_unique<T>(std::forward<Args>(args)..., this, loc);
    T* ptr = value.get();
    nodes_.push_back(std::move(value));
    return ptr;
  }

  std::string Emit(LineInfo* line_info = nullptr) const;

  verilog::Slice* Slice(IndexableExpression* subject, Expression* hi,
                        Expression* lo, const SourceInfo& loc) {
    return Make<verilog::Slice>(loc, subject, hi, lo);
  }
  verilog::Slice* Slice(IndexableExpression* subject, int64_t hi, int64_t lo,
                        const SourceInfo& loc) {
    XLS_CHECK_GE(hi, 0);
    XLS_CHECK_GE(lo, 0);
    return Make<verilog::Slice>(loc, subject, MaybePlainLiteral(hi, loc),
                                MaybePlainLiteral(lo, loc));
  }

  verilog::PartSelect* PartSelect(IndexableExpression* subject,
                                  Expression* start, Expression* width,
                                  const SourceInfo& loc) {
    return Make<verilog::PartSelect>(loc, subject, start, width);
  }
  verilog::PartSelect* PartSelect(IndexableExpression* subject,
                                  Expression* start, int64_t width,
                                  const SourceInfo& loc) {
    XLS_CHECK_GT(width, 0);
    return Make<verilog::PartSelect>(loc, subject, start,
                                     MaybePlainLiteral(width, loc));
  }

  verilog::Index* Index(IndexableExpression* subject, Expression* index,
                        const SourceInfo& loc) {
    return Make<verilog::Index>(loc, subject, index);
  }
  verilog::Index* Index(IndexableExpression* subject, int64_t index,
                        const SourceInfo& loc) {
    XLS_CHECK_GE(index, 0);
    return Make<verilog::Index>(loc, subject, MaybePlainLiteral(index, loc));
  }

  Unary* Negate(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, "-", expression, /*precedence=*/12);
  }
  Unary* BitwiseNot(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, "~", expression, /*precedence=*/12);
  }
  Unary* LogicalNot(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, "!", expression, /*precedence=*/12);
  }
  Unary* AndReduce(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, "&", expression, /*precedence=*/12);
  }
  Unary* OrReduce(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, "|", expression, /*precedence=*/12);
  }
  Unary* XorReduce(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, "^", expression, /*precedence=*/12);
  }

  xls::verilog::Concat* Concat(absl::Span<Expression* const> args,
                               const SourceInfo& loc) {
    return Make<xls::verilog::Concat>(loc, args);
  }
  xls::verilog::Concat* Concat(Expression* replication,
                               absl::Span<Expression* const> args,
                               const SourceInfo& loc) {
    return Make<xls::verilog::Concat>(loc, replication, args);
  }
  xls::verilog::Concat* Concat(int64_t replication,
                               absl::Span<Expression* const> args,
                               const SourceInfo& loc) {
    return Make<xls::verilog::Concat>(loc, MaybePlainLiteral(replication, loc),
                                      args);
  }

  xls::verilog::ArrayAssignmentPattern* ArrayAssignmentPattern(
      absl::Span<Expression* const> args, const SourceInfo& loc) {
    return Make<xls::verilog::ArrayAssignmentPattern>(loc, args);
  }

  BinaryInfix* Add(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "+", rhs, /*precedence=*/9);
  }
  BinaryInfix* LogicalAnd(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "&&", rhs, /*precedence=*/2);
  }
  BinaryInfix* BitwiseAnd(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "&", rhs, /*precedence=*/5);
  }
  BinaryInfix* NotEquals(Expression* lhs, Expression* rhs,
                         const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "!=", rhs, /*precedence=*/6);
  }
  BinaryInfix* Equals(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "==", rhs, /*precedence=*/6);
  }
  BinaryInfix* GreaterThanEquals(Expression* lhs, Expression* rhs,
                                 const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, ">=", rhs, /*precedence=*/7);
  }
  BinaryInfix* GreaterThan(Expression* lhs, Expression* rhs,
                           const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, ">", rhs, /*precedence=*/7);
  }
  BinaryInfix* LessThanEquals(Expression* lhs, Expression* rhs,
                              const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "<=", rhs, /*precedence=*/7);
  }
  BinaryInfix* LessThan(Expression* lhs, Expression* rhs,
                        const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "<", rhs, /*precedence=*/7);
  }
  BinaryInfix* Div(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "/", rhs, /*precedence=*/10);
  }
  BinaryInfix* Mod(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "%", rhs, /*precedence=*/10);
  }
  BinaryInfix* Mul(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "*", rhs, /*precedence=*/10);
  }
  BinaryInfix* BitwiseOr(Expression* lhs, Expression* rhs,
                         const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "|", rhs, /*precedence=*/3);
  }
  BinaryInfix* LogicalOr(Expression* lhs, Expression* rhs,
                         const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "||", rhs, /*precedence=*/1);
  }
  BinaryInfix* BitwiseXor(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "^", rhs, /*precedence=*/4);
  }
  BinaryInfix* Shll(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "<<", rhs, /*precedence=*/8);
  }
  BinaryInfix* Shra(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, ">>>", rhs, /*precedence=*/8);
  }
  BinaryInfix* Shrl(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, ">>", rhs, /*precedence=*/8);
  }
  BinaryInfix* Sub(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "-", rhs, /*precedence=*/9);
  }

  // Only for use in testing.
  BinaryInfix* NotEqualsX(Expression* lhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "!==", XLiteral(loc), /*precedence=*/6);
  }
  BinaryInfix* EqualsX(Expression* lhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, "===", XLiteral(loc), /*precedence=*/6);
  }

  verilog::Ternary* Ternary(Expression* cond, Expression* consequent,
                            Expression* alternate, const SourceInfo& loc) {
    return Make<verilog::Ternary>(loc, cond, consequent, alternate);
  }

  verilog::XLiteral* XLiteral(const SourceInfo& loc) {
    return Make<verilog::XLiteral>(loc);
  }

  // Creates an literal with the given value and bit_count.
  verilog::Literal* Literal(uint64_t value, int64_t bit_count,
                            const SourceInfo& loc,
                            FormatPreference format = FormatPreference::kHex) {
    return Make<verilog::Literal>(loc, UBits(value, bit_count), format);
  }

  // Creates an literal whose value and width is given by a Bits object.
  verilog::Literal* Literal(const Bits& bits, const SourceInfo& loc,
                            FormatPreference format = FormatPreference::kHex) {
    return Make<verilog::Literal>(loc, bits, format);
  }

  // Creates a one-bit literal.
  verilog::Literal* Literal1(int64_t value, const SourceInfo& loc) {
    // Avoid taking a bool argument as many types implicitly and undesirably
    // convert to bool
    XLS_CHECK((value == 0) || (value == 1));
    return Make<verilog::Literal>(loc, UBits(value, 1), FormatPreference::kHex);
  }

  // Creates a decimal literal representing a plain decimal number without a bit
  // count prefix (e.g., "42"). Use for clarity when bit width does not matter,
  // for example, as bit-slice indices.
  verilog::Literal* PlainLiteral(int32_t value, const SourceInfo& loc) {
    return Make<verilog::Literal>(loc, SBits(value, 32),
                                  FormatPreference::kDefault,
                                  /*emit_bit_count=*/false);
  }

  // Returns a scalar type. Example:
  //   wire foo;
  DataType* ScalarType(const SourceInfo& loc) {
    return Make<verilog::ScalarType>(loc);
  }

  // Returns an integer type. Example:
  //   integer foo;
  DataType* IntegerType(const SourceInfo& loc) {
    return Make<verilog::IntegerType>(loc);
  }

  // Returns a bit vector type for widths greater than one, and a scalar type
  // for a width of one. The motivation for this special case is avoiding types
  // with trivial single bit ranges "[0:0]" (as in "reg [0:0] foo"). This
  // matches the behavior of the XLS code generation where one-wide Bits types
  // are represented as Verilog scalars.
  DataType* BitVectorType(int64_t bit_count, const SourceInfo& loc,
                          bool is_signed = false);

  // As above, but does not produce a scalar value when the bit_count is 1.
  //
  // Generally BitVectorType() should be preferred, this is for use in
  // special-case Verilog operation contexts that cannot use scalars.
  verilog::BitVectorType* BitVectorTypeNoScalar(int64_t bit_count,
                                                const SourceInfo& loc,
                                                bool is_signed = false);

  // Returns a packed array type. Example:
  //   wire [7:0][41:0][122:0] foo;
  verilog::PackedArrayType* PackedArrayType(int64_t element_bit_count,
                                            absl::Span<const int64_t> dims,
                                            const SourceInfo& loc,
                                            bool is_signed = false);

  // Returns an unpacked array type. Example:
  //   wire [7:0] foo[42][123];
  verilog::UnpackedArrayType* UnpackedArrayType(int64_t element_bit_count,
                                                absl::Span<const int64_t> dims,
                                                const SourceInfo& loc,
                                                bool is_signed = false);

  verilog::Cover* Cover(LogicRef* clk, Expression* condition,
                        std::string_view label, const SourceInfo& loc) {
    return Make<verilog::Cover>(loc, clk, condition, label);
  }

  // Returns whether this is a SystemVerilog or Verilog file.
  bool use_system_verilog() const {
    return file_type_ == FileType::kSystemVerilog;
  }

 private:
  // Same as PlainLiteral if value fits in an int32_t. Otherwise creates a
  // 64-bit literal to hold the value.
  verilog::Literal* MaybePlainLiteral(int64_t value, const SourceInfo& loc) {
    return (value >= std::numeric_limits<int32_t>::min() &&
            value <= std::numeric_limits<int32_t>::max())
               ? PlainLiteral(value, loc)
               : Literal(SBits(value, 64), loc);
  }

  FileType file_type_;
  std::vector<FileMember> members_;
  std::vector<std::unique_ptr<VastNode>> nodes_;
};

template <typename T, typename... Args>
inline T* StatementBlock::Add(const SourceInfo& loc, Args&&... args) {
  T* ptr = file()->Make<T>(loc, std::forward<Args>(args)...);
  statements_.push_back(ptr);
  return ptr;
}

template <typename T, typename... Args>
inline T* VerilogFunction::AddStatement(const SourceInfo& loc, Args&&... args) {
  return statement_block_->Add<T>(loc, std::forward<Args>(args)...);
}

template <typename... Args>
inline LogicRef* VerilogFunction::AddRegDef(const SourceInfo& loc,
                                            Args&&... args) {
  RegDef* ptr = file()->Make<RegDef>(loc, std::forward<Args>(args)...);
  block_reg_defs_.push_back(ptr);
  return file()->Make<LogicRef>(loc, ptr);
}

template <typename T, typename... Args>
inline T* Module::Add(const SourceInfo& loc, Args&&... args) {
  T* ptr = file()->Make<T>(loc, std::forward<Args>(args)...);
  AddModuleMember(ptr);
  return ptr;
}

template <typename T, typename... Args>
inline T* ModuleSection::Add(const SourceInfo& loc, Args&&... args) {
  T* ptr = file()->Make<T>(loc, std::forward<Args>(args)...);
  AddModuleMember(ptr);
  return ptr;
}

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_VAST_H_
