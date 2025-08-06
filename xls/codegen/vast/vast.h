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

// Subset-of-verilog AST, suitable for combining as data structures before
// emission.

#ifndef XLS_CODEGEN_VAST_VAST_H_
#define XLS_CODEGEN_VAST_VAST_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace verilog {

enum class FileType {
  kVerilog,
  kSystemVerilog,
};

// Forward declarations.
class Enum;
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

  // Returns the nodes associated with this lineinfo.
  absl::Span<const VastNode* const> nodes() const { return nodes_; }

  // Returns the line spans that a given node covers.
  // Returns std::nullopt if the given node has a hanging span (Start w/o End).
  // Returns std::nullopt if the given node was never recorded.
  std::optional<std::vector<LineSpan>> LookupNode(const VastNode* node) const;

 private:
  int64_t current_line_number_ = 0;
  absl::flat_hash_map<const VastNode*, PartialLineSpans> spans_;
  // The vector of nodes in spans_ in insertion order.
  std::vector<const VastNode*> nodes_;
};

// Returns a sanitized identifier string based on the given name. Invalid
// characters are replaced with '_'. (System)Verilog keywords are
// suffixed with "_".
std::string SanitizeVerilogIdentifier(std::string_view name,
                                      bool system_verilog = true);

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

  // Returns whether this type represents a typedef, struct, enum, or an array
  // of a user-defined type.
  virtual bool IsUserDefined() const { return false; }

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
                                         std::string_view identifier) const;
};

// Represents a scalar type. Example:
//   wire foo;
class ScalarType final : public DataType {
 public:
  ScalarType(VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc), is_signed_(false) {}
  ScalarType(bool is_signed, VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc), is_signed_(is_signed) {}

  bool IsScalar() const final { return true; }
  bool is_signed() const final { return is_signed_; }
  absl::StatusOr<int64_t> WidthAsInt64() const final { return 1; }
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final { return 1; }
  std::optional<Expression*> width() const final { return std::nullopt; }
  std::string Emit(LineInfo* line_info) const final;

 private:
  bool is_signed_;
};

// Represents an integer type. Example:
//   integer foo;
class IntegerType final : public DataType {
 public:
  IntegerType(bool is_signed, VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc), is_signed_(is_signed) {}

  IntegerType(VerilogFile* file, const SourceInfo& loc)
      : IntegerType(/*is_signed=*/true, file, loc) {}

  bool IsScalar() const final { return false; }
  absl::StatusOr<int64_t> WidthAsInt64() const final {
    return absl::InvalidArgumentError("Cannot get width of integer types");
  }
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final { return 32; }
  std::optional<Expression*> width() const final { return std::nullopt; }
  std::string Emit(LineInfo* line_info) const final;

  bool is_signed() const final { return is_signed_; }

 private:
  bool is_signed_;
};

// Represents a bit-vector type. Example:
//   reg[7:0] foo;
class BitVectorType final : public DataType {
 public:
  BitVectorType(Expression* size_expr, bool is_signed, bool size_expr_is_max,
                VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc),
        size_expr_(size_expr),
        size_expr_is_max_(size_expr_is_max),
        is_signed_(is_signed) {}

  BitVectorType(Expression* size_expr, bool is_signed, VerilogFile* file,
                const SourceInfo& loc)
      : BitVectorType(size_expr, is_signed, /*size_expr_is_max=*/false, file,
                      loc) {}
  BitVectorType(int64_t width, bool is_signed, VerilogFile* file,
                const SourceInfo& loc);

  bool IsScalar() const final { return false; }
  absl::StatusOr<int64_t> WidthAsInt64() const final;
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final;

  std::optional<Expression*> width() const final {
    if (size_expr_is_max_) {
      return std::nullopt;
    }
    return size_expr_;
  }
  std::optional<Expression*> max() const {
    if (size_expr_is_max_) {
      return size_expr_;
    }
    return std::nullopt;
  }

  // Returns the expression for either the width or max; whichever was supplied
  // at construction time.
  Expression* size_expr() const { return size_expr_; }

  bool is_signed() const final { return is_signed_; }
  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* size_expr_;
  // Whether the `size_expr_` represents the max index as opposed to the width.
  // Currently this is only true for vectors originating from SystemVerilog
  // source code.
  bool size_expr_is_max_ = false;
  bool is_signed_;
};

class ArrayTypeBase : public DataType {
 public:
  ArrayTypeBase(DataType* element_type, absl::Span<Expression* const> dims,
                bool dims_are_max, VerilogFile* file, const SourceInfo& loc)
      : DataType(file, loc),
        element_type_(element_type),
        dims_(dims.begin(), dims.end()),
        dims_are_max_(dims_are_max) {
    CHECK(!dims.empty());
  }

  ArrayTypeBase(DataType* element_type, absl::Span<const int64_t> dims,
                bool dims_are_max, VerilogFile* file, const SourceInfo& loc);

  bool IsScalar() const final { return false; }

  bool IsUserDefined() const final { return element_type_->IsUserDefined(); }

  absl::StatusOr<int64_t> WidthAsInt64() const final {
    return element_type_->WidthAsInt64();
  }

  std::optional<Expression*> width() const final {
    return element_type_->width();
  }

  bool is_signed() const final { return element_type_->is_signed(); }

  DataType* element_type() const { return element_type_; }

  // Returns the dimensions for the type, excluding the element type. For
  // example, the net type for "wire [7:0][42:0][3:0] foo;" has {43, 4} as the
  // array dimensions. The inner-most dimension (of the contained
  // `BitVectorType`) is not included as that is considered for type purposes as
  // the underlying array element type ("wire [7:0]"). Similarly, for an
  // unpacked array, the net type for "wire [7:0][42:0] foo [123][7];" has {123,
  // 7} as the unpacked dimensions. The ordering of indices matches the order in
  // which they are emitted in the emitted Verilog.
  absl::Span<Expression* const> dims() const { return dims_; }

  bool dims_are_max() const { return dims_are_max_; }

 private:
  DataType* element_type_;
  std::vector<Expression*> dims_;
  bool dims_are_max_;
};

// Convenience function to normalize an array dimension, which may specify
// either the width or max index, to a width specification.
Expression* ArrayDimToWidth(Expression* dim, bool dim_is_max);

// Represents a packed array of bit-vectors type. Example:
//   wire [7:0][42:0][3:0] foo;
class PackedArrayType final : public ArrayTypeBase {
 public:
  PackedArrayType(Expression* width, absl::Span<Expression* const> packed_dims,
                  bool is_signed, VerilogFile* file, const SourceInfo& loc);

  PackedArrayType(int64_t width, absl::Span<const int64_t> packed_dims,
                  bool is_signed, VerilogFile* file, const SourceInfo& loc);

  PackedArrayType(DataType* element_type,
                  absl::Span<Expression* const> packed_dims, bool dims_are_max,
                  VerilogFile* file, const SourceInfo& loc)
      : ArrayTypeBase(element_type, packed_dims, dims_are_max, file, loc) {}

  PackedArrayType(DataType* element_type, absl::Span<const int64_t> packed_dims,
                  bool dims_are_max, VerilogFile* file, const SourceInfo& loc);

  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final;

  std::string Emit(LineInfo* line_info) const final;
};

// Represents an unpacked array of bit-vectors or packed array types. Example:
//   wire [7:0][42:0][3:0] foo [1:0];
// Where [1:0] is the unpacked array dimensions.
class UnpackedArrayType final : public ArrayTypeBase {
 public:
  UnpackedArrayType(DataType* element_type,
                    absl::Span<Expression* const> unpacked_dims,
                    VerilogFile* file, const SourceInfo& loc)
      : ArrayTypeBase(element_type, unpacked_dims, /*dims_are_max=*/false, file,
                      loc) {
    CHECK(dynamic_cast<UnpackedArrayType*>(element_type) == nullptr);
  }

  UnpackedArrayType(DataType* element_type,
                    absl::Span<const int64_t> unpacked_dims, VerilogFile* file,
                    const SourceInfo& loc);

  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final;

  std::string Emit(LineInfo* line_info) const final {
    LOG(FATAL) << "EmitWithIdentifier should be called rather than emit";
  }

  std::string EmitWithIdentifier(LineInfo* line_info,
                                 std::string_view identifier) const final;
};

// The kind of a net/variable. kReg, kWire, kLogic can be arbitrarily
// typed. kInteger definitions can only be of IntegerType.
enum class DataKind : int8_t {
  kReg,
  kWire,
  kLogic,
  kInteger,
  // Any user-defined type, such as a typedef, struct, or enum.
  kUser,
  // The data kind of an enum definition itself that has no specified kind, i.e.
  // "enum { elements }" as opposed to "enum int { elements }" or similar.
  kUntypedEnum,
  // Used as an integer during elaboration to evaluate a generate loop.
  kGenvar,
};

// Represents the definition of a variable or net.
class Def : public Statement {
 public:
  Def(std::string_view name, DataKind data_kind, DataType* data_type,
      VerilogFile* file, const SourceInfo& loc)
      : Def(name, data_kind, data_type, /*init=*/nullptr, file, loc) {}

  Def(std::string_view name, DataKind data_kind, DataType* data_type,
      Expression* init, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        name_(name),
        data_kind_(data_kind),
        data_type_(data_type),
        init_(init == nullptr ? std::nullopt : std::make_optional(init)) {}

  std::string Emit(LineInfo* line_info) const final;

  // Emit the definition without the trailing semicolon.
  std::string EmitNoSemi(LineInfo* line_info) const;

  const std::string& GetName() const { return name_; }
  DataKind data_kind() const { return data_kind_; }
  DataType* data_type() const { return data_type_; }

  // Returns the optional initialization expression.
  std::optional<Expression*> init() const { return init_; }

 private:
  std::string name_;
  DataKind data_kind_;
  DataType* data_type_;
  std::optional<Expression*> init_;
};

// A wire definition. Example:
//   wire [41:0] foo;
class WireDef final : public Def {
 public:
  WireDef(std::string_view name, DataType* data_type, VerilogFile* file,
          const SourceInfo& loc)
      : Def(name, DataKind::kWire, data_type, file, loc) {}
  WireDef(std::string_view name, DataType* data_type, Expression* init,
          VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kWire, data_type, init, file, loc) {}
};

// A user-defined type definition. Example:
//   FooBarT foo;
class UserDefinedDef final : public Def {
 public:
  UserDefinedDef(std::string_view name, DataType* data_type, VerilogFile* file,
                 const SourceInfo& loc)
      : Def(name, DataKind::kUser, data_type, file, loc) {}
  UserDefinedDef(std::string_view name, DataType* data_type, Expression* init,
                 VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kUser, data_type, init, file, loc) {}
};

// Register variable definition. Example:
//   reg [41:0] foo;
class RegDef final : public Def {
 public:
  RegDef(std::string_view name, DataType* data_type, VerilogFile* file,
         const SourceInfo& loc)
      : Def(name, DataKind::kReg, data_type, file, loc) {}
  RegDef(std::string_view name, DataType* data_type, Expression* init,
         VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kReg, data_type, init, file, loc) {}
};

// Logic variable definition. Example:
//   logic [41:0] foo;
class LogicDef final : public Def {
 public:
  LogicDef(std::string_view name, DataType* data_type, VerilogFile* file,
           const SourceInfo& loc)
      : Def(name, DataKind::kLogic, data_type, file, loc) {}
  LogicDef(std::string_view name, DataType* data_type, Expression* init,
           VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kLogic, data_type, init, file, loc) {}
};

// Variable definition with a type that is a user-defined name. Example:
//   foo_t [41:0] foo;
class UserDef final : public Def {
 public:
  UserDef(std::string_view name, DataType* data_type, VerilogFile* file,
          const SourceInfo& loc)
      : Def(name, DataKind::kUser, data_type, file, loc) {}
  UserDef(std::string_view name, DataType* data_type, Expression* init,
          VerilogFile* file, const SourceInfo& loc)
      : Def(name, DataKind::kUser, data_type, init, file, loc) {}
};

// Integer variable definition. Example:
//   integer foo;
class IntegerDef final : public Def {
 public:
  IntegerDef(std::string_view name, VerilogFile* file, const SourceInfo& loc);
  IntegerDef(std::string_view name, DataType* data_type, Expression* init,
             VerilogFile* file, const SourceInfo& loc);
};

// Represents a genvar definition, for use in generate loops. Example:
//   genvar i;
class GenvarDef final : public Def {
 public:
  GenvarDef(std::string_view name, VerilogFile* file, const SourceInfo& loc);
};

// Represents a #${delay} statement.
class DelayStatement final : public Statement {
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

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* delay_;
  Statement* delayed_statement_;
};

// Represents a wait statement.
class WaitStatement final : public Statement {
 public:
  WaitStatement(Expression* event, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), event_(event) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* event_;
};

// Represents a forever construct which runs a statement continuously.
class Forever final : public Statement {
 public:
  Forever(Statement* statement, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), statement_(statement) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  Statement* statement_;
};

class AssignmentBase : public Statement {
 public:
  AssignmentBase(Expression* lhs, Expression* rhs, VerilogFile* file,
                 const SourceInfo& loc)
      : Statement(file, loc), lhs_(ABSL_DIE_IF_NULL(lhs)), rhs_(rhs) {}

  Expression* lhs() const { return lhs_; }
  Expression* rhs() const { return rhs_; }

 private:
  Expression* lhs_;
  Expression* rhs_;
};

// Represents a blocking assignment ("lhs = rhs;")
class BlockingAssignment final : public AssignmentBase {
 public:
  BlockingAssignment(Expression* lhs, Expression* rhs, VerilogFile* file,
                     const SourceInfo& loc)
      : AssignmentBase(lhs, rhs, file, loc) {}

  std::string Emit(LineInfo* line_info) const final;
};

// Represents a nonblocking assignment  ("lhs <= rhs;").
class NonblockingAssignment final : public AssignmentBase {
 public:
  NonblockingAssignment(Expression* lhs, Expression* rhs, VerilogFile* file,
                        const SourceInfo& loc)
      : AssignmentBase(lhs, rhs, file, loc) {}

  std::string Emit(LineInfo* line_info) const final;
};

// Represents an explicit SystemVerilog function return statement. The
// alternative construct for this is an assignment to the function name.
// Currently an explicit return statement is only modeled in VAST trees coming
// from parsed SystemVerilog.
class ReturnStatement final : public Statement {
 public:
  ReturnStatement(Expression* expr, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), expr_(expr) {}

  std::string Emit(LineInfo* line_info) const final;

  Expression* expr() const { return expr_; }

 private:
  Expression* expr_;
};

// An abstraction representing a sequence of statements within a structured
// procedure (e.g., an "always" statement).
class StatementBlock final : public VastNode {
 public:
  using VastNode::VastNode;

  // Constructs and adds a statement to the block. Ownership is maintained by
  // the parent VerilogFile. Example:
  //   Case* c = Add<Case>(subject);
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  std::string Emit(LineInfo* line_info) const final;

  absl::Span<Statement* const> statements() const { return statements_; }

 private:
  std::vector<Statement*> statements_;
};

// Represents a generate loop construct. Example:
// ```verilog
// generate
//   for (i = 0; i < 32; i++) begin
//     assign output[i] = input[i];
//   end
// endgenerate
// ```
class GenerateLoop final : public Statement {
 public:
  GenerateLoop(LogicRef* genvar, Expression* init, Expression* limit,
               std::optional<std::string_view> label, VerilogFile* file,
               const SourceInfo& loc);

  void AddBodyNode(VastNode* node) { body_.push_back(node); }

  std::string Emit(LineInfo* line_info) const final;

 private:
  LogicRef* genvar_;
  GenvarDef* genvar_def_;

  Expression* init_;
  Expression* limit_;
  std::vector<VastNode*> body_;
  std::optional<std::string> label_;
};

// Similar to statement block,  but for use if `ifdef `else `endif blocks (no
// "begin" or "end").
class MacroStatementBlock final : public VastNode {
 public:
  using VastNode::VastNode;
  // Constructs and adds a statement to the block. Ownership is maintained by
  // the parent VerilogFile. Example:
  //   Case* c = Add<Case>(subject);
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  std::string Emit(LineInfo* line_info) const final;

  absl::Span<Statement* const> statements() const { return statements_; }

 private:
  std::vector<Statement*> statements_;
};

// Represents a 'default' case arm label.
struct DefaultSentinel {};

// Represents a label within a case statement.
using CaseLabel = std::variant<Expression*, DefaultSentinel>;

// Represents an arm of a case statement.
class CaseArm final : public VastNode {
 public:
  CaseArm(CaseLabel label, VerilogFile* file, const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const final;
  StatementBlock* statements() { return statements_; }

 private:
  CaseLabel label_;
  StatementBlock* statements_;
};

enum class CaseKeyword : uint8_t {
  kCase,
  kCasez,
};

enum class CaseModifier : uint8_t {
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
class Case final : public Statement {
 public:
  Case(Expression* subject, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc),
        subject_(subject),
        case_type_(CaseType(CaseKeyword::kCase)) {}

  Case(Expression* subject, CaseType case_type, VerilogFile* file,
       const SourceInfo& loc)
      : Statement(file, loc), subject_(subject), case_type_(case_type) {}

  StatementBlock* AddCaseArm(CaseLabel label);

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* subject_;
  std::vector<CaseArm*> arms_;
  CaseType case_type_;
};

// Represents an if statement with optional "else if" and "else" blocks.
class Conditional final : public Statement {
 public:
  Conditional(Expression* condition, VerilogFile* file, const SourceInfo& loc);

  // Returns a pointer to the statement block of the consequent.
  StatementBlock* consequent() const { return consequent_; }

  // Adds an alternate clause ("else if" or "else") and returns a pointer to the
  // consequent. The alternate is final (an "else") if condition is null. Dies
  // if a final alternate ("else") clause has been previously added.
  StatementBlock* AddAlternate(Expression* condition = nullptr);

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* condition_;
  StatementBlock* consequent_;

  // The alternate clauses ("else if's" and "else"). If the Expression* is null
  // then the alternate is unconditional ("else"). This can only appear as the
  // final alternate.
  std::vector<std::pair<Expression*, StatementBlock*>> alternates_;
};

enum class ConditionalDirectiveKind : uint8_t {
  kIfdef,
  kIfndef,
};

std::string ConditionalDirectiveKindToString(ConditionalDirectiveKind kind);
inline std::ostream& operator<<(std::ostream& os,
                                ConditionalDirectiveKind kind) {
  os << ConditionalDirectiveKindToString(kind);
  return os;
}

// Represents an `ifdef block within a statement context.
class StatementConditionalDirective final : public Statement {
 public:
  StatementConditionalDirective(ConditionalDirectiveKind kind,
                                std::string identifier, VerilogFile* file,
                                const SourceInfo& loc);

  // Returns a pointer to the statement block of the consequent.
  MacroStatementBlock* consequent() const { return consequent_; }

  // Adds an alternate clause ("`elsif" or "`else") and returns a pointer to the
  // consequent. The alternate is final (an "`else") if identifier is empty.
  // Dies if a final alternate ("`else") clause has been previously added.
  MacroStatementBlock* AddAlternate(std::string identifier = "");

  std::string Emit(LineInfo* line_info) const final;

 private:
  ConditionalDirectiveKind kind_;
  std::string identifier_;
  MacroStatementBlock* consequent_;

  // The alternate clauses ("`elsif" and "`else"). If the string is empty then
  // the alternate is unconditional ("`else"). This can only appear as the final
  // alternate.
  std::vector<std::pair<std::string, MacroStatementBlock*>> alternates_;
};

// Represents a while loop construct.
class WhileStatement final : public Statement {
 public:
  WhileStatement(Expression* condition, VerilogFile* file,
                 const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const final;

  StatementBlock* statements() const { return statements_; }

 private:
  Expression* condition_;
  StatementBlock* statements_;
};

// Represents a repeat construct.
class RepeatStatement final : public Statement {
 public:
  RepeatStatement(Expression* repeat_count, VerilogFile* file,
                  const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const final;

  StatementBlock* statements() const { return statements_; }

 private:
  Expression* repeat_count_;
  StatementBlock* statements_;
};

// Represents an event control statement. This is represented as "@(...);" where
// "..." is the event expression..
class EventControl final : public Statement {
 public:
  EventControl(Expression* event_expression, VerilogFile* file,
               const SourceInfo& loc)
      : Statement(file, loc), event_expression_(event_expression) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* event_expression_;
};

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
  // for the purposes of emission we consider them to have maximum precedence
  // so they are never wrapped in parentheses. Operator (derived from
  // Expression) are the only types with non-max precedence.
  //
  // Precedence of operators in Verilog operator precedence (from LRM):
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
class XSentinel final : public Expression {
 public:
  XSentinel(int64_t width, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), width_(width) {}

  std::string Emit(LineInfo* line_info) const final;

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
class FourValueBinaryLiteral final : public Expression {
 public:
  FourValueBinaryLiteral(absl::Span<FourValueBit const> value,
                         VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), bits_(value.begin(), value.end()) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::vector<FourValueBit> bits_;
};

// Specifies the kind of an `Operator` expression.
enum class OperatorKind {
  kEq,
  kCaseEq,
  kNe,
  kCaseNe,
  kEqX,
  kNeX,
  kGe,
  kGt,
  kLe,
  kLt,
  kAdd,
  kSub,
  kMul,
  kDiv,
  kMod,
  kShll,
  kShrl,
  kShra,
  kLogicalAnd,
  kLogicalOr,
  kBitwiseAnd,
  kBitwiseOr,
  kBitwiseXor,
  kPower,
  kNegate,
  kBitwiseNot,
  kLogicalNot,
  kAndReduce,
  kOrReduce,
  kXorReduce
};

// Returns the precedence for evaluation of an operator with the given kind.
int Precedence(OperatorKind kind);

// Returns the string used in Verilog code to represent the given operator kind.
std::string_view OperatorString(OperatorKind kind);

// Represents an operation (unary, binary, etc) with a particular precedence.
class Operator : public Expression {
 public:
  Operator(OperatorKind kind, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), kind_(kind), precedence_(Precedence(kind)) {}
  int64_t precedence() const final { return precedence_; }
  OperatorKind kind() const { return kind_; }

 private:
  OperatorKind kind_;
  int64_t precedence_;
};

// A posedge edge identifier expression.
class PosEdge final : public Expression {
 public:
  PosEdge(Expression* expression, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), expression_(expression) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* expression_;
};

// A negedge edge identifier expression.
class NegEdge final : public Expression {
 public:
  NegEdge(Expression* expression, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), expression_(expression) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* expression_;
};

// Represents a connection of either a module parameter or a port to its
// surrounding environment. That is, these are emitted in an instantiation like:
//
// ```verilog
//   .port_name(expression)
// ```
//
// Note that expression can be null, in which case we emit an empty port
// connection like so:
//
// ```verilog
//   .port_name()
// ```
//
// This can be useful in cases like stitching.
struct Connection {
  std::string port_name;
  Expression* absl_nullable expression;
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
class TemplateInstantiation final : public Instantiation {
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

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string template_text_;
};

// Represents a reference to an already-defined macro. For example: `MY_MACRO.
class MacroRef final : public Expression {
 public:
  MacroRef(std::string_view name, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), name_(name) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string name_;
};

// Defines a module parameter. A parameter must be assigned to an expression,
// and may have an explicit type def.
class Parameter final : public NamedTrait {
 public:
  Parameter(Def* def, Expression* rhs, VerilogFile* file, const SourceInfo& loc)
      : NamedTrait(file, loc), name_(def->GetName()), def_(def), rhs_(rhs) {}

  Parameter(std::string_view name, Expression* rhs, VerilogFile* file,
            const SourceInfo& loc)
      : NamedTrait(file, loc), name_(name), rhs_(rhs) {}

  std::string Emit(LineInfo* line_info) const final;
  std::string GetName() const final { return name_; }
  Def* def() const { return def_; }
  Expression* rhs() const { return rhs_; }

 private:
  // Agrees with `def_` in all cases where `def_` is non-null.
  std::string name_;
  // Currently this is only used for parameters originating from SystemVerilog
  // source code.
  Def* def_ = nullptr;
  Expression* rhs_;
};

// A user-defined type that gives a new name to another type, perhaps with some
// value set constraints, e.g. an enum or typedef.
class UserDefinedAliasType : public DataType {
 public:
  UserDefinedAliasType(DataType* base_type, VerilogFile* file,
                       const SourceInfo& loc)
      : DataType(file, loc), base_type_(base_type) {}

  DataType* BaseType() const { return base_type_; }

  bool IsScalar() const final { return base_type_->IsScalar(); }

  bool IsUserDefined() const final { return true; }

  absl::StatusOr<int64_t> WidthAsInt64() const final {
    return base_type_->WidthAsInt64();
  }

  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final {
    return base_type_->FlatBitCountAsInt64();
  }

  std::optional<Expression*> width() const final { return base_type_->width(); }

  bool is_signed() const final { return base_type_->is_signed(); }

 private:
  DataType* base_type_;
};

// The declaration of a typedef. This emits "typedef actual_type name;".
class Typedef final : public VastNode {
 public:
  Typedef(Def* def, VerilogFile* file, const SourceInfo& loc)
      : VastNode(file, loc), def_(def) {}

  std::string Emit(LineInfo* line_info) const final;

  std::string GetName() const { return def_->GetName(); }

  DataType* data_type() const { return def_->data_type(); }

  DataKind data_kind() const { return def_->data_kind(); }

 private:
  Def* def_;
};

// A type that is defined in an external package, where we may not know the
// underlying bit vector count as we request in `ExternType`.
class ExternPackageType : public DataType {
 public:
  explicit ExternPackageType(std::string_view package_name,
                             std::string_view type_name, VerilogFile* file,
                             const SourceInfo& loc)
      : DataType(file, loc),
        package_name_(package_name),
        type_name_(type_name) {}

  std::string Emit(LineInfo* line_info) const final;

  bool IsScalar() const final { return false; }
  bool IsUserDefined() const final { return true; }
  absl::StatusOr<int64_t> WidthAsInt64() const final {
    return absl::UnimplementedError(
        "WidthAsInt64 is not implemented for ExternPackageType.");
  }
  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final {
    return absl::UnimplementedError(
        "FlatBitCountAsInt64 is not implemented for ExternPackageType.");
  }
  std::optional<Expression*> width() const final { return std::nullopt; }
  bool is_signed() const final { return false; }

 private:
  std::string package_name_;
  std::string type_name_;
};

// The type of an entity when its type is a typedef. This emits just the name of
// the typedef.
class TypedefType final : public UserDefinedAliasType {
 public:
  explicit TypedefType(Typedef* type_def, VerilogFile* file,
                       const SourceInfo& loc)
      : UserDefinedAliasType(type_def->data_type(), file, loc),
        type_def_(type_def) {}

  std::string Emit(LineInfo* line_info) const final;

  Typedef* type_def() const { return type_def_; }

 private:
  Typedef* type_def_;
};

class ExternType final : public UserDefinedAliasType {
 public:
  explicit ExternType(DataType* bitvec_repr, std::string_view name,
                      VerilogFile* file, const SourceInfo& loc)
      : UserDefinedAliasType(bitvec_repr, file, loc), name_(name) {}

  // Just emits the name_
  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string name_;
};

// Represents the definition of a member of an enum.
class EnumMember final : public NamedTrait {
 public:
  EnumMember(std::string_view name, Expression* rhs, VerilogFile* file,
             const SourceInfo& loc)
      : NamedTrait(file, loc), name_(name), rhs_(rhs) {}

  std::string GetName() const final { return name_; }

  Expression* rhs() const { return rhs_; }

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string name_;
  Expression* rhs_;
};

// Refers to an enum item for use in expressions.
class EnumMemberRef final : public Expression {
 public:
  EnumMemberRef(Enum* enum_def, EnumMember* member, VerilogFile* file,
                const SourceInfo& loc)
      : Expression(file, loc), enum_def_(enum_def), member_(member) {}

  std::string Emit(LineInfo* line_info) const final {
    return member_->GetName();
  }

  Enum* enum_def() const { return enum_def_; }
  EnumMember* member() const { return member_; }

  // Duplicates this reference for use in another place. If performing type
  // inference on the VAST tree, the same exact ref object should not be used in
  // multiple places.
  EnumMemberRef* Duplicate() const;

 private:
  Enum* enum_def_;
  EnumMember* member_;
};

// Represents an enum definition.
class Enum final : public UserDefinedAliasType {
 public:
  Enum(DataKind kind, DataType* data_type, VerilogFile* file,
       const SourceInfo& loc)
      : UserDefinedAliasType(data_type, file, loc), kind_(kind) {}

  Enum(DataKind kind, DataType* data_type,
       absl::Span<EnumMember* const> members, VerilogFile* file,
       const SourceInfo& loc)
      : UserDefinedAliasType(data_type, file, loc),
        kind_(kind),
        members_(members.begin(), members.end()) {}

  EnumMemberRef* AddMember(std::string_view name, Expression* rhs,
                           const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const final;

  DataKind kind() const { return kind_; }

  absl::Span<EnumMember* const> members() const { return members_; }

 private:
  DataKind kind_;
  std::vector<EnumMember*> members_;
};

// Represents a struct type. Currently assumes packed and unsigned.
class Struct final : public DataType {
 public:
  Struct(absl::Span<Def* const> members, VerilogFile* file,
         const SourceInfo& loc)
      : DataType(file, loc), members_(members.begin(), members.end()) {}

  bool IsScalar() const final { return false; }

  bool IsUserDefined() const final { return true; }

  absl::StatusOr<int64_t> WidthAsInt64() const final {
    return absl::UnimplementedError(
        "WidthAsInt64 is not implemented for structs.");
  }

  absl::StatusOr<int64_t> FlatBitCountAsInt64() const final;

  std::optional<Expression*> width() const final { return std::nullopt; }

  bool is_signed() const final { return false; }

  std::string Emit(LineInfo* line_info) const final;

  absl::Span<Def* const> members() const { return members_; }

 private:
  std::vector<Def*> members_;
};

// Defines an item in a localparam.
class LocalParamItem final : public NamedTrait {
 public:
  LocalParamItem(std::string_view name, Expression* rhs, VerilogFile* file,
                 const SourceInfo& loc)
      : NamedTrait(file, loc), name_(name), rhs_(rhs) {}

  std::string GetName() const final { return name_; }

  std::string Emit(LineInfo* line_info) const final;

  Expression* rhs() const { return rhs_; }

 private:
  std::string name_;
  Expression* rhs_;
};

// Refers to an item in a localparam for use in expressions.
class LocalParamItemRef final : public Expression {
 public:
  LocalParamItemRef(LocalParamItem* item, VerilogFile* file,
                    const SourceInfo& loc)
      : Expression(file, loc), item_(item) {}

  std::string Emit(LineInfo* line_info) const final { return item_->GetName(); }

 private:
  LocalParamItem* item_;
};

// Defines a localparam.
class LocalParam final : public VastNode {
 public:
  using VastNode::VastNode;
  LocalParamItemRef* AddItem(std::string_view name, Expression* value,
                             const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::vector<LocalParamItem*> items_;
};

// Refers to a Parameter's definition for use in an expression.
class ParameterRef final : public Expression {
 public:
  ParameterRef(Parameter* parameter, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), parameter_(parameter) {}

  std::string Emit(LineInfo* line_info) const final {
    return parameter_->GetName();
  }

  Parameter* parameter() const { return parameter_; }

  // Duplicates this reference for use in another place. If performing type
  // inference on the VAST tree, the same exact ref object should not be used in
  // multiple places.
  ParameterRef* Duplicate() const;

 private:
  Parameter* parameter_;
};

// An indexable expression that can be bit-sliced or indexed.
class IndexableExpression : public Expression {
 public:
  using Expression::Expression;

  bool IsIndexableExpression() const final { return true; }
};

// Reference to the definition of a WireDef, RegDef, or LogicDef.
class LogicRef final : public IndexableExpression {
 public:
  LogicRef(Def* def, VerilogFile* file, const SourceInfo& loc)
      : IndexableExpression(file, loc), def_(ABSL_DIE_IF_NULL(def)) {}

  bool IsLogicRef() const final { return true; }

  std::string Emit(LineInfo* line_info) const final { return def_->GetName(); }

  // Returns the Def that this LogicRef refers to.
  Def* def() const { return def_; }

  // Returns the name of the underlying Def this object refers to.
  std::string GetName() const { return def()->GetName(); }

  // Duplicates this reference for use in another place. If performing type
  // inference on the VAST tree, the same exact ref object should not be used in
  // multiple places.
  LogicRef* Duplicate() const;

 private:
  // Logic signal definition.
  Def* def_;
};

// Represents a Verilog unary expression.
class Unary final : public Operator {
 public:
  Unary(Expression* arg, OperatorKind kind, VerilogFile* file,
        const SourceInfo& loc)
      : Operator(kind, file, loc), op_(OperatorString(kind)), arg_(arg) {}

  bool IsUnary() const final { return true; }

  // Returns true if this is reduction operation (OR, NOR, AND, NAND, XOR or
  // XNOR).
  bool IsReduction() const {
    return op_ == "|" || op_ == "~|" || op_ == "&" || op_ == "~&" ||
           op_ == "^" || op_ == "~^";
  }
  std::string Emit(LineInfo* line_info) const final;

  Expression* arg() const { return arg_; }

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
class AlwaysFlop final : public VastNode {
 public:
  AlwaysFlop(LogicRef* clk, VerilogFile* file, const SourceInfo& loc);
  AlwaysFlop(LogicRef* clk, Reset rst, VerilogFile* file,
             const SourceInfo& loc);

  // Add a register controlled by this AlwaysFlop. 'reset_value' can only have a
  // value if the AlwaysFlop has a reset signal.
  void AddRegister(LogicRef* reg, Expression* reg_next, const SourceInfo& loc,
                   Expression* reset_value = nullptr);

  std::string Emit(LineInfo* line_info) const final;

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
class Always final : public AlwaysBase {
 public:
  Always(absl::Span<const SensitivityListElement> sensitivity_list,
         VerilogFile* file, const SourceInfo& loc)
      : AlwaysBase(sensitivity_list, file, loc) {}

 protected:
  std::string name() const final { return "always"; }
};

// Defines an always_comb block.
class AlwaysComb final : public AlwaysBase {
 public:
  explicit AlwaysComb(VerilogFile* file, const SourceInfo& loc)
      : AlwaysBase({}, file, loc) {}
  std::string Emit(LineInfo* line_info) const final;

 protected:
  std::string name() const final { return "always_comb"; }
};

// Defines an always_ff block.
class AlwaysFf final : public AlwaysBase {
 public:
  using AlwaysBase::AlwaysBase;

 protected:
  std::string name() const final { return "always_ff"; }
};

// Defines an 'initial' block.
class Initial final : public StructuredProcedure {
 public:
  using StructuredProcedure::StructuredProcedure;

  std::string Emit(LineInfo* line_info) const final;
};

class Concat final : public Expression {
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

  std::string Emit(LineInfo* line_info) const final;

  absl::Span<Expression* const> args() const { return args_; }

 private:
  std::vector<Expression*> args_;
  Expression* replication_;
};

// An array assignment pattern such as: "'{foo, bar, baz}"
class ArrayAssignmentPattern final : public IndexableExpression {
 public:
  ArrayAssignmentPattern(absl::Span<Expression* const> args, VerilogFile* file,
                         const SourceInfo& loc)
      : IndexableExpression(file, loc), args_(args.begin(), args.end()) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::vector<Expression*> args_;
};

class BinaryInfix final : public Operator {
 public:
  BinaryInfix(Expression* lhs, Expression* rhs, OperatorKind kind,
              VerilogFile* file, const SourceInfo& loc)
      : Operator(kind, file, loc),
        op_(OperatorString(kind)),
        lhs_(ABSL_DIE_IF_NULL(lhs)),
        rhs_(ABSL_DIE_IF_NULL(rhs)) {}

  std::string Emit(LineInfo* line_info) const final;

  Expression* lhs() const { return lhs_; }

  Expression* rhs() const { return rhs_; }

 private:
  std::string op_;
  Expression* lhs_;
  Expression* rhs_;
};

// Defines a literal value (width and value).
class Literal final : public Expression {
 public:
  Literal(Bits bits, FormatPreference format, VerilogFile* file,
          const SourceInfo& loc)
      : Expression(file, loc),
        bits_(std::move(bits)),
        format_(format),
        emit_bit_count_(true),
        effective_bit_count_(bits_.bit_count()) {}

  Literal(Bits bits, FormatPreference format, bool emit_bit_count,
          VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        bits_(std::move(bits)),
        format_(format),
        emit_bit_count_(emit_bit_count),
        effective_bit_count_(bits_.bit_count()) {
    CHECK(emit_bit_count_ || bits_.bit_count() == 32);
  }

  Literal(Bits bits, FormatPreference format, int64_t declared_bit_count,
          bool emit_bit_count, bool declared_as_signed, VerilogFile* file,
          const SourceInfo& loc)
      : Expression(file, loc),
        bits_(std::move(bits)),
        format_(format),
        emit_bit_count_(emit_bit_count),
        declared_as_signed_(declared_as_signed),
        effective_bit_count_(declared_bit_count) {
    CHECK(declared_bit_count >= bits_.bit_count());
    CHECK(emit_bit_count_ || effective_bit_count_ == 32);
  }

  std::string Emit(LineInfo* line_info) const final;

  const Bits& bits() const { return bits_; }

  absl::StatusOr<int64_t> ToInt64() const {
    if (bits_.bit_count() != 64 &&
        !(declared_as_signed_ && bits_.bit_count() == effective_bit_count() &&
          bits_.Get(bits_.bit_count() - 1))) {
      return bits_ops::ZeroExtend(bits_, 64).ToInt64();
    }
    return bits_.ToInt64();
  }

  bool IsLiteral() const final { return true; }
  bool IsLiteralWithValue(int64_t target) const final;

  FormatPreference format() const { return format_; }

  bool is_declared_as_signed() const { return declared_as_signed_; }

  bool should_emit_bit_count() const { return emit_bit_count_; }

  int64_t effective_bit_count() const { return effective_bit_count_; }

 private:
  Bits bits_;
  FormatPreference format_;
  // Whether to emit the bit count when emitting the number. This can only be
  // false if the width of bits_ is 32 as the width of an undecorated number
  // literal in Verilog is 32.
  bool emit_bit_count_;
  // Currently only true for literals originating in SystemVerilog source code
  // that are marked by their prefix as signed.
  bool declared_as_signed_ = false;
  // Usually the same as `bits_.bit_count()`, but if created from a declaration
  // with specified bit count in SV source code, it is that specified count.
  int64_t effective_bit_count_;
};

// Represents a quoted literal string.
class QuotedString final : public Expression {
 public:
  QuotedString(std::string_view str, VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), str_(str) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string str_;
};

class XLiteral final : public Expression {
 public:
  using Expression::Expression;

  std::string Emit(LineInfo* line_info) const final { return "'X"; }
};

class ZeroLiteral final : public Expression {
 public:
  using Expression::Expression;

  std::string Emit(LineInfo* line_info) const final { return "'0"; }
};

// Represents a Verilog slice expression; e.g.
//
//    subject[hi:lo]
class Slice final : public Expression {
 public:
  Slice(IndexableExpression* subject, Expression* hi, Expression* lo,
        VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), subject_(subject), hi_(hi), lo_(lo) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  IndexableExpression* subject_;
  Expression* hi_;
  Expression* lo_;
};

// Represents a Verilog indexed part-select expression; e.g.
//
//    subject[start +: width]
class PartSelect final : public Expression {
 public:
  PartSelect(IndexableExpression* subject, Expression* start, Expression* width,
             VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        subject_(subject),
        start_(start),
        width_(width) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  IndexableExpression* subject_;
  Expression* start_;
  Expression* width_;
};

// Represents a Verilog indexing operation; e.g.
//
//    subject[index]
class Index final : public IndexableExpression {
 public:
  Index(IndexableExpression* subject, Expression* index, VerilogFile* file,
        const SourceInfo& loc)
      : IndexableExpression(file, loc), subject_(subject), index_(index) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  IndexableExpression* subject_;
  Expression* index_;
};

// Represents a Verilog ternary operator; e.g.
//
//    test ? consequent : alternate
class Ternary final : public Expression {
 public:
  Ternary(Expression* test, Expression* consequent, Expression* alternate,
          VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc),
        test_(test),
        consequent_(consequent),
        alternate_(alternate) {}

  std::string Emit(LineInfo* line_info) const final;
  int64_t precedence() const final { return 0; }

  Expression* test() const { return test_; }
  Expression* consequent() const { return consequent_; }
  Expression* alternate() const { return alternate_; }

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
class ContinuousAssignment final : public VastNode {
 public:
  ContinuousAssignment(Expression* lhs, Expression* rhs, VerilogFile* file,
                       const SourceInfo& loc)
      : VastNode(file, loc), lhs_(lhs), rhs_(rhs) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* lhs_;
  Expression* rhs_;
};

class BlankLine final : public Statement {
 public:
  BlankLine(VerilogFile* file, const SourceInfo& loc) : Statement(file, loc) {}
  using Statement::Statement;

  std::string Emit(LineInfo* line_info) const final { return ""; }
};

// Represents a SystemVerilog concurrent assert statement of the following
// form:
//
//   [LABEL:] assert property (
//       @(CLOCKING_EVENT)
//       [disable iff DISABLE_IFF] CONDITION)
//     else $fatal(0, message);
class ConcurrentAssertion final : public Statement {
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

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* condition_;
  Expression* clocking_event_;
  std::optional<Expression*> disable_iff_;
  std::string label_;
  std::string error_message_;
};

// Represents a SystemVerilog deferred assert statement of the following
// form:
//
//   [LABEL:] assert final (DISABLE_IFF || CONDITION)
//     else $fatal(0, message);
class DeferredImmediateAssertion final : public Statement {
 public:
  DeferredImmediateAssertion(Expression* condition,
                             std::optional<Expression*> disable_iff,
                             std::string_view label,
                             std::string_view error_message, VerilogFile* file,
                             const SourceInfo& loc);

  std::string Emit(LineInfo* line_info) const final;

 private:
  Expression* condition_;
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
class Cover final : public Statement {
 public:
  Cover(LogicRef* clk, Expression* condition, std::string_view label,
        VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), clk_(clk), condition_(condition), label_(label) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  LogicRef* clk_;
  Expression* condition_;
  std::string label_;
};

// Places a comment in statement position (we can think of comments as
// meaningless expression statements that do nothing).
class Comment final : public Statement {
 public:
  Comment(std::string_view text, VerilogFile* file, const SourceInfo& loc)
      : Statement(file, loc), text_(text) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string text_;
};

// A string which is emitted verbatim in the position of a statement.
class InlineVerilogStatement final : public Statement {
 public:
  InlineVerilogStatement(std::string_view text, VerilogFile* file,
                         const SourceInfo& loc)
      : Statement(file, loc), text_(text) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string text_;
};

// A reference to a definition defined within a InlineVerilogStatement. Can be
// used where a VAST node Ref is needed to refer to a value (wire, reg, etc)
// defined in a InlineVerilogStatement.
class InlineVerilogRef final : public IndexableExpression {
 public:
  InlineVerilogRef(std::string_view name, InlineVerilogStatement* raw_statement,
                   VerilogFile* file, const SourceInfo& loc)
      : IndexableExpression(file, loc), name_(name) {}

  std::string Emit(LineInfo* line_info) const final;

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

  std::string Emit(LineInfo* line_info) const final;

  const std::string& name() const { return name_; }
  std::optional<std::vector<Expression*>> args() const { return args_; }

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

  std::string Emit(LineInfo* line_info) const final;

  const std::string& name() const { return name_; }
  std::optional<std::vector<Expression*>> args() const { return args_; }

 private:
  std::string name_;
  std::optional<std::vector<Expression*>> args_;
};

// Represents a $display function call.
class Display final : public SystemTaskCall {
 public:
  Display(absl::Span<Expression* const> args, VerilogFile* file,
          const SourceInfo& loc)
      : SystemTaskCall("display", args, file, loc) {}
};

// Represents a $strobe function call.
class Strobe final : public SystemTaskCall {
 public:
  Strobe(absl::Span<Expression* const> args, VerilogFile* file,
         const SourceInfo& loc)
      : SystemTaskCall("strobe", args, file, loc) {}
};

// Represents a $monitor function call.
class Monitor final : public SystemTaskCall {
 public:
  Monitor(absl::Span<Expression* const> args, VerilogFile* file,
          const SourceInfo& loc)
      : SystemTaskCall("monitor", args, file, loc) {}
};

// Represents a $finish function call.
class Finish final : public SystemTaskCall {
 public:
  Finish(VerilogFile* file, const SourceInfo& loc)
      : SystemTaskCall("finish", file, loc) {}
};

// Represents a $signed function call which casts its argument to signed.
class SignedCast final : public SystemFunctionCall {
 public:
  SignedCast(Expression* value, VerilogFile* file, const SourceInfo& loc)
      : SystemFunctionCall("signed", {value}, file, loc) {}
};

// Represents a $unsigned function call which casts its argument to unsigned.
class UnsignedCast final : public SystemFunctionCall {
 public:
  UnsignedCast(Expression* value, VerilogFile* file, const SourceInfo& loc)
      : SystemFunctionCall("unsigned", {value}, file, loc) {}
};

// Represents the definition of a Verilog function.
class VerilogFunction final : public VastNode {
 public:
  VerilogFunction(std::string_view name, DataType* result_type,
                  VerilogFile* file, const SourceInfo& loc);

  // Adds an argument to the function and returns a reference to its value which
  // can be used in the body of the function.
  LogicRef* AddArgument(std::string_view name, DataType* type,
                        const SourceInfo& loc);

  LogicRef* AddArgument(Def* def, const SourceInfo& loc);

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

  RegDef* return_value_def() const { return return_value_def_; }
  absl::Span<Def* const> arguments() const { return argument_defs_; }
  StatementBlock* statement_block() const { return statement_block_; }
  absl::Span<RegDef* const> block_reg_defs() const { return block_reg_defs_; }

  // Returns the name of the function.
  std::string name() const { return name_; }

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string name_;
  RegDef* return_value_def_;

  // The block containing all of the statements of the function. SystemVerilog
  // allows multiple statements in a function without encapsulating them in a
  // begin/end block (StatementBlock), but Verilog does not so we choose the
  // least common denominator.
  StatementBlock* statement_block_;

  std::vector<Def*> argument_defs_;

  // The RegDefs of reg's defined in the function. These are emitted before the
  // statement block.
  std::vector<RegDef*> block_reg_defs_;
};

// Represents a call to a VerilogFunction.
class VerilogFunctionCall final : public Expression {
 public:
  VerilogFunctionCall(VerilogFunction* func, absl::Span<Expression* const> args,
                      VerilogFile* file, const SourceInfo& loc)
      : Expression(file, loc), func_(func), args_(args.begin(), args.end()) {}

  std::string Emit(LineInfo* line_info) const final;

  VerilogFunction* func() const { return func_; }
  absl::Span<Expression* const> args() const { return args_; }

 private:
  VerilogFunction* func_;
  std::vector<Expression*> args_;
};

class ModuleSection;
class ModuleConditionalDirective;

// Represents a member of a module.
using ModuleMember =
    std::variant<Def*,                     // Logic definition.
                 LocalParam*,              // Module-local parameter.
                 Parameter*,               // Module parameter.
                 Instantiation*,           // module instantiation.
                 ContinuousAssignment*,    // Continuous assignment.
                 StructuredProcedure*,     // Initial or always comb block.
                 AlwaysComb*,              // An always_comb block.
                 AlwaysFf*,                // An always_ff block.
                 AlwaysFlop*,              // "Flip-Flop" block.
                 Comment*,                 // Comment text.
                 BlankLine*,               // Blank line.
                 InlineVerilogStatement*,  // InlineVerilog string statement.
                 VerilogFunction*,         // Function definition
                 Typedef*, Enum*, Cover*, ConcurrentAssertion*,
                 DeferredImmediateAssertion*, ModuleConditionalDirective*,
                 ModuleSection*,
                 // Generate loop, can effectively generate more module members
                 // at elaboration time
                 GenerateLoop*>;

// A ModuleSection is a container of ModuleMembers used to organize the contents
// of a module. A Module contains a single top-level ModuleSection which may
// contain other ModuleSections. ModuleSections enables modules to be
// constructed a non-linear, random-access fashion by appending members to
// different sections rather than just appending to the end of module.
//
// TODO(meheff): Move Module methods AddReg*, AddWire*, etc to ModuleSection.
//
// TODO(tedhong): Alternatively, move AddX functions to free functions, and
// add extra inheritance levels/type traits constrain which VastNode
// (ModuleSection, Module, PackageSection, Package, etc...) supports adding X
// construct.
class ModuleSection final : public VastNode {
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

  const std::vector<ModuleMember>& members() const { return members_; }

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::vector<ModuleMember> members_;
};

// Represents an `ifdef block within a module context.
class ModuleConditionalDirective final : public VastNode {
 public:
  ModuleConditionalDirective(ConditionalDirectiveKind kind,
                             std::string identifier, VerilogFile* file,
                             const SourceInfo& loc);

  // Returns a pointer to the statement block of the consequent.
  ModuleSection* consequent() const { return consequent_; }

  // Adds an alternate clause ("`elsif" or "`else") and returns a pointer to the
  // consequent. The alternate is final (an "`else") if identifier is empty.
  // Dies if a final alternate ("`else") clause has been previously added.
  ModuleSection* AddAlternate(std::string identifier = "");

  std::string Emit(LineInfo* line_info) const final;

 private:
  ConditionalDirectiveKind kind_;
  std::string identifier_;
  ModuleSection* consequent_;

  // The alternate clauses ("`elsif" and "`else"). If the string is empty then
  // the alternate is unconditional ("`else"). This can only appear as the final
  // alternate.
  std::vector<std::pair<std::string, ModuleSection*>> alternates_;
};

// Specifies input/output direction.
enum class ModulePortDirection {
  kInput,
  kOutput,
};

std::string ToString(ModulePortDirection direction);
std::ostream& operator<<(std::ostream& os, ModulePortDirection d);

// Represents a module port.
struct ModulePort {
  const std::string& name() const { return wire->GetName(); }
  std::string ToString() const;

  ModulePortDirection direction;
  Def* wire;
};

// Represents a module definition.
class Module final : public VastNode {
 public:
  Module(std::string_view name, VerilogFile* file, const SourceInfo& loc)
      : VastNode(file, loc), name_(name), top_(file, loc) {}

  // Constructs and adds a node to the module. Ownership is maintained by the
  // parent VerilogFile. Most constructs should be added to the module. The
  // exceptions are the AddFoo convenience methods defined below which return a
  // Ref to the object created rather than the object itself.
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  absl::StatusOr<LogicRef*> AddInput(std::string_view name, DataType* type,
                                     const SourceInfo& loc);
  absl::StatusOr<LogicRef*> AddOutput(std::string_view name, DataType* type,
                                      const SourceInfo& loc);

  absl::StatusOr<LogicRef*> AddReg(std::string_view name, DataType* type,
                                   const SourceInfo& loc,
                                   Expression* init = nullptr,
                                   ModuleSection* section = nullptr);
  absl::StatusOr<LogicRef*> AddWire(std::string_view name, DataType* type,
                                    const SourceInfo& loc,
                                    ModuleSection* section = nullptr);
  // Variation of AddWire that takes an initializer expression.
  absl::StatusOr<LogicRef*> AddWire(std::string_view name, DataType* type,
                                    Expression* init, const SourceInfo& loc,
                                    ModuleSection* section = nullptr);
  absl::StatusOr<LogicRef*> AddInteger(std::string_view name,
                                       const SourceInfo& loc,
                                       ModuleSection* section = nullptr);
  absl::StatusOr<LogicRef*> AddGenvar(std::string_view name,
                                      const SourceInfo& loc,
                                      ModuleSection* section = nullptr);

  ParameterRef* AddParameter(std::string_view name, Expression* rhs,
                             const SourceInfo& loc);
  ParameterRef* AddParameter(Def* def, Expression* rhs, const SourceInfo& loc);

  Typedef* AddTypedef(Def* def, const SourceInfo& loc);

  // Adds a previously constructed VAST construct to the module.
  template <typename T>
  T AddModuleMember(T member) {
    top_.AddModuleMember(member);
    return member;
  }

  ModuleSection* top() { return &top_; }

  absl::Span<const ModulePort> ports() const { return ports_; }
  const std::string& name() const { return name_; }

  std::string Emit(LineInfo* line_info) const final;

 private:
  // Note: these `*Internal` variants don't do any already-defined checking.

  // Adds a (wire) port to this module with the given name and type. Returns
  // a reference to that wire.
  LogicRef* AddInputInternal(std::string_view name, DataType* type,
                             const SourceInfo& loc);
  LogicRef* AddOutputInternal(std::string_view name, DataType* type,
                              const SourceInfo& loc);

  // Adds a reg/wire definition to the module with the given type and, for regs,
  // initialized with the given value. Returns a reference to the definition.
  LogicRef* AddRegInternal(std::string_view name, DataType* type,
                           const SourceInfo& loc, Expression* init = nullptr,
                           ModuleSection* section = nullptr);
  LogicRef* AddWireInternal(std::string_view name, DataType* type,
                            const SourceInfo& loc,
                            ModuleSection* section = nullptr);
  // Variation of AddWire that takes an initializer expression.
  LogicRef* AddWireInternal(std::string_view name, DataType* type,
                            Expression* init, const SourceInfo& loc,
                            ModuleSection* section = nullptr);
  LogicRef* AddIntegerInternal(std::string_view name, const SourceInfo& loc,
                               ModuleSection* section = nullptr);
  LogicRef* AddGenvarInternal(std::string_view name, const SourceInfo& loc,
                              ModuleSection* section = nullptr);

  // Add the given Def as a port on the module.
  LogicRef* AddPortDef(ModulePortDirection direction, Def* def,
                       const SourceInfo& loc);

  std::string name_;
  std::vector<ModulePort> ports_;
  absl::flat_hash_set<std::string> defined_names_;

  ModuleSection top_;
};

class VerilogPackageSection;

// Represents a member of a package.
using VerilogPackageMember =
    std::variant<Parameter*,               // Package parameter/constant.
                 Comment*,                 // Comment text.
                 BlankLine*,               // Blank line.
                 InlineVerilogStatement*,  // InlineVerilog string statement.
                 Typedef*, VerilogPackageSection*>;

// A ParameterSection is a container of ParameterMembers used to organize the
// contents of a package. A Package contains a single top-level PackageSection
// which may contain other PackageSections. PackageSections enables packages
// to be constructed a non-linear, random-access fashion by appending members to
// different sections rather than just appending to the end of package.
class VerilogPackageSection final : public VastNode {
 public:
  using VastNode::VastNode;

  // Constructs and adds a package member of type T to the section. Ownership is
  // maintained by the parent VerilogFile. Templatized on T in order to return a
  // pointer to the derived type.  Most constructs should be added to the
  // package. The exceptions are the AddFoo convenience methods defined below
  // which return a Ref to the object created rather than the object itself.
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  template <typename T>
  T AddVerilogPackageMember(T member) {
    members_.push_back(member);
    return member;
  }

  // Adds a enum typedef to the package section.  Returns a type of the enum's
  // typedef rather than the enum itself.
  TypedefType* AddEnumTypedef(std::string_view name, Enum* enum_data_type,
                              const SourceInfo& loc);
  TypedefType* AddEnumTypedef(std::string_view name, DataKind kind,
                              DataType* data_type,
                              absl::Span<EnumMember*> enum_members,
                              const SourceInfo& loc);

  // Adds a struct typedef to the package section.  Returns a the type of the
  // typedef rather than the typedef itself.
  TypedefType* AddStructTypedef(std::string_view name, Struct* struct_data_type,
                                const SourceInfo& loc);
  TypedefType* AddStructTypedef(std::string_view name,
                                absl::Span<Def*> struct_members,
                                const SourceInfo& loc);

  // Adds a parameter to the package section.  Returns a reference to the
  // parameter rather than the parameter itself.
  ParameterRef* AddParameter(std::string_view name, Expression* rhs,
                             const SourceInfo& loc);
  ParameterRef* AddParameter(Def* def, Expression* rhs, const SourceInfo& loc);

  const std::vector<VerilogPackageMember>& members() const { return members_; }

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::vector<VerilogPackageMember> members_;
};

// Represents a package definition. Emits as:
// ```
// package ${name};
//   .. content ..
// endpackage
// ```
class VerilogPackage final : public VastNode {
 public:
  VerilogPackage(std::string_view name, VerilogFile* file,
                 const SourceInfo& loc)
      : VastNode(file, loc), name_(name), top_(file, loc) {}

  // Constructs and adds a node to the package's top section. Ownership is
  // maintained by the parent VerilogFile.
  template <typename T, typename... Args>
  inline T* Add(const SourceInfo& loc, Args&&... args);

  // Adds a previously constructed VAST construct to the module.
  template <typename T>
  T AddVerilogPackageMember(T member) {
    top_.AddVerilogPackageMember(member);
    return member;
  }

  VerilogPackageSection* top() { return &top_; }

  const std::string& name() const { return name_; }

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string name_;
  VerilogPackageSection top_;
};

// Represents a file-level inclusion directive.
class Include final : public VastNode {
 public:
  Include(std::string_view path, VerilogFile* file, const SourceInfo& loc)
      : VastNode(file, loc), path_(path) {}

  std::string Emit(LineInfo* line_info) const final;

 private:
  std::string path_;
};

using FileMember =
    std::variant<Module*, VerilogPackage*, Include*, BlankLine*, Comment*>;

// Represents a file (as a Verilog translation-unit equivalent).
class VerilogFile {
 public:
  explicit VerilogFile(FileType file_type) : file_type_(file_type) {}

  VerilogPackage* AddVerilogPackage(std::string_view name,
                                    const SourceInfo& loc) {
    return Add(Make<VerilogPackage>(loc, name));
  }
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
    CHECK_GE(hi, 0);
    CHECK_GE(lo, 0);
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
    CHECK_GT(width, 0);
    return Make<verilog::PartSelect>(loc, subject, start,
                                     MaybePlainLiteral(width, loc));
  }

  verilog::Index* Index(IndexableExpression* subject, Expression* index,
                        const SourceInfo& loc) {
    return Make<verilog::Index>(loc, subject, index);
  }
  verilog::Index* Index(IndexableExpression* subject, int64_t index,
                        const SourceInfo& loc) {
    CHECK_GE(index, 0);
    return Make<verilog::Index>(loc, subject, MaybePlainLiteral(index, loc));
  }

  Unary* Negate(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, expression, OperatorKind::kNegate);
  }
  Unary* BitwiseNot(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, expression, OperatorKind::kBitwiseNot);
  }
  Unary* LogicalNot(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, expression, OperatorKind::kLogicalNot);
  }
  Unary* AndReduce(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, expression, OperatorKind::kAndReduce);
  }
  Unary* OrReduce(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, expression, OperatorKind::kOrReduce);
  }
  Unary* XorReduce(Expression* expression, const SourceInfo& loc) {
    return Make<Unary>(loc, expression, OperatorKind::kXorReduce);
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
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kAdd);
  }
  BinaryInfix* LogicalAnd(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kLogicalAnd);
  }
  BinaryInfix* BitwiseAnd(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kBitwiseAnd);
  }
  BinaryInfix* NotEquals(Expression* lhs, Expression* rhs,
                         const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kNe);
  }
  BinaryInfix* CaseNotEquals(Expression* lhs, Expression* rhs,
                             const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kCaseNe);
  }
  BinaryInfix* Equals(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kEq);
  }
  BinaryInfix* CaseEquals(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kCaseEq);
  }
  BinaryInfix* GreaterThanEquals(Expression* lhs, Expression* rhs,
                                 const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kGe);
  }
  BinaryInfix* GreaterThan(Expression* lhs, Expression* rhs,
                           const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kGt);
  }
  BinaryInfix* LessThanEquals(Expression* lhs, Expression* rhs,
                              const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kLe);
  }
  BinaryInfix* LessThan(Expression* lhs, Expression* rhs,
                        const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kLt);
  }
  BinaryInfix* Div(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kDiv);
  }
  BinaryInfix* Mod(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kMod);
  }
  BinaryInfix* Mul(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kMul);
  }
  BinaryInfix* Power(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kPower);
  }
  BinaryInfix* BitwiseOr(Expression* lhs, Expression* rhs,
                         const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kBitwiseOr);
  }
  BinaryInfix* LogicalOr(Expression* lhs, Expression* rhs,
                         const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kLogicalOr);
  }
  BinaryInfix* BitwiseXor(Expression* lhs, Expression* rhs,
                          const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kBitwiseXor);
  }
  BinaryInfix* Shll(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kShll);
  }
  BinaryInfix* Shra(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kShra);
  }
  BinaryInfix* Shrl(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kShrl);
  }
  BinaryInfix* Sub(Expression* lhs, Expression* rhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, rhs, OperatorKind::kSub);
  }

  // Only for use in testing.
  BinaryInfix* NotEqualsX(Expression* lhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, XLiteral(loc), OperatorKind::kNeX);
  }
  BinaryInfix* EqualsX(Expression* lhs, const SourceInfo& loc) {
    return Make<BinaryInfix>(loc, lhs, XLiteral(loc), OperatorKind::kEqX);
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
    CHECK((value == 0) || (value == 1));
    return Make<verilog::Literal>(loc, UBits(value, 1), FormatPreference::kHex);
  }

  // Creates a decimal literal representing a plain decimal number without a bit
  // count prefix (e.g., "42"). Use for clarity when bit width does not matter,
  // for example, as bit-slice indices.
  verilog::Literal* PlainLiteral(int32_t value, const SourceInfo& loc) {
    return Make<verilog::Literal>(loc, SBits(value, 32),
                                  FormatPreference::kDefault, 32,
                                  /*emit_bit_count=*/false,
                                  /*declared_as_signed=*/true);
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

  DataType* ExternType(DataType* punable_type, std::string_view name,
                       const SourceInfo& loc);

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

  const std::vector<FileMember>& members() const { return members_; }

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
inline T* MacroStatementBlock::Add(const SourceInfo& loc, Args&&... args) {
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

template <typename T, typename... Args>
inline T* VerilogPackage::Add(const SourceInfo& loc, Args&&... args) {
  return top()->Add<T>(loc, std::forward<Args>(args)...);
}

template <typename T, typename... Args>
inline T* VerilogPackageSection::Add(const SourceInfo& loc, Args&&... args) {
  T* ptr = file()->Make<T>(loc, std::forward<Args>(args)...);
  AddVerilogPackageMember(ptr);
  return ptr;
}

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_VAST_VAST_H_
