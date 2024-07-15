#ifndef XLS_IR_OP_SPECIFICATION_H_
#define XLS_IR_OP_SPECIFICATION_H_

#include <string>
#include <optional>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/btree_set.h"
#include "absl/strings/str_join.h"

namespace xls {

// Describes an argument to the `Node` class constructor.
struct ConstructorArgument {
  // Name of the argument.
  std::string name;
  // The C++ type of the argument.
  std::string cpp_type;
  // The expression to use, when calling the constructor, when cloning.
  std::optional<std::string> clone_expression;
};

// Describes a data member of the Node class.
struct DataMember {
  // Name of the data member; should have a trailing '_' to follow the C++
  // style guide.
  std::string name;
  std::string cpp_type;
  std::string init;
  // A format string defining the expression for testing this member for
  // equality. The format fields are named 'lh' and 'rhs'.
  std::string equals_tmpl = "{lhs} == {rhs}";
};

struct MethodOptions {
  // True iff expression should be used as a body rather than a return value.
  bool expression_is_body = false;
  // Optional string of C++ parameters that get inserted into the method signature between "()"s.
  std::string params;
  // The method is a const method.
  bool is_const = true;
};

// Describes a method of the Node class.
class Method {
 public:
  explicit Method(std::string name,
                  std::string return_cpp_type,
                  std::optional<std::string> expression,
                  MethodOptions options = MethodOptions{})
    : name_(std::move(name)),
      return_cpp_type_(std::move(return_cpp_type)),
      expression_(std::move(expression)),
      options_(std::move(options))
  {}

  std::string_view name() const { return name_; }
  std::string_view return_cpp_type() const { return return_cpp_type_; }
  bool is_const() const { return options_.is_const; }
  bool expression_is_body() const { return options_.expression_is_body; }
  const std::optional<std::string>& expression() const { return expression_; }
  std::string_view params() const { return options_.params; }

 private:
  // Name of the method.
  std::string name_;
  // The C++ type of the value returned by the method.
  std::string return_cpp_type_;
  // The expression to produce the value returned by this method.
  std::optional<std::string> expression_;
  MethodOptions options_;
};

struct AttributeOptions {
  std::optional<std::string> arg_cpp_type;
  std::optional<std::string> return_cpp_type;
  std::string equals_tmpl = "{lhs} == {rhs}";
  std::vector<std::string> init_args;
};

// Desribes an attribute of a Node class.
//
// An Attribute desugars into a ConstructorArgument, DataMember, and an accessor Method.
class Attribute {
 public:
  explicit Attribute(
    std::string_view name,
    std::string_view cpp_type,
    AttributeOptions options = AttributeOptions{})
    : name_(name),
      constructor_argument_(ConstructorArgument{
        .name = std::string{name},
        .cpp_type = options.arg_cpp_type.has_value() ? options.arg_cpp_type.value() : std::string{cpp_type},
        .clone_expression = std::make_optional(absl::StrCat(name, "()"))
      }),
      data_member_(DataMember{
        .name = absl::StrCat(name, "_"),
        .cpp_type = std::string{cpp_type},
        .init = options.init_args.empty() ? std::string{name} : absl::StrJoin(options.init_args, ", "),
        .equals_tmpl = std::move(options.equals_tmpl),
      }),
      method_(Method{
        std::string{name},
        /*return_cpp_type=*/options.return_cpp_type.has_value() ? options.return_cpp_type.value() : std::string{cpp_type},
        /*expression=*/std::make_optional(data_member_.name)
      }) {}

  std::string_view name() const { return name_; }
  const ConstructorArgument& constructor_argument() const {
    return constructor_argument_;
  }
  const DataMember& data_member() const { return data_member_; }
  const Method& method() const { return method_; }
  std::string_view clone_expression() const {
    return constructor_argument_.clone_expression.value();
  }

 private:
  // Name of the attribute. The constructor argument and accessor method share this name. The data member is the same name with a '_' suffix.
  std::string name_;
  // The ConstructorArgument of the attribute.
  ConstructorArgument constructor_argument_;
  // The DatatMember of the attribute.
  DataMember data_member_;
  // The accessor Method of the attribute.
  Method method_;
};

class BoolAttribute : public Attribute {
 public:
  explicit BoolAttribute(std::string_view name) : Attribute{name, "bool"} {}
};

class Int64Attribute : public Attribute {
 public:
  explicit Int64Attribute(std::string_view name) : Attribute{name, "int64_t"} {}
};

class TypeAttribute : public Attribute {
  public:
    explicit TypeAttribute(std::string_view name) : Attribute{name, "Type*"} {}
};

class FunctionAttribute : public Attribute {
 public:
  explicit FunctionAttribute(std::string_view name)
    : Attribute(name, "Function*", AttributeOptions{
        .equals_tmpl = "{lhs}->IsDefinitelyEqualTo({rhs})",
      }) {}

};

class ValueAttribute : public Attribute {
 public:
  explicit ValueAttribute(std::string_view name)
    : Attribute{name, /*cpp_type=*/"Value", AttributeOptions{
        .return_cpp_type="const Value&"
      }} {}
};

class StringAttribute : public Attribute {
  public:
    explicit StringAttribute(std::string_view name) : Attribute{name, /*cpp_type=*/"std::string",
      AttributeOptions{
        .arg_cpp_type="std::string_view",
        .return_cpp_type="const std::string&",
      }} {}
};

class OptionalStringAttribute : public Attribute {
  public:
    explicit OptionalStringAttribute(std::string_view name) : Attribute{name, /*cpp_type=*/"std::optional<std::string>",
    AttributeOptions{
      .arg_cpp_type="std::optional<std::string>",
      .return_cpp_type="std::optional<std::string>",
    }} {}
};

class LsbOrMsbAttribute : public Attribute {
  public:
    explicit LsbOrMsbAttribute(std::string_view name) : Attribute{name, /*cpp_type=*/"LsbOrMsb",
      AttributeOptions{
        .arg_cpp_type="LsbOrMsb",
        .return_cpp_type="LsbOrMsb",
      }} {}
};

class TypeVectorAttribute : public Attribute {
  public:
    explicit TypeVectorAttribute(std::string_view name) : Attribute{name, /*cpp_type=*/"std::vector<Type*>", AttributeOptions{
      .arg_cpp_type = "absl::Span<Type* const>",
      .return_cpp_type = "absl::Span<Type* const>",
      .init_args = {absl::StrCat(name, ".begin()"), absl::StrCat(name, ".end()")},
    }} {
    }
};

class FormatStepsAttribute : public Attribute {
  public:
    explicit FormatStepsAttribute(std::string_view name) : Attribute{name, /*cpp_type=*/"std::vector<FormatStep>", 
      AttributeOptions{
        .arg_cpp_type="absl::Span<FormatStep const>",
        .return_cpp_type="absl::Span<FormatStep const>",
        .init_args = {absl::StrCat(name, ".begin()"), absl::StrCat(name, ".end()")},
      }} {}
};

class InstantiationAttribute : public Attribute {
  public:
    explicit InstantiationAttribute(std::string_view name) : Attribute{name, /*cpp_type=*/"Instantiation*"} {}
};

// Enumeration of properties of Ops.
//
// An Op can have zero or more properties.
enum class Property {
  kBitwise = 1,  // Ops such as kXor, kAnd, kOr, etc.
  kAssociative = 2,
  kCommutative = 3,
  kComparison = 4,
  kSideEffecting = 5,
};

enum class OperandKind {
  kDefault,
  kSpan,
  kOptional,
};

class OptionalOperand;

class Operand {
 public:
  explicit Operand(std::string name, OperandKind kind = OperandKind::kDefault)
    : name_(std::move(name)), kind_(kind) {}

  virtual ~Operand() = default;

  std::string_view name() const { return name_; }
  virtual std::string_view GetAddMethod() const { return "AddOperand"; }
  OperandKind kind() const { return kind_; }

  std::string CamelCaseName() const;

  const OptionalOperand& AsOptionalOrDie() const;

 private:
  std::string name_;
  OperandKind kind_;
};

class OperandSpan final : public Operand {
 public:
  explicit OperandSpan(std::string name) : Operand{std::move(name), OperandKind::kSpan} {}

  std::string_view GetAddMethod() const override { return "AddOperands"; }
};

// An operand that may be omitted.
class OptionalOperand final : public Operand {
 public:
  // Args:
  //  name: The name of the operand.
  //  manual_optional_implementation: If a '<name>_operand_number()' function,
  //    then has_<name>_ field and other helpers should be automatically created.
  //    This should only be set to true if the operand folows an operand-span and the node has a custom implementation.
  explicit OptionalOperand(std::string name, bool manual_optional_implementation)
    : Operand{std::move(name), OperandKind::kOptional}, manual_optional_implementation_(manual_optional_implementation) {}

  std::string_view GetAddMethod() const override { return "AddOptionalOperand"; }
  bool manual_optional_implementation() const { return manual_optional_implementation_; }

 private:
  bool manual_optional_implementation_;
};

// Holds information about a single operand.
class OperandInfo {
 public:
  explicit OperandInfo(const Operand& operand, size_t index)
    : operand_(operand), index_(index) {}

  const Operand& operand() const { return operand_; }
  size_t index() const { return index_; }

 private:
  // The operand data.
  const Operand& operand_;
  // What index this is provisionally assigned in the operand list (assuming
  // all spans are 0-length and all optional operands are present).
  size_t index_;
};

struct OpClassOptions {
  std::vector<std::unique_ptr<Attribute>> attributes;
  std::vector<ConstructorArgument> extra_constructor_args;
  std::vector<DataMember> extra_data_members;
  std::vector<Method> extra_methods;
  bool custom_clone_method;
};

// Describes a C++ subclass of `xls::Node`.
class OpClass {
 public:
  explicit OpClass(
    std::string name,
    std::string op,
    std::vector<std::unique_ptr<Operand>> operands,
    std::string xls_type_expression,
    OpClassOptions options = OpClassOptions{})
    : name_(std::move(name)),
      op_(std::move(op)),
      operands_(std::move(operands)),
      xls_type_expression_(std::move(xls_type_expression)),
      options_(std::move(options)) {}

  std::string_view name() const { return name_; }
  std::string_view op() const { return op_; }
  absl::Span<const std::unique_ptr<Operand>> operands() const { return operands_; }
  std::string_view xls_type_expression() const { return xls_type_expression_; }
  bool custom_clone_method() const { return options_.custom_clone_method; }

  std::vector<Method> GetMethods() const;

  std::vector<DataMember> GetDataMembers() const;

  bool HasDataMembers() const { return !GetDataMembers().empty(); }

  // Returns the arguments to pass to the constructor during cloning.
  //
  // Args:
  //  new_operands: The name of the span variable containing the new operands
  //    during cloning.
  std::string GetCloneArgsStr(std::string new_operands) const;

  absl::Span<const ConstructorArgument> extra_constructor_args() const {
    return options_.extra_constructor_args;
  }

  absl::flat_hash_set<std::string> GetExtraConstructorArgNames() const {
    absl::flat_hash_set<std::string> names;
    for (const ConstructorArgument& a : extra_constructor_args()) {
      names.insert(a.name);
    }
    return names;
  }

  // Get all the `std::optional<Node*>` operands.
  //
  // Returns OperandInfo for each optional operand in the order they appear.
  std::vector<OperandInfo> GetOptionalOperands() const {
    std::vector<OperandInfo> operand_info;
    for (size_t i = 0; i < operands_.size(); ++i) {
      const Operand& operand = *operands_[i];
      if (operand.kind() == OperandKind::kOptional) {
        operand_info.push_back(OperandInfo(operand, i));
      }
    }
    return operand_info;
  }

  std::string GetConstructorArgsStr() const;

  std::string GetBaseConstructorInvocation() const;

  // Returns expression used in IsDefinitelyEqualTo to compare expression.
  std::string GetEqualToExpr() const;

  // Is the operand one with an exact statically known offset?
  //
  // Args:
  //  op: The operand to query.
  //
  // Returns: true iff the operand is not a span or an optional.
  bool IsFixedOperand(const Operand& op) const {
    switch (op.kind()) {
     case OperandKind::kSpan:
     case OperandKind::kOptional:
      return false;
     default:
      return true;
    }
  }

  // Get all the operands with compile-time known offsets.
  //
  // Note that any operand after the first optional or variable-length (think
  // span) operand does not have a fixed offset.
  std::vector<OperandInfo> GetFixedOperands() const {
    std::vector<OperandInfo> operand_info;
    for (size_t i = 0; i < operands_.size(); ++i) {
      const Operand& operand = *operands_[i];
      if (!IsFixedOperand(operand)) {
        continue;
      }
      operand_info.push_back(OperandInfo(operand, i));
    }
    return operand_info;
  }

 private:
  // The name of the class.
  std::string name_;
  // The expression for the Op associated with this class (e.g. `Op::kParam`).
  std::string op_;
  std::vector<std::unique_ptr<Operand>> operands_;
  std::string xls_type_expression_;
  OpClassOptions options_;
};

struct Op {
  // The name of the C++ enum value (e.g. 'kParam').
  std::string enum_name;

  // The name of the op as it appears in the textual IR (e.g. 'param').
  std::string name;

  // The value indicating the C++ node subclass of the op.
  const OpClass& op_class;

  // A list of properties describing the op.
  //
  // We use a btree for stable traversal order in emitting text.
  absl::btree_set<Property> properties;
};

const absl::btree_map<std::string, OpClass>& GetOpClassKindsSingleton();

const std::vector<Op>& GetOpsSingleton();

}  // namespace xls

#endif  // XLS_IR_OP_SPECIFICATION_H_
