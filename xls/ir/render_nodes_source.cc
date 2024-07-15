#include "xls/ir/render_nodes_source.h"

#include "xls/ir/op_specification.h"
#include "absl/strings/str_replace.h"

namespace xls {
namespace {

std::string RenderDefinitelyEqualTo(const OpClass& op_class) {
  const std::string_view kTemplate = R"(bool {OP_CLASS_NAME}::IsDefinitelyEqualTo(const Node* other) const {
  if (this == other) {
    return true;
  }
  if (!Node::IsDefinitelyEqualTo(other)) {
    return false;
  }

  return {EQUAL_TO_EXPR};
}
)";
  return absl::StrReplaceAll(kTemplate, {
    {"{OP_CLASS_NAME}", op_class.name()},
    {"{EQUAL_TO_EXPR}", op_class.GetEqualToExpr()},
  });
}

}  // namespace

std::string RenderConstructor(const OpClass& op_class) {
  const std::string_view kTemplate = R"({OP_CLASS_NAME}::{OP_CLASS_NAME}({ARGS_STR})
    : {BASE_CONSTRUCTOR_INVOCATION}
      {INITIALIZER_LIST}
{
  CHECK(IsOpClass<{OP_CLASS_NAME}>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `{OP_CLASS_NAME}`.";
  {ADD_METHODS}
})";

  // Note: if the initializer list is non-empty it should start with a comma
  // since base constructor initialization precedes it.
  std::string base_constructor_invocation = op_class.GetBaseConstructorInvocation();

  std::vector<std::string> initializer_list_parts;
  for (const DataMember& member : op_class.GetDataMembers()) {
    initializer_list_parts.push_back(absl::StrFormat("%s(%s)", member.name, member.init));
  }

  const std::string initializer_list = absl::StrJoin(initializer_list_parts, ", ");
  if (!initializer_list.empty()) {
    base_constructor_invocation += ",";
  }

  std::vector<std::string> add_methods_parts;
  for (const auto& op : op_class.operands()) {
    add_methods_parts.push_back(absl::StrFormat("%s(%s);", op->GetAddMethod(), op->name()));
  }
  std::string add_methods = absl::StrJoin(add_methods_parts, "\n  ");

  return absl::StrReplaceAll(kTemplate, {
    {"{OP_CLASS_NAME}", op_class.name()},
    {"{ARGS_STR}", op_class.GetConstructorArgsStr()},
    {"{BASE_CONSTRUCTOR_INVOCATION}", base_constructor_invocation},
    {"{INITIALIZER_LIST}", initializer_list},
    {"{ADD_METHODS}", add_methods},
  });
}

std::string RenderStandardCloneMethod(const OpClass& op_class) {
  return absl::StrFormat(R"(absl::StatusOr<Node*>
%s::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<%s>(%s);
})", op_class.name(), op_class.name(), op_class.GetCloneArgsStr("new_operands"));
}

std::string RenderNodesSource() {
  const std::string_view kTemplate = R"(#include "xls/ir/nodes.h"

#include "absl/log/check.h"
#include "absl/status/statusor.h"

#include "xls/common/status/ret_check.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {

namespace {

Type* GetTupleType(Package* package, absl::Span<Node* const> operands) {
  std::vector<Type*> operand_types;
  for (Node* operand : operands) {
    operand_types.push_back(operand->GetType());
  }
  return package->GetTupleType(operand_types);
}

Type* GetConcatType(Package* package, absl::Span<Node* const> operands) {
  int64_t width = 0;
  for (Node* operand : operands) {
    width += operand->BitCountOrDie();
  }
  return package->GetBitsType(width);
}

Type* GetArrayConcatType(Package* package, absl::Span<Node* const> operands) {
  int64_t size = 0;
  Type* element_type = nullptr;

  for (Node* operand : operands) {
    auto operand_type = operand->GetType()->AsArrayOrDie();

    size += operand_type->size();
    if (element_type == nullptr) {
      // Set element_type to the first operand's element type
      element_type = operand_type->element_type();
    }
  }

  CHECK(element_type);
  return package->GetArrayType(size, element_type);
}

Type* GetMapType(Node* operand, Function* to_apply) {
  return operand->package()->GetArrayType(operand->GetType()->AsArrayOrDie()->size(),
                                 to_apply->return_value()->GetType());
}

Type* GetReceivePayloadType(FunctionBase* function_base, std::string_view channel_name) {
  return function_base->AsProcOrDie()->GetChannelReferenceType(channel_name).value();
}

Type* GetReceiveType(FunctionBase* function_base, std::string_view channel_name, bool is_blocking) {
  if(is_blocking) {
    return function_base->package()->GetTupleType(
        {function_base->package()->GetTokenType(),
         GetReceivePayloadType(function_base, channel_name)});
  }

  return function_base->package()->GetTupleType(
      {function_base->package()->GetTokenType(),
       GetReceivePayloadType(function_base, channel_name),
       function_base->package()->GetBitsType(1)});
}

}  // namespace

{OP_CLASS_CONSTRUCTORS}

{OP_CLASS_STANDARD_CLONE_METHODS}

{OP_CLASS_DEFINITELY_EQUAL_TO_METHODS}

SliceData Concat::GetOperandSliceData(int64_t operandno) const {
  CHECK_GE(operandno, 0);
  int64_t start = 0;
  for (int64_t i = operands().size()-1; i > operandno; --i) {
    Node* operand = this->operand(i);
    start += operand->BitCountOrDie();
  }
  return SliceData{start, this->operand(operandno)->BitCountOrDie()};
}

absl::StatusOr<Node*> Param::CloneInNewFunction(absl::Span<Node* const> new_operands,
                                           FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  XLS_ASSIGN_OR_RETURN(Type* new_type,
    new_function->package()->MapTypeFromOtherPackage(GetType()));
  return new_function->MakeNodeWithName<Param>(loc(), new_type, name_);
}

absl::StatusOr<Node*> Array::CloneInNewFunction(absl::Span<Node* const> new_operands,
                                           FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  XLS_RET_CHECK_EQ(size(), new_operands.size());
  XLS_ASSIGN_OR_RETURN(Type* new_element_type, 
    new_function->package()->MapTypeFromOtherPackage(element_type()));
  return new_function->MakeNodeWithName<Array>(loc(),
                                      new_operands,
                                      new_element_type,
                                      name_);
}

absl::StatusOr<Node*> CountedFor::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<CountedFor>(loc(), new_operands[0],
                                            new_operands.subspan(1),
                                            trip_count(),
                                            stride(),
                                            body(),
                                            name_);
}

absl::StatusOr<Node*> DynamicCountedFor::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<DynamicCountedFor>(loc(), new_operands[0],
                                            new_operands[1],
                                            new_operands[2],
                                            new_operands.subspan(3),
                                            body(),
                                            name_);
}

absl::StatusOr<Node*> Select::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  std::optional<Node*> new_default_value =
    default_value().has_value()
    ? std::optional<Node*>(new_operands.back())
    : std::nullopt;
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Select>(loc(), new_operands[0],
                                        new_operands.subspan(1, cases_size_),
                                        new_default_value,
                                        name_);
}

absl::StatusOr<Node*> OneHotSelect::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<OneHotSelect>(loc(), new_operands[0],
                                              new_operands.subspan(1),
                                              name_);
}

absl::StatusOr<Node*> PrioritySelect::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<PrioritySelect>(loc(), new_operands[0],
                                        new_operands.subspan(1, cases_size_),
                                        new_operands.back(),
                                        name_);
}

absl::StatusOr<Node*> ArrayIndex::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<ArrayIndex>(loc(),
                                      new_operands[0],
                                      new_operands.subspan(1),
                                      name_);
}

absl::StatusOr<Node*> ArrayUpdate::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<ArrayUpdate>(loc(),
                                      new_operands[0],
                                      new_operands[1],
                                      new_operands.subspan(2),
                                      name_);
}

absl::StatusOr<Node*> Trace::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(amfv): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Trace>(loc(),
                                      new_operands[0],
                                      new_operands[1],
                                      new_operands.subspan(2),
                                      format(),
                                      verbosity(),
                                      name_);
}

Type* Receive::GetPayloadType() const {
  return GetReceivePayloadType(function_base_, channel_name_);
}

void Receive::ReplaceChannel(std::string_view new_channel_name) {
  channel_name_ = new_channel_name;
}

void Send::ReplaceChannel(std::string_view new_channel_name) {
  channel_name_ = new_channel_name;
}

absl::StatusOr<Node*> Receive::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Receive>(
      loc(), new_operands[0],
      new_operands.size() > 1 ? std::optional<Node*>(new_operands[1])
                              : std::nullopt,
      channel_name(), is_blocking(), name_);
}

absl::StatusOr<Node*> Send::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Send>(
      loc(), new_operands[0], new_operands[1],
      new_operands.size() > 2 ? std::optional<Node*>(new_operands[2])
                              : std::nullopt,
      channel_name(), name_);
}

absl::StatusOr<Node*> Next::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  // TODO(meheff): Choose an appropriate name for the cloned node.
  return new_function->MakeNodeWithName<Next>(
      loc(), new_operands[0], new_operands[1],
      new_operands.size() > 2 ? std::optional<Node*>(new_operands[2])
                              : std::nullopt,
      name_);
}

bool Select::AllCases(std::function<bool(Node*)> p) const {
  for (Node* case_ : cases()) {
    if (!p(case_)) {
      return false;
    }
  }
  if (default_value().has_value()) {
    if (!p(default_value().value())) {
      return false;
    }
  }
  return true;
}

}  // namespace xls
)";

  std::vector<std::string> definitely_equal_to_seq;
  std::vector<std::string> constructors_seq;
  std::vector<std::string> standard_clone_seq;
  for (const auto& [name, op_class] : GetOpClassKindsSingleton()) {
    constructors_seq.push_back(RenderConstructor(op_class));
    if (op_class.HasDataMembers()) {
      definitely_equal_to_seq.push_back(RenderDefinitelyEqualTo(op_class));
    }
    if (!op_class.custom_clone_method()) {
      standard_clone_seq.push_back(RenderStandardCloneMethod(op_class));
    }
  }

  std::string definitely_equal_to = absl::StrJoin(definitely_equal_to_seq, "\n");
  std::string constructors = absl::StrJoin(constructors_seq, "\n");
  std::string standard_clone = absl::StrJoin(standard_clone_seq, "\n");

  return absl::StrReplaceAll(kTemplate, {
    {"{OP_CLASS_CONSTRUCTORS}", constructors},
    {"{OP_CLASS_STANDARD_CLONE_METHODS}", standard_clone},
    {"{OP_CLASS_DEFINITELY_EQUAL_TO_METHODS}", definitely_equal_to},
  });
}

}  // namespace xls
