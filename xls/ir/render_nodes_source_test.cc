#include "gtest/gtest.h"

#include "xls/ir/render_nodes_source.h"
#include "xls/ir/op_specification.h"
#include "xls/common/remove_empty_lines.h"

namespace xls {
namespace {

TEST(RenderOpClassTest, BinOpConstructor) {
  const OpClass& o = GetOpClassKindsSingleton().at("BIN_OP");
  EXPECT_EQ(RemoveEmptyLines(RenderConstructor(o)), R"(BinOp::BinOp(const SourceInfo& loc, Node* lhs, Node* rhs, Op op, std::string_view name, FunctionBase* function)
    : Node(op, lhs->GetType(), loc, name, function)
{
  CHECK(IsOpClass<BinOp>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `BinOp`.";
  AddOperand(lhs);
  AddOperand(rhs);
})");
}

TEST(RenderOpClassTest, ArithOpConstructor) {
  const OpClass& o = GetOpClassKindsSingleton().at("ARITH_OP");
  EXPECT_EQ(RemoveEmptyLines(RenderConstructor(o)), R"(ArithOp::ArithOp(const SourceInfo& loc, Node* lhs, Node* rhs, int64_t width, Op op, std::string_view name, FunctionBase* function)
    : Node(op, function->package()->GetBitsType(width), loc, name, function),
      width_(width)
{
  CHECK(IsOpClass<ArithOp>(op_))
      << "Op `" << op_
      << "` is not a valid op for Node class `ArithOp`.";
  AddOperand(lhs);
  AddOperand(rhs);
})");
}

TEST(RenderOpClassTest, ArithOpClone) {
  const OpClass& o = GetOpClassKindsSingleton().at("ARITH_OP");
  const std::string_view kWant = R"(absl::StatusOr<Node*>
ArithOp::CloneInNewFunction(
    absl::Span<Node* const> new_operands,
    FunctionBase* new_function) const {
  XLS_RET_CHECK_EQ(operand_count(), new_operands.size());
  return new_function->MakeNodeWithName<ArithOp>(loc(), new_operands[0], new_operands[1], width(), op(), name_);
})";
  EXPECT_EQ(RemoveEmptyLines(RenderStandardCloneMethod(o)), kWant);
}

}  // namespace
}  // namespace xls
