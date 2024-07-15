#include "gtest/gtest.h"

#include "xls/ir/render_nodes_header.h"
#include "xls/ir/op_specification.h"
#include "xls/common/remove_empty_lines.h"

namespace xls {
namespace {

TEST(RenderNodeSubclassTest, BinOp) {
  const OpClass& op_class = GetOpClassKindsSingleton().at("BIN_OP");
  std::string got = RemoveEmptyLines(RenderNodeSubclass(op_class));
  std::string want = R"(class BinOp final : public Node {
 public:
  static constexpr int64_t kLhsOperand = 0;
  static constexpr int64_t kRhsOperand = 1;
  BinOp(const SourceInfo& loc, Node* lhs, Node* rhs, Op op, std::string_view name, FunctionBase* function);
  absl::StatusOr<Node *>
  CloneInNewFunction(absl::Span<Node *const> new_operands,
                     FunctionBase *new_function) const final;
 private:
};)";
  EXPECT_EQ(got, want);
}

TEST(RenderNodeSubclassTest, ReceiveOp) {
  const std::string_view kWant = R"(class Receive final : public Node {
 public:
  static constexpr int64_t kTokenOperand = 0;

  Receive(const SourceInfo& loc, Node* token, std::optional<Node*> predicate, std::string_view channel_name, bool is_blocking, std::string_view name, FunctionBase* function);

  absl::StatusOr<Node *>
  CloneInNewFunction(absl::Span<Node *const> new_operands,
                     FunctionBase *new_function) const final;

  const std::string& channel_name() const {
    return channel_name_;
  }
  bool is_blocking() const {
    return is_blocking_;
  }
  Node* token() const {
    return operand(0);
  }
  std::optional<Node*> predicate() const {
    return predicate_operand_number().ok() ? std::optional<Node*>(operand(*predicate_operand_number())) : std::nullopt;
  }
  Type* GetPayloadType() const;
  void ReplaceChannel(std::string_view new_channel_name);

  absl::StatusOr<int64_t> predicate_operand_number() const {
    if (!has_predicate_) { return absl::InternalError("predicate is not present"); }
    int64_t ret = 1;
    return ret;
  }

  bool IsDefinitelyEqualTo(const Node *other) const final;

 private:
  std::string channel_name_;
  bool is_blocking_;
  bool has_predicate_;
};
)";
  const OpClass& op_class = GetOpClassKindsSingleton().at("RECEIVE");
  EXPECT_EQ(RenderNodeSubclass(op_class), kWant);
}

TEST(RenderNodeSubclassTest, ConcatOp) {
  const std::string_view kWant = R"(class Concat final : public Node {
 public:
  Concat(const SourceInfo& loc, absl::Span<Node* const> args, std::string_view name, FunctionBase* function);
  absl::StatusOr<Node *>
  CloneInNewFunction(absl::Span<Node *const> new_operands,
                     FunctionBase *new_function) const final;
  SliceData GetOperandSliceData(int64_t operandno) const;
 private:
};)";
  const OpClass& op_class = GetOpClassKindsSingleton().at("CONCAT");
  std::string got = RemoveEmptyLines(RenderNodeSubclass(op_class));
  EXPECT_EQ(got, kWant);
}

TEST(RenderNodeSubclassTest, RegisterWrite) {
  const std::string kWant = R"(class RegisterWrite final : public Node {
 public:
  static constexpr int64_t kDataOperand = 0;
  RegisterWrite(const SourceInfo& loc, Node* data, std::optional<Node*> load_enable, std::optional<Node*> reset, Register* reg, std::string_view name, FunctionBase* function);
  absl::StatusOr<Node *>
  CloneInNewFunction(absl::Span<Node *const> new_operands,
                     FunctionBase *new_function) const final;
  Node* data() const {
    return operand(0);
  }
  std::optional<Node*> load_enable() const {
    return load_enable_operand_number().ok() ? std::optional<Node*>(operand(*load_enable_operand_number())) : std::nullopt;
  }
  std::optional<Node*> reset() const {
    return reset_operand_number().ok() ? std::optional<Node*>(operand(*reset_operand_number())) : std::nullopt;
  }
  Register* GetRegister() const {
    return reg_;
  }
  absl::Status ReplaceExistingLoadEnable(Node* new_operand) {
    return has_load_enable_ ? ReplaceOperandNumber(*load_enable_operand_number(), new_operand) : absl::InternalError("Unable to replace load enable on RegisterWrite -- register does not have an existing load enable operand.");
  }
  absl::Status AddOrReplaceReset(Node* new_reset_node, Reset new_reset_info) {
    reg_->UpdateReset(new_reset_info);
    if (!has_reset_) {
      AddOperand(new_reset_node);
      has_reset_ = true;
      return absl::OkStatus();
    }
    return ReplaceOperandNumber(*reset_operand_number(), new_reset_node);
  }
  absl::StatusOr<int64_t> load_enable_operand_number() const {
    if (!has_load_enable_) { return absl::InternalError("load_enable is not present"); }
    int64_t ret = 1;
    return ret;
  }
  absl::StatusOr<int64_t> reset_operand_number() const {
    if (!has_reset_) { return absl::InternalError("reset is not present"); }
    int64_t ret = 2;
    if (!has_load_enable_) { ret--; }
    return ret;
  }
  bool IsDefinitelyEqualTo(const Node *other) const final;
 private:
  Register* reg_;
  bool has_load_enable_;
  bool has_reset_;
};)";
  const OpClass& op_class = GetOpClassKindsSingleton().at("REGISTER_WRITE");
  std::string got = RemoveEmptyLines(RenderNodeSubclass(op_class));
  EXPECT_EQ(got, kWant);
}

}  // namespace
}  // namespace xls
