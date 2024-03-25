// Copyright 2022 The XLS Authors
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
#include "xls/jit/ir_builder_visitor.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/casts.h"
#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/BasicBlock.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Instructions.h"
#include "llvm/include/llvm/IR/Intrinsics.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "llvm/include/llvm/IR/Type.h"
#include "llvm/include/llvm/IR/Value.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/block.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/events.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"

namespace xls {

bool ShouldMaterializeAtUse(Node* node) {
  // Only materialize Bits typed literals at their use. Array and tuple typed
  // literals are typically manipulated via pointer in the JITted code so these
  // values would have to be put in an alloca'd buffer anyway so there is no
  // advantage to doing this at their uses vs in HandleLiteral.
  return node->Is<Literal>() && node->GetType()->IsBits();
}

namespace {

// Abstraction representing a value carried across iterations of the loop.
struct LoopCarriedValue {
  std::string name;
  llvm::Value* initial_value;
};

// Abstraction representing a simple loop in LLVM.
//
// Loop structure is:
//
//   basic_block: // BB from constructor arg `builder`
//     ...
//     br preheader:
//
//   preheader:
//     index = phi(0, next_index)
//     cond = eq(index, loop_count)
//     br(cond, exit, body)
//
//   body:
//     ...
//     next_index = index + 1
//     br preheader
//
//   exit:
//     ...
class LlvmIrLoop {
 public:
  // Creates a simple loop in LLVM IR which iterates `loop_count` times.
  //
  // `loop_carried_values` are values which are carried across loop iterations.
  // LoopCarriedValue defines a variable name and initial value. When finalized,
  // the next-iteration value is specified. Loop carried values are defined in
  // the preheader:
  //
  //   basic_block:
  //     init_value = ...
  //     ...
  //
  //   preheader:
  //     my_variable = phi(init_value, next_my_variable)
  //     ...
  //     br(cond, exit, body)
  //
  //   body:
  //     ...
  //     next_my_variable = ...
  //     br preheader
  //
  //   exit:
  //     ...

  LlvmIrLoop(int64_t loop_count, llvm::IRBuilder<>& builder, int64_t stride = 1,
             llvm::BasicBlock* insert_before = nullptr,
             absl::Span<const LoopCarriedValue> loop_carried_values = {});
  ~LlvmIrLoop() { CHECK(finalized_); }

  // Index value ranging from 0 ... (N-1) * stride. Type is i64.
  llvm::Value* index() { return index_; }

  // Returns llvm::Value* for the loop-carried value with the given name. This
  // value can be used in the body of the loop and in the exit block of the
  // loop.
  llvm::Value* GetLoopCarriedValue(std::string_view name) const {
    return loop_carried_values_phis_.at(name);
  }

  // Builders for the loop body block and the exit block.
  llvm::IRBuilder<>& body_builder() { return *body_builder_; }
  llvm::IRBuilder<>& exit_builder() { return *exit_builder_; }

  // Consume body/exit builders. The `body_builder`/`exit_builder`
  // methods should not be called after this.
  std::unique_ptr<llvm::IRBuilder<>> ConsumeBodyBuilder() {
    return std::move(body_builder_);
  }
  std::unique_ptr<llvm::IRBuilder<>> ConsumeExitBuilder() {
    return std::move(exit_builder_);
  }

  // Finish constructing the loop by adding the backedge.
  // `final_body_block_builder` if specified is the builder for the final basic
  // block in the loop body. If not specified, then `body_builder` will be
  // used. This builder will be used to construct the back edge of the loop.
  //
  // `next_loop_carried_values` should contain an entry for each
  // LoopCarriedValue passed into the constructor. The entry in the map is the
  // loop-carried value of the variable in the next iteration of the loop.
  void Finalize(
      std::optional<llvm::IRBuilder<>*> final_body_block_builder = std::nullopt,
      const absl::flat_hash_map<std::string, llvm::Value*>&
          next_loop_carried_values = {}) {
    CHECK(!finalized_);

    llvm::IRBuilder<>* b = final_body_block_builder.has_value()
                               ? final_body_block_builder.value()
                               : body_builder_.get();
    for (const LoopCarriedValue& loop_carried_value : loop_carried_values_) {
      CHECK(next_loop_carried_values.contains(loop_carried_value.name))
          << absl::StrFormat(
                 "No next value specified for loop-carried value `%s`",
                 loop_carried_value.name);
      loop_carried_values_phis_.at(loop_carried_value.name)
          ->addIncoming(next_loop_carried_values.at(loop_carried_value.name),
                        b->GetInsertBlock());
    }

    llvm::Value* next_index = b->CreateAdd(index_, b->getInt64(stride_));
    next_index->setName("next_index");
    b->CreateBr(preheader_block_);
    index_->addIncoming(next_index, b->GetInsertBlock());
    finalized_ = true;
  }

 private:
  // The stride of the loop.
  int64_t stride_;

  std::vector<LoopCarriedValue> loop_carried_values_;

  // Preheader basic block. The preheader looks like:
  //
  //   preheader:
  //     index = phi(0, next_index)
  //     cond = eq(index, loop_count)
  //     br(cond, exit, body)
  llvm::BasicBlock* preheader_block_;

  // IRBuilder for the loop body basic block.
  std::unique_ptr<llvm::IRBuilder<>> body_builder_;

  // IRBuilder for the exit basic block of the loop.
  std::unique_ptr<llvm::IRBuilder<>> exit_builder_;

  // The index value (induction variable) for the loop. It may be accessed
  // anywhere in the preheader or body.
  llvm::PHINode* index_;

  // Phi nodes for loop-carried values. Indexed by name.
  absl::flat_hash_map<std::string, llvm::PHINode*> loop_carried_values_phis_;

  bool finalized_ = false;
};

LlvmIrLoop::LlvmIrLoop(int64_t loop_count, llvm::IRBuilder<>& builder,
                       int64_t stride, llvm::BasicBlock* insert_before,
                       absl::Span<const LoopCarriedValue> loop_carried_values)
    : stride_(stride),
      loop_carried_values_(loop_carried_values.begin(),
                           loop_carried_values.end()) {
  llvm::Function* function = builder.GetInsertBlock()->getParent();
  llvm::LLVMContext& context = builder.getContext();
  llvm::BasicBlock* entry_block = builder.GetInsertBlock();

  preheader_block_ =
      llvm::BasicBlock::Create(context, "preheader", function, insert_before);
  auto preheader_builder =
      std::make_unique<llvm::IRBuilder<>>(preheader_block_);
  body_builder_ = std::make_unique<llvm::IRBuilder<>>(
      llvm::BasicBlock::Create(context, "loop", function, insert_before));
  exit_builder_ = std::make_unique<llvm::IRBuilder<>>(
      llvm::BasicBlock::Create(context, "exit", function, insert_before));

  builder.CreateBr(preheader_block_);

  constexpr std::string_view kIndexName = "index";
  llvm::Value* init_index = preheader_builder->getInt64(0);
  index_ = preheader_builder->CreatePHI(llvm::Type::getInt64Ty(context), 2);
  index_->setName("index");

  for (const LoopCarriedValue& loop_carried_value : loop_carried_values_) {
    CHECK_NE(loop_carried_value.name, kIndexName) << absl::StrFormat(
        "Name `%s` is reserved for the loop index.", kIndexName);
    llvm::PHINode* phi = preheader_builder->CreatePHI(
        loop_carried_value.initial_value->getType(), 2);
    phi->setName(loop_carried_value.name);
    phi->addIncoming(loop_carried_value.initial_value, entry_block);
    loop_carried_values_phis_[loop_carried_value.name] = phi;
  }

  llvm::Value* index_limit = preheader_builder->getInt64(loop_count * stride);
  llvm::Value* loop_done = preheader_builder->CreateICmpEQ(index_, index_limit);
  loop_done->setName("loop_done");
  preheader_builder->CreateCondBr(loop_done, exit_builder_->GetInsertBlock(),
                                  body_builder_->GetInsertBlock());
  index_->addIncoming(init_index, entry_block);
}

// Abstraction representing a if-then construct in LLVM.
struct LlvmIfThen {
  // Builder for the "then" block.
  std::unique_ptr<llvm::IRBuilder<>> then_builder;

  // Builder for the join block.
  std::unique_ptr<llvm::IRBuilder<>> join_builder;

  // Finalizes construction by creating a branch from the `then` block to the
  // `join` block. `builder`, if specified, is used to construct the branch. If
  // not specified, then the builder for the `then` block is used.
  std::unique_ptr<llvm::IRBuilder<>> Finalize(
      std::optional<llvm::IRBuilder<>*> builder = std::nullopt) {
    if (builder.has_value()) {
      (*builder)->CreateBr(join_builder->GetInsertBlock());
    } else {
      then_builder->CreateBr(join_builder->GetInsertBlock());
    }
    return std::move(join_builder);
  }
};

LlvmIfThen CreateIfThen(llvm::Value* condition, llvm::IRBuilder<>& builder,
                        std::string_view prefix) {
  llvm::Function* function = builder.GetInsertBlock()->getParent();
  llvm::LLVMContext& context = builder.getContext();
  llvm::BasicBlock* then_block = llvm::BasicBlock::Create(
      context, absl::StrFormat("%s_then", prefix), function);
  llvm::BasicBlock* join_block = llvm::BasicBlock::Create(
      context, absl::StrFormat("%s_join", prefix), function);

  builder.CreateCondBr(condition, then_block, join_block);

  LlvmIfThen if_then;
  if_then.then_builder = std::make_unique<llvm::IRBuilder<>>(then_block);

  if_then.join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
  return if_then;
}

// Returns a sequence of numbered strings. Example: NumberedStrings("foo", 3)
// returns: {"foo_0", "foo_1", "foo_2"}
std::vector<std::string> NumberedStrings(std::string_view s, int64_t count) {
  std::vector<std::string> result(count);
  for (int64_t i = 0; i < count; ++i) {
    result[i] = absl::StrFormat("%s_%d", s, i);
  }
  return result;
}

// Returns the concatentation of two vectors.
std::vector<std::string> ConcatVectors(absl::Span<const std::string> a,
                                       absl::Span<const std::string> b) {
  std::vector<std::string> result(a.begin(), a.end());
  result.insert(result.end(), b.begin(), b.end());
  return result;
}

// Emit an LLVM shift operation corresponding to the semantics of the given XLS
// op.
llvm::Value* EmitShiftOp(Node* shift, llvm::Value* lhs, llvm::Value* rhs,
                         llvm::IRBuilder<>* builder,
                         LlvmTypeConverter* type_converter) {
  Op op = shift->op();
  // Shift operands are allowed to be different sizes in the [XLS] IR, so
  // we need to cast them to be the same size here (for LLVM).
  int common_width = std::max(lhs->getType()->getIntegerBitWidth(),
                              rhs->getType()->getIntegerBitWidth());
  llvm::Type* dest_type =
      llvm::Type::getIntNTy(builder->getContext(), common_width);
  llvm::Value* wide_lhs =
      op == Op::kShra
          ? type_converter->AsSignedValue(lhs, shift->operand(0)->GetType(),
                                          *builder, dest_type)
          : builder->CreateZExt(lhs, dest_type);
  llvm::Value* wide_rhs = builder->CreateZExt(rhs, dest_type);
  // In LLVM, an overshifted shift creates poison.
  llvm::Value* is_overshift = builder->CreateICmpUGE(
      wide_rhs, llvm::ConstantInt::get(dest_type, common_width));

  llvm::Value* inst;
  llvm::Value* zero = llvm::ConstantInt::get(dest_type, 0);
  llvm::Value* overshift_value = zero;
  // In the event of potential overshift (shift amount >= width of operand)
  // replace the shift amount with zero to avoid creating a poison value. The
  // motiviation is that LLVM has buggy optimizations which sometimes improperly
  // propagate poison values, particularly through selects. In the case where
  // the shift amount is replaced with zero, the shift valued is not even used
  // (selected in the Select instruction) so correctness is not affected.
  llvm::Value* safe_rhs = builder->CreateSelect(is_overshift, zero, wide_rhs);

  if (op == Op::kShll) {
    inst = builder->CreateShl(wide_lhs, safe_rhs);
  } else if (op == Op::kShra) {
    llvm::Value* high_bit = builder->CreateLShr(
        wide_lhs,
        llvm::ConstantInt::get(dest_type,
                               wide_lhs->getType()->getIntegerBitWidth() - 1));
    llvm::Value* high_bit_set =
        builder->CreateICmpEQ(high_bit, llvm::ConstantInt::get(dest_type, 1));
    overshift_value = builder->CreateSelect(
        high_bit_set, llvm::ConstantInt::getSigned(dest_type, -1), zero);
    inst = builder->CreateAShr(wide_lhs, safe_rhs);
  } else {
    CHECK_EQ(op, Op::kShrl);
    inst = builder->CreateLShr(wide_lhs, safe_rhs);
  }
  llvm::Value* result =
      builder->CreateSelect(is_overshift, overshift_value, inst);
  // The expected return type is the same as the original lhs. The shifted value
  // may be wider than this, so truncate in that case. This occurs if the
  // original rhs is wider than the original lhs.
  return result->getType() == lhs->getType()
             ? result
             : builder->CreateTrunc(result, lhs->getType());
}

llvm::Value* EmitDiv(llvm::Value* num, llvm::Value* denom, int64_t bit_count,
                     bool is_signed, LlvmTypeConverter* type_converter,
                     llvm::IRBuilder<>* builder) {
  // XLS div semantics differ from LLVM's (and most software's) here: in XLS,
  // division by zero returns the greatest value of that type, so 255 for an
  // unsigned byte, and either -128 or 127 for a signed one. Also, division of
  // the minimum signed value by -1 returns the minimum signed value, avoiding
  // overflow. Thus, a little more work is necessary to emit LLVM IR matching
  // the XLS div op than just IRBuilder::Create[SU]Div().
  llvm::Value* zero = llvm::ConstantInt::get(denom->getType(), 0);
  llvm::Value* denom_eq_zero = builder->CreateICmpEQ(denom, zero);

  // To avoid div by zero, make a never-zero denom. This works because in the
  // case of a zero denom this value is not used.
  llvm::Value* safe_denom = builder->CreateSelect(
      denom_eq_zero, llvm::ConstantInt::get(denom->getType(), 1), denom);
  if (!is_signed) {
    return builder->CreateSelect(
        denom_eq_zero,
        type_converter
            ->ToLlvmConstant(denom->getType(), Value(Bits::AllOnes(bit_count)))
            .value(),
        builder->CreateUDiv(num, safe_denom));
  }

  // Division by 0 gives the value furthest from zero with matching sign.
  llvm::Value* lhs_ge_zero = builder->CreateICmpSGE(num, zero);
  llvm::Value* max_value =
      type_converter
          ->ToLlvmConstant(denom->getType(), Value(Bits::MaxSigned(bit_count)))
          .value();
  llvm::Value* min_value =
      type_converter
          ->ToLlvmConstant(denom->getType(), Value(Bits::MinSigned(bit_count)))
          .value();
  llvm::Value* rhs_is_zero_result =
      builder->CreateSelect(lhs_ge_zero, max_value, min_value);

  // Division by -1 gets converted to negation; this prevents potential overflow
  // when dividing min_value by -1.
  llvm::Value* denom_eq_neg_one = builder->CreateICmpEQ(
      denom,
      type_converter
          ->ToLlvmConstant(denom->getType(), Value(Bits::AllOnes(bit_count)))
          .value());
  llvm::Value* denom_is_neg_one_result = builder->CreateSub(zero, num);

  // Since overflow is UB, make sure the denominator can't create overflow; this
  // works because it won't be used in the case of a -1 denom.
  safe_denom = builder->CreateSelect(
      denom_eq_neg_one, llvm::ConstantInt::get(denom->getType(), 1),
      safe_denom);
  llvm::Value* normal_result = builder->CreateSDiv(num, safe_denom);

  return builder->CreateSelect(
      denom_eq_zero, rhs_is_zero_result,
      builder->CreateSelect(denom_eq_neg_one, denom_is_neg_one_result,
                            normal_result));
}

llvm::Value* EmitMod(llvm::Value* lhs, llvm::Value* rhs, bool is_signed,
                     llvm::IRBuilder<>* builder) {
  // XLS mod semantics differ from LLVMs with regard to mod by zero. In XLS,
  // modulo by zero returns zero rather than undefined behavior.
  llvm::Value* zero = llvm::ConstantInt::get(rhs->getType(), 0);
  llvm::Value* rhs_eq_zero = builder->CreateICmpEQ(rhs, zero);
  // Replace a zero rhs with one to avoid SIGFPE even though the result is not
  // used.
  rhs = builder->CreateSelect(rhs_eq_zero,
                              llvm::ConstantInt::get(rhs->getType(), 1), rhs);
  return builder->CreateSelect(rhs_eq_zero, zero,
                               is_signed ? builder->CreateSRem(lhs, rhs)
                                         : builder->CreateURem(lhs, rhs));
}

// Local struct to hold the individual elements of a (possibly) compound
// comparison.
struct CompareTerm {
  llvm::Value* lhs;
  llvm::Value* rhs;
};

// Expand the lhs and rhs of a comparison into a vector of the individual leaf
// terms to compare.
absl::StatusOr<std::vector<CompareTerm>> ExpandTerms(
    Node* lhs, llvm::Value* llvm_lhs, Node* rhs, llvm::Value* llvm_rhs,
    Node* src, llvm::IRBuilder<>* builder) {
  XLS_RET_CHECK(lhs->GetType() == rhs->GetType()) << absl::StreamFormat(
      "The lhs and rhs of %s have different types: lhs %s rhs %s",
      src->ToString(), lhs->GetType()->ToString(), rhs->GetType()->ToString());

  struct ToExpand {
    Type* ty;
    llvm::Value* lhs;
    llvm::Value* rhs;
  };

  std::vector<ToExpand> unexpanded = {
      ToExpand{lhs->GetType(), llvm_lhs, llvm_rhs}};

  std::vector<CompareTerm> terms;

  while (!unexpanded.empty()) {
    ToExpand next = unexpanded.back();
    unexpanded.pop_back();

    switch (next.ty->kind()) {
      case TypeKind::kToken:
        // Tokens represent different points in time and are incomparable.
        return absl::InvalidArgumentError(absl::StrFormat(
            "Tokens are incomparable so this expression is illegal: %s",
            src->ToString()));
      case TypeKind::kBits:
        terms.push_back(CompareTerm{next.lhs, next.rhs});
        break;
      case TypeKind::kArray: {
        ArrayType* array_type = next.ty->AsArrayOrDie();
        Type* element_type = array_type->element_type();
        // Cast once so we do not have to cast when calling CreateExtractValue
        uint32_t array_size = static_cast<uint32_t>(array_type->size());
        for (uint32_t i = 0; i < array_size; i++) {
          llvm::Value* lhs_value = builder->CreateExtractValue(next.lhs, {i});
          llvm::Value* rhs_value = builder->CreateExtractValue(next.rhs, {i});
          unexpanded.push_back(ToExpand{element_type, lhs_value, rhs_value});
        }
        break;
      }
      case TypeKind::kTuple: {
        TupleType* tuple_type = next.ty->AsTupleOrDie();
        // Cast once so we do not have to cast when calling CreateExtractValue
        uint32_t tuple_size = static_cast<uint32_t>(tuple_type->size());
        for (uint32_t i = 0; i < tuple_size; i++) {
          Type* element_type = tuple_type->element_type(i);
          llvm::Value* lhs_value = builder->CreateExtractValue(next.lhs, {i});
          llvm::Value* rhs_value = builder->CreateExtractValue(next.rhs, {i});
          unexpanded.push_back(ToExpand{element_type, lhs_value, rhs_value});
        }
        break;
      }
    }
  }
  return terms;
}

// Returns an llvm::Value (i1 type) indicating whether `index` is an in-bounds
// index into an array of type `array_type`.
llvm::Value* IsIndexInBounds(llvm::Value* index, ArrayType* array_type,
                             llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = builder.getContext();
  int64_t array_size = array_type->size();
  int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
  int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
  llvm::Type* comparison_type =
      llvm::Type::getIntNTy(context, comparison_bitwidth);
  llvm::Value* is_inbounds = builder.CreateICmpULT(
      builder.CreateZExt(index, comparison_type),
      llvm::ConstantInt::get(comparison_type, array_size));
  is_inbounds->setName("is_inbounds");
  return is_inbounds;
}

// Returns the `index` clamped to the maximum element index of array type
// `array_type`. Return value is of type i64.
llvm::Value* ClampIndexInBounds(llvm::Value* index, ArrayType* array_type,
                                llvm::IRBuilder<>& builder) {
  llvm::LLVMContext& context = builder.getContext();
  llvm::Type* i64 = llvm::Type::getInt64Ty(context);
  int64_t array_size = array_type->size();
  llvm::Value* is_inbounds = IsIndexInBounds(index, array_type, builder);
  llvm::Value* inbounds_index = builder.CreateSelect(
      is_inbounds, builder.CreateIntCast(index, i64, /*isSigned=*/false),
      llvm::ConstantInt::get(i64, array_size - 1));
  inbounds_index->setName("clamped_index");
  return inbounds_index;
}

// This is a shim to let JIT code add a new trace fragment to an existing trace
// buffer.
void PerformStringStep(char* step_string, std::string* buffer) {
  buffer->append(step_string);
}

// Build the LLVM IR that handles string fragment format steps.
absl::Status InvokeStringStepCallback(llvm::IRBuilder<>* builder,
                                      const std::string& step_string,
                                      llvm::Value* buffer_ptr) {
  llvm::Constant* step_constant = builder->CreateGlobalStringPtr(step_string);

  std::vector<llvm::Type*> params = {step_constant->getType(),
                                     buffer_ptr->getType()};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {step_constant, buffer_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&PerformStringStep));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

void PerformFormatStep(JitRuntime* runtime, const xls::Type* type,
                       const uint8_t* value, uint64_t format_u64,
                       std::string* buffer) {
  FormatPreference format = static_cast<FormatPreference>(format_u64);
  Value ir_value = runtime->UnpackBuffer(value, type);
  absl::StrAppend(buffer, ir_value.ToHumanString(format));
}

// Build the LLVM IR that handles formatting a runtime value according to a
// format preference.
absl::Status InvokeFormatStepCallback(llvm::IRBuilder<>* builder,
                                      FormatPreference format,
                                      xls::Type* operand_type,
                                      llvm::Value* operand,
                                      llvm::Value* buffer_ptr,
                                      llvm::Value* jit_runtime_ptr) {
  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());
  auto* i64_type = llvm::Type::getInt64Ty(builder->getContext());

  llvm::ConstantInt* llvm_format =
      llvm::ConstantInt::get(i64_type, static_cast<uint64_t>(format));

  // Note: we assume the package lifetime is >= that of the JIT code by
  // capturing this type pointer as a value burned into the JIT code, which
  // should always be true.
  llvm::ConstantInt* llvm_operand_type =
      llvm::ConstantInt::get(i64_type, absl::bit_cast<uint64_t>(operand_type));

  std::vector<llvm::Type*> params = {
      jit_runtime_ptr->getType(), llvm_operand_type->getType(),
      operand->getType(), llvm_format->getType(), buffer_ptr->getType()};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {jit_runtime_ptr, llvm_operand_type, operand,
                                    llvm_format, buffer_ptr};

  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      i64_type, absl::bit_cast<uint64_t>(&PerformFormatStep));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

// This a shim to let JIT code record a completed trace as an interpreter event.
void RecordTrace(std::string* buffer, int64_t verbosity,
                 xls::InterpreterEvents* events) {
  events->trace_msgs.push_back(
      TraceMessage{.message = *buffer, .verbosity = verbosity});
  delete buffer;
}

// Build the LLVM IR to invoke the callback that records traces.
absl::Status InvokeRecordTraceCallback(llvm::IRBuilder<>* builder,
                                       int64_t verbosity,
                                       llvm::Value* buffer_ptr,
                                       llvm::Value* interpreter_events_ptr) {
  llvm::Type* int64_type = llvm::Type::getInt64Ty(builder->getContext());
  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  std::vector<llvm::Type*> params = {buffer_ptr->getType(), int64_type,
                                     ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  llvm::Value* verbosity_value = builder->getInt64(verbosity);
  std::vector<llvm::Value*> args = {buffer_ptr, verbosity_value,
                                    interpreter_events_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&RecordTrace));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

// This is a shim to let JIT code create a buffer for accumulating trace
// fragments.
std::string* CreateTraceBuffer() { return new std::string(); }

// Build the LLVM IR to invoke the callback that creates a trace buffer.
absl::StatusOr<llvm::Value*> InvokeCreateBufferCallback(
    llvm::IRBuilder<>* builder) {
  std::vector<llvm::Type*> params;

  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(ptr_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args;

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&CreateTraceBuffer));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  return builder->CreateCall(fn_type, fn_ptr, args);
}

// This a shim to let JIT code record an assertion failure as an interpreter
// event.
void RecordAssertion(char* msg, xls::InterpreterEvents* events) {
  events->assert_msgs.push_back(msg);
}

// Build the LLVM IR to invoke the callback that records assertions.
absl::Status InvokeAssertCallback(llvm::IRBuilder<>* builder,
                                  const std::string& message,
                                  llvm::Value* interpreter_events_ptr) {
  llvm::Constant* msg_constant = builder->CreateGlobalStringPtr(message);

  llvm::Type* msg_type = msg_constant->getType();

  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  std::vector<llvm::Type*> params = {msg_type, ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {msg_constant, interpreter_events_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(builder->getContext()),
                             absl::bit_cast<uint64_t>(&RecordAssertion));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

// This is a shim to let JIT code record the activation of a `next_value` node.
void RecordActiveNextValue(Next* next, InstanceContext* instance_context) {
  instance_context->active_next_values[next->param()->As<Param>()].insert(next);
}

// Build the LLVM IR to invoke the callback that records assertions.
absl::Status InvokeNextValueCallback(llvm::IRBuilder<>* builder, Next* next,
                                     llvm::Value* instance_context) {
  llvm::Type* addr_type =
      llvm::Type::getIntNTy(builder->getContext(), sizeof(uintptr_t) << 3);
  llvm::Constant* next_addr =
      llvm::ConstantInt::get(addr_type, reinterpret_cast<uintptr_t>(next));

  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);
  std::vector<llvm::Type*> params = {ptr_type, ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {
      builder->CreateIntToPtr(next_addr, ptr_type),
      builder->CreateIntToPtr(instance_context, ptr_type)};

  llvm::Constant* fn_addr = llvm::ConstantInt::get(
      addr_type, reinterpret_cast<uintptr_t>(&RecordActiveNextValue));
  llvm::Value* fn_ptr = builder->CreateIntToPtr(fn_addr, ptr_type);
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

absl::StatusOr<llvm::Function*> CreateLlvmFunction(
    std::string_view name, llvm::FunctionType* function_type,
    llvm::Module* module) {
  XLS_RET_CHECK_EQ(module->getFunction(name), nullptr) << absl::StreamFormat(
      "Function named `%s` already exists in LLVM module", name);
  return llvm::cast<llvm::Function>(
      module->getOrInsertFunction(std::string{name}, function_type)
          .getCallee());
}

// Abstraction gathering together the necessary context for emitting the LLVM IR
// for a given node. This data structure decouples IR generation for the
// top-level function from the IR generation of each node. This enables, for
// example, emitting a node as a separate function.
class NodeIrContext {
 public:
  // Possible node function signatures:
  //
  // bool f(void* operand_ptr_0, ..., void* operand_ptr_n,
  //        void* output_ptr_0,  ..., void* output_ptr_m)
  //
  // bool f(void* operand_ptr_0, ... , void* operand_ptr_n,
  //        void* output_ptr_0,  ... , void* output_ptr_m,
  //        void* global_inputs, void* global_outputs, void* tmp_buffer,
  //        void* events, void* instance_context, void* runtime)
  //
  // Args:
  //   node : The XLS node the LLVM IR is being generated for.
  //   operand_names : the names of the operand of `node`. If possible, these
  //     will be used to name the LLVM values of the operands.
  //   output_arg_count: The number of output pointer arguments in the generated
  //     function. Each argument points to a buffer which must be filled with
  //     the computed result.
  //   include_wrapper_args: whether to include top-level arguments (such as
  //     InterpreterEvents*, JitRuntime*, etc) in the node function. This is
  //     required when the node calls a top-level function (e.g, map, invoke,
  //     etc).
  //
  // The return value of the function indicates whether the execution of the
  // FunctionBase should be interrupted (return true) or continue (return
  // false). The return value is only used for continuation point nodes (ie,
  // blocking receives).
  static absl::StatusOr<NodeIrContext> Create(
      Node* node, absl::Span<const std::string> operand_names,
      int64_t output_arg_count, bool include_wrapper_args,
      const JitCompilationMetadata& metadata, JitBuilderContext& jit_context);

  // Completes the LLVM function by adding a return statement with the given
  // result (or pointer to result). If `exit_builder` is specified then it is
  // used to build the return statement. therwise `entry_builder()` is used.
  // If `return_value` is not specified then false is returned by the node
  // function.
  void FinalizeWithValue(llvm::Value* result,
                         std::optional<llvm::IRBuilder<>*> exit_builder,
                         std::optional<llvm::Value*> return_value,
                         std::optional<Type*> result_type = std::nullopt);
  void FinalizeWithPointerToValue(
      llvm::Value* result_buffer,
      std::optional<llvm::IRBuilder<>*> exit_builder,
      std::optional<llvm::Value*> return_value,
      std::optional<Type*> result_type = std::nullopt);

  Node* node() const { return node_; }
  LlvmTypeConverter& type_converter() const {
    return jit_context_.type_converter();
  }

  // Returns true if the underlying llvm::Function takes the top-level metadata
  // arguments (inputs, outputs, temp buffer, events, etc).
  bool has_metadata_args() const { return has_metadata_args_; }

  // The LLVM function that the generated code for the node is placed into,
  llvm::Function* llvm_function() const { return llvm_function_; }

  // Returns the IR builder to use for building code for this XLS node.
  llvm::IRBuilder<>& entry_builder() const { return *entry_builder_; }

  // Loads the given operand value from the pointer passed in as the respective
  // operand. `builder` is the builder used to construct any IR required to get
  // the operand ptr. If `builder` is not specified then the entry builder is
  // used.
  llvm::Value* LoadOperand(int64_t i, llvm::IRBuilder<>* builder = nullptr);

  // Loads the given global input argument. This may only be called if the
  // wrapper args were included.
  absl::StatusOr<llvm::Value*> LoadGlobalInput(
      Node* input, llvm::IRBuilder<>* builder = nullptr);

  // Returns the pointer to the `i-th` operand of `node_`. `builder` is the
  // builder used to construct any IR required to get the operand ptr. If
  // `builder` is not specified then the entry builder is used.
  llvm::Value* GetOperandPtr(int64_t i, llvm::IRBuilder<>* builder = nullptr);

  // Returns the output pointer arguments.
  absl::Span<llvm::Value* const> GetOutputPtrs() const { return output_ptrs_; }

  // Returns the `i-th` operand pointer argument.
  llvm::Value* GetOutputPtr(int64_t i) const { return output_ptrs_.at(i); }

  // Get one of the metadata arguments. CHECK fails the function was created
  // without metadata arguments.
  llvm::Value* GetInputPtrsArg() const {
    CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 6);
  }
  llvm::Value* GetOutputPtrsArg() const {
    CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 5);
  }
  llvm::Value* GetTempBufferArg() const {
    CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 4);
  }
  llvm::Value* GetInterpreterEventsArg() const {
    CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 3);
  }
  llvm::Value* GetInstanceContextArg() const {
    CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 2);
  }
  llvm::Value* GetJitRuntimeArg() const {
    CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 1);
  }

  // Returns the operands whose values should be passed into the node
  // function. This may contain fewer elements than node()->operands() because
  // operands are deduplicated.
  absl::Span<Node* const> GetOperandArgs() const { return operand_args_; }

 private:
  NodeIrContext(Node* node, bool has_metadata_args,
                const JitCompilationMetadata& metadata,
                JitBuilderContext& jit_context)
      : node_(node),
        has_metadata_args_(has_metadata_args),
        metadata_(metadata),
        jit_context_(jit_context) {}

  Node* node_;
  bool has_metadata_args_;
  const JitCompilationMetadata& metadata_;
  JitBuilderContext& jit_context_;

  llvm::Function* llvm_function_;
  std::unique_ptr<llvm::IRBuilder<>> entry_builder_;
  std::vector<llvm::Value*> output_ptrs_;

  // Map from operand to argument number in the LLVM function. Operands
  // are deduplicated so some operands may map to the same argument.
  absl::flat_hash_map<Node*, int64_t> operand_to_arg_;
  // Vector of nodes which should be passed in as the operand values of
  // `node_`. This is a deduplicated list of the operands of `node_`.
  std::vector<Node*> operand_args_;

  // Cache of calls to GetOperandPtr for values which are materialized at their
  // use. Indexed by node, and the LLVM basic block that the GetOperandPtr code
  // was generated in.
  absl::flat_hash_map<std::pair<Node*, llvm::BasicBlock*>, llvm::Value*>
      materialized_cache_;
};

absl::StatusOr<NodeIrContext> NodeIrContext::Create(
    Node* node, absl::Span<const std::string> operand_names,
    int64_t output_arg_count, bool include_wrapper_args,
    const JitCompilationMetadata& metadata, JitBuilderContext& jit_context) {
  XLS_RET_CHECK_GT(output_arg_count, 0);
  NodeIrContext nc(node, include_wrapper_args, metadata, jit_context);

  // Deduplicate the operands. If an operand appears more than once in the IR
  // node, map them to a single argument in the llvm Function for the mode.
  absl::flat_hash_map<Node*, int64_t> operand_to_operand_index;
  int64_t operand_count = 0;
  auto add_operand = [&](Node* operand) {
    operand_to_operand_index[operand] = operand_count++;
    if (ShouldMaterializeAtUse(operand)) {
      return;
    }
    if (nc.operand_to_arg_.contains(operand)) {
      return;
    }
    nc.operand_to_arg_[operand] = nc.operand_args_.size();
    nc.operand_args_.push_back(operand);
  };
  for (Node* operand : node->operands()) {
    add_operand(operand);
  }
  int64_t param_count = nc.operand_args_.size() + output_arg_count;
  if (include_wrapper_args) {
    param_count += 6;
  }
  std::vector<llvm::Type*> param_types(
      param_count,
      llvm::PointerType::get(jit_context.module()->getContext(), 0));
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getInt1Ty(jit_context.module()->getContext()), param_types,
      /*isVarArg=*/false);
  std::string function_name = absl::StrFormat(
      "__%s_%s_%d", node->function_base()->name(), node->GetName(), node->id());
  XLS_ASSIGN_OR_RETURN(
      nc.llvm_function_,
      CreateLlvmFunction(function_name, function_type, jit_context.module()));

  // Mark as private so function can be deleted after inlining.
  nc.llvm_function_->setLinkage(llvm::GlobalValue::PrivateLinkage);

  // Set names of LLVM function arguments to improve readability of LLVM
  // IR. Operands are deduplicated so some names passed in via `operand_names`
  // may not appear as argument names.
  XLS_RET_CHECK_EQ(operand_names.size(), node->operand_count());
  int64_t arg_no = 0;
  for (Node* operand : nc.operand_args_) {
    nc.llvm_function_->getArg(arg_no++)->setName(absl::StrCat(
        operand_names[operand_to_operand_index.at(operand)], "_ptr"));
  }
  for (int64_t i = 0; i < output_arg_count; ++i) {
    nc.llvm_function_->getArg(arg_no++)->setName(
        absl::StrFormat("output_%d_ptr", i));
  }
  if (include_wrapper_args) {
    nc.llvm_function_->getArg(arg_no++)->setName("inputs");
    nc.llvm_function_->getArg(arg_no++)->setName("outputs");
    nc.llvm_function_->getArg(arg_no++)->setName("temp_buffer");
    nc.llvm_function_->getArg(arg_no++)->setName("events");
    nc.llvm_function_->getArg(arg_no++)->setName("instance_context");
    nc.llvm_function_->getArg(arg_no++)->setName("jit_runtime");
  }

  nc.entry_builder_ =
      std::make_unique<llvm::IRBuilder<>>(llvm::BasicBlock::Create(
          jit_context.module()->getContext(), "entry", nc.llvm_function_,
          /*InsertBefore=*/nullptr));

  for (int64_t i = 0; i < output_arg_count; ++i) {
    nc.output_ptrs_.push_back(
        nc.llvm_function_->getArg(nc.operand_args_.size() + i));
  }
  return nc;
}

absl::StatusOr<llvm::Value*> NodeIrContext::LoadGlobalInput(
    Node* input, llvm::IRBuilder<>* builder) {
  XLS_RET_CHECK(has_metadata_args_);
  XLS_RET_CHECK(metadata_.IsInputNode(input));
  // XLS_RET_CHECK(this->GetInputPtrsArg())
  if (ShouldMaterializeAtUse(input)) {
    // If the operand is a bits constant, just return the constant as an
    // optimization.
    XLS_RET_CHECK(input->Is<Literal>())
        << "cannot materialize " << input->ToStringWithOperandTypes();
    return type_converter()
        .ToLlvmConstant(input->GetType(), input->As<Literal>()->value())
        .value();
  }
  llvm::IRBuilder<>& b = builder == nullptr ? entry_builder() : *builder;
  llvm::Type* input_type = type_converter().ConvertToLlvmType(input->GetType());
  llvm::Value* global_input = GetInputPtrsArg();
  XLS_ASSIGN_OR_RETURN(llvm::Value * input_ptr,
                       metadata_.GetInputBufferFrom(input, global_input, b));
  llvm::Value* load = b.CreateLoad(input_type, input_ptr);
  load->setName(input->GetName());
  return load;
}

llvm::Value* NodeIrContext::LoadOperand(int64_t i, llvm::IRBuilder<>* builder) {
  Node* operand = node()->operand(i);

  if (ShouldMaterializeAtUse(operand)) {
    // If the operand is a bits constant, just return the constant as an
    // optimization.
    CHECK(operand->Is<Literal>());
    return type_converter()
        .ToLlvmConstant(operand->GetType(), operand->As<Literal>()->value())
        .value();
  }
  llvm::IRBuilder<>& b = builder == nullptr ? entry_builder() : *builder;
  llvm::Type* operand_type =
      type_converter().ConvertToLlvmType(operand->GetType());
  llvm::Value* operand_ptr = GetOperandPtr(i, &b);
  llvm::Value* load = b.CreateLoad(operand_type, operand_ptr);
  load->setName(operand->GetName());
  return load;
}

llvm::Value* NodeIrContext::GetOperandPtr(int64_t i,
                                          llvm::IRBuilder<>* builder) {
  Node* operand = node()->operand(i);
  llvm::IRBuilder<>& b = builder == nullptr ? entry_builder() : *builder;

  std::pair<Node*, llvm::BasicBlock*> cache_key = {operand, b.GetInsertBlock()};
  if (ShouldMaterializeAtUse(operand)) {
    if (materialized_cache_.contains(cache_key)) {
      return materialized_cache_.at(cache_key);
    }
    llvm::Value* alloca =
        b.CreateAlloca(type_converter().ConvertToLlvmType(operand->GetType()));
    b.CreateStore(
        type_converter()
            .ToLlvmConstant(operand->GetType(), operand->As<Literal>()->value())
            .value(),
        alloca);
    materialized_cache_[cache_key] = alloca;
    return alloca;
  }
  return llvm_function_->getArg(operand_to_arg_.at(operand));
}

void NodeIrContext::FinalizeWithValue(
    llvm::Value* result, std::optional<llvm::IRBuilder<>*> exit_builder,
    std::optional<llvm::Value*> return_value,
    std::optional<Type*> return_type) {
  llvm::IRBuilder<>* b =
      exit_builder.has_value() ? exit_builder.value() : &entry_builder();
  result = type_converter().ClearPaddingBits(
      result, return_type.value_or(node()->GetType()), *b);
  if (GetOutputPtrs().empty()) {
    b->CreateRet(b->getFalse());
    return;
  }
  b->CreateStore(result, GetOutputPtr(0));
  return FinalizeWithPointerToValue(GetOutputPtr(0), exit_builder,
                                    return_value);
}

void NodeIrContext::FinalizeWithPointerToValue(
    llvm::Value* result_buffer, std::optional<llvm::IRBuilder<>*> exit_builder,
    std::optional<llvm::Value*> return_value,
    std::optional<Type*> result_type) {
  llvm::IRBuilder<>* b =
      exit_builder.has_value() ? exit_builder.value() : &entry_builder();
  for (int64_t i = 0; i < output_ptrs_.size(); ++i) {
    if (output_ptrs_[i] != result_buffer) {
      LlvmMemcpy(output_ptrs_[i], result_buffer,
                 type_converter().GetTypeByteSize(
                     result_type.value_or(node()->GetType())),
                 *b);
    }
  }
  b->CreateRet(return_value.has_value() ? *return_value : b->getFalse());
}

// Visitor to construct and LLVM function implementing an XLS IR node.
class IrBuilderVisitor : public DfsVisitorWithDefault {
 public:
  //  `output_arg_count` is the number of output arguments for the LLVM function
  IrBuilderVisitor(int64_t output_arg_count,
                   const JitCompilationMetadata& metadata,
                   JitBuilderContext& jit_context)
      : output_arg_count_(output_arg_count),
        metadata_(metadata),
        jit_context_(jit_context) {}

  NodeIrContext ConsumeNodeIrContext() { return std::move(*node_context_); }

  absl::Status DefaultHandler(Node* node) override;

  absl::Status HandleRegisterWrite(RegisterWrite* write) override;
  absl::Status HandleOutputPort(OutputPort* write) override;
  absl::Status HandleAdd(BinOp* binop) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* op) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
  absl::Status HandleMinDelay(MinDelay* min_delay) override;
  absl::Status HandleArray(Array* array) override;
  absl::Status HandleArrayIndex(ArrayIndex* index) override;
  absl::Status HandleArraySlice(ArraySlice* slice) override;
  absl::Status HandleArrayUpdate(ArrayUpdate* update) override;
  absl::Status HandleArrayConcat(ArrayConcat* concat) override;
  absl::Status HandleAssert(Assert* assert_op) override;
  absl::Status HandleBitSlice(BitSlice* bit_slice) override;
  absl::Status HandleBitSliceUpdate(BitSliceUpdate* update) override;
  absl::Status HandleDynamicBitSlice(
      DynamicBitSlice* dynamic_bit_slice) override;
  absl::Status HandleConcat(Concat* concat) override;
  absl::Status HandleCountedFor(CountedFor* counted_for) override;
  absl::Status HandleCover(Cover* cover) override;
  absl::Status HandleDecode(Decode* decode) override;
  absl::Status HandleDynamicCountedFor(
      DynamicCountedFor* dynamic_counted_for) override;
  absl::Status HandleEncode(Encode* encode) override;
  absl::Status HandleEq(CompareOp* eq) override;
  absl::Status HandleGate(Gate* gate) override;
  absl::Status HandleIdentity(UnOp* identity) override;
  absl::Status HandleInvoke(Invoke* invoke) override;
  absl::Status HandleLiteral(Literal* literal) override;
  absl::Status HandleMap(Map* map) override;
  absl::Status HandleNaryAnd(NaryOp* and_op) override;
  absl::Status HandleNaryNand(NaryOp* nand_op) override;
  absl::Status HandleNaryNor(NaryOp* nor_op) override;
  absl::Status HandleNaryOr(NaryOp* or_op) override;
  absl::Status HandleNaryXor(NaryOp* xor_op) override;
  absl::Status HandleNe(CompareOp* ne) override;
  absl::Status HandleNeg(UnOp* neg) override;
  absl::Status HandleNext(Next* next) override;
  absl::Status HandleNot(UnOp* not_op) override;
  absl::Status HandleOneHot(OneHot* one_hot) override;
  absl::Status HandleOneHotSel(OneHotSelect* sel) override;
  absl::Status HandleOrReduce(BitwiseReductionOp* op) override;
  absl::Status HandlePrioritySel(PrioritySelect* sel) override;
  absl::Status HandleReceive(Receive* recv) override;
  absl::Status HandleReverse(UnOp* reverse) override;
  absl::Status HandleSDiv(BinOp* binop) override;
  absl::Status HandleSGe(CompareOp* ge) override;
  absl::Status HandleSGt(CompareOp* gt) override;
  absl::Status HandleSLe(CompareOp* le) override;
  absl::Status HandleSLt(CompareOp* lt) override;
  absl::Status HandleSMod(BinOp* binop) override;
  absl::Status HandleSMul(ArithOp* mul) override;
  absl::Status HandleSMulp(PartialProductOp* mul) override;
  absl::Status HandleSel(Select* sel) override;
  absl::Status HandleSend(Send* send) override;
  absl::Status HandleShll(BinOp* binop) override;
  absl::Status HandleShra(BinOp* binop) override;
  absl::Status HandleShrl(BinOp* binop) override;
  absl::Status HandleSignExtend(ExtendOp* sign_ext) override;
  absl::Status HandleSub(BinOp* binop) override;
  absl::Status HandleTrace(Trace* trace_op) override;
  absl::Status HandleTuple(Tuple* tuple) override;
  absl::Status HandleTupleIndex(TupleIndex* index) override;
  absl::Status HandleUDiv(BinOp* binop) override;
  absl::Status HandleUGe(CompareOp* ge) override;
  absl::Status HandleUGt(CompareOp* gt) override;
  absl::Status HandleULe(CompareOp* le) override;
  absl::Status HandleULt(CompareOp* lt) override;
  absl::Status HandleUMod(BinOp* binop) override;
  absl::Status HandleUMul(ArithOp* mul) override;
  absl::Status HandleUMulp(PartialProductOp* mul) override;
  absl::Status HandleXorReduce(BitwiseReductionOp* op) override;
  absl::Status HandleZeroExtend(ExtendOp* zero_ext) override;

 protected:
  llvm::LLVMContext& ctx() { return jit_context_.context(); }
  llvm::Module* module() { return jit_context_.module(); }

  // Returns the top-level builder for the function. This builder is initialized
  // to the entry block of `dispatch_function()`.
  //  llvm::IRBuilder<>* dispatch_builder() { return dispatch_builder_.get(); }
  LlvmTypeConverter* type_converter() { return &jit_context_.type_converter(); }

  // Common handler for binary, unary, and nary nodes. `build_results` produces
  // the llvm Value to use as the result.
  absl::Status HandleUnaryOp(
      Node* node, std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>&)>,
      bool is_signed = false);
  absl::Status HandleBinaryOp(
      Node* node,
      std::function<llvm::Value*(llvm::Value*, llvm::Value*,
                                 llvm::IRBuilder<>&)>,
      bool is_signed = false);
  absl::Status HandleNaryOp(
      Node* node, std::function<llvm::Value*(absl::Span<llvm::Value* const>,
                                             llvm::IRBuilder<>&)>);

  // HandleBinaryOp variant which converts the operands to result type prior to
  // calling `build_result`.
  absl::Status HandleBinaryOpWithOperandConversion(
      Node* node,
      std::function<llvm::Value*(llvm::Value*, llvm::Value*,
                                 llvm::IRBuilder<>&)>
          build_result,
      bool is_signed);

  // Gets the built function representing the given XLS function.
  absl::StatusOr<llvm::Function*> GetFunction(Function* function) {
    return jit_context_.GetLlvmFunction(function);
  }

  // Creates and returns a new NodeIrContext for the given XLS node.
  absl::StatusOr<NodeIrContext> NewNodeIrContext(
      Node* node, absl::Span<const std::string> operand_names,
      bool include_wrapper_args = false);

  // Finalizes the given NodeIrContext (adds a return statement with the given
  // result) and adds a call in the top-level LLVM function to the node
  // function.
  absl::Status FinalizeNodeIrContextWithValue(
      NodeIrContext&& node_context, llvm::Value* result,
      std::optional<llvm::IRBuilder<>*> exit_builder = std::nullopt,
      std::optional<llvm::Value*> return_value = std::nullopt,
      std::optional<Type*> result_type = std::nullopt);
  absl::Status FinalizeNodeIrContextWithPointerToValue(
      NodeIrContext&& node_context, llvm::Value* result_buffer,
      std::optional<llvm::IRBuilder<>*> exit_builder = std::nullopt,
      std::optional<llvm::Value*> return_value = std::nullopt,
      std::optional<Type*> result_type = std::nullopt);

  llvm::Value* MaybeAsSigned(llvm::Value* v, Type* xls_type,
                             llvm::IRBuilder<>& builder, bool is_signed) {
    if (is_signed) {
      return type_converter()->AsSignedValue(v, xls_type, builder);
    }
    return v;
  }

  // Calls the given function `f`. `f` must have the same signature as top-level
  // jitted function (i.e., `JitFunctionType`) corrresponding to XLS functions,
  // procs, etc.
  absl::StatusOr<llvm::Value*> CallFunction(
      llvm::Function* f, absl::Span<llvm::Value* const> inputs,
      absl::Span<llvm::Value* const> outputs, llvm::Value* temp_buffer,
      llvm::Value* events, llvm::Value* instance_context, llvm::Value* runtime,
      llvm::IRBuilder<>& builder);

  // Invokes the receive callback function. The received data is written into
  // the buffer pointer to be `output_ptr`. Returns an i1 value indicating
  // whether the receive fired.
  absl::StatusOr<llvm::Value*> ReceiveFromQueue(llvm::IRBuilder<>* builder,
                                                int64_t queue_index,
                                                Receive* receive,
                                                llvm::Value* output_ptr,
                                                llvm::Value* instance_context);
  absl::Status SendToQueue(llvm::IRBuilder<>* builder, int64_t queue_index,
                           Send* send, llvm::Value* send_data_ptr,
                           llvm::Value* instance_context);

  int64_t output_arg_count_;
  const JitCompilationMetadata& metadata_;
  JitBuilderContext& jit_context_;
  std::optional<NodeIrContext> node_context_;
};

absl::Status IrBuilderVisitor::DefaultHandler(Node* node) {
  return absl::UnimplementedError(
      absl::StrCat("Unhandled node: ", node->ToString()));
}

absl::Status IrBuilderVisitor::HandleRegisterWrite(RegisterWrite* write) {
  XLS_RET_CHECK(write->function_base()->IsBlock())
      << "Register-write in non-block function.";
  XLS_ASSIGN_OR_RETURN(Node * paired_read,
                       write->function_base()->AsBlockOrDie()->GetRegisterRead(
                           write->GetRegister()));
  std::vector<std::string> names{"data"};
  names.reserve(write->operand_count() + 1);
  if (write->load_enable()) {
    names.push_back("load_enable");
  }
  if (write->reset()) {
    names.push_back("reset");
  }
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          write, names,
          /*include_wrapper_args=*/write->load_enable().has_value()));
  llvm::Function* function = node_context.llvm_function();
  llvm::Type* return_type =
      type_converter()->ConvertToLlvmType(write->data()->GetType());

  llvm::BasicBlock* current_step =
      node_context.entry_builder().GetInsertBlock();
  // llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  // entry:
  //   (reset present):       if reset == active { goto reset; }
  //   (load_enable present): if !load_enable { goto noload; }
  //   goto write_data;
  // reset:
  //   ret_data_reset := <reset_constant>
  //   goto return_value
  // noload:
  //   ret_data_noload := <old value>
  //   goto return_value
  // write_data:
  //   ret_data_write := <data>
  //   goto return_value
  // return_value:
  //   result := PHI(ret_data_reset, ret_data_noload, ret_data_write)
  //   return result
  std::optional<llvm::BasicBlock*> return_value_block =
      write->reset() || write->load_enable()
          ? std::make_optional(
                llvm::BasicBlock::Create(ctx(), "return_value", function))
          : std::nullopt;
  std::optional<llvm::Constant*> reset_value;
  std::optional<llvm::Value*> no_load_enable_value;
  std::optional<llvm::BasicBlock*> reset_selected;
  std::optional<llvm::BasicBlock*> no_load_enable_selected;
  if (write->reset()) {
    XLS_RET_CHECK(write->GetRegister()->reset())
        << "reset argument without reset behavior set";
    reset_selected =
        llvm::BasicBlock::Create(ctx(), "reset_selected", function);
    llvm::BasicBlock* no_reset_selected =
        llvm::BasicBlock::Create(ctx(), "value_not_being_reset", function);
    llvm::IRBuilder<> current_step_builder(current_step);
    llvm::IRBuilder<> reset_selected_builder(*reset_selected);
    XLS_ASSIGN_OR_RETURN(int64_t op_idx, write->reset_operand_number());
    auto reset_state = node_context.LoadOperand(op_idx, &current_step_builder);
    if (write->GetRegister()->reset()->active_low) {
      // current_step_builder.CreateCondBr(reset_state, *reset_selected,
      //                                   no_reset_selected);
      current_step_builder.CreateCondBr(reset_state, no_reset_selected,
                                        *reset_selected);
    } else {
      current_step_builder.CreateCondBr(reset_state, *reset_selected,
                                        no_reset_selected);
      // current_step_builder.CreateCondBr(reset_state, no_reset_selected,
      //                                   *reset_selected);
    }

    XLS_ASSIGN_OR_RETURN(
        reset_value,
        type_converter()->ToLlvmConstant(
            return_type, write->GetRegister()->reset()->reset_value));
    reset_selected_builder.CreateBr(*return_value_block);

    current_step = no_reset_selected;
  }
  if (write->load_enable()) {
    no_load_enable_selected =
        llvm::BasicBlock::Create(ctx(), "no_load_enable_selected", function);
    llvm::BasicBlock* load_enable_selected =
        llvm::BasicBlock::Create(ctx(), "load_enabled", function);
    llvm::IRBuilder<> no_load_enable_builder(*no_load_enable_selected);
    llvm::IRBuilder<> current_step_builder(current_step);
    auto load_enable_state = node_context.LoadOperand(
        write->load_enable_operand_number().value(), &current_step_builder);
    // the original value is at operand_count+1
    XLS_ASSIGN_OR_RETURN(
        no_load_enable_value,
        node_context.LoadGlobalInput(paired_read, &no_load_enable_builder));
    current_step_builder.CreateCondBr(load_enable_state, load_enable_selected,
                                      *no_load_enable_selected);
    no_load_enable_builder.CreateBr(*return_value_block);

    current_step = load_enable_selected;
  }
  llvm::Value* result_value;
  if (write->load_enable() || write->reset()) {
    llvm::IRBuilder<> current_step_builder(current_step);
    // need a phi.
    llvm::IRBuilder<> return_block_builder(*return_value_block);
    auto phi = return_block_builder.CreatePHI(
        return_type, write->load_enable() && write->reset() ? 3 : 2,
        absl::StrFormat("ONE_OF__WRITE%s%s",
                        write->load_enable() ? "_ORIGINAL" : "",
                        write->reset() ? "_RESET" : ""));
    phi->addIncoming(node_context.LoadOperand(RegisterWrite::kDataOperand,
                                              &current_step_builder),
                     current_step);
    if (write->load_enable()) {
      phi->addIncoming(*no_load_enable_value, *no_load_enable_selected);
    }
    if (write->reset()) {
      phi->addIncoming(*reset_value, *reset_selected);
    }
    result_value = phi;
    current_step_builder.CreateBr(*return_value_block);
    current_step = *return_value_block;
  } else {
    llvm::IRBuilder<> current_step_builder(current_step);
    result_value = node_context.LoadOperand(RegisterWrite::kDataOperand,
                                            &current_step_builder);
  }

  llvm::IRBuilder<> current_step_builder(current_step);
  return FinalizeNodeIrContextWithValue(
      std::move(node_context), result_value, &current_step_builder,
      /*return_value=*/current_step_builder.getFalse(),
      /*result_type=*/write->data()->GetType());
}

absl::Status IrBuilderVisitor::HandleOutputPort(OutputPort* write) {
  XLS_RET_CHECK(write->function_base()->IsBlock())
      << "output-port in non-block function.";
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(write, {"data"}, /*include_wrapper_args=*/false));
  auto value = node_context.LoadOperand(OutputPort::kOperandOperand);
  return FinalizeNodeIrContextWithValue(
      std::move(node_context), value, /*exit_builder=*/std::nullopt,
      /*return_value=*/std::nullopt,
      /*result_type=*/write->operand(0)->GetType());
}

absl::Status IrBuilderVisitor::HandleAdd(BinOp* binop) {
  return HandleBinaryOp(
      binop, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateAdd(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleAndReduce(BitwiseReductionOp* op) {
  return HandleUnaryOp(op, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
    // AND-reduce is equivalent to checking if every bit is set in the
    // input.
    return b.CreateICmpEQ(
        operand, type_converter()->PaddingMask(op->operand(0)->GetType(), b));
  });
}

absl::Status IrBuilderVisitor::HandleAfterAll(AfterAll* after_all) {
  // AfterAll is only meaningful to the compiler and does not actually perform
  // any computation.
  return HandleNaryOp(after_all, [&](absl::Span<llvm::Value* const> operands,
                                     llvm::IRBuilder<>& b) {
    return type_converter()->GetToken();
  });
}

absl::Status IrBuilderVisitor::HandleMinDelay(MinDelay* min_delay) {
  // MinDelay is only meaningful to the compiler and does not actually perform
  // any computation.
  return HandleUnaryOp(min_delay,
                       [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
                         return type_converter()->GetToken();
                       });
}

absl::Status IrBuilderVisitor::HandleAssert(Assert* assert_op) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(assert_op, {"tkn", "condition"},
                                        /*include_wrapper_args=*/true));

  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Function* function = node_context.llvm_function();

  std::string assert_label = assert_op->label().value_or("assert");

  llvm::BasicBlock* after_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_after"), function);

  llvm::BasicBlock* ok_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_ok"), function);

  llvm::IRBuilder<> ok_builder(ok_block);

  ok_builder.CreateBr(after_block);

  llvm::BasicBlock* fail_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(assert_label, "_fail"), function);
  llvm::IRBuilder<> fail_builder(fail_block);
  XLS_RETURN_IF_ERROR(
      InvokeAssertCallback(&fail_builder, assert_op->message(),
                           node_context.GetInterpreterEventsArg()));

  fail_builder.CreateBr(after_block);

  b.CreateCondBr(node_context.LoadOperand(1), ok_block, fail_block);

  auto after_builder = std::make_unique<llvm::IRBuilder<>>(after_block);
  llvm::Value* token = type_converter()->GetToken();
  return FinalizeNodeIrContextWithValue(std::move(node_context), token,
                                        after_builder.get());
}

absl::Status IrBuilderVisitor::HandleArray(Array* array) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(array,
                       NumberedStrings("operand", array->operand_count())));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* array_type =
      type_converter()->ConvertToLlvmType(array->GetType());
  int64_t element_size = type_converter()->GetTypeByteSize(
      array->GetType()->AsArrayOrDie()->element_type());
  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  for (uint32_t i = 0; i < array->size(); ++i) {
    llvm::Value* output_element = b.CreateGEP(
        array_type, output_buffer,
        {
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), i),
        });
    output_element->setName(absl::StrFormat("output_element_%d", i));
    LlvmMemcpy(output_element, node_context.GetOperandPtr(i), element_size, b);
  }

  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer);
}

absl::Status IrBuilderVisitor::HandleTrace(Trace* trace_op) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          trace_op,
          ConcatVectors({"tkn", "condition"},
                        NumberedStrings("arg", trace_op->args().size())),
          /*include_wrapper_args=*/true));

  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* condition = node_context.LoadOperand(1);
  llvm::Value* events_ptr = node_context.GetInterpreterEventsArg();
  llvm::Value* jit_runtime_ptr = node_context.GetJitRuntimeArg();

  std::string trace_name = trace_op->GetName();

  llvm::BasicBlock* after_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(trace_name, "_after"), node_context.llvm_function());

  llvm::BasicBlock* skip_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(trace_name, "_skip"), node_context.llvm_function());

  llvm::IRBuilder<> skip_builder(skip_block);

  skip_builder.CreateBr(after_block);

  llvm::BasicBlock* print_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(trace_name, "_print"), node_context.llvm_function());
  llvm::IRBuilder<> print_builder(print_block);

  XLS_ASSIGN_OR_RETURN(llvm::Value * buffer_ptr,
                       InvokeCreateBufferCallback(&print_builder));

  // Operands are: (tok, pred, ..data_operands..)
  XLS_RET_CHECK_EQ(trace_op->operand(0)->GetType(),
                   trace_op->package()->GetTokenType());
  XLS_RET_CHECK_EQ(trace_op->operand(1)->GetType(),
                   trace_op->package()->GetBitsType(1));

  size_t operand_index = 2;
  for (const FormatStep& step : trace_op->format()) {
    if (std::holds_alternative<std::string>(step)) {
      XLS_RETURN_IF_ERROR(InvokeStringStepCallback(
          &print_builder, std::get<std::string>(step), buffer_ptr));
    } else {
      xls::Node* o = trace_op->operand(operand_index);
      llvm::Value* operand = node_context.LoadOperand(operand_index);
      llvm::AllocaInst* alloca = print_builder.CreateAlloca(operand->getType());
      print_builder.CreateStore(operand, alloca);
      // The way our format strings are currently formed we implicitly refer to
      // the next operand after formatting this one.
      operand_index += 1;
      XLS_RETURN_IF_ERROR(InvokeFormatStepCallback(
          &print_builder, std::get<FormatPreference>(step), o->GetType(),
          alloca, buffer_ptr, jit_runtime_ptr));
    }
  }

  XLS_RETURN_IF_ERROR(InvokeRecordTraceCallback(
      &print_builder, trace_op->verbosity(), buffer_ptr, events_ptr));

  print_builder.CreateBr(after_block);

  b.CreateCondBr(condition, print_block, skip_block);

  auto after_builder = std::make_unique<llvm::IRBuilder<>>(after_block);
  llvm::Value* token = type_converter()->GetToken();
  return FinalizeNodeIrContextWithValue(std::move(node_context), token,
                                        after_builder.get());
}

absl::Status IrBuilderVisitor::HandleArrayIndex(ArrayIndex* index) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          index,
          ConcatVectors({"array"},
                        NumberedStrings("index", index->indices().size()))));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
  };

  Type* array_type = index->array()->GetType();
  for (int64_t i = 1; i < index->operand_count(); ++i) {
    llvm::Value* index_value = node_context.LoadOperand(i);
    gep_indices.push_back(
        ClampIndexInBounds(index_value, array_type->AsArrayOrDie(), b));
    array_type = array_type->AsArrayOrDie()->element_type();
  }
  llvm::Value* indexed_element = b.CreateGEP(
      type_converter()->ConvertToLlvmType(index->array()->GetType()),
      node_context.GetOperandPtr(0), gep_indices);
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 indexed_element);
}

absl::Status IrBuilderVisitor::HandleArraySlice(ArraySlice* slice) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(slice, {"array", "start"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  int64_t width = slice->width();
  llvm::Type* i64 = llvm::Type::getInt64Ty(ctx());
  llvm::Type* i32 = llvm::Type::getInt32Ty(ctx());
  ArrayType* input_array_type = slice->array()->GetType()->AsArrayOrDie();
  int64_t input_array_size = input_array_type->size();

  // An i64 is certainly large enough to hold the array size but may not be
  // large enough to hold the starting index which can be arbitrarily large. To
  // avoid having to work with arbitrarily large types for the index
  // calculations, clamp the start to the array size and cast to i64:
  //
  //   clamped_start = is_inbounds(start_operand) ?
  //                      i64{start} :
  //                      i64{input_array_size}
  llvm::Value* start = node_context.LoadOperand(1);
  llvm::Value* clamped_start =
      b.CreateSelect(IsIndexInBounds(start, input_array_type, b),
                     b.CreateIntCast(start, i64, /*isSigned=*/false),
                     llvm::ConstantInt::get(i64, input_array_size));
  clamped_start->setName("clamped_start");

  // Create a loop which copies one array element at a time.
  LlvmIrLoop loop(width, node_context.entry_builder());

  // Compute the index of the element in the operand array. The index should be
  // clamped to the maximum index value.
  llvm::Value* index =
      loop.body_builder().CreateAdd(clamped_start, loop.index());
  llvm::Value* max_index = llvm::ConstantInt::get(i64, input_array_size - 1);
  llvm::Value* clamped_index = loop.body_builder().CreateSelect(
      loop.body_builder().CreateICmpULE(index, max_index), index, max_index);
  clamped_index->setName("clamped_index");

  // Compute the address of the element in the operand array.
  llvm::Value* operand_buffer = node_context.GetOperandPtr(0);
  llvm::Type* src_array_type =
      type_converter()->ConvertToLlvmType(input_array_type);
  llvm::Value* src_element = loop.body_builder().CreateGEP(
      src_array_type, operand_buffer,
      {llvm::ConstantInt::get(i32, 0), clamped_index});
  src_element->setName("src_element");

  // Compute the address of the element in the result array.
  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  llvm::Type* tgt_array_type =
      type_converter()->ConvertToLlvmType(slice->GetType());
  llvm::Value* tgt_element = loop.body_builder().CreateGEP(
      tgt_array_type, output_buffer,
      {llvm::ConstantInt::get(i32, 0), loop.index()});
  tgt_element->setName("tgt_element");

  // Copy the element from the operand buffer to the result buffer.
  int64_t element_size = type_converter()->GetTypeByteSize(
      slice->array()->GetType()->AsArrayOrDie()->element_type());
  LlvmMemcpy(tgt_element, src_element, element_size, loop.body_builder());

  loop.Finalize();

  return FinalizeNodeIrContextWithPointerToValue(
      std::move(node_context), output_buffer, &loop.exit_builder());
}

absl::Status IrBuilderVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          update,
          ConcatVectors({"array", "update_value"},
                        NumberedStrings("index", update->indices().size()))));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  // First, copy the entire array to update (operand 0) to the output buffer.
  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  LlvmMemcpy(output_buffer, node_context.GetOperandPtr(0),
             type_converter()->GetTypeByteSize(update->GetType()), b);

  // Determine whether the indices are all inbounds. If any are out of bounds
  // then the array update operation is a NOP. Also, gather the GEP indices for
  // computing the address of the element to update if necessary.
  llvm::Value* is_inbounds =
      llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx()), 1);
  Type* array_type = update->array_to_update()->GetType();
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
  };
  llvm::Type* i64 = llvm::Type::getInt64Ty(ctx());
  for (int64_t i = 2; i < update->operand_count(); ++i) {
    llvm::Value* index_value = node_context.LoadOperand(i);
    is_inbounds = b.CreateAnd(
        is_inbounds,
        IsIndexInBounds(index_value, array_type->AsArrayOrDie(), b));
    // Cast the index to i64 for use as a gep index. LLVM does not like GEP
    // indices of unusual widths. This is safe because if the cast to i64 ends
    // up truncating the value, the gep is unused because the index is
    // necessarily out of bounds.
    gep_indices.push_back(
        b.CreateIntCast(index_value, i64, /*isSigned=*/false));
    array_type = array_type->AsArrayOrDie()->element_type();
  }

  llvm::BasicBlock* inbounds_block =
      llvm::BasicBlock::Create(ctx(), "inbounds", node_context.llvm_function());
  llvm::IRBuilder<> inbounds_builder(inbounds_block);

  llvm::BasicBlock* exit_block =
      llvm::BasicBlock::Create(ctx(), "exit", node_context.llvm_function());
  llvm::IRBuilder<> exit_builder = llvm::IRBuilder<>(exit_block);

  // In the inbounds block, compute the address of the element to update in the
  // output buffer using a GEP.
  llvm::Value* output_element = inbounds_builder.CreateGEP(
      type_converter()->ConvertToLlvmType(update->GetType()), output_buffer,
      gep_indices);
  LlvmMemcpy(output_element, node_context.GetOperandPtr(1),
             type_converter()->GetTypeByteSize(update->operand(1)->GetType()),
             inbounds_builder);
  inbounds_builder.CreateBr(exit_block);

  // From the entry block, branch to the inbounds block if the index is
  // inbounds. Otherwise branch to the exit block.
  b.CreateCondBr(is_inbounds, inbounds_block, exit_block);

  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer, &exit_builder);
}

absl::Status IrBuilderVisitor::HandleArrayConcat(ArrayConcat* concat) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(concat,
                       NumberedStrings("operand", concat->operand_count())));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* result_array_type =
      type_converter()->ConvertToLlvmType(concat->GetType());

  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  int64_t result_index = 0;
  for (int64_t i = 0; i < concat->operand_count(); ++i) {
    ArrayType* operand_array_type =
        concat->operand(i)->GetType()->AsArrayOrDie();

    llvm::Value* output_slice = b.CreateGEP(
        result_array_type, output_buffer,
        {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
         llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), result_index)});

    LlvmMemcpy(output_slice, node_context.GetOperandPtr(i),
               type_converter()->GetTypeByteSize(operand_array_type), b);
    result_index += operand_array_type->size();
  }

  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer);
}

absl::Status IrBuilderVisitor::HandleBitSlice(BitSlice* bit_slice) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(bit_slice, {"operand"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* value = node_context.LoadOperand(0);
  Value shift_amount(
      UBits(bit_slice->start(), value->getType()->getIntegerBitWidth()));
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * start,
      type_converter()->ToLlvmConstant(value->getType(), shift_amount));

  // Then shift and "mask" (by casting) the input value.
  llvm::Value* shifted_value = b.CreateLShr(value, start);
  llvm::Value* truncated_value = b.CreateTrunc(
      shifted_value, type_converter()->ConvertToLlvmType(bit_slice->GetType()));

  return FinalizeNodeIrContextWithValue(std::move(node_context),
                                        truncated_value);
}

absl::Status IrBuilderVisitor::HandleBitSliceUpdate(BitSliceUpdate* update) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(update, {"to_update", "start", "update_value"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* to_update = node_context.LoadOperand(0);
  llvm::Value* start = node_context.LoadOperand(1);
  llvm::Value* update_value = node_context.LoadOperand(2);

  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(update->GetType());

  // Zero extend each value to the max of any of the values' widths.
  int max_width = std::max(std::max(to_update->getType()->getIntegerBitWidth(),
                                    start->getType()->getIntegerBitWidth()),
                           update_value->getType()->getIntegerBitWidth());
  llvm::IntegerType* max_width_type = b.getIntNTy(max_width);
  llvm::Value* to_update_wide =
      b.CreateZExt(to_update, max_width_type, "to_update");
  llvm::Value* start_wide = b.CreateZExt(start, max_width_type, "start");
  llvm::Value* update_value_wide =
      b.CreateZExt(update_value, max_width_type, "update_value");

  // If start is greater than or equal to the width of to_update, then the
  // updated slice is entirely out of bounds the result of the operation is
  // simply to_update.
  llvm::Value* in_bounds =
      b.CreateICmpULT(start_wide,
                      llvm::ConstantInt::get(
                          max_width_type, update->operand(0)->BitCountOrDie()),
                      "start_is_inbounds");

  // Create a mask 00..0011..11 where the number of ones is equal to the
  // width of the update value. Then the updated value is:
  //
  //   (~(mask << start) & to_update) | (update_value << start)
  //
  // The shift is guaranteed to be non-poison because of the start value is
  // guarded by a compare against the width lhs of the shift (max_width).
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * mask,
      type_converter()->ToLlvmConstant(
          max_width_type,
          Value(bits_ops::ZeroExtend(
              Bits::AllOnes(update->update_value()->BitCountOrDie()),
              max_width))));
  // Set the shift amount to the 0 in the case of overshift (start >= max_width)
  // to avoid creating a poisonous shift value which can run afoul of LLVM
  // optimization bugs. The shifted value is not even used in this case.
  llvm::Value* shift_amount = b.CreateSelect(
      in_bounds, start_wide, llvm::ConstantInt::get(max_width_type, 0));
  llvm::Value* shifted_mask =
      b.CreateNot(b.CreateShl(mask, shift_amount), "mask");
  llvm::Value* masked_to_update = b.CreateAnd(shifted_mask, to_update_wide);
  llvm::Value* shifted_update_value =
      b.CreateShl(update_value_wide, shift_amount);
  llvm::Value* updated_slice =
      b.CreateTrunc(b.CreateOr(masked_to_update, shifted_update_value),
                    result_type, "updated_slice");

  llvm::Value* result =
      b.CreateSelect(in_bounds, updated_slice, to_update, "result");

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(dynamic_bit_slice, {"operand", "start"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* value = node_context.LoadOperand(0);
  llvm::Value* start = node_context.LoadOperand(1);

  int64_t value_width = value->getType()->getIntegerBitWidth();
  int64_t start_width = start->getType()->getIntegerBitWidth();
  // Either value or start may be wider, so we use the widest of both
  // since LLVM requires both arguments to be of the same type for
  // comparison and shifting.
  int64_t max_width = std::max(start_width, value_width);
  llvm::IntegerType* max_width_type = b.getIntNTy(max_width);
  llvm::Value* value_ext = b.CreateZExt(value, max_width_type);
  llvm::Value* start_ext = b.CreateZExt(start, max_width_type);

  Value operand_width(UBits(value_width, max_width));
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * bit_width,
      type_converter()->ToLlvmConstant(max_width_type, operand_width));

  // "out_of_bounds" indicates whether slice is completely out of bounds.
  llvm::Value* out_of_bounds = b.CreateICmpUGE(start_ext, bit_width);
  llvm::Type* return_type =
      type_converter()->ConvertToLlvmType(dynamic_bit_slice->GetType());
  XLS_ASSIGN_OR_RETURN(
      llvm::Constant * zeros,
      type_converter()->ToLlvmConstant(
          return_type, Value(Bits(dynamic_bit_slice->width()))));
  // Then shift and truncate the input value. Set the shift amount to the 0 in
  // the case of overshift to avoid creating a poisonous shift value which can
  // run afoul of LLVM optimization bugs. The shifted value is not even used in
  // this case.
  llvm::Value* shift_amount = b.CreateSelect(
      out_of_bounds, llvm::ConstantInt::get(max_width_type, 0), start_ext);
  llvm::Value* shifted_value = b.CreateLShr(value_ext, shift_amount);
  llvm::Value* truncated_value = b.CreateTrunc(shifted_value, return_type);
  llvm::Value* result = b.CreateSelect(out_of_bounds, zeros, truncated_value);

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleConcat(Concat* concat) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(concat,
                       NumberedStrings("operand", concat->operand_count())));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* dest_type =
      type_converter()->ConvertToLlvmType(concat->GetType());
  llvm::Value* base = llvm::ConstantInt::get(dest_type, 0);

  int current_shift = concat->BitCountOrDie();
  for (int64_t i = 0; i < concat->operand_count(); ++i) {
    Node* xls_operand = concat->operand(i);
    llvm::Value* operand = node_context.LoadOperand(i);

    // Widen each operand to the full size, shift to the right location, and
    // bitwise or into the result value.
    int64_t operand_width = xls_operand->BitCountOrDie();
    if (operand_width == 0) {
      continue;
    }
    operand = b.CreateZExt(operand, dest_type);
    llvm::Value* shifted_operand =
        b.CreateShl(operand, current_shift - operand_width);
    base = b.CreateOr(base, shifted_operand);

    current_shift -= operand_width;
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), base);
}

absl::Status IrBuilderVisitor::HandleCountedFor(CountedFor* counted_for) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          counted_for,
          ConcatVectors({"initial_value"},
                        NumberedStrings("invariant_arg",
                                        counted_for->invariant_args().size())),
          /*include_wrapper_args=*/true));
  XLS_ASSIGN_OR_RETURN(llvm::Function * body, GetFunction(counted_for->body()));

  // Create a buffer to hold the loop state and write in the initial value.
  llvm::Type* state_type = type_converter()->ConvertToLlvmType(
      counted_for->initial_value()->GetType());
  llvm::Value* loop_state_buffer =
      node_context.entry_builder().CreateAlloca(state_type);
  node_context.entry_builder().CreateStore(node_context.LoadOperand(0),
                                           loop_state_buffer);

  std::vector<llvm::Value*> invariant_arg_buffers;
  for (int64_t i = 1; i < counted_for->operand_count(); ++i) {
    invariant_arg_buffers.push_back(node_context.GetOperandPtr(i));
  }

  LlvmIrLoop loop(counted_for->trip_count(), node_context.entry_builder(),
                  counted_for->stride());

  llvm::Type* index_type = type_converter()->ConvertToLlvmType(
      counted_for->body()->param(0)->GetType());
  llvm::Value* cast_index = loop.body_builder().CreateIntCast(
      loop.index(), index_type, /*isSigned=*/false);
  llvm::Value* index_buffer = loop.body_builder().CreateAlloca(index_type);
  loop.body_builder().CreateStore(cast_index, index_buffer);

  // Signature of body function is:
  //
  //    f(bits[x] index,
  //      StateT loop_state,
  //      InvT_0 invariant_arg_0, ... , InvT_n invariant_arg_n)
  std::vector<llvm::Value*> input_arg_ptrs;
  input_arg_ptrs.push_back(index_buffer);
  input_arg_ptrs.push_back(loop_state_buffer);
  input_arg_ptrs.insert(input_arg_ptrs.end(), invariant_arg_buffers.begin(),
                        invariant_arg_buffers.end());

  llvm::Value* next_state_buffer = loop.body_builder().CreateAlloca(state_type);
  XLS_RETURN_IF_ERROR(CallFunction(body, input_arg_ptrs, {next_state_buffer},
                                   node_context.GetTempBufferArg(),
                                   node_context.GetInterpreterEventsArg(),
                                   node_context.GetInstanceContextArg(),
                                   node_context.GetJitRuntimeArg(),
                                   loop.body_builder())
                          .status());
  // TODO(meheff): 2022/09/09 Rather than loading the state and storing it in
  // the state buffer for the next iteration, simply swap the state and
  // next-state buffer pointers passed to the loop body function.
  llvm::Value* next_state =
      loop.body_builder().CreateLoad(state_type, next_state_buffer);
  loop.body_builder().CreateStore(next_state, loop_state_buffer);

  loop.Finalize();

  return FinalizeNodeIrContextWithPointerToValue(
      std::move(node_context), loop_state_buffer, &loop.exit_builder());
}

absl::Status IrBuilderVisitor::HandleCover(Cover* cover) {
  // TODO(https://github.com/google/xls/issues/499): 2021-09-17: Add coverpoint
  // support to the JIT.
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(cover, {"tkn", "condition"}));
  llvm::Value* token = type_converter()->GetToken();
  return FinalizeNodeIrContextWithValue(std::move(node_context), token);
}

absl::Status IrBuilderVisitor::HandleDecode(Decode* decode) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(decode, {"operand"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* input = node_context.LoadOperand(0);
  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(decode->GetType());

  llvm::Value* index = input;
  if (index->getType()->getIntegerBitWidth() <
      Bits::MinBitCountUnsigned(decode->width())) {
    // Extend index so it's comparable to decode->width().
    index =
        b.CreateZExt(input, type_converter()->GetLlvmBitsType(
                                Bits::MinBitCountUnsigned(decode->width())));
  }

  llvm::Value* shift = input;
  // Make sure shift has the same type as our result.
  if (shift->getType()->getIntegerBitWidth() <
      result_type->getIntegerBitWidth()) {
    shift = b.CreateZExt(shift, result_type);
  } else if (shift->getType()->getIntegerBitWidth() >
             result_type->getIntegerBitWidth()) {
    shift = b.CreateTrunc(shift, result_type);
  }

  // If the input value is greater than this op's width, then return 0.
  // In that case, the shl will produce a poison value, but it'll be unused.
  llvm::Value* overflow = b.CreateICmpUGE(
      index, llvm::ConstantInt::get(index->getType(), decode->width()));
  llvm::Value* result = b.CreateSelect(
      overflow, llvm::ConstantInt::get(result_type, 0),
      b.CreateShl(llvm::ConstantInt::get(result_type, 1), shift));

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleDynamicCountedFor(
    DynamicCountedFor* dynamic_counted_for) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          dynamic_counted_for,
          ConcatVectors(
              {"initial_value", "trip_count", "stride"},
              NumberedStrings("invariant_arg",
                              dynamic_counted_for->invariant_args().size())),
          /*include_wrapper_args=*/true));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* initial_value = node_context.LoadOperand(0);
  llvm::Value* trip_count = node_context.LoadOperand(1);
  llvm::Value* stride = node_context.LoadOperand(2);

  // Grab loop body.
  XLS_ASSIGN_OR_RETURN(llvm::Function * loop_body_function,
                       GetFunction(dynamic_counted_for->body()));

  llvm::Type* index_type = type_converter()->ConvertToLlvmType(
      dynamic_counted_for->body()->param(0)->GetType());
  llvm::Type* state_type = type_converter()->ConvertToLlvmType(
      dynamic_counted_for->initial_value()->GetType());

  llvm::Value* index_buffer = b.CreateAlloca(index_type);
  index_buffer->setName("index_buffer");
  llvm::Value* loop_state_buffer = b.CreateAlloca(state_type);
  loop_state_buffer->setName("loop_state_buffer");
  std::vector<llvm::Value*> invariant_arg_buffers;
  for (int64_t i = 0; i < dynamic_counted_for->invariant_args().size(); ++i) {
    invariant_arg_buffers.push_back(node_context.GetOperandPtr(i + 3));
  }

  // Signature of body function is:
  //
  //    f(bits[x] index,
  //      StateT loop_state,
  //      InvT_0 invariant_arg_0, ... , InvT_n invariant_arg_n)
  std::vector<llvm::Value*> input_arg_ptrs;
  input_arg_ptrs.push_back(index_buffer);
  input_arg_ptrs.push_back(loop_state_buffer);
  input_arg_ptrs.insert(input_arg_ptrs.end(), invariant_arg_buffers.begin(),
                        invariant_arg_buffers.end());

  llvm::Value* next_state_buffer = b.CreateAlloca(state_type);
  next_state_buffer->setName("next_state_buffer");

  // Create basic blocks and corresponding builders. We have 4 blocks:
  // Entry     - the code executed before the loop.
  // Preheader - checks the loop condition.
  // Loop      - calls the loop body, updates the loop carry and index.
  // Exit      - the block we jump to after the loop.
  // ------------------------------------------------------------------
  // Entry
  llvm::BasicBlock* entry_block = b.GetInsertBlock();
  // Preheader
  llvm::BasicBlock* preheader_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(dynamic_counted_for->GetName(), "_preheader"),
      node_context.llvm_function());
  std::unique_ptr<llvm::IRBuilder<>> preheader_builder =
      std::make_unique<llvm::IRBuilder<>>(preheader_block);
  // Loop
  llvm::BasicBlock* loop_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(dynamic_counted_for->GetName(), "_loop_block"),
      node_context.llvm_function());
  std::unique_ptr<llvm::IRBuilder<>> loop_builder =
      std::make_unique<llvm::IRBuilder<>>(loop_block);
  // Exit
  llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(dynamic_counted_for->GetName(), "_exit_block"),
      node_context.llvm_function());
  std::unique_ptr<llvm::IRBuilder<>> exit_builder =
      std::make_unique<llvm::IRBuilder<>>(exit_block);

  // Get initial index, loop carry.
  llvm::Value* init_index = llvm::ConstantInt::get(index_type, 0);
  llvm::Value* init_loop_carry = initial_value;

  // In entry, grab trip_count and stride, extended to match index type.
  // trip_count is zero-extended because the input trip_count is treated as
  // to be unsigned while stride is treated as signed.
  llvm::Value* trip_count_ext = b.CreateZExt(trip_count, index_type);
  trip_count_ext->setName("trip_count_ext");
  llvm::Value* stride_ext = type_converter()->AsSignedValue(
      stride, dynamic_counted_for->stride()->GetType(), b, index_type);
  stride_ext->setName("stride_ext");

  // Calculate index limit and jump entry loop predheader.
  llvm::Value* index_limit = b.CreateMul(trip_count_ext, stride_ext);
  index_limit->setName("index_limit");
  b.CreateBr(preheader_block);

  // Preheader
  // Check if trip_count interations completed.
  // If so, exit loop. Otherwise, keep looping.
  llvm::PHINode* index_phi = preheader_builder->CreatePHI(index_type, 2);
  llvm::PHINode* loop_carry_phi =
      preheader_builder->CreatePHI(init_loop_carry->getType(), 2);
  preheader_builder->CreateStore(index_phi, index_buffer);
  preheader_builder->CreateStore(loop_carry_phi, loop_state_buffer);
  llvm::Value* index_limit_reached =
      preheader_builder->CreateICmpEQ(index_phi, index_limit);
  preheader_builder->CreateCondBr(index_limit_reached, exit_block, loop_block);

  // Loop
  // Call loop body function and increment index before returning to
  // preheader_builder.
  XLS_RETURN_IF_ERROR(
      CallFunction(loop_body_function, input_arg_ptrs, {next_state_buffer},
                   node_context.GetTempBufferArg(),
                   node_context.GetInterpreterEventsArg(),
                   node_context.GetInstanceContextArg(),
                   node_context.GetJitRuntimeArg(), *loop_builder)
          .status());
  llvm::Value* loop_carry =
      loop_builder->CreateLoad(state_type, next_state_buffer);
  loop_carry->setName("next_state");
  loop_builder->CreateStore(loop_carry, loop_state_buffer);
  llvm::Value* inc_index = loop_builder->CreateAdd(index_phi, stride_ext);
  inc_index->setName("inc_index");
  loop_builder->CreateBr(preheader_block);

  // Set predheader Phi node inputs.
  index_phi->addIncoming(init_index, entry_block);
  index_phi->addIncoming(inc_index, loop_block);
  loop_carry_phi->addIncoming(init_loop_carry, entry_block);
  loop_carry_phi->addIncoming(loop_carry, loop_block);

  // Add a single-input PHI node for loop carry output so that,
  // under single static assignment, the loop carry is only used
  // in the loop. Llvm should do this for us in llvm::formLCSSA when we JIT
  // compile, but for some reason that functions fails it's own assertion
  // (L.isLCSSAForm(DT)) after its transformation is applied. So, we do this
  // manually here.
  llvm::PHINode* loop_carry_out =
      exit_builder->CreatePHI(init_loop_carry->getType(), 1);
  loop_carry_out->addIncoming(loop_carry_phi, preheader_block);

  llvm::Value* final_state =
      exit_builder->CreateLoad(state_type, loop_state_buffer);
  final_state->setName("final_state");
  return FinalizeNodeIrContextWithValue(std::move(node_context), final_state,
                                        exit_builder.get());
}

absl::Status IrBuilderVisitor::HandleEncode(Encode* encode) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(encode, {"operand"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* input = node_context.LoadOperand(0);
  llvm::Type* input_type = input->getType();
  llvm::Value* input_one = llvm::ConstantInt::get(input_type, 1);

  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(encode->GetType());
  llvm::Value* result_zero = llvm::ConstantInt::get(result_type, 0);

  const std::string kResultName = "result";
  LlvmIrLoop loop(encode->operand(0)->BitCountOrDie(), b, /*stride=*/1,
                  /*insert_before=*/nullptr,
                  {LoopCarriedValue{kResultName, result_zero}});
  llvm::Value* result = loop.GetLoopCarriedValue(kResultName);

  // For each bit in the input, if it's set, bitwise-OR its [numeric] value
  // with the result.
  llvm::Value* index_as_input_type = loop.body_builder().CreateZExtOrTrunc(
      loop.index(), input_type, "index_as_input_type");
  llvm::Value* index_as_result_type = loop.body_builder().CreateZExtOrTrunc(
      loop.index(), result_type, "index_as_result_type");

  llvm::Value* one_hot_mask = loop.body_builder().CreateShl(
      input_one, index_as_input_type, "one_hot_mask");
  llvm::Value* bit_set = loop.body_builder().CreateICmpEQ(
      loop.body_builder().CreateAnd(input, one_hot_mask), one_hot_mask,
      "bit_is_set");
  llvm::Value* or_value = loop.body_builder().CreateSelect(
      bit_set, index_as_result_type, result_zero);
  llvm::Value* next_result =
      loop.body_builder().CreateOr(result, or_value, "next_result");

  loop.Finalize(/*final_body_block_builder=*/std::nullopt,
                {{kResultName, next_result}});
  return FinalizeNodeIrContextWithValue(std::move(node_context), result,
                                        &loop.exit_builder());
}

absl::Status IrBuilderVisitor::HandleEq(CompareOp* eq) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(eq, {"lhs", "rhs"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* llvm_lhs = node_context.LoadOperand(0);
  llvm::Value* llvm_rhs = node_context.LoadOperand(1);

  Node* lhs = eq->operand(0);
  Node* rhs = eq->operand(1);

  // TODO(meheff): 2022/09/09 Rather than loading each element and comparing
  // individually, do a memcmp of the buffers.
  XLS_ASSIGN_OR_RETURN(std::vector<CompareTerm> eq_terms,
                       ExpandTerms(lhs, llvm_lhs, rhs, llvm_rhs, eq, &b));

  llvm::Value* result = b.getTrue();

  for (const auto& eq_term : eq_terms) {
    llvm::Value* term_test = b.CreateICmpEQ(eq_term.lhs, eq_term.rhs);
    result = b.CreateAnd(result, term_test);
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleGate(Gate* gate) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(gate, {"condition", "data"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* condition = node_context.LoadOperand(0);
  llvm::Value* data = node_context.LoadOperand(1);

  // TODO(meheff): 2022/09/09 Replace with a if/then/else block which does a
  // memcpy or writing zero to the output buffer.
  XLS_ASSIGN_OR_RETURN(llvm::Constant * zero,
                       type_converter()->ToLlvmConstant(
                           gate->GetType(), ZeroOfType(gate->GetType())));
  llvm::Value* result = b.CreateSelect(condition, data, zero);

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleIdentity(UnOp* identity) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(identity, {"operand"}));
  llvm::Value* operand_ptr = node_context.GetOperandPtr(0);
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 operand_ptr);
}

absl::Status IrBuilderVisitor::HandleInvoke(Invoke* invoke) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(invoke, NumberedStrings("arg", invoke->operand_count()),
                       /*include_wrapper_args=*/true));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  XLS_ASSIGN_OR_RETURN(llvm::Function * function,
                       GetFunction(invoke->to_apply()));

  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  std::vector<llvm::Value*> operand_ptrs;
  for (int64_t i = 0; i < invoke->operand_count(); ++i) {
    operand_ptrs.push_back(node_context.GetOperandPtr(i));
  }
  XLS_RETURN_IF_ERROR(CallFunction(function, operand_ptrs, {output_buffer},
                                   node_context.GetTempBufferArg(),
                                   node_context.GetInterpreterEventsArg(),
                                   node_context.GetInstanceContextArg(),
                                   node_context.GetJitRuntimeArg(), b)
                          .status());
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer);
}

absl::Status IrBuilderVisitor::HandleLiteral(Literal* literal) {
  // TODO(meheff): 2022/09/09 Avoid generating separate functions for
  // literals. Simply materialize the literals as constants at their uses.
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(literal, {}));
  Type* xls_type = literal->GetType();
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * llvm_literal,
      type_converter()->ToLlvmConstant(xls_type, literal->value()));
  return FinalizeNodeIrContextWithValue(std::move(node_context), llvm_literal);
}

absl::Status IrBuilderVisitor::HandleMap(Map* map) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(map, {"array"}, /*include_wrapper_args=*/true));
  ArrayType* input_array_type = map->operand(0)->GetType()->AsArrayOrDie();
  ArrayType* result_array_type = map->GetType()->AsArrayOrDie();

  llvm::Type* input_type =
      type_converter()->ConvertToLlvmType(input_array_type);
  llvm::Type* output_type =
      type_converter()->ConvertToLlvmType(result_array_type);

  llvm::Value* input_buffer = node_context.GetOperandPtr(0);
  llvm::Value* output_buffer = node_context.GetOutputPtr(0);

  // Loop through each index of the arrays.
  LlvmIrLoop loop(result_array_type->size(), node_context.entry_builder());

  // Compute address of input and output elements.
  llvm::Type* i32 = llvm::Type::getInt32Ty(ctx());
  llvm::Value* input_element = loop.body_builder().CreateGEP(
      input_type, input_buffer, {llvm::ConstantInt::get(i32, 0), loop.index()});
  llvm::Value* output_element = loop.body_builder().CreateGEP(
      output_type, output_buffer,
      {llvm::ConstantInt::get(i32, 0), loop.index()});

  // Call map function to compute the output element in situ.
  XLS_ASSIGN_OR_RETURN(llvm::Function * to_apply, GetFunction(map->to_apply()));
  XLS_RETURN_IF_ERROR(CallFunction(to_apply, {input_element}, {output_element},
                                   node_context.GetTempBufferArg(),
                                   node_context.GetInterpreterEventsArg(),
                                   node_context.GetInstanceContextArg(),
                                   node_context.GetJitRuntimeArg(),
                                   loop.body_builder())
                          .status());
  loop.Finalize();

  return FinalizeNodeIrContextWithPointerToValue(
      std::move(node_context), output_buffer, &loop.exit_builder());
}

absl::Status IrBuilderVisitor::HandleSMul(ArithOp* mul) {
  return HandleBinaryOpWithOperandConversion(
      mul,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateMul(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleUMul(ArithOp* mul) {
  return HandleBinaryOpWithOperandConversion(
      mul,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateMul(lhs, rhs);
      },
      /*is_signed=*/false);
}

namespace {
// Shared implementation for smulp and umulp.
absl::StatusOr<llvm::Value*> HandleMulp(PartialProductOp* mul,
                                        NodeIrContext* node_context,
                                        LlvmTypeConverter* type_converter,
                                        llvm::LLVMContext& ctx,
                                        bool is_signed) {
  llvm::IRBuilder<>& b = node_context->entry_builder();

  llvm::Value* lhs = node_context->LoadOperand(0);
  llvm::Value* rhs = node_context->LoadOperand(1);
  if (is_signed) {
    lhs = type_converter->AsSignedValue(lhs, mul->operand(0)->GetType(),
                                        node_context->entry_builder());
    rhs = type_converter->AsSignedValue(rhs, mul->operand(1)->GetType(),
                                        node_context->entry_builder());
  }

  xls::Type* result_type = mul->GetType();
  XLS_RET_CHECK(result_type->IsTuple() &&
                result_type->AsTupleOrDie()->element_types().size() == 2);
  llvm::Type* llvm_result_type = type_converter->ConvertToLlvmType(result_type);
  xls::Type* result_element_type =
      mul->GetType()->AsTupleOrDie()->element_type(0);
  llvm::Type* llvm_result_element_type =
      type_converter->ConvertToLlvmType(result_element_type);
  XLS_RET_CHECK(result_element_type->IsEqualTo(
      mul->GetType()->AsTupleOrDie()->element_type(1)));

  XLS_ASSIGN_OR_RETURN(
      llvm::Value * offset,
      type_converter->ToLlvmConstant(
          result_element_type,
          Value(MulpOffsetForSimulation(result_element_type->GetFlatBitCount(),
                                        /*shift_size=*/3))));
  // The outer int cast is unconditionally unsigned because smulp (like umulp)
  // returns a tuple of unsigned ints.
  llvm::Value* product =
      b.CreateIntCast(b.CreateMul(b.CreateIntCast(lhs, llvm_result_element_type,
                                                  /*isSigned=*/is_signed),
                                  b.CreateIntCast(rhs, llvm_result_element_type,
                                                  /*isSigned=*/is_signed)),
                      llvm_result_element_type, /*isSigned=*/false);
  llvm::Value* product_minus_offset = b.CreateSub(product, offset);

  llvm::Value* output_buffer = node_context->GetOutputPtr(0);
  llvm::Value* output_element0 =
      b.CreateGEP(llvm_result_type, output_buffer,
                  {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0),
                   llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0)});
  llvm::Value* output_element1 =
      b.CreateGEP(llvm_result_type, output_buffer,
                  {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0),
                   llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)});
  b.CreateStore(offset, output_element0);
  b.CreateStore(product_minus_offset, output_element1);

  return output_buffer;
}
}  // namespace

absl::Status IrBuilderVisitor::HandleSMulp(PartialProductOp* mul) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(mul, NumberedStrings("operand", mul->operand_count())));

  XLS_ASSIGN_OR_RETURN(llvm::Value * output_buffer,
                       HandleMulp(mul, &node_context, type_converter(), ctx(),
                                  /*is_signed=*/true));
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer);
}

absl::Status IrBuilderVisitor::HandleUMulp(PartialProductOp* mul) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(mul, NumberedStrings("operand", mul->operand_count())));
  XLS_ASSIGN_OR_RETURN(llvm::Value * output_buffer,
                       HandleMulp(mul, &node_context, type_converter(), ctx(),
                                  /*is_signed=*/false));
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer);
}

absl::Status IrBuilderVisitor::HandleNaryAnd(NaryOp* and_op) {
  return HandleNaryOp(and_op, [](absl::Span<llvm::Value* const> operands,
                                 llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateAnd(result, operands[i]);
    }
    return result;
  });
}

absl::Status IrBuilderVisitor::HandleNaryNand(NaryOp* nand_op) {
  return HandleNaryOp(nand_op, [](absl::Span<llvm::Value* const> operands,
                                  llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateAnd(result, operands[i]);
    }
    return b.CreateNot(result);
  });
}

absl::Status IrBuilderVisitor::HandleNaryNor(NaryOp* nor_op) {
  return HandleNaryOp(nor_op, [](absl::Span<llvm::Value* const> operands,
                                 llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateOr(result, operands[i]);
    }
    return b.CreateNot(result);
  });
}

absl::Status IrBuilderVisitor::HandleNaryOr(NaryOp* or_op) {
  return HandleNaryOp(
      or_op, [](absl::Span<llvm::Value* const> operands, llvm::IRBuilder<>& b) {
        llvm::Value* result = operands.front();
        for (int i = 1; i < operands.size(); ++i) {
          result = b.CreateOr(result, operands[i]);
        }
        return result;
      });
}

absl::Status IrBuilderVisitor::HandleNaryXor(NaryOp* xor_op) {
  return HandleNaryOp(xor_op, [](absl::Span<llvm::Value* const> operands,
                                 llvm::IRBuilder<>& b) {
    llvm::Value* result = operands.front();
    for (int i = 1; i < operands.size(); ++i) {
      result = b.CreateXor(result, operands[i]);
    }
    return result;
  });
}

absl::Status IrBuilderVisitor::HandleNe(CompareOp* ne) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(ne, {"lhs", "rhs"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* llvm_lhs = node_context.LoadOperand(0);
  llvm::Value* llvm_rhs = node_context.LoadOperand(1);

  Node* lhs = ne->operand(0);
  Node* rhs = ne->operand(1);

  // TODO(meheff): 2022/09/09 Use memcmp.
  XLS_ASSIGN_OR_RETURN(std::vector<CompareTerm> ne_terms,
                       ExpandTerms(lhs, llvm_lhs, rhs, llvm_rhs, ne, &b));

  llvm::Value* result = b.getFalse();

  for (const auto& ne_term : ne_terms) {
    llvm::Value* term_test = b.CreateICmpNE(ne_term.lhs, ne_term.rhs);
    result = b.CreateOr(result, term_test);
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleNeg(UnOp* neg) {
  return HandleUnaryOp(neg, [](llvm::Value* operand, llvm::IRBuilder<>& b) {
    return b.CreateNeg(operand);
  });
}

absl::Status IrBuilderVisitor::HandleNext(Next* next) {
  std::vector<std::string> param_names({"param", "value"});
  if (next->predicate().has_value()) {
    param_names.push_back("predicate");
  }
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(next, param_names, /*include_wrapper_args=*/true));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Value* value_ptr = node_context.GetOperandPtr(Next::kValueOperand);

  if (!next->predicate().has_value()) {
    LlvmMemcpy(node_context.GetOutputPtr(0), value_ptr,
               type_converter()->GetTypeByteSize(next->value()->GetType()), b);

    // Record that this Next node was activated.
    XLS_RETURN_IF_ERROR(InvokeNextValueCallback(
        &b, next, node_context.GetInstanceContextArg()));

    return FinalizeNodeIrContextWithPointerToValue(
        std::move(node_context), node_context.GetOutputPtr(0), &b);
  }

  // If the predicate is true, emulate the `next_value` node's effects.
  llvm::Value* predicate = node_context.LoadOperand(2);
  LlvmIfThen if_then = CreateIfThen(predicate, b, next->GetName());

  LlvmMemcpy(node_context.GetOutputPtr(0), value_ptr,
             type_converter()->GetTypeByteSize(next->value()->GetType()),
             *if_then.then_builder);

  // Record that this Next node was activated.
  XLS_RETURN_IF_ERROR(InvokeNextValueCallback(
      if_then.then_builder.get(), next, node_context.GetInstanceContextArg()));

  std::unique_ptr<llvm::IRBuilder<>> exit_builder = if_then.Finalize();
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 node_context.GetOutputPtr(0),
                                                 exit_builder.get());
}

absl::Status IrBuilderVisitor::HandleNot(UnOp* not_op) {
  return HandleUnaryOp(not_op, [](llvm::Value* operand, llvm::IRBuilder<>& b) {
    return b.CreateNot(operand);
  });
}

absl::Status IrBuilderVisitor::HandleOneHot(OneHot* one_hot) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(one_hot, {"operand"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* input = node_context.LoadOperand(0);

  llvm::Type* input_type = input->getType();
  int input_width = input_type->getIntegerBitWidth();

  llvm::Value* llvm_false = b.getFalse();
  llvm::Value* llvm_true = b.getTrue();

  llvm::Value* result;
  if (one_hot->operand(0)->GetType()->AsBitsOrDie()->bit_count() > 0) {
    llvm::Value* zeroes;
    if (one_hot->priority() == LsbOrMsb::kLsb) {
      llvm::Function* cttz = llvm::Intrinsic::getDeclaration(
          module(), llvm::Intrinsic::cttz, {input_type});
      // We don't need to pass user data to these intrinsics; they're leaf
      // nodes.
      zeroes = b.CreateCall(cttz, {input, llvm_false});
    } else {
      llvm::Function* ctlz = llvm::Intrinsic::getDeclaration(
          module(), llvm::Intrinsic::ctlz, {input_type});
      zeroes = b.CreateCall(ctlz, {input, llvm_false});
      zeroes = b.CreateSub(llvm::ConstantInt::get(input_type, input_width - 1),
                           zeroes);
    }

    // If the input is zero, then return the special high-bit value.
    llvm::Value* zero_value = llvm::ConstantInt::get(input_type, 0);
    llvm::Value* width_value = llvm::ConstantInt::get(
        input_type, one_hot->operand(0)->GetType()->GetFlatBitCount());
    llvm::Value* eq_zero = b.CreateICmpEQ(input, zero_value);
    llvm::Value* shift_amount = b.CreateSelect(eq_zero, width_value, zeroes);

    llvm::Type* result_type =
        type_converter()->ConvertToLlvmType(one_hot->GetType());
    result = b.CreateShl(llvm::ConstantInt::get(result_type, 1),
                         b.CreateZExt(shift_amount, result_type));
  } else {
    result = llvm_true;
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

// Bitwise ORs the value in `src_buffer` with the value in `tgt_buffer` of type
// `xls_type` and writes the result back to `tgt_buffer`. Any newly created
// basic blocks are added before `insert_before` (if non-null). This can be used
// to properly nest the basic blocks which form loops. `builder` is consumed and
// used for building the IR. Returns a builder which inserts after the entire OR
// operation.
std::unique_ptr<llvm::IRBuilder<>> OrInPlace(
    llvm::Value* src_buffer, llvm::Value* tgt_buffer, Type* xls_type,
    LlvmTypeConverter* type_converter, llvm::BasicBlock* insert_before,
    std::unique_ptr<llvm::IRBuilder<>> builder) {
  if (xls_type->GetFlatBitCount() == 0) {
    return builder;
  }
  if (xls_type->IsBits()) {
    llvm::Value* src = builder->CreateLoad(
        type_converter->ConvertToLlvmType(xls_type), src_buffer);
    llvm::Value* tgt = builder->CreateLoad(
        type_converter->ConvertToLlvmType(xls_type), tgt_buffer);
    llvm::Value* result = builder->CreateOr(src, tgt);
    builder->CreateStore(result, tgt_buffer);
    return builder;
  }
  if (xls_type->IsArray()) {
    // Create a loop in LLVM and iterate through each element.
    ArrayType* array_type = xls_type->AsArrayOrDie();
    LlvmIrLoop loop(array_type->size(), *builder, /*stride=*/1, insert_before);
    llvm::Value* element_src_buffer = loop.body_builder().CreateGEP(
        type_converter->ConvertToLlvmType(array_type), src_buffer,
        {
            loop.body_builder().getInt32(0),
            loop.index(),
        });
    llvm::Value* element_tgt_buffer = loop.body_builder().CreateGEP(
        type_converter->ConvertToLlvmType(array_type), tgt_buffer,
        {
            loop.body_builder().getInt32(0),
            loop.index(),
        });
    builder = OrInPlace(element_src_buffer, element_tgt_buffer,
                        array_type->element_type(), type_converter,
                        /*insert_before=*/loop.exit_builder().GetInsertBlock(),
                        loop.ConsumeBodyBuilder());
    loop.Finalize(builder.get());
    return loop.ConsumeExitBuilder();
  }
  // Iterate through each tuple element (unrolled).
  CHECK(xls_type->IsTuple());
  TupleType* tuple_type = xls_type->AsTupleOrDie();
  for (int64_t i = 0; i < tuple_type->size(); ++i) {
    llvm::Value* element_src_buffer = builder->CreateGEP(
        type_converter->ConvertToLlvmType(tuple_type), src_buffer,
        {
            builder->getInt32(0),
            builder->getInt32(i),
        });
    llvm::Value* element_tgt_buffer = builder->CreateGEP(
        type_converter->ConvertToLlvmType(tuple_type), tgt_buffer,
        {
            builder->getInt32(0),
            builder->getInt32(i),
        });
    builder = OrInPlace(element_src_buffer, element_tgt_buffer,
                        tuple_type->element_type(i), type_converter,
                        insert_before, std::move(builder));
  }
  return builder;
}

absl::Status IrBuilderVisitor::HandleOneHotSel(OneHotSelect* sel) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          sel, ConcatVectors({"selector"},
                             NumberedStrings("case", sel->cases().size()))));

  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  llvm::Value* selector = node_context.LoadOperand(0);

  // To make management of the builders easier, create a new block with a
  // std::unique_ptr builder. This builder variable will be updated as the IR
  // is built.
  llvm::BasicBlock* start_block =
      llvm::BasicBlock::Create(ctx(), "start", node_context.llvm_function());
  node_context.entry_builder().CreateBr(start_block);
  auto builder = std::make_unique<llvm::IRBuilder<>>(start_block);

  // Initially store a zero value in the output buffer then OR in any selected
  // cases.
  builder->CreateStore(LlvmTypeConverter::ZeroOfType(
                           type_converter()->ConvertToLlvmType(sel->GetType())),
                       output_buffer);
  for (int64_t i = 0; i < sel->cases().size(); ++i) {
    // Create a if-then construct where the `then` block is executed if the case
    // is selected. This `then` block ORs in the case value.
    llvm::Value* select_bit = builder->CreateTrunc(
        builder->CreateLShr(selector, i), llvm::Type::getInt1Ty(ctx()));
    LlvmIfThen if_then =
        CreateIfThen(select_bit, *builder, absl::StrFormat("case_%d", i));
    llvm::Value* case_buffer =
        node_context.GetOperandPtr(i + 1, if_then.then_builder.get());
    std::unique_ptr<llvm::IRBuilder<>> b =
        OrInPlace(case_buffer, output_buffer, sel->GetType(), type_converter(),
                  /*insert_before=*/if_then.join_builder->GetInsertBlock(),
                  std::move(if_then.then_builder));
    builder = if_then.Finalize(b.get());
  }

  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer, builder.get());
}

absl::Status IrBuilderVisitor::HandlePrioritySel(PrioritySelect* sel) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          sel, ConcatVectors({"selector"},
                             NumberedStrings("case", sel->cases().size()))));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* selector = node_context.LoadOperand(0);

  std::vector<llvm::Value*> cases;
  for (int64_t i = 1; i < sel->operand_count(); ++i) {
    cases.push_back(node_context.GetOperandPtr(i));
  }
  llvm::Type* sel_type = type_converter()->ConvertToLlvmType(sel->GetType());

  // Create a base case of a buffer containing all zeros.
  llvm::Value* typed_zero = b.CreateAlloca(sel_type);
  b.CreateStore(LlvmTypeConverter::ZeroOfType(sel_type), typed_zero);
  llvm::Value* llvm_false = b.getFalse();

  // Get index to select by counting trailing zeros
  llvm::Function* cttz = llvm::Intrinsic::getDeclaration(
      module(), llvm::Intrinsic::cttz, {selector->getType()});
  llvm::Value* selected_index = b.CreateCall(cttz, {selector, llvm_false});

  // Sel is implemented by a cascading series of select ops, e.g.,
  // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
  llvm::Value* llvm_sel = typed_zero;
  for (int i = sel->cases().size() - 1; i >= 0; i--) {
    llvm::Value* current_index = llvm::ConstantInt::get(selector->getType(), i);
    llvm::Value* cmp = b.CreateICmpEQ(selected_index, current_index);
    llvm_sel = b.CreateSelect(cmp, cases.at(i), llvm_sel);
  }

  llvm::Value* selector_is_zero =
      b.CreateICmpEQ(selector, llvm::ConstantInt::get(selector->getType(), 0));
  llvm_sel = b.CreateSelect(selector_is_zero, typed_zero, llvm_sel);
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 llvm_sel);
}

absl::Status IrBuilderVisitor::HandleOrReduce(BitwiseReductionOp* op) {
  return HandleUnaryOp(op, [](llvm::Value* operand, llvm::IRBuilder<>& b) {
    // OR-reduce is equivalent to checking if any bit is set in the input.
    return b.CreateICmpNE(operand,
                          llvm::ConstantInt::get(operand->getType(), 0));
  });
}

absl::Status IrBuilderVisitor::HandleReverse(UnOp* reverse) {
  return HandleUnaryOp(
      reverse, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
        llvm::Function* reverse_fn = llvm::Intrinsic::getDeclaration(
            module(), llvm::Intrinsic::bitreverse, {operand->getType()});
        // Shift right logically by native width - natural with.
        llvm::Value* result = b.CreateCall(reverse_fn, {operand});
        result = b.CreateLShr(
            result,
            llvm::ConstantInt::get(result->getType(),
                                   type_converter()->GetLlvmBitCount(
                                       reverse->GetType()->AsBitsOrDie()) -
                                       reverse->GetType()->GetFlatBitCount()));
        return result;
      });
}

absl::Status IrBuilderVisitor::HandleSDiv(BinOp* binop) {
  return HandleBinaryOp(
      binop,
      [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitDiv(lhs, rhs, binop->BitCountOrDie(), /*is_signed=*/true,
                       type_converter(), &b);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSMod(BinOp* binop) {
  return HandleBinaryOp(
      binop,
      [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitMod(lhs, rhs, /*is_signed=*/true, &b);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSel(Select* sel) {
  std::vector<std::string> operand_names =
      ConcatVectors({"selector"}, NumberedStrings("case", sel->cases().size()));
  if (sel->default_value().has_value()) {
    operand_names.push_back("default");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(sel, operand_names));

  llvm::IRBuilder<>& b = node_context.entry_builder();

  // Sel is implemented by a cascading series of select ops, e.g.,
  // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
  llvm::Value* selector = node_context.LoadOperand(0);
  llvm::Value* llvm_sel =
      sel->default_value()
          ? node_context.GetOperandPtr(sel->operand_count() - 1)
          : nullptr;
  for (int i = sel->cases().size() - 1; i >= 0; i--) {
    llvm::Value* llvm_case = node_context.GetOperandPtr(i + 1);
    if (llvm_sel == nullptr) {
      // The last element in the select tree isn't a sel, but an actual value.
      llvm_sel = llvm_case;
    } else {
      llvm::Value* index = llvm::ConstantInt::get(selector->getType(), i);
      llvm::Value* cmp = b.CreateICmpEQ(selector, index);
      llvm_sel = b.CreateSelect(cmp, llvm_case, llvm_sel);
    }
  }
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 llvm_sel);
}

absl::Status IrBuilderVisitor::HandleSGe(CompareOp* ge) {
  return HandleBinaryOp(
      ge,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSGE(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSGt(CompareOp* gt) {
  return HandleBinaryOp(
      gt,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSGT(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSignExtend(ExtendOp* sign_ext) {
  return HandleUnaryOp(
      sign_ext,
      [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
        llvm::Type* new_type =
            type_converter()->ConvertToLlvmType(sign_ext->GetType());
        return b.CreateSExt(operand, new_type);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSLe(CompareOp* le) {
  return HandleBinaryOp(
      le,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSLE(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleSLt(CompareOp* lt) {
  return HandleBinaryOp(
      lt,
      [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpSLT(lhs, rhs);
      },
      /*is_signed=*/true);
}

absl::Status IrBuilderVisitor::HandleShll(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitShiftOp(binop, lhs, rhs, &b, type_converter());
      });
}

absl::Status IrBuilderVisitor::HandleShra(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        // Only the LHS is treated as a signed number.
        return EmitShiftOp(binop,
                           type_converter()->AsSignedValue(
                               lhs, binop->operand(0)->GetType(), b),
                           rhs, &b, type_converter());
      });
}

absl::Status IrBuilderVisitor::HandleShrl(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitShiftOp(binop, lhs, rhs, &b, type_converter());
      });
}

absl::Status IrBuilderVisitor::HandleSub(BinOp* binop) {
  return HandleBinaryOp(
      binop, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateSub(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleTuple(Tuple* tuple) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(tuple,
                       NumberedStrings("operand", tuple->operand_count())));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* tuple_type =
      type_converter()->ConvertToLlvmType(tuple->GetType());

  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  for (int64_t i = 0; i < tuple->operand_count(); ++i) {
    std::vector<llvm::Value*> gep_indices = {
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), i)};
    llvm::Value* output_element =
        b.CreateGEP(tuple_type, output_buffer, gep_indices);
    int64_t element_size =
        type_converter()->GetTypeByteSize(tuple->operand(i)->GetType());
    LlvmMemcpy(output_element, node_context.GetOperandPtr(i), element_size, b);
  }

  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context),
                                                 output_buffer);
}

absl::Status IrBuilderVisitor::HandleTupleIndex(TupleIndex* index) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(index, {"operand"}, index->operand_count()));

  llvm::Value* gep = node_context.entry_builder().CreateGEP(
      type_converter()->ConvertToLlvmType(index->operand(0)->GetType()),
      node_context.GetOperandPtr(0),
      {
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), index->index()),
      });
  return FinalizeNodeIrContextWithPointerToValue(std::move(node_context), gep);
}

absl::Status IrBuilderVisitor::HandleUDiv(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitDiv(lhs, rhs, binop->BitCountOrDie(), /*is_signed=*/false,
                       type_converter(), &b);
      });
}

absl::Status IrBuilderVisitor::HandleUMod(BinOp* binop) {
  return HandleBinaryOp(
      binop, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return EmitMod(lhs, rhs, /*is_signed=*/false, &b);
      });
}

absl::Status IrBuilderVisitor::HandleUGe(CompareOp* ge) {
  return HandleBinaryOp(
      ge, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpUGE(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleUGt(CompareOp* gt) {
  return HandleBinaryOp(
      gt, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpUGT(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleULe(CompareOp* le) {
  return HandleBinaryOp(
      le, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpULE(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleULt(CompareOp* lt) {
  return HandleBinaryOp(
      lt, [](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        return b.CreateICmpULT(lhs, rhs);
      });
}

absl::Status IrBuilderVisitor::HandleXorReduce(BitwiseReductionOp* op) {
  return HandleUnaryOp(op, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
    // XOR-reduce is equivalent to checking if the number of set bits is odd.
    llvm::Function* ctpop = llvm::Intrinsic::getDeclaration(
        module(), llvm::Intrinsic::ctpop, {operand->getType()});
    // We don't need to pass user data to intrinsics; they're leaf nodes.
    llvm::Value* pop_count = b.CreateCall(ctpop, {operand});

    // Once we have the pop count, truncate to the first (i.e., "is odd") bit.
    return b.CreateTrunc(pop_count, llvm::IntegerType::get(ctx(), 1));
  });
}

absl::Status IrBuilderVisitor::HandleZeroExtend(ExtendOp* zero_ext) {
  return HandleUnaryOp(
      zero_ext, [&](llvm::Value* operand, llvm::IRBuilder<>& b) {
        llvm::Type* new_type =
            type_converter()->ConvertToLlvmType(zero_ext->GetType());
        return b.CreateZExt(operand, new_type);
      });
}

absl::Status IrBuilderVisitor::HandleUnaryOp(
    Node* node,
    std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>&)> build_result,
    bool is_signed) {
  XLS_RET_CHECK_EQ(node->operand_count(), 1);
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(node, {"operand"}));
  return FinalizeNodeIrContextWithValue(
      std::move(node_context),
      build_result(MaybeAsSigned(node_context.LoadOperand(0),
                                 node->operand(0)->GetType(),
                                 node_context.entry_builder(), is_signed),
                   node_context.entry_builder()));
}

absl::Status IrBuilderVisitor::HandleBinaryOp(
    Node* node,
    std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::IRBuilder<>&)>
        build_result,
    bool is_signed) {
  XLS_RET_CHECK_EQ(node->operand_count(), 2);
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(node, {"lhs", "rhs"}));
  llvm::Value* lhs =
      MaybeAsSigned(node_context.LoadOperand(0), node->operand(0)->GetType(),
                    node_context.entry_builder(), is_signed);
  llvm::Value* rhs =
      MaybeAsSigned(node_context.LoadOperand(1), node->operand(1)->GetType(),
                    node_context.entry_builder(), is_signed);
  return FinalizeNodeIrContextWithValue(
      std::move(node_context),
      build_result(lhs, rhs, node_context.entry_builder()));
}

absl::Status IrBuilderVisitor::HandleNaryOp(
    Node* node, std::function<llvm::Value*(absl::Span<llvm::Value* const>,
                                           llvm::IRBuilder<>&)>
                    build_result) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(node,
                       NumberedStrings("operand", node->operand_count())));
  std::vector<llvm::Value*> args;
  for (int64_t i = 0; i < node->operand_count(); ++i) {
    args.push_back(node_context.LoadOperand(i));
  }
  return FinalizeNodeIrContextWithValue(
      std::move(node_context),
      build_result(args, node_context.entry_builder()));
}

absl::Status IrBuilderVisitor::HandleBinaryOpWithOperandConversion(
    Node* node,
    std::function<llvm::Value*(llvm::Value*, llvm::Value*, llvm::IRBuilder<>&)>
        build_result,
    bool is_signed) {
  XLS_RET_CHECK_EQ(node->operand_count(), 2);
  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(node->GetType());
  return HandleBinaryOp(
      node, [&](llvm::Value* lhs, llvm::Value* rhs, llvm::IRBuilder<>& b) {
        llvm::Value* converted_lhs =
            MaybeAsSigned(lhs, node->operand(0)->GetType(), b, is_signed);
        llvm::Value* converted_rhs =
            MaybeAsSigned(rhs, node->operand(1)->GetType(), b, is_signed);
        return build_result(
            b.CreateIntCast(converted_lhs, result_type, is_signed),
            b.CreateIntCast(converted_rhs, result_type, is_signed), b);
      });
}

absl::StatusOr<NodeIrContext> IrBuilderVisitor::NewNodeIrContext(
    Node* node, absl::Span<const std::string> operand_names,
    bool include_wrapper_args) {
  return NodeIrContext::Create(node, operand_names, output_arg_count_,
                               include_wrapper_args, metadata_, jit_context_);
}

absl::Status IrBuilderVisitor::FinalizeNodeIrContextWithValue(
    NodeIrContext&& node_context, llvm::Value* result,
    std::optional<llvm::IRBuilder<>*> exit_builder,
    std::optional<llvm::Value*> return_value,
    std::optional<Type*> result_type) {
  node_context.FinalizeWithValue(result, exit_builder, return_value,
                                 result_type);
  node_context_.emplace(std::move(node_context));
  return absl::OkStatus();
}

absl::Status IrBuilderVisitor::FinalizeNodeIrContextWithPointerToValue(
    NodeIrContext&& node_context, llvm::Value* result_buffer,
    std::optional<llvm::IRBuilder<>*> exit_builder,
    std::optional<llvm::Value*> return_value,
    std::optional<Type*> result_type) {
  node_context.FinalizeWithPointerToValue(result_buffer, exit_builder,
                                          return_value, result_type);
  node_context_.emplace(std::move(node_context));
  return absl::OkStatus();
}

absl::StatusOr<llvm::Value*> IrBuilderVisitor::CallFunction(
    llvm::Function* f, absl::Span<llvm::Value* const> inputs,
    absl::Span<llvm::Value* const> outputs, llvm::Value* temp_buffer,
    llvm::Value* events, llvm::Value* instance_context, llvm::Value* runtime,
    llvm::IRBuilder<>& builder) {
  llvm::Type* input_pointer_array_type =
      llvm::ArrayType::get(llvm::PointerType::get(ctx(), 0), inputs.size());
  llvm::Value* input_arg_array = builder.CreateAlloca(input_pointer_array_type);
  input_arg_array->setName("input_arg_array");
  for (int64_t i = 0; i < inputs.size(); ++i) {
    llvm::Value* input_buffer = inputs[i];
    llvm::Value* gep = builder.CreateGEP(
        input_pointer_array_type, input_arg_array,
        {
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), i),
        });
    builder.CreateStore(input_buffer, gep);
  }

  llvm::Type* output_pointer_array_type =
      llvm::ArrayType::get(llvm::PointerType::get(ctx(), 0), outputs.size());
  llvm::Value* output_arg_array =
      builder.CreateAlloca(output_pointer_array_type);
  output_arg_array->setName("output_arg_array");
  for (int64_t i = 0; i < outputs.size(); ++i) {
    llvm::Value* output_buffer = outputs[i];
    llvm::Value* gep = builder.CreateGEP(
        output_pointer_array_type, output_arg_array,
        {
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), i),
        });
    builder.CreateStore(output_buffer, gep);
  }

  std::vector<llvm::Value*> args = {input_arg_array,
                                    output_arg_array,
                                    temp_buffer,
                                    events,
                                    instance_context,
                                    runtime,
                                    /*continuation_point=*/builder.getInt64(0)};
  return builder.CreateCall(f, args);
}

bool QueueReceiveWrapper(InstanceContext* instance_context, int64_t queue_index,
                         uint8_t* buffer) {
  return instance_context->channel_queues[queue_index]->ReadRaw(buffer);
}

absl::StatusOr<llvm::Value*> IrBuilderVisitor::ReceiveFromQueue(
    llvm::IRBuilder<>* builder, int64_t queue_index, Receive* receive,
    llvm::Value* output_ptr, llvm::Value* instance_context) {
  llvm::Type* i64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* bool_type = llvm::Type::getInt1Ty(ctx());
  llvm::Type* ptr_type = llvm::PointerType::get(ctx(), 0);

  // Call the user-provided function of type ProcJit::RecvFnT to receive the
  // value.
  std::vector<llvm::Type*> params = {ptr_type, i64_type, ptr_type};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(bool_type, params, /*isVarArg=*/false);

  // Call the wrapper to JitChannelQueue::Recv.
  llvm::Value* queue_index_value =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()), queue_index);

  std::vector<llvm::Value*> args = {
      builder->CreateIntToPtr(instance_context, ptr_type), queue_index_value,
      output_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()),
                             absl::bit_cast<uint64_t>(&QueueReceiveWrapper));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  llvm::Value* receive_fired = builder->CreateCall(fn_type, fn_ptr, args);
  return receive_fired;
}

absl::Status IrBuilderVisitor::HandleReceive(Receive* recv) {
  std::vector<std::string> operand_names = {"tkn"};
  if (recv->predicate().has_value()) {
    operand_names.push_back("predicate");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(recv, operand_names,
                                        /*include_wrapper_args=*/true));
  llvm::Value* instance_context = node_context.GetInstanceContextArg();

  int64_t queue_index =
      jit_context_.GetOrAllocateQueueIndex(recv->channel_name());

  llvm::Value* output_buffer = node_context.GetOutputPtr(0);
  // The data buffer is element 1 of the output tuple.
  llvm::Type* receive_type =
      type_converter()->ConvertToLlvmType(recv->GetType());
  llvm::Value* data_buffer = node_context.entry_builder().CreateGEP(
      receive_type, output_buffer,
      {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
       llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 1)});
  data_buffer->setName("data_buffer");

  if (recv->predicate().has_value()) {
    llvm::Value* predicate = node_context.LoadOperand(1);

    // First, declare the join block (so the case blocks can refer to it).
    llvm::BasicBlock* join_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_join"),
                                 node_context.llvm_function());

    // Create a block/branch for the true predicate case.
    llvm::BasicBlock* true_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_true"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> true_builder(true_block);
    XLS_ASSIGN_OR_RETURN(llvm::Value * true_receive_fired,
                         ReceiveFromQueue(&true_builder, queue_index, recv,
                                          data_buffer, instance_context));
    true_builder.CreateBr(join_block);

    // And the same for a false predicate - this will store a zero
    // value into the data buffer.
    llvm::BasicBlock* false_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_false"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    llvm::Type* data_type =
        type_converter()->ConvertToLlvmType(recv->GetPayloadType());
    false_builder.CreateStore(LlvmTypeConverter::ZeroOfType(data_type),
                              data_buffer);
    false_builder.CreateBr(join_block);

    // Next, create a branch op w/the original builder,
    node_context.entry_builder().CreateCondBr(predicate, true_block,
                                              false_block);

    // then join the two branches back together.
    llvm::IRBuilder<> join_builder(join_block);

    llvm::PHINode* receive_fired = join_builder.CreatePHI(
        llvm::Type::getInt1Ty(ctx()), /*NumReservedValues=*/2);
    receive_fired->addIncoming(true_receive_fired, true_block);
    receive_fired->addIncoming(llvm::ConstantInt::getFalse(ctx()), false_block);
    receive_fired->setName("receive_fired");
    if (!recv->is_blocking()) {
      // If the receive is non-blocking, the output of the receive has an
      // additional element (index 2) which indicates whether the receive fired.
      llvm::Value* receive_fired_buffer = join_builder.CreateGEP(
          receive_type, output_buffer,
          {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
           llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 2)});
      join_builder.CreateStore(receive_fired, receive_fired_buffer);
    }
    return FinalizeNodeIrContextWithPointerToValue(
        std::move(node_context), output_buffer,
        /*exit_builder=*/&join_builder,
        /*return_value=*/recv->is_blocking()
            ? join_builder.CreateAnd(predicate,
                                     join_builder.CreateNot(receive_fired))
            : join_builder.getFalse());
  }
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * receive_fired,
      ReceiveFromQueue(&node_context.entry_builder(), queue_index, recv,
                       data_buffer, instance_context));
  receive_fired->setName("receive_fired");
  if (!recv->is_blocking()) {
    // If the receive is non-blocking, the output of the receive has an
    // additional element (index 2) which indicates whether the receive fired.
    llvm::Value* receive_fired_buffer = node_context.entry_builder().CreateGEP(
        receive_type, output_buffer,
        {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0),
         llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 2)});
    node_context.entry_builder().CreateStore(receive_fired,
                                             receive_fired_buffer);
  }
  return FinalizeNodeIrContextWithPointerToValue(
      std::move(node_context), output_buffer,
      /*exit_builder=*/&node_context.entry_builder(),
      /*return_value=*/
      recv->is_blocking()
          ? node_context.entry_builder().CreateNot(receive_fired)
          : node_context.entry_builder().getFalse());
}

void QueueSendWrapper(InstanceContext* instance_context, int64_t queue_index,
                      const uint8_t* data) {
  instance_context->channel_queues[queue_index]->WriteRaw(data);
}

absl::Status IrBuilderVisitor::SendToQueue(llvm::IRBuilder<>* builder,
                                           int64_t queue_index, Send* send,
                                           llvm::Value* send_data_ptr,
                                           llvm::Value* instance_context) {
  llvm::Type* i64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* ptr_type = llvm::PointerType::get(ctx(), 0);

  // We do the same for sending/writing as we do for receiving/reading
  // above (set up and call an external function).
  std::vector<llvm::Type*> params = {ptr_type, i64_type, ptr_type};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  llvm::Value* queue_index_value =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()), queue_index);
  std::vector<llvm::Value*> args = {
      builder->CreateIntToPtr(instance_context, ptr_type), queue_index_value,
      send_data_ptr};

  llvm::ConstantInt* fn_addr =
      llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx()),
                             absl::bit_cast<uint64_t>(&QueueSendWrapper));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  builder->CreateCall(fn_type, fn_ptr, args);
  return absl::OkStatus();
}

absl::Status IrBuilderVisitor::HandleSend(Send* send) {
  std::vector<std::string> operand_names = {"tkn", "data"};
  if (send->predicate().has_value()) {
    operand_names.push_back("predicate");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(send, operand_names,
                                        /*include_wrapper_args=*/true));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* data_ptr = node_context.GetOperandPtr(1);
  llvm::Value* instance_context = node_context.GetInstanceContextArg();

  int64_t queue_index =
      jit_context_.GetOrAllocateQueueIndex(send->channel_name());

  if (send->predicate().has_value()) {
    llvm::Value* predicate = node_context.LoadOperand(2);

    // First, declare the join block (so the case blocks can refer to it).
    llvm::BasicBlock* join_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_join"),
                                 node_context.llvm_function());

    llvm::BasicBlock* true_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_true"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> true_builder(true_block);
    XLS_RETURN_IF_ERROR(SendToQueue(&true_builder, queue_index, send, data_ptr,
                                    instance_context));
    true_builder.CreateBr(join_block);

    llvm::BasicBlock* false_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_false"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    false_builder.CreateBr(join_block);

    b.CreateCondBr(predicate, true_block, false_block);

    auto exit_builder = std::make_unique<llvm::IRBuilder<>>(join_block);

    // The node function should return true if data was sent. This will trigger
    // an early exit from the top-level function.
    return FinalizeNodeIrContextWithValue(std::move(node_context),
                                          type_converter()->GetToken(),
                                          exit_builder.get(),
                                          /*return_value=*/predicate);
  }
  // Unconditional send.
  XLS_RETURN_IF_ERROR(
      SendToQueue(&b, queue_index, send, data_ptr, instance_context));

  // The node function should return true if data was sent. This will trigger
  // an early exit from the top-level function.
  return FinalizeNodeIrContextWithValue(
      std::move(node_context), type_converter()->GetToken(),
      /*exit_builder=*/&node_context.entry_builder(),
      /*return_value=*/node_context.entry_builder().getTrue());
}

}  // namespace

llvm::Value* LlvmMemcpy(llvm::Value* tgt, llvm::Value* src, int64_t size,
                        llvm::IRBuilder<>& builder) {
  CHECK(tgt->getType()->isPointerTy());
  CHECK(src->getType()->isPointerTy());
  return builder.CreateMemCpy(tgt, llvm::MaybeAlign(1), src,
                              llvm::MaybeAlign(1), size);
}

absl::StatusOr<NodeFunction> CreateNodeFunction(
    Node* node, int64_t output_arg_count,
    const JitCompilationMetadata& metadata, JitBuilderContext& jit_context) {
  IrBuilderVisitor visitor(output_arg_count, metadata, jit_context);
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  NodeIrContext node_context = visitor.ConsumeNodeIrContext();
  return NodeFunction{.node = node,
                      .function = node_context.llvm_function(),
                      .operand_arguments = std::vector<Node*>(
                          node_context.GetOperandArgs().begin(),
                          node_context.GetOperandArgs().end()),
                      .output_arg_count = output_arg_count,
                      .has_metadata_args = node_context.has_metadata_args()};
}

}  // namespace xls
