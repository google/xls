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

#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "llvm/include/llvm/IR/Constants.h"
#include "llvm/include/llvm/IR/DerivedTypes.h"
#include "llvm/include/llvm/IR/IRBuilder.h"
#include "llvm/include/llvm/IR/Instructions.h"
#include "llvm/include/llvm/IR/LLVMContext.h"
#include "llvm/include/llvm/IR/Module.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/value_helpers.h"
#include "xls/jit/jit_channel_queue.h"
#include "xls/jit/jit_runtime.h"
#include "xls/jit/llvm_type_converter.h"
#include "xls/jit/orc_jit.h"

namespace xls {
namespace {

// Returns a sequence of numbered strings. Example: NumberedStrings("foo", 3)
// returns: {"foo_0", "foo_1", "foo_2"}
std::vector<std::string> NumberedStrings(absl::string_view s, int64_t count) {
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
    XLS_CHECK_EQ(op, Op::kShrl);
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

llvm::Value* EmitDiv(llvm::Value* lhs, llvm::Value* rhs, bool is_signed,
                     LlvmTypeConverter* type_converter,
                     llvm::IRBuilder<>* builder) {
  // XLS div semantics differ from LLVM's (and most software's) here: in XLS,
  // division by zero returns the greatest value of that type, so 255 for an
  // unsigned byte, and either -128 or 127 for a signed one.
  // Thus, a little more work is necessary to emit LLVM IR matching the XLS
  // div op than just IRBuilder::Create[SU]Div().
  int type_width = rhs->getType()->getIntegerBitWidth();
  llvm::Value* zero = llvm::ConstantInt::get(rhs->getType(), 0);
  llvm::Value* rhs_eq_zero = builder->CreateICmpEQ(rhs, zero);
  llvm::Value* lhs_gt_zero = builder->CreateICmpSGT(lhs, zero);

  // If rhs is zero, make LHS = the max/min value and the RHS 1,
  // rather than introducing a proper conditional.
  rhs = builder->CreateSelect(rhs_eq_zero,
                              llvm::ConstantInt::get(rhs->getType(), 1), rhs);
  if (is_signed) {
    llvm::Value* max_value =
        type_converter
            ->ToLlvmConstant(rhs->getType(), Value(Bits::MaxSigned(type_width)))
            .value();
    llvm::Value* min_value =
        type_converter
            ->ToLlvmConstant(rhs->getType(), Value(Bits::MinSigned(type_width)))
            .value();

    lhs = builder->CreateSelect(
        rhs_eq_zero, builder->CreateSelect(lhs_gt_zero, max_value, min_value),
        lhs);
    return builder->CreateSDiv(lhs, rhs);
  }

  lhs = builder->CreateSelect(
      rhs_eq_zero,
      type_converter
          ->ToLlvmConstant(rhs->getType(), Value(Bits::AllOnes(type_width)))
          .value(),
      lhs);
  return builder->CreateUDiv(lhs, rhs);
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

// ORs together all elements in the two given values, be they Bits, Arrays, or
// Tuples.
llvm::Value* CreateAggregateOr(llvm::Value* lhs, llvm::Value* rhs,
                               llvm::IRBuilder<>* builder) {
  llvm::Type* arg_type = lhs->getType();
  if (arg_type->isIntegerTy()) {
    return builder->CreateOr(lhs, rhs);
  }

  llvm::Value* result = LlvmTypeConverter::ZeroOfType(arg_type);
  int num_elements = arg_type->isArrayTy() ? arg_type->getArrayNumElements()
                                           : arg_type->getNumContainedTypes();
  for (uint32_t i = 0; i < num_elements; ++i) {
    llvm::Value* iter_result =
        CreateAggregateOr(builder->CreateExtractValue(lhs, {i}),
                          builder->CreateExtractValue(rhs, {i}), builder);
    result = builder->CreateInsertValue(result, iter_result, {i});
  }

  return result;
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

// Returns the result of indexing into 'array' using the scalar index value
// 'index'. 'array_size' is the number of elements in the array.
absl::StatusOr<llvm::Value*> IndexIntoArray(llvm::Value* array,
                                            llvm::Value* index,
                                            int64_t array_size,
                                            LlvmTypeConverter* type_converter,
                                            llvm::IRBuilder<>* builder) {
  int64_t index_width = index->getType()->getIntegerBitWidth();

  // Check for out-of-bounds access. If the index is out of bounds it is set to
  // the maximum index value.
  int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
  int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
  llvm::Value* array_size_comparison_bitwidth = llvm::ConstantInt::get(
      llvm::Type::getIntNTy(builder->getContext(), comparison_bitwidth),
      array_size);
  llvm::Value* index_value_comparison_bitwidth = builder->CreateZExt(
      index, llvm::Type::getIntNTy(builder->getContext(), comparison_bitwidth));
  llvm::Value* is_index_inbounds = builder->CreateICmpULT(
      index_value_comparison_bitwidth, array_size_comparison_bitwidth);
  llvm::Value* inbounds_index = builder->CreateSelect(
      is_index_inbounds, index,
      llvm::ConstantInt::get(index->getType(), array_size - 1));

  // Our IR does not use negative indices, so we add a
  // zero MSb to prevent LLVM from interpreting this as such.
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(builder->getContext()), 0),
      builder->CreateZExt(inbounds_index,
                          type_converter->GetLlvmBitsType(index_width + 1))};

  llvm::Type* array_type = array->getType();
  // Ideally, we'd use IRBuilder::CreateExtractValue here, but that requires
  // constant indices. Since there's no other way to extract a value from an
  // aggregate, we're left with storing the value in a temporary alloca and
  // using that pointer to extract the value.
  llvm::Value* alloca = builder->CreateAlloca(array_type);
  builder->CreateStore(array, alloca);

  llvm::Type* element_type = array_type->getArrayElementType();
  llvm::Value* gep = builder->CreateGEP(array_type, alloca, gep_indices);
  return builder->CreateLoad(element_type, gep);
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
  Value ir_value = runtime->UnpackBuffer(value, type, /*unpoison=*/true);
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
void RecordTrace(std::string* buffer, xls::InterpreterEvents* events) {
  events->trace_msgs.push_back(*buffer);
  delete buffer;
}

// Build the LLVM IR to invoke the callback that records traces.
absl::Status InvokeRecordTraceCallback(llvm::IRBuilder<>* builder,
                                       llvm::Value* buffer_ptr,
                                       llvm::Value* interpreter_events_ptr) {
  llvm::Type* ptr_type = llvm::PointerType::get(builder->getContext(), 0);

  std::vector<llvm::Type*> params = {buffer_ptr->getType(), ptr_type};

  llvm::Type* void_type = llvm::Type::getVoidTy(builder->getContext());

  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  std::vector<llvm::Value*> args = {buffer_ptr, interpreter_events_ptr};

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

absl::StatusOr<llvm::Function*> CreateLlvmFunction(
    absl::string_view name, llvm::FunctionType* function_type,
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
  // void f(operand_ptr_0, ..., operand_ptr_n,
  //        output_ptr_0, ..., output_ptr_m)
  //
  // void f(void* operand_ptr_0, ... , void* operand_ptr_n,
  //        void* output_ptr_0,  ... , void* output_ptr_m,
  //        void* global_inputs, void* global_outputs, void* tmp_buffer,
  //        void* events, void* user_data, void* runtime)
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
  static absl::StatusOr<NodeIrContext> Create(
      Node* node, absl::Span<const std::string> operand_names,
      int64_t output_arg_count, bool include_wrapper_args,
      JitBuilderContext& jit_context);

  // Completes the LLVM function by adding a return statement with the given
  // result (or pointer to result). If `exit_builder` is specified then it is
  // used to build the return statement. Otherwise `entry_builder()` is used.
  void FinalizeWithValue(llvm::Value* result,
                         std::optional<llvm::IRBuilder<>*> exit_builder);
  void FinalizeWithPointerToValue(
      llvm::Value* result_buffer,
      std::optional<llvm::IRBuilder<>*> exit_builder);

  Node* node() const { return node_; }
  LlvmTypeConverter& type_converter() const {
    return orc_jit_.GetTypeConverter();
  }

  // Returns true if the underlying llvm::Function takes the top-level metadata
  // arguments (inputs, outputs, temp buffer, events, etc).
  bool has_metadata_args() const { return has_metadata_args_; }

  // The LLVM function that the generated code for the node is placed into,
  llvm::Function* llvm_function() const { return llvm_function_; }

  // Returns the IR builder to use for building code for this XLS node.
  llvm::IRBuilder<>& entry_builder() const { return *entry_builder_; }

  // Loads the given operand value from the pointer passed in as the respective
  // operand.
  llvm::Value* LoadOperand(int64_t i) const;

  // Returns the operand pointer arguments.
  absl::Span<llvm::Value* const> GetOperandPtrs() const {
    return operand_ptrs_;
  }

  // Returns the `i-th` operand pointer argument.
  llvm::Value* GetOperandPtr(int64_t i) const { return operand_ptrs_.at(i); }

  // Returns the output pointer arguments.
  absl::Span<llvm::Value* const> GetOutputPtrs() const { return output_ptrs_; }

  // Returns the `i-th` operand pointer argument.
  llvm::Value* GetOutputPtr(int64_t i) const { return output_ptrs_.at(i); }

  // Get one of the metadata arguments. CHECK fails the function was created
  // without metadata arguments.
  llvm::Value* GetInputPtrsArg() const {
    XLS_CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 6);
  }
  llvm::Value* GetOutputPtrsArg() const {
    XLS_CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 5);
  }
  llvm::Value* GetTempBufferArg() const {
    XLS_CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 4);
  }
  llvm::Value* GetInterpreterEventsArg() const {
    XLS_CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 3);
  }
  llvm::Value* GetUserDataArg() const {
    XLS_CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 2);
  }
  llvm::Value* GetJitRuntimeArg() const {
    XLS_CHECK(has_metadata_args_);
    return llvm_function_->getArg(llvm_function_->arg_size() - 1);
  }

 private:
  NodeIrContext(Node* node, bool has_metadata_args, OrcJit& orc_jit)
      : node_(node), has_metadata_args_(has_metadata_args), orc_jit_(orc_jit) {}

  Node* node_;
  bool has_metadata_args_;
  OrcJit& orc_jit_;

  llvm::Function* llvm_function_;
  std::unique_ptr<llvm::IRBuilder<>> entry_builder_;
  std::vector<llvm::Value*> operand_ptrs_;
  std::vector<llvm::Value*> output_ptrs_;
};

absl::StatusOr<NodeIrContext> NodeIrContext::Create(
    Node* node, absl::Span<const std::string> operand_names,
    int64_t output_arg_count, bool include_wrapper_args,
    JitBuilderContext& jit_context) {
  XLS_RET_CHECK_GT(output_arg_count, 0);
  NodeIrContext nc(node, include_wrapper_args, jit_context.orc_jit());
  nc.node_ = node;
  int64_t param_count = node->operand_count() + output_arg_count;
  if (include_wrapper_args) {
    param_count += 6;
  }
  std::vector<llvm::Type*> param_types(
      param_count,
      llvm::PointerType::get(jit_context.module()->getContext(), 0));
  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(jit_context.module()->getContext()), param_types,
      /*isVarArg=*/false);
  std::string function_name = absl::StrFormat(
      "__%s_%s_%d", node->function_base()->name(), node->GetName(), node->id());
  XLS_ASSIGN_OR_RETURN(
      nc.llvm_function_,
      CreateLlvmFunction(function_name, function_type, jit_context.module()));

  // Mark as private so function can be deleted after inlining.
  nc.llvm_function_->setLinkage(llvm::GlobalValue::PrivateLinkage);

  XLS_RET_CHECK_EQ(operand_names.size(), node->operand_count());
  int64_t arg_no = 0;
  for (const std::string& name : operand_names) {
    nc.llvm_function_->getArg(arg_no++)->setName(absl::StrCat(name, "_ptr"));
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
    nc.llvm_function_->getArg(arg_no++)->setName("user_data");
    nc.llvm_function_->getArg(arg_no++)->setName("jit_runtime");
  }

  nc.entry_builder_ =
      std::make_unique<llvm::IRBuilder<>>(llvm::BasicBlock::Create(
          jit_context.module()->getContext(), "entry", nc.llvm_function_,
          /*InsertBefore=*/nullptr));

  for (int64_t i = 0; i < node->operand_count(); ++i) {
    llvm::Value* operand_ptr = nc.llvm_function_->getArg(i);
    nc.operand_ptrs_.push_back(operand_ptr);
  }
  for (int64_t i = 0; i < output_arg_count; ++i) {
    nc.output_ptrs_.push_back(
        nc.llvm_function_->getArg(node->operand_count() + i));
  }
  return nc;
}

llvm::Value* NodeIrContext::LoadOperand(int64_t i) const {
  Node* operand = node()->operand(i);
  llvm::Type* operand_type =
      type_converter().ConvertToLlvmType(operand->GetType());
  llvm::Value* operand_ptr = llvm_function_->getArg(i);
  return entry_builder().CreateLoad(operand_type, operand_ptr);
}

void NodeIrContext::FinalizeWithValue(
    llvm::Value* result, std::optional<llvm::IRBuilder<>*> exit_builder) {
  llvm::IRBuilder<>* b =
      exit_builder.has_value() ? exit_builder.value() : &entry_builder();
  result = type_converter().ClearPaddingBits(result, node()->GetType(), *b);
  if (GetOutputPtrs().empty()) {
    b->CreateRetVoid();
    return;
  }
  b->CreateStore(result, GetOutputPtr(0));
  return FinalizeWithPointerToValue(GetOutputPtr(0), exit_builder);
}

void NodeIrContext::FinalizeWithPointerToValue(
    llvm::Value* result_buffer,
    std::optional<llvm::IRBuilder<>*> exit_builder) {
  llvm::IRBuilder<>* b =
      exit_builder.has_value() ? exit_builder.value() : &entry_builder();
  for (int64_t i = 0; i < output_ptrs_.size(); ++i) {
    if (output_ptrs_[i] != result_buffer) {
      LlvmMemcpy(output_ptrs_[i], result_buffer,
                 type_converter().GetTypeByteSize(node()->GetType()), *b);
    }
  }
  b->CreateRetVoid();
}

// Visitor to construct and LLVM function implementing an XLS IR node.
class IrBuilderVisitor : public DfsVisitorWithDefault {
 public:
  //  `output_arg_count` is the number of output arguments for the LLVM function
  IrBuilderVisitor(int64_t output_arg_count, JitBuilderContext& jit_context)
      : output_arg_count_(output_arg_count), jit_context_(jit_context) {}

  NodeIrContext ConsumeNodeIrContext() { return std::move(*node_context_); }

  absl::Status DefaultHandler(Node* node) override;

  absl::Status HandleAdd(BinOp* binop) override;
  absl::Status HandleAndReduce(BitwiseReductionOp* op) override;
  absl::Status HandleAfterAll(AfterAll* after_all) override;
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
  LlvmTypeConverter* type_converter() {
    return &jit_context_.orc_jit().GetTypeConverter();
  }

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
      std::optional<llvm::IRBuilder<>*> exit_builder = std::nullopt);
  absl::Status FinalizeNodeIrContextWithPointerToValue(
      NodeIrContext&& node_context, llvm::Value* result_buffer,
      std::optional<llvm::IRBuilder<>*> exit_builder = std::nullopt);

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
      llvm::Value* events, llvm::Value* user_data, llvm::Value* runtime,
      llvm::IRBuilder<>& builder);

  absl::StatusOr<llvm::Value*> InvokeRecvCallback(llvm::IRBuilder<>* builder,
                                                  JitChannelQueue* queue,
                                                  Receive* receive,
                                                  llvm::Value* output_ptr,
                                                  llvm::Value* user_data);

  absl::Status InvokeSendCallback(llvm::IRBuilder<>* builder,
                                  JitChannelQueue* queue, Send* send,
                                  llvm::Value* send_data_ptr,
                                  llvm::Value* user_data);

  int64_t output_arg_count_;
  JitBuilderContext& jit_context_;
  std::optional<NodeIrContext> node_context_;
};

absl::Status IrBuilderVisitor::DefaultHandler(Node* node) {
  return absl::UnimplementedError(
      absl::StrCat("Unhandled node: ", node->ToString()));
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

  // TODO(meheff): 2022/09/09 Rather than using insertvalue memcpy the
  // elements into place in the output buffer.
  llvm::Value* result = LlvmTypeConverter::ZeroOfType(array_type);
  for (uint32_t i = 0; i < array->size(); ++i) {
    result = b.CreateInsertValue(result, node_context.LoadOperand(i), {i});
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
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
    if (absl::holds_alternative<std::string>(step)) {
      XLS_RETURN_IF_ERROR(InvokeStringStepCallback(
          &print_builder, absl::get<std::string>(step), buffer_ptr));
    } else {
      xls::Node* o = trace_op->operand(operand_index);
      llvm::Value* operand = node_context.LoadOperand(operand_index);
      llvm::AllocaInst* alloca = print_builder.CreateAlloca(operand->getType());
      print_builder.CreateStore(operand, alloca);
      // The way our format strings are currently formed we implicitly refer to
      // the next operand after formatting this one.
      operand_index += 1;
      XLS_RETURN_IF_ERROR(InvokeFormatStepCallback(
          &print_builder, absl::get<FormatPreference>(step), o->GetType(),
          alloca, buffer_ptr, jit_runtime_ptr));
    }
  }

  XLS_RETURN_IF_ERROR(
      InvokeRecordTraceCallback(&print_builder, buffer_ptr, events_ptr));

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

  Type* element_type = index->array()->GetType();

  // TODO(meheff): 2022/09/09 Replace this with a single memcpy.
  llvm::Value* element = node_context.LoadOperand(0);
  // Index operands start at 1.
  for (int64_t i = 1; i < index->operand_count(); ++i) {
    llvm::Value* index_value = node_context.LoadOperand(i);
    XLS_ASSIGN_OR_RETURN(element,
                         IndexIntoArray(element, index_value,
                                        element_type->AsArrayOrDie()->size(),
                                        type_converter(), &b));
    element_type = element_type->AsArrayOrDie()->element_type();
  }
  return FinalizeNodeIrContextWithValue(std::move(node_context), element);
}

absl::Status IrBuilderVisitor::HandleArraySlice(ArraySlice* slice) {
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(slice, {"array", "start"}));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Value* array = node_context.LoadOperand(0);
  llvm::Value* start = node_context.LoadOperand(1);
  int64_t width = slice->width();

  // This overestimates the number of bits needed but in all practical
  // situations it should be fine. The only exception to that is if some code
  // uses a 64 bit index but doesn't actually make full use of that range, then
  // this will possibly push us over a performance cliff.
  int64_t index_bits = start->getType()->getIntegerBitWidth() +
                       Bits::MinBitCountSigned(width) + 1;
  llvm::Type* index_type = type_converter()->ConvertToLlvmType(
      slice->package()->GetBitsType(index_bits));
  llvm::Type* result_type =
      type_converter()->ConvertToLlvmType(slice->GetType());
  llvm::Type* result_element_type = type_converter()->ConvertToLlvmType(
      slice->GetType()->AsArrayOrDie()->element_type());
  llvm::Value* alloca = b.CreateAlloca(result_type, 0, "alloca");
  llvm::Value* start_big = b.CreateZExt(start, index_type, "start_big");

  for (int64_t i = 0; i < width; i++) {
    llvm::Value* index =
        b.CreateAdd(start_big, llvm::ConstantInt::get(index_type, i), "index");
    // TODO(meheff): 2022/09/09 Replace with memcpys.
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * value,
        IndexIntoArray(array, index,
                       slice->array()->GetType()->AsArrayOrDie()->size(),
                       type_converter(), &b));
    std::vector<llvm::Value*> gep_indices = {
        llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), i)};
    llvm::Value* gep = b.CreateGEP(result_element_type, alloca, gep_indices);
    b.CreateStore(value, gep);
  }

  llvm::Value* sliced_array = b.CreateLoad(result_type, alloca);

  return FinalizeNodeIrContextWithValue(std::move(node_context), sliced_array);
}

absl::Status IrBuilderVisitor::HandleArrayUpdate(ArrayUpdate* update) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          update,
          ConcatVectors({"array", "update_value"},
                        NumberedStrings("index", update->indices().size()))));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  // TODO(meheff): 2022/09/09 Replace with memcpy of whole array, followed by
  // memcpy of new element.
  llvm::Value* original_array = node_context.LoadOperand(0);
  llvm::Value* update_value = node_context.LoadOperand(1);
  std::vector<llvm::Value*> indices;
  for (int64_t i = 2; i < update->operand_count(); ++i) {
    indices.push_back(node_context.LoadOperand(i));
  }

  if (indices.empty()) {
    // An empty index replaces the entire array value.
    return FinalizeNodeIrContextWithValue(std::move(node_context),
                                          update_value);
  }

  llvm::Type* array_type = original_array->getType();
  llvm::AllocaInst* alloca = b.CreateAlloca(array_type);
  b.CreateStore(original_array, alloca);

  Type* element_type = update->array_to_update()->GetType();
  std::vector<llvm::Value*> gep_indices = {
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx()), 0)};
  llvm::Value* is_inbounds = b.getTrue();
  for (llvm::Value* index : indices) {
    int64_t index_bitwidth = index->getType()->getIntegerBitWidth();
    int64_t comparison_bitwidth = std::max(index_bitwidth, int64_t{64});
    llvm::Value* array_size_comparison_bitwidth = llvm::ConstantInt::get(
        b.getIntNTy(comparison_bitwidth), element_type->AsArrayOrDie()->size());
    llvm::Value* index_value_comparison_bitwidth =
        b.CreateZExt(index, llvm::Type::getIntNTy(ctx(), comparison_bitwidth));
    llvm::Value* is_index_inbounds =
        b.CreateICmpULT(index_value_comparison_bitwidth,
                        array_size_comparison_bitwidth, "idx_is_inbounds");

    gep_indices.push_back(index_value_comparison_bitwidth);
    is_inbounds = b.CreateAnd(is_inbounds, is_index_inbounds, "inbounds");

    element_type = element_type->AsArrayOrDie()->element_type();
  }

  // Create the join block which occurs after the conditional block (conditioned
  // on whether the index is inbounds).
  llvm::BasicBlock* join_block =
      llvm::BasicBlock::Create(ctx(), absl::StrCat(update->GetName(), "_join"),
                               node_context.llvm_function());

  // Create the inbounds block and fill with a store to the array elemnt.
  llvm::BasicBlock* inbounds_block = llvm::BasicBlock::Create(
      ctx(), absl::StrCat(update->GetName(), "_inbounds"),
      node_context.llvm_function(),
      /*InsertBefore=*/join_block);
  llvm::IRBuilder<> inbounds_builder(inbounds_block);
  llvm::Value* gep =
      inbounds_builder.CreateGEP(array_type, alloca, gep_indices);
  inbounds_builder.CreateStore(update_value, gep);
  inbounds_builder.CreateBr(join_block);

  // Create a conditional branch using the original builder (end of the BB
  // before the if/then).
  b.CreateCondBr(is_inbounds, inbounds_block, join_block);

  // Create a new BB at the join point.
  auto exit_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
  llvm::Value* update_array = exit_builder->CreateLoad(array_type, alloca);

  return FinalizeNodeIrContextWithValue(std::move(node_context), update_array,
                                        exit_builder.get());
}

absl::Status IrBuilderVisitor::HandleArrayConcat(ArrayConcat* concat) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(concat,
                       NumberedStrings("operand", concat->operand_count())));
  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* array_type =
      type_converter()->ConvertToLlvmType(concat->GetType());

  llvm::Value* result = LlvmTypeConverter::ZeroOfType(array_type);

  // TODO(meheff): 2022/09/09 Replace with memcpys of the operand arrays into
  // there location in the output array.
  int64_t result_index = 0;
  int64_t result_elements = array_type->getArrayNumElements();
  for (int64_t i = 0; i < concat->operand_count(); ++i) {
    llvm::Value* array = node_context.LoadOperand(i);
    llvm::Type* array_type = array->getType();

    int64_t element_count = array_type->getArrayNumElements();
    for (int64_t j = 0; j < element_count; ++j) {
      llvm::Value* element =
          b.CreateExtractValue(array, {static_cast<uint32_t>(j)});

      if (result_index >= result_elements) {
        return absl::InternalError(absl::StrFormat(
            "array-concat %s result and source have mismatched number of "
            "elements - expected %d",
            concat->ToString(), result_elements));
      }

      result = b.CreateInsertValue(result, element,
                                   {static_cast<uint32_t>(result_index)});
      ++result_index;
    }
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
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
  llvm::IRBuilder<>& b = node_context.entry_builder();

  XLS_ASSIGN_OR_RETURN(llvm::Function * body, GetFunction(counted_for->body()));

  llvm::Type* index_type = type_converter()->ConvertToLlvmType(
      counted_for->body()->param(0)->GetType());
  llvm::Type* state_type = type_converter()->ConvertToLlvmType(
      counted_for->initial_value()->GetType());

  llvm::Value* index_buffer = b.CreateAlloca(index_type);
  llvm::Value* loop_state_buffer = b.CreateAlloca(state_type);
  std::vector<llvm::Value*> invariant_arg_buffers;
  for (int64_t i = 1; i < counted_for->operand_count(); ++i) {
    invariant_arg_buffers.push_back(node_context.GetOperandPtr(i));
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

  llvm::Value* next_state = node_context.LoadOperand(0);
  b.CreateStore(next_state, loop_state_buffer);
  // TODO(meheff): 2022/09/09 Generate an LLVM loop rather than unrolling.
  for (int i = 0; i < counted_for->trip_count(); ++i) {
    b.CreateStore(llvm::ConstantInt::get(index_type, i * counted_for->stride()),
                  index_buffer);
    XLS_RETURN_IF_ERROR(CallFunction(body, input_arg_ptrs, {next_state_buffer},
                                     node_context.GetTempBufferArg(),
                                     node_context.GetInterpreterEventsArg(),
                                     node_context.GetUserDataArg(),
                                     node_context.GetJitRuntimeArg(), b)
                            .status());
    // TODO(meheff): 2022/09/09 Rather than loading the state and storing it in
    // the state buffer for the next iteration, simply swap the state and
    // next-state buffer pointers passed to the loop body function.
    next_state = b.CreateLoad(state_type, next_state_buffer);
    b.CreateStore(next_state, loop_state_buffer);
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), next_state);
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
  // If the input value is greater than this op's width, then return 0.
  // In that case, the shl will produce a poison value, but it'll be unused.
  llvm::Value* cast_input = b.CreateZExt(input, result_type);
  llvm::Value* overflow = b.CreateICmpUGE(
      cast_input, llvm::ConstantInt::get(result_type, decode->width()));
  llvm::Value* result = b.CreateSelect(
      overflow, llvm::ConstantInt::get(result_type, 0),
      b.CreateShl(llvm::ConstantInt::get(result_type, 1), cast_input));

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
  //  auto invariant_args = node_context.LoadOperands().subspan(3);

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
                   node_context.GetUserDataArg(),
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
  llvm::Value* result = llvm::ConstantInt::get(result_type, 0);

  llvm::Value* result_zero = llvm::ConstantInt::get(result_type, 0);

  // For each bit in the input, if it's set, bitwise-OR its [numeric] value
  // with the result.
  for (int i = 0; i < input_type->getIntegerBitWidth(); ++i) {
    llvm::Value* bit_set =
        b.CreateICmpEQ(b.CreateAnd(input, input_one), input_one);

    // Chained select, i.e., a = (b ? c : (d ? e : (...))), etc.
    llvm::Value* or_value = b.CreateSelect(
        bit_set, llvm::ConstantInt::get(result_type, i), result_zero);
    result = b.CreateOr(result, or_value);

    input = b.CreateLShr(input, input_one);
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
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
  XLS_RETURN_IF_ERROR(CallFunction(function, node_context.GetOperandPtrs(),
                                   {output_buffer},
                                   node_context.GetTempBufferArg(),
                                   node_context.GetInterpreterEventsArg(),
                                   node_context.GetUserDataArg(),
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
  llvm::IRBuilder<>& b = node_context.entry_builder();

  XLS_ASSIGN_OR_RETURN(llvm::Function * to_apply, GetFunction(map->to_apply()));

  llvm::Value* input = node_context.LoadOperand(0);
  llvm::Type* input_array_type = input->getType();
  llvm::Type* input_element_type = type_converter()->ConvertToLlvmType(
      map->operand(0)->GetType()->AsArrayOrDie()->element_type());

  llvm::Type* output_array_type =
      type_converter()->ConvertToLlvmType(map->GetType());
  llvm::Type* output_element_type = type_converter()->ConvertToLlvmType(
      map->to_apply()->return_value()->GetType());
  llvm::Value* result = LlvmTypeConverter::ZeroOfType(output_array_type);
  result->setName("init_result");

  // TODO(meheff): 2022/09/09 Avoid allocas and loading any values here. Map
  // should be possible to do simply by passing pointers on to the to-apply
  // function.
  llvm::Value* output_buffer =
      b.CreateAlloca(output_element_type,
                     /*ArraySize=*/nullptr, "output_buffer");
  llvm::Value* input_buffer =
      b.CreateAlloca(input_element_type, /*ArraySize=*/nullptr, "input_buffer");

  // TODO(meheff): 2022/09/09 Use an LLVM loop.
  for (uint32_t i = 0; i < input_array_type->getArrayNumElements(); ++i) {
    llvm::Value* input_element = b.CreateExtractValue(input, {i});
    input_element->setName(absl::StrFormat("input_element_%d", i));
    b.CreateStore(input_element, input_buffer);
    std::vector<llvm::Value*> inputs = {input_buffer};
    std::vector<llvm::Value*> outputs = {output_buffer};
    XLS_RETURN_IF_ERROR(CallFunction(to_apply, inputs, outputs,
                                     node_context.GetTempBufferArg(),
                                     node_context.GetInterpreterEventsArg(),
                                     node_context.GetUserDataArg(),
                                     node_context.GetJitRuntimeArg(), b)
                            .status());
    llvm::Value* iter_result = b.CreateLoad(output_element_type, output_buffer);
    result = b.CreateInsertValue(result, iter_result, {i});
    result->setName(absl::StrFormat("result_%d", i));
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
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

absl::Status IrBuilderVisitor::HandleSMulp(PartialProductOp* mul) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(mul, NumberedStrings("operand", mul->operand_count())));

  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* result_element_type = type_converter()->ConvertToLlvmType(
      mul->GetType()->AsTupleOrDie()->element_type(0));
  llvm::Type* tuple_type = type_converter()->ConvertToLlvmType(mul->GetType());
  llvm::Value* result = LlvmTypeConverter::ZeroOfType(tuple_type);

  llvm::Value* lhs = type_converter()->AsSignedValue(
      node_context.LoadOperand(0), mul->operand(0)->GetType(),
      node_context.entry_builder());
  llvm::Value* rhs = type_converter()->AsSignedValue(
      node_context.LoadOperand(1), mul->operand(1)->GetType(),
      node_context.entry_builder());

  result = b.CreateInsertValue(
      result,
      b.CreateMul(b.CreateIntCast(lhs, result_element_type, /*isSigned=*/true),
                  b.CreateIntCast(rhs, result_element_type, /*isSigned=*/true)),
      {1});

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandleUMulp(PartialProductOp* mul) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(mul, NumberedStrings("operand", mul->operand_count())));

  llvm::IRBuilder<>& b = node_context.entry_builder();

  llvm::Type* result_element_type = type_converter()->ConvertToLlvmType(
      mul->GetType()->AsTupleOrDie()->element_type(0));
  llvm::Type* tuple_type = type_converter()->ConvertToLlvmType(mul->GetType());
  llvm::Value* result = LlvmTypeConverter::ZeroOfType(tuple_type);

  result = b.CreateInsertValue(
      result,
      b.CreateMul(
          b.CreateIntCast(node_context.LoadOperand(0), result_element_type,
                          /*isSigned=*/false),
          b.CreateIntCast(node_context.LoadOperand(1), result_element_type,
                          /*isSigned=*/false)),
      {1});

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
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

absl::Status IrBuilderVisitor::HandleOneHotSel(OneHotSelect* sel) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          sel, ConcatVectors({"selector"},
                             NumberedStrings("case", sel->cases().size()))));
  // TODO(meheff): 2022/09/09 Only load operands which are actually selected.
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* selector = node_context.LoadOperand(0);
  std::vector<llvm::Value*> cases;
  for (int64_t i = 1; i < sel->operand_count(); ++i) {
    cases.push_back(node_context.LoadOperand(i));
  }
  llvm::Type* input_type = cases.front()->getType();

  llvm::Value* result;
  result = LlvmTypeConverter::ZeroOfType(input_type);

  llvm::Value* typed_zero = LlvmTypeConverter::ZeroOfType(input_type);
  llvm::Value* llvm_one = llvm::ConstantInt::get(selector->getType(), 1);

  for (llvm::Value* cse : cases) {
    // Extract the current selector bit & see if set (CreateSelect requires an
    // i1 argument, or we could directly use the AND result.
    llvm::Value* is_hot =
        b.CreateICmpEQ(b.CreateAnd(selector, llvm_one), llvm_one);

    // OR with zero might be slower than doing an if/else construct - if
    // it turns out to be performance-critical, we can update it.
    llvm::Value* or_value = b.CreateSelect(is_hot, cse, typed_zero);
    result = CreateAggregateOr(result, or_value, &b);
    selector = b.CreateLShr(selector, llvm_one);
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
}

absl::Status IrBuilderVisitor::HandlePrioritySel(PrioritySelect* sel) {
  XLS_ASSIGN_OR_RETURN(
      NodeIrContext node_context,
      NewNodeIrContext(
          sel, ConcatVectors({"selector"},
                             NumberedStrings("case", sel->cases().size()))));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* selector = node_context.LoadOperand(0);

  // TODO(meheff): 2022/09/09 Avoid loading cases and selecting among them.
  // Instead select the appropriate case pointer and call
  // FinalizeNodeIrContextWithPointer.

  std::vector<llvm::Value*> cases;
  for (int64_t i = 1; i < sel->operand_count(); ++i) {
    cases.push_back(node_context.LoadOperand(i));
  }
  llvm::Type* input_type = cases.front()->getType();

  llvm::Value* typed_zero = LlvmTypeConverter::ZeroOfType(input_type);
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
  return FinalizeNodeIrContextWithValue(std::move(node_context), llvm_sel);
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
        return EmitDiv(lhs, rhs, /*is_signed=*/true, type_converter(), &b);
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

  // TODO(meheff): 2022/09/09 Avoid loading cases and selecting among them.
  // Instead select the appropriate case pointer and call
  // FinalizeNodeIrContextWithPointer.

  // Sel is implemented by a cascading series of select ops, e.g.,
  // selector == 0 ? cases[0] : selector == 1 ? cases[1] : selector == 2 ? ...
  llvm::Value* selector = node_context.LoadOperand(0);
  llvm::Value* llvm_sel =
      sel->default_value() ? node_context.LoadOperand(sel->operand_count() - 1)
                           : nullptr;
  for (int i = sel->cases().size() - 1; i >= 0; i--) {
    llvm::Value* llvm_case = node_context.LoadOperand(i + 1);
    if (llvm_sel == nullptr) {
      // The last element in the select tree isn't a sel, but an actual value.
      llvm_sel = llvm_case;
    } else {
      llvm::Value* index = llvm::ConstantInt::get(selector->getType(), i);
      llvm::Value* cmp = b.CreateICmpEQ(selector, index);
      llvm_sel = b.CreateSelect(cmp, llvm_case, llvm_sel);
    }
  }
  return FinalizeNodeIrContextWithValue(std::move(node_context), llvm_sel);
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

  // TODO(meheff): 2022/09/09 Avoid loading elements and using
  // insertvalue. Instead memcpy each element into place.
  llvm::Value* result = LlvmTypeConverter::ZeroOfType(tuple_type);
  for (uint32_t i = 0; i < tuple->operand_count(); ++i) {
    if (tuple->operand(i)->GetType()->GetFlatBitCount() == 0) {
      continue;
    }
    result = b.CreateInsertValue(result, node_context.LoadOperand(i), {i});
  }

  return FinalizeNodeIrContextWithValue(std::move(node_context), result);
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
        return EmitDiv(lhs, rhs, /*is_signed=*/false, type_converter(), &b);
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
                               include_wrapper_args, jit_context_);
}

absl::Status IrBuilderVisitor::FinalizeNodeIrContextWithValue(
    NodeIrContext&& node_context, llvm::Value* result,
    std::optional<llvm::IRBuilder<>*> exit_builder) {
  node_context.FinalizeWithValue(result, exit_builder);
  node_context_.emplace(std::move(node_context));
  return absl::OkStatus();
}

absl::Status IrBuilderVisitor::FinalizeNodeIrContextWithPointerToValue(
    NodeIrContext&& node_context, llvm::Value* result_buffer,
    std::optional<llvm::IRBuilder<>*> exit_builder) {
  node_context.FinalizeWithPointerToValue(result_buffer, exit_builder);
  node_context_.emplace(std::move(node_context));
  return absl::OkStatus();
}

absl::StatusOr<llvm::Value*> IrBuilderVisitor::CallFunction(
    llvm::Function* f, absl::Span<llvm::Value* const> inputs,
    absl::Span<llvm::Value* const> outputs, llvm::Value* temp_buffer,
    llvm::Value* events, llvm::Value* user_data, llvm::Value* runtime,
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

  std::vector<llvm::Value*> args = {input_arg_array, output_arg_array,
                                    temp_buffer,     events,
                                    user_data,       runtime};
  return builder.CreateCall(f, args);
}

absl::StatusOr<llvm::Value*> IrBuilderVisitor::InvokeRecvCallback(
    llvm::IRBuilder<>* builder, JitChannelQueue* queue, Receive* receive,
    llvm::Value* output_ptr, llvm::Value* user_data) {
  llvm::Type* bool_type = llvm::Type::getInt1Ty(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  llvm::Type* ptr_type = llvm::PointerType::get(ctx(), 0);

  // Call the user-provided function of type ProcJit::RecvFnT to receive the
  // value.
  std::vector<llvm::Type*> params = {int64_type, int64_type, int8_ptr_type,
                                     int64_type, ptr_type};
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(bool_type, params, /*isVarArg=*/false);

  // recv_type is the full type of the receive.
  //   1. If blocking, then it is a tuple of (token, payload).
  //   2. If non-blocking, then it is a tuple of (token, payload, bool).
  llvm::Type* recv_type =
      type_converter()->ConvertToLlvmType(receive->GetType());

  // recv_payload_bytes is just the size of the payload.
  //
  // As token is zero size, it can also be considered the size of the
  // token + payload.
  int64_t recv_payload_bytes =
      type_converter()->GetTypeByteSize(receive->GetPayloadType());

  // Call the user-provided receive function.
  std::vector<llvm::Value*> args = {
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(receive)),
      output_ptr,
      llvm::ConstantInt::get(int64_type, recv_payload_bytes),
      user_data,
  };

  XLS_RET_CHECK(jit_context_.recv_fn().has_value());
  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()),
      absl::bit_cast<uint64_t>(jit_context_.recv_fn().value()));
  llvm::Value* fn_ptr =
      builder->CreateIntToPtr(fn_addr, llvm::PointerType::get(fn_type, 0));
  llvm::Value* data_valid = builder->CreateCall(fn_type, fn_ptr, args);

  // Load the reveive data from the bounce buffer.
  llvm::Value* data = builder->CreateLoad(recv_type, output_ptr);

  if (receive->is_blocking()) {
    return data;
  }

  // For non-blocking receives, add data_valid as the last entry in the
  // return tuple.
  return builder->CreateInsertValue(data, data_valid, {2});
}

absl::Status IrBuilderVisitor::HandleReceive(Receive* recv) {
  std::vector<std::string> operand_names = {"tkn"};
  if (recv->predicate().has_value()) {
    operand_names.push_back("predicate");
  }
  XLS_ASSIGN_OR_RETURN(NodeIrContext node_context,
                       NewNodeIrContext(recv, operand_names,
                                        /*include_wrapper_args=*/true));
  llvm::IRBuilder<>& b = node_context.entry_builder();
  llvm::Value* user_data = node_context.GetUserDataArg();

  XLS_RET_CHECK(jit_context_.queue_manager().has_value());
  XLS_ASSIGN_OR_RETURN(
      JitChannelQueue * queue,
      jit_context_.queue_manager().value()->GetQueueById(recv->channel_id()));
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
    XLS_ASSIGN_OR_RETURN(
        llvm::Value * true_result,
        InvokeRecvCallback(&true_builder, queue, recv,
                           node_context.GetOutputPtr(0), user_data));
    true_builder.CreateBr(join_block);

    // And the same for a false predicate - this will return an empty/zero
    // value. Creating an empty struct emits ops, so it needs a builder.
    llvm::BasicBlock* false_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(recv->GetName(), "_false"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    llvm::Type* result_type =
        type_converter()->ConvertToLlvmType(recv->GetType());
    llvm::Value* false_result = LlvmTypeConverter::ZeroOfType(result_type);
    false_builder.CreateBr(join_block);

    // Next, create a branch op w/the original builder,
    b.CreateCondBr(predicate, true_block, false_block);

    // then join the two branches back together.
    auto join_builder = std::make_unique<llvm::IRBuilder<>>(join_block);

    llvm::PHINode* phi =
        join_builder->CreatePHI(result_type, /*NumReservedValues=*/2);
    phi->addIncoming(true_result, true_block);
    phi->addIncoming(false_result, false_block);
    return FinalizeNodeIrContextWithValue(std::move(node_context), phi,
                                          /*exit_builder=*/join_builder.get());
  }
  XLS_ASSIGN_OR_RETURN(
      llvm::Value * invoke,
      InvokeRecvCallback(&b, queue, recv, node_context.GetOutputPtr(0),
                         user_data));
  return FinalizeNodeIrContextWithValue(std::move(node_context), invoke);
}

absl::Status IrBuilderVisitor::InvokeSendCallback(llvm::IRBuilder<>* builder,
                                                  JitChannelQueue* queue,
                                                  Send* send,
                                                  llvm::Value* send_data_ptr,
                                                  llvm::Value* user_data) {
  llvm::Type* void_type = llvm::Type::getVoidTy(ctx());
  llvm::Type* int64_type = llvm::Type::getInt64Ty(ctx());
  llvm::Type* int8_ptr_type = llvm::Type::getInt8PtrTy(ctx(), 0);

  llvm::Type* ptr_type = llvm::PointerType::get(ctx(), 0);

  // We do the same for sending/enqueuing as we do for receiving/dequeueing
  // above (set up and call an external function).
  std::vector<llvm::Type*> params = {
      int64_type, int64_type, int8_ptr_type, int64_type, ptr_type,
  };
  llvm::FunctionType* fn_type =
      llvm::FunctionType::get(void_type, params, /*isVarArg=*/false);

  int64_t send_type_size =
      type_converter()->GetTypeByteSize(send->data()->GetType());

  std::vector<llvm::Value*> args = {
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(queue)),
      llvm::ConstantInt::get(int64_type, absl::bit_cast<uint64_t>(send)),
      send_data_ptr,
      llvm::ConstantInt::get(int64_type, send_type_size),
      user_data,
  };

  XLS_RET_CHECK(jit_context_.send_fn().has_value());
  llvm::ConstantInt* fn_addr = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(ctx()),
      absl::bit_cast<uint64_t>(jit_context_.send_fn().value()));
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
  llvm::Value* user_data = node_context.GetUserDataArg();

  XLS_RET_CHECK(jit_context_.queue_manager().has_value());
  XLS_ASSIGN_OR_RETURN(
      JitChannelQueue * queue,
      jit_context_.queue_manager().value()->GetQueueById(send->channel_id()));
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
    XLS_RETURN_IF_ERROR(
        InvokeSendCallback(&true_builder, queue, send, data_ptr, user_data));
    true_builder.CreateBr(join_block);

    llvm::BasicBlock* false_block =
        llvm::BasicBlock::Create(ctx(), absl::StrCat(send->GetName(), "_false"),
                                 node_context.llvm_function(), join_block);
    llvm::IRBuilder<> false_builder(false_block);
    false_builder.CreateBr(join_block);

    b.CreateCondBr(predicate, true_block, false_block);

    auto exit_builder = std::make_unique<llvm::IRBuilder<>>(join_block);
    return FinalizeNodeIrContextWithValue(std::move(node_context),
                                          type_converter()->GetToken(),
                                          exit_builder.get());
  }
  // Unconditional send.
  XLS_RETURN_IF_ERROR(InvokeSendCallback(&b, queue, send, data_ptr, user_data));

  return FinalizeNodeIrContextWithValue(std::move(node_context),
                                        type_converter()->GetToken());
}

}  // namespace

llvm::Value* LlvmMemcpy(llvm::Value* tgt, llvm::Value* src, int64_t size,
                        llvm::IRBuilder<>& builder) {
  return builder.CreateMemCpy(tgt, llvm::MaybeAlign(1), src,
                              llvm::MaybeAlign(1), size);
}

absl::StatusOr<NodeFunction> CreateNodeFunction(
    Node* node, int64_t output_arg_count, JitBuilderContext& jit_context) {
  IrBuilderVisitor visitor(output_arg_count, jit_context);
  XLS_RETURN_IF_ERROR(node->VisitSingleNode(&visitor));
  NodeIrContext node_context = visitor.ConsumeNodeIrContext();
  return NodeFunction{.node = node,
                      .function = node_context.llvm_function(),
                      .output_arg_count = output_arg_count,
                      .has_metadata_args = node_context.has_metadata_args()};
}

}  // namespace xls
