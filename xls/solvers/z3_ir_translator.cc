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

#include "xls/solvers/z3_ir_translator.h"

#include "absl/debugging/leak_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xls/common/logging/logging.h"
#include "xls/common/logging/vlog_is_on.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/abstract_evaluator.h"
#include "xls/ir/abstract_node_evaluator.h"
#include "xls/ir/nodes.h"
#include "xls/solvers/z3_utils.h"
#include "../z3/src/api/z3_api.h"
#include "../z3/src/api/z3_fpa.h"

namespace xls {
namespace solvers {
namespace z3 {

std::string Predicate::ToString() const {
  switch (kind_) {
    case PredicateKind::kEqualToZero:
      return "eq zero";
    case PredicateKind::kNotEqualToZero:
      return "ne zero";
    case PredicateKind::kEqualToNode:
      return "eq " + node()->GetName();
  }
  return absl::StrFormat("<invalid predicate kind %d>",
                         static_cast<int>(kind_));
}

namespace {

class ScopedErrorHandler;

// Since the callback from z3 does not pass a void* as context we rely on RAII
// to establish this thread local value for retrieval from the static error
// handler.
thread_local ScopedErrorHandler* g_handler = nullptr;

// Helper class for establishing an error callback / turning it into a status
// via RAII.
class ScopedErrorHandler {
 public:
  explicit ScopedErrorHandler(Z3_context ctx) : ctx_(ctx) {
    Z3_set_error_handler(ctx_, Handler);
    prev_handler_ = g_handler;
    g_handler = this;
  }

  ~ScopedErrorHandler() {
    Z3_set_error_handler(ctx_, nullptr);
    XLS_CHECK_EQ(g_handler, this);
    g_handler = prev_handler_;
  }

  absl::Status status() const { return status_; }

 private:
  static void Handler(Z3_context c, Z3_error_code e) {
    g_handler->status_ = absl::InternalError(
        absl::StrFormat("Z3 error: %s", Z3_get_error_msg(c, e)));
    XLS_LOG(ERROR) << g_handler->status_;
  }

  Z3_context ctx_;
  ScopedErrorHandler* prev_handler_;
  absl::Status status_;
};

}  // namespace

// Helpers for Z3 translation that don't need to be part of IrTranslator's
// public interface.
class Z3OpTranslator {
 public:
  explicit Z3OpTranslator(Z3_context z3_ctx) : z3_ctx_(z3_ctx) {}
  // Helpers for building bit-vector operations, which are generally what we
  // use.
  Z3_ast Sub(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvsub(z3_ctx_, lhs, rhs); }
  Z3_ast And(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvand(z3_ctx_, lhs, rhs); }
  Z3_ast Or(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvor(z3_ctx_, lhs, rhs); }
  Z3_ast Xor(Z3_ast lhs, Z3_ast rhs) { return Z3_mk_bvxor(z3_ctx_, lhs, rhs); }
  Z3_ast Not(Z3_ast arg) { return Z3_mk_bvnot(z3_ctx_, arg); }
  Z3_ast ReduceOr(Z3_ast arg) { return Z3_mk_bvredor(z3_ctx_, arg); }
  Z3_ast EqZero(Z3_ast arg) { return Not(Z3_mk_bvredor(z3_ctx_, arg)); }
  Z3_ast Eq(Z3_ast lhs, Z3_ast rhs) { return EqZero(Xor(lhs, rhs)); }
  Z3_ast ZextBy1b(Z3_ast arg) { return Z3_mk_zero_ext(z3_ctx_, 1, arg); }
  Z3_ast SextBy1b(Z3_ast arg) { return Z3_mk_sign_ext(z3_ctx_, 1, arg); }
  Z3_ast Extract(Z3_ast arg, int64_t bitno) {
    return Z3_mk_extract(z3_ctx_, bitno, bitno, arg);
  }

  int64_t GetBvBitCount(Z3_ast arg) {
    Z3_sort sort = Z3_get_sort(z3_ctx_, arg);
    return Z3_get_bv_sort_size(z3_ctx_, sort);
  }

  // Explodes bits in the bit-vector Z3 value "arg" such that the LSb is in
  // index 0 of the return value.
  std::vector<Z3_ast> ExplodeBits(Z3_ast arg) {
    std::vector<Z3_ast> bits;
    int64_t bit_count = GetBvBitCount(arg);
    bits.reserve(bit_count);
    for (int64_t i = 0; i < bit_count; ++i) {
      bits.push_back(Extract(arg, i));
    }
    return bits;
  }

  Z3_ast Msb(Z3_ast arg) {
    int64_t bit_count = GetBvBitCount(arg);
    return Extract(arg, bit_count - 1);
  }

  Z3_ast SignExt(Z3_ast arg, int64_t new_bit_count) {
    int64_t input_bit_count = GetBvBitCount(arg);
    XLS_CHECK_GE(new_bit_count, input_bit_count);
    return Z3_mk_sign_ext(z3_ctx_, new_bit_count - input_bit_count, arg);
  }

  // Concatenates args such that arg[0]'s most significant bit is the most
  // significant bit of the result, and arg[args.size()-1]'s least significant
  // bit is the least significant bit of the result.
  Z3_ast ConcatN(absl::Span<const Z3_ast> args) {
    Z3_ast accum = args[0];
    for (int64_t i = 1; i < args.size(); ++i) {
      accum = Z3_mk_concat(z3_ctx_, accum, args[i]);
    }
    return accum;
  }

  // Returns whether lhs < rhs -- this is determined by zero-extending the
  // values and testing whether lhs - rhs < 0
  Z3_ast ULt(Z3_ast lhs, Z3_ast rhs) {
    return Msb(Sub(ZextBy1b(lhs), ZextBy1b(rhs)));
  }

  Z3_ast SLt(Z3_ast lhs, Z3_ast rhs) {
    return Msb(Sub(SextBy1b(lhs), SextBy1b(rhs)));
  }

  Z3_ast Min(Z3_ast lhs, Z3_ast rhs) {
    Z3_ast lt = Z3_mk_bvult(z3_ctx_, lhs, rhs);
    return Z3_mk_ite(z3_ctx_, lt, lhs, rhs);
  }

  // Returns a bit vector filled with "bit_count" digits of "value".
  Z3_ast Fill(bool value, int64_t bit_count) {
    std::unique_ptr<bool[]> bits(new bool[bit_count]);
    for (int64_t i = 0; i < bit_count; ++i) {
      bits[i] = value;
    }
    return Z3_mk_bv_numeral(z3_ctx_, bit_count, &bits[0]);
  }

  // For use in solver assertions, we have to use the "mk_eq" form that creates
  // a bool (in lieu of a bit vector). We put the "Bool" suffix on these helper
  // routines.
  Z3_ast EqZeroBool(Z3_ast arg) {
    int64_t bits = GetBvBitCount(arg);
    return Z3_mk_eq(z3_ctx_, arg, Fill(false, bits));
  }

  Z3_ast NeZeroBool(Z3_ast arg) { return Z3_mk_not(z3_ctx_, EqZeroBool(arg)); }

  Z3_ast NeBool(Z3_ast lhs, Z3_ast rhs) {
    return Z3_mk_not(z3_ctx_, Z3_mk_eq(z3_ctx_, lhs, rhs));
  }

  // Makes a bit count parameter.
  Z3_ast MakeBvParam(int64_t bit_count, absl::string_view name) {
    Z3_sort type = Z3_mk_bv_sort(z3_ctx_, bit_count);
    return Z3_mk_const(
        z3_ctx_, Z3_mk_string_symbol(z3_ctx_, std::string(name).c_str()), type);
  }

  Z3_context z3_ctx_;
};

namespace {

// Helper class for using the AbstractNodeEvaluator to enqueue Z3 expressions.
class Z3AbstractEvaluator
    : public AbstractEvaluator<Z3_ast, Z3AbstractEvaluator> {
 public:
  explicit Z3AbstractEvaluator(Z3_context z3_ctx) : translator_(z3_ctx) {}
  Element One() const { return translator_.Fill(true, 1); }
  Element Zero() const { return translator_.Fill(false, 1); }
  Element Not(const Element& a) const { return translator_.Not(a); }
  Element And(const Element& a, const Element& b) const {
    return translator_.And(a, b);
  }
  Element Or(const Element& a, const Element& b) const {
    return translator_.Or(a, b);
  }

 private:
  mutable Z3OpTranslator translator_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<IrTranslator>> IrTranslator::CreateAndTranslate(
    Function* function) {
  Z3_config config = Z3_mk_config();
  Z3_set_param_value(config, "proof", "true");
  auto translator = absl::WrapUnique(new IrTranslator(config, function));
  XLS_RETURN_IF_ERROR(function->Accept(translator.get()));
  return translator;
}

absl::StatusOr<std::unique_ptr<IrTranslator>> IrTranslator::CreateAndTranslate(
    Z3_context ctx, Function* function,
    absl::Span<const Z3_ast> imported_params) {
  auto translator =
      absl::WrapUnique(new IrTranslator(ctx, function, imported_params));
  XLS_RETURN_IF_ERROR(function->Accept(translator.get()));
  return translator;
}

absl::Status IrTranslator::Retranslate(
    const absl::flat_hash_map<const Node*, Z3_ast>& replacements) {
  ResetVisitedState();
  translations_ = replacements;
  return xls_function_->Accept(this);
}

IrTranslator::IrTranslator(Z3_config config, Function* xls_function)
    : config_(config),
      ctx_(Z3_mk_context(config_)),
      borrowed_context_(false),
      xls_function_(xls_function) {}

IrTranslator::IrTranslator(Z3_context ctx, Function* xls_function,
                           absl::Span<const Z3_ast> imported_params)
    : ctx_(ctx),
      borrowed_context_(true),
      imported_params_(imported_params),
      xls_function_(xls_function) {}

IrTranslator::~IrTranslator() {
  if (!borrowed_context_) {
    Z3_del_context(ctx_);
    Z3_del_config(config_);
  }
}

Z3_ast IrTranslator::GetTranslation(const Node* source) {
  return translations_.at(source);
}

Z3_ast IrTranslator::GetReturnNode() {
  return GetTranslation(xls_function_->return_value());
}

Z3_sort_kind IrTranslator::GetValueKind(Z3_ast value) {
  Z3_sort sort = Z3_get_sort(ctx_, value);
  return Z3_get_sort_kind(ctx_, sort);
}

void IrTranslator::SetTimeout(absl::Duration timeout) {
  std::string timeout_str = absl::StrCat(absl::ToInt64Milliseconds(timeout));
  Z3_update_param_value(ctx_, "timeout", timeout_str.c_str());
}

Z3_ast IrTranslator::FloatZero(Z3_sort sort) {
  return Z3_mk_fpa_zero(ctx_, sort, /*negative=*/false);
}

absl::StatusOr<Z3_ast> IrTranslator::FloatFlushSubnormal(Z3_ast value) {
  Z3_sort sort = Z3_get_sort(ctx_, value);
  Z3_sort_kind sort_kind = Z3_get_sort_kind(ctx_, sort);
  if (sort_kind != Z3_FLOATING_POINT_SORT) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Wrong sort for floating-point operations: %d.",
                        static_cast<int>(sort_kind)));
  }
  Z3_ast is_subnormal(Z3_mk_fpa_is_subnormal(ctx_, value));
  return Z3_mk_ite(ctx_, is_subnormal, FloatZero(sort), value);
}

absl::StatusOr<Z3_ast> IrTranslator::ToFloat32(absl::Span<const Z3_ast> nodes) {
  if (nodes.size() != 3) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Incorrect number of arguments - need 3, got ", nodes.size()));
  }

  // Does some validation and returns the node of interest.
  auto get_fp_component =
      [this, nodes](int64_t index,
                    int64_t expected_width) -> absl::StatusOr<Z3_ast> {
    Z3_sort sort = Z3_get_sort(ctx_, nodes[index]);
    Z3_sort_kind sort_kind = Z3_get_sort_kind(ctx_, sort);
    if (sort_kind != Z3_BV_SORT) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Wrong sort for floating-point components: need Z3_BV_SORT, got ",
          static_cast<int>(sort_kind)));
    }

    int bit_width = Z3_get_bv_sort_size(ctx_, sort);
    if (bit_width != expected_width) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid width for FP component %d: got %d, need %d",
                          index, bit_width, expected_width));
    }
    return nodes[index];
  };

  XLS_ASSIGN_OR_RETURN(Z3_ast sign, get_fp_component(0, 1));
  XLS_ASSIGN_OR_RETURN(Z3_ast exponent, get_fp_component(1, 8));
  XLS_ASSIGN_OR_RETURN(Z3_ast significand, get_fp_component(2, 23));

  return Z3_mk_fpa_fp(ctx_, sign, exponent, significand);
}

absl::StatusOr<Z3_ast> IrTranslator::ToFloat32(Z3_ast tuple) {
  std::vector<Z3_ast> components;
  Z3_sort tuple_sort = Z3_get_sort(ctx_, tuple);
  for (int i = 0; i < 3; i++) {
    Z3_func_decl func_decl = Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, i);
    components.push_back(Z3_mk_app(ctx_, func_decl, 1, &tuple));
  }

  return ToFloat32(components);
}

template <typename OpT, typename FnT>
absl::Status IrTranslator::HandleBinary(OpT* op, FnT f) {
  ScopedErrorHandler seh(ctx_);
  Z3_ast result = f(ctx_, GetBitVec(op->operand(0)), GetBitVec(op->operand(1)));
  NoteTranslation(op, result);
  return seh.status();
}

absl::Status IrTranslator::HandleAdd(BinOp* add) {
  return HandleBinary(add, Z3_mk_bvadd);
}
absl::Status IrTranslator::HandleSub(BinOp* sub) {
  return HandleBinary(sub, Z3_mk_bvsub);
}

absl::Status IrTranslator::HandleULe(CompareOp* ule) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    Z3OpTranslator op_translator(ctx);
    std::vector<Z3_ast> args;
    Z3_ast ult = op_translator.ULt(lhs, rhs);
    Z3_ast eq = op_translator.Eq(lhs, rhs);
    Z3_ast result = Z3_mk_bvor(ctx, ult, eq);
    return Z3_mk_bvredor(ctx, result);
  };
  return HandleBinary(ule, f);
}

absl::Status IrTranslator::HandleULt(CompareOp* lt) {
  auto f = [this](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    return Z3OpTranslator(ctx_).ULt(lhs, rhs);
  };
  return HandleBinary(lt, f);
}

absl::Status IrTranslator::HandleUGe(CompareOp* uge) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    Z3OpTranslator t(ctx);
    return t.Not(t.ULt(lhs, rhs));
  };
  return HandleBinary(uge, f);
}

absl::Status IrTranslator::HandleUGt(CompareOp* gt) {
  auto f = [this](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    // If the msb of the subtraction result is set, that means we underflowed,
    // so RHS is > LHS (that is LHS < RHS)
    //
    // Since we're trying to determine whether LHS > RHS we ask whether:
    //    (LHS == RHS) => false
    //    (LHS < RHS) => false
    //    _ => true
    Z3OpTranslator t(ctx_);
    return t.Not(t.Or(t.Eq(lhs, rhs), t.ULt(lhs, rhs)));
  };
  return HandleBinary(gt, f);
}

absl::Status IrTranslator::HandleSGt(CompareOp* sgt) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    Z3OpTranslator op_translator(ctx);
    Z3_ast slt = op_translator.SLt(lhs, rhs);
    Z3_ast eq = op_translator.Eq(lhs, rhs);

    Z3_ast result = Z3_mk_bvor(ctx, slt, eq);
    result = Z3_mk_bvredor(ctx, result);
    return Z3_mk_bvnot(ctx, result);
  };
  return HandleBinary(sgt, f);
}

absl::Status IrTranslator::HandleSLe(CompareOp* sle) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    Z3OpTranslator op_translator(ctx);
    std::vector<Z3_ast> args;
    Z3_ast slt = op_translator.SLt(lhs, rhs);
    Z3_ast eq = op_translator.Eq(lhs, rhs);
    Z3_ast result = Z3_mk_bvor(ctx, slt, eq);
    return Z3_mk_bvredor(ctx, result);
  };
  return HandleBinary(sle, f);
}

absl::Status IrTranslator::HandleSLt(CompareOp* slt) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    return Z3OpTranslator(ctx).SLt(lhs, rhs);
  };
  return HandleBinary(slt, f);
}

absl::Status IrTranslator::HandleSGe(CompareOp* sge) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    Z3OpTranslator t(ctx);
    return t.Not(t.SLt(lhs, rhs));
  };
  return HandleBinary(sge, f);
}

absl::Status IrTranslator::HandleEq(CompareOp* eq) {
  auto f = [](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    Z3OpTranslator t(ctx);
    return t.Not(t.ReduceOr(t.Xor(lhs, rhs)));
  };
  return HandleBinary(eq, f);
}

absl::Status IrTranslator::HandleNe(CompareOp* ne) {
  auto f = [](Z3_context ctx, Z3_ast a, Z3_ast b) {
    Z3OpTranslator t(ctx);
    return t.ReduceOr(t.Xor(a, b));
  };
  return HandleBinary(ne, f);
}

template <typename FnT>
absl::Status IrTranslator::HandleShift(BinOp* shift, FnT fshift) {
  auto f = [shift, fshift](Z3_context ctx, Z3_ast lhs, Z3_ast rhs) {
    int64_t lhs_bit_count = shift->operand(0)->BitCountOrDie();
    int64_t rhs_bit_count = shift->operand(1)->BitCountOrDie();
    if (rhs_bit_count < lhs_bit_count) {
      rhs = Z3_mk_zero_ext(ctx, lhs_bit_count - rhs_bit_count, rhs);
    } else if (rhs_bit_count > lhs_bit_count) {
      lhs = Z3_mk_zero_ext(ctx, rhs_bit_count - lhs_bit_count, lhs);
    }
    return fshift(ctx, lhs, rhs);
  };
  return HandleBinary(shift, f);
}

absl::Status IrTranslator::HandleShra(BinOp* shra) {
  return HandleShift(shra, Z3_mk_bvashr);
}

absl::Status IrTranslator::HandleShrl(BinOp* shrl) {
  return HandleShift(shrl, Z3_mk_bvlshr);
}

absl::Status IrTranslator::HandleShll(BinOp* shll) {
  return HandleShift(shll, Z3_mk_bvshl);
}

template <typename OpT, typename FnT>
absl::Status IrTranslator::HandleNary(OpT* op, FnT f, bool invert_result) {
  ScopedErrorHandler seh(ctx_);
  int64_t operands = op->operands().size();
  XLS_RET_CHECK_GT(operands, 0) << op->ToString();
  Z3_ast accum = GetBitVec(op->operand(0));
  for (int64_t i = 1; i < operands; ++i) {
    accum = f(ctx_, accum, GetBitVec(op->operand(i)));
  }
  if (invert_result) {
    accum = Z3OpTranslator(ctx_).Not(accum);
  }
  NoteTranslation(op, accum);
  return seh.status();
}

absl::Status IrTranslator::HandleNaryAnd(NaryOp* and_op) {
  return HandleNary(and_op, Z3_mk_bvand, /*invert_result=*/false);
}

absl::Status IrTranslator::HandleNaryNand(NaryOp* nand_op) {
  return HandleNary(nand_op, Z3_mk_bvand, /*invert_result=*/true);
}

absl::Status IrTranslator::HandleNaryNor(NaryOp* nor_op) {
  return HandleNary(nor_op, Z3_mk_bvor, /*invert_result=*/true);
}

absl::Status IrTranslator::HandleNaryOr(NaryOp* or_op) {
  return HandleNary(or_op, Z3_mk_bvor, /*invert_result=*/false);
}

absl::Status IrTranslator::HandleNaryXor(NaryOp* op) {
  return HandleNary(op, Z3_mk_bvxor, /*invert_result=*/false);
}

absl::Status IrTranslator::HandleConcat(Concat* concat) {
  return HandleNary(concat, Z3_mk_concat, /*invert_result=*/false);
}

Z3_ast IrTranslator::CreateTuple(Z3_sort tuple_sort,
                                 absl::Span<const Z3_ast> elements) {
  Z3_func_decl mk_tuple_decl = Z3_get_tuple_sort_mk_decl(ctx_, tuple_sort);
  return Z3_mk_app(ctx_, mk_tuple_decl, elements.size(), elements.data());
}

Z3_ast IrTranslator::CreateTuple(Type* tuple_type,
                                 absl::Span<const Z3_ast> elements) {
  Z3_sort tuple_sort = TypeToSort(ctx_, *tuple_type);
  Z3_func_decl mk_tuple_decl = Z3_get_tuple_sort_mk_decl(ctx_, tuple_sort);
  return Z3_mk_app(ctx_, mk_tuple_decl, elements.size(), elements.data());
}

absl::StatusOr<Z3_ast> IrTranslator::CreateZ3Param(
    Type* type, absl::string_view param_name) {
  return Z3_mk_const(ctx_,
                     Z3_mk_string_symbol(ctx_, std::string(param_name).c_str()),
                     TypeToSort(ctx_, *type));
}

absl::Status IrTranslator::HandleParam(Param* param) {
  ScopedErrorHandler seh(ctx_);
  Type* type = param->GetType();

  // If in "Use existing" mode, then all params must have been encountered
  // already - just copy them over.
  Z3_ast value;
  if (imported_params_) {
    // Find the index of this param in the function, and pull that one out of
    // the imported set.
    XLS_ASSIGN_OR_RETURN(int64_t param_index,
                         param->function_base()->GetParamIndex(param));
    value = imported_params_.value().at(param_index);
  } else {
    XLS_ASSIGN_OR_RETURN(value, CreateZ3Param(type, param->name()));
  }
  NoteTranslation(param, value);
  return seh.status();
}

Z3_ast IrTranslator::ZeroOfSort(Z3_sort sort) {
  // We represent tuples as bit vectors.
  Z3_sort_kind sort_kind = Z3_get_sort_kind(ctx_, sort);
  switch (sort_kind) {
    case Z3_BV_SORT:
      return Z3_mk_int(ctx_, 0, sort);
    case Z3_ARRAY_SORT: {
      // it's an array, so we need to create an array of zero-valued elements.
      Z3_sort index_sort = Z3_get_array_sort_domain(ctx_, sort);
      Z3_ast element = ZeroOfSort(Z3_get_array_sort_range(ctx_, sort));
      return Z3_mk_const_array(ctx_, index_sort, element);
    }
    case Z3_DATATYPE_SORT: {
      int num_elements = Z3_get_tuple_sort_num_fields(ctx_, sort);
      std::vector<Z3_ast> elements;
      elements.reserve(num_elements);
      for (int i = 0; i < num_elements; i++) {
        elements.push_back(ZeroOfSort(
            Z3_get_range(ctx_, Z3_get_tuple_sort_field_decl(ctx_, sort, i))));
      }
      return CreateTuple(sort, elements);
    }
    default:
      XLS_LOG(FATAL) << "Unknown/unsupported sort kind: "
                     << static_cast<int>(sort_kind);
  }
}

Z3_ast IrTranslator::CreateArray(ArrayType* type,
                                 absl::Span<const Z3_ast> elements) {
  Z3_sort element_sort = TypeToSort(ctx_, *type->element_type());

  // Zero-element arrays are A Thing, so we need to synthesize a Z3 zero value
  // for all our array element types.
  Z3_ast default_value = ZeroOfSort(element_sort);
  Z3_sort index_sort =
      Z3_mk_bv_sort(ctx_, Bits::MinBitCountUnsigned(type->size()));
  Z3_ast z3_array = Z3_mk_const_array(ctx_, index_sort, default_value);
  Z3OpTranslator op_translator(ctx_);
  for (int i = 0; i < type->size(); i++) {
    Z3_ast index = Z3_mk_int64(ctx_, i, index_sort);
    z3_array = Z3_mk_store(ctx_, z3_array, index, elements[i]);
  }

  return z3_array;
}

absl::Status IrTranslator::HandleAfterAll(AfterAll* after_all) {
  ScopedErrorHandler seh(ctx_);
  // Token types don't contain any data. A 0-field tuple is a convenient
  // way to let (most of) the rest of the z3 infrastructure treat a
  // token like a normal data-type.
  NoteTranslation(after_all,
                  CreateTuple(TypeToSort(ctx_, *after_all->GetType()),
                              /*elements=*/{}));
  return seh.status();
}

absl::Status IrTranslator::HandleArray(Array* array) {
  ScopedErrorHandler seh(ctx_);
  std::vector<Z3_ast> elements;
  elements.reserve(array->size());
  for (int i = 0; i < array->size(); i++) {
    elements.push_back(GetValue(array->operand(i)));
  }

  NoteTranslation(array,
                  CreateArray(array->GetType()->AsArrayOrDie(), elements));
  return seh.status();
}

absl::Status IrTranslator::HandleTuple(Tuple* tuple) {
  std::vector<Z3_ast> elements;
  elements.reserve(tuple->operand_count());
  for (int i = 0; i < tuple->operand_count(); i++) {
    elements.push_back(GetValue(tuple->operand(i)));
  }
  NoteTranslation(tuple, CreateTuple(tuple->GetType(), elements));

  return absl::OkStatus();
}

Z3_ast IrTranslator::GetAsFormattedArrayIndex(Z3_ast index,
                                              ArrayType* array_type) {
  // In XLS, array indices can be of any sort, whereas in Z3, index types need
  // to be declared w/the array (the "domain" argument - we declare that to be
  // the smallest bit vector that covers all indices. Thus, we need to "cast"
  // appropriately here.
  int target_width = Bits::MinBitCountUnsigned(array_type->size());
  int z3_width = Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, index));
  if (z3_width < target_width) {
    index = Z3_mk_zero_ext(ctx_, target_width - z3_width, index);
  } else if (z3_width > target_width) {
    index =
        Z3_mk_extract(ctx_, Bits::MinBitCountUnsigned(array_type->size()) - 1,
                      /*low=*/0, index);
  }

  return index;
}

Z3_ast IrTranslator::GetArrayElement(ArrayType* array_type, Z3_ast array,
                                     Z3_ast index) {
  index = GetAsFormattedArrayIndex(index, array_type);
  // To follow XLS semantics, if the index exceeds the array size, then return
  // the element at the max index.
  Z3OpTranslator t(ctx_);
  Z3_ast array_max_index =
      Z3_mk_int64(ctx_, array_type->size() - 1, Z3_get_sort(ctx_, index));
  index = t.Min(index, array_max_index);
  return Z3_mk_select(ctx_, array, index);
}

absl::Status IrTranslator::HandleArrayIndex(ArrayIndex* array_index) {
  ScopedErrorHandler seh(ctx_);
  Type* array_type = array_index->array()->GetType();
  Z3_ast element = GetValue(array_index->array());
  for (Node* index : array_index->indices()) {
    element =
        GetArrayElement(array_type->AsArrayOrDie(), element, GetValue(index));
    array_type = array_type->AsArrayOrDie()->element_type();
  }
  NoteTranslation(array_index, element);
  return seh.status();
}

Z3_ast IrTranslator::UpdateArrayElement(Type* type, Z3_ast array, Z3_ast value,
                                        Z3_ast cond,
                                        absl::Span<const Z3_ast> indices) {
  if (indices.empty()) {
    return Z3_mk_ite(ctx_, cond, value, array);
  }
  ArrayType* array_type = type->AsArrayOrDie();
  Z3_sort index_sort =
      Z3_mk_bv_sort(ctx_, Bits::MinBitCountUnsigned(array_type->size()));
  std::vector<Z3_ast> elements;
  for (int64_t i = 0; i < array_type->size(); ++i) {
    Z3_ast this_index =
        GetAsFormattedArrayIndex(Z3_mk_int64(ctx_, i, index_sort), array_type);
    Z3_ast updated_index =
        GetAsFormattedArrayIndex(indices.front(), array_type);
    // In the recursive call, the condition is updated by whether the current
    // index matches.
    Z3_ast and_args[] = {cond, Z3_mk_eq(ctx_, this_index, updated_index)};
    Z3_ast new_cond = Z3_mk_and(ctx_, 2, and_args);
    elements.push_back(UpdateArrayElement(
        /*type=*/array_type->element_type(),
        /*array=*/Z3_mk_select(ctx_, array, this_index),
        /*value=*/value, /*cond=*/new_cond, indices.subspan(1)));
  }
  return CreateArray(array_type, elements);
}

absl::Status IrTranslator::HandleArrayUpdate(ArrayUpdate* array_update) {
  ScopedErrorHandler seh(ctx_);

  std::vector<Z3_ast> indices;
  for (Node* index : array_update->indices()) {
    indices.push_back(GetValue(index));
  }
  Z3_ast new_array = UpdateArrayElement(
      /*type=*/array_update->GetType(),
      /*array=*/GetValue(array_update->array_to_update()),
      /*value=*/GetValue(array_update->update_value()),
      /*cond=*/Z3_mk_true(ctx_),
      /*indices=*/indices);
  NoteTranslation(array_update, new_array);
  return seh.status();
}

absl::Status IrTranslator::HandleArrayConcat(ArrayConcat* array_concat) {
  ScopedErrorHandler seh(ctx_);

  std::vector<Z3_ast> elements;
  for (Node* operand : array_concat->operands()) {
    // Get number of elements in this operand (which is an array)
    ArrayType* array_type = operand->GetType()->AsArrayOrDie();
    int64_t element_count = array_type->size();

    Z3_sort index_sort =
        Z3_mk_bv_sort(ctx_, Bits::MinBitCountUnsigned(element_count));

    Z3_ast array = GetValue(operand);

    for (int64_t i = 0; i < element_count; ++i) {
      Z3_ast index = Z3_mk_int64(ctx_, i, index_sort);
      Z3_ast element = Z3_mk_select(ctx_, array, index);
      elements.push_back(element);
    }
  }

  NoteTranslation(
      array_concat,
      CreateArray(array_concat->GetType()->AsArrayOrDie(), elements));
  return seh.status();
}

absl::Status IrTranslator::HandleArraySlice(ArraySlice* array_slice) {
  ScopedErrorHandler seh(ctx_);
  Z3_ast array_ast = GetValue(array_slice->array());
  Z3_ast start_ast = GetValue(array_slice->start());
  ArrayType* input_type = array_slice->array()->GetType()->AsArrayOrDie();
  ArrayType result_type(array_slice->width(), input_type->element_type());
  Z3_ast formatted_start_ast = GetAsFormattedArrayIndex(start_ast, input_type);

  std::vector<Z3_ast> elements;
  for (uint64_t i = 0; i < array_slice->width(); ++i) {
    Z3_ast i_ast = Z3_mk_int64(ctx_, i, Z3_get_sort(ctx_, formatted_start_ast));
    Z3_ast index_ast = Z3_mk_bvadd(ctx_, i_ast, formatted_start_ast);
    elements.push_back(GetArrayElement(input_type, array_ast, index_ast));
  }

  NoteTranslation(array_slice, CreateArray(&result_type, elements));
  return seh.status();
}

absl::Status IrTranslator::HandleTupleIndex(TupleIndex* tuple_index) {
  ScopedErrorHandler seh(ctx_);
  Z3_ast tuple = GetValue(tuple_index->operand(0));
  Z3_sort tuple_sort = Z3_get_sort(ctx_, tuple);
  Z3_func_decl proj_fn =
      Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, tuple_index->index());
  Z3_ast result = Z3_mk_app(ctx_, proj_fn, 1, &tuple);

  NoteTranslation(tuple_index, result);
  return seh.status();
}

// Handles the translation of unary node "op" by using the abstract node
// evaluator.
absl::Status IrTranslator::HandleUnaryViaAbstractEval(Node* op) {
  XLS_CHECK_EQ(op->operand_count(), 1);
  ScopedErrorHandler seh(ctx_);
  Z3AbstractEvaluator evaluator(ctx_);

  Z3_ast operand = GetBitVec(op->operand(0));
  Z3OpTranslator t(ctx_);
  XLS_CHECK_EQ(op->operand(0)->BitCountOrDie(), t.GetBvBitCount(operand));
  std::vector<Z3_ast> input_bits = t.ExplodeBits(operand);

  XLS_ASSIGN_OR_RETURN(
      std::vector<Z3_ast> output_bits,
      AbstractEvaluate(op, std::vector<Z3AbstractEvaluator::Vector>{input_bits},
                       &evaluator, nullptr));
  // The "output_bits" we are given have LSb in index 0, but ConcatN puts
  // argument 0 in the MSb position, so we must reverse.
  std::reverse(output_bits.begin(), output_bits.end());
  Z3_ast result = t.ConcatN(output_bits);
  XLS_CHECK_EQ(op->BitCountOrDie(), t.GetBvBitCount(result));
  NoteTranslation(op, result);
  return seh.status();
}

template <typename FnT>
absl::Status IrTranslator::HandleUnary(UnOp* op, FnT f) {
  ScopedErrorHandler seh(ctx_);
  Z3_ast result = f(ctx_, GetBitVec(op->operand(0)));
  NoteTranslation(op, result);
  return seh.status();
}

absl::Status IrTranslator::HandleEncode(Encode* encode) {
  return HandleUnaryViaAbstractEval(encode);
}

absl::Status IrTranslator::HandleOneHot(OneHot* one_hot) {
  return HandleUnaryViaAbstractEval(one_hot);
}

absl::Status IrTranslator::HandleNeg(UnOp* neg) {
  return HandleUnary(neg, Z3_mk_bvneg);
}

absl::Status IrTranslator::HandleNot(UnOp* not_op) {
  return HandleUnary(not_op, Z3_mk_bvnot);
}

absl::Status IrTranslator::HandleReverse(UnOp* reverse) {
  return HandleUnaryViaAbstractEval(reverse);
}

absl::Status IrTranslator::HandleIdentity(UnOp* identity) {
  NoteTranslation(identity, GetValue(identity->operand(0)));
  return absl::OkStatus();
}

absl::Status IrTranslator::HandleSignExtend(ExtendOp* sign_ext) {
  ScopedErrorHandler seh(ctx_);
  int64_t input_bit_count = sign_ext->operand(0)->BitCountOrDie();
  Z3_ast result =
      Z3_mk_sign_ext(ctx_, sign_ext->new_bit_count() - input_bit_count,
                     GetBitVec(sign_ext->operand(0)));
  NoteTranslation(sign_ext, result);
  return seh.status();
}

absl::Status IrTranslator::HandleZeroExtend(ExtendOp* zero_ext) {
  ScopedErrorHandler seh(ctx_);
  int64_t input_bit_count = zero_ext->operand(0)->BitCountOrDie();
  Z3_ast result =
      Z3_mk_zero_ext(ctx_, zero_ext->new_bit_count() - input_bit_count,
                     GetBitVec(zero_ext->operand(0)));
  NoteTranslation(zero_ext, result);
  return seh.status();
}

absl::Status IrTranslator::HandleBitSlice(BitSlice* bit_slice) {
  ScopedErrorHandler seh(ctx_);
  int64_t low = bit_slice->start();
  int64_t high = low + bit_slice->width() - 1;
  Z3_ast result =
      Z3_mk_extract(ctx_, high, low, GetBitVec(bit_slice->operand(0)));
  NoteTranslation(bit_slice, result);
  return seh.status();
}

absl::Status IrTranslator::HandleBitSliceUpdate(BitSliceUpdate* update) {
  ScopedErrorHandler seh(ctx_);
  Z3AbstractEvaluator evaluator(ctx_);
  std::vector<Z3_ast> to_update =
      Z3OpTranslator(ctx_).ExplodeBits(GetBitVec(update->to_update()));
  std::vector<Z3_ast> start =
      Z3OpTranslator(ctx_).ExplodeBits(GetBitVec(update->start()));
  std::vector<Z3_ast> update_value =
      Z3OpTranslator(ctx_).ExplodeBits(GetBitVec(update->update_value()));

  std::vector<Z3_ast> flat_results =
      evaluator.BitSliceUpdate(to_update, start, update_value);

  std::reverse(flat_results.begin(), flat_results.end());
  Z3_ast result = UnflattenZ3Ast(update->GetType(), flat_results);

  NoteTranslation(update, result);
  return seh.status();
}

absl::Status IrTranslator::HandleDynamicBitSlice(
    DynamicBitSlice* dynamic_bit_slice) {
  ScopedErrorHandler seh(ctx_);
  Z3_ast value = GetBitVec(dynamic_bit_slice->operand(0));
  Z3_ast start = GetBitVec(dynamic_bit_slice->operand(1));
  int64_t value_width = dynamic_bit_slice->operand(0)->BitCountOrDie();
  int64_t start_width = dynamic_bit_slice->operand(1)->BitCountOrDie();

  int64_t max_width = std::max(value_width, start_width);
  Z3_ast value_ext = Z3_mk_zero_ext(ctx_, max_width - value_width, value);
  Z3_ast start_ext = Z3_mk_zero_ext(ctx_, max_width - start_width, start);

  Value operand_width(UBits(value_width, max_width));
  BitsType max_width_type(max_width);
  XLS_ASSIGN_OR_RETURN(Z3_ast bit_width,
                       TranslateLiteralValue(&max_width_type, operand_width));

  // Indicates whether slice is completely out of bounds.
  Z3_ast out_of_bounds = Z3_mk_bvuge(ctx_, start_ext, bit_width);
  BitsType return_type(dynamic_bit_slice->width());
  XLS_ASSIGN_OR_RETURN(
      Z3_ast zeros, TranslateLiteralValue(
                        &return_type, Value(Bits(dynamic_bit_slice->width()))));
  Z3_ast shifted_value = Z3_mk_bvlshr(ctx_, value_ext, start_ext);
  Z3_ast truncated_value =
      Z3_mk_extract(ctx_, dynamic_bit_slice->width() - 1, 0, shifted_value);
  Z3_ast result = Z3_mk_ite(ctx_, out_of_bounds, zeros, truncated_value);
  NoteTranslation(dynamic_bit_slice, result);
  return seh.status();
}

absl::StatusOr<Z3_ast> IrTranslator::TranslateLiteralValue(Type* value_type,
                                                           const Value& value) {
  if (value.IsBits()) {
    const Bits& bits = value.bits();
    std::unique_ptr<bool[]> booleans(new bool[bits.bit_count()]);
    for (int64_t i = 0; i < bits.bit_count(); ++i) {
      booleans[i] = bits.Get(i);
    }
    return Z3_mk_bv_numeral(ctx_, bits.bit_count(), &booleans[0]);
  }

  if (value.IsArray()) {
    ArrayType* array_type = value_type->AsArrayOrDie();
    int num_elements = array_type->size();
    std::vector<Z3_ast> elements;
    elements.reserve(num_elements);

    for (int i = 0; i < value.elements().size(); i++) {
      XLS_ASSIGN_OR_RETURN(Z3_ast translated,
                           TranslateLiteralValue(array_type->element_type(),
                                                 value.elements()[i]));
      elements.push_back(translated);
    }

    return CreateArray(array_type, elements);
  }

  // Tuples!
  TupleType* tuple_type = value_type->AsTupleOrDie();
  int num_elements = tuple_type->size();
  std::vector<Z3_ast> elements;
  elements.reserve(num_elements);
  for (int i = 0; i < num_elements; i++) {
    XLS_ASSIGN_OR_RETURN(Z3_ast translated,
                         TranslateLiteralValue(tuple_type->element_type(i),
                                               value.elements()[i]));
    elements.push_back(translated);
  }

  return CreateTuple(tuple_type, elements);
}

absl::Status IrTranslator::HandleLiteral(Literal* literal) {
  ScopedErrorHandler seh(ctx_);
  XLS_ASSIGN_OR_RETURN(Z3_ast result, TranslateLiteralValue(literal->GetType(),
                                                            literal->value()));
  NoteTranslation(literal, result);
  return seh.status();
}

std::vector<Z3_ast> IrTranslator::FlattenValue(Type* type, Z3_ast value,
                                               bool little_endian) {
  Z3OpTranslator op_translator(ctx_);

  switch (type->kind()) {
    case TypeKind::kBits: {
      std::vector<Z3_ast> boom = op_translator.ExplodeBits(value);
      if (little_endian) {
        std::reverse(boom.begin(), boom.end());
      }
      return boom;
    }
    case TypeKind::kArray: {
      ArrayType* array_type = type->AsArrayOrDie();
      std::vector<Z3_ast> flattened;
      Z3_sort index_sort =
          Z3_mk_bv_sort(ctx_, Bits::MinBitCountUnsigned(array_type->size()));
      for (int i = 0; i < array_type->size(); i++) {
        Z3_ast index = Z3_mk_int64(ctx_, i, index_sort);
        Z3_ast element = GetArrayElement(array_type, value, index);
        std::vector<Z3_ast> flat_child =
            FlattenValue(array_type->element_type(), element, little_endian);
        flattened.insert(flattened.end(), flat_child.begin(), flat_child.end());
      }
      return flattened;
    }
    case TypeKind::kTuple: {
      TupleType* tuple_type = type->AsTupleOrDie();
      Z3_sort tuple_sort = Z3_get_sort(ctx_, value);

      std::vector<Z3_ast> flattened;
      for (int i = 0; i < tuple_type->size(); i++) {
        Z3_func_decl child_accessor =
            Z3_get_tuple_sort_field_decl(ctx_, tuple_sort, i);
        Z3_ast child = Z3_mk_app(ctx_, child_accessor, 1, &value);
        std::vector<Z3_ast> flat_child =
            FlattenValue(tuple_type->element_type(i), child, little_endian);
        flattened.insert(flattened.end(), flat_child.begin(), flat_child.end());
      }
      return flattened;
    }
    default:
      XLS_LOG(FATAL) << "Unsupported type kind: "
                     << TypeKindToString(type->kind());
  }
}

Z3_ast IrTranslator::UnflattenZ3Ast(Type* type, absl::Span<const Z3_ast> flat,
                                    bool little_endian) {
  Z3OpTranslator op_translator(ctx_);
  switch (type->kind()) {
    case TypeKind::kBits:
      if (little_endian) {
        std::vector<Z3_ast> flat_vec(flat.begin(), flat.end());
        std::reverse(flat_vec.begin(), flat_vec.end());
        return op_translator.ConcatN(flat_vec);
      } else {
        return op_translator.ConcatN(flat);
      }
    case TypeKind::kArray: {
      ArrayType* array_type = type->AsArrayOrDie();
      int num_elements = array_type->size();

      Type* element_type = array_type->element_type();
      int element_bits = element_type->GetFlatBitCount();
      std::vector<Z3_ast> elements;
      elements.reserve(num_elements);

      int high = array_type->GetFlatBitCount();
      for (int i = 0; i < num_elements; i++) {
        absl::Span<const Z3_ast> subspan =
            flat.subspan(high - element_bits, element_bits);
        elements.push_back(
            UnflattenZ3Ast(element_type, subspan, little_endian));
        high -= element_bits;
      }
      return CreateArray(array_type, elements);
    }
    case TypeKind::kTuple: {
      // For each tuple element, extract the sub-type's bits and unflatten, then
      // munge into a tuple.
      TupleType* tuple_type = type->AsTupleOrDie();
      std::vector<Z3_ast> elements;
      int high = tuple_type->GetFlatBitCount();
      for (Type* element_type : tuple_type->element_types()) {
        int64_t element_bits = element_type->GetFlatBitCount();
        absl::Span<const Z3_ast> subspan =
            flat.subspan(high - element_bits, element_bits);
        elements.push_back(
            UnflattenZ3Ast(element_type, subspan, little_endian));
        high -= element_bits;
      }
      return CreateTuple(tuple_type, elements);
    }
    default:
      XLS_LOG(FATAL) << "Unsupported type kind: "
                     << TypeKindToString(type->kind());
  }
}

template <typename NodeT>
absl::Status IrTranslator::HandleSelect(
    NodeT* node, std::function<FlatValue(const FlatValue& selector,
                                         const std::vector<FlatValue>& cases)>
                     evaluator) {
  // HandleSel could be implemented on its own terms (and not in the same way
  // as one-hot), if there's concern that flattening to bitwise Z3_asts loses
  // any semantic info.
  ScopedErrorHandler seh(ctx_);
  Z3OpTranslator op_translator(ctx_);
  std::vector<Z3_ast> selector =
      Z3OpTranslator(ctx_).ExplodeBits(GetBitVec(node->selector()));

  std::vector<std::vector<Z3_ast>> case_elements;
  for (Node* element : node->cases()) {
    case_elements.push_back(
        FlattenValue(element->GetType(), GetValue(element)));
  }

  std::vector<Z3_ast> flat_results = evaluator(selector, case_elements);
  std::reverse(flat_results.begin(), flat_results.end());
  Z3_ast result = UnflattenZ3Ast(node->GetType(), flat_results);

  NoteTranslation(node, result);
  return seh.status();
}

absl::Status IrTranslator::HandleOneHotSel(OneHotSelect* one_hot) {
  Z3AbstractEvaluator evaluator(ctx_);
  return HandleSelect(
      one_hot, [&evaluator](const std::vector<Z3_ast>& selector,
                            const std::vector<std::vector<Z3_ast>>& cases) {
        return evaluator.OneHotSelect(selector, cases,
                                      /*selector_can_be_zero=*/true);
      });
}

absl::Status IrTranslator::HandleSel(Select* sel) {
  Z3AbstractEvaluator evaluator(ctx_);
  Z3OpTranslator op_translator(ctx_);
  return HandleSelect(sel, [this, sel, &evaluator](
                               const std::vector<Z3_ast>& selector,
                               const std::vector<std::vector<Z3_ast>>& cases) {
    // Calculate the Z3-ified default value, if any.
    absl::optional<std::vector<Z3_ast>> default_value = absl::nullopt;
    if (sel->default_value()) {
      default_value = FlattenValue(sel->default_value().value()->GetType(),
                                   GetValue(sel->default_value().value()));
    }
    return evaluator.Select(selector, cases, default_value);
  });
}

absl::Status IrTranslator::HandleOrReduce(BitwiseReductionOp* or_reduce) {
  ScopedErrorHandler seh(ctx_);
  // OrReduce == is any bit set in the value? That's the same as "node != 0".
  Z3_ast operand = GetValue(or_reduce->operand(0));
  Z3_ast long_zero = Z3_mk_int(ctx_, 0, Z3_get_sort(ctx_, operand));

  Z3_sort bit_sort = Z3_mk_bv_sort(ctx_, 1);
  Z3_ast one = Z3_mk_int(ctx_, 1, bit_sort);
  Z3_ast zero = Z3_mk_int(ctx_, 0, bit_sort);

  Z3_ast eq = Z3_mk_eq(ctx_, operand, long_zero);
  Z3_ast result = Z3_mk_ite(ctx_, eq, zero, one);

  NoteTranslation(or_reduce, result);
  return seh.status();
}

void IrTranslator::HandleMul(ArithOp* mul, bool is_signed) {
  // In XLS IR, multiply operands can potentially be of different widths. In Z3,
  // they can't, so we need to zext (for a umul) the operands to the size of the
  // result.
  Z3_ast lhs = GetValue(mul->operand(0));
  Z3_ast rhs = GetValue(mul->operand(1));
  int lhs_size = Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, lhs));
  int rhs_size = Z3_get_bv_sort_size(ctx_, Z3_get_sort(ctx_, rhs));

  int result_size = mul->BitCountOrDie();
  int operation_size = std::max(result_size, std::max(lhs_size, rhs_size));
  // Do the mul at maximum width, then truncate if necessary to the result
  // width.
  if (is_signed) {
    if (lhs_size < operation_size) {
      lhs = Z3_mk_sign_ext(ctx_, operation_size - lhs_size, lhs);
    }
    if (rhs_size < operation_size) {
      rhs = Z3_mk_sign_ext(ctx_, operation_size - rhs_size, rhs);
    }
  } else {
    // If we're doing unsigned multiplication, add an extra 0 MSb to make sure
    // Z3 knows that.
    operation_size += 1;
    lhs = Z3_mk_zero_ext(ctx_, operation_size - lhs_size, lhs);
    rhs = Z3_mk_zero_ext(ctx_, operation_size - rhs_size, rhs);
  }

  Z3_ast result = Z3_mk_bvmul(ctx_, lhs, rhs);
  if (operation_size > result_size) {
    result = Z3_mk_extract(ctx_, result_size - 1, 0, result);
  }

  NoteTranslation(mul, result);
}

absl::Status IrTranslator::HandleSMul(ArithOp* mul) {
  ScopedErrorHandler seh(ctx_);
  HandleMul(mul, /*is_signed=*/true);
  return seh.status();
}

absl::Status IrTranslator::HandleUMul(ArithOp* mul) {
  ScopedErrorHandler seh(ctx_);
  HandleMul(mul, /*is_signed=*/false);
  return seh.status();
}

absl::Status IrTranslator::DefaultHandler(Node* node) {
  return absl::UnimplementedError("Unhandled node for conversion: " +
                                  node->ToString());
}

Z3_ast IrTranslator::GetValue(const Node* node) {
  auto it = translations_.find(node);
  XLS_CHECK(it != translations_.end()) << "Node not translated: " << node;
  return it->second;
}

// Wrapper around the above that verifies we're accessing a Bits value.
Z3_ast IrTranslator::GetBitVec(Node* node) {
  Z3_ast value = GetValue(node);
  Z3_sort value_sort = Z3_get_sort(ctx_, value);
  XLS_CHECK_EQ(Z3_get_sort_kind(ctx_, value_sort), Z3_BV_SORT);
  XLS_CHECK_EQ(node->GetType()->GetFlatBitCount(),
               Z3_get_bv_sort_size(ctx_, value_sort));
  return value;
}

void IrTranslator::NoteTranslation(Node* node, Z3_ast translated) {
  if (translations_.contains(node)) {
    XLS_VLOG(2) << "Skipping translation of " << node->GetName()
                << ", as it's already been recorded "
                << "(expected if we're retranslating).";
    return;
  }
  translations_[node] = translated;
}

absl::StatusOr<Z3_ast> PredicateToObjective(Predicate p, Z3_ast a,
                                            IrTranslator* translator) {
  ScopedErrorHandler seh(translator->ctx());
  Z3_ast objective;
  // Note that if the predicate we want to prove is "equal to zero" we return
  // that "not equal to zero" is not satisfiable.
  Z3OpTranslator t(translator->ctx());
  switch (p.kind()) {
    case PredicateKind::kEqualToZero:
      objective = t.NeZeroBool(a);
      break;
    case PredicateKind::kNotEqualToZero:
      objective = t.EqZeroBool(a);
      break;
    case PredicateKind::kEqualToNode: {
      Z3_ast value = translator->GetTranslation(p.node());
      if (translator->GetValueKind(value) != Z3_BV_SORT) {
        return absl::InvalidArgumentError(
            "Cannot compare to non-bits-valued node: " + p.node()->ToString());
      }
      Z3_ast b = value;
      objective = t.NeBool(a, b);
      break;
    }
    default:
      return absl::UnimplementedError("Unhandled predicate.");
  }
  XLS_RETURN_IF_ERROR(seh.status());
  return objective;
}

absl::StatusOr<bool> TryProve(Function* f, Node* subject, Predicate p,
                              absl::Duration timeout) {
  XLS_ASSIGN_OR_RETURN(auto translator, IrTranslator::CreateAndTranslate(f));
  translator->SetTimeout(timeout);
  Z3_ast value = translator->GetTranslation(subject);

  // All token types are equal.
  if (subject->GetType()->IsToken() &&
      p.kind() == PredicateKind::kEqualToNode &&
      p.node()->GetType()->IsToken()) {
    return true;
  }
  if (translator->GetValueKind(value) != Z3_BV_SORT) {
    return absl::InvalidArgumentError(
        "Cannot prove properties of non-bits-typed node: " +
        subject->ToString());
  }
  XLS_ASSIGN_OR_RETURN(Z3_ast objective,
                       PredicateToObjective(p, value, translator.get()));
  Z3_context ctx = translator->ctx();
  XLS_VLOG(2) << "objective:\n" << Z3_ast_to_string(ctx, objective);
  Z3_solver solver = solvers::z3::CreateSolver(ctx, 1);
  Z3_solver_assert(ctx, solver, objective);
  Z3_lbool satisfiable = Z3_solver_check(ctx, solver);
  XLS_VLOG(2) << solvers::z3::SolverResultToString(ctx, solver, satisfiable)
              << std::endl;
  Z3_solver_dec_ref(ctx, solver);

  if (Z3_solver_check(ctx, solver) == Z3_L_FALSE) {
    // We posit the inverse of the predicate we want to check -- when that is
    // unsatisfiable, the predicate has been proven (there was no way found that
    // we could not satisfy its inverse).
    return true;
  }

  return false;
}

}  // namespace z3
}  // namespace solvers
}  // namespace xls
