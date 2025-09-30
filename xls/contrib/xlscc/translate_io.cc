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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "clang/include/clang/AST/DeclTemplate.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/ExprCXX.h"
#include "clang/include/clang/AST/TypeBase.h"
#include "clang/include/clang/Basic/LLVM.h"
#include "clang/include/clang/Basic/OperatorKinds.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/tracked_bvalue.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/translator_types.h"
#include "xls/contrib/xlscc/xlscc_logging.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/nodes.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xlscc {

bool Translator::OpIsMasked(const IOOp& op) {
  const bool mask_write =
      context().mask_memory_writes && op.op == OpType::kWrite;
  const bool mask_other =
      context().mask_io_other_than_memory_writes && op.op != OpType::kWrite;

  return context().mask_side_effects || mask_write || mask_other;
}

absl::StatusOr<TrackedBValue> Translator::GetIOOpRetValueFromSlice(
    TrackedBValue slice_ret_val, const GeneratedFunctionSlice& slice,
    const xls::SourceInfo& loc) {
  XLSCC_CHECK_GT(slice_ret_val.GetType()->AsTupleOrDie()->size(), 0, loc);
  TrackedBValue op_out_value = context().fb->TupleIndex(
      slice_ret_val, slice_ret_val.GetType()->AsTupleOrDie()->size() - 1, loc,
      /*name=*/
      absl::StrFormat("%s_io_out_value", slice.function->name()));
  XLSCC_CHECK(op_out_value.valid(), loc);

  return op_out_value;
}

absl::StatusOr<IOOp*> Translator::AddOpToChannel(IOOp& op, IOChannel* channel,
                                                 const xls::SourceInfo& loc,
                                                 bool mask,
                                                 bool no_before_slice) {
  context().any_side_effects_requested = true;
  context().any_io_ops_requested = true;

  CHECK(op.op == OpType::kTrace || op.op == OpType::kLoopBegin ||
        op.op == OpType::kLoopEndJump || channel != nullptr);
  CHECK_EQ(op.channel, nullptr);

  if (mask || OpIsMasked(op)) {
    IOOpReturn ret;
    ret.generate_expr = false;
    if (op.op != OpType::kTrace && op.op != OpType::kLoopBegin &&
        op.op != OpType::kLoopEndJump) {
      // During IR generation, parameters are generated for masked IOs,
      // so that they are treated as unknown variables for analysis purposes,
      // such as loop unrolling. These are later removed and 0s are substituted
      // when the function [slice] is finished.
      XLS_ASSIGN_OR_RETURN(xls::Type * item_type_xls,
                           TranslateTypeToXLS(channel->item_type, loc));
      TrackedBValue default_bval = context().fb->Param(
          absl::StrFormat("__masked_%s_%i", channel->unique_name,
                          next_masked_op_param_number_++),
          item_type_xls, loc);

      context().sf->masked_op_params_to_remove.push_back(
          default_bval.node()->As<xls::Param>());

      op.input_value = CValue(default_bval, channel->item_type);
    }
    return nullptr;
  }

  if (op.op == OpType::kWrite) {
    context().any_writes_generated |= true;
  }

  op.channel = channel;
  op.op_location = loc;
  op.full_op_location = loc;

  if (channel == nullptr) {
    op.channel_op_index = context().sf->no_channel_op_count++;
  } else {
    op.channel_op_index = channel->total_ops++;
  }

  // When splitting activations on channel ops, a slice with no after_op
  // must be created in order to make it possible to transition activations
  // or "jump" to this slice, as it has no IO op input.
  if (generate_new_fsm_ && split_states_on_channel_ops_ && channel != nullptr &&
      !no_before_slice) {
    XLS_RETURN_IF_ERROR(NewContinuation(op.op, Debug_OpName(op),
                                        /*op_ret_value=*/TrackedBValue(),
                                        op.op_location, op.channel_op_index,
                                        /*create_slice_before=*/true,
                                        /*temp_name=*/false));
  }

  // Add the continuation before, not including the received parameter
  XLS_RETURN_IF_ERROR(NewContinuation(op.op, Debug_OpName(op), op.ret_value,
                                      op.op_location, op.channel_op_index,
                                      /*create_slice_before=*/false,
                                      /*temp_name=*/false));

  std::shared_ptr<CType> channel_item_type;

  // Channel must be inserted first by AddOpToChannel
  if (op.op == OpType::kRecv || op.op == OpType::kRead) {
    if (op.is_blocking) {
      channel_item_type = channel->item_type;
    } else {
      channel_item_type =
          std::make_shared<CInternalTuple>(std::vector<std::shared_ptr<CType>>(
              {channel->item_type, std::make_shared<CBoolType>()}));
    }
  }

  std::shared_ptr<CType> param_type = channel_item_type;

  xls::Type* xls_param_type = nullptr;
  if (param_type != nullptr) {
    XLS_ASSIGN_OR_RETURN(xls_param_type, TranslateTypeToXLS(param_type, loc));
    std::string safe_param_name;
    if (op.channel != nullptr) {
      const std::string channel_name = op.channel->unique_name;
      safe_param_name =
          absl::StrFormat("%s_op%i", channel_name, op.channel_op_index);
    } else {
      safe_param_name = absl::StrFormat("default_op%i", op.channel_op_index);
    }

    TrackedBValue pbval =
        context().fb->Param(safe_param_name, xls_param_type, loc);

    // Check for duplicate params
    if (!pbval.valid()) {
      return absl::InternalError(ErrorMessage(
          loc,
          "Failed to create implicit parameter %s, duplicate? See b/239861050",
          safe_param_name.c_str()));
    }

    const std::string final_param_name =
        pbval.node()->As<xls::Param>()->GetName();

    op.final_param_name = final_param_name;

    TrackedBValue input_io_value = pbval;

    if (channel_item_type) {
      XLSCC_CHECK(input_io_value.valid(), loc);
      op.input_value = CValue(input_io_value, channel_item_type);
    }
  }

  XLS_ASSIGN_OR_RETURN(std::optional<const IOOp*> last_op,
                       GetPreviousOp(op, loc));

  if (last_op.has_value()) {
    op.after_ops.push_back(last_op.value());
  }

  // Sequence after the previous op on the channel
  // TODO(seanhaskell): This is inefficient for memories. Parallelize operations
  // once "token phi" type features are available.
  context().sf->io_ops.push_back(op);

  if (param_type != nullptr) {
    XLSCC_CHECK_NE(xls_param_type, nullptr, loc);
    SideEffectingParameter side_effecting_param;
    side_effecting_param.type = SideEffectingParameterType::kIOOp;
    XLSCC_CHECK(!op.final_param_name.empty(), loc);
    side_effecting_param.param_name = op.final_param_name;
    side_effecting_param.xls_io_param_type = xls_param_type;
    side_effecting_param.io_op = &context().sf->io_ops.back();
    context().sf->side_effecting_parameters.push_back(side_effecting_param);
    if (!context().sf->slices.empty()) {
      context().sf->slices.back().side_effecting_parameters.push_back(
          side_effecting_param);
    }
  }

  IOOp* ret = &context().sf->io_ops.back();
  // TODO(seanhaskell): Turn this back into a CHECK once old pipelined loop
  // body generation is removed.
  if (!context().sf->slices.empty()) {
    context().sf->slices.back().after_op = ret;
  }
  return ret;
}

absl::StatusOr<std::optional<const IOOp*>> Translator::GetPreviousOp(
    const IOOp& op, const xls::SourceInfo& loc) {
  if (op_ordering_ == IOOpOrdering::kNone) {
    return std::nullopt;
  }
  if (op_ordering_ == IOOpOrdering::kChannelWise) {
    std::vector<const IOOp*> previous_ops_on_channel;

    for (const IOOp& existing_op : context().sf->io_ops) {
      if (existing_op.channel != op.channel) {
        continue;
      }
      if (existing_op.scheduling_option != IOSchedulingOption::kNone) {
        continue;
      }
      previous_ops_on_channel.push_back(&existing_op);
    }

    if (!previous_ops_on_channel.empty()) {
      const IOOp* last_op = previous_ops_on_channel.back();
      return last_op;
    }
    return std::nullopt;
  }
  if (op_ordering_ == IOOpOrdering::kLexical) {
    // Sequence after the previous op on any channel
    std::vector<const IOOp*> previous_ops_on_channel;

    for (const IOOp& existing_op : context().sf->io_ops) {
      if (existing_op.scheduling_option != IOSchedulingOption::kNone) {
        continue;
      }
      previous_ops_on_channel.push_back(&existing_op);
    }

    if (!previous_ops_on_channel.empty()) {
      const IOOp* last_op = previous_ops_on_channel.back();
      return last_op;
    }
    return std::nullopt;
  }
  return absl::UnimplementedError(
      ErrorMessage(loc, "IO op ordering %i", static_cast<int>(op_ordering_)));
}

absl::StatusOr<bool> Translator::TypeIsChannel(clang::QualType param,
                                               const xls::SourceInfo& loc) {
  // Ignore &
  XLS_ASSIGN_OR_RETURN(StrippedType stripped, StripTypeQualifiers(param));

  const clang::Type* type = stripped.base.getTypePtr();

  if (auto subst =
          clang::dyn_cast<const clang::SubstTemplateTypeParmType>(type)) {
    return TypeIsChannel(subst->getReplacementType(), loc);
  }

  if (type->getTypeClass() == clang::Type::TypeClass::TemplateSpecialization &&
      type->isRecordType()) {
    // Up-cast to avoid multiple inheritance of getAsRecordDecl()
    std::shared_ptr<CInstantiableTypeAlias> ret(
        new CInstantiableTypeAlias(type->getAsRecordDecl()));

    // TODO(seanhaskell): Put these strings in one place
    if (ret->base()->getNameAsString() == "__xls_channel" ||
        ret->base()->getNameAsString() == "__xls_memory") {
      return true;
    }
  }

  if (type->getTypeClass() == clang::Type::TypeClass::Typedef) {
    return TypeIsChannel(
        clang::QualType(type->getUnqualifiedDesugaredType(), 0), loc);
  }

  if (auto record = clang::dyn_cast<const clang::RecordType>(type)) {
    clang::RecordDecl* decl = record->getOriginalDecl();

    if (auto class_template_spec =
            clang::dyn_cast<const clang::ClassTemplateSpecializationDecl>(
                decl)) {
      const std::string template_name =
          class_template_spec->getSpecializedTemplate()->getNameAsString();

      // TODO(seanhaskell): Put these strings in one place
      if (template_name == "__xls_channel" || template_name == "__xls_memory") {
        return true;
      }
    }
  }

  return false;
}

absl::StatusOr<int64_t> Translator::GetIntegerTemplateArgument(
    const clang::TemplateArgument& arg, clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  if (arg.getKind() == clang::TemplateArgument::ArgKind::Expression) {
    XLS_ASSIGN_OR_RETURN(int64_t val,
                         EvaluateInt64(*arg.getAsExpr(), ctx, loc));
    return val;
  }
  if (arg.getKind() == clang::TemplateArgument::ArgKind::Integral) {
    return arg.getAsIntegral().getExtValue();
  }
  return absl::UnimplementedError(ErrorMessage(
      loc, "Expected integer or expression for second template argument"));
}

absl::StatusOr<std::shared_ptr<CChannelType>> Translator::GetChannelType(
    const clang::QualType& channel_type, clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  clang::ArrayRef<clang::TemplateArgument> template_arguments;
  std::string template_name;

  XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                       StripTypeQualifiers(channel_type));

  if (auto subst = clang::dyn_cast<const clang::SubstTemplateTypeParmType>(
          stripped.base.getTypePtr())) {
    return GetChannelType(subst->getReplacementType(), ctx, loc);
  }

  if (auto template_spec =
          clang::dyn_cast<const clang::TemplateSpecializationType>(
              stripped.base.getTypePtr());
      template_spec != nullptr) {
    template_arguments = template_spec->template_arguments();
    template_name =
        template_spec->getTemplateName().getAsTemplateDecl()->getNameAsString();
    if (template_spec->isTypeAlias()) {
      return GetChannelType(template_spec->getAliasedType(), ctx, loc);
    }
  } else if (auto typedef_type = clang::dyn_cast<const clang::TypedefType>(
                 stripped.base.getTypePtr());
             typedef_type != nullptr) {
    const clang::Type* type = stripped.base.getTypePtr();
    return GetChannelType(
        clang::QualType(type->getUnqualifiedDesugaredType(), 0), ctx, loc);
  } else if (auto record = clang::dyn_cast<const clang::RecordType>(
                 stripped.base.getTypePtr());
             record != nullptr) {
    clang::RecordDecl* decl = record->getOriginalDecl();
    if (auto class_template_spec =
            clang::dyn_cast<const clang::ClassTemplateSpecializationDecl>(
                decl)) {
      template_name =
          class_template_spec->getSpecializedTemplate()->getNameAsString();
      template_arguments = class_template_spec->getTemplateArgs().asArray();
    } else {
      return absl::UnimplementedError(ErrorMessage(
          loc, "Channel RecordDecl should be ClassTemplateSpecializationDecl"));
    }
  } else {
    return absl::UnimplementedError(ErrorMessage(
        loc,
        "Channel type should be a template specialization or record decl"));
  }

  if ((template_arguments.size() != 1) && (template_arguments.size() != 2)) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel should have 1 or 2 template args"));
  }
  OpType op_type = OpType::kNull;
  int64_t memory_size = -1;

  // TODO(seanhaskell): Put these strings in one place
  if (template_name == "__xls_memory") {
    if (template_arguments.size() != 2) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Memory should have 2 template args"));
    }
    XLS_ASSIGN_OR_RETURN(memory_size, GetIntegerTemplateArgument(
                                          template_arguments[1], ctx, loc));
  } else if (template_name == "__xls_channel" &&
             template_arguments.size() == 2) {
    int64_t op_type_int = -1;
    XLS_ASSIGN_OR_RETURN(op_type_int, GetIntegerTemplateArgument(
                                          template_arguments[1], ctx, loc));
    op_type = static_cast<OpType>(op_type_int);
  }
  const clang::TemplateArgument& arg = template_arguments[0];
  XLSCC_CHECK(arg.getKind() == clang::TemplateArgument::ArgKind::Type, loc);
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> item_type,
                       TranslateTypeFromClang(arg.getAsType(), loc));

  return std::make_shared<CChannelType>(item_type, memory_size, op_type);
}

absl::StatusOr<Translator::IOOpReturn> Translator::InterceptIOOp(
    const clang::Expr* expr, const xls::SourceInfo& loc,
    const CValue assignment_value) {
  const IOOpReturn no_op_return = {.generate_expr = true};

  TrackedBValue op_condition = context().full_condition_bval(loc);
  CHECK(op_condition.valid());

  const clang::Expr* object = nullptr;
  std::string op_name;

  if (auto member_call =
          clang::dyn_cast<const clang::CXXMemberCallExpr>(expr)) {
    object = member_call->getImplicitObjectArgument();
    const clang::FunctionDecl* funcdecl = member_call->getDirectCallee();
    op_name = funcdecl->getNameAsString();

    if (assignment_value.valid()) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Assignment to IO ops not supported"));
    }
  } else if (auto operator_call =
                 clang::dyn_cast<const clang::CXXOperatorCallExpr>(expr)) {
    if (operator_call->getOperator() !=
        clang::OverloadedOperatorKind::OO_Subscript) {
      return no_op_return;
    }
    object = operator_call->getArg(0);
    op_name = "__memory_by_operator";
  } else {
    return no_op_return;
  }

  std::vector<ConditionedIOChannel> channels;
  XLS_RETURN_IF_ERROR(GetChannelsForExprOrNull(object, &channels, loc));

  XLS_ASSIGN_OR_RETURN(bool type_is_channel,
                       TypeIsChannel(object->getType(), loc));
  if (type_is_channel && channels.empty()) {
    return absl::InvalidArgumentError(
        ErrorMessage(loc,
                     "Method call on channel type, but no channel found "
                     "(uninitialized?)"));
  }

  if (channels.empty()) {
    return no_op_return;
  }

  auto call = clang::dyn_cast<const clang::CallExpr>(expr);
  std::vector<CValue> arg_vals;
  arg_vals.resize(call->getNumArgs());

  for (int64_t arg = 0; arg < call->getNumArgs(); ++arg) {
    XLS_ASSIGN_OR_RETURN(
        arg_vals[arg],
        GenerateIR_Expr(call->getArg(static_cast<unsigned int>(arg)), loc));
  }

  IOOpReturn ret;
  ret.generate_expr = false;

  const clang::Expr* all_assign_ret_value_to = nullptr;
  std::optional<bool> all_is_blocking;

  // Combine mutually exclusive operations on different channels
  // For example: (cond?channel_A:channel_B).send(foo)
  for (const ConditionedIOChannel& conditioned_channel : channels) {
    const clang::Expr* assign_ret_value_to = nullptr;

    TrackedBValue this_channel_condition = conditioned_channel.condition;
    IOChannel* channel = conditioned_channel.channel;

    TrackedBValue channel_specific_condition =
        this_channel_condition.valid()
            ? TrackedBValue(
                  context().fb->And(op_condition, this_channel_condition, loc))
            : op_condition;

    // Short circuit the op condition if possible
    XLS_RETURN_IF_ERROR(ShortCircuitBVal(channel_specific_condition, loc));

    // Ignore IO ops that are definitely condition = 0
    // XLS opt also does this down-stream, but we try to do it here
    // for cases like "if(constexpr) {ch.read();} else {ch.write();}
    // which otherwise confuse XLS[cc] itself.
    bool do_default = false;

    absl::StatusOr<xls::Value> eval_result =
        EvaluateBVal(channel_specific_condition, loc, /*do_check=*/false);
    if (eval_result.ok()) {
      if (eval_result.value().IsAllZeros()) {
        do_default = true;
      }
    }

    IOOp op;

    if (op_name == "read") {
      if (channel->memory_size <= 0) {  // channel read()
        if (call->getNumArgs() == 1) {
          assign_ret_value_to = call->getArg(0);
        } else if (call->getNumArgs() != 0) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "IO read() should have one or zero argument(s)"));
        }
        op.op = OpType::kRecv;
        op.ret_value = channel_specific_condition;
        op.is_blocking = true;
      } else {  // memory read(addr)
        if (call->getNumArgs() != 1) {
          return absl::UnimplementedError(
              ErrorMessage(loc, "Memory read() should have one argument"));
        }
        CValue addr_val_unconverted = arg_vals.at(0);
        XLSCC_CHECK(addr_val_unconverted.valid(), loc);
        XLS_ASSIGN_OR_RETURN(
            TrackedBValue addr_val,
            GenTypeConvert(
                addr_val_unconverted,
                CChannelType::MemoryAddressType(channel->memory_size), loc));
        op.op = OpType::kRead;
        op.ret_value =
            context().fb->Tuple({addr_val, channel_specific_condition}, loc,
                                /*name=*/
                                absl::StrFormat("%s_op%i", channel->unique_name,
                                                channel->total_ops));
        op.is_blocking = true;
      }
    } else if (op_name == "nb_read") {
      if (call->getNumArgs() != 1) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "IO nb_read() should have one argument"));
      }
      assign_ret_value_to = call->getArg(0);
      op.op = OpType::kRecv;
      op.ret_value = channel_specific_condition;
      op.is_blocking = false;
    } else if (op_name == "write") {
      if (channel->memory_size <= 0) {  // channel write()
        if (call->getNumArgs() != 1) {
          return absl::UnimplementedError(
              ErrorMessage(loc, "IO write() should have one argument"));
        }

        CValue out_val = arg_vals.at(0);
        XLSCC_CHECK(out_val.valid(), loc);

        std::vector<TrackedBValue> sp = {out_val.rvalue(),
                                         channel_specific_condition};
        op.ret_value =
            context().fb->Tuple(ToNativeBValues(sp), loc,
                                /*name=*/
                                absl::StrFormat("%s_op%i", channel->unique_name,
                                                channel->total_ops));
        op.op = OpType::kSend;
        op.is_blocking = true;
      } else {  // memory write(addr, value)
        if (call->getNumArgs() != 2) {
          return absl::UnimplementedError(
              ErrorMessage(loc, "Memory write() should have two arguments"));
        }

        CValue addr_val_unconverted = arg_vals.at(0);
        XLSCC_CHECK(addr_val_unconverted.valid(), loc);
        XLS_ASSIGN_OR_RETURN(
            TrackedBValue addr_val,
            GenTypeConvert(
                addr_val_unconverted,
                CChannelType::MemoryAddressType(channel->memory_size), loc));
        CValue data_val = arg_vals.at(1);
        XLSCC_CHECK(data_val.valid(), loc);
        auto addr_val_tup = context().fb->Tuple(
            {addr_val, data_val.rvalue()}, loc,
            /*name=*/
            absl::StrFormat("addr_%s_%i", channel->unique_name,
                            channel->total_ops));

        std::vector<TrackedBValue> sp = {addr_val_tup,
                                         channel_specific_condition};
        op.ret_value =
            context().fb->Tuple(ToNativeBValues(sp), loc,
                                /*name=*/
                                absl::StrFormat("%s_op%i", channel->unique_name,
                                                channel->total_ops));
        op.op = OpType::kWrite;
        op.is_blocking = true;
      }
    } else if (op_name == "__memory_by_operator") {
      CValue addr_val = arg_vals.at(1);
      CHECK(addr_val.valid());

      const bool is_write = assignment_value.valid();

      op.is_blocking = true;

      XLS_ASSIGN_OR_RETURN(
          TrackedBValue addr_val_converted,
          GenTypeConvert(addr_val,
                         CChannelType::MemoryAddressType(channel->memory_size),
                         loc));

      if (is_write) {
        op.op = OpType::kWrite;
        CHECK(assignment_value.rvalue().valid());

        auto addr_val_tup = context().fb->Tuple(
            {addr_val_converted, assignment_value.rvalue()}, loc,
            /*name=*/
            absl::StrFormat("addr_%s_%i", channel->unique_name,
                            channel->total_ops));

        std::vector<TrackedBValue> sp = {addr_val_tup,
                                         channel_specific_condition};
        op.ret_value =
            context().fb->Tuple(ToNativeBValues(sp), loc,
                                /*name=*/
                                absl::StrFormat("%s_op%i", channel->unique_name,
                                                channel->total_ops));

      } else {
        op.op = OpType::kRead;
        std::vector<TrackedBValue> sp = {addr_val_converted,
                                         channel_specific_condition};
        op.ret_value =
            context().fb->Tuple(ToNativeBValues(sp), loc,
                                /*name=*/
                                absl::StrFormat("%s_op%i", channel->unique_name,
                                                channel->total_ops));
      }
    } else if (op_name == "size") {
      XLSCC_CHECK_GT(channel->memory_size, 0, loc);
      CValue value(context().fb->Literal(xls::UBits(channel->memory_size, 64)),
                   std::make_shared<CIntType>(64, /*is_signed=*/false));
      IOOpReturn const_ret = {.generate_expr = false, .value = value};
      return const_ret;
    } else {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Unsupported IO op: %s", op_name));
    }

    XLS_ASSIGN_OR_RETURN(IOOp * op_ptr,
                         AddOpToChannel(op, channel, loc, /*mask=*/do_default));
    (void)op_ptr;

    if (!ret.value.valid()) {
      ret.value = op.input_value;
    } else if (op.input_value.valid()) {
      // Combine values
      // TODO(seanhaskell): The conditions are always mutually exclusive.
      // Generate an assert?
      XLSCC_CHECK(*ret.value.type() == *op.input_value.type(), loc);
      ret.value =
          CValue(context().fb->Select(this_channel_condition,
                                      /*on_true=*/op.input_value.rvalue(),
                                      /*on_false=*/ret.value.rvalue(), loc),
                 ret.value.type());
    }

    XLSCC_CHECK(assign_ret_value_to == all_assign_ret_value_to ||
                    all_assign_ret_value_to == nullptr,
                loc);
    all_assign_ret_value_to = assign_ret_value_to;
    XLSCC_CHECK(!all_is_blocking.has_value() ||
                    all_is_blocking.value() == op.is_blocking,
                loc);
    all_is_blocking = op.is_blocking;
  }

  // Assign to parameter if requested
  if (all_assign_ret_value_to != nullptr) {
    XLSCC_CHECK(all_is_blocking.has_value(), loc);
    if (all_is_blocking.value()) {
      XLS_RETURN_IF_ERROR(Assign(all_assign_ret_value_to, ret.value, loc));
    } else {
      TrackedBValue read_ready =
          context().fb->TupleIndex(ret.value.rvalue(), 1);
      TrackedBValue read_value =
          context().fb->TupleIndex(ret.value.rvalue(), 0);

      if (!ret.value.type()->Is<CInternalTuple>()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unsupported IO op: %s", op_name));
      }

      CValue ret_object_value;
      ret_object_value = CValue(
          read_value, ret.value.type()->As<CInternalTuple>()->fields()[0]);

      {
        PushContextGuard condition_guard(*this, loc);
        XLS_RETURN_IF_ERROR(and_condition(read_ready, loc));
        XLS_RETURN_IF_ERROR(
            Assign(all_assign_ret_value_to, ret_object_value, loc));
      }
      CValue ret_struct = CValue(read_ready, std::make_shared<CBoolType>());

      XLSCC_CHECK(ret_struct.valid(), loc);
      ret.value = ret_struct;
    }
  }
  return ret;
}

IOChannel* Translator::AddChannel(const IOChannel& new_channel,
                                  const xls::SourceInfo& loc) {
  context().sf->io_channels.push_back(new_channel);
  IOChannel* ret = &context().sf->io_channels.back();

  ret->unique_name = GetUniqueChannelName(ret->unique_name);

  // Assertion for linear scan. If it fires, consider changing data structure
  XLSCC_CHECK_LE(context().sf->io_channels.size(), 128, loc);
  for (const IOChannel& existing_channel : context().sf->io_channels) {
    if (&existing_channel == ret) {
      continue;
    }
    XLSCC_CHECK_NE(ret->unique_name, existing_channel.unique_name, loc);
  }

  return ret;
}

std::string Translator::GetUniqueChannelName(const std::string& name) {
  std::string ret = name;
  while (used_channel_names_.contains(ret)) {
    ret = absl::StrFormat("%s__%i", ret, next_channel_number_++);
  }
  CHECK(!used_channel_names_.contains(ret));
  used_channel_names_.insert(ret);
  return ret;
}

absl::StatusOr<std::shared_ptr<LValue>> Translator::CreateChannelParam(
    const clang::NamedDecl* channel_name,
    const std::shared_ptr<CChannelType>& channel_type, bool declare_variable,
    const xls::SourceInfo& loc) {
  XLSCC_CHECK_NE(channel_name, nullptr, loc);
  IOChannel new_channel;

  new_channel.item_type = channel_type->GetItemType();
  new_channel.unique_name = channel_name->getNameAsString();

  new_channel.memory_size = channel_type->GetMemorySize();

  auto lvalue = std::make_shared<LValue>(AddChannel(new_channel, loc));

  if (!new_channel.generated.has_value()) {
    CHECK_NE(channel_name, nullptr);
    CHECK(!context().sf->lvalues_by_param.contains(channel_name));

    if (channel_name != nullptr) {
      context().sf->lvalues_by_param[channel_name] = lvalue;
    }
  }

  if (!declare_variable) {
    return lvalue;
  }

  CValue cval(/*rvalue=*/TrackedBValue(), channel_type,
              /*disable_type_check=*/true, lvalue);

  XLS_RETURN_IF_ERROR(DeclareVariable(channel_name, cval, loc));

  return lvalue;
}

absl::StatusOr<TrackedBValue> Translator::AddConditionToIOReturn(
    const IOOp& op, TrackedBValue retval, const xls::SourceInfo& loc) {
  TrackedBValue op_condition;

  switch (op.op) {
    case OpType::kNull:
    case OpType::kSendRecv:
      XLSCC_CHECK(
          "AddConditionToIOReturn() unsupported for Null and InOut "
          "directions" == nullptr,
          loc);
      break;
    case OpType::kLoopEndJump:
    case OpType::kLoopBegin: {
      op_condition = retval;
      break;
    }
    case OpType::kRecv:
      op_condition = retval;
      break;
    case OpType::kSend:
    case OpType::kWrite:
    case OpType::kRead:
      op_condition = context().fb->TupleIndex(retval, /*idx=*/1, loc);
      break;
    case OpType::kTrace: {
      switch (op.trace_type) {
        case TraceType::kNull:
          break;
        case TraceType::kAssert:
          op_condition = retval;
          break;
        case TraceType::kTrace:
          op_condition = context().fb->TupleIndex(retval, /*idx=*/0, loc);
          break;
      }
      if (!op_condition.valid()) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Unsupported trace type %i in AddConditionToIOReturn",
            op.trace_type));
      }
      break;
    }
  }

  if (!op_condition.valid()) {
    return absl::UnimplementedError(ErrorMessage(
        loc, "Unsupported IO op %i in AddConditionToIOReturn", op.op));
  }

  op_condition =
      context().fb->And(op_condition, context().full_condition_bval(loc), loc);

  // Short circuit the op condition if possible
  XLS_RETURN_IF_ERROR(ShortCircuitBVal(op_condition, loc));

  TrackedBValue new_retval;

  switch (op.op) {
    case OpType::kNull:
    case OpType::kSendRecv:
      XLSCC_CHECK(
          "AddConditionToIOReturn() unsupported for Null and InOut "
          "directions" == nullptr,
          loc);
      break;
    case OpType::kLoopBegin:
    case OpType::kLoopEndJump: {
      new_retval = op_condition;
      break;
    }
    case OpType::kRecv:
      new_retval = op_condition;
      break;
    case OpType::kSend:
    case OpType::kWrite:
    case OpType::kRead: {
      TrackedBValue data = context().fb->TupleIndex(retval, /*idx=*/0, loc);
      new_retval = context().fb->Tuple({data, op_condition}, loc);
      break;
    }
    case OpType::kTrace: {
      switch (op.trace_type) {
        case TraceType::kNull:
          break;
        case TraceType::kAssert:
          new_retval = op_condition;
          break;
        case TraceType::kTrace: {
          const uint64_t tuple_count = retval.GetType()->AsTupleOrDie()->size();
          std::vector<TrackedBValue> tuple_parts = {op_condition};
          for (int i = 1; i < tuple_count; ++i) {
            tuple_parts.push_back(
                context().fb->TupleIndex(retval, /*idx=*/i, loc));
          }
          new_retval = context().fb->Tuple(ToNativeBValues(tuple_parts), loc);
          break;
        }
      }
      if (!new_retval.valid()) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Unsupported trace type %i in AddConditionToIOReturn",
            op.trace_type));
      }
      break;
    }
      if (!new_retval.valid()) {
        return absl::UnimplementedError(ErrorMessage(
            loc, "Unsupported IO op %i in AddConditionToIOReturn", op.op));
      }
  }

  return new_retval;
}

absl::StatusOr<TrackedBValue> Translator::GetOpCondition(
    const IOOp& op, TrackedBValue op_out_value, xls::ProcBuilder& pb) {
  TrackedBValue ret_val = op_out_value;

  TrackedBValue io_condition;

  switch (op.op) {
    case OpType::kRecv:
      io_condition = ret_val;
      break;
    case OpType::kTrace: {
      switch (op.trace_type) {
        case TraceType::kNull:
          break;
        case TraceType::kTrace: {
          const uint64_t tuple_count =
              ret_val.GetType()->AsTupleOrDie()->size();
          XLSCC_CHECK_GE(tuple_count, 1, op.op_location);
          io_condition =
              pb.TupleIndex(ret_val, 0, op.op_location,
                            absl::StrFormat("%s_pred", Debug_OpName(op)));
          break;
        }
        case TraceType::kAssert: {
          io_condition = pb.Not(ret_val, op.op_location);
          break;
        }
      }
      break;
    }
    case OpType::kRead:
    case OpType::kWrite:
    case OpType::kSend: {
      io_condition =
          pb.TupleIndex(ret_val, 1, op.op_location,
                        absl::StrFormat("%s_pred", Debug_OpName(op)));
      break;
    }
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported op type in GetOpCondition: %s", Debug_OpName(op)));
  }

  XLSCC_CHECK(io_condition.valid(), op.op_location);
  XLSCC_CHECK(io_condition.GetType()->IsBits(), op.op_location);
  XLSCC_CHECK_EQ(io_condition.GetType()->GetFlatBitCount(), 1, op.op_location);

  io_condition =
      pb.And(io_condition, context().full_condition_bval(op.op_location),
             op.op_location);

  return io_condition;
}

absl::StatusOr<GenerateIOReturn> Translator::GenerateIO(
    const IOOp& op, TrackedBValue before_token, TrackedBValue op_out_value,
    xls::ProcBuilder& pb,
    const std::optional<ChannelBundle>& optional_channel_bundle,
    std::optional<TrackedBValue> extra_condition) {
  xls::SourceInfo op_loc = op.full_op_location;

  TrackedBValue ret_io_value = op_out_value;

  TrackedBValue arg_io_val;

  TrackedBValue new_token;
  ChannelBundle channel_bundle;

  TrackedBValue condition;

  XLS_ASSIGN_OR_RETURN(condition, GetOpCondition(op, op_out_value, pb));

  condition = ConditionWithExtra(pb, condition, extra_condition, op_loc);

  if (optional_channel_bundle.has_value()) {
    channel_bundle = optional_channel_bundle.value();
  }

  if (op.op == OpType::kRecv) {
    xls::Channel* xls_channel = channel_bundle.regular;

    unused_xls_channel_ops_.remove({xls_channel, /*is_send=*/false});

    CHECK_NE(xls_channel, nullptr);

    TrackedBValue receive;
    if (op.is_blocking) {
      receive = pb.ReceiveIf(xls_channel, before_token, condition, op_loc,
                             /*name=*/Debug_OpName(op));
    } else {
      receive = pb.ReceiveIfNonBlocking(xls_channel, before_token, condition,
                                        op_loc, /*name=*/Debug_OpName(op));
    }
    new_token =
        pb.TupleIndex(receive, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_token", Debug_OpName(op)));

    TrackedBValue in_val;
    if (op.is_blocking) {
      in_val =
          pb.TupleIndex(receive, 1, op_loc,
                        /*name=*/absl::StrFormat("%s_value", Debug_OpName(op)));
    } else {
      in_val = pb.Tuple(
          {pb.TupleIndex(
               receive, 1, op_loc,
               /*name=*/absl::StrFormat("%s_value", Debug_OpName(op))),
           pb.TupleIndex(
               receive, 2, op_loc,
               /*name=*/absl::StrFormat("%s_valid", Debug_OpName(op)))});
    }
    arg_io_val = in_val;
  } else if (op.op == OpType::kSend) {
    xls::Channel* xls_channel = channel_bundle.regular;

    unused_xls_channel_ops_.remove({xls_channel, /*is_send=*/true});

    CHECK_NE(xls_channel, nullptr);
    TrackedBValue val =
        pb.TupleIndex(ret_io_value, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_value", Debug_OpName(op)));

    new_token = pb.SendIf(xls_channel, before_token, condition, val, op_loc,
                          /*name=*/absl::StrFormat("%s", Debug_OpName(op)));
  } else if (op.op == OpType::kRead) {
    CHECK_EQ(channel_bundle.regular, nullptr);
    CHECK_NE(channel_bundle.read_request, nullptr);
    CHECK_NE(channel_bundle.read_response, nullptr);

    unused_xls_channel_ops_.remove(
        {channel_bundle.read_request, /*is_send=*/true});
    unused_xls_channel_ops_.remove(
        {channel_bundle.read_response, /*is_send=*/false});

    TrackedBValue addr =
        pb.TupleIndex(ret_io_value, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_addr", Debug_OpName(op)));

    // TODO(google/xls#861): supported masked memory operations.
    TrackedBValue mask = pb.Literal(xls::Value::Tuple({}), op_loc);
    TrackedBValue send_tuple_with_mask = pb.Tuple({addr, mask}, op_loc);
    new_token = pb.SendIf(channel_bundle.read_request, before_token, condition,
                          send_tuple_with_mask, op_loc,
                          /*name=*/absl::StrFormat("%s", Debug_OpName(op)));

    TrackedBValue receive =
        pb.ReceiveIf(channel_bundle.read_response, new_token, condition, op_loc,
                     /*name=*/Debug_OpName(op));

    new_token =
        pb.TupleIndex(receive, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_token", Debug_OpName(op)));
    TrackedBValue response_tup = pb.TupleIndex(
        receive, 1, op_loc,
        /*name=*/absl::StrFormat("%s_response_tup", Debug_OpName(op)));
    TrackedBValue response = pb.TupleIndex(
        response_tup, 0, op_loc,
        /*name=*/absl::StrFormat("%s_response", Debug_OpName(op)));

    arg_io_val = response;
  } else if (op.op == OpType::kWrite) {
    CHECK_EQ(channel_bundle.regular, nullptr);
    CHECK_NE(channel_bundle.write_request, nullptr);
    CHECK_NE(channel_bundle.write_response, nullptr);

    unused_xls_channel_ops_.remove(
        {channel_bundle.write_request, /*is_send=*/true});
    unused_xls_channel_ops_.remove(
        {channel_bundle.write_response, /*is_send=*/false});

    // This has (addr, value)
    TrackedBValue send_tuple = pb.TupleIndex(
        ret_io_value, 0, op_loc,
        /*name=*/absl::StrFormat("%s_send_tup", Debug_OpName(op)));

    // This has (addr, value, mask)
    TrackedBValue addr =
        pb.TupleIndex(send_tuple, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_addr", Debug_OpName(op)));
    TrackedBValue value =
        pb.TupleIndex(send_tuple, 1, op_loc,
                      /*name=*/absl::StrFormat("%s_value", Debug_OpName(op)));
    // TODO(google/xls#861): supported masked memory operations.
    TrackedBValue mask = pb.Literal(xls::Value::Tuple({}), op_loc);
    TrackedBValue send_tuple_with_mask = pb.Tuple({addr, value, mask}, op_loc);
    new_token = pb.SendIf(channel_bundle.write_request, before_token, condition,
                          send_tuple_with_mask, op_loc,
                          /*name=*/absl::StrFormat("%s", Debug_OpName(op)));

    TrackedBValue receive = pb.ReceiveIf(
        channel_bundle.write_response, new_token, condition, op_loc,
        /*name=*/absl::StrFormat("%s", Debug_OpName(op)));
    new_token =
        pb.TupleIndex(receive, 0, op_loc,
                      /*name=*/absl::StrFormat("%s_token", Debug_OpName(op)));
    // Ignore received value, should be an empty tuple
  } else if (op.op == OpType::kTrace) {
    // Condition not recorded
    XLS_ASSIGN_OR_RETURN(new_token, GenerateTrace(ret_io_value, before_token,
                                                  condition, op, pb));
  } else if (op.op == OpType::kLoopBegin || op.op == OpType::kLoopEndJump) {
    // TODO: Need to do anything here?
    new_token = before_token;
  } else {
    CHECK_EQ("Unknown IOOp type", nullptr);
  }

  return GenerateIOReturn{
      .token_out = new_token,
      .received_value = arg_io_val,
      .io_condition = condition,
  };
}

absl::StatusOr<TrackedBValue> Translator::GenerateTrace(
    TrackedBValue trace_out_value, TrackedBValue before_token,
    TrackedBValue condition, const IOOp& op, xls::ProcBuilder& pb) {
  switch (op.trace_type) {
    case TraceType::kNull:
      break;
    case TraceType::kTrace: {
      // Tuple is (condition, ... args ...)
      const uint64_t tuple_count =
          trace_out_value.GetType()->AsTupleOrDie()->size();
      CHECK_GE(tuple_count, 1);

      std::vector<TrackedBValue> args;
      for (int tuple_idx = 1; tuple_idx < tuple_count; ++tuple_idx) {
        TrackedBValue arg = pb.TupleIndex(
            trace_out_value, tuple_idx, op.op_location,
            /*name=*/absl::StrFormat("%s_arg_%i", Debug_OpName(op), tuple_idx));
        args.push_back(arg);
      }

      return pb.Trace(before_token, /*condition=*/condition,
                      ToNativeBValues(args),
                      /*format_string=*/op.trace_message_string,
                      /*verbosity=*/0, op.op_location);
    }
    case TraceType::kAssert: {
      // Assert condition is !fire
      return pb.Assert(before_token,
                       /*condition=*/condition,
                       /*message=*/op.trace_message_string,
                       /*label=*/op.label_string.empty()
                           ? std::nullopt
                           : std::optional<std::string>(op.label_string),
                       op.op_location);
    }
  }
  return absl::InternalError(
      ErrorMessage(op.op_location, "Unknown trace type %i", op.trace_type));
}

}  // namespace xlscc
