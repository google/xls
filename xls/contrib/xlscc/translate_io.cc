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

#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/ExprCXX.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/AST/Type.h"
#include "clang/include/clang/Basic/OperatorKinds.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"
#include "xls/contrib/xlscc/xlscc_logging.h"

namespace xlscc {

absl::StatusOr<IOOp*> Translator::AddOpToChannel(IOOp& op, IOChannel* channel,
                                                 const xls::SourceInfo& loc,
                                                 bool mask) {
  context().any_side_effects_requested = true;
  context().any_io_ops_requested = true;

  const bool mask_write =
      context().mask_memory_writes && op.op == OpType::kWrite;
  const bool mask_other =
      context().mask_io_other_than_memory_writes && op.op != OpType::kWrite;

  if (mask || context().mask_side_effects || mask_write || mask_other) {
    IOOpReturn ret;
    ret.generate_expr = false;
    XLS_ASSIGN_OR_RETURN(xls::BValue default_bval,
                         CreateDefaultValue(channel->item_type, loc));
    op.input_value = CValue(default_bval, channel->item_type);
    return nullptr;
  }

  if (op.op == OpType::kWrite) {
    context().any_writes_generated |= true;
  }

  XLS_CHECK_NE(channel, nullptr);
  XLS_CHECK_EQ(op.channel, nullptr);
  op.channel_op_index = channel->total_ops++;
  op.channel = channel;
  op.op_location = loc;

  // Channel must be inserted first by AddOpToChannel
  if (op.op == OpType::kRecv || op.op == OpType::kRead) {
    xls::Type* xls_item_type;
    std::shared_ptr<CType> channel_item_type;
    if (op.is_blocking) {
      XLS_ASSIGN_OR_RETURN(xls_item_type,
                           TranslateTypeToXLS(channel->item_type, loc));
      channel_item_type = channel->item_type;
    } else {
      XLS_ASSIGN_OR_RETURN(xls::Type * primary_item_type,
                           TranslateTypeToXLS(channel->item_type, loc));
      xls_item_type =
          package_->GetTupleType({primary_item_type, package_->GetBitsType(1)});

      channel_item_type =
          std::make_shared<CInternalTuple>(std::vector<std::shared_ptr<CType>>(
              {channel->item_type, std::make_shared<CBoolType>()}));
    }
    const int64_t channel_op_index = op.channel_op_index;

    std::string safe_param_name =
        absl::StrFormat("%s_op%i", op.channel->unique_name, channel_op_index);

    xls::BValue pbval =
        context().fb->Param(safe_param_name, xls_item_type, loc);

    // Check for duplicate params
    if (!pbval.valid()) {
      return absl::InternalError(ErrorMessage(
          loc,
          "Failed to create implicit parameter %s, duplicate? See b/239861050",
          safe_param_name.c_str()));
    }

    op.input_value = CValue(pbval, channel_item_type);
  }

  context().sf->io_ops.push_back(op);

  if (op.op == OpType::kRecv || op.op == OpType::kRead) {
    SideEffectingParameter side_effecting_param;
    side_effecting_param.type = SideEffectingParameterType::kIOOp;
    side_effecting_param.param_name =
        op.input_value.rvalue().node()->As<xls::Param>()->GetName();
    side_effecting_param.io_op = &context().sf->io_ops.back();
    context().sf->side_effecting_parameters.push_back(side_effecting_param);
  }

  return &context().sf->io_ops.back();
}

absl::StatusOr<bool> Translator::TypeIsChannel(clang::QualType param,
                                               const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(StrippedType stripped, StripTypeQualifiers(param));
  absl::StatusOr<std::shared_ptr<CType>> obj_type_ret =
      TranslateTypeFromClang(stripped.base, loc);

  // Ignore un-translatable types like pointers
  if (!obj_type_ret.ok()) {
    return false;
  }

  return obj_type_ret.value()->Is<CChannelType>();
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
  } else if (auto record = clang::dyn_cast<const clang::RecordType>(
                 stripped.base.getTypePtr());
             record != nullptr) {
    clang::RecordDecl* decl = record->getDecl();
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

absl::Status Translator::CreateChannelParam(
    const clang::NamedDecl* channel_name,
    const std::shared_ptr<CChannelType>& channel_type,
    const xls::SourceInfo& loc) {
  IOChannel new_channel;

  new_channel.item_type = channel_type->GetItemType();
  new_channel.unique_name = channel_name->getNameAsString();
  new_channel.memory_size = channel_type->GetMemorySize();

  auto lvalue = std::make_shared<LValue>(AddChannel(channel_name, new_channel));
  CValue cval(/*rvalue=*/xls::BValue(), channel_type,
              /*disable_type_check=*/true, lvalue);

  XLS_RETURN_IF_ERROR(DeclareVariable(channel_name, cval, loc));

  return absl::OkStatus();
}

absl::StatusOr<Translator::IOOpReturn> Translator::InterceptIOOp(
    const clang::Expr* expr, const xls::SourceInfo& loc,
    const CValue assignment_value) {
  xls::BValue op_condition = context().full_condition_bval(loc);
  XLS_CHECK(op_condition.valid());
  if (auto operator_call =
          clang::dyn_cast<const clang::CXXOperatorCallExpr>(expr)) {
    XLS_ASSIGN_OR_RETURN(IOChannel * channel,
                         GetChannelForExprOrNull(operator_call->getArg(0)));

    XLS_ASSIGN_OR_RETURN(
        bool type_is_channel,
        TypeIsChannel(operator_call->getArg(0)->getType(), loc));
    if (type_is_channel && channel == nullptr) {
      return absl::UnimplementedError(
          ErrorMessage(loc,
                       "Method call on channel type, but no channel found "
                       "(uninitialized?)"));
    }

    if (channel != nullptr) {
      if (operator_call->getOperator() ==
          clang::OverloadedOperatorKind::OO_Subscript) {
        XLS_ASSIGN_OR_RETURN(CValue addr_val,
                             GenerateIR_Expr(operator_call->getArg(1), loc));
        XLS_CHECK(addr_val.valid());

        const bool is_write = assignment_value.valid();

        IOOp op;

        op.is_blocking = true;

        XLS_ASSIGN_OR_RETURN(
            xls::BValue addr_val_converted,
            GenTypeConvert(
                addr_val, CChannelType::MemoryAddressType(channel->memory_size),
                loc));

        if (is_write) {
          op.op = OpType::kWrite;
          XLS_CHECK(assignment_value.rvalue().valid());

          auto addr_val_tup = context().fb->Tuple(
              {addr_val_converted, assignment_value.rvalue()}, loc);

          std::vector<xls::BValue> sp = {addr_val_tup, op_condition};
          op.ret_value = context().fb->Tuple(sp, loc);

        } else {
          op.op = OpType::kRead;
          std::vector<xls::BValue> sp = {addr_val_converted, op_condition};
          op.ret_value = context().fb->Tuple(sp, loc);
        }

        // TODO(seanhaskell): Short circuit as with IO ops?
        const bool do_default = false;
        XLS_ASSIGN_OR_RETURN(
            IOOp * op_ptr,
            AddOpToChannel(op, channel, loc, /*mask=*/do_default));
        (void)op_ptr;

        IOOpReturn ret;
        ret.generate_expr = false;
        ret.value = op.input_value;

        return ret;
      }
      return absl::UnimplementedError(ErrorMessage(
          loc, "Unsupported IO op from operator: %s",
          clang::getOperatorSpelling(operator_call->getOperator())));
    }
  }

  if (auto member_call =
          clang::dyn_cast<const clang::CXXMemberCallExpr>(expr)) {
    const clang::Expr* object = member_call->getImplicitObjectArgument();

    XLS_ASSIGN_OR_RETURN(IOChannel * channel, GetChannelForExprOrNull(object));

    XLS_ASSIGN_OR_RETURN(bool type_is_channel,
                         TypeIsChannel(object->getType(), loc));
    if (type_is_channel && channel == nullptr) {
      return absl::InvalidArgumentError(
          ErrorMessage(loc,
                       "Method call on channel type, but no channel found "
                       "(uninitialized?)"));
    }

    if (channel != nullptr) {
      if (assignment_value.valid()) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Assignment to IO ops not supported"));
      }

      const clang::FunctionDecl* funcdecl = member_call->getDirectCallee();
      const std::string op_name = funcdecl->getNameAsString();

      // Short circuit the op condition if possible
      XLS_RETURN_IF_ERROR(ShortCircuitBVal(op_condition, loc));

      // Ignore IO ops that are definitely condition = 0
      // XLS opt also does this down-stream, but we try to do it here
      // for cases like "if(constexpr) {ch.read();} else {ch.write();}
      // which otherwise confuse XLS[cc] itself.
      bool do_default = false;

      absl::StatusOr<xls::Value> eval_result =
          EvaluateBVal(op_condition, loc, /*do_check=*/false);
      if (eval_result.ok()) {
        if (eval_result.value().IsAllZeros()) {
          do_default = true;
        }
      }

      auto call = clang::dyn_cast<const clang::CallExpr>(expr);

      IOOpReturn ret;
      ret.generate_expr = false;

      IOOp op;
      const clang::Expr* assign_ret_value_to = nullptr;

      if (op_name == "read") {
        if (channel->memory_size <= 0) {  // channel read()
          if (call->getNumArgs() == 1) {
            assign_ret_value_to = call->getArg(0);
          } else if (call->getNumArgs() != 0) {
            return absl::UnimplementedError(ErrorMessage(
                loc, "IO read() should have one or zero argument(s)"));
          }
          op.op = OpType::kRecv;
          op.ret_value = op_condition;
          op.is_blocking = true;
        } else {  // memory read(addr)
          if (call->getNumArgs() != 1) {
            return absl::UnimplementedError(
                ErrorMessage(loc, "Memory read() should have one argument"));
          }
          XLS_ASSIGN_OR_RETURN(CValue addr_val_unconverted,
                               GenerateIR_Expr(call->getArg(0), loc));
          XLS_ASSIGN_OR_RETURN(
              xls::BValue addr_val,
              GenTypeConvert(
                  addr_val_unconverted,
                  CChannelType::MemoryAddressType(channel->memory_size), loc));
          op.op = OpType::kRead;
          op.ret_value = context().fb->Tuple({addr_val, op_condition}, loc);
          op.is_blocking = true;
        }
      } else if (op_name == "nb_read") {
        if (call->getNumArgs() != 1) {
          return absl::UnimplementedError(
              ErrorMessage(loc, "IO nb_read() should have one argument"));
        }
        assign_ret_value_to = call->getArg(0);
        op.op = OpType::kRecv;
        op.ret_value = op_condition;
        op.is_blocking = false;
      } else if (op_name == "write") {
        if (channel->memory_size <= 0) {  // channel write()
          if (call->getNumArgs() != 1) {
            return absl::UnimplementedError(
                ErrorMessage(loc, "IO write() should have one argument"));
          }

          XLS_ASSIGN_OR_RETURN(CValue out_val,
                               GenerateIR_Expr(call->getArg(0), loc));

          std::vector<xls::BValue> sp = {out_val.rvalue(), op_condition};
          op.ret_value = context().fb->Tuple(sp, loc);
          op.op = OpType::kSend;
          op.is_blocking = true;
        } else {  // memory write(addr, value)
          if (call->getNumArgs() != 2) {
            return absl::UnimplementedError(
                ErrorMessage(loc, "Memory write() should have two arguments"));
          }

          XLS_ASSIGN_OR_RETURN(CValue addr_val_unconverted,
                               GenerateIR_Expr(call->getArg(0), loc));
          XLS_ASSIGN_OR_RETURN(
              xls::BValue addr_val,
              GenTypeConvert(
                  addr_val_unconverted,
                  CChannelType::MemoryAddressType(channel->memory_size), loc));
          XLS_ASSIGN_OR_RETURN(CValue data_val,
                               GenerateIR_Expr(call->getArg(1), loc));
          auto addr_val_tup =
              context().fb->Tuple({addr_val, data_val.rvalue()}, loc);

          std::vector<xls::BValue> sp = {addr_val_tup, op_condition};
          op.ret_value = context().fb->Tuple(sp, loc);
          op.op = OpType::kWrite;
          op.is_blocking = true;
        }
      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unsupported IO op: %s", op_name));
      }

      XLS_ASSIGN_OR_RETURN(
          IOOp * op_ptr, AddOpToChannel(op, channel, loc, /*mask=*/do_default));
      (void)op_ptr;

      ret.value = op.input_value;
      if (assign_ret_value_to != nullptr) {
        if (op.is_blocking) {
          XLS_RETURN_IF_ERROR(Assign(assign_ret_value_to, ret.value, loc));
        } else {
          xls::BValue read_ready =
              context().fb->TupleIndex(ret.value.rvalue(), 1);
          xls::BValue read_value =
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
                Assign(assign_ret_value_to, ret_object_value, loc));
          }
          CValue ret_struct = CValue(read_ready, std::make_shared<CBoolType>());
          ret.value = ret_struct;
        }
      }
      return ret;
    }
  }

  IOOpReturn ret;
  ret.generate_expr = true;
  return ret;
}

IOChannel* Translator::AddChannel(const clang::NamedDecl* decl,
                                  IOChannel new_channel) {
  context().sf->io_channels.push_back(new_channel);
  IOChannel* ret = &context().sf->io_channels.back();

  if (decl != nullptr) {
    context().sf->io_channels_by_decl[decl] = ret;
    context().sf->decls_by_io_channel[ret] = decl;
  }

  return ret;
}

absl::StatusOr<xls::BValue> Translator::AddConditionToIOReturn(
    const IOOp& op, xls::BValue retval, const xls::SourceInfo& loc) {
  xls::BValue op_condition;

  switch (op.op) {
    case OpType::kRecv:
      op_condition = retval;
      break;
    case OpType::kSend:
    case OpType::kWrite:
    case OpType::kRead:
      op_condition = context().fb->TupleIndex(retval, /*idx=*/1, loc);
      break;
    default:
      return absl::UnimplementedError(ErrorMessage(
          loc, "Unsupported IO op %i in AddConditionToIOReturn", op.op));
  }

  XLS_CHECK(op_condition.valid());

  op_condition =
      context().fb->And(op_condition, context().full_condition_bval(loc), loc);

  // Short circuit the op condition if possible
  XLS_RETURN_IF_ERROR(ShortCircuitBVal(op_condition, loc));

  xls::BValue new_retval;

  switch (op.op) {
    case OpType::kRecv:
      new_retval = op_condition;
      break;
    case OpType::kSend:
    case OpType::kWrite:
    case OpType::kRead: {
      xls::BValue data = context().fb->TupleIndex(retval, /*idx=*/0, loc);
      new_retval = context().fb->Tuple({data, op_condition}, loc);
      break;
    }
    default:
      return absl::UnimplementedError(ErrorMessage(
          loc, "Unsupported IO op %i in AddConditionToIOReturn", op.op));
  }

  XLS_CHECK(new_retval.valid());

  return new_retval;
}

}  // namespace xlscc
