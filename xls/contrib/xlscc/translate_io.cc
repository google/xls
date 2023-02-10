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

#include "absl/status/status.h"
#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/ExprCXX.h"
#include "clang/include/clang/AST/OperationKinds.h"
#include "clang/include/clang/Basic/OperatorKinds.h"
#include "xls/common/status/status_macros.h"
#include "xls/contrib/xlscc/translator.h"

namespace xlscc {

absl::StatusOr<IOOp*> Translator::AddOpToChannel(IOOp& op, IOChannel* channel,
                                                 const xls::SourceInfo& loc,
                                                 bool mask) {
  context().any_side_effects_requested = true;

  if (context().mask_side_effects || mask) {
    IOOpReturn ret;
    ret.generate_expr = false;
    XLS_ASSIGN_OR_RETURN(xls::BValue default_bval,
                         CreateDefaultValue(channel->item_type, loc));
    op.input_value = CValue(default_bval, channel->item_type);
    return &op;
  }

  XLS_CHECK_NE(channel, nullptr);
  XLS_CHECK_EQ(op.channel, nullptr);
  op.channel_op_index = channel->total_ops++;
  op.channel = channel;
  op.op_location = loc;

  // Operation type is added late, as it's only known from the read()/write()
  // call(s)
  if (channel->channel_op_type == OpType::kNull) {
    channel->channel_op_type = op.op;
  } else {
    if (channel->channel_op_type != op.op) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Channels should be either input or output"));
    }
  }

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

absl::StatusOr<std::shared_ptr<CChannelType>> Translator::GetChannelType(
    const clang::QualType& channel_type, clang::ASTContext& ctx,
    const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                       StripTypeQualifiers(channel_type));
  auto template_spec = clang::dyn_cast<const clang::TemplateSpecializationType>(
      stripped.base.getTypePtr());
  if (template_spec == nullptr) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel type should be a template specialization"));
  }
  const std::string template_name =
      template_spec->getTemplateName().getAsTemplateDecl()->getNameAsString();
  if ((template_spec->template_arguments().size() != 1) &&
      (template_spec->template_arguments().size() != 2)) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel should have 1 or 2 template args"));
  }
  OpType op_type = OpType::kNull;
  int64_t memory_size = -1;

  // TODO(seanhaskell): Put these strings in one place
  if (template_name == "__xls_memory") {
    if (template_spec->template_arguments().size() != 2) {
      return absl::UnimplementedError(
          ErrorMessage(loc, "Memory should have 2 template args"));
    }
    const clang::TemplateArgument& arg = template_spec->template_arguments()[1];
    XLS_ASSIGN_OR_RETURN(memory_size,
                         EvaluateInt64(*arg.getAsExpr(), ctx, loc));
  } else if (template_name == "__xls_channel") {
    if (template_spec->template_arguments().size() == 2) {
      const clang::TemplateArgument& arg =
          template_spec->template_arguments()[1];
      XLS_ASSIGN_OR_RETURN(int64_t val,
                           EvaluateInt64(*arg.getAsExpr(), ctx, loc));
      op_type = static_cast<OpType>(val);
    }
  }
  const clang::TemplateArgument& arg = template_spec->template_arguments()[0];
  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> item_type,
                       TranslateTypeFromClang(arg.getAsType(), loc));

  return std::make_shared<CChannelType>(item_type, op_type, memory_size);
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
        if (call->getNumArgs() == 1) {
          assign_ret_value_to = call->getArg(0);
        } else if (call->getNumArgs() != 0) {
          return absl::UnimplementedError(ErrorMessage(
              loc, "IO read() should have one or zero argument(s)"));
        }
        op.op = OpType::kRecv;
        op.ret_value = op_condition;
        op.is_blocking = true;
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

}  // namespace xlscc
