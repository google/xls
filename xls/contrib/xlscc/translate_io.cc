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

#include "clang/include/clang/AST/Expr.h"
#include "clang/include/clang/AST/ExprCXX.h"
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
  if (op.op == OpType::kRecv) {
    XLS_ASSIGN_OR_RETURN(xls::Type * xls_item_type,
                         TranslateTypeToXLS(channel->item_type, loc));

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

    op.input_value = CValue(pbval, channel->item_type);
  }

  context().sf->io_ops.push_back(op);

  if (op.op == OpType::kRecv) {
    SideEffectingParameter side_effecting_param;
    side_effecting_param.type = SideEffectingParameterType::kIOOp;
    side_effecting_param.param_name =
        op.input_value.rvalue().node()->As<xls::Param>()->GetName();
    side_effecting_param.io_op = &context().sf->io_ops.back();
    context().sf->side_effecting_parameters.push_back(side_effecting_param);
  }

  return &context().sf->io_ops.back();
}

absl::StatusOr<std::shared_ptr<CType>> Translator::GetChannelType(
    const clang::ParmVarDecl* channel_param, const xls::SourceInfo& loc) {
  XLS_ASSIGN_OR_RETURN(StrippedType stripped,
                       StripTypeQualifiers(channel_param->getType()));
  auto template_spec = clang::dyn_cast<const clang::TemplateSpecializationType>(
      stripped.base.getTypePtr());
  if (template_spec == nullptr) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel type should be a template specialization"));
  }
  if (template_spec->getNumArgs() != 1) {
    return absl::UnimplementedError(
        ErrorMessage(loc, "Channel should have 1 template args"));
  }
  const clang::TemplateArgument& arg = template_spec->getArg(0);
  return TranslateTypeFromClang(arg.getAsType(), loc);
}

absl::Status Translator::CreateChannelParam(
    const clang::ParmVarDecl* channel_param, const xls::SourceInfo& loc) {
  XLS_CHECK(!context().sf->io_channels_by_param.contains(channel_param));

  XLS_ASSIGN_OR_RETURN(std::shared_ptr<CType> ctype,
                       GetChannelType(channel_param, loc));

  IOChannel new_channel;

  new_channel.item_type = ctype;
  new_channel.unique_name = channel_param->getNameAsString();

  context().sf->io_channels.push_back(new_channel);
  context().sf->io_channels_by_param[channel_param] =
      &context().sf->io_channels.back();
  context().sf->params_by_io_channel[&context().sf->io_channels.back()] =
      channel_param;

  context().channel_params.insert(channel_param);

  return absl::OkStatus();
}

absl::StatusOr<Translator::IOOpReturn> Translator::InterceptIOOp(
    const clang::Expr* expr, const xls::SourceInfo& loc) {
  if (auto member_call =
          clang::dyn_cast<const clang::CXXMemberCallExpr>(expr)) {
    const clang::Expr* object = member_call->getImplicitObjectArgument();

    XLS_ASSIGN_OR_RETURN(bool is_channel, ExprIsChannel(object, loc));
    if (is_channel) {
      // Duplicated code in GenerateIR_Call()?
      if (!clang::isa<clang::DeclRefExpr>(object)) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "IO ops should be on direct DeclRefs"));
      }
      auto object_ref = clang::dyn_cast<const clang::DeclRefExpr>(object);
      auto channel_param =
          clang::dyn_cast<const clang::ParmVarDecl>(object_ref->getDecl());
      if (channel_param == nullptr) {
        return absl::UnimplementedError(
            ErrorMessage(loc, "IO ops should be on channel parameters"));
      }
      const clang::FunctionDecl* funcdecl = member_call->getDirectCallee();
      const std::string op_name = funcdecl->getNameAsString();

      IOChannel* channel = context().sf->io_channels_by_param.at(channel_param);
      xls::BValue op_condition = context().full_condition_bval(loc);
      XLS_CHECK(op_condition.valid());

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

      } else {
        return absl::UnimplementedError(
            ErrorMessage(loc, "Unsupported IO op: %s", op_name));
      }

      XLS_ASSIGN_OR_RETURN(
          IOOp * op_ptr, AddOpToChannel(op, channel, loc, /*mask=*/do_default));
      (void)op_ptr;

      ret.value = op.input_value;
      if (assign_ret_value_to != nullptr) {
        XLS_RETURN_IF_ERROR(Assign(assign_ret_value_to, ret.value, loc));
      }

      return ret;
    }
  }

  IOOpReturn ret;
  ret.generate_expr = true;
  return ret;
}

}  // namespace xlscc
