#include "xls/ir/op_specification.h"

#include <string>

#include "absl/log/check.h"
#include "absl/container/btree_map.h"
#include "absl/strings/str_split.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_replace.h"
#include "absl/container/flat_hash_set.h"

namespace xls {

const OptionalOperand& Operand::AsOptionalOrDie() const {
  return *dynamic_cast<const OptionalOperand*>(this);
}

std::string Operand::CamelCaseName() const {
  std::vector<std::string_view> parts = absl::StrSplit(name_, '_');
  std::vector<std::string> capitalized;
  capitalized.reserve(parts.size());
  for (std::string_view part : parts) {
    capitalized.push_back(absl::StrCat(absl::AsciiStrToUpper(part.substr(0, 1)), part.substr(1)));
  }
  return absl::StrJoin(capitalized, "");
}

std::string OpClass::GetEqualToExpr() const {
  std::vector<std::string> pieces;

  for (const DataMember& data_member : GetDataMembers()) {
    std::string rhs = absl::StrFormat("other->As<%s>()->%s", name(), data_member.name);
    pieces.push_back(absl::StrReplaceAll(data_member.equals_tmpl, {
        {"{lhs}", data_member.name},
        {"{rhs}", rhs}
    }));
  }

  return absl::StrJoin(pieces, " && ");
}

std::vector<Method> OpClass::GetMethods() const {
  std::vector<Method> methods;
  methods.reserve(options_.attributes.size() + options_.extra_methods.size());
  for (const auto& attribute : options_.attributes) {
    methods.push_back(attribute->method());
  }
  for (const Method& extra : options_.extra_methods) {
    methods.push_back(extra);
  }
  return methods;
}

std::string OpClass::GetCloneArgsStr(std::string new_operands) const {
  CHECK(!custom_clone_method());
  std::vector<std::string> args = {"loc()"};
  if (operands_.size() == 1 && operands_[0]->kind() == OperandKind::kSpan) {
    args.push_back(new_operands);
  } else {
    for (size_t i = 0; i < operands_.size(); ++i) {
      args.push_back(absl::StrFormat("%s[%d]", new_operands, i));
    }
  }

  for (const auto& attribute : options_.attributes) {
    args.push_back(absl::StrFormat("%s()", attribute->name()));
  }
  for (const auto& attribute: extra_constructor_args()) {
    args.push_back(attribute.clone_expression.value());
  }
  if (!GetExtraConstructorArgNames().contains("name")) {
    args.push_back("name_");
  }
  return absl::StrJoin(args, ", ");
}

std::vector<DataMember> OpClass::GetDataMembers() const {
  std::vector<DataMember> data_members;
  for (const auto& attribute : options_.attributes) {
    data_members.push_back(attribute->data_member());
  }
  for (const auto& data_member : options_.extra_data_members) {
    data_members.push_back(data_member);
  }
  for (const OperandInfo& optional_operand: GetOptionalOperands()) {
    if (!optional_operand.operand().AsOptionalOrDie().manual_optional_implementation()) {
      data_members.push_back(DataMember{
        absl::StrFormat("has_%s_", optional_operand.operand().name()),
        "bool",
        absl::StrFormat("%s.has_value()", optional_operand.operand().name()),
      });
    }
  }
  return data_members;
}

std::string OpClass::GetBaseConstructorInvocation() const {
  return absl::StrFormat("Node(%s, %s, loc, name, function)",
      op_, xls_type_expression());
}

std::string OpClass::GetConstructorArgsStr() const {
  std::vector<ConstructorArgument> args = {
    ConstructorArgument("loc", "const SourceInfo&", "loc()"),
  };
  for (const std::unique_ptr<Operand>& o_ptr : operands_) {
    const Operand& o = *o_ptr;
    std::string name{o.name()};
    switch (o.kind()) {
     case OperandKind::kSpan:
      args.push_back(ConstructorArgument{name, "absl::Span<Node* const>"});
      break;
     case OperandKind::kOptional:
      args.push_back(ConstructorArgument{name, "std::optional<Node*>"});
      break;
     case OperandKind::kDefault:
      args.push_back(ConstructorArgument{name, "Node*"});
      break;
    }
  }
  for (const std::unique_ptr<Attribute>& a_ptr : options_.attributes) {
    args.push_back(a_ptr->constructor_argument());
  }

  absl::flat_hash_set<std::string> extra_constructor_arg_names;
  for (const ConstructorArgument& a : options_.extra_constructor_args) {
    extra_constructor_arg_names.insert(a.name);
    args.push_back(a);
  }

  if (!extra_constructor_arg_names.contains("name")) {
    args.push_back(ConstructorArgument{"name", "std::string_view", "name"});
  }

  args.push_back(ConstructorArgument{"function", "FunctionBase*", "function()"});

  std::vector<std::string> arg_strs;
  arg_strs.reserve(args.size());
  for (const ConstructorArgument& a : args) {
    arg_strs.push_back(absl::StrCat(a.cpp_type, " ", a.name));
  }
  return absl::StrJoin(arg_strs, ", ");
}

class OperandBuilder {
 public:
  OperandBuilder() = default;
  OperandBuilder& AddSpan(std::string name) {
    operands_.push_back(std::make_unique<OperandSpan>(std::move(name)));
    return *this;
  }
  OperandBuilder& Add(std::string name) {
    operands_.push_back(std::make_unique<Operand>(std::move(name)));
    return *this;
  }
  OperandBuilder& AddOptional(std::string name, bool manual_optional_implementation = false) {
    operands_.push_back(std::make_unique<OptionalOperand>(std::move(name), /*manual_optional_implementation=*/manual_optional_implementation));
    return *this;
  }
  std::vector<std::unique_ptr<Operand>> Build() { return std::move(operands_); }

 private:
  std::vector<std::unique_ptr<Operand>> operands_;
};

class AttributeBuilder {
 public:
  AttributeBuilder() = default;
  AttributeBuilder& Add(std::unique_ptr<Attribute> attribute) {
    attributes_.push_back(std::move(attribute));
    return *this;
  }
  AttributeBuilder& AddInt64(std::string_view name) {
    attributes_.push_back(std::make_unique<Int64Attribute>(name));
    return *this;
  }
  AttributeBuilder& AddString(std::string_view name) {
    attributes_.push_back(std::make_unique<StringAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddBool(std::string_view name) {
    attributes_.push_back(std::make_unique<BoolAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddOptionalString(std::string_view name) {
    attributes_.push_back(std::make_unique<OptionalStringAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddFormatSteps(std::string_view name) {
    attributes_.push_back(std::make_unique<FormatStepsAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddType(std::string_view name) {
    attributes_.push_back(std::make_unique<TypeAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddFunction(std::string_view name) {
    attributes_.push_back(std::make_unique<FunctionAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddValue(std::string_view name) {
    attributes_.push_back(std::make_unique<ValueAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddLsbOrMsb(std::string_view name) {
    attributes_.push_back(std::make_unique<LsbOrMsbAttribute>(name));
    return *this;
  }
  AttributeBuilder& AddInstantiation(std::string_view name) {
    attributes_.push_back(std::make_unique<InstantiationAttribute>(name));
    return *this;
  }
  std::vector<std::unique_ptr<Attribute>> Build() { return std::move(attributes_); }

 private:
  std::vector<std::unique_ptr<Attribute>> attributes_;
};

static absl::btree_map<std::string, OpClass>* CreateOpClassKindsSingleton() {
  auto* map = new absl::btree_map<std::string, OpClass>{};
  map->emplace("AFTER_ALL", OpClass{
      /*name=*/"AfterAll",
      /*op=*/"Op::kAfterAll",
      /*operands=*/OperandBuilder().AddSpan("dependencies").Build(),
      /*xls_type_expression=*/"function->package()->GetTokenType()",
  });
  map->emplace("MIN_DELAY", OpClass{
      /*name=*/"MinDelay",
      /*op=*/"Op::kMinDelay",
      /*operands=*/OperandBuilder().Add("token").Build(),
      /*xls_type_expression=*/"function->package()->GetTokenType()",
      OpClassOptions{
        .attributes = AttributeBuilder().AddInt64("delay").Build(),
      },
  });
  map->emplace("ARRAY", OpClass(
    /*name=*/"Array",
    /*op=*/"Op::kArray",
    /*operands=*/OperandBuilder().AddSpan("elements").Build(),
    /*xls_type_expression=*/"function->package()->GetArrayType(elements.size(), element_type)",
    /*options=*/OpClassOptions{
      .attributes = AttributeBuilder().AddType("element_type").Build(),
      .extra_methods = {
        Method("size", "int64_t", "operand_count()")
      },
      .custom_clone_method = true
    }
  ));
  map->emplace("ARRAY_INDEX", OpClass{
    /*name=*/"ArrayIndex",
    /*op=*/"Op::kArrayIndex",
    /*operands=*/OperandBuilder().Add("arg").AddSpan("indices").Build(),
    /*xls_type_expression=*/"GetIndexedElementType(arg->GetType(), indices.size()).value()",
    /*options=*/OpClassOptions{
    .extra_methods={
      Method("array", "Node*", "operand(0)"),
      Method("indices", "absl::Span<Node* const>", "operands().subspan(1)")
    },
    .custom_clone_method=true
    }
  });
  map->emplace("ARRAY_SLICE", OpClass{
    /*name=*/"ArraySlice",
    /*op=*/"Op::kArraySlice",
    /*operands=*/OperandBuilder().Add("array").Add("start").Build(),
    /*xls_type_expression=*/"function->package()->GetArrayType(width, array->GetType()->AsArrayOrDie()->element_type())",
    /*options=*/OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("width").Build(),
      .extra_methods={
        Method("array", "Node*", "operand(0)"),
        Method("start", "Node*", "operand(1)")
      },
    }
  });
  map->emplace("ARRAY_UPDATE", OpClass{
    /*name=*/"ArrayUpdate",
    /*op=*/"Op::kArrayUpdate",
    /*operands=*/OperandBuilder().Add("array").Add("update_value").AddSpan("indices").Build(),
    /*xls_type_expression=*/"array->GetType()",
    OpClassOptions{
      .extra_methods={
        Method("array_to_update", "Node*", "operand(0)"),
        Method("indices", "absl::Span<Node* const>", "operands().subspan(2)"),
        Method("update_value", "Node*", "operand(1)"),
      },
      .custom_clone_method=true
    },
  });
  map->emplace("ARRAY_CONCAT", OpClass{
    /*name=*/"ArrayConcat",
    /*op=*/"Op::kArrayConcat",
    /*operands=*/OperandBuilder().AddSpan("args").Build(),
    /*xls_type_expression=*/"GetArrayConcatType(function->package(), args)",
  });
  map->emplace("BIN_OP", OpClass{
    /*name=*/"BinOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("lhs").Add("rhs").Build(),
    /*xls_type_expression=*/"lhs->GetType()",
    OpClassOptions{
    .extra_constructor_args={
      ConstructorArgument{.name="op", .cpp_type="Op", .clone_expression="op()"}
    },
    }
  });
  map->emplace("ARITH_OP", OpClass{
    /*name=*/"ArithOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("lhs").Add("rhs").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(width)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("width").Build(),
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    }
  });

  map->emplace("PARTIAL_PRODUCT_OP", OpClass{
    /*name=*/"PartialProductOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("lhs").Add("rhs").Build(),
    /*xls_type_expression=*/"function->package()->GetTupleType({function->package()->GetBitsType(width), function->package()->GetBitsType(width)})",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("width").Build(),
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    },
  });

  map->emplace("ASSERT", OpClass{
    /*name=*/"Assert",
    /*op=*/"Op::kAssert",
    /*operands=*/OperandBuilder().Add("token").Add("condition").Build(),
    /*xls_type_expression=*/"function->package()->GetTokenType()",
    OpClassOptions{
      .attributes=AttributeBuilder().AddString("message").AddOptionalString("label").AddOptionalString("original_label").Build(),
      .extra_methods={
        Method("token", "Node*", "operand(0)"),
        Method("condition", "Node*", "operand(1)"),
        Method(
            /*name=*/"set_label",
            /*return_cpp_type=*/"void",
            /*expression=*/"label_ = std::move(new_label);",
            MethodOptions{
              .expression_is_body = true,
              .params = "std::string new_label",
              .is_const=false,
            }
        ),
      },
    }
  });

  map->emplace("TRACE", OpClass{
    /*name=*/"Trace",
    /*op=*/"Op::kTrace",
    /*operands=*/OperandBuilder().Add("token").Add("condition").AddSpan("args").Build(),
    /*xls_type_expression=*/"function->package()->GetTokenType()",
    OpClassOptions{
      .attributes=AttributeBuilder().AddFormatSteps("format").AddInt64("verbosity").Build(),
      .extra_methods={
        Method("token", "Node*", "operand(0)"),
        Method("condition", "Node*", "operand(1)"),
        Method("args", "absl::Span<Node* const>", "operands().subspan(2)"),
      },
      .custom_clone_method=true,
    },
  });

  map->emplace("COVER", OpClass{
    /*name=*/"Cover",
    /*op=*/"Op::kCover",
    /*operands=*/OperandBuilder().Add("condition").Build(),
    /*xls_type_expression=*/"function->package()->GetTupleType({})",
    OpClassOptions{
    .attributes=AttributeBuilder().AddString("label").AddOptionalString("original_label").Build(),
    .extra_methods={
      Method("condition", "Node*", "operand(0)"),
      Method(
        /*name=*/"set_label",
        /*return_cpp_type=*/"void",
        /*expression=*/"label_ = std::move(new_label);",
        MethodOptions{
          .expression_is_body = true,
          .params="std::string new_label",
          .is_const=false,
        }),
    },
    }
  });

  map->emplace("BITWISE_REDUCTION_OP", OpClass{
    /*name=*/"BitwiseReductionOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("operand").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(1)",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    }
  });
  map->emplace("RECEIVE", OpClass{
    /*name=*/"Receive",
    /*op=*/"Op::kReceive",
    /*operands=*/OperandBuilder().Add("token").AddOptional("predicate").Build(),
    /*xls_type_expression=*/"GetReceiveType(function, channel_name, is_blocking)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddString("channel_name").AddBool("is_blocking").Build(),
      .extra_methods={
        Method("token", "Node*", "operand(0)"),
        Method("predicate", "std::optional<Node*>", "predicate_operand_number().ok() ? std::optional<Node*>(operand(*predicate_operand_number())) : std::nullopt"),
        Method("GetPayloadType", "Type*", std::nullopt),
        Method(
          "ReplaceChannel",
          /*return_cpp_type=*/"void",
          /*expression=*/std::nullopt,
          MethodOptions{
            .params="std::string_view new_channel_name",
            .is_const=false,
          }
        ),
      },
      .custom_clone_method=true,
    },
  });

  map->emplace("SEND", OpClass{
    /*name=*/"Send",
    /*op=*/"Op::kSend",
    /*operands=*/OperandBuilder().Add("token").Add("data").AddOptional("predicate").Build(),
    /*xls_type_expression=*/"function->package()->GetTokenType()",
    OpClassOptions{
      .attributes=AttributeBuilder().AddString("channel_name").Build(),
      .extra_methods={
        Method("token", "Node*", "operand(0)"),
        Method("data", "Node*", "operand(1)"),
        Method("predicate", "std::optional<Node*>", "predicate_operand_number().ok() ? std::optional<Node*>(operand(*predicate_operand_number())) : std::nullopt"),
        Method("ReplaceChannel", "void", /*expression=*/std::nullopt, MethodOptions{
            .params = "std::string_view new_channel_name",
            .is_const=false
        }),
      },
      .custom_clone_method=true,
    }
  });

  map->emplace("NARY_OP", OpClass{
    /*name=*/"NaryOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().AddSpan("args").Build(),
    /*xls_type_expression=*/"args[0]->GetType()",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    }
  });
  map->emplace("BIT_SLICE", OpClass{
    /*name=*/"BitSlice",
    /*op=*/"Op::kBitSlice",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(width)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("start").AddInt64("width").Build(),
    },
  });
  map->emplace("DYNAMIC_BIT_SLICE", OpClass{
    /*name=*/"DynamicBitSlice",
    /*op=*/"Op::kDynamicBitSlice",
    /*operands=*/OperandBuilder().Add("arg").Add("start").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(width)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("width").Build(),
      .extra_methods={
        Method("to_slice", "Node*", "operand(0)"),
        Method("start", "Node*", "operand(1)"),
      },
    },
  });
  map->emplace("BIT_SLICE_UPDATE", OpClass{
    /*name=*/"BitSliceUpdate",
    /*op=*/"Op::kBitSliceUpdate",
    /*operands=*/OperandBuilder().Add("arg").Add("start").Add("value").Build(),
    /*xls_type_expression=*/"arg->GetType()",
    OpClassOptions{
      .extra_methods={
        Method("to_update", "Node*", "operand(0)"),
        Method("start", "Node*", "operand(1)"),
        Method("update_value", "Node*", "operand(2)"),
      },
    },
  });
  map->emplace("COMPARE_OP", OpClass{
    /*name=*/"CompareOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("lhs").Add("rhs").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(1)",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    },
  });
  map->emplace("CONCAT", OpClass{
    /*name=*/"Concat",
    /*op=*/"Op::kConcat",
    /*operands=*/OperandBuilder().AddSpan("args").Build(),
    /*xls_type_expression=*/"GetConcatType(function->package(), args)",
    OpClassOptions{
      .extra_methods={
        Method(
            /*name=*/"GetOperandSliceData",
            /*return_cpp_type=*/"SliceData",
            /*expression=*/std::nullopt,
            /*options=*/MethodOptions{
              .params = "int64_t operandno"
            }
        ),
      },
    },
  });

  map->emplace("COUNTED_FOR", OpClass{
    /*name=*/"CountedFor",
    /*op=*/"Op::kCountedFor",
    /*operands=*/OperandBuilder().Add("initial_value").AddSpan("invariant_args").Build(),
    /*xls_type_expression=*/"initial_value->GetType()",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("trip_count").AddInt64("stride").AddFunction("body").Build(),
      .extra_methods={
        Method("initial_value", "Node*", "operand(0)"),
        Method("invariant_args", "absl::Span<Node* const>", "operands().subspan(1)"),
      },
      .custom_clone_method=true,
    }
  });


  map->emplace("DYNAMIC_COUNTED_FOR", OpClass{
    /*name=*/"DynamicCountedFor",
    /*op=*/"Op::kDynamicCountedFor",
    /*operands=*/OperandBuilder().Add("initial_value").Add("trip_count").Add("stride").AddSpan("invariant_args").Build(),
    /*xls_type_expression=*/"initial_value->GetType()",
    OpClassOptions{
      .attributes=AttributeBuilder().AddFunction("body").Build(),
      .extra_methods={
        Method("initial_value", "Node*", "operand(0)"),
        Method("trip_count", "Node*", "operand(1)"),
        Method("stride", "Node*", "operand(2)"),
        Method("invariant_args", "absl::Span<Node* const>", "operands().subspan(3)"),
      },
      .custom_clone_method=true,
    },
  });

  map->emplace("EXTEND_OP", OpClass{
    /*name=*/"ExtendOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(new_bit_count)",
    OpClassOptions{
      .attributes = AttributeBuilder().AddInt64("new_bit_count").Build(),
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    },
  });

  map->emplace("INVOKE", OpClass{
    /*name=*/"Invoke",
    /*op=*/"Op::kInvoke",
    /*operands=*/OperandBuilder().AddSpan("args").Build(),
    /*xls_type_expression=*/"to_apply->return_value()->GetType()",
    OpClassOptions{
      .attributes=AttributeBuilder().AddFunction("to_apply").Build(),
    },
  });
  map->emplace("LITERAL", OpClass{
    /*name=*/"Literal",
    /*op=*/"Op::kLiteral",
    /*operands=*/{},
    /*xls_type_expression=*/"function->package()->GetTypeForValue(value)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddValue("value").Build(),
      .extra_methods={
        Method("IsZero", "bool", "value().IsBits() && value().bits().IsZero()"),
      },
    },
  });
  map->emplace("MAP", OpClass{
    /*name=*/"Map",
    /*op=*/"Op::kMap",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"GetMapType(arg, to_apply)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddFunction("to_apply").Build(),
    },
  });

  map->emplace("ONE_HOT", OpClass{
    /*name=*/"OneHot",
    /*op=*/"Op::kOneHot",
    /*operands=*/OperandBuilder().Add("input").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(input->BitCountOrDie() + 1)",
    OpClassOptions {
      .attributes=AttributeBuilder().AddLsbOrMsb("priority").Build(),
    },
  });
  map->emplace("ONE_HOT_SELECT", OpClass{
    /*name=*/"OneHotSelect",
    /*op=*/"Op::kOneHotSel",
    /*operands=*/OperandBuilder().Add("selector").AddSpan("cases").Build(),
    /*xls_type_expression=*/"cases[0]->GetType()",
    OpClassOptions{
      .extra_methods={
        Method("selector", "Node*", "operand(0)"),
        Method("cases", "absl::Span<Node* const>", "operands().subspan(1)"),
        Method("get_case", "Node*", /*expression=*/"cases().at(case_no)", MethodOptions{
          .params = "int64_t case_no",
        }),
      },
      .custom_clone_method=true,
    }
  });

  map->emplace("PRIORITY_SELECT", OpClass{
    /*name=*/"PrioritySelect",
    /*op=*/"Op::kPrioritySel",
    /*operands=*/OperandBuilder().Add("selector").AddSpan("cases").Add("default_value").Build(),
    /*xls_type_expression=*/"default_value->GetType()",
    OpClassOptions{
      .extra_data_members={
        DataMember("cases_size_", "int64_t", "cases.size()"),
      },
      .extra_methods={
        Method("selector", "Node*", "operand(0)"),
        Method("cases", "absl::Span<Node* const>", "operands().subspan(1, cases_size_)"),
        Method("get_case", "Node*", "cases().at(case_no)", MethodOptions{
          .params = "int64_t case_no", 
        }),
        Method("default_value", "Node*", "operand(1 + cases_size_)"),
      },
      .custom_clone_method=true,
    },
  });

  map->emplace("PARAM", OpClass{
    /*name=*/"Param",
    /*op=*/"Op::kParam",
    /*operands=*/{},
    /*xls_type_expression=*/"type",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"type", /*cpp_type=*/"Type*", /*clone_expression=*/"GetType()"},
        ConstructorArgument{/*name=*/"name", /*cpp_type=*/"std::string_view", /*clone_expression=*/"name()"},
      },
      .extra_methods={
        Method("name", "std::string_view", "name_"),
      },
      .custom_clone_method=true,
    },
  });
  map->emplace("NEXT", OpClass{
    /*name=*/"Next",
    /*op=*/"Op::kNext",
    /*operands=*/OperandBuilder().Add("param").Add("value").AddOptional("predicate").Build(),
    /*xls_type_expression=*/"function->package()->GetTupleType({})",
    OpClassOptions{
      .extra_methods={
        Method("param", "Node*", "operand(0)"),
        Method("value", "Node*", "operand(1)"),
        Method("predicate", "std::optional<Node*>", "predicate_operand_number().ok() ? std::optional<Node*>(operand(*predicate_operand_number())) : std::nullopt"),
      },
      .custom_clone_method=true,
    },
  });

  map->emplace("SELECT", OpClass{
    /*name=*/"Select",
    /*op=*/"Op::kSel",
    /*operands=*/OperandBuilder().Add("selector").AddSpan("cases").AddOptional("default_value", /*manual_optional_implementation=*/true).Build(),
    /*xls_type_expression=*/"cases[0]->GetType()",
    OpClassOptions{
      .extra_data_members={
        DataMember("cases_size_", "int64_t", "cases.size()"),
        DataMember("has_default_value_", "bool", "default_value.has_value()"),
      },
      .extra_methods={
        Method("selector", "Node*", "operand(0)"),
        Method("cases", "absl::Span<Node* const>", "operands().subspan(1, cases_size_)"),
        Method("get_case", "Node*", "cases().at(case_no)", MethodOptions{
          .params = "int64_t case_no",
        }),
        Method("default_value", "std::optional<Node*>", "has_default_value_ ? std::optional<Node*>(operands().back()) : std::nullopt"),
        Method("AllCases", /*return_cpp_type=*/"bool", /*expression=*/std::nullopt, MethodOptions{
            .params = "std::function<bool(Node*)> p"
        }),
        Method("any_case", /*return_cpp_type=*/"Node*", "!cases().empty() ? cases().front() : default_value().has_value() ? default_value().value() : nullptr"),
      },
      .custom_clone_method=true,
    },
  });
  map->emplace("TUPLE", OpClass{
    /*name=*/"Tuple",
    /*op=*/"Op::kTuple",
    /*operands=*/OperandBuilder().AddSpan("elements").Build(),
    /*xls_type_expression=*/"GetTupleType(function->package(), elements)",
    OpClassOptions{
      .extra_methods={
        Method("size", "int64_t", "operand_count()"),
      },
    },
  });
  map->emplace("TUPLE_INDEX", OpClass{
    /*name=*/"TupleIndex",
    /*op=*/"Op::kTupleIndex",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"arg->GetType()->AsTupleOrDie()->element_type(index)",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInt64("index").Build()
    },
  });

  map->emplace("UN_OP", OpClass{
    /*name=*/"UnOp",
    /*op=*/"op",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"arg->GetType()",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"op", /*cpp_type=*/"Op", /*clone_expression=*/"op()"}
      },
    },
  });
  map->emplace("DECODE", OpClass{
    /*name=*/"Decode",
    /*op=*/"Op::kDecode",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(width)",
    OpClassOptions{
      .attributes = AttributeBuilder().AddInt64("width").Build(),
    }
  });
  map->emplace("ENCODE", OpClass{
    /*name=*/"Encode",
    /*op=*/"Op::kEncode",
    /*operands=*/OperandBuilder().Add("arg").Build(),
    /*xls_type_expression=*/"function->package()->GetBitsType(CeilOfLog2(arg->BitCountOrDie()))",
  });

  map->emplace("INPUT_PORT", OpClass{
    /*name=*/"InputPort",
    /*op=*/"Op::kInputPort",
    /*operands=*/{},
    /*xls_type_expression=*/"type",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"name", /*cpp_type=*/"std::string_view", /*clone_expression=*/"name()"},
        ConstructorArgument{/*name=*/"type", /*cpp_type=*/"Type*", /*clone_expression=*/"GetType()"},
      },
      .extra_methods={
        Method("name", "std::string_view", "name_"),
      },
    },
  });
  map->emplace("OUTPUT_PORT", OpClass{
    /*name=*/"OutputPort",
    /*op=*/"Op::kOutputPort",
    /*operands=*/OperandBuilder().Add("operand").Build(),
    /*xls_type_expression=*/"function->package()->GetTupleType({})",
    OpClassOptions{
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"name", /*cpp_type=*/"std::string_view", /*clone_expression=*/"name()"},
      },
      .extra_methods={
        Method("name", "std::string_view", "name_"),
      },
    },
  });

  map->emplace("REGISTER_READ", OpClass{
    /*name=*/"RegisterRead",
    /*op=*/"Op::kRegisterRead",
    /*operands=*/{},
    /*xls_type_expression=*/"reg->type()",
    OpClassOptions{
      .extra_constructor_args={
        // `register` is a C++ keyword so an attribute of name register can't
        // be defined. Use `reg` for the data member and constructor arg, and
        // GetRegister for the accessor method.
        ConstructorArgument{/*name=*/"reg", /*cpp_type=*/"Register*", /*clone_expression=*/"GetRegister()"}
      },
      .extra_data_members={
        DataMember("reg_", "Register*", "reg")
      },
      .extra_methods={
        Method("GetRegister", "Register*", "reg_")
      },
    },
  });

  
  map->emplace("REGISTER_WRITE", OpClass{
    /*name=*/"RegisterWrite",
    /*op=*/"Op::kRegisterWrite",
    /*operands=*/OperandBuilder().Add("data").AddOptional("load_enable").AddOptional("reset").Build(),
    /*xls_type_expression=*/"function->package()->GetTupleType({})",
    OpClassOptions{
      // `register` is a C++ keyword so an attribute of name register can't
      // be defined. Use `reg` for the data member and constructor arg, and
      // GetRegister for the accessor method.
      .extra_constructor_args={
        ConstructorArgument{/*name=*/"reg", /*cpp_type=*/"Register*", /*clone_expression=*/"GetRegister()"}
      },
      .extra_data_members={
        DataMember("reg_", "Register*", "reg")
      },
      .extra_methods={
        Method("data", "Node*", "operand(0)"),
        Method("load_enable", "std::optional<Node*>", "load_enable_operand_number().ok() ? std::optional<Node*>(operand(*load_enable_operand_number())) : std::nullopt"),
        Method("reset", "std::optional<Node*>", "reset_operand_number().ok() ? std::optional<Node*>(operand(*reset_operand_number())) : std::nullopt"),
        Method("GetRegister", "Register*", "reg_"),
        Method("ReplaceExistingLoadEnable", /*return_cpp_type=*/"absl::Status", "has_load_enable_ ? ReplaceOperandNumber(*load_enable_operand_number(), new_operand) : absl::InternalError(\"Unable to replace load enable on RegisterWrite -- register does not have an existing load enable operand.\")", MethodOptions{
          .params = "Node* new_operand", 
          .is_const = false,
        }),
        Method("AddOrReplaceReset", "absl::Status", R"(reg_->UpdateReset(new_reset_info);
    if (!has_reset_) {
      AddOperand(new_reset_node);
      has_reset_ = true;
      return absl::OkStatus();
    }
    return ReplaceOperandNumber(*reset_operand_number(), new_reset_node);
)", MethodOptions{
            .expression_is_body = true,
            .params = "Node* new_reset_node, Reset new_reset_info",
            .is_const = false
        }),
      },
    },
  });

  map->emplace("INSTANTIATION_OUTPUT", OpClass{
    /*name=*/"InstantiationOutput",
    /*op=*/"Op::kInstantiationOutput",
    /*operands=*/{},
    /*xls_type_expression=*/"instantiation->GetOutputPort(port_name).value().type",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInstantiation("instantiation").AddString("port_name").Build(),
    }
  });
  map->emplace("INSTANTIATION_INPUT", OpClass{
    /*name=*/"InstantiationInput",
    /*op=*/"Op::kInstantiationInput",
    /*operands=*/OperandBuilder().Add("data").Build(),
    /*xls_type_expression=*/"function->package()->GetTupleType({})",
    OpClassOptions{
      .attributes=AttributeBuilder().AddInstantiation("instantiation").AddString("port_name").Build(),
      .extra_methods={
        Method("data", "Node*", "operand(0)"),
      },
    },
  });
  map->emplace("GATE", OpClass{
    /*name=*/"Gate",
    /*op=*/"Op::kGate",
    /*operands=*/OperandBuilder().Add("condition").Add("data").Build(),
    /*xls_type_expression=*/"data->GetType()",
    OpClassOptions{
      .extra_methods={
        Method("condition", "Node*", "operand(0)"),
        Method("data", "Node*", "operand(1)"),
      },
    }
  });

  return map;
}

const absl::btree_map<std::string, OpClass>& GetOpClassKindsSingleton() {
  static auto* op_class_kinds_singleton = CreateOpClassKindsSingleton();
  return *op_class_kinds_singleton;
}

const std::vector<Op>& GetOpsSingleton() {
  const absl::btree_map<std::string, OpClass>& op_classes = GetOpClassKindsSingleton();
  static const std::vector<Op>* singleton = new std::vector<Op>{
    Op{
      /*enum_name=*/"kAdd",
      /*name=*/"add",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{
        Property::kAssociative,
        Property::kCommutative,
      }
    },
    Op{
      /*enum_name=*/"kAnd",
      /*name=*/"and",
      /*op_class=*/op_classes.at("NARY_OP"),
      /*properties=*/{
        Property::kBitwise,
        Property::kAssociative,
        Property::kCommutative,
      }
    },
    Op{
      /*enum_name=*/"kAndReduce",
      /*name=*/"and_reduce",
      /*op_class=*/op_classes.at("BITWISE_REDUCTION_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kAssert",
      /*name=*/"assert",
      /*op_class=*/op_classes.at("ASSERT"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kCover",
      /*name=*/"cover",
      /*op_class=*/op_classes.at("COVER"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kReceive",
      /*name=*/"receive",
      /*op_class=*/op_classes.at("RECEIVE"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kSend",
      /*name=*/"send",
      /*op_class=*/op_classes.at("SEND"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kNand",
      /*name=*/"nand",
      /*op_class=*/op_classes.at("NARY_OP"),
      /*properties=*/{Property::kBitwise, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kNor",
      /*name=*/"nor",
      /*op_class=*/op_classes.at("NARY_OP"),
      /*properties=*/{Property::kBitwise, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kAfterAll",
      /*name=*/"after_all",
      /*op_class=*/op_classes.at("AFTER_ALL"),
      /*properties=*/{Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kArray",
      /*name=*/"array",
      /*op_class=*/op_classes.at("ARRAY"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kArrayIndex",
      /*name=*/"array_index",
      /*op_class=*/op_classes.at("ARRAY_INDEX"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kArraySlice",
      /*name=*/"array_slice",
      /*op_class=*/op_classes.at("ARRAY_SLICE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kArrayUpdate",
      /*name=*/"array_update",
      /*op_class=*/op_classes.at("ARRAY_UPDATE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kArrayConcat",
      /*name=*/"array_concat",
      /*op_class=*/op_classes.at("ARRAY_CONCAT"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kBitSlice",
      /*name=*/"bit_slice",
      /*op_class=*/op_classes.at("BIT_SLICE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kDynamicBitSlice",
      /*name=*/"dynamic_bit_slice",
      /*op_class=*/op_classes.at("DYNAMIC_BIT_SLICE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kBitSliceUpdate",
      /*name=*/"bit_slice_update",
      /*op_class=*/op_classes.at("BIT_SLICE_UPDATE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kConcat",
      /*name=*/"concat",
      /*op_class=*/op_classes.at("CONCAT"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kCountedFor",
      /*name=*/"counted_for",
      /*op_class=*/op_classes.at("COUNTED_FOR"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kDecode",
      /*name=*/"decode",
      /*op_class=*/op_classes.at("DECODE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kDynamicCountedFor",
      /*name=*/"dynamic_counted_for",
      /*op_class=*/op_classes.at("DYNAMIC_COUNTED_FOR"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kEncode",
      /*name=*/"encode",
      /*op_class=*/op_classes.at("ENCODE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kEq",
      /*name=*/"eq",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kIdentity",
      /*name=*/"identity",
      /*op_class=*/op_classes.at("UN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kInvoke",
      /*name=*/"invoke",
      /*op_class=*/op_classes.at("INVOKE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kInputPort",
      /*name=*/"input_port",
      /*op_class=*/op_classes.at("INPUT_PORT"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kLiteral",
      /*name=*/"literal",
      /*op_class=*/op_classes.at("LITERAL"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kMap",
      /*name=*/"map",
      /*op_class=*/op_classes.at("MAP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kNe",
      /*name=*/"ne",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kNeg",
      /*name=*/"neg",
      /*op_class=*/op_classes.at("UN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kNot",
      /*name=*/"not",
      /*op_class=*/op_classes.at("UN_OP"),
      /*properties=*/{Property::kBitwise},
    },
    Op{
      /*enum_name=*/"kOneHot",
      /*name=*/"one_hot",
      /*op_class=*/op_classes.at("ONE_HOT"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kOneHotSel",
      /*name=*/"one_hot_sel",
      /*op_class=*/op_classes.at("ONE_HOT_SELECT"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kPrioritySel",
      /*name=*/"priority_sel",
      /*op_class=*/op_classes.at("PRIORITY_SELECT"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kOr",
      /*name=*/"or",
      /*op_class=*/op_classes.at("NARY_OP"),
      /*properties=*/{Property::kBitwise, Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kOrReduce",
      /*name=*/"or_reduce",
      /*op_class=*/op_classes.at("BITWISE_REDUCTION_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kOutputPort",
      /*name=*/"output_port",
      /*op_class=*/op_classes.at("OUTPUT_PORT"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kParam",
      /*name=*/"param",
      /*op_class=*/op_classes.at("PARAM"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kNext",
      /*name=*/"next_value",
      /*op_class=*/op_classes.at("NEXT"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kRegisterRead",
      /*name=*/"register_read",
      /*op_class=*/op_classes.at("REGISTER_READ"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kRegisterWrite",
      /*name=*/"register_write",
      /*op_class=*/op_classes.at("REGISTER_WRITE"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kInstantiationOutput",
      /*name=*/"instantiation_output",
      /*op_class=*/op_classes.at("INSTANTIATION_OUTPUT"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kInstantiationInput",
      /*name=*/"instantiation_input",
      /*op_class=*/op_classes.at("INSTANTIATION_INPUT"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kReverse",
      /*name=*/"reverse",
      /*op_class=*/op_classes.at("UN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kSDiv",
      /*name=*/"sdiv",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kSMod",
      /*name=*/"smod",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kSel",
      /*name=*/"sel",
      /*op_class=*/op_classes.at("SELECT"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kSGe",
      /*name=*/"sge",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kSGt",
      /*name=*/"sgt",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kShll",
      /*name=*/"shll",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kShrl",
      /*name=*/"shrl",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kShra",
      /*name=*/"shra",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kSignExt",
      /*name=*/"sign_ext",
      /*op_class=*/op_classes.at("EXTEND_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kSLe",
      /*name=*/"sle",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kSLt",
      /*name=*/"slt",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kSMul",
      /*name=*/"smul",
      /*op_class=*/op_classes.at("ARITH_OP"),
      /*properties=*/{Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kSMulp",
      /*name=*/"smulp",
      /*op_class=*/op_classes.at("PARTIAL_PRODUCT_OP"),
      /*properties=*/{Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kSub",
      /*name=*/"sub",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kTuple",
      /*name=*/"tuple",
      /*op_class=*/op_classes.at("TUPLE"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kTupleIndex",
      /*name=*/"tuple_index",
      /*op_class=*/op_classes.at("TUPLE_INDEX"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kUDiv",
      /*name=*/"udiv",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kUMod",
      /*name=*/"umod",
      /*op_class=*/op_classes.at("BIN_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kUGe",
      /*name=*/"uge",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kUGt",
      /*name=*/"ugt",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kULe",
      /*name=*/"ule",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kULt",
      /*name=*/"ult",
      /*op_class=*/op_classes.at("COMPARE_OP"),
      /*properties=*/{Property::kComparison},
    },
    Op{
      /*enum_name=*/"kUMul",
      /*name=*/"umul",
      /*op_class=*/op_classes.at("ARITH_OP"),
      /*properties=*/{Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kUMulp",
      /*name=*/"umulp",
      /*op_class=*/op_classes.at("PARTIAL_PRODUCT_OP"),
      /*properties=*/{Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kXor",
      /*name=*/"xor",
      /*op_class=*/op_classes.at("NARY_OP"),
      /*properties=*/{Property::kBitwise, Property::kAssociative, Property::kCommutative},
    },
    Op{
      /*enum_name=*/"kXorReduce",
      /*name=*/"xor_reduce",
      /*op_class=*/op_classes.at("BITWISE_REDUCTION_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kZeroExt",
      /*name=*/"zero_ext",
      /*op_class=*/op_classes.at("EXTEND_OP"),
      /*properties=*/{},
    },
    Op{
      /*enum_name=*/"kGate",
      /*name=*/"gate",
      /*op_class=*/op_classes.at("GATE"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kTrace",
      /*name=*/"trace",
      /*op_class=*/op_classes.at("TRACE"),
      /*properties=*/{Property::kSideEffecting},
    },
    Op{
      /*enum_name=*/"kMinDelay",
      /*name=*/"min_delay",
      /*op_class=*/op_classes.at("MIN_DELAY"),
      /*properties=*/{},
    },
  };
  return *singleton;
}

}  // namespace xls
