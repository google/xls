// Copyright 2020 Google LLC
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

#include "xls/ir/ir_parser.h"

#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/verifier.h"

namespace xls {

xabsl::StatusOr<int64> Parser::ParseBitsTypeAndReturnWidth() {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_.PeekToken());
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("bits"));
  XLS_RETURN_IF_ERROR(
      scanner_.DropTokenOrError(LexicalTokenType::kBracketOpen));
  XLS_ASSIGN_OR_RETURN(int64 bit_count, ParseInt64());
  if (bit_count < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Only positive bit counts are permitted for bits types; found %d @ %s",
        bit_count, peek.pos().ToHumanString()));
  }
  XLS_RETURN_IF_ERROR(
      scanner_.DropTokenOrError(LexicalTokenType::kBracketClose));
  return bit_count;
}

xabsl::StatusOr<Type*> Parser::ParseBitsType(Package* package) {
  XLS_ASSIGN_OR_RETURN(int64 bit_count, ParseBitsTypeAndReturnWidth());
  return package->GetBitsType(bit_count);
}

xabsl::StatusOr<Type*> Parser::ParseType(Package* package) {
  Type* type;
  if (scanner_.PeekTokenIs(LexicalTokenType::kParenOpen)) {
    XLS_ASSIGN_OR_RETURN(type, ParseTupleType(package));
  } else if (scanner_.TryDropKeyword("token")) {
    return package->GetTokenType();
  } else {
    XLS_ASSIGN_OR_RETURN(type, ParseBitsType(package));
  }
  while (scanner_.TryDropToken(LexicalTokenType::kBracketOpen)) {
    // Array type.
    XLS_ASSIGN_OR_RETURN(int64 size, ParseInt64());
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kBracketClose));
    type = package->GetArrayType(size, type);
  }
  return type;
}

// Abstraction holding the value of a keyword argument.
template <typename T>
struct KeywordValue {
  bool is_optional;
  T value;
  absl::optional<T> optional_value;

  // Sets the value of this object to the value contains in 'value_status' or
  // returns an error if value_status is an error value.
  absl::Status SetOrReturn(xabsl::StatusOr<T> value_status) {
    if (is_optional) {
      XLS_ASSIGN_OR_RETURN(optional_value, value_status);
    } else {
      XLS_ASSIGN_OR_RETURN(value, value_status);
    }
    return absl::OkStatus();
  }
};

// Variant which gathers all the possible keyword argument types. New
// keywords arguments which require a new type should be added here.
using KeywordVariant =
    absl::variant<KeywordValue<int64>, KeywordValue<std::string>,
                  KeywordValue<BValue>, KeywordValue<std::vector<BValue>>,
                  KeywordValue<Value>, KeywordValue<SourceLocation>,
                  KeywordValue<bool>>;

// Abstraction for parsing the arguments of a node. The arguments include
// positional and keyword arguments. The positional arguments are exclusively
// the operands of the node. The keyword argument are the attributes such as
// counted_for loop stride, map function name, etc. Like python, the positional
// arguments are ordered and must be listed first. The keyword arguments may be
// listed in any order. Example:
//
//   operation.1: bits[32] = operation(x, y, z, foo=bar, baz=7)
//
// Here, x, y, and z are the positional arguments. foo and baz are the keyword
// arguments.
class ArgParser {
 public:
  ArgParser(const absl::flat_hash_map<std::string, BValue>& name_to_bvalue,
            Type* node_type, Parser* parser)
      : name_to_bvalue_(name_to_bvalue),
        node_type_(node_type),
        parser_(parser) {}

  // Adds a mandatory keyword to the parser. After calling ArgParser::Run the
  // value pointed to by the returned pointer is the keyword argument value.
  template <typename T>
  T* AddKeywordArg(std::string key) {
    mandatory_keywords_.insert(key);
    auto pair = keywords_.emplace(
        key, absl::make_unique<KeywordVariant>(KeywordValue<T>()));
    XLS_CHECK(pair.second);
    auto& keyword_value = absl::get<KeywordValue<T>>(*pair.first->second);
    keyword_value.is_optional = false;
    // Return a pointer into the KeywordValue which will be filled in when Run
    // is called.
    return &keyword_value.value;
  }

  // Adds an optional keyword to the parser. After calling ArgParser::Run the
  // absl::optional pointed to by the returned pointer will (optionally) contain
  // the keyword argument value.
  template <typename T>
  absl::optional<T>* AddOptionalKeywordArg(absl::string_view key) {
    auto pair = keywords_.emplace(
        key, absl::make_unique<KeywordVariant>(KeywordValue<T>()));
    XLS_CHECK(pair.second);
    auto& keyword_value = absl::get<KeywordValue<T>>(*keywords_.at(key));
    keyword_value.is_optional = true;
    // Return a pointer into the KeywordValue which will be filled in when Run
    // is called.
    return &keyword_value.optional_value;
  }

  template <typename T>
  T* AddOptionalKeywordArg(absl::string_view key, T default_value) {
    auto pair = keywords_.emplace(
        key, absl::make_unique<KeywordVariant>(KeywordValue<T>()));
    XLS_CHECK(pair.second);
    auto& keyword_value = absl::get<KeywordValue<T>>(*keywords_.at(key));
    keyword_value.optional_value = default_value;
    keyword_value.is_optional = true;
    // Return a pointer into the KeywordValue which may be filled in when Run is
    // called; however, if it is not filled, will remain the default value
    // provided.
    return &keyword_value.optional_value.value();
  }

  // Runs the argument parser. 'arity' is the expected number of operands
  // (positional arguments). Returns the BValues of the operands.
  static constexpr int64 kVariadic = -1;
  xabsl::StatusOr<std::vector<BValue>> Run(int64 arity) {
    absl::flat_hash_set<std::string> seen_keywords;
    std::vector<BValue> operands;
    XLS_ASSIGN_OR_RETURN(Token open_paren, parser_->scanner_.PopTokenOrError(
                                               LexicalTokenType::kParenOpen));
    if (!parser_->scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
      // Variable indicating whether we are parsing the keywords or still
      // parsing the positional arguments.
      bool parsing_keywords = false;
      do {
        XLS_ASSIGN_OR_RETURN(Token name,
                             parser_->scanner_.PopTokenOrError(
                                 LexicalTokenType::kIdent, "argument"));
        if (!parsing_keywords &&
            parser_->scanner_.PeekTokenIs(LexicalTokenType::kEquals)) {
          parsing_keywords = true;
        }
        if (parsing_keywords) {
          XLS_RETURN_IF_ERROR(
              parser_->scanner_.DropTokenOrError(LexicalTokenType::kEquals));
          if (seen_keywords.contains(name.value())) {
            return absl::InvalidArgumentError(
                absl::StrFormat("Duplicate keyword argument '%s' @ %s",
                                name.value(), name.pos().ToHumanString()));
          } else if (keywords_.contains(name.value())) {
            XLS_RETURN_IF_ERROR(ParseKeywordArg(name.value()));
          } else {
            return absl::InvalidArgumentError(
                absl::StrFormat("Invalid keyword @ %s: %s",
                                name.pos().ToHumanString(), name.value()));
          }
          seen_keywords.insert(name.value());
        } else {
          if (!name_to_bvalue_.contains(name.value())) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "Referred to a name @ %s that was not previously "
                "defined: \"%s\"",
                name.pos().ToHumanString(), name.value()));
          }
          operands.push_back(name_to_bvalue_.at(name.value()));
        }
      } while (parser_->scanner_.TryDropToken(LexicalTokenType::kComma));
    }
    XLS_RETURN_IF_ERROR(
        parser_->scanner_.DropTokenOrError(LexicalTokenType::kParenClose));

    // Verify all mandatory keywords are present.
    for (const std::string& key : mandatory_keywords_) {
      if (!seen_keywords.contains(key)) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Mandatory keyword argument '%s' not found @ %s",
                            key, open_paren.pos().ToHumanString()));
      }
    }

    // Verify the arity is as expected.
    if (arity != kVariadic && operands.size() != arity) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected %d operands, got %d @ %s", arity,
                          operands.size(), open_paren.pos().ToHumanString()));
    }
    return operands;
  }

 private:
  // Parses the keyword argument with the given key. The expected type of the
  // keyword argument value is determined by the template parameter type T used
  // when Add(Optional)KeywordArgument<T> was called.
  absl::Status ParseKeywordArg(absl::string_view key) {
    KeywordVariant& keyword_variant = *keywords_.at(key);
    return absl::visit(
        Visitor{[&](KeywordValue<bool>& v) {
                  return v.SetOrReturn(parser_->ParseBool());
                },
                [&](KeywordValue<int64>& v) {
                  return v.SetOrReturn(parser_->ParseInt64());
                },
                [&](KeywordValue<std::string>& v) {
                  return v.SetOrReturn(parser_->ParseIdentifierString());
                },
                [&](KeywordValue<Value>& v) {
                  return v.SetOrReturn(parser_->ParseValueInternal(node_type_));
                },
                [&](KeywordValue<BValue>& v) {
                  return v.SetOrReturn(
                      parser_->ParseIdentifierValue(name_to_bvalue_));
                },
                [&](KeywordValue<std::vector<BValue>>& v) {
                  return v.SetOrReturn(parser_->ParseNameList(name_to_bvalue_));
                },
                [&](KeywordValue<SourceLocation>& v) {
                  return v.SetOrReturn(parser_->ParseSourceLocation());
                }},
        keyword_variant);
    return absl::OkStatus();
  }

  const absl::flat_hash_map<std::string, BValue>& name_to_bvalue_;
  Type* node_type_;
  Parser* parser_;

  absl::flat_hash_set<std::string> mandatory_keywords_;
  absl::flat_hash_map<std::string, std::unique_ptr<KeywordVariant>> keywords_;
};

xabsl::StatusOr<int64> Parser::ParseInt64() {
  XLS_ASSIGN_OR_RETURN(Token literal,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  return literal.GetValueInt64();
}

xabsl::StatusOr<bool> Parser::ParseBool() {
  XLS_ASSIGN_OR_RETURN(Token literal,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  return literal.GetValueBool();
}

xabsl::StatusOr<std::string> Parser::ParseIdentifierString(TokenPos* pos) {
  XLS_ASSIGN_OR_RETURN(Token token,
                       scanner_.PopTokenOrError(LexicalTokenType::kIdent));
  if (pos != nullptr) {
    *pos = token.pos();
  }
  return token.value();
}

xabsl::StatusOr<BValue> Parser::ParseIdentifierValue(
    const absl::flat_hash_map<std::string, BValue>& name_to_value) {
  TokenPos start_pos;
  XLS_ASSIGN_OR_RETURN(std::string identifier,
                       ParseIdentifierString(&start_pos));
  auto it = name_to_value.find(identifier);
  if (it == name_to_value.end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Referred to a name @ %s that was not previously defined: \"%s\"",
        start_pos.ToHumanString(), identifier));
  }
  return it->second;
}

xabsl::StatusOr<Value> Parser::ParseValueInternal(absl::optional<Type*> type) {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_.PeekToken());
  const TokenPos start_pos = peek.pos();
  TypeKind type_kind;
  int64 bit_count = 0;
  if (type.has_value()) {
    type_kind = type.value()->kind();
    bit_count =
        type.value()->IsBits() ? type.value()->AsBitsOrDie()->bit_count() : 0;
  } else {
    if (scanner_.PeekTokenIs(LexicalTokenType::kKeyword)) {
      type_kind = TypeKind::kBits;
      XLS_ASSIGN_OR_RETURN(bit_count, ParseBitsTypeAndReturnWidth());
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
    } else if (scanner_.PeekTokenIs(LexicalTokenType::kBracketOpen)) {
      type_kind = TypeKind::kArray;
    } else {
      type_kind = TypeKind::kTuple;
    }
  }

  if (bit_count < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Invalid bit count: %d @ %s", bit_count, start_pos.ToHumanString()));
  }

  if (type_kind == TypeKind::kBits) {
    XLS_ASSIGN_OR_RETURN(Token literal,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_ASSIGN_OR_RETURN(Bits bits_value, literal.GetValueBits());
    if (bits_value.bit_count() > bit_count) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Value %s is not representable in %d bits @ %s", literal.value(),
          bit_count, literal.pos().ToHumanString()));
    }
    XLS_ASSIGN_OR_RETURN(bool is_negative, literal.IsNegative());
    if (is_negative) {
      return Value(bits_ops::SignExtend(bits_value, bit_count));
    } else {
      return Value(bits_ops::ZeroExtend(bits_value, bit_count));
    }
  }
  if (type_kind == TypeKind::kArray) {
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kBracketOpen));
    std::vector<Value> values;
    while (true) {
      if (scanner_.TryDropToken(LexicalTokenType::kBracketClose)) {
        break;
      }
      if (!values.empty()) {
        XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma,
                                                      "',' in array literal"));
      }
      absl::optional<Type*> element_type = absl::nullopt;
      if (type.has_value()) {
        element_type = type.value()->AsArrayOrDie()->element_type();
      }
      XLS_ASSIGN_OR_RETURN(Value element_value,
                           ParseValueInternal(element_type));
      values.push_back(std::move(element_value));
    }
    return Value::Array(values);
  }
  if (type_kind == TypeKind::kTuple) {
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));
    std::vector<Value> values;
    while (true) {
      if (scanner_.TryDropToken(LexicalTokenType::kParenClose)) {
        break;
      }
      if (!values.empty()) {
        XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma,
                                                      "',' in tuple literal"));
      }
      absl::optional<Type*> element_type = absl::nullopt;
      if (type.has_value()) {
        element_type =
            type.value()->AsTupleOrDie()->element_type(values.size());
      }
      XLS_ASSIGN_OR_RETURN(Value element_value,
                           ParseValueInternal(element_type));
      values.push_back(std::move(element_value));
    }
    return Value::Tuple(values);
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported type %s", TypeKindToString(type_kind)));
}

xabsl::StatusOr<std::vector<BValue>> Parser::ParseNameList(
    const absl::flat_hash_map<std::string, BValue>& name_to_value) {
  XLS_RETURN_IF_ERROR(
      scanner_.DropTokenOrError(LexicalTokenType::kBracketOpen));
  std::vector<BValue> result;
  bool must_end = false;
  while (true) {
    if (must_end) {
      XLS_RETURN_IF_ERROR(
          scanner_.DropTokenOrError(LexicalTokenType::kBracketClose));
      break;
    }
    if (scanner_.TryDropToken(LexicalTokenType::kBracketClose)) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(BValue value, ParseIdentifierValue(name_to_value));
    result.push_back(value);
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }
  return result;
}

xabsl::StatusOr<SourceLocation> Parser::ParseSourceLocation() {
  XLS_ASSIGN_OR_RETURN(Token fileno,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));
  XLS_ASSIGN_OR_RETURN(Token lineno,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));
  XLS_ASSIGN_OR_RETURN(Token colno,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  XLS_ASSIGN_OR_RETURN(int64 fileno_value, fileno.GetValueInt64());
  XLS_ASSIGN_OR_RETURN(int64 lineno_value, lineno.GetValueInt64());
  XLS_ASSIGN_OR_RETURN(int64 colno_value, colno.GetValueInt64());
  return SourceLocation(Fileno(fileno_value), Lineno(lineno_value),
                        Colno(colno_value));
}

xabsl::StatusOr<BValue> Parser::BuildBinaryOrUnaryOp(
    Op op, FunctionBuilder* fb, absl::optional<SourceLocation>* loc,
    ArgParser* arg_parser) {
  std::vector<BValue> operands;

  if (IsOpClass<BinOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(2));
    return fb->AddBinOp(op, operands[0], operands[1], *loc);
  }

  if (IsOpClass<UnOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(1));
    return fb->AddUnOp(op, operands[0], *loc);
  }

  if (IsOpClass<CompareOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(2));
    return fb->AddCompareOp(op, operands[0], operands[1], *loc);
  }

  if (IsOpClass<NaryOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(ArgParser::kVariadic));
    return fb->AddNaryOp(op, operands, *loc);
  }

  if (IsOpClass<BitwiseReductionOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(1));
    return fb->AddBitwiseReductionOp(op, operands[0], *loc);
  }

  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid operation name for IR parsing: \"%s\"", OpToString(op)));
}

namespace {

// GetLocalNode finds function-local BValues by name.
xabsl::StatusOr<Node*> GetLocalNode(
    std::string name, absl::flat_hash_map<std::string, BValue>* name_to_value) {
  auto it = name_to_value->find(name);
  if (it == name_to_value->end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Referred to a name that was not previously defined: %s", name));
  }
  return it->second.node();
}

}  // namespace

xabsl::StatusOr<BValue> Parser::ParseFunctionBody(
    FunctionBuilder* fb,
    absl::flat_hash_map<std::string, BValue>* name_to_value, Package* package) {
  BValue last_created;
  BValue return_value;
  while (!scanner_.PeekTokenIs(LexicalTokenType::kCurlClose)) {
    bool saw_ret = scanner_.TryDropKeyword("ret");

    // <output_name>: <type> = op(...)
    XLS_ASSIGN_OR_RETURN(
        Token output_name,
        scanner_.PopTokenOrError(LexicalTokenType::kIdent, "node output name"));
    if (saw_ret && !scanner_.PeekTokenIs(LexicalTokenType::kColon)) {
      XLS_ASSIGN_OR_RETURN(Node * ret,
                           GetLocalNode(output_name.value(), name_to_value));
      fb->function()->set_return_value(ret);
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(
          LexicalTokenType::kCurlClose, "'}' at end of function body"));
      return BValue(ret, fb);
    }

    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
    XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kEquals));
    XLS_ASSIGN_OR_RETURN(
        Token op_token,
        scanner_.PopTokenOrError(LexicalTokenType::kIdent, "operator"));

    XLS_ASSIGN_OR_RETURN(Op op, StringToOp(op_token.value()));

    ArgParser arg_parser(*name_to_value, type, this);
    absl::optional<SourceLocation>* loc =
        arg_parser.AddOptionalKeywordArg<SourceLocation>("pos");
    BValue bvalue;

    std::vector<BValue> operands;
    switch (op) {
      case Op::kBitSlice: {
        int64* start = arg_parser.AddKeywordArg<int64>("start");
        int64* width = arg_parser.AddKeywordArg<int64>("width");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        bvalue = fb->BitSlice(operands[0], *start, *width, *loc);
        break;
      }
      case Op::kDynamicBitSlice: {
        int64* width = arg_parser.AddKeywordArg<int64>("width");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        bvalue = fb->DynamicBitSlice(operands[0], operands[1], *width, *loc);
        break;
      }
      case Op::kConcat: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->Concat(operands, *loc);
        break;
      }
      case Op::kLiteral: {
        Value* value = arg_parser.AddKeywordArg<Value>("value");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
        bvalue = fb->Literal(*value, *loc);
        break;
      }
      case Op::kMap: {
        std::string* to_apply_name =
            arg_parser.AddKeywordArg<std::string>("to_apply");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        XLS_ASSIGN_OR_RETURN(Function * to_apply,
                             package->GetFunction(*to_apply_name));
        bvalue = fb->Map(operands[0], to_apply, *loc);
        break;
      }
      case Op::kParam: {
        // TODO(meheff): Params should not appear in the body of the
        // function. This is currently required because we have no way of
        // returning a param value otherwise.
        std::string* param_name = arg_parser.AddKeywordArg<std::string>("name");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
        auto it = name_to_value->find(*param_name);
        if (it == name_to_value->end()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Referred to parameter name that hadn't yet been "
                              "defined: %s @ %s",
                              *param_name, op_token.pos().ToHumanString()));
        }
        bvalue = it->second;
        break;
      }
      case Op::kCountedFor: {
        int64* trip_count = arg_parser.AddKeywordArg<int64>("trip_count");
        int64* stride = arg_parser.AddOptionalKeywordArg<int64>("stride", 1);
        std::string* body_name = arg_parser.AddKeywordArg<std::string>("body");
        std::vector<BValue>* invariant_args =
            arg_parser.AddOptionalKeywordArg<std::vector<BValue>>(
                "invariant_args", /*default_value=*/{});
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        XLS_ASSIGN_OR_RETURN(Function * body, package->GetFunction(*body_name));
        bvalue = fb->CountedFor(operands[0], *trip_count, *stride, body,
                                *invariant_args, *loc);
        break;
      }
      case Op::kOneHot: {
        bool* lsb_prio = arg_parser.AddKeywordArg<bool>("lsb_prio");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        bvalue = fb->OneHot(operands[0],
                            *lsb_prio ? LsbOrMsb::kLsb : LsbOrMsb::kMsb, *loc);
        break;
      }
      case Op::kOneHotSel: {
        std::vector<BValue>* case_args =
            arg_parser.AddKeywordArg<std::vector<BValue>>("cases");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        if (case_args->empty()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected at least 1 case @ %s", op_token.pos().ToHumanString()));
        }
        bvalue = fb->OneHotSelect(operands[0], *case_args, *loc);
        break;
      }
      case Op::kSel: {
        std::vector<BValue>* case_args =
            arg_parser.AddKeywordArg<std::vector<BValue>>("cases");
        std::optional<BValue>* default_value =
            arg_parser.AddOptionalKeywordArg<BValue>("default");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        if (case_args->empty()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected at least 1 case @ %s", op_token.pos().ToHumanString()));
        }
        bvalue = fb->Select(operands[0], *case_args, *default_value, *loc);
        break;
      }
      case Op::kTuple: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->Tuple(operands, *loc);
        break;
      }
      case Op::kAfterAll: {
        if (!type->IsToken()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected token type @ %s", op_token.pos().ToHumanString()));
        }
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->AfterAll(operands, *loc);
        break;
      }
      case Op::kArray: {
        if (!type->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected array type @ %s", op_token.pos().ToHumanString()));
        }
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue =
            fb->Array(operands, type->AsArrayOrDie()->element_type(), *loc);
        break;
      }
      case Op::kTupleIndex: {
        int64* index = arg_parser.AddKeywordArg<int64>("index");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        if (!operands[0].GetType()->IsTuple()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("tuple_index operand is not a tuple; got %s @ %s",
                              operands[0].GetType()->ToString(),
                              op_token.pos().ToHumanString()));
        }
        bvalue = fb->TupleIndex(operands[0], *index, *loc);
        break;
      }
      case Op::kArrayIndex: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        if (!operands[0].GetType()->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "array_index operand is not an array; got %s @ %s",
              operands[0].GetType()->ToString(),
              op_token.pos().ToHumanString()));
        }
        bvalue = fb->ArrayIndex(operands[0], operands[1], *loc);
        break;
      }
      case Op::kArrayUpdate: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/3));
        if (!operands[0].GetType()->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "array_update operand is not an array; got %s @ %s",
              operands[0].GetType()->ToString(),
              op_token.pos().ToHumanString()));
        }
        Type* element_type =
            operands[0].GetType()->AsArrayOrDie()->element_type();
        if (operands[2].GetType() != element_type) {
          return absl::InvalidArgumentError(
              absl::StrFormat("array_update update value is not the same type "
                              "as the array elements; got %s @ %s",
                              operands[2].GetType()->ToString(),
                              op_token.pos().ToHumanString()));
        }
        bvalue = fb->ArrayUpdate(operands[0], operands[1], operands[2], *loc);
        break;
      }
      case Op::kInvoke: {
        std::string* to_apply_name =
            arg_parser.AddKeywordArg<std::string>("to_apply");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        XLS_ASSIGN_OR_RETURN(Function * to_apply,
                             package->GetFunction(*to_apply_name));
        bvalue = fb->Invoke(operands, to_apply, *loc);
        break;
      }
      case Op::kZeroExt:
      case Op::kSignExt: {
        int64* new_bit_count = arg_parser.AddKeywordArg<int64>("new_bit_count");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        if (type->IsBits() &&
            type->AsBitsOrDie()->bit_count() != *new_bit_count) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Extend op has an annotated type %s that differs "
                              "from its new_bit_count annotation %d.",
                              type->ToString(), *new_bit_count));
        }
        bvalue = op == Op::kZeroExt
                     ? fb->ZeroExtend(operands[0], *new_bit_count, *loc)
                     : fb->SignExtend(operands[0], *new_bit_count, *loc);
        break;
      }
      case Op::kEncode: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        bvalue = fb->Encode(operands[0], *loc);
        break;
      }
      case Op::kDecode: {
        int64* width = arg_parser.AddKeywordArg<int64>("width");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        if (type->IsBits() && type->AsBitsOrDie()->bit_count() != *width) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Decode op has an annotated type %s that differs "
                              "from its width annotation %d.",
                              type->ToString(), *width));
        }
        bvalue = fb->Decode(operands[0], *width, *loc);
        break;
      }
      case Op::kSMul:
      case Op::kUMul: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        bvalue = fb->AddArithOp(op, operands[0], operands[1],
                                type->AsBitsOrDie()->bit_count(), *loc);
        break;
      }
      default:
        XLS_ASSIGN_OR_RETURN(bvalue,
                             BuildBinaryOrUnaryOp(op, fb, loc, &arg_parser));
    }

    // Verify name is unique
    if (name_to_value->contains(output_name.value())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Name '%s' has already been defined @ %s",
                          output_name.value(), op_token.pos().ToHumanString()));
    }
    (*name_to_value)[output_name.value()] = bvalue;

    // Verify that the type of the newly constructed node matches the parsed
    // type.
    if (bvalue.valid() && type != bvalue.node()->GetType()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Declared type %s does not match expected type %s @ %s",
          type->ToString(), bvalue.GetType()->ToString(),
          op_token.pos().ToHumanString()));
    }

    last_created = bvalue;

    // If the name in the IR dump suggested an ID, we use it directly.
    auto get_suggested_id =
        [](absl::string_view name) -> absl::optional<int64> {
      std::vector<absl::string_view> pieces = absl::StrSplit(name, '.');
      if (pieces.empty()) {
        return absl::nullopt;
      }
      int64 result;
      if (absl::SimpleAtoi(pieces.back(), &result)) {
        return result;
      }
      return absl::nullopt;
    };

    if (last_created.valid()) {
      if (absl::optional<int64> suggested_id =
              get_suggested_id(output_name.value())) {
        last_created.node()->set_id(suggested_id.value());
      }
    }

    if (saw_ret) {
      return_value = last_created;
    }
  }

  if (!return_value.valid()) {
    return_value = last_created;
  }

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlClose,
                                                "'}' at end of function body"));
  return return_value;
}

xabsl::StatusOr<Type*> Parser::ParseTupleType(Package* package) {
  std::vector<Type*> types;
  scanner_.PopToken();
  if (!scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
    do {
      XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
      types.push_back(type);
    } while (scanner_.TryDropToken(LexicalTokenType::kComma));
  }
  if (!scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected ')' to terminate tuple type; found %s",
                        scanner_.PopToken().value()));
  }
  scanner_.PopToken();
  return package->GetTupleType(types);
}

xabsl::StatusOr<std::pair<std::unique_ptr<FunctionBuilder>, Type*>>
Parser::ParseSignature(absl::flat_hash_map<std::string, BValue>* name_to_value,
                       Package* package) {
  XLS_ASSIGN_OR_RETURN(
      Token name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "function name"));
  auto fb = absl::make_unique<FunctionBuilder>(name.value(), package);
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in function parameters"));

  bool must_end = false;
  while (true) {
    if (must_end || scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(
          LexicalTokenType::kParenClose, "')' in function parameters"));
      break;
    }
    XLS_ASSIGN_OR_RETURN(Token param_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
    XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
    (*name_to_value)[param_name.value()] = fb->Param(param_name.value(), type);
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kRightArrow,
                                                "'->' in function signature"));
  XLS_ASSIGN_OR_RETURN(Type * return_type, ParseType(package));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlOpen,
                                                "start of function body"));
  return std::pair<std::unique_ptr<FunctionBuilder>, Type*>{std::move(fb),
                                                            return_type};
}

xabsl::StatusOr<std::string> Parser::ParsePackageName() {
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("package"));
  XLS_ASSIGN_OR_RETURN(
      Token package_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "package name"));
  return package_name.value();
}

xabsl::StatusOr<Function*> Parser::ParseFunction(Package* package) {
  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse function; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("fn"));

  absl::flat_hash_map<std::string, BValue> name_to_value;
  XLS_ASSIGN_OR_RETURN(auto function_data,
                       ParseSignature(&name_to_value, package));
  FunctionBuilder* fb = function_data.first.get();

  XLS_ASSIGN_OR_RETURN(BValue return_value,
                       ParseFunctionBody(fb, &name_to_value, package));

  if (return_value.valid() &&
      return_value.node()->GetType() != function_data.second) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Type of return value %s does not match declared function return type "
        "%s",
        return_value.node()->GetType()->ToString(),
        function_data.second->ToString()));
  }

  // TODO(leary): 2019-02-19 Could be an empty function body, need to decide
  // what to do for those. Accept that the return value can be null and handle
  // everywhere?
  return fb->BuildWithReturnValue(return_value);
}

xabsl::StatusOr<FunctionType*> Parser::ParseFunctionType(Package* package) {
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));
  std::vector<Type*> parameter_types;
  bool must_end = false;
  while (true) {
    if (must_end) {
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(
          LexicalTokenType::kParenClose,
          "expected end of function-type parameter list"));
      break;
    }
    if (scanner_.TryDropToken(LexicalTokenType::kParenClose)) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
    parameter_types.push_back(type);
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kRightArrow));
  XLS_ASSIGN_OR_RETURN(Type * return_type, ParseType(package));

  return package->GetFunctionType(parameter_types, return_type);
}

/* static */ xabsl::StatusOr<FunctionType*> Parser::ParseFunctionType(
    absl::string_view input_string, Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseFunctionType(package);
}

/* static */ xabsl::StatusOr<Type*> Parser::ParseType(
    absl::string_view input_string, Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseType(package);
}

// Verifies the given package. Replaces InternalError status codes with
// InvalidArgument status code which is more appropriate for the parser.
static absl::Status VerifyPackage(Package* package) {
  absl::Status status = Verify(package);
  if (!status.ok() && status.code() == absl::StatusCode::kInternal) {
    return absl::InvalidArgumentError(status.message());
  }
  return status;
}

/* static */
xabsl::StatusOr<Function*> Parser::ParseFunction(absl::string_view input_string,
                                                 Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  XLS_ASSIGN_OR_RETURN(Function * function, p.ParseFunction(package));

  // Verify the whole package because the addition of the function may break
  // package-scoped invariants (eg, duplicate function name).
  XLS_RETURN_IF_ERROR(VerifyPackage(package));
  return function;
}

/* static */
xabsl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackage(
    absl::string_view input_string,
    absl::optional<absl::string_view> filename) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ParsePackageNoVerify(input_string, filename));
  XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
  return package;
}

/* static */
xabsl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackageWithEntry(
    absl::string_view input_string, absl::string_view entry,
    absl::optional<absl::string_view> filename) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ParsePackageNoVerify(input_string, filename, entry));
  XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
  return package;
}

/* static */
xabsl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackageNoVerify(
    absl::string_view input_string, absl::optional<absl::string_view> filename,
    absl::optional<absl::string_view> entry) {
  return ParseDerivedPackageNoVerify<Package>(input_string, filename, entry);
}

/* static */
xabsl::StatusOr<Value> Parser::ParseValue(absl::string_view input_string,
                                          Type* expected_type) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseValueInternal(expected_type);
}

/* static */
xabsl::StatusOr<Value> Parser::ParseTypedValue(absl::string_view input_string) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseValueInternal(/*expected_type=*/absl::nullopt);
}

}  // namespace xls
