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

#include "xls/ir/ir_parser.h"

#include "google/protobuf/text_format.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/verifier.h"

namespace xls {

absl::StatusOr<int64> Parser::ParseBitsTypeAndReturnWidth() {
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

absl::StatusOr<Type*> Parser::ParseBitsType(Package* package) {
  XLS_ASSIGN_OR_RETURN(int64 bit_count, ParseBitsTypeAndReturnWidth());
  return package->GetBitsType(bit_count);
}

absl::StatusOr<Type*> Parser::ParseType(Package* package) {
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
  absl::Status SetOrReturn(absl::StatusOr<T> value_status) {
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
  absl::StatusOr<std::vector<BValue>> Run(int64 arity) {
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

absl::StatusOr<int64> Parser::ParseInt64() {
  XLS_ASSIGN_OR_RETURN(Token literal,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  return literal.GetValueInt64();
}

absl::StatusOr<bool> Parser::ParseBool() {
  XLS_ASSIGN_OR_RETURN(Token literal,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  return literal.GetValueBool();
}

absl::StatusOr<std::string> Parser::ParseIdentifierString(TokenPos* pos) {
  XLS_ASSIGN_OR_RETURN(Token token,
                       scanner_.PopTokenOrError(LexicalTokenType::kIdent));
  if (pos != nullptr) {
    *pos = token.pos();
  }
  return token.value();
}

absl::StatusOr<BValue> Parser::ParseIdentifierValue(
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

absl::StatusOr<Value> Parser::ParseValueInternal(absl::optional<Type*> type) {
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
      if (scanner_.TryDropKeyword("token")) {
        type_kind = TypeKind::kToken;
      } else {
        type_kind = TypeKind::kBits;
        XLS_ASSIGN_OR_RETURN(bit_count, ParseBitsTypeAndReturnWidth());
        XLS_RETURN_IF_ERROR(
            scanner_.DropTokenOrError(LexicalTokenType::kColon));
      }
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
  if (type_kind == TypeKind::kToken) {
    return Value::Token();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported type %s", TypeKindToString(type_kind)));
}

absl::StatusOr<std::vector<BValue>> Parser::ParseNameList(
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

absl::StatusOr<SourceLocation> Parser::ParseSourceLocation() {
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

absl::StatusOr<BValue> Parser::BuildBinaryOrUnaryOp(
    Op op, BuilderBase* fb, absl::optional<SourceLocation>* loc,
    absl::string_view node_name, ArgParser* arg_parser) {
  std::vector<BValue> operands;

  if (IsOpClass<BinOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(2));
    return fb->AddBinOp(op, operands[0], operands[1], *loc, node_name);
  }

  if (IsOpClass<UnOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(1));
    return fb->AddUnOp(op, operands[0], *loc, node_name);
  }

  if (IsOpClass<CompareOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(2));
    return fb->AddCompareOp(op, operands[0], operands[1], *loc, node_name);
  }

  if (IsOpClass<NaryOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(ArgParser::kVariadic));
    return fb->AddNaryOp(op, operands, *loc, node_name);
  }

  if (IsOpClass<BitwiseReductionOp>(op)) {
    XLS_ASSIGN_OR_RETURN(operands, arg_parser->Run(1));
    return fb->AddBitwiseReductionOp(op, operands[0], *loc, node_name);
  }

  return absl::InvalidArgumentError(absl::StrFormat(
      "Invalid operation name for IR parsing: \"%s\"", OpToString(op)));
}

namespace {

// GetLocalNode finds function-local BValues by name.
absl::StatusOr<Node*> GetLocalNode(
    std::string name, absl::flat_hash_map<std::string, BValue>* name_to_value) {
  auto it = name_to_value->find(name);
  if (it == name_to_value->end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Referred to a name that was not previously defined: %s", name));
  }
  return it->second.node();
}

// Splits node names of the form (.*)\.([0-9]+) into the string and integer
// components and returns them. For example, "add.42" will be returned
// as {"add", 42}. If the name does not match the pattern then nullopt is
// returned.
struct SplitName {
  std::string op_name;
  int64 node_id;
};
absl::optional<SplitName> SplitNodeName(absl::string_view name) {
  std::vector<absl::string_view> pieces = absl::StrSplit(name, '.');
  if (pieces.empty()) {
    return absl::nullopt;
  }
  int64 result;
  if (!absl::SimpleAtoi(pieces.back(), &result)) {
    return absl::nullopt;
  }
  pieces.pop_back();
  return SplitName{.op_name = absl::StrJoin(pieces, "."), .node_id = result};
}

}  // namespace

absl::StatusOr<BValue> Parser::ParseFunctionBody(
    BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value,
    Package* package) {
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
      XLS_RETURN_IF_ERROR(fb->function()->set_return_value(ret));
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
    absl::optional<int64>* id_attribute =
        arg_parser.AddOptionalKeywordArg<int64>("id");
    BValue bvalue;

    absl::optional<SplitName> split_name = SplitNodeName(output_name.value());
    // If output_name has the form <op>.<id> (e.g, "add.42"), then no name
    // should be given when constructing the node as the name is autogenerated
    // (the node has no meaningful given name). Otherwise, output_name is the
    // name of the node.
    std::string node_name = split_name.has_value() ? "" : output_name.value();

    std::vector<BValue> operands;
    switch (op) {
      case Op::kBitSlice: {
        int64* start = arg_parser.AddKeywordArg<int64>("start");
        int64* width = arg_parser.AddKeywordArg<int64>("width");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        bvalue = fb->BitSlice(operands[0], *start, *width, *loc, node_name);
        break;
      }
      case Op::kDynamicBitSlice: {
        int64* width = arg_parser.AddKeywordArg<int64>("width");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        bvalue = fb->DynamicBitSlice(operands[0], operands[1], *width, *loc,
                                     node_name);
        break;
      }
      case Op::kConcat: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->Concat(operands, *loc, node_name);
        break;
      }
      case Op::kLiteral: {
        Value* value = arg_parser.AddKeywordArg<Value>("value");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
        bvalue = fb->Literal(*value, *loc, node_name);
        break;
      }
      case Op::kMap: {
        std::string* to_apply_name =
            arg_parser.AddKeywordArg<std::string>("to_apply");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        XLS_ASSIGN_OR_RETURN(Function * to_apply,
                             package->GetFunction(*to_apply_name));
        bvalue = fb->Map(operands[0], to_apply, *loc, node_name);
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
                                *invariant_args, *loc, node_name);
        break;
      }
      case Op::kOneHot: {
        bool* lsb_prio = arg_parser.AddKeywordArg<bool>("lsb_prio");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        bvalue =
            fb->OneHot(operands[0], *lsb_prio ? LsbOrMsb::kLsb : LsbOrMsb::kMsb,
                       *loc, node_name);
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
        bvalue = fb->OneHotSelect(operands[0], *case_args, *loc, node_name);
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
        bvalue = fb->Select(operands[0], *case_args, *default_value, *loc,
                            node_name);
        break;
      }
      case Op::kTuple: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->Tuple(operands, *loc, node_name);
        break;
      }
      case Op::kAfterAll: {
        if (!type->IsToken()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected token type @ %s", op_token.pos().ToHumanString()));
        }
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->AfterAll(operands, *loc, node_name);
        break;
      }
      case Op::kArray: {
        if (!type->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected array type @ %s", op_token.pos().ToHumanString()));
        }
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->Array(operands, type->AsArrayOrDie()->element_type(), *loc,
                           node_name);
        break;
      }
      case Op::kTupleIndex: {
        int64* index = arg_parser.AddKeywordArg<int64>("index");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        if (operands[0].valid() && !operands[0].GetType()->IsTuple()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("tuple_index operand is not a tuple; got %s @ %s",
                              operands[0].GetType()->ToString(),
                              op_token.pos().ToHumanString()));
        }
        bvalue = fb->TupleIndex(operands[0], *index, *loc, node_name);
        break;
      }
      case Op::kArrayIndex: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        if (operands[0].valid() && !operands[0].GetType()->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "array_index operand is not an array; got %s @ %s",
              operands[0].GetType()->ToString(),
              op_token.pos().ToHumanString()));
        }
        bvalue = fb->ArrayIndex(operands[0], operands[1], *loc, node_name);
        break;
      }
      case Op::kArrayUpdate: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/3));
        if (operands[0].valid() && !operands[0].GetType()->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "array_update operand is not an array; got %s @ %s",
              operands[0].GetType()->ToString(),
              op_token.pos().ToHumanString()));
        }
        if (operands[0].valid() && operands[2].valid()) {
          Type* element_type =
              operands[0].GetType()->AsArrayOrDie()->element_type();
          if (operands[2].GetType() != element_type) {
            return absl::InvalidArgumentError(absl::StrFormat(
                "array_update update value is not the same type "
                "as the array elements; got %s @ %s",
                operands[2].GetType()->ToString(),
                op_token.pos().ToHumanString()));
          }
        }
        bvalue = fb->ArrayUpdate(operands[0], operands[1], operands[2], *loc,
                                 node_name);
        break;
      }
      case Op::kArrayConcat: {
        // fb->ArrayConcat will check that all operands are of an array
        // type and that all concat'ed arrays have the same element type.
        //
        // for now, just check that type is an Array
        if (!type->IsArray()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Expected array type @ %s, got %s",
              op_token.pos().ToHumanString(), type->ToString()));
        }

        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        bvalue = fb->ArrayConcat(operands, *loc, node_name);
        break;
      }
      case Op::kInvoke: {
        std::string* to_apply_name =
            arg_parser.AddKeywordArg<std::string>("to_apply");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
        XLS_ASSIGN_OR_RETURN(Function * to_apply,
                             package->GetFunction(*to_apply_name));
        bvalue = fb->Invoke(operands, to_apply, *loc, node_name);
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
        bvalue =
            op == Op::kZeroExt
                ? fb->ZeroExtend(operands[0], *new_bit_count, *loc, node_name)
                : fb->SignExtend(operands[0], *new_bit_count, *loc, node_name);
        break;
      }
      case Op::kEncode: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        bvalue = fb->Encode(operands[0], *loc, node_name);
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
        bvalue = fb->Decode(operands[0], *width, *loc, node_name);
        break;
      }
      case Op::kSMul:
      case Op::kUMul: {
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        bvalue =
            fb->AddArithOp(op, operands[0], operands[1],
                           type->AsBitsOrDie()->bit_count(), *loc, node_name);
        break;
      }
      case Op::kReceive: {
        int64* channel_id = arg_parser.AddKeywordArg<int64>("channel_id");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        // Get the channel from the package.
        if (!package->HasChannelWithId(*channel_id)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "No such channel with channel ID %d", *channel_id));
        }
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             package->GetChannel(*channel_id));
        Type* expected_type = package->GetReceiveType(channel);
        if (expected_type != type) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Receive op type is type: %s. Expected: %s",
                              type->ToString(), expected_type->ToString()));
        }
        bvalue = fb->Receive(channel, operands[0], *loc, node_name);
        break;
      }
      case Op::kReceiveIf: {
        int64* channel_id = arg_parser.AddKeywordArg<int64>("channel_id");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        // Get the channel from the package.
        if (!package->HasChannelWithId(*channel_id)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "No such channel with channel ID %d", *channel_id));
        }
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             package->GetChannel(*channel_id));
        Type* expected_type = package->GetReceiveType(channel);
        if (expected_type != type) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Receive_if op type is type: %s. Expected: %s",
                              type->ToString(), expected_type->ToString()));
        }
        bvalue =
            fb->ReceiveIf(channel, operands[0], operands[1], *loc, node_name);
        break;
      }
      case Op::kSend: {
        int64* channel_id = arg_parser.AddKeywordArg<int64>("channel_id");
        std::vector<BValue>* data_args =
            arg_parser.AddKeywordArg<std::vector<BValue>>("data");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
        // Get the channel from the package.
        if (!package->HasChannelWithId(*channel_id)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "No such channel with channel ID %d", *channel_id));
        }
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             package->GetChannel(*channel_id));
        // The first operand is the token, and the remaining are the data
        // values to send.
        bvalue = fb->Send(channel, operands[0], *data_args, *loc, node_name);
        break;
      }
      case Op::kSendIf: {
        int64* channel_id = arg_parser.AddKeywordArg<int64>("channel_id");
        std::vector<BValue>* data_args =
            arg_parser.AddKeywordArg<std::vector<BValue>>("data");
        XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
        // Get the channel from the package.
        if (!package->HasChannelWithId(*channel_id)) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "No such channel with channel ID %d", *channel_id));
        }
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             package->GetChannel(*channel_id));
        // The first operand is the token, and the remaining are the data
        // values to send.
        bvalue = fb->SendIf(channel, /*token=*/operands[0],
                            /*pred=*/operands[1], *data_args, *loc, node_name);
        break;
      }
      default:
        XLS_ASSIGN_OR_RETURN(
            bvalue, BuildBinaryOrUnaryOp(op, fb, loc, node_name, &arg_parser));
    }

    // Verify name is unique. Skip Params because these nodes are already added
    // to the name_to_value map during function signature parsing.
    if (name_to_value->contains(output_name.value()) &&
        (!bvalue.valid() || !bvalue.node()->Is<Param>())) {
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

    if (last_created.valid()) {
      Node* node = last_created.node();
      if (split_name.has_value()) {
        // If the name is a generated from opcode and id (e.g., "add.42") then
        // verify the opcode and id attribute (if given) match then set the id.
        if (id_attribute->has_value() &&
            id_attribute->value() != split_name->node_id) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "The id '%d' in node name %s does not match the id '%d' "
              "specified as an attribute @ %s",
              split_name->node_id, output_name.value(), id_attribute->value(),
              op_token.pos().ToHumanString()));
        }
        node->set_id(split_name->node_id);
        if (split_name->op_name != OpToString(node->op())) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "The substring '%s' in node name %s does not match the node op "
              "'%s' @ %s",
              split_name->op_name, output_name.value(), OpToString(node->op()),
              op_token.pos().ToHumanString()));
        }
      } else {
        // Otherwise, the output_name is a non-generated name. Verify a name
        // was assigned to the op. OK to XLS_RET_CHECK as a mismatch here is an
        // error in the parser not in the input file.
        XLS_RET_CHECK(node->HasAssignedName());
        // Also set the ID to the attribute ID (if given).
        if (id_attribute->has_value()) {
          node->set_id(id_attribute->value());
        }
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

absl::StatusOr<Type*> Parser::ParseTupleType(Package* package) {
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

absl::StatusOr<std::pair<std::unique_ptr<FunctionBuilder>, Type*>>
Parser::ParseFunctionSignature(
    absl::flat_hash_map<std::string, BValue>* name_to_value, Package* package) {
  XLS_ASSIGN_OR_RETURN(
      Token name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "function name"));
  // The parser does its own verification so pass should_verify=false. This
  // enables the parser to parse and construct malformed IR for tests.
  auto fb = absl::make_unique<FunctionBuilder>(name.value(), package,
                                               /*should_verify=*/false);
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

absl::StatusOr<std::unique_ptr<ProcBuilder>> Parser::ParseProcSignature(
    absl::flat_hash_map<std::string, BValue>* name_to_value, Package* package) {
  // Proc definition begins with something like:
  //
  //   proc foo(state: bits[32], tok: token, init=42) {
  //     ...
  //
  // The signature being parsed by this method starts at the proc name and ends
  // with the open brace.
  XLS_ASSIGN_OR_RETURN(Token name, scanner_.PopTokenOrError(
                                       LexicalTokenType::kIdent, "proc name"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in proc parameters"));

  // Parse the token parameter.
  XLS_ASSIGN_OR_RETURN(Token token_name,
                       scanner_.PopTokenOrError(LexicalTokenType::kIdent));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_.PeekToken());
  const TokenPos token_type_pos = peek.pos();
  XLS_ASSIGN_OR_RETURN(Type * token_type, ParseType(package));
  if (!token_type->IsToken()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected second argument of proc to be token type, is: %s @ %s",
        token_type->ToString(), token_type_pos.ToHumanString()));
  }

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));

  // Parse the state parameter.
  XLS_ASSIGN_OR_RETURN(Token state_name,
                       scanner_.PopTokenOrError(LexicalTokenType::kIdent));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
  XLS_ASSIGN_OR_RETURN(Type * state_type, ParseType(package));

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));

  // Parse "init=VALUE".
  XLS_ASSIGN_OR_RETURN(
      Token init_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "argument"));
  if (init_name.value() != "init") {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected 'init' attribute @ %s", token_type_pos.ToHumanString()));
  }
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kEquals));
  XLS_ASSIGN_OR_RETURN(Value init_value, ParseValueInternal(state_type));

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenClose,
                                                "')' in proc parameters"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlOpen,
                                                "start of proc body"));

  // The parser does its own verification so pass should_verify=false. This
  // enables the parser to parse and construct malformed IR for tests.
  auto builder = absl::make_unique<ProcBuilder>(
      name.value(), init_value, token_name.value(), state_name.value(), package,
      /*should_verify=*/false);
  (*name_to_value)[token_name.value()] = builder->GetTokenParam();
  (*name_to_value)[state_name.value()] = builder->GetStateParam();

  return std::move(builder);
}

absl::StatusOr<std::string> Parser::ParsePackageName() {
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("package"));
  XLS_ASSIGN_OR_RETURN(
      Token package_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "package name"));
  return package_name.value();
}

absl::StatusOr<Function*> Parser::ParseFunction(Package* package) {
  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse function; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("fn"));

  absl::flat_hash_map<std::string, BValue> name_to_value;
  XLS_ASSIGN_OR_RETURN(auto function_data,
                       ParseFunctionSignature(&name_to_value, package));
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

absl::StatusOr<Proc*> Parser::ParseProc(Package* package) {
  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse proc; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("proc"));

  absl::flat_hash_map<std::string, BValue> name_to_value;
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProcBuilder> pb,
                       ParseProcSignature(&name_to_value, package));

  XLS_ASSIGN_OR_RETURN(BValue return_value,
                       ParseFunctionBody(pb.get(), &name_to_value, package));

  if (return_value.valid() &&
      return_value.node()->GetType() != pb->proc()->ReturnType()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Type of return value %s does not match declared proc return type %s",
        return_value.node()->GetType()->ToString(),
        pb->proc()->ReturnType()->ToString()));
  }

  return pb->BuildWithReturnValue(return_value);
}

absl::StatusOr<Channel*> Parser::ParseChannel(Package* package) {
  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse channel; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("chan"));
  XLS_ASSIGN_OR_RETURN(
      Token channel_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "channel name"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in channel definition"));
  absl::optional<int64> id;
  absl::optional<ChannelKind> kind;
  absl::optional<ChannelMetadataProto> metadata;
  std::vector<DataElement> data_elements;
  bool must_end = false;

  // Iterate through the comma-separated elements in the channel definition.
  // Example:
  //
  //  chan my_channel(foo: bits[32], id=42, ...)
  //
  while (true) {
    if (must_end || scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(
          LexicalTokenType::kParenClose, "')' in channel definition"));
      break;
    }
    XLS_ASSIGN_OR_RETURN(Token field_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    if (scanner_.PeekTokenIs(LexicalTokenType::kColon)) {
      // Data element: "<name>: <type>"
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
      XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
      data_elements.push_back(DataElement{field_name.value(), type});
    } else if (scanner_.TryDropToken(LexicalTokenType::kEquals)) {
      // Attribute: "<name>=<value>"
      if (field_name.value() == "id") {
        XLS_ASSIGN_OR_RETURN(Token id_token, scanner_.PopTokenOrError(
                                                 LexicalTokenType::kLiteral));
        XLS_ASSIGN_OR_RETURN(id, id_token.GetValueInt64());
      } else if (field_name.value() == "kind") {
        XLS_ASSIGN_OR_RETURN(Token kind_token, scanner_.PopTokenOrError(
                                                   LexicalTokenType::kIdent));
        if (kind_token.value() == "send_only") {
          kind = ChannelKind::kSendOnly;
        } else if (kind_token.value() == "receive_only") {
          kind = ChannelKind::kReceiveOnly;
        } else if (kind_token.value() == "send_receive") {
          kind = ChannelKind::kSendReceive;
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Invalid channel kind \"%s\" @ %s. Expected: send_only, "
              "receive_only, or send_receive",
              kind_token.value(), field_name.pos().ToHumanString()));
        }
      } else if (field_name.value() == "metadata") {
        // The metadata is serialized as a text proto.
        XLS_ASSIGN_OR_RETURN(
            Token metadata_token,
            scanner_.PopTokenOrError(LexicalTokenType::kQuotedString));
        ChannelMetadataProto proto;
        bool success =
            google::protobuf::TextFormat::ParseFromString(metadata_token.value(), &proto);
        if (!success) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Invalid channel metadata @ %s",
                              metadata_token.pos().ToHumanString()));
        }
        metadata = proto;
      } else {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid channel attribute \"%s\" @ %s", field_name.value(),
            field_name.pos().ToHumanString()));
      }
    }
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }

  auto error = [&](absl::string_view message) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "%s @ %s", message, channel_name.pos().ToHumanString()));
  };
  if (!id.has_value()) {
    return error("Missing channel id");
  }
  if (!kind.has_value()) {
    return error("Missing channel kind");
  }
  if (!metadata.has_value()) {
    return error("Missing channel metadata");
  }
  if (data_elements.empty()) {
    return error("Channel has no data elements");
  }

  return package->CreateChannelWithId(channel_name.value(), *id, *kind,
                                      data_elements, *metadata);
}

absl::StatusOr<FunctionType*> Parser::ParseFunctionType(Package* package) {
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

/* static */ absl::StatusOr<FunctionType*> Parser::ParseFunctionType(
    absl::string_view input_string, Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseFunctionType(package);
}

/* static */ absl::StatusOr<Type*> Parser::ParseType(
    absl::string_view input_string, Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseType(package);
}

// Verifies the given package. Replaces InternalError status codes with
// InvalidArgument status code which is more appropriate for the parser.
static absl::Status VerifyAndSwapError(Package* package) {
  absl::Status status = VerifyPackage(package);
  if (!status.ok() && status.code() == absl::StatusCode::kInternal) {
    return absl::InvalidArgumentError(status.message());
  }
  return status;
}

/* static */
absl::StatusOr<Function*> Parser::ParseFunction(absl::string_view input_string,
                                                Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  XLS_ASSIGN_OR_RETURN(Function * function, p.ParseFunction(package));

  // Verify the whole package because the addition of the function may break
  // package-scoped invariants (eg, duplicate function name).
  XLS_RETURN_IF_ERROR(VerifyAndSwapError(package));
  return function;
}

/* static */
absl::StatusOr<Proc*> Parser::ParseProc(absl::string_view input_string,
                                        Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  XLS_ASSIGN_OR_RETURN(Proc * proc, p.ParseProc(package));

  // Verify the whole package because the addition of the proc may break
  // package-scoped invariants (eg, duplicate proc name).
  XLS_RETURN_IF_ERROR(VerifyAndSwapError(package));
  return proc;
}

/* static */
absl::StatusOr<Channel*> Parser::ParseChannel(absl::string_view input_string,
                                              Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseChannel(package);
}

/* static */
absl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackage(
    absl::string_view input_string,
    absl::optional<absl::string_view> filename) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ParsePackageNoVerify(input_string, filename));
  XLS_RETURN_IF_ERROR(VerifyAndSwapError(package.get()));
  return package;
}

/* static */
absl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackageWithEntry(
    absl::string_view input_string, absl::string_view entry,
    absl::optional<absl::string_view> filename) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ParsePackageNoVerify(input_string, filename, entry));
  XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
  return package;
}

/* static */
absl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackageNoVerify(
    absl::string_view input_string, absl::optional<absl::string_view> filename,
    absl::optional<absl::string_view> entry) {
  return ParseDerivedPackageNoVerify<Package>(input_string, filename, entry);
}

/* static */
absl::StatusOr<Value> Parser::ParseValue(absl::string_view input_string,
                                         Type* expected_type) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseValueInternal(expected_type);
}

/* static */
absl::StatusOr<Value> Parser::ParseTypedValue(absl::string_view input_string) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseValueInternal(/*expected_type=*/absl::nullopt);
}

}  // namespace xls
