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

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "google/protobuf/text_format.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/code_template.h"
#include "xls/ir/fileno.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/format_strings.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"

namespace xls {

absl::Status Parser::ParseKeywordArguments(
    const absl::flat_hash_map<std::string, std::function<absl::Status()>>&
        handlers,
    absl::Span<const std::string> mandatory_keywords) {
  absl::flat_hash_set<std::string> seen_keywords;
  while (scanner_.PeekTokenIs(LexicalTokenType::kIdent) ||
         scanner_.PeekTokenIs(LexicalTokenType::kKeyword)) {
    XLS_ASSIGN_OR_RETURN(Token name,
                         scanner_.PopKeywordOrIdentToken("argument"));
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kEquals));
    if (!seen_keywords.insert(name.value()).second) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate keyword argument `%s` @ %s", name.value(),
                          name.pos().ToHumanString()));
    }
    if (!handlers.contains(name.value())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid keyword argument `%s` @ %s", name.value(),
                          name.pos().ToHumanString()));
    }
    XLS_RETURN_IF_ERROR(handlers.at(name.value())());
    if (!scanner_.TryDropToken(LexicalTokenType::kComma)) {
      break;
    }
  }

  // Verify all mandatory keywords are present.
  for (const std::string& keyword : mandatory_keywords) {
    if (!seen_keywords.contains(keyword)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Mandatory keyword argument `%s` not found @ %s", keyword,
          scanner_.PeekTokenOrDie().pos().ToHumanString()));
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<std::vector<Parser::TypedArgument>> Parser::ParseTypedArguments(
    Package* package) {
  std::vector<Parser::TypedArgument> args;
  if (scanner_.PeekTokenIs(LexicalTokenType::kIdent)) {
    do {
      if (scanner_.PeekNthTokenIs(1, LexicalTokenType::kEquals)) {
        // Found `XXXX=...` which means that typed arguments are complete and
        // keyword arguments have begun.
        break;
      }
      XLS_ASSIGN_OR_RETURN(Token name,
                           scanner_.PopTokenOrError(LexicalTokenType::kIdent));
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
      XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
      args.push_back(TypedArgument{name.value(), type, name});
    } while (scanner_.TryDropToken(LexicalTokenType::kComma));
  }
  return args;
}

absl::StatusOr<int64_t> Parser::ParseBitsTypeAndReturnWidth() {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_.PeekToken());
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("bits"));
  XLS_RETURN_IF_ERROR(
      scanner_.DropTokenOrError(LexicalTokenType::kBracketOpen));
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, ParseInt64());
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
  XLS_ASSIGN_OR_RETURN(int64_t bit_count, ParseBitsTypeAndReturnWidth());
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
    XLS_ASSIGN_OR_RETURN(int64_t size, ParseInt64());
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
  std::optional<T> optional_value;

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

// Structs used in argument parsing for capturing string attributes. Example
// uses:
//
//   arg_parser.AddKeywordArg<IdentifierString>("to_apply");
//   arg_parser.AddKeywordArg<QuotedString>("message");
//
// These structs are necessary to differentiate between various attribute types
// which all are represented as std::string.
struct IdentifierString {
  std::string value;
};
struct QuotedString {
  std::string value;
};

// Variant which gathers all the possible keyword argument types. New
// keywords arguments which require a new type should be added here.
using KeywordVariant =
    std::variant<KeywordValue<int64_t>, KeywordValue<IdentifierString>,
                 KeywordValue<QuotedString>, KeywordValue<BValue>,
                 KeywordValue<std::vector<BValue>>, KeywordValue<Value>,
                 KeywordValue<SourceInfo>, KeywordValue<bool>>;

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
    mandatory_keywords_.push_back(key);
    auto pair = keywords_.emplace(
        key, std::make_unique<KeywordVariant>(KeywordValue<T>()));
    CHECK(pair.second);
    auto& keyword_value = std::get<KeywordValue<T>>(*pair.first->second);
    keyword_value.is_optional = false;
    // Return a pointer into the KeywordValue which will be filled in when Run
    // is called.
    return &keyword_value.value;
  }

  // Adds an optional keyword to the parser. After calling ArgParser::Run the
  // std::optional pointed to by the returned pointer will (optionally) contain
  // the keyword argument value.
  template <typename T>
  std::optional<T>* AddOptionalKeywordArg(std::string_view key) {
    auto pair = keywords_.emplace(
        key, std::make_unique<KeywordVariant>(KeywordValue<T>()));
    CHECK(pair.second);
    auto& keyword_value = std::get<KeywordValue<T>>(*keywords_.at(key));
    keyword_value.is_optional = true;
    // Return a pointer into the KeywordValue which will be filled in when Run
    // is called.
    return &keyword_value.optional_value;
  }

  template <typename T>
  T* AddOptionalKeywordArg(std::string_view key, T default_value) {
    auto pair = keywords_.emplace(
        key, std::make_unique<KeywordVariant>(KeywordValue<T>()));
    CHECK(pair.second);
    auto& keyword_value = std::get<KeywordValue<T>>(*keywords_.at(key));
    keyword_value.optional_value = default_value;
    keyword_value.is_optional = true;
    // Return a pointer into the KeywordValue which may be filled in when Run is
    // called; however, if it is not filled, will remain the default value
    // provided.
    return &keyword_value.optional_value.value();
  }

  // Runs the argument parser. 'arity' is the expected number of operands
  // (positional arguments). Returns the BValues of the operands.
  static constexpr int64_t kVariadic = -1;
  absl::StatusOr<std::vector<BValue>> Run(int64_t arity) {
    std::vector<BValue> operands;
    XLS_ASSIGN_OR_RETURN(Token open_paren, parser_->scanner_.PopTokenOrError(
                                               LexicalTokenType::kParenOpen));
    if (!parser_->scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
      // Variable indicating whether we are parsing the keywords or still
      // parsing the positional arguments.
      do {
        if (parser_->scanner_.PeekNthTokenIs(1, LexicalTokenType::kEquals)) {
          // Found `XXXX=...` which means that operands are complete and keyword
          // arguments have begun.
          break;
        }
        XLS_ASSIGN_OR_RETURN(
            Token name, parser_->scanner_.PopKeywordOrIdentToken("argument"));
        if (!name_to_bvalue_.contains(name.value())) {
          return absl::InvalidArgumentError(
              absl::StrFormat("Referred to a name @ %s that was not previously "
                              "defined: \"%s\"",
                              name.pos().ToHumanString(), name.value()));
        }
        operands.push_back(name_to_bvalue_.at(name.value()));
      } while (parser_->scanner_.TryDropToken(LexicalTokenType::kComma));
    }

    // Parse comma-separated list of keyword arguments (e.g., `foo=bar`, if any.
    absl::flat_hash_map<std::string, std::function<absl::Status()>>
        keyword_handlers;
    for (const auto& pair : keywords_) {
      const std::string& keyword = pair.first;
      keyword_handlers[keyword] = [this, keyword] {
        return ParseKeywordArg(keyword);
      };
    }
    XLS_RETURN_IF_ERROR(
        parser_->ParseKeywordArguments(keyword_handlers, mandatory_keywords_));

    XLS_RETURN_IF_ERROR(
        parser_->scanner_.DropTokenOrError(LexicalTokenType::kParenClose));

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
  absl::Status ParseKeywordArg(std::string_view key) {
    KeywordVariant& keyword_variant = *keywords_.at(key);
    return absl::visit(
        Visitor{[&](KeywordValue<bool>& v) {
                  return v.SetOrReturn(parser_->ParseBool());
                },
                [&](KeywordValue<int64_t>& v) {
                  return v.SetOrReturn(parser_->ParseInt64());
                },
                [&](KeywordValue<IdentifierString>& v) -> absl::Status {
                  XLS_ASSIGN_OR_RETURN(std::string identifier,
                                       parser_->ParseIdentifier());
                  return v.SetOrReturn(IdentifierString{identifier});
                },
                [&](KeywordValue<QuotedString>& v) -> absl::Status {
                  XLS_ASSIGN_OR_RETURN(std::string quoted_string,
                                       parser_->ParseQuotedString());
                  return v.SetOrReturn(QuotedString{quoted_string});
                },
                [&](KeywordValue<Value>& v) {
                  return v.SetOrReturn(parser_->ParseValueInternal(node_type_));
                },
                [&](KeywordValue<BValue>& v) {
                  return v.SetOrReturn(
                      parser_->ParseAndResolveIdentifier(name_to_bvalue_));
                },
                [&](KeywordValue<std::vector<BValue>>& v) {
                  return v.SetOrReturn(parser_->ParseNameList(name_to_bvalue_));
                },
                [&](KeywordValue<SourceInfo>& v) {
                  return v.SetOrReturn(parser_->ParseSourceInfo());
                }},
        keyword_variant);
    return absl::OkStatus();
  }

  const absl::flat_hash_map<std::string, BValue>& name_to_bvalue_;
  Type* node_type_;
  Parser* parser_;

  std::vector<std::string> mandatory_keywords_;
  absl::flat_hash_map<std::string, std::unique_ptr<KeywordVariant>> keywords_;
};

absl::StatusOr<int64_t> Parser::ParseInt64() {
  XLS_ASSIGN_OR_RETURN(Token literal,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  return literal.GetValueInt64();
}

absl::StatusOr<bool> Parser::ParseBool() {
  XLS_ASSIGN_OR_RETURN(Token literal,
                       scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
  return literal.GetValueBool();
}

absl::StatusOr<std::string> Parser::ParseIdentifier(TokenPos* pos) {
  XLS_ASSIGN_OR_RETURN(Token token,
                       scanner_.PopTokenOrError(LexicalTokenType::kIdent));
  if (pos != nullptr) {
    *pos = token.pos();
  }
  return token.value();
}

absl::StatusOr<std::string> Parser::ParseQuotedString(TokenPos* pos) {
  XLS_ASSIGN_OR_RETURN(
      Token token, scanner_.PopTokenOrError(LexicalTokenType::kQuotedString));
  if (pos != nullptr) {
    *pos = token.pos();
  }
  return token.value();
}

absl::StatusOr<BValue> Parser::ParseAndResolveIdentifier(
    const absl::flat_hash_map<std::string, BValue>& name_to_value) {
  TokenPos start_pos;
  XLS_ASSIGN_OR_RETURN(std::string identifier, ParseIdentifier(&start_pos));
  auto it = name_to_value.find(identifier);
  if (it == name_to_value.end()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Referred to a name @ %s that was not previously defined: \"%s\"",
        start_pos.ToHumanString(), identifier));
  }
  return it->second;
}

absl::StatusOr<Value> Parser::ParseValueInternal(std::optional<Type*> type) {
  XLS_ASSIGN_OR_RETURN(Token peek, scanner_.PeekToken());
  const TokenPos start_pos = peek.pos();
  TypeKind type_kind;
  int64_t bit_count = 0;
  if (type.has_value()) {
    type_kind = type.value()->kind();
    bit_count =
        type.value()->IsBits() ? type.value()->AsBitsOrDie()->bit_count() : 0;
  } else {
    if (scanner_.PeekTokenIs(LexicalTokenType::kKeyword)) {
      if (scanner_.TryDropKeyword("token")) {
        // A "typed" value of type token is just "token", the same as an
        // "untyped" value so just return the Value here.
        return Value::Token();
      }

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
    }
    return Value(bits_ops::ZeroExtend(bits_value, bit_count));
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
      std::optional<Type*> element_type = std::nullopt;
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
      std::optional<Type*> element_type = std::nullopt;
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
    XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("token"));
    return Value::Token();
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported type %s", TypeKindToString(type_kind)));
}

absl::StatusOr<std::vector<Value>> Parser::ParseCommaSeparatedValues(
    Type* type) {
  std::vector<Value> values;
  do {
    XLS_ASSIGN_OR_RETURN(Value value, ParseValueInternal(type));
    values.push_back(value);
  } while (scanner_.TryDropToken(LexicalTokenType::kComma));
  return values;
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
    XLS_ASSIGN_OR_RETURN(BValue value,
                         ParseAndResolveIdentifier(name_to_value));
    result.push_back(value);
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }
  return result;
}

absl::StatusOr<SourceInfo> Parser::ParseSourceInfo() {
  XLS_RETURN_IF_ERROR(
      scanner_.DropTokenOrError(LexicalTokenType::kBracketOpen));
  SourceInfo result;
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

    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));
    XLS_ASSIGN_OR_RETURN(Token fileno,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));
    XLS_ASSIGN_OR_RETURN(Token lineno,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));
    XLS_ASSIGN_OR_RETURN(Token colno,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kParenClose));

    XLS_ASSIGN_OR_RETURN(int64_t fileno_value, fileno.GetValueInt64());
    XLS_ASSIGN_OR_RETURN(int64_t lineno_value, lineno.GetValueInt64());
    XLS_ASSIGN_OR_RETURN(int64_t colno_value, colno.GetValueInt64());
    result.locations.push_back(SourceLocation(
        Fileno(fileno_value), Lineno(lineno_value), Colno(colno_value)));
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }
  SourceInfo copied = result;
  return result;
}

absl::StatusOr<BValue> Parser::BuildBinaryOrUnaryOp(Op op, BuilderBase* fb,
                                                    SourceInfo* loc,
                                                    std::string_view node_name,
                                                    ArgParser* arg_parser) {
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

// Splits node names of the form (.*)\.([0-9]+) into the string and integer
// components and returns them. For example, "add.42" will be returned
// as {"add", 42}. If the name does not match the pattern then nullopt is
// returned.
struct SplitName {
  std::string op_name;
  int64_t node_id;
};
std::optional<SplitName> SplitNodeName(std::string_view name) {
  std::vector<std::string_view> pieces = absl::StrSplit(name, '.');
  if (pieces.empty()) {
    return std::nullopt;
  }
  int64_t result;
  if (!absl::SimpleAtoi(pieces.back(), &result)) {
    return std::nullopt;
  }
  pieces.pop_back();
  return SplitName{.op_name = absl::StrJoin(pieces, "."), .node_id = result};
}

absl::StatusOr<BlockBuilder*> CastToBlockBuilderOrError(
    BuilderBase* base_builder, std::string_view error_message, TokenPos pos) {
  if (BlockBuilder* bb = dynamic_cast<BlockBuilder*>(base_builder)) {
    return bb;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("%s @ %s", error_message, pos.ToHumanString()));
}

absl::StatusOr<ProcBuilder*> CastToProcBuilderOrError(
    BuilderBase* base_builder, std::string_view error_message, TokenPos pos) {
  if (ProcBuilder* pb = dynamic_cast<ProcBuilder*>(base_builder)) {
    return pb;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("%s @ %s", error_message, pos.ToHumanString()));
}

}  // namespace

absl::StatusOr<BValue> Parser::ParseNode(
    BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value) {
  // <output_name>: <type> = op(...)
  XLS_ASSIGN_OR_RETURN(
      Token output_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "node output name"));

  Package* package = fb->function()->package();
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
  XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kEquals));
  XLS_ASSIGN_OR_RETURN(
      Token op_token,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "operator"));

  XLS_ASSIGN_OR_RETURN(Op op, StringToOp(op_token.value()));

  ArgParser arg_parser(*name_to_value, type, this);
  SourceInfo* loc =
      arg_parser.AddOptionalKeywordArg<SourceInfo>("pos", SourceInfo());
  std::optional<int64_t>* id_attribute =
      arg_parser.AddOptionalKeywordArg<int64_t>("id");
  BValue bvalue;

  std::optional<SplitName> split_name = SplitNodeName(output_name.value());
  // If output_name has the form <op>.<id> (e.g, "add.42"), then no name
  // should be given when constructing the node as the name is autogenerated
  // (the node has no meaningful given name). Otherwise, output_name is the
  // name of the node.
  std::string node_name = split_name.has_value() ? "" : output_name.value();

  std::vector<BValue> operands;
  switch (op) {
    case Op::kBitSlice: {
      int64_t* start = arg_parser.AddKeywordArg<int64_t>("start");
      int64_t* width = arg_parser.AddKeywordArg<int64_t>("width");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      bvalue = fb->BitSlice(operands[0], *start, *width, *loc, node_name);
      break;
    }
    case Op::kDynamicBitSlice: {
      int64_t* width = arg_parser.AddKeywordArg<int64_t>("width");
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
      IdentifierString* to_apply_name =
          arg_parser.AddKeywordArg<IdentifierString>("to_apply");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      XLS_ASSIGN_OR_RETURN(Function * to_apply,
                           package->GetFunction(to_apply_name->value));
      bvalue = fb->Map(operands[0], to_apply, *loc, node_name);
      break;
    }
    case Op::kParam: {
      // TODO(meheff): Params should not appear in the body of the
      // function. This is currently required because we have no way of
      // returning a param value otherwise.
      IdentifierString* param_name =
          arg_parser.AddKeywordArg<IdentifierString>("name");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
      auto it = name_to_value->find(param_name->value);
      if (it == name_to_value->end()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Referred to parameter name that hadn't yet been "
                            "defined: %s @ %s",
                            param_name->value, op_token.pos().ToHumanString()));
      }
      bvalue = it->second;
      break;
    }
    case Op::kNext: {
      XLS_ASSIGN_OR_RETURN(
          ProcBuilder * pb,
          CastToProcBuilderOrError(
              fb, "next operations only supported in procs", op_token.pos()));
      BValue* param = arg_parser.AddKeywordArg<BValue>("param");
      BValue* value = arg_parser.AddKeywordArg<BValue>("value");
      std::optional<BValue>* predicate =
          arg_parser.AddOptionalKeywordArg<BValue>("predicate");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
      bvalue = pb->Next(*param, *value, *predicate, *loc, node_name);
      break;
    }
    case Op::kCountedFor: {
      int64_t* trip_count = arg_parser.AddKeywordArg<int64_t>("trip_count");
      int64_t* stride = arg_parser.AddOptionalKeywordArg<int64_t>("stride", 1);
      IdentifierString* body_name =
          arg_parser.AddKeywordArg<IdentifierString>("body");
      std::vector<BValue>* invariant_args =
          arg_parser.AddOptionalKeywordArg<std::vector<BValue>>(
              "invariant_args", /*default_value=*/{});
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      XLS_ASSIGN_OR_RETURN(Function * body,
                           package->GetFunction(body_name->value));
      bvalue = fb->CountedFor(operands[0], *trip_count, *stride, body,
                              *invariant_args, *loc, node_name);
      break;
    }
    case Op::kDynamicCountedFor: {
      IdentifierString* body_name =
          arg_parser.AddKeywordArg<IdentifierString>("body");
      std::vector<BValue>* invariant_args =
          arg_parser.AddOptionalKeywordArg<std::vector<BValue>>(
              "invariant_args", /*default_value=*/{});
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/3));
      XLS_ASSIGN_OR_RETURN(Function * body,
                           package->GetFunction(body_name->value));
      bvalue = fb->DynamicCountedFor(operands[0], operands[1], operands[2],
                                     body, *invariant_args, *loc, node_name);
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
    case Op::kPrioritySel: {
      std::vector<BValue>* case_args =
          arg_parser.AddKeywordArg<std::vector<BValue>>("cases");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      if (case_args->empty()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected at least 1 case @ %s", op_token.pos().ToHumanString()));
      }
      bvalue = fb->PrioritySelect(operands[0], *case_args, *loc, node_name);
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
      bvalue =
          fb->Select(operands[0], *case_args, *default_value, *loc, node_name);
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
    case Op::kMinDelay: {
      if (!type->IsToken()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected token type @ %s", op_token.pos().ToHumanString()));
      }
      int64_t* delay = arg_parser.AddKeywordArg<int64_t>("delay");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      bvalue = fb->MinDelay(operands[0], *delay, *loc, node_name);
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
      int64_t* index = arg_parser.AddKeywordArg<int64_t>("index");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      if (operands[0].valid() && !operands[0].GetType()->IsTuple()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "tuple_index operand is not a tuple; got %s @ %s",
            operands[0].GetType()->ToString(), op_token.pos().ToHumanString()));
      }
      bvalue = fb->TupleIndex(operands[0], *index, *loc, node_name);
      break;
    }
    case Op::kArrayIndex: {
      std::vector<BValue>* index_args =
          arg_parser.AddKeywordArg<std::vector<BValue>>("indices");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      bvalue = fb->ArrayIndex(operands[0], *index_args, *loc, node_name);
      break;
    }
    case Op::kArrayUpdate: {
      std::vector<BValue>* index_args =
          arg_parser.AddKeywordArg<std::vector<BValue>>("indices");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      bvalue = fb->ArrayUpdate(operands[0], operands[1], *index_args, *loc,
                               node_name);
      break;
    }
    case Op::kArrayConcat: {
      // fb->ArrayConcat will check that all operands are of an array
      // type and that all concat'ed arrays have the same element type.
      //
      // for now, just check that type is an Array
      if (!type->IsArray()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Expected array type @ %s, got %s",
                            op_token.pos().ToHumanString(), type->ToString()));
      }

      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
      bvalue = fb->ArrayConcat(operands, *loc, node_name);
      break;
    }
    case Op::kArraySlice: {
      int64_t* width = arg_parser.AddKeywordArg<int64_t>("width");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      bvalue =
          fb->ArraySlice(operands[0], operands[1], *width, *loc, node_name);
      break;
    }
    case Op::kInvoke: {
      IdentifierString* to_apply_name =
          arg_parser.AddKeywordArg<IdentifierString>("to_apply");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(ArgParser::kVariadic));
      XLS_ASSIGN_OR_RETURN(Function * to_apply,
                           package->GetFunction(to_apply_name->value));
      bvalue = fb->Invoke(operands, to_apply, *loc, node_name);
      break;
    }
    case Op::kZeroExt:
    case Op::kSignExt: {
      int64_t* new_bit_count =
          arg_parser.AddKeywordArg<int64_t>("new_bit_count");
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
      int64_t* width = arg_parser.AddKeywordArg<int64_t>("width");
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
    case Op::kSMulp:
    case Op::kUMulp: {
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      if (!type->IsTuple()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s op has annotated type %s, but it should be a tuple.",
            OpToString(op), type->ToString()));
      }
      auto element_types = type->AsTupleOrDie()->element_types();
      if (element_types.size() != 2) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s op has annotated type with %d elements, should be 2.",
            OpToString(op), element_types.size()));
      }
      if (!element_types.at(0)->IsBits()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("%s op has tuple type with element %s, but it "
                            "should be a bits type.",
                            OpToString(op), element_types.at(0)->ToString()));
      }
      if (!element_types.at(1)->IsBits()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("%s op has tuple type with element %s, but it "
                            "should be a bits type.",
                            OpToString(op), element_types.at(1)->ToString()));
      }
      if (!element_types.at(0)->AsBitsOrDie()->IsEqualTo(
              element_types.at(1)->AsBitsOrDie())) {
        return absl::InvalidArgumentError(
            absl::StrFormat("%s op has a tuple type with elements (%s, %s) "
                            "that do not have the same type.",
                            OpToString(op), element_types.at(0)->ToString(),
                            element_types.at(1)->ToString()));
      }

      bvalue = fb->AddPartialProductOp(op, operands[0], operands[1],
                                       element_types.at(0)->GetFlatBitCount(),
                                       *loc, node_name);
      break;
    }
    case Op::kReceive: {
      XLS_ASSIGN_OR_RETURN(ProcBuilder * pb,
                           CastToProcBuilderOrError(
                               fb, "receive operations only supported in procs",
                               op_token.pos()));
      std::optional<BValue>* predicate =
          arg_parser.AddOptionalKeywordArg<BValue>("predicate");
      IdentifierString* channel_name =
          arg_parser.AddKeywordArg<IdentifierString>("channel");
      bool* is_blocking = arg_parser.AddOptionalKeywordArg<bool>(
          "blocking", /*default_value=*/true);
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      // Get the channel from the package.
      if (!pb->HasReceiveChannelRef(channel_name->value)) {
        if (!pb->HasSendChannelRef(channel_name->value)) {
          return absl::InvalidArgumentError(
              absl::StrFormat("No such channel `%s`", channel_name->value));
        }
        return absl::InvalidArgumentError(absl::StrFormat(
            "Cannot receive on channel `%s`", channel_name->value));
      }
      ReceiveChannelRef channel_ref;
      Type* channel_type;
      if (pb->proc()->is_new_style_proc()) {
        XLS_ASSIGN_OR_RETURN(
            channel_ref, pb->GetReceiveChannelReference(channel_name->value));
        channel_type = std::get<ReceiveChannelReference*>(channel_ref)->type();
      } else {
        XLS_ASSIGN_OR_RETURN(channel_ref,
                             package->GetChannel(channel_name->value));
        channel_type = std::get<Channel*>(channel_ref)->type();
      }

      Type* expected_type =
          (*is_blocking)
              ? package->GetTupleType({package->GetTokenType(), channel_type})
              : package->GetTupleType({package->GetTokenType(), channel_type,
                                       package->GetBitsType(1)});

      if (expected_type != type) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Receive op type is type: %s. Expected: %s",
                            type->ToString(), expected_type->ToString()));
      }
      if (predicate->has_value()) {
        if (*is_blocking) {
          bvalue = pb->ReceiveIf(channel_ref, operands[0], predicate->value(),
                                 *loc, node_name);
        } else {
          bvalue = pb->ReceiveIfNonBlocking(
              channel_ref, operands[0], predicate->value(), *loc, node_name);
        }
      } else {
        if (*is_blocking) {
          bvalue = pb->Receive(channel_ref, operands[0], *loc, node_name);
        } else {
          bvalue =
              pb->ReceiveNonBlocking(channel_ref, operands[0], *loc, node_name);
        }
      }
      break;
    }
    case Op::kSend: {
      XLS_ASSIGN_OR_RETURN(
          ProcBuilder * pb,
          CastToProcBuilderOrError(
              fb, "send operations only supported in procs", op_token.pos()));
      std::optional<BValue>* predicate =
          arg_parser.AddOptionalKeywordArg<BValue>("predicate");
      IdentifierString* channel_name =
          arg_parser.AddKeywordArg<IdentifierString>("channel");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      // Get the channel from the package.
      if (!pb->HasSendChannelRef(channel_name->value)) {
        if (!pb->HasReceiveChannelRef(channel_name->value)) {
          return absl::InvalidArgumentError(
              absl::StrFormat("No such channel `%s`", channel_name->value));
        }
        return absl::InvalidArgumentError(absl::StrFormat(
            "Cannot send on channel `%s`", channel_name->value));
      }
      SendChannelRef channel_ref;
      if (pb->proc()->is_new_style_proc()) {
        XLS_ASSIGN_OR_RETURN(channel_ref,
                             pb->GetSendChannelReference(channel_name->value));
      } else {
        XLS_ASSIGN_OR_RETURN(channel_ref,
                             package->GetChannel(channel_name->value));
      }
      if (predicate->has_value()) {
        bvalue = pb->SendIf(channel_ref, operands[0], predicate->value(),
                            operands[1], *loc, node_name);
      } else {
        bvalue =
            pb->Send(channel_ref, operands[0], operands[1], *loc, node_name);
      }
      break;
    }
    case Op::kAssert: {
      QuotedString* message = arg_parser.AddKeywordArg<QuotedString>("message");
      std::optional<QuotedString>* label =
          arg_parser.AddOptionalKeywordArg<QuotedString>("label");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      std::optional<std::string> label_string;
      if (label->has_value()) {
        label_string = label->value().value;
      }
      bvalue = fb->Assert(operands[0], operands[1], message->value,
                          label_string, *loc, node_name);
      break;
    }
    case Op::kTrace: {
      QuotedString* format_string =
          arg_parser.AddKeywordArg<QuotedString>("format");
      std::vector<BValue>* data_operands =
          arg_parser.AddKeywordArg<std::vector<BValue>>("data_operands");
      std::optional<int64_t>* verbosity =
          arg_parser.AddOptionalKeywordArg<int64_t>("verbosity");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      XLS_ASSIGN_OR_RETURN(std::vector<FormatStep> format,
                           ParseFormatString(format_string->value));
      if (verbosity->has_value() && verbosity->value() < 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Verbosity must be >= 0: got %d", verbosity->value()));
      }
      bvalue = fb->Trace(operands[0], operands[1], *data_operands, format,
                         verbosity->value_or(0), *loc, node_name);
      break;
    }
    case Op::kCover: {
      QuotedString* label = arg_parser.AddKeywordArg<QuotedString>("label");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      bvalue =
          fb->Cover(operands[0], operands[1], label->value, *loc, node_name);
      break;
    }
    case Op::kBitSliceUpdate: {
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/3));
      bvalue = fb->BitSliceUpdate(operands[0], operands[1], operands[2], *loc,
                                  node_name);
      break;
    }
    case Op::kInputPort: {
      XLS_ASSIGN_OR_RETURN(
          BlockBuilder * bb,
          CastToBlockBuilderOrError(
              fb, "input_port operations only supported in blocks",
              op_token.pos()));
      IdentifierString* name =
          arg_parser.AddKeywordArg<IdentifierString>("name");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
      bvalue = bb->InputPort(name->value, type, *loc);
      break;
    }
    case Op::kOutputPort: {
      XLS_ASSIGN_OR_RETURN(
          BlockBuilder * bb,
          CastToBlockBuilderOrError(
              fb, "output_port operations only supported in blocks",
              op_token.pos()));
      IdentifierString* name =
          arg_parser.AddKeywordArg<IdentifierString>("name");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      bvalue = bb->OutputPort(name->value, operands[0], *loc);
      break;
    }
    case Op::kRegisterRead: {
      XLS_ASSIGN_OR_RETURN(
          BlockBuilder * bb,
          CastToBlockBuilderOrError(
              fb, "register_read operations only supported in blocks",
              op_token.pos()));
      IdentifierString* register_name =
          arg_parser.AddKeywordArg<IdentifierString>("register");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
      absl::StatusOr<Register*> register_status =
          bb->block()->GetRegister(register_name->value);
      if (!register_status.ok()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("No such register named %s", register_name->value));
      }
      bvalue = bb->RegisterRead(register_status.value(), *loc, node_name);
      break;
    }
    case Op::kRegisterWrite: {
      XLS_ASSIGN_OR_RETURN(
          BlockBuilder * bb,
          CastToBlockBuilderOrError(
              fb, "register_write operations only supported in blocks",
              op_token.pos()));
      IdentifierString* register_name =
          arg_parser.AddKeywordArg<IdentifierString>("register");
      std::optional<BValue>* load_enable =
          arg_parser.AddOptionalKeywordArg<BValue>("load_enable");
      std::optional<BValue>* reset =
          arg_parser.AddOptionalKeywordArg<BValue>("reset");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      absl::StatusOr<Register*> register_status =
          bb->block()->GetRegister(register_name->value);
      if (!register_status.ok()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("No such register named %s", register_name->value));
      }
      bvalue = bb->RegisterWrite(register_status.value(), operands[0],
                                 *load_enable, *reset, *loc, node_name);
      break;
    }
    case Op::kGate: {
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/2));
      bvalue = fb->Gate(operands[0], operands[1], *loc, node_name);
      break;
    }
    case Op::kInstantiationInput: {
      XLS_ASSIGN_OR_RETURN(
          BlockBuilder * bb,
          CastToBlockBuilderOrError(
              fb, "instantiation_input operations only supported in blocks",
              op_token.pos()));
      IdentifierString* instantiation_name =
          arg_parser.AddKeywordArg<IdentifierString>("instantiation");
      IdentifierString* port_name =
          arg_parser.AddKeywordArg<IdentifierString>("port_name");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/1));
      absl::StatusOr<Instantiation*> instantiation_status =
          bb->block()->GetInstantiation(instantiation_name->value);
      if (!instantiation_status.ok()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "No instantiation named `%s`", instantiation_name->value));
      }
      bvalue =
          bb->InstantiationInput(instantiation_status.value(), port_name->value,
                                 operands[0], *loc, node_name);
      break;
    }
    case Op::kInstantiationOutput: {
      XLS_ASSIGN_OR_RETURN(
          BlockBuilder * bb,
          CastToBlockBuilderOrError(
              fb, "instantiation_output operations only supported in blocks",
              op_token.pos()));
      IdentifierString* instantiation_name =
          arg_parser.AddKeywordArg<IdentifierString>("instantiation");
      IdentifierString* port_name =
          arg_parser.AddKeywordArg<IdentifierString>("port_name");
      XLS_ASSIGN_OR_RETURN(operands, arg_parser.Run(/*arity=*/0));
      absl::StatusOr<Instantiation*> instantiation_status =
          bb->block()->GetInstantiation(instantiation_name->value);
      if (!instantiation_status.ok()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "No instantiation named `%s`", instantiation_name->value));
      }
      bvalue = bb->InstantiationOutput(instantiation_status.value(),
                                       port_name->value, *loc, node_name);
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

  if (bvalue.valid()) {
    Node* node = bvalue.node();

    // Verify that the type of the newly constructed node matches the parsed
    // type.
    if (type != node->GetType()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Declared type %s does not match expected type %s @ %s",
          type->ToString(), node->GetType()->ToString(),
          op_token.pos().ToHumanString()));
    }

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
      node->SetId(split_name->node_id);
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
      XLS_RET_CHECK(node->HasAssignedName()) << node->ToString();
      // Also set the ID to the attribute ID (if given).
      if (id_attribute->has_value()) {
        node->SetId(id_attribute->value());
      }
    }
  }
  return bvalue;
}

absl::StatusOr<Register*> Parser::ParseRegister(Block* block) {
  // A register declaration has the following form, for example (without reset):
  //
  //   reg foo(bits[32])
  //
  // With reset:
  //
  //   reg foo(bits[32], reset_value=42, asynchronous=false, active_low=false)
  XLS_ASSIGN_OR_RETURN(
      Token reg_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "register name"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));
  XLS_ASSIGN_OR_RETURN(Type * reg_type, ParseType(block->package()));
  // Parse optional reset attributes.
  std::optional<Value> reset_value;
  std::optional<bool> asynchronous;
  std::optional<bool> active_low;
  if (scanner_.TryDropToken(LexicalTokenType::kComma)) {
    absl::flat_hash_map<std::string, std::function<absl::Status()>> handlers;
    handlers["reset_value"] = [&]() -> absl::Status {
      XLS_ASSIGN_OR_RETURN(reset_value, ParseValueInternal(reg_type));
      return absl::OkStatus();
    };
    handlers["asynchronous"] = [&]() -> absl::Status {
      XLS_ASSIGN_OR_RETURN(asynchronous, ParseBool());
      return absl::OkStatus();
    };
    handlers["active_low"] = [&]() -> absl::Status {
      XLS_ASSIGN_OR_RETURN(active_low, ParseBool());
      return absl::OkStatus();
    };
    XLS_RETURN_IF_ERROR(ParseKeywordArguments(handlers));
  }
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenClose));

  // Either all reset attributes must be specified or none must be specified.
  std::optional<Reset> reset;
  if (reset_value.has_value() && asynchronous.has_value() &&
      active_low.has_value()) {
    reset = Reset{.reset_value = reset_value.value(),
                  .asynchronous = asynchronous.value(),
                  .active_low = active_low.value()};
  } else if (reset_value.has_value() || asynchronous.has_value() ||
             active_low.has_value()) {
    return absl::InvalidArgumentError(
        "Register reset incompletely specified, must include all reset "
        "attributes (reset_value, asynchronous, active_low)");
  }

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block->AddRegister(reg_name.value(), reg_type, reset));
  if (reg->name() != reg_name.value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Register already exists with name %s", reg_name.value()));
  }
  return reg;
}

absl::StatusOr<ProcInstantiation*> Parser::ParseProcInstantiation(Proc* proc) {
  // A proc instantiation declaration has the following forms:
  //
  //   proc_instantiation foo(ch0, ch1, proc=my_proc);
  //
  // When this function is called, the token `proc_instantiation` should already
  // have been popped.
  XLS_ASSIGN_OR_RETURN(
      Token instantiation_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "instantiation name"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));

  std::vector<std::string> channel_arg_names;
  // Parse channels arguments first.
  do {
    if (scanner_.PeekNthTokenIs(1, LexicalTokenType::kEquals)) {
      break;
    }
    XLS_ASSIGN_OR_RETURN(Token channel_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent,
                                                  "channel reference name"));
    channel_arg_names.push_back(channel_name.value());
  } while (scanner_.TryDropToken(LexicalTokenType::kComma));

  // Then parse keyword arguments.
  absl::flat_hash_map<std::string, std::function<absl::Status()>> handlers;

  std::optional<Proc*> instantiated_proc;
  handlers["proc"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token instantiated_proc_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    instantiated_proc =
        proc->package()->TryGetProc(instantiated_proc_name.value());
    if (!instantiated_proc.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No such proc '%s' @ %s", instantiated_proc_name.value(),
          instantiated_proc_name.pos().ToHumanString()));
    }
    if (!(*instantiated_proc)->is_new_style_proc()) {
      instantiated_proc = std::nullopt;
      return absl::InvalidArgumentError(absl::StrFormat(
          "Proc `%s` is not a new style proc and cannot be instantiated @ %s",
          instantiated_proc_name.value(),
          instantiated_proc_name.pos().ToHumanString()));
    }
    return absl::OkStatus();
  };
  XLS_RETURN_IF_ERROR(ParseKeywordArguments(handlers,
                                            /*mandatory_keywords=*/{"proc"}));

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenClose));

  // Convert channel names to ChannelReferences and create the instantiation.
  if (channel_arg_names.size() !=
      instantiated_proc.value()->interface().size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Proc `%s` expects %d channel arguments, got %d @ %s",
        instantiated_proc.value()->name(),
        instantiated_proc.value()->interface().size(), channel_arg_names.size(),
        instantiation_name.pos().ToHumanString()));
  }

  std::vector<ChannelReference*> channel_args;
  for (int64_t i = 0; i < channel_arg_names.size(); ++i) {
    ChannelReference* interface_channel =
        instantiated_proc.value()->interface()[i];
    if (!proc->HasChannelReference(channel_arg_names[i],
                                   interface_channel->direction())) {
      return absl::InvalidArgumentError(
          absl::StrFormat("No such %s channel `%s` for proc instantiation arg "
                          "%d in proc instantiation `%s` @ %s",
                          DirectionToString(interface_channel->direction()),
                          channel_arg_names[i], i, instantiation_name.value(),
                          instantiation_name.pos().ToHumanString()));
    }
    XLS_ASSIGN_OR_RETURN(
        ChannelReference * channel_arg,
        proc->GetChannelReference(channel_arg_names[i],
                                  interface_channel->direction()));
    channel_args.push_back(channel_arg);
  }

  return proc->AddProcInstantiation(instantiation_name.value(), channel_args,
                                    instantiated_proc.value());
};

absl::StatusOr<Instantiation*> Parser::ParseInstantiation(Block* block) {
  // A instantiation declaration has the following forms:
  //
  //   instantiation foo(kind=block, block=bar)
  XLS_ASSIGN_OR_RETURN(
      Token instantiation_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "instantiation name"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));

  absl::flat_hash_map<std::string, std::function<absl::Status()>> handlers;

  std::optional<InstantiationKind> kind;
  handlers["kind"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token kind_token,
                         scanner_.PopKeywordOrIdentToken("instantiation kind"));
    absl::StatusOr<InstantiationKind> kind_status =
        StringToInstantiationKind(kind_token.value());
    if (!kind_status.ok()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid instantiation kind `%s` @ %s", kind_token.value(),
          kind_token.pos().ToHumanString()));
    }
    kind = kind_status.value();
    return absl::OkStatus();
  };

  std::optional<Block*> instantiated_block;
  handlers["block"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token instantiated_block_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    instantiated_block =
        block->package()->TryGetBlock(instantiated_block_name.value());
    if (!instantiated_block.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "No such block '%s' @ %s", instantiated_block_name.value(),
          instantiated_block_name.pos().ToHumanString()));
    }
    return absl::OkStatus();
  };

  std::optional<Type*> data_type;
  handlers["data_type"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(data_type, ParseType(block->package()));
    return absl::OkStatus();
  };

  std::optional<int64_t> depth;
  handlers["depth"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token depth_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_ASSIGN_OR_RETURN(depth, depth_token.GetValueInt64());
    return absl::OkStatus();
  };

  std::optional<bool> bypass;
  handlers["bypass"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token bypass_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_ASSIGN_OR_RETURN(bypass, bypass_token.GetValueBool());
    return absl::OkStatus();
  };

  std::optional<std::string> channel;
  handlers["channel"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token channel_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    channel = channel_token.value();
    return absl::OkStatus();
  };

  XLS_RETURN_IF_ERROR(ParseKeywordArguments(handlers,
                                            /*mandatory_keywords=*/{"kind"}));

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenClose));

  struct Field {
    bool should_have = false;
    bool does_have = false;
    std::string name;
    static Field MustHave(bool has_value, std::string field_name) {
      return Field{.should_have = true,
                   .does_have = has_value,
                   .name = std::move(field_name)};
    }

    static Field MustNotHave(bool has_value, std::string field_name) {
      return Field{.should_have = false,
                   .does_have = has_value,
                   .name = std::move(field_name)};
    }
  };
  auto check_fields = [&instantiation_name](
                          absl::Span<Field const> fields,
                          std::string_view instantiation) -> absl::Status {
    for (const auto& field : fields) {
      if (!field.should_have && field.does_have) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Instantiated %s must not specify %s @ %s.", instantiation,
            field.name, instantiation_name.pos().ToHumanString()));
      }
      if (field.should_have && !field.does_have) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Instantiated %s must specify %s @ %s.", instantiation, field.name,
            instantiation_name.pos().ToHumanString()));
      }
    }
    return absl::OkStatus();
  };

  if (kind.value() == InstantiationKind::kBlock) {
    XLS_RETURN_IF_ERROR(check_fields(
        {
            Field::MustNotHave(data_type.has_value(), "data_type"),
            Field::MustNotHave(depth.has_value(), "depth"),
            Field::MustNotHave(bypass.has_value(), "bypass"),
            Field::MustNotHave(channel.has_value(), "channel"),
            Field::MustHave(instantiated_block.has_value(),
                            "instantiated_block"),
        },
        "block"));
    return block->AddBlockInstantiation(instantiation_name.value(),
                                        instantiated_block.value());
  }

  if (kind.value() == InstantiationKind::kFifo) {
    XLS_RETURN_IF_ERROR(check_fields(
        {
            Field::MustNotHave(instantiated_block.has_value(), "block"),
            Field::MustHave(data_type.has_value(), "data_type"),
            Field::MustHave(depth.has_value(), "depth"),
            Field::MustHave(bypass.has_value(), "bypass"),
        },
        "fifo"));
    return block->AddFifoInstantiation(
        instantiation_name.value(),
        FifoConfig{.depth = depth.value(), .bypass = bypass.value()},
        *data_type, channel);
  }

  return absl::InvalidArgumentError(
      absl::StrFormat("Unsupported instantiation kind `%s` @ %s",
                      InstantiationKindToString(kind.value()),
                      instantiation_name.pos().ToHumanString()));
}

absl::StatusOr<Parser::BodyResult> Parser::ParseBody(
    BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value,
    Package* package) {
  std::optional<BodyResult> result;
  while (!scanner_.PeekTokenIs(LexicalTokenType::kCurlClose)) {
    XLS_ASSIGN_OR_RETURN(Token peek, scanner_.PeekToken());

    // Handle register, instantiation and channel constructs.
    if (scanner_.TryDropKeyword("reg")) {
      if (!fb->function()->IsBlock()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("reg keyword only supported in blocks @ %s",
                            peek.pos().ToHumanString()));
      }
      XLS_RETURN_IF_ERROR(
          ParseRegister(fb->function()->AsBlockOrDie()).status());
      continue;
    }
    if (scanner_.TryDropKeyword("instantiation")) {
      if (!fb->function()->IsBlock()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "instantiation keyword only supported in blocks @ %s",
            peek.pos().ToHumanString()));
      }
      XLS_RETURN_IF_ERROR(
          ParseInstantiation(fb->function()->AsBlockOrDie()).status());
      continue;
    }
    if (scanner_.PeekTokenIs(LexicalTokenType::kKeyword) &&
        scanner_.PeekToken()->value() == "chan") {
      if (!fb->function()->IsProc()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("chan keyword only supported in procs @ %s",
                            peek.pos().ToHumanString()));
      }
      Proc* proc = fb->function()->AsProcOrDie();
      if (!proc->is_new_style_proc()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Channels can only be declared in new-style procs @ %s",
            peek.pos().ToHumanString()));
      }
      XLS_RETURN_IF_ERROR(
          ParseChannel(package, /*attributes=*/{}, proc).status());
      continue;
    }
    if (scanner_.TryDropKeyword("proc_instantiation")) {
      if (!fb->function()->IsProc()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "proc_instantiation keyword only supported in procs @ %s",
            peek.pos().ToHumanString()));
      }
      Proc* proc = fb->function()->AsProcOrDie();
      if (!proc->is_new_style_proc()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Proc instantiations can only be declared in new-style procs @ %s",
            peek.pos().ToHumanString()));
      }
      XLS_RETURN_IF_ERROR(ParseProcInstantiation(proc).status());
      continue;
    }

    // Handle "ret" or "next" depending on whether this is a Function or a Proc.
    bool saw_ret = scanner_.TryDropKeyword("ret");
    bool saw_next = !saw_ret && scanner_.TryDropKeyword("next");
    if ((saw_ret || saw_next) && result.has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "More than one ret/next found @ %s", peek.pos().ToHumanString()));
    }

    if (saw_next) {
      if (!fb->function()->IsProc()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("next keyword only supported in procs @ %s",
                            peek.pos().ToHumanString()));
      }

      // Parse 'next' statement:
      //  next (tkn, next_state0, next_state1, ...)
      XLS_RETURN_IF_ERROR(
          scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));
      XLS_ASSIGN_OR_RETURN(Token next_token_name,
                           scanner_.PopTokenOrError(LexicalTokenType::kIdent,
                                                    "proc next token name"));
      if (!name_to_value->contains(next_token_name.value())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Proc next token name @ %s  was not previously defined: \"%s\"",
            next_token_name.pos().ToHumanString(), next_token_name.value()));
      }

      // TODO: Remove this once fully transitioned over to `next_value` nodes.
      std::vector<BValue> next_state;
      while (scanner_.TryDropToken(LexicalTokenType::kComma)) {
        XLS_ASSIGN_OR_RETURN(Token next_state_name,
                             scanner_.PopTokenOrError(LexicalTokenType::kIdent,
                                                      "proc next state name"));
        if (!name_to_value->contains(next_state_name.value())) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Proc next state name @ %s  was not previously defined: \"%s\"",
              next_state_name.pos().ToHumanString(), next_state_name.value()));
        }
        next_state.push_back(name_to_value->at(next_state_name.value()));
      }
      if (Proc* proc = fb->function()->AsProcOrDie();
          !next_state.empty() && !proc->next_values().empty()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Proc includes both next_value nodes (e.g., %s) and next-state "
            "values on its 'next' line; both cannot be used at the same time.",
            proc->next_values().front()->GetName()));
      }

      XLS_RETURN_IF_ERROR(
          scanner_.DropTokenOrError(LexicalTokenType::kParenClose));
      result =
          ProcNext{.next_token = name_to_value->at(next_token_name.value()),
                   .next_state = next_state};
      continue;
    }

    XLS_ASSIGN_OR_RETURN(BValue bvalue, ParseNode(fb, name_to_value));

    if (saw_ret) {
      if (!fb->function()->IsFunction()) {
        return absl::InvalidArgumentError(
            absl::StrFormat("ret keyword only supported in functions @ %s",
                            peek.pos().ToHumanString()));
      }
      result = bvalue;
    }
  }

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlClose,
                                                "'}' at end of function body"));
  if (result.has_value()) {
    return result.value();
  }
  if (fb->function()->IsFunction()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected 'ret' in function."));
  }
  if (fb->function()->IsProc()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected 'next' in proc."));
  }
  // Return an empty BValue for blocks as no ret or next is supported.
  XLS_RET_CHECK(fb->function()->IsBlock());
  return BValue();
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
  auto fb = std::make_unique<FunctionBuilder>(name.value(), package,
                                              /*should_verify=*/false);
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in function parameters"));
  XLS_ASSIGN_OR_RETURN(std::vector<TypedArgument> params,
                       Parser::ParseTypedArguments(package));
  for (const TypedArgument& param : params) {
    (*name_to_value)[param.name] = fb->Param(param.name, param.type);
  }
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenClose,
                                                "')' in function parameters"));

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
  //   proc foo(tok: token, state0: bits[32], state1: bits[42], init={42, 33}) {
  //     ...
  //
  // OR for "new-style" procs with proc-scoped channels:
  //
  //   proc foo<in_ch: bits[32] streaming in: out_ch: bits[8] single_value out>
  //           (tok: token, state0: bits[32], state1: bits[42], init={42, 33}) {
  //     ...
  //
  // The signature being parsed by this method starts at the proc name and ends
  // with the open brace.
  XLS_ASSIGN_OR_RETURN(Token name, scanner_.PopTokenOrError(
                                       LexicalTokenType::kIdent, "proc name"));
  bool is_new_style_proc = false;
  std::vector<std::unique_ptr<ChannelReference>> interface_channels;
  if (scanner_.TryDropToken(LexicalTokenType::kLt)) {
    is_new_style_proc = true;
    if (!scanner_.TryDropToken(LexicalTokenType::kGt)) {
      do {
        XLS_ASSIGN_OR_RETURN(
            Token channel_name,
            scanner_.PopTokenOrError(LexicalTokenType::kIdent, "channel name"));
        XLS_RETURN_IF_ERROR(
            scanner_.DropTokenOrError(LexicalTokenType::kColon));
        XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
        XLS_ASSIGN_OR_RETURN(
            Token kind_token,
            scanner_.PopTokenOrError(LexicalTokenType::kIdent, "channel kind"));
        absl::StatusOr<ChannelKind> kind_status =
            StringToChannelKind(kind_token.value());
        if (!kind_status.ok()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Invalid channel kind \"%s\" @ %s", kind_token.value(),
              kind_token.pos().ToHumanString()));
        }
        ChannelKind kind = kind_status.value();

        XLS_ASSIGN_OR_RETURN(Token direction_token,
                             scanner_.PopTokenOrError(LexicalTokenType::kIdent,
                                                      "channel direction"));
        if (direction_token.value() == "in") {
          interface_channels.push_back(
              std::make_unique<ReceiveChannelReference>(channel_name.value(),
                                                        type, kind));
        } else if (direction_token.value() == "out") {
          interface_channels.push_back(std::make_unique<SendChannelReference>(
              channel_name.value(), type, kind));
        } else {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Invalid direction string `%s`, expected `in` or `out`",
              direction_token.value()));
        }
      } while (scanner_.TryDropToken(LexicalTokenType::kComma));
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kGt));
    }
  }

  XLS_ASSIGN_OR_RETURN(Token open_paren,
                       scanner_.PopTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in proc parameters"));

  // Parse the token parameter.
  XLS_ASSIGN_OR_RETURN(std::vector<TypedArgument> params,
                       Parser::ParseTypedArguments(package));
  if (params.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected proc to have at least one parameter @ %s",
                        open_paren.pos().ToHumanString()));
  }
  TypedArgument token_param = params.front();
  if (!token_param.type->IsToken()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected first argument of proc to be token type, is: %s @ %s",
        token_param.type->ToString(), token_param.token.pos().ToHumanString()));
  }

  absl::Span<const TypedArgument> state_params =
      absl::MakeSpan(params).subspan(1);

  absl::flat_hash_map<std::string, std::function<absl::Status()>> handlers;
  std::vector<Value> init_values;
  handlers["init"] = [&]() -> absl::Status {
    // Parse "{VALUE, VALUE, ...}".
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlOpen,
                                                  "start of init_values"));
    while (!scanner_.TryDropToken(LexicalTokenType::kCurlClose)) {
      if (!init_values.empty()) {
        XLS_RETURN_IF_ERROR(
            scanner_.DropTokenOrError(LexicalTokenType::kComma));
      }
      if (init_values.size() >= state_params.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Too many initial values given @ %s", name.pos().ToHumanString()));
      }
      XLS_ASSIGN_OR_RETURN(
          Value value,
          ParseValueInternal(state_params.at(init_values.size()).type));
      init_values.push_back(value);
    }
    return absl::OkStatus();
  };

  std::vector<std::string> mandatory_keywords;
  if (params.size() > 1) {
    // If the proc has a state element then an init field is required.
    mandatory_keywords.push_back("init");
  }
  XLS_RETURN_IF_ERROR(ParseKeywordArguments(handlers, mandatory_keywords));

  if (init_values.size() != state_params.size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Too few initial values given, expected %d, got %d @ %s",
        state_params.size(), init_values.size(), name.pos().ToHumanString()));
  }

  XLS_ASSIGN_OR_RETURN(Token paren_close,
                       scanner_.PopTokenOrError(LexicalTokenType::kParenClose,
                                                "')' in proc parameters"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlOpen,
                                                "start of proc body"));

  // The parser does its own verification so pass should_verify=false. This
  // enables the parser to parse and construct malformed IR for tests.
  std::unique_ptr<ProcBuilder> builder;
  if (is_new_style_proc) {
    builder = std::make_unique<ProcBuilder>(NewStyleProc(), name.value(),
                                            token_param.name, package,
                                            /*should_verify=*/false);
    for (const std::unique_ptr<ChannelReference>& channel_ref :
         interface_channels) {
      if (channel_ref->direction() == Direction::kReceive) {
        XLS_RETURN_IF_ERROR(builder
                                ->AddInputChannel(channel_ref->name(),
                                                  channel_ref->type(),
                                                  channel_ref->kind())
                                .status());
      } else {
        XLS_RETURN_IF_ERROR(builder
                                ->AddOutputChannel(channel_ref->name(),
                                                   channel_ref->type(),
                                                   channel_ref->kind())
                                .status());
      }
    }
  } else {
    builder =
        std::make_unique<ProcBuilder>(name.value(), token_param.name, package,
                                      /*should_verify=*/false);
  }
  (*name_to_value)[token_param.name] = builder->GetTokenParam();
  for (int64_t i = 0; i < state_params.size(); ++i) {
    (*name_to_value)[state_params[i].name] =
        builder->StateElement(state_params[i].name, init_values[i]);
  }

  return std::move(builder);
}

absl::StatusOr<Parser::BlockSignature> Parser::ParseBlockSignature(
    Package* package) {
  // A Block definition looks like:
  //
  //   block foo(clk: clock, a: bits[32], b: bits[32]) {
  //     ...
  //
  // The elements inside the parentheses are the ports and determine the order
  // of the ports in the emitted Verilog. These ports must have a corresponding
  // input_port or output_port node defined in the body of the block. A special
  // type `clock` defines the optional clock for the block.
  //
  // The signature being parsed by this method starts at the block name and ends
  // with the open brace.
  BlockSignature signature;
  XLS_ASSIGN_OR_RETURN(Token name, scanner_.PopTokenOrError(
                                       LexicalTokenType::kIdent, "block name"));
  signature.block_name = name.value();

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in block signature"));
  bool must_end = false;
  while (true) {
    if (must_end || scanner_.PeekTokenIs(LexicalTokenType::kParenClose)) {
      XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(
          LexicalTokenType::kParenClose, "')' in block ports"));
      break;
    }
    XLS_ASSIGN_OR_RETURN(Token port_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kColon));
    Type* type = nullptr;
    if (!scanner_.TryDropKeyword("clock")) {
      XLS_ASSIGN_OR_RETURN(type, ParseType(package));
    }
    signature.ports.push_back(Port{port_name.value(), type});
    must_end = !scanner_.TryDropToken(LexicalTokenType::kComma);
  }

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlOpen,
                                                "start of block body"));

  return std::move(signature);
}

absl::StatusOr<std::string> Parser::ParsePackageName() {
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("package"));
  XLS_ASSIGN_OR_RETURN(
      Token package_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "package name"));
  return package_name.value();
}

absl::Status Parser::ParseFileNumber(Package* package,
                                     const DeclAttributes& attributes) {
  if (!attributes.empty()) {
    return absl::InvalidArgumentError(
        "Attributes are not supported on file number declarations.");
  }

  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse function; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("file_number"));
  XLS_ASSIGN_OR_RETURN(
      Token file_number_token,
      scanner_.PopTokenOrError(LexicalTokenType::kLiteral, "file number"));
  XLS_ASSIGN_OR_RETURN(
      Token file_path_token,
      scanner_.PopTokenOrError(LexicalTokenType::kQuotedString, "file path"));
  XLS_ASSIGN_OR_RETURN(int64_t file_number, file_number_token.GetValueInt64());
  if (file_number > std::numeric_limits<int32_t>::max()) {
    return absl::InternalError("file_number declaration might overflow");
  }
  package->SetFileno(Fileno(static_cast<int32_t>(file_number)),
                     file_path_token.value());
  return absl::OkStatus();
}

absl::StatusOr<Function*> Parser::ParseFunction(
    Package* package, const DeclAttributes& attributes) {
  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse function; at EOF.");
  }

  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("fn"));

  absl::flat_hash_map<std::string, BValue> name_to_value;
  XLS_ASSIGN_OR_RETURN(auto function_data,
                       ParseFunctionSignature(&name_to_value, package));
  FunctionBuilder* fb = function_data.first.get();

  XLS_ASSIGN_OR_RETURN(BodyResult body_result,
                       ParseBody(fb, &name_to_value, package));

  XLS_RET_CHECK(std::holds_alternative<BValue>(body_result));
  BValue return_value = std::get<BValue>(body_result);

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
  XLS_ASSIGN_OR_RETURN(Function * result,
                       fb->BuildWithReturnValue(return_value));

  for (const auto& [attribute, literal] : attributes) {
    if (attribute == "initiation_interval") {
      XLS_ASSIGN_OR_RETURN(int64_t ii, literal.GetValueInt64());
      result->SetInitiationInterval(ii);
    } else if (attribute == "ffi_proto") {
      ForeignFunctionData ffi;
      if (!google::protobuf::TextFormat::ParseFromString(literal.value(), &ffi)) {
        return absl::InvalidArgumentError("Non-parseable FFI metadata proto.");
      }
      // Dummy parse to make sure it is a valid template.
      XLS_RETURN_IF_ERROR(CodeTemplate::Create(ffi.code_template()).status());
      result->SetForeignFunctionData(ffi);
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid attribute for function: %s", attribute));
    }
  }

  return result;
}

absl::StatusOr<Proc*> Parser::ParseProc(Package* package,
                                        const DeclAttributes& attributes) {
  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse proc; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("proc"));

  absl::flat_hash_map<std::string, BValue> name_to_value;
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<ProcBuilder> pb,
                       ParseProcSignature(&name_to_value, package));

  XLS_ASSIGN_OR_RETURN(BodyResult body_result,
                       ParseBody(pb.get(), &name_to_value, package));

  XLS_RET_CHECK(std::holds_alternative<ProcNext>(body_result));
  ProcNext proc_next = std::get<ProcNext>(body_result);

  XLS_ASSIGN_OR_RETURN(Proc * result,
                       pb->Build(proc_next.next_token, proc_next.next_state));

  for (const auto& [attribute, literal] : attributes) {
    if (attribute == "initiation_interval") {
      XLS_ASSIGN_OR_RETURN(int64_t ii, literal.GetValueInt64());
      result->SetInitiationInterval(ii);
    } else {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid attribute for proc: %s", attribute));
    }
  }

  return result;
}

absl::StatusOr<Block*> Parser::ParseBlock(Package* package,
                                          const DeclAttributes& attributes) {
  for (const auto& [attribute, token] : attributes) {
    if (attribute == "initiation_interval") {
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Attribute %s is not supported on blocks.", attribute));
  }

  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse block; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("block"));

  XLS_ASSIGN_OR_RETURN(BlockSignature signature, ParseBlockSignature(package));

  // The parser does its own verification so pass should_verify=false. This
  // enables the parser to parse and construct malformed IR for tests.
  auto bb = std::make_unique<BlockBuilder>(signature.block_name, package,
                                           /*should_verify=*/false);
  absl::flat_hash_map<std::string, BValue> name_to_value;
  XLS_ASSIGN_OR_RETURN(BodyResult body_result,
                       ParseBody(bb.get(), &name_to_value, package));
  XLS_RET_CHECK(std::holds_alternative<BValue>(body_result));

  XLS_ASSIGN_OR_RETURN(Block * block, bb->Build());

  // Verify the ports in the signature match one-to-one to input_ports and
  // output_ports.
  absl::flat_hash_map<std::string, Port> ports_by_name;
  std::vector<std::string> port_names;
  for (const Port& port : signature.ports) {
    if (ports_by_name.contains(port.name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Duplicate port name \"%s\"", port.name));
    }
    ports_by_name[port.name] = port;
    port_names.push_back(port.name);
  }
  absl::flat_hash_map<std::string, Node*> port_nodes;
  for (Node* node : block->nodes()) {
    if (node->Is<InputPort>() || node->Is<OutputPort>()) {
      if (!ports_by_name.contains(node->GetName())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Block signature does not contain port \"%s\"", node->GetName()));
      }
      if (node->Is<InputPort>() &&
          node->GetType() != ports_by_name.at(node->GetName()).type) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Type of input port \"%s\" in block signature %s does not match "
            "type of input_port operation: %s",
            node->GetName(), node->GetType()->ToString(),
            ports_by_name.at(node->GetName()).type->ToString()));
      }
      if (node->Is<OutputPort>() &&
          node->As<OutputPort>()->operand(0)->GetType() !=
              ports_by_name.at(node->GetName()).type) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Type of output port \"%s\" in block signature %s does not match "
            "type of output_port operation: %s",
            node->GetName(),
            node->As<OutputPort>()->operand(0)->GetType()->ToString(),
            ports_by_name.at(node->GetName()).type->ToString()));
      }
      port_nodes[node->GetName()] = node;
    }
  }

  for (const Port& port : signature.ports) {
    if (port.type == nullptr) {
      if (block->GetClockPort().has_value()) {
        return absl::InvalidArgumentError("Block has multiple clocks");
      }
      XLS_RETURN_IF_ERROR(block->AddClockPort(port.name));
      continue;
    }
    if (!port_nodes.contains(port.name)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Block port %s has no corresponding input_port or output_port node",
          port.name));
    }
  }

  XLS_RETURN_IF_ERROR(block->ReorderPorts(port_names));
  return block;
}

absl::StatusOr<Channel*> Parser::ParseChannel(Package* package,
                                              const DeclAttributes& attributes,
                                              Proc* proc) {
  if (!attributes.empty()) {
    return absl::InvalidArgumentError(
        "Attributes are not supported on channel declarations.");
  }

  if (AtEof()) {
    return absl::InvalidArgumentError("Could not parse channel; at EOF.");
  }
  XLS_RETURN_IF_ERROR(scanner_.DropKeywordOrError("chan"));
  XLS_ASSIGN_OR_RETURN(
      Token channel_name,
      scanner_.PopTokenOrError(LexicalTokenType::kIdent, "channel name"));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenOpen,
                                                "'(' in channel definition"));
  std::optional<int64_t> id;
  std::optional<ChannelOps> supported_ops;
  std::optional<ChannelMetadataProto> metadata;
  std::vector<Value> initial_values;
  std::optional<ChannelKind> kind;
  std::optional<FlowControl> flow_control;
  std::optional<ChannelStrictness> strictness;
  std::optional<int64_t> fifo_depth;
  std::optional<bool> bypass;

  // Iterate through the comma-separated elements in the channel definition.
  // Examples:
  //
  //  // No initial values.
  //  chan my_channel(bits[32], id=42, ...)
  //
  //  // Initial values.
  //  chan my_channel(bits[32], initial_values={1, 2, 3}, id=42, ...)
  //
  // First parse type.
  XLS_ASSIGN_OR_RETURN(Type * type, ParseType(package));
  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kComma));

  absl::flat_hash_map<std::string, std::function<absl::Status()>> handlers;
  handlers["initial_values"] = [&]() -> absl::Status {
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kCurlOpen));
    if (!scanner_.PeekTokenIs(LexicalTokenType::kCurlClose)) {
      XLS_ASSIGN_OR_RETURN(initial_values, ParseCommaSeparatedValues(type));
    }
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kCurlClose));
    return absl::OkStatus();
  };
  handlers["id"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token id_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_ASSIGN_OR_RETURN(id, id_token.GetValueInt64());
    return absl::OkStatus();
  };
  handlers["kind"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token kind_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    absl::StatusOr<ChannelKind> kind_status =
        StringToChannelKind(kind_token.value());
    if (!kind_status.ok()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid channel kind \"%s\" @ %s", kind_token.value(),
          kind_token.pos().ToHumanString()));
    }
    kind = kind_status.value();
    return absl::OkStatus();
  };
  handlers["ops"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token supported_ops_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    if (supported_ops_token.value() == "send_only") {
      supported_ops = ChannelOps::kSendOnly;
    } else if (supported_ops_token.value() == "receive_only") {
      supported_ops = ChannelOps::kReceiveOnly;
    } else if (supported_ops_token.value() == "send_receive") {
      supported_ops = ChannelOps::kSendReceive;
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid channel attribute ops \"%s\" @ %s. Expected: send_only,"
          "receive_only, or send_receive",
          supported_ops_token.value(),
          supported_ops_token.pos().ToHumanString()));
    }
    return absl::OkStatus();
  };
  handlers["metadata"] = [&]() -> absl::Status {
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
    return absl::OkStatus();
  };
  handlers["flow_control"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token flow_control_token,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    absl::StatusOr<FlowControl> flow_control_status =
        StringToFlowControl(flow_control_token.value());
    if (!flow_control_status.ok()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Invalid flow control value \"%s\" @ %s", flow_control_token.value(),
          flow_control_token.pos().ToHumanString()));
    }
    flow_control = flow_control_status.value();
    return absl::OkStatus();
  };
  handlers["fifo_depth"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token token,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_ASSIGN_OR_RETURN(fifo_depth, token.GetValueInt64());
    return absl::OkStatus();
  };
  handlers["strictness"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token token,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    absl::StatusOr<ChannelStrictness> strictness_status =
        ChannelStrictnessFromString(token.value());
    if (!strictness_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid strictness value \"%s\" @ %s", token.value(),
                          token.pos().ToHumanString()));
    }
    strictness = strictness_status.value();
    return absl::OkStatus();
  };
  auto token_to_bool = [](const Token& token) -> absl::StatusOr<bool> {
    absl::StatusOr<bool> bool_status = token.GetValueBool();
    if (!bool_status.ok()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid bool value \"%s\" @ %s", token.value(),
                          token.pos().ToHumanString()));
    }
    return bool_status.value();
  };
  handlers["bypass"] = [&]() -> absl::Status {
    XLS_ASSIGN_OR_RETURN(Token token,
                         scanner_.PopTokenOrError(LexicalTokenType::kLiteral));
    XLS_ASSIGN_OR_RETURN(bypass, token_to_bool(token));
    return absl::OkStatus();
  };

  XLS_RETURN_IF_ERROR(ParseKeywordArguments(
      handlers, /*mandatory_keywords=*/{"id", "ops", "metadata", "kind"}));

  XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kParenClose,
                                                "')' in channel definition"));

  for (const auto& [option_has_value, option_name] :
       std::initializer_list<std::tuple<bool, std::string_view>>{
           {flow_control.has_value(), "flow control"},
           {fifo_depth.has_value(), "fifo_depth"},
           {strictness.has_value(), "strictness"},
           {bypass.has_value(), "bypass"},
       }) {
    if (option_has_value && kind != ChannelKind::kStreaming) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Only streaming channels can have %s @ %s",
                          option_name, channel_name.pos().ToHumanString()));
    }
  }
  if (bypass.has_value() && !fifo_depth.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "fifo_depth must be specified if bypass is specified @ %s",
        channel_name.pos().ToHumanString()));
  }

  switch (kind.value()) {
    case ChannelKind::kStreaming: {
      std::optional<FifoConfig> fifo_config;
      if (fifo_depth.has_value()) {
        fifo_config.emplace();
        fifo_config->depth = *fifo_depth;
        fifo_config->bypass = bypass.value_or(true);
      }
      if (proc == nullptr) {
        return package->CreateStreamingChannel(
            channel_name.value(), *supported_ops, type, initial_values,
            fifo_config, flow_control.value_or(FlowControl::kNone),
            strictness.value_or(ChannelStrictness::kProvenMutuallyExclusive),
            *metadata, id);
      }
      return package->CreateStreamingChannelInProc(
          channel_name.value(), *supported_ops, type, proc, initial_values,
          fifo_config, flow_control.value_or(FlowControl::kNone),
          strictness.value_or(ChannelStrictness::kProvenMutuallyExclusive),
          *metadata, id);
    }
    case ChannelKind::kSingleValue: {
      if (proc == nullptr) {
        return package->CreateSingleValueChannel(
            channel_name.value(), *supported_ops, type, *metadata, id);
      }
      return package->CreateSingleValueChannelInProc(
          channel_name.value(), *supported_ops, type, proc, *metadata, id);
    }
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unknown channel kind: ", static_cast<int>(kind.value())));
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

absl::StatusOr<DeclAttributes> Parser::MaybeParseAttributes() {
  absl::flat_hash_map<std::string, Token> attributes;
  while (!AtEof() && scanner_.PeekTokenIs(LexicalTokenType::kHash)) {
    XLS_RETURN_IF_ERROR(scanner_.DropTokenOrError(LexicalTokenType::kHash));
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kBracketOpen));
    XLS_ASSIGN_OR_RETURN(Token attribute_name,
                         scanner_.PopTokenOrError(LexicalTokenType::kIdent));
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kParenOpen));
    XLS_ASSIGN_OR_RETURN(Token literal,
                         scanner_.PopLiteralOrStringToken("for attributes"));
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kParenClose));
    XLS_RETURN_IF_ERROR(
        scanner_.DropTokenOrError(LexicalTokenType::kBracketClose));
    attributes.emplace(attribute_name.value(), literal);
  }
  if (AtEof()) {
    return absl::InvalidArgumentError("Illegal attribute at end of file");
  }
  return attributes;
}

/* static */ absl::StatusOr<FunctionType*> Parser::ParseFunctionType(
    std::string_view input_string, Package* package) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseFunctionType(package);
}

/* static */ absl::StatusOr<Type*> Parser::ParseType(
    std::string_view input_string, Package* package) {
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

/* static */ absl::StatusOr<Function*> Parser::ParseFunction(
    std::string_view input_string, Package* package, bool verify_function_only,
    const DeclAttributes& attributes) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  XLS_ASSIGN_OR_RETURN(Function * function,
                       p.ParseFunction(package, attributes));

  if (verify_function_only) {
    XLS_RETURN_IF_ERROR(VerifyFunction(function));
  } else {
    // Verify the whole package because the addition of the function may break
    // package-scoped invariants (eg, duplicate function name).
    XLS_RETURN_IF_ERROR(VerifyAndSwapError(package));
  }

  return function;
}

/* static */ absl::StatusOr<Proc*> Parser::ParseProc(
    std::string_view input_string, Package* package,
    const DeclAttributes& attributes) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  XLS_ASSIGN_OR_RETURN(Proc * proc, p.ParseProc(package, attributes));

  // Verify the whole package because the addition of the proc may break
  // package-scoped invariants (eg, duplicate proc name).
  XLS_RETURN_IF_ERROR(VerifyAndSwapError(package));
  return proc;
}

/* static */ absl::StatusOr<Block*> Parser::ParseBlock(
    std::string_view input_string, Package* package,
    const DeclAttributes& attributes) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  XLS_ASSIGN_OR_RETURN(Block * proc, p.ParseBlock(package, attributes));

  // Verify the whole package because the addition of the block may break
  // package-scoped invariants (eg, duplicate block name).
  XLS_RETURN_IF_ERROR(VerifyAndSwapError(package));
  return proc;
}

/* static */ absl::StatusOr<Channel*> Parser::ParseChannel(
    std::string_view input_string, Package* package,
    const DeclAttributes& attributes) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseChannel(package, attributes);
}

/* static */ absl::StatusOr<std::unique_ptr<Package>> Parser::ParsePackage(
    std::string_view input_string, std::optional<std::string_view> filename) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ParsePackageNoVerify(input_string, filename));
  XLS_RETURN_IF_ERROR(VerifyAndSwapError(package.get()));
  return package;
}

/* static */ absl::StatusOr<std::unique_ptr<Package>>
Parser::ParsePackageWithEntry(std::string_view input_string,
                              std::string_view entry,
                              std::optional<std::string_view> filename) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<Package> package,
                       ParsePackageNoVerify(input_string, filename, entry));
  XLS_RETURN_IF_ERROR(VerifyPackage(package.get()));
  return package;
}

/* static */ absl::StatusOr<std::unique_ptr<Package>>
Parser::ParsePackageNoVerify(std::string_view input_string,
                             std::optional<std::string_view> filename,
                             std::optional<std::string_view> entry) {
  return ParseDerivedPackageNoVerify<Package>(input_string, filename, entry);
}

/* static */ absl::StatusOr<Value> Parser::ParseValue(
    std::string_view input_string, Type* expected_type) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseValueInternal(expected_type);
}

/* static */ absl::StatusOr<Value> Parser::ParseTypedValue(
    std::string_view input_string) {
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser p(std::move(scanner));
  return p.ParseValueInternal(/*expected_type=*/std::nullopt);
}

}  // namespace xls
