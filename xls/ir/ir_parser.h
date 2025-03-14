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

// The IR parser allows to build an IR from reading in and
// parsing textual IR.
//
// This is convenience functionality, great for debugging and
// construction of small test cases, it can be used by other
// front-ends to target XLS without having to fully link to it.

#ifndef XLS_IR_IR_PARSER_H_
#define XLS_IR_IR_PARSER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_scanner.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"

namespace xls {

class ArgParser;

struct InitiationInterval {
  int64_t value;
};

struct ResetAttribute {
  std::string port_name;
  ResetBehavior behavior;
};

using IrAttributePayload = std::variant<InitiationInterval, ChannelPortMetadata,
                                        ForeignFunctionData, ResetAttribute>;
struct IrAttribute {
  std::string name;
  IrAttributePayload payload;
};

class Parser {
 public:
  // Parses the given input string as a package.
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackage(
      std::string_view input_string,
      std::optional<std::string_view> filename = std::nullopt);

  // As above, but sets the entry function to be the given name in the returned
  // package.
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackageWithEntry(
      std::string_view input_string, std::string_view entry,
      std::optional<std::string_view> filename = std::nullopt);

  // Parse the input_string as a function into the given package.
  // If verify_function_only is true, then only this new function is verified,
  // otherwise the whole package is verified by default.
  // TODO(meheff): 2022/2/9 Remove `verify_function_only` argument.
  static absl::StatusOr<Function*> ParseFunction(
      std::string_view input_string, Package* package,
      bool verify_function_only = false,
      absl::Span<const IrAttribute> outer_attributes = {});

  // Parse the input_string as a proc into the given package.
  static absl::StatusOr<Proc*> ParseProc(
      std::string_view input_string, Package* package,
      absl::Span<const IrAttribute> outer_attributes = {});

  // Parse the input_string as a block into the given package.
  static absl::StatusOr<Block*> ParseBlock(
      std::string_view input_string, Package* package,
      absl::Span<const IrAttribute> outer_attributes = {});

  // Parse the input_string as a channel in the given package.
  static absl::StatusOr<Channel*> ParseChannel(
      std::string_view input_string, Package* package,
      absl::Span<const IrAttribute> outer_attributes = {});

  // Parse the input_string as a function type into the given package.
  static absl::StatusOr<FunctionType*> ParseFunctionType(
      std::string_view input_string, Package* package);

  // Parse the input_string as a type into the given package.
  static absl::StatusOr<Type*> ParseType(std::string_view input_string,
                                         Package* package);

  // Parses the given input string as a package skipping verification. This
  // should only be used in tests when malformed IR is desired.
  static absl::StatusOr<std::unique_ptr<Package>> ParsePackageNoVerify(
      std::string_view input_string,
      std::optional<std::string_view> filename = std::nullopt,
      std::optional<std::string_view> entry = std::nullopt);

  // As above but creates a package of type PackageT where PackageT must be
  // type derived from Package.
  template <typename PackageT>
  static absl::StatusOr<std::unique_ptr<PackageT>> ParseDerivedPackageNoVerify(
      std::string_view input_string,
      std::optional<std::string_view> filename = std::nullopt,
      std::optional<std::string_view> entry = std::nullopt);

  // Parses a literal value that should be of type "expected_type" and returns
  // it.
  static absl::StatusOr<Value> ParseValue(std::string_view input_string,
                                          Type* expected_type);

  // Parses a value with embedded type information, specifically 'bits[xx]:'
  // substrings indicating the width of literal values. Value::ToString emits
  // strings of this form. Examples of strings parsable with this method:
  //   bits[32]:0x42
  //   (bits[7]:0, bits[8]:1)
  //   [bits[2]:1, bits[2]:2, bits[2]:3]
  static absl::StatusOr<Value> ParseTypedValue(std::string_view input_string);

 private:
  friend class ArgParser;

  explicit Parser(Scanner scanner) : scanner_(scanner) {}

  // Parse a function starting at the current scanner position.
  absl::StatusOr<Function*> ParseFunction(
      Package* package, absl::Span<const IrAttribute> outer_attributes = {});

  // Parse a proc starting at the current scanner position.
  absl::StatusOr<Proc*> ParseProc(
      Package* package, absl::Span<const IrAttribute> outer_attributes = {});

  // Parse a block starting at the current scanner position.
  absl::StatusOr<Block*> ParseBlock(
      Package* package, absl::Span<const IrAttribute> outer_attributes = {});

  // Parse a channel starting at the current scanner position. If `proc` is not
  // null then this is a proc-scoped channel.
  absl::StatusOr<Channel*> ParseChannel(
      Package* package, absl::Span<const IrAttribute> outer_attributes = {},
      Proc* proc = nullptr);

  // Parse a channel interface starting at the current scanner position. The
  // scanner must be positioned within the body of the proc `proc` with
  // proc-scoped channels. Because of implementation details of the parser and
  // proc API, this method does not actually add an interface to the proc (this
  // is done when a channel is added or the signature is parsed). Rather, this
  // method sets various attributes on the interface.
  absl::StatusOr<ChannelInterface*> ParseChannelInterface(Package* package,
                                                          Proc* proc);

  // Parse starting from a function type.
  absl::StatusOr<FunctionType*> ParseFunctionType(Package* package);

  // A thin convenience function which parses a single boolean literal.
  absl::StatusOr<bool> ParseBool();

  // A thin convenience function which parses a single int64_t number.
  absl::StatusOr<int64_t> ParseInt64();

  // A thin convenience function which parses a single identifier string.
  absl::StatusOr<std::string> ParseIdentifier(TokenPos* pos = nullptr);

  // A thin convenience function which parses a quoted string.
  absl::StatusOr<std::string> ParseQuotedString(TokenPos* pos = nullptr);

  // Convenience function that parses an identifier and resolve it to a value,
  // or returns a status error if it cannot.
  absl::StatusOr<BValue> ParseAndResolveIdentifier(
      const absl::flat_hash_map<std::string, BValue>& name_to_value);

  // Parses a Value. Supports bits, array, and tuple types as well as their
  // nested variants. If expected_type is not given, the input string should
  // have embedded 'bits' types indicating the width of bits values as produced
  // by Value::ToString. For example: "(bits[32]:0x23, bits[0]:0x1)". If
  // expected_type is given, the string should NOT have embedded bits types as
  // produced by Value::ToHumanString. For example: "(0x23, 0x1)".
  absl::StatusOr<Value> ParseValueInternal(std::optional<Type*> expected_type);

  // Parses a comma-separated sequence of values of the given type. Must have at
  // least one element in the sequence.
  absl::StatusOr<std::vector<Value>> ParseCommaSeparatedValues(Type* type);

  // Parse a comma-separated list of typed arguments as might be see in a
  // function signature. For example:
  //
  //   a: bits[32], b: bits[44], c: (bits[32][2], bits[1])
  //
  // Arguments can include optional unique ids:
  //
  //   a: bits[32] id=3, b: bits[44] id=4, c: (bits[32][2], bits[1]) id=5
  //
  struct TypedArgument {
    std::string name;
    Type* type;
    std::optional<int64_t> id;
    Token token;
  };
  absl::StatusOr<std::vector<TypedArgument>> ParseTypedArguments(
      Package* package);

  // Parses a comma-delimited list of names surrounded by brackets; e.g.
  //
  //    "[foo, bar, baz]"
  //
  // Where the foo, bar, and baz identifiers are resolved via name_to_value.
  //
  // Returns an error if the parse fails or if any of the names cannot be
  // resolved via name_to_value.
  absl::StatusOr<std::vector<BValue>> ParseNameList(
      const absl::flat_hash_map<std::string, BValue>& name_to_value);

  // Parses a comma-delimited list of keyword argument values (e.g.,
  // `foo=bar`). `handlers` is a map of the supported keywords where the key is
  // the keyword and the value is a function which parses the right hand side of
  // the `lhs=rhs` keyword argument. `mandatory_keywords` is a list of keywords
  // which must be present.
  absl::Status ParseKeywordArguments(
      const absl::flat_hash_map<std::string, std::function<absl::Status()>>&
          handlers,
      absl::Span<const std::string> mandatory_keywords = {});

  // Parses a source location.
  // TODO(meheff): Currently the source location is a sequence of three
  // comma-separated numbers. Encapsulating the numbers in braces or something
  // would make the output less ambiguous. Example:
  // "and(x,y,pos={1,2,3},foo=bar)" vs "and(x,y,pos=[(1,2,3)],foo=bar)"
  absl::StatusOr<SourceInfo> ParseSourceInfo();

  // Parse type specifications.
  absl::StatusOr<Type*> ParseType(Package* package);

  // Parse a tuple type (which can contain nested tuples).
  absl::StatusOr<Type*> ParseTupleType(Package* package);

  // Parse a bits type.
  absl::StatusOr<Type*> ParseBitsType(Package* package);

  // Parses a bits types and returns the width.
  absl::StatusOr<int64_t> ParseBitsTypeAndReturnWidth();

  // Builds a binary or unary BValue with the given Op using the given
  // FunctionBuilder and arg parser.
  absl::StatusOr<BValue> BuildBinaryOrUnaryOp(Op op, BuilderBase* fb,
                                              SourceInfo* loc,
                                              std::string_view node_name,
                                              ArgParser* arg_parser);

  // Reassign IDs of nodes which were *not* explicitly assigned in the IR (e.g.,
  // `id=42`).
  // TODO(https://github.com/google/xls/issues/1601): Consider alternate
  // approaches if SetId is removed.
  static constexpr int64_t kUnassignedNodeId = 0;
  static void SetUnassignedNodeIds(
      Package* package, std::optional<FunctionBase*> scope = std::nullopt);

  // Parses a node in a function/proc body. Example: "foo: bits[32] = add(x, y)"
  absl::StatusOr<BValue> ParseNode(
      BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value);

  // Parses a register declaration. Only supported in blocks.
  absl::StatusOr<Register*> ParseRegister(Block* block);

  // Parses an instantiation declaration. Only supported in blocks.
  absl::StatusOr<Instantiation*> ParseInstantiation(Block* block);

  // Parses a proc instantiation declaration. Only supported in procs.
  absl::StatusOr<ProcInstantiation*> ParseProcInstantiation(Proc* proc);

  struct ProcBodyResult {
    std::vector<BValue> next_state;
    std::vector<ChannelInterface*> declared_channel_interfaces;
  };
  struct FunctionBodyResult {
    BValue return_value;
  };
  struct BlockBodyResult {};
  using BodyResult =
      std::variant<FunctionBodyResult, ProcBodyResult, BlockBodyResult>;

  // Parses the line-statements in the body of a function/proc. Returns the
  // return value if the body is a function, or the next token/state pair if the
  // body is a proc.
  absl::StatusOr<BodyResult> ParseBody(
      BuilderBase* fb, absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a function signature, starting after the 'fn' keyword up to and
  // including the opening brace. Returns the newly created builder and the
  // annotated return type (may be nullptr) after the opening brace has been
  // popped.
  absl::StatusOr<std::pair<std::unique_ptr<FunctionBuilder>, Type*>>
  ParseFunctionSignature(
      absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a proc signature, starting after the 'proc' keyword up to and
  // including the opening brace. Returns the newly created builder.
  absl::StatusOr<std::unique_ptr<ProcBuilder>> ParseProcSignature(
      absl::flat_hash_map<std::string, BValue>* name_to_value,
      Package* package);

  // Parses a block signature, starting after the 'block' keyword up to and
  // including the opening brace. Returns the newly created builder along with
  // information about the ports. The order of the returned Ports corresponds to
  // the order within the block.
  struct Port {
    std::string name;
    Type* type;
  };
  struct BlockSignature {
    std::string block_name;
    std::vector<Port> ports;
  };
  absl::StatusOr<BlockSignature> ParseBlockSignature(Package* package);

  // Pops the package name out of the scanner, of the form:
  //
  //  "package" <name>
  //
  // And returns the name.
  absl::StatusOr<std::string> ParsePackageName();

  // Pops a file_number declaration out of the scanner, of the form:
  //
  //  "file_number" <integer> <quoted-string>
  //
  // And adds the mapping to the given `Package`.
  absl::Status ParseFileNumber(Package* package,
                               absl::Span<const IrAttribute> attributes = {});

  // Parse a sequence of outer attributes of the form:
  //
  // #[Attr]
  absl::StatusOr<std::vector<IrAttribute>> MaybeParseOuterAttributes(
      Package* package);

  // Parse a sequence of inner attributes of the form:
  //
  // #![Attr]
  absl::StatusOr<std::vector<IrAttribute>> MaybeParseInnerAttributes(
      Package* package);

  // Parse an attribute. For example, this would be the `Attr` tokens in
  // `#![Attr]`.
  absl::StatusOr<IrAttribute> ParseAttribute(Package* package);

  bool AtEof() const { return scanner_.AtEof(); }

  Scanner scanner_;
};

/* static */ template <typename PackageT>
absl::StatusOr<std::unique_ptr<PackageT>> Parser::ParseDerivedPackageNoVerify(
    std::string_view input_string, std::optional<std::string_view> filename,
    std::optional<std::string_view> entry) {
  std::optional<Token> previous_top_token;
  XLS_ASSIGN_OR_RETURN(auto scanner, Scanner::Create(input_string));
  Parser parser(std::move(scanner));

  XLS_ASSIGN_OR_RETURN(std::string package_name, parser.ParsePackageName());

  auto package = std::make_unique<PackageT>(package_name);
  std::string filename_str =
      (filename.has_value() ? std::string(filename.value()) : "<unknown file>");
  while (!parser.AtEof()) {
    XLS_ASSIGN_OR_RETURN(std::vector<IrAttribute> outer_attributes,
                         parser.MaybeParseOuterAttributes(package.get()));

    XLS_ASSIGN_OR_RETURN(Token peek, parser.scanner_.PeekToken());

    bool is_top = false;
    // The fn, proc or block is a top entity.
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "top") {
      is_top = true;
      XLS_RETURN_IF_ERROR(parser.scanner_.DropKeywordOrError("top"));
      XLS_ASSIGN_OR_RETURN(peek, parser.scanner_.PeekToken());
      if (package.get()->HasTop() && previous_top_token.has_value()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Top declared more than once, previous declaration @ %s",
            previous_top_token.value().pos().ToHumanString()));
      }
      previous_top_token = peek;
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "fn") {
      XLS_ASSIGN_OR_RETURN(
          Function * fn, parser.ParseFunction(package.get(), outer_attributes),
          _ << "@ " << filename_str);
      if (is_top) {
        XLS_RETURN_IF_ERROR(package->SetTop(fn));
      }
      continue;
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "proc") {
      XLS_ASSIGN_OR_RETURN(Proc * proc,
                           parser.ParseProc(package.get(), outer_attributes),
                           _ << "@ " << filename_str);
      if (is_top) {
        XLS_RETURN_IF_ERROR(package->SetTop(proc));
      }
      continue;
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "block") {
      XLS_ASSIGN_OR_RETURN(Block * block,
                           parser.ParseBlock(package.get(), outer_attributes),
                           _ << "@ " << filename_str);
      if (is_top) {
        XLS_RETURN_IF_ERROR(package->SetTop(block));
      }
      continue;
    }
    if (is_top) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Expected fn, proc or block definition, got %s @ %s",
                          peek.value(), peek.pos().ToHumanString()));
    }
    if (peek.type() == LexicalTokenType::kKeyword && peek.value() == "chan") {
      XLS_RETURN_IF_ERROR(
          parser.ParseChannel(package.get(), outer_attributes).status())
          << "@ " << filename_str;
      continue;
    }
    if (peek.type() == LexicalTokenType::kKeyword &&
        peek.value() == "file_number") {
      XLS_RETURN_IF_ERROR(
          parser.ParseFileNumber(package.get(), outer_attributes))
          << "@ " << filename_str;
      continue;
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected attribute or declaration "
                        "(`fn`, `proc`, `block`, `chan`, `file_number`), "
                        "got %s @ %s",
                        peek.value(), peek.pos().ToHumanString()));
  }

  // Verify the given entry function exists in the package.
  if (entry.has_value()) {
    XLS_RETURN_IF_ERROR(package->SetTopByName(entry.value()));
    XLS_RETURN_IF_ERROR(package->GetFunction(*entry).status());
  }
  SetUnassignedNodeIds(package.get());
  return package;
}

}  // namespace xls

#endif  // XLS_IR_IR_PARSER_H_
