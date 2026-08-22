// Copyright 2026 The XLS Authors
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

#include "xls/jit/aot_cpp_header.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/cpp_name_utils.h"
#include "xls/common/status/status_macros.h"
#include "xls/jit/aot_entrypoint.pb.h"

namespace xls {
namespace {

// XLS packed buffers store values in a dense bit layout with all octets used.
// For a ``bits[3]`` value that means the packed buffer holds 8 bits even though
// the value only uses 3. This function computes the packed size in bits,
// rounded up to the next byte, mirroring
// ``LlvmTypeConverter::GetPackedTypeByteSize``.
int64_t PackedBitsFromFlat(int64_t flat_bit_count) {
  if (flat_bit_count == 0) {
    return 8;
  }
  return ((flat_bit_count + 7) / 8) * 8;
}

bool IsImplementationReservedIdentifier(std::string_view identifier) {
  return identifier.starts_with("_") ||
         identifier.find("__") != std::string_view::npos;
}

bool IsLexicallyValidIdentifier(std::string_view identifier) {
  if (identifier.empty()) {
    return false;
  }
  for (size_t i = 0; i < identifier.size(); ++i) {
    char c = identifier[i];
    bool is_alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9');
    bool is_underscore = c == '_';
    if (i == 0) {
      if (!is_alnum && !is_underscore) {
        return false;
      }
      if (c >= '0' && c <= '9') {
        return false;  // Identifiers must not start with a digit.
      }
    } else if (!is_alnum && !is_underscore) {
      return false;
    }
  }
  return true;
}

absl::Status ValidateIdentifier(std::string_view identifier,
                                std::string_view what) {
  if (!IsLexicallyValidIdentifier(identifier)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "'%s' is not a valid C++ identifier used as %s. Only C++ identifiers "
        "consisting of letters, digits and underscores (and not starting with "
        "a digit) are allowed.",
        identifier, what));
  }
  if (IsCppKeyword(identifier)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "'%s' is a C++ keyword and cannot be used as %s.", identifier, what));
  }
  if (IsImplementationReservedIdentifier(identifier)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "'%s' is reserved to the C++ implementation and cannot be used as %s.",
        identifier, what));
  }
  return absl::OkStatus();
}

// Validates a C++ namespace given as a sequence of identifiers joined by "::".
absl::Status ValidateCppNamespace(std::string_view cpp_namespace) {
  if (cpp_namespace.empty()) {
    return absl::OkStatus();
  }
  for (const std::string_view part :
       absl::StrSplit(cpp_namespace, absl::ByString("::"))) {
    XLS_RETURN_IF_ERROR(ValidateIdentifier(part, "namespace"));
  }
  return absl::OkStatus();
}

// Converts an XLS parameter name (typically snake_case) into a valid C++
// type-alias identifier.
absl::StatusOr<std::string> ToCppTypeName(std::string_view name) {
  for (char c : name) {
    bool alnum = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                 (c >= '0' && c <= '9');
    if (!alnum && c != '_') {
      return absl::InvalidArgumentError(absl::StrFormat(
          "XLS parameter name '%s' cannot be converted to a C++ identifier.",
          name));
    }
  }
  std::string result;
  bool capitalize = true;
  for (char c : name) {
    if (c == '_') {
      capitalize = true;
    } else {
      if (capitalize) {
        result.push_back(static_cast<char>(std::toupper(c)));
        capitalize = false;
      } else {
        result.push_back(c);
      }
    }
  }
  if (result.empty()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("XLS parameter name '%s' cannot be converted to a "
                        "meaningful C++ identifier.",
                        name));
  }
  if (std::isdigit(result[0])) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "XLS parameter name '%s' would produce a C++ identifier starting with "
        "a digit.",
        name));
  }
  XLS_RETURN_IF_ERROR(ValidateIdentifier(result, "generated type alias"));
  return result;
}

// Recursively computes the flat bit count of an XLS type proto.
absl::StatusOr<int64_t> FlatBitCount(const TypeProto& type) {
  switch (type.type_enum()) {
    case TypeProto::BITS:
      if (type.bit_count() < 0) {
        return absl::InvalidArgumentError("BITS type has a negative width.");
      }
      return type.bit_count();
    case TypeProto::ARRAY: {
      if (type.array_size() < 0) {
        return absl::InvalidArgumentError("ARRAY type has a negative size.");
      }
      XLS_ASSIGN_OR_RETURN(int64_t element_bits,
                           FlatBitCount(type.array_element()));
      if (type.array_size() != 0 &&
          element_bits >
              std::numeric_limits<int64_t>::max() / type.array_size()) {
        return absl::OutOfRangeError("ARRAY flat bit count overflows int64.");
      }
      return element_bits * type.array_size();
    }
    case TypeProto::TUPLE: {
      int64_t total = 0;
      for (const TypeProto& element : type.tuple_elements()) {
        XLS_ASSIGN_OR_RETURN(int64_t element_bits, FlatBitCount(element));
        if (element_bits > std::numeric_limits<int64_t>::max() - total) {
          return absl::OutOfRangeError("TUPLE flat bit count overflows int64.");
        }
        total += element_bits;
      }
      return total;
    }
    default:
      return absl::UnimplementedError(
          absl::StrFormat("Cannot compute the flat bit count of type kind %s.",
                          TypeProto::TypeEnum_Name(type.type_enum())));
  }
}

// Converts a TypeProto into the C++ type expression text for the structural
// templates (Bits/Array/Tuple).
absl::StatusOr<std::string> TypeProtoToCpp(const TypeProto& type) {
  switch (type.type_enum()) {
    case TypeProto::BITS:
      if (type.bit_count() < 0) {
        return absl::InvalidArgumentError("BITS type has a negative width.");
      }
      return absl::StrCat("Bits<", type.bit_count(), ">");
    case TypeProto::ARRAY: {
      if (type.array_size() < 0) {
        return absl::InvalidArgumentError("ARRAY type has a negative size.");
      }
      XLS_ASSIGN_OR_RETURN(std::string element,
                           TypeProtoToCpp(type.array_element()));
      return absl::StrCat("Array<", element, ", ", type.array_size(), ">");
    }
    case TypeProto::TUPLE: {
      std::vector<std::string> elements;
      for (const TypeProto& element : type.tuple_elements()) {
        XLS_ASSIGN_OR_RETURN(elements.emplace_back(), TypeProtoToCpp(element));
      }
      return absl::StrCat("Tuple<", absl::StrJoin(elements, ", "), ">");
    }
    default:
      return absl::UnimplementedError(
          absl::StrFormat("C++ packed header generation only supports BITS, "
                          "ARRAY and TUPLE types, got %s.",
                          TypeProto::TypeEnum_Name(type.type_enum())));
  }
}

// Escapes a raw symbol name for use inside a C++ string literal. XLS symbols
// are normally plain ASCII but the escaping keeps the generated header correct
// in the general case.
std::string EscapeCppStringLiteral(std::string_view literal) {
  std::string escaped;
  escaped.reserve(literal.size());
  for (unsigned char c : literal) {
    switch (c) {
      case '\\':
        escaped.append("\\\\");
        break;
      case '"':
        escaped.append("\\\"");
        break;
      case '\n':
        escaped.append("\\n");
        break;
      case '\t':
        escaped.append("\\t");
        break;
      default:
        if (c < 0x20 || c >= 0x7f) {
          absl::StrAppendFormat(&escaped, "\\%03o", c);
        } else {
          escaped.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  return escaped;
}

}  // namespace

absl::StatusOr<std::string> GenerateAotCppHeader(
    const AotPackageEntrypointsProto& package,
    const AotCppHeaderOptions& options) {
  XLS_RETURN_IF_ERROR(ValidateCppNamespace(options.cpp_namespace));
  XLS_RETURN_IF_ERROR(
      ValidateIdentifier(options.entrypoint_name, "cpp_entrypoint_name"));

  if (package.entrypoint_size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "C++ header generation requires exactly one entrypoint in the package "
        "but got %d.",
        package.entrypoint_size()));
  }
  const AotEntrypointProto& entrypoint = package.entrypoint(0);
  if (entrypoint.type() != AotEntrypointProto::FUNCTION) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "C++ packed header generation currently supports "
        "FUNCTION entrypoints only, got entrypoint '%s' of "
        "type %s.",
        entrypoint.xls_function_identifier(),
        AotEntrypointProto::XlsFunctionType_Name(entrypoint.type())));
  }
  if (!entrypoint.has_packed_function_symbol() ||
      entrypoint.packed_function_symbol().empty()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "C++ packed header generation requires a packed function symbol for "
        "entrypoint '%s' but none was found in the metadata.",
        entrypoint.xls_function_identifier()));
  }
  const int64_t temp_alignment = entrypoint.temp_buffer_alignment();
  if (!entrypoint.has_temp_buffer_size() || entrypoint.temp_buffer_size() < 0 ||
      !entrypoint.has_temp_buffer_alignment() ||
      temp_alignment < 0 ||
      (temp_alignment != 0 &&
       (temp_alignment & (temp_alignment - 1)) != 0)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "C++ packed header generation requires a nonnegative temporary buffer "
        "size and an alignment that is zero or a power of two for entrypoint "
        "'%s'.",
        entrypoint.xls_function_identifier()));
  }
  if (!entrypoint.has_function_metadata()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function entrypoint '%s' has no function metadata.",
        entrypoint.xls_function_identifier()));
  }

  const auto& function_interface =
      entrypoint.function_metadata().function_interface();
  const int64_t num_params = function_interface.parameters_size();
  if (entrypoint.inputs_names().size() != num_params) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Input name count (%d) and parameter count (%d) disagree for "
        "entrypoint '%s'.",
        entrypoint.inputs_names().size(), num_params,
        entrypoint.xls_function_identifier()));
  }

  // Structural types of arguments and the result.
  std::vector<std::string> param_type_exprs;
  param_type_exprs.reserve(num_params);
  for (int64_t i = 0; i < num_params; ++i) {
    XLS_ASSIGN_OR_RETURN(
        param_type_exprs.emplace_back(),
        TypeProtoToCpp(function_interface.parameters(i).type()));
  }
  if (!function_interface.has_result_type()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Function entrypoint '%s' has no result type.",
        entrypoint.xls_function_identifier()));
  }
  XLS_ASSIGN_OR_RETURN(std::string result_type_expr,
                       TypeProtoToCpp(function_interface.result_type()));

  // The packed sizes in the proto are in bits, rounded up to the next byte.
  // Verify that the structural types agree with the metadata. A mismatch here
  // means the atomic metadata (e.g. this header) would silently produce wrong
  // ABI guesses.
  const int64_t packed_input_count =
      entrypoint.packed_input_buffer_sizes().size();
  const int64_t packed_output_count =
      entrypoint.packed_output_buffer_sizes().size();
  if (num_params != packed_input_count || packed_output_count != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Packed buffer counts (%d inputs, %d outputs) disagree with function "
        "entrypoint '%s' (%d parameters, 1 result).",
        packed_input_count, packed_output_count,
        entrypoint.xls_function_identifier(), num_params));
  }

  for (int64_t i = 0; i < num_params; ++i) {
    XLS_ASSIGN_OR_RETURN(int64_t flat,
                         FlatBitCount(function_interface.parameters(i).type()));
    if (flat > std::numeric_limits<int64_t>::max() - 7) {
      return absl::OutOfRangeError("Packed input bit count overflows int64.");
    }
    const int64_t packed = entrypoint.packed_input_buffer_sizes(i);
    if (PackedBitsFromFlat(flat) != packed) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Packed size mismatch for input %d of entrypoint '%s': the type "
          "tree has %d flat bits (packed: %d) but the metadata states %d bits.",
          i, entrypoint.xls_function_identifier(), flat,
          PackedBitsFromFlat(flat), packed));
    }
  }
  {
    XLS_ASSIGN_OR_RETURN(int64_t flat,
                         FlatBitCount(function_interface.result_type()));
    if (flat > std::numeric_limits<int64_t>::max() - 7) {
      return absl::OutOfRangeError("Packed output bit count overflows int64.");
    }
    const int64_t expected = entrypoint.packed_output_buffer_sizes(0);
    if (PackedBitsFromFlat(flat) != expected) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Packed output size mismatch for entrypoint '%s': the type tree has "
          "%d flat bits (packed %d) but the metadata states %d bits.",
          entrypoint.xls_function_identifier(), flat, PackedBitsFromFlat(flat),
          expected));
    }
  }

  // Candidate semantic aliases derived from the parameter names. `Input{i}`
  // aliases are always emitted; a semantic alias is emitted only if the name
  // converts to a unique, non-colliding C++ identifier.
  std::vector<std::string> semantic_names;
  semantic_names.reserve(num_params);
  {
    absl::flat_hash_set<std::string> used_names;
    const absl::flat_hash_set<std::string> reserved = {
        "Bits",
        "Array",
        "Tuple",
        "Result",
        "PackedFunction",
        "InvokePacked",
        "InputCount",
        "OutputCount",
        "kInputCount",
        "kOutputCount",
        "kPackedInputBitCounts",
        "kPackedOutputBitCounts",
        "kTemporaryBufferSize",
        "kTemporaryBufferAlignment",
        "PackedBitCount",
        absl::StrCat(options.entrypoint_name, "InvokePacked")};
    auto collides_with_input_alias = [](const std::string& name) {
      if (name.size() <= 5 || name.substr(0, 5) != "Input") {
        return false;
      }
      return std::all_of(name.begin() + 5, name.end(),
                         [](char c) { return c >= '0' && c <= '9'; });
    };
    for (int64_t i = 0; i < num_params; ++i) {
      std::string semantic;
      if (entrypoint.inputs_names_size() > i) {
        auto name = ToCppTypeName(entrypoint.inputs_names(i));
        if (name.ok() && !used_names.contains(*name) &&
            !reserved.contains(*name) && !collides_with_input_alias(*name)) {
          semantic = *name;
        }
      }
      if (!semantic.empty()) {
        used_names.insert(semantic);
      }
      semantic_names.push_back(std::move(semantic));
    }
  }

  // Length-delimited spelling preserves case and component boundaries, making
  // the guard injective over valid namespace/entrypoint combinations.
  std::string guard = "XLS_AOT";
  for (const std::string_view part : absl::StrSplit(
           options.cpp_namespace, absl::ByString("::"), absl::SkipEmpty())) {
    absl::StrAppend(&guard, "_", part.size(), "_", part);
  }
  absl::StrAppend(&guard, "_", options.entrypoint_name.size(), "_",
                  options.entrypoint_name);
  const std::string include_guard = absl::StrCat(guard, "_H_");

  std::string out;
  out.reserve(8192);
  absl::StrAppend(&out,
                  "// Generated by XLS aot_compiler_main; do not edit.\n");
  absl::StrAppend(&out, "//\n");
  absl::StrAppend(&out,
                  "// Self-contained C++20 ABI description of an XLS AOT "
                  "packed entrypoint.\n");
  absl::StrAppend(&out, "//\n");
  absl::StrAppend(&out, "// Entrypoint: ", options.entrypoint_name, "\n");
  if (entrypoint.has_xls_package_name()) {
    absl::StrAppend(&out, "// XLS package: ", entrypoint.xls_package_name(),
                    "\n");
  }
  absl::StrAppend(
      &out, "// XLS function: ", entrypoint.xls_function_identifier(), "\n");
  absl::StrAppend(&out, "\n");
  absl::StrAppend(&out, "#ifndef ", include_guard, "\n");
  absl::StrAppend(&out, "#define ", include_guard, "\n");
  absl::StrAppend(&out, "\n");
  absl::StrAppend(
      &out,
      "#if !defined(__GNUC__) && !defined(__clang__)\n"
      "#error \"XLS AOT generated headers require GCC or Clang asm-label "
      "support.\"\n"
      "#endif\n\n");
  absl::StrAppend(&out, "#include <array>\n");
  absl::StrAppend(&out, "#include <cstddef>\n");
  absl::StrAppend(&out, "#include <cstdint>\n");
  absl::StrAppend(&out, "#include <tuple>\n");
  absl::StrAppend(&out, "\n");

  std::vector<std::string> namespace_parts;
  for (const std::string_view part : absl::StrSplit(
           options.cpp_namespace, absl::ByString("::"), absl::SkipEmpty())) {
    namespace_parts.push_back(std::string(part));
  }
  namespace_parts.push_back(options.entrypoint_name);
  for (const std::string& part : namespace_parts) {
    absl::StrAppend(&out, "namespace ", part, " {\n");
  }
  absl::StrAppend(&out, "\n");

  // ---------------------------------------------------------------------------
  // Support templates.
  absl::StrAppend(
      &out,
      "// Struct descriptions of packed XLS types. The templates carry only "
      "the\n"
      "// structure (element types, sizes and bit counts); numeric sizes are\n"
      "// reproduced here so the consumer never has to copy values from IR or\n"
      "// the .entrypoints.pb by hand.\n"
      "template <std::size_t Width>\n"
      "struct Bits {\n"
      "  static constexpr std::size_t kBitCount = Width;\n"
      "};\n"
      "\n"
      "template <typename ElementType, std::size_t Size>\n"
      "struct Array {\n"
      "  using Element = ElementType;\n"
      "  static constexpr std::size_t kSize = Size;\n"
      "  static constexpr std::size_t kBitCount = ElementType::kBitCount * "
      "Size;\n"
      "};\n"
      "\n"
      "template <typename... ElementTypes>\n"
      "struct Tuple {\n"
      "  static constexpr std::size_t kSize = sizeof...(ElementTypes);\n"
      "  static constexpr std::size_t kBitCount =\n"
      "      (ElementTypes::kBitCount + ... + 0);\n"
      "\n"
      "  template <std::size_t Index>\n"
      "  using Element =\n"
      "      std::tuple_element_t<Index, std::tuple<ElementTypes...>>;\n"
      "\n"
      "  // XLS packed tuples are declared MSB-to-LSB: element zero is the\n"
      "  // highest element. The offset of an element is the number of "
      "trailing\n"
      "  // bits (counted from bit zero) it occupies in the packed buffer.\n"
      "  template <std::size_t Index>\n"
      "  static consteval std::size_t Offset() {\n"
      "    static_assert(Index < kSize, \"Tuple element index out of "
      "range.\");\n"
      "    return TupleOffsetImpl<Index, ElementTypes...>::Value();\n"
      "  }\n"
      "\n"
      " private:\n"
      "  template <std::size_t Index, typename... Types>\n"
      "  struct TupleOffsetImpl;\n"
      "\n"
      "  template <std::size_t Index, typename First, typename... Rest>\n"
      "  struct TupleOffsetImpl<Index, First, Rest...> {\n"
      "    static consteval std::size_t Value() {\n"
      "      if constexpr (sizeof...(Rest) == 0) {\n"
      "        return 0;\n"
      "      } else {\n"
      "        if constexpr (Index == 0) {\n"
      "          return (Rest::kBitCount + ... + 0);\n"
      "        } else {\n"
      "          return TupleOffsetImpl<Index - 1, Rest...>::Value();\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  };\n"
      "};\n"
      "\n"
      "// The packed buffer of a value with 'BitCount' bits occupies this "
      "many\n"
      "// bits, rounded up to the nearest byte (matching the XLS packed ABI).\n"
      "// A zero-bit value occupies one byte for simplicity.\n"
      "template <std::size_t BitCount>\n"
      "static constexpr std::size_t PackedBitCount =\n"
      "    BitCount == 0 ? 8 : ((BitCount + 7) / 8) * 8;\n"
      "\n");

  // ---------------------------------------------------------------------------
  // ABI constants.
  absl::StrAppend(&out,
                  "// ABI constants taken from the AOT metadata.\n"
                  "inline constexpr std::size_t kInputCount = ",
                  num_params,
                  ";\n"
                  "inline constexpr std::size_t kOutputCount = ",
                  packed_output_count,
                  ";\n"
                  "inline constexpr std::size_t kTemporaryBufferSize = ",
                  entrypoint.temp_buffer_size(),
                  ";\n"
                  "inline constexpr std::size_t kTemporaryBufferAlignment = ",
                  entrypoint.temp_buffer_alignment(),
                  ";\n"
                  "\n"
                  "// Packed input/output buffer sizes are bit counts (rounded "
                  "up to a byte\n"
                  "// boundary), not byte counts.\n"
                  "inline constexpr std::array<std::size_t, kInputCount> "
                  "kPackedInputBitCounts = {");
  for (int64_t i = 0; i < num_params; ++i) {
    absl::StrAppend(&out, i == 0 ? "" : ", ",
                    entrypoint.packed_input_buffer_sizes(i));
  }
  absl::StrAppend(&out, "};\n");
  absl::StrAppend(
      &out,
      "inline constexpr std::array<std::size_t, kOutputCount> "
      "kPackedOutputBitCounts = {");
  for (int64_t i = 0; i < packed_output_count; ++i) {
    absl::StrAppend(&out, i == 0 ? "" : ", ",
                    entrypoint.packed_output_buffer_sizes(i));
  }
  absl::StrAppend(&out, "};\n\n");

  // The packed ABI function pointer type.
  const std::string packed_function_identifier =
      absl::StrCat(options.entrypoint_name, "InvokePacked");
  absl::StrAppend(
      &out,
      "// The ABI of the packed entrypoint: the actual symbol exported by the\n"
      "// AOT object file.\n"
      "using PackedFunction = std::int64_t (*)(\n"
      "    const std::uint8_t* const* inputs, std::uint8_t* const* outputs,\n"
      "    void* temporary_buffer, void* events, void* instance_context,\n"
      "    void* runtime, std::int64_t continuation_point);\n"
      "\n"
      "// The real packed symbol under a stable C++ identifier.\n"
      "extern \"C\" std::int64_t ",
      packed_function_identifier,
      "(\n"
      "    const std::uint8_t* const* inputs, std::uint8_t* const* outputs,\n"
      "    void* temporary_buffer, void* events, void* instance_context,\n"
      "    void* runtime, std::int64_t continuation_point) asm(\"",
      EscapeCppStringLiteral(entrypoint.packed_function_symbol()),
      "\");\n"
      "\n"
      "inline constexpr PackedFunction kPackedFunction = &",
      packed_function_identifier,
      ";\n"
      "\n");

  // ---------------------------------------------------------------------------
  // Argument/result type aliases.
  for (int64_t i = 0; i < num_params; ++i) {
    absl::StrAppend(&out, "using Input", i, " = ", param_type_exprs[i], ";\n");
    if (!semantic_names[i].empty()) {
      // Keeps semantic names (e.g. snake_case DSLX parameters) available.
      absl::StrAppend(&out, "using ", semantic_names[i], " = Input", i, ";\n");
    }
  }
  absl::StrAppend(&out, "using Result = ", result_type_expr, ";\n\n");

  for (int64_t i = 0; i < num_params; ++i) {
    absl::StrAppend(&out, "static_assert(PackedBitCount<Input", i,
                    "::kBitCount> == kPackedInputBitCounts[", i,
                    "], \"Input type mismatch with metadata.\");\n");
  }
  absl::StrAppend(
      &out,
      "static_assert(PackedBitCount<Result::kBitCount> == "
      "kPackedOutputBitCounts[0], \"Result type mismatch with metadata.\");\n");
  absl::StrAppend(&out, "\n");

  for (auto it = namespace_parts.rbegin(); it != namespace_parts.rend(); ++it) {
    absl::StrAppend(&out, "}  // namespace ", *it, "\n");
  }
  absl::StrAppend(&out, "\n#endif  // ", include_guard, "\n");

  return out;
}

}  // namespace xls
