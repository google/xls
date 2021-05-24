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

#include "xls/dslx/scanner.h"

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "absl/strings/strip.h"
#include "pybind11/functional.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xls/common/status/statusor_pybind_caster.h"
#include "xls/dslx/ast_builtin_types.inc"
#include "xls/dslx/pos.h"
#include "xls/dslx/python/errors.h"

namespace py = pybind11;

namespace xls::dslx {

const absl::Status& GetStatus(const absl::Status& status) { return status; }
template <typename T>
const absl::Status& GetStatus(const absl::StatusOr<T>& status_or) {
  return status_or.status();
}

template <typename ReturnT, typename... Args>
std::function<ReturnT(Scanner*, Args...)> ScanErrorWrap(
    ReturnT (Scanner::*f)(Args...)) {
  return [f](Scanner* s, Args... args) {
    auto statusor = ((*s).*f)(std::forward<Args>(args)...);
    TryThrowScanError(GetStatus(statusor));
    return statusor;
  };
}

PYBIND11_MODULE(scanner, m) {
  ImportStatusModule();

  py::enum_<Keyword>(m, "Keyword")
#define VALUE(__enum, __pyattr, ...) .value(#__pyattr, Keyword::__enum)
      XLS_DSLX_KEYWORDS(VALUE)
#undef VALUE
          .export_values()
          .def_property_readonly("value", [](Keyword keyword) {
            return KeywordToString(keyword);
          });

  py::enum_<TokenKind>(m, "TokenKind")
#define VALUE(__enum, __pyattr, ...) .value(#__pyattr, TokenKind::__enum)
      XLS_DSLX_TOKEN_KINDS(VALUE)
#undef VALUE
          .export_values()
          .def_property_readonly(
              "value", [](TokenKind kind) { return TokenKindToString(kind); });

  m.def("TokenKindFromString",
        [](absl::string_view s) { return TokenKindFromString(s); });

  static py::exception<ScanError> parse_exc(m, "ScanError");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const ScanError& e) {
      py::object& e_type = parse_exc;
      py::object instance = e_type();
      instance.attr("message") = e.what();
      instance.attr("span") = e.span();
      PyErr_SetObject(parse_exc.ptr(), instance.ptr());
    }
  });

  py::class_<Token>(m, "Token")
      .def(py::init([](TokenKind kind, Span span,
                       absl::optional<std::string> value) {
             return Token(kind, span, value);
           }),
           py::arg("kind"), py::arg("span"), py::arg("value"))
      .def(py::init(
               [](Span span, Keyword keyword) { return Token(span, keyword); }),
           py::arg("span"), py::arg("value"))
      .def_property_readonly("kind", &Token::kind)
      .def_property_readonly("span", &Token::span)
      .def_property_readonly("value", &Token::GetPayload)
      .def("to_error_str", &Token::ToErrorString)
      .def("is_keyword", &Token::IsKeyword)
      .def("is_keyword_in", &Token::IsKeywordIn)
      .def("is_type_keyword", &Token::IsTypeKeyword)
      .def("is_identifier", &Token::IsIdentifier)
      .def("is_number", &Token::IsNumber)
      .def("__str__", &Token::ToString)
      .def("__repr__", &Token::ToRepr);

  py::class_<Scanner>(m, "Scanner")
      .def(py::init([](std::string filename, std::string text,
                       bool include_whitespace_and_comments) {
             return Scanner(std::move(filename), std::move(text),
                            include_whitespace_and_comments);
           }),
           py::arg("filename"), py::arg("text"),
           py::arg("include_whitespace_and_comments") = false)
      .def("at_eof", &Scanner::AtEof)
      .def("pop", ScanErrorWrap(&Scanner::Pop))
      .def("pop_all", ScanErrorWrap(&Scanner::PopAll))
      .def_property_readonly("pos", &Scanner::GetPos);
}

}  // namespace xls::dslx
