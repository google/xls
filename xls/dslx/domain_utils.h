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

#ifndef XLS_DSLX_DOMAIN_UTILS_H_
#define XLS_DSLX_DOMAIN_UTILS_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/import_data.h"
#include "xls/dslx/interp_value.h"
#include "xls/dslx/type_system/type.h"
#include "xls/dslx/type_system/type_info.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

// Validates that a domain expression is compatible with the corresponding
// function parameter type. Returns an error if not compatible.
absl::Status ValidateDomain(const Expr* domain, const Type* param_type,
                            std::string_view param_str, const TypeInfo& ti,
                            WarningCollector& warning_collector,
                            const ImportData& import_data,
                            const FileTable& file_table);

// Validates that each domain expression in `domains` is compatible with the
// corresponding parameter type of `fn`. Returns an error if any domain is not
// compatible.
absl::Status ValidateFunctionDomains(const Function& fn,
                                     const XlsTuple* domains,
                                     const TypeInfo& ti,
                                     WarningCollector& warning_collector,
                                     const ImportData& import_data,
                                     const FileTable& file_table);

// Enumerates all values in the unconstrained ("arbitrary") domain for `type`,
// bounded by `max_points`.
//
// When `allow_composite` is false, only enum, bool, and fixed-width integer
// types are accepted; when true, struct, tuple, and array types are
// recursively enumerated as Cartesian products of their member/element domains.
absl::StatusOr<std::vector<InterpValue>> EnumerateArbitraryDomain(
    std::string_view param_name, const Type& type, int64_t max_points,
    bool allow_composite = true);

// Evaluates a DSLX domain expression `expr` for a single parameter (as used in
// `domains=` attributes such as `#[fuzz_test(domains=`...`)]` and
// `#[tabulate(domains=`...`)]`) against `target_type`, returning the enumerated
// sequence of `InterpValue`s up to `max_points`.
//
// Supports `()` arbitrary domains, exclusive (`..`) and inclusive (`..=`)
// ranges, ElementOf arrays (`[...]`), constant/imported/helper-function domain
// values, tuple domains, array domains with optional trailing unconstrained
// elements, inline struct domains, and `#[fuzz_domain]` derived domain structs.
//
// Returns `InvalidArgumentError` if `expr` is incompatible with `target_type`,
// produces an empty domain, or exceeds `max_points`. Callers that want DSLX
// source-span type-inference errors should still run `ValidateDomain` or
// `ValidateFunctionDomains` during typechecking.
absl::StatusOr<std::vector<InterpValue>> EnumerateDomainForOneParam(
    ImportData* import_data, TypeInfo* type_info, std::string_view param_name,
    const Type& target_type, const Expr* expr, int64_t max_points);

// Generates the Cartesian product of input parameter domains for `fn` up to
// `max_points` rows.
//
// If `domains_tuple` is present, each parameter's domain is evaluated from the
// corresponding tuple member expression (returning `InvalidArgumentError` on
// arity or type mismatches); otherwise, each parameter's full scalar input
// domain is enumerated.
absl::StatusOr<std::vector<std::vector<InterpValue>>> EnumerateFunctionDomains(
    ImportData* import_data, TypeInfo* type_info, const Function& fn,
    const FunctionType& fn_type, std::optional<const XlsTuple*> domains_tuple,
    int64_t max_points);

}  // namespace xls::dslx

#endif  // XLS_DSLX_DOMAIN_UTILS_H_
