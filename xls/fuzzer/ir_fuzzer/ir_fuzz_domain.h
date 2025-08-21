// Copyright 2025 The XLS Authors
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

#ifndef XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_
#define XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_

#include <memory>
#include <cstdint>
#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/types/span.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"

// Contains functions that return IR fuzz test domains.

namespace xls {

// What adjustable behavior we are going to use to generate new fuzz examples.
constexpr FuzzVersion kCurrentFuzzProtoVersion =
    FuzzVersion::BOUND_WITH_MODULO_VERSION;

// Helper to build a fuzz package domain.
class FuzzPackageDomainBuilder {
 public:
  fuzztest::Domain<FuzzPackage> Build() &&;
  FuzzPackageDomainBuilder WithOnlyBitsOperations() && {
    only_bits_ = true;
    return std::move(*this);
  }
  // Limit the package to only contain the given operations. Note that kParam
  // and kLiteral are always allowed, as are extend and slice. If non-bits
  // operations are allowed then the array/tuple index, slice etc operations are
  // also always allowed.
  FuzzPackageDomainBuilder WithOperations(absl::Span<Op const> ops) && {
    ops_ = ops;
    return std::move(*this);
  }
  FuzzPackageDomainBuilder MinOpCount(int64_t min) && {
    min_op_count_ = min;
    return std::move(*this);
  }

  FuzzPackageDomainBuilder WithArgs(bool val) && {
    with_args_ = val;
    return std::move(*this);
  }

  FuzzPackageDomainBuilder NoClz() && {
    allow_clz_ = false;
    return std::move(*this);
  }
  FuzzPackageDomainBuilder NoCtz() && {
    allow_ctz_ = false;
    return std::move(*this);
  }

 private:
  bool only_bits_ = false;
  absl::Span<Op const> ops_ = kAllOps;
  int64_t min_op_count_ = 1;
  bool with_args_ = true;
  bool allow_clz_ = true;
  bool allow_ctz_ = true;
};

// Helper to build a package with args domain.
class PackageWithArgsDomainBuilder {
 public:
  PackageWithArgsDomainBuilder(int64_t arg_set_count)
      : arg_set_count_(arg_set_count) {}
  fuzztest::Domain<FuzzPackageWithArgs> Build() &&;
  PackageWithArgsDomainBuilder WithOnlyBitsOperations() && {
    return PackageWithArgsDomainBuilder(
        std::move(base_).WithOnlyBitsOperations(), arg_set_count_);
  }
  // Limit the package to only contain the given operations. Note that kParam
  // and kLiteral are always allowed, as are extend and slice. If non-bits
  // operations are allowed then the array/tuple index, slice etc operations are
  // also always allowed.
  PackageWithArgsDomainBuilder WithOperations(absl::Span<Op const> ops) && {
    return PackageWithArgsDomainBuilder(std::move(base_).WithOperations(ops),
                                        arg_set_count_);
  }
  PackageWithArgsDomainBuilder MinOpCount(int64_t min) {
    return PackageWithArgsDomainBuilder(std::move(base_).MinOpCount(min),
                                        arg_set_count_);
  }

  PackageWithArgsDomainBuilder NoClz() && {
    return PackageWithArgsDomainBuilder(std::move(base_).NoClz(),
                                        arg_set_count_);
  }
  PackageWithArgsDomainBuilder NoCtz() && {
    return PackageWithArgsDomainBuilder(std::move(base_).NoCtz(),
                                        arg_set_count_);
  }

 private:
  explicit PackageWithArgsDomainBuilder(FuzzPackageDomainBuilder&& base,
                                        int64_t arg_set_count)
      : base_(std::move(base)), arg_set_count_(arg_set_count) {}
  FuzzPackageDomainBuilder base_;
  int64_t arg_set_count_;
};

// Helper to build a package domain.
class PackageDomainBuilder {
 public:
  PackageDomainBuilder() : base_(FuzzPackageDomainBuilder().WithArgs(false)) {}
  fuzztest::Domain<std::shared_ptr<Package>> Build() &&;
  PackageDomainBuilder WithOnlyBitsOperations() && {
    return PackageDomainBuilder(std::move(base_).WithOnlyBitsOperations());
  }
  // Limit the package to only contain the given operations. Note that kParam
  // and kLiteral are always allowed, as are extend and slice. If non-bits
  // operations are allowed then the array/tuple index, slice etc operations are
  // also always allowed.
  PackageDomainBuilder WithOperations(absl::Span<Op const> ops) && {
    return PackageDomainBuilder(std::move(base_).WithOperations(ops));
  }
  PackageDomainBuilder MinOpCount(int64_t min) {
    return PackageDomainBuilder(std::move(base_).MinOpCount(min));
  }
  PackageDomainBuilder NoClz() && {
    return PackageDomainBuilder(std::move(base_).NoClz());
  }
  PackageDomainBuilder NoCtz() && {
    return PackageDomainBuilder(std::move(base_).NoCtz());
  }

 private:
  explicit PackageDomainBuilder(FuzzPackageDomainBuilder&& base)
      : base_(std::move(base)) {}
  FuzzPackageDomainBuilder base_;
};

fuzztest::Domain<std::shared_ptr<Package>> IrFuzzDomain();
fuzztest::Domain<FuzzPackageWithArgs> IrFuzzDomainWithArgs(
    int64_t arg_set_count);

}  // namespace xls

#endif  // XLS_FUZZER_IR_FUZZER_IR_FUZZ_DOMAIN_H_
