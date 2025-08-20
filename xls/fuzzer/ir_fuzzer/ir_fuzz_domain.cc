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

#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>

#include "xls/common/fuzzing/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/fuzzer/ir_fuzzer/fuzz_program.pb.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_builder.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/verifier.h"

namespace xls {

// NB a lot of these functions need 'auto' because fuzztest returns
// fuzztest::internal:: values and requires you to use functions defined on
// them. This means naming the types could cause compilation issues if fuzztest
// ever changes its abi. To avoid that complexity just use auto and rely on
// c++ forwarding to keep the types straight.
//
// Weirdness about how fuzztest values get built also forces the use of a monad
// like implementation where we pass the domain around since there is no
// operator=(&&) implementation for the domains.
namespace {

// Turn off all ops in the 'ops' list.
auto RestrictOpsRecursive(auto&& domain, absl::Span<Op const> ops) {
  // Avoid annoying fuzztest type coercions by doing this recursively.
  if (ops.empty()) {
    return std::forward<decltype(domain)>(domain);
  }
  auto add_one = [](decltype(domain)&& domain, Op op) -> decltype(domain) {
    switch (op) {
      case Op::kAdd:
        return std::move(domain).WithFieldUnset("add");
      case Op::kSub:
        return std::move(domain).WithFieldUnset("subtract");
      case Op::kUMul:
        return std::move(domain).WithFieldUnset("umul");
      case Op::kUDiv:
        return std::move(domain).WithFieldUnset("udiv");
      case Op::kSMul:
        return std::move(domain).WithFieldUnset("smul");
      case Op::kSDiv:
        return std::move(domain).WithFieldUnset("sdiv");
      case Op::kUMod:
        return std::move(domain).WithFieldUnset("umod");
      case Op::kSMod:
        return std::move(domain).WithFieldUnset("smod");
      case Op::kAnd:
        return std::move(domain).WithFieldUnset("and_op");
      case Op::kOr:
        return std::move(domain).WithFieldUnset("or_op");
      case Op::kNor:
        return std::move(domain).WithFieldUnset("nor");
      case Op::kXor:
        return std::move(domain).WithFieldUnset("xor_op");
      case Op::kNand:
        return std::move(domain).WithFieldUnset("nand");
      case Op::kAndReduce:
        return std::move(domain).WithFieldUnset("and_reduce");
      case Op::kOrReduce:
        return std::move(domain).WithFieldUnset("or_reduce");
      case Op::kXorReduce:
        return std::move(domain).WithFieldUnset("xor_reduce");
      case Op::kUMulp:
        return std::move(domain).WithFieldUnset("umulp");
      case Op::kSMulp:
        return std::move(domain).WithFieldUnset("smulp");
      case Op::kConcat:
        return std::move(domain).WithFieldUnset("concat");
      case Op::kULe:
        return std::move(domain).WithFieldUnset("ule");
      case Op::kULt:
        return std::move(domain).WithFieldUnset("ult");
      case Op::kUGe:
        return std::move(domain).WithFieldUnset("uge");
      case Op::kUGt:
        return std::move(domain).WithFieldUnset("ugt");
      case Op::kSLe:
        return std::move(domain).WithFieldUnset("sle");
      case Op::kSLt:
        return std::move(domain).WithFieldUnset("slt");
      case Op::kSGe:
        return std::move(domain).WithFieldUnset("sge");
      case Op::kSGt:
        return std::move(domain).WithFieldUnset("sgt");
      case Op::kEq:
        return std::move(domain).WithFieldUnset("eq");
      case Op::kNe:
        return std::move(domain).WithFieldUnset("ne");
      case Op::kNeg:
        return std::move(domain).WithFieldUnset("negate");
      case Op::kNot:
        return std::move(domain).WithFieldUnset("not_op");
      case Op::kSel:
        return std::move(domain).WithFieldUnset("select");
      case Op::kOneHot:
        return std::move(domain).WithFieldUnset("one_hot");
      case Op::kOneHotSel:
        return std::move(domain).WithFieldUnset("one_hot_select");
      case Op::kPrioritySel:
        return std::move(domain).WithFieldUnset("priority_select");
      case Op::kTuple:
        return std::move(domain).WithFieldUnset("tuple");
      case Op::kArray:
        return std::move(domain).WithFieldUnset("array");
      case Op::kTupleIndex:
        return std::move(domain).WithFieldUnset("tuple_index");
      case Op::kArrayIndex:
        return std::move(domain).WithFieldUnset("array_index");
      case Op::kArraySlice:
        return std::move(domain).WithFieldUnset("array_slice");
      case Op::kArrayUpdate:
        return std::move(domain).WithFieldUnset("array_update");
      case Op::kArrayConcat:
        return std::move(domain).WithFieldUnset("array_concat");
      case Op::kReverse:
        return std::move(domain).WithFieldUnset("reverse");
      case Op::kIdentity:
        return std::move(domain).WithFieldUnset("identity");
      case Op::kSignExt:
        return std::move(domain).WithFieldUnset("sign_extend");
      case Op::kZeroExt:
        return std::move(domain).WithFieldUnset("zero_extend");
      case Op::kBitSlice:
        return std::move(domain).WithFieldUnset("bit_slice");
      case Op::kBitSliceUpdate:
        return std::move(domain).WithFieldUnset("bit_slice_update");
      case Op::kDynamicBitSlice:
        return std::move(domain).WithFieldUnset("dynamic_bit_slice");
      case Op::kEncode:
        return std::move(domain).WithFieldUnset("encode");
      case Op::kDecode:
        return std::move(domain).WithFieldUnset("decode");
      case Op::kShra:
        return std::move(domain).WithFieldUnset("shra");
      case Op::kShll:
        return std::move(domain).WithFieldUnset("shll");
      case Op::kShrl:
        return std::move(domain).WithFieldUnset("shrl");
      case Op::kGate:
        return std::move(domain).WithFieldUnset("gate");
      default:
        // Do nothing for ops not in FuzzOpProto.
        return std::move(domain);
    }
  };
  return RestrictOpsRecursive(
      add_one(std::forward<decltype(domain)>(domain), ops.front()),
      ops.subspan(1));
}

auto RestrictOps(auto&& domain, absl::Span<Op const> ops, bool allow_clz,
                 bool allow_ctz) {
  auto base = RestrictOpsRecursive(std::forward<decltype(domain)>(domain), ops);
  bool allow_match = absl::c_none_of(ops, [](Op op) {
    return op == Op::kSel || op == Op::kPrioritySel || op == Op::kOneHotSel;
  });
  auto sel = allow_match
                 ? std::move(base)
                 : std::move(base).WithFieldUnset("match").WithFieldUnset(
                       "match_true");
  auto clz = allow_clz ? std::move(sel) : std::move(sel).WithFieldUnset("clz");
  auto ctz = allow_ctz ? std::move(clz) : std::move(clz).WithFieldUnset("ctz");
  return ctz;
}
std::vector<Op> GetRestrictedSet(absl::Span<Op const> ops, bool only_bits) {
  std::vector<Op> restricted_set;
  restricted_set.reserve(kAllOps.size());
  absl::flat_hash_set<Op> kept(ops.begin(), ops.end());
  // Add always included ones.
  kept.insert(Op::kParam);
  kept.insert(Op::kLiteral);
  auto is_non_bits = [](Op op) {
    switch (op) {
      case Op::kArray:
      case Op::kArrayIndex:
      case Op::kArrayConcat:
      case Op::kArraySlice:
      case Op::kArrayUpdate:
      case Op::kTuple:
      case Op::kTupleIndex:
        return true;
      default:
        return false;
    }
  };
  absl::c_copy_if(kAllOps, std::back_inserter(restricted_set), [&](Op o) {
    if (only_bits && is_non_bits(o)) {
      return true;
    }
    return !kept.contains(o);
  });
  return restricted_set;
}
}  // namespace

fuzztest::Domain<FuzzPackage> FuzzPackageDomainBuilder::Build() && {
  fuzztest::Domain<FuzzOpProto> op_domain = RestrictOps(
      fuzztest::Arbitrary<FuzzOpProto>().WithOneofAlwaysSet("fuzz_op"),
      GetRestrictedSet(ops_, only_bits_), allow_clz_, allow_ctz_);
  auto args = with_args_
                  ? static_cast<fuzztest::Domain<std::string>>(
                        fuzztest::Arbitrary<std::string>().WithMinSize(1000))
                  : fuzztest::Just<std::string>("");
  return fuzztest::Map(
      [](FuzzProgramProto fuzz_program) {
        // Create the package.
        std::unique_ptr<Package> p =
            std::make_unique<VerifiedPackage>(kFuzzTestName);
        FunctionBuilder fb(kFuzzTestName, p.get());
        // Build the IR from the FuzzProgramProto.
        IrFuzzBuilder ir_fuzz_builder(fuzz_program, p.get(), &fb);
        BValue ir = ir_fuzz_builder.BuildIr();
        CHECK_OK(fb.BuildWithReturnValue(ir))
            << "Failed to build package from FuzzProgramProto: "
            << fuzz_program.DebugString();
        // Create the FuzzPackage object as the domain export.
        return FuzzPackage(std::move(p), fuzz_program);
      },
      fuzztest::Arbitrary<FuzzProgramProto>()
          .WithStringField("args_bytes", args)
          .WithEnumField("version",
                         fuzztest::Just<int>(kCurrentFuzzProtoVersion))
          .WithRepeatedProtobufField(
              "fuzz_ops",
              // Generate at least one FuzzOp.
              fuzztest::VectorOf(op_domain).WithMinSize(min_op_count_)));
}

fuzztest::Domain<FuzzPackageWithArgs> PackageWithArgsDomainBuilder::Build() && {
  return fuzztest::Map(
      [arg_set_count = arg_set_count_](FuzzPackage fuzz_package) {
        return GenArgSetsForPackage(std::move(fuzz_package), arg_set_count);
      },
      std::move(base_).Build());
}

fuzztest::Domain<std::shared_ptr<Package>> PackageDomainBuilder::Build() && {
  return fuzztest::Map(
      [](FuzzPackage fuzz_package) -> std::shared_ptr<Package> {
        return std::shared_ptr<Package>(fuzz_package.p.release());
      },
      std::move(base_).Build());
}

fuzztest::Domain<std::shared_ptr<Package>> IrFuzzDomain() {
  return PackageDomainBuilder().Build();
}

// Same as IrFuzzDomain but returns a FuzzPackageWithArgs domain which also
// contains the argument sets that are compatible with the function.
fuzztest::Domain<FuzzPackageWithArgs> IrFuzzDomainWithArgs(
    int64_t arg_set_count) {
  return PackageWithArgsDomainBuilder(arg_set_count).Build();
}

}  // namespace xls
