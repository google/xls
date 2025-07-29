# XLS: IR Fuzzer

[TOC]

## Usage

The most common use of the IR Fuzzer is to generate a fuzzed `Function` object
and perform some test on it. For example, the following test takes a fuzzed
function, plugs arguments into the function resulting in a return value,
optimizes the function using one of the XLS optimization passes, plugs the same
arguments into the optimized function resulting in another return value, and
compares the two return values, making sure they are the same. This test in
particular can be used in any optimization pass
([link to code](https://github.com/google/xls/blob/main/xls/passes/reassociation_pass_test.cc)):

```c++
void IrFuzzReassociation(FuzzPackageWithArgs fuzz_package_with_args) {
  ReassociationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzReassociation)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));
```

The FuzzTest accepts the `IrFuzzDomain()` or `IrFuzzDomainWithArgs()` domain
depending on whether or not you want arguments that can be plugged into the
fuzzed function. If you choose to use arguments, you may generate multiple sets
of arguments using the `arg_set_count` parameter. The
`OptimizationPassChangesOutputs()` test accepts the fuzzed function and the
optimization pass.

## Introduction

Compilers have bugs, but triggering them and detecting them is difficult. The
input space is huge as the compiler accepts all possible programs, after which
the compiled program also accepts its own inputs. It's usually impossible to
generalize test coverage. Just because one program is compiled correctly doesn't
say much about the next. Manually curating test programs does not scale. To
tackle these challenges, it's valuable to leverage automation to efficiently
explore the input space.

[Fuzzing](https://github.com/google/fuzztest/blob/main/doc/overview.md) is a
technique of randomly generating inputs to unit tests to attempt to find bugs.
Google FuzzTest purposefully chooses inputs that explore more parts of the
program. Google FuzzTest is coverage guided, meaning that it will try to
generate inputs that execute previously unvisited parts of the code. Fuzzing is
also quite efficient, allowing for many inputs to be generated in a short amount
of time. The IR Fuzzer uses Google fuzzing to generate a vast amount of compiler
inputs in hopes of finding bugs. More specifically, the IR Fuzzer generates
`xls::Function` objects.

To make the compilation pipeline more organized, XLS is split up into several
layers. You start at the [DSLX](https://google.github.io/xls/dslx_reference/)
programming language layer, which then gets compiled into the XLS intermediate
representation ([IR](https://google.github.io/xls/ir_overview/)) layer, which
then eventually becomes Verilog. The IR Fuzzer generates fuzzed IR because we
want to find bugs in the IR specifically. Additionally, we will be able to pass
the fuzzed IR to downstream XLS layers to find bugs in these layers as well.

XLS already has a [DSLX Fuzzer](https://google.github.io/xls/fuzzer/), which
randomly generates DSLX programs. While the DSLX Fuzzer is useful, it is
problematic in several ways:

                | Input Generation Method                                                          | Uses GoogleTest | Readable Errors | XLS Layers Covered
:-------------: | -------------------------------------------------------------------------------- | --------------- | --------------- | ------------------
**DSLX Fuzzer** | Randomization                                                                    | No              | Not really      | Full compilation pipeline and build-tools. Relatively slow but runs in parallel.
**IR Fuzzer**   | [FuzzTest](https://github.com/google/fuzztest/blob/main/doc/overview.md) Fuzzing | Yes             | Yes             | IR, individual passes or transformations. Significantly faster and the results are more focused on individual components.

## Implementation Details

### High-level Design

Implementation pipeline used by the IR Fuzzer:

1.  [**Protobuf**](https://protobuf.dev/overview/) **Message Template**: Struct
    domain that acts as an object template. Has names and fields used to define
    object attributes. Will contain a list field such that each element in the
    list corresponds to an IR operation. Every IR operation has its own protobuf
    message that acts as an instruction template to instantiate an IR node.
2.  [**FuzzTest**](https://github.com/google/fuzztest/blob/main/doc/overview.md)
    **Randomizer**: Takes in the protobuf message template and generates a
    protobuf object instantiations with randomized values for its attributes. If
    the function requires argument inputs, then arguments will be generated as
    well.
3.  **Generate [IR](https://google.github.io/xls/ir_overview/) Nodes**: Takes in
    the protobuf object and iterates over all of the IR operations. Each
    operation in the protobuf gets instantiated into an IR node object. These IR
    nodes are placed on a small stack VM so that future IR operations may use
    previous IR operations as operands.
4.  **Finalize IR Function**: Use the IR nodes generated by the previous step to
    create a final return value encapsulating the generated computation.
5.  **Perform Tests**: At this point, we have generated a fuzzed XLS function,
    and can perform a variety of tests on it. For example, perform an
    optimization pass on the fuzzed function, plug in random parameters into the
    function before and after optimization, and verify that the interpreted
    results are equal to each other.

### 1\. Protobuf Message Template

#### **High-level Design**

We have chosen to use [protobufs](https://protobuf.dev/overview/) with
[FuzzTest](https://github.com/google/fuzztest/blob/main/doc/overview.md) because
FuzzTest has the capability of
[accepting protobuf message templates](https://github.com/google/fuzztest/blob/main/doc/domains-reference.md#protocol-buffer-domains)
and generating fuzzed object instantiations of the template. This method is
great at generating complex message values to be used as testing inputs. The
main structure of the proto file is as follows
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/fuzz_program.proto)):

```protobuf
message FuzzProgramProto {
  repeated FuzzOpProto fuzz_ops = 3;
}

message FuzzOpProto {
  oneof fuzz_op {
    FuzzParamProto param = 1;
    FuzzShraProto shra = 2;
    FuzzShllProto shll = 3;
    // ...many more ops...
  }
}
```

`FuzzProgramProto` acts as the main object template which contains a repeated
list of `FuzzOpProtos`, where each `FuzzOpProto` represents a template to
instantiate an IR node. So FuzzTest will generate a random amount of `fuzz_ops`,
where each `fuzz_op` contains the instructional information required to generate
an actual IR node.

#### **FuzzOp Design**

Each `fuzz_op` stores information about its operands. There are a variety of
ways to represent its operands due to constraints defined in the protobuf. For
example, some operands must be of a specific type. In order to retrieve an
operand of a specific type, we have to retrieve an IR node in the stack/context
list with the same exact type. However, this is quite rare due to the fact that
all IR nodes are completely random. So to compensate, we also include type
coercion information in the `fuzz_op`, so that we have the instructions required
to coerce any existing IR node into the correct type.

Here is an example of a the add `fuzz_op`:
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/fuzz_program.proto)):

```protobuf
message FuzzAddProto {
  // References a bits operand.
  optional BitsOperandIdxProto lhs_idx = 1;
  optional BitsOperandIdxProto rhs_idx = 2;
  // Coercion instructions to coerce the operands into a specific bits type such
  // that they have the same bit width.
  optional BitsCoercedTypeProto operands_type = 3;
}
```

The `FuzzAddProto` accepts two operands, lhs and rhs. These refer to bits typed
operands because the AddOp only accepts bits typed operands of the same bit
width. To get the operands of the same bit width, we have the
`BitsCoercedTypeProto`, which contains information to coerce a bits type into a
different bits type such that it results in the same bit width. Once we perform
coercion, we end up with two bits operands retrieved off the context list that
have been coerced to be of the same bits type with the same bit width.

Here is an example of a the select `fuzz_op`:
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/fuzz_program.proto)):

```protobuf
message FuzzSelectProto {
  optional BitsOperandIdxProto selector_idx = 1;
  // OperandIdxProto retrieves an operand of any type.
  repeated OperandIdxProto case_idxs = 2;
  optional OperandIdxProto default_value_idx = 3;
  // Specifies the exact type that the cases and default value should be.
  // CoercedTypeProto contains coercion information to coerce any operand into
  // the specified type.
  optional CoercedTypeProto cases_and_default_type = 4;
}
```

The `FuzzSelectProto` accepts a bits type selector operand, a variable amount of
any typed case operands, and an any typed default value operand. The
`CoercedTypeProto` specifies that the cases and default type operand must be of
the same exact type. This means that they must be of the same categorical type,
meaning they must all be bits or they must all be arrays. It also means that
they must all be of the same exact type, so if they were arrays, the array
element type would also have to match.

#### **Coercion Instructions**

Coercion is required when a FuzzOp requires an operand of a specific type, but
the FuzzOp that we actually grab from the context list does not match this type.
In order to get the correctly typed operand, we coerce the grabbed operand into
the correct type. This involves modifying the grabbed operand in several ways.

Coercion only occurs among same categorical types. For example, to coerce a bits
type of u32 to u64, we change its bit width. In the case that they are not of
the same categorical type, we simply return a default value of the desired type.
This avoids complexity and relies on the fuzzers coverage guided ability to
choose interesting inputs.

The following are the bits type coercion methods
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/fuzz_program.proto)):

```protobuf
// Methods used to change the bit width of a bits BValue.
message ChangeBitWidthMethodProto {
  optional DecreaseWidthMethod decrease_width_method = 1;
  optional IncreaseWidthMethod increase_width_method = 2;
}
enum DecreaseWidthMethod {
  UNSET_DECREASE_WIDTH_METHOD = 0;
  BIT_SLICE_METHOD = 1;
}
enum IncreaseWidthMethod {
  UNSET_INCREASE_WIDTH_METHOD = 0;
  ZERO_EXTEND_METHOD = 1;
  SIGN_EXTEND_METHOD = 2;
}
```

The `ChangeBitWidthMethodProto` contains coercion instructions to coerce any
bits type to any other bits type. Note that two increase width methods are
present, that being zero extend and sign extend, in order to add randomness.

The following are the tuple type coercion methods
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/fuzz_program.proto)):

```protobuf
// Methods used to coerce a tuple to a different sized tuple.
message ChangeTupleSizeMethodProto {
  optional DecreaseTupleSizeMethod decrease_size_method = 1;
  optional IncreaseTupleSizeMethod increase_size_method = 2;
}
```

The `ChangeTupleSizeMethodProto` contains coercion instructions to coerce any
tuple type to any other tuple type. This is done by simply expanding or
shrinking the tuple to the specified size. Once the tuple is of the specified
size, we must also coerce its elements to be of the correct type, so we iterate
over the tuple elements and coerce them to the specified type.

#### **Type Instructions**

The XLS IR has 4 different types, that being bits, tuple, array, and token. The
IR Fuzzer deals with all of the types except tokens because they do not add much
to function generation. This is how types are defined
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/fuzz_program.proto)):

```protobuf
message FuzzTypeProto {
  oneof type {
    BitsTypeProto bits = 1;
    TupleTypeProto tuple = 2;
    ArrayTypeProto array = 3;
  }
}
message BitsTypeProto {
  optional int64 bit_width = 1;
}
message TupleTypeProto {
  repeated FuzzTypeProto tuple_elements = 1;
}
message ArrayTypeProto {
  optional int64 array_size = 1;
  optional FuzzTypeProto array_element = 2;
}

message CoercedTypeProto {
  oneof type {
    BitsCoercedTypeProto bits = 1;
    TupleCoercedTypeProto tuple = 2;
    ArrayCoercedTypeProto array = 3;
  }
}
message BitsCoercedTypeProto {
  optional int64 bit_width = 1;
  optional BitsCoercionMethodProto coercion_method = 2;
}
message TupleCoercedTypeProto {
  repeated CoercedTypeProto tuple_elements = 1;
  optional TupleCoercionMethodProto coercion_method = 2;
}
message ArrayCoercedTypeProto {
  optional int64 array_size = 1;
  optional CoercedTypeProto array_element = 2;
  optional ArrayCoercionMethodProto coercion_method = 3;
}
```

There are two different type protos, that being the `FuzzTypeProto` and
`CoercedTypeProto`. Both are pretty much the exact same thing, only that the
`CoercedTypeProto` also contains coercion information specifying how to coerce
any operand into its type. `FuzzTypeProto` is used when we require a fuzzed
type. For example, when creating a `ParamOp`, we require a type to define what
the input type must be for that param. `CoercedTypeProto` is used when we
require a type, but also coercion instructions to coerce an operand into that
type. The helper protobufs like `BitsCoercedTypeProto` can also be used if we
require a bits type instead of just a general type.

### 2\. FuzzTest Randomizer

#### **IR Fuzz Domain**

The IR Fuzz
[Domain](https://github.com/google/fuzztest/blob/main/doc/domains-reference.md)
acts as the main interface for IR FuzzTests. It returns a `FuzzPackage` object
which contains the fuzzed IR function. The following is the implementation of
the `IrFuzzDomain()`
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_domain.cc)):

```c++
struct FuzzPackage {
  std::unique_ptr<Package> p;
  FuzzProgramProto fuzz_program;
};

fuzztest::Domain<FuzzPackage> IrFuzzDomain() {
  return fuzztest::Map(
      [](FuzzProgramProto fuzz_program) {
        // Create the package.
        std::unique_ptr<Package> p =
            std::make_unique<VerifiedPackage>(kFuzzTestName);
        FunctionBuilder fb(kFuzzTestName, p.get());
        // Build the IR from the FuzzProgramProto.
        IrFuzzBuilder ir_fuzz_builder(fuzz_program, p.get(), &fb);
        BValue ir = ir_fuzz_builder.BuildIr();
        CHECK_OK(fb.BuildWithReturnValue(ir));
        // Create the FuzzPackage object.
        return FuzzPackage(std::move(p), fuzz_program);
      },
      // Specify the range of possible values for the FuzzProgramProto protobuf.
      fuzztest::Arbitrary<FuzzProgramProto>()
          .WithStringField("args_bytes",
                           fuzztest::Arbitrary<std::string>().WithMinSize(1000))
          .WithRepeatedProtobufField(
              "fuzz_ops",
              fuzztest::VectorOf(fuzztest::Arbitrary<FuzzOpProto>()
                                     // We want all FuzzOps to be defined.
                                     .WithOneofAlwaysSet("fuzz_op"))
                  // Generate at least one FuzzOp.
                  .WithMinSize(1)));
}
```

`IrFuzzDomain()` first defines the range of possible values that the
instantiated object can have, such as requiring at least one FuzzOp. It then
generates the instantiated `FuzzProgramProto`. It then builds the actual IR
function by generating the IR nodes and finalizing the function return value.

#### **IR Fuzz Domain With Arguments**

We also have a domain that contains the fuzzed function and some arguments that
can be plugged into the function
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_domain.cc)):

```c++
// Same as IrFuzzDomain but returns a FuzzPackageWithArgs domain which also
// contains the argument sets that are compatible with the function.
fuzztest::Domain<FuzzPackageWithArgs> IrFuzzDomainWithArgs(
    int64_t arg_set_count) {
  return fuzztest::Map(
      [arg_set_count](FuzzPackage fuzz_package) {
        return GenArgSetsForPackage(std::move(fuzz_package), arg_set_count);
      },
      IrFuzzDomain());
}
```

`IrFuzzDomainWithArgs()` adds onto `IrFuzzDomain()` by generating an additional
amount of argument sets, where each set is a complete input into the IR
function's parameters.

#### **Argument Generation**

The IR function arguments are generated in a similar way to how the IR nodes are
generated. A bytes field exists in the protobuf called `arg_bytes`, which
essentially generates a string of random byte values. These byte values are
converted into bits data and supplied to the `UnflattenBitsToValue()` function
([link to code](https://github.com/google/xls/blob/main/xls/ir/value_flattening.cc)):

```c++
absl::StatusOr<Value> UnflattenBitsToValue(const Bits& bits, const Type* type);
```

`UnflattenBitsToValue()` generates a value of a specified type, which can be
used as an argument to a function.

### 3\. Generate IR Nodes

#### **High-level Design**

At this stage, we have already fuzzed a `FuzzProgramProto`, and now we have to
iterate through its FuzzOps and generate IR nodes to be placed in the
stack/context list. Accomplishing this task is similar to other compilers, in
the sense that we have to process each FuzzOp individually
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/gen_ir_nodes_pass.h)):

```c++
class GenIrNodesPass : public IrFuzzVisitor {
 public:
  void GenIrNodes();

  // Functions to handle each FuzzOp.
  void HandleParam(const FuzzParamProto& param) override;
  void HandleShra(const FuzzShraProto& shra) override;
  // ...
 private:
  const FuzzProgramProto& fuzz_program_;
  Package* p_;
  FunctionBuilder* fb_;
  // IR nodes are generated from the FuzzOp and placed in this context list.
  IrNodeContextList& context_list_;
}
```

#### **IR Node Generation**

For every IR node, we are given its FuzzOp, and we must generate an IR node from
this FuzzOp. This requires retrieving operands, coercing the operands when
necessary, and making sure operands are in bounds. We are basically writing code
to make sure that no matter how random the FuzzOp is, it can be used to generate
an IR node that resembles the FuzzOp's values.

The following function generates an `AddOp`
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/gen_ir_nodes_pass.cc)):

```c++
void GenIrNodesPass::HandleAdd(const FuzzAddProto& add) {
  BValue lhs = GetCoercedBitsOperand(add.lhs_idx(), add.operands_type());
  BValue rhs = GetCoercedBitsOperand(add.rhs_idx(), add.operands_type());
  context_list_.AppendElement(fb_->Add(lhs, rhs));
}
```

`HandleAdd()` retrieves two operands, lhs and rhs, where both of them are
coerced to a bits type. This is necessary because in order to add two numbers,
they both must be of the same bits type and bit width. We then create an Add IR
node and add it to the context list.

The following function generates a `DecodeOp`
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/gen_ir_nodes_pass.cc)):

```c++
void GenIrNodesPass::HandleDecode(const FuzzDecodeProto& decode) {
  BValue operand = GetBitsOperand(decode.operand_idx());
  // The decode bit width cannot exceed 2 ** operand_bit_width.
  int64_t right_bound = 1000;
  if (operand.BitCountOrDie() < 64) {
    right_bound = std::min<int64_t>(1000, 1ULL << operand.BitCountOrDie());
  }
  int64_t bit_width =
      BoundedWidth(decode.bit_width(), /*left_bound=*/1, right_bound);
  context_list_.AppendElement(fb_->Decode(operand, bit_width));
}
```

`HandleDecode()` retrieves a single bits operand with no coercion necessary. It
then makes sure that the Decode bit width does not exceed 2 ^ operands bit
width, otherwise the operation cannot be performed. It then bounds the width
between 1 and 1000 as the IR Fuzzer only supports bit widths from 1-1000.
Finally, it adds the IR node to the context list.

#### **Context List Design**

The context list is its own data structure. Its similar to a stack VM in the
sense that nodes are placed on it and future nodes may use previous nodes as
operands
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_node_context_list.h)):

```c++
class IrNodeContextList {
 public:
  // List interface functions.
  BValue GetElementAt(int64_t list_idx,
                      ContextListType list_type = COMBINED_LIST) const;
  int64_t GetListSize(ContextListType list_type = COMBINED_LIST) const;
  bool IsEmpty(ContextListType list_type = COMBINED_LIST) const;
  void AppendElement(BValue element);

 private:
  Package* p_;
  FunctionBuilder* fb_;
  // Separate context lists.
  std::vector<BValue> combined_context_list_;
  std::vector<BValue> bits_context_list_;
  std::vector<BValue> tuple_context_list_;
  std::vector<BValue> array_context_list_;
};
```

The context list is made up of 4 separate lists, one for each type and one
combined list. The reason why we have a list for each type is because if we want
to easily retrieve a bits operand, it is easier to do so if we can grab from a
bits list. If we want an operand of any type, we simply retrieve from the
combined list.

#### **IR Fuzzer Type Traversals**

Due to the introduction of types, there is a need to traverse `Type` and
`TypeProto` objects. Due to the recursive nature of arrays and tuples, in that
they can contain sub-types, the type traversal functions must also be recursive.

Various different tasks are performed by type traversal functions:

*   **BValue
    [Coerced](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.cc)(BValue
    bvalue, CoercedTypeProto coerced\_type)** \- Accepts an IR node/bvalue and a
    type that the bvalue should be coerced to. The `CoercedTypeProto` contains
    coercion and type information.
*   **BValue
    [Fitted](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.cc)(BValue
    bvalue, CoercionMethodProto coercion\_method, Type target\_type)** \- Same
    as `Coerced()` but uses a `CoercionMethodProto` and a `Type` rather than a
    `CoercedTypeProto`.
*   **BValue
    [DefaultValueOfType](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.cc)(Type
    type)** \- Returns a default IR node of a specific type.
*   **Type
    [ConvertTypeProtoToType](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_helpers.cc)(TypeProto
    type\_proto)** \- Converts a `TypeProto` like `FuzzTypeProto` or
    `CoercedTypeProto` into a `Type` object.
*   **Value
    [UnflattenBitsToValue](https://github.com/google/xls/blob/main/xls/ir/value_flattening.cc)(Bits
    bits, Type type)** \- Uses bits data to generate a Value object of a
    specified type.

### 4\. Combine IR Nodes

At this stage, we have already filled our entire context list with IR nodes. Now
we must combine the IR nodes together such that a single IR node can represent
the entire context list as a function return value. We want to capture as much
context as possible into this return value.

The following are functions used to combine the context list
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/combine_context_list.cc)):

```c++
// Returns the last element of the combined context list.
BValue LastElement(FunctionBuilder* fb, const IrNodeContextList& context_list) {
  return context_list.GetElementAt(context_list.GetListSize() - 1);
}

// Tuples everything in the combined context list together.
BValue TupleList(FunctionBuilder* fb, const IrNodeContextList& context_list) {
  std::vector<BValue> elements;
  for (int64_t i = 0; i < context_list.GetListSize(); i += 1) {
    elements.push_back(context_list.GetElementAt(i));
  }
  return fb->Tuple(elements);
}
```

The `LastElement()` function simply returns the last element in the combined
context list. This is specifically useful for testing. The `TupleList()`
function tuples all of the elements in the context list together. This method is
preferable as it captures the entire context of this list into a single IR node.

### 5\. Perform Tests

Now that we have generated a fuzzed function and potentially some arguments that
can be plugged into that function, we can find ways to test other parts of the
XLS codebase using this function.

#### **Optimization Pass Changes Outputs Test**

This test takes a fuzzed function, plugs arguments into the function resulting
in a return value, optimizes the function using one of the XLS optimization
passes, plugs the same arguments into the optimized function resulting in
another return value, and compares the two return values, making sure they are
the same. Note that multiple sets of arguments may be used to increase the
chance of at least one of the return values being different. Here is the
implementation of the test
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.cc)):

```c++
void OptimizationPassChangesOutputs(FuzzPackageWithArgs fuzz_package_with_args,
                                    OptimizationPass& pass) {
  std::unique_ptr<Package>& p = fuzz_package_with_args.fuzz_package.p;
  FuzzProgramProto& fuzz_program =
      fuzz_package_with_args.fuzz_package.fuzz_program;
  std::vector<std::vector<Value>>& arg_sets = fuzz_package_with_args.arg_sets;
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, p->GetFunction(kFuzzTestName));
  // Interpret the IR function with the arguments before optimization.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<InterpreterResult<Value>> before_pass_results,
      EvaluateArgSets(f, arg_sets));
  // Run the optimization pass over the IR.
  PassResults results;
  OptimizationContext context;
  XLS_ASSERT_OK_AND_ASSIGN(
      bool ir_changed,
      pass.Run(p.get(), OptimizationPassOptions(), &results, context));
  // Interpret the IR function with the arguments after optimization.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<InterpreterResult<Value>> after_pass_results,
      EvaluateArgSets(f, arg_sets));
  // Check if the results are the same before and after optimization.
  bool results_changed =
      DoResultsChange(before_pass_results, after_pass_results);
  ASSERT_FALSE(results_changed);
}
```

```c++
void IrFuzzReassociation(FuzzPackageWithArgs fuzz_package_with_args) {
  ReassociationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzReassociation)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));
```

The `IrFuzzReassociation` FuzzTest uses `OptimizationPassChangesOutputs()` with
the reassociation optimization pass. We are specifying the use of 10 argument
sets with the `IrFuzzDomainWithArgs()` domain. You can also specify to use
multiple optimization passes, but we have chosen to isolate the fuzzing of each
pass individually.

#### **Verify Fuzz Package**

This test generates a fuzzed IR function and verifies if it is made with correct
values using the
[verifier](https://github.com/google/xls/blob/main/xls/ir/verifier.h). It is
great at finding bugs in the IR Fuzzer due to generating IR nodes with out of
bounds values
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_test.cc)):

```c++
void VerifyIrFuzzPackage(FuzzPackage fuzz_package) {
  std::unique_ptr<Package>& p = fuzz_package.p;
  XLS_ASSERT_OK(VerifyPackage(p.get()));
}
FUZZ_TEST(IrFuzzTest, VerifyIrFuzzPackage).WithDomains(IrFuzzDomain());
```

#### **Proto String Unit Tests**

This test instantiates a `FuzzProgramProto` by defining a proto string directly
in code. We then check if the IR Fuzzer correctly created the correct FuzzOp as
the proto string specifies
([link to code](https://github.com/google/xls/blob/main/xls/fuzzer/ir_fuzzer/ir_fuzz_builder_test.cc)):

```c++
TEST(IrFuzzBuilderTest, AddTwoLiterals) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          param {
            type {
              bits {
                bit_width: 64
              }
            }
          }
        }
      )");
  auto expected_ir_node = m::Param();
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}
```
