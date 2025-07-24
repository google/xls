# XLS: IR Fuzzer

[TOC]

## IR Fuzzer Introduction

Compilers are historically quite hard to debug due to their ability to accept a
large pool of inputs. For example, consider how many different programs you can
write in C++. Each program that is possible is a valid input to the compiler
that could cause a bug. So to catch bugs, we have to use techniques more
complicated than simple unit tests.

Fuzzing is a technique of randomly generating inputs to unit tests to attempt to
find bugs. Fuzzing purposefully chooses inputs that explore more parts of the
program with the intent to find bugs. We plan on using fuzzing to generate a
vast amount of compiler inputs in hopes of breaking code.

To make the compilation pipeline more organized, XLS is split up into several
layers. You start at the DSLX programming language layer, which then gets
compiled into the XLS intermediate representation (IR) layer, which then
eventually becomes Verilog. We plan on generating fuzzed IR because we want to
target bug finding in the IR specifically. Additionally, we will be able to pass
the fuzzed IR to downstream XLS layers in attempts to find bugs in these layers
as well.

## High-level Design

The following is a brief description of the implementation pipeline used for the
IR Fuzzer:

1.  **Protobuf Instruction Template**: Struct domain that acts as an object
    template. Has names and fields. Will contain a list field such that each
    element in the list corresponds to an IR operation such as AddOp. Every IR
    operation has its own protobuf that acts as an instruction template to
    instantiate an IR operation.
2.  **FuzzTest Randomizer**: Takes in the protobuf object template and generates
    a physical protobuf object with randomized values for its fields. The
    protobuf will have a random amount of IR operations.
3.  **Generate IR Nodes**: Takes in the randomized protobuf and iterates over
    all of the operations. Each operation in the protobuf gets instantiated into
    an IR operation node. No matter how random the values for an operation node
    are, it can be instantiated into a valid IR node. We place the instantiated
    IR nodes in a context list because some nodes may reference the use of
    previous nodes in the context list.
4.  **Combine IR Nodes**: Use the context list of instantiated IR nodes and
    combine them into a single IR node object. We are ultimately trying to
    encapsulate as much randomness/context into a single object.
5.  **Perform Tests**: Now that we have a random IR object, we can perform tests
    on it. For example, perform an optimization pass on some generated IR, plug
    in random parameters into the IR function before and after optimization, and
    verify that the interpreted results are equal to each other.

Our goal is to generate IR objects such that they have high random variance
potential. The higher the randomness, the more likely that one of the many
randomly generated objects will fail the test. We want to find as many failing
edge cases as possible because that is the goal of testing. To increase the
randomness, we must generate complex IR node structures with complex node
values.

We are performing a Protobuf to IR conversion because we must add a layer before
the IR in order to generate valid IR operation nodes. This is because
randomizing values in the IR commonly leads to invalid IR. For example, if we
were to create an IR add node, we would have to make sure that its two operands
reference valid existing IR nodes. This check can be avoided by using a context
list along with protobuf operations to simplify the generation process,
ultimately reducing the amount of minor test cases we have to deal with. The
protobuf add operation could reference a random index in the context list to act
as an operand rather than verifying if said index points to a valid IR node.

We are using the GoogleTest FUZZ\_TEST tool because it is able to accomplish our
randomization goals. The tool is designed to randomize protobuf structures, and
that is a reason why we decided to use protobufs as our struct domain.
GoogleTest is also compatible with Google testing infrastructure, making it easy
to integrate normal and continuous testing. The tool also offers verbose logging
of tests.

The biggest advantage of using the FuzzTest tool is its ability to use
randomness and strategy in order to generate values. During testing, FuzzTest
will strategically generate values that attempt to discover edge cases. For
example, if you had a program that breaks only when a value is between 500 and
600, out of the possible 2^31 integers in a 32 bit integer, a normal randomizer
would take forever to discover the bug. FuzzTest, on the other hand, would be
able to identify that the particular if-statement was never being executed, so
it would strategically choose a value to enter into the if-statement. This
strategic randomness allows for better bug hunting in a shorter amount of time.

The IR Fuzzer seeks to improve upon the DSLX Fuzzer that we currently have. The
DSLX Fuzzer is just like the IR Fuzzer, but generates DSLX instead of IR through
randomization rather than FuzzTest. The DSLX Fuzzer works and has found many
bugs, but it is quite hard to debug. The error logs that it gives are not very
helpful. It takes forever to run because it is purely random.

Now that I have explained the overview and goals of the IR Fuzzer, I will
discuss the lower level details of the implementation pipeline with the five
categories listed above.

## 1\. Protobuf Instruction Template

### Protobuf Introduction

A protobuf is an independent file with the .proto extension. It is essentially a
programming language with the very specific task of defining object structures.
It is very similar to defining structs in C++. The advantage of using protobufs
over C++ structs is that it is programming language independent. That means you
can define object structures in a single protobuf file and use that object
structure in most programming languages, avoiding the need to redefine the
object structure in each language. Many other tools have adopted the use of
protobufs for this reason.

The main tool we are using for this project is FuzzTest, which has the ability
to accept a protobuf object structure and generate physical objects with
randomly defined values. For example, if I defined a protobuf named
WeatherPredictor with two int32 fields called humidity\_index and temp\_index,
FuzzTest would accept this protobuf object template and generate a physical
object in C++ with random values, such as
WeatherPredictor(humidity\_index=2131412,temp\_index=-125123).

### High-level Design

For this project, I am using a protobuf file to define the structure of the XLS
IR. It essentially has the capability of generating a list of IR nodes. The main
structure of the proto file is as follows:

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

FuzzProgramProto acts as the main object template which contains a repeated list
of FuzzOpProtos, where each FuzzOpProto represents a template to instantiate an
IR node. So FuzzTest will generate a random amount of fuzz\_ops, where each
fuzz\_op contains the instructional information required to generate an actual
IR node.

When designing this protobuf file, you have to keep in mind that all of the
randomly generated values are purposely chosen by FuzzTest to break the program.
So think of things like empty fields, empty protos, empty enums, max integer
values, negative values, etc. The protobuf to IR compiler has to be able to deal
with all of these extreme/unset values, so the protobuf should be designed in a
way that is simple for a program to accomplish this compiler task.

So while the protobuf file should be representative of the XLS IR, it should not
replicate the IR directly. This is because the IR behaves like a typical
programming language structure, where each IR node uses other IR nodes through
references. If you want to create an IR add node, its lhs and rhs operands must
reference existing IR nodes. These references are hard to keep track of if you
are using randomly generated references via FuzzTest, so instead of using
references, we are using a context list. This context list is just a list that
stores all of the generated IR nodes. These existing IR nodes can then be used
by future IR nodes as operands.

### FuzzOp Design

Each fuzz\_op stores information about its operands. There are a variety of ways
to represent its operands due to constraints defined in the protobuf. For
example, some operands must be of a specific type. In order to retrieve an
operand of a specific type, we have to retrieve an IR node in the context list
with the same exact type. However, this is quite rare due to the fact that all
IR nodes are completely random. So to compensate, we also include type coercion
information in the fuzz\_op, so that we have the instructions required to coerce
any existing IR node into the correct type.

This type constraint only applies to some IR nodes where others are a lot less
restrictive. For example, here are some examples of fuzz\_ops:

```protobuf
message FuzzAddProto {
  // References a bits operand.
  optional BitsOperandIdxProto lhs_idx = 1;
  optional BitsOperandIdxProto rhs_idx = 2;
  // Coercion instructions to coerce the operands into a specific bits type such
  // that they have the same bit width.
  optional BitsCoercedTypeProto operands_type = 3;
}
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

For the FuzzAddProto, it accepts two operands, lhs and rhs. These refer to bits
typed operands because the AddOp only accepts bits typed operands of the same
bit width. To get the operands of the same bit width, we have the
BitsCoercedTypeProto, which contains information to coerce a bits type into a
different bits type such that it results in the same bit width. Once we perform
coercion, we end up with two bits operands retrieved off the context list that
have been coerced to be of the same bits type with the same bit width.

The FuzzSelectProto accepts a bits type selector operand, a variable amount of
any typed case operands, and an any typed default value operand. The
CoercedTypeProto specifies that the cases and default type operand must be of
the same exact type. This means that they must be of the same categorical type,
meaning they must all be bits or they must all be arrays. It also means that
they must all be of the same exact type, so if they were arrays, the array
element type would also have to match.

### Coercion Instructions

Coercion is required when a FuzzOp requires an operand of a specific type, but
the FuzzOp that we actually grab from the context list does not match this type.
In order to get the correctly typed operand, we coerce the grabbed operand into
the correct type. This involves modifying the grabbed operand in several ways.

I have decided to only coerce between same categorical types for simplicity. For
example, if we were to coerce a bits type of u32 to a bits type of u64, we would
simply change the bit width. However, if we wanted to coerce an array type to a
bits type, this coercion is quite subjective. We may choose to take the first
element of the array and coerce that element into a bits type. However, this is
somewhat complicated and tedious for not much gain in terms of randomness. So
instead, in the case of coercion between two different types, we simply place a
default value of the required type. This is much more simple and relies on the
fuzzers ability to avoid boring results.

The following is the actual design for coercion instructions:

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

// Methods used to coerce a tuple to a different sized tuple.
message ChangeTupleSizeMethodProto {
  optional DecreaseTupleSizeMethod decrease_size_method = 1;
  optional IncreaseTupleSizeMethod increase_size_method = 2;
}
enum DecreaseTupleSizeMethod {
  UNSET_DECREASE_TUPLE_SIZE_METHOD = 0;
  SHRINK_TUPLE_METHOD = 1;
}
enum IncreaseTupleSizeMethod {
  UNSET_INCREASE_TUPLE_SIZE_METHOD = 0;
  EXPAND_TUPLE_METHOD = 1;
}
```

The ChangeBitWidthMethodProto contains coercion instructions to coerce any bits
type to any other bits type. Note that two increase width methods are present,
that being zero extend and sign extend, in order to add randomness.

The ChangeTupleSizeMethodProto contains coercion instructions to coerce any
tuple type to any other tuple type. This is done by simply expanding or
shrinking the tuple to the specified size. Once the tuple is of the specified
size, we must also coerce its elements to be of the correct type, so we iterate
over the tuple elements and coerce them to the specified type.

### Type Instructions

The XLS IR has 4 different types, that being bits, tuple, array, and token. The
bits type is simply a list of bits, where the number of bits is defined by its
bit width. The tuple type is a list of individual types, where each element in
the tuple can have a distinct type. The array type is the same as a tuple, but
all of its elements must have the same type. The IR Fuzzer deals with all of the
types except tokens because they do not add much to function generation. This is
how types are defined:

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

There are two different type protos, that being the FuzzTypeProto and
CoercedTypeProto. Both are pretty much the exact same thing, only that the
CoercedTypeProto also contains coercion information specifying how to coerce any
operand into its type. FuzzTypeProto is used when we require a fuzzed type. For
example, when creating a ParamOp, we require a type to define what the input
type must be for that param. CoercedTypeProto is used when we require a type,
but also coercion instructions to coerce an operand into that type. The helper
protobufs like BitsCoercedTypeProto can also be used if we require a bits type
instead of just a general type.

Due to how similar these type instructions are, we can iterate over them using
the same templated C++ functions instead of making a unique function for each
protobuf.

### Combine Context List Instructions

During runtime, the IR Fuzzer iterates over all of the FuzzOps, generates each
one, and then adds them to the context list. After all of the IR nodes are in
the context list, we need a way to return a final IR node that encompasses all
of the IR nodes in the list. To do this, we have several methods listed here:

```protobuf
message FuzzProgramProto {
  optional CombineListMethod combine_list_method = 1;
}

// Specifies the method used to combine the context list of BValues into a
// single IR object.
enum CombineListMethod {
  UNSET_COMBINE_LIST_METHOD = 0;
  LAST_ELEMENT_METHOD = 1;
  TUPLE_LIST_METHOD = 2;
}
```

CombineListMethod is just an enum with several methods associated. The
LAST\_ELEMENT\_METHOD simply returns the last IR node in the context list. This
is useful in testing. The TUPLE\_LIST\_METHOD tuples all of the IR nodes
together, which is a great way of gathering all of the context into a single IR
node.

### Value Generation

Several FuzzOps require physical values, so we must provide value information to
these FuzzOps such that they can use that information in order to generate a
value. For example, the LiteralOp requires a value to be created:

```protobuf
message FuzzLiteralProto {
  optional FuzzTypeProto type = 1;
  // Bytes used to fill the literal with an actual value.
  optional bytes value_bytes = 2;
}
```

The value\_bytes field generates a bytes list, which is basically just a string
of randomly generated characters. These bytes represent binary, which is used to
fill the literal type with bit values.

If we want to generate parameter values, we use an args\_bytes field:

```protobuf
message FuzzProgramProto {
  // Will generate a list of bytes that may be used to create arguments that are
  // compatible with the parameters of the function.
  optional bytes args_bytes = 2;
}
```

This args\_bytes field generates bytes for all ParamOps. I chose to implement
this long bytes list because it is simple to just grab bit data from the list
rather than grabbing bit data from multiple sources. Additionally, a single
ParamOp may generate multiple arguments.

## 2\. FuzzTest Randomizer

### FuzzTest Introduction

FuzzTest is a GoogleTest unit testing tool. Like any other unit test, you
specify physical inputs, you run those inputs against some code, and you verify
if the code works correctly and has the correct output from the processed
inputs. Fuzzing is different from normal unit testing in the sense that the
input values are randomly generated. Randomly generated input values allow you
to have a lot more input coverage, which typically is more likely to discover
bugs. FuzzTest does not use complete randomness to generate inputs, but rather
algorithmically attempts to choose inputs that explore all parts of your code.

FuzzTest also has a feature of being able to generate protobuf inputs. FuzzTest
can be passed a protobuf object and fill all of the fields with fuzzed values.
In the IR Fuzzer, I am providing FuzzTest with the FuzzProgramProto. I then take
this FuzzProgramProto with fuzzed values and generate a bunch of IR nodes to be
placed in the context list.

### FuzzTest Domain Introduction

A domain is a wrapper around any object that includes information about the
range of possible values that an object can have. So a domain can be any object,
with additional instructions specifying how to generate fuzzed versions of that
object. For example, we can create an int64\_t domain, where int64\_t is the
object, and the domain wrapper contains instructions on how to randomly generate
int64\_t values, resulting in fuzztest::Domain\<int64\_t\>. I could specify that
the domain should only fuzz values between 10 and 20\.

Domains are used as the recipe to a FuzzTest. They tell the FuzzTest what
objects to be generating and how they should be fuzzed. Here is an example:

```c++
fuzztest::Domain<FuzzPackage> IrFuzzDomain() {
  // ...
}

void VerifyIrFuzzPackage(FuzzPackage fuzz_package) {
  // ...
}
FUZZ_TEST(IrFuzzTest, VerifyIrFuzzPackage).WithDomains(IrFuzzDomain());
```

FUZZ\_TEST is the actual GoogleTest interfacing function which defines the unit
test. It accepts two parameters. The first parameter is the test suite, just
like any other GoogleTest. The second parameter is a function, which is
VerifyIrFuzzPackage in this example. You then have to specify the domain for the
FuzzTest, which is IrFuzzDomain() in this example. IrFuzzDomain() is a function
that returns a fuzztest::Domain\<FuzzPackage\> object. The VerifyIrFuzzPackage
function requires a FuzzPackage parameter because its parameters represent the
generated object from the domain. Because IrFuzzDomain() returns a FuzzPackage
domain, VerifyIrFuzzPackage must accept a FuzzPackage, which represents the
randomly generated object after fuzzing.

### IR Fuzz Domain

The following is the implementation of the IrFuzzDomain():

```c++
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

The IrFuzzDomain() returns a FuzzPackage object, which contains a Package
object, which contains the randomly generated function. IrFuzzDomain() uses
fuzztest::Map() which accepts two map arguments. The first map argument is a
function which has a parameter that accepts the return value of the second map
argument. The second map argument specifies what object to generate and the
fuzzing instructions on how to generate it.

In this example, the second map argument is a FuzzProgramProto. It has several
fuzzing conditions like how all FuzzOps must be defined and at least one FuzzOp
is generated. The first map argument accepts a FuzzProgramProto for its
parameter, and performs a transformation to convert the FuzzProgramProto into a
FuzzPackage. This FuzzPackage actually contains the useful fuzzed function. This
fuzzed function is the resulting product that we desire out of the IR fuzzer,
and for that reason, we created a domain that returns this object.

Inside of the first map argument function, we actually build the FuzzPackage.
This involves creating a Package and FunctionBuilder object, using the
IrFuzzBuilder to get the IR which represents the function return value, building
the actual function, then creating the FuzzPackage which contains the package
and fuzz program.

### IR Fuzz Domain With Arguments

We also have a domain that contains the fuzzed function and some arguments that
can be plugged into the function:

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

IrFuzzDomainWithArgs() uses IrFuzzDomain() for its second map argument, so we
require that the first map argument function has a FuzzPackage for its
parameter. We then transform the FuzzPackage into a FuzzPackageWithArgs so that
we return the generated arguments with the package.

### Argument Generation

The protobuf provides the args\_bytes field for generating bit data. So to
generate physical arguments, we just have to take that byte data and create
argument values that match the type of the params for the function. We may also
want multiple sets of arguments in order to increase randomness, so we may end
up generating multiple arguments per param.

```c++
FuzzPackageWithArgs GenArgSetsForPackage(FuzzPackage fuzz_package,
                                         int64_t arg_set_count) {
  Function* f = fuzz_package.p->GetFunction(kFuzzTestName).value();
  std::vector<std::vector<Value>> arg_sets;
  std::string args_bytes = fuzz_package.fuzz_program.args_bytes();
  // Convert the args_bytes into a Bits object.
  Bits args_bits = Bits::FromBytes(
      absl::MakeSpan(reinterpret_cast<const uint8_t*>(args_bytes.data()),
                     args_bytes.size()),
      args_bytes.size() * 8);
  int64_t bits_idx = 0;
  // Retrieve arg_set_count amount of arguments for each parameter.
  for (Param* param : f->params()) {
    arg_sets.push_back(
        GenArgsForParam(arg_set_count, param->GetType(), args_bits, bits_idx));
  }
  // ...
  FuzzPackageWithArgs fuzz_package_with_args =
      FuzzPackageWithArgs(std::move(fuzz_package), transposed_arg_sets);
  return fuzz_package_with_args;
}
```

This GenArgSetsForPackage() function retrieves the bits data. For each param, it
will generate arg\_set\_count amount of arguments. Once a bit has been used, the
bits\_idx will skip over it indicating that it cannot be used again for a
different param. The arguments are returned with the package in a
FuzzPackageWithArgs object.

We ultimately decided that the best way to generate fuzzed arguments would be to
define a bytes field in the protobuf and used the generated byte values to
generate arguments. We tried other methods before this, such as using a
fuzztest::FlatMap() to generate the bytes data directly using FuzzTest without
protobuf involvement. This approach wasn’t great because it was overcomplicated.

We also tried using normal randomization through absl::BitGen. This involves
generating random bytes during runtime to produce arguments. The problem with
this is the inability to reproduce randomness using a seed. We thought we could
generate a seed in the protobuf and provide it to the randomizer for reproduced
results, however, absl::BitGen is not seed stable, so even with the same seed,
you get differently generated values for different program executions.

## 3\. Generate IR Nodes

### High-level Design

At this stage, we have already fuzzed a FuzzProgramProto, and now we just have
to iterate through its FuzzOps and generate IR nodes to be placed in the context
list. Accomplishing this task is similar to other compilers, in the sense that
we have to process each Op individually and perform some form of logic. There
may be accompanying data structures as well such as the context list. When
setting up this process, I decided that a visitor pattern would be ideal:

```c++
class IrFuzzVisitor {
 public:
  // These functions correlate to an IR Node.
  virtual void HandleParam(const FuzzParamProto& param) = 0;
  virtual void HandleShra(const FuzzShraProto& shra) = 0;
  // ...
}

class GenIrNodesPass : public IrFuzzVisitor {
 public:
  void GenIrNodes();

  void HandleParam(const FuzzParamProto& param) override;
  void HandleShra(const FuzzShraProto& shra) override;
  // ...
 private:
  const FuzzProgramProto& fuzz_program_;
  Package* p_;
  FunctionBuilder* fb_;
  IrNodeContextList& context_list_;
}
```

Despite GenIrNodesPass being the only class to inherit the visitor, I still
think it’s ideal for organization as it represents the independence of each IR
node.

### IR Node Generation

For every IR node, we are given its FuzzOp, and we must generate an IR node from
this FuzzOp. This requires the retrieval of operands, coercion when necessary,
and bounds checking when necessary. We are basically writing code to make sure
that no matter how random the FuzzOp is, we will be able to generate an IR node
that resembles the FuzzOp values. The following code has two examples.

```c++
void GenIrNodesPass::HandleAdd(const FuzzAddProto& add) {
  BValue lhs = GetCoercedBitsOperand(add.lhs_idx(), add.operands_type());
  BValue rhs = GetCoercedBitsOperand(add.rhs_idx(), add.operands_type());
  context_list_.AppendElement(fb_->Add(lhs, rhs));
}

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

HandleAdd retrieves two operands, lhs and rhs, where both of them are coerced to
a bits type. This is necessary because in order to add two numbers, they both
must be of the same bits type and bit width. We then create an Add IR node and
add it to the context list.

HandleDecode retrieves a single bits operand with no coercion necessary. It then
makes sure that the Decode bit width does not exceed 2 ^ operands bit width,
otherwise the operation cannot be performed. It then bounds the width between 1
and 1000 as the IR Fuzzer only supports bit widths from 1-1000. Finally, it can
add the IR node to the context list.

We must take random inputs and constrain them into valid inputs into the IR node
functions.

### Context List Design

The context list is its own data structure:

```c++
class IrNodeContextList {
 public:
  BValue GetElementAt(int64_t list_idx,
                      ContextListType list_type = COMBINED_LIST) const;
  int64_t GetListSize(ContextListType list_type = COMBINED_LIST) const;
  bool IsEmpty(ContextListType list_type = COMBINED_LIST) const;

  void AppendElement(BValue element);

 private:
  Package* p_;
  FunctionBuilder* fb_;
  std::vector<BValue> combined_context_list_;
  std::vector<BValue> bits_context_list_;
  std::vector<BValue> tuple_context_list_;
  std::vector<BValue> array_context_list_;
};
```

The context list is made up of four separate lists, one for each type and one
combined list. The reason why we have a list for each type is because if we want
to easily retrieve a bits operand, it is easier to do so if we can grab from a
bits list, rather than having to traverse through the combined list and locating
a bits operand. If we want an operand of any type, we simply retrieve from the
combined list. A variety of helper functions are provided to do this
appending/retrieval.

### IR Fuzzer Type Traversals

Due to the introduction of types, there was a need to traverse Type and
TypeProto objects. Due to the recursive nature of arrays and tuples, in that
they can contain sub-types, the type traversal functions must also be recursive.

Various different tasks are performed by type traversal functions:

*   **BValue Coerced(BValue bvalue, CoercedTypeProto coerced_type)** \- Accepts
    an IR node/bvalue and a type that the bvalue should be coerced to. The
    CoercedTypeProto contains coercion and type information. There is more
    information about this function below.
*   **BValue Fitted(BValue bvalue, CoercionMethodProto coercion_method, Type
    target_type)** \- Same as Coerced() but uses a CoercionMethodProto and a
    Type rather than a CoercedTypeProto.
*   **BValue DefaultValueOfType(Type type)** \- Returns a default IR node of a
    specific type.
*   **Type ConvertTypeProtoToType(TypeProto type_proto)** \- Converts a
    TypeProto like FuzzTypeProto or CoercedTypeProto into a Type object.
*   **Value UnflattenBitsToValue(Bits bits, Type type)** \- Located in
    ir/value\_flattening.h. Uses bits data to generate a Value object of a
    specified type.

### Operand Coercion

When an operand is not of the specified type, it must be coerced such that it is
of the correct type. The following is the Coerced() type traversal function
(don’t read it too hard, just understand the structure):

```c++
BValue Coerced(Package* p, FunctionBuilder* fb, BValue bvalue,
               const CoercedTypeProto& coerced_type, Type* target_type) {
  switch (coerced_type.type_case()) {
    case CoercedTypeProto::kBits:
      return CoercedBits(p, fb, bvalue, coerced_type.bits(), target_type);
    case CoercedTypeProto::kTuple:
      return CoercedTuple(p, fb, bvalue, coerced_type.tuple(), target_type);
    case CoercedTypeProto::kArray:
      return CoercedArray(p, fb, bvalue, coerced_type.array(), target_type);
    default:
      return DefaultValue(p, fb);
  }
}

BValue CoercedBits(Package* p, FunctionBuilder* fb, BValue bvalue,
                   const BitsCoercedTypeProto& coerced_type,
                   Type* target_type) {
  Type* bvalue_type = bvalue.GetType();
  // If the bvalue is already the specified type, return it as is.
  if (bvalue_type == target_type) {
    return bvalue;
  }
  // If the bvalue differs in categorical type to bits, simply return a default
  // value of the specified type.
  if (!bvalue_type->IsBits()) {
    return DefaultValueOfBitsType(p, fb, target_type);
  }
  BitsType* bits_type = target_type->AsBitsOrDie();
  auto coercion_method = coerced_type.coercion_method();
  // Change the bit width to the specified bit width.
  return ChangeBitWidth(fb, bvalue, bits_type->bit_count(),
                        coercion_method.change_bit_width_method());
}

BValue CoercedTuple(Package* p, FunctionBuilder* fb, BValue bvalue,
                    const TupleCoercedTypeProto& coerced_type,
                    Type* target_type) {
  Type* bvalue_type = bvalue.GetType();
  if (bvalue_type == target_type) {
    return bvalue;
  }
  if (!bvalue_type->IsTuple()) {
    return DefaultValueOfTupleType(p, fb, target_type);
  }
  TupleType* tuple_type = target_type->AsTupleOrDie();
  auto coercion_method = coerced_type.coercion_method();
  // Change the size of the tuple to match the specified size.
  bvalue = ChangeTupleSize(fb, bvalue, tuple_type->size(),
                           coercion_method.change_list_size_method());
  std::vector<BValue> coerced_elements;
  // Coerce each tuple element and create a new tuple with the coerced
  // elements.
  for (int64_t i = 0; i < tuple_type->size(); i += 1) {
    BValue element = fb->TupleIndex(bvalue, i);
    // Recursive call.
    BValue coerced_element =
        Coerced(p, fb, element, coerced_type.tuple_elements(i),
                tuple_type->element_type(i));
    coerced_elements.push_back(coerced_element);
  }
  return fb->Tuple(coerced_elements);
}
```

We start at the Coerced() function, which accepts a bvalue representing the
operand. It also requires a CoercedTypeProto which contains the coercion
instructions. It also requires a Type\* object to specify the type in which the
bvalue should be coerced to. First, we simply determine which type the bvalue is
and call the associated helper function.

The CoercedBits() function first checks if bvalue is supposed to be a bits type.
If it is not, then it will just return a default value. This is a situation
where we are trying to coerce between different categorical types, so we use a
default value. However, if we expect the bvalue to be of that type, then we
perform same categorical type coercion. In this case, we simply change the bit
width of to match that of the target type.

The CoercedTuple() function pretty much does the same thing as the CoercedBits()
function with a few differences. For example, instead of changing the bit width,
we change the tuple size to match that of the target type. If we end up
extending the tuple, we will fill its extended elements with default values.
Additionally, we must coerce each of its elements to also have to correct tuple
element type. This introduces the recursion I was talking about. So we simply
call the Coerced() function on each of its elements.

## 4\. Combine IR Nodes

At this stage, we have already filled our entire context list with IR nodes. Now
we must combine the IR nodes together such that a single IR node can represent
the entire context list. It is important to capture all of the context into a
single IR node because this IR node acts as the return value for the function.
We want to capture as much randomness as possible into this return value.

The following are functions used to combine the context list:

```c++
BValue CombineContextList(const FuzzProgramProto& fuzz_program,
                          FunctionBuilder* fb,
                          const IrNodeContextList& context_list) {
  switch (fuzz_program.combine_list_method()) {
    case CombineListMethod::TUPLE_LIST_METHOD:
      return TupleList(fb, context_list);
    case CombineListMethod::LAST_ELEMENT_METHOD:
    default:
      return LastElement(fb, context_list);
  }
}

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

The CombineListMethod protobuf field defines which list combining method to use.
The LastElement() function simply returns the last element in the combined
context list. This is specifically useful for testing. The TupleList() functions
tuples all of the elements in the context list together. This method is
preferable as it captures the entire context of this list into a single IR node.

## 5\. Perform Tests

Now that we have generated a fuzzed function and potentially some arguments that
can be plugged into that function, we can find ways to test other parts of the
XLS codebase using this function.

### Optimization Pass Changes Outputs Test

This test takes a fuzzed function, plugs arguments into the function resulting
in a return value, optimizes the function using one of the XLS optimization
passes, plugs the same arguments into the optimized function resulting in
another return value, and compares the two return values, making sure they are
the same. Note that multiple sets of arguments may be used to increase the
chance of at least one of the return values being different. The IR Fuzzer was
basically built to perform this specific test. The following code is the test:

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

void IrFuzzReassociation(FuzzPackageWithArgs fuzz_package_with_args) {
  ReassociationPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzReassociation)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));
```

The FuzzPackageWithArgs struct contains the Package object which contains the
fuzzed Function object. It also contains the argument sets.

The IrFuzzReassociation FuzzTest uses OptimizationPassChangesOutputs with the
reassociation optimization pass. We are specifying the use of 10 argument sets.
You can also specify to use multiple optimization passes, but we have chosen to
isolate the fuzzing of each pass individually.

### Verify Fuzz Package

This test simply creates the fuzz package and verifies if it is made correctly.
This verification is done by the pre-existing XLS verifier. The verifier makes
sure that all operations are using the correct type, correctly bounded values,
correct amount of operands, etc. This test is great at finding bugs in the IR
node generator as in development, it is common to accidentally create invalid
operations:

```c++
void VerifyIrFuzzPackage(FuzzPackage fuzz_package) {
  std::unique_ptr<Package>& p = fuzz_package.p;
  XLS_ASSERT_OK(VerifyPackage(p.get()));
}
FUZZ_TEST(IrFuzzTest, VerifyIrFuzzPackage).WithDomains(IrFuzzDomain());
```

### Proto String Unit Tests

While the Verify Fuzz Package test is great at finding random IR fuzzer bugs
using the fuzzer directly, normal unit tests with no fuzzing at all are good for
test driven development and finding edge cases that are difficult for the fuzzer
to find. These unit tests rely on defining the protobuf in a string format,
building the IR from this proto string, and performing some sort of test on the
generated function. Because the proto string is used to define the protobuf, no
fuzzing actually occurs here. The following is a proto string unit test:

```c++
absl::Status EquateProtoToIrTest(
    std::string proto_string, testing::Matcher<const Node*> expected_ir_node) {
  XLS_ASSIGN_OR_RETURN(FuzzPackage fuzz_package,
                       BuildPackageFromProtoString(proto_string));
  std::unique_ptr<Package>& p = fuzz_package.p;
  XLS_ASSIGN_OR_RETURN(Function * f, p->GetFunction(kFuzzTestName));
  // Verify that the proto_ir_node matches the expected_ir_node.
  EXPECT_THAT(f->return_value(), expected_ir_node);
  return absl::OkStatus();
}

TEST(IrFuzzBuilderTest, AddTwoLiterals) {
  std::string proto_string = absl::StrFormat(
      R"(
        combine_list_method: LAST_ELEMENT_METHOD
        fuzz_ops {
          literal {
            type {
              bits {
                bit_width: 64
              }
            }
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          literal {
            type {
              bits {
                bit_width: 64
              }
            }
            value_bytes: "\x%x"
          }
        }
        fuzz_ops {
          add {
            lhs_idx {
              list_idx: 0
            }
            rhs_idx {
              list_idx: 1
            }
            operands_type {
              bit_width: 64
            }
          }
        }
      )",
      10, 20);
  auto expected_ir_node =
      m::Add(m::Literal(UBits(10, 64)), m::Literal(UBits(20, 64)));
  XLS_ASSERT_OK(EquateProtoToIrTest(proto_string, expected_ir_node));
}
```

The EquateProtoToIrTest function accepts a proto string. This string is built
into a function. We then verify that the function return value matches that of
the expected\_ir\_node.

The AddTwoLiterals unit test defines the proto string. Three fuzz ops are
present, that being two literals and an AddOp that adds the two literals
together. The combine\_list\_method is set to use the last element as the return
value. So when defining expected\_ir\_node, we expect it to be an add IR node
with two literal IR node operands, one of which has the value 10 and the other
with a value of 20\.
