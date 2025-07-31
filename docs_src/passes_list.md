<!-- Generated file. Do not edit. -->
<!-- To regenerate build `xls/passes/tools:rebuild_documentation` -->
# Optimization Passes List

<!-- Don't bother formatting this. -->
<!-- mdformat global-off -->

This is an automatically generated list of all optimization passes available
with the default `opt_main`. This is generated automatically based on comments
in the header files.

If the opt level is set below 'Min opt level' the pass will act as a no-op.

If the opt level is set above 'Cap opt level' the pass (or passes within the
compound pass) will be executed with the opt level capped to the specified
value.


## default_pipeline - The default pipeline. {#default_pipeline}



[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [simplify-and-inline](#simplify-and-inline)
- [post-inlining](#post-inlining)



## arith_simp - Arithmetic Simplifications {#arith_simp}


This pass performs various arithmetic optimizations such as replacement of divide by a constant with non-divide operations.


[Header](http://github.com/google/xls/tree/main/xls/passes/arith_simplification_pass.h)






## array_simp - Array Simplification {#array_simp}


Pass which simplifies or eliminates some array-type operations such as ArrayIndex.


[Header](http://github.com/google/xls/tree/main/xls/passes/array_simplification_pass.h)






## array_untuple - Array UnTuple {#array_untuple}


Pass which changes any (non-external) array-of-tuple into a tuple-of-arrays. We can see through tuples quite well but can't see through arrays to anywhere near the same extent. Therefore the struct-of-array representation is always superior.

 Note that this pass makes no attempt to unpack or repack arrays which escape the function-base. This means that anything which comes in through a function param, or a procs recv or escapes through a function return or a proc send is not untuple'd. TODO(allight): We could do this at the cost of a significant number of ir nodes. We should experiment to see if this is worth doing.


[Header](http://github.com/google/xls/tree/main/xls/passes/array_untuple_pass.h)






## basic_simp - Basic Simplifications {#basic_simp}


This pass does simple pattern-matching optimizations which are ~always a good idea to do (replacing a node with a constant, removing operands of nodes, etc). They improve QoR, do not increase the number of nodes in the graph, preserve the same abstraction level, and do not impede later optimizations via obfuscation. These optimizations require no analyses beyond looking at the node and its operands. Examples include: not(not(x)) => x, x + 0 => x, etc.


[Header](http://github.com/google/xls/tree/main/xls/passes/basic_simplification_pass.h)






## bdd_cse - BDD-based Common Subexpression Elimination {#bdd_cse}


Pass which commons equivalent expressions in the graph using binary decision diagrams.


[Header](http://github.com/google/xls/tree/main/xls/passes/bdd_cse_pass.h)






## bdd_simp - BDD-based Simplification {#bdd_simp}


Runs BDD-based simplifications on the function. Currently this is a very limited set of optimization including one-hot removal and replacement of statically known values with literals. TODO(meheff): Add more BDD-based optimizations.


[Header](http://github.com/google/xls/tree/main/xls/passes/bdd_simplification_pass.h)






## bdd_simp(2) - BDD-based Simplification with opt_level <= 2 {#bdd_simp2}


Runs BDD-based simplifications on the function. Currently this is a very limited set of optimization including one-hot removal and replacement of statically known values with literals. TODO(meheff): Add more BDD-based optimizations.


[Header](http://github.com/google/xls/tree/main/xls/passes/bdd_simplification_pass.h)






## bdd_simp(3) - BDD-based Simplification with opt_level <= 3 {#bdd_simp3}


Runs BDD-based simplifications on the function. Currently this is a very limited set of optimization including one-hot removal and replacement of statically known values with literals. TODO(meheff): Add more BDD-based optimizations.


[Header](http://github.com/google/xls/tree/main/xls/passes/bdd_simplification_pass.h)






## bitslice_simp - Bit-slice simplification {#bitslice_simp}


Pass which simplifies bit-slices. This includes collapsing sequential bit-slices, eliminating degenerate full-width slices, and others.


[Header](http://github.com/google/xls/tree/main/xls/passes/bit_slice_simplification_pass.h)






## bool_simp - boolean simplification {#bool_simp}


Attempts to simplify bitwise / boolean expressions (e.g. of multiple variables).


[Header](http://github.com/google/xls/tree/main/xls/passes/boolean_simplification_pass.h)






## canon - Canonicalization {#canon}


class CanonicalizationPass iterates over nodes and tries to canonicalize the expressions found. For example, for an add between a node and a literal, the literal should only be the 2nd operand. This preprocessing of the IR helps to simplify later passes.


[Header](http://github.com/google/xls/tree/main/xls/passes/canonicalization_pass.h)






## channel_legalization - Legalize multiple send/recvs per channel {#channel_legalization}


Pass that legalizes multiple send/receive operations per channel.

 This pass adds cross-activation tokens to guarantee that later activations of a proc cannot send or receive on a channel until all previous activations have completed working with that channel.


[Header](http://github.com/google/xls/tree/main/xls/passes/channel_legalization_pass.h)






## comparison_simp - Comparison Simplification {#comparison_simp}


Simplifies logical operations on the results of comparison operations. For example:

   eq(x, 0) && ne(x, 1) => eq(x, 0)   eq(x, 0) && ne(x, 0) => 0   eq(x, 0) || ne(x, 0) => 1


[Header](http://github.com/google/xls/tree/main/xls/passes/comparison_simplification_pass.h)






## concat_simp - Concat simplification {#concat_simp}


Pass which simplifies concats. This includes removing single-operand concats, flattening trees of dependent concats, and others.


[Header](http://github.com/google/xls/tree/main/xls/passes/concat_simplification_pass.h)






## cond_spec(Bdd) - Conditional specialization {#cond_specBdd}


Pass which specializes arms of select operations based on their selector value.


[Header](http://github.com/google/xls/tree/main/xls/passes/conditional_specialization_pass.h)






## cond_spec(false) - Conditional specialization {#cond_specfalse}


Pass which specializes arms of select operations based on their selector value.


[Header](http://github.com/google/xls/tree/main/xls/passes/conditional_specialization_pass.h)






## cond_spec(noBdd) - Conditional specialization {#cond_specnoBdd}


Pass which specializes arms of select operations based on their selector value.


[Header](http://github.com/google/xls/tree/main/xls/passes/conditional_specialization_pass.h)






## cond_spec(true) - Conditional specialization {#cond_spectrue}


Pass which specializes arms of select operations based on their selector value.


[Header](http://github.com/google/xls/tree/main/xls/passes/conditional_specialization_pass.h)






## const_fold - Constant folding {#const_fold}


Pass which performs constant folding. Every op with only literal operands is replaced by a equivalent literal. Runs DCE after constant folding.


[Header](http://github.com/google/xls/tree/main/xls/passes/constant_folding_pass.h)






## cse - Common subexpression elimination {#cse}


Pass which performs common subexpression elimination. Equivalent ops with the same operands are commoned. The pass can find arbitrarily large common expressions.


[Header](http://github.com/google/xls/tree/main/xls/passes/cse_pass.h)






## dataflow - Dataflow Optimization {#dataflow}


An optimization which uses a lattice-based dataflow analysis to find equivalent nodes in the graph and replace them with a simpler form. The analysis traces through tuples, arrays, and select operations. Optimizations which can be performed by this pass:

    tuple_index(tuple(x, y), index=1)  =>  y

    select(selector, {z, z})  =>  z

    array_index(array_update(A, x, index={42}), index={42})  =>  x


[Header](http://github.com/google/xls/tree/main/xls/passes/dataflow_simplification_pass.h)






## dce - Dead Code Elimination {#dce}


class DeadCodeEliminationPass iterates up from a functions result nodes and marks all visited node. After that, all unvisited nodes are considered dead.


[Header](http://github.com/google/xls/tree/main/xls/passes/dce_pass.h)






## dfe - Dead Function Elimination {#dfe}


This pass removes unreachable procs/blocks/functions from the package. The pass requires `top` be set in order remove any constructs.


[Header](http://github.com/google/xls/tree/main/xls/passes/dfe_pass.h)






## fixedpoint_proc_state_flattening - Proc State Flattening {#fixedpoint_proc_state_flattening}


Prepare proc state for further analysis by removing arrays and tuples.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set


Run to a fixedpoint.








### Invoked Passes


- [proc_state_array_flat](#proc_state_array_flat)
- [proc_state_tuple_flat](#proc_state_tuple_flat)



## fixedpoint_simp - Fixed-point Simplification {#fixedpoint_simp}


Standard simplification pipeline.

This is run to a fixedpoint and avoids many time-consuming analyses.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set


Run to a fixedpoint.








### Invoked Passes


- [simp](#simp)



## fixedpoint_simp(2) - Max-2 Fixed-point Simplification {#fixedpoint_simp2}


Standard simplification pipeline.

Opt level is capped at 2


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set


Run to a fixedpoint.





Cap opt level: 2




### Invoked Passes


- [simp](#simp)



## fixedpoint_simp(3) - Max-3 Fixed-point Simplification {#fixedpoint_simp3}


Standard simplification pipeline.

Opt level is capped at 3


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set


Run to a fixedpoint.





Cap opt level: 3




### Invoked Passes


- [simp](#simp)



## fixedpoint_simp(>=1,<=2) - Min-1 Max-2 Fixedpoint Simplification {#fixedpoint_simp12}


Standard simplification pipeline.

Opt level is capped at 2 and skipped if less than 1


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set


Run to a fixedpoint.



Min opt level: 1



Cap opt level: 2




### Invoked Passes


- [simp](#simp)



## full-inlining - full function inlining passes {#full-inlining}


Fully inline all functions in a single step.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [loop_unroll](#loop_unroll)
- [map_inlining](#map_inlining)
- [inlining](#inlining)
- [dfe](#dfe)



## ident_remove - Identity Removal {#ident_remove}


class IdentityRemovalPass eliminates all identity() expressions by forward substituting it's parameters to the uses of the identity's def.


[Header](http://github.com/google/xls/tree/main/xls/passes/identity_removal_pass.h)






## inlining - Inlines invocations {#inlining}


Inlines a package toward the `top` function/proc.

 If `full` then all functions are inlined into the `top`.

 If `leaf` then only leaf functions are inlined into their caller. This allows other passes to optimize on smaller graphs.


[Header](http://github.com/google/xls/tree/main/xls/passes/inlining_pass.h)






## label-recovery - LabelRecovery {#label-recovery}


At the end of the pass pipeline (when inlining and optimizations have been performed) attempts to recover original names for coverpoints and assertions to whatever degree possible so they're more human-readable -- we mangle them for inlining to ensure they're unique, but often those names are way overqualified.


[Header](http://github.com/google/xls/tree/main/xls/passes/label_recovery_pass.h)






## leaf-inlining - Inlines invocations {#leaf-inlining}


Inlines a package toward the `top` function/proc.

 If `full` then all functions are inlined into the `top`.

 If `leaf` then only leaf functions are inlined into their caller. This allows other passes to optimize on smaller graphs.


[Header](http://github.com/google/xls/tree/main/xls/passes/inlining_pass.h)






## loop_unroll - Unroll counted loops {#loop_unroll}



[Header](http://github.com/google/xls/tree/main/xls/passes/unroll_pass.h)






## lut_conversion - LUT Conversion {#lut_conversion}


Pass which opportunistically converts nodes to lookup tables (selects) where we can prove it's beneficial.


[Header](http://github.com/google/xls/tree/main/xls/passes/lut_conversion_pass.h)






## map_inlining - Inline map operations {#map_inlining}


A pass to convert map nodes to in-line Invoke nodes. We don't directly lower maps to Verilog.


[Header](http://github.com/google/xls/tree/main/xls/passes/map_inlining_pass.h)






## narrow - Narrowing {#narrow}


A pass which reduces the width of operations eliminating redundant or unused bits.


[Header](http://github.com/google/xls/tree/main/xls/passes/narrowing_pass.h)






## narrow(Context) - Narrowing {#narrowContext}


A pass which reduces the width of operations eliminating redundant or unused bits.


[Header](http://github.com/google/xls/tree/main/xls/passes/narrowing_pass.h)






## narrow(OptionalContext) - Narrowing {#narrowOptionalContext}


A pass which reduces the width of operations eliminating redundant or unused bits.


[Header](http://github.com/google/xls/tree/main/xls/passes/narrowing_pass.h)






## narrow(Range) - Narrowing {#narrowRange}


A pass which reduces the width of operations eliminating redundant or unused bits.


[Header](http://github.com/google/xls/tree/main/xls/passes/narrowing_pass.h)






## narrow(Ternary) - Narrowing {#narrowTernary}


A pass which reduces the width of operations eliminating redundant or unused bits.


[Header](http://github.com/google/xls/tree/main/xls/passes/narrowing_pass.h)






## next_value_opt - Next Value Optimization {#next_value_opt}


Pass which tries to optimize `next_value` nodes.

 Optimizations include: - removing literal predicates on `next_value` nodes (removing the   `next_value` node if dead), - splitting `next_value` nodes with `select`-based values (if small), - splitting `next_value` nodes with `priority_sel`-based values, and - splitting `next_value` nodes with `one_hot_sel`-based values (where safe).

 For best results, first modernizes old-style values on `next (...)` lines, converting them to `next_value` nodes.


[Header](http://github.com/google/xls/tree/main/xls/passes/next_value_optimization_pass.h)






## next_value_opt(3) - max-3 next value optimization {#next_value_opt3}


Next value opt capped at 3


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set






Cap opt level: 3




### Invoked Passes


- [next_value_opt](#next_value_opt)



## one-leaf-inlining - leaf function inlining passes {#one-leaf-inlining}


inline one level of functions.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [loop_unroll](#loop_unroll)
- [map_inlining](#map_inlining)
- [leaf-inlining](#leaf-inlining)
- [dfe](#dfe)



## post-inlining - Post-inlining passes {#post-inlining}


Passes performed after inlining


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [post-inlining-opt(>=1)](#post-inlining-opt1)
- [dce](#dce)
- [label-recovery](#label-recovery)
- [resource_sharing](#resource_sharing)



## post-inlining-opt - post-inlining optimization passes {#post-inlining-opt}


Passes performed after inlining


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [fixedpoint_simp(2)](#fixedpoint_simp2)
- [bdd_simp(2)](#bdd_simp2)
- [dce](#dce)
- [bdd_cse](#bdd_cse)
- [dce](#dce)
- [cond_spec(Bdd)](#cond_specBdd)
- [dce](#dce)
- [fixedpoint_simp(2)](#fixedpoint_simp2)
- [narrow(OptionalContext)](#narrowOptionalContext)
- [dce](#dce)
- [basic_simp](#basic_simp)
- [dce](#dce)
- [arith_simp](#arith_simp)
- [dce](#dce)
- [cse](#cse)
- [sparsify_select](#sparsify_select)
- [dce](#dce)
- [useless_assert_remove](#useless_assert_remove)
- [ram_rewrite](#ram_rewrite)
- [useless_io_remove](#useless_io_remove)
- [dce](#dce)
- [cond_spec(Bdd)](#cond_specBdd)
- [channel_legalization](#channel_legalization)
- [token_dependency](#token_dependency)
- [fixedpoint_simp(2)](#fixedpoint_simp2)
- [fixedpoint_proc_state_flattening](#fixedpoint_proc_state_flattening)
- [proc_state_bits_shatter](#proc_state_bits_shatter)
- [proc_state_tuple_flat](#proc_state_tuple_flat)
- [ident_remove](#ident_remove)
- [dataflow](#dataflow)
- [next_value_opt(3)](#next_value_opt3)
- [dce](#dce)
- [proc_state_narrow](#proc_state_narrow)
- [dce](#dce)
- [proc_state_opt](#proc_state_opt)
- [dce](#dce)
- [proc_state_provenance_narrow](#proc_state_provenance_narrow)
- [dce](#dce)
- [proc_state_opt](#proc_state_opt)
- [dce](#dce)
- [bdd_simp(3)](#bdd_simp3)
- [dce](#dce)
- [bdd_cse](#bdd_cse)
- [select_lifting](#select_lifting)
- [dce](#dce)
- [lut_conversion](#lut_conversion)
- [dce](#dce)
- [cond_spec(Bdd)](#cond_specBdd)
- [dce](#dce)
- [fixedpoint_simp(3)](#fixedpoint_simp3)
- [select_range_simp](#select_range_simp)
- [dce](#dce)
- [fixedpoint_simp(3)](#fixedpoint_simp3)
- [bdd_simp(3)](#bdd_simp3)
- [dce](#dce)
- [bdd_cse](#bdd_cse)
- [dce](#dce)
- [proc_state_bits_shatter](#proc_state_bits_shatter)
- [proc_state_tuple_flat](#proc_state_tuple_flat)
- [fixedpoint_simp(3)](#fixedpoint_simp3)
- [useless_assert_remove](#useless_assert_remove)
- [useless_io_remove](#useless_io_remove)
- [next_value_opt(3)](#next_value_opt3)
- [proc_state_opt](#proc_state_opt)
- [dce](#dce)
- [cond_spec(Bdd)](#cond_specBdd)
- [dce](#dce)
- [select_merge](#select_merge)
- [dce](#dce)
- [fixedpoint_simp(3)](#fixedpoint_simp3)



## post-inlining-opt(>=1) - min-1 post-inlining optimization passes {#post-inlining-opt1}


Passes performed after inlining


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set




Min opt level: 1






### Invoked Passes


- [post-inlining-opt](#post-inlining-opt)



## pre-inlining - pre-inlining passes {#pre-inlining}


Passes performed before each inlining.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [dfe](#dfe)
- [dce](#dce)
- [simp(>=1,<=2)](#simp12)



## proc_state_array_flat - Proc State Array Flattening {#proc_state_array_flat}


Pass which flattens array elements of the proc state into their constituent elements. Tuples are flattened in a different pass. Flattening improves optimizability because each state element can be considered and transformed in isolation. Flattening also gives the scheduler more flexibility; without flattening, each element in the aggregate must have the same lifetime.


[Header](http://github.com/google/xls/tree/main/xls/passes/proc_state_array_flattening_pass.h)






## proc_state_bits_shatter - Proc State Bits Shattering {#proc_state_bits_shatter}


Pass which transforms Bits-type elements of the proc state into tuples of components. Only flattens where it can show that doing so will enable dynamic state feedback opportunities later (assuming it's followed by a ProcStateTupleFlatteningPass).


[Header](http://github.com/google/xls/tree/main/xls/passes/proc_state_bits_shattering_pass.h)






## proc_state_narrow - Proc State Narrowing {#proc_state_narrow}


Pass which tries to minimize the size and total number of elements of the proc state.  The optimizations include removal of dead state elements and zero-width elements.


[Header](http://github.com/google/xls/tree/main/xls/passes/proc_state_narrowing_pass.h)






## proc_state_opt - Proc State Optimization {#proc_state_opt}


Pass which tries to minimize the size and total number of elements of the proc state.  The optimizations include removal of dead state elements and zero-width elements.


[Header](http://github.com/google/xls/tree/main/xls/passes/proc_state_optimization_pass.h)






## proc_state_provenance_narrow - Proc State Provenance Narrowing {#proc_state_provenance_narrow}


Pass which tries to minimize the size and total number of elements of the proc state. This pass works by examining the provenance of the bits making up the next value to determine which (if any) bits are never actually modified.

 NB This is a separate pass from ProcStateNarrowing for simplicity of implementation. That pass mostly assumes we'll have a range-analysis which this does not need.


[Header](http://github.com/google/xls/tree/main/xls/passes/proc_state_provenance_narrowing_pass.h)






## proc_state_tuple_flat - Proc State Tuple Flattening {#proc_state_tuple_flat}


Pass which flattens tuple elements of the proc state into their constituent components. Array elements are flattened in a different pass. Flattening improves optimizability because each state element can be considered and transformed in isolation. Flattening also gives the scheduler more flexibility; without flattening, each element in the aggregate must have the same lifetime.


[Header](http://github.com/google/xls/tree/main/xls/passes/proc_state_tuple_flattening_pass.h)






## ram_rewrite - RAM Rewrite {#ram_rewrite}


Pass that rewrites RAMs of one type to a new type. Generally this will be some kind of lowering from more abstract to concrete RAMs.


[Header](http://github.com/google/xls/tree/main/xls/passes/ram_rewrite_pass.h)






## reassociation - Reassociation {#reassociation}


Reassociates associative operations to reduce delay by transforming chains of operations to a balanced tree of operations, and gathering together constants in the expression for folding.


[Header](http://github.com/google/xls/tree/main/xls/passes/reassociation_pass.h)






## recv_default - Receive default value simplification {#recv_default}


Optimization which removes useless selects between the data value of a conditional or non-blocking receive and the default value of the receive (all zeros).


[Header](http://github.com/google/xls/tree/main/xls/passes/receive_default_value_simplification_pass.h)






## resource_sharing - Resource Sharing {#resource_sharing}



[Header](http://github.com/google/xls/tree/main/xls/passes/resource_sharing_pass.h)






## select_lifting - Select Lifting {#select_lifting}


Pass which replace the pattern   v = sel (c, array[i], array[j], ...) where all cases of the select reference the same array, to the code   z = sel (c, i, j, ...)   v = array[z]


[Header](http://github.com/google/xls/tree/main/xls/passes/select_lifting_pass.h)






## select_merge - Select Merging {#select_merge}



[Header](http://github.com/google/xls/tree/main/xls/passes/select_merging_pass.h)






## select_range_simp - Select Range Simplification {#select_range_simp}


Pass which simplifies selects and one-hot-selects. Example optimizations include removing dead arms and eliminating selects with constant selectors. Uses range analysis to determine possible values.


[Header](http://github.com/google/xls/tree/main/xls/passes/select_simplification_pass.h)






## select_simp - Select Simplification {#select_simp}


Pass which simplifies selects and one-hot-selects. Example optimizations include removing dead arms and eliminating selects with constant selectors. Uses ternary analysis to determine possible values.


[Header](http://github.com/google/xls/tree/main/xls/passes/select_simplification_pass.h)






## simp - Simplification {#simp}


Standard simplification pipeline.

This is run a large number of times and avoids many time-consuming analyses.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)




### Invoked Passes


- [ident_remove](#ident_remove)
- [const_fold](#const_fold)
- [dce](#dce)
- [canon](#canon)
- [dce](#dce)
- [basic_simp](#basic_simp)
- [dce](#dce)
- [arith_simp](#arith_simp)
- [dce](#dce)
- [comparison_simp](#comparison_simp)
- [dce](#dce)
- [table_switch](#table_switch)
- [dce](#dce)
- [recv_default](#recv_default)
- [dce](#dce)
- [select_simp](#select_simp)
- [dce](#dce)
- [dataflow](#dataflow)
- [dce](#dce)
- [cond_spec(noBdd)](#cond_specnoBdd)
- [dce](#dce)
- [reassociation](#reassociation)
- [dce](#dce)
- [const_fold](#const_fold)
- [dce](#dce)
- [bitslice_simp](#bitslice_simp)
- [dce](#dce)
- [concat_simp](#concat_simp)
- [dce](#dce)
- [array_untuple](#array_untuple)
- [dce](#dce)
- [dataflow](#dataflow)
- [dce](#dce)
- [strength_red](#strength_red)
- [dce](#dce)
- [array_simp](#array_simp)
- [dce](#dce)
- [cse](#cse)
- [dce](#dce)
- [basic_simp](#basic_simp)
- [dce](#dce)
- [arith_simp](#arith_simp)
- [dce](#dce)
- [narrow(Ternary)](#narrowTernary)
- [dce](#dce)
- [bool_simp](#bool_simp)
- [dce](#dce)
- [token_simp](#token_simp)
- [dce](#dce)



## simp(2) - Max-2 Simplification {#simp2}


Standard simplification pipeline.

Opt level is capped at 2


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set






Cap opt level: 2




### Invoked Passes


- [simp](#simp)



## simp(3) - Max-3 Simplification {#simp3}


Standard simplification pipeline.

Opt level is capped at 3


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set






Cap opt level: 3




### Invoked Passes


- [simp](#simp)



## simp(>=1,<=2) - Min-1 Max-2 Simplification {#simp12}


Standard simplification pipeline.

Opt level is capped at 2 and skipped if less than 1


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set




Min opt level: 1



Cap opt level: 2




### Invoked Passes


- [simp](#simp)



## simplify-and-inline - Iteratively inline and simplify {#simplify-and-inline}


Inlining segment of the pipeline

This is performed to fixedpoint and each run a single layer of the function hierarchy is inlined away.


[Text-proto](http://github.com/google/xls/tree/main/xls/passes/optimization_pass_pipeline.txtpb)


### Options Set


Run to a fixedpoint.








### Invoked Passes


- [pre-inlining](#pre-inlining)
- [one-leaf-inlining](#one-leaf-inlining)



## sparsify_select - Sparsify Select {#sparsify_select}


The SparsifySelectPass is a type of range analysis-informed dead code elimination that removes cases from selects when range analysis proves that they can never occur. It does this by splitting a select into many selects, each of which covers a single interval from the selector interval set.


[Header](http://github.com/google/xls/tree/main/xls/passes/sparsify_select_pass.h)






## strength_red - Strength Reduction {#strength_red}


Replaces operations with equivalent cheaper operations. For example, multiply by a power-of-two constant may be replaced with a shift left.


[Header](http://github.com/google/xls/tree/main/xls/passes/strength_reduction_pass.h)






## table_switch - Table switch conversion {#table_switch}


TableSwitchPass converts chains of Select nodes into ArrayIndex ops. These chains have the form: sel.(N)(eq.X, literal.A, literal.B) sel.(N+1)(eq.Y, sel.(N), literal.C) sel.(N+2)(eq.Z, sel.(N+1), literal.D) And so on. In these chains, eq.X, eq.Y, and eq.Z must all be comparisons of the same value against different literals.

 Current limitations:  - Either the start or end index in the chain must be 0.  - The increment between indices must be positive or negative 1.  - There can be no "gaps" between indices.  - The Select ops have to be binary (i.e., selecting between only two cases).


[Header](http://github.com/google/xls/tree/main/xls/passes/table_switch_pass.h)






## token_dependency - Convert data dependencies between effectful operations into token dependencies {#token_dependency}


Pass which turns data dependencies between certain effectful operations into token dependencies. In particular, transitive data dependencies between receives and other effectful ops are turned into token dependencies whenever no such token dependency already exists.


[Header](http://github.com/google/xls/tree/main/xls/passes/token_dependency_pass.h)






## token_simp - Simplify token networks {#token_simp}


Pass that simplifies token networks. For example, if an AfterAll node has operands where one operand is an ancestor of another in the token graph, then the ancestor can be omitted. Similarly, duplicate operands can be removed and AfterAlls with one operand can be replaced with their operand.


[Header](http://github.com/google/xls/tree/main/xls/passes/token_simplification_pass.h)






## useless_assert_remove - Remove useless (always true) asserts {#useless_assert_remove}


Pass which removes asserts that have a literal 1 as the condition, meaning they are never triggered. Rewires tokens to ensure nothing breaks.


[Header](http://github.com/google/xls/tree/main/xls/passes/useless_assert_removal_pass.h)






## useless_io_remove - Remove useless send/receive {#useless_io_remove}


Pass which removes sends/receives that have literal false as their condition. Also removes the condition from sends/receives that have literal true as their condition.


[Header](http://github.com/google/xls/tree/main/xls/passes/useless_io_removal_pass.h)





