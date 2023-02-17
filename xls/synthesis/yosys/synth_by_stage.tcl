# Based on bazel_rules_hdl/synthesis/synth.tcl.
#
# Tcl script for the synthesize_rtl rule, customized to break each
# XLS pipeline stage into its own module.
#
# It can be replaced by a user-defined script by overriding the synth_tcl
# argument of that rule.
#
# User-defined synthesis scripts need to consult the following environment
# variables for their parameters:
# FLIST = a file that lists verilog sources (one file per line)
# UHDM_FLIST = a file that lists UHDM sources (one file per line)
# TOP = top module for synthesis
# LIBERTY = liberty file for the target technology library
# OUTPUT = verilog file for synthesis output

yosys -import

# read design
set srcs_flist_path $::env(FLIST)
set srcs_flist_file [open $srcs_flist_path "r"]
set srcs_flist_data [read $srcs_flist_file]
set srcs [split $srcs_flist_data "\n"]
puts $srcs
foreach src $srcs {
    # Skip empty lines, including the implict one after the last \n delimiter
    # for files that end with a newline.
    if {$src eq ""} continue
    yosys read_verilog -sv -defer $src
}

# read UHDM designs
set srcs_uhdm_flist_path $::env(UHDM_FLIST)
set srcs_uhdm_flist_file [open $srcs_uhdm_flist_path "r"]
set srcs_uhdm_flist_data [read $srcs_uhdm_flist_file]
set srcs [split $srcs_uhdm_flist_data "\n"]
puts $srcs
foreach src $srcs {
    # Skip empty lines, including the implict one after the last \n delimiter
    # for files that end with a newline.
    if {$src eq ""} continue
    read_uhdm $src
}

# generic synthesis
set top $::env(TOP)
yosys synth -top $top

# create module for each XLS stage
set max_stage 20
for {set i 0} {$i < $max_stage} {incr i} {
    log doing sel p${i}
    select -none
    #
    # Documentation of the following select command:
    #
    # select */w:p${i}_*                        Add wires with names pN_*
    # select */w:p${i}_* %ci:+\[Q\]             Add cells that drive those wires using a pin Q (i.e. flops)
    # select */w:p${i}_* %ci:+\[Q\] */w:* %d    Remove wires from active set
    # select */w:p${i}_* %ci:+\[Q\] */w:* %d %ci:+\[D,S,R\]             Add input wires driving flop D/S/R pins
    # select */w:p${i}_* %ci:+\[Q\] */w:* %d %ci:+\[D,S,R\] %cie*       Add fanin cones, stopping at flops
    #    ... and a final */w:* %d to remove wires (we just want cells)
    #
    select */w:p${i}_* %ci:+\[Q\] */w:* %d %ci:+\[D,S,R\] %cie* */w:* %d
    select -count
    set mcount [scratchpad -copy select.count result.string]
    if { $mcount > 0 } {
        log -n Cell count in p${i}mod: $mcount
        submod -name "p${i}mod"
    }
}
select -clear
opt_clean -purge

# ====== mapping to liberty
set liberty $::env(LIBERTY)
dfflibmap -liberty $liberty
if { [info exists ::env(CLOCK_PERIOD) ] } {
  abc -liberty $liberty -dff -g aig -D $::env(CLOCK_PERIOD)
} else {
  abc -liberty $liberty -dff -g aig
}

# ====== write synthesized design
set output $::env(OUTPUT)
write_verilog $output

# ====== print stats / info ======
select -clear
stat -liberty $liberty
read_liberty -lib -ignore_miss_func $liberty
ltp -noff
yosys log -n Flop count:\ 
yosys select -count t:*__df* t:DF* t:*_DFF* t:*_SDFF* t:*_ADFF* t:*dff

# ====== per-module flop count ======
for {set i 0} {$i < $max_stage} {incr i} {
    set mname "p${i}mod"

    # make sure module exists and is nonempty
    select -clear
    select -count ${mname}/*
    set mcount [scratchpad -copy select.count result.string]
    if { $mcount == 0 } { continue }

    yosys select -clear
    yosys log -n Flop count $mname:\ 
    yosys select -module $mname -count t:*__df* t:DF* t:*_DFF* t:*_SDFF* t:*_ADFF* t:*dff
}

set base_liberty [file tail $liberty]
yosys log Liberty: $base_liberty
