# XLS stage-aware openSTA timing tcl script for the run_opensta rule.
# It can be replaced by a user-defined script by overriding the sta_tcl
# argument of that rule.

# User-defined timing scripts need to consult the following environment
# variables for their parameters:
# NETLIST = synthesized netlist (Verilog)
# TOP = top module in NETLIST
# LIBERTY = liberty file for the target technology library
# ADDITIONAL_LIBERTIES = all lib files required for the design, e.g. multi Vt.
# LOGFILE = file for analysis output

set sta_log $::env(LOGFILE)
set netlist $::env(NETLIST)
set liberty $::env(LIBERTY)
set top $::env(TOP)
set all_liberties [split $::env(ADDITIONAL_LIBERTIES) ","]
lappend all_liberties $liberty

sta::redirect_file_begin $sta_log

foreach lib $all_liberties {
    puts "reading lib file: $lib"
    read_liberty $lib
}

#
# Force ps for time units regardless of Liberty default
#
set_cmd_units -time ps
report_units

#
# Longest path overall
#
puts "Total design"
sta::clear_sta
read_verilog $netlist
link_design  $top
report_checks -unconstrained

#
# Find stage module names
#
sta::clear_sta
read_verilog $netlist
link_design  $top
set stagemods [get_cells "p*mod"]
set snames {}
foreach s $stagemods {
    lappend snames [get_full_name $s]
}
puts -nonewline "STAGES: "
puts $snames
puts "\n\n"

#
# Longest path in each stage
#
foreach stage $snames {
    puts "\n\nTiming ${stage}"
    sta::clear_sta
    read_verilog $netlist
    link_design $stage
    report_checks -unconstrained -path_delay max -fields {slew cap input nets fanout} -format full_clock_expanded

    #
    # print out endpoint info
    #
    set tpaths [find_timing_paths -unconstrained -path_delay max]
    foreach p $tpaths {
        set sp [get_full_name [get_property $p "startpoint"]]
        puts "$stage Startpoint: $sp"
        set ep [get_full_name [get_property $p "endpoint"]]
        puts "$stage Endpoint: ${ep}"
        set n [get_nets -of_objects $ep]
        if {$n != ""} {
            set n_name [get_property $n name]
            puts "$stage Endpoint net name: $n_name"
        }
        #
        # Assuming endpoint is a flop input, get flop and the net it drives
        #
        try {
            set inst [get_cells -of_objects $ep]
        } on error { } {
            set inst ""
        }
        if {$inst != ""} {
            set qp [get_pins -of_objects $inst -filter “direction==output”]
            foreach pin $qp {
                set pinname [get_full_name $pin]
                set qn [get_nets -of_objects $pinname]
                set qn_name [get_full_name $qn]
                puts "$stage Endpoint flop outpout net: $qn_name"
            }
        }
    }

}

sta::redirect_file_end
