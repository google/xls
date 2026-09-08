const VERBOSITY_LEV_0 = u32:0;
const VERBOSITY_LEV_1 = u32:2;
const VERBOSITY_LEV_2 = u32:4;
const VERBOSITY_LEV_3 = u32:8;
const VERBOSITY_LEV_4 = u32:16;
const VERBOSITY_LEV_5 = u32:32;
const VERBOSITY_LEV_6 = u32:64;

proc Vprinter {
    trigger: chan<bool> in;

    config(trigger: chan<bool> in) { (trigger,) }

    init { () }

    next(st: ()) {
        vtrace_fmt!(VERBOSITY_LEV_0, "Verbosity level {:d}", VERBOSITY_LEV_0);
        vtrace_fmt!(VERBOSITY_LEV_1, "Verbosity level {:d}", VERBOSITY_LEV_1);
        vtrace_fmt!(VERBOSITY_LEV_2, "Verbosity level {:d}", VERBOSITY_LEV_2);
        vtrace_fmt!(VERBOSITY_LEV_3, "Verbosity level {:d}", VERBOSITY_LEV_3);
        vtrace_fmt!(VERBOSITY_LEV_4, "Verbosity level {:d}", VERBOSITY_LEV_4);
        vtrace_fmt!(VERBOSITY_LEV_5, "Verbosity level {:d}", VERBOSITY_LEV_5);
        vtrace_fmt!(VERBOSITY_LEV_6, "Verbosity level {:d}", VERBOSITY_LEV_6);
        trace_fmt!("Trace verification.");
        let (tok, _) = recv(join(), trigger);
    }
}
