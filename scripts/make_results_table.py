#!/usr/bin/env python3
"""Emit the paper's two LaTeX results tables on stdout.

Two benchmark folders go in: the first becomes the "Cut Value and Run time"
table's left column group, the second the right, each headed with the S that
folder actually ran (read from its own report.txt).  A head-to-head table
follows, scoring each instance against the cited SB result and preferring the
smaller budget; an instance with no budget that wins on both cut and time is
left as "---".  Nothing is printed until every number has been
re-derived from the raw logs: partitions are re-scored against the original
Gset graphs, .out is cross-checked against .err, the summary line, and the
CSV row, and the sweep count is verified everywhere it was recorded.  Any
disagreement is a hard failure: diagnostics to stderr, no LaTeX, non-zero
exit.  An instance absent from a folder just leaves that half of its row
blank (noted on stderr).  READ-ONLY: this script never writes anything.

Typical use, from the artifact root:
  python3 scripts/make_results_table.py results/full_gset_<short-run> \
                                        results/full_gset_<long-run>
"""

import argparse
import csv
import re
import sys
from pathlib import Path

# Resolved from this script's own location, so the artifact can be moved.
HERE = Path(__file__).resolve().parent
ARTIFACT = HERE.parent
GSET = ARTIFACT / "Gset"

# The table's rows, in order.  Hard-coded by the paper, not by the data.
INSTANCES = [
    "G60", "G61", "G62", "G63", "G64", "G65",
    "G66", "G67", "G70", "G72", "G77", "G81",
]

# Reference results transcribed verbatim from the cited Simulated Bifurcation
# paper: instance -> (q_best, t).  Not produced or verified by this artifact.
SB = {
    "G60": (14186, 21827),
    "G61": (5796, 10883),
    "G62": (4862, 20501),
    "G63": (27023, 31279),
    "G64": (8739, 31448),
    "G65": (5546, 5651),
    "G66": (6342, 27408),
    "G67": (6922, 6340),
    "G70": (9578, 31599),
    "G72": (6982, 6142),
    "G77": (9904, 46760),
    "G81": (13992, 62194),
}

# logs/<G>.out: one 3-line block per chain, then one summary line.
OUT_DEVICE_RE = re.compile(
    r"^device:.*?--iters\s+([\d,]+)\s+"
    r"best-cut-value:\s*(-?\d+)\s+final-cut-value:\s*(-?\d+)\s+"
    r"time-sec:\s*([\d.]+)"
)
# Every field is required: ./main always prints all five.  hits_at_best is
# additionally verified against the per-chain values.
OUT_SUMMARY_RE = re.compile(
    r"^summary:\s*chains:\s*(\d+)\s+best-chain:\s*(\d+)\s+"
    r"best-cut-value:\s*(-?\d+)\s+time-sec:\s*([\d.]+)"
    r"\s+chains-at-best:\s*(\d+)"
)

# logs/<G>.err: one "Loaded:" header line then one line per chain.
ERR_LOADED_RE = re.compile(r"^Loaded:\s*n=(\d+)\s+m=(\d+)\s+weight-scale=([\d.]+)")
ERR_CHAIN_RE = re.compile(
    r"^chain\s+(\d+):\s*kernel-best-val:\s*(-?\d+)\s+kernel-final-val:\s*(-?\d+)"
)

# report.txt lines.  --step is optional (a default run passes none); S then
# falls back to the per-phase count of REPORT_SWEEPS_RE.
REPORT_STEP_RE = re.compile(r"^#\s*invocation:(?:.*?\s--step\s+(\d+))?")
REPORT_SWEEPS_RE = re.compile(r"^#\s*sweeps per chain:\s*(\d+)\s*\((\d+) x (\d+) phases\)")
REPORT_ITERS_RE = re.compile(r"^#\s*--iters\s+([\d,]+)")

# The optional quote lets a shell-quoted "10000,10000" parse the same as bare.
ITERS_FLAG_RE = re.compile(r"--iters\s+[\"']?([\d,]+)")


class Problems:
    """Collects every failed check so one run reports all of them."""

    def __init__(self):
        self.items = []

    def check(self, ok, where, msg):
        if not ok:
            self.items.append(f"{where}: {msg}")
        return ok

    def fail(self, where, msg):
        self.items.append(f"{where}: {msg}")


def load_gset(path):
    """Read a Gset file into the solver's dense vertex numbering.

    The binary prints one bit per CSR vertex and the CSR drops edgeless
    vertices (G61: 7000-vertex header, 6957 used), so a naive 1..n indexing
    mis-scores.  Dense id = rank of the original id among used endpoints.
    """
    edges = []
    with open(path, "r") as fh:
        header = fh.readline().split()
        n_header, m_header = int(header[0]), int(header[1])
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                edges.append((int(parts[0]), int(parts[1]), int(parts[2])))
    used = sorted({u for u, _, _ in edges} | {v for _, v, _ in edges})
    dense = {old: i for i, old in enumerate(used)}
    return {
        "n_header": n_header,
        "m_header": m_header,
        "n_dense": len(used),
        "edges": [(dense[u], dense[v], w) for u, v, w in edges],
    }


def cut_value(edges, bits):
    return sum(w for u, v, w in edges if bits[u] != bits[v])


def parse_out(path, edges, n_dense, prob, where):
    """Stream logs/<G>.out, scoring each chain's two partitions as they arrive
    (the files reach several MB, so bitstrings are scored and dropped)."""
    chains, summary = [], None
    pending = None
    best_bits = None
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("device:"):
                m = OUT_DEVICE_RE.match(line)
                if not m:
                    prob.fail(where, f"unparsable device line: {line[:120].strip()}")
                    return chains, summary
                pending = {
                    "iters": m.group(1),
                    "best": int(m.group(2)),
                    "final": int(m.group(3)),
                    "time_sec": float(m.group(4)),
                }
                best_bits = None
            elif line.startswith("Solution-best-partition:"):
                best_bits = line.split(": ", 1)[1].strip()
            elif line.startswith("Solution-final-partition:"):
                final_bits = line.split(": ", 1)[1].strip()
                k = len(chains)
                if pending is None or best_bits is None:
                    prob.fail(where, f"chain {k}: partition line without a device line")
                    return chains, summary
                for label, bits, reported in (
                    ("best_cut", best_bits, pending["best"]),
                    ("ending_cut", final_bits, pending["final"]),
                ):
                    if len(bits) != n_dense:
                        prob.fail(where, f"chain {k} {label}: partition has {len(bits)} "
                                         f"bits, graph has {n_dense} dense vertices")
                        continue
                    got = cut_value(edges, bits)
                    prob.check(got == reported, where,
                               f"chain {k} {label}: log says {reported}, "
                               f"recomputed from its partition {got}")
                chains.append(pending)
                pending = None
            elif line.startswith("summary:"):
                m = OUT_SUMMARY_RE.match(line)
                if not m:
                    # Keep going: the per-chain checks stand on their own.
                    prob.fail(where, f"unparsable summary line: {line[:120].strip()}")
                    continue
                summary = {
                    "chains": int(m.group(1)),
                    "best_chain": int(m.group(2)),
                    "best_cut": int(m.group(3)),
                    "time_sec": float(m.group(4)),
                    "hits_at_best": int(m.group(5)),
                }
            elif line.startswith("weight-scale:"):
                # Only printed when scale != 1, i.e. cuts would not be raw.
                prob.fail(where, f"unexpected rescaled output: {line[:120].strip()}")
    if summary is None:
        prob.fail(where, "no summary line")
    return chains, summary


def parse_err(path, prob, where):
    loaded, chains = None, []
    with open(path, "r") as fh:
        for line in fh:
            m = ERR_LOADED_RE.match(line)
            if m:
                loaded = {
                    "n": int(m.group(1)),
                    "m": int(m.group(2)),
                    "weight_scale": float(m.group(3)),
                }
                continue
            m = ERR_CHAIN_RE.match(line)
            if m:
                if int(m.group(1)) != len(chains):
                    prob.fail(where, f"chain lines out of order at index {len(chains)}")
                chains.append((int(m.group(2)), int(m.group(3))))
    if loaded is None:
        prob.fail(where, "no 'Loaded:' line")
    return loaded, chains


def verify_instance(row, folder, gset_dir, graphs, prob):
    """Re-derive one results.csv row from its logs; appends to `prob` on mismatch."""
    name = row["instance"]
    where = f"{folder.name}/{name}"
    out_path = folder / "logs" / f"{name}.out"
    err_path = folder / "logs" / f"{name}.err"
    for path in (out_path, err_path):
        if not path.is_file():
            prob.fail(where, f"missing log file {path}")
            return
    gset_path = gset_dir / name
    if not gset_path.is_file():
        prob.fail(where, f"missing graph file {gset_path}")
        return

    if name not in graphs:
        graphs[name] = load_gset(gset_path)
    g = graphs[name]

    # A crashed instance leaves blank numeric cells; turn those into named
    # diagnostics, not ValueError tracebacks.
    def num(field, conv):
        raw = row.get(field, "")
        try:
            return conv(raw)
        except (TypeError, ValueError):
            prob.fail(where, f"results.csv {field}={raw!r} is not a number "
                             f"(the run recorded no result for this instance)")
            return None

    csv_n = num("n", int)
    csv_m = num("m", int)
    csv_chains = num("chains", int)
    csv_best = num("best_cut", int)
    csv_hits = num("hits_at_best", int)
    csv_best_chain = num("best_chain", int)
    csv_kernel_s = num("kernel_s", float)
    csv_wall_s = num("wall_s", float)
    if None in (csv_n, csv_m, csv_chains, csv_best, csv_hits, csv_best_chain,
                csv_kernel_s, csv_wall_s):
        return

    prob.check(csv_n == g["n_header"], where,
               f"csv n={csv_n} but {gset_path.name} header says {g['n_header']}")
    prob.check(csv_m == g["m_header"], where,
               f"csv m={csv_m} but {gset_path.name} header says {g['m_header']}")

    loaded, err_chains = parse_err(err_path, prob, where)
    if loaded is not None:
        prob.check(loaded["n"] == csv_n, where,
                   f"csv n={csv_n} but .err loaded n={loaded['n']}")
        prob.check(loaded["m"] == csv_m, where,
                   f"csv m={csv_m} but .err loaded m={loaded['m']}")
        prob.check(loaded["weight_scale"] == 1.0, where,
                   f"weight-scale={loaded['weight_scale']}, cuts are not raw")

    chains, summary = parse_out(out_path, g["edges"], g["n_dense"], prob, where)
    if not prob.check(len(chains) == csv_chains, where,
                      f"csv chains={csv_chains} but .out has {len(chains)} chain blocks"):
        return
    if summary is None:
        prob.fail(where, "no usable summary line; per-chain checks still applied")
    else:
        prob.check(summary["chains"] == csv_chains, where,
                   f"csv chains={csv_chains} but summary says {summary['chains']}")
    if prob.check(len(err_chains) == len(chains), where,
                  f".out has {len(chains)} chains but .err has {len(err_chains)}"):
        for k, ((e_best, e_final), c) in enumerate(zip(err_chains, chains)):
            prob.check(e_best == c["best"], where,
                       f"chain {k} best_cut: .out {c['best']} vs .err {e_best}")
            prob.check(e_final == c["final"], where,
                       f"chain {k} ending_cut: .out {c['final']} vs .err {e_final}")

    # Per-chain values are authoritative (each just recomputed from its own
    # partition); the summary line is a second, independent witness.
    best_vals = [c["best"] for c in chains]
    prob.check(csv_best == max(best_vals), where,
               f"csv best_cut={csv_best} but max over chains is {max(best_vals)}")
    hits = best_vals.count(csv_best)
    prob.check(csv_hits == hits, where,
               f"csv hits_at_best={csv_hits} but {hits} chains reach {csv_best}")
    if 0 <= csv_best_chain < len(chains):
        prob.check(chains[csv_best_chain]["best"] == csv_best, where,
                   f"chain {csv_best_chain} is named best but scored "
                   f"{chains[csv_best_chain]['best']}, not {csv_best}")
    else:
        prob.fail(where, f"csv best_chain={csv_best_chain} out of range")

    if summary is not None:
        prob.check(csv_best == summary["best_cut"], where,
                   f"csv best_cut={csv_best} but summary says {summary['best_cut']}")
        prob.check(csv_best_chain == summary["best_chain"], where,
                   f"csv best_chain={csv_best_chain} but summary says "
                   f"{summary['best_chain']}")
        prob.check(csv_hits == summary["hits_at_best"], where,
                   f"csv hits_at_best={csv_hits} but summary chains-at-best="
                   f"{summary['hits_at_best']}")
        prob.check(abs(csv_kernel_s - summary["time_sec"]) <= 1e-3, where,
                   f"csv kernel_s={csv_kernel_s} but summary time-sec="
                   f"{summary['time_sec']}")
    prob.check(csv_kernel_s > 0.0, where, f"kernel_s={csv_kernel_s} is not positive")
    prob.check(csv_wall_s >= csv_kernel_s, where,
               f"wall_s={csv_wall_s} is below kernel_s={csv_kernel_s}")

    return [c["iters"] for c in chains]


def read_report(folder, prob):
    """Parse report.txt into {step, sweeps, iters}, or None if unreadable."""
    path = folder / "report.txt"
    if not path.is_file():
        prob.fail(folder.name, "missing report.txt")
        return None
    step = sweeps = report_iters = None
    with open(path, "r") as fh:
        for line in fh:
            m = REPORT_STEP_RE.match(line)
            if m and m.group(1) is not None:
                step = int(m.group(1))
            m = REPORT_SWEEPS_RE.match(line)
            if m:
                sweeps = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            m = REPORT_ITERS_RE.match(line)
            if m:
                report_iters = m.group(1)
    return {"step": step, "sweeps": sweeps, "iters": report_iters}


def derive_s(folder, report, prob):
    """This folder's S, from its OWN report.txt: the invocation's --step when
    present, else the "sweeps per chain" per-phase count.  No particular S is
    required."""
    if report is None:
        return None
    if report["step"] is not None:
        s = report["step"]
    elif report["sweeps"] is not None:
        s = report["sweeps"][1]
    else:
        prob.fail(folder.name, "report.txt has neither an invocation --step nor a "
                               "'sweeps per chain' line; cannot derive S")
        return None
    if not prob.check(s > 0, folder.name, f"report.txt gives S={s}, not positive"):
        return None
    return s


def check_sweeps(folder, rows, expected_s, report, prob):
    """Require S == expected_s everywhere the run recorded it.  expected_s is
    None only when report.txt was unusable (already a failure); the remaining
    internal-consistency checks still run."""
    where = folder.name
    iters_field = None
    for row in rows:
        name = row["instance"]
        phases = row["phases"].split(",")
        values = row["iters"].split(",")
        if not prob.check(len(values) == len(phases), where,
                          f"{name}: {len(values)} iters for {len(phases)} phases"):
            continue
        if not prob.check(len(set(values)) == 1, where,
                          f"{name}: iters are not all equal: {row['iters']}"):
            continue
        if expected_s is not None and not prob.check(
                int(values[0]) == expected_s, where,
                f"{name}: S={int(values[0])}, report.txt says {expected_s}"):
            continue
        if iters_field is None:
            iters_field = row["iters"]
        else:
            prob.check(row["iters"] == iters_field, where,
                       f"{name}: iters {row['iters']} differ from the rest of the folder")
    if iters_field is None:
        prob.fail(where, "no usable iters field"
                         + ("" if expected_s is None else f"; report.txt says S={expected_s}"))
        return None

    n_phases = len(iters_field.split(","))
    if report is not None:
        # Re-check the parts of report.txt that did NOT feed the S derivation
        # against the parts that did.
        step, sweeps = report["step"], report["sweeps"]
        if expected_s is not None and step is not None:
            prob.check(step == expected_s, where,
                       f"report.txt invocation --step {step}, expected {expected_s}")
        prob.check(report["iters"] == iters_field, where,
                   f"report.txt --iters {report['iters']} differ from "
                   f"results.csv {iters_field}")
        if prob.check(sweeps is not None, where, "report.txt has no 'sweeps per chain' line"):
            total, per_phase, phases = sweeps
            if expected_s is not None:
                prob.check(per_phase == expected_s, where,
                           f"report.txt sweeps per chain is {per_phase} x {phases}, "
                           f"expected {expected_s}")
            prob.check(phases == n_phases, where,
                       f"report.txt says {phases} phases, results.csv says {n_phases}")
            prob.check(total == per_phase * phases, where,
                       f"report.txt sweeps per chain {total} != {per_phase} x {phases}")

    repro = folder / "reproduce.sh"
    if repro.is_file():
        # Only lines carrying the flag are runs; a comment about --iters isn't.
        seen = 0
        with open(repro, "r") as fh:
            for lineno, line in enumerate(fh, 1):
                if "--iters" not in line or line.lstrip().startswith("#"):
                    continue
                seen += 1
                m = ITERS_FLAG_RE.search(line)
                if not m:
                    prob.fail(where, f"reproduce.sh line {lineno} has an unreadable "
                                     f"--iters value")
                else:
                    prob.check(m.group(1) == iters_field, where,
                               f"reproduce.sh line {lineno} --iters {m.group(1)} differ "
                               f"from results.csv {iters_field}")
        prob.check(seen > 0, where, "reproduce.sh has no ./main line carrying --iters")
    else:
        prob.fail(where, "missing reproduce.sh")

    return iters_field


def read_rows(folder, prob):
    path = folder / "results.csv"
    if not path.is_file():
        prob.fail(folder.name, f"missing {path}")
        return []
    with open(path, "r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen = set()
    for row in rows:
        name = row["instance"]
        if name in seen:
            prob.fail(folder.name, f"duplicate row for {name}")
        seen.add(name)
    return rows


def collect(folder, expected_s, report, gset_dir, graphs, prob):
    """Verify one folder and return {instance: (q_best, c, t)}."""
    rows = read_rows(folder, prob)
    if not rows:
        return {}
    iters_field = check_sweeps(folder, rows, expected_s, report, prob)

    cells = {}
    for row in rows:
        name = row["instance"]
        if name not in SB:
            continue  # not in the table
        out_iters = verify_instance(row, folder, gset_dir, graphs, prob)
        if iters_field is not None and out_iters is not None:
            for k, got in enumerate(out_iters):
                if got != iters_field:
                    prob.fail(f"{folder.name}/{name}",
                              f"chain {k} ran --iters {got}, results.csv says {iters_field}")
                    break
        # Blank cells (crashed instance) were already flagged above; skip
        # rather than raise on the conversion.
        try:
            cells[name] = (int(row["best_cut"]), int(row["hits_at_best"]),
                           round(float(row["wall_s"])))
        except (TypeError, ValueError):
            continue
    return cells


def format_s(value):
    """10000000 -> 'S=10M'."""
    for unit, scale in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if value >= scale and value % scale == 0:
            return f"S={value // scale}{unit}"
    return f"S={value}"


def emit_table(cells_a, cells_b, s_a, s_b, out=sys.stdout):
    w = out.write
    w("\\begin{table}[htbp]\n")
    w("\\caption{Cut Value and Run time}\n")
    w("\\begin{center}\n")
    w("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n")
    w("\\hline\n")
    w(f"\\textbf{{Graph}}&\\multicolumn{{3}}{{|c|}}{{\\textbf{{{format_s(s_a)}}}}} "
      f"&\\multicolumn{{3}}{{|c|}}{{\\textbf{{{format_s(s_b)}}}}}  "
      "&\\multicolumn{2}{|c|}{\\textbf{SB \\cite{SB2}}} \\\\\n")
    w("\\cline{2-9} \n")
    w("\\textbf{} & \\textbf{\\textit{$q_{best}$}}& \\textbf{\\textit{$c$}}& "
      "\\textbf{\\textit{$t$}}  & \\textbf{\\textit{$q_{best}$}} & "
      "\\textbf{\\textit{$c$}} & \\textbf{\\textit{$t$}} & \\textit{$q_{best}$} & "
      "\\textbf{\\textit{$t$}} \\\\\n")
    w("\\hline\n")
    for name in INSTANCES:
        fields = [name]
        for cells in (cells_a, cells_b):
            cell = cells.get(name)
            fields.extend(["", "", ""] if cell is None else [str(v) for v in cell])
        fields.extend(str(v) for v in SB[name])
        w(" & ".join(fields) + " \\\\\n")
        w("\\hline\n")
    w("\\end{tabular}\n")
    w("\\label{tab:results}\n")
    w("\\end{center}\n")
    w("\\end{table}\n")


def emit_summary(cells_a, cells_b, s_a, s_b, out=sys.stdout):
    """The same table again, with each budget's q_best and t stated relative to
    SB: signed cut-value delta and speedup.  Every instance is shown, losses
    included; SB's own columns stay absolute.  A blank cell (crashed instance,
    already noted on stderr) stays blank, as in the raw table."""
    w = out.write
    w("\\begin{table}[htbp]\n")
    w("\\caption{Improvement over SB on V100: cut-value delta $\\Delta q$ and "
      "speedup $t_{SB}/t$}\n")
    w("\\begin{center}\n")
    w("\\begin{tabular}{|c|c|c|c|c|c|c|c|c|}\n")
    w("\\hline\n")
    w(f"\\textbf{{Graph}}&\\multicolumn{{3}}{{|c|}}{{\\textbf{{{format_s(s_a)}}}}} "
      f"&\\multicolumn{{3}}{{|c|}}{{\\textbf{{{format_s(s_b)}}}}}  "
      "&\\multicolumn{2}{|c|}{\\textbf{SB \\cite{SB2}}} \\\\\n")
    w("\\cline{2-9} \n")
    w("\\textbf{} & \\textbf{\\textit{$\\Delta q$}}& \\textbf{\\textit{$c$}}& "
      "\\textbf{\\textit{$t_{SB}/t$}}  & \\textbf{\\textit{$\\Delta q$}} & "
      "\\textbf{\\textit{$c$}} & \\textbf{\\textit{$t_{SB}/t$}} & "
      "\\textit{$q_{best}$} & \\textbf{\\textit{$t$}} \\\\\n")
    w("\\hline\n")
    for name in INSTANCES:
        sb_f, sb_t = SB[name]
        fields = [name]
        for cells in (cells_a, cells_b):
            cell = cells.get(name)
            if cell is None:
                fields.extend(["", "", ""])
            else:
                q_best, c, t = cell
                fields.extend([f"{q_best - sb_f:+d}", str(c), f"{sb_t / t:.2f}"])
        fields.extend([str(sb_f), str(sb_t)])
        w(" & ".join(fields) + " \\\\\n")
        w("\\hline\n")
    w("\\end{tabular}\n")
    w("\\label{tab:results}\n")
    w("\\end{center}\n")
    w("\\end{table}\n")


def main():
    ap = argparse.ArgumentParser(
        description="Emit the paper's two LaTeX results tables on stdout. "
                    "Read-only: no file is ever written.")
    ap.add_argument("dir_a", type=Path,
                    help="result folder for the table's first column group; its S "
                         "is read from its own report.txt")
    ap.add_argument("dir_b", type=Path,
                    help="result folder for the table's second column group; its S "
                         "is read from its own report.txt")
    ap.add_argument("--gset", type=Path, default=GSET,
                    help=f"directory holding the Gset graph files (default: {GSET})")
    args = ap.parse_args()

    prob = Problems()
    graphs = {}
    folders = (args.dir_a, args.dir_b)
    for folder in folders:
        if not folder.is_dir():
            prob.fail(str(folder), "not a directory")
    if not args.gset.is_dir():
        prob.fail(str(args.gset), "not a directory")
    if prob.items:
        return report_failure(prob)

    reports = [read_report(folder, prob) for folder in folders]
    s_values = [derive_s(folder, report, prob)
                for folder, report in zip(folders, reports)]

    cells = [collect(folder, s, report, args.gset, graphs, prob)
             for folder, s, report in zip(folders, s_values, reports)]
    if prob.items:
        return report_failure(prob)

    for folder, s, got in zip(folders, s_values, cells):
        missing = [g for g in INSTANCES if g not in got]
        if missing:
            print(f"note: {folder.name} (S={s}) has no results for "
                  f"{', '.join(missing)}; those cells are left blank",
                  file=sys.stderr)

    emit_table(cells[0], cells[1], s_values[0], s_values[1])
    print()
    emit_summary(cells[0], cells[1], s_values[0], s_values[1])
    return 0


def report_failure(prob):
    print(f"verification FAILED: {len(prob.items)} problem(s); no table emitted",
          file=sys.stderr)
    for item in prob.items:
        print(f"  {item}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
