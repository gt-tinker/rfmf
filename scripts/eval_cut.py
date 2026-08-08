#!/usr/bin/env python3
"""Print the weighted cut size of a Gset instance + partition bit string.

This is the artifact's independent check on its own headline numbers.  It
shares NOTHING with the solver: no CUDA, no C++, no CSR, not even a helper
module -- it re-reads the original Gset edge list and adds up the weight of
every edge whose endpoints landed in different halves.  A reviewer who does
not trust a line of the kernel can still take a "Solution-best-partition"
straight out of a committed log, feed it here, and confirm the "best-cut-value"
printed on that chain's own "device:" line.  That is the whole reason this file
exists, so it is kept deliberately dumb and short: standard library only, one
obvious loop, auditable end to end in about thirty seconds.  Please do not make
it clever.

The one subtlety, and it is not optional: the solver numbers vertices densely.
A Gset header may declare more vertices than the edge list ever mentions, and
those edgeless vertices never enter the CSR, so the printed bit string has one
bit per USED vertex in increasing original-id order -- not one bit per declared
vertex.  Three of the paper's twelve instances are affected (G60 and G61 use
6957 of a declared 7000, G70 uses 8646 of 10000); indexing them as "id - 1"
scores the wrong vertices.  Everywhere else the two numberings coincide, so the
mapping below is the same single code path for every instance.

Both arguments are required.  There is no default instance and no default bit
string on purpose -- a reviewer who runs this bare must get a usage message,
never a plausible-looking number for a solution they did not ask for.

Usage:
  eval_cut.py INSTANCE BITS
  eval_cut.py INSTANCE -        # read BITS from stdin, for long partitions

  INSTANCE   a file name in the artifact's Gset/ directory, e.g. G60
  BITS       one character per vertex the solver used, in vertex order; any two
             distinct characters work, only equality between them is tested

Examples, from the artifact root:
  python3 scripts/eval_cut.py G60 0011011111...
  sed -n '152p' results/V100-SXM2-10M/logs/G60.out | cut -d' ' -f2 \\
      | python3 scripts/eval_cut.py G60 -
"""
import pathlib
import sys

argv = sys.argv[1:]
if argv and argv[0] in ("-h", "--help"):
    print(__doc__)
    sys.exit(0)
if len(argv) != 2:
    sys.exit(f"usage: {sys.argv[0]} INSTANCE BITS   (BITS of '-' reads stdin)"
             f"\ntry '{sys.argv[0]} --help'")
INSTANCE, BITS = argv
if BITS == "-":
    BITS = sys.stdin.read().strip()

# Gset/ sits next to scripts/ in the artifact, so this resolves wherever the
# unpacked artifact happens to live.
path = pathlib.Path(__file__).resolve().parent.parent / "Gset" / INSTANCE
if not path.is_file():
    sys.exit(f"no such instance: {path}")
lines = [l for l in path.read_text().split("\n") if l.strip() and not l.lstrip().startswith("#")]
n_header = int(lines[0].split()[0])
edges = [tuple(int(x) for x in line.split()) for line in lines[1:]]

# Original 1-based vertex id -> the solver's dense 0-based id, which is just
# that id's rank among the vertices an edge actually touches (see above).
rank = {old: i for i, old in enumerate(sorted({u for u, _, _ in edges} | {v for _, v, _ in edges}))}

# Deliberately NOT an assert: python3 -O and PYTHONOPTIMIZE strip asserts, and
# this is the only check standing between a misaddressed bit string and a
# plausible-looking wrong number.  A string that is too LONG still indexes
# fine, so under -O an assert would let it score a silent prefix and exit 0 --
# the one failure this script must never produce.
if len(BITS) != len(rank):
    sys.exit(f"{INSTANCE} has {len(rank)} vertices carrying an edge "
             f"(header declares {n_header}), bit string has {len(BITS)}")

cut = 0
for u, v, w in edges:
    if BITS[rank[u]] != BITS[rank[v]]:
        cut += w
print(cut)
