#!/usr/bin/env python3
"""CRC32 collision audit (reviewer point #8).

The callable-cell concordance check compares CRC32 hashes of allele sequences.
CRC32 is a 32-bit non-cryptographic checksum, so in principle two *distinct*
DNA sequences could share a hash. This audit checks, per schema, whether any
CRC32 value is produced by more than one distinct allele sequence. If there are
zero such collisions, then "equal CRC32" is equivalent to "equal sequence" on
these schemas, and the 0-discordance concordance result is exact (not an
artefact of hash collisions).

CRC32 uses the IEEE 802.3 polynomial (zlib.crc32), matching chewcall
(crc32fast) and chewBBACA (binascii.crc32).

Usage: collision_audit.py <schema_dir> [<schema_dir> ...]
"""
import sys
import zlib
from pathlib import Path


def audit_schema(schema_dir: Path):
    by_crc = {}            # crc32 -> set of distinct sequences
    n_alleles = 0
    for fa in sorted(schema_dir.glob("*.fasta")):
        seq_lines, seq = [], None
        with open(fa) as f:
            for line in f:
                if line.startswith(">"):
                    if seq is not None:
                        _add(by_crc, "".join(seq_lines)); n_alleles += 1
                    seq_lines, seq = [], True
                else:
                    seq_lines.append(line.strip().upper())
            if seq is not None:
                _add(by_crc, "".join(seq_lines)); n_alleles += 1
    distinct_seqs = sum(len(v) for v in by_crc.values())
    collisions = {c: v for c, v in by_crc.items() if len(v) > 1}
    return n_alleles, len(by_crc), distinct_seqs, collisions


def _add(by_crc, seq):
    if not seq:
        return
    c = zlib.crc32(seq.encode())
    by_crc.setdefault(c, set()).add(seq)


def main():
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr); return 2
    total_coll = 0
    print(f"{'schema':<28}{'alleles':>12}{'distinct_seq':>14}{'distinct_crc':>14}{'collisions':>12}")
    for d in sys.argv[1:]:
        p = Path(d)
        n_all, n_crc, n_seq, coll = audit_schema(p)
        total_coll += len(coll)
        print(f"{p.name:<28}{n_all:>12,}{n_seq:>14,}{n_crc:>14,}{len(coll):>12}")
        for c, seqs in list(coll.items())[:3]:
            print(f"    COLLISION crc={c}: {len(seqs)} distinct sequences", file=sys.stderr)
    print()
    if total_coll == 0:
        print("RESULT: zero CRC32 collisions among distinct allele sequences -> "
              "CRC32 equality == sequence equality on these schemas.")
    else:
        print(f"RESULT: {total_coll} colliding CRC32 value(s) found -> report and "
              "switch the concordance check to SHA-256 / direct comparison.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
