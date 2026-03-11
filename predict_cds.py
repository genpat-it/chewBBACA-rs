#!/usr/bin/env python3
"""Pre-compute CDS prediction using pyrodigal, compatible with chewBBACA.
Writes per-genome FASTA files in the output directory."""

import sys
import os
import argparse
from pathlib import Path

import pyrodigal
from Bio import SeqIO


def predict_genome(genome_path, training_info, translation_table, output_dir):
    """Predict CDS for one genome and write FASTA."""
    genome_name = Path(genome_path).stem
    output_file = os.path.join(output_dir, f"{genome_name}.cds.fasta")

    orf_finder = pyrodigal.GeneFinder(meta=False, training_info=training_info)

    records = list(SeqIO.parse(genome_path, "fasta"))
    contig_lengths = {rec.id: len(rec.seq) for rec in records}

    with open(output_file, "w") as fh:
        protein_idx = 0
        for rec in records:
            genes = orf_finder.find_genes(str(rec.seq).encode())
            for gene in genes:
                seq = gene.sequence()
                # Header format: >contig_name # start # stop # strand # ID=N;...
                strand = 1 if gene.strand == 1 else -1
                header = (
                    f"{rec.id}_{protein_idx} # {gene.begin} # {gene.end} # {strand} "
                    f"# contig_len={contig_lengths[rec.id]}"
                )
                fh.write(f">{header}\n{seq}\n")
                protein_idx += 1

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Predict CDS with pyrodigal")
    parser.add_argument("-i", "--input", required=True, help="Input genome directory")
    parser.add_argument("-g", "--schema", required=True, help="Schema directory (for training file)")
    parser.add_argument("-o", "--output", required=True, help="Output directory for CDS FASTA files")
    parser.add_argument("-t", "--translation-table", type=int, default=11)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Find training file
    training_info = None
    for f in os.listdir(args.schema):
        if f.endswith(".trn"):
            trn_path = os.path.join(args.schema, f)
            with open(trn_path, "rb") as fh:
                training_info = pyrodigal.TrainingInfo.load(fh)
            print(f"Using training file: {trn_path}", file=sys.stderr)
            break

    if training_info is None:
        print("WARNING: No training file found, will self-train per genome", file=sys.stderr)

    # Process genomes
    genome_dir = Path(args.input)
    extensions = {".fasta", ".fa", ".fna", ".fas", ".fsa"}
    genome_files = sorted(
        f for f in genome_dir.iterdir()
        if f.suffix.lower() in extensions
    )

    for gf in genome_files:
        out = predict_genome(str(gf), training_info, args.translation_table, args.output)
        print(f"  {gf.name} -> {Path(out).name}", file=sys.stderr)

    print(f"Done. {len(genome_files)} genomes processed.", file=sys.stderr)


if __name__ == "__main__":
    main()
