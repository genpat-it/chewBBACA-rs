#!/bin/bash
set -euo pipefail

###############################################################################
# Prepare ALL missing data for benchmarks
# Run this ONCE before run_all.sh
###############################################################################

BENCH_DIR="/mnt/disk2/a.deruvo/chewcall_benchmarks"
MAMEDE="/mnt/disk2/a.deruvo/chewcall_paper/07_chewbbaca3_benchmark"
PALMA="/mnt/disk2/a.deruvo/palma_validation"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

###############################################################################
# 1. Benchmark 02: Generate missing Lm CDS for circular genomes
###############################################################################

prepare_02_circular_cds() {
    log "=== Benchmark 02: Circular genome CDS ==="
    local cds_dir="$BENCH_DIR/02_circular/cds"
    mkdir -p "$cds_dir"/{lm,se,ec,cj}

    # Lm EGDe - need to generate
    if [ ! -f "$cds_dir/lm/Lm_EGDe.cds.fasta" ]; then
        log "Generating CDS for Lm EGDe..."
        python3 -c "
import pyrodigal
from pathlib import Path

# Read genome
genome_path = '$PALMA/circular_all/Lm_EGDe.fna'
seqs = []
with open(genome_path) as f:
    seq = []
    for line in f:
        if line.startswith('>'):
            if seq: seqs.append(''.join(seq))
            seq = []
        else:
            seq.append(line.strip())
    if seq: seqs.append(''.join(seq))

# Predict with pyrodigal (closed ends, mask)
p = pyrodigal.GeneFinder(meta=False, closed=True, mask=True)
p.train(*[s.encode() for s in seqs], translation_table=11)

out = open('$cds_dir/lm/Lm_EGDe.cds.fasta', 'w')
for i, seq_str in enumerate(seqs):
    genes = p.find_genes(seq_str.encode())
    for gene in genes:
        dna = gene.sequence()
        out.write(f'>gene_{i}_{gene.begin}_{gene.end}_{gene.strand}\n{dna}\n')
out.close()
print(f'Done: {Path(\"$cds_dir/lm/Lm_EGDe.cds.fasta\").stat().st_size} bytes')
"
    fi

    # Lm ATCC19115
    if [ ! -f "$cds_dir/lm/Lm_ATCC19115.cds.fasta" ]; then
        log "Generating CDS for Lm ATCC19115..."
        python3 -c "
import pyrodigal
from pathlib import Path

genome_path = '$PALMA/references/ATCC19115.fna'
seqs = []
headers = []
with open(genome_path) as f:
    seq = []
    for line in f:
        if line.startswith('>'):
            if seq: seqs.append(''.join(seq))
            headers.append(line.strip())
            seq = []
        else:
            seq.append(line.strip())
    if seq: seqs.append(''.join(seq))

p = pyrodigal.GeneFinder(meta=False, closed=True, mask=True)
p.train(*[s.encode() for s in seqs], translation_table=11)

out = open('$cds_dir/lm/Lm_ATCC19115.cds.fasta', 'w')
for i, seq_str in enumerate(seqs):
    genes = p.find_genes(seq_str.encode())
    for gene in genes:
        dna = gene.sequence()
        out.write(f'>gene_{i}_{gene.begin}_{gene.end}_{gene.strand}\n{dna}\n')
out.close()
print(f'Done: {Path(\"$cds_dir/lm/Lm_ATCC19115.cds.fasta\").stat().st_size} bytes')
"
    fi

    # Se, Ec, Cj - symlink from existing
    for f in SE_14028S.cds.fasta SE_LT2.cds.fasta; do
        [ -f "$cds_dir/se/$f" ] || ln -sf "$PALMA/circular_all/se_cds/$f" "$cds_dir/se/$f"
    done
    [ -f "$cds_dir/ec/EC_K12.cds.fasta" ] || ln -sf "$PALMA/circular_all/ec_cds/EC_K12.cds.fasta" "$cds_dir/ec/"
    [ -f "$cds_dir/cj/CJ_NCTC11168.cds.fasta" ] || ln -sf "$PALMA/circular_all/cj_cds/CJ_NCTC11168.cds.fasta" "$cds_dir/cj/"

    # Genomes symlinks
    local gdir="$BENCH_DIR/02_circular/genomes"
    mkdir -p "$gdir"/{lm,se,ec,cj}
    [ -f "$gdir/lm/Lm_EGDe.fna" ] || ln -sf "$PALMA/circular_all/Lm_EGDe.fna" "$gdir/lm/"
    [ -f "$gdir/lm/Lm_ATCC19115.fna" ] || ln -sf "$PALMA/references/ATCC19115.fna" "$gdir/lm/Lm_ATCC19115.fna"
    [ -f "$gdir/se/SE_LT2.fna" ] || ln -sf "$PALMA/circular_all/Se_LT2.fna" "$gdir/se/SE_LT2.fna"
    [ -f "$gdir/se/SE_14028S.fna" ] || ln -sf "$PALMA/circular_all/Se_14028S.fna" "$gdir/se/SE_14028S.fna"
    [ -f "$gdir/ec/EC_K12.fna" ] || ln -sf "$PALMA/circular_all/Ec_K12.fna" "$gdir/ec/EC_K12.fna"
    [ -f "$gdir/cj/CJ_NCTC11168.fna" ] || ln -sf "$PALMA/circular_all/CJ_NCTC11168.fna" "$gdir/cj/CJ_NCTC11168.fna"

    log "Benchmark 02 data ready."
    ls -R "$cds_dir" | head -20
}

###############################################################################
# 2. Benchmark 03: Link degradation assemblies and CDS
###############################################################################

prepare_03_degradation() {
    log "=== Benchmark 03: Degradation data ==="
    local base="$BENCH_DIR/03_degradation"

    # Structure: 03_degradation/{lm,se,ec,cj}/{coverage}x_rep{N}/{genomes/,cds/}
    declare -A ORG_PREFIX
    ORG_PREFIX[lm]="Lm_EGDe"
    ORG_PREFIX[se]="Se_LT2"
    ORG_PREFIX[ec]="Ec_K12"
    ORG_PREFIX[cj]="Cj_NCTC11168"

    for org in lm se ec cj; do
        local prefix="${ORG_PREFIX[$org]}"
        for cov in 10 20 30 40 50 60 70 80 90 100; do
            for rep in 1 2 3; do
                local src_dir="$PALMA/circular_degradation/${prefix}_${cov}x_rep${rep}"
                local dst_dir="$base/$org/${cov}x_rep${rep}"

                if [ ! -d "$src_dir" ]; then
                    log "  WARN: $src_dir not found, skipping"
                    continue
                fi

                mkdir -p "$dst_dir/genomes" "$dst_dir/cds"

                # Assembly (genome)
                local assembly="$src_dir/assembly/contigs.fa"
                [ -f "$assembly" ] || assembly="$src_dir/assembly/scaffolds.fasta"
                [ -f "$assembly" ] || assembly=$(find "$src_dir/assembly" -name "*.fa" -o -name "*.fasta" 2>/dev/null | head -1)
                if [ -f "$assembly" ]; then
                    ln -sf "$assembly" "$dst_dir/genomes/${prefix}_${cov}x_rep${rep}.fasta" 2>/dev/null || true
                fi

                # CDS from degradation_results
                local cds_file="$PALMA/circular_degradation_results/$prefix/cds/${prefix}_${cov}x_rep${rep}.cds.fasta"
                if [ -f "$cds_file" ]; then
                    ln -sf "$cds_file" "$dst_dir/cds/${prefix}_${cov}x_rep${rep}.cds.fasta" 2>/dev/null || true
                else
                    # CDS from assembly dir
                    if [ -f "$src_dir/cds.fna" ]; then
                        ln -sf "$src_dir/cds.fna" "$dst_dir/cds/${prefix}_${cov}x_rep${rep}.cds.fasta" 2>/dev/null || true
                    fi
                fi
            done
        done
    done

    log "Benchmark 03 data ready."
    echo "  Directories: $(find "$base" -mindepth 2 -maxdepth 2 -type d | wc -l)"
}

###############################################################################
# 3. Benchmark 06: Extract genomes from .agc for all organisms and sizes
###############################################################################

prepare_06_genomes() {
    log "=== Benchmark 06: Extract genomes from Mamede .agc archives ==="

    local genomes_dir="$BENCH_DIR/06_c3_scalability/genomes"
    local datasets="$MAMEDE/Datasets/Datasets"

    declare -A ORG_AGC ORG_SUBDIR
    ORG_AGC[lm]="$datasets/lm_datasets/lm_draft_genomes/lm_draft_genomes.agc"
    ORG_AGC[se]="$datasets/se_datasets/se_draft_genomes/se_draft_genomes.agc"
    ORG_AGC[sp]="$datasets/sp_datasets/sp_draft_genomes/sp_draft_genomes.agc"
    ORG_SUBDIR[lm]="$datasets/lm_datasets/lm_draft_genomes/lm_draft_genomes_subdatasets"
    ORG_SUBDIR[se]="$datasets/se_datasets/se_draft_genomes/se_draft_genomes_subdatasets"
    ORG_SUBDIR[sp]="$datasets/sp_datasets/sp_draft_genomes/sp_draft_genomes_subdatasets"

    # Sizes matching Mamede et al. paper
    declare -A ORG_SIZES
    ORG_SIZES[lm]="128 256 512 1024 2048 4096"
    ORG_SIZES[se]="128 512 1024 2048 4096"
    ORG_SIZES[sp]="128 512 1024 2048 4096"

    for org in lm se sp; do
        local agc="${ORG_AGC[$org]}"
        local subdir="${ORG_SUBDIR[$org]}"

        if [ ! -f "$agc" ]; then
            log "ERROR: AGC not found: $agc"
            continue
        fi

        for size in ${ORG_SIZES[$org]}; do
            local outdir="$genomes_dir/$org/$size"
            local existing
            existing=$(ls "$outdir" 2>/dev/null | wc -l)

            if [ "$existing" -ge "$size" ]; then
                log "  $org/$size: already has $existing genomes, skip"
                continue
            fi

            # Use replicate 1 list (ss{size}_1.txt)
            local list_file="$subdir/$size/ss${size}_1.txt"
            if [ ! -f "$list_file" ]; then
                log "  WARN: list not found: $list_file"
                continue
            fi

            mkdir -p "$outdir"
            local count=0
            local total
            total=$(wc -l < "$list_file")
            log "  Extracting $org/$size: $total genomes from agc..."

            while IFS= read -r accession; do
                local outfasta="$outdir/${accession}.fasta"
                if [ ! -f "$outfasta" ]; then
                    agc getset "$agc" "$accession" > "$outfasta" 2>/dev/null || {
                        log "    WARN: failed to extract $accession"
                        rm -f "$outfasta"
                    }
                fi
                count=$((count + 1))
                if [ $((count % 100)) -eq 0 ]; then
                    log "    $count / $total extracted"
                fi
            done < "$list_file"

            local final_count
            final_count=$(ls "$outdir"/*.fasta 2>/dev/null | wc -l)
            log "  $org/$size: $final_count genomes extracted"
        done
    done
}

###############################################################################
# 4. Verify all data
###############################################################################

verify_all() {
    log ""
    log "=== VERIFICATION ==="

    echo ""
    echo "--- Benchmark 01: BeONE ---"
    for org in lm se ec cj; do
        echo "  $org: $(ls /mnt/disk2/a.deruvo/beone_benchmarks/data/$org/genomes/ | wc -l) genomes, $(ls /mnt/disk2/a.deruvo/beone_benchmarks/data/$org/cds/ | wc -l) CDS"
    done

    echo ""
    echo "--- Benchmark 02: Circular ---"
    for org in lm se ec cj; do
        echo "  $org genomes: $(ls $BENCH_DIR/02_circular/genomes/$org/ 2>/dev/null | wc -l)"
        echo "  $org CDS:     $(ls $BENCH_DIR/02_circular/cds/$org/ 2>/dev/null | wc -l)"
    done

    echo ""
    echo "--- Benchmark 03: Degradation ---"
    for org in lm se ec cj; do
        local ndirs
        ndirs=$(find "$BENCH_DIR/03_degradation/$org" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        echo "  $org: $ndirs coverage×rep combinations"
    done

    echo ""
    echo "--- Benchmark 04: Synthetic ---"
    for org in lm se ec cj; do
        echo "  $org: $(ls /mnt/disk2/a.deruvo/chew_results/synthetic_cds_$org/cds/ | wc -l) CDS files"
    done

    echo ""
    echo "--- Benchmark 05: Outbreak ---"
    for org in lm se ec cj; do
        echo "  $org: $(ls /mnt/disk2/a.deruvo/chewcall_paper/06_outbreak_validation/$org/assemblies/ | wc -l) genomes, $(ls /mnt/disk2/a.deruvo/chewcall_paper/06_outbreak_validation/$org/cds/ | wc -l) CDS"
    done

    echo ""
    echo "--- Benchmark 06: c3 Scalability ---"
    for org in lm se sp; do
        echo -n "  $org: "
        for size in 128 256 512 1024 2048 4096; do
            local n
            n=$(ls "$BENCH_DIR/06_c3_scalability/genomes/$org/$size/" 2>/dev/null | wc -l)
            echo -n "${size}=$n "
        done
        echo ""
    done

    echo ""
    log "=== VERIFICATION DONE ==="
}

###############################################################################
# Main
###############################################################################

main() {
    log "=========================================="
    log "  Data preparation for chewcall benchmarks"
    log "=========================================="

    prepare_02_circular_cds
    prepare_03_degradation
    prepare_06_genomes
    verify_all

    log "All data prepared. Ready to run benchmarks."
}

main "$@"
