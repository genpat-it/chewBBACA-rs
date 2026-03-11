# chewBBACA-rs: Rust AlleleCall

Reimplementazione in Rust del modulo AlleleCall di chewBBACA.
Obiettivo: stessi risultati (matrice allelica identica), 10-20x più veloce.

## Pipeline (7 fasi)

### Fase 1: CDS Prediction
- Input: FASTA assemblaggi genomici
- Chiama `prodigal` come subprocess (o binding C via FFI)
- Output: CDS predetti + coordinate (contig, start, stop, strand)
- Crate: `std::process::Command` per subprocess

### Fase 2: Deduplicazione CDS
- SHA256 hash di ogni sequenza DNA
- HashMap<Hash, Vec<GenomeId>> per raggruppare duplicati
- Tiene solo il "rappresentativo" (primo visto)
- Output: FASTA con CDS distinti

### Fase 3a: Exact Matching DNA
- Carica hash table pre-computate dallo schema (pickle → bisogna convertire o rigenerare)
- Intersezione hash input ∩ hash schema
- Match → classificazione **EXC** (BSR = 1.0)

### Fase 3b: Traduzione
- Traduce CDS non classificati → proteine
- Tavola codoni standard (genetic code 11 default)
- Filtra: lunghezza minima, codoni ambigui, stop prematuri
- Crate: implementazione custom (triviale, ~50 righe)

### Fase 3c: Exact Matching Proteine
- Come 3a ma su hash proteici
- Match → classificazione **INF** (primo match) o **EXC** (già visto)

### Fase 4: Clustering + Allineamento
- Minimizer index (k=5, w=5) sui rappresentativi dello schema
- Clustering proteine non classificate (threshold similarità 0.2)
- Smith-Waterman BLOSUM62 (gap_open=11, gap_extend=1) su ogni cluster
- BSR = raw_score / self_score
- BSR >= 0.6 → classificazione (vedi sotto)
- BSR in [0.6, 0.7) → candidati rappresentativi per fase 5

### Fase 5: Representative Determination (iterativo)
- BLAST/SW rappresentativi vs proteine non classificate
- Classifica match con BSR >= 0.6
- Seleziona nuovi rappresentativi da candidati BSR [0.6, 0.7)
- Ripete finché non trova nuovi rappresentativi

### Fase 6: Classificazione
Per ogni match inesatto (BSR >= threshold):

```
1. Posizione sul contig (solo se input = assemblaggio):
   - contig_length < rep_length → LOTSC
   - allineamento esce dal 5' → PLOT5
   - allineamento esce dal 3' → PLOT3

2. Dimensione CDS:
   - length < mode × 0.8 → ASM
   - length > mode × 1.2 → ALM

3. Match multipli stesso locus:
   - tutti EXC → NIPHEM
   - altrimenti → NIPH

4. Match multipli loci diversi → PAMA

5. Nessun match → LNF

6. Default → INF (nuovo allele inferito)
```

### Fase 7: Output
- `results_alleles.tsv` — matrice allelica (genoma × locus)
- `results_statistics.tsv` — conteggi per classe per genoma
- `results_contigsInfo.tsv` — coordinate CDS per EXC/INF
- `loci_summary_stats.tsv` — conteggi per classe per locus
- `novel_alleles.fasta` — nuovi alleli inferiti

## Struttura Rust

```
chewbbacca-rs/
├── Cargo.toml
├── src/
│   ├── main.rs          # CLI (clap)
│   ├── cds.rs           # Fase 1: CDS prediction (subprocess prodigal)
│   ├── dedup.rs         # Fase 2: deduplicazione SHA256
│   ├── exact_match.rs   # Fase 3: exact matching DNA + proteine
│   ├── translate.rs     # Fase 3b: traduzione codoni
│   ├── cluster.rs       # Fase 4: minimizer clustering
│   ├── sw.rs            # Smith-Waterman BLOSUM62 con SIMD
│   ├── bsr.rs           # BSR computation + thresholds
│   ├── classify.rs      # Fase 6: logica classificazione
│   ├── repdet.rs        # Fase 5: representative determination
│   ├── schema.rs        # Lettura/scrittura schema (FASTA + hash tables)
│   ├── output.rs        # Fase 7: output TSV/FASTA
│   └── types.rs         # Tipi condivisi
```

## Dipendenze Rust

```toml
[dependencies]
clap = { version = "4", features = ["derive"] }    # CLI
rayon = "1.10"                                       # parallelismo
needletail = "0.5"                                   # parsing FASTA (velocissimo)
sha2 = "0.10"                                        # SHA256 hashing
rustc-hash = "2"                                     # FxHashMap (veloce)
memchr = "2"                                         # ricerca byte veloce
csv = "1.3"                                          # output TSV

# Per SW SIMD:
# Implementazione custom con std::arch (AVX2/SSE4.1)
# Oppure parasail-rs (binding a parasail C library)
```

## Smith-Waterman SIMD

L'allineamento SW è il cuore computazionale. Due opzioni:

### Opzione A: parasail-rs (binding C)
- parasail già installato in ~/parasail/
- Supporta AVX2, SSE4.1, NEON
- 50-100 GCUPS su CPU moderna
- Zero sforzo implementativo

### Opzione B: Implementazione custom
- Striped SW come in geo_aligner
- BLOSUM62 hardcoded, gap_open=11, gap_extend=1
- AVX2: 32 celle per istruzione (int8) o 8 celle (int32)
- Più controllo, meno dipendenze

**Raccomandazione**: inizia con parasail-rs, poi reimplementa se serve.

## Performance attesa

| Fase | Python | Rust (stima) | Speedup |
|------|--------|-------------|---------|
| CDS prediction | 60s | 55s (prodigal subprocess, stesso) | 1.1x |
| Dedup + hash | 27s | 2-3s (rayon + SHA256-ni) | 10x |
| Clustering | 7s | 0.5s | 14x |
| SW alignment | 17s (GPU 4 GCUPS) | 2-3s (CPU 50 GCUPS parasail) | 6x |
| Wrapping up | 30s | 3-5s | 8x |
| **Totale** | **~140s** | **~65s** | **~2x** |

Nota: CDS prediction domina. Se si usa pyrodigal multi-thread
o una reimplementazione, si può scendere a ~30s totali.

## Schema Compatibility

Lo schema chewBBACA usa:
- File FASTA per locus (alleli)
- Pickle per hash tables pre-computate
- Pickle per loci_modes

Per compatibilità:
1. **Leggere pickle**: usare `serde` + formato custom, oppure
   script Python di conversione pickle → JSON/bincode
2. **Rigenerare hash tables**: più semplice, calcola SHA256 da FASTA
3. **loci_modes**: calcola al volo dalla distribuzione alleli

**Approccio consigliato**: ignora pickle, rigenera tutto da FASTA.
Lo schema è definito dai file FASTA, il resto è cache.

## Validazione

Per verificare identità risultati:
```bash
# Run Python chewBBACA
chewBBACA.py AlleleCall -i genomes/ -g schema/ -o output_py/

# Run Rust chewBBACA
chewbbacca-rs allele-call -i genomes/ -g schema/ -o output_rs/

# Compare CRC32 hashed profiles
diff output_py/results_alleles.tsv output_rs/results_alleles.tsv
```

## Step-by-step di implementazione

1. **Settimana 1**: CLI + schema reader + traduzione + dedup + exact matching
   - Obiettivo: produrre EXC/INF da exact matching
2. **Settimana 2**: SW alignment (parasail) + BSR + clustering + classificazione
   - Obiettivo: pipeline completo con tutte le classi
3. **Settimana 3**: RepDet iterativo + output files + validazione vs Python
   - Obiettivo: matrice identica a chewBBACA Python
