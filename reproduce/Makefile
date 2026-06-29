# Makefile — full reproduction of the chewcall paper, with a convergence gate.
#
#   make help        list targets
#   make all         build -> run chewcall + chewBBACA refs + FDA -> tables/figures
#   make verify      assert chewcall and chewBBACA converge (cell-by-cell CRC32)
#   make brute       demo: chewcall --brute-residual on Lm, then verify convergence
#
# All paths come from config.sh; override any of them in the environment, e.g.
#   make BENCH_BASE=/data/01_beone CPU=16 all
#
# chewBBACA reference runs need the conda envs named in config.sh
# (chewbbaca_354 / chewbbaca_3310).

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

REPRO_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO_ROOT := $(abspath $(REPRO_DIR)/../..)
OUT_TEX   := $(REPRO_DIR)/output
PY        := python3
DATASETS  := lm se ec cj

.PHONY: help all build chewcall refs fda tables figures supp supp-ablation verify brute \
        remedy pareto determinism gpu ext-table extensions repro-all check-inputs clean

help:
	@echo "chewcall paper reproduction"
	@echo " core:"
	@echo "  build       build chewcall + constructive_remedy + schema_audit_pareto"
	@echo "  chewcall    run chewcall on the 8 BeONE datasets (shcds + e2e x3 reps)"
	@echo "  refs        run chewBBACA v3.5.4 and v3.3.10 reference calls"
	@echo "  fda         run chewcall + chewBBACA on the 4 FDA outbreak datasets"
	@echo "  tables      regenerate paper Tables 1-5,7 into $(OUT_TEX)/"
	@echo "  figures     regenerate Figure 2 into $(OUT_TEX)/"
	@echo "  supp        regenerate Supplementary S2/S3/S4 into $(OUT_TEX)/"
	@echo "  verify      convergence gate: chewcall vs chewBBACA, all datasets"
	@echo " algorithmic extensions (script set 10-15):"
	@echo "  remedy      filter-safety audit + constructive remedy, all 4 organisms"
	@echo "  pareto      (k,w,tau,kappa) Pareto sweep, all 4 organisms"
	@echo "  determinism Lm hash vs lexicographic minimizer ordering (cell diff)"
	@echo "  gpu         Lm GPU vs CPU concordance (cell diff)"
	@echo "  brute       chewcall --brute-residual on Lm + convergence check"
	@echo "  ext-table   render Table tab:filter_safety from the remedy summary"
	@echo "  extensions  remedy + pareto + determinism + gpu + ext-table"
	@echo "  supp-ablation minimizer ablation table (Suppl tab:ablation)"
	@echo " everything:"
	@echo "  check-inputs report which tools / data / conda envs are present"
	@echo "  all         core pipeline (build -> chewcall -> refs -> fda -> tables ...)"
	@echo "  repro-all   all + extensions  (total reproduction)"

# ---- Input doctor: what is present, what is missing ----------------
check-inputs:
	@cd $(REPRO_DIR) && source config_extensions.sh
	echo "== binaries =="
	for b in "$$CHEWCALL_BIN" "$$CONSTRUCTIVE_REMEDY_BIN" "$$PARETO_BIN" "$$SCHEMA_AUDIT_BIN"; do
	  [ -x "$$b" ] && echo "  OK   $$b" || echo "  MISS $$b   (run 'make build')"; done
	echo "== libraries =="
	[ -e "$$PARASAIL_LIB/libparasail.so" ] && echo "  OK   parasail $$PARASAIL_LIB" || echo "  MISS parasail at $$PARASAIL_LIB"
	[ -d "$$CUDA_LIB" ] && echo "  OK   CUDA $$CUDA_LIB" || echo "  --   CUDA not found (GPU target unavailable)"
	echo "== conda envs (chewBBACA references) =="
	for e in chewbbaca_354 chewbbaca_3310; do
	  conda env list 2>/dev/null | grep -q "$$e" && echo "  OK   $$e" || echo "  MISS $$e   (needed for 'make refs')"; done
	echo "== per-organism schema / genomes / CDS =="
	for org in $$ORGANISMS; do
	  s=$$(audit_schema_for "$$org")
	  [ -d "$$s/short" ] && echo "  OK   $$org schema  $$s" || echo "  MISS $$org schema  $$s"; done
	[ -d "$$GENOMES_lm" ] && echo "  OK   lm genomes  $$GENOMES_lm" || echo "  MISS lm genomes (determinism/gpu)"
	[ -d "$$CDS_lm" ]     && echo "  OK   lm CDS      $$CDS_lm"     || echo "  MISS lm CDS (determinism/gpu)"
	echo "(core targets all/chewcall/refs/fda use config.sh BENCH_BASE staging; see README.)"

# ---- Supplementary ablation table ----------------------------------
supp-ablation:
	cd $(REPRO_DIR) && bash benchmark_scripts/ablation_minimizer.sh

# ---- 0. Build ------------------------------------------------------
build:
	cd $(REPO_ROOT) && RUSTFLAGS="-C target-cpu=native" cargo build --release
	@echo "Built chewcall, constructive_remedy, schema_audit_pareto in $(REPO_ROOT)/target/release/"

# ---- 1-3. Run the tools --------------------------------------------
chewcall:
	cd $(REPRO_DIR) && ./01_run_chewcall_BeONE.sh all

refs:
	cd $(REPRO_DIR)
	./02_run_chewbbaca_BeONE.sh 354  all
	./02_run_chewbbaca_BeONE.sh 3310 all

fda:
	cd $(REPRO_DIR) && ./03_run_FDA_outbreak.sh

# ---- 4-9. Tables and figures ---------------------------------------
$(OUT_TEX):
	mkdir -p $(OUT_TEX)

tables: | $(OUT_TEX)
	cd $(REPRO_DIR) && source config.sh
	$(PY) 04_tables_runtime_concordance.py --mode shcds > $(OUT_TEX)/tables_1_2.tex
	$(PY) 04_tables_runtime_concordance.py --mode e2e   > $(OUT_TEX)/tables_3_4.tex
	$(PY) 09_table_cgmlst_split.py                       > $(OUT_TEX)/table5_cgmlst_split.tex
	$(PY) 05_table_clustering.py                         > $(OUT_TEX)/table7_clustering.tex
	@echo "Tables written to $(OUT_TEX)/"

figures: | $(OUT_TEX)
	cd $(REPRO_DIR) && source config.sh
	$(PY) 06_figure2_discordance_anatomy.py > $(OUT_TEX)/figure2.tex
	@echo "Figure 2 written to $(OUT_TEX)/"

supp: | $(OUT_TEX)
	cd $(REPRO_DIR) && source config.sh
	$(PY) 07_supp_S3_classification_transitions.py > $(OUT_TEX)/tables_S3_S4.tex
	@echo "Supplementary tables written to $(OUT_TEX)/"

# ---- Convergence gate ----------------------------------------------
# For each dataset, locate the chewcall and chewBBACA v3.5.4 hashed profiles
# and assert cell-by-cell CRC32 agreement on jointly-callable cells.
verify:
	cd $(REPRO_DIR) && source config.sh
	rc=0
	for ds in $(DATASETS); do
	  cc=$$(find "$$OUT_CHEWCALL/shcds_cons_$${ds}_rep1" -name results_alleles_hashed.tsv 2>/dev/null | head -1)
	  cb=$$(find "$$OUT_C354/shcds_cons_$${ds}_rep1" -name results_alleles_hashed.tsv 2>/dev/null | head -1)
	  if [[ -z "$$cc" || -z "$$cb" ]]; then
	    echo "[$${ds}] SKIP (missing profiles: cc='$$cc' cb='$$cb')"
	    continue
	  fi
	  $(PY) verify_convergence.py "$$cc" "$$cb" --label "$$ds" || rc=1
	done
	exit $$rc

# ---- Brute-residual demo + convergence -----------------------------
brute:
	cd $(REPRO_DIR) && source config.sh
	out="$${OUT_ROOT:-/tmp}/cc_brute_lm"
	mkdir -p "$$out"
	echo ">> chewcall --brute-residual on Lm ($$GENOMES_LM_CONS)"
	LD_LIBRARY_PATH="$$PARASAIL_LIB" "$$CHEWCALL_BIN" \
	  -i "$$GENOMES_LM_CONS" -g "$$SCHEMA_LM" -o "$$out" \
	  --cpu "$${CPU:-8}" --cds-input "$$CDS_LM_CONS" --brute-residual
	cc=$$(find "$$out" -name results_alleles_hashed.tsv | head -1)
	cb=$$(find "$$OUT_C354/lm" -name results_alleles_hashed.tsv 2>/dev/null | head -1)
	if [[ -n "$$cb" ]]; then
	  $(PY) verify_convergence.py "$$cc" "$$cb" --label "lm-brute"
	else
	  echo "chewBBACA v3.5.4 Lm reference not found under $$OUT_C354/lm; run 'make refs' first."
	fi

# ---- Algorithmic extensions (script set 10-15) ---------------------
remedy:
	cd $(REPRO_DIR) && bash 10_schema_filter_safety.sh

pareto:
	cd $(REPRO_DIR) && bash 11_pareto_sweep.sh

determinism:
	cd $(REPRO_DIR) && bash 12_determinism.sh

gpu:
	cd $(REPRO_DIR) && bash 13_gpu_vs_cpu.sh

ext-table: remedy | $(OUT_TEX)
	cd $(REPRO_DIR) && $(PY) 15_table_filter_safety.py \
	  output/extensions/filter_safety/filter_safety_summary.tsv > $(OUT_TEX)/table_filter_safety.tex
	@echo "Wrote $(OUT_TEX)/table_filter_safety.tex"

extensions: build remedy pareto determinism gpu ext-table
	@echo "Algorithmic-extension reproduction complete."

all: build chewcall refs fda tables figures supp
	@echo "Core reproduction complete. Run 'make verify' for the convergence gate."

repro-all: all extensions verify
	@echo "TOTAL reproduction complete (core + extensions + convergence gate)."

clean:
	rm -rf $(OUT_TEX)
