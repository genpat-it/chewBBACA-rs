#!/bin/bash
# Toggle sort bug fix in chewBBACA allele_call.py
# Usage: toggle_sort_fix.sh [original|fixed]

ALLELE_CALL="/home/IZSNT/a.deruvo/miniconda3/envs/chewbbaca_original/lib/python3.11/site-packages/CHEWBBACA/AlleleCall/allele_call.py"

case "${1:-}" in
    original)
        # Restore the bug: sort by x[5] (slen) instead of x[6] (score)
        sed -i 's/lambda x: int(x\[6\])/lambda x: int(x[5])/' "$ALLELE_CALL"
        echo "SET: c3_original (sort by x[5] = slen, BUGGY)"
        grep -n "lambda x: int(x\[" "$ALLELE_CALL"
        ;;
    fixed)
        # Apply the fix: sort by x[6] (score)
        sed -i 's/lambda x: int(x\[5\])/lambda x: int(x[6])/' "$ALLELE_CALL"
        echo "SET: c3_fixed (sort by x[6] = score, CORRECT)"
        grep -n "lambda x: int(x\[" "$ALLELE_CALL"
        ;;
    *)
        echo "Usage: $0 [original|fixed]"
        echo "Current state:"
        grep -n "lambda x: int(x\[" "$ALLELE_CALL"
        ;;
esac
