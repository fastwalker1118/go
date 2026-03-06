#!/bin/bash
# Convert SGF folder to training data (End-to-End)
# Usage: ./convert_dataset.sh -i /path/to/sgfs -o data/my_output_name [-c config.cfg]

set -e

# Default values
CONFIG_FILE="simple_training_config.cfg"
TRAIN_RATIO=0.9
FILES_PER_CHUNK=100000
ONLY_PLAYER=""
VAL_GAME_COUNT=""
ENRICH_CHECKPOINT=""
ENRICH_ALPHA=0.5
ENRICH_BETA=0.5
ENRICH_BATCH_SIZE=256

# Function to print usage
usage() {
    echo "Usage: $0 -i <input_sgf_dir> -o <output_dir> [-c <config_file>] [--train-ratio <0.0-1.0>] [--only-player <name>] [--val-games <count>]"
    echo "  -i, --input             Directory containing .sgf files"
    echo "  -o, --output            Directory to store processed data (e.g., data/dataset_name)"
    echo "  -c, --config            KataGo training config file (default: simple_training_config.cfg)"
    echo "  --train-ratio           Fraction of data to use for training (default: 0.9)"
    echo "  --only-player           Only extract moves played by this player name"
    echo "  --val-games             Fixed number of games to use for validation (overrides --train-ratio)"
    echo "  --enrich-checkpoint     KataGo model checkpoint for soft policy enrichment (optional)"
    echo "  --enrich-alpha          Blend weight for human move vs quality-band (default: 0.5)"
    echo "  --enrich-beta           Deviation weight scale (default: 0.5)"
    echo "  --enrich-batch-size     Batch size for model inference (default: 256)"
    exit 1
}

# Parse arguments
PARSED_ARGS=$(getopt -o i:o:c:h --long input:,output:,config:,train-ratio:,only-player:,val-games:,enrich-checkpoint:,enrich-alpha:,enrich-beta:,enrich-batch-size:,help --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    usage
fi

eval set -- "$PARSED_ARGS"

while true; do
    case "$1" in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --train-ratio)
            TRAIN_RATIO="$2"
            shift 2
            ;;
        --only-player)
            ONLY_PLAYER="$2"
            shift 2
            ;;
        --val-games)
            VAL_GAME_COUNT="$2"
            shift 2
            ;;
        --enrich-checkpoint)
            ENRICH_CHECKPOINT="$2"
            shift 2
            ;;
        --enrich-alpha)
            ENRICH_ALPHA="$2"
            shift 2
            ;;
        --enrich-beta)
            ENRICH_BETA="$2"
            shift 2
            ;;
        --enrich-batch-size)
            ENRICH_BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Programming error"
            exit 3
            ;;
    esac
done

# Validate inputs
if [[ -z "$INPUT_DIR" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    echo "Error: Input and Output directories are required."
    usage
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    exit 1
fi

# Resolve to absolute paths so output locations are unambiguous
INPUT_DIR=$(realpath "$INPUT_DIR")
OUTPUT_DIR=$(realpath "$OUTPUT_DIR")

if [[ ! -f "$CONFIG_FILE" ]]; then
    # Try finding it in the root if not found
    REPO_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
    if [[ -f "$REPO_ROOT/$CONFIG_FILE" ]]; then
        CONFIG_FILE="$REPO_ROOT/$CONFIG_FILE"
    else
        echo "Error: Config file '$CONFIG_FILE' not found."
        exit 1
    fi
fi

# Locate KataGo binary and shuffle script
REPO_ROOT=$(dirname "$(dirname "$(realpath "$0")")")
KATAGO_BIN="$REPO_ROOT/cpp/build/katago"
if [[ ! -f "$KATAGO_BIN" ]]; then
    KATAGO_BIN="$REPO_ROOT/cpp/katago"
fi
SHUFFLE_SCRIPT="$REPO_ROOT/python/shuffle.py"

if [[ ! -f "$KATAGO_BIN" ]]; then
    echo "Error: KataGo binary not found at cpp/build/katago or cpp/katago"
    echo "Please build KataGo first."
    exit 1
fi

echo "======================================================================"
echo "🎯 KataGo Data Conversion Pipeline"
echo "======================================================================"
echo "Input SGFs:   $INPUT_DIR"
echo "Output Dir:   $OUTPUT_DIR"
echo "Config:       $CONFIG_FILE"
echo "Train Ratio:  $TRAIN_RATIO"
if [[ -n "$ENRICH_CHECKPOINT" ]]; then
echo "Enrich Model: $ENRICH_CHECKPOINT"
echo "Enrich Alpha: $ENRICH_ALPHA"
echo "Enrich Beta:  $ENRICH_BETA"
fi
echo "======================================================================"

# ============================================
# STEP 1: Convert SGFs to NPZ
# ============================================
echo ""
echo "Step 1: Converting SGFs to NPZ..."

RAW_DIR="$OUTPUT_DIR/raw_npz"
mkdir -p "$RAW_DIR"

# Count SGFs
TOTAL_SGFS=$(find "$INPUT_DIR" -name "*.sgf" | wc -l)
if [[ "$TOTAL_SGFS" -eq 0 ]]; then
    echo "Error: No SGF files found in $INPUT_DIR"
    exit 1
fi
echo "Found $TOTAL_SGFS SGF files."

# Calculate chunks
NUM_CHUNKS=$(( ($TOTAL_SGFS + $FILES_PER_CHUNK - 1) / $FILES_PER_CHUNK ))
echo "Splitting into $NUM_CHUNKS chunks (~$FILES_PER_CHUNK files each)..."

# Create temp chunk dirs and distribute files
for i in $(seq 0 $(($NUM_CHUNKS - 1))); do
    mkdir -p "$RAW_DIR/chunk_$i"
done

# We use a temp file list to distribute files safely
# Using find and distributing round-robin or linear
echo "Distributing files..."
find "$INPUT_DIR" -name "*.sgf" > "$RAW_DIR/all_sgfs.txt"
split -l $FILES_PER_CHUNK -d --suffix-length=3 "$RAW_DIR/all_sgfs.txt" "$RAW_DIR/filelist_"

CHUNK_IDX=0
for filelist in "$RAW_DIR"/filelist_*; do
    # Skip if we created more filelists than expected (safety)
    if [[ ! -d "$RAW_DIR/chunk_$CHUNK_IDX" ]]; then
        break
    fi
    
    # Link files (use path-based names to avoid duplicate basename collisions)
    INPUT_ABS=$(realpath "$INPUT_DIR")
    while IFS= read -r sgf; do
        SGF_ABS=$(realpath "$sgf")
        REL="${SGF_ABS#$INPUT_ABS/}"
        LINK_NAME="${REL//\//__}"
        ln -s "$SGF_ABS" "$RAW_DIR/chunk_$CHUNK_IDX/$LINK_NAME"
    done < "$filelist"
    rm "$filelist"
    
    CHUNK_IDX=$((CHUNK_IDX + 1))
done
rm "$RAW_DIR/all_sgfs.txt"

# Run conversion on chunks
for i in $(seq 0 $(($NUM_CHUNKS - 1))); do
    echo "Converting chunk $i..."
    
    # Build command args array
    CMD_ARGS=(
        "$KATAGO_BIN" writetrainingdata_simple 
        -config "$CONFIG_FILE" 
        -sgfdir "$RAW_DIR/chunk_$i" 
        -output "$RAW_DIR/data$i.npz" 
        -verbosity 1
    )
    
    if [[ -n "$ONLY_PLAYER" ]]; then
        CMD_ARGS+=(-only-player "$ONLY_PLAYER")
    fi
    
    "${CMD_ARGS[@]}"
    
    # Cleanup chunk dir
    rm -rf "$RAW_DIR/chunk_$i"
done

echo "SGF to NPZ conversion complete."

# ============================================
# STEP 1.5: Enrich NPZ with soft policy (optional)
# Enriches and compresses one file at a time to avoid large intermediate disk use.
# ============================================
if [[ -n "$ENRICH_CHECKPOINT" ]]; then
    echo ""
    echo "Step 1.5: Enriching and compressing NPZ (quality-band, one file at a time)..."
    ENRICH_SCRIPT="$(dirname "$(realpath "$0")")/enrich_npz.py"
    ENRICHED_COMPRESSED_DIR="$OUTPUT_DIR/enriched_npz_compressed"

    python3 "$ENRICH_SCRIPT" \
        --input-dir "$RAW_DIR" \
        --output-dir "$ENRICHED_COMPRESSED_DIR" \
        --checkpoint "$ENRICH_CHECKPOINT" \
        --alpha "$ENRICH_ALPHA" \
        --beta "$ENRICH_BETA" \
        --batch-size "$ENRICH_BATCH_SIZE" \
        --compress-output

    echo "Soft policy enrichment complete (compressed output)."
    RAW_DIR="$ENRICHED_COMPRESSED_DIR"
fi

# ============================================
# STEP 2: Split Train/Val
# ============================================
echo ""
echo "Step 2: Splitting Train/Val..."

RAW_TRAIN="$OUTPUT_DIR/raw_train"
RAW_VAL="$OUTPUT_DIR/raw_val"
mkdir -p "$RAW_TRAIN" "$RAW_VAL"

# Count available NPZ files
NPZ_FILES=("$RAW_DIR"/*.npz)
NUM_NPZ=${#NPZ_FILES[@]}

if [[ "$NUM_NPZ" -eq 0 ]]; then
    echo "Error: No NPZ files produced! (Did the player filter remove all games?)"
    exit 1
fi

if [[ "$NUM_NPZ" -eq 1 ]]; then
    echo "Warning: Only 1 NPZ file produced."
    echo "Note: The shuffle script handles row-based splitting, so we will pass this single file to BOTH train and val."
    echo "The filtering logic will happen during the shuffle step."
    ln -f "${NPZ_FILES[0]}" "$RAW_TRAIN/"
    ln -f "${NPZ_FILES[0]}" "$RAW_VAL/"
    
    # Recalculate positions based on file size if possible?
    # Shuffle.py needs -min-rows. 
    # Let's assume user knows what they are doing with small datasets.
else
    # Determine split point
    if [[ -n "$VAL_GAME_COUNT" ]]; then
        # Heuristic: We don't know exactly how many games are in each NPZ.
        # But we know total games.
        # Let's assume uniform distribution.
        # If we want 20 games for val, and we have 285 total.
        # 20 / 285 ~ 7%.
        # If we have 14 files, 7% is ~1 file.
        # This is imprecise at file level.
        # Better approach: Just take the LAST file for validation if it exists.
        SPLIT_IDX=$((NUM_NPZ - 1))
        if [[ "$SPLIT_IDX" -lt 1 ]]; then SPLIT_IDX=1; fi
        echo "Using last file for validation (approximate split for small dataset)"
    else
        SPLIT_IDX=$(python3 -c "print(int($NUM_NPZ * $TRAIN_RATIO))")
        if [[ "$SPLIT_IDX" -eq "$NUM_NPZ" ]]; then SPLIT_IDX=$((NUM_NPZ - 1)); fi
    fi
    
    # Ensure at least one file for val if we have multiple
    if [[ "$SPLIT_IDX" -eq "$NUM_NPZ" ]]; then SPLIT_IDX=$((NUM_NPZ - 1)); fi
    if [[ "$SPLIT_IDX" -lt 0 ]]; then SPLIT_IDX=0; fi # Should not happen if >1 file
    
    echo "Using first $SPLIT_IDX files for Training, remaining $((NUM_NPZ - SPLIT_IDX)) files for Validation."
    
    for ((i=0; i<NUM_NPZ; i++)); do
        FILE="${NPZ_FILES[$i]}"
        BASENAME=$(basename "$FILE")
        if [[ $i -lt $SPLIT_IDX ]]; then
            ln -f "$FILE" "$RAW_TRAIN/$BASENAME"
        else
            ln -f "$FILE" "$RAW_VAL/$BASENAME"
        fi
    done
fi

# ============================================
# STEP 3: Shuffle
# ============================================
echo ""
echo "Step 3: Shuffling data..."

SHUF_OUT="$OUTPUT_DIR/shuffled"
SHUF_TRAIN="$SHUF_OUT/train"
SHUF_VAL="$SHUF_OUT/val"
TMP_DIR="$OUTPUT_DIR/tmp_shuffle"

mkdir -p "$SHUF_TRAIN" "$SHUF_VAL" "$TMP_DIR"

# Check if output directories already exist and are not empty
if [ "$(ls -A $SHUF_TRAIN)" ]; then
    echo "Warning: $SHUF_TRAIN is not empty. Clearing it..."
    rm -rf "$SHUF_TRAIN"/*
fi
if [ "$(ls -A $SHUF_VAL)" ]; then
    echo "Warning: $SHUF_VAL is not empty. Clearing it..."
    rm -rf "$SHUF_VAL"/*
fi


# Heuristic for total positions: ~200 per game (filtered by player = 100 per game)
MOVES_PER_GAME=100
if [[ -z "$ONLY_PLAYER" ]]; then
    MOVES_PER_GAME=200
fi

ESTIMATED_POSITIONS=$((TOTAL_SGFS * MOVES_PER_GAME))

if [[ -n "$VAL_GAME_COUNT" ]]; then
    VAL_POS_TARGET=$(($VAL_GAME_COUNT * $MOVES_PER_GAME))
    TRAIN_POS_TARGET=$(($ESTIMATED_POSITIONS - $VAL_POS_TARGET))
else
    TRAIN_POS_TARGET=$(python3 -c "print(int($ESTIMATED_POSITIONS * $TRAIN_RATIO))")
    VAL_POS_TARGET=$(python3 -c "print(int($ESTIMATED_POSITIONS * (1-$TRAIN_RATIO)))")
fi

# Ensure sensible minimums
if [[ "$VAL_POS_TARGET" -lt 1000 ]]; then VAL_POS_TARGET=1000; fi
if [[ "$TRAIN_POS_TARGET" -lt 1000 ]]; then TRAIN_POS_TARGET=1000; fi

echo "Estimated Total Positions: $ESTIMATED_POSITIONS"
echo "Target Train Positions:    $TRAIN_POS_TARGET"
echo "Target Val Positions:      $VAL_POS_TARGET"

# Shuffle Train
echo "Shuffling Training Set..."
# Ensure the output directory does NOT exist, as shuffle.py expects to create it
if [ -d "$SHUF_TRAIN" ]; then
    rm -rf "$SHUF_TRAIN"
fi

python3 "$SHUFFLE_SCRIPT" "$RAW_TRAIN" \
    -out-dir "$SHUF_TRAIN" \
    -out-tmp-dir "$TMP_DIR/train" \
    -batch-size 64 \
    -num-processes 1 \
    -worker-group-size 20000 \
    -min-rows 1 \
    -keep-target-rows "$TRAIN_POS_TARGET" \
    -expand-window-per-row 0.4 \
    -taper-window-exponent 0.65 \
    -output-npz

# Shuffle Val
echo "Shuffling Validation Set..."
if [ -d "$SHUF_VAL" ]; then
    rm -rf "$SHUF_VAL"
fi

python3 "$SHUFFLE_SCRIPT" "$RAW_VAL" \
    -out-dir "$SHUF_VAL" \
    -out-tmp-dir "$TMP_DIR/val" \
    -batch-size 64 \
    -num-processes 1 \
    -worker-group-size 20000 \
    -min-rows 1 \
    -keep-target-rows "$VAL_POS_TARGET" \
    -expand-window-per-row 0.4 \
    -taper-window-exponent 0.65 \
    -output-npz

echo "======================================================================"
echo "✅ Conversion Complete!"
echo "======================================================================"
echo "Data is ready at: $SHUF_OUT"
echo "  Train: $SHUF_TRAIN"
echo "  Val:   $SHUF_VAL"
echo ""
echo "To train, run:"
echo "python training/train_with_freezing.py -datadir $SHUF_OUT ..."

