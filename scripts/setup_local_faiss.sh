#!/bin/bash
# Setup script to copy FAISS index to local storage for faster loading
# This significantly improves startup time on NFS-based systems

set -e

SOURCE_INDEX="${FAISS_INDEX_PATH:-faiss_index.bin}"
LOCAL_DIR="/tmp/ml-pipeline-cache"
LOCAL_INDEX="$LOCAL_DIR/faiss_index.bin"

echo "=== FAISS Local Cache Setup ==="
echo "Source: $SOURCE_INDEX"
echo "Target: $LOCAL_INDEX"

# Check if source exists
if [[ ! -f "$SOURCE_INDEX" ]]; then
    echo "ERROR: Source index not found: $SOURCE_INDEX"
    exit 1
fi

# Check available space in /tmp
REQUIRED_MB=$(( $(stat -c%s "$SOURCE_INDEX") / 1024 / 1024 + 100 ))
AVAILABLE_MB=$(df -m /tmp | tail -1 | awk '{print $4}')

echo "Required: ${REQUIRED_MB} MB"
echo "Available in /tmp: ${AVAILABLE_MB} MB"

if [[ $AVAILABLE_MB -lt $REQUIRED_MB ]]; then
    echo "WARNING: Not enough space in /tmp for local copy"
    echo "Falling back to mmap mode instead"
    export FAISS_USE_MMAP=true
    echo "Set FAISS_USE_MMAP=true"
    exit 0
fi

# Create local directory
mkdir -p "$LOCAL_DIR"

# Check if already cached (compare size and mtime)
if [[ -f "$LOCAL_INDEX" ]]; then
    SOURCE_SIZE=$(stat -c%s "$SOURCE_INDEX")
    LOCAL_SIZE=$(stat -c%s "$LOCAL_INDEX" 2>/dev/null || echo 0)
    
    if [[ "$SOURCE_SIZE" == "$LOCAL_SIZE" ]]; then
        echo "Index already cached locally (size matches)"
        echo "export FAISS_INDEX_PATH=$LOCAL_INDEX"
        exit 0
    else
        echo "Local cache outdated, refreshing..."
        rm -f "$LOCAL_INDEX"
    fi
fi

# Copy with progress
echo "Copying index to local storage..."
echo "This may take a few minutes for large indexes..."

# Use pv if available for progress, otherwise dd
if command -v pv &> /dev/null; then
    pv "$SOURCE_INDEX" > "$LOCAL_INDEX"
else
    dd if="$SOURCE_INDEX" of="$LOCAL_INDEX" bs=4M status=progress
fi

echo ""
echo "=== Setup Complete ==="
echo "Local index: $LOCAL_INDEX"
echo ""
echo "To use the local index, run:"
echo "  export FAISS_INDEX_PATH=$LOCAL_INDEX"
echo ""
echo "Or source this script:"
echo "  source <(./scripts/setup_local_faiss.sh)"
