cd /home/hpc4090/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b/

TARGET_DIR="/home/hpc4090/.cache/huggingface/videomme"
mkdir -p $TARGET_DIR/data

for i in {01..20}; do
    echo "=== Extracting videos_chunked_${i}.zip ==="
    if [ -f "videos_chunked_${i}.zip" ]; then
        unzip -o "videos_chunked_${i}.zip" -d "$TARGET_DIR"
        echo "Chunk ${i} done. Current video count:"
        find $TARGET_DIR/data/ -name "*.mp4" 2>/dev/null | wc -l
    else
        echo "File videos_chunked_${i}.zip not found!"
    fi
done

echo "=== Extracting subtitles ==="
unzip -o subtitle.zip -d "$TARGET_DIR"

echo "=== All done! ==="
echo "Total videos:"
ls -1 $TARGET_DIR/data/*.mp4 2>/dev/null | wc -l