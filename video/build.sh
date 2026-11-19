#!/usr/bin/env bash
# Render every scene and join them into video/output/diffdodree_explainer.mp4 (1920x1080, 30 fps).
# Usage: bash video/build.sh [preview]   (preview = 960x540 at 15 fps)
set -euo pipefail
cd "$(dirname "$0")"
source ~/.venvs/manim-ko/bin/activate
MEDIA=${MEDIA_DIR:-$HOME/.cache/diffdodree_video_media}
if [ "${1:-}" = "preview" ]; then RES=960,540; FPS=15; OUT=output/diffdodree_explainer_preview.mp4
else RES=1920,1080; FPS=30; OUT=output/diffdodree_explainer.mp4; fi
SCENES=$(python -c "from narration import SCENES; print(' '.join(SCENES))")
python make_audio.py "${VOICE:-F1}" "${SPEED:-1.0}"
mkdir -p output
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
LIST="$TMP/list.txt"
for S in $SCENES; do
  manim -q h -r "$RES" --fps "$FPS" --media_dir "$MEDIA" --disable_caching -o "$S" explainer.py "$S" >/dev/null
  F=$(find "$MEDIA/videos/explainer" -name "$S.mp4" -path "*${RES#*,}p*" | head -1)
  # Manim's audio track ends with the last sound, so pad it to the video length;
  # otherwise the concat drifts and later narration plays early and gets cut.
  V=$(ffprobe -v error -select_streams v:0 -show_entries stream=duration -of csv=p=0 "$F")
  ffmpeg -y -loglevel error -i "$F" -af "apad=whole_dur=$V" -t "$V" -c:v copy -c:a pcm_s16le -ar 48000 "$TMP/$S.mov"
  echo "file '$TMP/$S.mov'" >> "$LIST"; echo "rendered $S ($V s)"
done
# join, re-encode, and bring the loudness to −16 LUFS
ffmpeg -y -loglevel error -f concat -safe 0 -i "$LIST" -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
  -af "loudnorm=I=-16:TP=-1.5:LRA=11" -c:a aac -b:a 192k -ar 48000 -movflags +faststart "$OUT"
ffprobe -v error -show_entries format=duration -of csv=p=0 "$OUT" | xargs printf "wrote $OUT (%.1f s)\n"
