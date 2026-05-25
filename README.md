folder="/path/to/files"
out_dir="/path/to/matched_txt"

mkdir -p "$out_dir"

for png in "$folder"/*.png; do
    base="$(basename "$png" .png)"
    [ -f "$folder/$base.txt" ] && cp "$folder/$base.txt" "$out_dir/"
done
