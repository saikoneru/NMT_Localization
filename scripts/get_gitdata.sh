#!/bin/bash
lg=$1
ROOT="../data/"
out=$ROOT/"po_data_${lg}/"
parse_dir=$ROOT/"gitdata_parsed_"${lg}/
mkdir -p $out
mkdir -p $parse_dir
src search -stream -json "context:global type:path .*/${lg}.${lg}\.po$|.*/${lg}\.po$ count:all patterntype:regexp case:no" > $out/po_jsons.txt
chmod +rwx $out/po_jsons.txt
cat $out/po_jsons.txt | ./json-to-raw-url.sh | grep '.po$' > $out/po_urls.txt # Remove urls with weird file names that are parsed incorrectly
chmod +rwx $out/po_urls.txt
ulimit -n 30000
echo "Parsed query results and created urls"
cat $out/po_urls.txt | python3 gather_downloader.py  -o $out
find $out -type f -name "*.po" > $out/filepaths.txt
echo "Parsing po files, takes time "
python extract_git_data.py --file_paths $out/filepaths.txt --output_path $parse_dir --src_lang "en" --tgt_lang $lg
cat $out/filepaths.txt | xargs dirname > $out/dirpaths.txt
ulimit -n 1024
