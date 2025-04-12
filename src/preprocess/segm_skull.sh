
path_inputs="/data/aneurysm/external/crop_0.4"
output_dir="/data/aneurysm/external/crop_0.4_totalseg"

mkdir -p $output_dir


# for every file in path_inputs generate an  output file in output_dir

for file in $path_inputs/*; do
    # get the filename without the path
    filename=$(basename "$file")
    echo $filename 
    # get the filename without the extension
    filename_no_ext="${filename%.*}" 
    filename_no_ext="${filename_no_ext%.*}" 
    file_output="$output_dir/$filename_no_ext"
    TotalSegmentator -i $path_inputs/$filename -o $output_dir/$filename_no_ext --roi_subset brain  --device gpu:0
done 