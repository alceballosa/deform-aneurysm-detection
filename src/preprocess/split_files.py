# Given a folder with a number of files, this script splits them into subfolders with 100 files each.

# The script takes two arguments: the input folder and the output folder.

import os 
import shutil
import sys 

def split_files(input_folder, output_folder):
    files = os.listdir(input_folder)
    files.sort()
    for i in range(0, len(files), 100):
        subfolder = os.path.join(output_folder, str(i))
        os.makedirs(subfolder, exist_ok=True)
        for j in range(i, min(i + 100, len(files))):
            # move 
            shutil.copy(os.path.join(input_folder, files[j]), subfolder)
            # create a symlink to each file 
            #os.symlink(os.path.join(input_folder, files[j]), os.path.join(subfolder, files[j]))


if __name__ == "__main__":
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    split_files(input_folder, output_folder)