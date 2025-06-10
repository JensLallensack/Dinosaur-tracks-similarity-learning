from syntracks import syntracks
import os
import glob


current_directory = os.getcwd()
ods_files = glob.glob(os.path.join(current_directory, '*.ods'))

# syntracks(ods_files[0],n=1,magnitude=1.1,include_standard=True)

for i in range(len(ods_files)):
    print("Generating ",ods_files[i])
    #syntracks(ods_files[i],n=50,magnitude=1,include_standard=True)
    syntracks(ods_files[i],n=5,magnitude=2,include_standard=False)  #set n to specify number of outlines per class
