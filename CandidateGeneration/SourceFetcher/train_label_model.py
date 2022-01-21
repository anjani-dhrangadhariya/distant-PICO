import os
import glob


indir = '/mnt/nas2/results/Results/systematicReview/distant_pico/candidate_generation'
directory = os.fsencode(indir)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    print(filename)

# XXX: doing - Read all the labeled candidates into a dictionary


# Combine UMLS candidate labels into partitions

# Keep the rest stagnant and change UMLS for training PIO * (partition functions) label models

# build the full pipeline before executing it....