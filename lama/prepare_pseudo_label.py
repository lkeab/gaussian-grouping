import os
import sys
from PIL import Image

# Check if the user provided an argument
if len(sys.argv) != 3:
    print("Usage: python3 {} <pseudo_mask_folder> <dataset_folder>".format(sys.argv[0]))
    sys.exit(1)

in_dir = os.path.join(sys.argv[1],'label')
out_dir = os.path.join(sys.argv[2],'images_inpaint_unseen')
train_dir = os.path.join(sys.argv[2],'images')
os.makedirs(out_dir,exist_ok=True)
train_names = sorted(os.listdir(train_dir))
in_names = sorted(os.listdir(in_dir))

assert len(train_names) == len(in_names), "We need to provide pseudo labels for the whole dataset!"

for i,name in enumerate(in_names):
    src_path = os.path.join(in_dir,name)
    image =  Image.open(src_path)
    tgt_name = train_names[i]
    tgt_path = os.path.join(out_dir,tgt_name)
    image.save(tgt_path)
    print("Copy ", src_path, "........to........",tgt_path)

