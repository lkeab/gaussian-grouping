# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

We provide dataset format and custom dataset preparation in the [training doc](./train.md). Here we introduce the LERF-Mask dataset proposed in our paper and its evaluation.


## 1. LERF-Mask dataset
You can download LERF-Mask dataset from [this hugging-face link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data/lerf_mask). Test set of LERF-Mask dataset includes 2-4 novel view images. The mask annotations are saved in `test_mask` folder. The name of each mask image corresponds to the input text-prompt.

```
lerf_mask
|____figurines
| |____distorted
| |____images
| |____images_train
| |____object_mask
| |____sparse
| |____stereo
| |____test_mask
|   |____<novel view 0>
|   | |____<text prompt 0>.png
|   | |____...
|   |____<novel view 1>
|   | |____<text prompt 0>.png
|   | |____...
|____ramen
| |____...
|____teatime
| |____...
```

## 2. Render mask with text-prompt
For semantic information of each mask output, since SAM masks are class-agnostic, we can use a vision-language detector's mask output, for example [grounded-sam](https://github.com/IDEA-Research/Grounded-Segment-Anything), to match our mask to give semantic information.

We test our segmentation with a simple strategy using grounded-sam on the first frame for text-prompt. You can use the following command with the provided checkpoints on [hugging face](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/checkpoint) or your own training result. In the future we can also explore better detectors and prompt formats.

```
python render_lerf_mask.py -m output/lerf_pretrain/figurines --skip_train
python render_lerf_mask.py -m output/lerf_pretrain/ramen --skip_train
python render_lerf_mask.py -m output/lerf_pretrain/teatime --skip_train
```




## 3. LERF-Mask evaluation
We provide our result on [hugging face](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/result). We also provide a script for evaluating IoU and Boundary-IoU. You can change the output path to your output folder and run the script.

For example,
```
python script/eval_lerf_mask.py figurines
python script/eval_lerf_mask.py ramen
python script/eval_lerf_mask.py teatime
```

