# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

## 1. Prepare associated SAM masks

### 1.1 Pre-converted datasets
We provide converted datasets in our paper, You can use directly train on datasets from [hugging face link](https://huggingface.co/mqye/Gaussian-Grouping/tree/main)

```
data
|____bear
|____lerf
| |____figurines
|____mipnerf360
| |____counter
```


### 1.2 (Optional) Prepare your own datasets
For your custom dataset, you can follow this step to create masks for training. If you want to prepare masks on your own dataset, you will need [DEVA](../Tracking-Anything-with-DEVA/README.md) python environment and checkpoints.


```
<location>
|---input
    |---<image 0>
    |---<image 1>
    |---...
```
Firstly, convert initial camera pose and point cloud with colmap
```
python convert.py -s <location>
```

Then, convert SAM associated object masks. Note that the quality of converted-mask will largely affect the results of 3D segmentation and editing. And getting the mask is very fast. So it is best to adjust the parameters of anything segment first to make the mask as consistent and reasonable as possible from multiple views.

Example1. Bear dataset
```
bash script/prepare_pseudo_label.sh bear 1
```

Example2. figurines dataset
```
bash script/prepare_pseudo_label.sh lerf/figurines 1
```

Example3. counter dataset
```
bash script/prepare_pseudo_label.sh mipnerf360/counter 2
```

## 2. Training and Masks Rendering

For Gaussian Grouping training and segmentation rendering of trained 3D Gaussian Grouping model:

Example1. Bear dataset
```
bash script/train.sh bear 1
```

Example2. figurines dataset
```
bash script/train_lerf.sh lerf/figurines 1
```

Example3. counter dataset
```
bash script/train.sh mipnerf360/counter 2
```

