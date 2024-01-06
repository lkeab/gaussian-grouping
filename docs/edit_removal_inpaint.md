# Gaussian Grouping: Segment and Edit Anything in 3D Scenes

## 1. 3D Object Removal

### 1.1 Training

First finish training as described in [training doc](./train.md). Save the output file then we can edit on it.


### 1.2 Remove the selected object

You can choose one or more object id(s) for removal and indicate it in the config file.


Example1. Bear dataset
```
bash script/edit_object_removal.sh output/bear config/object_removal/bear.json
```

Example2. Kitchen dataset
```
bash script/edit_object_removal.sh output/mipnerf360/kitchen config/object_removal/mipnerf360/kitchen.json
```


## 2. 3D Object inpainting

For 3D object inpainting, our pipeline includes three steps.

1. Remove the object

2. Inpaint the unseen region (always invisible due to occlusion) in 2D

3. Use 2D inpainting as pseudo label, finetune 3D Gaussians

For your custom datasets, you can follow these three steps to inpaint the object.

For our example datasets, we provide the pseudo labels on [hugging face](https://huggingface.co/mqye/Gaussian-Grouping/tree/main/data) and you can skip the first two steps below and directly finetune 3D Gaussians in **2.3 3D inpaint**. 

### 2.1 (Optional) Unseen mask preparation

First finish training and remove the object you want to inpaint. After removal, we can get the **unseen region mask** for inpainting.

Unseen region mask is the empty region left after removing the object, and we can perform 2D inpainting on it. An example is shown in the bottom of fig8 in our paper. We can obtain the unseen region mask with DEVA. For example,

```bash
cd Tracking-Anything-with-DEVA/

python demo/demo_with_text.py   --chunk_size 4    --img_path ${removal_render_path}  --amp \
  --temporal_setting semionline --size 480   --output ${inpaint_2d_unseen_mask}  \
  --prompt "black blurry hole"
```

You can also try other prompts like "black region" to get the best unseen region mask.



### 2.2 (Optional) 2D inpaint

We follow [SPIN-NeRF](https://github.com/SamsungLabs/SPIn-NeRF) pipeline of 2D guidance for inpainting. We use LAMA to inpaint on 2D images rendered after removing the object. We only need 2D inpainting on RGB and do not need inpainting on depth map.

You can follow LAMA steps of 2D inpainting on RGB image with the images rendered after removal and the unseen region mask. We will provide detail steps on custom datasets later.


### 2.3 3D inpaint
Example1. Bear dataset
```
bash script/edit_object_inpaint.sh output/bear config/object_inpaint/bear.json
```

Example2. Kitchen dataset
```
bash script/edit_object_inpaint.sh output/mipnerf360/kitchen config/object_inpaint/mipnerf360/kitchen.json
```
