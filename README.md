# Gaussian Grouping

> [**Gaussian Grouping: Segment and Edit Anything in 3D Scenes**](https://arxiv.org/abs/xxxx.xxxx)           
> arXiv 2023  
> ETH Zurich

We propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. Refer to our [paper](https://arxiv.org/abs/xxx) for more details. Our code is under preparation.

<img width="1000" alt="image" src='media/teaser_github_demo.gif'>

# Introduction
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by SAM, along with introduced 3D spatial consistency regularization Comparing to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization and scene recomposition.

<img width="1096" alt="image" src='media/github_method.png'>

# Application Overview

## 3D Object Removal
Note: For box-prompting-based evaluation, we feed SAM, MobileSAM and our HQ-SAM with the same image/video bounding boxes and adopt the single mask output mode of SAM. 

## 3D Object Inpainting
Note: For box-prompting-based evaluation, we feed SAM, MobileSAM and our HQ-SAM with the same image/video bounding boxes and adopt the single mask output mode of SAM. 

We provide comprehensive performance, model size and speed comparison on SAM variants:
<img width="1096" alt="image" src='figs/sam_variants_comp.png'>

## 3D Object Style Transfer
![backbones](figs/sam_vs_hqsam_backbones.png)
Note: For the COCO dataset, we use a SOTA detector FocalNet-DINO trained on the COCO dataset as our box prompt generator.

## 3D Open-world Segmentation
Note:Using ViT-L backbone. We adopt the SOTA detector Mask2Former trained on the YouTubeVIS 2019 dataset as our video boxes prompt generator while reusing its object association prediction.
![ytvis](figs/ytvis.png)

## 3D Multi-Object Editing
Note: Using ViT-L backbone. We adopt the SOTA model XMem as our video boxes prompt generator while reusing its object association prediction.
![davis](figs/davis.png)


Citation
---------------
If you find HQ-SAM useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
```