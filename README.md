# Gaussian Grouping

> [**Gaussian Grouping: Segment and Edit Anything in 3D Scenes**](https://arxiv.org/abs/xxxx.xxxx)           
> arXiv 2023  
> ETH Zurich

We propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. Refer to our [paper](https://arxiv.org/abs/xxx) for more details. Our code is under preparation, please stay tuned!

<img width="1000" alt="image" src='media/teaser_github_demo.gif'>

# Introduction
The recent Gaussian Splatting achieves high-quality and real-time novel-view synthesis of the 3D scenes. However, it is solely concentrated on the appearance and geometry modeling, while lacking in fine-grained object-level scene understanding. To address this issue, we propose Gaussian Grouping, which extends Gaussian Splatting to jointly reconstruct and segment anything in open-world 3D scenes. We augment each Gaussian with a compact Identity Encoding, allowing the Gaussians to be grouped according to their object instance or stuff membership in the 3D scene. Instead of resorting to expensive 3D labels, we supervise the Identity Encodings during the differentiable rendering by leveraging the 2D mask predictions by SAM, along with introduced 3D spatial consistency regularization Comparing to the implicit NeRF representation, we show that the discrete and grouped 3D Gaussians can reconstruct, segment and edit anything in 3D with high visual quality, fine granularity and efficiency. Based on Gaussian Grouping, we further propose a local Gaussian Editing scheme, which shows efficacy in versatile scene editing applications, including 3D object removal, inpainting, colorization and scene recomposition.

<img width="1096" alt="image" src='media/github_method.png'>

# Application Overview
Grouped Gaussians after training. Each group represents a specific instance / stuff of the 3D scene and can be fully decoupled.
<img width="1096" alt="image" src='media/editing_operation.png'>

## 3D Object Removal
Our Gaussian Grouping can remove the large-scale objects on the Tanks & Temples dataset, from the whole 3D scene with greatly reduced artifacts.

https://github.com/lkeab/gaussian-grouping/assets/17427852/f3b0f964-a610-49ab-8332-f2caa64fbf45

## 3D Object Inpainting
Comparison on 3D object inpainting cases, where SPIn-NeRF requires 5h training while our method with better inpainting quality only needs 1 hour training and 20 minutes tuning.

https://github.com/lkeab/gaussian-grouping/assets/17427852/9f5050da-6a50-4a5f-a755-3bdc55eab1bc

https://github.com/lkeab/gaussian-grouping/assets/17427852/3ed0203c-0047-4333-8bf0-0c10f5a078d1

## 3D Object Style Transfer
Comparison on 3D object style transfer cases, Our Gaussian Grouping produces more coherent and natural transfer results across views, with faithfully preserved background.

https://github.com/lkeab/gaussian-grouping/assets/17427852/2f00eab5-590b-4295-bb1c-2076acc63d4a

## 3D Open-world Segmentation
Our Gaussian Grouping approach jointly reconstructs and segments anything in full open-world 3D scenes. The masks predicted by Gaussian Group contains much sharp and accurate boundary than LERF.

https://github.com/lkeab/gaussian-grouping/assets/17427852/d972f552-cd89-4dc0-8953-2cde9a438192

## 3D Multi-Object Editing
Our Gaussian Grouping approach jointly reconstructs and segments anything in full open-world 3D scenes. Then we concurrently perform 3D object editing for several objects.

https://github.com/lkeab/gaussian-grouping/assets/17427852/0f01aeb6-8c6d-4ede-9b71-4d2f059459b9


Citation
---------------
If you find Gaussian Grouping useful in your research or refer to the provided baseline results, please star :star: this repository and consider citing :pencil::
```
@article{gaussian_grouping,
  title={Gaussian Grouping: Segment and Edit Anything in 3D Scenes},
  author={Ye, Mingqiao and Danelljan, Martin and Yu, Fisher and Ke, Lei},
  journal={arXiv preprint arXiv:xxx.xxx},
  year={2023}
}
```
