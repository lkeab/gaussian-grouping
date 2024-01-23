# Copyright (C) 2023, Gaussian-Grouping
# Gaussian-Grouping research group, https://github.com/lkeab/gaussian-grouping
# All rights reserved.
#
# ------------------------------------------------------------------------
# Modified from codes in Gaussian-Splatting 
# GRAPHDECO research group, https://team.inria.fr/graphdeco

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import cv2

from ext.grounded_sam import grouned_sam_output, load_model_hf, select_obj_ioa
from segment_anything import sam_model_registry, SamPredictor

from render import feature_to_rgb, visualize_obj


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT, threshold=0.2):
    render_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "gt")
    colormask_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "objects_feature16")
    pred_obj_path = os.path.join(model_path, name, "ours_{}_text".format(iteration), "test_mask")
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(colormask_path, exist_ok=True)
    makedirs(pred_obj_path, exist_ok=True)

    # Use Grounded-SAM on the first frame
    results0 = render(views[0], gaussians, pipeline, background)
    rendering0 = results0["render"]
    rendering_obj0 = results0["render_object"]
    logits = classifier(rendering_obj0)
    pred_obj = torch.argmax(logits,dim=0)

    image = (rendering0.permute(1,2,0) * 255).cpu().numpy().astype('uint8')
    text_mask, annotated_frame_with_mask = grouned_sam_output(groundingdino_model, sam_predictor, TEXT_PROMPT, image)
    Image.fromarray(annotated_frame_with_mask).save(os.path.join(render_path[:-8],'grounded-sam---'+TEXT_PROMPT+'.png'))
    selected_obj_ids = select_obj_ioa(pred_obj, text_mask)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        pred_obj_img_path = os.path.join(pred_obj_path,str(idx))
        makedirs(pred_obj_img_path, exist_ok=True)

        results = render(view, gaussians, pipeline, background)
        rendering = results["render"]
        rendering_obj = results["render_object"]
        logits = classifier(rendering_obj)

        if len(selected_obj_ids) > 0:
            prob = torch.softmax(logits,dim=0)

            pred_obj_mask = prob[selected_obj_ids, :, :] > threshold
            pred_obj_mask = pred_obj_mask.any(dim=0)
            pred_obj_mask = (pred_obj_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
        else:
            pred_obj_mask = torch.zeros_like(view.objects).cpu().numpy()

        gt_objects = view.objects
        gt_rgb_mask = visualize_obj(gt_objects.cpu().numpy().astype(np.uint8))

        rgb_mask = feature_to_rgb(rendering_obj)
        Image.fromarray(rgb_mask).save(os.path.join(colormask_path, '{0:05d}'.format(idx) + ".png"))
        Image.fromarray(pred_obj_mask).save(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        print(os.path.join(pred_obj_img_path, TEXT_PROMPT + ".png"))
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))



def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        num_classes = dataset.num_classes
        print("Num classes: ",num_classes)

        classifier = torch.nn.Conv2d(gaussians.num_objects, num_classes, kernel_size=1)
        classifier.cuda()
        classifier.load_state_dict(torch.load(os.path.join(dataset.model_path,"point_cloud","iteration_"+str(scene.loaded_iter),"classifier.pth")))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # grounding-dino
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
        groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

        # sam-hq
        sam_checkpoint = 'Tracking-Anything-with-DEVA/saves/sam_vit_h_4b8939.pth'
        sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        # Text prompt
        if 'figurines' in dataset.model_path:
            positive_input = "green apple;green toy chair;old camera;porcelain hand;red apple;red toy chair;rubber duck with red hat"
        elif 'ramen' in dataset.model_path:
            positive_input = "chopsticks;egg;glass of water;pork belly;wavy noodles in bowl;yellow bowl"
        elif 'teatime' in dataset.model_path:
            positive_input = "apple;bag of cookies;coffee mug;cookies on a plate;paper napkin;plate;sheep;spoon handle;stuffed bear;tea in a glass"
        else:
            raise NotImplementedError   # You can provide your text prompt here
        
        positives = positive_input.split(";")
        print("Text prompts:    ", positives)

        for TEXT_PROMPT in positives:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT)
            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, classifier, groundingdino_model, sam_predictor, TEXT_PROMPT)


             

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)