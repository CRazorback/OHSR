# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from models import nnunet
from monai.data import decollate_batch
from utils.nnunet_data_utils import get_loader
from utils.utils import resample_3d
import nibabel as nib
from utils.utils import dice
import argparse

parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='/dataset/dataset0/', type=str, help='dataset directory')
parser.add_argument('--exp_name', default='test1', type=str, help='experiment name')
parser.add_argument('--json_list', default='dataset.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt', type=str, help='pretrained model name')
parser.add_argument('--feature_size', default=48, type=int, help='feature size')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=14, type=int, help='number of output channels')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--spatial_dims', default=3, type=int, help='spatial dimension of input data')
parser.add_argument('--use_checkpoint', action='store_true', help='use gradient checkpointing to save memory')
parser.add_argument('--dataset', default='JHU', type=str, help='dataset name')
parser.add_argument('--fold', default=0, type=int, help='fold number')


def main():
    args = parser.parse_args()
    args.test_mode = False
    args.use_normal_dataset = True
    args.batch_size = 1
    output_directory = './outputs/'+args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    _, val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = nnunet.UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=args.out_channels,
            strides=(2, 2, 2, 2),
            )
    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []
        dice_list_organ = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch['label_meta_dict']['affine'][0].numpy()
            _, _, h, w, d = val_labels.shape
            target_shape = (h, w, d)
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            print("Inference on case {}".format(img_name))
            if os.path.exists(os.path.join(output_directory, img_name)):
                print("Already exists, skip")
                continue
            val_outputs = sliding_window_inference(val_inputs,
                                                   (args.roi_x,
                                                    args.roi_y,
                                                    args.roi_z),
                                                   args.sw_batch_size,
                                                   model,
                                                   overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs[:,:], 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint16)[0]
            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            dice_list_sub = []
            for i in range(1, args.out_channels):
                organ_Dice = dice(val_outputs == i, val_labels == i)
                print("Dice of {}: {}".format(i, organ_Dice))
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("Mean Organ Dice: {}".format(mean_dice))
            dice_list_case.append(mean_dice)
            dice_list_organ.append(dice_list_sub)
            nib.save(nib.Nifti1Image(val_outputs.astype(np.uint16), original_affine),
                     os.path.join(output_directory, img_name))
            nib.save(nib.Nifti1Image(val_labels.astype(np.uint16), original_affine),
                     os.path.join(output_directory, img_name.replace('.nii.gz', '_label.nii.gz')))

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
        np.save(os.path.join(pretrained_dir, 'results.npy'), np.array(dice_list_organ))

if __name__ == '__main__':
    main()
