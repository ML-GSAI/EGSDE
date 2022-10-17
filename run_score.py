from tool.eval_score import fid_l2_psnr_ssim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
if __name__ == '__main__':
    task = 'cat2dog'

    if task == 'cat2dog':
        translate_path = 'runs/cat2dog/0'
        source_path = 'data/afhq/val/cat'
        gt_path = 'data/afhq/val/dog'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)

    if task == 'wild2dog':
        translate_path = 'runs/wild2dog/0'
        source_path = 'data/afhq/val/wild'
        gt_path = 'data/afhq/val/dog'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)

    if task == 'male2female':
        translate_path = 'runs/male2female/0'
        source_path = 'data/celeba_hq/val/male'
        gt_path = 'data/celeba_hq/train/female' #'tool/fid_celebahq_female.npz'
        fid_l2_psnr_ssim(task, translate_path, source_path, gt_path)





