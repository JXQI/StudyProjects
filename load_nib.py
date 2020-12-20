"""
function:通过nib加载文件
"""
import nibabel as nib
import numpy as np
from skimage.measure import label, regionprops
import pandas as pd
import os
from tqdm import tqdm
from settings import NII_GA_PRE,NII_GZ_SAVE
'''
function: 生成csv文件
'''
def _make_submission_files(pred, image_id, affine):
    # print(np.unique(pred))
    pred_label = label(pred > 0).astype(np.int16)
    pred_regions = regionprops(pred_label, pred)
    pred_index = [0] + [region.label for region in pred_regions]
    pred_proba = [0.0] + [region.mean_intensity for region in pred_regions]
    # placeholder for label class since classifaction isn't included
    # pred_label_code = [0] + [1] * int(pred_label.max())
    pred_label_code = [0]+[pred[tuple(np.array(i.centroid,dtype="int64").tolist())] for i in pred_regions]
    pred_image = nib.Nifti1Image(pred_label, affine)
    pred_info = pd.DataFrame({
        "public_id": [image_id] * len(pred_index),
        "label_id": pred_index,
        "confidence": pred_proba,
        "label_code": pred_label_code
    })

    return pred_image, pred_info

if __name__=='__main__':
    path=NII_GZ_SAVE #生成的mask对象路径
    pred_dir=NII_GA_PRE #将要保存的预测对象
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    image_path_list = sorted([os.path.join(path, file) for file in os.listdir(path) if "nii" in file])
    image_id_list = [os.path.basename(path).split("-")[0] for path in image_path_list]
    progress = tqdm(total=len(image_id_list))
    pred_info_list = []
    for image_id, image_path in zip(image_id_list, image_path_list):
        print(image_path)
        image=nib.load(image_path)
        image_arr = image.get_fdata().astype(np.uint8)
        image_affine = image.affine
        pred_image,pred_info=_make_submission_files(image_arr,image_id,image_affine)
        pred_info_list.append(pred_info)
        pred_path = os.path.join(pred_dir, f"{image_id}_pred.nii.gz")
        nib.save(pred_image, pred_path)
        progress.update()
    pred_info = pd.concat(pred_info_list, ignore_index=True)
    pred_info.to_csv(os.path.join(pred_dir, "pred_info.csv"),index=False)
