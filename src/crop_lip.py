from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys


def main(youtube_id = '1'):
    of_aligneddir = Path('/home/jphacks/LipNet-JP/data/processed/{0}/{0}_aligned/'.format(youtube_id))
    aligned_lm_path = Path('/home/jphacks/LipNet-JP/data/processed2/{0}/{0}_aligned.csv'.format(youtube_id))
    cropped_outdir = Path('/home/jphacks/LipNet-JP/data/processed2/{0}/{0}_aligned_aligned_cropped/'.format(youtube_id))
    cropped_outdir.mkdir(exist_ok=True)

    lm_df = pd.read_csv(aligned_lm_path)

    lip_idexies = list(range(48, 68))
    lip_x_list = [' x_{}'.format(i) for i in lip_idexies]
    lip_y_list = [' y_{}'.format(i) for i in lip_idexies]
    lm_df[' x_lip_mean'] = lm_df[lip_x_list].mean(axis=1)
    lm_df[' y_lip_mean'] = lm_df[lip_y_list].mean(axis=1)

    for i, row in tqdm(lm_df.iterrows()):
        frame = i + 1
        aligned_imgpath = of_aligneddir / 'frame_det_00_{:06d}.bmp'.format(frame)
        assert aligned_imgpath.exists(), aligned_imgpath
        
        img = cv2.imread(str(aligned_imgpath))
        x_lip_mean = round(row[' x_lip_mean'])
        y_lip_mean = round(row[' y_lip_mean'])
        
        w = 160
        h = 80
        
        w_size, h_size = img.shape[0], img.shape[1]
        
        crop_img = img[max(0, int(y_lip_mean - h/2)):min(h_size, int(y_lip_mean + h/2)), 
                    max(0, int(x_lip_mean - w/2)):min(w_size, int(x_lip_mean + w/2))]

        cv2.imwrite(str(cropped_outdir / 'frame_det_00_{:06d}.bmp'.format(frame)), crop_img)


if __name__ == '__main__':
    youtube_id = sys.argv[1]
    print('Processing against:', youtube_id)
    main(youtube_id)