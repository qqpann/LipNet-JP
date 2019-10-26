from pathlib import Path
import shutil
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sys


def main(youtube_id: str = '1'):
    workdir = Path('/home/jphacks/LipNet-JP/')
    txtpath = workdir / 'data/align' / 'output{}.align'.format(youtube_id)
    aligned_lm_path = Path('/home/jphacks/LipNet-JP/data/processed2/{0}/{0}_aligned.csv'.format(youtube_id))
    lm_path = Path('/home/jphacks/LipNet-JP/data/processed/{0}/{0}.csv'.format(youtube_id))
    croppeddir = Path('/home/jphacks/LipNet-JP/data/processed2/{0}/{0}_aligned_aligned_cropped'.format(youtube_id))
    assert croppeddir.exists()
    datadir = Path('/home/jphacks/LipNet-JP/data')
    videodir = datadir / 'lip_video'
    txtdir = datadir / 'align_txt'

    with open(txtpath, 'r') as f:
        txt = json.load(f)

    aligned_lm_df = pd.read_csv(str(aligned_lm_path))
    # Get timestamp from frame
    aligned_lm_df[' timestamp'] = (aligned_lm_df['frame'] - 1) * (1/30)

    spk = 's{}'.format(youtube_id)

    train_rate = 0.8
    all_len = len(txt)
    train_f = open(str(datadir / 'jp_train.txt'), 'w')
    val_f = open(str(datadir / 'jp_val.txt'), 'w')
    for idx, word in tqdm(enumerate(txt)):
        start0 = 10000000
        end0 = 0
        for w in word:
            start = w['start']
            end = w['end']
            start0 = min(start0, start)
            end0 = max(end0, end)

        video = aligned_lm_df[(aligned_lm_df[' timestamp'] > start0) & (aligned_lm_df[' timestamp'] < end0)]
        # TODO: exclude rotated faces
    #     print(video.frame.values)
    #     print(word)
        
        frames_dir = videodir / spk / 'video/mpg_6000' / '{:06d}'.format(idx)
        frames_dir.mkdir(parents=True, exist_ok=True)
    #     print(videodir / spk / 'video/mpg_6000' / '{:06d}'.format(idx))
    #     print(txtdir / spk / 'align' / '{:06d}.align'.format(idx))
        
        # Copy video frames
        for frame in video.frame.values:
            frame_path = croppeddir / 'frame_det_00_{:06d}.bmp'.format(frame)
            assert frame_path.exists()
            frame_outpath = frames_dir / 'frame_det_00_{:06d}.bmp'.format(frame)
            shutil.copy(str(frame_path), str(frame_outpath))   
            
            f = train_f if (idx < all_len * train_rate) else val_f
            f.write('{}\n'.format('{}/video/mpg_6000/{:06d}'.format(spk, idx)))
        # Write txt file
        (txtdir / spk / 'align').mkdir(parents=True, exist_ok=True)
        with open(str(txtdir / spk / 'align' / '{:06d}.align'.format(idx)), 'w') as f:
            for w in word:
                start = w['start']
                end = w['end']
                start0 = min(start0, start)
                end0 = max(end0, end)
                f.write('{} {} {}\n'.format(start, end, w['word']))


if __name__ == '__main__':
    youtube_id = sys.argv[1]
    main(youtube_id)