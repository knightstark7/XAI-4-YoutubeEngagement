import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
EMBEDDING_DIR = '/mnt/d/Thesis/Prep/SnapUGC/embedding'
OUTPUT_DIR = '/mnt/d/Thesis/Prep/SnapUGC/embedding/agg'

# create separate file for each video
video_embedding_dir = glob.glob(os.path.join(EMBEDDING_DIR, 'video_embedding', '*'))
audio_embedding_dir = glob.glob(os.path.join(EMBEDDING_DIR, 'audio_embedding', '*'))
title_embedding_dir = glob.glob(os.path.join(EMBEDDING_DIR, 'title_embedding', '*'))

for bi in tqdm(range(len(video_embedding_dir)), desc='Aggregating batches of embeddings'):
    video_df = pd.read_parquet(video_embedding_dir[bi])
    audio_df = pd.read_parquet(audio_embedding_dir[bi])
    title_df = pd.read_parquet(title_embedding_dir[bi])

    for i, row in video_df.iterrows():
        vid_em = np.array(row['embedding_video'])
        aud_em = np.array(audio_df.iloc[i]['embedding_audio'])
        title_em = np.array(title_df.iloc[i]['embedding_title'])
        np.savez_compressed(os.path.join(OUTPUT_DIR, row['Id']) + '.npz', 
                            video_embedding=vid_em, 
                            audio_embedding=aud_em, 
                            title_embedding=title_em)
