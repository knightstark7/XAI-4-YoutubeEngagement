import pandas as pd
import torch
import argparse
import os
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
import numpy as np

def setup_data(snapugc_path):
    print('Setup data snapugc')
    return pd.read_csv(snapugc_path)

def setup_model():
    print('Setup model VGGish')
    model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.eval().to(device)

def setup_audio_df(snapugc, data_sample_dir):
    print('Setup path audio')
    format_path = os.path.join(data_sample_dir, '{}/{}.mp4')
    video_df = snapugc[['Id', 'Set']].copy()

    video_df['path'] = video_df.apply(lambda row: format_path.format(row.Set, row.Id), axis=1)
    return video_df

def embed_audio_handler(video_path, model):
    try:
        # .mp4 to .wav
        dir_path, file_name = os.path.split(video_path)
        file_name = os.path.splitext(file_name)[0] + '.wav'
        audio_path = os.path.join(dir_path, 'audio', file_name)
        if not os.path.exists(audio_path):
            ffmpeg_extract_audio(video_path, audio_path, bitrate=128, fps=16000, logger=None)
        
        embed = model.forward(audio_path)
        # pad to fixed size
        def pad_to_fixed_size(arr, target_rows=62, target_cols=128):
            assert arr.shape[1] == target_cols, "Each row must be of size 128"
            
            if arr.shape[0] >= target_rows:
                return arr[:target_rows]
            else:
                pad_rows = target_rows - arr.shape[0]
                padding = np.zeros((pad_rows, target_cols), dtype=arr.dtype)
                return np.vstack([arr, padding])
        
        embed = pad_to_fixed_size(embed.cpu().detach().numpy(), target_rows=62, target_cols=128)
        return embed.tolist()
    except Exception as e:
        print(f'Error at {video_path}: {e}')
        return None

def embed_audio(audio_df, model, batch_size, start, end, embedded_data_dir):
    print('Start Embedding...')
    for idx in tqdm(range(start, end, batch_size)): # range(start, end, batch_size):
        batch_end = min(idx+batch_size, len(audio_df))
        
        output_file = os.path.join(embedded_data_dir, f'audio_embedding_{idx}_{batch_end - 1}.parquet')
        if os.path.exists(output_file):
            print(f'File {output_file} already exists, skipping...')
            continue
        
        audios_df_batch = audio_df.iloc[idx:batch_end].copy()
        audios_df_batch['embedding_audio'] = audios_df_batch['path'].apply(
            lambda x: embed_audio_handler(x, model)
        )
        audios_df_batch.to_parquet(output_file, index=False)
        # print(f'Index {idx}->{idx+batch_size} is Done')
    print("All is done")

def main(args):
    snapugc = setup_data(args.snapugc_path)
    model = setup_model()
    audio_df = setup_audio_df(snapugc, args.data_sample_dir)
    embed_audio(audio_df, model, args.batch_size, args.start, args.end, args.embedded_data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio embedding script')
    parser.add_argument("--snapugc_path", type=str, help="Path to the SnapUGC csv file", default='/home/datlinux/XAI-4-YoutubeEngagement/data/SnapUGC/prepped_df.csv')
    parser.add_argument("--embedded_data_dir", type=str, help="Directory for embedded data output", default="/mnt/d/Thesis/Prep/SnapUGC/embedding/audio_embedding")
    parser.add_argument("--data_sample_dir", type=str, help="Directory containing sample data", default="/mnt/d/Thesis/Data/SnapUGC")
    parser.add_argument("--start", type=int, default=0, help='Start index for processing')
    parser.add_argument("--end", type=int, default=None, help='End index for processing')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for processing')
    args = parser.parse_args()
    
    if args.end is None:
        args.end = len(pd.read_csv(args.snapugc_path))
    
    main(args)
