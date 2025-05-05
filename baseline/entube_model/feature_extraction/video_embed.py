import argparse
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

def setup_model():
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
    model = torch.nn.Sequential(*(list(model.modules())[1][:-1]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval().to(device)
    return model, device

class PackPathway(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames: torch.Tensor):
        fast_pathway = frames
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long(),
        )
        return [slow_pathway, fast_pathway]

def setup_transform(side_size, mean, std, crop_size, num_frames, alpha):
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(num_frames),
            Lambda(lambda x: x / 255.0),
            NormalizeVideo(mean, std),
            ShortSideScale(size=side_size),
            CenterCropVideo(crop_size),
            PackPathway(alpha)
        ]),
    )

def embed_video_handler(video_path, transform, model, device, clip_duration):
    try:
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start_sec=0, end_sec=clip_duration)
        video_data = transform(video_data)
        inputs = [i.to(device)[None, ...] for i in video_data["video"]]
        embedding = model(inputs)[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Error path {video_path}: {e}")
        return None

def process_videos(video_df, transform, model, device, clip_duration, output_dir, start, end, batch_size):
    for idx in tqdm(range(start, end, batch_size)): # range(start, end, batch_size):
        batch_end = min(idx+batch_size, len(video_df))
        
        output_file = os.path.join(output_dir, f'video_embedding_{idx}_{batch_end - 1}.parquet')
        if os.path.exists(output_file):
            print(f'File {output_file} already exists, skipping...')
            continue
        
        videos_df_batch = video_df.iloc[idx:batch_end].copy()
        videos_df_batch['embedding_video'] = videos_df_batch['path'].apply(
            lambda x: embed_video_handler(x, transform, model, device, clip_duration)
        )
        videos_df_batch.to_parquet(output_file, index=False)
        # print(f'Index {idx}->{idx+batch_size} is Done')

def main(args):
    print('Setup data snapugc')
    snapugc = pd.read_csv(args.snapugc)

    print('Setup model Slowfast')
    model, device = setup_model()

    side_size = 256
    mean = [0.45, 0.45, 0.45]
    std = [0.225, 0.225, 0.225]
    crop_size = 256
    num_frames = 32
    sampling_rate = 2
    frames_per_second = 30
    slowfast_alpha = 4

    transform = setup_transform(side_size, mean, std, crop_size, num_frames, slowfast_alpha)
    clip_duration = (num_frames * sampling_rate) / frames_per_second

    print('Setup path video')
    format_path = os.path.join(args.data_sample_dir, '{}/{}.mp4')
    video_df = snapugc[['Id', 'Set']].copy()

    video_df['path'] = video_df.apply(lambda row: format_path.format(row.Set, row.Id), axis=1)
    print(video_df.columns)

    print('Start Embedding...')
    end = args.end if args.end else len(video_df)
    process_videos(video_df, transform, model, device, clip_duration, args.embedded_data_dir, args.start, end, args.batch_size)

    print("All is done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video embedding process")
    parser.add_argument("--snapugc", type=str, help="Path to the SnapUGC csv file", default='/home/datlinux/XAI-4-YoutubeEngagement/data/SnapUGC/prepped_df.csv')
    parser.add_argument("--embedded_data_dir", type=str, help="Directory for embedded data output", default="/mnt/d/Thesis/Prep/SnapUGC/embedding/video_embedding")
    parser.add_argument("--data_sample_dir", type=str, help="Directory containing sample data", default="/mnt/d/Thesis/Data/SnapUGC")
    parser.add_argument("--start", type=int, default=0, help='Start index for processing')
    parser.add_argument("--end", type=int, default=None, help='End index for processing')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for processing')
    args = parser.parse_args()

    main(args)
