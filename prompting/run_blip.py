import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from decord import VideoReader, cpu, gpu
import argparse
from glob import glob
from tqdm import tqdm
import os
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='/gallery_louvre/dayoon.ko/dataset/videohumor/*', type=str)
    parser.add_argument('--num_video', default=10, type=int)
    parser.add_argument('--store_cap', default=False, action='store_true')
    parser.add_argument('--dest_dir', default='/gallery_louvre/dayoon.ko/dataset/blip_intern/blip', type=str)
    parser.add_argument('--num', default=1000, type=int)
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--inter', default=0, type=int)
    return parser.parse_args()
    

def load_model(device):
    """
    Tải mô hình BLIP-2 với xử lý lỗi GPU
    """
    try:
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2", model_type="coco", is_eval=True, device=device
        )
        return model, vis_processors, device
    except RuntimeError as e:
        # Nếu gặp lỗi CUDA out of memory, thử lại với CPU
        if "CUDA out of memory" in str(e) and device == "cuda":
            print("WARNING: CUDA out of memory. Falling back to CPU for model loading.")
            model, vis_processors, _ = load_model_and_preprocess(
                name="blip2", model_type="coco", is_eval=True, device="cpu"
            )
            return model, vis_processors, "cpu"
        else:
            # Nếu là lỗi khác, ném lại ngoại lệ
            raise


def generate_captions(images, model, vis_processors, device, iter=20):
    """
    Tạo caption cho danh sách hình ảnh, tự động chuyển sang CPU nếu GPU bị lỗi
    """
    # Thử xử lý trên device ban đầu (có thể là GPU hoặc CPU)
    try:
        # Chuyển đổi hình ảnh thành tensor và đưa lên device
        processed_images = torch.stack([vis_processors['eval'](img) for img in images]).to(device)
        
        caps = []
        for i in range(iter):
            out = model.generate({'image': processed_images, "prompt": "Question: Who is doing what? Answer:"}, use_nucleus_sampling=True)
            caps.append(out)
        
        # Định dạng lại kết quả
        caps = [[caps[i][b] for i in range(len(caps))] for b in range(len(caps[0]))]
        return caps, device
        
    except RuntimeError as e:
        # Nếu gặp lỗi CUDA out of memory và device hiện tại là CUDA
        if "CUDA out of memory" in str(e) and device == "cuda":
            print("WARNING: CUDA out of memory during caption generation. Falling back to CPU...")
            
            # Chuyển model sang CPU nếu nó đang ở GPU
            model = model.to("cpu")
            
            # Thử lại với CPU
            processed_images = torch.stack([vis_processors['eval'](img) for img in images]).to("cpu")
            
            caps = []
            for i in range(min(iter, 10)):  # Giảm số lượng lặp để tiết kiệm thời gian trên CPU
                out = model.generate({'image': processed_images, "prompt": "Question: Who is doing what? Answer:"}, use_nucleus_sampling=True)
                caps.append(out)
            
            # Định dạng lại kết quả
            caps = [[caps[i][b] for i in range(len(caps))] for b in range(len(caps[0]))]
            return caps, "cpu"
        else:
            # Nếu là lỗi khác, ném lại ngoại lệ
            raise


def caption_video(video_pth, model, vis_processors, device, sample_fps=5.0, num_captions=2):
    """
    Tạo caption cho video, tự động chuyển sang CPU nếu GPU bị lỗi
    Xử lý lỗi khung hình bị hỏng
    """
    # Thử tối đa 3 lần
    max_tries = 3
    for attempt in range(max_tries):
        try:
            # Đọc video với thời gian chờ giữa các lần thử 
            if attempt > 0:
                print(f"Attempt {attempt+1}/{max_tries} to process video...")
                time.sleep(1)  # Chờ 1 giây trước khi thử lại
                
            try:
                vr = VideoReader(video_pth, ctx=cpu(0))
                org_fps = vr.get_avg_fps()
                len_frames = len(vr)
            except Exception as e:
                print(f"Error opening video {os.path.basename(video_pth)}: {e}")
                if attempt < max_tries - 1:
                    continue  # Thử lại việc mở video
                else:
                    return [[f"Error opening video: {os.path.basename(video_pth)}"]]
            
            # Tính toán bước nhảy dựa trên fps
            t_stride = max(1, int(round(float(org_fps)/float(sample_fps))))
            
            # Lấy mẫu các khung hình theo lô để tiết kiệm bộ nhớ
            frame_indices = list(range(0, len_frames, t_stride))
            
            # Nếu có quá nhiều khung hình, chỉ lấy một tập nhỏ hơn
            if len(frame_indices) > 50:
                # Lấy tối đa 50 khung hình phân bố đều
                step = len(frame_indices) // 50
                frame_indices = frame_indices[::step][:50]
            
            # Nếu có nhiều khung hình, phân thành các lô nhỏ hơn
            batch_size = 8  # Giảm batch size để giảm việc sử dụng bộ nhớ
            all_caps = []
            
            for i in range(0, len(frame_indices), batch_size):
                # Lấy một lô khung hình
                batch_indices = frame_indices[i:i+batch_size]
                images = []
                
                for f in batch_indices:
                    # Thử đọc khung hình với xử lý lỗi
                    for retry in range(3):  # Thử tối đa 3 lần cho mỗi khung hình
                        try:
                            # Đọc khung hình bằng indexing thay vì việc lấy toàn bộ khung hình
                            frame = vr[f]
                            if frame is not None:
                                img = Image.fromarray(frame.numpy())
                                # Giảm kích thước để tiết kiệm bộ nhớ
                                if img.width > 480 or img.height > 360:
                                    img.thumbnail((480, 360), Image.LANCZOS)
                                images.append(img)
                                break  # Thoát vòng lặp thử lại nếu thành công
                        except Exception as e:
                            if "Error sending packet" in str(e) or "Check failed" in str(e):
                                print(f"Warning: Could not process frame {f}, retry {retry+1}/3: {str(e)[:100]}...")
                                # Đợi một chút trước khi thử lại
                                time.sleep(0.1)
                            else:
                                print(f"Error processing frame {f}: {str(e)[:100]}...")
                                break  # Không thử lại với các lỗi khác
                
                if not images:
                    print(f"No valid frames extracted from batch {i} to {i+batch_size-1}")
                    continue
                    
                try:
                    # Xử lý lô khung hình hiện tại
                    batch_caps, used_device = generate_captions(images, model, vis_processors, device, num_captions)
                    all_caps.extend(batch_caps)
                    
                    # Cập nhật device nếu đã chuyển
                    if used_device != device:
                        device = used_device
                        print(f"Switched to {device} for processing")
                    
                    # Giải phóng bộ nhớ
                    del images, batch_caps
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e) and device == "cuda":
                        print(f"CUDA out of memory during batch processing. Switching to CPU and reducing quality...")
                        # Chuyển model sang CPU
                        model = model.to("cpu")
                        device = "cpu"
                        
                        # Xử lý lại với cài đặt thấp hơn
                        reduced_images = []
                        for img in images:
                            img_small = img.copy()
                            img_small.thumbnail((240, 180), Image.LANCZOS)
                            reduced_images.append(img_small)
                        
                        batch_caps, _ = generate_captions(reduced_images, model, vis_processors, "cpu", min(2, num_captions))
                        all_caps.extend(batch_caps)
                        
                        # Giải phóng bộ nhớ
                        del images, reduced_images, batch_caps
                    else:
                        raise
            
            # Giải phóng bộ nhớ video
            del vr
            
            if all_caps:
                return all_caps
            else:
                if attempt < max_tries - 1:
                    print("No captions were generated. Retrying...")
                    continue
                else:
                    return [[f"No captions could be generated for: {os.path.basename(video_pth)}"]]
        
        except RuntimeError as e:
            # Kiểm tra nếu đây là lỗi liên quan đến GPU
            if "CUDA out of memory" in str(e) and device == "cuda":
                print(f"CUDA out of memory error during video processing: {e}")
                print("Retrying with minimal settings on CPU...")
                
                try:
                    # Chuyển model sang CPU nếu chưa chuyển
                    model = model.to("cpu")
                    device = "cpu"
                    
                    # Thử lại với các cài đặt tối thiểu
                    vr = VideoReader(video_pth, ctx=cpu(0))
                    len_frames = len(vr)
                    
                    # Chỉ lấy một số ít khung hình
                    indices = [int(len_frames * i / 5) for i in range(5)]
                    
                    # Chuyển đổi thành đối tượng PIL Image với kích thước nhỏ
                    images = []
                    for f in indices:
                        if f < len_frames:
                            try:
                                frame = vr[f]
                                if frame is not None:
                                    img = Image.fromarray(frame.numpy())
                                    # Giảm kích thước nhiều hơn
                                    img.thumbnail((240, 180), Image.LANCZOS)
                                    images.append(img)
                            except Exception as e:
                                print(f"Warning: Could not process frame {f}: {str(e)[:100]}...")
                    
                    # Giải phóng bộ nhớ
                    del vr
                    
                    if images:
                        # Tạo caption trên CPU với ít lặp lại hơn
                        caps, _ = generate_captions(images, model, vis_processors, "cpu", 1)
                        print("Successfully completed minimal caption generation on CPU")
                        return caps
                    else:
                        if attempt < max_tries - 1:
                            print("No valid frames extracted. Retrying...")
                            continue
                        else:
                            return [[f"Could not extract any frames from: {os.path.basename(video_pth)}"]]
                except Exception as inner_e:
                    print(f"Failed even with minimal settings: {str(inner_e)}")
                    if attempt < max_tries - 1:
                        continue
                    else:
                        # Trả về caption giả để tránh lỗi
                        return [[f"Error processing video: {os.path.basename(video_pth)}"]]
            else:
                # Thử lại với lỗi khác nếu còn lần thử
                print(f"Error during processing: {str(e)[:100]}...")
                if attempt < max_tries - 1:
                    continue
                else:
                    # Ném lại ngoại lệ nếu hết số lần thử
                    print(f"Failed after {max_tries} attempts")
                    return [[f"Processing failed after multiple attempts: {os.path.basename(video_pth)}"]]
    
    # Fallback cuối cùng nếu tất cả các lần thử đều thất bại
    return [[f"Could not process video after multiple attempts: {os.path.basename(video_pth)}"]]
        