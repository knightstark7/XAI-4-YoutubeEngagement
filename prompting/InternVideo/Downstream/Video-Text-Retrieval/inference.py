from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import numpy as np
import random
import os
from metrics import compute_metrics, tensor_text_to_video_metrics, tensor_video_to_text_sim
import time
import argparse
from modules.tokenization_clip import SimpleTokenizer as ClipTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CLIP4Clip
from modules.optimization import BertAdam

from util import parallel_apply
from dataloaders.data_dataloaders import DATALOADER_DICT
import torch.distributed as dist
import subprocess
from ipdb import set_trace
import torch.nn as nn
from scipy.special import softmax
from tqdm import tqdm
import json


def normalize_path(path):
    """Normalize a path to use forward slashes, avoiding double backslash issues on Windows."""
    return os.path.normpath(path).replace('\\', '/')


def get_args(video_dir, caption_filename, segmentation_filename, description='CLIP4Clip on Retrieval Task'):
    # Chuẩn hóa đường dẫn trước khi sử dụng
    video_dir = normalize_path(video_dir)
    
    # Loại bỏ trùng lặp videos/videos nếu có
    if 'videos/videos' in video_dir or 'videos\\videos' in video_dir:
        video_dir = video_dir.replace('videos/videos', 'videos').replace('videos\\videos', 'videos')
        print(f"Đã chuẩn hóa đường dẫn thành: {video_dir}")
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.", default=False)
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.", default=False)
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.", default=True)

    parser.add_argument('--video_dir', type=str, default=video_dir, help='Root directory of videos')
    parser.add_argument('--caption_filename', type=str, default=caption_filename)
    parser.add_argument('--segmentation_filename', type=str, default=segmentation_filename)
    parser.add_argument('--force_cpu', action='store_true', default=False, help='Force using CPU even if CUDA is available')

    parser.add_argument('--num_thread_reader', type=int, default=0, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=1, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')
    
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--resume_model", default=None, type=str, required=False, help="Resume train model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="retrieval", type=str, help="Point the task `retrieval` to finetune.")
    parser.add_argument("--datatype", default="videohumor", type=str, help="Point the dataset to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument("--rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=12, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=4, help="Layer NO. of cross.")

    parser.add_argument('--loose_type', default=True, action='store_true', help="Default using meanP type for retrieval.")
    parser.add_argument('--expand_msrvtt_sentences', action='store_true', help="")

    parser.add_argument('--train_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")
    parser.add_argument('--eval_frame_order', type=int, default=0, choices=[0, 1, 2],
                        help="Frame order, 0: ordinary order; 1: reverse order; 2: random order.")

    parser.add_argument('--freeze_layer_num', type=int, default=0, help="Layer NO. of CLIP need to freeze.")
    parser.add_argument('--slice_framepos', type=int, default=0, choices=[0, 1, 2],
                        help="0: cut from head frames; 1: cut from tail frames; 2: extract frames uniformly.")
    parser.add_argument('--linear_patch', type=str, default="2d", choices=["2d", "3d"],
                        help="linear projection of flattened patches.")
    parser.add_argument('--sim_header', type=str, default="meanP",
                        choices=["meanP", "seqLSTM", "seqTransf", "tightTransf"],
                        help="choice a similarity header.")

    #### CLIP KC/EVL ######
    parser.add_argument("--zeroshot", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--pretrained_clip_name", default="ViT-L/14", type=str, help="Choose a CLIP version")
    parser.add_argument("--clip_evl", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--clip_kc", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--use_dsl", action='store_true', help="Choose a CLIP version")
    parser.add_argument("--clip_kc2", action='store_true', help="This is for ViT-B/16")
    parser.add_argument("--clip_kc4", action='store_true', help="This is for ViT-L/14") 


    ### DRL ###
    parser.add_argument("--interaction", type=str, default='no', help="Choose a CLIP version")
    parser.add_argument("--wti_arch", type=int, default=0, help="Choose a CLIP version")
    parser.add_argument("--cdcr", type=int, default=0, help="Choose a CLIP version")
    parser.add_argument("--pretrained_path", type=str, default='/gallery_louvre/dayoon.ko/research/models/InternVideo/Downstream/Video-Text-Retrieval/InternVideo-MM-L-14.ckpt', help="Choose a CLIP version")
    parser.add_argument("--mergeclip", type=bool, default=False, help="Choose a CLIP version")
    parser.add_argument("--mergeweight", type=float, default=0.5, help="Choose a CLIP version")
    parser.add_argument("--use_capdecoder", type=bool, default=False, help="Choose a CLIP version")
    
    args = parser.parse_args()

    if args.sim_header == "tightTransf":
        
        args.loose_type = False

    # Check paramenters
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
    
def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        #args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29500')
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.gpu = int(proc_id % num_gpus)
        #print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}' )
        
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        #print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        #print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    #print('| distributed init (rank {}): {}'.format(
    #    args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

    
def setup(args):
    # predefining random initial seeds
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    # Kiểm tra xem có nên sử dụng tính năng phân tán không
    use_distributed = True
    
    # Kiểm tra nếu đang chạy trên Windows
    if os.name == 'nt':  # Windows
        print("Running on Windows, disabling distributed mode")
        use_distributed = False
    
    # Kiểm tra xem PyTorch có hỗ trợ NCCL không
    try:
        if not hasattr(torch.distributed, 'is_nccl_available') or not torch.distributed.is_nccl_available():
            print("NCCL backend not available, disabling distributed mode")
            use_distributed = False
    except:
        print("Error checking NCCL availability, disabling distributed mode")
        use_distributed = False
        
    # Chỉ khởi chạy chế độ phân tán nếu đủ điều kiện
    if use_distributed:
        try:
            init_distributed_mode(args)
        except Exception as e:
            print(f"Error initializing distributed mode: {e}")
            print("Falling back to non-distributed mode")
            args.rank = 0
            args.gpu = 0
            args.world_size = 1
            if torch.cuda.is_available():
                torch.cuda.set_device(args.gpu)
    else:
        # Cấu hình cho chế độ không phân tán
        args.rank = 0
        args.gpu = 0
        args.world_size = 1
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
    
    return args

def init_device(args, local_rank):
    
    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    
    # Log device initialization
    print(f"\n====== DEVICE INITIALIZATION ======")
    print(f"Force CPU: {args.force_cpu if hasattr(args, 'force_cpu') else 'Not set'}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
    print(f"==================================\n")
    
    n_gpu = torch.cuda.device_count()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Thay logger.info() bằng print() thông thường
    if local_rank == 0:
        print(f"Device: {device}, n_gpu: {n_gpu}")

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):


    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')['state_dict']
    else:
        model_state_dict = None

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
    model = CLIP4Clip.from_pretrained(args.cross_model, cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)

    model.cuda()

    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):

    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    decay_clip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." in n]
    decay_noclip_param_tp = [(n, p) for n, p in decay_param_tp if "clip." not in n]

    no_decay_clip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." in n]
    no_decay_noclip_param_tp = [(n, p) for n, p in no_decay_param_tp if "clip." not in n]

    weight_decay = 0.2
    optimizer_grouped_parameters = [
        {'params': [p for n, p in decay_clip_param_tp], 'weight_decay': weight_decay, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_noclip_param_tp], 'weight_decay': weight_decay},
        {'params': [p for n, p in no_decay_clip_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_noclip_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_cosine', b1=0.9, b2=0.98, e=1e-6,
                         t_total=num_train_optimization_steps, weight_decay=weight_decay,
                         max_grad_norm=1.0)
    
    # Only use DistributedDataParallel if not on Windows and world_size > 1
    if os.name != 'nt' and args.world_size > 1:
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                                            output_device=torch.cuda.current_device(),
                                                            find_unused_parameters=False)
        except RuntimeError as e:
            if "use_libuv" in str(e):
                print("Warning: Caught libuv error, falling back to single GPU mode")
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
            else:
                raise
    else:
        # For single GPU or Windows, use DataParallel or single GPU
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.to(device)

    return optimizer, scheduler, model

def _run_on_single_gpu(model, batch_list_t, batch_list_v, batch_sequence_output_list, batch_visual_output_list):
    sim_matrix = []
    for idx1, b1 in enumerate(batch_list_t):
        input_mask, segment_ids, *_tmp = b1
        sequence_output = batch_sequence_output_list[idx1]
        each_row = []
        for idx2, b2 in enumerate(batch_list_v):
            video_mask, *_tmp = b2
            visual_output = batch_visual_output_list[idx2]
            b1b2_logits, *_tmp = model.get_similarity_logits(sequence_output, visual_output, input_mask, video_mask,
                                                                     loose_type=model.loose_type)
            b1b2_logits = b1b2_logits.cpu().detach().numpy()
            each_row.append(b1b2_logits)
        each_row = np.concatenate(tuple(each_row), axis=-1)
        sim_matrix.append(each_row)
    return sim_matrix


def eval_epoch(args, model, test_dataloader, device, n_gpu, store=True):
    if hasattr(model, 'module'):
        model = model.module.to(device)
    else:
        model = model.to(device)
    model.eval()
    
    # In ra thông tin thiết bị đang sử dụng
    print(f"\n====== DEVICE INFO IN EVAL_EPOCH ======")
    print(f"Device type: {device}")
    if str(device) == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Free Memory: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    print(f"========================================\n")
    
    # Tạo file để lưu trạng thái đã xử lý
    processed_file = os.path.join(os.path.dirname(args.video_dir), "processed_videos.txt")
    processed_videos = set()
    
    if os.path.exists(processed_file):
        with open(processed_file, 'r') as f:
            processed_videos = set(line.strip() for line in f.readlines())
        print(f"Đã tìm thấy {len(processed_videos)} video đã xử lý, sẽ bỏ qua")
    
    with torch.no_grad():
        for bid, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            # Giải phóng bộ nhớ GPU sau mỗi video để tránh lỗi CUDA OOM
            if str(device) == 'cuda':
                torch.cuda.empty_cache()
                
            output = []
            vid = batch[0][0]
            
            # Xử lý đường dẫn video
            if vid.startswith("videos\\") or vid.startswith("videos/"):
                vid = vid.replace("videos\\", "").replace("videos/", "")
                print(f"Đã chuẩn hóa ID video: {vid}")
            
            # Kiểm tra xem video này đã xử lý chưa
            if vid in processed_videos:
                print(f"Bỏ qua video {vid} vì đã xử lý trước đó")
                continue
            
            # load video segments
            segment_path = os.path.join(args.video_dir, vid, args.segmentation_filename)
            segment_path = normalize_path(segment_path)
            
            print(f"Đang kiểm tra file: {segment_path}")
            if not os.path.exists(segment_path):
                print(f"Không tìm thấy file segments.json cho video {vid} tại {segment_path}")
                continue
                
            try:
                with open(segment_path) as f:
                    segments = json.load(f)
                    # Kiểm tra nếu đã có vcap
                    if segments and any('vcap' in segment for segment in segments):
                        print(f"Video {vid} đã có vcap, bỏ qua")
                        # Thêm vào danh sách đã xử lý
                        with open(processed_file, 'a') as pf:
                            pf.write(f"{vid}\n")
                        continue
            except Exception as e:
                print(f"Lỗi khi đọc file segments.json cho video {vid}: {e}")
                continue
            
            # Xác định số lượng phân đoạn video cần xử lý
            segments_count = len(segments)
            print(f"Video {vid} có {segments_count} phân đoạn cần xử lý")
                
            try:
                for b in batch:
                    # ----------------------------
                    # 1. cache the features
                    # ----------------------------
                    batch_list_t = []
                    batch_list_v = []
                    batch_sequence_output_list, batch_visual_output_list = [], []
                        
                    video_id, input_ids, input_mask, segment_ids, video, video_mask, sentences = b
                    
                    if len(input_ids) == 0:
                        output.append([])
                        continue
                    
                    print(f"Đang xử lý {len(input_ids)} đoạn văn bản với {video.shape[0]} khung hình...")
                    
                    # Thêm xử lý lỗi CUDA out of memory
                    try:
                        # Tính toán bộ nhớ CUDA sẽ sử dụng trước khi chuyển
                        if str(device) == 'cuda':
                            free_memory = (torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / (1024**2)
                            # Ước tính bộ nhớ sẽ cần (rất thô sơ)
                            estimated_memory = sum(i.numel() * 4 for i in input_ids) + video.numel() * 4  # bytes
                            estimated_memory = estimated_memory / (1024**2)  # MB
                            print(f"Bộ nhớ CUDA trống: {free_memory:.2f}MB, Ước tính cần: {estimated_memory:.2f}MB")
                            
                            if estimated_memory > free_memory * 0.8:  # Nếu cần > 80% bộ nhớ trống
                                print(f"⚠️ Cần nhiều bộ nhớ, phân mảnh dữ liệu để xử lý an toàn")
                                # Sẽ xử lý từng đoạn nhỏ thay vì chuyển toàn bộ vào GPU
                                use_chunking = True
                            else:
                                use_chunking = False
                        else:
                            use_chunking = True  # CPU luôn phân mảnh
                        
                        if not use_chunking:
                            input_ids = [i.to(device) for i in input_ids]
                            input_mask = [i.to(device) for i in input_mask]
                            segment_ids = [i.to(device) for i in segment_ids]
                            video = video.to(device)
                            video_mask = video_mask.to(device)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"❌ CUDA out of memory khi chuyển dữ liệu sang GPU, sẽ xử lý từng phần nhỏ")
                            torch.cuda.empty_cache()
                            use_chunking = True
                        else:
                            raise
                    
                    sequence_output, visual_output = [],[]
                    
                    # Xử lý từng phần nhỏ để tránh CUDA OOM
                    chunk_size = 1  # Xử lý từng mẫu một để an toàn
                    
                    for i in range(0, len(input_ids), chunk_size):
                        chunk_end = min(i + chunk_size, len(input_ids))
                        try:
                            # Xử lý batch này
                            chunk_input_ids = [inp_id.to(device) if use_chunking else inp_id for inp_id in input_ids[i:chunk_end]]
                            chunk_input_mask = [inp_mask.to(device) if use_chunking else inp_mask for inp_mask in input_mask[i:chunk_end]]
                            chunk_segment_ids = [seg_id.to(device) if use_chunking else seg_id for seg_id in segment_ids[i:chunk_end]]
                            
                            chunk_video = video[i:chunk_end].to(device) if use_chunking else video[i:chunk_end]
                            chunk_video_mask = video_mask[i:chunk_end].to(device) if use_chunking else video_mask[i:chunk_end]
                            
                            for j in range(len(chunk_input_ids)):
                                try:
                                    sequence_output_sub, visual_output_sub = model.get_sequence_visual_output(
                                        chunk_input_ids[j].unsqueeze(0),
                                        chunk_segment_ids[j].unsqueeze(0),
                                        chunk_input_mask[j].unsqueeze(0),
                                        chunk_video[j:j+1].unsqueeze(1),
                                        chunk_video_mask[j:j+1].unsqueeze(1)
                                    )
                                    sequence_output.append(sequence_output_sub)
                                    visual_output.append(visual_output_sub)
                                    
                                    # Từng mẫu một - giải phóng bộ nhớ ngay
                                    if str(device) == 'cuda':
                                        torch.cuda.empty_cache()
                                except RuntimeError as e:
                                    if "CUDA out of memory" in str(e):
                                        print(f"❌ CUDA out of memory khi xử lý mẫu, chuyển sang CPU cho mẫu này")
                                        torch.cuda.empty_cache()
                                        
                                        # Di chuyển lên CPU nếu cần
                                        temp_model = model
                                        if str(device) == 'cuda':
                                            temp_model = model.to('cpu')
                                        
                                        # Đảm bảo dữ liệu đều ở CPU
                                        cpu_input_id = chunk_input_ids[j].cpu()
                                        cpu_segment_id = chunk_segment_ids[j].cpu()
                                        cpu_input_mask = chunk_input_mask[j].cpu()
                                        cpu_video = chunk_video[j:j+1].cpu()
                                        cpu_video_mask = chunk_video_mask[j:j+1].cpu()
                                        
                                        # Xử lý trên CPU
                                        sequence_output_sub, visual_output_sub = temp_model.get_sequence_visual_output(
                                            cpu_input_id.unsqueeze(0),
                                            cpu_segment_id.unsqueeze(0),
                                            cpu_input_mask.unsqueeze(0),
                                            cpu_video.unsqueeze(1),
                                            cpu_video_mask.unsqueeze(1)
                                        )
                                        
                                        # Đưa model về GPU nếu cần
                                        if str(device) == 'cuda':
                                            temp_model = temp_model.to(device)
                                            
                                        sequence_output.append(sequence_output_sub)
                                        visual_output.append(visual_output_sub)
                                    else:
                                        print(f"Lỗi không phải CUDA OOM: {e}")
                                        raise
                        except Exception as e:
                            print(f"Lỗi khi xử lý chunk {i}-{chunk_end}: {e}")
                            # Tiếp tục với chunk tiếp theo
                            continue
                                
                        # Đảm bảo GPU đã được giải phóng sau mỗi chunk
                        if str(device) == 'cuda':
                            torch.cuda.empty_cache()
                    
                    # Chuẩn bị đầu vào cho phần tính toán similarity
                    for i, seq_out in enumerate(sequence_output):
                        batch_sequence_output_list.append(seq_out.squeeze())
                        batch_list_t.append((input_mask[i].to(device) if str(device) == 'cuda' else input_mask[i], 
                                             segment_ids[i].to(device) if str(device) == 'cuda' else segment_ids[i]))
                        
                        batch_visual_output_list.append(visual_output[i])
                        batch_list_v.append((video_mask[i].to(device) if str(device) == 'cuda' else video_mask[i],))

                    # ----------------------------------
                    # 2. calculate the similarity
                    # ----------------------------------
                    idxs = []
                    for i in range(len(sequence_output)):
                        try:
                            sim_matrix = _run_on_single_gpu(model, [batch_list_t[i]], [batch_list_v[i]], 
                                                         [batch_sequence_output_list[i]], [batch_visual_output_list[i]])
                            idxs.append(np.argmax(sim_matrix[0].squeeze()))
                            
                            # Giải phóng bộ nhớ sau mỗi lần tính toán 
                            if str(device) == 'cuda':
                                torch.cuda.empty_cache()
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                print(f"❌ CUDA out of memory khi tính similarity, chọn phương án đơn giản hơn")
                                torch.cuda.empty_cache()
                                # Chọn caption đầu tiên trong trường hợp không tính được similarity
                                idxs.append(0)
                            else:
                                raise
                    
                    # Kết quả với xử lý ngoại lệ
                    try:
                        result = [sentences[i][idx] if idx < len(sentences[i]) else "" for i, idx in enumerate(idxs)]
                        output.append(result)
                    except Exception as e:
                        print(f"Lỗi khi lấy kết quả: {e}")
                        output.append([""] * len(input_ids))
                
                if store:
                    try:
                        # update the result
                        for i, v in enumerate(output):
                            if i < len(segments):
                                if 'vcap' not in segments[i]:
                                    segments[i]['vcap'] = v
                        
                        # store the updated result
                        with open(segment_path, 'w') as f:
                            json.dump(segments, f, indent=2)
                        print(f"✓ Đã cập nhật vcap cho video {vid}")
                        
                        # Đánh dấu là đã xử lý
                        with open(processed_file, 'a') as pf:
                            pf.write(f"{vid}\n")
                    except Exception as e:
                        print(f"Lỗi khi lưu file segments.json cho video {vid}: {e}")
            except Exception as e:
                print(f"Lỗi tổng thể khi xử lý video {vid}: {e}")
                continue
            
            # Giải phóng bộ nhớ sau khi xử lý xong một video
            if str(device) == 'cuda':
                torch.cuda.empty_cache()

def caption_retrieval(video_dir, caption_filename, segmentation_filename):
    # Normalize paths to prevent double backslash issues
    video_dir = normalize_path(video_dir)
    caption_filename = normalize_path(caption_filename)
    segmentation_filename = normalize_path(segmentation_filename)
    
    # Remove duplicate 'videos' in path if exists (handle both slash styles)
    if 'videos/videos' in video_dir or 'videos\\videos' in video_dir:
        video_dir = video_dir.replace('videos/videos', 'videos').replace('videos\\videos', 'videos')
    
    # Log the actual directory being processed
    print(f"Processing videos from directory: {video_dir}")
    
    # Ensure video_dir is a valid directory
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    
    # Tạo một copy của các tham số để tránh thay đổi các tham số gốc
    args = get_args(video_dir, caption_filename, segmentation_filename)
    
    # Thêm tham số dist_url nếu chưa có
    if not hasattr(args, 'dist_url'):
        args.dist_url = 'env://'
    
    # Đặt force_cpu=False để ưu tiên sử dụng GPU
    args.force_cpu = False
    
    # Giảm batch_size để tiết kiệm bộ nhớ GPU
    args.batch_size = 1
    args.batch_size_val = 1
    
    # Giảm max_frames để tiết kiệm bộ nhớ
    args.max_frames = 50 
    
    # Giải phóng bộ nhớ CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    args = setup(args)
    
    device, n_gpu = init_device(args, args.local_rank)
    
    # In ra thông tin thiết bị đang sử dụng
    print(f"\n====== DEVICE INFO IN CAPTION_RETRIEVAL ======")
    print(f"Device type: {device}")
    if str(device) == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Free Memory: {(torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    print(f"n_gpu: {n_gpu}")
    print(f"================================================\n")
    
    # Initialize model and tokenizer
    tokenizer = ClipTokenizer()
    model = init_model(args, device, n_gpu, args.local_rank)
    
    # Tạo file để lưu trạng thái đã xử lý
    processed_file = os.path.join(os.path.dirname(args.video_dir), "processed_videos.txt")
    if not os.path.exists(processed_file):
        with open(processed_file, 'w') as f:
            f.write("")  # Tạo file rỗng
    
    with torch.no_grad():
        test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer)
        if args.local_rank == 0:
            print("***** Running test *****")
            print("  Num examples = %d" % test_length)
            print("  Batch size = %d" % args.batch_size_val)
            print("  Num steps = %d" % len(test_dataloader))
            eval_epoch(args, model, test_dataloader, device, n_gpu)
    