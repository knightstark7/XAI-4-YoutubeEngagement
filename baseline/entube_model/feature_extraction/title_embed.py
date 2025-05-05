import pandas as pd
import torch
import argparse
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

def setup_data(snapugc_path):
    print('Setup data snapugc')
    return pd.read_csv(snapugc_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model():
    print('Setup model Bert')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return model.eval().to(device), tokenizer

def setup_title_df(snapugc):
    print('Setup title')
    title_df = snapugc[['Id', 'Set', 'Title']].copy()
    
    title_df['Title'] = title_df['Title'].fillna('')
    title_df['Title'] = title_df['Title'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True)
    title_df['Title'] = title_df['Title'].str.replace(r'\s+', ' ', regex=True)
    title_df['Title'] = title_df['Title'].str.strip().str.lower()
    return title_df

def embed_title_handler(title, model, tokenizer):
    try:
        inputs = tokenizer(title, return_tensors='pt', truncation=True, padding='max_length', max_length=32)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get BERT outputs
        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: [1, 768]
            return cls_embedding[0].tolist()
    except Exception as e:
        print(f'Error at {title}: {e}')
        return None

def embed_title(title_df, model, tokenizer, batch_size, start, end, embedded_data_dir):
    print('Start Embedding...')
    for idx in tqdm(range(start, end, batch_size)): # range(start, end, batch_size):
        batch_end = min(idx+batch_size, len(title_df))
        
        output_file = os.path.join(embedded_data_dir, f'title_embedding_{idx}_{batch_end - 1}.parquet')
        if os.path.exists(output_file):
            print(f'File {output_file} already exists, skipping...')
            continue
        
        audios_df_batch = title_df.iloc[idx:batch_end].copy()
        audios_df_batch['embedding_title'] = audios_df_batch['Title'].apply(
            lambda x: embed_title_handler(x, model, tokenizer)
        )
        audios_df_batch.to_parquet(output_file, index=False)
        # print(f'Index {idx}->{idx+batch_size} is Done')
    print("All is done")

def main(args):
    snapugc = setup_data(args.snapugc_path)
    model, tokenizer = setup_model()
    title_df = setup_title_df(snapugc)
    embed_title(title_df, model, tokenizer, args.batch_size, args.start, args.end, args.embedded_data_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio embedding script')
    parser.add_argument("--snapugc_path", type=str, help="Path to the SnapUGC csv file", default='/home/datlinux/XAI-4-YoutubeEngagement/data/SnapUGC/prepped_df.csv')
    parser.add_argument("--embedded_data_dir", type=str, help="Directory for embedded data output", default="/mnt/d/Thesis/Prep/SnapUGC/embedding/title_embedding")
    parser.add_argument("--data_sample_dir", type=str, help="Directory containing sample data", default="/mnt/d/Thesis/Data/SnapUGC")
    parser.add_argument("--start", type=int, default=0, help='Start index for processing')
    parser.add_argument("--end", type=int, default=None, help='End index for processing')
    parser.add_argument("--batch_size", type=int, default=128, help='Batch size for processing')
    args = parser.parse_args()
    
    if args.end is None:
        args.end = len(pd.read_csv(args.snapugc_path))
    
    main(args)
