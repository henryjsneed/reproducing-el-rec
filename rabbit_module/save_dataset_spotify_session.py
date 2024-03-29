import os
import glob

# tools for data preproc/loading
import torch
import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader

import multiprocessing as mp
import re
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--reorder', type=int, default=0)

global access_list

def time_wrap():
    torch.cuda.synchronize()
    return time.time()

def get_col(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:20]
    x_int = torch.cat(list(x_both.values())[20:], 1)
    length = y.shape[0]
    y = y.reshape(length, 1)

    for j in range(20):
        if access_list[j].shape[0] > 0:
            x_cat[j] = access_list[j][x_cat[j]]
    x_cat = torch.stack(x_cat)
    return y, x_cat, x_int

        

if __name__ == "__main__":
    CONTINUOUS_COLUMNS = ["session_position", "session_length", "hour_of_day", "hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback"]
    CATEGORICAL_COLUMNS = ["session_id", "track_id_clean", "skip_1", "skip_2", "skip_3", "not_skipped", "context_switch", "no_pause_before_play", "short_pause_before_play", "long_pause_before_play", "hist_user_behavior_is_shuffle", "date", "premium", "context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end"]
    LABEL_COLUMNS = ["label"]

    BATCH_SIZE = 8192
    PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))

    args = parser.parse_args()
    if_reorder = args.reorder
    input_path = "/workspace/SC_artifacts_eval/processed_data/workspace/spotify_session_workspace/output"

    train_paths = glob.glob(os.path.join(input_path, "train", "*.parquet"))
    valid_paths = glob.glob(os.path.join(input_path, "valid", "*.parquet"))
    train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)
    valid_data = nvt.Dataset(valid_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)

    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK,
    )
    
    train_dataloader = DLDataLoader(
        train_data_itrs, collate_fn=get_col, batch_size=None, pin_memory=False, num_workers=0
    )
    
    workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))
    embeddings = list(get_embedding_sizes(workflow).values())
    # We limit the output dimension to 16
    embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings] # embedding table size

    embedding_table_size = []  #embedding table size
    for emb in embeddings:
        embedding_table_size.append(emb[0])
    print(embedding_table_size)

    device = 'cuda:0'
    length = len(embedding_table_size)
    access_list = []

    print("Access list shapes:")
    for idx, tensor in enumerate(access_list):
        print(f"access_list[{idx}]: {tensor.shape}")

    for i in range(length):
        if if_reorder == 1:
            x = torch.load("/workspace/SC_artifacts_eval/Access_Index/spotify_session/access_index/access_index_"+ str(i) +"_new.pt").to(device)
        else:
            x = torch.load("/workspace/SC_artifacts_eval/Access_Index/spotify_session/access_index/access_index_"+ str(i) +".pt").to(device)
        access_list.append(x)

    train_iter = iter(train_dataloader)
    label, sparse, dense = train_iter.next()  

    for i in tqdm(range(500-1)):
        tmp_label, tmp_sparse, tmp_dense = train_iter.next()  
        sparse = torch.cat((sparse, tmp_sparse), 1)
        label = torch.cat((label, tmp_label), 0)
        dense = torch.cat((dense, tmp_dense), 0)

    # ====================== save ========================
    if if_reorder == 1:
        torch.save(sparse, '/workspace/SC_artifacts_eval/Access_Index/spotify_session/training_data/reordered_sparse.pt')
        torch.save(dense, '/workspace/SC_artifacts_eval/Access_Index/spotify_session/training_data/reordered_dense.pt')
        torch.save(label, '/workspace/SC_artifacts_eval/Access_Index/spotify_session/training_data/reordered_label.pt')
    else:
        torch.save(sparse, '/workspace/SC_artifacts_eval/Access_Index/spotify_session/training_data/sparse.pt')
        torch.save(dense, '/workspace/SC_artifacts_eval/Access_Index/spotify_session/training_data/dense.pt')
    torch.save(label, '/workspace/SC_artifacts_eval/Access_Index/spotify_session/training_data/label.pt')

