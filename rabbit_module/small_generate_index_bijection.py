import os
from time import time
import glob
import argparse
import subprocess
import time
import torch.multiprocessing as mp
import numpy as np
import rabbit
import atexit
from filelock import FileLock
import torch

# tools for data preproc/loading
import torch
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def find_available_gpu(memory_threshold=3000, memory_utilization_threshold=5):
    lock_file_prefix = "/tmp/gpu_lock_"
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    gpu_info = result.stdout.strip().split("\n")

    for line in gpu_info:
        gpu_idx, gpu_util, memory_used, memory_total = line.split(", ")
        gpu_idx = int(gpu_idx)
        gpu_util = int(gpu_util.strip("%"))
        memory_used = int(memory_used.strip("MiB"))
        memory_total = int(memory_total.strip("MiB"))
        memory_utilization = (memory_used / memory_total) * 100
        print("MEMORY UTILIzation: " + str(memory_utilization))
        print("Memory threshold: " + str(memory_utilization_threshold))
        if memory_utilization < memory_utilization_threshold:
            print("memory utiliztion is less than threshhold.")
            lock_file = f"{lock_file_prefix}{gpu_idx}.lock"

            # Check if the lock file exists, meaning the GPU is in use
            if not os.path.exists(lock_file):
                print(f"No lock file found for GPU {gpu_idx}")
                # Create the lock file
                with open(lock_file, 'w') as lock:
                    lock.write('1')
                print(f"Lock file created for GPU {gpu_idx}")

                return gpu_idx
            else:
                print(f"Lock file found for GPU {gpu_idx}")

    raise RuntimeError("No available GPU found")


def _release_gpu(lock_file):
    if os.path.exists(lock_file):
        os.remove(lock_file)
available_gpu = find_available_gpu()
os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpu)



import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes
from nvtabular.loader.torch import TorchAsyncItr, DLDataLoader




_in_use_gpus = set()

def get_train_paths(input_path):
    for path in glob.iglob(os.path.join(input_path, "train", "*.parquet")):
        yield path

global access_list

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 
parser.add_argument('--table_idx', type=int, default=10)
parser.add_argument('--batch_num', type=int, default=65536)

def time_wrap():
    torch.cuda.synchronize()
    return time.time()


def get_col_kaggle(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:26]
    x_int = torch.cat(list(x_both.values())[26:39],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    for j in range(26):
        x_cat[j] = access_list[j][x_cat[j]]
    
    return y, x_cat, x_int

def get_col_avazu(inputBatch):
    x_both, y = inputBatch
    x_cat = list(x_both.values())[0:20]
    x_int = torch.cat(list(x_both.values())[20:23],1)
    length = y.shape[0]
    y = y.reshape(length,1)

    for j in range(20):
        x_cat[j] = access_list[j][x_cat[j]]
    
    return y, x_cat, x_int


if __name__ == "__main__":
    args = parser.parse_args()

    LABEL_COLUMNS = ["label"]
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 256))
    PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))

    if torch.cuda.is_available():
      print(f"Total number of GPUs: {torch.cuda.device_count()}")
      print(f"Selected GPU: {available_gpu}")

    dataset = args.dataset

    input_path = ""
    if dataset == "kaggle":
        input_path = os.path.join(BASE_DIR, "workspace/kaggle_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
    elif dataset == "avazu":
        input_path = os.path.join(BASE_DIR, "workspace/avazu_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 4)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 21)]
    elif dataset == "terabyte":
        input_path = os.path.join(BASE_DIR, "workspace/terabyte_workspace/output")
        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]

    train_paths = get_train_paths(input_path)
    train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.04 / PARTS_PER_CHUNK)

    train_data_itrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        parts_per_chunk=PARTS_PER_CHUNK,
    )

    if dataset == "kaggle":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=1
        )
    elif dataset == "terabyte":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_kaggle, batch_size=None, pin_memory=False, num_workers=1
        )
    elif dataset == "avazu":
        train_dataloader = DLDataLoader(
            train_data_itrs, collate_fn=get_col_avazu, batch_size=None, pin_memory=False, num_workers=1
        )

    workflow = nvt.Workflow.load(os.path.join(input_path, "workflow"))
    embeddings = list(get_embedding_sizes(workflow).values())
    embeddings = [[emb[0], min(16, emb[1])] for emb in embeddings] # embedding table size

    embedding_table_size = []  # embedding table size
    for emb in embeddings:
        embedding_table_size.append(emb[0])

    device_count = torch.cuda.device_count()
    # assert device_count >= 4, f"Only {device_count} GPUs detected. Please make sure you have 4 GPUs available."
    print("GPUS DETECTED:" + str(device_count))

    table_idx = args.table_idx
    print(embedding_table_size)  # total length of the table
    print(embedding_table_size[table_idx])  # total length of the table
    if dataset == "kaggle": #[2, 3, 11, 15, 20]
        input_file = "/workspace/SC_artifacts_eval/Access_Index/kaggle/access_index/access_index_" + str(table_idx) + ".pt"
        cat_num = 26
    elif dataset == "terabyte": #[0, 9, 10, 19, 20, 21]
        input_file = "/workspace/SC_artifacts_eval/Access_Index/terabyte/access_index/access_index_" + str(table_idx) + ".pt"
        cat_num = 26
    elif dataset == "avazu": #[7, 8]
        input_file = "/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_" + str(table_idx) + ".pt"
        cat_num = 20

    train_iter = iter(train_dataloader)
    total_batch_num = len(train_dataloader)
    emb_index = torch.load(input_file)

    length = embedding_table_size[table_idx]
    hot_idx = int(length*0.05)

    start = time_wrap()
    edge_list = []

    batch_num = args.batch_num # 70656

    idx = 0
    for i, inputBatch in enumerate(train_data_itrs):
        x_both, y = inputBatch
        x_cat = list(x_both.values())[0:cat_num]

        x_cat_cpu = x_cat[table_idx].cpu()

        x_cat[table_idx] = emb_index[x_cat_cpu]
        x_cat[table_idx] = torch.clamp(x_cat[table_idx],min=hot_idx)-hot_idx

        batch_index = (x_cat[table_idx].view(-1)).to(torch.int).unique()

        edge_pairs = torch.combinations(batch_index[1:])
        edge_list.append(edge_pairs)
       
        if i % 1024 == 0:
            tmp = time_wrap()
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(torch.cuda.current_device())}")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved(torch.cuda.current_device())}")
            print("batch:",i,"finish ratio:",round(i/batch_num,4), "total ratio:",round(i/total_batch_num,4), "time:",tmp-start)
        if i == batch_num:
            break

    end = time_wrap()
    print("total time for generate edge_list:", end - start)

    torch_edge = torch.cat(edge_list,0).transpose(0, 1).detach().contiguous()

    del(edge_list)
    torch.cuda.empty_cache()

    print("start reordering")
    start = time_wrap()

    num_chunks = 10
    chunk_size = torch_edge.size(1) // num_chunks
    new_edge_index_list = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else torch_edge.size(1)
        
        chunk = torch_edge[:, start_idx:end_idx].contiguous()
        print(f"Processing chunk {i+1}/{num_chunks}")
        
        new_edge_index_chunk = rabbit.generate_new_index(chunk, hot_idx)
        new_edge_index_list.append(new_edge_index_chunk)

    new_edge_index = torch.cat(new_edge_index_list, dim=0)

    end = time_wrap()
    print("reordering time:", end - start)

    final_index = []
    print(f"Length: {length}")
    print(f"New edge index shape: {new_edge_index.shape[0]}")
    chunk_size = 1000  # Choose a suitable chunk size
    num_chunks = math.ceil(length / chunk_size)

    final_index_torch = torch.zeros(length, dtype=torch.int32)

    for chunk in range(num_chunks):
        start = chunk * chunk_size
        end = min((chunk + 1) * chunk_size, length)

        for i in range(start, end):
            k = emb_index[i].item()
            if k > hot_idx and k - hot_idx < new_edge_index.shape[0]:
                k = new_edge_index[k - hot_idx]
            final_index_torch[i] = k

    print("complted for loop")
    tensor_index = torch.tensor(final_index)
    print("completed torch.tensor")
    if dataset == "kaggle":
        output_file = "/workspace/SC_artifacts_eval/Access_Index/kaggle/access_index/access_index_" + str(table_idx) + "_new.pt"
    elif dataset == "avazu":
        output_file = "/workspace/SC_artifacts_eval/Access_Index/avazu/access_index/access_index_" + str(table_idx) + "_new.pt"
    elif dataset == "terabyte":
        output_file = "/workspace/SC_artifacts_eval/Access_Index/terabyte/access_index/access_index_" + str(table_idx) + "_new.pt"

    torch.save(tensor_index, output_file)
    print("saved index bijection")
