import os
import re
import shutil
import warnings

# External Dependencies
import numpy as np
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
import rmm

# NVTabular
import nvtabular as nvt
from nvtabular.ops import (
    Categorify,
    Clip,
    FillMissing,
    Normalize,
)
from nvtabular.utils import _pynvml_mem_size, device_mem_size

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset', type=str, default="kaggle") 

if __name__ == "__main__":
    args = parser.parse_args()
    dataset = args.dataset
    print(dataset)
    if dataset == "spotify":
        BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"
        INPUT_DATA_DIR  = os.path.join(BASE_DIR, "spotify/processed")
        Workspace = "workspace/spotify_workspace/"
        dask_workdir = os.path.join(BASE_DIR, Workspace + "workdir")
        OUTPUT_DATA_DIR = os.path.join(BASE_DIR, Workspace + "output")
        stats_path = os.path.join(BASE_DIR, Workspace + "stats")
        
        if os.path.isdir(dask_workdir):
            shutil.rmtree(dask_workdir)
        os.makedirs(dask_workdir)

        if os.path.isdir(stats_path):
            shutil.rmtree(stats_path)
        os.mkdir(stats_path)

        if os.path.isdir(OUTPUT_DATA_DIR):
            shutil.rmtree(OUTPUT_DATA_DIR)
        os.mkdir(OUTPUT_DATA_DIR)
        #TODO: Add validation set
        train_paths = os.path.join(INPUT_DATA_DIR, "spotify/train_dataset.parquet")
        valid_paths = os.path.join(INPUT_DATA_DIR, "spotify/valid_dataset.parquet")

        CONTINUOUS_COLUMNS = ["session_position", "session_length", "hist_user_behavior_n_seekfwd", "hist_user_behavior_n_seekback", "hour_of_day"]
        CATEGORICAL_COLUMNS = ["session_id", "track_id_clean", "context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end", "date"]
        BOOLEAN_COLUMNS = ["skip_1", "skip_2", "skip_3", "not_skipped", "context_switch", "no_pause_before_play", "short_pause_before_play", "long_pause_before_play", "hist_user_behavior_is_shuffle", "premium"]
        LABEL_COLUMNS = ["not_skipped"]
        COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + BOOLEAN_COLUMNS
        
    if dataset == "kaggle":
        Workspace = "workspace/kaggle_workspace/"
        BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"
        INPUT_DATA_DIR  = os.path.join(BASE_DIR, "kaggle/processed")
        dask_workdir = os.path.join(BASE_DIR, Workspace + "workdir")
        OUTPUT_DATA_DIR = os.path.join(BASE_DIR, Workspace + "output") 
        stats_path = os.path.join(BASE_DIR, Workspace + "stats")

        # Make sure we have a clean worker space for Dask
        if os.path.isdir(dask_workdir):
            shutil.rmtree(dask_workdir)
        os.makedirs(dask_workdir)

        # Make sure we have a clean stats space for Dask
        if os.path.isdir(stats_path):
            shutil.rmtree(stats_path)
        os.mkdir(stats_path)

        # Make sure we have a clean output path
        if os.path.isdir(OUTPUT_DATA_DIR):
            shutil.rmtree(OUTPUT_DATA_DIR)
        os.mkdir(OUTPUT_DATA_DIR)

        train_paths = os.path.join(INPUT_DATA_DIR, "kaggle_ad_train/train_subset.txt.parquet")
        valid_paths = os.path.join(INPUT_DATA_DIR, "kaggle_ad_val/val_subset.txt.parquet")

        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)] #(1,14)
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)] #(1,27)
        LABEL_COLUMNS = ["label"]
        COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + LABEL_COLUMNS

    elif dataset == "terabyte":
        Workspace = "workspace/terabyte_workspace/"
        # # BASE_DIR = "/home/ubuntu/zheng/Nvidia-Merlin/"
        BASE_DIR = "/workspace/SC_artifacts_eval/processed_data"
        INPUT_DATA_DIR  = os.path.join(BASE_DIR, "terabyte/processed/terabyte")
        dask_workdir = os.path.join(BASE_DIR, Workspace + "workdir")
        OUTPUT_DATA_DIR = os.path.join(BASE_DIR, Workspace + "output") 
        stats_path = os.path.join(BASE_DIR, Workspace + "stats")

        # Make sure we have a clean worker space for Dask
        if os.path.isdir(dask_workdir):
            shutil.rmtree(dask_workdir)
        os.makedirs(dask_workdir)

        # Make sure we have a clean stats space for Dask
        if os.path.isdir(stats_path):
            shutil.rmtree(stats_path)
        os.mkdir(stats_path)

        # Make sure we have a clean output path
        if os.path.isdir(OUTPUT_DATA_DIR):
            shutil.rmtree(OUTPUT_DATA_DIR)
        os.mkdir(OUTPUT_DATA_DIR)


        fname = "day_{}.parquet"
        num_days = 3
        train_paths = [os.path.join(INPUT_DATA_DIR, fname.format(day)) for day in range(num_days - 1)]
        valid_paths = [
            os.path.join(INPUT_DATA_DIR, fname.format(day)) for day in range(num_days - 1, num_days)
        ]
        print(train_paths)
        print(valid_paths)

        CONTINUOUS_COLUMNS = ["I" + str(x) for x in range(1, 14)]
        CATEGORICAL_COLUMNS = ["C" + str(x) for x in range(1, 27)]
        LABEL_COLUMNS = ["label"]
        COLUMNS = CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS + LABEL_COLUMNS

        
    # Dask dashboard
    dashboard_port = "8787"

    # Deploy a Single-Machine Multi-GPU Cluster
    protocol = "tcp"  # "tcp" or "ucx"
    NUM_GPUS = [0, 1, 2, 3]
    visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Delect devices to place workers
    device_limit_frac = 0.7  # Spill GPU-Worker memory to host at this limit.  default:0.7 terabyte:0.5
    device_pool_frac = 0.8  # default:0.8  terabyte:0.9
    part_mem_frac = 0.15  # default:0.15   terabyte:0.05

    # Use total device size to calculate args.device_limit_frac
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)
    part_size = int(part_mem_frac * device_size)

    # Check if any device memory is already occupied
    for dev in visible_devices.split(","):
        fmem = _pynvml_mem_size(kind="free", index=int(dev))
        used = (device_size - fmem) / 1e9
        if used > 1.0:
            warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")

    cluster = None  # (Optional) Specify existing scheduler port
    if cluster is None:
        cluster = LocalCUDACluster(
            protocol=protocol,
            n_workers=len(visible_devices.split(",")),
            CUDA_VISIBLE_DEVICES=visible_devices,
            device_memory_limit=device_limit,
            local_directory=dask_workdir,
            dashboard_address=":" + dashboard_port,
        )

    # Create the distributed client
    client = Client(cluster)
    print(client)
    # client

    # Initialize RMM pool on ALL workers
    def _rmm_pool():
        rmm.reinitialize(
            # RMM may require the pool size to be a multiple of 256.
            pool_allocator=True,
            initial_pool_size=(device_pool_size // 256) * 256,  # Use default size
        )


    client.run(_rmm_pool)
    # define our dataset schema
    

    num_buckets = 50000000
    categorify_op = Categorify(out_path=stats_path, max_size=num_buckets)
    # categorify_op = Categorify(out_path=stats_path)
    cat_features = CATEGORICAL_COLUMNS >> categorify_op
    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()
    features = cat_features + cont_features + LABEL_COLUMNS

    workflow = nvt.Workflow(features, client=client)

    dict_dtypes = {}
    for col in CATEGORICAL_COLUMNS:
        dict_dtypes[col] = np.int64

    for col in CONTINUOUS_COLUMNS:
        dict_dtypes[col] = np.float32

    for col in LABEL_COLUMNS:
        dict_dtypes[col] = np.float32

    train_dataset = nvt.Dataset(train_paths, engine="parquet", part_size=part_size)
    valid_dataset = nvt.Dataset(valid_paths, engine="parquet", part_size=part_size)

    output_train_dir = os.path.join(OUTPUT_DATA_DIR, "train/")
    output_valid_dir = os.path.join(OUTPUT_DATA_DIR, "valid/")

    workflow.fit(train_dataset)

    workflow.transform(train_dataset).to_parquet(
        output_path=output_train_dir,
        shuffle=nvt.io.Shuffle.PER_PARTITION,
        dtypes=dict_dtypes,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )

    workflow.transform(valid_dataset).to_parquet(
        output_path=output_valid_dir,
        dtypes=dict_dtypes,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
    )

    workflow.save(os.path.join(OUTPUT_DATA_DIR, "workflow"))


   