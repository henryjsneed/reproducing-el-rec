import os
from os import path
import glob

import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt
from ntabular.utils ivmport device_mem_size, get_rmm_size


if __name__ == "__main__":
    BASE_DIR = "/workspace/SC_artifacts_eval/processed_data/spotify"
    INPUT_PATH  = "/workspace/SC_artifacts_eval/dlrm_dataset/spotify"
    OUTPUT_PATH  = os.path.join(BASE_DIR, "processed")
    CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    frac_size = 0.10

    cluster = None
    if cluster is None:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=CUDA_VISIBLE_DEVICES,
            rmm_pool_size=get_rmm_size(0.8 * device_mem_size()),
            local_directory=os.path.join(OUTPUT_PATH, "dask-space"),
        )
    client = Client(cluster)

    file_list = ['/workspace/SC_artifacts_eval/dlrm_dataset/spotify/spotify_session.csv']

    dtypes = {
        "session_id": "object",
        "session_position": np.int32,
        "session_length": np.int32,
        "track_id_clean": "object",
        "skip_1": "bool",
        "skip_2": "bool",
        "skip_3": "bool",
        "not_skipped": "bool",
        "context_switch": np.int32,
        "no_pause_before_play": np.int32,
        "short_pause_before_play": np.int32,
        "long_pause_before_play": np.int32,
        "hist_user_behavior_n_seekfwd": np.int32,
        "hist_user_behavior_n_seekback": np.int32,
        "hist_user_behavior_is_shuffle": "bool",
        "hour_of_day": np.int32,
        "date": "object",
        "premium": "bool",
        "context_type": "object",
        "hist_user_behavior_reason_start": "object",
        "hist_user_behavior_reason_end": "object",
    }

    dataset = nvt.Dataset(
        file_list,
        engine="csv",
        part_mem_fraction=frac_size,
        sep=",",
        dtypes=dtypes,
        client=client,
    )

    dataset.to_parquet(
        os.path.join(OUTPUT_PATH, "spotify"),
        preserve_files=True,
    )
