import os
from os import path
import glob

import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

import nvtabular as nvt
from nvtabular.utils import device_mem_size, get_rmm_size

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

    # Update column names based on the data
    feature_names = ['track_id', 'duration', 'release_year', 'us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness', 'danceability', 'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key', 'liveness', 'loudness', 'mechanism', 'mode', 'organism', 'speechiness', 'tempo', 'time_signature', 'valence', 'acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']

    dtypes = {
        'track_id': 'str',
        'duration': np.float32,
        'release_year': np.int32,
        'us_popularity_estimate': np.float32,
        'acousticness': np.float32,
        'beat_strength': np.float32,
        'bounciness': np.float32,
        'danceability': np.float32,
        'dyn_range_mean': np.float32,
        'energy': np.float32,
        'flatness': np.float32,
        'instrumentalness': np.float32,
        'key': np.int32,
        'liveness': np.float32,
        'loudness': np.float32,
        'mechanism': np.float32,
        'mode': 'str',
        'organism': np.float32,
        'speechiness': np.float32,
        'tempo': np.float32,
        'time_signature': np.int32,
        'valence': np.float32,
        'acoustic_vector_0': np.float32,
        'acoustic_vector_1': np.float32,
        'acoustic_vector_2': np.float32,
        'acoustic_vector_3': np.float32,
        'acoustic_vector_4': np.float32,
        'acoustic_vector_5': np.float32,
        'acoustic_vector_6': np.float32,
        'acoustic_vector_7': np.float32,
    }

    file_list = glob.glob(os.path.join(INPUT_PATH, "*.csv"))

    dataset = nvt.Dataset(
        file_list,
        engine="csv",
        names=feature_names,
        part_mem_fraction=frac_size,
        dtypes=dtypes,
        client=client,
    )

    dataset.to_parquet(
        os.path.join(OUTPUT_PATH, "spotify_track"),
        preserve_files=True,
    )