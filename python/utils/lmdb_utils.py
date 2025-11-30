import os
import lmdb
import msgpack

def list_shard_paths(root_dir: str):
    shards = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("shard_"):
            shards.append(full)
    shards.sort()
    return shards

def count_entries(env_path: str) -> int:
    env = lmdb.open(env_path, readonly=True, lock=False, readahead=True, max_readers=16)
    with env.begin() as txn:
        st = txn.stat()
    env.close()
    return st.get("entries", 0)

def scan_all_shards(root_dir: str):
    total = 0
    shards = list_shard_paths(root_dir)
    for p in shards:
        n = count_entries(p)
        print(f"{os.path.basename(p)}: {n}")
        total += n
    print(f"TOTAL: {total}")

def sample_entry(env_path: str, idx: int = 0):
    key = f"{idx:08d}".encode()
    env = lmdb.open(env_path, readonly=True, lock=False, readahead=True, max_readers=16)
    with env.begin() as txn:
        data = txn.get(key)
    env.close()
    if data is None:
        raise KeyError(f"missing key {idx}")
    return msgpack.unpackb(data, raw=False)
