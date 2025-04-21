#!/usr/bin/env python

import os
import subprocess
from time import sleep

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.distributed as dist


# def _setup_dist_env_from_slurm(args):
#     while not os.environ.get("MASTER_ADDR", ""):
#         os.environ["MASTER_ADDR"] = (
#             subprocess.check_output(
#                 "sinfo -Nh -n %s | head -n 1 | awk '{print $1}'" % os.environ["SLURM_NODELIST"],
#                 shell=True,
#             )
#             .decode()
#             .strip()
#         )
#         sleep(1)
#     os.environ["MASTER_PORT"] = str(args.master_port)
#     os.environ["RANK"] = os.environ["SLURM_PROCID"]
#     # os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]
#     os.environ["WORLD_SIZE"] = 4
#     os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
#     # os.environ["LOCAL_WORLD_SIZE"] = os.environ["SLURM_NTASKS_PER_NODE"]
#     os.environ["LOCAL_WORLD_SIZE"] = 4

def _setup_dist_env_from_launcher(args):
    """
    Sets up distributed environment, relying primarily on variables
    set by a launcher like torchrun. Sets a default MASTER_PORT if needed.
    """
    # torchrun sets MASTER_ADDR, RANK, WORLD_SIZE, LOCAL_RANK, LOCAL_WORLD_SIZE
    # It might also set MASTER_PORT, but we can provide a default.

    if "MASTER_ADDR" not in os.environ:
        # If launcher didn't set it (unlikely for torchrun), default for single node
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        print("WARNING: MASTER_ADDR not set by launcher, defaulting to 127.0.0.1")

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(args.master_port)
        print(f"INFO: MASTER_PORT not set by launcher, using args.master_port: {args.master_port}")
    else:
        # If launcher set it, respect that value
        print(f"INFO: Using MASTER_PORT set by launcher: {os.environ['MASTER_PORT']}")

    # Optional: Check if essential variables are set
    required_vars = ["RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"]
    missing_vars = [v for v in required_vars if v not in os.environ]
    if missing_vars:
        print(f"ERROR: Required environment variables not set by launcher: {missing_vars}")
        print("       Ensure you are using a PyTorch distributed launcher like torchrun.")


_INTRA_NODE_PROCESS_GROUP, _INTER_NODE_PROCESS_GROUP = None, None
_LOCAL_RANK, _LOCAL_WORLD_SIZE = -1, -1


def get_local_rank() -> int:
    return _LOCAL_RANK


def get_local_world_size() -> int:
    return _LOCAL_WORLD_SIZE


def distributed_init(args):
    if any([x not in os.environ for x in ["RANK", "WORLD_SIZE", "MASTER_PORT", "MASTER_ADDR"]]):
        # _setup_dist_env_from_slurm(args)
        _setup_dist_env_from_launcher(args)

    dist.init_process_group("nccl")
    fs_init.initialize_model_parallel(args.model_parallel_size)
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

    global _LOCAL_RANK, _LOCAL_WORLD_SIZE
    _LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    _LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])

    global _INTRA_NODE_PROCESS_GROUP, _INTER_NODE_PROCESS_GROUP
    local_ranks, local_world_sizes = [
        torch.empty([dist.get_world_size()], dtype=torch.long, device="cuda") for _ in (0, 1)
    ]
    dist.all_gather_into_tensor(local_ranks, torch.tensor(get_local_rank(), device="cuda"))
    dist.all_gather_into_tensor(local_world_sizes, torch.tensor(get_local_world_size(), device="cuda"))
    local_ranks, local_world_sizes = local_ranks.tolist(), local_world_sizes.tolist()
    node_ranks = [[0]]
    for i in range(1, dist.get_world_size()):
        if len(node_ranks[-1]) == local_world_sizes[i - 1]:
            node_ranks.append([])
        else:
            assert local_world_sizes[i] == local_world_sizes[i - 1]
        node_ranks[-1].append(i)
    for ranks in node_ranks:
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            assert _INTRA_NODE_PROCESS_GROUP is None
            _INTRA_NODE_PROCESS_GROUP = group
    assert _INTRA_NODE_PROCESS_GROUP is not None

    if min(local_world_sizes) == max(local_world_sizes):
        for i in range(get_local_world_size()):
            group = dist.new_group(list(range(i, dist.get_world_size(), get_local_world_size())))
            if i == get_local_rank():
                assert _INTER_NODE_PROCESS_GROUP is None
                _INTER_NODE_PROCESS_GROUP = group
        assert _INTER_NODE_PROCESS_GROUP is not None


def get_intra_node_process_group():
    assert _INTRA_NODE_PROCESS_GROUP is not None, "Intra-node process group is not initialized."
    return _INTRA_NODE_PROCESS_GROUP


def get_inter_node_process_group():
    assert _INTRA_NODE_PROCESS_GROUP is not None, "Intra- and inter-node process groups are not initialized."
    return _INTER_NODE_PROCESS_GROUP
