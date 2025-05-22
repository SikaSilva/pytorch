import os
import torch
import torch.distributed as dist

# torchrun --nnodes=1:2 --nproc-per-node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --rdzv-id=5247 test/test_my_torchrun.py


def main():
    # 自动选择适合的后端
    dist.init_process_group(backend='gloo')

    rank = dist.get_rank()
    print(f"\n{'=' * 40}")
    print(f"Process Rank: {rank}")
    print("Environment Variables:")

    # # 打印所有环境变量（按字母顺序排序）
    # for k, v in sorted(os.environ.items()):
    #     print(f"[Rank {rank}] {k}={v}")

    rank = os.getenv("RANK", "-1")
    local_rank = os.getenv("LOCAL_RANK", "-1")
    group_rank = os.getenv("GROUP_RANK", "-1")
    world_size = os.getenv("WORLD_SIZE", "-1")
    role_rank = os.getenv("ROLE_RANK", "-1")
    role_world_size = os.getenv("ROLE_WORLD_SIZE", "-1")
    # print('rank =', rank)
    # print('local_rank =', local_rank)
    # print('group_rank =', group_rank)
    # print('world_size =', world_size)
    # print('role_rank =', role_rank)
    # print('role_world_size =', role_world_size)
    print(f'rank = {rank}, local_rank = {local_rank}, group_rank = {group_rank}, world_size = {world_size}, role_rank = {role_rank}, role_world_size = {role_world_size}, i am started...')

    while True:
        pass

    # dist.destroy_process_group()


if __name__ == "__main__":
    main()
