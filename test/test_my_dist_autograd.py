import torch
import torch.distributed.rpc as rpc
import torch.distributed.autograd as dist_autograd
from torch.distributed.rpc import TensorPipeRpcBackendOptions

# 定义一个简单的加法函数，该函数会被远程调用


def my_add(t1, t2):
    return torch.add(t1, t2)


def worker0():
    # 在worker0上执行的主要逻辑
    with dist_autograd.context() as context_id:  # 创建分布式自动梯度上下文
        # 创建两个需要梯度的本地张量
        t1 = torch.rand((3, 3), requires_grad=True)
        t2 = torch.rand((3, 3), requires_grad=True)

        # 第一阶段：远程计算
        # 通过RPC同步调用worker1执行my_add操作
        t3 = rpc.rpc_sync("worker1", my_add, args=(t1, t2))

        # 本地计算：使用远程结果进行矩阵乘法
        t4 = torch.rand((3, 3), requires_grad=True)
        t5 = torch.mul(t3, t4)

        # 计算损失（标量值）
        loss = t5.sum()

        # 第二阶段：分布式反向传播
        # 执行分布式反向传播，从loss开始
        dist_autograd.backward(context_id, [loss])

        # 从上下文中获取梯度字典
        gradients = dist_autograd.get_gradients(context_id)

        # 打印关键信息
        print(f"Loss: {loss.item()}")
        print("Gradients on worker0:")
        for param, grad in gradients.items():
            print(f"Parameter shape: {param.shape}, Gradient shape: {grad.shape}")


def run_worker(rank, world_size):
    """
    RPC工作节点初始化函数
    rank: 当前进程的等级（0或1）
    world_size: 总进程数
    """
    # 配置TensorPipe后端选项
    rpc_backend_options = TensorPipeRpcBackendOptions()
    rpc_backend_options.init_method = "tcp://localhost:29501"

    # 根据rank初始化不同的工作节点
    if rank == 0:
        # 初始化worker0，设置名称、rank和世界大小
        rpc.init_rpc(
            "worker0",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        worker0()  # 执行worker0的主逻辑
    elif rank == 1:
        # 初始化worker1，该节点只接收RPC请求
        rpc.init_rpc(
            "worker1",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=rpc_backend_options,
        )
        # worker1不需要主动执行操作，只需等待RPC请求

    # 阻塞直到所有RPC操作完成
    rpc.shutdown()


if __name__ == "__main__":
    # 设置分布式参数
    world_size = 2

    # 使用多进程启动器启动两个工作节点
    torch.multiprocessing.spawn(
        run_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
