import torch
from torch import Tensor
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.autograd import gradcheck, gradgradcheck


class MyAttentionPyOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q: Tensor, k: Tensor, v: Tensor):
        assert q.dim() == 2, f"dim of q should be 2 but got {q.dim()}"
        assert k.dim() == 2, f"dim of k should be 2 but got {k.dim()}"
        assert v.dim() == 2, f"dim of v should be 2 but got {v.dim()}"
        assert q.shape[0] == k.shape[0] == v.shape[0]
        assert q.shape[1] == k.shape[1]

        x = torch.matmul(q, k.transpose(0, 1))
        a = torch.tanh(x)
        o = torch.matmul(a, v)

        ctx.save_for_backward(q, k, a, v)
        return o, a

    @staticmethod
    def backward(ctx, grad_out_o, grad_out_a):
        q, k, a, v = ctx.saved_tensors

        # Gradients of the loss to a, 两条出边
        # m*m = m*k @ (m*k)^T
        grad_a = torch.matmul(grad_out_o, v.transpose(0, 1)) + grad_out_a

        # Gradients of the loss to x
        # m*m = m*m * m*m
        grad_x = grad_a * (1.0 - (a ** 2))

        # Gradients of the loss to q
        # m*n = m*m @ m*n
        grad_q = torch.matmul(grad_x, k)

        # Gradients of the loss to k
        # m*n = (m*m)^T @ m*n
        grad_k = torch.matmul(grad_x.transpose(0, 1), q)

        # Gradients of the loss to v
        grad_v = torch.matmul(a.transpose(0, 1), grad_out_o)

        return grad_q, grad_k, grad_v


class TestMyAttention(TestCase):
    def test_py_method(self):
        dtype = torch.float64
        q = torch.randn(3, 3, dtype=dtype, requires_grad=False)
        k = torch.randn(3, 3, dtype=dtype, requires_grad=False)
        v = torch.randn(3, 2, dtype=dtype, requires_grad=False)
        result = torch.my_attention(q, k, v)
        print(result)

        q = torch.randn(10, 5, dtype=dtype, requires_grad=True)
        k = torch.randn(10, 5, dtype=dtype, requires_grad=True)
        v = torch.randn(10, 8, dtype=dtype, requires_grad=True)
        inputs = (q, k, v)
        assert gradcheck(MyAttentionPyOp.apply, inputs)
        assert gradgradcheck(MyAttentionPyOp.apply, inputs)

    def test_native_method(self):
        dtype = torch.float64
        print('test native method')

        q = torch.randn(3, 4, dtype=dtype, requires_grad=True)
        k = torch.randn(3, 4, dtype=dtype, requires_grad=True)
        v = torch.randn(3, 2, dtype=dtype, requires_grad=True)
        inputs = (q, k, v)
        assert gradcheck(torch.my_attention, inputs, check_undefined_grad=False)
        assert gradgradcheck(torch.my_attention, inputs, check_undefined_grad=False)

    def test_my_ping_native_method(self):
        x = torch.randn(2, 2, requires_grad=False)
        y = torch.my_ping(x)
        print("x =", x)
        print("y =", y)

    def test_my_native_backward(self):
        dtype = torch.float64
        print('test native function backward')
        q = torch.randn(3, 4, dtype=dtype, requires_grad=True)
        k = torch.randn(3, 4, dtype=dtype, requires_grad=True)
        v = torch.randn(3, 2, dtype=dtype, requires_grad=True)
        x = torch.ones(3, 2, dtype=dtype)
        o, a = torch.my_attention(q, k, v)
        y = torch.matmul(a, x)
        z = o + y
        print("z: ")
        print(z)
        sumans = z.sum()
        print("sum: ")
        print(sumans)
        sumans.backward()
        print("backward of q: ")
        print(q.grad)

# instantiate_device_type_tests(TestMyAttention, globals(), only_for=("cpu"))


if __name__ == '__main__':
    run_tests()
