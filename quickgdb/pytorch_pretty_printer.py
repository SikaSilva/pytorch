import gdb
import struct


class TensorPrinter:
    """Pretty printer for torch::Tensor类型"""

    def __init__(self, val):
        self.val = val  # 保存GDB的value对象

    def _get_dtype(self):
        """提取Tensor的数据类型"""
        # 获取dtype_字段（不同版本路径可能不同）
        dtype = self.val['dtype_']
        # 提取标量类型枚举值（torch::kFloat等）
        scalar_type = dtype['scalar_type_']
        return scalar_type

    def _get_sizes(self):
        """提取Tensor维度信息"""
        # 访问TensorImpl的sizes字段
        sizes = self.val['impl_']['sizes_']['sizes_']
        # 转换为Python列表
        return [sizes[i] for i in range(sizes.type.sizeof // sizes.type.target().sizeof)]

    def _get_strides(self):
        """提取Tensor步长信息"""
        # 访问TensorImpl的strides字段
        strides = self.val['impl_']['strides_']['sizes_']
        return [strides[i] for i in range(strides.type.sizeof // strides.type.target().sizeof)]

    def _get_data_ptr(self):
        """获取原始数据指针"""
        # 通过storage_访问数据指针（注意不同版本可能路径不同）
        return self.val['storage_']['data_']['ptr_']

    def _read_memory(self, address, dtype, count):
        """从内存地址读取数据"""
        # 获取当前调试进程
        inferior = gdb.selected_inferior()

        # 根据类型确定格式字符和字节大小
        type_map = {
            0: ('B', 1),   # torch::kByte
            1: ('b', 1),   # torch::kChar
            2: ('h', 2),   # torch::kShort
            3: ('i', 4),   # torch::kInt
            4: ('l', 8),   # torch::kLong
            5: ('f', 4),   # torch::kFloat
            6: ('d', 8)    # torch::kDouble
        }

        fmt, size = type_map.get(int(self._get_dtype()), ('B', 1))

        # 读取原始字节数据
        buffer = inferior.read_memory(address, count * size)

        # 使用struct解包二进制数据
        return struct.unpack(f'{"@"}{count}{fmt}', buffer)

    def _format_matrix(self, data, shape):
        """将数据格式化为矩阵字符串"""
        # 使用numpy处理多维数组展示
        # arr = np.array(data).reshape(shape)
        # return str(arr)

    def to_string(self):
        """主打印入口"""
        print('in gdb here')
        try:
            # 检查空Tensor
            if not self.val['impl_']:
                return "Empty Tensor"

            # 获取元数据
            sizes = self._get_sizes()
            strides = self._get_strides()
            dtype = self._get_dtype()
            data_ptr = self._get_data_ptr()
            print('sizes = {}, type = {}', str(sizes), type(sizes))
            print('strides = {}, type = {}', str(strides), type(strides))
            print('dtype = {}, type = {}', str(dtype), type(dtype))
            print('data_ptr = {}, type = {}', str(data_ptr), type(data_ptr))

            # # 计算元素总数
            # numel = 1
            # for dim in sizes:
            #     numel *= dim

            # if numel == 0:
            #     return "Empty Tensor"

            # # 读取数据（这里简化处理，假设内存连续）
            # raw_data = self._read_memory(data_ptr, dtype, numel)

            # # 格式化为矩阵（这里展示2D情况）
            # if len(sizes) == 2:
            #     return self._format_matrix(raw_data, sizes)
            # else:
            #     # 高维Tensor显示简要信息
            #     return f"Tensor(shape={sizes}, dtype={dtype}, data={raw_data[:10]}...)"

        except Exception as e:
            return f"Tensor Printing Error: {str(e)}"


def register_pretty_printers():
    """注册Tensor的pretty printer"""
    pp = gdb.printing.RegexpCollectionPrettyPrinter("torch_tensor")
    pp.add_printer('torch::Tensor', '^torch::Tensor$', TensorPrinter)
    pp.add_printer('at::Tensor', '^at::Tensor$', TensorPrinter)
    gdb.printing.register_pretty_printer(gdb.current_objfile(), pp)


register_pretty_printers()
