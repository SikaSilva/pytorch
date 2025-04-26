import json
from pathlib import Path


def generate_gdb_commands(input_json_path, output_txt_path):
    """
    生成gdb初始化命令文件
    :param input_json_path: 输入的JSON文件路径
    :param output_txt_path: 输出的gdb命令文件路径
    """
    with open(input_json_path, 'r') as f:
        entries = json.load(f)

    commands = [
        "add-auto-load-safe-path /root/workspace/pytorch/.gdbinit"
    ]

    for entry in entries:
        file_path = entry["uri"]["path"]
        if not file_path.endswith(".py"):
            # 取第一个range的起始行号
            line = entry["range"][0]["line"]
            commands.append(f"break {file_path}:{line}")

    with open(output_txt_path, 'w') as f:
        f.write('\n'.join(commands))


if __name__ == "__main__":
    curr_path = Path(__file__).parent
    bp_path = curr_path / 'breakpoints.json'
    gdb_commands_path = curr_path / 'gdb_commands.txt'
    generate_gdb_commands(bp_path, gdb_commands_path)
