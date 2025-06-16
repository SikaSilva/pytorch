from pathlib import Path
from typing import List, Tuple
import functools
import argparse
import threading

DEFAULT_PATHS = [
    './aten/**',
    './build/aten/src/ATen/**',
    './c10/**',
    './caffe2/**',
    './torch/**',
    './torchgen/**'
]

DEFAULT_FILE_TYPES = (
    ".h",
    ".cpp",
    ".py",
    ".yaml",
)


def search_paths(patterns: List[str]) -> List[Path]:
    """
    Search for files based on directory patterns.

    Args:
        patterns: A list of string paths. For each pattern:
            - Ends with '/**': recursively search this directory and all subdirectories.
            - Otherwise (e.g., 'dir/' or 'dir/*'): search only top-level files in this directory.

    Returns:
        A list of pathlib.Path objects pointing to found files.
    """
    results: List[Path] = []

    def _dfs(dir_path: Path):
        # Depth-first traversal: yield files first then recurse into directories
        for entry in dir_path.iterdir():
            if entry.is_file():
                results.append(entry)
            elif entry.is_dir():
                _dfs(entry)

    def _first_diff(p1: Path, p2: Path) -> Tuple[Tuple[str, bool], Tuple[str, bool]]:
        """
        找到两个 Path 的第一个不同 component。
        返回两个元组： (组件字符串, is_filename)
        - is_filename = True 当该 component 是 path 的最后一部分
        """
        parts1 = p1.parts
        parts2 = p2.parts
        # 遍历到较短长度，找不同
        for idx, (c1, c2) in enumerate(zip(parts1, parts2)):
            if c1 != c2:
                is_file1 = (idx == len(parts1) - 1)
                is_file2 = (idx == len(parts2) - 1)
                return (c1, is_file1), (c2, is_file2)
        assert False, f"p1 = {p1}, p2 = {p2}"

    def _path_cmp(p1: Path, p2: Path) -> bool:

        (c1, is_file1), (c2, is_file2) = _first_diff(p1, p2)
        if is_file1 == is_file2:
            # 字典序比较
            return (c1 > c2) - (c1 < c2)
        # 目录 < 文件
        return -1 if not is_file1 else 1

    for pat in patterns:
        path = Path(pat.rstrip('/*'))
        if pat.endswith('/**'):
            # Recursive search
            if path.is_dir():
                _dfs(path)
        elif path.is_dir():
            # Non-recursive: list only files in this directory
            for entry in path.iterdir():
                if entry.is_file():
                    results.append(entry)
        elif path.is_file():
            results.append(path)

    results = sorted(results, key=functools.cmp_to_key(_path_cmp))
    return results


def search_content(part_ps: List[Path], keyword: str) -> List[str]:
    ans_list = []
    for path in part_ps:
        with open(path, mode='r', encoding='utf-8') as infile:
            lines = infile.readlines()
            for line_no, line in enumerate(lines):
                if keyword in line:
                    ans_list.append((path, line_no + 1, line))
    return ans_list


def parse_args():
    p = argparse.ArgumentParser(description="多线程分片执行 search() 示例")
    p.add_argument(
        "-j", type=int, default=1,
        help="并行线程总数（包括主线程）。省略或 -j1 则只用主线程。"
    )
    p.add_argument(
        "-k", "--keyword", required=True,
        help="要在文件名中查找的关键字"
    )
    return p.parse_args()


def chunkify(lst: List, n: int) -> List[List]:
    """
    把 lst 平均切分成 n 份，长度差不超过 1
    """
    k, m = divmod(len(lst), n)
    chunks = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        chunks.append(lst[start:start + size])
        start += size
    return chunks


def main():
    args = parse_args()
    total_threads = max(1, args.j)

    # 切分 ps 为 total_threads 份
    ps = search_paths(DEFAULT_PATHS)
    # print(ps[0].parts)
    ps = [p for p in ps if str(p.parts[-1]).endswith(DEFAULT_FILE_TYPES)]
    parts = chunkify(ps, total_threads)

    # 用来收集各个线程的返回值
    results: List[List[str]] = [None] * total_threads

    def worker(idx: int, part_ps: List[Path]):
        results[idx] = search_content(part_ps, args.keyword)

    threads = []
    # 启动子线程（index 从 1 开始）
    for idx in range(1, total_threads):
        t = threading.Thread(target=worker, args=(idx, parts[idx]))
        t.start()
        threads.append(t)

    # 主线程也跑一个 slice（index 0）
    worker(0, parts[0])

    # 等待所有子线程结束
    for t in threads:
        t.join()

    # 汇总所有结果
    all_hits: List[str] = []
    for sublist in results:
        all_hits.extend(sublist)

    # 按行输出
    for hit in all_hits:
        path: Path
        path, line_no, _ = hit
        print(f'{str(path.absolute())}:{line_no}')


if __name__ == "__main__":
    main()
