import numpy as np
from typing import Optional


def parse_dot_bracket(dot_bracket_str):
    """
    将 RNA 二级结构字符串解析为接触矩阵 (Adjacency Matrix)
    支持: . ( ) [ ] { } < >
    """
    n = len(dot_bracket_str)
    # 初始化全 0 矩阵
    matrix = np.zeros((n, n), dtype=int)
    # 使用栈 (Stack) 来匹配不同类型的括号
    # 键为左括号，值为对应的右括号
    bracket_map = {')': '(', ']': '[', '}': '{', '>': '<'}
    stacks = {
        '(': [],
        '[': [],
        '{': [],
        '<': []
    }
    for i, char in enumerate(dot_bracket_str):
        if char in stacks:
            # 如果是左括号，记录索引入栈
            stacks[char].append(i)
        elif char in bracket_map:
            # 如果是右括号，弹出对应栈顶索引并填充矩阵
            left_char = bracket_map[char]
            if stacks[left_char]:
                j = stacks[left_char].pop()
                matrix[i, j] = 1
                matrix[j, i] = 1
            # else:
            #     print(f"警告: 发现未匹配的右括号 '{char}' 位于索引 {i}")
        elif char == '.':
            continue
        elif char == '&':
            # 处理多链分隔符，通常跳过或根据需要记录
            continue
        else:
            # 处理字母类伪结 (A-Z 配对 a-z)
            if char.isalpha():
                if char.isupper(): # 左括号 A, B, C...
                    if char not in stacks: stacks[char] = []
                    stacks[char].append(i)
                else: # 右括号 a, b, c...
                    upper_char = char.upper()
                    if upper_char in stacks and stacks[upper_char]:
                        j = stacks[upper_char].pop()
                        matrix[i, j] = 1
                        matrix[j, i] = 1
    return matrix


def load_sec_struct_file(filepath: str) -> str:
    """
    Reads a secondary structure file (e.g., .dbn).
    Assumes the file might contain header lines and sequence/structure lines.
    Returns the secondary structure string.
    """
    try:
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # Simple heuristic: find the longest line containing only '().' 
        # (and maybe specific characters for pseudoknots if needed like '[]{}')
        # Here we stick to standard dot-bracket characters
        # lines = lines.replace('&', '.')
        allowed = set(".():[]{}<>ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        candidates = [l for l in lines if all(c in allowed for c in l)]
        if candidates:
            return max(candidates, key=len)
        return ""
    except Exception:
        return ""
