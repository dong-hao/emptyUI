"""Hunk-level staging helper for batch commits (dev tool, not part of the app).

Usage (run from the repository root):
    python scripts/dev/stage_block.py list  <path>
    python scripts/dev/stage_block.py stage <path> <block>[,<block>...]

`list` prints the change blocks between HEAD and the working file, numbered.
`stage` writes the index blob = HEAD with only the selected blocks applied.
The working file is never modified. Block numbers are only valid until the
next commit: list again before staging the next batch.
"""

from __future__ import annotations

import difflib
import subprocess
import sys


def head_lines(path: str) -> list[str]:
    out = subprocess.run(["git", "show", f"HEAD:{path}"],
                         capture_output=True, text=True, check=True).stdout
    return out.splitlines(keepends=True)


def work_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        return fh.read().splitlines(keepends=True)


def change_blocks(path: str):
    head, work = head_lines(path), work_lines(path)
    matcher = difflib.SequenceMatcher(None, head, work, autojunk=False)
    ops = [op for op in matcher.get_opcodes() if op[0] != "equal"]
    return head, work, ops


def cmd_list(path: str) -> None:
    head, work, ops = change_blocks(path)
    for number, (tag, i1, i2, j1, j2) in enumerate(ops):
        print(f"#{number} {tag} HEAD[{i1}:{i2}] WORK[{j1}:{j2}]")
        for line in head[i1:i2]:
            print("   - " + line.rstrip("\n"))
        for line in work[j1:j2]:
            print("   + " + line.rstrip("\n"))


def cmd_stage(path: str, selected: set[int]) -> None:
    head, work, ops = change_blocks(path)
    out: list[str] = []
    cursor = 0
    for number, (_tag, i1, i2, j1, j2) in enumerate(ops):
        out.extend(head[cursor:i1])
        out.extend(work[j1:j2] if number in selected else head[i1:i2])
        cursor = i2
    out.extend(head[cursor:])
    blob = subprocess.run(["git", "hash-object", "-w", "--stdin"],
                          input="".join(out).encode("utf-8"),
                          capture_output=True, check=True).stdout.decode().strip()
    subprocess.run(["git", "update-index", "--add", "--cacheinfo",
                    f"100644,{blob},{path}"], check=True)
    print(f"staged {path}: blocks {sorted(selected)} -> blob {blob[:10]}")


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "list":
        cmd_list(argv[2])
        return 0
    if len(argv) == 4 and argv[1] == "stage":
        cmd_stage(argv[2], {int(x) for x in argv[3].split(",")})
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
