#!/usr/bin/env python3
"""Post-process a freshly generated Doxygen flat output into our tidy subfolder
layout.

Expected starting state: Doxygen 1.9.x has just regenerated into the repo root
as it does by default (hundreds of files: *.html, *.css, *.js, *.png, *.jpeg,
*.svg, *.map, *.md5 plus a search/ subdirectory).

After running this script you get:

    css/   - *.css
    js/    - *.js  (+ patched navtree.js)
    images/ui/      - Doxygen UI pngs (folderopen, nav_f, tab_b, sync_*, ...)
    images/authors/ - author photos (hard-coded list below)
    images/board/   - board photos (hard-coded list below)
    graphs/         - *.svg + *.map + *.md5 (call/include graphs)
    search/         - untouched (Doxygen ships it as a subfolder)
    *.html, README.md stay at repo root.

All referring paths are rewritten:
    - href=/src= in HTML -> prefixed with the new subfolder
    - url(...) in CSS   -> prefixed with ../images/ui/
    - xlink:href in SVG -> prefixed with ../ (so standalone SVG points at root HTML)
    - js/navtree.js     -> dynamic getScript() prepends js/, sync icons use images/ui/

Run from the repo root: `python3 tools/reorg.py`.

Idempotent: running it a second time is a no-op.
"""
import os
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def atomic_write(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write atomically via a sibling .tmp + os.replace, so an interrupted run
    never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding=encoding)
        os.replace(tmp, path)
    except OSError as e:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise OSError(f"failed to write {path}: {e}") from e


def safe_read(path: Path, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding)
    except OSError as e:
        raise OSError(f"failed to read {path}: {e}") from e

# ---- hard-coded classification (extend here when adding non-Doxygen assets) --
UI_ICONS = {
    "bc_s.png", "bdwn.png", "closed.png", "doc.png",
    "folderclosed.png", "folderopen.png",
    "nav_f.png", "nav_g.png", "nav_h.png",
    "open.png", "splitbar.png",
    "sync_off.png", "sync_on.png",
    "tab_a.png", "tab_b.png", "tab_h.png", "tab_s.png",
}
AUTHOR_PHOTOS = {
    "AcquadroPatrizio.png", "GiacomoColosio.png",
    "SebastianoColosio.jpeg", "TitoNicolaDrugman.jpeg",
}
BOARD_PHOTOS = {
    "STM32N6570-DK_top_view.jpeg", "STM32N6570-DK_bottom_view.jpeg",
}


def move_if_exists(src: Path, dst_dir: Path):
    if src.exists() and src.is_file():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst_dir / src.name))


def reorganise_files():
    """Move files from root into subfolders. No-op for already-moved files."""
    # CSS
    for p in list(ROOT.glob("*.css")):
        move_if_exists(p, ROOT / "css")
    # JS
    for p in list(ROOT.glob("*.js")):
        move_if_exists(p, ROOT / "js")
    # UI icons
    for name in UI_ICONS:
        move_if_exists(ROOT / name, ROOT / "images" / "ui")
    # Authors
    for name in AUTHOR_PHOTOS:
        move_if_exists(ROOT / name, ROOT / "images" / "authors")
    # Board photos
    for name in BOARD_PHOTOS:
        move_if_exists(ROOT / name, ROOT / "images" / "board")
    # Graphs: svg + map + md5
    for ext in ("*.svg", "*.map", "*.md5"):
        for p in list(ROOT.glob(ext)):
            move_if_exists(p, ROOT / "graphs")


def build_moves_map() -> dict:
    """filename -> new relative path (as seen from a page at repo root)."""
    moves = {}
    for sub in ("css", "js", "images/ui", "images/authors", "images/board", "graphs"):
        d = ROOT / sub
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.is_file():
                moves[p.name] = f"{sub}/{p.name}"
    return moves


def rewrite_root_pages(moves: dict):
    """Rewrite href/src references to moved files inside HTML + README in root."""
    if not moves:
        return 0
    names = sorted(moves.keys(), key=len, reverse=True)
    pat = re.compile(
        r'(?P<left>["\'(=,\s>])(' + "|".join(re.escape(n) for n in names) + r')(?P<right>["\')\s,#?])'
    )

    def sub(m):
        return m.group("left") + moves[m.group(2)] + m.group("right")

    count = 0
    targets = list(ROOT.glob("*.html")) + [ROOT / "README.md"]
    for f in targets:
        if not f.exists():
            continue
        text = safe_read(f)
        new = pat.sub(sub, text)
        if new != text:
            atomic_write(f, new)
            count += 1
    return count


def rewrite_css_urls():
    """url(nav_f.png) -> url(../images/ui/nav_f.png) inside css/*.css."""
    css_dir = ROOT / "css"
    if not css_dir.exists():
        return 0
    pat = re.compile(
        r'url\(\s*(?P<q>["\']?)(' + "|".join(re.escape(n) for n in UI_ICONS) + r')(?P=q)\s*\)'
    )
    count = 0
    for f in css_dir.glob("*.css"):
        text = safe_read(f)
        new = pat.sub(lambda m: f'url({m.group("q")}../images/ui/{m.group(2)}{m.group("q")})', text)
        if new != text:
            atomic_write(f, new)
            count += 1
    return count


def rewrite_svg_hrefs():
    """xlink:href="xxx.html" -> "../xxx.html" in graphs/*.svg, so the graphs
    still navigate to root HTML pages when opened standalone."""
    graphs_dir = ROOT / "graphs"
    if not graphs_dir.exists():
        return 0
    # Skip already-prefixed (./, ../) + absolute (http…) + pure anchors (#)
    pat = re.compile(r'(xlink:href=")(?![./]|https?:|#)([^"#]+\.html)(["#])')
    count = 0
    for f in graphs_dir.glob("*.svg"):
        text = safe_read(f)
        new = pat.sub(r"\1../\2\3", text)
        if new != text:
            atomic_write(f, new)
            count += 1
    return count


def patch_navtree_js():
    """navtree.js needs three surgical edits that the generic regex can't make:
    - getScript() dynamically loads navtreeindex*.js etc. — prepend 'js/'.
    - showSyncOff/On use '+relpath+filename' — insert 'images/ui/' before filename.
    Idempotent (looks for already-patched marker before editing)."""
    f = ROOT / "js" / "navtree.js"
    if not f.exists():
        return False
    text = safe_read(f)
    before = text
    text = text.replace(
        "script.src = scriptName+'.js';",
        "script.src = 'js/'+scriptName+'.js';",
    )
    text = text.replace(
        "'<img src=\"'+relpath+'sync_off.png\"",
        "'<img src=\"'+relpath+'images/ui/sync_off.png\"",
    )
    text = text.replace(
        "'<img src=\"'+relpath+'sync_on.png\"",
        "'<img src=\"'+relpath+'images/ui/sync_on.png\"",
    )
    if text != before:
        atomic_write(f, text)
        return True
    return False


def main():
    print("[reorg] moving files into subfolders...")
    reorganise_files()
    moves = build_moves_map()
    print(f"[reorg] {len(moves)} moved filenames tracked")
    n_pages = rewrite_root_pages(moves)
    print(f"[reorg] rewrote paths in {n_pages} root HTML/README files")
    n_css = rewrite_css_urls()
    print(f"[reorg] rewrote url() in {n_css} CSS files")
    n_svg = rewrite_svg_hrefs()
    print(f"[reorg] rewrote xlink:href in {n_svg} SVG files")
    patched = patch_navtree_js()
    print(f"[reorg] navtree.js patched: {patched}")
    print("[reorg] done")


if __name__ == "__main__":
    main()
