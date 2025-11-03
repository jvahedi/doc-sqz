"""
DOCX conversion utilities for docsqz.
Pandoc-based conversion: DOCX → HTML/Markdown/LaTeX/etc.
"""

from pathlib import Path
import pypandoc

def convert_docx(
    input_path,
    to="html",                   # "html", "gfm", "md", "latex", etc.
    out_path=None,
    extract_media_dir="media",   # images saved here; set None to skip
    math="mathml",               # "mathml" (HTML), "latex" (MathJax), or None
    add_toc=False
):
    input_path = Path(input_path)
    assert input_path.suffix.lower() == ".docx", "Input must be a .docx file"

    extra = []
    if math == "mathml":
        extra += ["--mathml"]
    elif math == "latex":
        extra += ["--mathjax"]   # renders via MathJax in HTML
    if add_toc:
        extra += ["--toc"]
    if extract_media_dir:
        extra += [f"--extract-media={extract_media_dir}"]

    result = pypandoc.convert_file(str(input_path), to, extra_args=extra)

    if out_path:
        Path(out_path).write_text(result, encoding="utf-8")

    return result

def df_to_markdown_table(df, newline_as=" "):
    """
    Convert a masked df containing only one table's cells
    into a markdown table. Assumes columns ['row','col','text'].
    newline_as = " " for flat text, "<br>" to keep visible breaks.
    """
    n_rows = int(df["row"].max()) + 1
    n_cols = int(df["col"].max()) + 1
    grid = [["" for _ in range(n_cols)] for __ in range(n_rows)]

    for _, r in df.iterrows():
        text = str(r["text"] or "")
        text = text.replace("\n", newline_as)   # 👈 handle line breaks
        grid[int(r["row"])][int(r["col"])] = text

    # build markdown
    to_md = lambda row: "| " + " | ".join(row) + " |"
    lines = [to_md(grid[0])]
    lines.append("|" + "|".join("---" for _ in grid[0]) + "|")
    for row in grid[1:]:
        lines.append(to_md(row))
    return "\n".join(lines)