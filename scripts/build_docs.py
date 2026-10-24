"""Build the pdoc HTML documentation into docs/.

pdoc prints the values of module constants, and the paths in
``fdm_defect.config`` are absolute (e.g. ``C:/Users/<name>/.../data``). This
script strips the machine-specific part above the repository folder, so the
committed docs neither leak local paths nor change between machines.

Usage:
  python scripts/build_docs.py
"""

import shutil
from pathlib import Path

import pdoc
import pdoc.render

from fdm_defect import __version__, config

OUT = Path("docs")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    pdoc.render.configure(docformat="numpy", footer_text=f"fdm-defect {__version__}")
    pdoc.pdoc("fdm_defect", output_directory=OUT)

    # e.g. "C:/Users/me/Downloads/" in front of "fdm-defect-detection/data/..."
    prefix = config.PROJECT_ROOT.parent
    prefixes = {prefix.as_posix() + "/", str(prefix) + "\\", str(prefix).replace("\\", "\\\\") + "\\\\"}
    for page in OUT.rglob("*"):
        if page.suffix in (".html", ".js"):
            text = page.read_text(encoding="utf-8")
            for p in prefixes:
                text = text.replace(p, "")
            page.write_text(text, encoding="utf-8")
    print(f"docs written to {OUT}/")


if __name__ == "__main__":
    main()
