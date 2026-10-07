import subprocess
import zipfile
from collections.abc import Iterable
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

# Directory/segment names never worth snapshotting.
_SKIP = {"__pycache__", ".git"}


def _capture(cmd: list[str]) -> str:
    """Run ``cmd`` and return stdout, or "" if the command is missing/fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def manifest_dump(
    manifest_dir: Path,
    config: DictConfig | ListConfig,
    src_dir: Path,
    extra_files: Iterable[Path] = (),
) -> None:
    """Snapshot an experiment for reproducibility.

    Writes into ``manifest_dir``:
      - ``config.yaml``    -- the full resolved training config.
      - ``git_commit.txt`` -- ``src_dir``'s repo HEAD, suffixed ``-dirty`` if the tree diverges.
      - ``uv_freeze.txt``  -- exact installed package versions in the active venv.
      - ``code.zip``       -- a zip of ``src_dir`` plus any ``extra_files``
                              (e.g. the runner and its sbatch wrapper).
    """
    manifest_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, manifest_dir / "config.yaml")

    # Provenance: commit (+dirty flag) and the resolved dependency versions.
    head = _capture(["git", "-C", str(src_dir), "rev-parse", "HEAD"]).strip()
    dirty = bool(_capture(["git", "-C", str(src_dir), "status", "--porcelain"]).strip())
    (manifest_dir / "git_commit.txt").write_text(f"{head}{'-dirty' if dirty else ''}\n")
    (manifest_dir / "uv_freeze.txt").write_text(_capture(["uv", "pip", "freeze"]))

    with zipfile.ZipFile(manifest_dir / "code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in src_dir.rglob("*"):
            if p.is_file() and not (_SKIP & set(p.parts)) and p.suffix != ".pyc":
                z.write(p, p.relative_to(src_dir.parent))
        for f in extra_files:
            z.write(f, f.name)
