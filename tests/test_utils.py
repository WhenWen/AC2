from pathlib import Path
from hydra import compose, initialize_config_dir
import time

import verl.trainer.main_ppo as main_ppo
from ac2.utils.experiment_utils import manifest_dump

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent

def test_manifest():
    start = time.perf_counter()

    VERL_CONFIG_DIR = str(Path(main_ppo.__file__).parent / "config")
    with initialize_config_dir(config_dir=VERL_CONFIG_DIR, version_base=None):
        cfg = compose(config_name="ppo_trainer")

    MANIFEST_DIR = Path(__file__).resolve().parent / "test_manifest"
    manifest_dump(
        manifest_dir=MANIFEST_DIR,
        config=cfg,
        src_dir=REPO_ROOT / "src",
        extra_files=(Path(__file__).resolve(),)
    )

    end = time.perf_counter()

    print(f"Manifest write complete, time: {end-start}")

if __name__=="__main__":
    test_manifest()