"""步骤5：纹理优化与贴图导出。"""

import argparse
import subprocess
from pathlib import Path


def run_step(args, cwd):
    print("[step_texture]", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def build_parser():
    parser = argparse.ArgumentParser(description="Step 5: run texture optimization")
    parser.add_argument("--repo-root", type=str, required=True)
    parser.add_argument("--python-exec", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--num-view", type=int, default=16)
    return parser


def main():
    args = build_parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    texture_root = repo_root / "pipelines" / "steps" / "TexturePipeline"
    data_root = Path(args.data_root)

    required_inputs = [
        data_root / "images",
        data_root / "2dgs_recon.obj",
        data_root / "transforms.json",
    ]
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        print(
            f"[step_texture] skip dataset={data_root.name}: "
            f"missing inputs: {', '.join(missing)}"
        )
        return

    cmd = [
        args.python_exec,
        "run.py",
        "--img_root",
        str(data_root / "images"),
        "--mesh_path",
        str(data_root / "2dgs_recon.obj"),
        "--cam_path",
        str(data_root / "transforms.json"),
        "--save_root",
        str(data_root / "texture"),
        "--num_view",
        str(args.num_view),
    ]
    run_step(cmd, str(texture_root))


if __name__ == "__main__":
    main()
