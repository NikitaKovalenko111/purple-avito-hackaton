from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"
INFER_SCRIPT = MODEL_DIR / "infer_csv.py"
SERVER_ENV_FILE = ROOT_DIR / "server" / ".env"
SERVER_VENV_PYTHON = ROOT_DIR / "server" / ".venv" / "Scripts" / "python.exe"
MODEL_VENV_PYTHON = ROOT_DIR / "model" / "venv" / "Scripts" / "python.exe"


def _load_env_file(path: Path) -> dict[str, str]:
    env_map: dict[str, str] = {}
    if not path.exists():
        return env_map

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env_map[key] = value
    return env_map


def _set_if_missing_or_empty(target: dict[str, str], key: str, value: str | None) -> None:
    if not value:
        return
    if not target.get(key):
        target[key] = value


def _python_has_torch(python_exec: Path) -> bool:
    if not python_exec.exists():
        return False
    try:
        result = subprocess.run(
            [str(python_exec), "-c", "import torch"],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def _resolve_python_exec(raw_path: Path | None) -> Path:
    if raw_path is not None:
        resolved = raw_path.expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Python executable not found: {resolved}")
        return resolved

    candidates = [
        Path(sys.executable).resolve(),
        SERVER_VENV_PYTHON.resolve(),
        MODEL_VENV_PYTHON.resolve(),
    ]

    for candidate in candidates:
        if _python_has_torch(candidate):
            return candidate

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("No Python executable found for inference run")


def _resolve_checkpoint_path(raw_path: Path | None) -> Path:
    if raw_path is not None:
        path = raw_path.expanduser().resolve()
        if path.suffix.lower() == ".json":
            candidate_best = path.parent / "cv_best_fold_checkpoint.pt"
            candidate_default = path.parent / "model_checkpoint.pt"
            if candidate_best.exists():
                return candidate_best
            if candidate_default.exists():
                return candidate_default
            raise FileNotFoundError(
                "JSON file was provided instead of checkpoint, and no .pt checkpoint was found nearby: "
                f"{candidate_best} or {candidate_default}"
            )
        return path

    env_path = os.getenv("MODEL_CHECKPOINT_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    model_checkpoint = CHECKPOINTS_DIR / "model_checkpoint.pt"
    if model_checkpoint.exists():
        return model_checkpoint

    cv_best_checkpoint = CHECKPOINTS_DIR / "cv_best_fold_checkpoint.pt"
    if cv_best_checkpoint.exists():
        return cv_best_checkpoint

    return model_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference for rnc_test.csv via model/infer_csv.py")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=ROOT_DIR / "rnc_test.csv",
        help="Input CSV path (default: ./rnc_test.csv)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=MODEL_DIR / "reports" / "rnc_test_predictions.csv",
        help="Output CSV path (default: ./model/reports/rnc_test_predictions.csv)",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint path (.pt). If omitted, uses MODEL_CHECKPOINT_PATH or model/checkpoints defaults.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding passed to infer_csv.py",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Progress print frequency passed to infer_csv.py",
    )
    parser.add_argument(
        "--use-llm-drafts",
        dest="use_llm_drafts",
        action="store_true",
        default=True,
        help="Enable OpenRouter-based draft generation (default: enabled).",
    )
    parser.add_argument(
        "--no-llm-drafts",
        dest="use_llm_drafts",
        action="store_false",
        help="Disable OpenRouter-based draft generation.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=SERVER_ENV_FILE,
        help="Path to .env file with OPENROUTER_* variables (default: ./server/.env)",
    )
    parser.add_argument(
        "--python-exec",
        type=Path,
        default=None,
        help="Python executable to run infer_csv.py (default: auto-select interpreter with torch).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_csv = args.input_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint_path)
    python_exec = _resolve_python_exec(args.python_exec)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    if not INFER_SCRIPT.exists():
        raise FileNotFoundError(f"Inference script not found: {INFER_SCRIPT}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    child_env = os.environ.copy()
    env_file = args.env_file.expanduser().resolve()
    env_values = _load_env_file(env_file)
    _set_if_missing_or_empty(child_env, "OPENROUTER_API_KEY", env_values.get("OPENROUTER_API_KEY"))
    _set_if_missing_or_empty(child_env, "OPENROUTER_MODEL", env_values.get("OPENROUTER_MODEL"))
    _set_if_missing_or_empty(child_env, "LLM_API_URL", env_values.get("LLM_API_URL"))
    _set_if_missing_or_empty(child_env, "OPENROUTER_BASE_URL", env_values.get("OPENROUTER_BASE_URL"))

    cmd = [
        str(python_exec),
        str(INFER_SCRIPT),
        "--input-csv",
        str(input_csv),
        "--output-csv",
        str(output_csv),
        "--checkpoint-path",
        str(checkpoint_path),
        "--encoding",
        args.encoding,
        "--print-every",
        str(args.print_every),
    ]
    if args.use_llm_drafts:
        cmd.append("--use-llm-drafts")
        if child_env.get("OPENROUTER_MODEL"):
            cmd.extend(["--openrouter-model", child_env["OPENROUTER_MODEL"]])
        openrouter_base_url = child_env.get("OPENROUTER_BASE_URL") or child_env.get("LLM_API_URL")
        if openrouter_base_url:
            cmd.extend(["--openrouter-base-url", openrouter_base_url])

    print(f"[RUN] Input CSV: {input_csv}")
    print(f"[RUN] Output CSV: {output_csv}")
    print(f"[RUN] Checkpoint: {checkpoint_path}")
    print(f"[RUN] Python executable: {python_exec}")
    print(f"[RUN] LLM drafts: {args.use_llm_drafts}")
    print(f"[RUN] Env file: {env_file}")
    print(f"[RUN] OpenRouter key present: {bool(child_env.get('OPENROUTER_API_KEY'))}")
    print(f"[RUN] OpenRouter model: {child_env.get('OPENROUTER_MODEL', '')}")
    print(f"[RUN] Command: {' '.join(cmd)}")

    subprocess.run(cmd, check=True, cwd=str(ROOT_DIR), env=child_env)


if __name__ == "__main__":
    main()
