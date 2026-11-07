#!/usr/bin/env python3
"""Install one pinned local music model. Never accepts or opens audio files."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import urllib.request
from urllib.parse import urlparse
import venv

ROOT = Path(__file__).resolve().parent
MODEL_ID = "mlx-community/MOSS-Music-8B-Thinking-8bit"
REPO = "Neywa-Labs/mlx-audio"
MODEL = ROOT / "models" / MODEL_ID.split("/")[1]
PYTHON = ROOT / "venv" / "bin" / "python"


def save(path: Path, data: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "local-moss-music-setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def run(args: list[str], env: dict[str, str]) -> None:
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def protected_state() -> dict:
    project = ROOT.parent
    state = {}
    for name in ("app.py", "requirements.txt", ".venv/pyvenv.cfg"):
        path = project / name
        if path.is_file():
            state[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    existing_python = project / ".venv/bin/python"
    if existing_python.is_file():
        state["packages"] = subprocess.check_output(
            [str(existing_python), "-m", "pip", "freeze"], text=True,
            stderr=subprocess.DEVNULL,
        )
    return state


def lock_requirements(report: dict, source: str) -> str:
    lines = ["# Exact runtime versions; source is fixed to an immutable GitHub commit."]
    names = set()
    for item in report["install"]:
        name = item["metadata"]["name"]
        version = item["metadata"]["version"]
        normalized = name.lower().replace("_", "-")
        names.add(normalized)
        url = item["download_info"]["url"]
        if normalized == "mlx-audio":
            if url != source:
                raise RuntimeError("Unexpected mlx-audio source in dependency resolution.")
            lines.append(f"{name} @ {source}")
        else:
            if urlparse(url).hostname not in {"files.pythonhosted.org", "pypi.org"}:
                raise RuntimeError(f"Dependency outside official PyPI distribution: {name}")
            lines.append(f"{name}=={version}")
    if not {"mlx-audio", "mlx", "huggingface-hub"}.issubset(names):
        raise RuntimeError("The dependency report is incomplete.")
    return "\n".join(lines) + "\n"


DOWNLOAD_CODE = r'''
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path.cwd()
lock = json.loads((root / 'setup-lock.json').read_text())
model = root / 'models' / 'MOSS-Music-8B-Thinking-8bit'
snapshot_download(
    repo_id=lock['model_id'], revision=lock['model_revision'], local_dir=model,
    allow_patterns=['*.safetensors', '*.json', '*.txt', '*.model', '*.jinja',
                    'LICENSE*', 'NOTICE*', 'README.md'],
    max_workers=2, token=False,
)
weights = list(model.glob('*.safetensors'))
if not weights or not (model / 'config.json').is_file():
    raise RuntimeError('Missing model weights or config.')
index = model / 'model.safetensors.index.json'
if index.is_file():
    for shard in set(json.loads(index.read_text())['weight_map'].values()):
        if not (model / shard).is_file():
            raise RuntimeError('Missing model shard: ' + shard)
print('DESCARGA COMPLETA:', model, flush=True)
print('Pesos descargados: %.2f GB' % (sum(p.stat().st_size for p in weights) / 1e9), flush=True)
'''

OFFLINE_CODE = r'''
import os, sys
from pathlib import Path
# Also reject Python network calls if an accidental online code path is reached.
def audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        raise RuntimeError('Network disabled for the local model check')
sys.addaudithook(audit)
import mlx.core as mx
from mlx_audio.stt import load
if not mx.metal.is_available():
    raise RuntimeError('MLX Metal is unavailable')
model = load(str(Path.cwd() / 'models' / 'MOSS-Music-8B-Thinking-8bit'))
if not callable(getattr(model, 'generate', None)):
    raise RuntimeError('Loaded model has no generation API')
print('CARGA LOCAL SIN AUDIO: OK. Metal disponible; modelo cargado.', flush=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("This installer requires native Apple Silicon macOS.")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Run this script with native Python 3.12 (python3.12).")
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise RuntimeError("At least 30 GiB free space is required for setup.")
    before = protected_state()
    status = {"model_id": MODEL_ID, "audio_used": False, "status": "setting_up"}
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("PYTHON", "PIP_", "HF_", "HUGGINGFACE_")):
            env.pop(key)
    for name in ("cache", "tmp", "models"):
        (ROOT / name).mkdir(exist_ok=True)
    env.update({
        "HF_HOME": str(ROOT / "cache/huggingface"),
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "PIP_CACHE_DIR": str(ROOT / "cache/pip"), "TMPDIR": str(ROOT / "tmp"),
        "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    try:
        if not PYTHON.is_file():
            venv.EnvBuilder(with_pip=True).create(ROOT / "venv")
        run([str(PYTHON), "-c", "import platform,sys; assert sys.version_info[:2] == (3,12); assert platform.machine() == 'arm64'; assert sys.prefix != sys.base_prefix"], env)
        lock_path = ROOT / "setup-lock.json"
        if lock_path.is_file():
            lock = json.loads(lock_path.read_text())
            if lock["model_id"] != MODEL_ID or lock["source_repo"] != REPO:
                raise RuntimeError("Unexpected existing setup lock.")
        else:
            print("Fijando revisiones de las fuentes oficiales…", flush=True)
            commit = fetch_json(f"https://api.github.com/repos/{REPO}/commits/main")["sha"]
            revision = fetch_json(f"https://huggingface.co/api/models/{MODEL_ID}")["sha"]
            lock = {"source_repo": REPO, "source_commit": commit,
                    "model_id": MODEL_ID, "model_revision": revision,
                    "python": platform.python_version()}
            save(lock_path, lock)
        source = f"https://github.com/{REPO}/archive/{lock['source_commit']}.tar.gz"
        pip = [str(PYTHON), "-m", "pip", "--isolated", "--disable-pip-version-check",
               "--cache-dir", str(ROOT / "cache/pip")]
        install = ["install", "--index-url", "https://pypi.org/simple"]
        requirements = ROOT / "runtime-lock.txt"
        if not requirements.is_file():
            print("Resolviendo y fijando todas las dependencias…", flush=True)
            report = ROOT / "dependency-resolution.json"
            run(pip + install + ["--dry-run", "--ignore-installed", "--report", str(report), source], env)
            requirements.write_text(lock_requirements(json.loads(report.read_text()), source))
        print("Instalando versiones fijadas en el entorno separado…", flush=True)
        run(pip + install + ["--no-deps", "-r", str(requirements)], env)
        run(pip + ["check"], env)
        with (ROOT / "installed-versions.txt").open("w") as handle:
            subprocess.run(pip + ["freeze", "--all"], cwd=ROOT, env=env, check=True, stdout=handle)
        run([str(PYTHON), "-c", "import mlx_audio.stt.models.moss_music"], env)
        print("Descargando únicamente MOSS-Music-8B-Thinking-8bit (~10 GB). Puede tardar…", flush=True)
        run([str(PYTHON), "-c", DOWNLOAD_CODE], env)
        status.update(status="downloaded", model_downloaded=True, model_revision=lock["model_revision"])
        save(ROOT / "status.json", status)
        offline = dict(env, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        print("Comprobando carga local en modo offline; no se usa audio…", flush=True)
        run([str(PYTHON), "-c", OFFLINE_CODE], offline)
        status.update(status="ready_for_real_wav", model_loaded=True)
        print("\nLISTO: descarga y carga local verificadas. Elige un WAV real de 15–30 segundos para la prueba auditiva.", flush=True)
    except Exception as exc:
        status.update(status="setup_failed", error=str(exc))
        raise
    finally:
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        status["original_files_and_environment_unchanged"] = protected_state() == before
        save(ROOT / "status.json", status)
        if not status["original_files_and_environment_unchanged"]:
            raise RuntimeError("The original project state changed during setup; inspect it before continuing.")


if __name__ == "__main__":
    main()
