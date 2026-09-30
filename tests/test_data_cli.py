import json
import subprocess
import sys
from pathlib import Path


def test_generate_and_validate_dataset_cli(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "minimum"
    generate = subprocess.run(
        [
            sys.executable,
            "scripts/generate_dataset.py",
            "--task",
            "minimum",
            "--count",
            "12",
            "--seed",
            "42",
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Generated dataset manifest" in generate.stdout
    manifest_path = output_dir / "manifest.json"
    validate = subprocess.run(
        [sys.executable, "scripts/validate_dataset.py", "--manifest", str(manifest_path)],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Dataset validation ok" in validate.stdout
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["count"] == 12
