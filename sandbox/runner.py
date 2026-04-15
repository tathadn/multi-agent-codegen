from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

SANDBOX_IMAGE = "multi-agent-sandbox"
TIMEOUT_SECONDS = 30
SANDBOX_DIR = Path(__file__).parent

_image_ready = False


@dataclass
class CodeFile:
    filename: str
    content: str


@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    exit_code: int


def _ensure_image() -> None:
    """Build the sandbox Docker image on first use if it isn't already present."""
    global _image_ready
    if _image_ready:
        return

    inspect = subprocess.run(
        ["docker", "image", "inspect", SANDBOX_IMAGE],
        capture_output=True,
        text=True,
    )
    if inspect.returncode == 0:
        _image_ready = True
        return

    build = subprocess.run(
        ["docker", "build", "-t", SANDBOX_IMAGE, str(SANDBOX_DIR)],
        capture_output=True,
        text=True,
    )
    if build.returncode != 0:
        raise RuntimeError(
            f"Failed to build sandbox image '{SANDBOX_IMAGE}':\n{build.stderr}"
        )
    _image_ready = True


def run_in_sandbox(files: list[CodeFile]) -> SandboxResult:
    """Write files to a temp dir, run pytest in the sandbox container, return the result."""
    _ensure_image()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        for f in files:
            dest = tmp / f.filename
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f.content)

        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "-v", f"{tmpdir}:/workspace",
                    SANDBOX_IMAGE,
                ],
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
            )
            return SandboxResult(
                success=proc.returncode == 0,
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )

        except subprocess.TimeoutExpired:
            return SandboxResult(
                success=False,
                stdout="",
                stderr=f"Sandbox timed out after {TIMEOUT_SECONDS} seconds.",
                exit_code=-1,
            )
