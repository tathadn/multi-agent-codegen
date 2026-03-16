from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

SANDBOX_IMAGE = "multi-agent-sandbox"
TIMEOUT_SECONDS = 30


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


def run_in_sandbox(files: list[CodeFile]) -> SandboxResult:
    """Write files to a temp directory, run pytest inside the sandbox container, and return the result."""
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
