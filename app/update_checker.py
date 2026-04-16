import logging
import os
import re
import shlex
import subprocess
import sys
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

logger = logging.getLogger(__name__)


def _format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _run_command(command: list[str], cwd: str, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out: {_format_command(command)}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        if not detail:
            detail = f"exit code {result.returncode}"
        raise RuntimeError(f"{_format_command(command)} failed: {detail}")

    return (result.stdout or "").strip()


class _GitWorkerBase(QObject):
    error = Signal(str)
    finished = Signal()

    def __init__(self, repo_root: str, remote: str, branch: Optional[str], repo_web_url: str):
        super().__init__()
        self.repo_root = repo_root
        self.remote = (remote or "origin").strip() or "origin"
        self.branch = branch.strip() if branch else None
        self.repo_web_url = repo_web_url.rstrip("/")

    def _run_git(self, args: list[str], timeout: int = 60) -> str:
        return _run_command(["git", *args], cwd=self.repo_root, timeout=timeout)

    def _ensure_git_repo(self) -> None:
        inside = self._run_git(["rev-parse", "--is-inside-work-tree"])
        if inside.lower() != "true":
            raise RuntimeError(f"'{self.repo_root}' is not a Git repository.")

    def _resolve_branch(self) -> str:
        if self.branch:
            return self.branch

        branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if branch == "HEAD":
            raise RuntimeError(
                "Cannot auto-update in detached HEAD state. "
                "Checkout a branch first (for example: git checkout main)."
            )
        return branch


class UpdateChecker(QObject):
    """
    Checks and applies updates using Git pull instead of release installers.
    """

    update_available = Signal(str, str, str)  # remote_sha, compare_url, summary
    up_to_date = Signal()
    error_occurred = Signal(str)
    update_finished = Signal(str)

    REPO_OWNER = "armrabbit"
    REPO_NAME = "translate"
    REPO_URL = "https://github.com/armrabbit/translate"

    def __init__(self):
        super().__init__()
        self.repo_owner, self.repo_name = self._resolve_repo_target()
        self.repo_web_url = f"https://github.com/{self.repo_owner}/{self.repo_name}"
        self.repo_root = self._resolve_repo_root()
        self.remote = (os.getenv("COMICTRANSLATE_UPDATE_REMOTE", "origin") or "origin").strip() or "origin"
        configured_branch = (os.getenv("COMICTRANSLATE_UPDATE_BRANCH", "") or "").strip()
        self.branch = configured_branch or None

        self._worker_thread = None
        self._worker = None

    def _resolve_repo_target(self) -> tuple[str, str]:
        repo_url = (os.getenv("COMICTRANSLATE_UPDATE_REPO_URL", self.REPO_URL) or self.REPO_URL).strip()
        parsed = self._parse_github_repo(repo_url)
        if parsed is not None:
            return parsed

        owner = (os.getenv("COMICTRANSLATE_UPDATE_REPO_OWNER", self.REPO_OWNER) or self.REPO_OWNER).strip() or self.REPO_OWNER
        name = (os.getenv("COMICTRANSLATE_UPDATE_REPO_NAME", self.REPO_NAME) or self.REPO_NAME).strip() or self.REPO_NAME
        return owner, name

    def _resolve_repo_root(self) -> str:
        configured = (os.getenv("COMICTRANSLATE_UPDATE_WORKDIR", "") or "").strip()
        if configured:
            return os.path.abspath(configured)
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    @staticmethod
    def _parse_github_repo(repo_url: str) -> tuple[str, str] | None:
        if not repo_url:
            return None

        cleaned = repo_url.strip()
        if not cleaned:
            return None

        cleaned = cleaned.split("#", 1)[0].rstrip("/")
        if cleaned.endswith(".git"):
            cleaned = cleaned[:-4]

        pattern = r"(?:github\.com[:/]+)?([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)$"
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            return None

        return match.group(1), match.group(2)

    def _safe_stop_thread(self):
        try:
            if self._worker_thread and self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait()
        except RuntimeError:
            pass
        except Exception as exc:
            logger.error("Error stopping update thread: %s", exc)

        self._worker_thread = None
        self._worker = None

    def check_for_updates(self):
        self._safe_stop_thread()

        worker = CheckWorker(
            repo_root=self.repo_root,
            remote=self.remote,
            branch=self.branch,
            repo_web_url=self.repo_web_url,
        )

        self._worker_thread = QThread()
        self._worker = worker
        worker.moveToThread(self._worker_thread)

        worker.finished.connect(self._worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        worker.update_available.connect(self.update_available)
        worker.up_to_date.connect(self.up_to_date)
        worker.error.connect(self.error_occurred)

        self._worker_thread.started.connect(worker.run)
        self._worker_thread.start()

    def apply_update(self):
        self._safe_stop_thread()

        worker = ApplyWorker(
            repo_root=self.repo_root,
            remote=self.remote,
            branch=self.branch,
            repo_web_url=self.repo_web_url,
        )

        self._worker_thread = QThread()
        self._worker = worker
        worker.moveToThread(self._worker_thread)

        worker.finished.connect(self._worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        worker.update_finished.connect(self.update_finished)
        worker.error.connect(self.error_occurred)

        self._worker_thread.started.connect(worker.run)
        self._worker_thread.start()

    def shutdown(self):
        self._safe_stop_thread()
        self._worker_thread = None
        self._worker = None


class CheckWorker(_GitWorkerBase):
    update_available = Signal(str, str, str)  # remote_sha, compare_url, summary
    up_to_date = Signal()

    def run(self):
        try:
            self._ensure_git_repo()
            branch = self._resolve_branch()
            remote_ref = f"{self.remote}/{branch}"

            self._run_git(["fetch", self.remote, branch, "--prune"], timeout=90)

            local_sha = self._run_git(["rev-parse", "HEAD"])
            remote_sha = self._run_git(["rev-parse", remote_ref])

            if local_sha == remote_sha:
                self.up_to_date.emit()
                return

            counts = self._run_git(["rev-list", "--left-right", "--count", f"HEAD...{remote_ref}"])
            ahead, behind = self._parse_counts(counts)
            latest_subject = self._run_git(["log", "-1", "--pretty=%s", remote_ref])
            compare_url = f"{self.repo_web_url}/compare/{local_sha}...{remote_sha}"
            summary = f"{branch}: behind {behind}, ahead {ahead}. Latest: {latest_subject}"
            self.update_available.emit(remote_sha, compare_url, summary)
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    @staticmethod
    def _parse_counts(raw_counts: str) -> tuple[int, int]:
        parts = raw_counts.replace("\t", " ").split()
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 0, 0


class ApplyWorker(_GitWorkerBase):
    update_finished = Signal(str)

    def run(self):
        try:
            self._ensure_git_repo()
            branch = self._resolve_branch()
            remote_ref = f"{self.remote}/{branch}"

            self._run_git(["fetch", self.remote, branch, "--prune"], timeout=90)

            tracked_changes = self._run_git(["status", "--porcelain", "--untracked-files=no"])
            if tracked_changes:
                raise RuntimeError(
                    "Local tracked changes detected. Please commit or stash them before updating."
                )

            local_sha = self._run_git(["rev-parse", "HEAD"])
            remote_sha = self._run_git(["rev-parse", remote_ref])
            if local_sha == remote_sha:
                self.update_finished.emit("Already up to date.")
                return

            pull_output = self._run_git(["pull", "--ff-only", self.remote, branch], timeout=180)

            details: list[str] = ["Updated from Git successfully."]
            if pull_output:
                details.append(pull_output.strip())

            requirements_path = os.path.join(self.repo_root, "requirements.txt")
            if os.path.isfile(requirements_path):
                details.append(self._install_requirements(requirements_path))

            self.update_finished.emit("\n".join(line for line in details if line))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()

    def _install_requirements(self, requirements_path: str) -> str:
        command = [sys.executable, "-m", "pip", "install", "-r", requirements_path]
        try:
            output = _run_command(command, cwd=self.repo_root, timeout=600)
            if output:
                return "Dependencies were updated."
            return "Dependencies are already satisfied."
        except Exception as exc:
            logger.warning("Dependency update failed after pull: %s", exc)
            return (
                "Code updated, but dependency install failed. "
                f"Please run manually: {sys.executable} -m pip install -r requirements.txt"
            )
