import os
import platform
import logging
import re
import requests
import subprocess
import tempfile
from packaging import version
from PySide6.QtCore import QObject, Signal, QThread, QStandardPaths
from app.version import __version__

logger = logging.getLogger(__name__)

class UpdateChecker(QObject):
    """
    Checks for updates on GitHub and handles downloading/running installers.
    """
    update_available = Signal(str, str, str)  # version, release_notes, download_url
    up_to_date = Signal()
    error_occurred = Signal(str)
    download_progress = Signal(int)
    download_finished = Signal(str) # file_path

    REPO_OWNER = "armrabbit"
    REPO_NAME = "translate"
    REPO_URL = "https://github.com/armrabbit/translate#"

    def __init__(self):
        super().__init__()
        self.repo_owner, self.repo_name = self._resolve_repo_target()
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

        # Supports:
        # - https://github.com/owner/repo
        # - https://github.com/owner/repo#
        # - git@github.com:owner/repo
        # - owner/repo
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
            # The C++ object has been deleted
            pass
        except Exception as e:
            logger.error(f"Error stopping thread: {e}")
        self._worker_thread = None
        self._worker = None

    def check_for_updates(self):
        """Starts the check in a background thread."""
        self._safe_stop_thread()
            
        self._worker_thread = QThread()
        self._worker = UpdateWorker(self.repo_owner, self.repo_name, __version__)
        self._worker.moveToThread(self._worker_thread)
        
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        
        self._worker.update_available.connect(self.update_available)
        self._worker.up_to_date.connect(self.up_to_date)
        self._worker.error.connect(self.error_occurred)
        
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def download_installer(self, url, filename):
        """Starts the download in a background thread."""
        self._safe_stop_thread()

        self._worker_thread = QThread()
        self._worker = DownloadWorker(url, filename)
        self._worker.moveToThread(self._worker_thread)
        
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        
        self._worker.progress.connect(self.download_progress)
        self._worker.finished_path.connect(self.download_finished)
        self._worker.error.connect(self.error_occurred)
        
        self._worker_thread.started.connect(self._worker.run)
        self._worker_thread.start()

    def run_installer(self, file_path):
        """Executes the installer based on the platform."""
        try:
            system = platform.system()
            if system == "Windows":
                # Use os.startfile; Windows will parse the installer manifest
                # and trigger UAC only if the installer requires it.
                os.startfile(file_path)
            elif system == "Darwin": # macOS
                subprocess.Popen(["open", file_path])
            else:
                self.error_occurred.emit(f"Unsupported platform for installer launch: {system}")
        except Exception as e:
            self.error_occurred.emit(f"Failed to launch installer: {e}")

    def shutdown(self):
        """Stops any active worker thread (best-effort)."""
        self._safe_stop_thread()
        self._worker_thread = None
        self._worker = None


class UpdateWorker(QObject):
    update_available = Signal(str, str, str)
    up_to_date = Signal()
    error = Signal(str)
    finished = Signal()

    def __init__(self, owner, repo, current_version):
        super().__init__()
        self.owner = owner
        self.repo = repo
        self.current_version = current_version

    def run(self):
        try:
            url = f"https://api.github.com/repos/{self.owner}/{self.repo}/releases/latest"
            response = requests.get(
                url,
                timeout=10,
                headers={"Accept": "application/vnd.github+json"},
            )
            if response.status_code == 404:
                # Repo may not publish GitHub Releases (only commits/tags).
                # Treat as "no update metadata" instead of surfacing an error popup/log.
                logger.info(
                    "No GitHub release metadata for %s/%s; skipping release-based update check.",
                    self.owner,
                    self.repo,
                )
                self.up_to_date.emit()
                return
            response.raise_for_status()
            data = response.json()
            
            latest_tag = data.get("tag_name", "").lstrip("v")
            if not latest_tag:
                 self.error.emit("Could not parse version from release.")
                 return

            if version.parse(latest_tag) > version.parse(self.current_version):
                # Find appropriate asset
                asset_url = None
                system = platform.system()
                if system == "Windows":
                    for asset in data.get("assets", []):
                        asset_name = asset.get("name", "")
                        if asset_name.endswith(".exe") or asset_name.endswith(".msi"):
                            asset_url = asset.get("browser_download_url")
                            break
                elif system == "Darwin":
                    for asset in data.get("assets", []):
                        asset_name = asset.get("name", "")
                        if asset_name.endswith(".dmg") or asset_name.endswith(".pkg"):
                            asset_url = asset.get("browser_download_url")
                            break
                
                if asset_url:
                    self.update_available.emit(latest_tag, data.get("html_url", ""), asset_url)
                else:
                    self.error.emit(f"New version {latest_tag} available, but no installer found for your OS.")
            else:
                self.up_to_date.emit()

        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()


class DownloadWorker(QObject):
    progress = Signal(int)
    finished_path = Signal(str)
    error = Signal(str)
    finished = Signal()

    def __init__(self, url, filename):
        super().__init__()
        self.url = url
        self.filename = filename

    def run(self):
        try:
            # Download to Downloads directory
            download_dir = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
            if not download_dir:
                download_dir = os.path.join(os.path.expanduser("~"), "Downloads")
            
            # Fallback to temp if Downloads doesn't exist
            if not os.path.exists(download_dir):
                download_dir = tempfile.gettempdir()

            save_path = os.path.join(download_dir, self.filename)
            
            response = requests.get(self.url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            percent = int((downloaded_size / total_size) * 100)
                            self.progress.emit(percent)
            
            self.finished_path.emit(save_path)
            
        except Exception as e:
            self.error.emit(str(e))
        finally:
            self.finished.emit()

