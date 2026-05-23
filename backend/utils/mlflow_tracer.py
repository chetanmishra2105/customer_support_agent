import json
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from backend.config import config
from backend.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import mlflow
except ImportError:  # pragma: no cover
    mlflow = None
    logger.warning("MLflow is not installed. Tracing will be disabled.")


class MLflowTracer:
    def __init__(self):
        self.tracking_uri = self._normalize_tracking_uri(config.MLFLOW_TRACKING_URI)
        self.experiment_name = config.MLFLOW_EXPERIMENT_NAME
        self.artifact_dir = Path(config.MLFLOW_ARTIFACT_DIR)
        self.ui_host = config.MLFLOW_UI_HOST
        self.ui_port = config.MLFLOW_UI_PORT
        self.ui_process: Optional[subprocess.Popen] = None
        self.enabled = mlflow is not None

        if self.enabled:
            try:
                self._setup()
            except Exception as exc:
                logger.exception("MLflow initialization failed, disabling tracing: %s", exc)
                self.enabled = False
        else:
            logger.warning("MLflowTracer initialized in disabled mode.")

    def _setup(self) -> None:
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logger.info("MLflow tracking initialized.")
        logger.info("MLflow UI available at %s", self.get_ui_url())

    def _normalize_tracking_uri(self, uri: str) -> str:
        parsed = urlparse(uri)
        if self._looks_like_windows_path(uri) or parsed.scheme == "":
            path = Path(uri)
            if not path.is_absolute():
                path = Path.cwd() / path
            if path.suffix.lower() in {".db", ".sqlite"}:
                return f"sqlite:///{path.resolve().as_posix()}"
            return path.resolve().as_uri()
        return uri

    @staticmethod
    def _looks_like_windows_path(uri: str) -> bool:
        return len(uri) >= 2 and uri[1] == ":" and uri[0].isalpha()

    def start_span(self, name: str, attributes: Dict[str, Any] | None = None):
        if not self.enabled:
            return nullcontext()
        return mlflow.start_span(name=name, attributes=attributes)

    def set_trace_tag(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        mlflow.set_tag(key, value)

    def get_ui_url(self) -> str:
        return f"http://{self.ui_host}:{self.ui_port}"

    def start_ui(self) -> None:
        if not self.enabled:
            logger.warning("Cannot start MLflow UI because MLflow is not installed.")
            return

        if self.ui_process is not None and self.ui_process.poll() is None:
            logger.info("MLflow UI already running at %s", self.get_ui_url())
            return

        try:
            cmd = [
                sys.executable,
                "-m",
                "mlflow",
                "ui",
                "--backend-store-uri",
                str(self.tracking_uri),
                "--default-artifact-root",
                str(self.artifact_dir),
                "--host",
                self.ui_host,
                "--port",
                str(self.ui_port),
            ]
            self.ui_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
            )
            time.sleep(1)
            if self.ui_process.poll() is None:
                logger.info("Started MLflow UI at %s", self.get_ui_url())
            else:
                logger.warning("MLflow UI process exited immediately.")
        except Exception as exc:  # pragma: no cover
            logger.exception("Failed to launch MLflow UI: %s", exc)
            self.ui_process = None

    def stop_ui(self) -> None:
        if self.ui_process is None:
            return
        try:
            self.ui_process.terminate()
            self.ui_process.wait(timeout=5)
            logger.info("Stopped MLflow UI process.")
        except Exception:
            logger.exception("Error stopping MLflow UI process.")
        finally:
            self.ui_process = None

    @contextmanager
    def start_run(self, run_name: str, nested: bool = True, **params: Any):
        if not self.enabled:
            yield None
            return

        run = mlflow.start_run(run_name=run_name, nested=nested)
        try:
            for key, value in params.items():
                if value is None:
                    continue
                mlflow.log_param(key, str(value))
            yield run
        finally:
            mlflow.end_run()

    def log_metric(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        mlflow.log_metric(key, value)

    def log_tag(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        mlflow.set_tag(key, value)

    def log_json_artifact(self, payload: Dict[str, Any], filename: str) -> None:
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(path))

    def log_text_artifact(self, text: str, filename: str) -> None:
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / filename
            path.write_text(text, encoding="utf-8")
            mlflow.log_artifact(str(path))
