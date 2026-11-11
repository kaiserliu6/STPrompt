import csv
import os
import time
from contextlib import contextmanager
from datetime import datetime


class RunTimeRecorder:
    HEADER = [
        "Dataset",
        "Model",
        "Save Name",
        "Backbone",
        "Run Index",
        "Total Runs",
        "Start Time",
        "End Time",
        "Elapsed Seconds",
        "Elapsed Minutes",
        "Status",
        "Error",
    ]

    def __init__(self, enabled=True, csv_path="result/run_time.csv"):
        self.enabled = bool(enabled)
        self.csv_path = csv_path
        if self.enabled:
            self._ensure_csv()

    @contextmanager
    def track(self, model_name, dataset_name, run_index=None, total_runs=None, save_name=None):
        record = _RunTimeRecord(
            enabled=self.enabled,
            model_name=model_name,
            dataset_name=dataset_name,
            run_index=run_index,
            total_runs=total_runs,
            save_name=save_name,
        )

        if not self.enabled:
            yield record
            return

        start = time.perf_counter()
        record.start_time = datetime.now()
        try:
            yield record
            record.status = "success"
        except Exception as e:
            record.status = "failed"
            record.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            end = time.perf_counter()
            record.end_time = datetime.now()
            record.elapsed_seconds = end - start
            self._write(record)

    def _ensure_csv(self):
        csv_dir = os.path.dirname(self.csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        if not os.path.exists(self.csv_path) or os.path.getsize(self.csv_path) == 0:
            with open(self.csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.HEADER)

    def _write(self, record):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(record.to_row())


class _RunTimeRecord:
    def __init__(self, enabled, model_name, dataset_name, run_index, total_runs, save_name):
        self.enabled = enabled
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.run_index = run_index
        self.total_runs = total_runs
        self.save_name = save_name or model_name
        self.backbone = ""
        self.start_time = None
        self.end_time = None
        self.elapsed_seconds = 0
        self.status = "success"
        self.error = ""

    def set_backbone(self, backbone):
        if self.enabled:
            self.backbone = backbone or ""

    def to_row(self):
        elapsed_minutes = self.elapsed_seconds / 60
        return [
            self.dataset_name,
            self.model_name,
            self.save_name,
            self.backbone,
            self.run_index,
            self.total_runs,
            self._format_time(self.start_time),
            self._format_time(self.end_time),
            round(self.elapsed_seconds, 5),
            round(elapsed_minutes, 5),
            self.status,
            self.error,
        ]

    @staticmethod
    def _format_time(value):
        if value is None:
            return ""
        return value.strftime("%Y-%m-%d %H:%M:%S")
