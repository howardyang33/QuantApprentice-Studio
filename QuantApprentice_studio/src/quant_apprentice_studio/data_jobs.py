from __future__ import annotations

import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping

from .contracts import build_run_contract, dataset_job_path, dataset_manifest_path, ensure_contract_dirs
from .data_adapters.kline_downloader import KlineDownloader, MAX_ROUNDS, parse_stock_codes
from .guided_entry import build_online_kline_dataset_manifest
from .provenance import read_json, write_json


logger = logging.getLogger("quant_apprentice_studio.data_jobs")


class KlineDownloadJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def _now(self) -> str:
        return datetime.now().isoformat(timespec="seconds")

    def _persist(self, contract: Mapping[str, Any], payload: Dict[str, Any]) -> None:
        write_json(dataset_job_path(dict(contract), str(payload["job_id"])), payload)

    def _append_log(self, payload: Dict[str, Any], message: str) -> None:
        logs = list(payload.get("logs") or [])
        logs.append(f"[{self._now()}] {message}")
        payload["logs"] = logs[-400:]

    def get_job(self, *, profile_id: str, project_id: str, dataset_id: str, run_id: str, job_id: str) -> Dict[str, Any]:
        contract = build_run_contract(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            allow_imported_fallback=True,
            allow_demo_fallback=False,
        )
        path = dataset_job_path(contract, job_id)
        if not path.exists():
            raise FileNotFoundError(f"K-line download job not found: {path}")
        payload = read_json(path)
        out = payload if isinstance(payload, dict) else {}
        manifest_json = str(out.get("manifest_json", "")).strip()
        if manifest_json:
            try:
                out["manifest_preview"] = read_json(Path(manifest_json))
            except Exception:
                out["manifest_preview"] = {}
        return out

    def start_job(
        self,
        *,
        profile_id: str,
        project_id: str,
        dataset_id: str,
        run_id: str,
        task_type: str,
        allow_imported_fallback: bool,
        allow_demo_fallback: bool,
        stock_codes: str | List[str],
        earliest_date: str,
        adjust_type: str,
        full_refresh: bool,
        update_indexes: bool,
    ) -> Dict[str, Any]:
        contract = build_run_contract(
            profile_id=profile_id,
            project_id=project_id,
            dataset_id=dataset_id,
            run_id=run_id,
            allow_imported_fallback=allow_imported_fallback,
            allow_demo_fallback=allow_demo_fallback,
        )
        ensure_contract_dirs(contract)
        job_id = f"kline-job-{uuid.uuid4().hex[:12]}"
        codes = parse_stock_codes(stock_codes)
        if not codes:
            raise ValueError("Provide at least one A-share stock code for online K-line download.")

        payload: Dict[str, Any] = {
            "job_id": job_id,
            "profile_id": profile_id,
            "project_id": contract["project_id"],
            "dataset_id": contract["dataset_id"],
            "run_id": contract["run_id"],
            "task_type": task_type,
            "source_type": "online_kline_downloader",
            "status": "queued",
            "progress": 0.0,
            "success_count": 0,
            "failed_count": 0,
            "failed_codes": [],
            "logs": [],
            "output_paths": {
                "dataset_root": contract["dataset_root"],
                "stock_kline_root": contract["dataset_stock_klines_root"],
                "index_kline_root": contract["dataset_index_klines_root"],
                "listed_dates_cache": str(Path(contract["dataset_cache_root"]) / "listed_dates_cache.csv"),
                "dataset_manifest_json": str(dataset_manifest_path(contract)),
            },
            "config": {
                "stock_codes": codes,
                "earliest_date": earliest_date,
                "adjust_type": adjust_type,
                "full_refresh": bool(full_refresh),
                "update_indexes": bool(update_indexes),
            },
            "manifest_json": "",
            "started_at": "",
            "finished_at": "",
        }
        self._append_log(payload, f"Queued online K-line download for {len(codes)} stock codes.")
        self._persist(contract, payload)
        with self._lock:
            self._jobs[job_id] = dict(payload)

        thread = threading.Thread(
            target=self._run_job,
            kwargs={
                "contract": contract,
                "job_id": job_id,
                "task_type": task_type,
                "codes": codes,
                "earliest_date": earliest_date,
                "adjust_type": adjust_type,
                "full_refresh": full_refresh,
                "update_indexes": update_indexes,
            },
            daemon=True,
            name=f"kline_download_{job_id}",
        )
        thread.start()
        return payload

    def _update_job(self, contract: Mapping[str, Any], job_id: str, mutate: Any) -> Dict[str, Any]:
        path = dataset_job_path(dict(contract), job_id)
        current = read_json(path) if path.exists() else {}
        payload = dict(current if isinstance(current, dict) else {})
        mutate(payload)
        self._persist(contract, payload)
        with self._lock:
            self._jobs[job_id] = dict(payload)
        return payload

    def _run_job(
        self,
        *,
        contract: Mapping[str, Any],
        job_id: str,
        task_type: str,
        codes: List[str],
        earliest_date: str,
        adjust_type: str,
        full_refresh: bool,
        update_indexes: bool,
    ) -> None:
        def mark_running(payload: Dict[str, Any]) -> None:
            payload["status"] = "running"
            payload["started_at"] = self._now()
            self._append_log(payload, "Job started.")

        self._update_job(contract, job_id, mark_running)

        listed_dates_cache = str(Path(contract["dataset_cache_root"]) / "listed_dates_cache.csv")
        downloader = KlineDownloader(
            cache_dir=contract["dataset_stock_klines_root"],
            index_cache_dir=contract["dataset_index_klines_root"],
            listed_dates_cache_path=listed_dates_cache,
            earliest_date=earliest_date,
            full_refresh=bool(full_refresh),
            adjust_type=adjust_type,
        )

        total_work_units = len(codes) + (8 if update_indexes else 0)
        progress_state = {"processed": 0, "success": 0, "failed_codes": []}

        def on_progress(phase: str, info: Dict[str, Any]) -> None:
            code = str(info.get("code", "")).strip()
            ok = bool(info.get("ok", False))
            if code and info.get("status") == "item_processed":
                progress_state["processed"] += 1 if (ok or int(info.get("round", 0) or 0) >= MAX_ROUNDS) else 0
                if ok:
                    progress_state["success"] += 1
                elif code not in progress_state["failed_codes"] and int(info.get("round", 0) or 0) >= MAX_ROUNDS:
                    progress_state["failed_codes"].append(code)

            def mutate(payload: Dict[str, Any]) -> None:
                payload["progress"] = round(min(progress_state["processed"] / max(total_work_units, 1), 1.0), 4)
                payload["success_count"] = int(progress_state["success"])
                payload["failed_codes"] = list(progress_state["failed_codes"])
                payload["failed_count"] = len(payload["failed_codes"])
                if code:
                    label = "ok" if ok else "retrying_or_failed"
                    self._append_log(payload, f"{phase}: {code} -> {label}")

            self._update_job(contract, job_id, mutate)

        try:
            index_result = {"total": 0, "success": 0, "failed": 0, "failed_codes": []}
            if update_indexes:
                self._update_job(contract, job_id, lambda payload: self._append_log(payload, "Downloading benchmark index K-lines."))
                index_result = downloader.update_indexes(progress_callback=on_progress)
            else:
                self._update_job(contract, job_id, lambda payload: self._append_log(payload, "Index download skipped by user choice."))

            self._update_job(contract, job_id, lambda payload: self._append_log(payload, f"Downloading stock K-lines for {len(codes)} symbols."))
            stock_result = downloader.update_all_kline_cache(codes, progress_callback=on_progress)
            final_failed_codes = list(index_result.get("failed_codes") or []) + list(stock_result.get("failed_codes") or [])
            manifest = build_online_kline_dataset_manifest(
                profile_id=str(contract["profile_id"]),
                project_id=str(contract["project_id"]),
                dataset_id=str(contract["dataset_id"]),
                run_id=str(contract["run_id"]),
                task_type=task_type,
                adjust_type=adjust_type,
                earliest_date=earliest_date,
                update_indexes=update_indexes,
                stock_kline_root=str(contract["dataset_stock_klines_root"]),
                index_kline_root=str(contract["dataset_index_klines_root"]),
                failed_codes=final_failed_codes,
                allow_imported_fallback=bool(contract["allow_imported_fallback"]),
                allow_demo_fallback=bool(contract["allow_demo_fallback"]),
            )

            def mark_completed(payload: Dict[str, Any]) -> None:
                success_count = int(index_result.get("success", 0) or 0) + int(stock_result.get("success", 0) or 0)
                payload["status"] = "completed" if success_count > 0 else "failed"
                payload["progress"] = 1.0
                payload["success_count"] = success_count
                payload["failed_codes"] = final_failed_codes
                payload["failed_count"] = len(final_failed_codes)
                payload["manifest_json"] = str(dataset_manifest_path(dict(contract)))
                payload["finished_at"] = self._now()
                log_prefix = "Job completed" if success_count > 0 else "Job failed"
                self._append_log(payload, f"{log_prefix}. Manifest written to {payload['manifest_json']}.")
                payload["summary"] = {
                    "stocks": stock_result,
                    "indexes": index_result,
                    "manifest_data_readiness": manifest.get("data_readiness", ""),
                    "full_pipeline_ready": bool(manifest.get("full_pipeline_ready", False)),
                }

            self._update_job(contract, job_id, mark_completed)
        except Exception as exc:
            logger.exception("K-line download job %s failed", job_id)

            def mark_failed(payload: Dict[str, Any]) -> None:
                payload["status"] = "failed"
                payload["finished_at"] = self._now()
                self._append_log(payload, f"Job failed: {exc}")

            self._update_job(contract, job_id, mark_failed)
