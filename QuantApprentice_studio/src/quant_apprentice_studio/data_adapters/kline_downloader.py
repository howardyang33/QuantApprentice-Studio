from __future__ import annotations

import datetime as dt
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..secret_config import resolve_secret


logger = logging.getLogger("quant_apprentice_studio.kline_downloader")

KLINE_HISTORY_DAYS = 90
RETRY_INTERVAL_INITIAL = 2.0
RETRY_INTERVAL_MIN = 0.4
RETRY_INTERVAL_MAX = 60.0
RETRY_INTERVAL_BACKOFF_FACTOR = 2.0
RETRY_INTERVAL_SUCCESS_FACTOR = 0.8
MAX_ROUNDS = 4

STANDARD_KLINE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_chg",
    "high_limit",
    "low_limit",
]

INDEX_CODE_MAP = {
    "000016": "上证50",
    "000300": "沪深300",
    "000688": "科创50",
    "399006": "创业板指",
    "000905": "中证500",
    "000852": "中证1000",
    "932000": "中证2000",
    "931483": "微盘股指数",
}


ProgressCallback = Callable[[str, Dict[str, Any]], None]


def parse_stock_codes(raw: str | List[str]) -> List[str]:
    if isinstance(raw, list):
        items = [str(item or "").strip() for item in raw]
    else:
        text = str(raw or "").replace("\r", "\n")
        items = [part.strip() for block in text.split("\n") for part in block.split(",")]
    codes: List[str] = []
    seen = set()
    for item in items:
        digits = "".join(ch for ch in item if ch.isdigit())
        if not digits:
            continue
        code = digits.zfill(6)
        if code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def _clean_proxy_env() -> None:
    for key in list(os.environ.keys()):
        if "PROXY" in key.upper():
            os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


def _safe_call_no_proxy(func: Callable[..., Any], *args: Any, max_retries: int = 2, fast_fail: bool = True, **kwargs: Any) -> Any:
    def _call_once() -> Any:
        return func(*args, **kwargs)

    def _call_bypass() -> Any:
        proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
        saved = {key: os.environ.pop(key, None) for key in proxy_keys}
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        try:
            return func(*args, **kwargs)
        finally:
            for key, value in saved.items():
                if value is not None:
                    os.environ[key] = value
            os.environ.pop("NO_PROXY", None)
            os.environ.pop("no_proxy", None)

    last_err: Exception | None = None
    bypass_tried = False
    for attempt in range(1, max_retries + 1):
        try:
            return _call_once()
        except Exception as exc:  # pragma: no cover - network/library dependent
            last_err = exc
            err_lower = str(exc).lower()
            is_proxy = any(
                kw in err_lower
                for kw in (
                    "proxyerror",
                    "proxy",
                    "handshake",
                    "cannot connect to proxy",
                    "remotedisconnected",
                    "remote end closed connection",
                    "connection aborted",
                )
            )
            if is_proxy and not bypass_tried:
                bypass_tried = True
                logger.debug("Proxy-related fetch failure detected, retrying direct connection: %s", exc)
                try:
                    return _call_bypass()
                except Exception as exc2:  # pragma: no cover - network/library dependent
                    last_err = exc2
            if fast_fail:
                return None
            time.sleep(random.uniform(5, 15) * attempt)
    if last_err:
        logger.debug("Downloader call exhausted retries: %s", last_err)
    return None


def _resolve_tushare_token() -> str:
    return resolve_secret("TUSHARE_TOKEN")


def _load_listed_dates(cache_path: Path) -> Dict[str, dt.date]:
    if cache_path.exists():
        try:
            mtime = dt.datetime.fromtimestamp(cache_path.stat().st_mtime).date()
            age = (dt.date.today() - mtime).days
            df_cache = pd.read_csv(cache_path, dtype={"code": str})
            df_cache["listed_date"] = pd.to_datetime(df_cache["listed_date"]).dt.date
            cached = dict(zip(df_cache["code"].str.zfill(6), df_cache["listed_date"]))
            logger.info("Loaded listed-date cache from %s (%s symbols, age=%s days)", cache_path, len(cached), age)
            if age <= 7:
                return cached
        except Exception as exc:
            logger.debug("Failed to read listed-date cache %s: %s", cache_path, exc)

    token = _resolve_tushare_token()
    if not token:
        logger.warning("TUSHARE_TOKEN is not configured. Continuing without fresh listed-date metadata.")
        return {}

    try:
        import tushare as ts

        _clean_proxy_env()
        ts.set_token(token)
        pro = ts.pro_api()
        df = pro.stock_basic(exchange="", list_status="L", fields="code,list_date")
        if df is None or df.empty or "list_date" not in df.columns:
            return {}
        df["code"] = df["code"].astype(str).str.zfill(6)
        df["listed_date"] = pd.to_datetime(df["list_date"], format="%Y%m%d", errors="coerce").dt.date
        result = {
            row["code"]: row["listed_date"]
            for _, row in df.iterrows()
            if pd.notna(row["listed_date"])
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df[["code", "listed_date"]].to_csv(cache_path, index=False)
        logger.info("Refreshed listed-date cache into %s (%s symbols)", cache_path, len(result))
        return result
    except Exception as exc:  # pragma: no cover - network/library dependent
        logger.warning("Failed to refresh listed-date cache via tushare: %s", exc)
        if cache_path.exists():
            try:
                df_cache = pd.read_csv(cache_path, dtype={"code": str})
                df_cache["listed_date"] = pd.to_datetime(df_cache["listed_date"]).dt.date
                return dict(zip(df_cache["code"].str.zfill(6), df_cache["listed_date"]))
            except Exception:
                return {}
        return {}


def _fetch_kline_sina(code: str, start_str: str, end_str: str, *, is_index: bool = False, adjust_type: str = "qfq") -> Optional[pd.DataFrame]:
    try:
        import akshare as ak

        if is_index:
            if code.startswith(("000", "880", "93")):
                prefix = "sh"
            elif code.startswith(("399", "899")):
                prefix = "sz"
            else:
                prefix = "sh"
            symbol = f"{prefix}{code}"
            df = _safe_call_no_proxy(
                ak.stock_zh_index_daily,
                symbol=symbol,
                max_retries=1,
                fast_fail=True,
            )
        else:
            prefix = "sh" if code.startswith(("60", "68", "900")) else "sz"
            symbol = f"{prefix}{code}"
            df = _safe_call_no_proxy(
                ak.stock_zh_a_daily,
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                adjust=adjust_type,
                max_retries=1,
                fast_fail=True,
            )
        if df is None or df.empty:
            return None
        if "outstanding_share" in df.columns:
            df = df.drop(columns=["outstanding_share"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)
    except Exception as exc:  # pragma: no cover - network/library dependent
        logger.warning("Failed to fetch %s %s: %s", "index" if is_index else "stock", code, exc)
        return None


class KlineDownloader:
    def __init__(
        self,
        *,
        cache_dir: str,
        index_cache_dir: str,
        listed_dates_cache_path: str,
        history_days: int = KLINE_HISTORY_DAYS,
        earliest_date: Optional[str] = None,
        listed_dates: Optional[Dict[str, dt.date]] = None,
        force: bool = False,
        full_refresh: bool = False,
        adjust_type: str = "qfq",
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.index_cache_dir = Path(index_cache_dir).expanduser().resolve()
        self.listed_dates_cache_path = Path(listed_dates_cache_path).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_cache_dir.mkdir(parents=True, exist_ok=True)
        self.listed_dates_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_days = int(history_days)
        self._force = bool(force)
        self._full_refresh = bool(full_refresh)
        self._adjust_type = str(adjust_type).strip() or "qfq"
        self._listed_dates = dict(listed_dates or _load_listed_dates(self.listed_dates_cache_path))
        self._earliest_date: Optional[dt.date] = None
        if earliest_date:
            text = str(earliest_date).replace("-", "")
            if len(text) != 8 or not text.isdigit():
                raise ValueError(f"Invalid earliest_date: {earliest_date}")
            self._earliest_date = dt.date(int(text[:4]), int(text[4:6]), int(text[6:]))

    def _get_cache_path(self, code: str, is_index: bool = False) -> Path:
        return (self.index_cache_dir if is_index else self.cache_dir) / f"{code.zfill(6)}.csv"

    def _ensure_kline_columns(self, df: pd.DataFrame, code: str, is_index: bool = False) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.copy()
        if "amount" not in out.columns:
            out["amount"] = np.nan
        if "turnover" not in out.columns:
            out["turnover"] = np.nan
        if "pct_chg" not in out.columns and "close" in out.columns:
            out["pct_chg"] = out["close"].pct_change() * 100
        if not is_index:
            if "high_limit" not in out.columns or "low_limit" not in out.columns:
                out = self._estimate_limit_prices(out, code)
        else:
            if "high_limit" not in out.columns:
                out["high_limit"] = np.nan
            if "low_limit" not in out.columns:
                out["low_limit"] = np.nan
        return out

    def _estimate_limit_prices(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        if df.empty:
            return df
        out = df.reset_index(drop=True).copy()
        is_gem = code.startswith("30")
        is_star = code.startswith("68")
        limit_pct = 0.20 if (is_gem or is_star) else 0.10
        pre_close = out["close"].shift(1)
        out["high_limit"] = (pre_close * (1 + limit_pct)).round(2)
        out["low_limit"] = (pre_close * (1 - limit_pct)).round(2)
        out.loc[0, "high_limit"] = round(out.loc[0, "close"] * (1 + limit_pct), 2)
        out.loc[0, "low_limit"] = round(out.loc[0, "close"] * (1 - limit_pct), 2)
        return out

    def _decide_fetch_range(self, code: str, is_index: bool = False) -> Tuple[Optional[dt.date], Optional[dt.date], bool, str]:
        today = dt.date.today()
        now = dt.datetime.now()
        is_trading_hours_over = now.hour >= 16
        cache_valid_cutoff = today if is_trading_hours_over else today - dt.timedelta(days=1)
        fetch_end_date = today if is_trading_hours_over else today - dt.timedelta(days=1)

        cache_path = self._get_cache_path(code, is_index)
        cache_first_date: Optional[dt.date] = None
        cache_last_date: Optional[dt.date] = None
        if cache_path.exists():
            try:
                old_df = pd.read_csv(cache_path, parse_dates=["date"])
                old_df = self._ensure_kline_columns(old_df, code, is_index)
                if not old_df.empty:
                    cache_first_date = old_df["date"].min().date()
                    cache_last_date = old_df["date"].max().date()
            except Exception:
                pass

        if self._full_refresh:
            target_start = self._earliest_date if self._earliest_date else today - dt.timedelta(days=self.history_days)
        else:
            if cache_last_date is not None:
                target_start = cache_last_date + dt.timedelta(days=1)
            else:
                target_start = self._earliest_date if self._earliest_date else today - dt.timedelta(days=self.history_days)

        if cache_last_date is None:
            return target_start, fetch_end_date, True, "no_cache"

        latest_ok = cache_last_date >= cache_valid_cutoff
        if self._full_refresh:
            if not is_index:
                stock_start = self._listed_dates.get(code)
                if self._earliest_date:
                    if stock_start is not None and stock_start > self._earliest_date:
                        hist_cutoff = stock_start - dt.timedelta(days=15)
                    else:
                        hist_cutoff = self._earliest_date - dt.timedelta(days=7)
                else:
                    hist_cutoff = stock_start or (today - dt.timedelta(days=self.history_days))
            else:
                hist_cutoff = self._earliest_date or (today - dt.timedelta(days=self.history_days))
            has_complete_history = cache_first_date is not None and cache_first_date <= hist_cutoff
            if latest_ok and has_complete_history:
                return None, None, False, "cache_complete"
        else:
            if latest_ok:
                return None, None, False, "cache_complete"

        fetch_start = cache_last_date + dt.timedelta(days=1)
        return fetch_start, fetch_end_date, False, "incremental"

    def _fetch_and_save(self, code: str, is_index: bool = False) -> bool:
        fetch_start, fetch_end, full_pull, _reason = self._decide_fetch_range(code, is_index)
        if fetch_start is None:
            return True

        end_d = fetch_end or (dt.date.today() - dt.timedelta(days=1))
        start_str = fetch_start.strftime("%Y%m%d")
        end_str = end_d.strftime("%Y%m%d")

        old_df: Optional[pd.DataFrame] = None
        cache_path = self._get_cache_path(code, is_index)
        if self._force and cache_path.exists():
            cache_path.unlink()
        if not full_pull and cache_path.exists():
            try:
                old_df = pd.read_csv(cache_path, parse_dates=["date"])
                old_df = self._ensure_kline_columns(old_df, code, is_index)
            except Exception:
                old_df = None

        new_df = _fetch_kline_sina(code, start_str, end_str, is_index=is_index, adjust_type=self._adjust_type)
        if new_df is None or new_df.empty:
            return False

        new_df["date"] = pd.to_datetime(new_df["date"])
        if not is_index:
            if "high_limit" not in new_df.columns or "low_limit" not in new_df.columns:
                new_df = self._estimate_limit_prices(new_df, code)
        else:
            if "high_limit" not in new_df.columns:
                new_df["high_limit"] = np.nan
            if "low_limit" not in new_df.columns:
                new_df["low_limit"] = np.nan

        for col in STANDARD_KLINE_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = np.nan
        new_df = new_df[STANDARD_KLINE_COLUMNS].sort_values("date").reset_index(drop=True)

        if old_df is not None and not old_df.empty:
            for col in STANDARD_KLINE_COLUMNS:
                if col not in old_df.columns:
                    old_df[col] = np.nan
            merged = pd.concat([old_df[STANDARD_KLINE_COLUMNS], new_df], ignore_index=True)
        else:
            merged = new_df.copy()

        merged = merged.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
        merged["pct_chg"] = merged["close"].pct_change() * 100
        cutoff = self._earliest_date if self._earliest_date else dt.date.today() - dt.timedelta(days=self.history_days)
        merged = merged[merged["date"] >= pd.Timestamp(cutoff)].reset_index(drop=True)
        merged.to_csv(cache_path, index=False)
        return True

    def _run_code_loop(
        self,
        *,
        codes: List[str],
        phase: str,
        is_index: bool,
        progress_callback: Optional[ProgressCallback],
    ) -> Dict[str, Any]:
        total = len(codes)
        if total == 0:
            return {"total": 0, "success": 0, "failed": 0, "failed_codes": [], "interrupted": False}

        pending: List[str] = []
        skipped = 0
        if self._force:
            pending = list(codes)
        else:
            for code in codes:
                fetch_start, _, _, _reason = self._decide_fetch_range(code, is_index=is_index)
                if fetch_start is None:
                    skipped += 1
                else:
                    pending.append(code)

        processed_overall = skipped
        if progress_callback:
            progress_callback(
                phase,
                {
                    "phase": phase,
                    "total": total,
                    "pending": len(pending),
                    "processed": processed_overall,
                    "success": skipped,
                    "failed": 0,
                    "failed_codes": [],
                    "code": "",
                    "status": "prefiltered",
                },
            )

        if not pending:
            return {"total": total, "success": skipped, "failed": 0, "failed_codes": [], "interrupted": False}

        to_fetch = list(pending)
        retry_interval = float(RETRY_INTERVAL_INITIAL)
        for round_num in range(1, MAX_ROUNDS + 1):
            if not to_fetch:
                break
            next_round_fail: List[str] = []
            for code in to_fetch:
                ok = self._fetch_and_save(code, is_index=is_index)
                if ok:
                    retry_interval = max(RETRY_INTERVAL_MIN, retry_interval * RETRY_INTERVAL_SUCCESS_FACTOR)
                else:
                    retry_interval = min(RETRY_INTERVAL_MAX, retry_interval * RETRY_INTERVAL_BACKOFF_FACTOR)
                    next_round_fail.append(code)
                processed_overall += 1 if ok or round_num == MAX_ROUNDS else 0
                if progress_callback:
                    progress_callback(
                        phase,
                        {
                            "phase": phase,
                            "round": round_num,
                            "total": total,
                            "pending": len(to_fetch),
                            "processed": processed_overall,
                            "success": total - skipped - len(next_round_fail) - max(len(to_fetch) - (to_fetch.index(code) + 1), 0),
                            "failed": len(next_round_fail),
                            "failed_codes": list(next_round_fail),
                            "code": code,
                            "ok": ok,
                            "status": "item_processed",
                        },
                    )
                time.sleep(retry_interval if not ok else max(0.5, retry_interval * 0.5))
            if not next_round_fail:
                to_fetch = []
                break
            to_fetch = list(next_round_fail)
            if round_num < MAX_ROUNDS:
                wait = min(60, max(5, len(to_fetch) // 10))
                time.sleep(wait)

        final_fail = list(to_fetch)
        success_total = total - skipped - len(final_fail)
        return {
            "total": total,
            "success": success_total,
            "failed": len(final_fail),
            "failed_codes": final_fail,
            "interrupted": False,
        }

    def update_indexes(self, progress_callback: Optional[ProgressCallback] = None) -> Dict[str, Any]:
        return self._run_code_loop(
            codes=list(INDEX_CODE_MAP.keys()),
            phase="indexes",
            is_index=True,
            progress_callback=progress_callback,
        )

    def update_all_kline_cache(self, codes: List[str], progress_callback: Optional[ProgressCallback] = None) -> Dict[str, Any]:
        normalized_codes = [str(code).zfill(6) for code in codes if str(code).strip()]
        return self._run_code_loop(
            codes=normalized_codes,
            phase="stocks",
            is_index=False,
            progress_callback=progress_callback,
        )
