from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from ..live_runtime import describe_live_model_config, load_live_model_config, score_live_signal
from ..paths import studio_root
from ..provenance import read_json, to_jsonable, write_json
from ..schemas import SignalScore
from .base import BaseAgent


class SignalScoringAgent(BaseAgent):
    def __init__(self, registry) -> None:
        super().__init__(registry)
        self._score_cache: Dict[str, Dict[Tuple[str, str], Dict]] = {}
        self._signal_schema_cache: Dict[str, Dict[str, Any]] = {}

    def _load_market_scores(self, alias: str) -> Dict[Tuple[str, str], Dict]:
        if alias in self._score_cache:
            return self._score_cache[alias]
        catalog = self.registry.load_runtime_catalog()
        payload = read_json(Path(catalog["market_runs"][alias]["llm_signal_scores_json"]))
        results = list(payload.get("results") or [])
        lookup: Dict[Tuple[str, str], Dict] = {}
        for row in results:
            signal_date = str(row.get("signal_date", "")).strip()
            symbol = str(row.get("symbol", "")).strip()
            if signal_date and symbol:
                lookup[(signal_date, symbol)] = row
        self._score_cache[alias] = lookup
        return lookup

    def _default_market_run_alias(self) -> str:
        catalog = self.registry.load_runtime_catalog()
        return str(catalog.get("defaults", {}).get("market_run_alias", "")).strip()

    def _resolve_signal_key(
        self,
        market_run_alias: str,
        *,
        signal_date: str = "",
        symbol: str = "",
    ) -> Tuple[str, str]:
        lookup = self._load_market_scores(market_run_alias)
        if signal_date and symbol:
            key = (str(signal_date).strip(), str(symbol).strip())
            if key not in lookup:
                raise KeyError(f"signal not found for {market_run_alias}: {signal_date} {symbol}")
            return key
        if signal_date:
            keys = [key for key in sorted(lookup.keys()) if key[0] == str(signal_date).strip()]
            if not keys:
                raise KeyError(f"no signals found for {market_run_alias} on {signal_date}")
            return keys[0]
        keys = sorted(lookup.keys())
        if not keys:
            raise KeyError(f"no recorded signals found for {market_run_alias}")
        return keys[0]

    def sample_signal_record(
        self,
        *,
        market_run_alias: str = "",
        signal_date: str = "",
        symbol: str = "",
    ) -> Dict[str, Any]:
        alias = str(market_run_alias or self._default_market_run_alias()).strip()
        if not alias:
            raise ValueError("No market run alias available for signal schema sampling.")
        key = self._resolve_signal_key(alias, signal_date=signal_date, symbol=symbol)
        return dict(self._load_market_scores(alias)[key].get("signal_record") or {})

    def describe_signal_schema(
        self,
        *,
        market_run_alias: str = "",
        signal_date: str = "",
        symbol: str = "",
    ) -> Dict[str, Any]:
        alias = str(market_run_alias or self._default_market_run_alias()).strip()
        cache_key = f"{alias}::{signal_date}::{symbol}"
        if cache_key in self._signal_schema_cache:
            return dict(self._signal_schema_cache[cache_key])

        sample = self.sample_signal_record(
            market_run_alias=alias,
            signal_date=signal_date,
            symbol=symbol,
        )
        features = dict(sample.get("features") or {})
        schema = {
            "market_run_alias": alias,
            "reference_signal": {
                "signal_date": str(sample.get("signal_date", "")).strip(),
                "symbol": str(sample.get("symbol", "")).strip(),
            },
            "required_top_level_keys": ["symbol", "signal_date", "features"],
            "recommended_top_level_keys": ["entry_date", "exit_date"],
            "required_feature_keys": sorted(features.keys()),
            "feature_count": len(features),
            "top_level_types": {
                key: ("object" if isinstance(value, dict) else type(value).__name__)
                for key, value in sample.items()
            },
            "feature_value_type": "number",
            "example_signal_record": sample,
        }
        self._signal_schema_cache[cache_key] = dict(schema)
        return schema

    def write_signal_template(
        self,
        output_path: str,
        *,
        market_run_alias: str = "",
        signal_date: str = "",
        symbol: str = "",
    ) -> Dict[str, Any]:
        template = self.sample_signal_record(
            market_run_alias=market_run_alias,
            signal_date=signal_date,
            symbol=symbol,
        )
        path = Path(output_path).expanduser().resolve()
        write_json(path, template)
        return {
            "output_path": str(path),
            "market_run_alias": str(market_run_alias or self._default_market_run_alias()).strip(),
            "symbol": str(template.get("symbol", "")).strip(),
            "signal_date": str(template.get("signal_date", "")).strip(),
        }

    def normalize_signal_record(
        self,
        signal_record: Mapping[str, Any],
        *,
        market_run_alias: str = "",
    ) -> Dict[str, Any]:
        schema = self.describe_signal_schema(market_run_alias=market_run_alias)
        normalized = dict(signal_record or {})
        normalized["symbol"] = str(normalized.get("symbol", "")).strip()
        normalized["signal_date"] = str(normalized.get("signal_date", "")).strip()
        if "entry_date" in normalized:
            normalized["entry_date"] = str(normalized.get("entry_date", "")).strip()
        if "exit_date" in normalized:
            normalized["exit_date"] = str(normalized.get("exit_date", "")).strip()

        feature_map = dict(normalized.get("features") or {})
        numeric_feature_keys = set(schema["required_feature_keys"])
        for key, value in list(feature_map.items()):
            if key in numeric_feature_keys and isinstance(value, str):
                text = value.strip()
                if text:
                    try:
                        feature_map[key] = float(text)
                    except ValueError:
                        pass
        normalized["features"] = feature_map
        return normalized

    def validate_signal_record(
        self,
        signal_record: Mapping[str, Any],
        *,
        market_run_alias: str = "",
    ) -> Dict[str, Any]:
        schema = self.describe_signal_schema(market_run_alias=market_run_alias)
        record = self.normalize_signal_record(signal_record, market_run_alias=market_run_alias)
        required_top_level = list(schema["required_top_level_keys"])
        recommended_top_level = list(schema["recommended_top_level_keys"])

        missing_top_level = [key for key in required_top_level if key not in record or record.get(key) in ("", None, {})]
        missing_recommended = [key for key in recommended_top_level if key not in record or record.get(key) in ("", None)]
        extra_top_level = sorted([key for key in record.keys() if key not in set(required_top_level + recommended_top_level)])

        features = record.get("features")
        feature_map = dict(features or {}) if isinstance(features, dict) else {}
        feature_keys = set(schema["required_feature_keys"])
        missing_feature_keys = sorted([key for key in feature_keys if key not in feature_map or feature_map.get(key) in ("", None)])
        extra_feature_keys = sorted([key for key in feature_map.keys() if key not in feature_keys])

        non_numeric_feature_keys = []
        for key in feature_keys.intersection(feature_map.keys()):
            value = feature_map.get(key)
            if not isinstance(value, (int, float)):
                non_numeric_feature_keys.append(key)

        valid = (
            not missing_top_level
            and isinstance(features, dict)
            and not missing_feature_keys
            and not non_numeric_feature_keys
        )

        return {
            "valid": bool(valid),
            "required_top_level_keys": required_top_level,
            "recommended_top_level_keys": recommended_top_level,
            "missing_top_level_keys": missing_top_level,
            "missing_recommended_keys": missing_recommended,
            "extra_top_level_keys": extra_top_level,
            "required_feature_count": len(feature_keys),
            "provided_feature_count": len(feature_map),
            "missing_feature_keys": missing_feature_keys,
            "extra_feature_keys": extra_feature_keys,
            "non_numeric_feature_keys": sorted(non_numeric_feature_keys),
            "reference_market_run_alias": schema["market_run_alias"],
            "reference_signal": dict(schema["reference_signal"]),
            "normalized_signal_record": record,
        }

    def _persist_live_run(
        self,
        *,
        lesson_alias: str,
        signal_record: Mapping[str, Any],
        result_payload: Mapping[str, Any],
        signal_schema_validation: Mapping[str, Any],
        source_tag: str,
        run_label: str,
    ) -> str:
        symbol = str(signal_record.get("symbol", "")).strip() or "unknown_symbol"
        signal_date = str(signal_record.get("signal_date", "")).strip() or "unknown_date"
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        signal_text = json.dumps(dict(signal_record), sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha1(signal_text.encode("utf-8")).hexdigest()[:12]
        path = (
            studio_root()
            / "runs"
            / "live_score_runs"
            / lesson_alias
            / f"{timestamp}__{signal_date}__{symbol}__{digest}.json"
        )
        payload = {
            "saved_at": timestamp,
            "profile_id": self.registry.profile_id,
            "lesson_alias": lesson_alias,
            "source_tag": source_tag,
            "run_label": str(run_label or "").strip(),
            "signal_schema_validation": dict(signal_schema_validation),
            "signal_record": dict(signal_record),
            "result": dict(result_payload),
        }
        write_json(path, payload)
        return str(path)

    def score_recorded(self, market_run_alias: str, signal_date: str, symbol: str) -> SignalScore:
        lookup = self._load_market_scores(market_run_alias)
        key = (str(signal_date).strip(), str(symbol).strip())
        if key not in lookup:
            raise KeyError(f"recorded score not found for {market_run_alias}: {signal_date} {symbol}")
        row = lookup[key]
        return SignalScore(
            market_run_alias=market_run_alias,
            signal_date=str(row.get("signal_date", "")),
            symbol=str(row.get("symbol", "")),
            total_score=float(row.get("total_score", 0.0)),
            short_reason=str(row.get("short_reason", "")),
            future_return_5d=float(row.get("future_return_5d", 0.0)),
            signal_record=dict(row.get("signal_record") or {}),
            parsed_payload=dict(row.get("parsed_payload") or {}),
            subscores=dict(row.get("subscores") or {}),
        )

    def runtime_activation_summary(
        self,
        *,
        lesson_alias: str = "",
        final_lesson_state_json: str = "",
    ) -> Dict[str, Any]:
        catalog = self.registry.load_runtime_catalog()
        resolved_alias = str(lesson_alias).strip() or str(catalog.get("defaults", {}).get("alignment_seed_alias", "")).strip()
        return {
            "profile_id": self.registry.profile_id,
            "resolved_lesson_alias": resolved_alias,
            "final_lesson_state_json_hint": str(final_lesson_state_json).strip(),
            "live_model_config": to_jsonable(load_live_model_config()),
            "live_model_status": describe_live_model_config(),
            "default_market_run_alias": self._default_market_run_alias(),
        }

    def build_signal_record_from_recorded(self, market_run_alias: str, signal_date: str, symbol: str) -> Dict:
        lookup = self._load_market_scores(market_run_alias)
        key = (str(signal_date).strip(), str(symbol).strip())
        if key not in lookup:
            raise KeyError(f"recorded score not found for {market_run_alias}: {signal_date} {symbol}")
        return dict(lookup[key].get("signal_record") or {})

    def live_config_status(self) -> Dict:
        return describe_live_model_config()

    def list_recorded_signal_keys(self, market_run_alias: str) -> List[Dict[str, str]]:
        lookup = self._load_market_scores(market_run_alias)
        keys = sorted(lookup.keys())
        return [{"signal_date": signal_date, "symbol": symbol} for signal_date, symbol in keys]

    def list_recorded_signal_keys_window(
        self,
        market_run_alias: str,
        *,
        signal_date: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> Dict:
        keys = self.list_recorded_signal_keys(market_run_alias)
        if signal_date:
            keys = [row for row in keys if row["signal_date"] == str(signal_date).strip()]
        sliced = keys[int(offset) : int(offset) + max(0, int(limit))]
        return {
            "market_run_alias": market_run_alias,
            "signal_date_filter": str(signal_date or "").strip(),
            "offset": int(offset),
            "limit": int(limit),
            "total_count": len(keys),
            "items": sliced,
        }

    def score_live(
        self,
        *,
        lesson_alias: str,
        final_lesson_state_json: str = "",
        signal_record: Mapping[str, object],
        prompt_only: bool = False,
        reuse_cache: bool = True,
        persist_run: bool = False,
        run_label: str = "",
        source_tag: str = "external_signal",
        schema_market_run_alias: str = "",
    ) -> Dict:
        validation = self.validate_signal_record(signal_record, market_run_alias=schema_market_run_alias)
        normalized_signal_record = dict(validation["normalized_signal_record"])
        if not validation["valid"]:
            raise ValueError(
                "Signal record does not match the canonical schema. "
                f"Missing top-level: {validation['missing_top_level_keys']}; "
                f"missing features: {validation['missing_feature_keys']}; "
                f"non-numeric features: {validation['non_numeric_feature_keys']}"
            )
        live_config = load_live_model_config()
        payload = score_live_signal(
            self.registry,
            lesson_alias=lesson_alias,
            final_lesson_state_json=final_lesson_state_json,
            signal_record=normalized_signal_record,
            prompt_only=prompt_only,
            reuse_cache=reuse_cache,
            live_config=live_config,
        )
        payload["signal_schema_validation"] = {
            key: value for key, value in validation.items() if key != "normalized_signal_record"
        }
        if persist_run and not prompt_only:
            payload["saved_run_path"] = self._persist_live_run(
                lesson_alias=lesson_alias,
                signal_record=normalized_signal_record,
                result_payload=payload,
                signal_schema_validation=payload["signal_schema_validation"],
                source_tag=source_tag,
                run_label=run_label,
            )
        return payload

    def compare_live_to_recorded(
        self,
        *,
        lesson_alias: str,
        final_lesson_state_json: str = "",
        market_run_alias: str,
        signal_date: str,
        symbol: str,
        prompt_only: bool = False,
        reuse_cache: bool = True,
    ) -> Dict:
        recorded = self.score_recorded(market_run_alias, signal_date, symbol)
        live = self.score_live(
            lesson_alias=lesson_alias,
            final_lesson_state_json=final_lesson_state_json,
            signal_record=recorded.signal_record,
            prompt_only=prompt_only,
            reuse_cache=reuse_cache,
            persist_run=False,
            source_tag="recorded_replay",
            schema_market_run_alias=market_run_alias,
        )
        payload = {
            "market_run_alias": market_run_alias,
            "signal_date": signal_date,
            "symbol": symbol,
            "lesson_alias": lesson_alias,
            "recorded": recorded.__dict__,
            "live": live,
        }
        if not prompt_only and "total_score" in live:
            payload["score_delta"] = float(live["total_score"]) - float(recorded.total_score)
        return payload

    def score_live_batch_from_recorded(
        self,
        *,
        lesson_alias: str,
        final_lesson_state_json: str = "",
        market_run_alias: str,
        limit: int = 5,
        offset: int = 0,
        signal_date: str = "",
        prompt_only: bool = False,
        reuse_cache: bool = True,
    ) -> Dict:
        lookup = self._load_market_scores(market_run_alias)
        keys = sorted(lookup.keys())
        if signal_date:
            keys = [key for key in keys if key[0] == str(signal_date).strip()]
        sliced = keys[int(offset) : int(offset) + max(0, int(limit))]
        rows: List[Dict] = []
        live_scores: List[float] = []
        recorded_scores: List[float] = []
        for day, symbol in sliced:
            item = self.compare_live_to_recorded(
                lesson_alias=lesson_alias,
                final_lesson_state_json=final_lesson_state_json,
                market_run_alias=market_run_alias,
                signal_date=day,
                symbol=symbol,
                prompt_only=prompt_only,
                reuse_cache=reuse_cache,
            )
            rows.append(item)
            if not prompt_only and "total_score" in item["live"]:
                live_scores.append(float(item["live"]["total_score"]))
                recorded_scores.append(float(item["recorded"]["total_score"]))
        summary = {
            "market_run_alias": market_run_alias,
            "lesson_alias": lesson_alias,
            "requested_limit": int(limit),
            "offset": int(offset),
            "signal_date_filter": str(signal_date or "").strip(),
            "returned_count": len(rows),
            "prompt_only": bool(prompt_only),
        }
        if live_scores and recorded_scores:
            summary.update(
                {
                    "recorded_score_mean": sum(recorded_scores) / len(recorded_scores),
                    "live_score_mean": sum(live_scores) / len(live_scores),
                    "mean_score_delta": (sum(live_scores) - sum(recorded_scores)) / len(live_scores),
                }
            )
        return {"summary": summary, "items": rows}
