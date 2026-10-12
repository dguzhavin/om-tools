"""
sdk_probe.py — стенд для проверки ВСЕХ публичных методов os_ws_sdk 1.0 (WS 9.200).

Что делает:
  1. Берёт настройки из .env (рядом со скриптом).
  2. По группам вызывает каждый публичный метод SDK и записывает факт:
     что вернул (тип, длина, колонки, первые строки), сколько занял,
     какие файлы создал (имя, размер, кодировка, разделитель, первые строки),
     какую ошибку выбросил (тип, модуль, текст).
  3. Пишущие методы (импорт, clear, справочники, формулы, права, UAM, run_script)
     работают только при OM_SKIP_WRITE=0 и только на тестовых объектах из .env.
     Временные справочники создаёт сам и удаляет в конце.
  4. Управление моделью (recalculate, backup, stop/start) — только при OM_TEST_ADMIN=1.
     restore() — отдельно, только при OM_TEST_RESTORE=1.
  5. Складывает результат в OM_WORK_DIR/report.json и report.md.
     Отчёт перезаписывается после каждой группы: если стенд упадёт,
     всё, что успело выполниться, останется в отчёте.

Запуск:  python sdk_probe.py
Первый прогон — с OM_SKIP_WRITE=1 и OM_TEST_ADMIN=0 (только чтение).
"""

import asyncio
import dataclasses
import enum
import importlib
import inspect
import json
import logging
import os
import sys
import time
import traceback
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Пути импорта — из 02_os_ws_sdk_api_signatures.md (проверены по коду wheel)
from os_ws_sdk.api.connector_manager import ConnectorManager
from os_ws_sdk.api.ws_version import WsVersion
from os_ws_sdk.common.exceptions import ModelNotFoundError, WorkspaceException
from os_ws_sdk.components.download_format import DownloadFormat, DownloadFormatOM
from os_ws_sdk.components.encoding_type import EncodingTypeExport, EncodingTypeImport
from os_ws_sdk.components.scripts_history_filter import (
    ScriptRunType,
    ScriptStatus,
    build_scripts_history_filter,
)
from os_ws_sdk.objects.cube_formula_manager import CubeFormulaManager
from os_ws_sdk.objects.permission_change_manager import PermissionChangeManager
from os_ws_sdk.objects.permission_grid_exporter import PermissionGridExporter
from os_ws_sdk.objects.system_grid_exporter import GridTypeIdentifiers, SystemGridExporter
from os_ws_sdk.objects.uam_manager import UamManager
from os_ws_sdk.objects.workspace import Workspace

# ---------------------------------------------------------------------------
# Логи — в консоль
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("sdk_probe")

# Все группы в порядке выполнения. Пишущие и админские — в конце.
ALL_GROUPS = [
    "imports", "connect", "workspace", "model", "system_grids",
    "multicube", "mc_view", "catalog", "catalog_view",
    "formulas_read", "permissions_read", "uam_read", "users", "scripts",
    "dashboards", "optimizer", "backups",
    "catalog_write", "multicube_write", "formulas_write",
    "permissions_write", "uam_write", "scripts_write", "optimizer_write",
    "admin",
]

RESULTS: list = []      # все записи о вызовах
SURFACE: dict = {}      # реальный набор публичных методов у объектов SDK
S: dict = {}            # настройки
CTX: dict = {}          # объекты, полученные по ходу прогона (ws, model, mc, ...)


# ---------------------------------------------------------------------------
# Настройки из .env
# ---------------------------------------------------------------------------
def load_settings() -> dict:
    load_dotenv(Path(__file__).with_name(".env"))

    def req(name: str) -> str:
        value = os.getenv(name, "").strip()
        if not value:
            log.error("В .env не задана переменная %s", name)
            sys.exit(1)
        return value

    def opt(name: str) -> str:
        return os.getenv(name, "").strip()

    def opt_int(name: str):
        value = opt(name)
        if not value:
            return None
        try:
            return int(value)
        except ValueError:
            log.error("Переменная %s должна быть числом, сейчас: %r", name, value)
            sys.exit(1)

    def flag(name: str, default: str) -> bool:
        return os.getenv(name, default).strip() == "1"

    s = {
        "ws_url": req("OM_WS_URL").rstrip("/"),
        "token": req("OM_TOKEN"),
        "model_id": opt("OM_MODEL_ID"),
        "model_name": opt("OM_MODEL_NAME"),
        "work_dir": Path(opt("OM_WORK_DIR") or "./sdk_probe_files").resolve(),
        # режимы
        "skip_write": flag("OM_SKIP_WRITE", "1"),
        "admin": flag("OM_TEST_ADMIN", "0"),
        "restore": flag("OM_TEST_RESTORE", "0"),
        "download_backup": flag("OM_TEST_DOWNLOAD_BACKUP", "0"),
        "mc_copy": flag("OM_TEST_MC_COPY", "0"),
        "save_dfs": flag("OM_SAVE_DATAFRAMES", "1"),
        "timeout": float(opt("OM_PROBE_TIMEOUT") or "600"),
        "groups": [g.strip() for g in opt("OM_GROUPS").split(",") if g.strip()],
        # объекты для чтения
        "mc_name": req("OM_TEST_MULTICUBE"),
        "mc_view": opt("OM_TEST_MC_VIEW"),
        "catalog_name": req("OM_TEST_CATALOG"),
        "catalog_view": opt("OM_TEST_CATALOG_VIEW"),
        "cube_id": opt_int("OM_TEST_CUBE_ID"),
        "bool_cube_id": opt_int("OM_TEST_BOOLEAN_CUBE_ID"),
        "filter_dim_id": opt_int("OM_TEST_FILTER_DIMENSION_ID"),
        "filter_formula": opt("OM_TEST_FILTER_FORMULA"),
        "dashboard_id": opt_int("OM_TEST_DASHBOARD_ID"),
        "optimization_id": opt_int("OM_TEST_OPTIMIZATION_ID"),
        "script_id": opt_int("OM_TEST_SCRIPT_ID"),
        "model_user_id": opt("OM_TEST_MODEL_USER_ID"),
        "ws_user_id": opt("OM_TEST_WS_USER_ID"),
        "backup_row_id": opt("OM_TEST_BACKUP_ROW_ID"),
        "backup_id": opt("OM_TEST_BACKUP_ID"),
        "uam_list_id": opt_int("OM_TEST_UAM_LIST_ID"),
        "time_dim_id": opt_int("OM_TEST_TIME_DIMENSION_ID"),
        "filter_dim_alt": [int(x) for x in opt("OM_TEST_FILTER_DIMENSION_ALT").replace(" ", "").split(",") if x],
        # объекты для записи
        "target_mc_name": opt("OM_TEST_TARGET_MULTICUBE"),
        "cell_value": opt("OM_TEST_CELL_VALUE"),
        "cell_cube_id": opt_int("OM_TEST_CELL_CUBE_ID"),
        "cell_dim_id": opt_int("OM_TEST_CELL_DIMENSION_ID"),
        "formula_cube_id": opt_int("OM_TEST_FORMULA_CUBE_ID"),
        "formula_new": opt("OM_TEST_FORMULA_NEW"),
        "role_id": opt("OM_TEST_ROLE_ID"),
        "version_id": opt_int("OM_TEST_VERSION_ID"),
        "perm_set": opt("OM_TEST_PERMISSION_SET") or "read",
        "perm_restore": opt("OM_TEST_PERMISSION_RESTORE"),
        "uam_item_id": opt("OM_TEST_UAM_ITEM_ID"),
        "uam_user": opt("OM_TEST_UAM_USER"),
        "uam_set": opt("OM_TEST_UAM_SET") or "read",
        "uam_restore": opt("OM_TEST_UAM_RESTORE"),
        "solution_zip": opt("OM_TEST_SOLUTION_ZIP"),
    }
    if not s["model_id"] and not s["model_name"]:
        log.error("Укажите в .env OM_MODEL_ID или OM_MODEL_NAME")
        sys.exit(1)
    unknown = [g for g in s["groups"] if g not in ALL_GROUPS]
    if unknown:
        log.error("В OM_GROUPS неизвестные группы: %s. Доступны: %s", unknown, ", ".join(ALL_GROUPS))
        sys.exit(1)
    return s


# ---------------------------------------------------------------------------
# Описание результата: тип, размер, превью. Без токена и без внутренностей SDK.
# ---------------------------------------------------------------------------
HIDDEN_ATTRS = ("token", "connector", "session", "header", "password", "secret")


def short(text, limit: int = 500) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} симв.)"


def type_name(value) -> str:
    t = type(value)
    return f"{t.__module__}.{t.__qualname__}"


def public_attrs(obj) -> dict:
    """Простые публичные атрибуты объекта SDK (имя, id и т.п.), без коннектора и токена."""
    out = {}
    try:
        items = vars(obj).items()
    except TypeError:
        return out
    for key, value in items:
        if key.startswith("_") or any(h in key.lower() for h in HIDDEN_ATTRS):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
            out[key] = value if not isinstance(value, str) else short(value, 200)
        elif isinstance(value, enum.Enum):
            out[key] = str(value)
        else:
            out[key] = f"<{type_name(value)}>"
    return out


def describe(value, depth: int = 0):
    d = {"type": type_name(value)}
    if value is None:
        return d
    if isinstance(value, enum.Enum):
        d["value"] = str(value)
    elif isinstance(value, (bool, int, float)):
        d["value"] = value
    elif isinstance(value, str):
        d["len"] = len(value)
        d["preview"] = short(value)
    elif isinstance(value, (bytes, bytearray)):
        d["len"] = len(value)
        d["head_hex"] = bytes(value[:16]).hex()
        d["looks_like"] = "zip" if value[:2] == b"PK" else "unknown"
    elif isinstance(value, pd.DataFrame):
        d["shape"] = list(value.shape)
        d["columns"] = [str(c) for c in value.columns[:80]]
        d["dtypes"] = {str(c): str(t) for c, t in list(value.dtypes.items())[:80]}
        d["index"] = type_name(value.index)
        d["head"] = json.loads(value.head(5).astype(str).to_json(orient="records", force_ascii=False))
    elif isinstance(value, pd.Series):
        d["len"] = len(value)
        d["head"] = [short(v, 100) for v in value.head(5).tolist()]
    elif isinstance(value, dict):
        d["len"] = len(value)
        d["keys"] = [str(k) for k in list(value.keys())[:50]]
        if depth < 2:
            d["items"] = {str(k): describe(v, depth + 1) for k, v in list(value.items())[:15]}
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
        d["len"] = len(items)
        d["item_types"] = dict(Counter(type_name(i) for i in items))
        if depth < 2:
            d["first"] = [describe(i, depth + 1) for i in items[:3]]
    elif dataclasses.is_dataclass(value):
        fields = {}
        for f in dataclasses.fields(value):
            if any(h in f.name.lower() for h in HIDDEN_ATTRS):
                continue
            v = getattr(value, f.name, None)
            fields[f.name] = describe(v, depth + 1) if depth < 2 else type_name(v)
        d["fields"] = fields
    else:
        d["attrs"] = public_attrs(value)
    return d


def api_surface(obj) -> dict:
    """Какие публичные методы реально есть у объекта и какие из них async."""
    out = {}
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            attr = getattr(type(obj), name, None) or getattr(obj, name)
        except Exception:
            continue
        if callable(attr) and not inspect.isclass(attr):
            out[name] = "async" if inspect.iscoroutinefunction(attr) else "sync"
    return out


# ---------------------------------------------------------------------------
# Файлы, созданные вызовом: имя, размер, кодировка, разделитель, первые строки
# ---------------------------------------------------------------------------
def snapshot(folder) -> dict:
    if not folder or not Path(folder).exists():
        return {}
    return {p: p.stat().st_mtime for p in Path(folder).rglob("*") if p.is_file()}


def describe_file(path: Path) -> dict:
    info = {"name": path.name, "size": path.stat().st_size}
    raw = path.read_bytes()[:20000]
    if raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(path) as z:
                info["zip_entries"] = [{"name": i.filename, "size": i.file_size} for i in z.infolist()[:50]]
                # первые строки текстовых файлов внутри zip (LP, map, csv)
                previews = {}
                for i in z.infolist()[:10]:
                    if i.filename.lower().endswith((".lp", ".map", ".csv", ".txt")):
                        with z.open(i) as f:
                            head = f.read(3000).decode("utf-8", errors="replace").splitlines()[:5]
                        previews[i.filename] = [short(line, 300) for line in head]
                if previews:
                    info["zip_previews"] = previews
        except zipfile.BadZipFile:
            info["zip_error"] = "BadZipFile"
        info["kind"] = "zip/xlsx"
        return info
    if raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        info["kind"] = "xls (OLE2)"
        return info
    info["bom"] = raw[:3] == b"\xef\xbb\xbf"
    for enc in ("utf-8", "cp1251"):
        try:
            text = raw.decode(enc)
            info["encoding_guess"] = enc
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
        info["encoding_guess"] = "unknown"
    lines = text.lstrip("\ufeff").splitlines()
    first = lines[0] if lines else ""
    counts = {sep: first.count(sep) for sep in (";", ",", "\t", "|")}
    info["delimiter_guess"] = repr(max(counts, key=counts.get)) if any(counts.values()) else None
    info["first_lines"] = [short(line, 300) for line in lines[:5]]
    info["kind"] = "text"
    return info


def new_files(before: dict, folder) -> list:
    after = snapshot(folder)
    changed = [p for p, m in after.items() if p not in before or before[p] != m]
    return [describe_file(p) for p in sorted(changed)]


# ---------------------------------------------------------------------------
# Один вызов метода SDK с записью результата
# ---------------------------------------------------------------------------
def slug(text: str) -> str:
    keep = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(filter(None, keep.split("_")))[:80]


def probe_dir(group: str, call: str) -> Path:
    """Отдельная пустая папка под каждый вызов, чтобы видеть, какой файл он создал."""
    d = S["work_dir"] / "export" / group / slug(call)
    d.mkdir(parents=True, exist_ok=True)
    return d


async def probe(group, call, fn, *, kind="read", out_dir=None, expect_error=False, note=""):
    """
    fn — lambda без аргументов, возвращает корутину или обычное значение.
    kind: read | write | admin. expect_error=True — негативный тест (ошибка = норма).
    Возвращает результат вызова или None при ошибке/пропуске.
    """
    rec = {"group": group, "call": call, "kind": kind, "expect_error": expect_error, "note": note}
    if kind == "write" and S["skip_write"]:
        return record_skip(rec, "OM_SKIP_WRITE=1")
    if kind == "admin" and not S["admin"]:
        return record_skip(rec, "OM_TEST_ADMIN=0")

    log.info("[%s] %s", group, call)
    before = snapshot(out_dir)
    value = None
    t0 = time.perf_counter()
    try:
        value = fn()
        if inspect.isawaitable(value):
            value = await asyncio.wait_for(value, S["timeout"])
        rec["status"] = "unexpected_ok" if expect_error else "ok"
        rec["result"] = describe(value)
    except Exception as e:  # noqa: BLE001 — стенду нужны любые ошибки
        rec["status"] = "expected_error" if expect_error else "error"
        rec["error"] = {
            "type": type_name(e),
            "message": short(e, 1500),
            "traceback_tail": [short(x, 300) for x in traceback.format_exc().strip().splitlines()[-6:]],
        }
        value = None
    rec["elapsed_s"] = round(time.perf_counter() - t0, 3)
    if out_dir is not None:
        rec["files"] = new_files(before, out_dir)
    save_dataframe(rec, value)
    RESULTS.append(rec)
    log_result(rec)
    return value


def record_skip(rec: dict, reason: str):
    rec["status"] = "skipped"
    rec["reason"] = reason
    RESULTS.append(rec)
    log.info("[%s] %s — пропуск: %s", rec["group"], rec["call"], reason)
    return None


def skip(group: str, call: str, reason: str, kind: str = "read"):
    return record_skip({"group": group, "call": call, "kind": kind, "expect_error": False, "note": ""}, reason)


def save_dataframe(rec: dict, value) -> None:
    """Полные DataFrame (до 2000 строк) сохраняются в results/ — по ним видно реальные колонки."""
    if not S["save_dfs"] or not isinstance(value, pd.DataFrame):
        return
    out = S["work_dir"] / "results" / f"{len(RESULTS) + 1:03d}_{slug(rec['call'])}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    value.head(2000).to_csv(out, sep=";", index=False, encoding="utf-8")
    rec["saved_dataframe"] = str(out.relative_to(S["work_dir"]))


def log_result(rec: dict) -> None:
    status = rec["status"]
    took = f"{rec['elapsed_s']:.2f} с"
    if status in ("ok", "expected_error"):
        res = rec.get("result") or {}
        what = res.get("type", "")
        if "shape" in res:
            what += f" shape={res['shape']}"
        elif "len" in res:
            what += f" len={res['len']}"
        elif "value" in res:
            what += f" = {res['value']!r}"
        if status == "expected_error":
            what = f"ожидаемая ошибка {rec['error']['type']}: {short(rec['error']['message'], 150)}"
        log.info("    ок за %s → %s", took, what)
        for f in rec.get("files", []):
            log.info("    файл: %s (%s байт)", f["name"], f["size"])
    elif status == "unexpected_ok":
        log.warning("    ждали ошибку, а вызов прошёл за %s → %s", took, rec["result"].get("type"))
    else:
        log.warning("    ОШИБКА за %s: %s: %s", took, rec["error"]["type"], short(rec["error"]["message"], 300))


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------
async def obj_id(obj):
    """id объекта SDK: сначала атрибут, иначе info().id."""
    value = getattr(obj, "id", None)
    if value is not None:
        return value
    info = obj.info()
    if inspect.isawaitable(info):
        info = await info
    return getattr(info, "id", None)


def find_col(columns, candidates, contains=None):
    """Ищет колонку по точным именам, потом по подстроке. ПРОВЕРИТЬ: имена колонок на стенде."""
    low = {str(c).strip().lower(): i for i, c in enumerate(columns)}
    for cand in candidates:
        if cand in low:
            return low[cand]
    if contains:
        for name, i in low.items():
            if contains in name:
                return i
    return None


def element_ids_by_name(columns, rows, names) -> dict:
    """Карта имя → id элемента по результату get_elements()."""
    id_i = find_col(columns, ["id", "longid", "long id", "item id"], contains="id")
    name_i = find_col(columns, ["item name", "name", "имя", "название", "наименование"])
    if id_i is None:
        return {}
    out = {}
    for row in rows:
        cells = list(row) if isinstance(row, (list, tuple)) else [row]
        if id_i >= len(cells):
            continue
        candidates = [cells[name_i]] if name_i is not None and name_i < len(cells) else cells
        for n in names:
            if n in [str(c) for c in candidates]:
                out[n] = cells[id_i]
    return out


def write_report() -> None:
    """report.json (всё) и report.md (сводка + детали). Токен вырезается."""
    meta = {
        "started_at": CTX.get("started_at"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "ws_url": S["ws_url"],
        "model": CTX.get("model_label"),
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "sdk_version": CTX.get("sdk_version"),
        "skip_write": S["skip_write"],
        "admin": S["admin"],
        "groups": S["groups"] or ALL_GROUPS,
    }
    data = {"meta": meta, "api_surface": SURFACE, "results": RESULTS}
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    text = text.replace(S["token"], "***")
    (S["work_dir"] / "report.json").write_text(text, encoding="utf-8")
    (S["work_dir"] / "report.md").write_text(render_md(meta).replace(S["token"], "***"), encoding="utf-8")


def render_md(meta: dict) -> str:
    counts = Counter(r["status"] for r in RESULTS)
    lines = [
        "# Отчёт sdk_probe",
        "",
        f"- Запуск: {meta['started_at']} → {meta['finished_at']}",
        f"- Workspace: {meta['ws_url']}, модель: {meta['model']}",
        f"- SDK: {meta['sdk_version']}, Python {meta['python']}, pandas {meta['pandas']}",
        f"- Запись: {'выключена' if meta['skip_write'] else 'ВКЛЮЧЕНА'}, админка: {'да' if meta['admin'] else 'нет'}",
        f"- Итог: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        "| # | Группа | Вызов | Тип | Статус | Время, с | Результат |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(RESULTS, 1):
        res = r.get("result") or {}
        if r["status"] == "skipped":
            summary = r["reason"]
        elif "error" in r:
            summary = f"{r['error']['type'].split('.')[-1]}: {short(r['error']['message'], 80)}"
        else:
            summary = res.get("type", "").split(".")[-1]
            if "shape" in res:
                summary += f" {res['shape'][0]}×{res['shape'][1]}"
            elif "len" in res:
                summary += f" len={res['len']}"
            elif "value" in res:
                summary += f" = {short(res['value'], 40)}"
            if r.get("files"):
                summary += " · файлы: " + ", ".join(f["name"] for f in r["files"])
        summary = summary.replace("|", "\\|").replace("\n", " ")
        call = r["call"].replace("|", "\\|")
        lines.append(f"| {i} | {r['group']} | `{call}` | {r['kind']} | {r['status']} | "
                     f"{r.get('elapsed_s', '')} | {summary} |")
    lines += ["", "## Детали", ""]
    for i, r in enumerate(RESULTS, 1):
        if r["status"] == "skipped":
            continue
        body = {k: v for k, v in r.items() if k not in ("group", "call")}
        lines += [f"### {i}. `{r['call']}`", "", "```json",
                  short(json.dumps(body, ensure_ascii=False, indent=2, default=str), 6000), "```", ""]
    lines += ["## Публичные методы объектов (dir)", "", "```json",
              json.dumps(SURFACE, ensure_ascii=False, indent=2), "```"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Группы проверок
# ---------------------------------------------------------------------------
async def g_imports():
    """Какие пути импорта реально работают (в документации описаны не все)."""
    g = "imports"
    try:
        from importlib.metadata import version
        CTX["sdk_version"] = version("os_ws_sdk")
    except Exception:
        CTX["sdk_version"] = "не определена"
    log.info("Версия os_ws_sdk: %s", CTX["sdk_version"])
    paths = [
        ("os_ws_sdk", "ScriptRunType"),            # так в документации
        ("os_ws_sdk", "ScriptStatus"),             # так в документации
        ("os_ws_sdk", "Workspace"),                # есть ли реэкспорт в корне
        ("os_ws_sdk", "ConnectorManager"),
        ("os_ws_sdk.components.position", "Position"),  # из документации, в файле 02 модуля нет
        ("os_ws_sdk.api.v1.connector_v1", "ConnectorV1"),
        ("os_ws_sdk.logging_sdk", "logger"),
        ("os_ws_sdk.objects.permission_grid_exporter", "PermissionGridData"),
        ("os_ws_sdk.objects.uam_manager", "UamGridData"),
    ]
    for module, name in paths:
        await probe(g, f"from {module} import {name}",
                    lambda m=module, n=name: getattr(importlib.import_module(m), n))
    CTX["Position"] = None
    try:
        CTX["Position"] = importlib.import_module("os_ws_sdk.components.position").Position
    except Exception:
        pass
    await probe(g, "list(WsVersion)", lambda: [str(v) for v in WsVersion])
    await probe(g, "list(DownloadFormat)", lambda: [f"{v.name}={v.value}" for v in DownloadFormat])
    await probe(g, "list(DownloadFormatOM)", lambda: [f"{v.name}={v.value}" for v in DownloadFormatOM])
    await probe(g, "list(EncodingTypeExport)", lambda: [f"{v.name}={v.value}" for v in EncodingTypeExport])
    await probe(g, "list(EncodingTypeImport)", lambda: [f"{v.name}={v.value}" for v in EncodingTypeImport])
    await probe(g, "list(GridTypeIdentifiers)", lambda: [f"{v.name}={v.value}" for v in GridTypeIdentifiers])
    await probe(g, "list(ScriptStatus)", lambda: [f"{v.name}={v.value}" for v in ScriptStatus])
    await probe(g, "list(ScriptRunType)", lambda: [f"{v.name}={v.value}" for v in ScriptRunType])
    if CTX["Position"] is not None:
        await probe(g, "list(Position)", lambda: [f"{v.name}={v.value}" for v in CTX["Position"]])
    # Чистая функция: показывает, какой фильтр реально уходит на сервер
    await probe(g, "build_scripts_history_filter(status=SUCCESS, run_type=API, show_archive=True)",
                lambda: build_scripts_history_filter(status=ScriptStatus.SUCCESS,
                                                     run_type=ScriptRunType.API, show_archive=True))
    await probe(g, "build_scripts_history_filter(date_from=now-7d, date_to=now)",
                lambda: build_scripts_history_filter(
                    date_from=datetime.now(timezone.utc) - timedelta(days=7),
                    date_to=datetime.now(timezone.utc)))


async def g_connect():
    g = "connect"
    connector = await probe(g, "ConnectorManager.create(version=VERSION_9_200, token=***)",
                            lambda: ConnectorManager.create(version=WsVersion.VERSION_9_200, token=S["token"]))
    if connector is None:
        raise RuntimeError("Не удалось создать коннектор — дальше проверять нечего")
    SURFACE["connector"] = api_surface(connector)
    ws = await probe(g, "Workspace(url, connector)", lambda: Workspace(S["ws_url"], connector))
    if ws is None:
        raise RuntimeError("Не удалось создать Workspace")
    SURFACE["workspace"] = api_surface(ws)
    CTX["connector"], CTX["ws"] = connector, ws


async def g_workspace():
    g = "workspace"
    ws = CTX["ws"]
    await probe(g, "ws.info()", lambda: ws.info())
    models = await probe(g, "ws.get_models()", lambda: ws.get_models())
    if models:
        await probe(g, "models[0].info()", lambda: models[0].info())
    await probe(g, "ws.get_model(id='000…0')  # нет такой", expect_error=True,
                fn=lambda: ws.get_model(id="0" * 32))
    await probe(g, "ws.get_model_by_name('__sdk_probe_nope__')",
                lambda: ws.get_model_by_name(name="__sdk_probe_nope__"),
                note="Что будет, если модели нет: None или исключение")


async def resolve_model():
    ws = CTX["ws"]
    if S["model_id"]:
        model = await probe("model", f"ws.get_model(id='{S['model_id']}')", lambda: ws.get_model(id=S["model_id"]))
    else:
        model = await probe("model", f"ws.get_model_by_name('{S['model_name']}')",
                            lambda: ws.get_model_by_name(name=S["model_name"]))
    if model is None:
        log.error("Модель не найдена: %s", S["model_id"] or S["model_name"])
        sys.exit(1)
    info = model.info()
    CTX["model"] = model
    CTX["model_label"] = f"{info.model_name} ({info.model_id})"
    SURFACE["model"] = api_surface(model)
    log.info("Модель: %s | ячеек %s | офлайн %s", info.model_name, info.cell_count, info.is_offline)


async def resolve_objects():
    """Тестовые мультикуб, вью и справочник — нужны группам, даже если группа model не выбрана."""
    m = CTX["model"]
    try:
        mc = await m.get_multicube_by_name(S["mc_name"])
        if mc is not None:
            CTX["mc"], CTX["mc_id"] = mc, await obj_id(mc)
            if S["mc_view"]:
                CTX["view"] = await mc.get_view_by_name(S["mc_view"])
            else:
                views = await mc.get_views()
                CTX["view"] = views[0] if views else None
        else:
            log.error("Тестовый мультикуб «%s» не найден", S["mc_name"])
    except Exception:
        log.exception("Не удалось получить тестовый мультикуб")
    try:
        cat = await m.get_catalog_by_name(S["catalog_name"])
        if cat is not None:
            CTX["catalog"], CTX["catalog_id"] = cat, await obj_id(cat)
            if S["catalog_view"]:
                CTX["catalog_view_obj"] = await cat.get_view_by_name(S["catalog_view"])
            else:
                views = await cat.get_views()
                CTX["catalog_view_obj"] = views[0] if views else None
    except Exception:
        log.exception("Не удалось получить тестовый справочник «%s»", S["catalog_name"])
    for key, label in (("mc", "multicube"), ("view", "view"), ("catalog", "catalog"),
                       ("catalog_view_obj", "catalog_view")):
        if CTX.get(key) is not None:
            SURFACE[label] = api_surface(CTX[key])
    log.info("Тестовые объекты: мультикуб id=%s, вью=%s, справочник id=%s, вью справочника=%s",
             CTX.get("mc_id"), "есть" if CTX.get("view") else "нет",
             CTX.get("catalog_id"), "есть" if CTX.get("catalog_view_obj") else "нет")


async def g_model():
    g = "model"
    m = CTX["model"]
    await probe(g, "model.info()  # sync", lambda: m.info())
    mcs = await probe(g, "model.get_multicubes()", lambda: m.get_multicubes())
    mc = await probe(g, f"model.get_multicube_by_name('{S['mc_name']}')", lambda: m.get_multicube_by_name(S["mc_name"]))
    if mc is None:
        log.error("Тестовый мультикуб «%s» не найден — группы multicube* будут пропущены", S["mc_name"])
    else:
        CTX["mc"] = mc
        CTX["mc_id"] = await obj_id(mc)
        SURFACE["multicube"] = api_surface(mc)
        await probe(g, f"model.get_multicube({CTX['mc_id']})", lambda: m.get_multicube(CTX["mc_id"]))
    await probe(g, "model.get_multicube_by_name('__nope__')", lambda: m.get_multicube_by_name("__sdk_probe_nope__"),
                note="Ожидаем None (по коду SDK)")
    await probe(g, "model.get_multicube(999999999999)", lambda: m.get_multicube(999999999999),
                note="Несуществующий id: None или исключение")
    await probe(g, f"model.get_id_multicube_by_name('{S['mc_name']}')", lambda: m.get_id_multicube_by_name(S["mc_name"]))
    await probe(g, "model.get_id_multicube_by_name('__nope__')", expect_error=True,
                fn=lambda: m.get_id_multicube_by_name("__sdk_probe_nope__"))

    await probe(g, "model.get_catalogs()", lambda: m.get_catalogs())
    await probe(g, "model.get_catalogs(start=0, count=3)", lambda: m.get_catalogs(start=0, count=3),
                note="Проверка пагинации")
    cat = await probe(g, f"model.get_catalog_by_name('{S['catalog_name']}')",
                      lambda: m.get_catalog_by_name(S["catalog_name"]))
    if cat is not None:
        CTX["catalog"] = cat
        CTX["catalog_id"] = await obj_id(cat)
        SURFACE["catalog"] = api_surface(cat)
        await probe(g, f"model.get_catalog({CTX['catalog_id']})", lambda: m.get_catalog(CTX["catalog_id"]))
        await probe(g, f"model.get_catalog_views({CTX['catalog_id']})", lambda: m.get_catalog_views(CTX["catalog_id"]))
        if S["catalog_view"]:
            await probe(g, f"model.get_catalog_view_by_name({CTX['catalog_id']}, '{S['catalog_view']}')",
                        lambda: m.get_catalog_view_by_name(CTX["catalog_id"], S["catalog_view"]))
        await probe(g, f"model.get_catalog_view_by_name({CTX['catalog_id']}, '__nope__')",
                    lambda: m.get_catalog_view_by_name(CTX["catalog_id"], "__sdk_probe_nope__"),
                    note="По документации — None")
    else:
        log.error("Тестовый справочник «%s» не найден — группы catalog* будут пропущены", S["catalog_name"])
    await probe(g, "model.get_catalog_by_name('__nope__')", expect_error=True,
                fn=lambda: m.get_catalog_by_name("__sdk_probe_nope__"),
                note="Прогон 1: WorkspaceException 404, в отличие от get_multicube_by_name (None)")
    await probe(g, "model.get_catalog(999999999999)", expect_error=True, fn=lambda: m.get_catalog(999999999999))
    for label, action_id in (("optimization", S["optimization_id"]), ("script", S["script_id"])):
        if action_id:
            await probe(g, f"model.action_info({action_id})  # {label}", lambda a=action_id: m.action_info(a))


async def g_system_grids():
    g = "system_grids"
    exporter = SystemGridExporter.from_model(CTX["model"])
    SURFACE["system_grid_exporter"] = api_surface(exporter)
    # Каждый шаблонный грид отдельно — в отчёте будут реальные колонки каждого
    for grid in GridTypeIdentifiers:
        await probe(g, f"exporter.export_df(GridTypeIdentifiers.{grid.name})  # {grid.value}",
                    lambda gr=grid: exporter.export_df(gr))
    await probe(g, "exporter.export_df(GridTypeIdentifiers.CUBES_GRID, row_count=5)",
                lambda: exporter.export_df(GridTypeIdentifiers.CUBES_GRID, row_count=5),
                note="row_count — размер страницы или общий лимит?")
    await probe(g, "exporter.export_df(137000000002)  # числовой id вместо enum",
                lambda: exporter.export_df(137000000002))
    await probe(g, "exporter.export_df(137000000051)  # OPTIMIZER_REQUESTS_TABLES, нет в enum",
                lambda: exporter.export_df(137000000051))
    await probe(g, "exporter.grid(GridTypeIdentifiers.FOLDERS_GRID).export_df()",
                lambda: exporter.grid(GridTypeIdentifiers.FOLDERS_GRID).export_df())
    # Прогон 1: LIST_PROPERTIES_GRID и TIME_*_SUBSETS_GRID без контекста дают
    # 'viewReference entity not valid'. Пробуем передать контекст.
    if CTX.get("catalog_id"):
        await probe(g, f"exporter.export_df(LIST_PROPERTIES_GRID, context_entity_long_id={CTX['catalog_id']})",
                    lambda: exporter.export_df(GridTypeIdentifiers.LIST_PROPERTIES_GRID,
                                               context_entity_long_id=CTX["catalog_id"]))
    if S["time_dim_id"]:
        for grid in (GridTypeIdentifiers.TIME_MONTHS_SUBSETS_GRID, GridTypeIdentifiers.TIME_YEARS_SUBSETS_GRID):
            await probe(g, f"exporter.export_df({grid.name}, context_entity_long_id={S['time_dim_id']})",
                        lambda gr=grid: exporter.export_df(gr, context_entity_long_id=S["time_dim_id"]),
                        note="ПРОВЕРИТЬ: какой контекст нужен временным гридам — догадка")
    await probe(g, "exporter.export_many()  # все гриды по умолчанию, часть из них падает",
                lambda: exporter.export_many(),
                note="Валит ли одна ошибочная таблица весь вызов")
    pair = (GridTypeIdentifiers.MULTICUBES_GRID, GridTypeIdentifiers.CUBES_GRID)
    await probe(g, "exporter.export_many([MULTICUBES_GRID, CUBES_GRID])", lambda: exporter.export_many(pair))
    await probe(g, "exporter.export_many_as_one_df([MULTICUBES_GRID, CUBES_GRID])",
                lambda: exporter.export_many_as_one_df(pair))


async def g_multicube():
    g = "multicube"
    mc = CTX.get("mc")
    if mc is None:
        return skip(g, "*", "тестовый мультикуб не найден")
    await probe(g, "mc.info()", lambda: mc.info())
    views = await probe(g, "mc.get_views()", lambda: mc.get_views())
    await probe(g, "mc.get_views(start=0, count=1)", lambda: mc.get_views(start=0, count=1))
    if S["mc_view"]:
        view = await probe(g, f"mc.get_view_by_name('{S['mc_view']}')", lambda: mc.get_view_by_name(S["mc_view"]))
        if view is not None:
            CTX["view"] = view
            SURFACE["view"] = api_surface(view)
    elif views:
        CTX["view"] = views[0]
        SURFACE["view"] = api_surface(views[0])
    await probe(g, "mc.get_view_by_name('__nope__')", lambda: mc.get_view_by_name("__sdk_probe_nope__"),
                note="None или исключение")

    # Обычный экспорт — все форматы, каждый в свою папку
    for fmt in DownloadFormat:
        call = f"mc.export_file(path, format=DownloadFormat.{fmt.name})"
        d = probe_dir(g, call)
        await probe(g, call, lambda d=d, f=fmt: mc.export_file(path=str(d), format=f), out_dir=d)
    call = "mc.export_file(path)  # всё по умолчанию"
    d = probe_dir(g, call)
    await probe(g, call, lambda: mc.export_file(path=str(d)), out_dir=d)
    call = "mc.export_file(path, CSV, delimiter=',', decimal_separator=',')"
    d = probe_dir(g, call)
    await probe(g, call, lambda: mc.export_file(path=str(d), format=DownloadFormat.CSV, delimiter=",",
                                                decimal_separator=","), out_dir=d)
    call = "mc.export_file(path, CSV, encoding=EncodingTypeExport.WINDOWS_1251)"
    d = probe_dir(g, call)
    await probe(g, call, lambda: mc.export_file(path=str(d), format=DownloadFormat.CSV,
                                                encoding=EncodingTypeExport.WINDOWS_1251), out_dir=d)

    # Экспорт в формате ОМ
    for fmt in DownloadFormatOM:
        call = f"mc.export_om_file(path, format=DownloadFormatOM.{fmt.name})"
        d = probe_dir(g, call)
        await probe(g, call, lambda d=d, f=fmt: mc.export_om_file(path=str(d), format=f), out_dir=d)

    # Фильтры (только если заданы в .env)
    if S["bool_cube_id"] and S["filter_dim_id"]:
        flt = [S["bool_cube_id"], S["filter_dim_id"]]
        call = f"mc.export_file(path, filter_boolean_cube={flt})"
        d = probe_dir(g, call)
        await probe(g, call, lambda: mc.export_file(path=str(d), filter_boolean_cube=flt), out_dir=d)
        await probe(g, f"mc.export_df(filter_boolean_cube={flt})", lambda: mc.export_df(filter_boolean_cube=flt))
    else:
        skip(g, "mc.export_file / export_df с filter_boolean_cube",
             "не заданы OM_TEST_BOOLEAN_CUBE_ID и OM_TEST_FILTER_DIMENSION_ID")
    # Прогон 3: filter_boolean_cube с измерением 13000000003 не меняет файл, filter_formula
    # убирает все месяцы. Пробуем другие id измерений из OM_TEST_FILTER_DIMENSION_ALT.
    for alt in S["filter_dim_alt"]:
        if S["bool_cube_id"]:
            flt = [S["bool_cube_id"], alt]
            call = f"mc.export_file(path, filter_boolean_cube={flt})  # другое измерение"
            d = probe_dir(g, call)
            await probe(g, call, lambda d=d, f=flt: mc.export_file(path=str(d), filter_boolean_cube=f), out_dir=d)
        if S["filter_formula"]:
            flt = [S["filter_formula"], alt]
            call = f"mc.export_file(path, filter_formula=[formula, {alt}])  # другое измерение"
            d = probe_dir(g, call)
            await probe(g, call, lambda d=d, f=flt: mc.export_file(path=str(d), filter_formula=f), out_dir=d)
    if S["bool_cube_id"]:
        call = f"mc.export_om_file(path, boolean_cube_filter_long_id={S['bool_cube_id']})"
        d = probe_dir(g, call)
        await probe(g, call, lambda: mc.export_om_file(path=str(d), boolean_cube_filter_long_id=S["bool_cube_id"]),
                    out_dir=d)
    if S["filter_formula"] and S["filter_dim_id"]:
        flt = [S["filter_formula"], S["filter_dim_id"]]
        call = "mc.export_file(path, filter_formula=[formula, dim_id])"
        d = probe_dir(g, call)
        await probe(g, call, lambda: mc.export_file(path=str(d), filter_formula=flt), out_dir=d)
        await probe(g, "mc.export_df(filter_formula=[formula, dim_id])", lambda: mc.export_df(filter_formula=flt))
    if S["filter_formula"]:
        call = "mc.export_om_file(path, filter_formula=formula)  # строка, не список"
        d = probe_dir(g, call)
        await probe(g, call, lambda: mc.export_om_file(path=str(d), filter_formula=S["filter_formula"]), out_dir=d)
    else:
        skip(g, "экспорт с filter_formula", "не задан OM_TEST_FILTER_FORMULA")

    # DataFrame и выгрузка всех вью
    await probe(g, "mc.export_df()", lambda: mc.export_df())
    await probe(g, "mc.export_df(dtype=str)  # kwargs уходят в pd.read_csv?", lambda: mc.export_df(dtype=str))
    # Прогон 2: export_df() вернул DataFrame из ОДНОЙ колонки — CSV с ';' читается как с ','.
    await probe(g, "mc.export_df(sep=';')  # обход: явный разделитель для read_csv", lambda: mc.export_df(sep=";"),
                note="Если kwargs уходят в read_csv, колонок станет больше одной")
    await probe(g, "mc.export_df(delimiter=';')", lambda: mc.export_df(delimiter=";"),
                note="delimiter — параметр экспорта или read_csv")
    call = "mc.export_views(path)"
    d = probe_dir(g, call)
    await probe(g, call, lambda: mc.export_views(path=str(d)), out_dir=d)
    missing = S["work_dir"] / "нет_папки" / "x"

    async def export_to_missing():
        await mc.export_file(path=str(missing))
        found = [str(p.relative_to(S["work_dir"])) for p in S["work_dir"].rglob("*")
                 if p.is_file() and "нет_папки" in str(p)]
        return {"folder_exists_after": missing.exists(), "files": found}

    await probe(g, "mc.export_file(path='не_существует/...')", export_to_missing,
                note="Прогон 2: ошибки нет, файла в ожидаемом месте нет — проверяем, создалась ли папка")

    # Параллельность: два экспорта через gather
    d = probe_dir(g, "gather")

    async def two_exports():
        t0 = time.perf_counter()
        df, _ = await asyncio.gather(mc.export_df(), mc.export_om_file(path=str(d)))
        return {"rows_df": len(df), "elapsed_parallel_s": round(time.perf_counter() - t0, 3)}

    await probe(g, "asyncio.gather(mc.export_df(), mc.export_om_file(path))", two_exports, out_dir=d)


async def g_mc_view():
    g = "mc_view"
    view = CTX.get("view")
    if view is None:
        return skip(g, "*", "у тестового мультикуба нет вью (или не задан OM_TEST_MC_VIEW)")
    await probe(g, "view.info()", lambda: view.info())
    for fmt in (DownloadFormat.CSV, DownloadFormat.XLSX):
        call = f"view.export_file(path, format=DownloadFormat.{fmt.name})"
        d = probe_dir(g, call)
        await probe(g, call, lambda d=d, f=fmt: view.export_file(path=str(d), format=f), out_dir=d)
    call = "view.export_om_file(path)"
    d = probe_dir(g, call)
    await probe(g, call, lambda: view.export_om_file(path=str(d)), out_dir=d)
    await probe(g, "view.export_df()", lambda: view.export_df())
    await probe(g, "view.export_df(sep=';')", lambda: view.export_df(sep=";"))


async def g_catalog():
    g = "catalog"
    cat = CTX.get("catalog")
    if cat is None:
        return skip(g, "*", "тестовый справочник не найден")
    await probe(g, "catalog.info()", lambda: cat.info())
    await probe(g, "catalog.get_elements()", lambda: cat.get_elements())
    await probe(g, "catalog.get_elements(row_count=3, column_count=2)",
                lambda: cat.get_elements(row_count=3, column_count=2))
    views = await probe(g, "catalog.get_views()", lambda: cat.get_views())
    if S["catalog_view"]:
        cv = await probe(g, f"catalog.get_view_by_name('{S['catalog_view']}')",
                         lambda: cat.get_view_by_name(S["catalog_view"]))
    else:
        cv = views[0] if views else None
    if cv is not None:
        CTX["catalog_view_obj"] = cv
        SURFACE["catalog_view"] = api_surface(cv)
    await probe(g, "catalog.get_view_by_name('__nope__')", lambda: cat.get_view_by_name("__sdk_probe_nope__"))
    call = "catalog.export_file(path)  # по умолчанию XLSX"
    d = probe_dir(g, call)
    await probe(g, call, lambda: cat.export_file(path=str(d)), out_dir=d)
    for fmt in (DownloadFormat.CSV, DownloadFormat.TXT, DownloadFormat.XLS):
        call = f"catalog.export_file(path, format=DownloadFormat.{fmt.name})"
        d = probe_dir(g, call)
        await probe(g, call, lambda d=d, f=fmt: cat.export_file(path=str(d), format=f), out_dir=d)
    for fmt in DownloadFormatOM:
        call = f"catalog.export_om_file(path, format=DownloadFormatOM.{fmt.name})"
        d = probe_dir(g, call)
        await probe(g, call, lambda d=d, f=fmt: cat.export_om_file(path=str(d), format=f), out_dir=d)
    await probe(g, "catalog.export_df()", lambda: cat.export_df())
    call = "catalog.export_views(path)"
    d = probe_dir(g, call)
    await probe(g, call, lambda: cat.export_views(path=str(d)), out_dir=d)


async def g_catalog_view():
    g = "catalog_view"
    cv = CTX.get("catalog_view_obj")
    if cv is None:
        return skip(g, "*", "у тестового справочника нет вью")
    await probe(g, "catalog_view.info()", lambda: cv.info())
    await probe(g, "catalog_view.get_elements()", lambda: cv.get_elements())
    call = "catalog_view.export_file(path, format=DownloadFormat.CSV)"
    d = probe_dir(g, call)
    await probe(g, call, lambda: cv.export_file(path=str(d), format=DownloadFormat.CSV), out_dir=d)
    call = "catalog_view.export_om_file(path)"
    d = probe_dir(g, call)
    await probe(g, call, lambda: cv.export_om_file(path=str(d)), out_dir=d)
    await probe(g, "catalog_view.export_df()", lambda: cv.export_df())


async def g_formulas_read():
    g = "formulas_read"
    m = CTX["model"]
    if not S["cube_id"]:
        return skip(g, "*", "не задан OM_TEST_CUBE_ID")
    await probe(g, f"model.get_formula({S['cube_id']})", lambda: m.get_formula(S["cube_id"]))
    fm = CubeFormulaManager.from_model(m)
    SURFACE["cube_formula_manager"] = api_surface(fm)
    await probe(g, f"CubeFormulaManager.get_formula({S['cube_id']})", lambda: fm.get_formula(S["cube_id"]))
    await probe(g, f"CubeFormulaManager.get_formula('{S['cube_id']}')  # id строкой",
                lambda: fm.get_formula(str(S["cube_id"])))
    await probe(g, "model.get_formula(999999999999)", expect_error=True, fn=lambda: m.get_formula(999999999999))


async def g_permissions_read():
    g = "permissions_read"
    m = CTX["model"]
    await probe(g, "model.get_multicube_permissions()", lambda: m.get_multicube_permissions())
    if CTX.get("mc_id"):
        await probe(g, f"model.get_multicube_permissions({CTX['mc_id']})",
                    lambda: m.get_multicube_permissions(CTX["mc_id"]))
    await probe(g, "model.get_version_permissions()", lambda: m.get_version_permissions())
    await probe(g, "model.get_list_permissions()", lambda: m.get_list_permissions())
    if CTX.get("catalog_id"):
        await probe(g, f"model.get_list_permissions({CTX['catalog_id']})",
                    lambda: m.get_list_permissions(CTX["catalog_id"]))
    await probe(g, "model.get_macro_permissions()", lambda: m.get_macro_permissions())
    await probe(g, "model.get_macros_permissions()  # алиас", lambda: m.get_macros_permissions())
    await probe(g, "model.get_user_permissions()", lambda: m.get_user_permissions())
    await probe(g, "model.get_user_permissions(ignore_missing_roles=False)",
                lambda: m.get_user_permissions(ignore_missing_roles=False))
    await probe(g, "model.get_userpermisions()  # алиас с опечаткой", lambda: m.get_userpermisions())
    exporter = PermissionGridExporter.from_model(m)
    SURFACE["permission_grid_exporter"] = api_surface(exporter)
    await probe(g, "PermissionGridExporter.export(137000000019)  # мультикубы",
                lambda: exporter.export(137000000019))
    await probe(g, "PermissionGridExporter.export(137000000019, row_page_size=10, column_page_size=5)",
                lambda: exporter.export(137000000019, row_page_size=10, column_page_size=5),
                note="Тот же результат при мелких страницах = пагинация работает")


async def g_uam_read():
    g = "uam_read"
    m = CTX["model"]
    uam = UamManager.from_model(m)
    SURFACE["uam_manager"] = api_surface(uam)
    await probe(g, "model.get_list_uam()", lambda: m.get_list_uam())
    ids = await probe(g, "uam.get_enabled_list_ids()", lambda: uam.get_enabled_list_ids())
    await probe(g, "uam.get_uam_grid_mapping()", lambda: uam.get_uam_grid_mapping())
    list_id = S["uam_list_id"] or (ids[0] if ids else None)
    if list_id:
        await probe(g, f"model.get_uam({list_id})", lambda: m.get_uam(list_id))
        await probe(g, f"uam.get_uam({list_id})", lambda: uam.get_uam(list_id))
    else:
        skip(g, "get_uam", "нет справочников с UAM и не задан OM_TEST_UAM_LIST_ID")
    if CTX.get("catalog_id") and CTX["catalog_id"] not in (ids or []):
        await probe(g, f"uam.get_uam({CTX['catalog_id']})  # справочник без UAM", expect_error=True,
                    fn=lambda: uam.get_uam(CTX["catalog_id"]))


async def g_users():
    g = "users"
    m = CTX["model"]
    await probe(g, "model.get_model_user_list()", lambda: m.get_model_user_list())
    if S["model_user_id"]:
        await probe(g, f"model.get_model_user_info({S['model_user_id']})",
                    lambda: m.get_model_user_info(S["model_user_id"]))
    else:
        skip(g, "model.get_model_user_info", "не задан OM_TEST_MODEL_USER_ID (возьмите из get_model_user_list)")
    await probe(g, "model.get_ws_user_list()", lambda: m.get_ws_user_list(),
                note="Без прав на пользователей WS — 'Grid access denied'")
    if S["ws_user_id"]:
        await probe(g, f"model.get_ws_user_info({S['ws_user_id']})", lambda: m.get_ws_user_info(S["ws_user_id"]))
    else:
        skip(g, "model.get_ws_user_info", "не задан OM_TEST_WS_USER_ID")


async def g_scripts():
    g = "scripts"
    m = CTX["model"]
    ids = await probe(g, "model.get_script_list()", lambda: m.get_script_list())
    script_id = S["script_id"] or (ids[0] if ids else None)
    if script_id:
        await probe(g, f"model.get_script_info({script_id})", lambda: m.get_script_info(script_id))
    await probe(g, "model.get_script_info(999999999999)", lambda: m.get_script_info(999999999999),
                note="По документации — None")
    now = datetime.now(timezone.utc)
    await probe(g, "model.get_script_history()  # последние 24 ч", lambda: m.get_script_history())
    await probe(g, "model.get_script_history(date_from=now-30d, date_to=now)",
                lambda: m.get_script_history(date_from=now - timedelta(days=30), date_to=now))
    await probe(g, "model.get_script_history(status=SUCCESS, run_type=API, show_archive=True)",
                lambda: m.get_script_history(status=ScriptStatus.SUCCESS, run_type=ScriptRunType.API,
                                             show_archive=True))
    if script_id:
        await probe(g, f"model.get_script_history(identifiers=[{script_id}], date_from=now-30d)",
                    lambda: m.get_script_history(identifiers=[script_id], date_from=now - timedelta(days=30)))
    await probe(g, "model.get_script_chain_history()", lambda: m.get_script_chain_history())
    await probe(g, "model.get_script_chain_history(date_from=now-30d)",
                lambda: m.get_script_chain_history(date_from=now - timedelta(days=30)))


async def g_dashboards():
    g = "dashboards"
    m = CTX["model"]
    await probe(g, "model.get_dashboard_multicube_links()", lambda: m.get_dashboard_multicube_links())
    await probe(g, "model.export_dashboard_multicube_links_grid()", lambda: m.export_dashboard_multicube_links_grid())
    await probe(g, "model.get_dashboard_multicube_links(ignore_errors=False)",
                lambda: m.get_dashboard_multicube_links(ignore_errors=False),
                note="Прогон 2: с ignore_errors=True вернулся пустой DataFrame — не прячет ли он ошибки")
    if S["dashboard_id"]:
        await probe(g, f"model.get_dashboard_multicubes({S['dashboard_id']})",
                    lambda: m.get_dashboard_multicubes(S["dashboard_id"]))
        await probe(g, f"model.get_dashboard_multicube_links(dashboard_ids=[{S['dashboard_id']}])",
                    lambda: m.get_dashboard_multicube_links(dashboard_ids=[S["dashboard_id"]]))
    else:
        skip(g, "model.get_dashboard_multicubes", "не задан OM_TEST_DASHBOARD_ID (возьмите из DASHBOARDS_GRID)")
    await probe(g, "model.get_dashboard_multicubes(999999999999)", lambda: m.get_dashboard_multicubes(999999999999),
                note="Несуществующий дашборд")


async def g_optimizer():
    g = "optimizer"
    m = CTX["model"]
    await probe(g, "model.get_optimizer_requests_grid()", lambda: m.get_optimizer_requests_grid())
    await probe(g, "model.get_optimizer_requests_grid(entity_default_view_entity_long_id=137000000051)",
                lambda: m.get_optimizer_requests_grid(entity_default_view_entity_long_id=137000000051))
    await probe(g, "model.get_optimizer_requests()  # алиас", lambda: m.get_optimizer_requests())
    if not S["optimization_id"]:
        return skip(g, "get_optimizer_request_fields / export_lp_mapping", "не задан OM_TEST_OPTIMIZATION_ID")
    oid = S["optimization_id"]
    await probe(g, f"model.get_optimizer_request_fields({oid})", lambda: m.get_optimizer_request_fields(oid))
    await probe(g, f"model.get_optimizer_request_fields({oid}, entity_default_view_entity_long_id=137000000051)",
                lambda: m.get_optimizer_request_fields(oid, entity_default_view_entity_long_id=137000000051))
    await probe(g, f"model.action_info({oid})", lambda: m.action_info(oid))
    call = f"model.export_lp_mapping({oid}, path)"
    d = probe_dir(g, call)
    name = await probe(g, call, lambda: m.export_lp_mapping(oid, str(d)), out_dir=d,
                       note="Возвращает имя файла, не путь (по коду SDK); timeoutInMinutes=5 зашит")
    if name:
        CTX["lp_zip"] = d / Path(str(name)).name


async def g_backups():
    g = "backups"
    m = CTX["model"]
    await probe(g, "model.get_backup_list()", lambda: m.get_backup_list())
    if S["backup_row_id"]:
        await probe(g, f"model.get_backup_info({S['backup_row_id']})", lambda: m.get_backup_info(S["backup_row_id"]))
    else:
        skip(g, "model.get_backup_info", "не задан OM_TEST_BACKUP_ROW_ID (long id строки из get_backup_list)")
    if S["backup_id"] and S["download_backup"]:
        data = await probe(g, f"model.get_backup('{S['backup_id']}')", lambda: m.get_backup(S["backup_id"]))
        if data:
            out = S["work_dir"] / "export" / g / f"backup_{slug(S['backup_id'])}.zip"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            log.info("    бэкап сохранён: %s (%s байт)", out, len(data))
    else:
        skip(g, "model.get_backup", "нужны OM_TEST_BACKUP_ID и OM_TEST_DOWNLOAD_BACKUP=1")
    await probe(g, "model.get_backup(999999)", expect_error=True, fn=lambda: m.get_backup(999999))


# ----------------------------- пишущие группы ------------------------------
async def g_catalog_write():
    """Полный жизненный цикл на временных справочниках. В конце всё удаляется."""
    g = "catalog_write"
    m = CTX["model"]
    stamp = datetime.now().strftime("%H%M%S")
    name1, name2, name3 = f"SDK_PROBE_{stamp}_A", f"SDK_PROBE_{stamp}_B", f"SDK_PROBE_{stamp}_C"
    created = []
    try:
        await probe(g, f"model.create_catalogs(names=['{name1}', '{name2}'])",
                    lambda: m.create_catalogs(names=[name1, name2]), kind="write")
        c1 = await probe(g, f"model.get_catalog_by_name('{name1}')", lambda: m.get_catalog_by_name(name1), kind="write")
        c2 = await probe(g, f"model.get_catalog_by_name('{name2}')", lambda: m.get_catalog_by_name(name2), kind="write")
        if c1 is None or c2 is None:
            return skip(g, "остальные шаги", "временные справочники не создались или не нашлись", kind="write")
        id1, id2 = await obj_id(c1), await obj_id(c2)
        created += [id1, id2]
        info1 = await c1.info()
        if CTX.get("Position") is not None:
            await probe(g, f"model.create_catalogs(names=['{name3}'], position=AFTER, relative_to='{name1}')",
                        lambda: m.create_catalogs(names=[name3], position=CTX["Position"].AFTER, relative_to=name1),
                        kind="write")
        else:
            await probe(g, f"catalog.create(model_id, names=['{name3}'])",
                        lambda: c1.create(model_id=info1.model_id, names=[name3]), kind="write")
        c3 = await probe(g, f"model.get_catalog_by_name('{name3}')", lambda: m.get_catalog_by_name(name3), kind="write")
        if c3 is not None:
            created.append(await obj_id(c3))

        # Элементы
        await probe(g, "c1.add_elements(names=['A', 'B', 'C'])", lambda: c1.add_elements(names=["A", "B", "C"]),
                    kind="write")
        await probe(g, "c1.add_elements(count=2)  # имена генерирует платформа", lambda: c1.add_elements(count=2),
                    kind="write")
        if CTX.get("Position") is not None:
            await probe(g, "c1.add_elements(names=['A0'], position=BEFORE, relative_to='A')",
                        lambda: c1.add_elements(names=["A0"], position=CTX["Position"].BEFORE, relative_to="A"),
                        kind="write")
        els = await probe(g, "c1.get_elements()  # после добавления", lambda: c1.get_elements(), kind="write")
        if els:
            columns, rows = els
            ids = element_ids_by_name(columns, rows, ["A", "C"])
            log.info("    колонки справочника: %s; найденные id: %s", columns, ids)
            if "A" in ids:
                await probe(g, f"c1.rename_elements(names=['A_renamed'], elements_ids=[{ids['A']}])",
                            lambda: c1.rename_elements(names=["A_renamed"], elements_ids=[ids["A"]]), kind="write")
            else:
                skip(g, "c1.rename_elements", f"не удалось найти id элемента по колонкам {columns}", kind="write")
            if "C" in ids:
                await probe(g, f"c1.delete_elements(ids=[{ids['C']}])",
                            lambda: c1.delete_elements(ids=[ids["C"]]), kind="write")
        await probe(g, "c1.delete_elements(names=['B'])", lambda: c1.delete_elements(names=["B"]), kind="write")
        await probe(g, "c1.delete_elements(names=['__nope__'])", lambda: c1.delete_elements(names=["__nope__"]),
                    kind="write", note="Удаление несуществующего: тихо или ошибка")
        await probe(g, "c1.get_elements()  # после удаления/переименования", lambda: c1.get_elements(), kind="write")

        # Справочник → справочник (пока в c1 есть элементы)
        await probe(g, f"c1.export_catalog(destination_catalog_id={id2})", lambda: c1.export_catalog(id2), kind="write")
        await probe(g, "c2.get_elements()  # после export_catalog", lambda: c2.get_elements(), kind="write")
        await probe(g, "c2.clear()", lambda: c2.clear(), kind="write")
        await probe(g, f"c2.import_catalog(source_catalog_id={id1})", lambda: c2.import_catalog(id1), kind="write")
        await probe(g, "c2.get_elements()  # после import_catalog", lambda: c2.get_elements(), kind="write")

        # Круговой тест импорта. Прогон 3: импорт собственного CSV-экспорта после clear
        # дал 0 элементов без ошибки. Пробуем три формата файла.
        csv_dir = probe_dir(g, "roundtrip_csv")
        xlsx_dir = probe_dir(g, "roundtrip_xlsx")
        await probe(g, "c1.export_file(path, CSV)  # для кругового теста",
                    lambda: c1.export_file(path=str(csv_dir), format=DownloadFormat.CSV), out_dir=csv_dir, kind="write")
        await probe(g, "c1.export_file(path, XLSX)  # для кругового теста",
                    lambda: c1.export_file(path=str(xlsx_dir), format=DownloadFormat.XLSX), out_dir=xlsx_dir,
                    kind="write")
        simple_csv = S["work_dir"] / "import" / "catalog_simple.csv"
        simple_csv.parent.mkdir(parents=True, exist_ok=True)
        simple_csv.write_text("Item Name\nP1\nP2\nP3\n", encoding="utf-8")
        variants = [("свой CSV-экспорт", sorted(csv_dir.glob("*.csv"))),
                    ("свой XLSX-экспорт", sorted(xlsx_dir.glob("*.xlsx"))),
                    ("простой CSV: заголовок Item Name + 3 имени", [simple_csv])]
        for label, files in variants:
            if not files:
                skip(g, f"c1.import_file({label})", "файл экспорта не появился", kind="write")
                continue
            src = files[-1]
            await probe(g, f"c1.clear()  # перед импортом: {label}", lambda: c1.clear(), kind="write")
            await probe(g, f"c1.import_file('{src.name}')  # {label}", lambda s=src: c1.import_file(str(s)),
                        kind="write")
            await probe(g, f"c1.get_elements()  # после импорта: {label}", lambda: c1.get_elements(), kind="write")
        await probe(g, "c1.import_file(простой CSV повторно, encoding=UTF)  # дубли или upsert",
                    lambda: c1.import_file(str(simple_csv), encoding=EncodingTypeImport.UTF), kind="write")
        await probe(g, "c1.get_elements()  # после повторного импорта", lambda: c1.get_elements(), kind="write")
        await probe(g, "c1.import_file('нет_файла.csv')", expect_error=True, kind="write",
                    fn=lambda: c1.import_file(str(S["work_dir"] / "нет_файла.csv")))

        # Переименование
        await probe(g, f"c1.rename(['{name1}_R'])", lambda: c1.rename([f"{name1}_R"]), kind="write")
        await probe(g, f"model.rename_catalogs(['{name2}_R'], [{id2}])",
                    lambda: m.rename_catalogs([f"{name2}_R"], [id2]), kind="write")
        await probe(g, "model.get_catalog(id1).info()  # имя после rename",
                    lambda: _catalog_info(m, id1), kind="write")
        await probe(g, "c1.info()  # старый объект после rename — кэш или свежие данные",
                    lambda: c1.info(), kind="write")
        await probe(g, "model.delete_catalogs([999999999999])", expect_error=True, kind="write",
                    fn=lambda: m.delete_catalogs([999999999999]))
    finally:
        # Уборка: один справочник через catalog.delete(), остальные через model.delete_catalogs
        if created:
            first, rest = created[0], created[1:]
            await probe(g, f"catalog.delete()  # уборка {first}", lambda: _delete_one(m, first), kind="write")
            if rest:
                await probe(g, f"model.delete_catalogs({rest})  # уборка", lambda: m.delete_catalogs(rest), kind="write")


async def _catalog_info(m, cid):
    cat = await m.get_catalog(cid)
    return await cat.info()


async def _delete_one(m, cid):
    cat = await m.get_catalog(cid)
    return await cat.delete()


async def g_multicube_write():
    """Круговые тесты на мультикубе-приёмнике. Данные сохраняются и возвращаются в конце."""
    g = "multicube_write"
    m = CTX["model"]
    if not S["target_mc_name"]:
        return skip(g, "*", "не задан OM_TEST_TARGET_MULTICUBE", kind="write")
    dst = await probe(g, f"model.get_multicube_by_name('{S['target_mc_name']}')",
                      lambda: m.get_multicube_by_name(S["target_mc_name"]), kind="write")
    if dst is None:
        return skip(g, "*", "мультикуб-приёмник не найден", kind="write")

    # 1. Сохраняем текущие данные приёмника (формат ОМ и обычный CSV)
    backup_dir = probe_dir(g, "backup_om")
    await probe(g, "dst.export_om_file(path)  # сохранить данные", lambda: dst.export_om_file(path=str(backup_dir)),
                out_dir=backup_dir, kind="write")
    om_files = sorted(backup_dir.glob("*.csv"))
    std_dir = probe_dir(g, "backup_std")
    await probe(g, "dst.export_file(path, CSV)  # сохранить данные", lambda: dst.export_file(path=str(std_dir)),
                out_dir=std_dir, kind="write")
    std_files = sorted(std_dir.glob("*.csv"))
    if not om_files:
        return skip(g, "остальные шаги", "не удалось сохранить данные приёмника — дальше не трогаем", kind="write")
    om_backup = om_files[-1]

    try:
        await probe(g, "dst.export_df()  # строк до", lambda: dst.export_df(), kind="write")
        await probe(g, "dst.clear()", lambda: dst.clear(), kind="write")
        await probe(g, "dst.export_df()  # строк после clear", lambda: dst.export_df(), kind="write")
        await probe(g, f"dst.import_om_file('{om_backup.name}')", lambda: dst.import_om_file(str(om_backup)),
                    kind="write")
        await probe(g, "dst.export_df()  # строк после import_om_file", lambda: dst.export_df(), kind="write")
        if std_files:
            await probe(g, f"dst.import_file('{std_files[-1].name}')  # обычный CSV грида",
                        lambda: dst.import_file(str(std_files[-1])), kind="write")
        await probe(g, "dst.import_om_file(файл не ОМ-формата)", kind="write",
                    fn=lambda: dst.import_om_file(str(std_files[-1])) if std_files else None,
                    note="Как SDK реагирует на неверный формат: ошибка или тихий частичный импорт")

        # Вью приёмника
        views = await dst.get_views()
        if views:
            v = views[0]
            vdir = probe_dir(g, "view_om")
            await probe(g, "dst_view.export_om_file(path)", lambda: v.export_om_file(path=str(vdir)), out_dir=vdir,
                        kind="write")
            vfiles = sorted(vdir.glob("*.csv"))
            if vfiles:
                await probe(g, "dst_view.import_om_file(тот же файл)", lambda: v.import_om_file(str(vfiles[-1])),
                            kind="write")
            vdir2 = probe_dir(g, "view_std")
            await probe(g, "dst_view.export_file(path)", lambda: v.export_file(path=str(vdir2)), out_dir=vdir2,
                        kind="write")
            vfiles2 = sorted(vdir2.glob("*.csv"))
            if vfiles2:
                await probe(g, "dst_view.import_file(тот же файл)", lambda: v.import_file(str(vfiles2[-1])),
                            kind="write")

        # Изменение одной ячейки
        if S["cell_value"] and S["cell_cube_id"] and S["cell_dim_id"]:
            await probe(g, f"dst.change_cell_value({S['cell_value']!r}, cube_id={S['cell_cube_id']}, "
                           f"dimension_id={S['cell_dim_id']})",
                        lambda: dst.change_cell_value(S["cell_value"], S["cell_cube_id"], S["cell_dim_id"]),
                        kind="write", note="ПРОВЕРИТЬ: смысл dimension_id в сигнатуре не документирован")
        else:
            skip(g, "dst.change_cell_value", "не заданы OM_TEST_CELL_VALUE / _CUBE_ID / _DIMENSION_ID", kind="write")

        # Копирование мультикуб → мультикуб
        if S["mc_copy"] and CTX.get("mc_id"):
            dst_id = await obj_id(dst)
            await probe(g, f"dst.import_from_multicube({CTX['mc_id']})",
                        lambda: dst.import_from_multicube(CTX["mc_id"]), kind="write")
            await probe(g, "dst.export_df()  # после import_from_multicube", lambda: dst.export_df(), kind="write")
            await probe(g, f"src.export_to_multicube({dst_id})",
                        lambda: CTX["mc"].export_to_multicube(dst_id), kind="write")
        else:
            skip(g, "import_from_multicube / export_to_multicube", "OM_TEST_MC_COPY=0", kind="write")
    finally:
        # Возвращаем исходные данные приёмника
        await probe(g, "dst.clear()  # восстановление", lambda: dst.clear(), kind="write")
        await probe(g, f"dst.import_om_file('{om_backup.name}')  # восстановление",
                    lambda: dst.import_om_file(str(om_backup)), kind="write")
        await probe(g, "dst.export_df(sep=';')  # проверка восстановления", lambda: dst.export_df(sep=";"),
                    kind="write")
        log.info("Данные приёмника сохранены также в %s", om_backup)


async def g_formulas_write():
    g = "formulas_write"
    m = CTX["model"]
    cube = S["formula_cube_id"]
    if not cube:
        return skip(g, "*", "не задан OM_TEST_FORMULA_CUBE_ID", kind="write")
    fm = CubeFormulaManager.from_model(m)
    original = await probe(g, f"get_formula({cube})  # исходная", lambda: fm.get_formula(cube), kind="write")
    if original is None and not S["skip_write"]:
        return skip(g, "change_formula", "не удалось прочитать исходную формулу", kind="write")
    await probe(g, f"model.change_formula({cube}, та же формула)  # ждём 0 (no-op)",
                lambda: m.change_formula(cube, original), kind="write")
    if not S["formula_new"]:
        return skip(g, "change_formula(новая)", "не задан OM_TEST_FORMULA_NEW", kind="write")
    try:
        await probe(g, f"CubeFormulaManager.change_formula({cube}, OM_TEST_FORMULA_NEW)  # ждём 1",
                    lambda: fm.change_formula(cube, S["formula_new"]), kind="write")
        await probe(g, f"get_formula({cube})  # после изменения", lambda: fm.get_formula(cube), kind="write")
        await probe(g, f"change_formula({cube}, 'синтаксическая ошибка (((')",
                    lambda: fm.change_formula(cube, "SDK_PROBE((("), kind="write", expect_error=True)
    finally:
        await probe(g, f"change_formula({cube}, исходная)  # откат", lambda: fm.change_formula(cube, original),
                    kind="write")


async def g_permissions_write():
    g = "permissions_write"
    m = CTX["model"]
    if not S["role_id"] or not S["perm_restore"]:
        return skip(g, "*", "нужны OM_TEST_ROLE_ID и OM_TEST_PERMISSION_RESTORE", kind="write")
    pm = PermissionChangeManager.from_model(m)
    SURFACE["permission_change_manager"] = api_surface(pm)
    role, new, old = S["role_id"], S["perm_set"], S["perm_restore"]
    if not str(role).isdigit():
        # В .env указано имя роли — ищем её longId в ролевых колонках матрицы прав
        role_name = role
        data = await PermissionGridExporter.from_model(m).export(137000000019)
        ids = {getattr(c, "role", None): getattr(c, "role_id", None) for c in data.role_columns}
        log.info("    роли в матрице прав: %s", ids)
        if role_name not in ids:
            return skip(g, "*", f"роль «{role_name}» не найдена среди {list(ids)}", kind="write")
        role = ids[role_name]
        if CTX.get("mc_id"):
            await probe(g, f"change_multicube_permission({CTX['mc_id']}, '{role_name}', '{new}')  # роль по имени",
                        lambda: pm.change_multicube_permission(CTX["mc_id"], role_name, new), kind="write",
                        note="Принимает ли SDK имя роли вместо longId")
    targets = []
    if CTX.get("mc_id"):
        targets.append(("multicube", CTX["mc_id"], pm.change_multicube_permission, m.change_multicube_permission))
    if CTX.get("catalog_id"):
        targets.append(("list", CTX["catalog_id"], pm.change_list_permission, m.change_list_permission))
    if S["version_id"]:
        targets.append(("version", S["version_id"], pm.change_version_permission, m.change_version_permission))
    if S["script_id"]:
        targets.append(("macros", S["script_id"], pm.change_macros_permission, m.change_macros_permission))
    for kind_name, obj, via_manager, via_model in targets:
        await probe(g, f"permissions.change_{kind_name}_permission({obj}, {role}, '{new}')",
                    lambda f=via_manager, o=obj: f(o, role, new), kind="write")
        await probe(g, f"permissions.change_{kind_name}_permission({obj}, {role}, '{new.upper()}')  # регистр, повтор",
                    lambda f=via_manager, o=obj: f(o, role, new.upper()), kind="write",
                    note="Тот же уровень повторно: True/False?")
        if kind_name == "macros":
            # Прогон 3: 'none' для макроса отвергается, хотя есть в тексте ошибки. Откат — через bool.
            await probe(g, f"permissions.change_macros_permission({obj}, {role}, 'none')",
                        lambda o=obj: pm.change_macros_permission(o, role, "none"), kind="write",
                        note="Прогон 3: отвергнуто через model.*; проверяем менеджер")
            await probe(g, f"permissions.change_macros_permission({obj}, {role}, 'false')",
                        lambda o=obj: pm.change_macros_permission(o, role, "false"), kind="write")
            restore_value = old.lower() not in ("none", "false", "")
            await probe(g, f"model.change_macros_permission({obj}, {role}, {restore_value})  # откат через bool",
                        lambda f=via_model, o=obj: f(o, role, restore_value), kind="write")
        else:
            await probe(g, f"model.change_{kind_name}_permission({obj}, {role}, '{old}')  # откат",
                        lambda f=via_model, o=obj: f(o, role, old), kind="write")
    if CTX.get("mc_id"):
        await probe(g, f"change_multicube_permission({CTX['mc_id']}, {role}, 'admin')  # неверное значение",
                    expect_error=True, kind="write",
                    fn=lambda: pm.change_multicube_permission(CTX["mc_id"], role, "admin"))
        await probe(g, f"change_multicube_permission({CTX['mc_id']}, 999999999999, 'read')  # нет роли",
                    expect_error=True, kind="write",
                    fn=lambda: pm.change_multicube_permission(CTX["mc_id"], 999999999999, "read"))
    if S["script_id"]:
        await probe(g, f"model.change_macro_permission({S['script_id']}, {role}, True)  # bool вместо строки",
                    lambda: m.change_macro_permission(S["script_id"], role, True), kind="write")
        await probe(g, f"model.change_mscros_permission({S['script_id']}, {role}, False)  # алиас с опечаткой, откат",
                    lambda: m.change_mscros_permission(S["script_id"], role, False), kind="write")
        await probe(g, f"model.get_macro_permissions({S['script_id']})  # итог после отката",
                    lambda: m.get_macro_permissions(S["script_id"]), kind="write")


async def g_uam_write():
    g = "uam_write"
    m = CTX["model"]
    need = (S["uam_list_id"], S["uam_item_id"], S["uam_user"], S["uam_restore"])
    if not all(need):
        return skip(g, "*", "нужны OM_TEST_UAM_LIST_ID, _ITEM_ID, _USER, _RESTORE", kind="write")
    uam = UamManager.from_model(m)
    lid, item, user = S["uam_list_id"], S["uam_item_id"], S["uam_user"]
    await probe(g, f"uam.change_uam({lid}, {item}, user, '{S['uam_set']}')",
                lambda: uam.change_uam(lid, item, user, S["uam_set"]), kind="write")
    await probe(g, f"model.get_uam({lid})  # после изменения", lambda: m.get_uam(lid), kind="write")
    await probe(g, f"model.change_uam({lid}, {item}, user, '{S['uam_restore']}')  # откат",
                lambda: m.change_uam(lid, item, user, S["uam_restore"]), kind="write")
    await probe(g, f"uam.change_uam({lid}, {item}, '__nobody__', 'read')  # нет пользователя",
                expect_error=True, kind="write", fn=lambda: uam.change_uam(lid, item, "__sdk_probe_nobody__", "read"))


async def g_scripts_write():
    g = "scripts_write"
    m = CTX["model"]
    if not S["script_id"]:
        return skip(g, "*", "не задан OM_TEST_SCRIPT_ID", kind="write")
    await probe(g, f"model.run_script({S['script_id']})", lambda: m.run_script(S["script_id"]), kind="write",
                note="Синхронно ждёт окончания или только ставит в очередь — смотрим время и историю")
    # identifiers сервер отвергает (прогон 2), поэтому смотрим всю историю за последний час
    await probe(g, "model.get_script_history(date_from=now-1h)  # сразу после запуска",
                lambda: m.get_script_history(date_from=datetime.now(timezone.utc) - timedelta(hours=1)),
                kind="write")
    await probe(g, "model.get_script_chain_history(date_from=now-1h)",
                lambda: m.get_script_chain_history(date_from=datetime.now(timezone.utc) - timedelta(hours=1)),
                kind="write")
    await probe(g, "model.run_script(999999999999)", expect_error=True, kind="write",
                fn=lambda: m.run_script(999999999999))


async def g_optimizer_write():
    g = "optimizer_write"
    m = CTX["model"]
    if not S["solution_zip"]:
        return skip(g, "import_solution_mapping", "не задан OM_TEST_SOLUTION_ZIP", kind="write")
    zip_path = Path(S["solution_zip"]).resolve()
    await probe(g, f"model.import_solution_mapping('{zip_path.name}')",
                lambda: m.import_solution_mapping(str(zip_path)), kind="write",
                note="Формат CSV решения не описан — ответ сервера здесь главный результат")
    if CTX.get("lp_zip") and CTX["lp_zip"].exists():
        await probe(g, "model.import_solution_mapping(zip из export_lp_mapping, без решения)", expect_error=True,
                    kind="write", fn=lambda: m.import_solution_mapping(str(CTX["lp_zip"])))


async def g_admin():
    """Управление моделью. Модель будет недоступна пользователям на время stop/start."""
    g = "admin"
    m, ws = CTX["model"], CTX["ws"]
    await probe(g, "model.recalculate()", lambda: m.recalculate(), kind="admin")
    await probe(g, "model.backup()", lambda: m.backup(), kind="admin")
    await probe(g, "model.get_backup_list()  # после backup()", lambda: m.get_backup_list(), kind="admin")
    try:
        await probe(g, "model.stop()", lambda: m.stop(), kind="admin")
        await probe(g, "model.info().is_offline  # тот же объект после stop", lambda: m.info().is_offline, kind="admin",
                    note="Кэшируется ли info()")
        await probe(g, "ws.get_model(id).info().is_offline  # свежий объект", lambda: _fresh_offline(ws, m),
                    kind="admin")
        await probe(g, "model.stop()  # повторно, ждём False", lambda: m.stop(), kind="admin")
        await probe(g, "mc.export_df() на остановленной модели", kind="admin",
                    fn=lambda: CTX["mc"].export_df() if CTX.get("mc") else None)
    finally:
        await probe(g, "model.start()", lambda: m.start(), kind="admin")
    await probe(g, "model.start()  # повторно, на онлайн-модели", lambda: m.start(), kind="admin")
    if S["restore"]:
        await probe(g, "model.restore()  # ОТКАТ МОДЕЛИ", lambda: m.restore(), kind="admin")
    else:
        skip(g, "model.restore()", "OM_TEST_RESTORE=0", kind="admin")


async def _fresh_offline(ws, m):
    fresh = await ws.get_model(id=m.info().model_id)
    return fresh.info().is_offline


GROUP_FUNCS = {
    "imports": g_imports, "connect": g_connect, "workspace": g_workspace, "model": g_model,
    "system_grids": g_system_grids, "multicube": g_multicube, "mc_view": g_mc_view,
    "catalog": g_catalog, "catalog_view": g_catalog_view, "formulas_read": g_formulas_read,
    "permissions_read": g_permissions_read, "uam_read": g_uam_read, "users": g_users,
    "scripts": g_scripts, "dashboards": g_dashboards, "optimizer": g_optimizer, "backups": g_backups,
    "catalog_write": g_catalog_write, "multicube_write": g_multicube_write,
    "formulas_write": g_formulas_write, "permissions_write": g_permissions_write,
    "uam_write": g_uam_write, "scripts_write": g_scripts_write, "optimizer_write": g_optimizer_write,
    "admin": g_admin,
}
WRITE_GROUPS = {"catalog_write", "multicube_write", "formulas_write", "permissions_write",
                "uam_write", "scripts_write", "optimizer_write"}


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
async def main() -> None:
    S.update(load_settings())
    S["work_dir"].mkdir(parents=True, exist_ok=True)
    CTX["started_at"] = datetime.now().isoformat(timespec="seconds")
    selected = S["groups"] or ALL_GROUPS

    log.info("Стенд os_ws_sdk: %s", S["ws_url"])
    log.info("Рабочая папка: %s", S["work_dir"])
    log.info("Запись в модель: %s | админка: %s | restore: %s",
             "ВЫКЛ (OM_SKIP_WRITE=1)" if S["skip_write"] else "ВКЛ",
             "ВКЛ" if S["admin"] else "выкл", "ВКЛ" if S["restore"] else "выкл")
    log.info("Группы: %s", ", ".join(selected))

    t_all = time.perf_counter()
    # Подключение и модель нужны всем группам, поэтому выполняются всегда
    await g_connect()
    await resolve_model()
    await resolve_objects()
    for name in ALL_GROUPS:
        if name in ("connect",) or name not in selected:
            continue
        if name in WRITE_GROUPS and S["skip_write"]:
            skip(name, "*", "OM_SKIP_WRITE=1", kind="write")
            continue
        if name == "admin" and not S["admin"]:
            skip(name, "*", "OM_TEST_ADMIN=0", kind="admin")
            continue
        log.info("=" * 70)
        log.info("Группа: %s", name)
        t0 = time.perf_counter()
        try:
            await GROUP_FUNCS[name]()
        except Exception:
            # группа упала вне probe (ошибка в самом стенде) — записываем и идём дальше
            log.exception("Группа %s прервана", name)
            RESULTS.append({"group": name, "call": "<группа прервана>", "kind": "-", "status": "error",
                            "expect_error": False, "note": "",
                            "error": {"type": "stand", "message": short(traceback.format_exc(), 1500)}})
        log.info("Группа %s: %.1f с", name, time.perf_counter() - t0)
        write_report()

    write_report()
    counts = Counter(r["status"] for r in RESULTS)
    log.info("=" * 70)
    log.info("Готово за %.1f с. Вызовов: %s (%s)", time.perf_counter() - t_all, len(RESULTS),
             ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    log.info("Отчёт: %s и report.md", S["work_dir"] / "report.json")


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except SystemExit:
        raise
    except Exception:
        log.exception("Стенд упал с ошибкой")
        if S:
            write_report()
        sys.exit(1)
