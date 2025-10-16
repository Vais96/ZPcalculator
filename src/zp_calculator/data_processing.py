from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ReconciliationRow",
    "load_reconciliation_file",
    "aggregate_by_partner_and_buyer",
    "aggregate_overall",
    "load_expenses_file",
    "aggregate_expenses_by_buyer",
    "aggregate_expenses_totals",
    "load_spend_sheet",
    "load_spend_workbook",
    "aggregate_spend_by_buyer",
    "aggregate_spend_by_sheet",
]

SPACE_PATTERN = re.compile(r"[\s\u00A0\u202F]")

SUMMARY_ROW_PREFIXES = ("Всего", "Итого", "Итог", "Total", "Subtotal", "Grandtotal")

METRIC_HEADER_KEYS = {
    "ftd",
    "ftdcount",
    "payout",
    "commissiontype",
    "commisiontype",
    "totals",
    "total",
    "итого",
    "итог",
    "всего",
}

HEADER_ALIASES = {
    "buyer": ["buyer", "байер", "имя"],
    "commission": ["commissiontype", "commisiontype", "типкомиссии", "commission"],
    "ftd": [
        "суммапополюftdcount",
        "ftdcount",
        "ftd",
        "депозитов",
        "количествoftd",
        "count",
    ],
    "payout": [
        "суммапополюpayout",
        "payout",
        "revshare",
        "revenue",
        "выплаты",
    ],
}

SPEND_HEADER_ALIASES = {
    "buyer": ["buyer", "байер", "байер"],
    "account_label": [
        "accountname",
        "имяаккаунта",
        "аккаунт",
        "account",
        "карта",
        "имя",
    ],
    "account_id": [
        "accountid",
        "idаккаунта",
        "idакк",
        "id",
        "idak",
        "idan",
        "idakkaunta",
        "idakkaunt",
    ],
    "spend": ["spend", "spending", "спенд", "спендинг", "расход"],
    "total": ["total", "итого", "totals"],
}

CURRENCY_SYMBOLS = {
    "$": "USD",
    "€": "EUR",
    "₽": "RUB",
    "₴": "UAH",
    "£": "GBP",
    "₸": "KZT",
    "¥": "CNY",
}

BUYER_OVERRIDES = {
    "банк": "Банк",
}

EXCEL_EXTENSIONS = {"xlsx", "xlsm", "xls"}


def _derive_spend_currency(sheet_name: str) -> str:
    normalized = str(sheet_name or "").strip().upper()
    if "DTD" in normalized:
        return "EUR"
    return "USD"


@dataclass
class ReconciliationRow:
    buyer: str
    commission_type: str
    ftd_count: int
    payout: float
    partner_program: str
    source_file: str
    currency: Optional[str] = None


def _read_text(file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase) -> str:
    """Return UTF-8 text from a file-like object."""
    position: Optional[int] = None
    if hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
        try:
            position = file_obj.tell()
            file_obj.seek(0)
        except (OSError, io.UnsupportedOperation):
            position = None
    raw = file_obj.read()
    if isinstance(raw, bytes):
        text = raw.decode("utf-8-sig")
    else:
        text = raw
    if position is not None:
        try:
            file_obj.seek(position)
        except (OSError, io.UnsupportedOperation):
            pass
    return text


def _get_file_extension(file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase) -> str:
    name = getattr(file_obj, "name", "") or ""
    if not isinstance(name, str):
        return ""
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[-1].lower()


def _remember_position(file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase) -> Optional[int]:
    if hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
        try:
            return file_obj.tell()
        except (OSError, io.UnsupportedOperation):
            return None
    return None


def _restore_position(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
    position: Optional[int],
) -> None:
    if position is None:
        return
    try:
        file_obj.seek(position)
    except (OSError, io.UnsupportedOperation):
        pass


def _read_first_sheet_rows(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
    sheet_index: int = 0,
) -> Optional[List[List[str]]]:
    position = _remember_position(file_obj)
    try:
        frame = pd.read_excel(file_obj, sheet_name=sheet_index, dtype=str, header=None)
    except Exception:
        _restore_position(file_obj, position)
        return None

    rows = frame.fillna("").astype(str).values.tolist()
    _restore_position(file_obj, position)
    return rows


def _read_excel_sheets(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
) -> List[tuple[str, List[List[str]]]]:
    position = _remember_position(file_obj)
    try:
        workbook = pd.read_excel(file_obj, sheet_name=None, dtype=str, header=None)
    except Exception as exc:
        _restore_position(file_obj, position)
        raise exc

    sheets: List[tuple[str, List[List[str]]]] = []
    for name, frame in workbook.items():
        rows = frame.fillna("").astype(str).values.tolist()
        if any(any(str(cell).strip() for cell in row) for row in rows):
            sheets.append((str(name), rows))

    _restore_position(file_obj, position)
    return sheets


def _read_structured_rows(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
) -> List[List[str]]:
    extension = _get_file_extension(file_obj)
    rows: Optional[List[List[str]]] = None

    if extension in EXCEL_EXTENSIONS:
        rows = _read_first_sheet_rows(file_obj)
        if rows is None:
            raise ValueError("Не удалось прочитать данные из Excel-файла")
    else:
        rows = _read_first_sheet_rows(file_obj)

    if rows is not None and any(any(str(cell).strip() for cell in row) for row in rows):
        return rows

    text = _read_text(file_obj)
    csv_reader = csv.reader(io.StringIO(text))
    return [row for row in csv_reader]


def _normalize_numeric(value: str | float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    cleaned = value.replace("\u00A0", " ")
    cleaned = SPACE_PATTERN.sub("", cleaned)
    cleaned = cleaned.replace("\u2212", "-")  # minus sign variant
    cleaned = cleaned.replace(",", ".")
    cleaned = cleaned.replace('"', "")
    lowered = cleaned.lower()
    for token in ("usd", "uah", "byn", "eur", "rub", "₽", "руб", "som", "kzt"):
        lowered = lowered.replace(token, "")
    cleaned = lowered.replace("$", "").replace("€", "").replace("£", "")
    return cleaned.strip()


def _looks_numeric(value: str | float | int | None) -> bool:
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return True
    text = str(value)
    return any(char.isdigit() for char in text)


def _parse_int(value: str | float | int | None) -> int:
    cleaned = _normalize_numeric(value)
    if cleaned in {"", "-", "--"}:
        return 0
    try:
        return int(float(cleaned))
    except ValueError:
        raise ValueError(f"Не удалось преобразовать значение '{value}' в целое число")


def _parse_float(value: str | float | int | None) -> float:
    cleaned = _normalize_numeric(value)
    if cleaned in {"", "-", "--"}:
        return 0.0
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"Не удалось преобразовать значение '{value}' в число с плавающей точкой")


def _combine_rows(row_a: List[str], row_b: List[str]) -> List[str]:
    width = max(len(row_a), len(row_b))
    combined: List[str] = []
    for idx in range(width):
        left = row_a[idx] if idx < len(row_a) else ""
        right = row_b[idx] if idx < len(row_b) else ""
        left_text = str(left).strip()
        right_text = str(right).strip()
        if left_text and right_text:
            if left_text.lower() == right_text.lower():
                combined.append(left_text)
            else:
                combined.append(f"{left_text} {right_text}".strip())
        elif left_text:
            combined.append(left_text)
        elif right_text:
            combined.append(right_text)
        else:
            combined.append("")
    return combined


def _get_reconciliation_markers(column_map: dict[str, int]) -> tuple[Optional[int], Optional[int], Optional[int]]:
    buyer_idx = _resolve_column(column_map, HEADER_ALIASES["buyer"])
    ftd_idx = _resolve_column(column_map, HEADER_ALIASES["ftd"])
    payout_idx = _resolve_column(column_map, HEADER_ALIASES["payout"])
    return buyer_idx, ftd_idx, payout_idx


def _guess_buyer_header_index(
    row: List[str],
    column_map: dict[str, int],
    ftd_idx: Optional[int],
    payout_idx: Optional[int],
) -> Optional[int]:
    skip_indices = {idx for idx in (ftd_idx, payout_idx) if idx is not None}
    candidates: List[int] = []
    for idx, cell in enumerate(row):
        if idx in skip_indices:
            continue
        text = str(cell).strip()
        if not text:
            continue
        normalized = _normalize_header_name(text)
        if not normalized:
            continue
        if normalized in METRIC_HEADER_KEYS:
            continue
        if normalized in column_map and column_map.get(normalized) in skip_indices:
            continue
        candidates.append(idx)

    if not candidates:
        return None

    if ftd_idx is not None:
        left_candidates = [idx for idx in candidates if idx < ftd_idx]
        if left_candidates:
            return left_candidates[-1]

    return candidates[0]


def _find_header_row(rows: List[List[str]]) -> tuple[int, int, List[str]]:
    lookahead_limit = 6

    for idx, row in enumerate(rows):
        if not row or not any(str(cell).strip() for cell in row):
            continue

        column_map = _build_column_index(row)
        buyer_idx, ftd_idx, payout_idx = _get_reconciliation_markers(column_map)
        if ftd_idx is not None and payout_idx is not None:
            if buyer_idx is None:
                buyer_idx = _guess_buyer_header_index(row, column_map, ftd_idx, payout_idx)
            if buyer_idx is not None:
                return idx, 1, row

        combined_row = row[:]
        span = 1
        for offset in range(1, lookahead_limit + 1):
            if idx + offset >= len(rows):
                break
            combined_row = _combine_rows(combined_row, rows[idx + offset])
            span += 1
            combined_map = _build_column_index(combined_row)
            buyer_idx, ftd_idx, payout_idx = _get_reconciliation_markers(combined_map)
            if ftd_idx is not None and payout_idx is not None:
                if buyer_idx is None:
                    buyer_idx = _guess_buyer_header_index(combined_row, combined_map, ftd_idx, payout_idx)
                if buyer_idx is None:
                    continue
                return idx, span, combined_row

    raise ValueError("В файле не найдена строка заголовка 'Buyer'")


def _extract_currency(rows: List[List[str]]) -> Optional[str]:
    currency_candidates = {"usd", "eur", "uah", "rub", "byn", "gbp", "cad", "aud"}
    for row in rows:
        for cell in reversed(row):
            normalized = cell.strip().lower()
            if normalized in currency_candidates:
                return normalized.upper()
    return None


def _normalize_header_name(name: str) -> str:
    cleaned = unicodedata.normalize("NFKD", name or "")
    cleaned = cleaned.replace("\u00A0", " ").replace("\u202F", " ")
    cleaned = cleaned.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    cleaned = cleaned.replace("\ufeff", "")
    cleaned = cleaned.strip().lower()
    cleaned = cleaned.replace("ё", "е")
    cleaned = re.sub(r"[^a-z0-9а-я]+", "", cleaned)
    return cleaned


def _strip_leading_symbols(text: str) -> str:
    if not text:
        return ""
    text = text.lstrip()
    for idx, ch in enumerate(text):
        category = unicodedata.category(ch)
        if category and category[0] in {"L", "N"}:
            return text[idx:]
    return ""


def _is_summary_label(value: str) -> bool:
    if not value:
        return False
    normalized = value.replace("\u00A0", " ").replace("\u202F", " ")
    normalized = normalized.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    normalized = normalized.replace("\ufeff", "")
    normalized = _strip_leading_symbols(normalized)
    normalized = normalized.strip()
    if not normalized:
        return False
    lowered = normalized.casefold()
    for prefix in SUMMARY_ROW_PREFIXES:
        token = prefix.casefold()
        if lowered.startswith(token):
            return True
        if lowered.endswith(token):
            return True
        if lowered.startswith("(") and lowered.endswith(")"):
            inner = lowered[1:-1].strip()
            if inner.startswith(token) or inner.endswith(token):
                return True
    return False


def _normalize_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = normalized.replace("\u00A0", " ").replace("\u202F", " ")
    normalized = normalized.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    normalized = normalized.replace("\ufeff", "")
    normalized = "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")
    return normalized.casefold()


def _canonicalize_buyer_name(name: str) -> str:
    normalized = name.replace("\u00A0", " ").replace("\u202F", " ")
    normalized = normalized.replace("\u200B", "").replace("\u200C", "").replace("\u200D", "")
    normalized = normalized.replace("\ufeff", "")
    normalized = _strip_leading_symbols(normalized)
    normalized = normalized.strip()
    lowered = normalized.casefold()
    if lowered in BUYER_OVERRIDES:
        return BUYER_OVERRIDES[lowered]
    if normalized.islower() and len(normalized) <= 20:
        # Для коротких кириллических названий приводим к Capitalized виду
        try:
            return normalized.capitalize()
        except Exception:  # pragma: no cover - safety net
            return normalized
    return normalized


def _buyer_key(name: str) -> str:
    return name.replace("\u00A0", " ").strip().casefold()


def _build_column_index(row: List[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, cell in enumerate(row):
        normalized = _normalize_header_name(cell)
        if normalized and normalized not in mapping:
            mapping[normalized] = idx
    return mapping


def _resolve_column(column_map: dict[str, int], keys: list[str]) -> Optional[int]:
    candidates = list(column_map.items())
    for key in keys:
        normalized = _normalize_header_name(key)
        if normalized in column_map:
            return column_map[normalized]
        for existing_key, index in candidates:
            if existing_key.startswith(normalized) or normalized in existing_key:
                return index
    return None


def _get_cell(row: List[str], index: Optional[int]) -> str:
    if index is None:
        return ""
    if index < len(row):
        return row[index]
    return ""


def _parse_reconciliation_rows(
    rows: List[List[str]],
    partner_program: str,
    source_name: Optional[str],
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "buyer",
                "commission_type",
                "ftd_count",
                "payout",
                "partner_program",
                "source_file",
                "currency",
                "is_chargeback",
            ]
        )

    header_idx, header_span, header_row = _find_header_row(rows)
    column_map = _build_column_index(header_row)
    buyer_idx = _resolve_column(column_map, HEADER_ALIASES["buyer"])
    ftd_idx = _resolve_column(column_map, HEADER_ALIASES["ftd"])
    payout_idx = _resolve_column(column_map, HEADER_ALIASES["payout"])
    commission_idx = _resolve_column(column_map, HEADER_ALIASES["commission"])

    if buyer_idx is None:
        buyer_idx = _guess_buyer_header_index(header_row, column_map, ftd_idx, payout_idx)
    if buyer_idx is None:
        raise ValueError("В файле отсутствует колонка с именем байера")
    if ftd_idx is None:
        raise ValueError("В файле отсутствует колонка с количеством депозитов (FTD)")
    if payout_idx is None:
        raise ValueError("В файле отсутствует колонка с суммой выплат (Payout)")

    data_rows = rows[header_idx + header_span :]
    currency = _extract_currency(rows[:header_idx])

    records: List[ReconciliationRow] = []
    current_buyer: Optional[str] = None
    source = source_name or partner_program

    for raw_row in data_rows:
        buyer_cell = _get_cell(raw_row, buyer_idx).strip()
        commission_cell = _get_cell(raw_row, commission_idx).strip()
        ftd_cell = _get_cell(raw_row, ftd_idx)
        payout_cell = _get_cell(raw_row, payout_idx)

        if buyer_cell:
            normalized_candidate = _canonicalize_buyer_name(buyer_cell)
            if _is_summary_label(buyer_cell) or _is_summary_label(normalized_candidate):
                continue
            current_buyer = buyer_cell
            buyer_value = buyer_cell
        else:
            if current_buyer is None:
                continue
            buyer_value = current_buyer

        normalized_buyer = _canonicalize_buyer_name(buyer_value)
        if _is_summary_label(buyer_value) or _is_summary_label(normalized_buyer):
            continue
        if normalized_buyer == "":
            continue
        if normalized_buyer.lower() in {"buyer", "totals"}:
            continue

        if commission_cell.lower().startswith("сумма") or "тотал" in commission_cell.lower():
            continue

        try:
            ftd_value = _parse_int(ftd_cell)
        except ValueError:
            if not _looks_numeric(ftd_cell):
                continue
            raise
        try:
            payout_value = _parse_float(payout_cell)
        except ValueError:
            if not _looks_numeric(payout_cell):
                continue
            raise
        records.append(
            ReconciliationRow(
                buyer=normalized_buyer,
                commission_type=commission_cell or "Unknown",
                ftd_count=ftd_value,
                payout=payout_value,
                partner_program=partner_program,
                source_file=source,
                currency=currency,
            )
        )

    df = pd.DataFrame([record.__dict__ for record in records])
    if df.empty:
        return df

    df["is_chargeback"] = df["commission_type"].str.contains("chargeback", case=False, na=False)
    return df


def load_reconciliation_file(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
    partner_program: str,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    """Return cleaned reconciliation details as DataFrame.

    Columns: buyer, commission_type, ftd_count, payout, partner_program,
    source_file, currency, is_chargeback.
    """

    extension = _get_file_extension(file_obj)
    if extension in EXCEL_EXTENSIONS:
        try:
            sheets = _read_excel_sheets(file_obj)
        except Exception:
            sheets = []
        preferred: List[tuple[str, List[List[str]]]] = []
        others: List[tuple[str, List[List[str]]]] = []
        for sheet_name, rows in sheets:
            normalized = _normalize_label(sheet_name.strip())
            if "сводная" in normalized and ("байер" in normalized or "баер" in normalized or "buyer" in normalized):
                preferred.append((sheet_name, rows))
            else:
                others.append((sheet_name, rows))

        ordered_sheets = preferred + others
        errors: List[str] = []
        missing_header = False
        for sheet_name, rows in ordered_sheets:
            try:
                return _parse_reconciliation_rows(rows, partner_program, source_name)
            except ValueError as exc:
                message = str(exc)
                if "строка заголовка 'Buyer'" in message:
                    missing_header = True
                    continue
                errors.append(f"{sheet_name}: {message}")
                continue
        if errors:
            raise ValueError("; ".join(errors))
        if missing_header:
            raise ValueError("В файле не найдена строка заголовка 'Buyer'")
        raise ValueError("В Excel-файле не найдены листы с данными")

    rows = _read_structured_rows(file_obj)
    return _parse_reconciliation_rows(rows, partner_program, source_name)


def aggregate_by_partner_and_buyer(details: pd.DataFrame) -> pd.DataFrame:
    if details.empty:
        return details

    df = details.copy()

    df["deposits"] = np.where(df["is_chargeback"], 0, df["ftd_count"])
    df["chargebacks"] = np.where(df["is_chargeback"], np.abs(df["ftd_count"]), 0)
    df["net_deposits"] = df["ftd_count"]

    df["payout_positive"] = np.where(df["payout"] > 0, df["payout"], 0.0)
    df["payout_negative"] = np.where(df["payout"] < 0, np.abs(df["payout"]), 0.0)
    df["net_payout"] = df["payout"]

    aggregation = {
        "deposits": "sum",
        "chargebacks": "sum",
        "net_deposits": "sum",
        "payout_positive": "sum",
        "payout_negative": "sum",
        "net_payout": "sum",
        "currency": lambda x: _first_non_null(x),
    }

    grouped = (
        df.groupby(["partner_program", "buyer"], dropna=False)
        .agg(aggregation)
        .reset_index()
        .rename(
            columns={
                "payout_positive": "payout",
                "payout_negative": "chargeback_amount",
            }
        )
    )

    grouped["net_payout"] = grouped["net_payout"].round(2)
    grouped["payout"] = grouped["payout"].round(2)
    grouped["chargeback_amount"] = grouped["chargeback_amount"].round(2)

    return grouped.sort_values(["partner_program", "buyer"]).reset_index(drop=True)


def aggregate_overall(partner_summary: pd.DataFrame) -> pd.DataFrame:
    if partner_summary.empty:
        return partner_summary

    aggregation = {
        "deposits": "sum",
        "chargebacks": "sum",
        "net_deposits": "sum",
        "payout": "sum",
        "chargeback_amount": "sum",
        "net_payout": "sum",
        "partner_program": lambda x: ", ".join(sorted(set(x))),
    }

    grouped = (
        partner_summary.groupby(["buyer", "currency"], dropna=False)
        .agg(aggregation)
        .reset_index()
    )

    grouped["net_payout"] = grouped["net_payout"].round(2)
    grouped["payout"] = grouped["payout"].round(2)
    grouped["chargeback_amount"] = grouped["chargeback_amount"].round(2)

    return grouped.sort_values(["buyer"]).reset_index(drop=True)


def _first_non_null(values: Iterable[Optional[str]]) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _has_non_empty(values: Iterable[Optional[str]]) -> bool:
    for value in values:
        if isinstance(value, str) and value.strip():
            return True
    return False


def load_expenses_file(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    """Parse expense spreadsheet exported as CSV.

    Returns columns: buyer, expense_type, item_count, amount, source_file, notes.
    """
    rows = _read_structured_rows(file_obj)

    if not rows:
        return pd.DataFrame(
            columns=["buyer", "expense_type", "item_count", "amount", "source_file", "notes"]
        )

    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    raw = pd.DataFrame(normalized_rows, dtype=str).fillna("")

    records: list[dict[str, object]] = []

    for row_idx in range(1, raw.shape[0]):
        header_row = raw.iloc[row_idx]
        if not any(str(cell).strip().lower() == "байер" for cell in header_row):
            continue

        section_row = raw.iloc[row_idx - 1] if row_idx > 0 else pd.Series(dtype=str)

        for col_idx, cell in header_row.items():
            if str(cell).strip().lower() != "байер":
                continue

            expense_type = str(section_row.iloc[col_idx]).strip() if col_idx < len(section_row) else ""
            if not expense_type:
                continue

            count_col = col_idx + 1
            amount_col = col_idx + 2
            note_col = col_idx + 3 if col_idx + 3 < raw.shape[1] else None

            data_row_idx = row_idx + 1
            while data_row_idx < raw.shape[0]:
                buyer_raw = str(raw.iat[data_row_idx, col_idx]).strip() if col_idx < raw.shape[1] else ""
                count_raw = (
                    str(raw.iat[data_row_idx, count_col]).strip()
                    if count_col < raw.shape[1]
                    else ""
                )
                amount_raw = (
                    str(raw.iat[data_row_idx, amount_col]).strip()
                    if amount_col < raw.shape[1]
                    else ""
                )
                note_raw = (
                    str(raw.iat[data_row_idx, note_col]).strip()
                    if note_col is not None and note_col < raw.shape[1]
                    else ""
                )

                if not _has_non_empty([buyer_raw, count_raw, amount_raw]):
                    break

                normalized_buyer = _canonicalize_buyer_name(buyer_raw.replace("\u00A0", " ").strip())
                if not normalized_buyer:
                    data_row_idx += 1
                    continue

                lowered_buyer = normalized_buyer.lower()
                if any(lowered_buyer.startswith(prefix.lower()) for prefix in SUMMARY_ROW_PREFIXES):
                    break

                item_count = _parse_float(count_raw)
                amount_value = _parse_float(amount_raw)

                records.append(
                    {
                        "buyer": normalized_buyer,
                        "expense_type": expense_type,
                        "item_count": item_count,
                        "amount": amount_value,
                        "source_file": source_name or getattr(file_obj, "name", "uploaded"),
                        "notes": note_raw,
                    }
                )

                data_row_idx += 1

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["item_count"] = pd.to_numeric(df["item_count"], errors="coerce").fillna(0.0)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)

    if (df["item_count"] % 1 == 0).all():
        df["item_count"] = df["item_count"].astype(int)

    df["amount"] = df["amount"].round(2)
    return df.reset_index(drop=True)


def aggregate_expenses_by_buyer(expense_details: pd.DataFrame) -> pd.DataFrame:
    if expense_details.empty:
        return expense_details

    grouped = (
        expense_details.groupby(["buyer", "expense_type"], dropna=False)
        .agg(item_count=("item_count", "sum"), amount=("amount", "sum"))
        .reset_index()
    )

    if (grouped["item_count"] % 1 == 0).all():
        grouped["item_count"] = grouped["item_count"].astype(int)
    grouped["amount"] = grouped["amount"].round(2)

    return grouped.sort_values(["buyer", "expense_type"]).reset_index(drop=True)


def aggregate_expenses_totals(expense_details: pd.DataFrame) -> pd.DataFrame:
    if expense_details.empty:
        return expense_details

    grouped = (
        expense_details.groupby("buyer", dropna=False)
        .agg(
            total_amount=("amount", "sum"),
            total_items=("item_count", "sum"),
            expense_types=("expense_type", "nunique"),
            entries=("expense_type", "count"),
        )
        .reset_index()
    )

    grouped["total_amount"] = grouped["total_amount"].round(2)
    if (grouped["total_items"] % 1 == 0).all():
        grouped["total_items"] = grouped["total_items"].astype(int)

    return grouped.sort_values(["total_amount", "buyer"], ascending=[False, True]).reset_index(drop=True)


def _detect_currency_code(value: str | float | int | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    text = str(value)
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    lowered = text.lower()
    for token in ("usd", "eur", "uah", "rub", "byn", "kzt", "cad", "aud", "gbp", "pln", "czk"):
        if token in lowered:
            return token.upper()
    return None


def _clean_identifier(value: str) -> str:
    cleaned = value.replace("\u00A0", " ").strip()
    cleaned = cleaned.strip("/")
    cleaned = cleaned.replace(" ", "")
    return cleaned


def _parse_spend_rows(
    rows: List[List[str]],
    sheet_name: str,
    source_file: str,
) -> pd.DataFrame:
    result_columns = [
        "buyer",
        "buyer_key",
        "account_label",
        "account_id",
        "spend",
        "total",
        "currency",
        "sheet_name",
        "source_file",
        "notes",
    ]

    def _has_meaningful(cell: object) -> bool:
        text = str(cell).strip()
        lowered = text.lower()
        if not text:
            return False
        if lowered in {"nan", "none"}:
            return False
        if lowered.startswith("unnamed:"):
            return False
        return True

    if not rows:
        return pd.DataFrame(columns=result_columns)

    width = max(len(row) for row in rows)
    normalized_rows = [list(row) + [""] * (width - len(row)) for row in rows]

    header_row = normalized_rows[0]
    data_has_content = any(_has_meaningful(cell) for row in normalized_rows[1:] for cell in row)
    if not any(_has_meaningful(cell) for cell in header_row) and not data_has_content:
        return pd.DataFrame(columns=result_columns)
    column_map = _build_column_index(header_row)

    buyer_idx = _resolve_column(column_map, SPEND_HEADER_ALIASES["buyer"])
    spend_idx = _resolve_column(column_map, SPEND_HEADER_ALIASES["spend"])
    account_label_idx = _resolve_column(column_map, SPEND_HEADER_ALIASES["account_label"].copy())
    account_id_idx = _resolve_column(column_map, SPEND_HEADER_ALIASES["account_id"].copy())
    total_idx = _resolve_column(column_map, SPEND_HEADER_ALIASES["total"])

    if buyer_idx is None or spend_idx is None:
        if not data_has_content:
            return pd.DataFrame(columns=result_columns)
        missing = "байером" if buyer_idx is None else "спендом"
        raise ValueError(f"Лист '{sheet_name}': не найдена колонка с {missing}")

    records: list[dict[str, object]] = []
    sheet_currency: str = _derive_spend_currency(sheet_name)

    skip_indices = {buyer_idx, spend_idx}
    if account_label_idx is not None:
        skip_indices.add(account_label_idx)
    if account_id_idx is not None:
        skip_indices.add(account_id_idx)
    if total_idx is not None:
        skip_indices.add(total_idx)

    for row in normalized_rows[1:]:
        buyer_raw = str(row[buyer_idx]).replace("\u00A0", " ").strip()
        spend_raw = str(row[spend_idx]).strip()
        total_raw = str(row[total_idx]).strip() if total_idx is not None else ""

        if not _has_non_empty([buyer_raw, spend_raw, total_raw]):
            continue

        if any(buyer_raw.lower().startswith(prefix.lower()) for prefix in SUMMARY_ROW_PREFIXES):
            continue

        if not buyer_raw:
            continue

        buyer_display = _canonicalize_buyer_name(buyer_raw)
        buyer_identifier = _buyer_key(buyer_display or buyer_raw)

        account_label_raw = (
            str(row[account_label_idx]).strip()
            if account_label_idx is not None and account_label_idx < len(row)
            else ""
        )
        account_id_raw = (
            str(row[account_id_idx]).strip()
            if account_id_idx is not None and account_id_idx < len(row)
            else ""
        )

        notes_parts: list[str] = []

        try:
            spend_value = _parse_float(spend_raw)
        except ValueError:
            stripped = spend_raw.strip()
            if stripped:
                notes_parts.append(f"SPEND:{stripped}")
            # если не удалось распарсить спенд — пропускаем строку
            continue

        if spend_value == 0 and not spend_raw:
            continue

        if total_idx is not None and total_raw:
            try:
                total_value = _parse_float(total_raw)
            except ValueError:
                stripped_total = total_raw.strip()
                if stripped_total:
                    notes_parts.append(f"TOTAL:{stripped_total}")
                total_value = float("nan")
        else:
            total_value = float("nan")

        account_id = _clean_identifier(account_id_raw)
        account_label = account_label_raw.replace("\u00A0", " ").strip(" /")

        if not account_label and account_id:
            account_label = account_id

        for idx, cell in enumerate(row):
            if idx in skip_indices:
                continue
            cell_text = str(cell).strip()
            if cell_text:
                notes_parts.append(cell_text)

        record_currency = sheet_currency

        records.append(
            {
                "buyer": buyer_display,
                "buyer_key": buyer_identifier,
                "account_label": account_label,
                "account_id": account_id,
                "spend": spend_value,
                "total": total_value,
                "currency": record_currency,
                "sheet_name": sheet_name,
                "source_file": source_file,
                "notes": " | ".join(dict.fromkeys(notes_parts)) if notes_parts else "",
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0).round(2)
    if "total" in df:
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
        df["total"] = df["total"].round(2)

    if sheet_currency and "currency" in df.columns:
        df["currency"] = df["currency"].fillna(sheet_currency)

    df["currency"] = df["currency"].where(df["currency"].notna(), None)
    df["buyer"] = df["buyer"].fillna("").astype(str)
    df["buyer_key"] = df.get("buyer_key", df["buyer"].str.casefold())
    df["buyer_key"] = df["buyer_key"].fillna(df["buyer"].str.casefold())
    df["account_label"] = df["account_label"].fillna("")
    df["account_id"] = df["account_id"].fillna("")
    df["notes"] = df["notes"].fillna("")

    return df.reset_index(drop=True)


def load_spend_sheet(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
    sheet_name: str,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    text = _read_text(file_obj)
    csv_reader = csv.reader(io.StringIO(text))
    rows = [row for row in csv_reader]
    source = source_name or getattr(file_obj, "name", "uploaded")
    return _parse_spend_rows(rows, sheet_name, source)


def load_spend_workbook(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    source = source_name or getattr(file_obj, "name", "workbook")
    position: Optional[int] = None
    if hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
        try:
            position = file_obj.tell()
            file_obj.seek(0)
        except (OSError, io.UnsupportedOperation):
            position = None

    workbook = pd.read_excel(file_obj, sheet_name=None, dtype=str)

    frames: list[pd.DataFrame] = []
    warnings: list[str] = []
    for sheet, frame in workbook.items():
        header = [str(col) if not pd.isna(col) else "" for col in frame.columns]
        data_rows = [list(row) for row in frame.fillna("").astype(str).itertuples(index=False, name=None)]
        rows = [header] + data_rows
        try:
            parsed = _parse_spend_rows(rows, sheet, source)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        if not parsed.empty:
            frames.append(parsed)

    if position is not None:
        try:
            file_obj.seek(position)
        except (OSError, io.UnsupportedOperation):
            pass

    if not frames:
        empty = pd.DataFrame(
            columns=[
                "buyer",
                "account_label",
                "account_id",
                "spend",
                "total",
                "currency",
                "sheet_name",
                "source_file",
                "notes",
            ]
        )
        if warnings:
            empty.attrs["warnings"] = warnings
        return empty

    result = pd.concat(frames, ignore_index=True)
    if warnings:
        result.attrs["warnings"] = warnings
    return result


def aggregate_spend_by_buyer(spend_details: pd.DataFrame) -> pd.DataFrame:
    if spend_details.empty:
        return spend_details
    df = spend_details.copy()
    if "buyer_key" not in df.columns:
        df["buyer_key"] = df["buyer"].astype(str).str.casefold()

    df["buyer_display"] = df["buyer"].apply(_canonicalize_buyer_name)
    buyer_display_map = (
        df.groupby("buyer_key")["buyer_display"].agg(lambda series: _first_non_null(series)).to_dict()
    )

    grouped = (
        df.groupby(["buyer_key", "currency"], dropna=False)
        .agg(
            total_spend=("spend", "sum"),
            total_reported=("total", "sum"),
            accounts=("account_id", "nunique"),
            rows=("account_id", "count"),
            sheets=("sheet_name", "nunique"),
        )
        .reset_index()
    )

    grouped["buyer"] = grouped["buyer_key"].map(buyer_display_map).fillna("")
    grouped["total_spend"] = grouped["total_spend"].round(2)
    grouped["total_reported"] = grouped["total_reported"].round(2)

    grouped = grouped[
        ["buyer", "currency", "total_spend", "total_reported", "accounts", "rows", "sheets"]
    ]

    return grouped.sort_values(["buyer", "currency"], na_position="last").reset_index(drop=True)


def aggregate_spend_by_sheet(spend_details: pd.DataFrame) -> pd.DataFrame:
    if spend_details.empty:
        return spend_details

    df = spend_details.copy()
    if "buyer_key" not in df.columns:
        df["buyer_key"] = df["buyer"].astype(str).str.casefold()

    grouped = (
        df.groupby(["sheet_name", "currency"], dropna=False)
        .agg(
            total_spend=("spend", "sum"),
            total_reported=("total", "sum"),
            buyers=("buyer_key", "nunique"),
            rows=("account_id", "count"),
        )
        .reset_index()
    )

    grouped["total_spend"] = grouped["total_spend"].round(2)
    grouped["total_reported"] = grouped["total_reported"].round(2)

    return grouped.sort_values(["sheet_name", "currency"], na_position="last").reset_index(drop=True)
