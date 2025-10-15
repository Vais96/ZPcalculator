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
]

SPACE_PATTERN = re.compile(r"[\s\u00A0\u202F]")

SUMMARY_ROW_PREFIXES = ("Всего", "Итого")

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


def _find_header_index(rows: List[List[str]]) -> int:
    for idx, row in enumerate(rows):
        if not row:
            continue
        first_cell = row[0].strip()
        if first_cell.lower() == "buyer":
            return idx
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
    cleaned = unicodedata.normalize("NFKD", name or "").strip().lower()
    cleaned = cleaned.replace("ё", "е")
    cleaned = re.sub(r"[^a-z0-9а-я]+", "", cleaned)
    return cleaned


def _build_column_index(row: List[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, cell in enumerate(row):
        normalized = _normalize_header_name(cell)
        if normalized and normalized not in mapping:
            mapping[normalized] = idx
    return mapping


def _resolve_column(column_map: dict[str, int], keys: list[str]) -> Optional[int]:
    for key in keys:
        normalized = _normalize_header_name(key)
        if normalized in column_map:
            return column_map[normalized]
    return None


def _get_cell(row: List[str], index: Optional[int]) -> str:
    if index is None:
        return ""
    if index < len(row):
        return row[index]
    return ""


def load_reconciliation_file(
    file_obj: io.BufferedIOBase | io.BytesIO | io.StringIO | io.TextIOBase,
    partner_program: str,
    source_name: Optional[str] = None,
) -> pd.DataFrame:
    """Return cleaned reconciliation details as DataFrame.

    Columns: buyer, commission_type, ftd_count, payout, partner_program,
    source_file, currency, is_chargeback.
    """
    text = _read_text(file_obj)
    csv_reader = csv.reader(io.StringIO(text), delimiter=",")
    rows = [row for row in csv_reader]
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

    header_idx = _find_header_index(rows)
    header_row = rows[header_idx]
    column_map = _build_column_index(header_row)
    buyer_idx = _resolve_column(column_map, HEADER_ALIASES["buyer"])
    ftd_idx = _resolve_column(column_map, HEADER_ALIASES["ftd"])
    payout_idx = _resolve_column(column_map, HEADER_ALIASES["payout"])
    commission_idx = _resolve_column(column_map, HEADER_ALIASES["commission"])

    if buyer_idx is None:
        raise ValueError("В файле отсутствует колонка с именем байера")
    if ftd_idx is None:
        raise ValueError("В файле отсутствует колонка с количеством депозитов (FTD)")
    if payout_idx is None:
        raise ValueError("В файле отсутствует колонка с суммой выплат (Payout)")

    data_rows = rows[header_idx + 1 :]
    currency = _extract_currency(rows[:header_idx])

    records: List[ReconciliationRow] = []
    current_buyer: Optional[str] = None
    source = source_name or partner_program

    for raw_row in data_rows:
        buyer_cell = _get_cell(raw_row, buyer_idx).strip()
        commission_cell = _get_cell(raw_row, commission_idx).strip()
        ftd_cell = _get_cell(raw_row, ftd_idx)
        payout_cell = _get_cell(raw_row, payout_idx)

        if not buyer_cell:
            if current_buyer is None:
                continue
            buyer_value = current_buyer
        else:
            buyer_value = buyer_cell
            current_buyer = buyer_value

        normalized_buyer = buyer_value.replace("\u00A0", " ").strip()
        if any(normalized_buyer.startswith(prefix) for prefix in SUMMARY_ROW_PREFIXES):
            # Skip subtotal rows; we will compute totals ourselves.
            continue
        if normalized_buyer == "":
            continue
        if normalized_buyer.lower() in {"buyer", "totals"}:
            continue

        if commission_cell.lower().startswith("сумма") or "тотал" in commission_cell.lower():
            # Skip helper rows coming from analytics exports
            continue

        ftd_value = _parse_int(ftd_cell)
        payout_value = _parse_float(payout_cell)
        is_chargeback = "chargeback" in commission_cell.lower()

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

    text = _read_text(file_obj)
    csv_reader = csv.reader(io.StringIO(text))
    rows = [row for row in csv_reader]

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

                normalized_buyer = buyer_raw.replace("\u00A0", " ").strip()
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
