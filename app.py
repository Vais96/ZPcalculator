from __future__ import annotations

import re
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from zp_calculator.data_processing import (
    aggregate_by_partner_and_buyer,
    aggregate_expenses_by_buyer,
    aggregate_expenses_totals,
    aggregate_overall,
    load_expenses_file,
    load_reconciliation_file,
)


def create_download_button(label: str, df: pd.DataFrame, filename: str) -> None:
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
    csv_buffer.seek(0)
    st.download_button(
        label=label,
        data=csv_buffer,
        file_name=filename,
        mime="text/csv",
    )


def append_total_row(df: pd.DataFrame, label: str, numeric_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return df

    total = {column: df[column].sum() if column in numeric_columns else "" for column in df.columns}
    first_column = df.columns[0]
    total[first_column] = label
    total_df = pd.DataFrame([total])
    # Preserve integer formatting for count columns when possible
    for column in numeric_columns:
        if column in df.columns and pd.api.types.is_integer_dtype(df[column]):
            total_df[column] = total_df[column].astype(int)
    return pd.concat([df, total_df], ignore_index=True)


def resolve_rate(value: str | None, rates: dict[str, float]) -> float:
    if value is None:
        return 1.0
    normalized = str(value).strip().upper()
    if not normalized or normalized == "NAN":
        return 1.0
    return rates.get(normalized, 1.0)


PARTNER_SUFFIX_PATTERN = re.compile(r"\s*-\s*сводн[аa][^$]*$", re.IGNORECASE)


def infer_partner_program(filename: str) -> str:
    stem = Path(filename).stem
    cleaned = PARTNER_SUFFIX_PATTERN.sub("", stem)
    cleaned = cleaned.strip()
    return cleaned or stem.strip()


def render_reconciliation_module(sidebar: DeltaGenerator) -> None:
    st.subheader("🧾 Сводная по байерам из файлов сверок")
    st.caption(
        "Загрузите один или несколько CSV-файлов сверок, чтобы получить агрегированную таблицу по байерам."
    )

    uploaded_files = st.file_uploader(
        "Загрузите CSV-файлы",
        type=["csv"],
        accept_multiple_files=True,
        key="reconciliation_files",
    )

    if not uploaded_files:
        st.info("Добавьте один или несколько CSV-файлов, чтобы увидеть результат.")
        return

    all_details: list[pd.DataFrame] = []
    errors: list[str] = []

    for uploaded in uploaded_files:
        partner_program = infer_partner_program(uploaded.name)
        try:
            details = load_reconciliation_file(uploaded, partner_program, source_name=uploaded.name)
        except Exception as exc:  # pragma: no cover - отображение ошибки для пользователя
            errors.append(f"{uploaded.name}: {exc}")
            continue

        if details.empty:
            errors.append(f"{uploaded.name}: не удалось извлечь данные")
            continue

        all_details.append(details)

    if errors:
        st.warning("\n".join(errors))

    if not all_details:
        st.error("Не получилось прочитать данные ни из одного файла.")
        return

    combined_details = pd.concat(all_details, ignore_index=True)
    combined_details["partner_program"] = combined_details["partner_program"].astype(str)
    combined_details["buyer"] = combined_details["buyer"].astype(str)
    combined_details["currency"] = (
        combined_details["currency"].fillna("USD").replace("", "USD").astype(str).str.upper()
    )

    available_currencies = sorted(
        {
            currency
            for currency in combined_details["currency"].unique()
            if isinstance(currency, str) and currency and currency != "NAN"
        }
    )
    if not available_currencies:
        available_currencies = ["USD"]

    sidebar.subheader("Курсы валют → $")
    sidebar.caption("Введите, сколько долларов соответствует 1 единице валюты партнёрской программы.")
    exchange_rates: dict[str, float] = {}
    for currency in available_currencies:
        exchange_rates[currency] = sidebar.number_input(
            f"{currency} → USD",
            min_value=0.0,
            value=1.0,
            step=0.01,
            format="%.4f",
            key=f"rate_{currency}",
        )

    exchange_rates.setdefault("USD", 1.0)

    partner_options = sorted(combined_details["partner_program"].unique())
    selected_programs = st.multiselect(
        "Партнерские программы",
        options=partner_options,
        default=partner_options,
    )
    filtered_details = combined_details[combined_details["partner_program"].isin(selected_programs)]

    buyer_options = sorted(filtered_details["buyer"].unique())
    selected_buyers = st.multiselect(
        "Байеры",
        options=buyer_options,
        default=buyer_options,
    )
    filtered_details = filtered_details[filtered_details["buyer"].isin(selected_buyers)]

    if filtered_details.empty:
        st.warning("По выбранным фильтрам нет данных.")
        return

    st.subheader("Детальные строки")
    st.dataframe(
        filtered_details[
            [
                "partner_program",
                "buyer",
                "commission_type",
                "ftd_count",
                "payout",
                "currency",
                "is_chargeback",
                "source_file",
            ]
        ]
    )

    summary = aggregate_by_partner_and_buyer(filtered_details)
    summary["currency"] = summary["currency"].fillna("USD").replace("", "USD").str.upper()
    summary["conversion_rate"] = summary["currency"].apply(lambda cur: resolve_rate(cur, exchange_rates))
    summary["payout_usd"] = (summary["payout"] * summary["conversion_rate"]).round(2)
    summary["chargeback_amount_usd"] = (
        summary["chargeback_amount"] * summary["conversion_rate"]
    ).round(2)
    summary["net_payout_usd"] = (summary["net_payout"] * summary["conversion_rate"]).round(2)

    st.subheader("Сводка по партнерским программам и байерам")
    st.dataframe(
        summary[
            [
                "partner_program",
                "buyer",
                "deposits",
                "chargebacks",
                "net_deposits",
                "payout",
                "payout_usd",
                "chargeback_amount",
                "chargeback_amount_usd",
                "net_payout",
                "net_payout_usd",
                "currency",
            ]
        ]
    )
    create_download_button("Скачать сводку по программам", summary, "partner_summary.csv")

    overall = aggregate_overall(summary)
    overall["currency"] = overall["currency"].fillna("USD").replace("", "USD").str.upper()
    overall["conversion_rate"] = overall["currency"].apply(lambda cur: resolve_rate(cur, exchange_rates))
    overall["payout_usd"] = (overall["payout"] * overall["conversion_rate"]).round(2)
    overall["chargeback_amount_usd"] = (
        overall["chargeback_amount"] * overall["conversion_rate"]
    ).round(2)
    overall["net_payout_usd"] = (overall["net_payout"] * overall["conversion_rate"]).round(2)

    st.subheader("Общая сводка по байерам")
    st.dataframe(
        overall[
            [
                "buyer",
                "currency",
                "deposits",
                "chargebacks",
                "net_deposits",
                "payout",
                "payout_usd",
                "chargeback_amount",
                "chargeback_amount_usd",
                "net_payout",
                "net_payout_usd",
                "partner_program",
            ]
        ]
    )
    create_download_button("Скачать общую сводку", overall, "overall_summary.csv")

    buyer_detail_options = sorted(summary["buyer"].unique())

    if buyer_detail_options:
        st.subheader("Отчет по выбранному байеру")
        buyer_for_report = st.selectbox("Выберите байера для отдельной выгрузки", buyer_detail_options)

        buyer_summary = summary[summary["buyer"] == buyer_for_report].copy()

        if buyer_summary.empty:
            st.info("Для выбранного байера нет данных после применения фильтров.")
            return

        offers_table = buyer_summary[
            ["partner_program", "deposits", "payout", "payout_usd", "net_payout", "net_payout_usd"]
        ].rename(
            columns={
                "partner_program": "Оффер",
                "deposits": "Депозитов",
                "payout": "Ревеню в валюте ПП",
                "payout_usd": "Ревеню в $$",
                "net_payout": "Ревеню (net) в валюте ПП",
                "net_payout_usd": "Ревеню (net) в $$",
            }
        )
        offers_table["Депозитов"] = offers_table["Депозитов"].astype(int)
        offers_table = offers_table[
            [
                "Оффер",
                "Депозитов",
                "Ревеню в валюте ПП",
                "Ревеню в $$",
                "Ревеню (net) в валюте ПП",
                "Ревеню (net) в $$",
            ]
        ]
        offers_table = append_total_row(
            offers_table,
            "Сумма",
            [
                "Депозитов",
                "Ревеню в валюте ПП",
                "Ревеню в $$",
                "Ревеню (net) в валюте ПП",
                "Ревеню (net) в $$",
            ],
        )

        st.markdown("**Выплаты**")
        st.dataframe(offers_table)
        create_download_button(
            f"Скачать отчет по байеру — {buyer_for_report}",
            offers_table,
            f"buyer_report_{buyer_for_report}.csv",
        )

        chargebacks_table = buyer_summary[buyer_summary["chargebacks"] > 0][
            ["partner_program", "chargebacks", "chargeback_amount", "chargeback_amount_usd"]
        ].rename(
            columns={
                "partner_program": "Оффер",
                "chargebacks": "Чарджбеков",
                "chargeback_amount": "Чардж в валюте ПП",
                "chargeback_amount_usd": "Чардж в $$",
            }
        )

        if not chargebacks_table.empty:
            chargebacks_table["Чарджбеков"] = chargebacks_table["Чарджбеков"].astype(int)
            chargebacks_table = append_total_row(
                chargebacks_table,
                "Сумма",
                ["Чарджбеков", "Чардж в валюте ПП", "Чардж в $$"],
            )

            st.markdown("**Чарджбеки**")
            st.dataframe(chargebacks_table)
            create_download_button(
                f"Скачать чарджбеки — {buyer_for_report}",
                chargebacks_table,
                f"buyer_chargebacks_{buyer_for_report}.csv",
            )


def render_expenses_module(sidebar: DeltaGenerator) -> None:
    st.subheader("💸 Расходники по байерам")
    st.caption("Загрузите CSV с расходами — модуль объединит блоки и посчитает итоги по байерам.")

    uploaded_files = st.file_uploader(
        "Загрузите CSV-файлы расходников",
        type=["csv"],
        accept_multiple_files=True,
        key="expenses_files",
    )

    if not uploaded_files:
        st.info("Добавьте хотя бы один CSV-файл, чтобы увидеть данные по расходам.")
        return

    expense_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for uploaded in uploaded_files:
        try:
            details = load_expenses_file(uploaded, source_name=uploaded.name)
        except Exception as exc:  # pragma: no cover - отображение ошибки для пользователя
            errors.append(f"{uploaded.name}: {exc}")
            continue

        if details.empty:
            errors.append(f"{uploaded.name}: не удалось извлечь данные")
            continue

        expense_frames.append(details)

    if errors:
        st.warning("\n".join(errors))

    if not expense_frames:
        st.error("Не удалось получить данные ни из одного файла расходников.")
        return

    combined = pd.concat(expense_frames, ignore_index=True)
    combined["amount"] = combined["amount"].astype(float).round(2)

    buyer_options = sorted(combined["buyer"].unique())
    selected_buyers = st.multiselect(
        "Байеры (расходы)",
        options=buyer_options,
        default=buyer_options,
        key="expense_buyers",
    )

    type_options = sorted(combined["expense_type"].unique())
    selected_types = st.multiselect(
        "Типы расходов",
        options=type_options,
        default=type_options,
        key="expense_types",
    )

    filtered = combined[
        combined["buyer"].isin(selected_buyers) & combined["expense_type"].isin(selected_types)
    ]

    if filtered.empty:
        st.warning("По выбранным фильтрам нет данных.")
        return

    detail_display = filtered.rename(
        columns={
            "buyer": "Байер",
            "expense_type": "Тип расхода",
            "item_count": "Кол-во",
            "amount": "Сумма, $",
            "source_file": "Источник",
            "notes": "Комментарий",
        }
    )

    st.subheader("Детальная таблица расходов")
    st.dataframe(detail_display, use_container_width=True)
    create_download_button("Скачать детализацию расходников", detail_display, "expenses_details.csv")

    by_type = aggregate_expenses_by_buyer(filtered)
    by_type_display = by_type.rename(
        columns={
            "buyer": "Байер",
            "expense_type": "Тип расхода",
            "item_count": "Кол-во",
            "amount": "Сумма, $",
        }
    )
    st.subheader("Сводка по типам расходов")
    st.dataframe(by_type_display, use_container_width=True)
    create_download_button("Скачать сводку по типам", by_type_display, "expenses_by_type.csv")

    totals = aggregate_expenses_totals(filtered)
    totals_display = totals.rename(
        columns={
            "buyer": "Байер",
            "total_amount": "Всего, $",
            "total_items": "Кол-во позиций",
            "expense_types": "Типов расходов",
            "entries": "Строк",
        }
    )

    numeric_columns = ["Всего, $", "Кол-во позиций", "Типов расходов", "Строк"]
    for column in numeric_columns[1:]:
        if column in totals_display.columns:
            totals_display[column] = totals_display[column].astype(int)
    totals_display["Всего, $"] = totals_display["Всего, $"].round(2)
    totals_display = append_total_row(totals_display, "Сумма", numeric_columns)

    st.subheader("Итого по байерам")
    st.dataframe(totals_display, use_container_width=True)
    create_download_button("Скачать итоги по байерам", totals_display, "expenses_totals.csv")

    sidebar.subheader("Быстрые метрики")
    total_spent = filtered["amount"].sum()
    unique_buyers = filtered["buyer"].nunique()
    sidebar.metric("Расходов всего, $", f"{total_spent:,.2f}")
    sidebar.metric("Активных байеров", unique_buyers)

    buyer_expense_options = sorted(filtered["buyer"].unique())
    if buyer_expense_options:
        st.subheader("Отчет по расходам выбранного байера")
        buyer_for_expenses = st.selectbox(
            "Выберите байера (расходы)",
            options=buyer_expense_options,
            key="expense_buyer_report",
        )

        buyer_expenses = filtered[filtered["buyer"] == buyer_for_expenses].copy()

        if buyer_expenses.empty:
            st.info("Для выбранного байера нет записей после фильтрации.")
        else:
            buyer_detail = buyer_expenses.rename(
                columns={
                    "expense_type": "Тип расхода",
                    "item_count": "Кол-во",
                    "amount": "Сумма, $",
                    "notes": "Комментарий",
                    "source_file": "Источник",
                }
            )[
                ["Тип расхода", "Кол-во", "Сумма, $", "Комментарий", "Источник"]
            ]
            buyer_detail = append_total_row(buyer_detail, "Сумма", ["Кол-во", "Сумма, $"])

            st.dataframe(buyer_detail, use_container_width=True)
            create_download_button(
                f"Скачать расходник — {buyer_for_expenses}",
                buyer_detail,
                f"expenses_{buyer_for_expenses}.csv",
            )

st.set_page_config(page_title="ZP Calculator", layout="wide")

sidebar = st.sidebar
module = sidebar.radio("Выберите модуль", ["Сверки", "Расходники"], index=0)

if module == "Сверки":
    sidebar.header("Правила импорта сверок")
    sidebar.markdown(
        """
        * Принимаются файлы в формате CSV (UTF-8).
        * Необходимые колонки: **Buyer**, **Commision/Commission Type**, **FTD Count**, **Payout**.
        * Строки "Всего" и "Итого" игнорируются — итог пересчитывается автоматически.
        """
    )
    st.title("ZP Calculator — Сверки")
    render_reconciliation_module(sidebar)
else:
    sidebar.header("Правила импорта расходников")
    sidebar.markdown(
        """
        * Экспортируйте таблицу с расходами в CSV (UTF-8, без объединённых ячеек).
        * Для каждого блока должны быть заголовки "Байер", "Кол-во", "Сумма" по типу расхода.
        * Строки "Итого" и "Всего" пропускаются автоматически.
        """
    )
    st.title("ZP Calculator — Расходники")
    render_expenses_module(sidebar)
