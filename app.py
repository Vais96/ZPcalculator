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
    load_spend_sheet,
    load_spend_workbook,
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
        "Загрузите один или несколько CSV- или XLSX-файлов сверок (в Excel читается только первый лист), чтобы получить агрегированную таблицу по байерам."
    )

    uploaded_files = st.file_uploader(
        "Загрузите CSV или XLSX",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="reconciliation_files",
    )

    if not uploaded_files:
        st.info("Добавьте один или несколько файлов, чтобы увидеть результат.")
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
    st.caption(
        "Загрузите CSV- или XLSX-файл с расходами (из Excel берётся только первый лист) — модуль объединит блоки и посчитает итоги по байерам."
    )

    uploaded_files = st.file_uploader(
        "Загрузите CSV или XLSX",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="expenses_files",
    )

    if not uploaded_files:
        st.info("Добавьте хотя бы один файл, чтобы увидеть данные по расходам.")
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
                ["Тип расхода", "Сумма, $", "Кол-во", "Комментарий", "Источник"]
            ]
            buyer_detail = append_total_row(buyer_detail, "Сумма", ["Кол-во", "Сумма, $"])

            st.dataframe(buyer_detail, use_container_width=True)
            create_download_button(
                f"Скачать расходник — {buyer_for_expenses}",
                buyer_detail,
                f"expenses_{buyer_for_expenses}.csv",
            )


def render_spend_module(sidebar: DeltaGenerator) -> None:
    st.subheader("📊 Спенды по листам Excel")
    st.caption(
        "Загрузите один или несколько XLSX-файлов (или отдельные CSV-листы) — приложение соберёт спенды по байерам."
    )

    uploaded_files = st.file_uploader(
        "Загрузите файлы со спендами",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="spend_files",
    )

    if not uploaded_files:
        st.info("Добавьте хотя бы один файл, чтобы построить отчёты по спендам.")
        return

    spend_frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix.lower()
        try:
            if suffix in {".xlsx", ".xls"}:
                frame = load_spend_workbook(uploaded, source_name=uploaded.name)
            else:
                sheet_name = Path(uploaded.name).stem
                frame = load_spend_sheet(uploaded, sheet_name=sheet_name, source_name=uploaded.name)
        except Exception as exc:  # pragma: no cover - пользователь видит ошибку
            errors.append(f"{uploaded.name}: {exc}")
            continue

        warnings = getattr(frame, "attrs", {}).get("warnings", [])
        if warnings:
            errors.extend(f"{uploaded.name}: {warning}" for warning in warnings)
            frame = frame.copy()
            try:
                del frame.attrs["warnings"]
            except KeyError:
                pass

        if frame.empty:
            errors.append(f"{uploaded.name}: не удалось извлечь данные")
            continue

        spend_frames.append(frame)

    if errors:
        st.warning("\n".join(errors))

    if not spend_frames:
        st.error("Не получилось прочитать спенды ни из одного файла.")
        return

    combined = pd.concat(spend_frames, ignore_index=True)
    combined["currency"] = combined["currency"].fillna("USD").replace("", "USD")
    combined["sheet_name"] = combined["sheet_name"].fillna("Без названия")
    combined["buyer"] = combined["buyer"].astype(str)
    combined["spend"] = combined["spend"].astype(float).round(2)
    if "total" in combined:
        combined["total"] = combined["total"].astype(float)

    sheet_options = sorted({sheet for sheet in combined["sheet_name"].unique() if isinstance(sheet, str)})
    selected_sheets = st.multiselect(
        "Листы / агентства",
        options=sheet_options,
        default=sheet_options,
        key="spend_sheet_filter",
    )

    buyer_options = sorted(combined["buyer"].unique())
    selected_buyers = st.multiselect(
        "Байеры (спенды)",
        options=buyer_options,
        default=buyer_options,
        key="spend_buyer_filter",
    )

    currency_options = sorted({cur for cur in combined["currency"].unique() if isinstance(cur, str)})
    if not currency_options:
        currency_options = ["USD"]
    selected_currencies = st.multiselect(
        "Валюты",
        options=currency_options,
        default=currency_options,
        key="spend_currency_filter",
    )

    sidebar.subheader("Курс спендов → $")
    sidebar.caption("Укажите, сколько долларов соответствует 1 единице валюты спенда.")
    spend_rates: dict[str, float] = {}
    for currency in currency_options:
        default_rate = 1.0
        spend_rates[currency] = sidebar.number_input(
            f"{currency} → USD",
            min_value=0.0,
            value=default_rate,
            step=0.01,
            format="%.4f",
            key=f"spend_rate_{currency}",
        )

    spend_rates.setdefault("USD", 1.0)

    filtered = combined[
        combined["sheet_name"].isin(selected_sheets)
        & combined["buyer"].isin(selected_buyers)
        & combined["currency"].isin(selected_currencies)
    ]

    if filtered.empty:
        st.warning("По выбранным фильтрам нет строк со спендами.")
        return

    filtered = filtered.copy()
    filtered["currency"] = filtered["currency"].fillna("USD")
    rate_series = filtered["currency"].map(spend_rates).fillna(1.0)
    filtered["spend_usd"] = (filtered["spend"] * rate_series).round(2)

    display_details = filtered.copy()
    display_details = display_details.rename(
        columns={
            "buyer": "Байер",
            "account_label": "Аккаунт / карта",
            "account_id": "ID аккаунта",
            "sheet_name": "Лист",
            "notes": "Комментарий",
            "currency": "Валюта",
            "source_file": "Файл",
            "spend_usd": "Спенд (USD)",
        }
    )
    id_is_empty = display_details["ID аккаунта"].astype(str).str.strip().isin({"", "nan", "None"})
    display_details.loc[id_is_empty, "ID аккаунта"] = display_details.loc[
        id_is_empty, "Аккаунт / карта"
    ]
    display_details = display_details[
        [
            "ID аккаунта",
            "Спенд (USD)",
            "Лист",
            "Комментарий",
            "Аккаунт / карта",
            "Байер",
            "Валюта",
            "Файл",
        ]
    ]
    display_details["Спенд (USD)"] = display_details["Спенд (USD)"].apply(lambda value: f"{value:,.2f}")
    display_details["Валюта"] = display_details["Валюта"].replace({"UNKNOWN": "—"})

    st.subheader("Детальная таблица спендов")
    st.dataframe(display_details, use_container_width=True)

    download_details = filtered.rename(
        columns={
            "sheet_name": "sheet",
            "buyer": "buyer",
            "account_label": "account_label",
            "account_id": "account_id",
            "spend": "spend",
            "spend_usd": "spend_usd",
            "currency": "currency",
            "notes": "notes",
            "source_file": "source",
        }
    )
    download_details = download_details[
        [
            "sheet",
            "buyer",
            "account_label",
            "account_id",
            "spend",
            "spend_usd",
            "currency",
            "notes",
            "source",
        ]
    ]
    create_download_button("Скачать детализацию спендов", download_details, "spends_details.csv")

    buyer_summary = (
        filtered.groupby("buyer", dropna=False)
        .agg(
            total_spend_usd=("spend_usd", "sum"),
            accounts=("account_id", "nunique"),
            rows=("account_id", "count"),
            sheets=("sheet_name", "nunique"),
        )
        .reset_index()
        .sort_values(["buyer"], na_position="last")
    )
    buyer_summary["buyer"] = buyer_summary["buyer"].fillna("")
    buyer_summary["total_spend_usd"] = buyer_summary["total_spend_usd"].round(2)

    buyer_display = buyer_summary.rename(
        columns={
            "buyer": "Байер",
            "total_spend_usd": "Сумма спендов (USD)",
            "accounts": "Аккаунтов",
            "rows": "Строк",
            "sheets": "Листов",
        }
    )
    buyer_display["Сумма спендов (USD)"] = buyer_display["Сумма спендов (USD)"].apply(lambda v: f"{v:,.2f}")

    st.subheader("Сводка по байерам")
    st.dataframe(buyer_display, use_container_width=True)
    create_download_button("Скачать сводку по байерам", buyer_summary, "spends_by_buyer.csv")

    sheet_summary = (
        filtered.groupby("sheet_name", dropna=False)
        .agg(
            total_spend_usd=("spend_usd", "sum"),
            buyers=("buyer", "nunique"),
            rows=("account_id", "count"),
        )
        .reset_index()
        .sort_values(["sheet_name"], na_position="last")
    )
    sheet_summary["sheet_name"] = sheet_summary["sheet_name"].fillna("Без названия")
    sheet_summary["total_spend_usd"] = sheet_summary["total_spend_usd"].round(2)

    sheet_display = sheet_summary.rename(
        columns={
            "sheet_name": "Лист",
            "total_spend_usd": "Сумма спендов (USD)",
            "buyers": "Байеров",
            "rows": "Строк",
        }
    )
    sheet_display["Сумма спендов (USD)"] = sheet_display["Сумма спендов (USD)"].apply(lambda v: f"{v:,.2f}")

    st.subheader("Сводка по листам")
    st.dataframe(sheet_display, use_container_width=True)
    create_download_button("Скачать сводку по листам", sheet_summary, "spends_by_sheet.csv")

    buyer_report_options = sorted(filtered["buyer"].unique())
    if buyer_report_options:
        st.subheader("Спенды выбранного байера по листам")
        buyer_for_lines = st.selectbox(
            "Выберите байера для детализации",
            options=buyer_report_options,
            key="spend_buyer_lines",
        )

        buyer_lines = filtered[filtered["buyer"] == buyer_for_lines].copy()

        if buyer_lines.empty:
            st.info("Для выбранного байера нет строк после применения фильтров.")
        else:
            buyer_detail = buyer_lines.rename(
                columns={
                    "sheet_name": "Лист",
                    "account_id": "ID аккаунта",
                    "account_label": "Аккаунт / карта",
                    "spend_usd": "Спенд (USD)",
                    "currency": "Валюта",
                    "notes": "Комментарий",
                    "source_file": "Файл",
                }
            )[
                [
                    "ID аккаунта",
                    "Спенд (USD)",
                    "Лист",
                    "Комментарий",
                    "Аккаунт / карта",
                    "Валюта",
                    "Файл",
                ]
            ]

            detail_id_empty = buyer_detail["ID аккаунта"].astype(str).str.strip().isin({"", "nan", "None"})
            buyer_detail.loc[detail_id_empty, "ID аккаунта"] = buyer_detail.loc[
                detail_id_empty, "Аккаунт / карта"
            ]
            buyer_detail = buyer_detail.sort_values(["Лист", "Аккаунт / карта", "ID аккаунта"])

            buyer_detail_with_total = append_total_row(
                buyer_detail,
                "Сумма",
                ["Спенд (USD)"],
            )

            buyer_detail_display = buyer_detail_with_total.copy()
            buyer_detail_display["Спенд (USD)"] = buyer_detail_display["Спенд (USD)"].apply(
                lambda value: "" if pd.isna(value) else f"{value:,.2f}"
            )

            st.dataframe(buyer_detail_display, use_container_width=True)
            create_download_button(
                f"Скачать строки — {buyer_for_lines}",
                buyer_detail_with_total,
                f"spends_{buyer_for_lines}.csv",
            )

            sheet_breakdown = (
                buyer_lines.groupby(["sheet_name", "currency"], dropna=False)
                .agg(
                    total_spend_usd=("spend_usd", "sum"),
                    accounts=("account_id", "nunique"),
                    rows=("account_id", "count"),
                )
                .reset_index()
            )

            if not sheet_breakdown.empty:
                sheet_breakdown = sheet_breakdown.rename(
                    columns={
                        "sheet_name": "Лист",
                        "currency": "Валюта",
                        "total_spend_usd": "Сумма спендов (USD)",
                        "accounts": "Аккаунтов",
                        "rows": "Строк",
                    }
                )
                sheet_breakdown["Сумма спендов (USD)"] = sheet_breakdown["Сумма спендов (USD)"].round(2)

                sheet_breakdown_display = sheet_breakdown.copy()
                sheet_breakdown_display["Сумма спендов (USD)"] = sheet_breakdown_display[
                    "Сумма спендов (USD)"
                ].apply(lambda value: f"{value:,.2f}")

                st.markdown("**Разрез по листам**")
                st.dataframe(sheet_breakdown_display, use_container_width=True)
                create_download_button(
                    f"Скачать разрез по листам — {buyer_for_lines}",
                    sheet_breakdown,
                    f"spends_by_sheet_{buyer_for_lines}.csv",
                )

    sidebar.subheader("Метрики по спендам")
    sidebar.metric("Сумма спендов (USD)", f"{filtered['spend_usd'].sum():,.2f}")
    sidebar.metric("Байеров", int(filtered["buyer"].nunique()))
    sidebar.metric("Листов", int(filtered["sheet_name"].nunique()))


def render_combined_module(sidebar: DeltaGenerator) -> None:
    st.subheader("🧩 Объединённый отчёт по байерам")
    st.caption(
        "Загрузите файлы со сверками, расходами и спендами — приложение покажет все три блока сразу по выбранному байеру."
    )

    reconciliation_files = st.file_uploader(
        "Сверки (CSV/XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="combined_reconciliation_files",
    )
    expenses_files = st.file_uploader(
        "Расходники (CSV/XLSX)",
        type=["csv", "xlsx"],
        accept_multiple_files=True,
        key="combined_expenses_files",
    )
    spend_files = st.file_uploader(
        "Спенды (XLSX/CSV)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        key="combined_spend_files",
    )

    if not reconciliation_files and not expenses_files and not spend_files:
        st.info("Чтобы построить отчёт, загрузите хотя бы один файл со сверками, расходами или спендами.")
        return

    reconciliation_frames: list[pd.DataFrame] = []
    reconciliation_errors: list[str] = []

    if reconciliation_files:
        for uploaded in reconciliation_files:
            partner_program = infer_partner_program(uploaded.name)
            try:
                details = load_reconciliation_file(uploaded, partner_program, source_name=uploaded.name)
            except Exception as exc:  # pragma: no cover - сообщение пользователю
                reconciliation_errors.append(f"{uploaded.name}: {exc}")
                continue

            if details.empty:
                reconciliation_errors.append(f"{uploaded.name}: не удалось извлечь данные")
                continue

            reconciliation_frames.append(details)

    if reconciliation_errors:
        st.warning("\n".join(reconciliation_errors))

    if reconciliation_frames:
        reconciliation_details = pd.concat(reconciliation_frames, ignore_index=True)
    else:
        reconciliation_details = pd.DataFrame()

    expense_frames: list[pd.DataFrame] = []
    expense_errors: list[str] = []

    if expenses_files:
        for uploaded in expenses_files:
            try:
                details = load_expenses_file(uploaded, source_name=uploaded.name)
            except Exception as exc:  # pragma: no cover - сообщение пользователю
                expense_errors.append(f"{uploaded.name}: {exc}")
                continue

            if details.empty:
                expense_errors.append(f"{uploaded.name}: не удалось извлечь данные")
                continue

            expense_frames.append(details)

    if expense_errors:
        st.warning("\n".join(expense_errors))

    if expense_frames:
        expense_details = pd.concat(expense_frames, ignore_index=True)
        expense_details["amount"] = expense_details["amount"].astype(float).round(2)
    else:
        expense_details = pd.DataFrame()

    spend_frames: list[pd.DataFrame] = []
    spend_errors: list[str] = []

    if spend_files:
        for uploaded in spend_files:
            suffix = Path(uploaded.name).suffix.lower()
            try:
                if suffix in {".xlsx", ".xls"}:
                    frame = load_spend_workbook(uploaded, source_name=uploaded.name)
                else:
                    sheet_name = Path(uploaded.name).stem
                    frame = load_spend_sheet(uploaded, sheet_name=sheet_name, source_name=uploaded.name)
            except Exception as exc:  # pragma: no cover - сообщение пользователю
                spend_errors.append(f"{uploaded.name}: {exc}")
                continue

            warnings = getattr(frame, "attrs", {}).get("warnings", [])
            if warnings:
                spend_errors.extend(f"{uploaded.name}: {warning}" for warning in warnings)
                frame = frame.copy()
                try:
                    del frame.attrs["warnings"]
                except KeyError:
                    pass

            if frame.empty:
                spend_errors.append(f"{uploaded.name}: не удалось извлечь данные")
                continue

            spend_frames.append(frame)

    if spend_errors:
        st.warning("\n".join(spend_errors))

    if spend_frames:
        spend_details = pd.concat(spend_frames, ignore_index=True)
        spend_details["currency"] = spend_details["currency"].fillna("USD").replace("", "USD")
        spend_details["sheet_name"] = spend_details["sheet_name"].fillna("Без названия")
        spend_details["buyer"] = spend_details["buyer"].astype(str)
        spend_details["spend"] = spend_details["spend"].astype(float).round(2)
    else:
        spend_details = pd.DataFrame()

    buyer_options = set()
    for df in (reconciliation_details, expense_details, spend_details):
        if not df.empty and "buyer" in df.columns:
            buyer_options.update(df["buyer"].astype(str).str.strip())

    buyer_options = sorted({buyer for buyer in buyer_options if buyer})

    if not buyer_options:
        st.warning("Нет доступных байеров. Загрузите корректные файлы со сверками, расходами или спендами.")
        return

    currency_candidates: set[str] = set()
    if not reconciliation_details.empty and "currency" in reconciliation_details.columns:
        currency_candidates.update(
            reconciliation_details["currency"].dropna().astype(str).str.upper().replace({"", "NAN"}, "USD")
        )
    if not spend_details.empty and "currency" in spend_details.columns:
        currency_candidates.update(
            spend_details["currency"].dropna().astype(str).str.upper().replace({"", "NAN"}, "USD")
        )

    currency_candidates = {cur for cur in currency_candidates if cur and cur != "UNKNOWN"}
    if not currency_candidates:
        currency_candidates = {"USD"}

    sidebar.subheader("Курсы валют → $")
    sidebar.caption("Введите курс: сколько долларов соответствует 1 единице выбранной валюты.")
    exchange_rates: dict[str, float] = {}
    for currency in sorted(currency_candidates):
        exchange_rates[currency] = sidebar.number_input(
            f"{currency} → USD",
            min_value=0.0,
            value=1.0,
            step=0.01,
            format="%.4f",
            key=f"combined_rate_{currency}",
        )

    exchange_rates.setdefault("USD", 1.0)

    selected_buyer = st.selectbox(
        "Выберите байера",
        options=buyer_options,
        key="combined_buyer_select",
    )

    st.subheader("Сверки")
    if reconciliation_details.empty:
        st.info("Файлы сверок не загружены или не содержат данных.")
    else:
        buyer_reconciliation = reconciliation_details[
            reconciliation_details["buyer"].astype(str) == selected_buyer
        ].copy()

        if buyer_reconciliation.empty:
            st.info("Для выбранного байера нет строк в загруженных сверках.")
        else:
            buyer_reconciliation["currency"] = (
                buyer_reconciliation["currency"].fillna("USD").replace("", "USD").astype(str).str.upper()
            )

            buyer_summary = aggregate_by_partner_and_buyer(buyer_reconciliation)
            buyer_summary["currency"] = (
                buyer_summary["currency"].fillna("USD").replace("", "USD").astype(str).str.upper()
            )
            buyer_summary["conversion_rate"] = buyer_summary["currency"].apply(
                lambda cur: resolve_rate(cur, exchange_rates)
            )
            buyer_summary["payout_usd"] = (buyer_summary["payout"] * buyer_summary["conversion_rate"]).round(2)
            buyer_summary["chargeback_amount_usd"] = (
                buyer_summary["chargeback_amount"] * buyer_summary["conversion_rate"]
            ).round(2)
            buyer_summary["net_payout_usd"] = (
                buyer_summary["net_payout"] * buyer_summary["conversion_rate"]
            ).round(2)

            offers_table = buyer_summary[
                [
                    "partner_program",
                    "deposits",
                    "payout",
                    "payout_usd",
                    "net_payout",
                    "net_payout_usd",
                ]
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
            st.dataframe(offers_table, use_container_width=True)
            create_download_button(
                f"Скачать сверки — {selected_buyer}",
                offers_table,
                f"reconciliation_{selected_buyer}.csv",
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

            if chargebacks_table.empty:
                st.markdown("_Чарджбеков для выбранного байера нет._")
            else:
                chargebacks_table["Чарджбеков"] = chargebacks_table["Чарджбеков"].astype(int)
                chargebacks_table = append_total_row(
                    chargebacks_table,
                    "Сумма",
                    ["Чарджбеков", "Чардж в валюте ПП", "Чардж в $$"],
                )

                st.markdown("**Чарджбеки**")
                st.dataframe(chargebacks_table, use_container_width=True)
                create_download_button(
                    f"Скачать чарджбеки — {selected_buyer}",
                    chargebacks_table,
                    f"reconciliation_chargebacks_{selected_buyer}.csv",
                )

    st.subheader("Расходники")
    if expense_details.empty:
        st.info("Файлы расходников не загружены или не содержат данных.")
    else:
        buyer_expenses = expense_details[expense_details["buyer"].astype(str) == selected_buyer].copy()

        if buyer_expenses.empty:
            st.info("Для выбранного байера нет строк в загруженных расходниках.")
        else:
            buyer_expense_table = buyer_expenses.rename(
                columns={
                    "expense_type": "Тип расхода",
                    "item_count": "Кол-во",
                    "amount": "Сумма, $",
                    "notes": "Комментарий",
                    "source_file": "Источник",
                }
            )[["Тип расхода", "Сумма, $", "Кол-во", "Комментарий", "Источник"]]
            buyer_expense_table = append_total_row(
                buyer_expense_table,
                "Сумма",
                ["Кол-во", "Сумма, $"],
            )

            st.dataframe(buyer_expense_table, use_container_width=True)
            create_download_button(
                f"Скачать расходник — {selected_buyer}",
                buyer_expense_table,
                f"expenses_{selected_buyer}.csv",
            )

    st.subheader("Спенды")
    if spend_details.empty:
        st.info("Файлы со спендами не загружены или не содержат данных.")
    else:
        spend_details = spend_details.copy()
        rate_series = spend_details["currency"].map(exchange_rates).fillna(1.0)
        spend_details["spend_usd"] = (spend_details["spend"] * rate_series).round(2)

        buyer_spends = spend_details[spend_details["buyer"].astype(str) == selected_buyer].copy()

        if buyer_spends.empty:
            st.info("Для выбранного байера нет строк в загруженных спендах.")
        else:
            buyer_detail = buyer_spends.rename(
                columns={
                    "sheet_name": "Лист",
                    "account_id": "ID аккаунта",
                    "account_label": "Аккаунт / карта",
                    "spend_usd": "Спенд (USD)",
                    "currency": "Валюта",
                    "notes": "Комментарий",
                    "source_file": "Файл",
                }
            )[
                [
                    "ID аккаунта",
                    "Спенд (USD)",
                    "Лист",
                    "Комментарий",
                    "Аккаунт / карта",
                    "Валюта",
                    "Файл",
                ]
            ]

            id_missing = buyer_detail["ID аккаунта"].astype(str).str.strip().isin({"", "nan", "None"})
            buyer_detail.loc[id_missing, "ID аккаунта"] = buyer_detail.loc[
                id_missing, "Аккаунт / карта"
            ]
            buyer_detail = buyer_detail.sort_values(["Лист", "Аккаунт / карта", "ID аккаунта"])

            buyer_detail_with_total = append_total_row(
                buyer_detail,
                "Сумма",
                ["Спенд (USD)"],
            )

            buyer_detail_display = buyer_detail_with_total.copy()
            buyer_detail_display["Спенд (USD)"] = buyer_detail_display["Спенд (USD)"].apply(
                lambda value: "" if pd.isna(value) else f"{value:,.2f}"
            )

            st.dataframe(buyer_detail_display, use_container_width=True)
            create_download_button(
                f"Скачать спенды — {selected_buyer}",
                buyer_detail_with_total,
                f"spends_{selected_buyer}.csv",
            )

            sheet_breakdown = (
                buyer_spends.groupby(["sheet_name", "currency"], dropna=False)
                .agg(
                    total_spend_usd=("spend_usd", "sum"),
                    accounts=("account_id", "nunique"),
                    rows=("account_id", "count"),
                )
                .reset_index()
            )

            if sheet_breakdown.empty:
                st.markdown("_Данных по листам нет._")
            else:
                sheet_breakdown = sheet_breakdown.rename(
                    columns={
                        "sheet_name": "Лист",
                        "currency": "Валюта",
                        "total_spend_usd": "Сумма спендов (USD)",
                        "accounts": "Аккаунтов",
                        "rows": "Строк",
                    }
                )
                sheet_breakdown["Сумма спендов (USD)"] = sheet_breakdown["Сумма спендов (USD)"].round(2)

                sheet_breakdown_display = sheet_breakdown.copy()
                sheet_breakdown_display["Сумма спендов (USD)"] = sheet_breakdown_display[
                    "Сумма спендов (USD)"
                ].apply(lambda value: f"{value:,.2f}")

                st.markdown("**Разрез по листам**")
                st.dataframe(sheet_breakdown_display, use_container_width=True)
                create_download_button(
                    f"Скачать разрез по листам — {selected_buyer}",
                    sheet_breakdown,
                    f"spends_by_sheet_{selected_buyer}.csv",
                )

st.set_page_config(page_title="ZP Calculator", layout="wide")

sidebar = st.sidebar
module = sidebar.radio("Выберите модуль", ["Сверки", "Расходники", "Спенды", "Объединение"], index=0)

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
elif module == "Расходники":
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
elif module == "Спенды":
    sidebar.header("Правила импорта спендов")
    sidebar.markdown(
        """
        * Принимаются Excel-файлы (XLSX/XLS) или CSV-листы, один файл = один лист.
        * Обязательные колонки: **Байер**, **Имя аккаунта** (или **Карта**), **ID аккаунта**, **Spend**.
        * Дополнительные колонки сохраняются в комментариях.
        * Листы с названием `DTD` учитываются в EUR, остальные — в USD.
        * Курс каждой валюты к доллару можно задать в сайдбаре; отчёты показываются в USD.
        """
    )
    st.title("ZP Calculator — Спенды")
    render_spend_module(sidebar)
else:
    sidebar.header("Комбинированный отчёт по байеру")
    sidebar.markdown(
        """
        * Загрузите файлы со сверками, расходниками и спендами.
        * Поддерживаются форматы: CSV/XLSX для сверок и расходников, XLSX/XLS/CSV для спендов.
        * Все суммы в отчёте приводятся к USD по указанным в сайдбаре курсам.
        * После загрузки выберите одного байера, чтобы увидеть три блока: сверки, расходы и спенды.
        """
    )
    st.title("ZP Calculator — Объединение")
    render_combined_module(sidebar)
