from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from zp_calculator.data_processing import (
    aggregate_by_partner_and_buyer,
    aggregate_overall,
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

st.set_page_config(
    page_title="ZP Calculator",
    layout="wide",
)

st.title("🧾 Сводная по байерам из файлов сверок")
st.caption(
    "Загрузите один или несколько CSV-файлов сверок, чтобы получить агрегированную таблицу по байерам."
)

with st.sidebar:
    st.header("Правила импорта")
    st.markdown(
        """
        * Принимаются файлы в формате CSV (UTF-8).
        * Из данных используются колонки **Buyer**, **Commision/Commission Type**, **Сумма по полю FTD Count**, **Сумма по полю Payout**.
        * "Всего" и "Итого" строки игнорируются — totals считаются заново.
        """
    )

uploaded_files = st.file_uploader(
    "Загрузите CSV-файлы", type=["csv"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Добавьте один или несколько CSV-файлов, чтобы увидеть результат.")
    st.stop()

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
    st.stop()

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

st.sidebar.subheader("Курсы валют → $")
st.sidebar.caption("Введите, сколько долларов соответствует 1 единице валюты партнёрской программы.")
exchange_rates: dict[str, float] = {}
for currency in available_currencies:
    default_rate = 1.0 if currency == "USD" else 1.0
    exchange_rates[currency] = st.sidebar.number_input(
        f"{currency} → USD",
        min_value=0.0,
        value=float(default_rate),
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
    st.stop()

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
    else:
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
