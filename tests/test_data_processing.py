from __future__ import annotations

import csv
import io
import math

import pytest
import pandas as pd

from zp_calculator.data_processing import (
    aggregate_by_partner_and_buyer,
    aggregate_overall,
    aggregate_expenses_by_buyer,
    aggregate_expenses_totals,
    aggregate_spend_by_buyer,
    aggregate_spend_by_sheet,
    load_expenses_file,
    load_reconciliation_file,
    load_spend_sheet,
    load_spend_workbook,
)



def test_load_reconciliation_file_parses_commission_rows():
    csv_content = (
        "Тотал ПП,,,,,\n"
        "Buyer,Commision Type,Сумма по полю FTD Count,Сумма по полю Payout,,\n"
        "Vladyslav,Chargeback,- 4 ,- 720,,,\n"
        ",CPA,  84 ,  15\u00A0200,,,\n"
        "Всего (Vladyslav),,  80 ,  14\u00A0480,,,\n"
        "Итого,,  432 ,  78\u00A0910,,,\n"
    )

    data = load_reconciliation_file(io.StringIO(csv_content), "Rock", source_name="file.csv")

    assert len(data) == 2
    assert set(data["buyer"]) == {"Vladyslav"}
    chargeback_row = data[data["commission_type"].str.contains("Chargeback", case=False)].iloc[0]
    cpa_row = data[data["commission_type"].str.contains("CPA", case=False)].iloc[0]

    assert chargeback_row["ftd_count"] == -4
    assert chargeback_row["payout"] == -720.0
    assert bool(chargeback_row["is_chargeback"])

    assert cpa_row["ftd_count"] == 84
    assert cpa_row["payout"] == 15200.0
    assert not bool(cpa_row["is_chargeback"])


def test_load_reconciliation_file_reads_header_not_in_first_column():
    rows = [
        ["", "Buyer", "Commision Type", "Сумма по полю FTD Count", "Сумма по полю Payout"],
        ["", "Vladyslav", "Chargeback", "- 4 ", "- 720"],
        ["", "", "CPA", "  84 ", "  15\u00A0200"],
        ["", "Всего (Vladyslav)", "", "  80 ", "  14\u00A0480"],
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сверка", index=False, header=False)

    buffer.seek(0)
    data = load_reconciliation_file(buffer, "Rock", source_name="file.xlsx")

    assert len(data) == 2
    assert set(data["buyer"]) == {"Vladyslav"}


def test_load_reconciliation_file_handles_buyer_name_header():
    rows = [
        ["", "Buyer Name", "Commision Type", "FTD", "Payout"],
        ["", "Vladyslav", "CPA", "84", "15200"],
        ["", "Всего (Vladyslav)", "", "84", "15200"],
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Sheet1", index=False, header=False)

    buffer.seek(0)
    data = load_reconciliation_file(buffer, "Referon", source_name="referon.xlsx")

    assert not data.empty
    assert set(data["buyer"]) == {"Vladyslav"}


def test_load_reconciliation_file_handles_multi_row_header():
    rows = [
        ["", "Buyer", "Commission", "FTD", ""],
        ["", "Name", "Type", "Count", "Payout"],
        ["", "Vladyslav", "CPA", "84", "15200"],
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Sheet1", index=False, header=False)

    buffer.seek(0)
    data = load_reconciliation_file(buffer, "Wowpartners", source_name="wow.xlsx")

    assert not data.empty
    assert set(data["buyer"]) == {"Vladyslav"}


def test_load_reconciliation_file_handles_nbsp_header_cells():
    rows = [
        [
            "\u00A0Buyer",
            "\u00A0Сумма по полю FTD Count",
            "\u00A0Сумма по полю Payout",
        ],
        ["EgorVarivonchik", "38", "7 600"],
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сводная", index=False, header=False)

    buffer.seek(0)
    data = load_reconciliation_file(buffer, "Wowpartners", source_name="wow_nbsp.xlsx")

    assert not data.empty
    assert set(data["buyer"]) == {"EgorVarivonchik"}


def test_load_reconciliation_file_handles_bom_header_cells():
    rows = [
        [
            "\ufeffBuyer",
            "\ufeffСумма по полю FTD Count",
            "\ufeffСумма по полю Payout",
        ],
        ["Tatiana", "30", "6 960"],
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сводная", index=False, header=False)

    buffer.seek(0)
    data = load_reconciliation_file(buffer, "Wowpartners", source_name="wow_bom.xlsx")

    assert not data.empty
    assert set(data["buyer"]) == {"Tatiana"}


def test_load_reconciliation_file_skips_summary_rows():
    csv_content = (
        "Buyer,Commission Type,Сумма по полю FTD Count,Сумма по полю Payout\n"
        "Tatiana,CPA,30,6 960\n"
        "Всего (Tatiana),,30,6 960\n"
        "Итого,,30,6 960\n"
    )

    details = load_reconciliation_file(io.StringIO(csv_content), "Wowpartners", source_name="summary.csv")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Tatiana"
    assert row["ftd_count"] == 30
    assert row["payout"] == pytest.approx(6960.0)


def test_load_reconciliation_file_skips_suffix_summary_labels():
    csv_content = (
        "Buyer,Commission Type,Сумма по полю FTD Count,Сумма по полю Payout\n"
        "Tatiana Итог,CPA,30,6 960\n"
        "Tatiana,CPA,30,6 960\n"
    )

    details = load_reconciliation_file(io.StringIO(csv_content), "Starcrown", source_name="suffix.csv")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Tatiana"
    assert row["ftd_count"] == 30
    assert row["payout"] == pytest.approx(6960.0)


def test_aggregations_compute_totals_correctly():
    csv_content = (
        "Сумма ПП,,,\n"
        "Buyer,Commission Type,Сумма по полю FTD Count,Сумма по полю Payout\n"
        "Dmytro,Chargeback,- 2 ,- 360\n"
        ",CPA (Netherlands),  65 ,  11\u00A0760\n"
        "Всего (Dmytro),,  63 ,  11\u00A0400\n"
        "Итого,,  41 ,  12\u00A0300\n"
    )

    details = load_reconciliation_file(io.StringIO(csv_content), "Maroon", source_name="file2.csv")
    summary = aggregate_by_partner_and_buyer(details)

    assert len(summary) == 1
    row = summary.iloc[0]
    assert row["deposits"] == 65
    assert row["chargebacks"] == 2
    assert row["net_deposits"] == 63
    assert row["payout"] == 11760.0
    assert row["chargeback_amount"] == 360.0
    assert row["net_payout"] == 11400.0

    overall = aggregate_overall(summary)
    assert len(overall) == 1
    overall_row = overall.iloc[0]
    assert overall_row["deposits"] == 65
    assert overall_row["chargebacks"] == 2
    assert overall_row["net_deposits"] == 63
    assert overall_row["net_payout"] == 11400.0
    assert "Maroon" in overall_row["partner_program"]


def test_load_reconciliation_file_skips_rows_without_numeric_values():
    csv_content = (
        "Buyer,Commission Type,Сумма по полю FTD Count,Сумма по полю Payout\n"
        "Dmytro,CPA,sale,800\n"
        "Dmytro,CPA,4,800\n"
    )

    details = load_reconciliation_file(io.StringIO(csv_content), "Starcrown", source_name="non_numeric.csv")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Dmytro"
    assert row["ftd_count"] == 4
    assert row["payout"] == pytest.approx(800.0)


def test_load_reconciliation_file_header_after_long_preamble():
    rows = [
        ["", "", "", "", "Тотал ПП", "", "4", ""],
        ["", "", "", "", "Сумма ПП", "", "800", "eur"],
        ["", "", "", "", "Тотал Кейтаро", "", "4", ""],
        ["", "", "", "", "Сумма Кейтаро", "", "935,81", "usd"],
        ["", "", "", "", "Chargeback", "", "-", ""],
        ["Buyer", "FTD Count", "Сумма по полю Payout", "", "", "", "", ""],
        ["Dmytro", "4", "800", "", "", "", "", ""],
        ["Итого", "4", "800", "", "", "", "", ""],
    ]

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name="Сводная по байерам", index=False, header=False)

    buffer.seek(0)
    buffer.name = "starcrown_affilka.xlsx"
    details = load_reconciliation_file(buffer, "Starcrown", source_name="starcrown_affilka.xlsx")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Dmytro"
    assert row["ftd_count"] == 4
    assert row["payout"] == pytest.approx(800.0)


def test_load_reconciliation_file_prefers_sheet_named_svodnaya():
    front_sheet = pd.DataFrame(
        [
            ["Тотал ПП", "4"],
            ["Сумма ПП", "800"],
        ]
    )
    target_sheet = pd.DataFrame(
        [
            ["Buyer", "FTD Count", "Сумма по полю Payout"],
            ["Dmytro", "4", "800"],
        ]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        front_sheet.to_excel(writer, sheet_name="report_28_2025-10-09", index=False, header=False)
        target_sheet.to_excel(writer, sheet_name="Сводная по байерам", index=False, header=False)

    buffer.seek(0)
    buffer.name = "starcrown_affilka.xlsx"
    details = load_reconciliation_file(buffer, "Starcrown", source_name="starcrown_affilka.xlsx")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Dmytro"
    assert row["ftd_count"] == 4
    assert row["payout"] == pytest.approx(800.0)


def test_load_handles_missing_commission_column():
    csv_content = (
        "Buyer,Сумма по полю FTD Count,Сумма по полю Payout\n"
        "Diana,41,7 380\n"
        "Всего (Diana),42,7 560\n"
    )

    details = load_reconciliation_file(io.StringIO(csv_content), "Starcrown", source_name="file3.csv")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Diana"
    assert row["commission_type"] == "Unknown"
    assert row["ftd_count"] == 41
    assert row["payout"] == 7380.0

    summary = aggregate_by_partner_and_buyer(details)
    summary_row = summary.iloc[0]
    assert summary_row["deposits"] == 41
    assert summary_row["payout"] == 7380.0
    assert summary_row["net_payout"] == 7380.0


def test_load_expenses_file_and_aggregate():
    csv_content = (
        ",Фанки,,,,Рекламные аккаунты,,\n"
        ",Байер,Кол-во,Сумма,,Байер,Кол-во,Сумма,\n"
        ",Arseniy Simich,3,\"$359,00\",,Arseniy Simich,1,\"$45,00\",\n"
        ",Итого:,3,\"$359,00\",,Итого:,1,\"$45,00\",\n"
        ",Прокси,,,\n"
        ",Байер,Кол-во,Сумма,\n"
        ",Arseniy Simich,2,\"$12,00\",\n"
        ",Итого:,2,\"$12,00\",\n"
    )

    data = load_expenses_file(io.StringIO(csv_content), source_name="expenses.csv")

    assert len(data) == 3
    assert set(data["expense_type"]) == {"Фанки", "Рекламные аккаунты", "Прокси"}

    grouped = aggregate_expenses_by_buyer(data)
    assert len(grouped) == 3
    funk_row = grouped[grouped["expense_type"] == "Фанки"].iloc[0]
    assert funk_row["item_count"] == 3
    assert funk_row["amount"] == 359.0

    totals = aggregate_expenses_totals(data)
    assert len(totals) == 1
    total_row = totals.iloc[0]
    assert total_row["buyer"] == "Arseniy Simich"
    assert total_row["total_amount"] == 416.0


def test_load_spend_sheet_parses_accounts_and_currency():
    csv_content = (
        "Байер,account name,account id,spending,total,,\n"
        "Банк,ULPWU US-Under 0909-8 01,/ 1177972760761934,\"$6,66\",\"$273 906,75\",/ ,/ / 1177972760761934,\n"
        "AleksandrRamanovich ,ULPWU US-Under 0910-8 03,/ 1234209211769191,\"$5,63\",,/ ,/ / 1234209211769191,\n"
        "Итого,,,,,,\n"
    )

    details = load_spend_sheet(io.StringIO(csv_content), sheet_name="UL", source_name="ul.csv")

    assert len(details) == 2
    first_row = details.iloc[0]
    assert first_row["buyer"] == "Банк"
    assert first_row["account_id"] == "1177972760761934"
    assert first_row["currency"] == "USD"
    assert first_row["spend"] == 6.66
    assert first_row["total"] == 273906.75
    assert first_row["sheet_name"] == "UL"
    assert first_row["source_file"] == "ul.csv"

    grouped = aggregate_spend_by_buyer(details)
    assert len(grouped) == 2
    bank_row = grouped[grouped["buyer"] == "Банк"].iloc[0]
    assert bank_row["total_spend"] == 6.66
    assert bank_row["currency"] == "USD"

    sheet_summary = aggregate_spend_by_sheet(details)
    assert len(sheet_summary) == 1
    summary_row = sheet_summary.iloc[0]
    assert summary_row["sheet_name"] == "UL"
    assert summary_row["total_spend"] == 12.29


def test_load_spend_sheet_skips_comment_rows():
    csv_content = (
        "Байер ,Имя аккаунта,id аккаунта,spend,total\n"
        "банк,ULLIT US-EP-AC Underdog 0828-8,1006452728611052,\"$2,40\",\"$7,53\"\n"
        "банк,ULGP US-EP-AdCow-Underdog 0903-8 01,1048295084029712,\"$5,13\",\n"
        ",,,,\n"
        ",\"Мы его только прогревали, его заказали только с октября\",, ,\n"
    )

    details = load_spend_sheet(io.StringIO(csv_content), sheet_name="AdCow", source_name="adcow.csv")

    assert len(details) == 2
    assert set(details["buyer"]) == {"Банк"}
    assert details["spend"].sum() == pytest.approx(7.53)
    assert details["currency"].unique().tolist() == ["USD"]


def test_load_spend_sheet_merges_buyer_case():
    csv_content = (
        "Байер,Имя аккаунта,id аккаунта,spend,total\n"
        "Банк,UL-1,/ 123,\"$10,00\",\n"
        "банк,UL-2,/ 456,\"$5,00\",\n"
    )

    details = load_spend_sheet(io.StringIO(csv_content), sheet_name="UL", source_name="ul.csv")

    assert details["buyer"].tolist() == ["Банк", "Банк"]

    grouped = aggregate_spend_by_buyer(details)
    assert len(grouped) == 1
    row = grouped.iloc[0]
    assert row["buyer"] == "Банк"
    assert row["total_spend"] == pytest.approx(15.0)


def test_load_spend_sheet_notes_non_numeric_total():
    csv_content = (
        "Байер,account name,account id,spend,total,comment\n"
        "Vladyslav Serhiienko,ULFUN-Under 0604-8 02,/ 1891469591589462,\"$5 242,59\",SiniavinOleh,правильно Олег\n"
    )

    details = load_spend_sheet(io.StringIO(csv_content), sheet_name="UL", source_name="ul.csv")

    assert len(details) == 1
    row = details.iloc[0]
    assert row["buyer"] == "Vladyslav Serhiienko"
    assert math.isnan(row["total"])
    assert "TOTAL:SiniavinOleh" in row["notes"]


def test_load_spend_workbook_collects_warnings():
    valid = pd.DataFrame(
        {
            "Байер": ["Банк"],
            "Имя аккаунта": ["UL-1"],
            "id аккаунта": ["/ 123"],
            "spend": ["$10,00"],
        }
    )
    junk = pd.DataFrame({"Unnamed: 0": ["visa7918", ""]})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        valid.to_excel(writer, sheet_name="Valid", index=False)
        junk.to_excel(writer, sheet_name="Лист14", index=False)

    buffer.seek(0)
    workbook_df = load_spend_workbook(buffer, source_name="test.xlsx")

    warnings = workbook_df.attrs.get("warnings", [])
    assert any("Лист14" in warning for warning in warnings)
    assert not workbook_df.empty
    assert "Valid" in workbook_df["sheet_name"].values


def test_load_reconciliation_file_reads_first_sheet_from_xlsx():
    csv_content = (
        "Тотал ПП,,,,,\n"
        "Buyer,Commision Type,Сумма по полю FTD Count,Сумма по полю Payout,,\n"
        "Vladyslav,Chargeback,- 4 ,- 720,,\n"
        ",CPA,  84 ,  15\u00A0200,,\n"
        "Всего (Vladyslav),,  80 ,  14\u00A0480,,\n"
    )

    rows = list(csv.reader(io.StringIO(csv_content)))
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    frame = pd.DataFrame(normalized_rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Сверка", index=False, header=False)
        pd.DataFrame([["Buyer", "Ignored"], ["Someone", "Value"]]).to_excel(
            writer, sheet_name="Другой", index=False, header=False
        )

    buffer.seek(0)
    data = load_reconciliation_file(buffer, "Rock", source_name="file.xlsx")

    assert len(data) == 2
    assert set(data["buyer"]) == {"Vladyslav"}
    chargeback_row = data[data["commission_type"].str.contains("Chargeback", case=False)].iloc[0]
    cpa_row = data[data["commission_type"].str.contains("CPA", case=False)].iloc[0]

    assert chargeback_row["ftd_count"] == -4
    assert chargeback_row["payout"] == -720.0
    assert bool(chargeback_row["is_chargeback"])

    assert cpa_row["ftd_count"] == 84
    assert cpa_row["payout"] == 15200.0
    assert not bool(cpa_row["is_chargeback"])


def test_load_reconciliation_file_scans_all_excel_sheets_for_headers():
    first_sheet = pd.DataFrame(
        [
            ["", "", "", ""],
            ["Всего", "", "", ""],
        ]
    )
    second_sheet = pd.DataFrame(
        [
            ["", "Buyer", "Commision Type", "Сумма по полю FTD Count", "Сумма по полю Payout"],
            ["", "Vladyslav", "CPA", "84", "15200"],
        ]
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        first_sheet.to_excel(writer, sheet_name="Summary", index=False, header=False)
        second_sheet.to_excel(writer, sheet_name="Data", index=False, header=False)

    buffer.seek(0)
    buffer.name = "wowpartners.xlsx"
    data = load_reconciliation_file(buffer, "Wowpartners", source_name="wowpartners.xlsx")

    assert not data.empty
    assert set(data["buyer"]) == {"Vladyslav"}
def test_load_expenses_file_reads_first_sheet_from_xlsx():
    csv_content = (
        ",Фанки,,,,Рекламные аккаунты,,\n"
        ",Байер,Кол-во,Сумма,,Байер,Кол-во,Сумма,\n"
        ",Arseniy Simich,3,\"$359,00\",,Arseniy Simich,1,\"$45,00\",\n"
        ",Итого:,3,\"$359,00\",,Итого:,1,\"$45,00\",\n"
        ",Прокси,,,,\n"
        ",Байер,Кол-во,Сумма,\n"
        ",Arseniy Simich,2,\"$12,00\",\n"
        ",Итого:,2,\"$12,00\",\n"
    )

    rows = list(csv.reader(io.StringIO(csv_content)))
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    frame = pd.DataFrame(normalized_rows)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Расходы", index=False, header=False)
        pd.DataFrame([["Байер", "Кол-во", "Сумма"], ["Иван", 1, "$10"]]).to_excel(
            writer, sheet_name="Другой", index=False, header=False
        )

    buffer.seek(0)
    data = load_expenses_file(buffer, source_name="expenses.xlsx")

    assert len(data) == 3
    assert set(data["expense_type"]) == {"Фанки", "Рекламные аккаунты", "Прокси"}

    grouped = aggregate_expenses_by_buyer(data)
    funk_row = grouped[grouped["expense_type"] == "Фанки"].iloc[0]
    assert funk_row["item_count"] == 3
    assert funk_row["amount"] == 359.0
