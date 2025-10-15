from __future__ import annotations

import io

from zp_calculator.data_processing import (
    aggregate_by_partner_and_buyer,
    aggregate_overall,
    aggregate_expenses_by_buyer,
    aggregate_expenses_totals,
    load_expenses_file,
    load_reconciliation_file,
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
