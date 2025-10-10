import argparse
import math
import os
from typing import List, Tuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from matplotlib.ticker import MultipleLocator


# ===================== Параметры по умолчанию =====================

DEF_DELTA_N_TIME = 0.2
DEF_DELTA_N_VOLT = 0.2
DEF_REL_TIME_PCT = 3.0
DEF_REL_VOLT_PCT = 3.0


# ===================== Вспомогательные функции =====================

def ask_float(prompt: str) -> float:
    s = input(prompt).strip().replace(',', '.')
    return float(s)

def ask_int(prompt: str) -> int:
    s = input(prompt).strip()
    return int(s)

def gather_rows_time(n_rows: int):
    dt_list, ndiv_list = [], []
    print("\n--- Таблица 1.1 ---")
    for i in range(1, n_rows + 1):
        dt = ask_float(f"[{i}] Переключатель (с/дел): ")
        n = ask_float(f"[{i}] Делений (дел): ")
        dt_list.append(dt)
        ndiv_list.append(n)
    return dt_list, ndiv_list

def gather_rows_voltage(n_rows: int):
    vpd_list, ndiv_list = [], []
    print("\n--- Таблица 1.2 ---")
    for i in range(1, n_rows + 1):
        vpd = ask_float(f"[{i}] Переключатель (В/дел): ")
        n = ask_float(f"[{i}] Делений (дел): ")
        vpd_list.append(vpd)
        ndiv_list.append(n)
    return vpd_list, ndiv_list


def compute_time_table(dt_list, n_list, d_n_div: float, rel_scale_percent: float) -> pd.DataFrame:
    rel = rel_scale_percent / 100.0
    rows = []
    for i, (dt, n) in enumerate(zip(dt_list, n_list), start=1):
        T = n * dt
        d_dt = dt * rel
        dT = math.sqrt((d_n_div * dt) ** 2 + (n * d_dt) ** 2)
        f = 1.0 / T if T != 0 else float("nan")
        df = f * (dT / T) if T != 0 else float("nan")
        rows.append({
            "№": i,
            "Показания Переключателя (с/дел)": dt,
            "Число Делений (дел)": n,
            "T (с)": T,
            "ΔT (с)": dT,
            "f (Гц)": f,
            "Δf (Гц)": df,
        })
    return pd.DataFrame(rows)

def compute_voltage_table(vpd_list, n_list, d_n_div: float, rel_scale_percent: float) -> pd.DataFrame:
    rel = rel_scale_percent / 100.0
    rows = []
    for i, (vpd, n) in enumerate(zip(vpd_list, n_list), start=1):
        Upp   = n * vpd
        d_vpd = vpd * rel
        dUpp  = math.sqrt((d_n_div * vpd) ** 2 + (n * d_vpd) ** 2)
        Uamp  = Upp / 2.0
        dUamp = dUpp / 2.0
        Urms  = Uamp / math.sqrt(2.0)
        dUrms = dUamp / math.sqrt(2.0)
        rows.append({
            "№": i,
            "Показания Переключателя (В/дел)": vpd,
            "Число Делений (дел)": n,
            "Upp (В)": Upp,
            "ΔUpp (В)": dUpp,
            "Uamp (В)": Uamp,
            "ΔUamp (В)": dUamp,
            "Urms (В)": Urms,
            "ΔUrms (В)": dUrms,
        })
    return pd.DataFrame(rows)


# ===================== Графики =====================

def plot_with_errorbars(x_vals, y_vals, y_errs, title, xlabel, ylabel, out_path=None, align_grid=True):
    plt.figure()
    plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o-', capsize=4)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # --- выравниваем сетку по точкам ---
    if align_grid:
        plt.xticks(x_vals)
        plt.yticks(np.round(np.linspace(min(y_vals), max(y_vals), len(x_vals)), 3))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
    else:
        plt.grid(True)

    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()


def plot_with_fit(x_vals, y_vals, y_errs, title, xlabel, ylabel, out_path=None):
    plt.figure(figsize=(6, 4))

    # --- точки с погрешностями ---
    plt.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o-', capsize=4, color='blue', label='Эксп. данные')

    # --- МНК ---
    coeffs = np.polyfit(x_vals, y_vals, 1)
    line = np.poly1d(coeffs)
    x_fit = np.linspace(min(x_vals), max(x_vals), 100)
    plt.plot(x_fit, line(x_fit), 'r--', linewidth=1.5,
             label=f"МНК: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")

    # --- оформление ---
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()

    # --- сетка по точкам ---
    plt.xticks(x_vals)
    plt.yticks(np.round(np.linspace(min(y_vals), max(y_vals), len(x_vals)), 3))

    # --- сохранение ---
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close()



# ===================== Таблица → PDF =====================

def df_as_mpl_table(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(max(6, len(df.columns)*1.2), max(2.5, len(df)*0.35 + 1.5)))
    ax.axis('off')
    ax.set_title(title, pad=12)
    tbl = ax.table(cellText=df.round(6).values, colLabels=df.columns, loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    return fig

# ===================== Чтение CSV или ввод данных =====================

def read_or_ask_time(args) -> pd.DataFrame:
    if args.time_csv and os.path.isfile(args.time_csv):
        df = pd.read_csv(args.time_csv)
        dt_list = df.iloc[:, 0].astype(float).tolist()
        n_list  = df.iloc[:, 1].astype(float).tolist()
    else:
        n = ask_int("Сколько строк для Табл. 1.1? ")
        dt_list, n_list = gather_rows_time(n)
    return compute_time_table(
        dt_list, n_list,
        d_n_div=args.delta_n_time,
        rel_scale_percent=args.rel_time_pct
    )


def read_or_ask_voltage(args) -> pd.DataFrame:
    if args.volt_csv and os.path.isfile(args.volt_csv):
        df = pd.read_csv(args.volt_csv)
        vpd_list = df.iloc[:, 0].astype(float).tolist()
        n_list   = df.iloc[:, 1].astype(float).tolist()
    else:
        n = ask_int("Сколько строк для Табл. 1.2? ")
        vpd_list, n_list = gather_rows_voltage(n)
    return compute_voltage_table(
        vpd_list, n_list,
        d_n_div=args.delta_n_volt,
        rel_scale_percent=args.rel_volt_pct
    )



# ===================== Главная логика =====================

def main():
    parser = argparse.ArgumentParser(description="Oscilloscope tables → расчёт величин, погрешностей и отчёт с графиками")
    parser.add_argument("--time-csv")
    parser.add_argument("--volt-csv")
    parser.add_argument("--out", default="out")
    parser.add_argument("--delta-n-time", type=float, default=DEF_DELTA_N_TIME)
    parser.add_argument("--delta-n-volt", type=float, default=DEF_DELTA_N_VOLT)
    parser.add_argument("--rel-time-pct", type=float, default=DEF_REL_TIME_PCT)
    parser.add_argument("--rel-volt-pct", type=float, default=DEF_REL_VOLT_PCT)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df_time = read_or_ask_time(args)
    df_volt = read_or_ask_voltage(args)

    csv_time = os.path.join(args.out, "table_1_1_time_results.csv")
    csv_volt = os.path.join(args.out, "table_1_2_voltage_results.csv")
    df_time.to_csv(csv_time, index=False, float_format="%.3f")
    df_volt.to_csv(csv_volt, index=False, float_format="%.3f")

    # === Графики ===
    x1 = df_time["№"].values
    plot_with_errorbars(x1, df_time["f (Гц)"].values, df_time["Δf (Гц)"].values,
                        "График 1.1B: Частота f с погрешностями", "Показания ГСК, дел", "f, Гц",
                        out_path=os.path.join(args.out, "time_f.png"))

    x2 = df_volt["№"].values
    plot_with_fit(x2, df_volt["Urms (В)"].values, df_volt["ΔUrms (В)"].values,
                  "График 1.2: Действующее значение с аппроксимацией МНК", "Показания ГСК, дел", "U_rms, В",
                  out_path=os.path.join(args.out, "volt_rms_fit.png"))

    # === PDF ===
    pdf_path = os.path.join(args.out, "oscill_report.pdf")
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(df_as_mpl_table(df_time, "Таблица 1.1 — Время/Частота"), bbox_inches="tight")
        pdf.savefig(df_as_mpl_table(df_volt, "Таблица 1.2 — Амплитуда/Напряжение"), bbox_inches="tight")

        for png in ["time_f.png", "volt_rms_fit.png"]:
            p = os.path.join(args.out, png)
            if os.path.exists(p):
                img = plt.imread(p)
                fig, ax = plt.subplots()
                ax.imshow(img)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    print("ГОТОВО.\nРезультаты сохранены в:", args.out)


if __name__ == "__main__":
    main()
