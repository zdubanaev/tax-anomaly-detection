"""
Детекция аномалий в синтетических налоговых декларациях.

Все параметры генератора — предположения автора (они НЕ откалиброваны по
реальным данным ГНС КР). Код воспроизводим: запуск с теми же seed даёт те же числа.

Запуск:  python experiments.py
Зависимости: numpy, pandas, scikit-learn, scipy
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Параметры генератора (предположения)
# ---------------------------------------------------------------------------
VAT_RATE = 0.12      # ставка НДС в КР
PROFIT_TAX = 0.10    # ставка налога на прибыль в КР
SOC_RATE = 0.2725    # суммарные страховые взносы (упрощённо)

# отрасль: (выручка на сотрудника, доля расходов, доля материальных затрат,
#           средняя месячная зарплата, фондоёмкость)
INDUSTRIES = {
    "торговля":       (3.0e6, 0.90, 0.75, 25e3, 0.3),
    "строительство":  (2.0e6, 0.85, 0.55, 35e3, 0.6),
    "производство":   (1.8e6, 0.82, 0.60, 30e3, 1.2),
    "IT":             (1.5e6, 0.70, 0.10, 80e3, 0.2),
    "транспорт":      (1.6e6, 0.80, 0.40, 32e3, 1.5),
    "общепит":        (1.0e6, 0.85, 0.45, 20e3, 0.4),
    "сельское_хоз":   (0.8e6, 0.80, 0.50, 15e3, 1.0),
    "услуги":         (1.2e6, 0.75, 0.20, 28e3, 0.3),
}
REGIONS = ["Бишкек", "Ош", "Чуйская", "Ошская", "Джалал-Абадская",
           "Иссык-Кульская", "Нарынская", "Таласская", "Баткенская"]
REGION_P = [0.38, 0.10, 0.16, 0.08, 0.09, 0.07, 0.04, 0.04, 0.04]

NUM_FEATURES = [
    "revenue", "costs", "material_costs", "payroll", "headcount", "vat_out",
    "vat_in", "vat_payable", "deduction_share", "profit", "profit_tax",
    "fixed_assets", "depreciation", "other_income", "prev_revenue",
    "revenue_growth", "payroll_per_employee", "revenue_per_employee",
    "cost_ratio", "soc_contrib", "months_active",
]
CAT_FEATURES = ["industry", "region", "regime"]   # всего 21 + 3 = 24 признака
MONEY = ["revenue", "costs", "material_costs", "payroll", "vat_out", "vat_in",
         "vat_payable", "profit", "profit_tax", "fixed_assets", "depreciation",
         "other_income", "prev_revenue", "payroll_per_employee",
         "revenue_per_employee", "soc_contrib"]


def generate(n, profile, rng):
    """Генерирует n «нормальных» деклараций.

    profile: 'vat'    — плательщики НДС, общий режим, умеренный разброс
             'profit' — общий режим, тяжёлые хвосты по размеру
             'mixed'  — смесь режимов, много микропредприятий
    """
    ind_names = list(INDUSTRIES)
    industry = rng.choice(ind_names, size=n)
    region = rng.choice(REGIONS, size=n, p=REGION_P)

    if profile == "vat":
        headcount = np.round(rng.lognormal(3.0, 0.8, n)) + 1
        regime = np.full(n, "общий")
    elif profile == "profit":
        headcount = np.round(rng.pareto(1.6, n) * 10) + 1       # тяжёлый хвост
        regime = np.full(n, "общий")
    else:
        headcount = np.round(rng.lognormal(1.2, 1.2, n)) + 1
        regime = rng.choice(["общий", "единый_налог", "патент"], size=n,
                            p=[0.35, 0.45, 0.20])

    p = np.array([INDUSTRIES[i] for i in industry])
    rev_per_emp, cost_ratio0, mat_share, salary, asset_int = p.T

    revenue = headcount * rev_per_emp * rng.lognormal(0, 0.35, n)
    cost_ratio = np.clip(cost_ratio0 + rng.normal(0, 0.05, n), 0.3, 1.1)
    costs = revenue * cost_ratio
    material_costs = costs * np.clip(mat_share + rng.normal(0, 0.05, n), 0.02, 0.95)
    payroll = headcount * salary * 12 * rng.lognormal(0, 0.15, n)
    soc_contrib = payroll * SOC_RATE
    fixed_assets = revenue * asset_int * rng.lognormal(0, 0.4, n)
    depreciation = fixed_assets * rng.uniform(0.05, 0.15, n)
    other_income = revenue * rng.exponential(0.02, n)

    vat_payer = regime == "общий"
    vat_out = np.where(vat_payer, revenue * VAT_RATE, 0.0)
    vat_in = np.where(vat_payer, material_costs * VAT_RATE * rng.uniform(0.85, 1.0, n), 0.0)
    vat_payable = vat_out - vat_in
    deduction_share = np.where(vat_out > 0, vat_in / np.maximum(vat_out, 1), 0.0)

    profit = revenue + other_income - costs
    profit_tax = np.where(vat_payer, np.maximum(profit, 0) * PROFIT_TAX, 0.0)

    growth = rng.normal(0.08, 0.15, n)
    prev_revenue = revenue / (1 + growth)
    months_active = np.where(rng.random(n) < 0.9, 12, rng.integers(1, 12, n))

    df = pd.DataFrame(dict(
        revenue=revenue, costs=costs, material_costs=material_costs,
        payroll=payroll, headcount=headcount, vat_out=vat_out, vat_in=vat_in,
        vat_payable=vat_payable, deduction_share=deduction_share,
        profit=profit, profit_tax=profit_tax, fixed_assets=fixed_assets,
        depreciation=depreciation, other_income=other_income,
        prev_revenue=prev_revenue, revenue_growth=growth,
        payroll_per_employee=payroll / headcount,
        revenue_per_employee=revenue / headcount,
        cost_ratio=cost_ratio, soc_contrib=soc_contrib,
        months_active=months_active,
        industry=industry, region=region, regime=regime,
    ))
    df["label"] = 0
    df["anomaly_type"] = "нет"
    return df


def inject_anomalies(df, share, rng):
    """Внедряет аномалии трёх типов (поровну) в долю share деклараций."""
    df = df.copy()
    n_anom = int(round(share * len(df)))
    idx = rng.choice(len(df), size=n_anom, replace=False)
    types = np.array(["точечная", "вычеты_НДС", "занижение_ФОТ"])[np.arange(n_anom) % 3]

    for i, t in zip(idx, types):
        r = df.iloc[i]
        if t == "точечная" or (t == "вычеты_НДС" and r.vat_out == 0):
            # искажение одной строки без согласования зависимых строк
            col = rng.choice(["revenue", "costs", "material_costs", "fixed_assets"])
            factor = rng.choice([rng.uniform(0.1, 0.3), rng.uniform(3, 8)])
            df.iat[i, df.columns.get_loc(col)] = r[col] * factor
            t = "точечная"
        elif t == "вычеты_НДС":
            # фиктивные ЭСФ: входной НДС завышен, всё пересчитано согласованно
            vat_in = r.vat_in * rng.uniform(1.3, 2.0)
            df.loc[df.index[i], ["vat_in", "vat_payable", "deduction_share"]] = [
                vat_in, r.vat_out - vat_in, vat_in / max(r.vat_out, 1)]
        else:
            # занижение ФОТ при неизменной численности
            payroll = r.payroll * rng.uniform(0.2, 0.5)
            df.loc[df.index[i], ["payroll", "payroll_per_employee", "soc_contrib"]] = [
                payroll, payroll / r.headcount, payroll * SOC_RATE]
        df.iat[i, df.columns.get_loc("label")] = 1
        df.iat[i, df.columns.get_loc("anomaly_type")] = t
    return df


# ---------------------------------------------------------------------------
# 2. Предобработка
# ---------------------------------------------------------------------------
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def make_preprocessor():
    """Денежные признаки -> signed log; все числовые -> StandardScaler;
    категориальные -> one-hot."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


def transform_money(df):
    df = df.copy()
    df[MONEY] = signed_log(df[MONEY].astype(float))
    return df


# ---------------------------------------------------------------------------
# 3. Методы: каждая функция возвращает оценки аномальности (больше = аномальнее)
# ---------------------------------------------------------------------------
def score_iforest(Xtr, Xte, seed):
    m = IsolationForest(n_estimators=200, random_state=seed).fit(Xtr)
    return -m.score_samples(Xtr), -m.score_samples(Xte)


def score_lof(Xtr, Xte, seed):
    m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr)
    return -m.score_samples(Xtr), -m.score_samples(Xte)


def score_ocsvm(Xtr, Xte, seed):
    m = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(Xtr)
    return -m.score_samples(Xtr), -m.score_samples(Xte)


def score_autoencoder(Xtr, Xte, seed):
    d = Xtr.shape[1]
    m = MLPRegressor(hidden_layer_sizes=(16, 8, 16), activation="relu",
                     solver="adam", max_iter=150, random_state=seed).fit(Xtr, Xtr)
    err = lambda X: ((m.predict(X) - X) ** 2).mean(axis=1)
    return err(Xtr), err(Xte)


def score_zscore(tr_df, te_df):
    """Базовое правило: max(|z|) по log-выручке и доле вычетов, порог 3σ."""
    out = []
    for df in (tr_df, te_df):
        z = []
        for c in ["revenue", "deduction_share"]:
            mu, sd = tr_df[c].mean(), tr_df[c].std() + 1e-9
            z.append(np.abs((df[c] - mu) / sd))
        out.append(np.maximum(*z).values)
    return out


METHODS = {
    "Isolation Forest": score_iforest,
    "Автокодер": score_autoencoder,
    "LOF": score_lof,
    "One-Class SVM": score_ocsvm,
}


def minmax(train_scores, test_scores):
    lo, hi = np.percentile(train_scores, [1, 99])
    return np.clip((test_scores - lo) / (hi - lo + 1e-12), 0, 1)


# ---------------------------------------------------------------------------
# 4. Эксперимент
# ---------------------------------------------------------------------------
def evaluate(y, s, top_share):
    """Флаг = верхние top_share деклараций по оценке аномальности."""
    thr = np.quantile(s, 1 - top_share)
    pred = (s >= thr).astype(int)
    return dict(Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), F1=f1_score(y, pred),
                ROC_AUC=roc_auc_score(y, s), PR_AUC=average_precision_score(y, s))


def run_experiment(profile, n, seed, anomaly_share=0.021, top_share=0.03):
    rng = np.random.default_rng(seed)
    df = inject_anomalies(generate(n, profile, rng), anomaly_share, rng)
    tr, te = train_test_split(df, test_size=0.3, stratify=df.label, random_state=seed)

    # Модели обучаются на обучающей части БЕЗ меток (аномалии в ней присутствуют,
    # как и в реальной жизни). Метки используются только для оценки на тестовой части.
    tr_l, te_l = transform_money(tr), transform_money(te)
    pre = make_preprocessor().fit(tr_l)
    Xtr, Xte = pre.transform(tr_l), pre.transform(te_l)
    y = te.label.values

    rows, normed = [], []
    for name, fn in METHODS.items():
        s_tr, s_te = fn(Xtr, Xte, seed)
        rows.append(dict(method=name, **evaluate(y, s_te, top_share)))
        normed.append(minmax(s_tr, s_te))
    ens = np.mean(normed, axis=0)
    rows.append(dict(method="Ансамбль (среднее)", **evaluate(y, ens, top_share)))
    z_tr, z_te = score_zscore(tr_l, te_l)
    rows.append(dict(method="Z-скоринг (бенчмарк)", **evaluate(y, z_te, top_share)))

    # Recall ансамбля по типам аномалий
    thr = np.quantile(ens, 1 - top_share)
    by_type = (te.assign(flag=ens >= thr)[te.label == 1]
                 .groupby("anomaly_type").flag.mean().to_dict())
    return pd.DataFrame(rows), by_type


EXPERIMENTS = [("1. Декларации по НДС", "vat", 7850),
               ("2. Декларации по налогу на прибыль", "profit", 6280),
               ("3. Смешанный набор", "mixed", 4290)]
SEEDS = [0, 1, 2, 3, 4]

# ---------------------------------------------------------------------------
# 6. Дополнительный анализ: полнота каждого метода по типам аномалий
# ---------------------------------------------------------------------------
def recall_by_type(profile, n, seed, numeric_only=False, anomaly_share=0.021, top_share=0.03):
    """Recall каждого метода отдельно по типам аномалий + ROC-AUC.
    numeric_only=True — без категориальных признаков (проверка влияния one-hot)."""
    rng = np.random.default_rng(seed)
    df = inject_anomalies(generate(n, profile, rng), anomaly_share, rng)
    tr, te = train_test_split(df, test_size=0.3, stratify=df.label, random_state=seed)
    tr_l, te_l = transform_money(tr), transform_money(te)
    if numeric_only:
        sc = StandardScaler().fit(tr_l[NUM_FEATURES])
        Xtr, Xte = sc.transform(tr_l[NUM_FEATURES]), sc.transform(te_l[NUM_FEATURES])
    else:
        pre = make_preprocessor().fit(tr_l)
        Xtr, Xte = pre.transform(tr_l), pre.transform(te_l)
    out = {}
    for name, fn in METHODS.items():
        _, s = fn(Xtr, Xte, seed)
        thr = np.quantile(s, 1 - top_share)
        r = te.assign(flag=s >= thr)[te.label == 1].groupby("anomaly_type").flag.mean()
        r["ROC_AUC"] = roc_auc_score(te.label, s)
        out[name] = r
    return pd.DataFrame(out)


def additional_analysis():
    for numeric_only in (False, True):
        tables = [recall_by_type(p, n, s, numeric_only)
                  for _, p, n in EXPERIMENTS for s in SEEDS]
        avg = sum(tables) / len(tables)
        title = ("без категориальных признаков" if numeric_only
                 else "все признаки")
        print(f"\n=== Recall по типам аномалий и ROC-AUC, среднее по 3 экспериментам "
              f"и {len(SEEDS)} запускам ({title}) ===")
        print(avg.round(3).to_string())

    # Что изолирует Isolation Forest в первую очередь? Сравниваем его оценку
    # с «экстремальностью» масштаба предприятия |z| по log-выручке (seed = 0).
    from scipy.stats import spearmanr
    print("\n=== Isolation Forest и масштаб предприятия (seed = 0) ===")
    for title, profile, n in EXPERIMENTS:
        rng = np.random.default_rng(0)
        df = inject_anomalies(generate(n, profile, rng), 0.021, rng)
        tr, te = train_test_split(df, test_size=0.3, stratify=df.label, random_state=0)
        tr_l, te_l = transform_money(tr), transform_money(te)
        pre = make_preprocessor().fit(tr_l)
        _, s = score_iforest(pre.transform(tr_l), pre.transform(te_l), 0)
        z = np.abs((te_l.revenue - tr_l.revenue.mean()) / tr_l.revenue.std())
        top = s >= np.quantile(s, 0.97)
        print(f"{title}: Спирмен(оценка IF, |z|) = {spearmanr(s, z)[0]:.2f}; "
              f"средний |z| среди отмеченных = {z[top].mean():.2f} (по всем {z.mean():.2f}); "
              f"доля нормальных среди отмеченных = {1 - te.label[top].mean():.2f}")


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    all_results = []
    for title, profile, n in EXPERIMENTS:
        runs, types = [], []
        for seed in SEEDS:
            res, bt = run_experiment(profile, n, seed)
            runs.append(res.assign(seed=seed)); types.append(bt)
        runs = pd.concat(runs)
        agg = runs.groupby("method", sort=False).agg(["mean", "std"]).drop(columns="seed")
        agg = agg.sort_values(("F1", "mean"), ascending=False)
        print(f"\n=== Эксперимент {title} (n={n}, {len(SEEDS)} запусков) ===")
        print(agg.round(3).to_string())
        print("Recall ансамбля по типам аномалий:",
              pd.DataFrame(types).mean().round(3).to_dict())
        all_results.append(runs.assign(experiment=title))
    pd.concat(all_results).to_csv("results_raw.csv", index=False)
    print("\nСырые результаты сохранены в results_raw.csv")
    additional_analysis()
