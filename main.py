from scipy.optimize import differential_evolution
from simulate import simulate_house, _DAILY_PROFILE_KWH_PER_KWP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# FITNESS FUNCTIONS


def fitness_SC(x, house_id):
    n_mod = max(1, int(round(x[0])))
    Pm    = float(x[1])
    SC, SS, NEEG, NPV, PBP = simulate_house(house_id, n_mod, Pm)
    return -SC if SC is not None else 1e9  # maximize SC → minimize -SC

def fitness_SS(x, house_id):
    n_mod = max(1, int(round(x[0])))
    Pm    = float(x[1])
    SC, SS, NEEG, NPV, PBP = simulate_house(house_id, n_mod, Pm)
    return -SS if SS is not None else 1e9

def fitness_NPV(x, house_id):
    n_mod = max(1, int(round(x[0])))
    Pm    = float(x[1])
    SC, SS, NEEG, NPV, PBP = simulate_house(house_id, n_mod, Pm)
    return -NPV if NPV is not None else 1e9

def fitness_NEEG(x, house_id):
    n_mod = max(1, int(round(x[0])))
    Pm    = float(x[1])
    SC, SS, NEEG, NPV, PBP = simulate_house(house_id, n_mod, Pm)
    return NEEG if NEEG is not None else 1e9

def fitness_PBP(x, house_id):
    n_mod = max(1, int(round(x[0])))
    Pm    = float(x[1])
    SC, SS, NEEG, NPV, PBP = simulate_house(house_id, n_mod, Pm)
    return PBP if PBP is not None else 1e9


# OPTIMIZATION WRAPPER


def optimize_house(house_id, fitness_func, label):
    bounds = [
        (1, 40),      # number of panels
        (300, 600)    # panel peak power in Wp
    ]
    result = differential_evolution(
        lambda x: fitness_func(x, house_id),
        bounds,
        strategy='best1bin',
        maxiter=40, popsize=12,
        tol=1e-2, polish=True, seed=42,
        workers=1
    )
    n_mod = max(1, int(round(result.x[0])))
    Pm    = float(result.x[1])
    SC, SS, NEEG, NPV, PBP = simulate_house(house_id, n_mod, Pm)

    print(f"\nHouse {house_id} - Optimization objective: {label}:")
    print(f"  Optimal number of panels: {n_mod}")
    print(f"  Optimal panel power: {Pm:.1f} W")
    print(f"  SC: {SC:.3f}")
    print(f"  SS: {SS:.3f}")
    print(f"  NEEG: {NEEG:.2f} kWh/year")
    print(f"  NPV: {NPV:.2f} EUR")
    print(f"  PBP: {PBP if PBP else 'Not recovered within lifetime'} years")

    return {
        "Objective": label,
        "n_mod": n_mod,
        "Pm": Pm,
        "SC": SC,
        "SS": SS,
        "NEEG": NEEG,
        "NPV": NPV,
        "PBP": PBP
    }


# MAIN


if __name__ == "__main__":
    house_id = 2000900

    objectives = {
        "Max SC": fitness_SC,
        "Max SS": fitness_SS,
        "Max NPV": fitness_NPV,
        "Min NEEG": fitness_NEEG,
        "Min PBP": fitness_PBP
    }

    results = []
    for label, func in objectives.items():
        results.append(optimize_house(house_id, func, label))

    # Build comparison DataFrame
    df = pd.DataFrame(results).set_index("Objective")
    print("\n=== Comparative Results ===")
    print(df)

    # Export results to Excel
    df.to_excel(f"results_house_{house_id}.xlsx")

    best = results[1]  # Max SS objective
    n_mod, Pm = best["n_mod"], best["Pm"]

    df_cons = pd.read_csv(f"case/casa_{house_id}.csv", parse_dates=["datetime"])
    df_cons["datetime"] = pd.to_datetime(df_cons["datetime"]).dt.tz_localize(None)
    df_cons = df_cons.set_index("datetime").sort_index()
    daily_consum = df_cons["Value"].resample("D").sum()

    system_kWp = (n_mod * Pm) / 1000.0
    profile = _DAILY_PROFILE_KWH_PER_KWP[:len(daily_consum)]

    daily_production = profile * system_kWp
    daily_production = pd.Series(daily_production, index=daily_consum.index)

    df_comparison = pd.DataFrame({
        "Consumption_kWh": daily_consum / 1000.0,
        "Production_kWh": daily_production
    }).dropna()

    plt.figure(figsize=(12,6))
    df_comparison.plot(ax=plt.gca())
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.title(f"Consumption vs. PV Production - House {house_id}\n"
              f"{n_mod} panels, {Pm:.1f} W each")
    plt.ylabel("Energy (kWh/day)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
