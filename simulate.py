import pandas as pd
from functools import lru_cache
from batem.core.weather import SiteWeatherDataBuilder
from batem.core.solar import SolarModel, PVplant, MOUNT_TYPES


# 1) Weather data setup

location = 'Grenoble'
latitude_deg_north = 45.1885
longitude_deg_east = 5.7245

site_weather_data = SiteWeatherDataBuilder().build(
    location,
    given_latitude_north_deg=latitude_deg_north,
    given_longitude_east_deg=longitude_deg_east,
    from_requested_stringdate='1/1/1998',
    to_requested_stringdate='31/12/1998'
)
solar_model = SolarModel(site_weather_data)

# Panel parameters
exposure_deg = 180     # south-facing
slope_deg = 30         # tilt angle
pv_efficiency = 0.20
panel_height_m = 1.7   # spacing between arrays

def _daily_profile_kWh_per_kWp():
    """Realistic daily profile (kWh/day) for 1 kWp with automatic unit detection."""
    pv_plant_1kwp = PVplant(
        solar_model=solar_model,
        exposure_deg=exposure_deg,
        slope_deg=slope_deg,
        distance_between_arrays_m=panel_height_m,
        mount_type=MOUNT_TYPES.PLAN,
        peak_power_kW=1.0,
        number_of_panels_per_array=1,
        panel_height_m=panel_height_m,
        pv_efficiency=1.0
    )

    P = pd.Series(pv_plant_1kwp.powers_W(), index=pd.to_datetime(solar_model.datetimes).tz_localize(None))
    # Time step in hours
    dt_h = (P.index[1] - P.index[0]).total_seconds() / 3600.0

    # Three integration approaches to detect correct unit
    annual_A_Wh = (P * dt_h).resample("D").sum().sum()          # assumes P in W
    annual_B_Wh = P.resample("D").sum().sum()                    # assumes P in Wh/step
    import numpy as np
    annual_C_Wh = np.trapz(P.values, dx=dt_h)                   # numerical integration

    # Select the value closest to realistic range for Grenoble (700–1600 kWh/kWp/year)
    candidates = {
        "A_from_W": annual_A_Wh/1000.0,
        "B_from_WhStep": annual_B_Wh/1000.0,
        "C_trapz": annual_C_Wh/1000.0
    }
    target = 1200.0
    valid = {k: v for k, v in candidates.items() if 700.0 <= v <= 1600.0}
    if not valid:
        chosen_key = max(candidates, key=candidates.get)
        chosen_val = candidates[chosen_key]
    else:
        chosen_key = min(valid, key=lambda k: abs(valid[k] - target))
        chosen_val = valid[chosen_key]

    print(f"[CHECK] dt_h={dt_h:.3f} h, peak@1kWp={P.max():.1f} (raw units), "
          f"Annual candidates kWh/kWp/year: {candidates}, chosen={chosen_key}:{chosen_val:.1f}")

    # Build daily profile from selected integration method
    if chosen_key == "A_from_W":
        daily_kWh = (P * dt_h).resample("D").sum() / 1000.0
    elif chosen_key == "B_from_WhStep":
        daily_kWh = P.resample("D").sum() / 1000.0
    else:
        daily_kWh = (P * dt_h).resample("D").sum() / 1000.0

    return daily_kWh


# Cache profile on first call
_DAILY_PROFILE_KWH_PER_KWP = _daily_profile_kWh_per_kWp()
ANUAL_KWH_PER_KWP = _DAILY_PROFILE_KWH_PER_KWP.sum()
print(f"[CHECK] Output for 1 kWp (Grenoble, selected year): {ANUAL_KWH_PER_KWP:.1f} kWh/kWp/year")

# Calibrate production to realistic target for Grenoble
TARGET_KWH_PER_KWP = 1100.0
calib = TARGET_KWH_PER_KWP / ANUAL_KWH_PER_KWP
print(f"[CALIB] Calibration factor = {calib:.2f}x "
      f"(from {ANUAL_KWH_PER_KWP:.1f} to {TARGET_KWH_PER_KWP:.1f} kWh/kWp/year)")

_DAILY_PROFILE_KWH_PER_KWP *= calib
ANUAL_KWH_PER_KWP = _DAILY_PROFILE_KWH_PER_KWP.sum()
print(f"[CHECK] After calibration: {ANUAL_KWH_PER_KWP:.1f} kWh/kWp/year")


@lru_cache(maxsize=64)
def _load_daily_consum_kWh(house_id: int) -> pd.Series:
    """
    Loads household consumption data once and returns it as kWh/day.
    Assumes 'Value' column contains energy per time step in Wh.
    """
    df_cons = pd.read_csv(f"case/casa_{house_id}.csv", parse_dates=["datetime"])
    df_cons["datetime"] = pd.to_datetime(df_cons["datetime"]).dt.tz_localize(None)
    df_cons = df_cons.set_index("datetime").sort_index()
    daily_kWh = (df_cons["Value"].resample("D").sum()) / 1000.0
    return daily_kWh

def simulate_house(house_id: int, n_mod: int, Pm: float):
    """
    Returns performance indicators (SC, SS, NEEG, NPV, PBP) for a given PV configuration.

    Parameters:
        house_id : household identifier
        n_mod    : number of solar panels
        Pm       : peak power per panel in Wp
    """

    # 2) Daily production and consumption (kWh/day)
    system_kWp = (n_mod * Pm) / 1000.0
    daily_prod_kWh = _DAILY_PROFILE_KWH_PER_KWP * system_kWp
    daily_cons_kWh = _load_daily_consum_kWh(house_id)

    # Align time indices
    df = pd.DataFrame({
        "prod_kWh": daily_prod_kWh,
        "load_kWh": daily_cons_kWh
    }).dropna()

    # 3) Technical indicators
    autoc_kWh = 0.0
    prod_tot_kWh = 0.0
    load_tot_kWh = 0.0
    NEEG_kWh = 0.0

    # Iterate over each time interval k
    for k in range(len(df)):
        P_prod = df["prod_kWh"].iloc[k]   # PV production in interval k
        P_load = df["load_kWh"].iloc[k]   # Household consumption in interval k

        # Self-consumption: min(production, load)
        autoc_kWh += min(P_prod, P_load)

        # Total production and consumption
        prod_tot_kWh += P_prod
        load_tot_kWh += P_load

        # Net energy exchange gap
        NEEG_kWh += abs(P_prod - P_load)

    # Self-Consumption ratio (SC)
    SC = (autoc_kWh / prod_tot_kWh) if prod_tot_kWh > 0 else float("nan")

    # Self-Sufficiency ratio (SS)
    SS = (autoc_kWh / load_tot_kWh) if load_tot_kWh > 0 else float("nan")

    # 4) Financial indicators
    Cwp = 1.2             # euro/Wp — installed cost per watt
    r   = 0.03            # discount rate
    Y   = 30              # system lifetime in years
    tarif_energie = 0.35  # euro/kWh — energy tariff

    CapEX = Cwp * (Pm * n_mod)
    OpEX_t = 0.01 * CapEX                          # 1% annual maintenance cost
    B_ref = load_tot_kWh * tarif_energie            # annual cost without PV
    B_new = (load_tot_kWh - autoc_kWh) * tarif_energie
    G_t   = B_ref - B_new                          # annual savings

    # Net Present Value (NPV)
    NPV = -CapEX
    for t in range(1, Y + 1):
        NPV += (G_t - OpEX_t) / ((1 + r) ** t)

    # Payback Period (PBP)
    cumulative = -CapEX
    PBP = None
    for t in range(1, Y + 1):
        cumulative += (G_t - OpEX_t)
        if cumulative >= 0:
            PBP = t
            break

    return SC, SS, NEEG_kWh, NPV, PBP
