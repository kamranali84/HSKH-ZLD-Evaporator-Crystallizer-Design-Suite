"""
mvr_thermo.py
Steam-table and MVR blower / VFD helper functions for the Vacom crystallizer app.

Saturation pressure/temperature use the IAPWS-IF97 Region 4 equations
(valid 0.000611 - 22.064 MPa), so boiling temperature tracks vessel
pressure exactly instead of using a coarse lookup table.
"""

import math

PSI_TO_MPA = 0.00689476
PSI_TO_BAR = 0.0689476

# IAPWS-IF97 Region 4 coefficients
_N = [
    0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
    0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
    -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
    0.65017534844798e3,
]


def tsat_c_from_psia(p_psia: float) -> float:
    """Saturation temperature of pure water (deg C) at pressure p (psia)."""
    n = _N
    p = max(p_psia, 0.1) * PSI_TO_MPA
    beta = p ** 0.25
    e = beta ** 2 + n[2] * beta + n[5]
    f = n[0] * beta ** 2 + n[3] * beta + n[6]
    g = n[1] * beta ** 2 + n[4] * beta + n[7]
    d = 2.0 * g / (-f - math.sqrt(f ** 2 - 4.0 * e * g))
    t_k = (n[9] + d - math.sqrt((n[9] + d) ** 2 - 4.0 * (n[8] + n[9] * d))) / 2.0
    return t_k - 273.15


def psat_psia_from_c(t_c: float) -> float:
    """Saturation pressure of pure water (psia) at temperature t (deg C)."""
    n = _N
    t = t_c + 273.15
    th = t + n[8] / (t - n[9])
    a = th ** 2 + n[0] * th + n[1]
    b = n[2] * th ** 2 + n[3] * th + n[4]
    c = n[5] * th ** 2 + n[6] * th + n[7]
    p_mpa = (2.0 * c / (-b + math.sqrt(b ** 2 - 4.0 * a * c))) ** 4
    return p_mpa / PSI_TO_MPA


def c_to_f(t_c: float) -> float:
    return t_c * 1.8 + 32.0


KJKG_TO_BTULB = 0.429923


def hfg_kj_kg(t_c: float) -> float:
    """Latent heat of vaporization of water (kJ/kg) at t (deg C). +/-0.3% for 20-180 C."""
    return 2501.6 - 2.3693 * t_c - 0.0013 * t_c ** 2


def hfg_btu_lb(t_c: float) -> float:
    return hfg_kj_kg(t_c) * KJKG_TO_BTULB


FHX_MODES = ["Calculate from effectiveness", "Fixed outlet temperature"]


def preheater(
    feed_lb_hr: float,
    t_feed_in_c: float,
    dist_lb_hr: float,
    t_dist_in_c: float,
    effectiveness: float = 0.80,
    feed_cp: float = 0.95,
    dist_cp: float = 1.0,
    mode: str = FHX_MODES[0],
    t_feed_out_fixed_c: float = 80.0,
) -> dict:
    """
    Counter-flow feed pre-heater: cold feed heated by hot distillate.

    mode "Calculate from effectiveness":
        Q = eff x Cmin x (T_dist_in - T_feed_in)
    mode "Fixed outlet temperature":
        Q = C_feed x (T_feed_out - T_feed_in)
    Temperatures deg C, duty MMBTU/hr, C = lb/hr x Cp (BTU/hr-F).
    """
    F = 1.8
    c_feed = feed_lb_hr * feed_cp
    c_dist = dist_lb_hr * dist_cp
    c_min = min(c_feed, c_dist)
    dt_max = t_dist_in_c - t_feed_in_c

    if mode == FHX_MODES[1]:
        q = c_feed * (t_feed_out_fixed_c - t_feed_in_c) * F
    else:
        q = effectiveness * c_min * max(dt_max, 0.0) * F

    t_feed_out = t_feed_in_c + (q / (c_feed * F) if c_feed > 0 else 0.0)
    t_dist_out = t_dist_in_c - (q / (c_dist * F) if c_dist > 0 else 0.0)
    eff_actual = q / (c_min * dt_max * F) if c_min > 0 and dt_max > 0 else 0.0

    dt_hot = t_dist_in_c - t_feed_out      # hot end
    dt_cold = t_dist_out - t_feed_in_c     # cold end
    if dt_hot > 0 and dt_cold > 0:
        lmtd_c = (dt_hot - dt_cold) / math.log(dt_hot / dt_cold) if abs(dt_hot - dt_cold) > 1e-6 else dt_hot
    else:
        lmtd_c = 0.0

    return {
        "t_feed_in_c": t_feed_in_c,
        "t_feed_out_c": t_feed_out,
        "t_dist_in_c": t_dist_in_c,
        "t_dist_out_c": t_dist_out,
        "duty_mmbtu": q / 1e6,
        "effectiveness": eff_actual,
        "approach_hot_c": dt_hot,
        "approach_cold_c": dt_cold,
        "lmtd_c": lmtd_c,
        "feasible": dt_hot > 0 and dt_cold > 0 and eff_actual <= 1.0,
    }


def recirc_requirement(
    vapor_lb_hr: float,
    vessel_sat_c: float,
    boil_c: float,
    feed_lb_hr: float,
    feed_in_c: float,
    flash_dt_f: float,
    brine_cp: float = 0.80,
    brine_sg: float = 1.20,
    feed_cp: float = 0.95,
) -> dict:
    """
    Forced-circulation loop heat balance (used by Recirc Pump and Primary HX tabs).

    HX duty   = evaporation (vapor x hfg) + feed heat-up (pre-HX outlet -> boiling)
    Recirc    = duty / (brine Cp x flash dT)
    """
    hfg_vessel = hfg_btu_lb(vessel_sat_c)
    evap_duty = vapor_lb_hr * hfg_vessel / 1e6
    feed_heat_duty = max(feed_lb_hr * feed_cp * (boil_c - feed_in_c) * 1.8, 0.0) / 1e6
    hx_duty = evap_duty + feed_heat_duty
    flow_lb_hr = hx_duty * 1e6 / (brine_cp * flash_dt_f) if flash_dt_f > 0 else 0.0
    flow_gpm = flow_lb_hr / (8.34 * brine_sg * 60.0)
    return {
        "hfg_vessel": hfg_vessel,
        "evap_duty": evap_duty,
        "feed_heat_duty": feed_heat_duty,
        "hx_duty": hx_duty,
        "flow_lb_hr": flow_lb_hr,
        "flow_gpm": flow_gpm,
        "flash_fraction_pct": brine_cp * flash_dt_f / hfg_vessel * 100.0,
    }


def boiler_capacity(
    feed_lb_hr: float,
    salt_lb_hr: float,
    cake_water_lb_hr: float,
    distillate_lb_hr: float,
    t_feed_c: float,
    t_preheat_out_c: float,
    t_distillate_in_c: float,
    t_boil_c: float,
    t_sat_vessel_c: float,
    blower_shaft_kw: float,
    boiler_psia: float,
    boiler_feedwater_c: float,
    boiler_eff: float,
    feed_cp: float = 0.95,
    salt_cp: float = 0.21,
    vent_frac: float = 0.0075,
    radiation_frac: float = 0.025,
    makeup_factor: float = 2.5,
    brine_inventory_lb: float = 135000.0,
    brine_cp: float = 0.80,
    steel_lb: float = 70000.0,
    steel_cp: float = 0.12,
    heatup_loss_frac: float = 0.13,
    heatup_hr: float = 4.0,
    priming_lb_hr: float = 0.0,
) -> dict:
    """
    Electric boiler sizing for an MVR crystallizer.
    Boiler capacity = max(design makeup steam, start-up heat-up steam, blower priming steam).
    Temperatures deg C, Cp BTU/lb-F, heat MMBTU/hr unless noted.
    """
    F = 1.8  # deg C difference -> deg F difference

    # Steam properties
    t_sat_boiler = tsat_c_from_psia(boiler_psia)
    hfg_boiler = hfg_btu_lb(t_sat_boiler)                   # heat delivered per lb steam
    boiler_rise = (4.186 * (t_sat_boiler - boiler_feedwater_c) * KJKG_TO_BTULB
                   + hfg_boiler)                            # electric input per lb steam
    hfg_vessel = hfg_btu_lb(t_sat_vessel_c)

    # Heat exchangers
    q_preheat = feed_lb_hr * feed_cp * (t_preheat_out_c - t_feed_c) * F / 1e6
    t_dist_out = (t_distillate_in_c - q_preheat * 1e6 / (distillate_lb_hr * F)
                  if distillate_lb_hr > 0 else t_distillate_in_c)
    evap_duty = distillate_lb_hr * hfg_vessel / 1e6

    # Steady-state losses
    loss_distillate = max(0.0, distillate_lb_hr * (t_dist_out - t_feed_c) * F / 1e6)
    loss_salt = (salt_lb_hr * salt_cp + cake_water_lb_hr) * (t_boil_c - t_feed_c) * F / 1e6
    h_vent = hfg_vessel + (t_boil_c - t_feed_c) * F
    loss_vent = distillate_lb_hr * vent_frac * h_vent / 1e6
    loss_radiation = evap_duty * radiation_frac
    losses = loss_distillate + loss_salt + loss_vent + loss_radiation

    blower_heat = blower_shaft_kw * 3412.14 / 1e6
    net_makeup = max(0.0, losses - blower_heat)
    surplus = max(0.0, blower_heat - losses)
    makeup_lb_hr = net_makeup * 1e6 / hfg_boiler
    makeup_design = makeup_lb_hr * makeup_factor

    # Start-up heat-up
    q_brine = brine_inventory_lb * brine_cp * (t_boil_c - t_feed_c) * F / 1e6
    q_steel = steel_lb * steel_cp * (t_boil_c - t_feed_c) * F / 1e6
    q_startup = (q_brine + q_steel) * (1.0 + heatup_loss_frac)
    startup_lb_hr = q_startup * 1e6 / (hfg_boiler * heatup_hr) if heatup_hr > 0 else 0.0

    required = max(makeup_design, startup_lb_hr, priming_lb_hr)
    if required == priming_lb_hr and priming_lb_hr > 0:
        governing = "Blower priming"
    elif required == startup_lb_hr:
        governing = "Start-up heat-up"
    else:
        governing = "Steady-state makeup"

    elec_kw = required * boiler_rise / 3412.14 / boiler_eff if boiler_eff > 0 else 0.0

    return {
        "t_sat_boiler": t_sat_boiler,
        "hfg_boiler": hfg_boiler,
        "boiler_rise": boiler_rise,
        "q_preheat": q_preheat,
        "t_dist_out": t_dist_out,
        "evap_duty": evap_duty,
        "loss_distillate": loss_distillate,
        "loss_salt": loss_salt,
        "loss_vent": loss_vent,
        "loss_radiation": loss_radiation,
        "losses": losses,
        "blower_heat": blower_heat,
        "net_makeup": net_makeup,
        "surplus": surplus,
        "makeup_lb_hr": makeup_lb_hr,
        "makeup_design": makeup_design,
        "q_startup": q_startup,
        "startup_lb_hr": startup_lb_hr,
        "priming_lb_hr": priming_lb_hr,
        "required_lb_hr": required,
        "governing": governing,
        "elec_kw": elec_kw,
    }


def mvr_state(
    vessel_psia: float,
    discharge_psia: float,
    bpe_c: float,
    blower_eff: float,
    motor_eff: float,
    vapor_lb_hr: float,
    k: float = 1.33,
    cp_vapor: float = 1.9,
) -> dict:
    """
    Thermodynamic state of crystallizer + MVR blower.

    vessel_psia     crystallizer (blower suction) pressure, psia
    discharge_psia  blower discharge pressure, psia
    bpe_c           boiling point elevation of brine, deg C
    blower_eff      isentropic efficiency, fraction (0.75)
    motor_eff       motor + VFD efficiency, fraction (0.95)
    vapor_lb_hr     vapor flow through blower, lb/hr
    """
    t_sat_vessel = tsat_c_from_psia(vessel_psia)
    t_boil = t_sat_vessel + bpe_c          # brine boils / vapor leaves at this temp
    t_sat_disch = tsat_c_from_psia(discharge_psia)  # condensing temp in primary HX

    pr = discharge_psia / vessel_psia
    t1_k = t_boil + 273.15
    dt_isen = t1_k * (pr ** ((k - 1.0) / k) - 1.0)
    dt_actual = dt_isen / blower_eff
    t_disch_actual = t_boil + dt_actual   # superheated discharge before desuperheating

    dh_kj_kg = cp_vapor * dt_actual
    m_kg_s = vapor_lb_hr / 2.20462 / 3600.0
    shaft_kw = m_kg_s * dh_kj_kg
    elec_kw = shaft_kw / motor_eff if motor_eff > 0 else 0.0

    return {
        "vessel_psia": vessel_psia,
        "vessel_bar": vessel_psia * PSI_TO_BAR,
        "t_sat_vessel": t_sat_vessel,
        "t_boil": t_boil,
        "discharge_psia": discharge_psia,
        "blower_dp": discharge_psia - vessel_psia,
        "pressure_ratio": pr,
        "t_sat_disch": t_sat_disch,
        "sat_temp_lift": t_sat_disch - t_sat_vessel,
        "hx_driving_dt": t_sat_disch - t_boil,   # condensing vapor vs boiling brine
        "t_disch_actual": t_disch_actual,
        "superheat": t_disch_actual - t_sat_disch,
        "shaft_kw": shaft_kw,
        "elec_kw": elec_kw,
    }


def vfd_response(
    vessel_pv_psia: float,
    vessel_sp_psia: float,
    design_speed_pct: float,
    kp_pct_per_psi: float,
    min_speed_pct: float,
    max_speed_pct: float,
    design_dp_psi: float,
    design_vapor_lb_hr: float,
) -> dict:
    """
    Pressure-control loop: vessel pressure (PV) vs setpoint (SP) -> blower VFD speed.

    PV > SP : vapor is forming faster than the blower removes it -> speed up.
    PV < SP : vapor is forming slower than the blower removes it -> slow down.

    Proportional action:  speed = design_speed + Kp x (PV - SP), clipped to VFD limits.
    Fan affinity laws:    flow ~ N,  head (dP) ~ N^2,  power ~ N^3.
    """
    error = vessel_pv_psia - vessel_sp_psia
    speed = design_speed_pct + kp_pct_per_psi * error
    speed = min(max(speed, min_speed_pct), max_speed_pct)
    ratio = speed / design_speed_pct if design_speed_pct > 0 else 0.0

    dp = design_dp_psi * ratio ** 2
    return {
        "error_psi": error,
        "speed_pct": speed,
        "speed_ratio": ratio,
        "blower_dp": dp,
        "discharge_psia": vessel_pv_psia + dp,
        "vapor_lb_hr": design_vapor_lb_hr * ratio,
        "at_limit": speed in (min_speed_pct, max_speed_pct),
        "action": "Speed UP" if error > 0.005 else ("Slow DOWN" if error < -0.005 else "Hold"),
    }
