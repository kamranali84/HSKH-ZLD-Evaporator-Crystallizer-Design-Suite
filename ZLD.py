# ============================================================
# IMPORTS
# ============================================================
import math
import json
import os
import re
import copy
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mvr_thermo import (tsat_c_from_psia, c_to_f, mvr_state, vfd_response,
                        boiler_capacity as calc_boiler_capacity)

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle


##############################################################################
# PDF REPORT FUNCTION (simple text report from a Parameter/Value table)
##############################################################################
def create_design_report_pdf(report_df):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "MVR Crystallizer Design Report")
    c.setFont("Helvetica", 10)
    y = 720
    for _, row in report_df.iterrows():
        c.drawString(50, y, f"{row['Parameter']} : {row['Value']}")
        y -= 15
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 750
    c.save()
    pdf = buffer.getvalue()
    buffer.close()
    return pdf



##############################################################################
# REPORT EXTRAS  (flow-diagram cover page, effluent page, PDF merge)
# Formerly report_extras.py - now part of this file.
# Merging the pages needs pypdf (pip install pypdf); without it the app falls
# back to the original report plus separate PDFs for the two extra pages.
##############################################################################
import html as _html
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch

_FD_COL = {"b": "#1D4ED8", "o": "#C2410C", "g": "#15803D", "s": "#92400E",
           "k": "#475569", "c": "#7C3AED", "u": "#0369A1", "a": "#B45309", "w": "#0E7490"}
_FD_W, _FD_H = 980, 560


def _fnum(p, key, default=0.0):
    try:
        v = p.get(key, default)
        return float(v) if v is not None else float(default)
    except (TypeError, ValueError):
        return float(default)


# ============================================================================
# EFFLUENT STREAMS
# ============================================================================
def effluent_data(p):
    feed_gpm = _fnum(p, "design_flow", 36)
    feed_lb = _fnum(p, "feed_lb_hr", feed_gpm * 500)
    salt_in = _fnum(p, "salt_lb_hr_calc")
    vapor = _fnum(p, "vapor_flow_lb_hr", _fnum(p, "distillate_lb_hr_calc"))
    vent = vapor * _fnum(p, "vent_pct", 0.75) / 100.0
    dist = vapor - vent
    dist_gpm = dist / 500.0
    cond = _fnum(p, "distillate_cond")
    tds_factor = _fnum(p, "tds_cond_factor", 0.65)
    dist_tds = cond * tds_factor
    dist_c = _fnum(p, "distillate_out_c", 40)
    dist_f = dist_c * 9 / 5 + 32

    cake = _fnum(p, "cake_production_lb_hr")
    cake_solids = _fnum(p, "cake_solids", 90)
    cake_moist = 100 - cake_solids
    cake_tpd = cake * 24 / 2000
    cake_ton_yr = cake * _fnum(p, "operating_hours", 8000) / 2000
    bulk = _fnum(p, "cake_bulk_density", 75)
    cake_yd3_day = cake * 24 / max(bulk, 1) / 27
    container = max(_fnum(p, "container_yd3", 20), 1)
    roll_offs = cake_yd3_day / container

    total_out = dist + vent + cake
    closure = total_out / max(feed_lb, 1e-9) * 100
    water_in = feed_lb - salt_in
    recovery = dist / max(water_in, 1e-9) * 100

    rows = [
        ["Distillate (product water)", f"{dist:,.0f}", f"{dist_gpm:,.1f} GPM",
         f"TDS {dist_tds:,.1f} mg/L ({cond:,.0f} uS/cm)",
         f"{dist_c:,.0f} C / {dist_f:,.0f} F", "Reuse / discharge point (V-410)"],
        ["Salt cake (wet)", f"{cake:,.0f}", f"{cake_tpd:,.1f} TPD",
         f"{cake_solids:,.0f}% solids / {cake_moist:,.0f}% moisture",
         "ambient", "Roll-off containers, landfill / resale"],
        ["NCG vent", f"{vent:,.0f}", "-",
         "Non-condensable gas + trace water vapor", "-", "Vacuum system vent"],
        ["Liquid discharge to drain", "0", "0 GPM",
         "None - zero liquid discharge", "-", "-"],
    ]
    return {
        "rows": rows, "feed_lb": feed_lb, "feed_gpm": feed_gpm, "total_out": total_out,
        "closure": closure, "recovery": recovery, "dist": dist, "dist_gpm": dist_gpm,
        "dist_tds": dist_tds, "dist_cond": cond, "dist_c": dist_c, "vent": vent,
        "cake": cake, "cake_tpd": cake_tpd, "cake_ton_yr": cake_ton_yr,
        "cake_solids": cake_solids, "cake_yd3_day": cake_yd3_day, "roll_offs": roll_offs,
    }


def chemical_rows(p):
    """Chemicals / utility water that enter the circulation pump suction header."""
    return [
        ["Sulfamic acid (100%)", f"{_fnum(p, 'acid_lb_hr'):,.1f} lb/hr",
         f"{_fnum(p, 'acid_solution_gph'):,.1f} gph solution",
         f"{_fnum(p, 'acid_lb_yr') / 2000:,.1f} ton/yr incl. cleaning"],
        ["Defoamer", f"{_fnum(p, 'defoamer_lb_hr'):,.2f} lb/hr",
         f"{_fnum(p, 'defoamer_gph'):,.2f} gph",
         f"{_fnum(p, 'defoamer_lb_hr') * _fnum(p, 'operating_hours', 8000) / 2000:,.1f} ton/yr"],
        ["Utility water to process", f"{_fnum(p, 'uw_to_process_lb_hr'):,.0f} lb/hr",
         f"{_fnum(p, 'uw_to_process_gpm'):,.2f} GPM",
         f"avg demand {_fnum(p, 'uw_avg_gpm'):,.1f} GPM"],
    ]


# ============================================================================
# FLOW DIAGRAM SPEC  (shared by the on-screen SVG and the PDF page)
# Follows the P&ID sheets: electric steam boiler skid (feed water / condensate N,
# steam M) and the crystallizer skid (feed B, slurry D, distillate out H).
# primitives: box, arrow, text, circle + decorations (_DECO_KINDS) - coordinates in a
# 980 x 560 box, y down
# ============================================================================
def flow_spec(p, hl=()):
    g = lambda k, d=0.0: _fnum(p, k, d)
    e = effluent_data(p)
    feed_gpm, feed_lb = e["feed_gpm"], e["feed_lb"]
    vapor = g("vapor_flow_lb_hr", g("distillate_lb_hr_calc"))
    recirc = g("recirculation_flow")
    slurry = g("slurry_blowdown_gpm")
    filtrate = g("filtrate_return_gpm")
    steam = g("steam_rate")
    boiler_kw = g("boiler_selected_kw", g("boiler_power_kw"))

    # ---- process temperatures / pressures (from the Basis, Feed HX, Primary HX and Boiler tabs) ----
    atm = g("atmospheric_pressure", 14.696)
    vp = g("vessel_pressure_sp")                                  # crystallizer pressure, psia
    t_boil, t_cond = g("crystallizer_boiling_temp"), g("mvr_discharge_sat_temp")
    t_disch = g("mvr_discharge_temp_actual")
    d_psig = g("blower_discharge_psig")
    feed_c = g("fhx_feed_temp_c", g("feed_temp_c"))
    pre_out_c, dist_in_c, dist_out_c = g("preheat_out_c"), g("distillate_temp_c"), g("distillate_out_c")
    hx_out_c = (g("cryst_hx_out") - 32) / 1.8 if g("cryst_hx_out") > 0 else t_boil
    steam_psig = g("boiler_pressure", g("steam_pressure"))
    try:
        steam_c = float(tsat_c_from_psia(steam_psig + atm))
    except Exception:
        steam_c = 0.0
    Tc = lambda c: f"{c:.1f}°C" if c > 0 else "-"
    P = []

    def box(x, y, w, h, title, sub="", fill="#EAF2FC", stroke="#1D4ED8", sw=2):
        if title in hl:  # equipment of the open tab: orange outline
            stroke, sw = "#F59E0B", 4
        P.append(("box", x, y, w, h, title, sub, fill, stroke, sw))

    def arrow(pts, c, dash=False):
        P.append(("arrow", pts, c, dash))

    def text(x, y, t, anchor="middle", bold=False, size=10, color="#000000"):
        P.append(("text", x, y, t, anchor, bold, size, "#000000"))       # all diagram text is black

    def deco(*a):
        P.append(tuple(a))

    # ---- skid boundaries (dashed, as on the P&ID) ----
    deco("frame", 255, 62, 140, 236, "#64748B")      # crystallizer vessel
    deco("frame", 456, 70, 198, 118, "#64748B")      # blower + silencer
    deco("frame", 775, 312, 200, 160, "#64748B")     # electric boiler skid
    text(972, 328, "BOILER SKID", "end", True, 9, "#475569")

    # ---- equipment ----
    box(8, 430, 64, 50, "FEED", f"{feed_gpm:.1f} GPM", "#E6F6EC", "#15803D")
    box(150, 266, 90, 56, "E-120", "operating")                  # 2 x 100 % preheaters, 1 operating / 1 standby
    box(150, 350, 90, 56, "E-130", "standby")
    box(270, 420, 80, 48, "Sulfamic acid", f"{g('acid_lb_hr'):,.1f} lb/hr", "#FFF7E0", _FD_COL["a"])
    box(360, 420, 72, 48, "Defoamer", f"{g('defoamer_lb_hr'):,.2f} lb/hr", "#F3E8FF", _FD_COL["c"])
    box(442, 420, 80, 48, "Utility water", f"{g('uw_to_process_gpm'):,.2f} GPM", "#E0F2FE", _FD_COL["u"])
    box(270, 80, 110, 200, "V-200", "Crystallizer", "#DCE9FB")
    box(470, 86, 72, 84, "B-310", "", "#FFF1E0", "#C2410C")
    deco("sil", 575, 100, 64, 24)
    if "P-201" in hl:
        P.append(("circle", 520, 350, 21, "#FDE68A", "#F59E0B"))
    P.append(("circle", 520, 350, 15, "#FFFFFF", "#1D4ED8"))
    text(520, 353, "P-201", size=8.5, bold=True)
    box(640, 220, 170, 54, "E-220", "", "#EAF2FC")
    box(540, 420, 100, 44, "Oberlin filter", "", "#F5EBDD", "#92400E")
    box(656, 420, 76, 44, "Salt cake", f"{e['cake']:,.0f} lb/hr", "#F5EBDD", "#92400E", 3)
    box(850, 400, 110, 46, "Boiler", f"{steam:,.0f} lb/hr", "#F1F5F9", "#475569")
    box(890, 18, 80, 44, "Distillate", f"{e['dist_gpm']:.1f} GPM", "#E6F6EC", "#15803D", 3)

    # ---- feed (B) -> pre-HX -> circulation pump suction header (chemicals tie in here) ----
    # cold side: feed header -> E-120 (operating) / E-130 (standby) -> suction header
    arrow([(72, 455), (108, 455), (108, 282), (150, 282)], "b")
    arrow([(108, 392), (150, 392)], "b")
    arrow([(240, 282), (264, 282), (264, 350), (505, 350)], "b")
    arrow([(240, 392), (264, 392), (264, 350)], "b")
    arrow([(325, 280), (325, 350)], "b")
    arrow([(310, 420), (310, 350)], "a")
    arrow([(396, 420), (396, 350)], "c")
    arrow([(482, 420), (482, 350)], "u")

    # ---- recirculation loop: P-201 -> E-220 tubes -> vessel ----
    arrow([(535, 350), (665, 350), (665, 274)], "b")
    arrow([(640, 247), (380, 247)], "b")

    # ---- vapor -> blower -> silencer -> E-220 shell; NCG vent; boiler steam makeup ----
    arrow([(380, 112), (462, 112)], "o")
    arrow([(420, 112), (420, 62)], "o")
    arrow([(542, 112), (575, 112)], "o")
    arrow([(639, 112), (720, 112), (720, 220)], "o")
    arrow([(905, 400), (905, 170), (785, 170), (785, 220)], "k")      # boiler steam -> E-220 shell (own inlet)

    # ---- slurry -> Oberlin filter -> cake; filtrate back to the feed ----
    arrow([(590, 350), (590, 420)], "s")
    arrow([(640, 442), (656, 442)], "s")
    arrow([(550, 420), (550, 400), (500, 400), (500, 356)], "b", True)

    # ---- distillate: E-220 -> distillate pump -> pre-HX -> TP -> distillate out (H) ----
    # hot side: E-220 -> distillate pump -> supply header -> E-120 / E-130 -> return header -> TP -> out (H)
    arrow([(745, 274), (745, 530), (248, 530), (248, 306), (240, 306)], "g")
    arrow([(248, 364), (240, 364)], "g")
    arrow([(150, 306), (124, 306)], "g")
    arrow([(150, 364), (124, 364), (124, 40), (890, 40)], "g")

    # ---- steam condensate: E-220 -> condensate pump -> boiler (N) ----
    arrow([(790, 274), (790, 423), (850, 423)], "w")

    # ---- valves, instruments, connectors ----
    for vx, vy in ((310, 385), (396, 385), (482, 385), (590, 392)):
        deco("valve", vx, vy, "v")
    # preheater isolation valves (cold in / cold out / hot out of each exchanger)
    for vx, vy in ((138, 282), (138, 392), (252, 282), (256, 392), (137, 306), (137, 364)):
        deco("valve", vx, vy, "h")
    deco("cv", 850, 170, "h")                        # steam control valve
    deco("cv", 790, 392, "v")                        # condensate control valve
    deco("check", 828, 423, "h")                     # boiler feed check valve
    deco("valve", 800, 40, "h")                      # distillate out valve
    deco("relief", 300, 80)                          # crystallizer pressure relief
    deco("relief", 940, 400)                         # boiler safety valve
    deco("gauge", 555, 322, 350)
    deco("pump", 790, 350)                           # condensate pump
    deco("pump", 745, 500)                           # distillate pump
    deco("bubble", 850, 20, 9, 40)                   # TP (temperature point) on distillate out
    for cx_, cy_, ltr in ((96, 455, "B"), (590, 368, "D"), (760, 40, "H"), (905, 250, "M"), (790, 296, "N")):
        deco("conn", cx_, cy_, 9)
        text(cx_, cy_ + 3.5, ltr, "middle", True, 9)
    text(850, 23, "TP", "middle", True, 7.5)

    # ---- labels ----
    text(40, 424, f"{feed_lb:,.0f} lb/hr")
    text(195, 250, "2 x 100% PREHEATERS", "middle", True, 8.5, "#334155")
    text(195, 261, "1 operating, 1 standby", "middle", False, 8, "#334155")
    text(440, 342, "Pump suction header", size=9)
    text(510, 238, f"Recirc. {recirc:,.0f} GPM", bold=True)
    text(632, 264, "E-220", "end", True, 10)
    text(632, 276, "Primary HX", "end", False, 9)
    text(398, 127, "Vapor", "start", True, 8.5)
    text(398, 138, Tc(t_boil), "start", False, 8.5)
    text(398, 149, f"{vp:.2f} psia" if vp > 0 else "-", "start", False, 8.5)
    text(398, 160, f"{vapor:,.0f} lb/hr", "start", False, 8.5)
    text(426, 58, f"NCG vent {e['vent']:,.0f} lb/hr", "start", False, 9.5)
    text(607, 142, "SILENCER", "middle", False, 8)
    text(728, 140, "Compressed", "start", False, 9)
    text(728, 151, "vapor", "start", False, 9)
    text(898, 196, f"Steam {steam:,.0f} lb/hr", "end", False, 9.5)
    text(898, 207, f"{steam_psig:.0f} psig / {steam_c:.0f}°C" if steam_c > 0 else "-", "end", True, 9)
    text(804, 300, f"Condensate ~{steam_c:.0f}°C" if steam_c > 0 else "Condensate", "start", False, 9)
    text(738, 292, "Distillate", "end", False, 9)
    text(598, 386, f"Slurry {slurry:.1f} GPM", "start")
    text(608, 399, f"{Tc(t_boil)} / {vp:.1f} psia" if vp > 0 else "-", "start", False, 9)
    text(536, 414, "Filtrate", "end", False, 9)
    text(400, 494, f"Filtrate {filtrate:.1f} GPM returns to the pump suction header", "middle", False, 9.5)
    text(802, 354, "Condensate pump", "start", False, 9)
    text(753, 504, "Distillate pump", "start", False, 9)
    text(450, 522, f"Distillate {e['dist_gpm']:.1f} GPM: {Tc(dist_in_c)} in -> {Tc(dist_out_c)} out of pre-HX (heat recovered)",
         "middle", False, 9.5)
    text(905, 460, f"Electric steam boiler {boiler_kw:,.0f} kW" if boiler_kw else "Electric steam boiler",
         "middle", False, 8.5)
    text(930, 12, "EFFLUENT", "middle", True, 9, "#B91C1C")
    text(694, 414, "EFFLUENT", "middle", True, 9, "#B91C1C")

    # ---- temperature / pressure tags ----
    text(40, 413, f"Feed {Tc(feed_c)}", "middle", True, 8.5)
    text(195, 339, f"Feed out {Tc(pre_out_c)}", "middle", True, 8.5)
    text(130, 196, "Distillate out", "start", False, 8.5)
    text(130, 207, Tc(dist_out_c), "start", True, 8.5)
    text(333, 309, f"Vessel {vp:.2f} psia ({vp - atm:+.1f} psig)" if vp > 0 else "Vessel -", "start", True, 8.5)
    text(333, 320, f"Brine boils {Tc(t_boil)}", "start", False, 8.5)
    text(662, 97, f"Discharge {d_psig:.1f} psig ({d_psig + atm:.1f} psia)" if d_psig > 0 else "Discharge -", "start", True, 8.5)
    text(662, 86, f"{t_disch:.0f}°C dry, condenses {Tc(t_cond)}" if t_disch > 0 else "-", "start", False, 8.5)
    text(490, 262, f"Brine out {Tc(hx_out_c)}", "middle", False, 8.5)
    text(604, 341, Tc(t_boil), "middle", False, 8.5)
    text(632, 288, f"Shell {Tc(t_cond)}", "end", False, 8.5)
    text(838, 16, Tc(dist_out_c), "end", True, 8.5)

    # ---- legend ----
    lx, ly = 8, 110
    for i, (k, t) in enumerate([("b", "Brine / feed"), ("o", "Vapor"), ("g", "Distillate"),
                                ("s", "Solids / slurry"), ("k", "Steam"), ("w", "Condensate"),
                                ("a", "Sulfamic acid"), ("c", "Defoamer"), ("u", "Utility water")]):
        P.append(("arrow", [(lx, ly + i * 14), (lx + 24, ly + i * 14)], k, False))
        text(lx + 30, ly + i * 14 + 3.5, t, "start", size=9.5)
    return P


def _fd_head(p0, p1, size):
    import math
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    bx, by = p1[0] - size * math.cos(ang), p1[1] - size * math.sin(ang)
    px, py = -math.sin(ang) * size * 0.5, math.cos(ang) * size * 0.5
    return [(p1[0], p1[1]), (bx + px, by + py), (bx - px, by - py)]


# ============================================================================
# 3D EQUIPMENT RENDERERS (SVG on screen + reportlab canvas for the PDF)
# Equipment is drawn as shaded cylinders / vessels / pumps, pipes as tubes.
# ============================================================================
def _shade(hexc, f):
    """f > 0 mixes the colour toward white, f < 0 toward black (|f| <= 1)."""
    h = hexc.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    t = 255 if f > 0 else 0
    a = abs(f)
    return "#%02X%02X%02X" % tuple(int(round(v + (t - v) * a)) for v in (r, g, b))


# cylinder shading: (position, shade factor) from one edge to the other
_GRAD = [(0.0, -0.22), (0.30, 0.55), (0.62, 0.05), (1.0, -0.38)]

# Drawing ops (shared by SVG and PDF):
#   ('rect', x, y, w, h, fill, stroke, lw, grad)      grad: None | 'h' | 'v'
#   ('poly', pts, fill, stroke, lw, grad)
#   ('ell', cx, cy, rx, ry, fill, stroke, lw, None)
#   ('line', pts, color, lw)
#   ('sh', cx, cy, rx, ry)                            ground shadow


def _cyl_v(o, x, y, w, h, fill, st, lw, ry=None, cone=0.0, shadow=True):
    """Vertical cylinder (tank / vessel); cone > 0 adds a conical bottom."""
    if ry is None:
        ry = max(3.0, min(w * 0.14, h * 0.17, 9.0))
    cx = x + w / 2
    if shadow:
        o.append(('sh', cx, y + h, w * 0.55, max(2.5, ry * 0.5)))
    if cone > 0:
        yb = y + h * (1 - cone)
        o.append(('poly', [(x, yb), (x + w, yb), (cx + w * 0.10, y + h), (cx - w * 0.10, y + h)],
                  fill, st, lw, 'h'))
    else:
        yb = y + h - ry
        o.append(('ell', cx, yb, w / 2, ry, _shade(fill, -0.12), st, lw, None))
    o.append(('rect', x, y + ry, w, yb - y - ry, fill, None, 0, 'h'))
    o.append(('line', [(x, y + ry), (x, yb)], st, lw))
    o.append(('line', [(x + w, y + ry), (x + w, yb)], st, lw))
    o.append(('ell', cx, y + ry, w / 2, ry, _shade(fill, 0.6), st, lw, None))


def _cyl_h(o, x, y, w, h, fill, st, lw, shadow=True):
    """Horizontal cylinder (heat exchanger shell, boiler, filter, motor)."""
    rx = max(3.0, min(h * 0.16, 8.0))
    cy = y + h / 2
    if shadow:
        o.append(('sh', x + w / 2, y + h + 1, w * 0.5, 2.5))
    o.append(('ell', x + rx, cy, rx, h / 2, _shade(fill, -0.2), st, lw, None))
    o.append(('rect', x + rx, y, w - 2 * rx, h, fill, None, 0, 'v'))
    o.append(('line', [(x + rx, y), (x + w - rx, y)], st, lw))
    o.append(('line', [(x + rx, y + h), (x + w - rx, y + h)], st, lw))
    o.append(('ell', x + w - rx, cy, rx, h / 2, _shade(fill, 0.35), st, lw, None))


def _bin(o, x, y, w, h, fill, st, lw):
    """Open roll-off container (salt cake)."""
    d = max(5.0, min(10.0, h * 0.25))
    o.append(('sh', x + w / 2, y + h + 1, w * 0.5, 2.5))
    side = [(x + w - d - 2, y + d), (x + w - 2, y), (x + w - 6, y + h - d), (x + w - d - 6, y + h)]
    front = [(x + 2, y + d), (x + w - d - 2, y + d), (x + w - d - 6, y + h), (x + 6, y + h)]
    top = [(x + 2, y + d), (x + w - d - 2, y + d), (x + w - 2, y), (x + d + 2, y)]
    o.append(('poly', side, _shade(fill, -0.30), st, lw, None))
    o.append(('poly', front, fill, st, lw, 'v'))
    o.append(('poly', top, _shade(fill, 0.75), st, lw, None))


def _equip(title, x, y, w, h, fill, st, sw):
    """Ops for one piece of equipment inside its footprint (x, y, w, h).
    Returns ops, text-centre x/y, optional custom text list."""
    lw = max(1.2, sw * 0.8)
    o = []
    cx, cy = x + w / 2, y + h / 2
    tcx, tcy, texts = cx, cy, None
    if title in ("FEED", "Distillate", "Sulfamic acid", "Defoamer", "Utility water"):
        _cyl_v(o, x, y, w, h, fill, st, lw)
        tcy = cy + max(3.0, min(w * 0.14, h * 0.17, 9.0)) * 0.4
    elif title in ("E-120/130", "E-120", "E-130"):  # plate / shell-and-tube pre-HX
        o.append(('sh', cx, y + h, w * 0.5, 2.5))
        o.append(('rect', cx - 4, y, 8, 8, "#9AA9BC", st, lw, 'h'))
        o.append(('rect', cx - 4, y + h - 8, 8, 8, "#9AA9BC", st, lw, 'h'))
        _cyl_h(o, x, y + h * 0.2, 18, h * 0.6, _shade(fill, -0.05), st, lw, False)
        _cyl_h(o, x + w - 18, y + h * 0.2, 18, h * 0.6, _shade(fill, -0.05), st, lw, False)
        _cyl_h(o, x + 12, y + h * 0.1, w - 24, h * 0.8, fill, st, lw, False)
        o.append(('rect', x + 10, y + h * 0.07, 3, h * 0.86, "#64748B", st, 1, None))
        o.append(('rect', x + w - 13, y + h * 0.07, 3, h * 0.86, "#64748B", st, 1, None))
    elif title == "E-220":                          # primary HX: three shell-and-tube sections (as on the P&ID)
        o.append(('sh', cx, y + h, w * 0.5, 3))
        o.append(('rect', x, y, w, h, fill, st, lw, 'v'))
        r = h * 0.40
        for k in range(3):
            ccx = x + w * (2 * k + 1) / 6.0
            o.append(('ell', ccx, cy, r, r, "#FFFFFF", st, 1.2, None))
            zz = [(ccx - r * 0.55 + i * r * 0.275, cy + (-r * 0.45 if i % 2 == 0 else r * 0.45)) for i in range(5)]
            o.append(('line', zz, "#334155", 1.1))
        texts = []                                  # labels are placed beside the exchanger in flow_spec
    elif title == "V-200":                          # crystallizer, conical bottom
        _cyl_v(o, x, y, w, h, fill, st, lw, ry=10, cone=0.2)
        tcy = y + h * 0.42
    elif title == "B-310":                          # lobe blower: housing with two rotors + motor below
        hh = h * 0.62
        o.append(('sh', cx, y + h, w * 0.5, 2.5))
        o.append(('rect', x + 6, y + hh - 2, w - 12, h - hh + 2, _shade(fill, -0.08), st, lw, 'v'))   # motor
        o.append(('rect', x, y, w, hh, fill, st, lw, 'h'))                                           # housing
        o.append(('rect', x - 10, y + hh / 2 - 4, 10, 8, "#E2C9A6", st, 1, 'v'))                      # inlet stub
        o.append(('rect', x + w, y + hh / 2 - 4, 10, 8, "#E2C9A6", st, 1, 'v'))                       # discharge stub
        for yy in (y + hh * 0.28, y + hh * 0.72):
            o.append(('ell', cx, yy, 12, 12, _shade(fill, -0.18), st, 1.2, None))
            o.append(('ell', cx - 2.5, yy - 3, 7, 7, _shade(fill, 0.55), None, 0, None))
        texts = [(cx, y + h - 6, "B-310", 9, True), (cx, y + h + 12, "MVR blower", 9, False)]
    elif title == "Boiler":                         # electric boiler on saddles
        o.append(('sh', cx, y + h, w * 0.5, 2.5))
        o.append(('rect', cx - 4, y + h - 9, 8, 9, "#9AA9BC", st, lw, 'h'))
        o.append(('rect', x + 12, y + h - 9, 9, 9, "#64748B", st, lw, 'h'))
        o.append(('rect', x + w - 21, y + h - 9, 9, 9, "#64748B", st, lw, 'h'))
        o.append(('rect', cx - 9, y, 18, 7, "#94A3B8", st, lw, 'h'))               # terminal box
        _cyl_h(o, x, y + 5, w, h - 14, fill, st, lw, False)
        tcy = y + 5 + (h - 14) / 2
    elif title == "Oberlin filter":                 # filter on legs with discharge hopper
        o.append(('sh', cx, y + h, w * 0.5, 2.5))
        o.append(('rect', cx - 4, y, 8, 6, "#9AA9BC", st, lw, 'h'))
        o.append(('rect', x + 14, y + h - 8, 10, 8, "#64748B", st, lw, 'h'))
        o.append(('rect', x + w - 24, y + h - 8, 10, 8, "#64748B", st, lw, 'h'))
        o.append(('poly', [(cx - 9, y + h - 9), (cx + 9, y + h - 9), (cx + 4, y + h), (cx - 4, y + h)],
                  _shade(fill, -0.15), st, lw, 'h'))
        _cyl_h(o, x, y + 4, w, h - 12, fill, st, lw, False)
        o.append(('rect', x + 7, y + 4, 4, h - 12, _shade(fill, -0.3), st, 1, None))
        o.append(('rect', x + w - 11, y + 4, 4, h - 12, _shade(fill, -0.3), st, 1, None))
        tcy = y + 4 + (h - 12) / 2
    else:                                           # salt cake and anything else: container
        d = max(5.0, min(10.0, h * 0.25))
        _bin(o, x, y, w, h, fill, st, lw)
        tcx, tcy = x + (w - d) / 2, y + d + (h - d) / 2 + 1
    return o, tcx, tcy, texts


def _circle_ops(cx, cy, r, fill, st):
    """Pump P-201 (volute + motor + base) or the orange highlight halo."""
    o = []
    if fill.upper() != "#FFFFFF":
        o.append(('ell', cx, cy, r, r, fill, st, 2, None))
        return o
    o.append(('sh', cx, cy + r + 24, 14, 2.5))
    _cyl_v(o, cx - 8, cy + r - 3, 16, 24, "#9AA9BC", "#334155", 1.2, ry=3, shadow=False)
    o.append(('rect', cx - 13, cy + r + 20, 26, 3, "#475569", "#1E293B", 1, None))
    o.append(('ell', cx, cy, r, r, "#7FA3E8", st, 2, None))
    o.append(('ell', cx - r * 0.14, cy - r * 0.2, r * 0.74, r * 0.74, "#C7D9F8", None, 0, None))
    o.append(('ell', cx - r * 0.36, cy - r * 0.46, r * 0.3, r * 0.2, "#FFFFFF", None, 0, None))
    return o


_DECO_KINDS = ("valve", "gauge", "cv", "check", "relief", "pump", "conn", "bubble", "sil", "frame")


def _dashed(x0, y0, x1, y1, col, dash=7.0, gap=5.0, lw=1.2):
    """Dashed straight line as short solid segments (works in the SVG and the PDF renderer)."""
    out = []
    L = math.hypot(x1 - x0, y1 - y0)
    if L <= 0:
        return out
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    t = 0.0
    while t < L:
        e = min(t + dash, L)
        out.append(('line', [(x0 + ux * t, y0 + uy * t), (x0 + ux * e, y0 + uy * e)], col, lw))
        t += dash + gap
    return out


def _deco_ops(kind, *a):
    """Valves, control valves, check / relief valves, small pumps, connectors, instruments, skid frames."""
    o = []
    gold, dk = "#E3B341", "#6B4E0E"
    if kind == "valve":
        x, y, orient = a
        if orient == "h":
            o.append(('poly', [(x - 6, y - 5), (x - 6, y + 5), (x, y)], gold, dk, 1.1, None))
            o.append(('poly', [(x + 6, y - 5), (x + 6, y + 5), (x, y)], gold, dk, 1.1, None))
            o.append(('line', [(x, y), (x, y - 9)], dk, 1.4))
            o.append(('ell', x, y - 10, 4.5, 1.8, "#B91C1C", dk, 0.8, None))
        else:
            o.append(('poly', [(x - 5, y - 6), (x + 5, y - 6), (x, y)], gold, dk, 1.1, None))
            o.append(('poly', [(x - 5, y + 6), (x + 5, y + 6), (x, y)], gold, dk, 1.1, None))
            o.append(('line', [(x, y), (x + 9, y)], dk, 1.4))
            o.append(('ell', x + 10, y, 1.8, 4.5, "#B91C1C", dk, 0.8, None))
    elif kind == "gauge":
        x, y, py = a
        o.append(('line', [(x, y + 5), (x, py)], "#475569", 1.6))
        o.append(('ell', x, y, 6, 6, "#FFFFFF", "#334155", 1.4, None))
        o.append(('line', [(x, y), (x + 3, y - 3.5)], "#B91C1C", 1.2))
        o.append(('ell', x, y, 1.2, 1.2, "#334155", None, 0, None))
    elif kind == "cv":                                   # control valve with diaphragm actuator
        x, y, orient = a
        if orient == "h":
            o.append(('poly', [(x - 6, y - 5), (x - 6, y + 5), (x, y)], gold, dk, 1.1, None))
            o.append(('poly', [(x + 6, y - 5), (x + 6, y + 5), (x, y)], gold, dk, 1.1, None))
            o.append(('line', [(x, y), (x, y - 12)], dk, 1.4))
            o.append(('ell', x, y - 16, 6, 4, "#FFFFFF", dk, 1.1, None))
        else:
            o.append(('poly', [(x - 5, y - 6), (x + 5, y - 6), (x, y)], gold, dk, 1.1, None))
            o.append(('poly', [(x - 5, y + 6), (x + 5, y + 6), (x, y)], gold, dk, 1.1, None))
            o.append(('line', [(x, y), (x + 12, y)], dk, 1.4))
            o.append(('ell', x + 16, y, 4, 6, "#FFFFFF", dk, 1.1, None))
    elif kind == "check":                                # check valve, flow to the right
        x, y, _orient = a
        o.append(('poly', [(x - 6, y - 5), (x - 6, y + 5), (x + 6, y)], gold, dk, 1.1, None))
        o.append(('line', [(x + 6, y - 6), (x + 6, y + 6)], dk, 1.6))
    elif kind == "relief":                               # pressure relief / safety valve, vent up-right
        x, y = a
        o.append(('line', [(x, y), (x, y - 6)], "#475569", 2.0))
        o.append(('poly', [(x - 5, y - 6), (x + 5, y - 6), (x, y - 14)], gold, dk, 1.1, None))
        o.append(('line', [(x - 4, y - 18), (x + 4, y - 18)], dk, 1.6))
        o.append(('line', [(x, y - 14), (x, y - 18)], dk, 1.4))
        o.append(('line', [(x + 6, y - 10), (x + 16, y - 10)], "#B91C1C", 1.6))
        o.append(('poly', [(x + 18, y - 10), (x + 13, y - 13), (x + 13, y - 7)], "#B91C1C", "#B91C1C", 0.5, None))
    elif kind == "pump":                                 # small pump symbol
        x, y = a
        o.append(('ell', x, y, 8, 8, "#FFFFFF", "#1D4ED8", 1.6, None))
        o.append(('poly', [(x - 4, y - 5), (x - 4, y + 5), (x + 5, y)], "#1D4ED8", "#1D4ED8", 0.5, None))
    elif kind == "conn":                                 # off-sheet connector (diamond with a letter)
        x, y, r = a
        o.append(('poly', [(x, y - r), (x + r, y), (x, y + r), (x - r, y)], "#FFFFFF", "#334155", 1.4, None))
    elif kind == "bubble":                               # instrument bubble with stem to the pipe
        x, y, r, py = a
        o.append(('line', [(x, y + r), (x, py)], "#475569", 1.6))
        o.append(('ell', x, y, r, r, "#FFFFFF", "#334155", 1.4, None))
    elif kind == "sil":                                  # silencer
        x, y, w, h = a
        r = h / 2.0
        o.append(('sh', x + w / 2, y + h + 1, w * 0.45, 2.5))
        o.append(('ell', x + r, y + r, r, r, "#A8B5C6", "#334155", 1.3, None))
        o.append(('rect', x + r, y, w - 2 * r, h, "#CBD5E1", None, 0, 'v'))
        o.append(('line', [(x + r, y), (x + w - r, y)], "#334155", 1.3))
        o.append(('line', [(x + r, y + h), (x + w - r, y + h)], "#334155", 1.3))
        o.append(('ell', x + w - r, y + r, r, r, "#E2E8F0", "#334155", 1.3, None))
    elif kind == "frame":                                # dashed skid boundary
        x, y, w, h, col = a
        o += _dashed(x, y, x + w, y, col)
        o += _dashed(x + w, y, x + w, y + h, col)
        o += _dashed(x + w, y + h, x, y + h, col)
        o += _dashed(x, y + h, x, y, col)
    return o


def _ops_svg(ops, out, gid):
    def fillspec(fill, grad):
        if not grad:
            return fill
        gid[0] += 1
        n = gid[0]
        x2, y2 = ("1", "0") if grad == 'h' else ("0", "1")
        out.append(f'<linearGradient id="g{n}" x1="0" y1="0" x2="{x2}" y2="{y2}">' + "".join(
            f'<stop offset="{p}" stop-color="{_shade(fill, f)}"/>' for p, f in _GRAD) + '</linearGradient>')
        return f"url(#g{n})"

    def stk(st, lw):
        return f' stroke="{st}" stroke-width="{lw:.2f}"' if st else ' stroke="none"'

    for op in ops:
        k = op[0]
        if k == 'sh':
            _, cx, cy, rx, ry = op
            out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
                       f'fill="#0F172A" fill-opacity="0.14"/>')
        elif k == 'rect':
            _, x, y, w, h, fill, st, lw, grad = op
            out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                       f'fill="{fillspec(fill, grad)}"{stk(st, lw)}/>')
        elif k == 'poly':
            _, pts, fill, st, lw, grad = op
            p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            out.append(f'<polygon points="{p}" fill="{fillspec(fill, grad)}"{stk(st, lw)} '
                       f'stroke-linejoin="round"/>')
        elif k == 'ell':
            _, cx, cy, rx, ry, fill, st, lw, grad = op
            out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{rx:.1f}" ry="{ry:.1f}" '
                       f'fill="{fill}"{stk(st, lw)}/>')
        elif k == 'line':
            _, pts, col, lw = op
            d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            out.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="{lw:.2f}" '
                       f'stroke-linecap="round"/>')


def flow_diagram_svg(p, hl=()):
    out = ['<svg viewBox="0 0 %d %d" xmlns="http://www.w3.org/2000/svg" '
           'style="display:block;margin:0 auto;max-width:980px;width:100%%;height:auto;'
           'font-family:Segoe UI,Arial,sans-serif">' % (_FD_W, _FD_H),
           f'<rect x="0" y="0" width="{_FD_W}" height="{_FD_H}" rx="12" fill="#FFFFFF" stroke="#9FB3CB"/>']
    gid = [0]
    for pr in flow_spec(p, hl):
        kind = pr[0]
        if kind == "box":
            _, x, y, w, h, title, sub, fill, stroke, sw = pr
            ops, tcx, tcy, texts = _equip(title, x, y, w, h, fill, stroke, sw)
            _ops_svg(ops, out, gid)
            if texts is None:
                ty = tcy + (-4 if sub else 4)
                out.append(f'<text x="{tcx}" y="{ty}" text-anchor="middle" font-size="11" '
                           f'font-weight="700" fill="#000">{_html.escape(title)}</text>')
                if sub:
                    out.append(f'<text x="{tcx}" y="{ty + 13}" text-anchor="middle" '
                               f'font-size="10" fill="#000">{_html.escape(sub)}</text>')
            else:
                for tx_, ty_, tt, sz, bd in texts:
                    out.append(f'<text x="{tx_}" y="{ty_}" text-anchor="middle" font-size="{sz}" '
                               f'font-weight="{700 if bd else 500}" fill="#000">{_html.escape(tt)}</text>')
        elif kind == "circle":
            _, cx, cy, r, fill, stroke = pr
            _ops_svg(_circle_ops(cx, cy, r, fill, stroke), out, gid)
        elif kind in _DECO_KINDS:
            _ops_svg(_deco_ops(kind, *pr[1:]), out, gid)
        elif kind == "arrow":
            _, pts_, c, dash = pr
            col = _FD_COL[c]
            dd = "M" + " L".join(f"{x},{y}" for x, y in pts_)
            da = ' stroke-dasharray="6 4"' if dash else ""
            out.append(f'<path d="{dd}" fill="none" stroke="{_shade(col, -0.45)}" stroke-width="6" '
                       f'stroke-linejoin="round"{da}/>')
            out.append(f'<path d="{dd}" fill="none" stroke="{col}" stroke-width="4.2" '
                       f'stroke-linejoin="round"{da}/>')
            out.append(f'<path d="{dd}" fill="none" stroke="{_shade(col, 0.55)}" stroke-width="1.4" '
                       f'stroke-opacity="0.85" stroke-linejoin="round"{da}/>')
            hd = _fd_head(pts_[-2], pts_[-1], 9)
            out.append('<polygon points="%s" fill="%s" stroke="%s" stroke-width="0.8"/>' %
                       (" ".join(f"{x:.1f},{y:.1f}" for x, y in hd), col, _shade(col, -0.45)))
        elif kind == "text":
            _, x, y, t, anchor, bold, size, color = pr
            out.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
                       f'font-weight="{700 if bold else 500}" fill="{color}">{_html.escape(t)}</text>')
    out.append("</svg>")
    return "".join(out)


def draw_flow(c, p, x0, y0, w, h):
    """Draw the 3D equipment diagram on a reportlab canvas inside (x0, y0, w, h); y0 = bottom."""
    s = min(w / _FD_W, h / _FD_H)
    ox = x0 + (w - _FD_W * s) / 2
    oy = y0 + (h - _FD_H * s) / 2
    tx = lambda x: ox + x * s
    ty = lambda y: oy + (_FD_H - y) * s

    def make_path(op):
        pa = c.beginPath()
        k = op[0]
        if k == 'rect':
            _, x, y, bw, bh, *_r = op
            pa.rect(tx(x), ty(y + bh), bw * s, bh * s)
            bb = (tx(x), ty(y + bh), tx(x + bw), ty(y))
        elif k == 'poly':
            pts = op[1]
            pa.moveTo(tx(pts[0][0]), ty(pts[0][1]))
            for a, b in pts[1:]:
                pa.lineTo(tx(a), ty(b))
            pa.close()
            xs, ys = [tx(a) for a, _ in pts], [ty(b) for _, b in pts]
            bb = (min(xs), min(ys), max(xs), max(ys))
        else:  # 'ell'
            _, cx, cy, rx, ry = op[:5]
            pa.ellipse(tx(cx - rx), ty(cy + ry), 2 * rx * s, 2 * ry * s)
            bb = (tx(cx - rx), ty(cy + ry), tx(cx + rx), ty(cy - ry))
        return pa, bb

    def draw_ops(ops):
        for op in ops:
            k = op[0]
            if k == 'sh':
                _, cx, cy, rx, ry = op
                c.setFillColor(colors.HexColor("#CBD5E1"))
                c.ellipse(tx(cx - rx), ty(cy + ry), tx(cx + rx), ty(cy - ry), stroke=0, fill=1)
            elif k == 'line':
                _, pts, col, lw = op
                c.setStrokeColor(colors.HexColor(col))
                c.setLineWidth(lw * s)
                c.setLineCap(1)
                pa = c.beginPath()
                pa.moveTo(tx(pts[0][0]), ty(pts[0][1]))
                for a, b in pts[1:]:
                    pa.lineTo(tx(a), ty(b))
                c.drawPath(pa, stroke=1, fill=0)
            else:
                if k == 'rect':
                    fill, st, lw, grad = op[5], op[6], op[7], op[8]
                elif k == 'poly':
                    fill, st, lw, grad = op[2], op[3], op[4], op[5]
                else:
                    fill, st, lw, grad = op[5], op[6], op[7], None
                pa, bb = make_path(op)
                done = False
                if grad:
                    c.saveState()
                    try:
                        c.clipPath(pa, stroke=0, fill=0)
                        cols = [colors.HexColor(_shade(fill, f)) for _, f in _GRAD]
                        pos = [q for q, _ in _GRAD]
                        if grad == 'h':
                            c.linearGradient(bb[0], bb[1], bb[2], bb[1], cols, pos, extend=True)
                        else:
                            c.linearGradient(bb[0], bb[3], bb[0], bb[1], cols, pos, extend=True)
                        done = True
                    except Exception:
                        done = False
                    c.restoreState()
                if not done:
                    c.setFillColor(colors.HexColor(fill))
                    c.drawPath(pa, stroke=0, fill=1)
                if st:
                    c.setStrokeColor(colors.HexColor(st))
                    c.setLineWidth(lw * s)
                    c.setLineJoin(1)
                    c.drawPath(pa, stroke=1, fill=0)

    def put_text(x, y, t, anchor, bold, size, color):
        c.setFillColor(colors.HexColor(color))
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size * s)
        if anchor == "middle":
            c.drawCentredString(tx(x), ty(y), t)
        elif anchor == "end":
            c.drawRightString(tx(x), ty(y), t)
        else:
            c.drawString(tx(x), ty(y), t)

    c.setFillColor(colors.white)
    c.setStrokeColor(colors.HexColor("#9FB3CB"))
    c.roundRect(ox, oy, _FD_W * s, _FD_H * s, 8, fill=1, stroke=1)

    for pr in flow_spec(p):
        kind = pr[0]
        if kind == "box":
            _, x, y, bw, bh, title, sub, fill, stroke, sw = pr
            ops, tcx, tcy, texts = _equip(title, x, y, bw, bh, fill, stroke, sw)
            draw_ops(ops)
            if texts is None:
                yy = tcy + (-4 if sub else 4)
                put_text(tcx, yy, title, "middle", True, 11, "#000000")
                if sub:
                    put_text(tcx, yy + 13, sub, "middle", False, 10, "#000000")
            else:
                for tx_, ty_, tt, sz, bd in texts:
                    put_text(tx_, ty_, tt, "middle", bd, sz, "#000000")
        elif kind == "circle":
            _, cx, cy, r, fill, stroke = pr
            draw_ops(_circle_ops(cx, cy, r, fill, stroke))
        elif kind in _DECO_KINDS:
            draw_ops(_deco_ops(kind, *pr[1:]))
        elif kind == "arrow":
            _, pts_, k, dash = pr
            col = _FD_COL[k]
            c.setLineCap(0)
            c.setLineJoin(1)
            c.setDash(*((5 * s, 3 * s) if dash else (1, 0)))
            for wd, cc in ((6, _shade(col, -0.45)), (4.2, col), (1.4, _shade(col, 0.55))):
                c.setStrokeColor(colors.HexColor(cc))
                c.setLineWidth(wd * s)
                path = c.beginPath()
                path.moveTo(tx(pts_[0][0]), ty(pts_[0][1]))
                for a, b in pts_[1:]:
                    path.lineTo(tx(a), ty(b))
                c.drawPath(path, stroke=1, fill=0)
            c.setDash(1, 0)
            hd = _fd_head(pts_[-2], pts_[-1], 9)
            c.setFillColor(colors.HexColor(col))
            c.setStrokeColor(colors.HexColor(_shade(col, -0.45)))
            c.setLineWidth(0.8 * s)
            hp = c.beginPath()
            hp.moveTo(tx(hd[0][0]), ty(hd[0][1]))
            for a, b in hd[1:]:
                hp.lineTo(tx(a), ty(b))
            hp.close()
            c.drawPath(hp, stroke=1, fill=1)
        elif kind == "text":
            _, x, y, t, anchor, bold, size, color = pr
            put_text(x, y, t, anchor, bold, size, color)
    c.setDash(1, 0)


# ============================================================================
# PAGE 1  -  PROCESS FLOW DIAGRAM
# ============================================================================
def build_cover_pdf(p, meta):
    buf = BytesIO()
    pw, ph = landscape(letter)  # 792 x 612
    c = canvas.Canvas(buf, pagesize=(pw, ph))
    navy = colors.HexColor("#0B1D36")

    c.setFillColor(navy)
    c.rect(0, ph - 62, pw, 62, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, ph - 30, "MVR Crystallizer - Process Flow Diagram")
    c.setFont("Helvetica", 10)
    c.drawString(30, ph - 48, f"{meta.get('project', '')}   |   {meta.get('client', '')}")
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(pw - 30, ph - 30, meta.get("company", ""))
    c.setFont("Helvetica", 10)
    c.drawRightString(pw - 30, ph - 48,
                      f"Project {meta.get('project_no', '')}   Rev {meta.get('revision', '')}")

    draw_flow(c, p, 30, 118, pw - 60, ph - 62 - 118 - 12)

    # key figures strip
    e = effluent_data(p)
    kp = [
        ("Feed", f"{e['feed_gpm']:.1f} GPM"),
        ("Distillate", f"{e['dist_gpm']:.1f} GPM"),
        ("Salt cake", f"{e['cake_tpd']:.1f} TPD"),
        ("Recirculation", f"{_fnum(p, 'recirculation_flow'):,.0f} GPM"),
        ("Blower", f"{_fnum(p, 'blower_kw'):,.0f} kW"),
        ("Boiler", f"{_fnum(p, 'boiler_selected_kw'):,.0f} kW"),
    ]
    bw = (pw - 60 - 5 * 8) / 6
    for i, (lab, val) in enumerate(kp):
        x = 30 + i * (bw + 8)
        c.setFillColor(colors.HexColor("#EAF2FC"))
        c.setStrokeColor(colors.HexColor("#9FB3CB"))
        c.roundRect(x, 70, bw, 38, 5, fill=1, stroke=1)
        c.setFillColor(colors.HexColor("#475569"))
        c.setFont("Helvetica", 8)
        c.drawCentredString(x + bw / 2, 96, lab.upper())
        c.setFillColor(colors.black)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(x + bw / 2, 78, val)

    # title block
    tb = Table([
        ["Doc. No.", meta.get("doc_no", "") or "-", "Prepared", meta.get("prepared_by", "") or "-",
         "Checked", meta.get("checked_by", "") or "-", "Approved", meta.get("approved_by", "") or "-",
         "Date", datetime.now().strftime("%d %b %Y")],
    ], colWidths=[48, 80, 48, 80, 48, 80, 52, 80, 30, 66])
    tb.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTNAME", (0, 0), (0, 0), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, 0), "Helvetica-Bold"),
        ("FONTNAME", (4, 0), (4, 0), "Helvetica-Bold"), ("FONTNAME", (6, 0), (6, 0), "Helvetica-Bold"),
        ("FONTNAME", (8, 0), (8, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9FB3CB")),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#EAF2FC")),
        ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#EAF2FC")),
        ("BACKGROUND", (4, 0), (4, 0), colors.HexColor("#EAF2FC")),
        ("BACKGROUND", (6, 0), (6, 0), colors.HexColor("#EAF2FC")),
        ("BACKGROUND", (8, 0), (8, 0), colors.HexColor("#EAF2FC")),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    tw, th = tb.wrapOn(c, pw, ph)
    tb.drawOn(c, (pw - tw) / 2, 26)
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawCentredString(pw / 2, 14, "Flows are design values from the calculation; flow diagram follows the P&ID layout, "
                                    "not a P&ID.")
    c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================================
# LAST PAGE  -  EFFLUENT OUTPUT
# ============================================================================
def build_effluent_pdf(p, meta):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, textColor=colors.HexColor("#0B1D36"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#0B1D36"),
                        spaceBefore=12)
    cell = ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8, leading=10)
    cellb = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)

    def mk(header, rows, widths):
        data = [[Paragraph(h, cellb) for h in header]] + [[Paragraph(str(x), cell) for x in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13294B")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9FB3CB")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F7FB")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    e = effluent_data(p)
    el = []
    el.append(Paragraph("Effluent Output Summary", h1))
    el.append(Paragraph(
        f"{meta.get('project', '')} - {meta.get('client', '')}. Design feed "
        f"{e['feed_gpm']:.1f} GPM ({e['feed_lb']:,.0f} lb/hr). The plant is a zero-liquid-discharge "
        "system: all feed water leaves as distillate, the dissolved salt leaves as wet cake and a "
        "small non-condensable vent leaves the vacuum system.", body))

    el.append(Paragraph("Effluent streams", h2))
    el.append(mk(["Stream", "lb/hr", "Flow", "Quality / composition", "Temperature", "Destination"],
                 e["rows"], [1.35 * inch, 0.6 * inch, 0.75 * inch, 2.0 * inch, 0.85 * inch, 1.75 * inch]))

    el.append(Paragraph("Closure and recovery", h2))
    el.append(mk(["Item", "Value"], [
        ["Total feed in", f"{e['feed_lb']:,.0f} lb/hr"],
        ["Total effluent out (distillate + vent + cake)", f"{e['total_out']:,.0f} lb/hr"],
        ["Overall mass closure", f"{e['closure']:.1f} %"],
        ["Recovery of feed water as distillate", f"{e['recovery']:.1f} %"],
        ["Salt cake per year", f"{e['cake_ton_yr']:,.0f} ton/yr"],
        ["Cake volume / roll-off containers", f"{e['cake_yd3_day']:,.0f} yd3/day / {e['roll_offs']:.1f} per day"],
    ], [4.2 * inch, 3.1 * inch]))

    el.append(Paragraph("Chemicals and utility water entering the process", h2))
    el.append(mk(["Item", "Rate", "Volume", "Annual / note"], chemical_rows(p),
                 [2.2 * inch, 1.4 * inch, 1.8 * inch, 1.9 * inch]))
    el.append(Spacer(1, 8))
    el.append(Paragraph(
        "Sulfamic acid, defoamer and utility water are injected into the circulation-pump suction "
        "header (P&amp;ID). Utility water that enters the brine is evaporated and leaves with the "
        "distillate. The boiler steam / condensate loop is closed and does not cross the "
        "process boundary.", body))
    doc.build(el)
    return buf.getvalue()


# ============================================================================
# MERGE
# ============================================================================
def merge_pdfs(parts):
    """Concatenate PDF byte strings. Returns None if pypdf / PyPDF2 is not installed."""
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError:
        try:
            from PyPDF2 import PdfReader, PdfWriter
        except ImportError:
            return None
    w = PdfWriter()
    for b in parts:
        for page in PdfReader(BytesIO(b)).pages:
            w.add_page(page)
    out = BytesIO()
    w.write(out)
    return out.getvalue()


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="HSKH ZLD Evaporator & Crystallizer Design Suite",
    page_icon="🪸",
    layout="wide"
)

# ============================================================
# PROJECT MANAGER  (create / save / open - everything is inside this file)
# Projects are stored as  projects/<name>.json  next to this file.
# ============================================================
import hashlib
import tempfile

PROJECT_DIR = Path(__file__).resolve().parent / "projects"
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PROJECT_DATA = {
    "feed_flow": 60.0, "operating_hours": 8000, "tds": 72000, "sulfate": 46252,
    "calcium": 4322, "evaporator_recovery": 90, "crystal_recovery": 85,
}


def _safe_name(name):
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", str(name or ""))
    return re.sub(r"\s+", " ", s).strip().strip(".").strip()[:80].strip()


def _pfile(name):
    return PROJECT_DIR / f"{name}.json"


def _json_default(o):
    if hasattr(o, "item") and callable(o.item):  # numpy numbers
        try:
            return o.item()
        except Exception:
            pass
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(data):
    """Plain JSON-safe dict (accepts old files that wrapped everything in {"project_data": ...})."""
    if isinstance(data, dict) and isinstance(data.get("project_data"), dict):
        data = data["project_data"]
    if not isinstance(data, dict):
        return {}
    d = json.loads(json.dumps(data, default=_json_default, skipkeys=True))
    d.pop("_meta", None)
    return d


def _hash(data):
    return hashlib.md5(json.dumps(_clean(data), sort_keys=True).encode("utf-8")).hexdigest()


def list_projects():
    """[(name, last_modified)] newest first."""
    items = []
    for f in PROJECT_DIR.glob("*.json"):
        if not f.name.startswith((".", "_")):
            items.append((f.stem, f.stat().st_mtime))
    return sorted(items, key=lambda x: x[1], reverse=True)


def get_project_list():
    return sorted(n for n, _ in list_projects())


def _unique_name(base):
    base = _safe_name(base) or "New Project"
    existing = {n.lower() for n, _ in list_projects()}
    if base.lower() not in existing:
        return base
    i = 2
    while f"{base} ({i})".lower() in existing:
        i += 1
    return f"{base} ({i})"


def _write(name, data):
    """Safe write: temp file then replace; the previous version is kept as <name>.json.bak."""
    payload = _clean(data)
    payload["_meta"] = {"project": name, "saved": datetime.now().isoformat(timespec="seconds")}
    target = _pfile(name)
    if target.exists():
        try:
            (PROJECT_DIR / f"{name}.json.bak").write_bytes(target.read_bytes())
        except OSError:
            pass
    fd, tmp = tempfile.mkstemp(dir=str(PROJECT_DIR), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=_json_default, skipkeys=True)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read(name):
    f = _pfile(name)
    for cand in (f, PROJECT_DIR / f"{name}.json.bak"):  # falls back to the backup copy
        try:
            return _clean(json.loads(cand.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return None


def _remember_last(name):
    try:
        (PROJECT_DIR / ".last_project").write_text(name, encoding="utf-8")
    except OSError:
        pass


def _last_project():
    try:
        return (PROJECT_DIR / ".last_project").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


for _k, _v in (("pm_hash", {}), ("pm_saved_at", {}), ("pm_autosave", True),
               ("pm_show_flow", True), ("pm_flow_expanded", True)):
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ---- first start: re-open the last project, or create one ----
if "project_data" not in st.session_state or "current_project" not in st.session_state:
    _name, _data = _last_project(), None
    if _name:
        _data = _read(_name)
    if _data is None:
        for _n, _m in list_projects():
            _data = _read(_n)
            if _data is not None:
                _name = _n
                break
    if _data is None:
        _name, _data = _unique_name("New Project"), copy.deepcopy(DEFAULT_PROJECT_DATA)
        _write(_name, _data)
    st.session_state.project_data = _data
    st.session_state.current_project = _name
    st.session_state.pm_hash[_name] = _hash(_data)
    st.session_state.pm_rename_name = _name
    _remember_last(_name)


def _persist(name, data, force=False):
    """Write the project only when something changed (or force). Returns True if a file was written."""
    h = _hash(data)
    if not force and st.session_state.pm_hash.get(name) == h and _pfile(name).exists():
        return False
    try:
        _write(name, data)
    except OSError as e:
        st.session_state["pm_error"] = str(e)
        return False
    st.session_state.pm_hash[name] = h
    st.session_state.pm_saved_at[name] = datetime.now().strftime("%H:%M:%S")
    st.session_state.pop("pm_error", None)
    _remember_last(name)
    return True


def save_project(project_name, data=None):
    """Called by every tab: saves the project when auto-save is on and something changed."""
    if not st.session_state.get("pm_autosave", True):
        return None
    return _persist(project_name, st.session_state.project_data if data is None else data)


def load_project(project_name):
    return _read(project_name) or {}


def _flash(kind, text):
    st.session_state["pm_flash"] = (kind, text)


def _activate(name, data, message):
    """Switch to another project; widget values are cleared so every tab re-reads the project data."""
    for k in list(st.session_state.keys()):
        if k not in ("project_data", "current_project") and not str(k).startswith("pm_"):
            del st.session_state[k]
    st.session_state.project_data = _clean(data)
    st.session_state.current_project = name
    st.session_state.pm_hash[name] = _hash(data)
    st.session_state.pm_rename_name = name
    st.session_state.pop("pm_error", None)
    _remember_last(name)
    _flash("success", message)


def _cb_save():
    name = st.session_state.current_project
    if _persist(name, st.session_state.project_data, force=True):
        _flash("success", f"Saved “{name}”")
    else:
        _flash("error", f"Could not save: {st.session_state.get('pm_error', 'unknown error')}")


def _cb_new():
    raw = _safe_name(st.session_state.get("pm_new_name", ""))
    if not raw:
        _flash("error", "Enter a project name.")
        return
    if st.session_state.get("pm_autosave", True):
        _persist(st.session_state.current_project, st.session_state.project_data)
    name, data = _unique_name(raw), copy.deepcopy(DEFAULT_PROJECT_DATA)
    try:
        _write(name, data)
    except OSError as e:
        _flash("error", f"Could not create the project: {e}")
        return
    _activate(name, data, f"Created “{name}”")
    st.session_state["pm_new_name"] = "New Project"


def _cb_open():
    name = st.session_state.get("pm_open_select")
    if not name or name == st.session_state.current_project:
        return
    data = _read(name)
    if data is None:
        _flash("error", f"“{name}” could not be opened.")
        return
    if st.session_state.get("pm_autosave", True):
        _persist(st.session_state.current_project, st.session_state.project_data)
    _activate(name, data, f"Opened “{name}”")


def _status():
    name = st.session_state.current_project
    if st.session_state.get("pm_error"):
        return "bad", "Could not save: " + st.session_state["pm_error"]
    dirty = _hash(st.session_state.project_data) != st.session_state.pm_hash.get(name)
    if not st.session_state.get("pm_autosave", True):
        return "warn", "Auto-save OFF · " + ("unsaved changes" if dirty else "saved")
    when = st.session_state.pm_saved_at.get(name)
    if not when:
        try:
            when = datetime.fromtimestamp(_pfile(name).stat().st_mtime).strftime("%d %b %H:%M")
        except OSError:
            when = "not saved yet"
    return ("warn", "Unsaved changes") if dirty else ("ok", f"Auto-saved {when}")


def _card_html():
    p = st.session_state.project_data
    level, text = _status()
    chips = []
    try:
        flow = p.get("design_flow") or p.get("feed_flow")
        if flow:
            chips.append(f"{float(flow):,.0f} GPM")
        if p.get("tds"):
            chips.append(f"TDS {float(p['tds']):,.0f} mg/L")
        if p.get("vessel_pressure_sp"):
            chips.append(f"{float(p['vessel_pressure_sp']):.2f} psia")
    except (TypeError, ValueError):
        pass
    chip_html = "".join(f'<span class="pm-chip">{_html.escape(c)}</span>' for c in chips)
    return (f'<div class="pm-card"><div class="pm-label">ACTIVE PROJECT</div>'
            f'<div class="pm-name">{_html.escape(st.session_state.current_project)}</div>'
            f'<div class="pm-status {level}">● {_html.escape(text)}</div>'
            f'<div class="pm-chips">{chip_html}</div></div>')


def _autosave():
    """Last line of the script: saves the final state of every tab and refreshes the project card."""
    if st.session_state.get("pm_autosave", True):
        _persist(st.session_state.current_project, st.session_state.project_data)
    ph = st.session_state.get("pm_card_ph")
    if ph is not None:
        try:
            ph.markdown(_card_html(), unsafe_allow_html=True)
        except Exception:
            pass


def _wide(fn, *args, **kw):
    try:
        return fn(*args, width="stretch", **kw)
    except TypeError:
        return fn(*args, use_container_width=True, **kw)


st.markdown("""
<style>
/* ---- sidebar: light panel, black text ---- */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#F4F7FB 0%,#DDE6F1 100%) !important;
    border-right:1px solid #9FB3CB !important; box-shadow:none !important;
}
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{ background:transparent !important; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] li,
section[data-testid="stSidebar"] summary,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{ color:#000000 !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] summary svg{ fill:#000000 !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details{
    background:rgba(255,255,255,.75); border:1px solid #9FB3CB !important; border-radius:10px; }
section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea{
    background:#FFFFFF !important; color:#000000 !important; -webkit-text-fill-color:#000000 !important; caret-color:#000000 !important; }
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="select"] > div{
    background:#FFFFFF !important; border:1px solid #7E95B2 !important; border-radius:8px !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] *{
    color:#000000 !important; -webkit-text-fill-color:#000000 !important; background-color:#FFFFFF !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] svg{ fill:#000000 !important; }
section[data-testid="stSidebar"] .stButton > button{
    min-height:2.3rem; padding:.4rem .6rem; border-radius:9px; font-weight:700; border:none;
    background:linear-gradient(180deg,#3B82F6,#2563EB) !important;
    box-shadow:0 3px 0 #1E40AF, 0 5px 10px rgba(0,0,0,.18) !important; }
section[data-testid="stSidebar"] .stButton > button *{ color:#FFFFFF !important; }
section[data-testid="stSidebar"] .stButton > button:disabled{ background:#B7C4D6 !important; box-shadow:none !important; opacity:1 !important; }
section[data-testid="stSidebar"] .stButton > button:disabled *{ color:#333333 !important; }
.pm-brand{ display:flex; align-items:center; gap:10px; margin:2px 0 12px 0; }
.pm-logo{ width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center;
          font-size:20px; background:linear-gradient(135deg,#2563EB,#0EA5E9); }
.pm-title{ color:#000000 !important; font-weight:800; font-size:15px; line-height:1.15; }
.pm-sub{ color:#000000 !important; font-size:11.5px; }
.pm-card{ background:#FFFFFF; border:1px solid #9FB3CB; border-radius:12px; padding:12px 14px; margin-bottom:12px;
          box-shadow:0 2px 6px rgba(0,0,0,.10); }
.pm-label{ color:#000000 !important; font-size:10.5px; font-weight:800; letter-spacing:1.1px; }
.pm-name{ color:#000000 !important; font-size:17px; font-weight:800; margin:3px 0 6px 0; word-break:break-word; }
.pm-status{ font-size:12px; font-weight:700; }
.pm-status.ok{ color:#0B5A2F !important; } .pm-status.warn{ color:#7A4B00 !important; } .pm-status.bad{ color:#9B1C1C !important; }
.pm-chips{ display:flex; flex-wrap:wrap; gap:6px; margin-top:9px; }
.pm-chip{ background:#DCE9FB; color:#000000 !important; border-radius:20px; padding:2px 10px; font-size:11px; font-weight:700; }
.pm-section{ color:#000000 !important; font-size:10.5px; font-weight:800; letter-spacing:1.1px; margin:14px 0 6px 0; }
.pm-footer{ color:#000000 !important; font-size:10.5px; margin-top:16px; text-align:center; word-break:break-all; }
</style>
""", unsafe_allow_html=True)

_projects = list_projects()
_mtimes = dict(_projects)

with st.sidebar:
    st.markdown(
        '<div class="pm-brand"><div class="pm-logo">🪸</div><div>'
        '<div class="pm-title">ZLD Design Suite</div>'
        '<div class="pm-sub">Evaporator &amp; Crystallizer</div></div></div>', unsafe_allow_html=True)

    st.session_state["pm_card_ph"] = st.empty()
    st.session_state["pm_card_ph"].markdown(_card_html(), unsafe_allow_html=True)

    _fl = st.session_state.pop("pm_flash", None)
    if _fl:
        try:
            st.toast(_fl[1], icon="✅" if _fl[0] == "success" else "⚠️")
        except Exception:
            pass
        (st.success if _fl[0] == "success" else st.error)(_fl[1])

    # ---- SAVE ----
    _wide(st.button, "💾  Save project", type="primary", on_click=_cb_save, key="pm_btn_save")
    st.checkbox("Auto-save every change", key="pm_autosave")
    st.checkbox("🗺️ Show flow diagram in every tab", key="pm_show_flow")
    st.checkbox("Open flow diagrams by default", key="pm_flow_expanded",
                disabled=not st.session_state.get("pm_show_flow", True),
                help="Each diagram can also be minimised with its own arrow.")

    # ---- CREATE ----
    with st.expander("＋ Create new project"):
        if "pm_new_name" not in st.session_state:
            st.session_state["pm_new_name"] = "New Project"
        st.text_input("Project name", key="pm_new_name")
        _wide(st.button, "Create project", on_click=_cb_new, key="pm_btn_new")

    # ---- OPEN ----
    st.markdown('<div class="pm-section">OPEN PROJECT</div>', unsafe_allow_html=True)
    if _projects:
        _names = [n for n, _ in _projects]
        if st.session_state.get("pm_open_select") not in _names:
            st.session_state["pm_open_select"] = (
                st.session_state.current_project if st.session_state.current_project in _names else _names[0])
        st.selectbox("Saved projects", _names, key="pm_open_select", label_visibility="collapsed",
                     format_func=lambda
                         n: f"{n}  ·  {datetime.fromtimestamp(_mtimes.get(n, 0)).strftime('%d %b %Y, %H:%M')}")
        _wide(st.button, "📂  Open project", on_click=_cb_open, key="pm_btn_open",
              disabled=st.session_state.get("pm_open_select") == st.session_state.current_project)
    else:
        st.caption("No saved projects yet.")

    st.markdown(f'<div class="pm-footer">{len(_projects)} project(s)<br>{_html.escape(str(PROJECT_DIR))}</div>',
                unsafe_allow_html=True)

project_name = st.session_state.current_project


def _idx(options, value, default=0):
    """Index of a saved value in a list of options (restores selectboxes when a project is opened)."""
    options = list(options)
    return options.index(value) if value in options else default


#####################################################################
# TITLE BLOCK
#####################################################################
st.markdown("""
<style>
.app-header{
    position: fixed;
    top: 15px;
    left: 50%;
    transform: translateX(-50%);
    width: calc(90% - 20px);
    z-index: 999999;
    height: 70px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    color: #000000;
    background: transparent;
}
.block-container{ padding-top:0px !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
🪸 HSKH ZLD Evaporator & Crystallizer Design Suite
</div>
""", unsafe_allow_html=True)

############################################################
# COMPLETE VIBRANT ENGINEERING THEME
############################################################
st.markdown("""
<style>
/* ---------- APP BACKGROUND ---------- */
.stApp{
    background: linear-gradient(135deg, #E2E8F0 0%, #8FA8C2 50%, #4F667D 100%);
    border:5px solid #0F172A;
    border-radius:20px;
    margin:10px;
    min-height:98vh;
    box-sizing:border-box;
}

/* ---------- MAIN WORK AREA (75 % of the screen) ---------- */
.block-container{
    max-width:75% !important;
    margin:auto !important;
    background:transparent;
    border:none;
    box-shadow:none;
    padding:25px;
}

/* ---------- TITLE BANNER ---------- */
.title-banner{
    position: sticky; top: 50px; z-index: 999;
    background: linear-gradient(135deg, #0F172A, #1D4ED8, #0EA5E9);
    padding: 20px; border-radius: 15px; margin-bottom: 20px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.25);
}
.title-banner h1{ color:white; text-align:center; margin:0; font-size:42px; font-weight:700; }

/* ---------- KPI CARDS ---------- */
[data-testid="stMetric"]{
    background:linear-gradient(180deg, #FFFFFF, #F8FAFC);
    border-radius:16px;
    border-left:6px solid #2563EB;
    padding:12px;
    box-shadow:0 4px 10px rgba(0,0,0,0.15);
    transition:all .2s ease;
}
[data-testid="stMetric"]:hover{ transform:translateY(-3px); box-shadow:0 10px 18px rgba(0,0,0,0.20); }
[data-testid="stMetricLabel"]{ font-size:11px !important; font-weight:600 !important; color:#64748B !important; }
[data-testid="stMetricValue"]{ font-size:20px !important; font-weight:700 !important; color:#0F172A !important; }

/* ---------- 3D BUTTONS ---------- */
.stButton > button{
    width:100%; color:white; font-weight:700; border:none; border-radius:12px; padding:12px;
    background:linear-gradient(180deg, #38BDF8, #2563EB);
    box-shadow:0 5px 0 #1E40AF, 0 10px 18px rgba(0,0,0,0.25);
    transition:all .15s ease;
}
.stButton > button:hover{ transform:translateY(-2px); box-shadow:0 7px 0 #1E40AF, 0 14px 24px rgba(0,0,0,0.30); }
.stButton > button:active{ transform:translateY(3px); box-shadow:0 2px 0 #1E40AF; }

/* ---------- INPUT BOXES ---------- */
.stTextInput input, .stNumberInput input{
    border-radius:10px !important; border:2px solid #CBD5E1 !important; background:white !important;
}
.stSelectbox{ border-radius:10px; }

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab"]{
    background:white; border-radius:10px; padding:10px 15px; margin-right:4px; font-weight:600;
    box-shadow:0 3px 6px rgba(0,0,0,0.10);
}
.stTabs [aria-selected="true"]{
    background:linear-gradient(135deg, #1D4ED8, #0EA5E9) !important; color:white !important;
}

/* ---------- DATAFRAME ---------- */
[data-testid="stDataFrame"]{
    background:white; border-radius:15px; overflow:hidden; box-shadow:0 6px 14px rgba(0,0,0,0.12);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Top header: dark text on the light gradient */
[data-testid="stHeader"]{
    background: linear-gradient(135deg, #E2E8F0 0%, #8FA8C2 50%, #4F667D 100%) !important;
    backdrop-filter: blur(8px);
}
[data-testid="stToolbar"],
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
[data-testid="stAppDeployButton"],
[data-testid="stDecoration"],
#MainMenu,
footer{ display: none !important; visibility: hidden !important; }
header{ background: transparent !important; }
[data-testid="stHeader"] *{ color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

#####################################################################
# Professional Dashboard Theme
#####################################################################
st.markdown("""
<style>
.main{ background-color:#F3F7FB; }
.block-container{ padding-top:0px !important; }
.main-title{ font-size:34px; font-weight:bold; color:#003366; }
.project-bar{ background-color:white; padding:12px; border-radius:10px; margin-bottom:15px; }
.section{ background:white; padding:15px; border-radius:10px; box-shadow:0px 3px 10px rgba(0,0,0,0.08); }
.stTabs [data-baseweb="tab"]{
    font-size:16px; font-weight:bold; background-color:#E6EDF5; border-radius:8px; padding:10px 18px;
}
.stTabs [aria-selected="true"]{ background:#003366 !important; color:white !important; }
</style>
""", unsafe_allow_html=True)

#####################################################################
# READABLE WIDGETS  (Streamlit may run in dark mode: its light text, inputs and
# alert boxes were unreadable on this light page. Everything in the main area
# is forced to dark text on white / light boxes.)
#####################################################################
st.markdown("""
<style>
:root{ color-scheme: light; }

/* ---------- headings, labels, normal text ---------- */
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] h4, [data-testid="stMain"] h5, [data-testid="stMain"] h6,
[data-testid="stMain"] [data-testid="stHeading"] *,
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] li,
[data-testid="stMain"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stMain"] label,
[data-testid="stMain"] [data-testid="stCheckbox"] *,
[data-testid="stMain"] [data-testid="stRadio"] *,
[data-testid="stMain"] [data-testid="stExpander"] summary *{
    color:#000000 !important;
}
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] *{ color:#000000 !important; }

/* ---------- input boxes: white with dark text ---------- */
[data-testid="stMain"] input, [data-testid="stMain"] textarea{
    background:#FFFFFF !important; color:#000000 !important;
    -webkit-text-fill-color:#000000 !important; caret-color:#000000 !important;
}
[data-testid="stMain"] [data-baseweb="input"],
[data-testid="stMain"] [data-baseweb="textarea"],
[data-testid="stMain"] [data-testid="stNumberInputContainer"],
[data-testid="stMain"] [data-baseweb="select"] > div{
    background:#FFFFFF !important; border:1px solid #7E95B2 !important; border-radius:8px !important;
}
[data-testid="stMain"] [data-baseweb="select"] div,
[data-testid="stMain"] [data-baseweb="select"] span,
[data-testid="stMain"] [data-baseweb="select"] input{
    background-color:#FFFFFF !important; color:#000000 !important;
    -webkit-text-fill-color:#000000 !important;
}
[data-testid="stMain"] [data-baseweb="select"] svg{ fill:#000000 !important; }

/* number input  - / +  buttons */
[data-testid="stMain"] [data-testid="stNumberInput"] button{
    background:#E6EEF8 !important; color:#000000 !important; border:none !important; box-shadow:none !important;
}
[data-testid="stMain"] [data-testid="stNumberInput"] button svg{ fill:#000000 !important; color:#000000 !important; }

/* disabled inputs */
[data-testid="stMain"] input:disabled, [data-testid="stMain"] textarea:disabled{
    background:#E3E8EE !important; color:#4A5F7A !important; -webkit-text-fill-color:#4A5F7A !important;
}

/* ---------- alert boxes (st.info / success / warning / error) ---------- */
[data-testid="stMain"] [data-testid="stAlert"]{
    background:#EAF2FC !important; border:1px solid #9FB3CB !important; border-left:6px solid #2563EB !important;
    border-radius:10px !important;
}
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentSuccess"]){ background:#E8F6EE !important; border-left-color:#16A34A !important; }
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentWarning"]){ background:#FFF5E0 !important; border-left-color:#E08A00 !important; }
[data-testid="stMain"] [data-testid="stAlert"]:has([data-testid="stAlertContentError"]){ background:#FDECEC !important; border-left-color:#DC2626 !important; }
[data-testid="stMain"] [data-testid="stAlert"] *{ color:#000000 !important; }

/* ---------- expanders, slider values, file uploader ---------- */
[data-testid="stMain"] [data-testid="stExpander"] details{
    background:rgba(255,255,255,.85); border:1px solid #9FB3CB !important; border-radius:10px;
}
[data-testid="stMain"] [data-testid="stSliderThumbValue"],
[data-testid="stMain"] [data-testid="stTickBarMin"],
[data-testid="stMain"] [data-testid="stTickBarMax"]{ color:#000000 !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"]{ background:#FFFFFF !important; border:1px dashed #6B829E !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] *{ color:#000000 !important; }
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] button{ background:#E6EEF8 !important; }

/* ---------- buttons: blue with white text (new Streamlit button test-ids included) ---------- */
[data-testid="stMain"] .stButton button, [data-testid="stMain"] .stDownloadButton button,
[data-testid="stMain"] [data-testid="stBaseButton-secondary"], [data-testid="stMain"] [data-testid="stBaseButton-primary"],
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button{
    background:linear-gradient(180deg,#38BDF8,#2563EB) !important; border:none !important;
    border-radius:12px !important; box-shadow:0 4px 0 #1E40AF, 0 8px 14px rgba(0,0,0,.22) !important;
}
[data-testid="stMain"] .stButton button *, [data-testid="stMain"] .stDownloadButton button *,
[data-testid="stMain"] [data-testid="stBaseButton-secondary"] *, [data-testid="stMain"] [data-testid="stBaseButton-primary"] *,
[data-testid="stMain"] [data-testid="stFormSubmitButton"] button *{ color:#FFFFFF !important; font-weight:700 !important; }

/* ---------- metric cards: readable sizes ---------- */
[data-testid="stMain"] [data-testid="stMetricLabel"], [data-testid="stMain"] [data-testid="stMetricLabel"] *{
    color:#334155 !important; font-size:12px !important; font-weight:600 !important; }
[data-testid="stMain"] [data-testid="stMetricValue"], [data-testid="stMain"] [data-testid="stMetricValue"] *{
    color:#0F172A !important; font-size:18px !important; font-weight:700 !important; }
[data-testid="stMain"] [data-testid="stMetricDelta"], [data-testid="stMain"] [data-testid="stMetricDelta"] *{
    font-size:11px !important; }
</style>
""", unsafe_allow_html=True)

###############################################################################
# TOP TAB NAVIGATION
# Navy toolbar, Material icons, tabs stretch to fill each row.
# Selectors match Streamlit 1.64 (tabs are div[role="tab"]).
###############################################################################
st.markdown("""
<style>
/* ---- remove the empty gaps left by st.markdown(<style>) blocks above the tabs ---- */
[data-testid="stElementContainer"]:has(style){ display:none !important; }
[data-testid="stElementContainer"]:has(.app-header){ margin-bottom:-1rem !important; }

/* tabs start right under the fixed title banner (top 15px + 70px high) */
.stTabs{ margin-top:68px !important; }

/* ---- tab bar panel: frozen at the top, 2 rows of 8 ---- */
.stTabs [role="tablist"]{
    position:sticky !important;
    top:68px !important;
    z-index:990 !important;
    display:grid !important;
    grid-template-columns:repeat(8, minmax(0, 1fr)) !important;
    gap:4px !important;
    padding:5px !important;
    margin-bottom:14px !important;
    background:linear-gradient(180deg, #13294B 0%, #0B1D36 100%) !important;
    border:1px solid #0B1D36 !important;
    border-radius:0 0 10px 10px !important;
    box-shadow:0 6px 18px rgba(11,29,54,.35) !important;
    overflow:visible !important;
}
/* cover strip above the frozen tab bar: page-coloured */
.stTabs [role="tablist"]::before{
    content:"";
    position:absolute;
    left:-1px;
    right:-1px;
    bottom:100%;
    height:90px;
    z-index:-1;
    pointer-events:none;
    background:linear-gradient(135deg, #E2E8F0 0%, #8FA8C2 50%, #4F667D 100%) fixed;
}
.stTabs .react-aria-SelectionIndicator{ display:none !important; }

/* ---- individual tabs ---- */
.stTabs [role="tab"]{
    justify-content:center !important;
    margin:0 !important;
    min-width:0 !important;
    height:30px !important;
    padding:0 6px !important;
    overflow:hidden !important;
    background:rgba(255,255,255,.05) !important;
    border:1px solid rgba(255,255,255,.08) !important;
    border-radius:6px !important;
    box-shadow:none !important;
    color:#C3D1E6 !important;
    cursor:pointer;
    transition:background .15s ease, color .15s ease, border-color .15s ease;
}
.stTabs [role="tab"] [data-testid="stMarkdownContainer"]{ min-width:0 !important; overflow:hidden !important; }
.stTabs [role="tab"] p{
    font-size:12px !important; font-weight:600 !important; letter-spacing:.1px;
    white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
    color:inherit !important; margin:0 !important;
}
.stTabs [role="tab"] span[role="img"]{ font-size:15px !important; color:#6CB2F0 !important; margin-right:3px; }
.stTabs [role="tab"]:hover{
    background:rgba(255,255,255,.13) !important;
    border-color:rgba(255,255,255,.22) !important;
    color:#FFFFFF !important;
}

/* ---- selected tab ---- */
.stTabs [role="tab"][aria-selected="true"]{
    background:#FFFFFF !important;
    border-color:#FFFFFF !important;
    color:#0B1D36 !important;
    box-shadow:inset 0 -3px 0 #F59E0B, 0 2px 6px rgba(0,0,0,.25) !important;
}
.stTabs [role="tab"][aria-selected="true"] span[role="img"]{ color:#1D4ED8 !important; }
</style>
""", unsafe_allow_html=True)

(
    tab1, tab2, tab3, tab_chem, tab4, tab5, tab6,
    tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14,
) = st.tabs([
    ":material/water_drop: Water",
    ":material/tune: Design Basis",
    ":material/swap_horiz: Feed Preheater",
    ":material/biotech: Chem. & Utilities",
    ":material/autorenew: Recirc Pump",
    ":material/thermostat: Primary HX",
    ":material/science: Cryst. Thermal",
    ":material/propane_tank: Cryst. Vessel",
    ":material/air: MVR Blower",
    ":material/local_fire_department: Boiler",
    ":material/filter_alt: Filter & Filtrate",
    ":material/balance: Mass Balance",
    ":material/inventory_2: Equipment",
    ":material/picture_as_pdf: Report",
    ":material/account_tree: Controls",
])

###############################################################################
# FLOW DIAGRAM IN EVERY TAB
# Each tab reserves a slot at its top (flow_slot); all slots are filled at the
# very end of the script (_render_flow_slots), so the diagram always shows the
# values of the current run. Sidebar switch hides it everywhere; the expander
# arrow minimises one panel. The equipment of the open tab is outlined orange.
###############################################################################
import streamlit.components.v1 as components

_FLOW_SLOTS = []


def flow_slot(hl=()):
    _FLOW_SLOTS.append((st.container(), tuple(hl)))


def _render_flow_slots():
    if not st.session_state.get("pm_show_flow", True):
        return
    expanded = bool(st.session_state.get("pm_flow_expanded", True))
    cache = {}
    for _box, _hl in _FLOW_SLOTS:
        if _hl not in cache:
            cache[_hl] = flow_diagram_svg(st.session_state.project_data, _hl)
        with _box:
            with st.expander("🗺️ Process flow diagram", expanded=expanded):
                components.html(cache[_hl], height=_FD_H + 30)


##############################################################################
# WATER & FEED TANK SYSTEM
##############################################################################

with tab1:
    st.header("💧 Feed Water & Feed Tank System")
    flow_slot(('FEED',))

    st.markdown("### Upload Water Analysis")

    uploaded_excel = st.file_uploader("Upload Water Quality XLSX", type=["xlsx"], key="water_upload")

    # ==========================================================
    # FEED WATER BASIS
    # ==========================================================
    WATER_BASIS_OPTIONS = ["Typical RO Concentrate", "Maximum RO Concentrate", "Upset Unsoftened Feed"]
    water_basis = st.selectbox(
        "Feed Water Basis", WATER_BASIS_OPTIONS,
        index=_idx(WATER_BASIS_OPTIONS, st.session_state.project_data.get("water_basis")))

    # ==========================================================
    # WATER CHEMISTRY TABLES
    # ==========================================================
    parameter_list = [
        "Arsenic", "Bromide", "Cadmium", "Calcium", "Chloride", "Chromium", "Hardness", "Lead",
        "Magnesium", "Mercury", "Nitrate/Nitrite", "Potassium", "Selenium", "Silver", "Sodium",
        "Sulfate", "TDS", "TSS", "Silica", "Conductivity", "Alkalinity", "Bicarbonate Alkalinity",
        "pH", "Temperature",
    ]

    unit_list = [
        "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L", "mg/L as CaCO3", "mg/L",
        "mg/L", "ng/L", "mg/L as N", "mg/L", "mg/L", "mg/L", "mg/L",
        "mg/L", "mg/L", "mg/L", "mg/L", "µS/cm", "mg/L as CaCO3", "mg/L as CaCO3",
        "SU", "°F",
    ]

    if water_basis == "Typical RO Concentrate":
        value_list = [
            0.01, 439, 0.02, 514, 3779, 0.45, 2310, 0.21,
            251, 1629, 586, 1779, 0.87, 0.21, 19807,
            37720, 72000, 100, 50, 120000, 1029, 1029,
            7.50, 60,
        ]
    elif water_basis == "Maximum RO Concentrate":
        value_list = [
            2.57, 1029, 1.44, 2057, 12343, 7.2, 18500, 7.2,
            1003, 3600, 6171, 2367, 5.14, 7.2, 26343,
            49000, 80000, 100, 50, 130000, 2057, 2057,
            7.50, 60,
        ]
    else:
        value_list = [
            0.01, 703, 0.02, 4322, 5728, 0.45, 52800, 0.21,
            10234, 1629, 789, 1520, 0.87, 0.21, 1502,
            46252, 72000, 100, 50, 120000, 764, 764,
            7.79, 60,
        ]

    default_water = pd.DataFrame({
        "Parameter": parameter_list,
        "Units": unit_list,
        "Value": value_list
    })

    # ==========================================================
    # LOAD SAVED PROJECT
    # ==========================================================
    if "water_quality" in st.session_state.project_data:
        water_df = pd.DataFrame(st.session_state.project_data["water_quality"])
    else:
        water_df = default_water

    # ==========================================================
    # EXCEL IMPORT
    # ==========================================================
    if uploaded_excel is not None:
        try:
            water_df = pd.read_excel(uploaded_excel)
            st.success("✅ Water Quality Loaded")
        except Exception:
            st.error("Excel format not recognized")

    # ==========================================================
    # FEED CONDITIONS
    # ==========================================================
    st.subheader("Feed Conditions")

    c1, c2 = st.columns(2)

    with c1:
        feed_flow = st.number_input(
            "Feed Flow (GPM)", min_value=1.0,
            value=float(st.session_state.project_data.get("feed_flow", 36.0)))

    with c2:
        PROJECT_CASE_OPTIONS = ["ZLD", "Salt Recovery", "Volume Reduction"]
        project_case = st.selectbox(
            "Project Basis", PROJECT_CASE_OPTIONS,
            index=_idx(PROJECT_CASE_OPTIONS, st.session_state.project_data.get("project_case")))

    st.divider()

    # ==========================================================
    # FEED TANK SYSTEM
    # ==========================================================
    st.subheader("🏗️ Crystallizer Feed Tank System")

    c4, c5, c6 = st.columns(3)

    with c4:
        feed_tank_volume = st.number_input(
            "Single Feed Tank Volume (gal)",
            value=float(st.session_state.project_data.get("feed_tank_volume", 15500)))

    with c5:
        number_of_feed_tanks = st.number_input(
            "Number of Feed Tanks", min_value=1,
            value=int(st.session_state.project_data.get("number_of_feed_tanks", 3)))

    with c6:
        feed_pumps = st.number_input(
            "Feed Pumps (Duty + Standby)", min_value=1,
            value=int(st.session_state.project_data.get("feed_pumps", 2)))

    total_feed_storage = feed_tank_volume * number_of_feed_tanks
    retention_hours = total_feed_storage / (feed_flow * 60)

    st.info(
        f"Total Feed Storage = {total_feed_storage:,.0f} gal | "
        f"Retention Time = {retention_hours:.1f} hr"
    )

    st.divider()

    # ==========================================================
    # WATER CHEMISTRY
    # ==========================================================
    st.subheader("Incoming Water Constituents")

    edited_water = st.data_editor(
        water_df,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="water_editor"
    )

    water_dict = dict(zip(edited_water["Parameter"], edited_water["Value"]))

    # ==========================================================
    # FEED TEMPERATURE
    # ==========================================================
    feed_temp_f = water_dict.get("Temperature", 60)
    feed_temp_c = (feed_temp_f - 32) * 5 / 9

    st.session_state.project_data["feed_temp_f"] = feed_temp_f
    st.session_state.project_data["feed_temp_c"] = feed_temp_c

    # ==========================================================
    # SAVE DATA
    # ==========================================================
    st.session_state.project_data["feed_flow"] = feed_flow
    st.session_state.project_data["project_case"] = project_case
    st.session_state.project_data["water_basis"] = water_basis

    st.session_state.project_data["feed_tank_volume"] = feed_tank_volume
    st.session_state.project_data["number_of_feed_tanks"] = number_of_feed_tanks
    st.session_state.project_data["feed_pumps"] = feed_pumps
    st.session_state.project_data["total_feed_storage"] = total_feed_storage
    st.session_state.project_data["retention_hours"] = retention_hours

    st.session_state.project_data["water_quality"] = edited_water.to_dict()

    for parameter, value in water_dict.items():
        key = parameter.lower().replace("/", "_").replace(" ", "_")
        st.session_state.project_data[key] = value

    save_project(project_name, st.session_state.project_data)


    # ==========================================================
    # KPI SUMMARY
    # ==========================================================
    st.subheader("📊 Water System Summary")

    k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)

    with k1:
        st.metric("Feed Flow", f"{feed_flow:,.1f} GPM")
    with k2:
        st.metric("TDS", f"{water_dict.get('TDS', 0):,.0f} mg/L")
    with k3:
        st.metric("TSS", f"{water_dict.get('TSS', 0):,.0f} mg/L")
    with k4:
        st.metric("Sulfate", f"{water_dict.get('Sulfate', 0):,.0f} mg/L")
    with k5:
        st.metric("Conductivity", f"{water_dict.get('Conductivity', 0):,.0f} µS/cm")
    with k6:
        st.metric("Storage", f"{total_feed_storage:,.0f} gal")
    with k7:
        st.metric("Feed Temp", f"{feed_temp_f:.1f} °F")
    with k8:
        st.metric("Retention", f"{retention_hours:.1f} hr")

    st.success("✅ Water & Feed Tank Data Auto Saved")

##############################################################################
# DESIGN BASIS
# Requires mvr_thermo.py in the same folder as the app.
##############################################################################

with tab2:
    st.header("📊 Design Basis")
    flow_slot(('V-200', 'B-310', 'E-220'))

    st.markdown(
        "Master design parameters used throughout Evaporator, Crystallizer, Boiler, MVR and Oberlin calculations."
    )

    st.divider()

    ##########################################################################
    # PLANT DESIGN BASIS
    ##########################################################################
    st.subheader("Feed water parameters")

    c1, c2, c3, c4 = st.columns(4)

    CAPACITY_BASIS = {
        "Average Design (36 GPM)": 36.0,
        "Maximum Design (44 GPM)": 44.0,
        "Maximum Capacity (60 GPM)": 60.0,
    }

    # Feed TDS comes from the Feed Water tab (saved there as "tds")
    feed_tds = float(st.session_state.project_data.get("tds", 72000.0))

    # Keyed widget: the value lives in st.session_state, so the capacity-basis
    # dropdown can overwrite it, and typed values are not reset on rerun.
    if "design_flow_input" not in st.session_state:
        st.session_state["design_flow_input"] = float(
            st.session_state.project_data.get("design_flow", 36.0)
        )


    def _apply_capacity_basis():
        st.session_state["design_flow_input"] = CAPACITY_BASIS[
            st.session_state["capacity_basis_select"]
        ]


    with c1:
        capacity_basis = st.selectbox(
            "Capacity Basis",
            list(CAPACITY_BASIS),
            index=_idx(list(CAPACITY_BASIS), st.session_state.project_data.get("capacity_basis")),
            key="capacity_basis_select",
            on_change=_apply_capacity_basis
        )

        design_flow = st.number_input(
            "Design Feed Flow (GPM)",
            min_value=0.0,
            step=1.0,
            key="design_flow_input",
            help="Select a capacity basis or type any flow. Drives the mass balance and MVR sizing below."
        )

    with c2:
        operating_hours = st.number_input(
            "Operating Hours/Year",
            value=float(st.session_state.project_data.get("operating_hours", 8000)),
            key="basis_operating_hours"
        )

    with c3:
        plant_availability = st.number_input(
            "Plant Availability (%)", min_value=50.0, max_value=100.0,
            value=float(st.session_state.project_data.get("plant_availability", 95.0)))

    with c4:
        design_factor = st.number_input(
            "Design Margin (%)", min_value=0.0, max_value=50.0,
            value=float(st.session_state.project_data.get("design_factor", 10.0)))

    st.divider()

    ##########################################################################
    # VACOM DESIGN TARGETS
    ##########################################################################
    st.subheader("Vacom Design Targets")

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        design_distillate_flow = st.number_input(
            "Design Distillate (GPM)",
            value=float(st.session_state.project_data.get("design_distillate_flow", 35.0)))

    with d2:
        salt_production_rate = st.number_input(
            "Salt Production (lb/hr)",
            value=float(st.session_state.project_data.get("salt_production_rate", 2500.0)))

    with d3:
        boiler_capacity = st.number_input(
            "Boiler Capacity (lb/hr)",
            value=float(st.session_state.project_data.get("boiler_capacity", 6900.0)))

    with d4:
        boiler_pressure = st.number_input(
            "Boiler Pressure (psig)",
            value=float(st.session_state.project_data.get("boiler_pressure", 30.0)))

    ##########################################################################
    # CALCULATED MASS BALANCE (driven by Design Feed Flow + Feed TDS)
    ##########################################################################
    st.subheader("Calculated Mass Balance")

    mb0, mb1, mb2, mb3, mb4 = st.columns(5)

    with mb0:
        cake_moisture_pct = st.number_input(
            "Salt Cake Moisture (%)", min_value=0.0, max_value=50.0,
            value=float(st.session_state.project_data.get("cake_moisture_pct", 10.0)),
            help="Free water leaving with the discharged salt."
        )

    feed_sg = 1.0 + 0.00069 * feed_tds / 1000.0  # brine SG estimate
    feed_lb_hr = design_flow * 8.34 * feed_sg * 60.0
    salt_lb_hr = design_flow * 3.785 * feed_tds / 1000.0 * 60.0 / 453.592
    cake_water_lb_hr = (
        salt_lb_hr * cake_moisture_pct / (100.0 - cake_moisture_pct)
        if cake_moisture_pct < 100 else 0.0
    )
    distillate_lb_hr = max(feed_lb_hr - salt_lb_hr - cake_water_lb_hr, 0.0)
    distillate_gpm = distillate_lb_hr / 500.0  # lb/hr -> GPM (8.34 x 60)

    with mb1:
        st.metric("Feed", f"{feed_lb_hr:,.0f} lb/hr",
                  help=f"SG {feed_sg:.3f} at TDS {feed_tds:,.0f} mg/L")

    with mb2:
        st.metric("Distillate (calc.)", f"{distillate_gpm:,.1f} GPM",
                  f"{distillate_gpm - design_distillate_flow:+.1f} vs Vacom target")

    with mb3:
        st.metric("Salt, dry (calc.)", f"{salt_lb_hr:,.0f} lb/hr",
                  f"{salt_lb_hr - salt_production_rate:+,.0f} vs Vacom target")

    with mb4:
        st.metric("Vapor to Blower", f"{distillate_lb_hr:,.0f} lb/hr")

    st.caption(
        f"Feed TDS {feed_tds:,.0f} mg/L taken from the Feed Water tab. "
        "Distillate = feed − dry salt − cake moisture (ZLD, no liquid purge)."
    )

    st.divider()

    ##########################################################################
    # SYSTEM CONFIGURATION
    ##########################################################################
    st.subheader("Process Configuration")

    c5, c6, c7, c8 = st.columns(4)

    _pdx = st.session_state.project_data
    CRYST_SYSTEM_OPTIONS = ["Forced Circulation MVR", "Forced Circulation Steam", "Hybrid MVR + Steam"]
    CRYST_TYPE_OPTIONS = ["Forced Circulation", "DTB (Draft Tube Baffle)"]
    STEAM_SOURCE_OPTIONS = ["Electric Boiler", "Natural Gas Boiler", "Plant Steam"]
    OBJECTIVE_OPTIONS = ["Zero Liquid Discharge", "Salt Recovery", "Volume Reduction"]
    STANDBY_OPTIONS = ["None", "N+1", "2N"]

    with c5:
        crystallizer_system = st.selectbox(
            "Crystallizer System", CRYST_SYSTEM_OPTIONS,
            index=_idx(CRYST_SYSTEM_OPTIONS, _pdx.get("crystallizer_system")))

    with c6:
        crystallizer_type = st.selectbox(
            "Crystallizer Type", CRYST_TYPE_OPTIONS,
            index=_idx(CRYST_TYPE_OPTIONS, _pdx.get("crystallizer_type")))

    with c7:
        steam_source = st.selectbox(
            "Steam Source", STEAM_SOURCE_OPTIONS,
            index=_idx(STEAM_SOURCE_OPTIONS, _pdx.get("steam_source")))

    with c8:
        project_objective = st.selectbox(
            "Project Objective", OBJECTIVE_OPTIONS,
            index=_idx(OBJECTIVE_OPTIONS, _pdx.get("project_objective")))

    st.divider()

    ##########################################################################
    # RECOVERY TARGETS
    ##########################################################################
    st.subheader("Recovery Targets")

    c9, c10, c11, c12 = st.columns(4)

    with c9:
        water_recovery = st.slider(
            "Water Recovery (%)", 80, 99,
            int(st.session_state.project_data.get("water_recovery", 95)))

    with c10:
        crystallizer_solids = st.slider(
            "Crystallizer Slurry Solids (%)", 5, 25,
            int(st.session_state.project_data.get("crystallizer_solids", 15)))

    with c11:
        crystal_recovery = st.slider(
            "Crystal Recovery (%)", 50, 100,
            int(st.session_state.project_data.get("crystal_recovery", 85)))

    with c12:
        overall_recovery = st.slider(
            "Overall ZLD Recovery (%)", 80, 99,
            int(st.session_state.project_data.get("overall_recovery", 95)))

    st.divider()

    ##########################################################################
    # OPERATING CONDITIONS
    ##########################################################################
    st.subheader("Boiler make up water steam operating conditions")

    c13, c14, c15, c16, c17 = st.columns(5)

    with c13:
        operating_pressure = st.number_input(
            "Operating Pressure (psig)",
            value=float(st.session_state.project_data.get("operating_pressure", 3.0)),
            key="design_operating_pressure"
        )

    with c14:
        design_temperature = st.number_input(
            "Design Temperature (°F)",
            value=float(st.session_state.project_data.get("design_temperature", 228)))

    with c15:
        number_of_trains = st.number_input(
            "Number of Trains", min_value=1, max_value=10,
            value=int(st.session_state.project_data.get("number_of_trains", 1)))

    with c16:
        standby_configuration = st.selectbox(
            "Standby Configuration", STANDBY_OPTIONS,
            index=_idx(STANDBY_OPTIONS, _pdx.get("standby_configuration")))

    with c17:
        atmospheric_pressure = st.number_input(
            "Atmospheric Pressure (psia)",
            min_value=10.0, max_value=15.0,
            value=float(st.session_state.project_data.get("atmospheric_pressure", 14.696)),
            step=0.01, format="%.3f",
            help="Used to convert psig to psia. Lower it for site elevation."
        )

    c18, c19 = st.columns(2)

    with c18:
        tss_open = st.number_input(
            "Slurry TSS Low (%)",
            value=float(st.session_state.project_data.get("tss_open", 10)))

    with c19:
        tss_high = st.number_input(
            "Slurry TSS High (%)",
            value=float(st.session_state.project_data.get("tss_high", 15)))

    st.divider()

    ##############################################################################
    # VACOM CONTROL PHILOSOPHY - SETPOINTS
    ##############################################################################
    st.subheader("Vacom Control Philosophy")

    # ---- Design basis presets ------------------------------------------------
    # Vacom basis = operating values from Vacom P&ID line list
    # (ENG-55767-03-00001 Rev B):
    #   24-301  ME-230 -> B-310 (blower suction)   3 psig, 222 F
    #   20-321  V-320 -> E-220 F1 (blower disch.) 15 psig, 250 F
    #   20-202 / 24-221  brine in / out of E-220   238 F -> 243 F
    #   3-400   E-220 F4 -> V-410 (condensate)     210 F
    #   2-121 / 2-131  feed out of E-120 / E-130   200 F
    # Presets write project_data and clear the widget keys, so every tab
    # rebuilds its inputs from the new values on this run.
    from mvr_thermo import FHX_MODES as _FHX_MODES


    def _apply_basis(which):
        p = st.session_state.project_data
        atm = float(p.get("atmospheric_pressure", 14.696))
        if which == "vacom":
            vessel = 3.0 + atm
            disch = 15.0
            cond_c = tsat_c_from_psia(disch + atm)
            vals = {
                "vessel_pressure_sp": round(vessel, 2),
                "blower_discharge_psig": disch,
                "boiling_point_elevation": round((238 - 32) / 1.8 - tsat_c_from_psia(vessel), 1),
                "fhx_subcool_c": round(cond_c - (210 - 32) / 1.8, 1),
                "fhx_mode": _FHX_MODES[1],
                "fhx_out_fixed_c": round((200 - 32) / 1.8, 1),
                "flash_delta_t_input": 5.0,
            }
        else:
            vals = {
                "vessel_pressure_sp": 7.25,
                "blower_discharge_psig": 3.0,
                "boiling_point_elevation": 6.0,
                "fhx_subcool_c": 10.0,
                "fhx_mode": _FHX_MODES[0],
                "fhx_out_fixed_c": 80.0,
                "preheater_efficiency": 80.0,
                "flash_delta_t_input": 10.8,
            }
        vals["vessel_pressure_pv"] = vals["vessel_pressure_sp"]
        vals["flash_delta_t"] = vals["flash_delta_t_input"]
        vals["design_basis_name"] = which
        p.update(vals)
        for k in ("fhx_subcool_c", "fhx_mode", "fhx_out_fixed_c", "fhx_eff", "phx_flash_dt"):
            st.session_state.pop(k, None)


    pb1, pb2, pb3 = st.columns([1, 1, 2])
    with pb1:
        st.button("Load Vacom line-list basis", on_click=_apply_basis, args=("vacom",),
                  help="Vessel 3 psig, discharge 15 psig, brine 238→243 °F, "
                       "distillate 210 °F, feed out of pre-HX 200 °F (P&ID Rev B line list).")
    with pb2:
        st.button("Load UCC 7.25 psia basis", on_click=_apply_basis, args=("ucc",),
                  help="Vessel 7.25 psia, discharge 3 psig, BPE 6 °C, flash ΔT 10.8 °F, "
                       "subcool 10 °C, pre-HX 80% effectiveness.")
    with pb3:
        _bn = st.session_state.project_data.get("design_basis_name")
        if _bn == "vacom":
            st.info("Active basis: **Vacom P&ID line list (Rev B)**")
        elif _bn == "ucc":
            st.info("Active basis: **UCC 7.25 psia**")

    vc1, vc2, vc3, vc4 = st.columns(4)

    with vc1:
        vessel_level_sp = st.number_input(
            "Vessel Level Setpoint (%)", min_value=0.0, max_value=100.0,
            value=float(st.session_state.project_data.get("vessel_level_sp", 50.0)))

    with vc2:
        vessel_pressure_sp = st.number_input(
            "Crystallizer Pressure SP (psia)", min_value=1.0, max_value=40.0,
            value=float(st.session_state.project_data.get("vessel_pressure_sp", 7.25)),
            step=0.05,
            help="Held constant by the blower VFD (pressure controller output = blower speed)."
        )

    with vc3:
        blower_discharge_psig = st.number_input(
            "Blower Discharge Pressure (psig)", min_value=0.0, max_value=40.0,
            value=float(st.session_state.project_data.get("blower_discharge_psig", 3.0)),
            step=0.1,
            help="Compressed vapor pressure to the primary HX shell at design speed."
        )

    with vc4:
        boiling_point_elevation = st.number_input(
            "Boiling Point Elevation (°C)", min_value=0.0, max_value=20.0,
            value=float(st.session_state.project_data.get("boiling_point_elevation", 6.0)),
            step=0.5)

    ##############################################################################
    # MVR CONFIGURATION
    ##############################################################################
    st.subheader("Blower / MVR Configuration")

    c20, c21, c22, c23 = st.columns(4)

    with c20:
        mvr_enabled = st.checkbox(
            "MVR Included", value=st.session_state.project_data.get("mvr_enabled", True))

    with c21:
        steam_economy_target = st.number_input(
            "Steam Economy Target",
            value=float(st.session_state.project_data.get("steam_economy_target", 8.0)))

    with c22:
        blower_efficiency = st.number_input(
            "Blower Efficiency (%)",
            value=float(st.session_state.project_data.get("blower_efficiency", 75.0)))

    with c23:
        motor_efficiency = st.number_input(
            "Motor Efficiency (%)",
            value=float(st.session_state.project_data.get("motor_efficiency", 95.0)))

    st.divider()

    ##############################################################################
    # CRYSTALLIZER + MVR THERMAL DESIGN (AT SETPOINT)
    ##############################################################################
    st.subheader("🌡️ Crystallizer & MVR Thermal Design (at setpoint)")

    blower_discharge_psia = blower_discharge_psig + atmospheric_pressure
    design_vapor_lb_hr = distillate_lb_hr  # from calculated mass balance

    design = mvr_state(
        vessel_psia=vessel_pressure_sp,
        discharge_psia=blower_discharge_psia,
        bpe_c=boiling_point_elevation,
        blower_eff=blower_efficiency / 100.0,
        motor_eff=motor_efficiency / 100.0,
        vapor_lb_hr=design_vapor_lb_hr,
    )

    blower_dp_setpoint = design["blower_dp"]

    if vessel_pressure_sp > atmospheric_pressure:
        st.warning(
            f"Crystallizer pressure SP {vessel_pressure_sp:.2f} psia is ABOVE atmospheric "
            f"({atmospheric_pressure:.2f} psia): the vessel is pressurised, not under vacuum, and brine boils at "
            f"{design['t_boil']:.1f} °C. This is what the 'Vacom line-list basis' preset loads (blower suction 3 psig). "
            "For a vacuum crystallizer use the 'UCC 7.25 psia basis' preset or type a lower setpoint.")

    if design["hx_driving_dt"] <= 0:
        st.error(
            "Condensing temperature of compressed vapor is not above brine boiling "
            "temperature - primary HX cannot transfer heat. Raise discharge pressure "
            "or check BPE."
        )

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown("**Crystallizer vessel**")
        _vp_g = vessel_pressure_sp - atmospheric_pressure
        st.metric("Vessel Pressure", f"{vessel_pressure_sp:.2f} psia",
                  f"{_vp_g:+.2f} psig - " + ("ABOVE atmospheric" if _vp_g > 0 else "vacuum"),
                  delta_color="off")
        st.metric("Water Sat Temp", f"{design['t_sat_vessel']:.1f} °C")
        st.metric("Brine Boiling Temp", f"{design['t_boil']:.1f} °C",
                  help="Saturation temp + BPE. Vapor leaves vessel at this temp (superheated by BPE).")

    with r2:
        st.markdown("**MVR blower**")
        st.metric("Blower ΔP", f"{blower_dp_setpoint:.2f} psi")
        st.metric("Pressure Ratio", f"{design['pressure_ratio']:.2f}")
        st.metric("Sat Temp Lift", f"{design['sat_temp_lift']:.1f} °C",
                  help="Condensing sat temp at discharge minus vessel sat temp.")

    with r3:
        st.markdown("**Blower discharge vapor**")
        st.metric("Discharge Pressure",
                  f"{blower_discharge_psig:.2f} psig ({blower_discharge_psia:.2f} psia)")
        st.metric("Condensing (Sat) Temp", f"{design['t_sat_disch']:.1f} °C",
                  help="Temperature at which compressed vapor condenses in the primary HX.")
        st.metric("Actual Discharge Temp", f"{design['t_disch_actual']:.0f} °C",
                  help="Superheated temperature leaving blower before desuperheating.")

    with r4:
        st.markdown("**Primary HX / power**")
        st.metric("HX Driving ΔT", f"{design['hx_driving_dt']:.1f} °C",
                  help="Condensing temp minus brine boiling temp.")
        st.metric("Superheat to Remove", f"{design['superheat']:.0f} °C",
                  help="Removed by distillate desuperheating spray.")
        st.metric("Blower Power", f"{design['elec_kw']:,.0f} kW",
                  help=f"At {design_vapor_lb_hr:,.0f} lb/hr vapor (calculated distillate).")

    st.caption(
        f"Vessel {vessel_pressure_sp:.2f} psia ({design['vessel_bar']:.3f} bar abs) → "
        f"boils at {design['t_boil']:.1f} °C | Blower ΔP {blower_dp_setpoint:.2f} psi → "
        f"discharge {blower_discharge_psia:.2f} psia, condenses at "
        f"{design['t_sat_disch']:.1f} °C ({c_to_f(design['t_sat_disch']):.0f} °F). "
        "Saturation temps from IAPWS-IF97."
    )

    st.divider()

    ##############################################################################
    # VESSEL PRESSURE CONTROL - BLOWER VFD RESPONSE
    ##############################################################################
    st.subheader("🎛️ Vessel Pressure Control – Blower VFD Response")

    st.markdown(
        "- **Vessel pressure rises above SP** → vapor is forming faster than the blower removes it "
        "→ VFD **speeds up** the blower.\n"
        "- **Vessel pressure falls below SP** → vapor is forming slower → VFD **slows down** the blower.\n"
        "- Boiling temperature follows vessel pressure; blower ΔP and discharge "
        "condensing temperature follow blower speed (fan laws: flow ∝ N, ΔP ∝ N², power ∝ N³)."
    )

    v1, v2, v3, v4, v5 = st.columns(5)

    with v1:
        vessel_pressure_pv = st.number_input(
            "Actual Vessel Pressure PV (psia)", min_value=1.0, max_value=40.0,
            value=float(st.session_state.project_data.get("vessel_pressure_pv", vessel_pressure_sp)),
            step=0.05,
            help="Enter a measured / upset pressure to see the controller response."
        )

    with v2:
        vfd_design_speed = st.number_input(
            "Design Speed at SP (%)", min_value=10.0, max_value=100.0,
            value=float(st.session_state.project_data.get("vfd_design_speed", 85.0)))

    with v3:
        vfd_kp = st.number_input(
            "Controller Gain Kp (% speed / psi)", min_value=0.0, max_value=200.0,
            value=float(st.session_state.project_data.get("vfd_kp", 20.0)))

    with v4:
        vfd_min_speed = st.number_input(
            "VFD Min Speed (%)", min_value=0.0, max_value=100.0,
            value=float(st.session_state.project_data.get("vfd_min_speed", 40.0)),
            help="Keep above blower surge limit."
        )

    with v5:
        vfd_max_speed = st.number_input(
            "VFD Max Speed (%)", min_value=0.0, max_value=110.0,
            value=float(st.session_state.project_data.get("vfd_max_speed", 100.0)))


    def operating_point(pv_psia):
        ctrl = vfd_response(
            vessel_pv_psia=pv_psia,
            vessel_sp_psia=vessel_pressure_sp,
            design_speed_pct=vfd_design_speed,
            kp_pct_per_psi=vfd_kp,
            min_speed_pct=vfd_min_speed,
            max_speed_pct=vfd_max_speed,
            design_dp_psi=blower_dp_setpoint,
            design_vapor_lb_hr=design_vapor_lb_hr,
        )
        state = mvr_state(
            vessel_psia=pv_psia,
            discharge_psia=ctrl["discharge_psia"],
            bpe_c=boiling_point_elevation,
            blower_eff=blower_efficiency / 100.0,
            motor_eff=motor_efficiency / 100.0,
            vapor_lb_hr=ctrl["vapor_lb_hr"],
        )
        return ctrl, state


    ctrl, op = operating_point(vessel_pressure_pv)

    o1, o2, o3, o4, o5 = st.columns(5)

    with o1:
        st.metric("Pressure Error", f"{ctrl['error_psi']:+.2f} psi", help="PV - SP")
        st.metric("Controller Action", ctrl["action"])

    with o2:
        st.metric("Blower Speed", f"{ctrl['speed_pct']:.1f} %",
                  f"{ctrl['speed_pct'] - vfd_design_speed:+.1f} %")
        if ctrl["at_limit"]:
            st.warning("VFD at speed limit")

    with o3:
        st.metric("Brine Boiling Temp", f"{op['t_boil']:.1f} °C",
                  f"{op['t_boil'] - design['t_boil']:+.1f} °C")
        st.metric("Vapor Flow", f"{ctrl['vapor_lb_hr']:,.0f} lb/hr")

    with o4:
        st.metric("Blower ΔP", f"{op['blower_dp']:.2f} psi",
                  f"{op['blower_dp'] - blower_dp_setpoint:+.2f} psi")
        st.metric("Discharge Pressure",
                  f"{op['discharge_psia'] - atmospheric_pressure:.2f} psig")

    with o5:
        st.metric("Condensing Temp", f"{op['t_sat_disch']:.1f} °C",
                  f"{op['t_sat_disch'] - design['t_sat_disch']:+.1f} °C")
        st.metric("Blower Power", f"{op['elec_kw']:,.0f} kW",
                  f"{op['elec_kw'] - design['elec_kw']:+,.0f} kW")

    # Sensitivity table: SP ± 1 psi
    rows = []
    for i in range(-4, 5):
        pv = round(vessel_pressure_sp + 0.25 * i, 2)
        if pv <= 1.0:
            continue
        c_, s_ = operating_point(pv)
        rows.append({
            "Vessel P (psia)": pv,
            "Error (psi)": round(c_["error_psi"], 2),
            "Action": c_["action"],
            "Speed (%)": round(c_["speed_pct"], 1),
            "Boil Temp (°C)": round(s_["t_boil"], 1),
            "Blower ΔP (psi)": round(s_["blower_dp"], 2),
            "Discharge (psig)": round(s_["discharge_psia"] - atmospheric_pressure, 2),
            "Condensing T (°C)": round(s_["t_sat_disch"], 1),
            "Discharge T actual (°C)": round(s_["t_disch_actual"], 0),
            "HX ΔT (°C)": round(s_["hx_driving_dt"], 1),
            "Power (kW)": round(s_["elec_kw"], 0),
        })

    with st.expander("Response table – vessel pressure SP ± 1 psi", expanded=False):
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Proportional-only estimate of the steady VFD response. In the plant, a PI "
            "pressure controller trims speed until PV returns to SP."
        )

    st.divider()

    ##############################################################################
    # ELECTRIC BOILER CAPACITY
    ##############################################################################
    st.subheader("🔥 Electric Boiler Capacity")

    st.markdown(
        "Boiler capacity = **max** of (1) steady-state makeup steam, "
        "(2) start-up heat-up steam, (3) blower priming steam."
    )

    pd_ = st.session_state.project_data


    def _num(label, key, default, **kw):
        return st.number_input(
            label,
            value=float(pd_.get(key, default)),
            key=f"blr_{key}",
            **kw
        )


    # Pre-HX: same calculation as the Feed HX tab, using that tab's current inputs
    from mvr_thermo import preheater, FHX_MODES


    def _fhx_in(widget_key, pdata_key, default):
        return st.session_state.get(widget_key, pd_.get(pdata_key, default))


    feed_temp_c_in = float(_fhx_in("fhx_feed_temp_c", "fhx_feed_temp_c",
                                   pd_.get("feed_temp_c", 25.0)))
    distillate_subcool_c = float(_fhx_in("fhx_subcool_c", "fhx_subcool_c", 10.0))
    _fhx_mode = _fhx_in("fhx_mode", "fhx_mode", FHX_MODES[0])
    _fhx = preheater(
        feed_lb_hr=feed_lb_hr,
        t_feed_in_c=feed_temp_c_in,
        dist_lb_hr=distillate_lb_hr,
        t_dist_in_c=design["t_sat_disch"] - distillate_subcool_c,
        effectiveness=float(_fhx_in("fhx_eff", "preheater_efficiency", 80.0)) / 100.0,
        feed_cp=float(st.session_state.get("phx_feed_cp", pd_.get("feed_cp", 0.95))),
        mode=_fhx_mode if _fhx_mode in FHX_MODES else FHX_MODES[0],
        t_feed_out_fixed_c=float(_fhx_in("fhx_out_fixed_c", "fhx_out_fixed_c", 80.0)),
    )
    preheat_out_c = _fhx["t_feed_out_c"]

    st.markdown("**Heat exchangers & losses**")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        st.metric("Feed Temp to Pre-HX", f"{feed_temp_c_in:.1f} °C", help="Set on the Feed HX tab.")
    with b2:
        st.metric("Feed Temp out of Pre-HX", f"{preheat_out_c:.1f} °C", help="Calculated on the Feed HX tab.")
    with b3:
        st.metric("Distillate to Pre-HX",
                  f"{design['t_sat_disch'] - distillate_subcool_c:.1f} °C",
                  help=f"Condensing temp − {distillate_subcool_c:.1f} °C subcool (Feed HX tab).")
    with b4:
        vent_pct = _num("NCG Vent Loss (% of vapor)", "vent_pct", 0.75, min_value=0.0)
    with b5:
        radiation_pct = _num("Radiation Loss (% of evap duty)", "radiation_pct", 2.5, min_value=0.0)

    st.markdown("**Start-up & boiler**")
    b6, b7, b8, b9, b10 = st.columns(5)
    with b6:
        brine_inventory_lb = _num("Brine Inventory (lb)", "brine_inventory_lb", 135000.0,
                                  min_value=0.0, step=1000.0,
                                  help="Vessel + recirculation loop + HX holdup.")
    with b7:
        steel_lb = _num("Steel Weight Heated (lb)", "steel_lb", 70000.0, min_value=0.0,
                        step=1000.0, help="Replace with vendor equipment weight.")
    with b8:
        heatup_hr = _num("Allowable Heat-up Time (hr)", "heatup_hr", 4.0, min_value=0.5, step=0.5)
    with b9:
        makeup_factor = _num("Makeup Design Factor (x)", "makeup_factor", 2.5, min_value=1.0, step=0.1)
    with b10:
        boiler_eff_pct = _num("Boiler Efficiency (%)", "boiler_eff_pct", 98.0,
                              min_value=50.0, max_value=100.0)

    b11, b12, b13, b14 = st.columns(4)
    with b11:
        boiler_feedwater_c = _num("Boiler Feedwater Temp (°C)", "boiler_feedwater_c", 25.0)
    with b12:
        heatup_loss_pct = _num("Heat-up Losses (%)", "heatup_loss_pct", 13.0, min_value=0.0)
    with b13:
        surge_lb_hr = _num("Blower Surge Flow (lb/hr)", "surge_lb_hr", 0.0, min_value=0.0,
                           step=100.0, help="From blower vendor. 0 if unknown.")
    with b14:
        boiler_primes_blower = st.checkbox(
            "Boiler must supply surge flow at start-up",
            value=bool(pd_.get("boiler_primes_blower", False)),
            key="blr_boiler_primes_blower"
        )


    def _boiler(hr):
        return calc_boiler_capacity(
            feed_lb_hr=feed_lb_hr,
            salt_lb_hr=salt_lb_hr,
            cake_water_lb_hr=cake_water_lb_hr,
            distillate_lb_hr=distillate_lb_hr,
            t_feed_c=feed_temp_c_in,
            t_preheat_out_c=preheat_out_c,
            t_distillate_in_c=design["t_sat_disch"] - distillate_subcool_c,
            t_boil_c=design["t_boil"],
            t_sat_vessel_c=design["t_sat_vessel"],
            blower_shaft_kw=design["shaft_kw"],
            boiler_psia=boiler_pressure + atmospheric_pressure,
            boiler_feedwater_c=boiler_feedwater_c,
            boiler_eff=boiler_eff_pct / 100.0,
            vent_frac=vent_pct / 100.0,
            radiation_frac=radiation_pct / 100.0,
            makeup_factor=makeup_factor,
            brine_inventory_lb=brine_inventory_lb,
            steel_lb=steel_lb,
            heatup_loss_frac=heatup_loss_pct / 100.0,
            heatup_hr=hr,
            priming_lb_hr=surge_lb_hr if boiler_primes_blower else 0.0,
        )


    blr = _boiler(heatup_hr)

    if design["t_sat_disch"] - distillate_subcool_c <= preheat_out_c:
        st.warning("Distillate enters the pre-HX colder than the feed outlet target - "
                   "pre-HX cannot reach the set feed outlet temperature.")
    if blr["t_dist_out"] < feed_temp_c_in:
        st.warning("Pre-HX duty exceeds what the distillate can supply - "
                   "lower the feed outlet temperature.")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Required Boiler Capacity", f"{blr['required_lb_hr']:,.0f} lb/hr",
                  f"{blr['required_lb_hr'] - boiler_capacity:+,.0f} vs Vacom target",
                  delta_color="off")
    with k2:
        st.metric("Electric Boiler Power", f"{blr['elec_kw']:,.0f} kW",
                  help=f"{blr['boiler_rise']:,.0f} BTU/lb from {boiler_feedwater_c:.0f} °C feedwater")
    with k3:
        st.metric("Governing Case", blr["governing"])
    with k4:
        st.metric("Boiler Steam",
                  f"{boiler_pressure:.0f} psig / {blr['t_sat_boiler']:.1f} °C",
                  help=f"Latent heat {blr['hfg_boiler']:,.0f} BTU/lb")

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("**1. Steady-state makeup**")
        st.dataframe(pd.DataFrame([
            ["Distillate leaving pre-HX warm", blr["loss_distillate"]],
            ["Salt + cake moisture", blr["loss_salt"]],
            ["NCG vent", blr["loss_vent"]],
            ["Radiation / convection", blr["loss_radiation"]],
            ["Total losses", blr["losses"]],
            ["Less blower heat input", -blr["blower_heat"]],
            ["Net makeup heat", blr["net_makeup"]],
        ], columns=["Item", "MMBTU/hr"]).round(2), hide_index=True, width="stretch")
        st.metric("Design Makeup Steam", f"{blr['makeup_design']:,.0f} lb/hr",
                  help=f"{blr['makeup_lb_hr']:,.0f} lb/hr x {makeup_factor:.1f}")
        if blr["surplus"] > 0:
            st.info(f"Blower adds {blr['surplus']:.2f} MMBTU/hr more than losses - "
                    "no steady-state makeup; remove surplus by desuperheating / venting.")
    with s2:
        st.markdown("**2. Start-up heat-up**")
        st.metric("Start-up Heat", f"{blr['q_startup']:.2f} MMBTU")
        st.metric("Start-up Steam", f"{blr['startup_lb_hr']:,.0f} lb/hr",
                  help=f"Heat-up {feed_temp_c_in:.0f} → {design['t_boil']:.1f} °C "
                       f"in {heatup_hr:.1f} hr")
    with s3:
        st.markdown("**3. Blower priming**")
        st.metric("Priming Steam", f"{blr['priming_lb_hr']:,.0f} lb/hr")
        if distillate_lb_hr > 0:
            st.caption(f"Required capacity = "
                       f"{blr['required_lb_hr'] / distillate_lb_hr * 100:.1f}% of vapor flow")

    with st.expander("Heat-up time vs boiler size", expanded=False):
        hrows = []
        for hr in [2, 3, 4, 5, 6, 8, 10, 12, 16, 24]:
            b = _boiler(hr)
            hrows.append({
                "Heat-up (hr)": hr,
                "Start-up steam (lb/hr)": round(b["startup_lb_hr"]),
                "Boiler capacity (lb/hr)": round(b["required_lb_hr"]),
                "Electric power (kW)": round(b["elec_kw"]),
                "Governing": b["governing"],
            })
        st.dataframe(pd.DataFrame(hrows), hide_index=True, width="stretch")

    ##############################################################################
    # SAVE VARIABLES
    ##############################################################################
    pdata = st.session_state.project_data

    pdata.update({
        "design_flow": design_flow,
        "capacity_basis": capacity_basis,
        "feed_tds": feed_tds,
        "feed_sg": feed_sg,
        "feed_lb_hr": feed_lb_hr,
        "cake_moisture_pct": cake_moisture_pct,
        "salt_lb_hr_calc": salt_lb_hr,
        "cake_water_lb_hr": cake_water_lb_hr,
        "distillate_lb_hr_calc": distillate_lb_hr,
        "distillate_gpm_calc": distillate_gpm,
        "mvr_vapor_lb_hr": design_vapor_lb_hr,
        "design_distillate_flow": design_distillate_flow,
        "salt_production_rate": salt_production_rate,
        "boiler_capacity": boiler_capacity,
        "boiler_pressure": boiler_pressure,
        "atmospheric_pressure": atmospheric_pressure,

        # inputs that were not saved before - now restored when a project is opened
        "plant_availability": plant_availability,
        "design_factor": design_factor,
        "crystallizer_system": crystallizer_system,
        "crystallizer_type": crystallizer_type,
        "steam_source": steam_source,
        "project_objective": project_objective,
        "standby_configuration": standby_configuration,
        "crystal_recovery": crystal_recovery,
        "overall_recovery": overall_recovery,
        "operating_pressure": operating_pressure,
        "design_temperature": design_temperature,
        "number_of_trains": number_of_trains,

        "vessel_level_sp": vessel_level_sp,
        "vessel_pressure_sp": vessel_pressure_sp,
        "blower_discharge_psig": blower_discharge_psig,
        "blower_discharge_psia": blower_discharge_psia,
        "blower_dp_setpoint": blower_dp_setpoint,
        "blower_dp": blower_dp_setpoint,  # kept for existing modules
        "boiling_point_elevation": boiling_point_elevation,

        "crystallizer_pressure_bar": design["vessel_bar"],
        "saturation_temp": design["t_sat_vessel"],
        "crystallizer_boiling_temp": design["t_boil"],
        "mvr_pressure_ratio": design["pressure_ratio"],
        "mvr_discharge_sat_temp": design["t_sat_disch"],
        "mvr_discharge_temp_actual": design["t_disch_actual"],
        "mvr_sat_temp_lift": design["sat_temp_lift"],
        "mvr_superheat": design["superheat"],
        "primary_hx_driving_dt": design["hx_driving_dt"],
        "blower_power_kw": design["elec_kw"],

        "vessel_pressure_pv": vessel_pressure_pv,
        "vfd_design_speed": vfd_design_speed,
        "vfd_kp": vfd_kp,
        "vfd_min_speed": vfd_min_speed,
        "vfd_max_speed": vfd_max_speed,
        "vfd_speed_pct": ctrl["speed_pct"],

        "boiler_feed_temp_c": feed_temp_c_in,
        "preheat_out_c": preheat_out_c,
        "distillate_subcool_c": distillate_subcool_c,
        "vent_pct": vent_pct,
        "radiation_pct": radiation_pct,
        "brine_inventory_lb": brine_inventory_lb,
        "steel_lb": steel_lb,
        "heatup_hr": heatup_hr,
        "makeup_factor": makeup_factor,
        "boiler_eff_pct": boiler_eff_pct,
        "boiler_feedwater_c": boiler_feedwater_c,
        "heatup_loss_pct": heatup_loss_pct,
        "surge_lb_hr": surge_lb_hr,
        "boiler_primes_blower": boiler_primes_blower,
        "boiler_capacity_calc": blr["required_lb_hr"],
        "boiler_power_kw": blr["elec_kw"],
        "boiler_governing_case": blr["governing"],
        "boiler_makeup_design_lb_hr": blr["makeup_design"],
        "boiler_startup_lb_hr": blr["startup_lb_hr"],
        "heat_losses_mmbtu_hr": blr["losses"],
        "blower_heat_mmbtu_hr": blr["blower_heat"],

        "tss_open": tss_open,
        "tss_high": tss_high,
        "mvr_enabled": mvr_enabled,
        "steam_economy_target": steam_economy_target,
        "blower_efficiency": blower_efficiency,
        "motor_efficiency": motor_efficiency,
    })

##############################################################################################
# FEED PUMP & FEED PREHEATER  (all temperatures in °C, °F shown for reference)
#
# Feed     : design feed flow + feed lb/hr from Basis mass balance
# Distillate in : condensing temp of compressed vapor (Basis) - subcool in primary HX
# Pre-HX outlet : calculated from effectiveness, or fixed - this tab is the ONE place
#                 it is set. Basis (boiler), Recirc Pump and Primary HX tabs use it.
##############################################################################################

with tab3:
    from mvr_thermo import preheater, FHX_MODES

    st.header("🔥 Feed Pump & Feed Preheater")
    flow_slot(('E-120', 'E-130'))

    pdata = st.session_state.project_data
    c2f = lambda t: t * 9 / 5 + 32

    ##########################################################################
    # VALUES FROM WATER & BASIS TABS
    ##########################################################################
    feed_flow = float(pdata.get("design_flow", pdata.get("feed_flow", 36.0)))
    feed_lb_hr = float(pdata.get("feed_lb_hr", feed_flow * 8.34 * 60))
    feed_sg = float(pdata.get("feed_sg", 1.05))
    dist_lb_hr = float(pdata.get("distillate_lb_hr_calc", feed_lb_hr * 0.92))
    cond_c = float(pdata.get("mvr_discharge_sat_temp", 105.3))
    boil_c = float(pdata.get("crystallizer_boiling_temp", 87.3))
    retention_hours = float(pdata.get("retention_hours", 0.0))
    feed_tanks = pdata.get("number_of_feed_tanks", 3)
    feed_storage = float(pdata.get("total_feed_storage", 46500))
    feed_cp = float(st.session_state.get("phx_feed_cp", pdata.get("feed_cp", 0.95)))

    st.info(
        f"{feed_tanks} Feed Tanks | {feed_storage:,.0f} gal Total Storage | "
        f"Feed {feed_flow:.1f} GPM ({feed_lb_hr:,.0f} lb/hr) | "
        f"Distillate {dist_lb_hr:,.0f} lb/hr | "
        f"Vapor condenses at {cond_c:.1f} °C | Brine boils at {boil_c:.1f} °C"
    )

    ##########################################################################
    # PREHEATER INPUTS
    ##########################################################################
    st.subheader("Feed Preheater (distillate → feed)")

    p1, p2, p3 = st.columns(3)

    with p1:
        feed_temp_c = st.number_input(
            "Feed Temperature In (°C)",
            min_value=0.0, max_value=95.0,
            value=float(pdata.get("fhx_feed_temp_c", pdata.get("feed_temp_c", 25.0))),
            step=0.5,
            key="fhx_feed_temp_c",
            help="Defaults to the Water tab temperature."
        )

    with p2:
        distillate_subcool_c = st.number_input(
            "Distillate Subcool in Primary HX (°C)",
            min_value=0.0, max_value=40.0,
            value=float(pdata.get("fhx_subcool_c", pdata.get("distillate_subcool_c", 10.0))),
            step=0.5,
            key="fhx_subcool_c",
            help="Distillate leaves the primary HX shell this much below the condensing temp."
        )

    with p3:
        fhx_mode = st.radio(
            "Pre-HX Outlet",
            FHX_MODES,
            index=FHX_MODES.index(pdata.get("fhx_mode", FHX_MODES[0]))
            if pdata.get("fhx_mode") in FHX_MODES else 0,
            key="fhx_mode",
            horizontal=False
        )

    p4, p5, p6 = st.columns(3)

    with p4:
        preheater_efficiency = st.number_input(
            "Preheater Effectiveness (%)",
            min_value=10.0, max_value=98.0,
            value=float(pdata.get("preheater_efficiency", 80.0)),
            key="fhx_eff",
            disabled=fhx_mode != FHX_MODES[0],
            help="Heat recovered ÷ maximum possible. Plate HX typically 75-90%."
        )

    with p5:
        fixed_out_c = st.number_input(
            "Fixed Pre-HX Outlet (°C)",
            min_value=0.0, max_value=110.0,
            value=float(pdata.get("fhx_out_fixed_c", pdata.get("preheat_out_c", 80.0))),
            step=0.5,
            key="fhx_out_fixed_c",
            disabled=fhx_mode != FHX_MODES[1]
        )

    with p6:
        fhx_u = st.number_input(
            "Pre-HX U Value (BTU/hr-ft²-°F)",
            min_value=10.0,
            value=float(pdata.get("fhx_u", 450.0)),
            key="fhx_u",
            help="Plate & frame, water/water: typically 400-700."
        )

    ##########################################################################
    # PREHEATER CALCULATION
    ##########################################################################
    distillate_temp_c = cond_c - distillate_subcool_c

    fhx = preheater(
        feed_lb_hr=feed_lb_hr,
        t_feed_in_c=feed_temp_c,
        dist_lb_hr=dist_lb_hr,
        t_dist_in_c=distillate_temp_c,
        effectiveness=preheater_efficiency / 100.0,
        feed_cp=feed_cp,
        mode=fhx_mode,
        t_feed_out_fixed_c=fixed_out_c,
    )

    hx_outlet_temp_c = fhx["t_feed_out_c"]
    distillate_out_c = fhx["t_dist_out_c"]
    heat_duty = fhx["duty_mmbtu"]
    approach_temp = boil_c - hx_outlet_temp_c  # °C still to heat in primary HX
    lmtd_f = fhx["lmtd_c"] * 1.8
    fhx_area = heat_duty * 1e6 / (fhx_u * lmtd_f) if lmtd_f > 0 else 0.0

    if not fhx["feasible"]:
        st.error(
            f"Pre-HX not feasible: distillate in {distillate_temp_c:.1f} °C, feed out "
            f"{hx_outlet_temp_c:.1f} °C, distillate out {distillate_out_c:.1f} °C. "
            "Feed outlet must stay below distillate inlet, and distillate outlet above feed inlet. "
            "Lower the fixed outlet / effectiveness or the distillate subcool."
        )
    elif min(fhx["approach_hot_c"], fhx["approach_cold_c"]) < 3.0:
        st.warning("Pre-HX temperature approach below 3 °C - very large exchanger needed.")

    if approach_temp < 3:
        st.warning("⚠️ Preheater outlet is close to the crystallizer boiling temperature - "
                   "risk of flashing / scaling in the pre-HX.")
    elif approach_temp > 25:
        st.info("ℹ️ More heat could be recovered from the distillate (raise effectiveness).")

    ##########################################################################
    # FEED PUMP
    ##########################################################################
    st.subheader("Feed Pump")

    f1, f2, f3 = st.columns(3)

    with f1:
        feed_tdh = st.number_input(
            "Pump TDH (ft)", value=float(pdata.get("feed_tdh", 120.0)), key="feed_tdh")

    with f2:
        pump_efficiency = st.number_input(
            "Pump Efficiency (%)", min_value=10.0, max_value=95.0,
            value=float(pdata.get("feed_pump_eff", 75.0)), key="feed_pump_eff")

    with f3:
        feed_pumps = st.number_input(
            "Feed Pumps Installed", min_value=1,
            value=int(pdata.get("feed_pumps", 2)), key="fhx_feed_pumps")

    pump_hp = feed_flow * feed_tdh * feed_sg / (3960 * (pump_efficiency / 100))

    ##########################################################################
    # RESULTS
    ##########################################################################
    st.subheader("Preheater Performance")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric("Feed In → Out", f"{feed_temp_c:.1f} → {hx_outlet_temp_c:.1f} °C",
                  f"{c2f(feed_temp_c):.0f} → {c2f(hx_outlet_temp_c):.0f} °F", delta_color="off")
    with k2:
        st.metric("Distillate In → Out", f"{distillate_temp_c:.1f} → {distillate_out_c:.1f} °C",
                  f"{c2f(distillate_temp_c):.0f} → {c2f(distillate_out_c):.0f} °F", delta_color="off")
    with k3:
        st.metric("Preheater Duty", f"{heat_duty:.2f} MMBTU/hr",
                  f"{heat_duty * 293.07:,.0f} kW", delta_color="off")
    with k4:
        st.metric("Effectiveness", f"{fhx['effectiveness'] * 100:.0f} %")
    with k5:
        st.metric("LMTD / Area", f"{fhx['lmtd_c']:.1f} °C",
                  f"{fhx_area:,.0f} ft²", delta_color="off")
    with k6:
        st.metric("Remaining to Boiling", f"{approach_temp:.1f} °C",
                  help="Heated in the primary HX loop after the pre-HX.")

    k7, k8, k9, k10 = st.columns(4)

    with k7:
        st.metric("Feed Pump", f"{pump_hp:.1f} HP")
    with k8:
        st.metric("Hot-end Approach", f"{fhx['approach_hot_c']:.1f} °C")
    with k9:
        st.metric("Cold-end Approach", f"{fhx['approach_cold_c']:.1f} °C")
    with k10:
        st.metric("Retention Time", f"{retention_hours:.1f} hr")

    st.caption(
        f"Distillate inlet = condensing temp {cond_c:.1f} °C − subcool "
        f"{distillate_subcool_c:.1f} °C. Brine boils at {boil_c:.1f} °C "
        f"({c2f(boil_c):.1f} °F). The Basis (boiler), Recirc Pump and Primary HX tabs use "
        "the pre-HX outlet calculated here."
    )

    ##########################################################################
    # SAVE DATA (°C keys used by other tabs; °F keys kept for older code)
    ##########################################################################
    pdata.update({
        "fhx_feed_temp_c": feed_temp_c,
        "fhx_subcool_c": distillate_subcool_c,
        "fhx_mode": fhx_mode,
        "fhx_out_fixed_c": fixed_out_c,
        "fhx_u": fhx_u,
        "preheater_efficiency": preheater_efficiency,
        "preheat_out_c": hx_outlet_temp_c,
        "hx_outlet_temp_c": hx_outlet_temp_c,
        "hx_outlet_temp": c2f(hx_outlet_temp_c),
        "pre_hx_outlet_temp": c2f(hx_outlet_temp_c),
        "distillate_temp_c": distillate_temp_c,
        "distillate_temp": c2f(distillate_temp_c),
        "distillate_out_c": distillate_out_c,
        "feed_hx_duty": heat_duty,
        "feed_hx_area": fhx_area,
        "feed_hx_lmtd_c": fhx["lmtd_c"],
        "feed_tdh": feed_tdh,
        "feed_pump_eff": pump_efficiency,
        "feed_pumps": feed_pumps,
        "feed_pump_hp": pump_hp,
    })

    save_project(project_name, st.session_state.project_data)

    st.success("✅ Feed Pump & Preheater Data Saved")

##############################################################################
# RECIRCULATION PUMP
# Flow : from the primary HX heat balance  ->  duty / (Cp x flash ΔT)
# Head : sum of loop friction losses (closed loop, no static lift)
# Also : boiling suppression (HX submergence) and NPSH available
#
# Flash ΔT, brine SG / Cp and feed Cp are entered on the Primary HX tab.
# Their current values are read from st.session_state (widget keys), so this
# tab updates in the same run even though it comes before the HX tab.
##############################################################################

with tab4:
    from mvr_thermo import recirc_requirement, psat_psia_from_c
    import math

    st.header("🌡️ Recirculation Pump")
    flow_slot(('P-201',))

    pdata = st.session_state.project_data


    def _shared(widget_key, pdata_key, default):
        """Current value of a Primary HX tab input (widget state first, then saved)."""
        return float(st.session_state.get(widget_key, pdata.get(pdata_key, default)))


    ##########################################################################
    # VALUES FROM DESIGN BASIS / PRIMARY HX
    ##########################################################################
    feed_flow = float(pdata.get("design_flow", 36.0))
    crystallizer_pressure_psia = float(pdata.get("vessel_pressure_sp", 7.25))
    crystallizer_pressure_bar = float(pdata.get("crystallizer_pressure_bar", 0.50))
    boiling_point_elevation = float(pdata.get("boiling_point_elevation", 6.0))
    crystallizer_boiling_temp = float(pdata.get("crystallizer_boiling_temp", 87.3))
    vessel_sat_c = float(pdata.get("saturation_temp", 81.3))
    vapor_lb_hr = float(pdata.get("distillate_lb_hr_calc", 17500.0))
    feed_lb_hr = float(pdata.get("feed_lb_hr", 18900.0))
    feed_in_c = float(pdata.get("preheat_out_c", 80.0))

    flash_dt_f = _shared("phx_flash_dt", "flash_delta_t_input", 10.8)
    brine_sg = _shared("phx_brine_sg", "brine_sg", 1.20)
    brine_cp = _shared("phx_brine_cp", "brine_cp", 0.80)
    feed_cp = _shared("phx_feed_cp", "feed_cp", 0.95)

    req = recirc_requirement(
        vapor_lb_hr=vapor_lb_hr,
        vessel_sat_c=vessel_sat_c,
        boil_c=crystallizer_boiling_temp,
        feed_lb_hr=feed_lb_hr,
        feed_in_c=feed_in_c,
        flash_dt_f=flash_dt_f,
        brine_cp=brine_cp,
        brine_sg=brine_sg,
        feed_cp=feed_cp,
    )

    st.info(
        f"HX duty {req['hx_duty']:.2f} MMBTU/hr ÷ (Cp {brine_cp:.2f} × flash ΔT "
        f"{flash_dt_f:.1f} °F) → required recirculation "
        f"**{req['flow_gpm']:,.0f} GPM** (SG {brine_sg:.2f}). "
        "Flash ΔT, SG and Cp are set on the Primary Heat Exchanger tab."
    )

    ##########################################################################
    # FLOW BASIS
    ##########################################################################
    st.subheader("Flow Basis")

    f1, f2, f3 = st.columns(3)

    with f1:
        flow_basis = st.radio(
            "Recirculation Flow Basis",
            ["Calculated from HX duty", "Manual recirculation ratio"],
            index=0 if pdata.get("recirc_flow_basis", "Calculated from HX duty")
                       == "Calculated from HX duty" else 1,
            key="rp_flow_basis"
        )

    with f2:
        flow_margin = st.number_input(
            "Flow Design Margin (%)",
            min_value=0.0, max_value=50.0,
            value=float(pdata.get("recirc_flow_margin", 10.0)),
            key="rp_flow_margin",
            disabled=flow_basis != "Calculated from HX duty"
        )

    with f3:
        manual_ratio = st.number_input(
            "Recirculation Ratio (recirc / feed)",
            min_value=1.0,
            value=max(float(pdata.get("recirc_manual_ratio", 170.0)), 1.0),
            key="rp_manual_ratio",
            disabled=flow_basis == "Calculated from HX duty",
            help="Forced-circulation crystallizers are typically 100-300x feed."
        )

    if flow_basis == "Calculated from HX duty":
        recirc_flow = req["flow_gpm"] * (1 + flow_margin / 100.0)
    else:
        recirc_flow = feed_flow * manual_ratio
        if recirc_flow < req["flow_gpm"]:
            st.warning(
                f"Manual flow {recirc_flow:,.0f} GPM is below the {req['flow_gpm']:,.0f} GPM "
                "needed to carry the HX duty at this flash ΔT - actual flash ΔT will be "
                f"{req['flow_gpm'] / recirc_flow * flash_dt_f:.1f} °F."
            )

    ##########################################################################
    # HEAD (TDH) BUILD-UP
    ##########################################################################
    st.subheader("Pump Head (TDH) Build-up")
    st.caption("Closed loop from vessel back to vessel - no static lift, friction only.")

    h1, h2, h3, h4, h5 = st.columns(5)

    with h1:
        loss_tubes = st.number_input(
            "HX Tubes (ft)", min_value=0.0,
            value=float(pdata.get("rp_loss_tubes", 6.0)), key="rp_loss_tubes",
            help="Tube friction + entry/exit at 6-8 ft/s, 20 ft tubes: ~4-8 ft.")
    with h2:
        loss_channels = st.number_input(
            "HX Channels / Heads (ft)", min_value=0.0,
            value=float(pdata.get("rp_loss_channels", 1.5)), key="rp_loss_channels")
    with h3:
        loss_piping = st.number_input(
            "Piping & Fittings (ft)", min_value=0.0,
            value=float(pdata.get("rp_loss_piping", 4.0)), key="rp_loss_piping")
    with h4:
        loss_nozzle = st.number_input(
            "Vessel Inlet Nozzle (ft)", min_value=0.0,
            value=float(pdata.get("rp_loss_nozzle", 3.0)), key="rp_loss_nozzle")
    with h5:
        head_margin = st.number_input(
            "Head Margin (%)", min_value=0.0, max_value=50.0,
            value=float(pdata.get("rp_head_margin", 10.0)), key="rp_head_margin")

    tdh_friction = loss_tubes + loss_channels + loss_piping + loss_nozzle
    tdh = tdh_friction * (1 + head_margin / 100.0)

    ##########################################################################
    # VACOM VENDOR DATA - back-calculate flash ΔT and head from their selection
    ##########################################################################
    st.subheader("🏭 Vacom Vendor Data")
    st.caption(
        "Enter Vacom's pump selection. Leave Rated Flow or Rated TDH at 0 if unknown: "
        "flow is then estimated from the discharge pipe size, and TDH from the motor hp."
    )

    STD_MOTORS_HP = [5, 7.5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150,
                     200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]


    def _std_motor(hp_needed):
        return next((m for m in STD_MOTORS_HP if m >= hp_needed),
                    math.ceil(hp_needed / 50) * 50)


    vacom_distillate_gpm = float(pdata.get("design_distillate_flow", 35.0))
    duty_options = [
        f"App case ({feed_flow:.0f} GPM feed)",
        f"Vacom design target ({vacom_distillate_gpm:.0f} GPM distillate)",
    ]

    vd1, vd2, vd3, vd4 = st.columns(4)
    with vd1:
        vendor_motor_hp = st.number_input(
            "Vacom Motor (hp)", min_value=0.0,
            value=float(pdata.get("vendor_motor_hp", 600.0)), step=25.0, key="vd_motor_hp")
    with vd2:
        vendor_pipe_nominal = st.number_input(
            "Vacom Discharge Pipe (in, nominal)", min_value=1.0,
            value=float(pdata.get("vendor_pipe_nominal", 20.0)), key="vd_pipe_nominal")
    with vd3:
        vendor_pipe_id = st.number_input(
            "Pipe ID (in)", min_value=1.0,
            value=float(pdata.get("vendor_pipe_id", 19.25)), step=0.05, key="vd_pipe_id",
            help="20 in standard wall (0.375 in) = 19.25 in ID.")
    with vd4:
        vendor_duty_basis = st.selectbox(
            "Vacom Selection Is For", duty_options,
            index=int(pdata.get("vendor_duty_basis_idx", 0)), key="vd_duty_basis",
            help="Which duty Vacom sized the pump for - used to back-calculate their flash ΔT.")

    vd5, vd6, vd7, vd8, vd9 = st.columns(5)
    with vd5:
        vendor_flow_in = st.number_input(
            "Vacom Rated Flow (GPM)", min_value=0.0,
            value=float(pdata.get("vendor_flow_gpm", 0.0)), step=100.0, key="vd_flow",
            help="0 = unknown (estimated from pipe size).")
    with vd6:
        vendor_tdh_in = st.number_input(
            "Vacom Rated TDH (ft)", min_value=0.0,
            value=float(pdata.get("vendor_tdh_ft", 0.0)), step=5.0, key="vd_tdh",
            help="0 = unknown (estimated from motor hp).")
    with vd7:
        vendor_eff = st.number_input(
            "Vacom Pump Efficiency (%)", min_value=30.0, max_value=95.0,
            value=float(pdata.get("vendor_pump_eff", 80.0)), key="vd_eff")
    with vd8:
        vendor_load = st.number_input(
            "Motor Load (% of nameplate)", min_value=30.0, max_value=115.0,
            value=float(pdata.get("vendor_motor_load", 85.0)), key="vd_load",
            help="Used only when Rated TDH is unknown.")
    with vd9:
        vendor_assumed_velocity = st.number_input(
            "Assumed Pipe Velocity (ft/s)", min_value=3.0, max_value=15.0,
            value=float(pdata.get("vendor_assumed_velocity", 10.0)), key="vd_velocity",
            help="Used only when Rated Flow is unknown. 8-12 ft/s typical.")

    vendor_pipe_area = math.pi / 4 * (vendor_pipe_id / 12.0) ** 2
    gpm_per_fps = vendor_pipe_area * 448.83

    if vendor_flow_in > 0:
        v_flow, v_flow_src = vendor_flow_in, "Vacom"
    else:
        v_flow, v_flow_src = gpm_per_fps * vendor_assumed_velocity, "Estimated from pipe"
    v_velocity = v_flow / gpm_per_fps

    if vendor_duty_basis == duty_options[0]:
        v_duty = req["hx_duty"]
    else:
        v_duty = req["hx_duty"] * (vacom_distillate_gpm * 500.0) / max(vapor_lb_hr, 1.0)

    v_lbhr = v_flow * 8.34 * brine_sg * 60.0
    v_flash_dt = v_duty * 1e6 / (brine_cp * v_lbhr) if v_lbhr > 0 else 0.0

    if vendor_tdh_in > 0:
        v_tdh, v_tdh_src = vendor_tdh_in, "Vacom"
        v_bhp = v_flow * v_tdh * brine_sg / (3960.0 * vendor_eff / 100.0)
    else:
        v_bhp = vendor_motor_hp * vendor_load / 100.0
        v_tdh = (v_bhp * 3960.0 * vendor_eff / 100.0 / (v_flow * brine_sg)
                 if v_flow > 0 else 0.0)
        v_tdh_src = "Estimated from motor hp"

    v_extra_head = v_tdh - tdh_friction

    use_vendor = st.checkbox(
        "Use Vacom flow and TDH as the design basis for this tab",
        value=bool(pdata.get("use_vendor_basis", False)),
        key="vd_use_vendor"
    )

    # Keep the app's own calculation for the comparison table
    app_flow_calc = recirc_flow
    app_tdh_calc = tdh

    if use_vendor:
        recirc_flow = v_flow
        tdh = v_tdh

    recirc_ratio = recirc_flow / max(feed_flow, 1.0)
    loop_turnover = recirc_ratio
    recirc_lbhr = recirc_flow * 8.34 * brine_sg * 60.0
    actual_flash_dt = (
        req["hx_duty"] * 1e6 / (brine_cp * recirc_lbhr) if recirc_lbhr > 0 else 0.0
    )

    ##########################################################################
    # PUMP & MOTOR
    ##########################################################################
    st.subheader("Pump & Motor")

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        pump_type = st.selectbox(
            "Pump Type",
            ["Axial flow (elbow propeller)", "Mixed flow", "Centrifugal"],
            index=0,
            key="rp_pump_type",
            help="Axial-flow is standard for high-flow, low-head crystallizer loops."
        )
    with p2:
        efficiency = st.number_input(
            "Pump Efficiency (%)", min_value=30.0, max_value=95.0,
            value=float(pdata.get("recirc_eff", 80.0)), key="rp_eff")
    with p3:
        motor_eff = st.number_input(
            "Motor + VFD Efficiency (%)", min_value=50.0, max_value=100.0,
            value=float(pdata.get("recirc_motor_eff", 94.0)), key="rp_motor_eff")
    with p4:
        motor_sf = st.number_input(
            "Motor Sizing Factor", min_value=1.0, max_value=1.5,
            value=float(pdata.get("recirc_motor_sf", 1.15)), step=0.05, key="rp_motor_sf")

    brake_hp = recirc_flow * tdh * brine_sg / (3960.0 * efficiency / 100.0)
    recirc_hp = brake_hp
    recirc_kw = brake_hp * 0.746 / (motor_eff / 100.0)  # electrical input
    hydraulic_kw = recirc_flow * tdh * brine_sg / 3960.0 * 0.746

    motor_needed_hp = brake_hp * motor_sf
    recommended_motor_hp = _std_motor(motor_needed_hp)

    ##########################################################################
    # PIPING, BOILING SUPPRESSION, NPSH
    ##########################################################################
    st.subheader("Piping, Boiling Suppression & NPSH")

    n1, n2, n3, n4, n5 = st.columns(5)

    with n1:
        pipe_velocity = st.number_input(
            "Pipe Velocity (ft/s)", min_value=3.0, max_value=15.0,
            value=float(pdata.get("recirc_pipe_velocity", 9.0)), key="rp_pipe_velocity",
            help="8-12 ft/s keeps crystals suspended.")
    with n2:
        hx_submergence = st.number_input(
            "HX Top Below Liquid Level (ft)", min_value=0.0,
            value=float(pdata.get("hx_submergence_ft", 10.0)), key="rp_hx_submergence",
            help="Static head on the HX tubes that stops boiling inside them.")
    with n3:
        suction_static = st.number_input(
            "Liquid Level Above Pump Suction (ft)", min_value=0.0,
            value=float(pdata.get("pump_static_head_ft", 20.0)), key="rp_suction_static")
    with n4:
        suction_losses = st.number_input(
            "Suction Line Losses (ft)", min_value=0.0,
            value=float(pdata.get("pump_suction_losses_ft", 2.0)), key="rp_suction_losses")
    with n5:
        npshr = st.number_input(
            "Pump NPSHr (ft, vendor)", min_value=0.0,
            value=float(pdata.get("pump_npshr_ft", 12.0)), key="rp_npshr")

    # Pipe size
    flow_cfs = recirc_flow / 448.83
    pipe_diameter_in = math.sqrt(4 * flow_cfs / (math.pi * pipe_velocity)) * 12.0

    # Boiling suppression: brine at HX outlet must stay below its boiling point
    outlet_c = crystallizer_boiling_temp + actual_flash_dt / 1.8
    water_equiv_c = outlet_c - boiling_point_elevation
    p_boil_outlet = psat_psia_from_c(water_equiv_c)
    extra_psi = max(p_boil_outlet - crystallizer_pressure_psia, 0.0)
    submergence_required = extra_psi * 2.31 / brine_sg
    submergence_margin = hx_submergence - submergence_required

    # NPSH: liquid enters pump at its boiling point -> vessel P = vapor P
    npsha = suction_static - suction_losses
    npsh_margin = npsha - npshr

    ##########################################################################
    # RESULTS
    ##########################################################################
    st.subheader("Recirculation Loop Performance")

    k1, k2, k3, k4, k5, k6 = st.columns(6)

    with k1:
        st.metric("Feed Flow", f"{feed_flow:.1f} GPM")
    with k2:
        st.metric("Recirc Flow", f"{recirc_flow:,.0f} GPM",
                  f"{recirc_flow - req['flow_gpm']:+,.0f} vs required", delta_color="off")
    with k3:
        st.metric("TDH", f"{tdh:.1f} ft")
    with k4:
        st.metric("Brake Power", f"{brake_hp:,.1f} HP")
    with k5:
        st.metric("Motor", f"{recommended_motor_hp:g} HP")
    with k6:
        st.metric("Electrical Load", f"{recirc_kw:,.1f} kW")

    k7, k8, k9, k10, k11, k12 = st.columns(6)

    with k7:
        st.metric("Recirc / Feed", f"{recirc_ratio:,.0f} x")
    with k8:
        st.metric("Actual Flash ΔT", f"{actual_flash_dt:.1f} °F")
    with k9:
        st.metric("Pipe Diameter", f"{pipe_diameter_in:.1f} in")
    with k10:
        st.metric("Submergence Needed", f"{submergence_required:.1f} ft",
                  f"{submergence_margin:+.1f} ft margin", delta_color="normal")
    with k11:
        st.metric("NPSH Available", f"{npsha:.1f} ft",
                  f"{npsh_margin:+.1f} ft vs NPSHr", delta_color="normal")
    with k12:
        st.metric("Loop Temp", f"{crystallizer_boiling_temp:.1f} °C")

    if submergence_margin < 0:
        st.error("HX is not submerged enough - brine will boil in the tubes and scale them.")
    if npsh_margin < 3.0:
        st.warning("NPSH margin below 3 ft - lower the pump or reduce suction losses.")
    if pump_type == "Centrifugal" and tdh < 30:
        st.info("Low head / very high flow - an axial-flow pump is normally a better fit.")

    st.caption(
        f"Crystallizer {crystallizer_pressure_psia:.2f} psia ({crystallizer_pressure_bar:.3f} bar a) | "
        f"BPE {boiling_point_elevation:.1f} °C | boiling {crystallizer_boiling_temp:.1f} °C | "
        f"HX outlet {outlet_c:.1f} °C boils at {p_boil_outlet:.2f} psia → needs "
        f"{extra_psi:.2f} psi static head."
    )

    ##########################################################################
    # APP vs VACOM COMPARISON
    ##########################################################################
    st.subheader("⚖️ App Calculation vs Vacom Selection")

    if use_vendor:
        st.info("Vacom flow and TDH are being used as the design basis above. "
                "The 'App' column below shows the app's own calculation for reference.")

    app_bhp_calc = app_flow_calc * app_tdh_calc * brine_sg / (3960.0 * efficiency / 100.0)
    app_motor_calc = _std_motor(app_bhp_calc * motor_sf)
    app_velocity_in_vendor_pipe = app_flow_calc / gpm_per_fps
    app_flash_dt_calc = (
        req["hx_duty"] * 1e6 / (brine_cp * app_flow_calc * 8.34 * brine_sg * 60.0)
        if app_flow_calc > 0 else 0.0
    )


    def _pct(a, b):
        return (b - a) / a * 100.0 if a else 0.0


    flow_diff = _pct(app_flow_calc, v_flow)
    flow_flag = "✓ Close" if abs(flow_diff) <= 15 else "⚠ Different flash ΔT basis"

    dt_ref = app_flash_dt_calc if vendor_duty_basis == duty_options[0] else flash_dt_f
    dt_flag = ("✓ Close" if abs(v_flash_dt - dt_ref) <= 1.5
               else f"⚠ Set HX flash ΔT ≈ {v_flash_dt:.1f} °F to match")

    if v_tdh > 2 * app_tdh_calc:
        tdh_flag = "⚠ Vacom head much higher - ask for head breakdown"
    elif abs(_pct(app_tdh_calc, v_tdh)) <= 25:
        tdh_flag = "✓ Close"
    else:
        tdh_flag = "⚠ Check head losses"

    bhp_flag = "✓ Close" if abs(_pct(app_bhp_calc, v_bhp)) <= 25 else "⚠ Follows flow / head gap"
    motor_flag = ("✓ Same" if app_motor_calc == vendor_motor_hp
                  else f"⚠ {vendor_motor_hp / max(app_motor_calc, 1):.1f}x app motor")
    vel_flag = ("✓ 8-12 ft/s" if 8.0 <= v_velocity <= 12.0
                else "⚠ Outside 8-12 ft/s")

    cmp_df = pd.DataFrame({
        "Parameter": [
            "Recirculation flow (GPM)",
            "Flash ΔT (°F)",
            "Recirc / feed ratio",
            "TDH (ft)",
            "Brake power (hp)",
            "Motor (hp)",
            f"Velocity in {vendor_pipe_nominal:g} in pipe (ft/s)",
        ],
        "App": [
            f"{app_flow_calc:,.0f}",
            f"{app_flash_dt_calc:.1f}",
            f"{app_flow_calc / max(feed_flow, 1):,.0f}",
            f"{app_tdh_calc:.1f}",
            f"{app_bhp_calc:,.0f}",
            f"{app_motor_calc:g}",
            f"{app_velocity_in_vendor_pipe:.1f}",
        ],
        "Vacom": [
            f"{v_flow:,.0f}  ({v_flow_src})",
            f"{v_flash_dt:.1f}",
            f"{v_flow / max(feed_flow, 1):,.0f}",
            f"{v_tdh:.1f}  ({v_tdh_src})",
            f"{v_bhp:,.0f}",
            f"{vendor_motor_hp:g}",
            f"{v_velocity:.1f}",
        ],
        "Difference": [
            f"{flow_diff:+.0f} %",
            f"{v_flash_dt - app_flash_dt_calc:+.1f} °F",
            "",
            f"{v_tdh - app_tdh_calc:+.1f} ft",
            f"{v_bhp - app_bhp_calc:+,.0f} hp",
            f"{vendor_motor_hp - app_motor_calc:+g} hp",
            "",
        ],
        "Flag": [flow_flag, dt_flag, "", tdh_flag, bhp_flag, motor_flag, vel_flag],
    })

    st.dataframe(cmp_df, width="stretch", hide_index=True)

    if v_extra_head > 10:
        st.warning(
            f"Vacom TDH {v_tdh:.0f} ft is {v_extra_head:.0f} ft more than the app's loop "
            f"friction ({tdh_friction:.0f} ft). Ask Vacom what makes up this head: a "
            "throttling valve or orifice for boiling suppression, multi-pass / high-velocity "
            "HX tubes, extra equipment in the loop, or motor sized for a harder case "
            "(start-up, high slurry density)."
        )
    if v_flow_src != "Vacom" or v_tdh_src != "Vacom":
        st.caption("Values marked 'Estimated' are back-calculated. Enter Vacom's datasheet "
                   "flow and TDH for an exact comparison.")

    ##########################################################################
    # CONCENTRATE BLEED
    ##########################################################################
    concentrate_bleed_flow = st.number_input(
        "Average Concentrate Bleed (GPM)",
        value=float(pdata.get("concentrate_bleed_flow", 35.0)),
        key="rp_bleed",
        help="Typical Vacom concentrate bleed 28-42 GPM"
    )
    bleed_ratio = concentrate_bleed_flow / max(recirc_flow, 1.0) * 100.0

    ##########################################################################
    # DESIGN SUMMARY
    ##########################################################################
    st.subheader("📋 Pump Design Summary")

    recirc_m3hr = recirc_flow * 0.2271
    specific_energy = recirc_kw / max(feed_flow, 1.0)

    summary_df = pd.DataFrame({
        "Parameter": [
            "Feed Flow",
            "HX Duty",
            "Flash ΔT (design / actual)",
            "Required Recirculation Flow",
            "Design Recirculation Flow",
            "Recirculation Mass Flow",
            "Recirculation / Feed Ratio",
            "TDH (friction + margin)",
            "Pump Type",
            "Pump Efficiency",
            "Hydraulic Power",
            "Brake Power",
            "Electrical Load",
            "Recommended Motor",
            "Specific Energy",
            "Estimated Pipe Size",
            "HX Submergence (needed / provided)",
            "NPSH (available / required)",
            "Concentrate Bleed / Bleed Ratio",
            "Crystallizer Pressure",
            "Crystallizer Boiling Temp",
        ],
        "Value": [
            f"{feed_flow:.1f} GPM",
            f"{req['hx_duty']:.2f} MMBTU/hr",
            f"{flash_dt_f:.1f} / {actual_flash_dt:.1f} °F",
            f"{req['flow_gpm']:,.0f} GPM",
            f"{recirc_flow:,.0f} GPM ({recirc_m3hr:,.0f} m³/h)",
            f"{recirc_lbhr:,.0f} lb/hr",
            f"{recirc_ratio:,.0f} : 1",
            f"{tdh:.1f} ft ({tdh_friction:.1f} + {head_margin:.0f}%)",
            pump_type,
            f"{efficiency:.0f} %",
            f"{hydraulic_kw:,.1f} kW",
            f"{brake_hp:,.1f} HP",
            f"{recirc_kw:,.1f} kW",
            f"{recommended_motor_hp:g} HP",
            f"{specific_energy:.2f} kW per GPM feed",
            f"{pipe_diameter_in:.1f} in at {pipe_velocity:.1f} ft/s",
            f"{submergence_required:.1f} / {hx_submergence:.1f} ft",
            f"{npsha:.1f} / {npshr:.1f} ft",
            f"{concentrate_bleed_flow:.1f} GPM / {bleed_ratio:.2f} %",
            f"{crystallizer_pressure_psia:.2f} psia ({crystallizer_pressure_bar:.3f} bar a)",
            f"{crystallizer_boiling_temp:.1f} °C",
        ],
    })

    st.dataframe(summary_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE DATA (existing keys kept for other tabs)
    ##########################################################################
    pdata.update({
        "recirc_flow_basis": flow_basis,
        "recirc_flow_margin": flow_margin,
        "recirculation_ratio": recirc_ratio,
        "recirc_manual_ratio": manual_ratio,
        "recirculation_flow": recirc_flow,
        "recirculation_flow_required": req["flow_gpm"],
        "recirc_lbhr": recirc_lbhr,
        "recirc_actual_flash_dt": actual_flash_dt,
        "loop_temp": crystallizer_boiling_temp,
        "recirc_tdh": tdh,
        "rp_loss_tubes": loss_tubes,
        "rp_loss_channels": loss_channels,
        "rp_loss_piping": loss_piping,
        "rp_loss_nozzle": loss_nozzle,
        "rp_head_margin": head_margin,
        "recirc_eff": efficiency,
        "recirc_motor_eff": motor_eff,
        "recirc_motor_sf": motor_sf,
        "recirc_hp": recirc_hp,
        "recirc_kw": recirc_kw,
        "recirc_motor_hp": recommended_motor_hp,
        "recirc_pipe_velocity": pipe_velocity,
        "recirc_pipe_diameter_in": pipe_diameter_in,
        "hx_submergence_ft": hx_submergence,
        "hx_submergence_required_ft": submergence_required,
        "pump_static_head_ft": suction_static,
        "pump_suction_losses_ft": suction_losses,
        "pump_npshr_ft": npshr,
        "pump_npsha_ft": npsha,
        "liquor_density": brine_sg * 1000.0,
        "vendor_motor_hp": vendor_motor_hp,
        "vendor_pipe_nominal": vendor_pipe_nominal,
        "vendor_pipe_id": vendor_pipe_id,
        "vendor_duty_basis_idx": duty_options.index(vendor_duty_basis),
        "vendor_flow_gpm": vendor_flow_in,
        "vendor_tdh_ft": vendor_tdh_in,
        "vendor_pump_eff": vendor_eff,
        "vendor_motor_load": vendor_load,
        "vendor_assumed_velocity": vendor_assumed_velocity,
        "use_vendor_basis": use_vendor,
        "vendor_flow_calc_gpm": v_flow,
        "vendor_tdh_calc_ft": v_tdh,
        "vendor_bhp_calc": v_bhp,
        "vendor_flash_dt_calc": v_flash_dt,
        "vendor_pipe_velocity": v_velocity,
        "concentrate_bleed_flow": concentrate_bleed_flow,
        "loop_turnover": loop_turnover,
        "bleed_ratio": bleed_ratio,
    })

    save_project(project_name, st.session_state.project_data)

    st.success("✅ Recirculation Pump Auto Saved")

##############################################################################
# PRIMARY HEAT EXCHANGER
# Forced-circulation crystallizer heater: recirculating brine (tube side) is
# heated by compressed MVR vapor condensing on the shell side.
#
#   HX inlet  = brine boiling temp in vessel          (from Design Basis)
#   HX outlet = boiling temp + flash ΔT               (flash ΔT is an input)
#   Duty      = evaporation duty + feed heat-up       (from Design Basis)
#   Recirc    = duty / (Cp x flash ΔT)                (calculated)
#   Area      = duty / (U x LMTD), LMTD vs condensing vapor temp
##############################################################################

with tab5:
    from mvr_thermo import hfg_btu_lb, recirc_requirement
    import math

    st.header("🌡️ Primary Heat Exchanger")
    flow_slot(('E-220',))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM DESIGN BASIS
    ##########################################################################
    boil_c = float(pdata.get("crystallizer_boiling_temp", 87.3))
    vessel_sat_c = float(pdata.get("saturation_temp", 81.3))
    cond_c = float(pdata.get("mvr_discharge_sat_temp", 105.3))
    vapor_lb_hr = float(pdata.get("distillate_lb_hr_calc", 29100.0))
    feed_lb_hr = float(pdata.get("feed_lb_hr", 31500.0))
    feed_in_c = float(pdata.get("preheat_out_c", 80.0))
    recirc_flow_tab4 = float(pdata.get("recirculation_flow", 0.0))
    design_flow = float(pdata.get("design_flow", 60.0))

    c_to_f_ = lambda t: t * 9 / 5 + 32
    boil_f = c_to_f_(boil_c)
    cond_f = c_to_f_(cond_c)
    feed_in_f = c_to_f_(feed_in_c)

    st.info(
        f"From Design Basis: brine boils at {boil_f:.1f} °F ({boil_c:.1f} °C) | "
        f"compressed vapor condenses at {cond_f:.1f} °F ({cond_c:.1f} °C) | "
        f"vapor {vapor_lb_hr:,.0f} lb/hr | feed {feed_lb_hr:,.0f} lb/hr at {feed_in_f:.0f} °F"
    )

    ##########################################################################
    # USER INPUTS
    ##########################################################################
    st.subheader("Design Inputs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        flash_delta_t = st.number_input(
            "Flash ΔT (°F)", min_value=0.5, max_value=20.0,
            value=float(pdata.get("flash_delta_t_input", 10.8)), step=0.5,
            key="phx_flash_dt",
            help="Brine temperature rise through the HX = superheat flashed off in the "
                 "vessel. Design 10.8 °F (6 °C); typical range 5-15 °F (3-8 °C)."
        )

    with c2:
        brine_sg = st.number_input(
            "Brine SG", min_value=1.0, max_value=1.6,
            value=float(pdata.get("brine_sg", 1.20)), step=0.01, key="phx_brine_sg")

    with c3:
        brine_cp = st.number_input(
            "Brine Cp (BTU/lb-°F)", min_value=0.5, max_value=1.0,
            value=float(pdata.get("brine_cp", 0.80)), step=0.01, key="phx_brine_cp")

    with c4:
        feed_cp = st.number_input(
            "Feed Cp (BTU/lb-°F)", min_value=0.5, max_value=1.0,
            value=float(pdata.get("feed_cp", 0.95)), step=0.01, key="phx_feed_cp")

    c5, c6, c7, c8 = st.columns(4)

    with c5:
        u_value = st.number_input(
            "Clean U Value (BTU/hr-ft²-°F)", min_value=10.0,
            value=float(pdata.get("cryst_hx_u", 350.0)), key="cryst_hx_u",
            help="Condensing vapor / high-velocity brine. Typical 300-500 clean.")

    with c6:
        fouling_factor = st.number_input(
            "Fouling Factor (hr-ft²-°F/BTU)", min_value=0.0,
            value=float(pdata.get("hx_fouling_factor", 0.0005)),
            format="%.4f", step=0.0001, key="phx_fouling")

    with c7:
        area_margin = st.number_input(
            "Area Margin (%)", min_value=0.0, max_value=100.0,
            value=float(pdata.get("hx_area_margin", 10.0)), key="phx_area_margin")

    with c8:
        tube_passes = st.number_input(
            "Tube Passes", min_value=1, max_value=4,
            value=int(pdata.get("hx_tube_passes", 1)), key="phx_passes",
            help="Crystallizer heaters are usually single pass.")

    c9, c10, c11, c12 = st.columns(4)

    with c9:
        tube_od = st.number_input(
            "Tube OD (in)", min_value=0.5, max_value=2.0,
            value=float(pdata.get("hx_tube_od", 1.0)), step=0.125, key="phx_tube_od")

    with c10:
        tube_wall = st.number_input(
            "Tube Wall (in)", min_value=0.02, max_value=0.2,
            value=float(pdata.get("hx_tube_wall", 0.049)), step=0.001,
            format="%.3f", key="phx_tube_wall",
            help="0.049 in = 18 BWG (typical titanium).")

    with c11:
        tube_length = st.number_input(
            "Tube Length (ft)", min_value=4.0, max_value=40.0,
            value=float(pdata.get("hx_tube_length", 20.0)), key="phx_tube_length")

    with c12:
        target_velocity = st.number_input(
            "Target Tube Velocity (ft/s)", min_value=2.0, max_value=15.0,
            value=float(pdata.get("hx_tube_velocity", 7.0)), step=0.5,
            key="phx_target_velocity",
            help="6-8 ft/s typical to limit scaling and suppress boiling in tubes.")

    ##########################################################################
    # TEMPERATURES
    ##########################################################################
    inlet_temp = boil_f  # brine leaves vessel at boiling temp
    outlet_temp = boil_f + flash_delta_t  # superheated, flashes in vessel
    delta_temp = outlet_temp - inlet_temp

    dt1 = cond_f - inlet_temp  # hot end
    dt2 = cond_f - outlet_temp  # cold end (approach)
    approach_temp = dt2

    hx_ok = dt2 > 0
    if hx_ok:
        delta_t_lm = (dt1 - dt2) / math.log(dt1 / dt2) if abs(dt1 - dt2) > 1e-6 else dt1
    else:
        delta_t_lm = 0.0
        st.error(
            f"HX outlet ({outlet_temp:.1f} °F) is not below the condensing vapor "
            f"temperature ({cond_f:.1f} °F). Reduce flash ΔT or raise blower discharge "
            "pressure on the Design Basis tab."
        )

    ##########################################################################
    # DUTY & RECIRCULATION FLOW
    ##########################################################################
    # Same calculation the Recirculation Pump tab uses
    req = recirc_requirement(
        vapor_lb_hr=vapor_lb_hr,
        vessel_sat_c=vessel_sat_c,
        boil_c=boil_c,
        feed_lb_hr=feed_lb_hr,
        feed_in_c=feed_in_c,
        flash_dt_f=flash_delta_t,
        brine_cp=brine_cp,
        brine_sg=brine_sg,
        feed_cp=feed_cp,
    )

    hfg_vessel = req["hfg_vessel"]
    hfg_cond = hfg_btu_lb(cond_c)

    evap_duty = req["evap_duty"]  # MMBTU/hr
    feed_heat_duty = req["feed_heat_duty"]
    hx_duty = req["hx_duty"]
    duty_kw = hx_duty * 293.07

    flow_lb_hr = req["flow_lb_hr"]
    recirculation_flow = req["flow_gpm"]  # GPM
    flow_m3hr = recirculation_flow * 0.2271
    recirc_ratio_required = recirculation_flow / max(design_flow, 1.0)
    flash_fraction = req["flash_fraction_pct"]  # % per pass

    # Compressed vapor condensed on the shell side
    steam_required = hx_duty * 1e6 / hfg_cond

    ##########################################################################
    # AREA
    ##########################################################################
    u_dirty = 1.0 / (1.0 / u_value + fouling_factor)
    if hx_ok:
        clean_area = hx_duty * 1e6 / (u_value * delta_t_lm)
        dirty_area = hx_duty * 1e6 / (u_dirty * delta_t_lm)
    else:
        clean_area = dirty_area = 0.0
    design_area = dirty_area * (1 + area_margin / 100.0)
    area = design_area
    heat_flux = hx_duty * 1e6 / max(design_area, 1.0)

    ##########################################################################
    # TUBES & VELOCITY
    ##########################################################################
    tube_id = tube_od - 2 * tube_wall
    tube_area = math.pi * (tube_od / 12.0) * tube_length  # ft² per tube
    tube_flow_area = math.pi / 4 * (tube_id / 12.0) ** 2  # ft² per tube
    flow_cfs = flow_lb_hr / (62.4 * brine_sg) / 3600.0

    tube_count_area = design_area / max(tube_area, 0.01)
    tube_count_velocity = (
            flow_cfs / (target_velocity * tube_flow_area) * tube_passes
    )
    tube_count = max(tube_count_area, tube_count_velocity)
    actual_area = tube_count * tube_area
    tube_velocity = (flow_cfs / (tube_flow_area * tube_count / tube_passes)
                     if tube_count > 0 and tube_flow_area > 0 else 0.0)
    tubes_govern = "Velocity" if tube_count_velocity > tube_count_area else "Area"

    ##########################################################################
    # PERFORMANCE RESULTS
    ##########################################################################
    st.subheader("Heat Exchanger Performance")

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.metric("Recirc Flow (required)", f"{recirculation_flow:,.0f} GPM")
    with k2:
        st.metric("Brine In → Out", f"{inlet_temp:.1f} → {outlet_temp:.1f} °F")
    with k3:
        st.metric("HX Duty", f"{hx_duty:.2f} MMBTU/hr")
    with k4:
        st.metric("LMTD", f"{delta_t_lm:.1f} °F")
    with k5:
        st.metric("Design Area", f"{design_area:,.0f} ft²")

    k6, k7, k8, k9, k10 = st.columns(5)

    with k6:
        st.metric("Flash ΔT", f"{flash_delta_t:.1f} °F")
    with k7:
        st.metric("Approach (cold end)", f"{approach_temp:.1f} °F")
    with k8:
        st.metric("Flashed per Pass", f"{flash_fraction:.2f} %")
    with k9:
        st.metric("Recirc / Feed Ratio", f"{recirc_ratio_required:,.0f} x")
    with k10:
        st.metric("Tube Velocity", f"{tube_velocity:.1f} ft/s")

    if 0 < recirc_flow_tab4 < recirculation_flow * 0.99:
        st.warning(
            f"Recirculation pump tab uses {recirc_flow_tab4:,.0f} GPM, but this duty and "
            f"flash ΔT need {recirculation_flow:,.0f} GPM. Use 'Calculated from HX duty' on "
            f"that tab, or set its ratio to at least {recirc_ratio_required:,.0f}."
        )

    if not 5.0 <= tube_velocity <= 10.0:
        st.warning(f"Tube velocity {tube_velocity:.1f} ft/s is outside the usual 5-10 ft/s range.")

    if flash_delta_t > 15.0:
        st.warning("Flash ΔT above ~15 °F (8 °C) risks boiling in the tubes, scaling and fine crystals.")

    ##########################################################################
    # THERMAL PERFORMANCE
    ##########################################################################
    st.subheader("🔥 Thermal Performance")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.metric("Duty", f"{duty_kw:,.0f} kW")
    with t2:
        st.metric("Vapor Condensed (shell)", f"{steam_required:,.0f} lb/hr",
                  help="Compressed MVR vapor needed on the shell side = duty / latent heat "
                       "at condensing temp.")
    with t3:
        st.metric("Heat Flux", f"{heat_flux:,.0f} BTU/hr-ft²")
    with t4:
        st.metric("Dirty U", f"{u_dirty:,.0f} BTU/hr-ft²-°F")

    st.caption(
        f"Duty = evaporation {evap_duty:.2f} MMBTU/hr ({vapor_lb_hr:,.0f} lb/hr x "
        f"{hfg_vessel:.0f} BTU/lb) + feed heat-up {feed_heat_duty:.2f} MMBTU/hr "
        f"({feed_in_f:.0f} → {boil_f:.1f} °F)."
    )

    ##########################################################################
    # HX MECHANICAL DESIGN
    ##########################################################################
    st.subheader("🔧 Mechanical Design Data")

    b1, b2, b3, b4, b5 = st.columns(5)

    with b1:
        st.metric("Flow", f"{flow_m3hr:,.0f} m³/hr")
    with b2:
        st.metric("Clean Area", f"{clean_area:,.0f} ft²")
    with b3:
        st.metric("Dirty Area", f"{dirty_area:,.0f} ft²")
    with b4:
        st.metric("Tube Count", f"{tube_count:,.0f}", help=f"Governed by {tubes_govern}")
    with b5:
        st.metric("Installed Area", f"{actual_area:,.0f} ft²")

    ##########################################################################
    # DESIGN SUMMARY
    ##########################################################################
    st.subheader("📋 HX Design Summary")

    hx_df = pd.DataFrame({
        "Parameter": [
            "Recirculation Flow (required)",
            "Recirculation / Feed Ratio",
            "Brine Inlet Temperature (= boiling temp)",
            "Brine Outlet Temperature",
            "Flash ΔT",
            "Condensing Vapor Temperature",
            "Approach Temperature (cold end)",
            "LMTD",
            "Evaporation Duty",
            "Feed Heat-up Duty",
            "Total Heat Duty",
            "Duty (kW)",
            "Vapor Condensed (shell side)",
            "Clean U / Dirty U",
            "Clean Area",
            "Dirty Area",
            "Design Area (with margin)",
            "Tube OD x Wall x Length",
            "Tube Count / Passes",
            "Tube Velocity",
            "Installed Area",
        ],
        "Value": [
            f"{recirculation_flow:,.0f} GPM ({flow_lb_hr:,.0f} lb/hr)",
            f"{recirc_ratio_required:,.0f} x",
            f"{inlet_temp:.1f} °F",
            f"{outlet_temp:.1f} °F",
            f"{flash_delta_t:.1f} °F",
            f"{cond_f:.1f} °F",
            f"{approach_temp:.1f} °F",
            f"{delta_t_lm:.1f} °F",
            f"{evap_duty:.2f} MMBTU/hr",
            f"{feed_heat_duty:.2f} MMBTU/hr",
            f"{hx_duty:.2f} MMBTU/hr",
            f"{duty_kw:,.0f} kW",
            f"{steam_required:,.0f} lb/hr",
            f"{u_value:.0f} / {u_dirty:.0f} BTU/hr-ft²-°F",
            f"{clean_area:,.0f} ft²",
            f"{dirty_area:,.0f} ft²",
            f"{design_area:,.0f} ft²",
            f"{tube_od:.3f} in x {tube_wall:.3f} in x {tube_length:.0f} ft",
            f"{tube_count:,.0f} / {tube_passes}  (governed by {tubes_govern.lower()})",
            f"{tube_velocity:.1f} ft/s",
            f"{actual_area:,.0f} ft²",
        ],
    })

    st.dataframe(hx_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE RESULTS (existing keys kept for other tabs)
    ##########################################################################
    pdata.update({
        "cryst_hx_area": area,
        "cryst_hx_duty": hx_duty,
        "cryst_hx_in": inlet_temp,
        "cryst_hx_out": outlet_temp,
        "flash_delta_t": flash_delta_t,
        "flash_delta_t_input": flash_delta_t,
        "cryst_hx_u": u_value,
        "cryst_hx_lmtd": delta_t_lm,
        "cryst_hx_condensing_temp_f": cond_f,
        "recirculation_flow_required": recirculation_flow,
        "recirc_ratio_required": recirc_ratio_required,
        "brine_sg": brine_sg,
        "brine_cp": brine_cp,
        "feed_cp": feed_cp,
        "hx_fouling_factor": fouling_factor,
        "hx_area_margin": area_margin,
        "hx_tube_passes": tube_passes,
        "hx_tube_od": tube_od,
        "hx_tube_wall": tube_wall,
        "hx_tube_length": tube_length,
        "hx_tube_velocity": target_velocity,
        "hx_tube_velocity_actual": tube_velocity,
        "steam_required": steam_required,
        "dirty_area": dirty_area,
        "tube_count": tube_count,
        "duty_kw": duty_kw,
        "approach_temp": approach_temp,
    })

    save_project(project_name, st.session_state.project_data)

    st.success("✅ Primary HX Calculations Auto Saved")

##############################################################################
# CRYSTALLIZER THERMAL DESIGN
# Read-only heat balance built from the Design Basis, Feed HX and Primary HX
# tabs. Change vessel pressure / BPE / etc. on the Basis tab.
##############################################################################

with tab6:
    from mvr_thermo import hfg_btu_lb, c_to_f

    st.header("💎 Crystallizer Thermal Design")
    flow_slot(('V-200', 'E-220'))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM OTHER TABS
    ##########################################################################
    feed_gpm = float(pdata.get("design_flow", 60.0))
    feed_lb_hr = float(pdata.get("feed_lb_hr", feed_gpm * 500.0))
    salt_lb_hr = float(pdata.get("salt_lb_hr_calc", 0.0))
    cake_water_lb_hr = float(pdata.get("cake_water_lb_hr", 0.0))
    dist_lb_hr = float(pdata.get("distillate_lb_hr_calc", feed_lb_hr * 0.92))
    dist_gpm = float(pdata.get("distillate_gpm_calc", dist_lb_hr / 500.0))
    feed_tds = float(pdata.get("feed_tds", pdata.get("tds", 72000.0)))

    vessel_psia = float(pdata.get("vessel_pressure_sp", 7.25))
    t_sat = float(pdata.get("saturation_temp", 90.0))
    bpe_c = float(pdata.get("boiling_point_elevation", 6.0))
    t_boil = float(pdata.get("crystallizer_boiling_temp", t_sat + bpe_c))
    t_cond = float(pdata.get("mvr_discharge_sat_temp", t_boil + 10.0))
    hx_dt = float(pdata.get("primary_hx_driving_dt", t_cond - t_boil))
    feed_in_c = float(pdata.get("preheat_out_c", 80.0))  # pre-HX outlet

    hx_duty = float(pdata.get("cryst_hx_duty", 0.0))  # MMBTU/hr
    hx_area = float(pdata.get("cryst_hx_area", 0.0))
    hx_u = float(pdata.get("cryst_hx_u", 0.0))
    hx_lmtd = float(pdata.get("cryst_hx_lmtd", 0.0))
    flash_dt = float(pdata.get("flash_delta_t", 0.0))
    recirc_req = float(pdata.get("recirculation_flow_required", 0.0))

    losses = float(pdata.get("heat_losses_mmbtu_hr", 0.0))
    blower_heat = float(pdata.get("blower_heat_mmbtu_hr", 0.0))
    makeup_factor = float(pdata.get("makeup_factor", 2.5))
    makeup_design = float(pdata.get("boiler_makeup_design_lb_hr", 0.0))
    makeup_steady = makeup_design / makeup_factor if makeup_factor > 0 else makeup_design

    st.info(
        f"From Design Basis: {feed_gpm:.0f} GPM feed | vessel {vessel_psia:.2f} psia "
        f"(sat. {t_sat:.1f} °C) + BPE {bpe_c:.1f} °C = boiling {t_boil:.1f} °C "
        f"({c_to_f(t_boil):.0f} °F) | vapor condenses at {t_cond:.1f} °C "
        f"({c_to_f(t_cond):.0f} °F). Change these on the Basis tab."
    )

    ##########################################################################
    # HEAT BALANCE (MMBTU/hr)
    ##########################################################################
    hfg_vessel = hfg_btu_lb(t_sat)
    evap_duty = dist_lb_hr * hfg_vessel / 1e6
    feed_heatup = max(feed_lb_hr * 0.95 * (t_boil - feed_in_c) * 1.8, 0.0) / 1e6
    duty_calc = evap_duty + feed_heatup
    if hx_duty <= 0:
        hx_duty = duty_calc

    ##########################################################################
    # DISTILLATE QUALITY (entrainment through the mist eliminator)
    ##########################################################################
    st.subheader("Distillate Quality Inputs")
    q1, q2, q3 = st.columns(3)
    with q1:
        ml_tds = st.number_input(
            "Mother Liquor TDS (mg/L)", min_value=0.0, step=10000.0,
            value=float(pdata.get("mother_liquor_tds", 350000.0)), key="ct_ml_tds",
            help="Dissolved solids in the boiling liquor (near saturation in a ZLD crystallizer).")
    with q2:
        entrainment_ppm = st.number_input(
            "Entrainment After Demister (ppm w/w of vapor)", min_value=0.0, step=5.0,
            value=float(pdata.get("entrainment_ppm", 20.0)), key="ct_entrain",
            help="Liquor droplets carried over per million lb vapor. 5-50 ppm is typical for a "
                 "mesh or chevron demister.")
    with q3:
        tds_per_us = st.number_input(
            "TDS / Conductivity Factor", min_value=0.3, max_value=1.0, step=0.05,
            value=float(pdata.get("tds_cond_factor", 0.65)), key="ct_tds_factor",
            help="mg/L per µS/cm. About 0.65 for sulfate/chloride water.")

    dist_tds = ml_tds * entrainment_ppm / 1e6
    dist_cond = dist_tds / max(tds_per_us, 0.01)

    ##########################################################################
    # RESULTS
    ##########################################################################
    st.subheader("Operating Point")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Vessel Pressure", f"{vessel_psia:.2f} psia",
              f"{vessel_psia - float(pdata.get('atmospheric_pressure', 14.696)):+.2f} psig",
              delta_color="off")
    r2.metric("Saturation Temp", f"{t_sat:.1f} °C")
    r3.metric("BPE", f"{bpe_c:.1f} °C")
    r4.metric("Boiling Temp", f"{t_boil:.1f} °C", f"{c_to_f(t_boil):.0f} °F", delta_color="off")
    r5.metric("Condensing Temp", f"{t_cond:.1f} °C", f"{c_to_f(t_cond):.0f} °F", delta_color="off")
    r6.metric("HX Driving ΔT", f"{hx_dt:.1f} °C")

    st.subheader("Heat Balance")
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Evaporation", f"{dist_lb_hr:,.0f} lb/hr", f"{dist_gpm:.1f} GPM", delta_color="off")
    h2.metric("Latent Duty", f"{evap_duty:.2f} MMBTU/hr")
    h3.metric("Feed Heat-up", f"{feed_heatup:.2f} MMBTU/hr",
              f"{feed_in_c:.1f} → {t_boil:.1f} °C", delta_color="off")
    h4.metric("Primary HX Duty", f"{hx_duty:.2f} MMBTU/hr")
    h5.metric("Losses / Blower Heat", f"{losses:.2f} / {blower_heat:.2f}",
              help="MMBTU/hr, from the Basis tab boiler calculation.")
    h6.metric("Steady Makeup Steam", f"{makeup_steady:,.0f} lb/hr",
              f"design {makeup_design:,.0f} lb/hr", delta_color="off")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Installed HX Area", f"{hx_area:,.0f} ft²")
    k2.metric("HX U / LMTD", f"{hx_u:.0f} / {hx_lmtd:.1f} °F")
    k3.metric("Flash ΔT", f"{flash_dt:.1f} °F")
    k4.metric("Recirc Required", f"{recirc_req:,.0f} GPM")
    k5.metric("Distillate Quality", f"{dist_cond:.0f} µS/cm", f"{dist_tds:.1f} mg/L TDS",
              delta_color="off")

    if abs(duty_calc - hx_duty) > 0.05 * max(hx_duty, 0.01):
        st.warning(f"Heat balance here gives {duty_calc:.2f} MMBTU/hr but the Primary HX tab "
                   f"has {hx_duty:.2f}. Open the Primary HX tab to refresh it.")
    if hx_dt < 5:
        st.error(f"HX driving ΔT is only {hx_dt:.1f} °C - raise the blower discharge pressure.")
    elif hx_dt < 8:
        st.warning(f"HX driving ΔT {hx_dt:.1f} °C is tight (8-12 °C usual for MVR) - "
                   f"large HX area.")

    st.caption(
        "Scale control: in a ZLD crystallizer gypsum precipitates by design. It is kept off "
        "the tubes by seeded slurry (see the Vessel tab), high recirculation and no boiling "
        "in the tubes (HX submergence on the Recirc Pump tab)."
    )

    summary_df = pd.DataFrame({
        "Parameter": [
            "Feed", "Salt to cake", "Cake moisture", "Distillate / vapor",
            "Vessel pressure", "Boiling temperature", "Condensing temperature",
            "Latent duty", "Feed heat-up duty", "Primary HX duty", "Heat losses",
            "Blower heat input", "Steady makeup steam", "Design makeup steam",
            "Distillate conductivity",
        ],
        "Value": [
            f"{feed_gpm:.1f} GPM ({feed_lb_hr:,.0f} lb/hr, {feed_tds:,.0f} mg/L)",
            f"{salt_lb_hr:,.0f} lb/hr",
            f"{cake_water_lb_hr:,.0f} lb/hr",
            f"{dist_lb_hr:,.0f} lb/hr ({dist_gpm:.1f} GPM)",
            f"{vessel_psia:.2f} psia",
            f"{t_boil:.1f} °C ({c_to_f(t_boil):.0f} °F)",
            f"{t_cond:.1f} °C ({c_to_f(t_cond):.0f} °F)",
            f"{evap_duty:.2f} MMBTU/hr",
            f"{feed_heatup:.2f} MMBTU/hr",
            f"{hx_duty:.2f} MMBTU/hr",
            f"{losses:.2f} MMBTU/hr",
            f"{blower_heat:.2f} MMBTU/hr",
            f"{makeup_steady:,.0f} lb/hr",
            f"{makeup_design:,.0f} lb/hr (× {makeup_factor:g})",
            f"{dist_cond:.0f} µS/cm",
        ],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE - old key names kept, now filled with the Basis-tab values
    ##########################################################################
    pdata.update({
        "distillate_flow": dist_gpm,
        "vapor_flow_lb_hr": dist_lb_hr,
        "heat_duty_mmbtu": hx_duty,
        "crystallizer_heat_duty": hx_duty,
        "latent_duty_mmbtu": evap_duty,
        "feed_heatup_mmbtu": feed_heatup,
        "makeup_steam_lb_hr": makeup_steady,
        "steam_required_lb_hr": makeup_steady,
        "boiling_temp": c_to_f(t_boil),  # °F (old key)
        "steam_chest_temp": c_to_f(t_cond),  # °F (old key)
        "vessel_pressure": float(pdata.get("crystallizer_pressure_bar", vessel_psia * 0.0689476)),
        "mother_liquor_tds": ml_tds,
        "entrainment_ppm": entrainment_ppm,
        "tds_cond_factor": tds_per_us,
        "distillate_cond": dist_cond,
        "water_recovery": dist_lb_hr / max(feed_lb_hr - salt_lb_hr, 1.0) * 100.0,
    })

    save_project(project_name, st.session_state.project_data)
    st.success("✅ Crystallizer Thermal Auto Saved")

##############################################################################
# CRYSTALLIZER VESSEL
#   salt production  = feed x TDS  (Basis tab) - all salt leaves as cake in ZLD
#   slurry inventory = brine inventory (Basis tab)
#   crystal residence time = crystals held in the slurry / salt production
#   vapor head diameter = Souders-Brown vapor velocity at vessel pressure
#   controls follow the Vacom philosophy on the Basis tab
##############################################################################

with tab7:
    from mvr_thermo import c_to_f

    st.header("💎 Crystallizer Vessel (V-200)")
    flow_slot(('V-200',))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM OTHER TABS
    ##########################################################################
    salt_lb_hr = float(pdata.get("salt_lb_hr_calc", 0.0))
    cake_water_lb_hr = float(pdata.get("cake_water_lb_hr", 0.0))
    vapor_lb_hr = float(pdata.get("distillate_lb_hr_calc", pdata.get("vapor_flow_lb_hr", 0.0)))
    vessel_psia = float(pdata.get("vessel_pressure_sp", 7.25))
    atm_psia = float(pdata.get("atmospheric_pressure", 14.696))
    t_boil = float(pdata.get("crystallizer_boiling_temp", 90.0))
    inventory_lb = float(pdata.get("brine_inventory_lb", 135000.0))
    recirc_gpm = float(pdata.get("recirculation_flow", 0.0))
    level_sp = float(pdata.get("vessel_level_sp", 60.0))
    tss_low = float(pdata.get("tss_open", 10.0))
    tss_high = float(pdata.get("tss_high", 15.0))
    pv_psia = float(pdata.get("vessel_pressure_pv", vessel_psia))
    vfd_speed = float(pdata.get("vfd_speed_pct", 0.0))

    st.info(
        f"From Design Basis: dry salt {salt_lb_hr:,.0f} lb/hr | vapor {vapor_lb_hr:,.0f} lb/hr | "
        f"vessel {vessel_psia:.2f} psia ({vessel_psia - atm_psia:+.2f} psig), boiling "
        f"{t_boil:.1f} °C ({c_to_f(t_boil):.0f} °F) | slurry inventory {inventory_lb:,.0f} lb."
    )

    ##########################################################################
    # INPUTS
    ##########################################################################
    st.subheader("Vessel Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        slurry_solids = st.number_input(
            "Slurry Suspended Solids (% wt)", min_value=1.0, max_value=40.0, step=1.0,
            value=float(pdata.get("crystallizer_solids", (tss_low + tss_high) / 2.0)), key="cv_solids",
            help="Seed crystals held in the circulating slurry. 15-25% is typical for a "
                 "seeded gypsum/sodium sulfate crystallizer.")
    with c2:
        liquor_sg = st.number_input(
            "Mother Liquor SG", min_value=1.0, max_value=1.6, step=0.01,
            value=float(pdata.get("mother_liquor_sg", 1.30)), key="cv_liquor_sg")
    with c3:
        crystal_sg = st.number_input(
            "Crystal SG", min_value=1.5, max_value=3.0, step=0.05,
            value=float(pdata.get("crystal_sg", 2.32)), key="cv_crystal_sg",
            help="Gypsum 2.32, sodium sulfate 2.66, sodium chloride 2.16.")
    with c4:
        min_residence = st.number_input(
            "Min. Crystal Residence Time (hr)", min_value=0.5, step=0.5,
            value=float(pdata.get("residence_time", 4.0)), key="cv_min_res",
            help="Needed for crystal growth to filterable size.")

    d1, d2, d3 = st.columns(3)
    with d1:
        k_sb = st.number_input(
            "Souders-Brown K (ft/s)", min_value=0.02, max_value=0.5, step=0.01,
            value=float(pdata.get("vessel_k_factor", 0.12)), key="cv_k",
            help="Vapor-head disengagement factor. 0.10-0.15 ft/s with a mesh demister.")
    with d2:
        vendor_id_ft = st.number_input(
            "Vacom Vessel ID (ft, 0 = unknown)", min_value=0.0, step=0.5,
            value=float(pdata.get("vendor_vessel_id_ft", 0.0)), key="cv_vendor_id")
    with d3:
        vessel_share = st.number_input(
            "Inventory in Vessel (%)", min_value=10.0, max_value=100.0, step=5.0,
            value=float(pdata.get("vessel_inventory_share", 70.0)), key="cv_share",
            help="Rest of the inventory is in the recirculation piping and HX.")

    ##########################################################################
    # SLURRY AND RESIDENCE TIME
    ##########################################################################
    x = slurry_solids / 100.0
    slurry_sg = 1.0 / (x / crystal_sg + (1 - x) / liquor_sg)
    slurry_density = slurry_sg * 62.4  # lb/ft³
    slurry_gal = inventory_lb / (slurry_sg * 8.34)
    vessel_gal = slurry_gal * vessel_share / 100.0
    crystal_inventory_lb = inventory_lb * x
    residence_hr = crystal_inventory_lb / max(salt_lb_hr, 1e-9) if salt_lb_hr > 0 else 0.0
    loop_turnover_min = slurry_gal / max(recirc_gpm, 1e-9) if recirc_gpm > 0 else 0.0

    # Slurry blowdown to the filter at the vessel solids concentration
    blowdown_lb_hr = salt_lb_hr / max(x, 1e-9)
    blowdown_gpm = blowdown_lb_hr / (slurry_sg * 8.34 * 60.0)
    liquor_in_blowdown_lb_hr = blowdown_lb_hr - salt_lb_hr
    liquor_in_blowdown_gpm = liquor_in_blowdown_lb_hr / (liquor_sg * 8.34 * 60.0)

    # Vapor head (Souders-Brown)
    t_boil_r = c_to_f(t_boil) + 459.67
    vapor_ft3_lb = 85.76 * t_boil_r / (vessel_psia * 144.0)  # ideal gas, steam
    rho_v = 1.0 / vapor_ft3_lb
    rho_l = liquor_sg * 62.4
    v_allow = k_sb * ((rho_l - rho_v) / rho_v) ** 0.5
    vapor_cfs = vapor_lb_hr * vapor_ft3_lb / 3600.0
    min_area = vapor_cfs / max(v_allow, 1e-9)
    min_id_ft = (4 * min_area / 3.14159) ** 0.5
    check_id = vendor_id_ft if vendor_id_ft > 0 else min_id_ft
    actual_v = vapor_cfs / (3.14159 * check_id ** 2 / 4) if check_id > 0 else 0.0
    liquid_height_ft = (vessel_gal / 7.4805) / (3.14159 * check_id ** 2 / 4) if check_id > 0 else 0.0

    ##########################################################################
    # RESULTS
    ##########################################################################
    st.subheader("📈 Vessel Performance")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Crystal Production", f"{salt_lb_hr:,.0f} lb/hr", f"{salt_lb_hr * 24 / 2000:.1f} TPD",
              delta_color="off")
    r2.metric("Slurry SG", f"{slurry_sg:.3f}", f"{slurry_density:.1f} lb/ft³", delta_color="off")
    r3.metric("Total Slurry Volume", f"{slurry_gal:,.0f} gal",
              f"vessel {vessel_gal:,.0f} gal", delta_color="off")
    r4.metric("Crystal Inventory", f"{crystal_inventory_lb:,.0f} lb")
    r5.metric("Crystal Residence", f"{residence_hr:.1f} hr",
              f"min {min_residence:.1f} hr", delta_color="off")
    r6.metric("Loop Turnover", f"{loop_turnover_min:.1f} min",
              help="Total slurry volume ÷ recirculation flow.")

    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric("Slurry Blowdown", f"{blowdown_gpm:.1f} GPM", f"{blowdown_lb_hr:,.0f} lb/hr",
              delta_color="off")
    v2.metric("Liquor in Blowdown", f"{liquor_in_blowdown_gpm:.1f} GPM",
              help="Returns as filtrate after the filter.")
    v3.metric("Vapor Volume", f"{vapor_cfs:,.0f} ft³/s", f"{vapor_ft3_lb:.1f} ft³/lb",
              delta_color="off")
    v4.metric("Allowable Vapor Velocity", f"{v_allow:.1f} ft/s")
    v5.metric("Min. Vapor Head ID", f"{min_id_ft:.1f} ft")
    v6.metric("Velocity at " + ("Vacom ID" if vendor_id_ft > 0 else "min. ID"),
              f"{actual_v:.1f} ft/s",
              f"liquid ≈ {liquid_height_ft:.1f} ft deep" if check_id > 0 else None,
              delta_color="off")

    if residence_hr < min_residence:
        st.warning(f"Crystal residence {residence_hr:.1f} hr is below the {min_residence:.1f} hr "
                   f"minimum. Raise the slurry solids or the inventory.")
    if vendor_id_ft > 0 and actual_v > v_allow:
        st.error(f"Vapor velocity {actual_v:.1f} ft/s exceeds the allowable {v_allow:.1f} ft/s "
                 f"at {vendor_id_ft:.1f} ft ID - entrainment risk.")

    ##########################################################################
    # CONTROLS (Vacom philosophy - setpoints from the Basis tab)
    ##########################################################################
    st.subheader("🎛️ Control Check (Vacom philosophy)")
    st.caption("Move the sliders to see how each loop responds. Setpoints come from the Basis tab.")

    p1, p2 = st.columns(2)
    with p1:
        level_pv = st.slider("Vessel Level PV (%)", 0, 100, int(round(level_sp)), key="cv_level_pv")
    with p2:
        tss_pv = st.slider("Slurry TSS PV (% wt)", 0.0, 40.0, float(round(slurry_solids, 1)),
                           step=0.5, key="cv_tss_pv")

    level_err = level_pv - level_sp
    if level_err < -5:
        feed_valve = "OPENING"
    elif level_err > 5:
        feed_valve = "CLOSING"
    else:
        feed_valve = "MODULATING"

    if tss_pv >= tss_high:
        blowdown_valve = "OPEN - to filter"
    elif tss_pv <= tss_low:
        blowdown_valve = "CLOSED - build seed"
    else:
        blowdown_valve = "MODULATING"

    p_err = pv_psia - vessel_psia
    pressure_action = ("Blower speeds UP" if p_err > 0.005 else
                       "Blower slows DOWN" if p_err < -0.005 else "Hold")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("LIC → Feed Valve", feed_valve, f"SP {level_sp:.0f}%", delta_color="off")
    k2.metric("TSS → Blowdown Valve", blowdown_valve,
              f"{tss_low:.0f}-{tss_high:.0f}% band", delta_color="off")
    k3.metric("PIC → Blower VFD", pressure_action,
              f"PV {pv_psia:.2f} / SP {vessel_psia:.2f} psia", delta_color="off")
    k4.metric("VFD Speed", f"{vfd_speed:.0f} %")

    if not (tss_low <= slurry_solids <= tss_high):
        st.warning(f"Design slurry solids {slurry_solids:.1f}% is outside the {tss_low:.0f}-{tss_high:.0f}% TSS "
                   f"control band (Design Basis tab). Set 'Slurry Suspended Solids' between {tss_low:.0f} and "
                   f"{tss_high:.0f}% - the band midpoint is {(tss_low + tss_high) / 2:.1f}%.")
    else:
        st.success(f"Design slurry solids {slurry_solids:.1f}% is inside the {tss_low:.0f}-{tss_high:.0f}% TSS control band.")

    ##########################################################################
    # SUMMARY
    ##########################################################################
    summary_df = pd.DataFrame({
        "Parameter": [
            "Vessel pressure / boiling temp", "Crystal production",
            "Slurry solids / SG", "Total slurry inventory", "Slurry in vessel",
            "Crystal residence time", "Slurry blowdown to filter",
            "Vapor flow / volume", "Allowable vapor velocity", "Min. vapor head ID",
        ],
        "Value": [
            f"{vessel_psia:.2f} psia / {t_boil:.1f} °C",
            f"{salt_lb_hr:,.0f} lb/hr dry",
            f"{slurry_solids:.0f}% / {slurry_sg:.3f}",
            f"{inventory_lb:,.0f} lb ({slurry_gal:,.0f} gal)",
            f"{vessel_gal:,.0f} gal",
            f"{residence_hr:.1f} hr",
            f"{blowdown_gpm:.1f} GPM ({blowdown_lb_hr:,.0f} lb/hr)",
            f"{vapor_lb_hr:,.0f} lb/hr ({vapor_cfs:,.0f} ft³/s)",
            f"{v_allow:.1f} ft/s (K {k_sb:.2f})",
            f"{min_id_ft:.1f} ft",
        ],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE (old key names kept)
    ##########################################################################
    pdata.update({
        "crystallizer_solids": slurry_solids,
        "mother_liquor_sg": liquor_sg,
        "crystal_sg": crystal_sg,
        "residence_time": min_residence,
        "vessel_k_factor": k_sb,
        "vendor_vessel_id_ft": vendor_id_ft,
        "vessel_inventory_share": vessel_share,
        "salt_lb_hr": salt_lb_hr,
        "crystal_lb_hr": salt_lb_hr,
        "slurry_sg": slurry_sg,
        "slurry_density": slurry_density,
        "crystallizer_volume": vessel_gal,
        "slurry_volume_gal": slurry_gal,
        "crystal_inventory_lb": crystal_inventory_lb,
        "crystal_residence_hr": residence_hr,
        "slurry_tss": tss_pv,
        "slurry_blowdown_gpm": blowdown_gpm,
        "slurry_blowdown_lb_hr": blowdown_lb_hr,
        "mother_liquor_flow": liquor_in_blowdown_gpm,
        "mother_liquor_lb_hr": liquor_in_blowdown_lb_hr,
        "concentrate_flow": blowdown_gpm,
        "vessel_min_id_ft": min_id_ft,
        "vessel_vapor_velocity": actual_v,
        "vessel_allow_velocity": v_allow,
        "feed_valve": feed_valve,
        "blowdown_valve": blowdown_valve,
        "crystallizer_temp": c_to_f(t_boil),
    })

    save_project(project_name, st.session_state.project_data)
    st.success("✅ Crystallizer Vessel Auto Saved")

##############################################################################
# MVR BLOWER SYSTEM
# Blower power from compression of the crystallizer vapor:
#   suction  = vessel pressure (Design Basis), vapor at boiling temp
#   discharge = blower discharge pressure (Design Basis)
# Plus a Vacom vendor-data check: what feed flow their blower motor supports.
##############################################################################

with tab8:
    from mvr_thermo import hfg_btu_lb, tsat_c_from_psia
    import math

    st.header("🌪️ MVR Blower System")
    flow_slot(('B-310',))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM DESIGN BASIS
    ##########################################################################
    feed_gpm = float(pdata.get("design_flow", 60.0))
    vapor_flow_lb_hr = float(pdata.get("distillate_lb_hr_calc",
                                       pdata.get("vapor_flow_lb_hr", 29100.0)))
    distillate_gpm = float(pdata.get("distillate_gpm_calc", vapor_flow_lb_hr / 500.0))
    vessel_psia = float(pdata.get("vessel_pressure_sp", 7.25))
    atm_psia = float(pdata.get("atmospheric_pressure", 14.696))
    discharge_psia = float(pdata.get("blower_discharge_psia",
                                     pdata.get("blower_discharge_psig", 3.0) + atm_psia))
    bpe_c = float(pdata.get("boiling_point_elevation", 6.0))
    blower_efficiency = float(pdata.get("blower_efficiency", 75.0))
    motor_efficiency = float(pdata.get("motor_efficiency", 95.0))
    boiler_psig = float(pdata.get("boiler_pressure", 30.0))
    flash_delta_t = float(pdata.get("flash_delta_t", 10.8))

    d = mvr_state(
        vessel_psia=vessel_psia,
        discharge_psia=discharge_psia,
        bpe_c=bpe_c,
        blower_eff=blower_efficiency / 100.0,
        motor_eff=motor_efficiency / 100.0,
        vapor_lb_hr=vapor_flow_lb_hr,
    )

    compression_ratio = d["pressure_ratio"]
    suction_temp = d["t_boil"]
    discharge_temp = d["t_disch_actual"]
    shaft_kw = d["shaft_kw"]
    blower_kw = d["elec_kw"]
    shaft_hp = shaft_kw / 0.746

    st.info(
        f"From Design Basis: {vapor_flow_lb_hr:,.0f} lb/hr vapor ({feed_gpm:.0f} GPM feed) | "
        f"suction {vessel_psia:.2f} psia at {suction_temp:.1f} °C → discharge "
        f"{discharge_psia - atm_psia:.2f} psig ({discharge_psia:.2f} psia) | pressure ratio "
        f"{compression_ratio:.2f} | blower η {blower_efficiency:.0f}%, motor η "
        f"{motor_efficiency:.0f}%. Change these on the Basis tab."
    )

    ##########################################################################
    # ECONOMIC INPUTS
    ##########################################################################
    st.subheader("Operating & Economic Inputs")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        operating_hours = st.number_input(
            "Operating Hours/Year", min_value=0.0, max_value=8760.0,
            value=float(pdata.get("mvr_operating_hours", pdata.get("operating_hours", 8000))),
            key="mvr_operating_hours")
    with c2:
        electricity_cost = st.number_input(
            "Electricity Cost ($/kWh)", min_value=0.0,
            value=float(pdata.get("electricity_cost", 0.10)), step=0.01,
            key="mvr_electricity_cost")
    with c3:
        steam_cost = st.number_input(
            "Steam Cost ($/1000 lb)", min_value=0.0,
            value=float(pdata.get("steam_cost", 12.0)), key="mvr_steam_cost",
            help="Cost of steam a conventional steam-driven crystallizer would use.")
    with c4:
        grid_co2 = st.number_input(
            "Grid CO₂ Factor (kg/kWh)", min_value=0.0,
            value=float(pdata.get("grid_co2_factor", 0.40)), step=0.05,
            key="mvr_grid_co2")

    ##########################################################################
    # MVR PERFORMANCE
    ##########################################################################
    hfg_vessel = hfg_btu_lb(d["t_sat_vessel"])
    hfg_boiler = hfg_btu_lb(tsat_c_from_psia(boiler_psig + atm_psia))

    latent_recycled_kw = vapor_flow_lb_hr * hfg_vessel / 3412.14  # heat reused by MVR
    cop = latent_recycled_kw / max(blower_kw, 1.0)

    # Steam a single-effect steam-heated crystallizer would need for the same duty
    steam_saved_lb_hr = vapor_flow_lb_hr * hfg_vessel / hfg_boiler
    heat_recovery_percent = latent_recycled_kw / max(latent_recycled_kw + blower_kw, 1e-9) * 100

    distillate_m3_hr = distillate_gpm * 0.2271
    specific_energy = blower_kw / max(distillate_m3_hr, 0.001)  # kWh/m³
    specific_energy_kgal = blower_kw / max(distillate_gpm * 60 / 1000, 0.001)

    annual_energy = blower_kw * operating_hours
    annual_energy_cost = annual_energy * electricity_cost
    annual_steam_value = steam_saved_lb_hr * operating_hours * steam_cost / 1000
    annual_net_savings = annual_steam_value - annual_energy_cost

    # CO2: natural-gas boiler (≈1,400 BTU fuel per lb steam, 53.06 kg CO2/MMBTU) vs grid power
    co2_steam_t = steam_saved_lb_hr * 1400 / 1e6 * 53.06 * operating_hours / 1000
    co2_power_t = annual_energy * grid_co2 / 1000
    co2_reduction = co2_steam_t - co2_power_t

    heat_duty_kw = float(pdata.get("cryst_hx_duty", latent_recycled_kw / 293.07)) * 293.07

    st.subheader("MVR Performance")

    r1, r2, r3, r4, r5, r6 = st.columns(6)
    with r1:
        st.metric("Vapor Flow", f"{vapor_flow_lb_hr:,.0f} lb/hr")
    with r2:
        st.metric("Pressure Ratio", f"{compression_ratio:.2f}")
    with r3:
        st.metric("Shaft Power", f"{shaft_kw:,.0f} kW", f"{shaft_hp:,.0f} hp", delta_color="off")
    with r4:
        st.metric("Electrical Power", f"{blower_kw:,.0f} kW")
    with r5:
        st.metric("Suction Temp", f"{suction_temp:.1f} °C")
    with r6:
        st.metric("Discharge Temp (dry)", f"{discharge_temp:.0f} °C",
                  help="Superheated, before desuperheating.")

    r7, r8, r9, r10, r11, r12 = st.columns(6)
    with r7:
        st.metric("Condensing Temp", f"{d['t_sat_disch']:.1f} °C")
    with r8:
        st.metric("Sat. Temp Lift", f"{d['sat_temp_lift']:.1f} °C")
    with r9:
        st.metric("COP", f"{cop:.1f}", help="Latent heat recycled ÷ blower electrical power.")
    with r10:
        st.metric("Specific Energy", f"{specific_energy:.1f} kWh/m³",
                  f"{specific_energy_kgal:.0f} kWh/kgal", delta_color="off")
    with r11:
        st.metric("Equivalent Steam Saved", f"{steam_saved_lb_hr:,.0f} lb/hr")
    with r12:
        st.metric("Heat Recovery", f"{heat_recovery_percent:.0f} %")

    ##########################################################################
    # ECONOMICS
    ##########################################################################
    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.metric("Annual Energy Cost", f"${annual_energy_cost:,.0f}")
    with e2:
        st.metric("Steam Value Avoided", f"${annual_steam_value:,.0f}")
    with e3:
        st.metric("Net Savings", f"${annual_net_savings:,.0f}")
    with e4:
        st.metric("Net CO₂ Reduction", f"{co2_reduction:,.0f} t/yr",
                  help="Gas-boiler CO₂ avoided minus grid CO₂ of blower power.")

    st.divider()

    ##########################################################################
    # VACOM VENDOR DATA
    ##########################################################################
    st.subheader("🏭 Vacom Blower Vendor Data")
    st.caption(
        "Enter Vacom's blower selection. Design vapor flow and absorbed power are optional - "
        "if entered, the app back-calculates Vacom's blower efficiency."
    )

    STD_MOTORS_HP = [100, 125, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800,
                     900, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500, 4000]


    def _std_motor(hp):
        return next((m for m in STD_MOTORS_HP if m >= hp), math.ceil(hp / 500) * 500)


    v1, v2, v3, v4 = st.columns(4)
    with v1:
        vendor_motor_hp = st.number_input(
            "Vacom Blower Motor (hp)", min_value=0.0,
            value=float(pdata.get("vendor_blower_hp", 900.0)), step=50.0, key="vb_motor_hp")
    with v2:
        max_load = st.number_input(
            "Max Continuous Motor Load (%)", min_value=50.0, max_value=115.0,
            value=float(pdata.get("vendor_blower_max_load", 90.0)), key="vb_max_load",
            help="Shaft power should stay below this % of nameplate in normal operation.")
    with v3:
        vendor_vapor = st.number_input(
            "Vacom Design Vapor Flow (lb/hr)", min_value=0.0,
            value=float(pdata.get("vendor_blower_vapor", 0.0)), step=500.0, key="vb_vapor",
            help="0 = unknown.")
    with v4:
        vendor_kw = st.number_input(
            "Vacom Absorbed Shaft Power (kW)", min_value=0.0,
            value=float(pdata.get("vendor_blower_kw", 0.0)), step=10.0, key="vb_kw",
            help="0 = unknown.")

    vendor_motor_kw = vendor_motor_hp * 0.746
    kw_per_lb = shaft_kw / max(vapor_flow_lb_hr, 1.0)  # app shaft kW per lb/hr
    lb_per_feed_gpm = vapor_flow_lb_hr / max(feed_gpm, 1.0)

    max_vapor = vendor_motor_kw * max_load / 100.0 / max(kw_per_lb, 1e-9)
    max_feed_gpm = max_vapor / max(lb_per_feed_gpm, 1e-9)
    max_vapor_100 = vendor_motor_kw / max(kw_per_lb, 1e-9)
    max_feed_100 = max_vapor_100 / max(lb_per_feed_gpm, 1e-9)

    current_load = shaft_kw / max(vendor_motor_kw, 1e-9) * 100.0
    app_motor_hp = _std_motor(shaft_hp * 1.10)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Load at Current Case", f"{current_load:.0f} %",
                  f"{feed_gpm:.0f} GPM feed", delta_color="off")
    with k2:
        st.metric(f"Max Feed at {max_load:.0f}% Load", f"{max_feed_gpm:,.1f} GPM",
                  f"{max_vapor:,.0f} lb/hr vapor", delta_color="off")
    with k3:
        st.metric("Max Feed at 100% Load", f"{max_feed_100:,.1f} GPM",
                  f"{max_vapor_100:,.0f} lb/hr vapor", delta_color="off")
    with k4:
        st.metric("App Motor for Current Case", f"{app_motor_hp:g} hp",
                  f"{app_motor_hp - vendor_motor_hp:+g} vs Vacom", delta_color="off")

    if current_load > 100:
        st.error(
            f"At {feed_gpm:.0f} GPM feed the blower needs {shaft_hp:,.0f} shaft hp - "
            f"{current_load:.0f}% of Vacom's {vendor_motor_hp:g} hp motor. Vacom's selection "
            f"supports about {max_feed_gpm:.0f} GPM at {max_load:.0f}% load on this basis "
            f"(pressure ratio {compression_ratio:.2f}, η {blower_efficiency:.0f}%)."
        )
    elif current_load > max_load:
        st.warning(f"Motor load {current_load:.0f}% is above the {max_load:.0f}% continuous limit.")
    else:
        st.success(f"Vacom's {vendor_motor_hp:g} hp motor covers {feed_gpm:.0f} GPM feed "
                   f"at {current_load:.0f}% load.")

    # Capacity cases
    case_rows = []
    for label, gpm in [("Average Design", 36.0), ("Maximum Design", 44.0),
                       ("Maximum Capacity", 60.0), ("Current case", feed_gpm)]:
        vap = lb_per_feed_gpm * gpm
        skw = kw_per_lb * vap
        load = skw / max(vendor_motor_kw, 1e-9) * 100.0
        case_rows.append({
            "Case": label,
            "Feed (GPM)": round(gpm, 1),
            "Vapor (lb/hr)": round(vap),
            "Shaft (kW)": round(skw),
            "Shaft (hp)": round(skw / 0.746),
            f"Load on {vendor_motor_hp:g} hp (%)": round(load),
            "Status": ("✓ OK" if load <= max_load else
                       ("⚠ Above continuous limit" if load <= 100 else "✗ Exceeds motor")),
        })
    st.dataframe(pd.DataFrame(case_rows), width="stretch", hide_index=True)

    # Back-calculated Vacom efficiency
    if vendor_vapor > 0 and vendor_kw > 0:
        isen_kw_per_lb = kw_per_lb * blower_efficiency / 100.0  # isentropic kW per lb/hr
        vendor_eff = isen_kw_per_lb * vendor_vapor / vendor_kw * 100.0
        vendor_load = vendor_kw / max(vendor_motor_kw, 1e-9) * 100.0
        vendor_feed = vendor_vapor / max(lb_per_feed_gpm, 1e-9)
        b1, b2, b3 = st.columns(3)
        with b1:
            st.metric("Vacom Design Point ≈ Feed", f"{vendor_feed:,.1f} GPM")
        with b2:
            st.metric("Implied Vacom Efficiency", f"{vendor_eff:.0f} %",
                      f"{vendor_eff - blower_efficiency:+.0f} vs app", delta_color="off")
        with b3:
            st.metric("Vacom Motor Load", f"{vendor_load:.0f} %")
        if vendor_eff > 92 or vendor_eff < 50:
            st.warning("Implied efficiency is unrealistic - check that Vacom's pressures "
                       "match the Basis tab (vessel 7.25 psia, discharge pressure).")
    else:
        vendor_eff = 0.0
        st.caption("Enter Vacom's design vapor flow and absorbed power to back-calculate "
                   "their blower efficiency.")

    ##########################################################################
    # SUMMARY TABLE
    ##########################################################################
    st.subheader("📋 MVR Summary")

    summary_df = pd.DataFrame({
        "Parameter": [
            "Vapor Flow",
            "Suction / Discharge Pressure",
            "Pressure Ratio",
            "Suction Temp / Condensing Temp",
            "Discharge Temp (dry)",
            "Shaft Power",
            "Electrical Power",
            "COP",
            "Specific Energy",
            "Equivalent Steam Saved",
            "Annual Net Savings",
            "Vacom Motor / Load at Current Case",
            f"Max Feed on Vacom Motor ({max_load:.0f}% load)",
        ],
        "Value": [
            f"{vapor_flow_lb_hr:,.0f} lb/hr",
            f"{vessel_psia:.2f} psia / {discharge_psia:.2f} psia",
            f"{compression_ratio:.2f}",
            f"{suction_temp:.1f} °C / {d['t_sat_disch']:.1f} °C",
            f"{discharge_temp:.0f} °C",
            f"{shaft_kw:,.0f} kW ({shaft_hp:,.0f} hp)",
            f"{blower_kw:,.0f} kW",
            f"{cop:.1f}",
            f"{specific_energy:.1f} kWh/m³",
            f"{steam_saved_lb_hr:,.0f} lb/hr",
            f"${annual_net_savings:,.0f}",
            f"{vendor_motor_hp:g} hp / {current_load:.0f} %",
            f"{max_feed_gpm:,.1f} GPM",
        ],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE (existing keys kept for other tabs and the PDF report)
    ##########################################################################
    pdata.update({
        "compression_ratio": compression_ratio,
        "blower_kw": blower_kw,
        "mvr_kw": blower_kw,
        "blower_shaft_kw": shaft_kw,
        "steam_saved_lb_hr": steam_saved_lb_hr,
        "heat_recovery_percent": heat_recovery_percent,
        "specific_energy": specific_energy,
        "annual_mvr_cost": annual_energy_cost,
        "annual_net_savings": annual_net_savings,
        "mvr_suction_temp": suction_temp,
        "mvr_discharge_temp": discharge_temp,
        "mvr_cop": cop,
        "heat_duty_kw": heat_duty_kw,
        "mvr_operating_hours": operating_hours,
        "electricity_cost": electricity_cost,
        "steam_cost": steam_cost,
        "grid_co2_factor": grid_co2,
        "mvr_co2_reduction_t": co2_reduction,
        "vendor_blower_hp": vendor_motor_hp,
        "vendor_blower_max_load": max_load,
        "vendor_blower_vapor": vendor_vapor,
        "vendor_blower_kw": vendor_kw,
        "vendor_blower_load_pct": current_load,
        "vendor_blower_max_feed_gpm": max_feed_gpm,
        "vendor_blower_eff_calc": vendor_eff,
    })

    save_project(project_name, st.session_state.project_data)

    st.success("✅ MVR Blower System Auto Saved")

##############################################################################
# BOILER MAKEUP STEAM SYSTEM
# Boiler size comes from the Basis tab calculation (IAPWS steam properties):
#   capacity = max(design makeup, start-up heat-up, blower priming)
# Per the Vacom P&ID the boiler steam heats E-220 (shell B1) and its
# condensate returns B4 -> V-510 -> P-520 -> boiler, so it is a closed loop
# and is NOT part of the process mass balance.
##############################################################################

with tab9:
    from mvr_thermo import tsat_c_from_psia, hfg_btu_lb, KJKG_TO_BTULB, c_to_f

    st.header("♨️ Boiler Makeup Steam System")
    flow_slot(('Boiler',))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM DESIGN BASIS
    ##########################################################################
    atm_psia = float(pdata.get("atmospheric_pressure", 14.696))
    boiler_psig = float(pdata.get("boiler_pressure", 30.0))
    boiler_eff = float(pdata.get("boiler_eff_pct", 99.0))
    feedwater_c = float(pdata.get("boiler_feedwater_c", 90.0))
    required_lb_hr = float(pdata.get("boiler_capacity_calc", 0.0))
    governing = str(pdata.get("boiler_governing_case", "-"))
    makeup_design = float(pdata.get("boiler_makeup_design_lb_hr", 0.0))
    makeup_factor = float(pdata.get("makeup_factor", 2.5))
    makeup_steady = makeup_design / makeup_factor if makeup_factor > 0 else makeup_design
    startup_lb_hr = float(pdata.get("boiler_startup_lb_hr", 0.0))
    heatup_hr = float(pdata.get("heatup_hr", 4.0))
    vendor_capacity = float(pdata.get("boiler_capacity", 0.0))
    evap_lb_hr = float(pdata.get("distillate_lb_hr_calc", pdata.get("vapor_flow_lb_hr", 0.0)))
    losses = float(pdata.get("heat_losses_mmbtu_hr", 0.0))
    blower_heat = float(pdata.get("blower_heat_mmbtu_hr", 0.0))

    t_sat_boiler = tsat_c_from_psia(boiler_psig + atm_psia)
    hfg = hfg_btu_lb(t_sat_boiler)
    sensible = 4.186 * (t_sat_boiler - feedwater_c) * KJKG_TO_BTULB
    rise = hfg + sensible  # BTU per lb steam generated
    kw_per_lb = rise / 3412.14 / max(boiler_eff / 100.0, 0.01)

    st.info(
        f"From Design Basis: {boiler_psig:.0f} psig steam (sat. {t_sat_boiler:.1f} °C / "
        f"{c_to_f(t_sat_boiler):.0f} °F) | feedwater {feedwater_c:.0f} °C | efficiency "
        f"{boiler_eff:.0f}% | required {required_lb_hr:,.0f} lb/hr, governed by "
        f"**{governing}**. Change these on the Basis tab."
    )

    ##########################################################################
    # INPUTS
    ##########################################################################
    st.subheader("Boiler Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        boiler_type = st.selectbox(
            "Boiler Type", ["Electric Boiler", "Natural Gas Boiler"],
            index=0 if pdata.get("boiler_type", "Electric Boiler") == "Electric Boiler" else 1,
            key="bl_type")
    with c2:
        condensate_return = st.number_input(
            "Condensate Return (%)", min_value=0.0, max_value=100.0, step=1.0,
            value=float(pdata.get("condensate_return_percent", 98.0)), key="bl_cond_return",
            help="Closed loop via V-510 / P-520. Losses are flash and trap leakage.")
    with c3:
        size_step = st.number_input(
            "Round Boiler Size Up To (kW)", min_value=10.0, step=10.0,
            value=float(pdata.get("boiler_size_step_kw", 100.0)), key="bl_size_step")
    with c4:
        makeup_run_pct = st.number_input(
            "Boiler Running in Normal Operation (% of time)", min_value=0.0, max_value=100.0,
            step=5.0, value=float(pdata.get("boiler_run_pct", 100.0)), key="bl_run_pct",
            help="100% if the boiler supplies steady makeup steam continuously.")

    e1, e2, e3 = st.columns(3)
    with e1:
        operating_hours = st.number_input(
            "Operating Hours/Year", min_value=0.0, max_value=8760.0, step=100.0,
            value=float(pdata.get("operating_hours", 8000)), key="bl_operating_hours")
    with e2:
        electricity_cost = st.number_input(
            "Electricity Cost ($/kWh)", min_value=0.0, step=0.01,
            value=float(pdata.get("electricity_cost", 0.10)), key="bl_electricity_cost")
    with e3:
        gas_cost = st.number_input(
            "Natural Gas Cost ($/MMBTU)", min_value=0.0, step=0.5,
            value=float(pdata.get("gas_cost", 4.0)), key="bl_gas_cost",
            disabled=boiler_type == "Electric Boiler")

    ##########################################################################
    # CALCULATIONS
    ##########################################################################
    installed_kw = required_lb_hr * kw_per_lb
    selected_kw = math.ceil(installed_kw / size_step) * size_step if installed_kw > 0 else 0.0
    selected_lb_hr = selected_kw / kw_per_lb if kw_per_lb > 0 else 0.0
    steady_kw = makeup_steady * kw_per_lb
    startup_kw = startup_lb_hr * kw_per_lb
    startup_mwh = startup_kw * heatup_hr / 1000.0

    condensate_lb_hr = makeup_steady * condensate_return / 100.0
    makeup_water_lb_hr = makeup_steady - condensate_lb_hr

    run_hours = operating_hours * makeup_run_pct / 100.0
    if boiler_type == "Electric Boiler":
        annual_energy = steady_kw * run_hours  # kWh
        annual_cost = annual_energy * electricity_cost
        energy_text = f"{annual_energy / 1000:,.0f} MWh/yr"
    else:
        annual_energy = makeup_steady * rise / max(boiler_eff / 100, 0.01) / 1e6 * run_hours
        annual_cost = annual_energy * gas_cost  # MMBTU
        energy_text = f"{annual_energy:,.0f} MMBTU/yr"

    mvr_share = 100.0 - makeup_steady / max(evap_lb_hr, 1.0) * 100.0

    ##########################################################################
    # RESULTS
    ##########################################################################
    st.subheader("Boiler Sizing")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Required Capacity", f"{required_lb_hr:,.0f} lb/hr", governing, delta_color="off")
    r2.metric("Installed Power", f"{installed_kw:,.0f} kW")
    r3.metric("Selected Boiler", f"{selected_kw:,.0f} kW", f"{selected_lb_hr:,.0f} lb/hr",
              delta_color="off")
    r4.metric("Start-up Steam", f"{startup_lb_hr:,.0f} lb/hr",
              f"{heatup_hr:g} hr heat-up, {startup_mwh:.1f} MWh", delta_color="off")
    r5.metric("Design Makeup", f"{makeup_design:,.0f} lb/hr",
              f"steady {makeup_steady:,.0f} × {makeup_factor:g}", delta_color="off")

    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Steady Boiler Load", f"{steady_kw:,.0f} kW")
    b2.metric("Condensate Return", f"{condensate_lb_hr:,.0f} lb/hr")
    b3.metric("Boiler Makeup Water", f"{makeup_water_lb_hr:,.0f} lb/hr")
    b4.metric("Annual Energy", energy_text)
    b5.metric("Annual Cost", f"${annual_cost:,.0f}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Heat Losses", f"{losses:.2f} MMBTU/hr")
    m2.metric("Blower Heat Input", f"{blower_heat:.2f} MMBTU/hr")
    m3.metric("Evaporation Supplied by MVR", f"{mvr_share:.1f} %",
              help="1 - steady makeup steam ÷ evaporation.")

    if vendor_capacity > 0:
        if required_lb_hr > vendor_capacity:
            st.warning(f"Required {required_lb_hr:,.0f} lb/hr exceeds the target/vendor boiler "
                       f"capacity {vendor_capacity:,.0f} lb/hr on the Basis tab.")
        else:
            st.success(f"Target/vendor boiler {vendor_capacity:,.0f} lb/hr covers the required "
                       f"{required_lb_hr:,.0f} lb/hr ({required_lb_hr / vendor_capacity * 100:.0f}%).")

    summary_df = pd.DataFrame({
        "Parameter": [
            "Boiler type", "Steam pressure", "Governing case", "Required capacity",
            "Start-up heat-up steam", "Design makeup steam", "Steady makeup steam",
            "Energy per lb steam", "Installed power", "Selected boiler",
            "Condensate return", "Boiler makeup water", "Annual energy", "Annual cost",
        ],
        "Value": [
            boiler_type,
            f"{boiler_psig:.0f} psig ({t_sat_boiler:.1f} °C)",
            governing,
            f"{required_lb_hr:,.0f} lb/hr",
            f"{startup_lb_hr:,.0f} lb/hr",
            f"{makeup_design:,.0f} lb/hr",
            f"{makeup_steady:,.0f} lb/hr",
            f"{rise:,.0f} BTU/lb ({kw_per_lb * 1000:.0f} W per lb/hr)",
            f"{installed_kw:,.0f} kW",
            f"{selected_kw:,.0f} kW",
            f"{condensate_lb_hr:,.0f} lb/hr ({condensate_return:.0f}%)",
            f"{makeup_water_lb_hr:,.0f} lb/hr",
            energy_text,
            f"${annual_cost:,.0f}",
        ],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE (old key names kept)
    ##########################################################################
    pdata.update({
        "boiler_type": boiler_type,
        "condensate_return_percent": condensate_return,
        "boiler_size_step_kw": size_step,
        "boiler_run_pct": makeup_run_pct,
        "operating_hours": operating_hours,
        "electricity_cost": electricity_cost,
        "gas_cost": gas_cost,
        "steam_rate": makeup_steady,
        "boiler_capacity_lb_hr": required_lb_hr,
        "boiler_kw": installed_kw,
        "boiler_selected_kw": selected_kw,
        "boiler_steady_kw": steady_kw,
        "boiler_mw": installed_kw / 1000.0,
        "boiler_annual_cost": annual_cost,
        "condensate_return_lb_hr": condensate_lb_hr,
        "boiler_makeup_water_lb_hr": makeup_water_lb_hr,
        "steam_pressure": boiler_psig,
        "mvr_steam_offset_lb_hr": evap_lb_hr - makeup_steady,
        "steam_offset_percent": mvr_share,
    })

    save_project(project_name, st.session_state.project_data)
    st.success("✅ Boiler Makeup Steam System Auto Saved")

##############################################################################
# OBERLIN FILTER & FILTRATE RECYCLE
#   dry solids = salt from the Basis tab (feed x TDS)
#   cake       = dry solids + cake moisture (Basis tab)
#   filtrate   = slurry blowdown (Vessel tab) - cake  -> back to the crystallizer
#   filter area is sized on DRY SOLIDS loading (lb dry solids / hr / ft²)
##############################################################################

with tab10:
    st.header("🧂 Oberlin Filter & Filtrate Recycle")
    flow_slot(('Oberlin filter', 'Salt cake'))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM OTHER TABS
    ##########################################################################
    dry_solids = float(pdata.get("salt_lb_hr_calc", pdata.get("crystal_lb_hr", 0.0)))
    cake_moisture = float(pdata.get("cake_moisture_pct", 20.0))
    cake_water = float(pdata.get("cake_water_lb_hr",
                                 dry_solids * cake_moisture / max(100 - cake_moisture, 1)))
    slurry_lb_hr = float(pdata.get("slurry_blowdown_lb_hr", 0.0))
    slurry_gpm = float(pdata.get("slurry_blowdown_gpm", 0.0))
    liquor_sg = float(pdata.get("mother_liquor_sg", 1.30))
    slurry_solids = float(pdata.get("crystallizer_solids", 20.0))

    st.info(
        f"From Basis and Vessel tabs: dry salt {dry_solids:,.0f} lb/hr | cake moisture "
        f"{cake_moisture:.0f}% | slurry blowdown {slurry_gpm:.1f} GPM "
        f"({slurry_lb_hr:,.0f} lb/hr at {slurry_solids:.0f}% solids)."
    )

    ##########################################################################
    # INPUTS
    ##########################################################################
    st.subheader("Filter Inputs")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        filter_loading = st.number_input(
            "Filter Loading (lb dry solids/hr·ft²)", min_value=0.5, step=1.0,
            value=float(pdata.get("filter_loading_dry", 10.0)), key="of_loading",
            help="Get the design value from Oberlin's filtration test. 5-20 is a typical "
                 "range for gypsum/salt cake on a pressure filter.")
    with c2:
        area_per_unit = st.number_input(
            "Area per Filter (ft²)", min_value=1.0, step=10.0,
            value=float(pdata.get("filter_area_per_unit", 100.0)), key="of_area_unit")
    with c3:
        spare_units = st.number_input(
            "Spare Filters", min_value=0, max_value=4, step=1,
            value=int(pdata.get("filter_spares", 1)), key="of_spares")
    with c4:
        cycle_time = st.number_input(
            "Cycle Time (min)", min_value=1.0, step=1.0,
            value=float(pdata.get("cycle_time", 15.0)), key="of_cycle")

    d1, d2, d3 = st.columns(3)
    with d1:
        cake_bulk_density = st.number_input(
            "Cake Bulk Density (lb/ft³)", min_value=20.0, max_value=150.0, step=5.0,
            value=float(pdata.get("cake_bulk_density", 75.0)), key="of_bulk")
    with d2:
        container_yd3 = st.number_input(
            "Roll-off Container (yd³)", min_value=1.0, step=5.0,
            value=float(pdata.get("container_yd3", 20.0)), key="of_container")
    with d3:
        operating_hours = float(pdata.get("operating_hours", 8000))
        st.metric("Operating Hours/Year", f"{operating_hours:,.0f}",
                  help="Set on the Boiler tab.")

    ##########################################################################
    # CALCULATIONS
    ##########################################################################
    cake_lb_hr = dry_solids + cake_water
    cake_solids_pct = dry_solids / max(cake_lb_hr, 1e-9) * 100.0
    if slurry_lb_hr <= 0:
        slurry_lb_hr = dry_solids / max(slurry_solids / 100.0, 0.01)
    filtrate_lb_hr = max(slurry_lb_hr - cake_lb_hr, 0.0)
    filtrate_gpm = filtrate_lb_hr / (liquor_sg * 8.34 * 60.0)
    recycle_pct = filtrate_lb_hr / max(slurry_lb_hr, 1e-9) * 100.0

    required_area = dry_solids / filter_loading
    duty_units = max(1, math.ceil(required_area / area_per_unit))
    total_units = duty_units + int(spare_units)
    installed_area = total_units * area_per_unit

    cycles_per_hr = 60.0 / cycle_time
    cake_per_cycle_lb = cake_lb_hr / cycles_per_hr / duty_units
    cake_per_cycle_ft3 = cake_per_cycle_lb / cake_bulk_density

    cake_tpd = cake_lb_hr * 24 / 2000.0
    annual_cake_ton = cake_lb_hr * operating_hours / 2000.0
    cake_yd3_day = cake_lb_hr * 24 / cake_bulk_density / 27.0
    containers_day = cake_yd3_day / container_yd3

    ##########################################################################
    # RESULTS
    ##########################################################################
    st.subheader("Filter Performance")
    r1, r2, r3, r4, r5, r6 = st.columns(6)
    r1.metric("Slurry Feed", f"{slurry_lb_hr:,.0f} lb/hr", f"{slurry_gpm:.1f} GPM", delta_color="off")
    r2.metric("Dry Solids", f"{dry_solids:,.0f} lb/hr")
    r3.metric("Wet Cake", f"{cake_lb_hr:,.0f} lb/hr", f"{cake_solids_pct:.0f}% solids",
              delta_color="off")
    r4.metric("Required Area", f"{required_area:,.0f} ft²")
    r5.metric("Filters", f"{duty_units} + {int(spare_units)} spare",
              f"{installed_area:,.0f} ft² installed", delta_color="off")
    r6.metric("Cake per Cycle", f"{cake_per_cycle_lb:,.0f} lb", f"{cake_per_cycle_ft3:.1f} ft³ / filter",
              delta_color="off")

    st.subheader("♻️ Filtrate Recycle & Cake Disposal")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Filtrate Return", f"{filtrate_gpm:.1f} GPM", f"{filtrate_lb_hr:,.0f} lb/hr",
              delta_color="off")
    f2.metric("Recycle", f"{recycle_pct:.0f} % of slurry")
    f3.metric("Cake", f"{cake_tpd:.1f} TPD")
    f4.metric("Annual Cake", f"{annual_cake_ton:,.0f} ton/yr")
    f5.metric("Roll-offs", f"{containers_day:.1f} /day", f"{cake_yd3_day:.0f} yd³/day",
              delta_color="off")

    summary_df = pd.DataFrame({
        "Parameter": [
            "Slurry feed", "Dry solids", "Cake moisture", "Wet cake",
            "Filtrate return to crystallizer", "Filter loading (dry solids)",
            "Required area", "Filters (duty + spare)", "Cycle time / cake per cycle",
            "Cake production", "Cake volume",
        ],
        "Value": [
            f"{slurry_lb_hr:,.0f} lb/hr ({slurry_gpm:.1f} GPM)",
            f"{dry_solids:,.0f} lb/hr",
            f"{cake_water:,.0f} lb/hr ({cake_moisture:.0f}%)",
            f"{cake_lb_hr:,.0f} lb/hr",
            f"{filtrate_lb_hr:,.0f} lb/hr ({filtrate_gpm:.1f} GPM)",
            f"{filter_loading:g} lb/hr·ft²",
            f"{required_area:,.0f} ft²",
            f"{duty_units} + {int(spare_units)} × {area_per_unit:,.0f} ft²",
            f"{cycle_time:g} min / {cake_per_cycle_lb:,.0f} lb per filter",
            f"{cake_tpd:.1f} TPD ({annual_cake_ton:,.0f} ton/yr)",
            f"{cake_yd3_day:.0f} yd³/day ({containers_day:.1f} × {container_yd3:g} yd³)",
        ],
    })
    st.dataframe(summary_df, width="stretch", hide_index=True)

    ##########################################################################
    # SAVE (old key names kept)
    ##########################################################################
    pdata.update({
        "filter_loading_dry": filter_loading,
        "filter_area_per_unit": area_per_unit,
        "filter_spares": int(spare_units),
        "cycle_time": cycle_time,
        "cake_bulk_density": cake_bulk_density,
        "container_yd3": container_yd3,
        "cake_solids": cake_solids_pct,
        "filter_feed_lb_hr": slurry_lb_hr,
        "required_filter_area": required_area,
        "number_of_filters": total_units,
        "filters_duty": duty_units,
        "cake_production_lb_hr": cake_lb_hr,
        "filtrate_return_lb_hr": filtrate_lb_hr,
        "filtrate_return_gpm": filtrate_gpm,
        "recycle_percent": recycle_pct,
        "cake_tpd": cake_tpd,
        "annual_cake_ton": annual_cake_ton,
    })

    save_project(project_name, st.session_state.project_data)
    st.success("✅ Oberlin Filter Auto Saved")

##############################################################################
# OVERALL MASS BALANCE
# Process boundary, lb/hr:
#   IN  = feed (water + dissolved salt)
#   OUT = distillate + NCG vent loss + wet cake
# Internal streams (recirculation, slurry blowdown, filtrate) and the closed
# boiler steam/condensate loop do not cross the boundary.
# Each OUT stream comes from a different tab, so the closure checks that the
# tabs agree with each other.
##############################################################################

with tab11:
    st.header("⚖️ Overall Mass Balance")
    flow_slot(('FEED', 'Distillate', 'Salt cake'))

    pdata = st.session_state.project_data

    ##########################################################################
    # STREAMS FROM OTHER TABS
    ##########################################################################
    feed_gpm = float(pdata.get("design_flow", 60.0))
    feed_lb = float(pdata.get("feed_lb_hr", feed_gpm * 500.0))  # Basis
    salt_in = float(pdata.get("salt_lb_hr_calc", 0.0))  # Basis
    water_in = feed_lb - salt_in

    vapor_lb = float(pdata.get("vapor_flow_lb_hr",  # Thermal tab
                               pdata.get("distillate_lb_hr_calc", 0.0)))
    vent_pct = float(pdata.get("vent_pct", 0.75))  # Basis
    vent_lb = vapor_lb * vent_pct / 100.0
    dist_lb = vapor_lb - vent_lb
    dist_gpm = dist_lb / 500.0
    dist_tds = float(pdata.get("distillate_cond", 0.0)) * float(pdata.get("tds_cond_factor", 0.65))
    salt_in_dist = dist_lb * dist_tds / 1e6

    cake_lb = float(pdata.get("cake_production_lb_hr", 0.0))  # Filter tab
    cake_solids_pct = float(pdata.get("cake_solids", 80.0))
    cake_salt = cake_lb * cake_solids_pct / 100.0
    cake_water = cake_lb - cake_salt

    slurry_lb = float(pdata.get("slurry_blowdown_lb_hr", 0.0))  # Vessel tab
    slurry_gpm = float(pdata.get("slurry_blowdown_gpm", 0.0))
    filtrate_lb = float(pdata.get("filtrate_return_lb_hr", 0.0))  # Filter tab
    filtrate_gpm = float(pdata.get("filtrate_return_gpm", 0.0))
    recirc_lb = float(pdata.get("recirc_lbhr", 0.0))  # Recirc tab
    recirc_gpm = float(pdata.get("recirculation_flow", 0.0))

    boiler_steam = float(pdata.get("steam_rate", 0.0))  # Boiler tab
    boiler_cond = float(pdata.get("condensate_return_lb_hr", 0.0))
    boiler_mu = float(pdata.get("boiler_makeup_water_lb_hr", 0.0))

    ##########################################################################
    # CLOSURES
    ##########################################################################
    total_out = dist_lb + vent_lb + cake_lb
    overall_closure = total_out / max(feed_lb, 1e-9) * 100.0
    water_out = dist_lb + vent_lb + cake_water - salt_in_dist
    water_closure = water_out / max(water_in, 1e-9) * 100.0
    salt_out = cake_salt + salt_in_dist
    salt_closure = salt_out / max(salt_in, 1e-9) * 100.0
    filter_closure = (cake_lb + filtrate_lb) / max(slurry_lb, 1e-9) * 100.0 if slurry_lb else 0.0
    water_recovery = dist_lb / max(water_in, 1e-9) * 100.0
    zld_check = 100.0 - (dist_lb + cake_lb) / max(feed_lb, 1e-9) * 100.0  # vent %


    def _status(c):
        if abs(c - 100) <= 1:
            return "🟢 GOOD"
        if abs(c - 100) <= 5:
            return "🟡 CHECK"
        return "🔴 REVIEW"


    closure_status = _status(overall_closure)

    ##########################################################################
    # DISPLAY
    ##########################################################################
    st.subheader("Mass Balance Closure")
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Overall", f"{overall_closure:.1f}%", _status(overall_closure), delta_color="off")
    k2.metric("Water", f"{water_closure:.1f}%", _status(water_closure), delta_color="off")
    k3.metric("Salt", f"{salt_closure:.1f}%", _status(salt_closure), delta_color="off")
    k4.metric("NCG Vent Loss", f"{vent_lb:,.0f} lb/hr", f"{zld_check:.2f}% of feed",
              delta_color="off")
    k5.metric("Recovery of Feed Water", f"{water_recovery:.1f}%",
              f"{dist_gpm:.1f} GPM distillate", delta_color="off",
              help="Distillate product (after vent loss) ÷ water in the feed.")

    if abs(overall_closure - 100) > 1 or abs(salt_closure - 100) > 1:
        st.warning("Tabs disagree. Open the Basis, Thermal, Vessel and Filter tabs in order "
                   "so each one refreshes from the Basis values.")

    st.subheader("Process Boundary Streams")
    boundary_df = pd.DataFrame([
        ["IN", "Feed (water + salt)", feed_lb, feed_gpm, "Basis"],
        ["  ", "  of which water", water_in, None, "Basis"],
        ["  ", "  of which dissolved salt", salt_in, None, "Basis"],
        ["OUT", "Distillate product", dist_lb, dist_gpm, "Thermal - vent"],
        ["OUT", "NCG vent loss", vent_lb, None, f"Basis ({vent_pct:g}% of vapor)"],
        ["OUT", "Wet cake", cake_lb, None, "Filter"],
        ["  ", "  of which salt", cake_salt, None, "Filter"],
        ["  ", "  of which moisture", cake_water, None, "Filter"],
        ["", "Total OUT", total_out, None, ""],
    ], columns=["Dir", "Stream", "lb/hr", "GPM", "Source tab"])
    boundary_df["lb/hr"] = boundary_df["lb/hr"].round(0)
    boundary_df["GPM"] = boundary_df["GPM"].round(1)
    st.dataframe(boundary_df, width="stretch", hide_index=True)

    st.subheader("Internal Streams (inside the boundary)")
    internal_df = pd.DataFrame([
        ["Recirculation (P-201 → E-220 → V-200)", recirc_lb, recirc_gpm],
        ["Slurry blowdown to filter", slurry_lb, slurry_gpm],
        ["Filtrate return to crystallizer", filtrate_lb, filtrate_gpm],
        ["Vapor to blower (B-310)", vapor_lb, vapor_lb / 500.0],
    ], columns=["Stream", "lb/hr", "GPM"]).round({"lb/hr": 0, "GPM": 1})
    st.dataframe(internal_df, width="stretch", hide_index=True)

    st.subheader("Boiler Loop (closed - not in process balance)")
    b1, b2, b3 = st.columns(3)
    b1.metric("Steam to E-220 Shell", f"{boiler_steam:,.0f} lb/hr")
    b2.metric("Condensate to V-510", f"{boiler_cond:,.0f} lb/hr")
    b3.metric("Boiler Makeup Water", f"{boiler_mu:,.0f} lb/hr")


    ##########################################################################
    # SAVE (water_recovery is no longer overwritten here - the Thermal tab owns it)
    ##########################################################################
    pdata.update({
        "overall_closure": overall_closure,
        "water_closure": water_closure,
        "salt_closure": salt_closure,
        "filter_closure": filter_closure,
        "thermal_closure": water_closure,
        "closure_status": closure_status,
        "distillate_product_lb_hr": dist_lb,
        "vent_loss_lb_hr": vent_lb,
    })

    save_project(project_name, st.session_state.project_data)
    st.success("✅ Mass Balance Auto Saved")

##############################################################################
# EQUIPMENT SUMMARY
# Uses the same data as the PDF report (report_pdf.collect_report) plus the
# P&ID items the report does not size, so the two always agree.
##############################################################################

with tab12:
    from report_pdf import collect_report

    st.header("🏭 Equipment Summary")
    flow_slot(())

    pdata = st.session_state.project_data
    rep = collect_report(pdata)


    def _v(key, fmt, unit="", default="-"):
        val = pdata.get(key)
        try:
            return f"{float(val):{fmt}} {unit}".strip()
        except (TypeError, ValueError):
            return default


    st.caption(f"Design basis: {rep['basis_name']}")

    ##########################################################################
    # KPIs
    ##########################################################################
    for col, (value, label) in zip(st.columns(len(rep["kpis"])), rep["kpis"]):
        col.metric(label, value)

    ##########################################################################
    # MAJOR EQUIPMENT (same rows as the PDF) + other P&ID items
    ##########################################################################
    st.subheader("Major Equipment")
    eq = pd.DataFrame(rep["equipment"],
                      columns=["Tag", "Equipment", "Design duty", "Size / rating", "Power"])
    n_filters = pdata.get("number_of_filters")
    if n_filters:
        eq.loc[eq["Tag"] == "Filter", "Size / rating"] = (
            f"{_v('required_filter_area', ',.0f', 'ft²')} req'd, {int(n_filters)} units "
            f"× {_v('filter_area_per_unit', ',.0f', 'ft²')}")
    eq.loc[eq["Tag"] == "V-200", "Size / rating"] = (
        f"{_v('crystallizer_volume', ',.0f', 'gal')} slurry, min ID "
        f"{_v('vessel_min_id_ft', '.1f', 'ft')}")
    eq.loc[eq["Tag"] == "Boiler", "Size / rating"] = (
        f"{_v('boiler_selected_kw', ',.0f', 'kW')} selected "
        f"({pdata.get('boiler_governing_case', '-')})")

    other = pd.DataFrame([
        ["Feed tank", "Feed storage", _v("total_feed_storage", ",.0f", "gal"),
         _v("retention_hours", ".1f", "hr retention"), "-"],
        ["V-410", "Distillate tank", _v("distillate_gpm_calc", ".1f", "GPM"),
         _v("distillate_temp_c", ".0f", "°C in"), "-"],
        ["V-510", "Boiler condensate receiver", _v("condensate_return_lb_hr", ",.0f", "lb/hr"),
         "-", "-"],
        ["P-520", "Boiler condensate return pump", _v("condensate_return_lb_hr", ",.0f", "lb/hr"),
         "-", "-"],
        ["-", "Slurry blowdown to filter", _v("slurry_blowdown_gpm", ".1f", "GPM"),
         _v("crystallizer_solids", ".0f", "% solids"), "-"],
    ], columns=eq.columns)
    st.dataframe(pd.concat([eq, other], ignore_index=True), width="stretch", hide_index=True)

    ##########################################################################
    # ELECTRICAL LOAD
    ##########################################################################
    st.subheader("⚡ Electrical Load")
    power = list(rep["power"])
    steady_boiler = pdata.get("boiler_steady_kw")
    rows = [[n, round(v)] for n, v in power]
    if steady_boiler:
        rows.append(["Boiler (steady makeup)", round(float(steady_boiler))])
    running_kw = sum(r[1] for r in rows)
    installed_boiler = float(pdata.get("boiler_selected_kw", pdata.get("boiler_power_kw", 0.0)) or 0.0)

    e1, e2, e3 = st.columns(3)
    e1.metric("Running Load", f"{running_kw:,.0f} kW")
    e2.metric("Boiler Installed", f"{installed_boiler:,.0f} kW")
    e3.metric("Connected Load", f"{running_kw - float(steady_boiler or 0) + installed_boiler:,.0f} kW",
              help="Motors + full boiler rating (start-up case).")
    if rows:
        st.dataframe(pd.DataFrame(rows, columns=["Load", "kW"]), width="stretch", hide_index=True)

    ##########################################################################
    # PROCESS SUMMARY
    ##########################################################################
    st.subheader("📋 Process Summary")
    p1, p2 = st.columns(2)
    with p1:
        st.dataframe(pd.DataFrame(rep["basis"], columns=["Parameter", "Value"]),
                     width="stretch", hide_index=True)
    with p2:
        st.dataframe(pd.DataFrame(rep["temps"], columns=["Stream", "°C", "°F"]),
                     width="stretch", hide_index=True)

    pdata["running_kw"] = running_kw
    save_project(project_name, st.session_state.project_data)

##############################################################################
# FINAL DESIGN REPORT
# Requires report_pdf.py in the same folder as this file.
##############################################################################

with tab13:
    from report_pdf import collect_report, build_design_report_pdf

    st.header("📄 Final Design Report")
    flow_slot(())

    pdata = st.session_state.project_data
    rep = collect_report(pdata)

    ##########################################################################
    # REPORT DETAILS (title block of the PDF - saved with the project)
    ##########################################################################
    with st.expander("📝 Report details (PDF title block)", expanded=False):
        m1, m2, m3 = st.columns(3)
        with m1:
            rp_company = st.text_input("Company", pdata.get("report_company", "UCC Environmental"))
            rp_project = st.text_input("Project", pdata.get("report_project",
                                                            "Plant Scherer FGD Crystallizer"))
            rp_client = st.text_input("Client / Site", pdata.get("report_client",
                                                                 "Georgia Power Company - Plant Scherer"))
        with m2:
            rp_project_no = st.text_input("Project No.", pdata.get("report_project_no", "55767-03"))
            rp_doc_no = st.text_input("Document No.", pdata.get("report_doc_no", ""))
            rp_vendor = st.text_input("Vendor", pdata.get("report_vendor", "Vacom"))
        with m3:
            rp_rev = st.text_input("Revision", pdata.get("report_rev", "A"))
            rp_prepared = st.text_input("Prepared by", pdata.get("report_prepared", ""))
            rp_checked = st.text_input("Checked by", pdata.get("report_checked", ""))
        rp_approved = st.text_input("Approved by", pdata.get("report_approved", ""))
        rp_summary = st.text_area(
            "Executive summary (leave blank for automatic text)",
            pdata.get("report_summary", ""), height=90)
        rp_notes = st.text_area(
            "Additional notes (one per line - printed in Notes & Assumptions)",
            pdata.get("report_notes", ""), height=90)

    pdata.update({
        "report_company": rp_company, "report_project": rp_project,
        "report_client": rp_client, "report_project_no": rp_project_no,
        "report_doc_no": rp_doc_no, "report_vendor": rp_vendor, "report_rev": rp_rev,
        "report_prepared": rp_prepared, "report_checked": rp_checked,
        "report_approved": rp_approved, "report_summary": rp_summary,
        "report_notes": rp_notes,
    })

    ##########################################################################
    # EXECUTIVE SUMMARY
    ##########################################################################
    st.subheader("📊 Executive Summary")
    st.caption(f"Design basis: {rep['basis_name']}")

    for col, (value, label) in zip(st.columns(len(rep["kpis"])), rep["kpis"]):
        with col:
            st.metric(label, value)

    st.divider()

    ##########################################################################
    # PROCESS FLOW DIAGRAM (page 1 of the PDF)
    ##########################################################################
    import streamlit.components.v1 as components

    ##########################################################################
    # EFFLUENT OUTPUT (last page of the PDF)
    ##########################################################################
    st.subheader("💧 Effluent Output")
    _eff = effluent_data(pdata)
    ef1, ef2, ef3, ef4, ef5 = st.columns(5)
    ef1.metric("Distillate", f"{_eff['dist_gpm']:.1f} GPM", f"{_eff['dist']:,.0f} lb/hr", delta_color="off")
    ef2.metric("Distillate TDS", f"{_eff['dist_tds']:.1f} mg/L", f"{_eff['dist_cond']:.0f} µS/cm", delta_color="off")
    ef3.metric("Salt Cake", f"{_eff['cake_tpd']:.1f} TPD", f"{_eff['cake']:,.0f} lb/hr", delta_color="off")
    ef4.metric("NCG Vent", f"{_eff['vent']:,.0f} lb/hr")
    ef5.metric("Liquid Discharge", "0 GPM", "ZLD", delta_color="off")
    st.dataframe(pd.DataFrame(_eff["rows"],
                              columns=["Stream", "lb/hr", "Flow", "Quality", "Temperature", "Destination"]),
                 hide_index=True, width="stretch")
    st.caption(f"Total effluent {_eff['total_out']:,.0f} lb/hr vs feed {_eff['feed_lb']:,.0f} lb/hr "
               f"(closure {_eff['closure']:.1f}%) | feed-water recovery {_eff['recovery']:.1f}% | "
               f"salt cake {_eff['cake_ton_yr']:,.0f} ton/yr.")
    st.dataframe(pd.DataFrame(chemical_rows(pdata),
                              columns=["Chemicals / utility water in", "Rate", "Volume", "Annual / note"]),
                 hide_index=True, width="stretch")

    st.divider()

    ##########################################################################
    # TABLES (same content as the PDF)
    ##########################################################################
    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Design Basis")
        st.dataframe(pd.DataFrame(rep["basis"], columns=["Parameter", "Value"]),
                     hide_index=True, width="stretch")
        st.subheader("Mass Balance")
        st.dataframe(pd.DataFrame(rep["mass"], columns=["Stream", "lb/hr", "GPM"]),
                     hide_index=True, width="stretch")
    with t2:
        st.subheader("Process Temperatures")
        st.dataframe(pd.DataFrame(rep["temps"], columns=["Stream", "°C", "°F"]),
                     hide_index=True, width="stretch")
        st.subheader("Energy & Utilities")
        st.dataframe(pd.DataFrame(rep["energy"], columns=["Item", "Value"]),
                     hide_index=True, width="stretch")

    st.subheader("🏭 Major Equipment Summary")
    st.dataframe(pd.DataFrame(rep["equipment"],
                              columns=["Tag", "Equipment", "Design duty", "Size / rating", "Power"]),
                 hide_index=True, width="stretch")

    st.subheader("Vendor Comparison")
    st.dataframe(pd.DataFrame(rep["vendor"], columns=["Item", "Calculated", "Vendor", "Remark"]),
                 hide_index=True, width="stretch")

    ##########################################################################
    # DESIGN CHECKS
    ##########################################################################
    st.subheader("✅ Design Checks")
    if rep["checks"]:
        icon = {"OK": "🟢", "CHECK": "🟡", "REVIEW": "🔴"}
        st.dataframe(pd.DataFrame(
            [[i, f"{icon.get(s, '')} {s}", c] for i, s, c in rep["checks"]],
            columns=["Check", "Status", "Comment"]), hide_index=True, width="stretch")
        n_review = sum(1 for _, s, _ in rep["checks"] if s == "REVIEW")
        n_check = sum(1 for _, s, _ in rep["checks"] if s == "CHECK")
        if n_review:
            st.error(f"{n_review} item(s) need review before the report is issued.")
        elif n_check:
            st.warning(f"{n_check} item(s) to check - report can be issued with comments.")
        else:
            st.success("All design checks passed.")
    else:
        st.info("Open the Basis, Feed HX, Recirc Pump, Primary HX and Blower tabs to populate checks.")

    ##########################################################################
    # PDF DOWNLOAD
    ##########################################################################
    st.divider()

    meta = {
        "company": rp_company, "project": rp_project, "client": rp_client,
        "project_no": rp_project_no, "doc_no": rp_doc_no, "vendor": rp_vendor,
        "revision": rp_rev, "prepared_by": rp_prepared, "checked_by": rp_checked,
        "approved_by": rp_approved, "summary": rp_summary, "notes": rp_notes,
    }

    try:
        main_pdf = build_design_report_pdf(pdata, meta)
        pdf_bytes = main_pdf
        try:
            cover_pdf = build_cover_pdf(pdata, meta)
            effluent_pdf = build_effluent_pdf(pdata, meta)
            merged = merge_pdfs([cover_pdf, main_pdf, effluent_pdf])
            if merged is None:
                st.warning("Install pypdf (pip install pypdf) to put the flow diagram first and the "
                           "effluent page last in one PDF. Downloading the original report; the "
                           "extra pages are available separately below.")
                st.download_button("📄 Flow diagram page (PDF)", data=cover_pdf,
                                   file_name="PFD_cover_page.pdf", mime="application/pdf")
                st.download_button("📄 Effluent output page (PDF)", data=effluent_pdf,
                                   file_name="Effluent_output_page.pdf", mime="application/pdf")
            else:
                pdf_bytes = merged
        except Exception as _exc:
            st.warning(f"Flow diagram / effluent pages could not be added: {_exc}")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in
                       (rp_project_no or "MVR") + "_Rev" + (rp_rev or "A"))
        st.download_button(
            label="📄 Download Design Report PDF",
            data=pdf_bytes,
            file_name=f"{safe}_MVR_Crystallizer_Design_Report.pdf",
            mime="application/pdf",
            type="primary",
        )
    except Exception as exc:
        st.error(f"PDF could not be created: {exc}")

    save_project(project_name, st.session_state.project_data)

##############################################################################
# CONTROL PHILOSOPHY
##############################################################################

with tab14:
    st.header("🎛️ Control Philosophy")
    flow_slot(())

    control_text = """
### Crystallizer Level Control

The crystallizer vessel level is controlled by automatic
modulation of the feed control valve.

• Low Level (<40%) → Feed Valve Opens

• Normal Level (40-80%) → Feed Valve Modulates

• High Level (>80%) → Feed Valve Closes

### Crystallizer Pressure Control

The crystallizer operates under vacuum at approximately
0.35 bar absolute pressure.

This vacuum operation reduces boiling temperature,
improves MVR performance and minimizes thermal stress.

• Low Pressure → Steam Valve Opens

• Normal Pressure → Steam Valve Modulates

• High Pressure → Steam Valve Closes

### Recirculation Loop Control

The recirculation pump continuously circulates slurry
through the Primary Heat Exchanger and Crystallizer.

Typical Design Basis:

• Feed Flow = 60 GPM

• Recirculation Flow = 720 GPM

• Recirculation Ratio = 12:1

The high recirculation rate is required to:

• Maintain crystal suspension

• Prevent tube scaling

• Improve heat transfer coefficient

• Minimize boiling inside tubes

• Maintain uniform supersaturation

### MVR Blower Control

The MVR blower compresses vapor generated from the
crystallizer and returns latent heat to the process.

Blower operation is monitored by:

• Compression Ratio

• Vapor Flow

• Motor Load

• Differential Pressure

### Differential Pressure Control

The concentrate discharge valve is automatically
modulated using blower differential pressure.

• DP > 5 psi → Open Concentrate Valve Further

• DP 2-5 psi → Normal Modulation

• DP < 2 psi → Close Concentrate Valve

### Slurry Solids Control

Crystallizer slurry solids concentration is maintained
within the desired operating range.

• TSS <10% → Retain Slurry In Loop

• TSS 10-15% → Normal Operation

• TSS >15% → Controlled slurry discharge to the filter feed tank / Oberlin filter

### Oberlin Filter Control

The Oberlin Filter separates crystal solids from slurry.

• Automatic cake discharge

• Continuous filtrate collection

• Filtrate recycled to crystallizer

• Automatic cycle sequencing

• Filter loading monitoring

### Boiler Control

The boiler provides supplemental steam during:

• Startup

• Low MVR performance

• Process upset conditions

• Peak thermal demand

### Condensate Recovery Control

Condensed vapor from the Primary Heat Exchanger is
routed through the Feed Preheater.

This recovered energy reduces steam consumption
and improves overall thermal efficiency.

### Chemical Dosing and Utility Water

Sulfamic acid, defoamer and utility water are injected into the
circulation pump suction header (see P&ID).

• Sulfamic acid supply pump - scale control / periodic descaling

• Defoamer supply pump - dosed on foam indication / proportional to feed

• Utility water - seal flush, demister wash and acid make-up

### Safety Interlocks

• High Crystallizer Pressure

• Low Crystallizer Level

• High Crystallizer Level

• Recirculation Pump Failure

• MVR High Vibration

• MVR Motor Overload

• Boiler Low Water Cutoff

• Oberlin Filter High Level

• Emergency Shutdown Signal
"""

    st.markdown(control_text)

    ##########################################################################
    # OPERATING SEQUENCE
    ##########################################################################
    st.subheader("🔄 Normal Operating Sequence")

    sequence_text = """
### Process Sequence

1. Feedwater is transferred from the Feed Tank.

2. Feedwater passes through the Pre-HX where recovered
   heat from hot distillate is utilized.

3. Preheated feed enters the recirculation loop.

4. The Recirculation Pump circulates slurry through the
   Primary Heat Exchanger.

5. Heated slurry enters the Crystallizer Vessel.

6. Water evaporates within the Crystallizer under vacuum.

7. Vapor generated from the crystallizer flows to the MVR.

8. The MVR compresses vapor and raises its temperature.

9. Compressed vapor enters the Primary Heat Exchanger.

10. Vapor condenses and transfers latent heat to the
    recirculation loop.

11. Distillate exits the HX and passes through the Pre-HX.

12. Distillate is collected in the Distillate Tank.

13. Crystal slurry is withdrawn from the Crystallizer.

14. Slurry is sent to the Oberlin Filter.

15. Crystal cake is discharged.

16. Filtrate is recycled back to the Crystallizer.

17. Boiler provides startup and supplemental steam
    whenever necessary.

---

### Recirculation Philosophy

Feed Flow = 60 GPM

Recirculation Ratio = 12 : 1

Recirculation Flow = 720 GPM

The crystallizer requires high recirculation rates to:

• Keep crystals suspended

• Prevent scaling and fouling

• Improve heat transfer

• Maintain stable supersaturation

• Prevent tube boiling

---

### Typical Industry Recirculation Ratios

Falling Film Evaporator ............ 1 - 3

Forced Circulation Evaporator ...... 5 - 15

Crystallizer ....................... 8 - 25

High Salt ZLD Crystallizer ......... 15 - 40

---

### Typical Temperatures

Feed Tank ..................... 25°C

Pre-HX Outlet ................. 45-55°C

Recirculation Loop ............ 75-80°C

Primary HX Outlet ............. 78-82°C

Crystallizer Boiling .......... 78°C

MVR Suction ................... 78°C

MVR Discharge ................. 84-88°C

Distillate Tank ............... 40-50°C

---

### Crystallizer Boiling Temperature

Boiling temperature depends on:

• Operating vacuum

• Boiling Point Elevation (BPE)

• Dissolved salt concentration

Typical Values:

Deep Vacuum (~0.2 bar abs) ........ 60-65°C

Medium Vacuum (~0.3 bar abs) ...... 70-75°C

Vacuum (~0.35-0.4 bar abs) ........ 75-80°C

Atmospheric ....................... 100°C + BPE

Typical MVR Crystallizer .......... 75-90°C

Current Design Basis .............. 78°C


Pressure maintain in vessel
Parameter                                                     	Description

Crystallizer Operating Pressure	                           7.25 psia (0.50 bar abs)
Equivalent Vacuum                                       	7.45 psi below atmospheric pressure
Water Boiling Temperature	                                                ~81°C
Assumed Boiling Point Elevation (BPE)                                         	~6°C
Crystallizer Liquor Temperature                                       	~87°C
Primary Objective	                                Maintain constant vessel pressure for stable evaporation and crystal growth
Pressure Measurement                            	Pressure Transmitter (PT) on crystallizer vapor space
Primary Pressure Control Device                 	MVR Blower / Vapor Compressor VFD
Startup Vacuum Source	                                    Liquid Ring Vacuum Pump or Steam Ejector
Normal Vacuum Source                                    	MVR blower suction removes vapor continuously
Control Setpoint	                                            7.25 psia (0.50 bar abs)


#VACOM SYSTEM
Vessel pressure control system

Vessel pressure is the controlled variable, and blower speed is how the controller holds it. Blower ΔP is not set directly. It settles at whatever value the primary heat exchanger needs to transfer the heat.

What is maintained, and why
Maintained	How	Why
Vessel pressure, 7.25 psia	Blower VFD speed	Fixes the boiling temperature (Tsat 81.3 °C + BPE ≈ 87 °C). That gives stable crystal growth, steady scaling behaviour, stable suction conditions for the blower, and NPSH for the recirculation pump.
Energy balance	Boiler makeup steam (heat deficit) or vent / desuperheat (heat surplus)	The blower’s work and the plant’s heat losses rarely match exactly.
Vessel level	Feed flow	Keeps the brine inventory constant.
Slurry density / TSS	Salt slurry withdrawal	Controls crystal size and prevents plugging.

Blower ΔP is the result:

ΔP needed → condensing temp = brine boiling temp + BPE effect + HX ΔT
HX ΔT = Q ÷ (U × A)

Heat to the brine comes only from vapor condensing in the primary heat exchanger. The compressed vapor therefore has to condense hotter than the brine boils, by enough to push the evaporation duty through the exchanger. With 3 psig discharge, that’s 105.3 °C condensing against about 87 °C boiling, an 18 °C driving force.

What happens when vessel pressure changes

Pressure rises (vapor forming faster than the blower removes it). This could be from hotter feed, a cleaner heat exchanger, or extra boiler steam.

Vapor accumulates and vessel pressure goes above 7.25 psia.
Boiling temperature rises (for example 87.3 → 89.0 °C at 7.75 psia).
The pressure controller speeds up the blower.
The blower pulls more vapor, so vessel pressure comes back to 7.25 psia.
At the higher speed, discharge pressure and ΔP rise, so the vapor condenses hotter. The heat exchanger then transfers more heat and the plant evaporates more.
It settles at a higher throughput, with the vessel back at 7.25 psia and a somewhat higher ΔP.

Pressure falls (vapor forming slower). This could be from heat exchanger fouling, colder feed, or heat losses.

Vessel pressure drops and boiling temperature drops.
The controller slows the blower, so less vapor is pulled and pressure recovers.
ΔP and condensing temperature fall, so throughput drops.
If throughput is now too low, boiler makeup steam is added to restore heat.
Using ΔP as a health indicator

At constant vessel pressure, a steadily rising ΔP and blower speed for the same distillate flow mean the heat exchanger needs more ΔT. That usually means fouling or scaling in the primary HX. When the blower nears its maximum speed or ΔP limit, it’s time to clean.

Two control setups seen in MVR plants
Your scheme: blower speed holds vessel pressure. Throughput floats, and boiler steam or the vent corrects the heat balance.
Common alternative: blower speed holds ΔP or distillate flow, which sets throughput. Vessel pressure is then held by makeup steam (pressure low) and a vent valve (pressure high).

Your app has both a vessel pressure setpoint and a “Blower ΔP setpoint”. Ask Vacom which one the VFD actually controls, and which one the steam and vent valves control. Only one loop can drive the blower speed.

How the code models this

The code uses fan laws (ΔP ∝ speed²), which is a simplification. In the real plant, ΔP is set where the blower curve meets the heat exchanger’s “system curve”. I can revise the code so ΔP comes from the heat exchanger duty (U, area, fouling), and model the steam and vent loop for the heat balance.


Inputs that change the results

Input	Where	What it changes
Feed TDS	Feed Water tab, “TDS” row	Feed SG, salt, distillate, vapor flow, blower power
Salt cake moisture (%)	Mass balance	Cake water, distillate, vapor flow, blower power
Crystallizer pressure SP (psia)	Control philosophy	Saturation temp, boiling temp, ΔP, pressure ratio, temperature lift, discharge temp, HX ΔT, power, VFD setpoint
Blower discharge pressure (psig)	Control philosophy	ΔP, pressure ratio, condensing temp, discharge temp, superheat, HX ΔT, power
Atmospheric pressure (psia)	Operating conditions	Discharge psia, so everything in the row above
Boiling point elevation (°C)	Control philosophy	Boiling temp, HX ΔT, blower suction temp, discharge temp, power (slightly)
Blower efficiency (%)	MVR config	Actual discharge temp, superheat, power
Motor efficiency (%)	MVR config	Electrical power
Actual vessel pressure PV (psia)	VFD section	Error, action, speed, boiling temp at PV, ΔP, discharge pressure, condensing temp, power
Design speed, Kp, min/max speed	VFD section	Speed, ΔP, vapor flow and power at PV
Vacom distillate and salt targets	Design targets	Only the comparison numbers (+/−)

Inputs that change nothing right now

Input	Note
Operating hours, plant availability, design margin	Not used, and not saved to the project
Boiler capacity, boiler pressure	Saved only; boiler capacity isn’t calculated
Process configuration (system, type, steam source, objective)	Not used
Recovery sliders (water, solids, crystal, overall)	Not used. The 95% water recovery slider doesn’t match the calculated value: 29,112 ÷ 31,525 ≈ 92% at 60 GPM
Operating pressure (3 psig), design temperature (228 °F)	Not used; this 3 psig duplicates the blower discharge pressure input
Number of trains, standby configuration	Not used, so flow isn’t split per train
Vessel level SP, TSS low/high, MVR included, steam economy target	Saved or shown only

Fixed constants in the code (could be turned into inputs):

Constant	Value
Vapor k / Cp	1.33 / 1.9 kJ/kg·K
Feed SG	Estimated as 1 + 0.00069 × TDS (g/L)
Liquid purge	None (pure ZLD)
Water density	8.34 lb/gal

The most useful inputs to connect next would be:
"""

    st.markdown(sequence_text)

    st.success("✅ Control Philosophy Generated")

##############################################################################
# CHEMICALS & UTILITIES
# P&ID: sulfamic acid supply pump, defoamer supply pump and utility water tie
# into the circulation-pump suction header; instrument air from the air receiver.
##############################################################################

with tab_chem:
    st.header("🧪 Chemicals & Utilities")
    flow_slot(('Sulfamic acid', 'Defoamer', 'Utility water'))

    pdata = st.session_state.project_data

    ##########################################################################
    # VALUES FROM OTHER TABS
    ##########################################################################
    feed_gpm = float(pdata.get("design_flow", 60.0))
    feed_lb_hr = float(pdata.get("feed_lb_hr", feed_gpm * 500.0))
    op_hours = float(pdata.get("operating_hours", 8000))
    loop_gal = float(pdata.get("slurry_volume_gal", 16000.0))  # total slurry inventory
    recirc_gpm = float(pdata.get("recirculation_flow", 0.0))

    feed_gph = feed_gpm * 60.0
    feed_L_hr = feed_gph * 3.785  # litres of feed per hour
    LB_PER_MG = 1.0 / 453592.0

    st.info(
        f"From Design Basis: feed {feed_gpm:.1f} GPM ({feed_lb_hr:,.0f} lb/hr) | "
        f"slurry loop {loop_gal:,.0f} gal | recirculation {recirc_gpm:,.0f} GPM | "
        f"{op_hours:,.0f} operating hr/yr. Chemicals are injected into the circulation "
        "pump suction header (P&ID)."
    )

    ##########################################################################
    # SULFAMIC ACID
    ##########################################################################
    st.subheader("🧴 Sulfamic Acid (scale control / descaling)")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        acid_ppm = st.number_input(
            "Continuous Dose (mg/L of feed, 100% acid)", min_value=0.0, step=5.0,
            value=float(pdata.get("acid_dose_ppm", 50.0)), key="cu_acid_ppm",
            help="Set to 0 if the acid is used only for periodic cleaning. "
                 "Confirm the dose with the water chemist / vendor.")
    with a2:
        acid_strength = st.number_input(
            "Solution Strength (% w/w)", min_value=1.0, max_value=15.0, step=0.5,
            value=float(pdata.get("acid_strength_pct", 10.0)), key="cu_acid_strength",
            help="Sulfamic acid solubility is about 15% at 20 °C. 5-10% is usual.")
    with a3:
        acid_sg = st.number_input(
            "Solution SG", min_value=1.0, max_value=1.2, step=0.01,
            value=float(pdata.get("acid_sg", 1.05)), key="cu_acid_sg")
    with a4:
        acid_cost = st.number_input(
            "Acid Cost ($/lb, 100%)", min_value=0.0, step=0.05,
            value=float(pdata.get("acid_cost_lb", 0.90)), key="cu_acid_cost")

    a5, a6, a7, a8 = st.columns(4)
    with a5:
        acid_tank_days = st.number_input(
            "Day Tank Storage (days)", min_value=0.5, step=0.5,
            value=float(pdata.get("acid_tank_days", 7.0)), key="cu_acid_days")
    with a6:
        acid_pump_margin = st.number_input(
            "Metering Pump Margin (%)", min_value=0.0, max_value=200.0, step=10.0,
            value=float(pdata.get("acid_pump_margin", 50.0)), key="cu_acid_pmargin")
    with a7:
        cip_conc = st.number_input(
            "Cleaning (CIP) Strength (% w/w)", min_value=1.0, max_value=15.0, step=0.5,
            value=float(pdata.get("acid_cip_pct", 5.0)), key="cu_cip_conc")
    with a8:
        cips_per_year = st.number_input(
            "Cleanings per Year", min_value=0, step=1,
            value=int(pdata.get("acid_cips_per_year", 6)), key="cu_cips")

    cip_fill_pct = st.slider(
        "CIP volume as % of slurry loop inventory", 10, 100,
        int(pdata.get("acid_cip_fill_pct", 40)), key="cu_cip_fill",
        help="Primary HX, recirculation piping and vessel cone that are filled with "
             "cleaning solution.")

    # continuous dosing
    acid_lb_hr = feed_L_hr * acid_ppm * LB_PER_MG  # 100 % acid
    acid_soln_lb_hr = acid_lb_hr / (acid_strength / 100.0)
    acid_soln_gph = acid_soln_lb_hr / (acid_sg * 8.34)
    acid_dilution_lb_hr = acid_soln_lb_hr - acid_lb_hr  # water to make solution
    acid_dilution_gpm = acid_dilution_lb_hr / 500.0
    acid_pump_gph = acid_soln_gph * (1 + acid_pump_margin / 100.0)
    acid_tank_gal = acid_soln_gph * 24.0 * acid_tank_days
    acid_lb_yr_cont = acid_lb_hr * op_hours

    # cleaning
    cip_gal = loop_gal * cip_fill_pct / 100.0
    cip_acid_lb = cip_gal * 8.34 * acid_sg * cip_conc / 100.0
    cip_acid_lb_yr = cip_acid_lb * cips_per_year

    acid_lb_yr = acid_lb_yr_cont + cip_acid_lb_yr
    acid_cost_yr = acid_lb_yr * acid_cost

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Acid, 100 %", f"{acid_lb_hr:,.1f} lb/hr", f"{acid_lb_hr * 24:,.0f} lb/day",
              delta_color="off")
    s2.metric("Solution Feed", f"{acid_soln_gph:,.1f} gph", f"{acid_soln_gph / 60:.2f} GPM",
              delta_color="off")
    s3.metric("Metering Pump", f"{acid_pump_gph:,.1f} gph",
              f"+{acid_pump_margin:.0f}% margin", delta_color="off")
    s4.metric("Day Tank", f"{acid_tank_gal:,.0f} gal", f"{acid_tank_days:g} days", delta_color="off")
    s5.metric("Dilution Water", f"{acid_dilution_gpm:.2f} GPM",
              "from utility water", delta_color="off")

    s6, s7, s8, s9 = st.columns(4)
    s6.metric("CIP Volume", f"{cip_gal:,.0f} gal")
    s7.metric("Acid per Cleaning", f"{cip_acid_lb:,.0f} lb", f"{cip_conc:g}% solution",
              delta_color="off")
    s8.metric("Acid per Year", f"{acid_lb_yr / 2000:,.1f} ton",
              f"{acid_lb_yr_cont / 2000:,.1f} dosing + {cip_acid_lb_yr / 2000:,.1f} CIP",
              delta_color="off")
    s9.metric("Acid Cost", f"${acid_cost_yr:,.0f} /yr")

    st.caption("Sulfamic acid is a solid; it is dissolved in utility water in the supply "
               "tank (dilution water above). P&ID: 1 supply pump (duty), discharge to the "
               "circulation pump suction header.")

    st.divider()

    ##########################################################################
    # DEFOAMER
    ##########################################################################
    st.subheader("🫧 Defoamer / Antifoam")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        df_ppm = st.number_input(
            "Dose (mg/L of feed, as supplied)", min_value=0.0, step=1.0,
            value=float(pdata.get("defoamer_dose_ppm", 10.0)), key="cu_df_ppm",
            help="Typical antifoam dose 5-50 mg/L; get the figure from the vendor / jar test.")
    with d2:
        df_sg = st.number_input(
            "Product SG", min_value=0.8, max_value=1.3, step=0.01,
            value=float(pdata.get("defoamer_sg", 1.00)), key="cu_df_sg")
    with d3:
        df_cost = st.number_input(
            "Product Cost ($/lb)", min_value=0.0, step=0.1,
            value=float(pdata.get("defoamer_cost_lb", 2.50)), key="cu_df_cost")
    with d4:
        df_tote_gal = st.number_input(
            "Tote / Drum Size (gal)", min_value=5.0, step=5.0,
            value=float(pdata.get("defoamer_tote_gal", 275.0)), key="cu_df_tote",
            help="275 gal tote or 55 gal drum.")

    d5, d6 = st.columns(2)
    with d5:
        df_pump_margin = st.number_input(
            "Metering Pump Margin (%)", min_value=0.0, max_value=200.0, step=10.0,
            value=float(pdata.get("defoamer_pump_margin", 100.0)), key="cu_df_pmargin",
            help="Foaming events need a burst dose well above the average.")
    with d6:
        df_tank_days = st.number_input(
            "Storage (days)", min_value=0.5, step=0.5,
            value=float(pdata.get("defoamer_tank_days", 30.0)), key="cu_df_days")

    df_lb_hr = feed_L_hr * df_ppm * LB_PER_MG
    df_gph = df_lb_hr / (df_sg * 8.34)
    df_pump_gph = df_gph * (1 + df_pump_margin / 100.0)
    df_gal_day = df_gph * 24.0
    df_storage_gal = df_gal_day * df_tank_days
    df_totes_yr = df_gph * op_hours / df_tote_gal
    df_lb_yr = df_lb_hr * op_hours
    df_cost_yr = df_lb_yr * df_cost

    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Defoamer", f"{df_lb_hr:,.2f} lb/hr", f"{df_gph:.2f} gph", delta_color="off")
    f2.metric("Usage", f"{df_gal_day:,.1f} gal/day")
    f3.metric("Metering Pump", f"{df_pump_gph:.2f} gph", f"+{df_pump_margin:.0f}% margin",
              delta_color="off")
    f4.metric("Storage", f"{df_storage_gal:,.0f} gal", f"{df_tank_days:g} days", delta_color="off")
    f5.metric("Annual", f"{df_totes_yr:,.1f} totes", f"${df_cost_yr:,.0f} /yr", delta_color="off")

    st.divider()

    ##########################################################################
    # UTILITY WATER
    ##########################################################################
    st.subheader("🚰 Utility Water")

    u1, u2, u3, u4 = st.columns(4)
    with u1:
        uw_pressure = st.number_input(
            "Supply Pressure (psig)", min_value=0.0, step=5.0,
            value=float(pdata.get("uw_pressure_psig", 60.0)), key="cu_uw_p")
    with u2:
        uw_velocity = st.number_input(
            "Header Velocity (ft/s)", min_value=2.0, max_value=10.0, step=0.5,
            value=float(pdata.get("uw_velocity", 5.0)), key="cu_uw_v")
    with u3:
        uw_simul = st.number_input(
            "Simultaneous Use Factor", min_value=0.1, max_value=1.0, step=0.05,
            value=float(pdata.get("uw_simultaneity", 0.6)), key="cu_uw_sim",
            help="Fraction of the intermittent demand that can run at the same time.")
    with u4:
        uw_cost = st.number_input(
            "Water Cost ($/kgal)", min_value=0.0, step=0.5,
            value=float(pdata.get("uw_cost_kgal", 3.0)), key="cu_uw_cost")

    st.caption("Consumer list - placeholder values, replace with vendor / P&ID data. "
               "Add or delete rows as needed. 'To process %' is the share that ends up "
               "in the crystallizer brine and must be evaporated.")

    default_uw = [
        {"Consumer": "Recirc pump mechanical seal flush", "GPM": 2.0, "Hr/day": 24.0, "To process %": 100.0},
        {"Consumer": "Feed / blowdown / filtrate pump seals", "GPM": 1.5, "Hr/day": 24.0, "To process %": 100.0},
        {"Consumer": "Mist eliminator / demister wash", "GPM": 20.0, "Hr/day": 0.5, "To process %": 100.0},
        {"Consumer": "Vessel & HX flush (shutdown)", "GPM": 100.0, "Hr/day": 0.25, "To process %": 0.0},
        {"Consumer": "Hose stations / washdown", "GPM": 10.0, "Hr/day": 0.5, "To process %": 0.0},
    ]
    saved_uw = pdata.get("uw_consumers")
    uw_df0 = pd.DataFrame(saved_uw if saved_uw else default_uw,
                          columns=["Consumer", "GPM", "Hr/day", "To process %"])

    uw_df = st.data_editor(
        uw_df0, hide_index=True, num_rows="dynamic", width="stretch", key="cu_uw_editor",
        column_config={
            "GPM": st.column_config.NumberColumn(min_value=0.0, format="%.2f"),
            "Hr/day": st.column_config.NumberColumn(min_value=0.0, max_value=24.0, format="%.2f"),
            "To process %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.0f"),
        })
    uw_df = uw_df.dropna(subset=["Consumer"]).fillna(0.0)

    # add the acid dilution as a calculated consumer
    calc_rows = pd.DataFrame([{
        "Consumer": "Sulfamic acid solution make-up (calculated)",
        "GPM": acid_dilution_gpm, "Hr/day": 24.0, "To process %": 100.0}])
    all_uw = pd.concat([uw_df, calc_rows], ignore_index=True)
    all_uw["Avg GPM"] = all_uw["GPM"] * all_uw["Hr/day"] / 24.0
    all_uw["Gal/day"] = all_uw["Avg GPM"] * 1440.0
    all_uw["To process GPM"] = all_uw["Avg GPM"] * all_uw["To process %"] / 100.0

    uw_avg_gpm = float(all_uw["Avg GPM"].sum())
    uw_connected = float(all_uw["GPM"].sum())
    cont = all_uw[all_uw["Hr/day"] >= 23.5]["GPM"].sum()
    inter = all_uw[all_uw["Hr/day"] < 23.5]["GPM"].sum()
    uw_peak_gpm = float(cont + inter * uw_simul)
    uw_gal_day = float(all_uw["Gal/day"].sum())
    uw_to_process_gpm = float(all_uw["To process GPM"].sum())
    uw_to_process_lb_hr = uw_to_process_gpm * 500.0
    uw_pct_feed = uw_to_process_gpm / max(feed_gpm, 1e-9) * 100.0
    flow_cfs = uw_peak_gpm / 448.83
    uw_pipe_in = math.sqrt(4 * flow_cfs / (math.pi * uw_velocity)) * 12.0 if flow_cfs > 0 else 0.0
    uw_cost_yr = uw_gal_day * (op_hours / 24.0) / 1000.0 * uw_cost

    w1, w2, w3, w4, w5, w6 = st.columns(6)
    w1.metric("Average Demand", f"{uw_avg_gpm:.1f} GPM", f"{uw_gal_day:,.0f} gal/day", delta_color="off")
    w2.metric("Connected Flow", f"{uw_connected:.0f} GPM")
    w3.metric("Design Peak", f"{uw_peak_gpm:.0f} GPM", f"factor {uw_simul:.2f}", delta_color="off")
    w4.metric("Header Size", f"{uw_pipe_in:.1f} in", f"at {uw_velocity:g} ft/s", delta_color="off")
    w5.metric("Water to Process", f"{uw_to_process_gpm:.2f} GPM", f"{uw_pct_feed:.1f}% of feed",
              delta_color="off",
              help="Adds to the evaporation load: feed + this = water to evaporate.")
    w6.metric("Water Cost", f"${uw_cost_yr:,.0f} /yr")

    if uw_pct_feed > 3.0:
        st.warning(f"Utility water entering the process is {uw_pct_feed:.1f}% of feed - "
                   "it increases distillate, vapor and blower load. Check the wash and seal "
                   "flows.")

    st.dataframe(all_uw.round(2), hide_index=True, width="stretch")

    st.divider()

    ##########################################################################
    # INSTRUMENT AIR (from the air receiver on the P&ID)
    ##########################################################################
    st.subheader("💨 Instrument Air")

    i1, i2, i3 = st.columns(3)
    with i1:
        air_scfm = st.number_input(
            "Air Demand (scfm)", min_value=0.0, step=1.0,
            value=float(pdata.get("air_demand_scfm", 20.0)), key="cu_air_scfm",
            help="Valves, actuators, level / pressure purges and diaphragm pumps. "
                 "Placeholder - total from the valve / instrument list.")
    with i2:
        air_pressure = st.number_input(
            "Header Pressure (psig)", min_value=0.0, step=5.0,
            value=float(pdata.get("air_pressure_psig", 100.0)), key="cu_air_p")
    with i3:
        air_receiver_min = st.number_input(
            "Receiver Hold-up (min at demand)", min_value=1.0, step=1.0,
            value=float(pdata.get("air_receiver_min", 10.0)), key="cu_air_min",
            help="Minutes of air to ride through a compressor trip, down to 60 psig.")

    # receiver volume: V = Q * t * Patm / (P1 - P2)  (ft3)
    p_atm = float(pdata.get("atmospheric_pressure", 14.696))
    air_receiver_ft3 = air_scfm * air_receiver_min * p_atm / max(air_pressure - 60.0, 1.0)
    air_receiver_gal = air_receiver_ft3 * 7.4805

    ia1, ia2 = st.columns(2)
    ia1.metric("Air Demand", f"{air_scfm:.0f} scfm @ {air_pressure:.0f} psig")
    ia2.metric("Receiver Volume", f"{air_receiver_gal:,.0f} gal", f"{air_receiver_ft3:,.0f} ft³",
               delta_color="off")

    st.divider()

    ##########################################################################
    # SUMMARY
    ##########################################################################
    st.subheader("📋 Chemicals & Utilities Summary")

    cu_df = pd.DataFrame({
        "Item": [
            "Sulfamic acid dose", "Sulfamic acid solution", "Acid metering pump",
            "Acid day tank", "Acid per year (dosing + CIP)",
            "Defoamer dose", "Defoamer metering pump", "Defoamer storage",
            "Utility water average / peak", "Utility water to process",
            "Utility water header", "Instrument air / receiver",
        ],
        "Value": [
            f"{acid_ppm:g} mg/L = {acid_lb_hr:,.1f} lb/hr (100%)",
            f"{acid_strength:g}% -> {acid_soln_gph:,.1f} gph",
            f"{acid_pump_gph:,.1f} gph (1 duty)",
            f"{acid_tank_gal:,.0f} gal ({acid_tank_days:g} days)",
            f"{acid_lb_yr / 2000:,.1f} ton (${acid_cost_yr:,.0f})",
            f"{df_ppm:g} mg/L = {df_lb_hr:,.2f} lb/hr ({df_gph:.2f} gph)",
            f"{df_pump_gph:.2f} gph (1 duty)",
            f"{df_storage_gal:,.0f} gal ({df_tank_days:g} days)",
            f"{uw_avg_gpm:.1f} / {uw_peak_gpm:.0f} GPM at {uw_pressure:.0f} psig",
            f"{uw_to_process_gpm:.2f} GPM ({uw_pct_feed:.1f}% of feed)",
            f"{uw_pipe_in:.1f} in at {uw_velocity:g} ft/s",
            f"{air_scfm:.0f} scfm @ {air_pressure:.0f} psig, receiver {air_receiver_gal:,.0f} gal",
        ],
    })
    st.dataframe(cu_df, hide_index=True, width="stretch")

    ##########################################################################
    # SAVE
    ##########################################################################
    pdata.update({
        "acid_dose_ppm": acid_ppm, "acid_strength_pct": acid_strength, "acid_sg": acid_sg,
        "acid_cost_lb": acid_cost, "acid_tank_days": acid_tank_days,
        "acid_pump_margin": acid_pump_margin, "acid_cip_pct": cip_conc,
        "acid_cips_per_year": int(cips_per_year), "acid_cip_fill_pct": cip_fill_pct,
        "acid_lb_hr": acid_lb_hr, "acid_solution_gph": acid_soln_gph,
        "acid_pump_gph": acid_pump_gph, "acid_tank_gal": acid_tank_gal,
        "acid_dilution_gpm": acid_dilution_gpm, "acid_lb_yr": acid_lb_yr,
        "acid_cost_yr": acid_cost_yr,

        "defoamer_dose_ppm": df_ppm, "defoamer_sg": df_sg, "defoamer_cost_lb": df_cost,
        "defoamer_tote_gal": df_tote_gal, "defoamer_pump_margin": df_pump_margin,
        "defoamer_tank_days": df_tank_days, "defoamer_lb_hr": df_lb_hr,
        "defoamer_gph": df_gph, "defoamer_pump_gph": df_pump_gph,
        "defoamer_storage_gal": df_storage_gal, "defoamer_cost_yr": df_cost_yr,

        "uw_pressure_psig": uw_pressure, "uw_velocity": uw_velocity,
        "uw_simultaneity": uw_simul, "uw_cost_kgal": uw_cost,
        "uw_consumers": uw_df.to_dict("records"),
        "uw_avg_gpm": uw_avg_gpm, "uw_peak_gpm": uw_peak_gpm,
        "uw_to_process_gpm": uw_to_process_gpm,
        "uw_to_process_lb_hr": uw_to_process_lb_hr,
        "uw_header_in": uw_pipe_in, "uw_cost_yr": uw_cost_yr,

        "air_demand_scfm": air_scfm, "air_pressure_psig": air_pressure,
        "air_receiver_min": air_receiver_min, "air_receiver_gal": air_receiver_gal,
    })

    save_project(project_name, st.session_state.project_data)
    st.success("✅ Chemicals & Utilities Auto Saved")

# ============================================================
# FINAL STEP: save the final state of every tab and refresh the project card
# (keep this as the LAST line of the script)
# ============================================================
_render_flow_slots()
_autosave()
