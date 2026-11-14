"""
Professional PDF design report for the ZLD forced-circulation MVR crystallizer app.

    from report_pdf import collect_report, build_design_report_pdf

collect_report(pdata)            -> dict of report sections (rows of strings), used by
                                    both the Streamlit Report tab and the PDF
build_design_report_pdf(pdata, meta) -> PDF bytes for st.download_button

Missing values print as "-" (never as a misleading 0).
"""

from datetime import date
from io import BytesIO

from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

# ---------------------------------------------------------------------------
# Palette / styles
# ---------------------------------------------------------------------------
NAVY = colors.HexColor("#14325C")
ACCENT = colors.HexColor("#1F7A8C")
LIGHT = colors.HexColor("#EEF3F8")
ZEBRA = colors.HexColor("#F7F9FC")
GRID = colors.HexColor("#C9D3DF")
TEXT = colors.HexColor("#1E2630")
MUTED = colors.HexColor("#5B6878")
OK_C = colors.HexColor("#2E7D32")
CHECK_C = colors.HexColor("#B26A00")
BAD_C = colors.HexColor("#C62828")

_S = {
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=22, leading=27,
                            textColor=NAVY, spaceAfter=4),
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=12, leading=16,
                               textColor=MUTED, spaceAfter=14),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13, leading=17,
                         textColor=NAVY, spaceBefore=14, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=9, leading=12.5,
                           textColor=TEXT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7.5, leading=10,
                            textColor=MUTED),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.5, leading=11,
                           textColor=TEXT),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                            textColor=TEXT),
    "head": ParagraphStyle("head", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                           textColor=colors.white),
    "kpi_v": ParagraphStyle("kpi_v", fontName="Helvetica-Bold", fontSize=14, leading=17,
                            textColor=NAVY, alignment=TA_CENTER),
    "kpi_l": ParagraphStyle("kpi_l", fontName="Helvetica", fontSize=7.5, leading=9.5,
                            textColor=MUTED, alignment=TA_CENTER),
}


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------
def _num(pdata, *keys):
    """First key present with a usable number, else None."""
    for k in keys:
        v = pdata.get(k)
        if v is None or v == "":
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _f(v, spec=",.1f", unit=""):
    if v is None:
        return "-"
    s = format(v, spec)
    return f"{s} {unit}".strip()


def _usd(v):
    if v is None:
        return "-"
    return f"-${-v:,.0f} /yr" if v < 0 else f"${v:,.0f} /yr"


def _pct_of(calc, vendor):
    if calc is None or not vendor:
        return "-"
    return f"{calc / vendor * 100:.0f}% of vendor"


def _c_f(c):
    """'87.3 °C / 189.1 °F'"""
    if c is None:
        return "-", "-"
    return f"{c:,.1f}", f"{c * 9 / 5 + 32:,.1f}"


def _status(level):
    return {"OK": OK_C, "CHECK": CHECK_C, "REVIEW": BAD_C}.get(level, MUTED)


# ---------------------------------------------------------------------------
# Data collection (shared by Streamlit tab and PDF)
# ---------------------------------------------------------------------------
def collect_report(pdata):
    g = lambda *k: _num(pdata, *k)

    feed_gpm = g("design_flow", "feed_flow")
    feed_lb = g("feed_lb_hr")
    dist_lb = g("distillate_lb_hr_calc")
    dist_gpm = g("distillate_gpm_calc", "distillate_flow")
    salt_lb = g("salt_lb_hr_calc", "crystal_lb_hr")
    cake_w = g("cake_water_lb_hr")
    feed_water = (feed_lb - (salt_lb or 0.0)) if feed_lb else None
    recovery = (dist_lb / feed_water * 100) if (dist_lb and feed_water) else g("water_recovery")

    boil_c = g("crystallizer_boiling_temp")
    flash_f = g("flash_delta_t", "flash_delta_t_input")
    hx_out_c = boil_c + flash_f / 1.8 if (boil_c is not None and flash_f is not None) else None
    cond_c = g("mvr_discharge_sat_temp")

    blower_kw = g("blower_power_kw", "blower_kw")
    shaft_kw = g("blower_shaft_kw")
    shaft_hp = shaft_kw / 0.746 if shaft_kw is not None else None
    recirc_kw = g("recirc_kw")
    feed_pump_hp = g("feed_pump_hp")
    feed_pump_kw = feed_pump_hp * 0.746 if feed_pump_hp is not None else None
    boiler_kw = g("boiler_power_kw", "boiler_kw")

    basis_name = {"vacom": "Vacom P&ID line list (Rev B)",
                  "ucc": "UCC 7.25 psia basis"}.get(pdata.get("design_basis_name"), "User inputs")

    kpis = [
        (_f(feed_gpm, ",.1f", "GPM"), "Design feed flow"),
        (_f(dist_gpm, ",.1f", "GPM"), "Distillate"),
        (_f(recovery, ".1f", "%"), "Water recovery"),
        (_f(salt_lb, ",.0f", "lb/hr"), "Dry salt"),
        (_f(blower_kw, ",.0f", "kW"), "MVR blower power"),
        (_f(g("boiler_capacity_calc", "boiler_capacity_lb_hr"), ",.0f", "lb/hr"), "Boiler capacity"),
    ]

    basis = [
        ["Design basis", basis_name],
        ["Capacity case", str(pdata.get("capacity_basis", "-"))],
        ["Design feed flow", _f(feed_gpm, ",.1f", "GPM")],
        ["Feed TDS", _f(g("feed_tds", "tds"), ",.0f", "mg/L")],
        ["Feed temperature", _f(g("fhx_feed_temp_c", "feed_temp_c"), ".1f", "°C")],
        ["Feed specific gravity", _f(g("feed_sg"), ".3f")],
        ["Crystallizer pressure (setpoint)", _f(g("vessel_pressure_sp"), ".2f", "psia")],
        ["Boiling point elevation", _f(g("boiling_point_elevation"), ".1f", "°C")],
        ["Blower discharge pressure", _f(g("blower_discharge_psig"), ".2f", "psig")],
        ["Blower pressure ratio / dP",
         f"{_f(g('mvr_pressure_ratio'), '.2f')}  /  {_f(g('blower_dp_setpoint'), '.2f', 'psi')}"],
        ["Primary HX flash dT", _f(flash_f, ".1f", "°F")],
        ["Boiler steam pressure", _f(g("boiler_pressure"), ".0f", "psig")],
        ["Operating hours", _f(g("mvr_operating_hours", "operating_hours"), ",.0f", "hr/yr")],
    ]

    mass = [
        ["Feed", _f(feed_lb, ",.0f"), _f(feed_gpm, ",.1f")],
        ["Dry salt", _f(salt_lb, ",.0f"), "-"],
        ["Cake moisture", _f(cake_w, ",.0f"), "-"],
        ["Distillate / vapor to blower", _f(dist_lb, ",.0f"), _f(dist_gpm, ",.1f")],
        ["Recirculation flow", _f(g("recirc_lbhr"), ",.0f"),
         _f(g("recirculation_flow", "recirculation_flow_required"), ",.0f")],
    ]

    temps = []
    for label, c in [
        ("Feed to pre-HX", g("fhx_feed_temp_c", "feed_temp_c")),
        ("Feed from pre-HX", g("preheat_out_c")),
        ("Distillate to pre-HX", g("distillate_temp_c")),
        ("Distillate from pre-HX", g("distillate_out_c")),
        ("Vessel water saturation", g("saturation_temp")),
        ("Brine boiling / primary HX inlet", boil_c),
        ("Primary HX brine outlet", hx_out_c),
        ("Compressed vapor condensing", cond_c),
        ("Blower discharge (superheated)", g("mvr_discharge_temp_actual")),
    ]:
        temps.append([label, *_c_f(c)])

    equipment = [
        ["E-120/130", "Feed preheaters",
         f"{_f(g('feed_hx_duty'), '.2f', 'MMBTU/hr')}",
         f"{_f(g('feed_hx_area'), ',.0f', 'ft²')}, LMTD {_f(g('feed_hx_lmtd_c'), '.1f', '°C')}",
         "-"],
        ["V-200", "Crystallizer vessel",
         f"{_f(g('vessel_pressure_sp'), '.2f', 'psia')}, {_f(boil_c, '.1f', '°C')}",
         _f(g("crystallizer_volume"), ",.0f", "gal"), "-"],
        ["P-201", "Recirculation pump",
         f"{_f(g('recirculation_flow'), ',.0f', 'GPM')} @ {_f(g('recirc_tdh'), ',.0f', 'ft')}",
         f"{_f(g('recirc_motor_hp'), ',.0f', 'hp')} motor",
         _f(recirc_kw, ",.0f", "kW")],
        ["E-220", "Primary heat exchanger",
         _f(g("cryst_hx_duty"), ".2f", "MMBTU/hr"),
         f"{_f(g('cryst_hx_area'), ',.0f', 'ft²')}, {_f(g('tube_count'), ',.0f')} tubes",
         "-"],
        ["B-310", "MVR blower",
         f"{_f(g('mvr_vapor_lb_hr', 'distillate_lb_hr_calc'), ',.0f', 'lb/hr')}, PR {_f(g('mvr_pressure_ratio'), '.2f')}",
         f"{_f(shaft_hp, ',.0f', 'shaft hp')}",
         _f(blower_kw, ",.0f", "kW")],
        ["Boiler", "Electric steam boiler",
         f"{_f(g('boiler_capacity_calc'), ',.0f', 'lb/hr')} @ {_f(g('boiler_pressure'), '.0f', 'psig')}",
         str(pdata.get("boiler_governing_case", "-")),
         _f(boiler_kw, ",.0f", "kW")],
        ["Feed pump", "Feed pumps",
         f"{_f(feed_gpm, ',.1f', 'GPM')} @ {_f(g('feed_tdh'), ',.0f', 'ft')}",
         f"{_f(g('feed_pumps'), '.0f')} installed",
         _f(feed_pump_kw, ",.1f", "kW")],
        ["Filter", "Oberlin filter press",
         _f(g("cake_production_lb_hr"), ",.0f", "lb/hr cake"),
         _f(g("required_filter_area"), ",.0f", "ft²"), "-"],
    ]

    power = [(n, v) for n, v in [
        ("MVR blower", blower_kw),
        ("Recirculation pump", recirc_kw),
        ("Feed pump", feed_pump_kw),
    ] if v]
    total_kw = sum(v for _, v in power) if power else None

    energy = [
        ["Continuous electrical load", _f(total_kw, ",.0f", "kW")],
        ["Specific energy (blower)", _f(g("specific_energy"), ",.1f", "kWh/m³ distillate")],
        ["MVR COP", _f(g("mvr_cop"), ".1f")],
        ["Steam displaced by MVR", _f(g("steam_saved_lb_hr"), ",.0f", "lb/hr")],
        ["Annual MVR energy cost", _usd(g("annual_mvr_cost"))],
        ["Annual net savings vs steam", _usd(g("annual_net_savings"))],
        ["CO2 reduction", _f(g("mvr_co2_reduction_t"), ",.0f", "t/yr")],
        ["Boiler installed power (start-up / makeup)", _f(boiler_kw, ",.0f", "kW")],
    ]

    vendor = [
        ["MVR blower shaft power", _f(shaft_hp, ",.0f", "hp"),
         _f(g("vendor_blower_hp"), ",.0f", "hp motor"),
         _f(g("vendor_blower_load_pct"), ".0f", "% load")],
        ["Max feed on vendor blower", "-", _f(g("vendor_blower_max_feed_gpm"), ",.1f", "GPM"), "-"],
    ]
    for label, calc, vend, unit, spec in [
        ("Recirculation flow", g("recirculation_flow"), g("vendor_flow_calc_gpm"), "GPM", ",.0f"),
        ("Recirculation pump TDH", g("recirc_tdh"), g("vendor_tdh_calc_ft"), "ft", ",.0f"),
        ("Recirculation pump motor", g("recirc_motor_hp"), g("vendor_motor_hp"), "hp", ",.0f"),
        ("Boiler capacity", g("boiler_capacity_calc"), g("boiler_capacity"), "lb/hr", ",.0f"),
        ("Distillate", dist_gpm, g("design_distillate_flow"), "GPM", ",.1f"),
        ("Salt production", salt_lb, g("salt_production_rate"), "lb/hr", ",.0f"),
    ]:
        vendor.append([label, _f(calc, spec, unit), _f(vend, spec, unit), _pct_of(calc, vend)])

    cap = str(pdata.get("capacity_basis", "") or "")
    if feed_gpm is not None and cap and f"({feed_gpm:g} GPM)" not in cap:
        cap = f"Custom ({feed_gpm:g} GPM)"
    basis[1][1] = cap or "-"

    # ---- automatic design checks ----
    checks = []

    def add(item, level, comment):
        checks.append([item, level, comment])

    load = g("vendor_blower_load_pct")
    if load is not None:
        if load > 100:
            add("MVR blower motor", "REVIEW", f"Required shaft power is {load:.0f}% of vendor motor.")
        elif load > 90:
            add("MVR blower motor", "CHECK", f"Motor load {load:.0f}% (above 90% continuous).")
        else:
            add("MVR blower motor", "OK", f"Motor load {load:.0f}%.")

    dt = g("primary_hx_driving_dt")
    if dt is not None:
        if dt <= 0:
            add("Primary HX driving dT", "REVIEW", "Condensing temp not above boiling temp.")
        elif dt < 8:
            add("Primary HX driving dT", "CHECK", f"{dt:.1f} °C - large HX area required.")
        else:
            add("Primary HX driving dT", "OK", f"{dt:.1f} °C.")

    vel = g("hx_tube_velocity_actual", "hx_tube_velocity")
    if vel is not None:
        add("Primary HX tube velocity", "OK" if 5 <= vel <= 10 else "CHECK",
            f"{vel:.1f} ft/s (target 5-10 ft/s).")

    pre_out = g("preheat_out_c")
    if pre_out is not None and boil_c is not None:
        margin = boil_c - pre_out
        add("Feed preheater outlet", "OK" if margin >= 3 else "CHECK",
            f"{margin:.1f} °C below boiling" + ("" if margin >= 3 else " - flashing risk."))
    lmtd = g("feed_hx_lmtd_c")
    if lmtd is not None:
        add("Feed preheater LMTD", "OK" if lmtd >= 5 else "CHECK",
            f"{lmtd:.1f} °C" + ("" if lmtd >= 5 else " - very large exchanger."))

    bc, bt = g("boiler_capacity_calc"), g("boiler_capacity")
    if bc is not None and bt:
        add("Boiler capacity", "OK" if bc <= bt else "CHECK",
            f"Required {bc:,.0f} vs vendor {bt:,.0f} lb/hr.")

    na, nr = g("pump_npsha_ft"), g("pump_npshr_ft")
    if na is not None and nr is not None:
        add("Recirc pump NPSH", "OK" if na - nr >= 3 else "CHECK",
            f"NPSHa {na:.1f} ft vs NPSHr {nr:.1f} ft.")

    clos = g("overall_closure")
    if clos is not None:
        add("Mass balance closure", "OK" if clos >= 99 else ("CHECK" if clos >= 95 else "REVIEW"),
            f"{clos:.1f}%.")

    return dict(kpis=kpis, basis=basis, mass=mass, temps=temps, equipment=equipment,
                power=power, total_kw=total_kw, energy=energy, vendor=vendor,
                checks=checks, basis_name=basis_name)


# ---------------------------------------------------------------------------
# PDF building blocks
# ---------------------------------------------------------------------------
def _esc(text):
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _p(text, style="cell"):
    return Paragraph(_esc(text), _S[style])


def _table(header, rows, widths, align_right_from=None, bold_first=True):
    data = [[_p(h, "head") for h in header]]
    for r in rows:
        data.append([_p(c, "cellb" if (i == 0 and bold_first) else "cell") for i, c in enumerate(r)])
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def _section(n, title):
    return Paragraph(f"{n}.&nbsp;&nbsp;{title}", _S["h1"])


def _kpi_tiles(kpis, width):
    row_v = [_p(v, "kpi_v") for v, _ in kpis]
    row_l = [_p(l, "kpi_l") for _, l in kpis]
    w = width / len(kpis)
    t = Table([row_v, row_l], colWidths=[w] * len(kpis))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
        ("LINEABOVE", (0, 0), (-1, 0), 2.5, ACCENT),
        ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white),
        ("TOPPADDING", (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 9),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def _power_chart(power, total, width):
    if not power:
        return None
    row_h, label_w, val_w = 18, 120, 70
    bar_w = width - label_w - val_w
    h = row_h * len(power) + 6
    d = Drawing(width, h)
    vmax = max(v for _, v in power)
    for i, (name, v) in enumerate(power):
        y = h - (i + 1) * row_h
        d.add(String(0, y + 5, name, fontName="Helvetica", fontSize=8.5, fillColor=TEXT))
        d.add(Rect(label_w, y + 2, bar_w, 11, fillColor=LIGHT, strokeColor=None))
        d.add(Rect(label_w, y + 2, bar_w * v / vmax, 11, fillColor=ACCENT, strokeColor=None))
        share = v / total * 100 if total else None
        pct = "" if share is None else ("  (<1%)" if share < 1 else f"  ({share:.0f}%)")
        d.add(String(label_w + bar_w + 6, y + 5, f"{v:,.0f} kW{pct}",
                     fontName="Helvetica-Bold", fontSize=8.5, fillColor=TEXT))
    return d


def _checks_table(checks, widths):
    data = [[_p(h, "head") for h in ("Check", "Status", "Comment")]]
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID),
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for i, (item, level, comment) in enumerate(checks, start=1):
        data.append([_p(item, "cellb"),
                     Paragraph(level, ParagraphStyle("st", parent=_S["cellb"],
                                                     textColor=colors.white, alignment=TA_CENTER)),
                     _p(comment)])
        style.append(("BACKGROUND", (1, i), (1, i), _status(level)))
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle(style))
    return t


class _NumberedCanvas(rl_canvas.Canvas):
    """Adds header/footer with 'Page x of y' after all pages are known."""

    meta = {}

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._saved)
        for state in self._saved:
            self.__dict__.update(state)
            self._decorate(total)
            super().showPage()
        super().save()

    def _decorate(self, total):
        m = self.meta
        w, h = letter
        self.saveState()
        # header
        self.setFillColor(NAVY)
        self.rect(0, h - 0.42 * inch, w, 0.42 * inch, fill=1, stroke=0)
        self.setFillColor(colors.white)
        self.setFont("Helvetica-Bold", 10)
        self.drawString(0.6 * inch, h - 0.27 * inch, m.get("company", ""))
        self.setFont("Helvetica", 9)
        self.drawRightString(w - 0.6 * inch, h - 0.27 * inch,
                             "MVR Crystallizer Design Report")
        self.setFillColor(ACCENT)
        self.rect(0, h - 0.46 * inch, w, 0.04 * inch, fill=1, stroke=0)
        # footer
        self.setStrokeColor(GRID)
        self.setLineWidth(0.5)
        self.line(0.6 * inch, 0.55 * inch, w - 0.6 * inch, 0.55 * inch)
        self.setFillColor(MUTED)
        self.setFont("Helvetica", 7.5)
        left = "  |  ".join(x for x in [m.get("doc_no", ""), f"Rev {m.get('revision', '')}",
                                        m.get("date", "")] if x.strip())
        self.drawString(0.6 * inch, 0.38 * inch, left)
        self.drawCentredString(w / 2, 0.38 * inch, m.get("confidential", ""))
        self.drawRightString(w - 0.6 * inch, 0.38 * inch,
                             f"Page {self._pageNumber} of {total}")
        self.restoreState()


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------
def build_design_report_pdf(pdata, meta=None):
    meta = dict(meta or {})
    meta.setdefault("company", "")
    meta.setdefault("date", date.today().strftime("%d %b %Y"))
    meta.setdefault("revision", "A")
    meta.setdefault("confidential", "CONFIDENTIAL - FOR PROPOSAL USE")
    r = collect_report(pdata)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.6 * inch,
                            rightMargin=0.6 * inch, topMargin=0.75 * inch,
                            bottomMargin=0.75 * inch,
                            title=meta.get("title", "MVR Crystallizer Design Report"),
                            author=meta.get("prepared_by", ""),
                            subject=meta.get("project", ""))
    W = doc.width
    story = []

    # ---- title block ----
    story.append(Paragraph(_esc(meta.get("title", "MVR Crystallizer Design Report")), _S["title"]))
    story.append(Paragraph(_esc(
        meta.get("subtitle", "ZLD Forced-Circulation MVR Crystallizer - Process Design Summary")),
        _S["subtitle"]))

    info = [
        ("Project", meta.get("project", "-")), ("Project No.", meta.get("project_no", "-")),
        ("Client / Site", meta.get("client", "-")), ("Document No.", meta.get("doc_no", "-")),
        ("Vendor", meta.get("vendor", "-")), ("Revision / Date",
                                              f"{meta['revision']}  /  {meta['date']}"),
        ("Prepared by", meta.get("prepared_by", "-")), ("Checked by", meta.get("checked_by", "-")),
    ]
    rows = []
    for i in range(0, len(info), 2):
        (a, b), (c, d) = info[i], info[i + 1]
        rows.append([_p(a, "small"), _p(b, "cellb"), _p(c, "small"), _p(d, "cellb")])
    t = Table(rows, colWidths=[0.95 * inch, W / 2 - 0.95 * inch, 1.05 * inch, W / 2 - 1.05 * inch])
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("BACKGROUND", (0, 0), (0, -1), LIGHT),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story += [t, Spacer(1, 14)]

    # ---- 1 executive summary ----
    story.append(_section(1, "Executive Summary"))
    story.append(_kpi_tiles(r["kpis"], W))
    story.append(Spacer(1, 8))
    summary = _esc(meta["summary"]) if meta.get("summary") else (
        f"Design basis: <b>{_esc(r['basis_name'])}</b>. The crystallizer operates at "
        f"{_f(_num(pdata, 'vessel_pressure_sp'), '.2f', 'psia')} with brine boiling at "
        f"{_f(_num(pdata, 'crystallizer_boiling_temp'), '.1f', '°C')}. Vapor is compressed by the "
        f"MVR blower to {_f(_num(pdata, 'blower_discharge_psig'), '.1f', 'psig')} and condenses at "
        f"{_f(_num(pdata, 'mvr_discharge_sat_temp'), '.1f', '°C')} in the primary heat exchanger. "
        "Distillate preheats the feed; the electric boiler supplies start-up and makeup steam."
    )
    story.append(Paragraph(summary, _S["body"]))

    # ---- 2 design basis ----
    story.append(_section(2, "Design Basis"))
    story.append(_table(["Parameter", "Value"], r["basis"], [W * 0.45, W * 0.55]))

    # ---- 3 mass balance + 4 temperatures ----
    story.append(KeepTogether([
        _section(3, "Mass Balance"),
        _table(["Stream", "lb/hr", "GPM"], r["mass"], [W * 0.5, W * 0.25, W * 0.25]),
    ]))
    story.append(KeepTogether([
        _section(4, "Process Temperatures"),
        _table(["Stream", "°C", "°F"], r["temps"], [W * 0.5, W * 0.25, W * 0.25]),
    ]))

    # ---- 5 equipment ----
    story.append(_section(5, "Major Equipment Summary"))
    story.append(_table(["Tag", "Equipment", "Design duty", "Size / rating", "Power"],
                        r["equipment"],
                        [W * 0.11, W * 0.2, W * 0.27, W * 0.27, W * 0.15]))

    # ---- 6 energy ----
    block = [_section(6, "Energy & Utilities")]
    chart = _power_chart(r["power"], r["total_kw"], W)
    if chart:
        block += [Paragraph("Continuous electrical load", _S["small"]), Spacer(1, 3), chart,
                  Spacer(1, 8)]
    block.append(_table(["Item", "Value"], r["energy"], [W * 0.55, W * 0.45]))
    story.append(KeepTogether(block))

    # ---- 7 vendor comparison ----
    story.append(KeepTogether([
        _section(7, "Vendor Comparison" + (f" - {_esc(meta['vendor'])}" if meta.get("vendor") else "")),
        _table(["Item", "Calculated", "Vendor", "Remark"], r["vendor"],
               [W * 0.34, W * 0.22, W * 0.24, W * 0.2]),
    ]))

    # ---- 8 design checks ----
    if r["checks"]:
        story.append(KeepTogether([
            _section(8, "Design Checks"),
            _checks_table(r["checks"], [W * 0.3, W * 0.13, W * 0.57]),
        ]))

    # ---- 9 notes ----
    notes = [n.strip() for n in (meta.get("notes") or "").splitlines() if n.strip()]
    notes += [
        "Saturation temperatures from IAPWS-IF97; blower power by polytropic compression "
        "(k = 1.33) at the stated efficiency.",
        "Values are preliminary process-design estimates and must be confirmed against the "
        "vendor heat and material balance and equipment datasheets.",
    ]
    story.append(_section(9, "Notes & Assumptions"))
    for i, n in enumerate(notes, 1):
        story.append(Paragraph(f"{i}.&nbsp;&nbsp;{_esc(n)}", _S["body"]))
        story.append(Spacer(1, 2))

    # ---- sign-off ----
    story.append(Spacer(1, 16))
    sign = Table(
        [[_p("Prepared", "small"), _p("Checked", "small"), _p("Approved", "small")],
         [_p(meta.get("prepared_by", ""), "cellb"), _p(meta.get("checked_by", ""), "cellb"),
          _p(meta.get("approved_by", ""), "cellb")],
         [_p("Signature / date", "small")] * 3],
        colWidths=[W / 3] * 3, rowHeights=[14, 30, 14])
    sign.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.6, GRID),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, GRID),
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(KeepTogether([sign]))

    canvas_cls = type("_Canvas", (_NumberedCanvas,), {"meta": meta})
    doc.build(story, canvasmaker=canvas_cls)
    return buf.getvalue()
