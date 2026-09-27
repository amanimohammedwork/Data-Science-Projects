from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import os

# DATA
KPIS = [
    ("5,646", "Historical hours analyzed"),
    ("2,762 MW", "Avg. demand — DK1"),
    ("1,767 MW", "Avg. demand — DK2"),
    ("+7.6 MW", "ENTSO-E mean error"),
    ("57.1 MW", "XGBoost MAE"),
]

ZONE_STATS = {
    "DK1 — Jutland & Funen": [
        ("Mean demand", "2,762.1 MW"),
        ("Std. deviation", "571.8 MW"),
        ("Minimum", "1,087.2 MW"),
        ("Maximum", "4,356.9 MW"),
    ],
    "DK2 — Zealand": [
        ("Mean demand", "1,767.1 MW"),
        ("Std. deviation", "354.4 MW"),
        ("Minimum", "425.2 MW"),
        ("Maximum", "2,732.8 MW"),
    ],
}

PATTERN_FINDINGS = [
    ("Seasonal", "Demand is generally higher during Q1 and declines gradually as "
                 "temperatures rise through spring and summer — consistent with "
                 "heating-driven consumption in colder months."),
    ("Daily cycle", "Demand rises through the morning and peaks around midday, "
                    "tracking periods of increased economic and human activity."),
    ("Weekly", "Demand is highest during the working week — Wednesday records the "
               "highest average demand, Sunday the lowest."),
    ("Weekday vs weekend", "Weekday demand is consistently higher than weekend "
                           "demand, in line with Denmark's industrial and "
                           "agricultural work schedules."),
]

ERROR_DATA = {
    "hour": [
        ("00", 32.3), ("01", 18.0), ("02", 14.8), ("03", -1.7), ("04", -17.7), ("05", -7.5),
        ("06", 52.2), ("07", 58.9), ("08", 35.4), ("09", 34.0), ("10", 41.9), ("11", 41.5),
        ("12", 43.2), ("13", 29.3), ("14", 6.1), ("15", 10.0), ("16", 20.6), ("17", 30.6),
        ("18", 25.3), ("19", -1.5), ("20", -25.7), ("21", -49.4), ("22", -90.1), ("23", -119.9),
    ],
    "day": [
        ("Mon", 4.9), ("Tue", -2.8), ("Wed", 5.9), ("Thu", 5.8),
        ("Fri", 4.4), ("Sat", 19.0), ("Sun", 15.4),
    ],
    "month": [
        ("Jan", 80.2), ("Feb", 8.5), ("Mar", 9.4), ("Apr", -8.9),
        ("May", -16.0), ("Jun", -9.6), ("Jul", -1.9), ("Aug", -4.9),
    ],
}

ERROR_FINDINGS = [
    ("Weekday vs weekend", "Weekday forecast error averages 76.2 MW (MAE) vs "
                           "58.9 MW on weekends — the model has a harder time "
                           "on working days."),
    ("Holidays", "Holiday forecasts run a much larger positive bias (+27.1 MW "
                 "vs +6.9 MW on regular days) — ENTSO-E tends to under-forecast "
                 "demand on Danish public holidays."),
    ("Night hours", "Hours 22:00–23:00 show both the largest error magnitude "
                    "(MAE up to 167.9 MW) and the strongest negative bias "
                    "(−119.9 MW at 23:00) — a consistent over-forecast late "
                    "at night."),
    ("Extremes", "The single largest forecast miss was +1,114.8 MW; the "
                 "largest over-forecast was −815.9 MW."),
]

# BENCHMARK
METHOD_DATA = {
    "mae":  {"Previous hour": 97.5, "Previous day": 190.8, "Previous week": 239.5, "ENTSO-E": 71.2, "XGBoost": 57.1},
    "rmse": {"Previous hour": 129.0, "Previous day": 268.6, "Previous week": 325.9, "ENTSO-E": 127.5, "XGBoost": 82.0},
    "p95":  {"Previous hour": 258.1, "Previous day": 601.4, "Previous week": 675.3, "ENTSO-E": 283.2, "XGBoost": 163.1},
    "bias": {"Previous hour": -0.2, "Previous day": 2.0, "Previous week": 22.6, "ENTSO-E": -7.6, "XGBoost": 3.9},
}
METHOD_LABELS = {"mae": "MAE (MW)", "rmse": "RMSE (MW)", "p95": "95th percentile (MW)", "bias": "Bias (MW)"}

SHAP_FINDINGS = [
    ("01", "Previous-hour demand (lag feature)", "dominant predictor"),
    ("02", "Hour of day", "SHAP ≈ 90"),
]

# Demand-quartile comparison: ENTSO-E vs XGBoost MAE, by quartile
QUARTILES = ["Low", "Medium-Low", "Medium-High", "High"]
QUARTILE_MAE = {
    "ENTSO-E": [30.1, 37.2, 41.8, 37.0],
    "XGBoost": [40.5, 46.4, 65.4, 76.1],
}

EXTREMES = {
    "Highest demand hours": [
        ("Actual 4,356.93 MW", "Forecast 4,231.00"),
        ("Actual 4,347.07 MW", "Forecast 4,282.00"),
        ("Actual 4,328.99 MW", "Forecast 4,308.00"),
    ],
    "Lowest demand hours": [
        ("Actual 1,087.22 MW", "Forecast 1,353.00"),
        ("Actual 1,110.43 MW", "Forecast 1,229.10"),
        ("Actual 1,265.46 MW", "Forecast 1,250.00"),
    ],
}

RECOMMENDATIONS = [
    "Use XGBoost as a complement, not a replacement. It wins on average and "
    "dramatically reduces worst-case error overall, but ENTSO-E holds its "
    "own — and often wins — at higher demand levels.",
    "Investigate the high-demand regime specifically. Understanding why "
    "XGBoost underperforms exactly when demand peaks is the clearest next "
    "step before wider deployment.",
    "Watch the night hours. 22:00–23:00 show the largest and most "
    "consistently biased errors in the official forecast — a natural "
    "target for a lightweight correction model.",
    "Keep the lag features. Previous-hour demand and hour-of-day dominate "
    "the model's decisions — any production pipeline needs reliable, "
    "low-latency access to the most recent observed load.",
]

COLORS = {
    "paper": "#F5F3EC", "card": "#FFFFFF", "ink": "#1B2A38", "ink_soft": "#5B6B78",
    "line": "#DED8C9", "bad": "#B54A3F", "bad_soft": "#F1DCD6",
    "good": "#2F6E5B", "good_soft": "#DCE9E2", "gold": "#B5852C", "gold_soft": "#F1E4C9",
}
FONT_FAMILY = "IBM Plex Sans, sans-serif"
HEADLINE_FAMILY = "Fraunces, serif"
MONO_FAMILY = "IBM Plex Mono, monospace"


# BUILD ILLUSTRATION
def make_error_chart(view):
    data = ERROR_DATA[view]
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [COLORS["bad"] if v >= 0 else COLORS["good"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" for v in values], textposition="outside",
        textfont=dict(family=MONO_FAMILY, size=11),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=40),
        height=max(220, 22 * len(data)),
        xaxis=dict(zeroline=True, zerolinecolor=COLORS["ink_soft"],
                   title="Mean forecast error, Actual − Forecast (MW)", gridcolor=COLORS["line"]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_method_chart(metric):
    vals = METHOD_DATA[metric]
    labels = list(vals.keys())
    values = list(vals.values())
    colors = [COLORS["good"] if l == "XGBoost" else COLORS["gold"] if l == "ENTSO-E" else COLORS["ink_soft"] for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.1f}" if metric == "bias" else f"{v:.1f}" for v in values],
        textposition="outside", textfont=dict(family=MONO_FAMILY, size=12),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=50),
        height=260,
        xaxis=dict(title=METHOD_LABELS[metric], gridcolor=COLORS["line"], zeroline=True, zerolinecolor=COLORS["ink_soft"]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_quartile_chart():
    fig = go.Figure()
    fig.add_trace(go.Bar(name="ENTSO-E", x=QUARTILES, y=QUARTILE_MAE["ENTSO-E"], marker_color=COLORS["gold"]))
    fig.add_trace(go.Bar(name="XGBoost", x=QUARTILES, y=QUARTILE_MAE["XGBoost"], marker_color=COLORS["bad"]))
    fig.update_layout(
        barmode="group",
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        yaxis=dict(title="MAE (MW)", gridcolor=COLORS["line"]),
        xaxis=dict(title="Demand quartile"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig

# APP

DemandForecasting = Dash(__name__)
DemandForecasting.title = "Denmark Electricity Demand — Forecasting Dashboard"

DemandForecasting.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,420;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            body { background-color: #F5F3EC; margin: 0; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

CARD_STYLE = {
    "background": COLORS["card"], "border": f"1px solid {COLORS['line']}",
    "borderRadius": "10px", "padding": "16px 18px", "fontSize": "14.5px", "lineHeight": "1.55",
}
SECTION_STYLE = {"maxWidth": "1000px", "margin": "0 auto", "padding": "40px 24px", "borderBottom": f"1px solid {COLORS['line']}"}
H2_STYLE = {"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "26px", "marginBottom": "8px"}
SUB_STYLE = {"color": COLORS["ink_soft"], "fontSize": "14.5px", "maxWidth": "64ch", "marginBottom": "28px", "lineHeight": "1.55"}


def kpi_card(num, label):
    return html.Div([
        html.Span(num, style={"fontFamily": MONO_FAMILY, "fontSize": "20px", "fontWeight": 600, "display": "block"}),
        html.Span(label, style={"fontSize": "11.5px", "color": COLORS["ink_soft"]}),
    ], style={"background": COLORS["card"], "padding": "18px 14px"})


def zone_card(title, stats):
    return html.Div([
        html.H3(title, style={"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "17px", "marginBottom": "12px"}),
        *[html.Div([html.Span(label), html.Span(val, style={"fontFamily": MONO_FAMILY, "fontWeight": 600})],
                   style={"display": "flex", "justifyContent": "space-between", "fontSize": "13.5px",
                          "padding": "6px 0", "borderBottom": f"1px solid {COLORS['line']}"})
          for label, val in stats],
    ], style={"background": COLORS["card"], "border": f"1px solid {COLORS['line']}", "borderRadius": "10px", "padding": "18px 20px"})


def finding_card(title, text):
    return html.Div([html.B(title + ". "), text], style=CARD_STYLE)


def callout(children):
    return html.Div(children, style={"background": COLORS["gold_soft"], "borderRadius": "10px",
                                       "padding": "18px 20px", "fontSize": "14.5px", "lineHeight": "1.6", "marginTop": "20px"})


header = html.Header([
    html.P("Denmark Electricity Demand — Forecasting Project",
           style={"fontFamily": MONO_FAMILY, "fontSize": "12.5px", "color": COLORS["ink_soft"]}),
    html.H1("Can a model beat the grid operator's own forecast?",
            style={"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "40px",
                   "lineHeight": "1.1", "maxWidth": "16ch", "margin": "0 0 16px"}),
    html.P("An interactive summary of an hourly electricity-demand forecasting project for "
           "Denmark's two price zones (DK1 / DK2), using ENTSO-E data — covering demand "
           "patterns, where the official day-ahead forecast breaks down, and whether a "
           "gradient-boosted model can do better.",
           style={"fontSize": "16.5px", "color": COLORS["ink_soft"], "maxWidth": "64ch", "lineHeight": "1.6"}),
    html.Div([kpi_card(n, l) for n, l in KPIS],
             style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "1px",
                    "background": COLORS["line"], "border": f"1px solid {COLORS['line']}",
                    "marginTop": "28px", "borderRadius": "10px", "overflow": "hidden"}),
], style={**SECTION_STYLE, "paddingTop": "50px"})


overview_section = html.Section([
    html.H2("Two grids, two demand profiles", style=H2_STYLE),
    html.P("DK1 (Jutland/Funen) and DK2 (Zealand) are Denmark's two bidding zones. Both are "
           "tracked hourly from Jan 1 through Aug 24, 2026 — 5,646 hours with observed demand, "
           "out of 8,760 in the full dataset.", style=SUB_STYLE),
    html.Div([zone_card(title, stats) for title, stats in ZONE_STATS.items()],
             style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "24px"}),
    finding_card("Zone comparison", "DK1 scores higher across every metric — likely because "
                 "Jutland is geographically larger and hosts heavier industrial and "
                 "agricultural activity than Zealand."),
], style=SECTION_STYLE)


patterns_section = html.Section([
    html.H2("The demand pattern", style=H2_STYLE),
    html.P("Electricity demand follows a strongly seasonal and cyclical pattern: higher in "
           "the colder first quarter, declining through the warmer months, rising through "
           "the morning to a midday peak, and consistently higher on weekdays than weekends.",
           style=SUB_STYLE),
    html.Div([finding_card(t, d) for t, d in PATTERN_FINDINGS],
              style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px"}),
], style=SECTION_STYLE)


error_section = html.Section([
    html.H2("Where the official forecast breaks down", style=H2_STYLE),
    html.P("ENTSO-E's day-ahead forecast is precise on average — a mean error of just "
           "+7.6 MW against demand that averages ~2,760 MW — but that average hides real "
           "structure. Forecast Error is Actual − Forecast, so positive bars mean demand "
           "came in higher than predicted.", style=SUB_STYLE),
    dcc.RadioItems(
        id="error-view",
        options=[{"label": " By hour of day", "value": "hour"},
                 {"label": " By day of week", "value": "day"},
                 {"label": " By month", "value": "month"}],
        value="hour", inline=True,
        style={"marginBottom": "16px"},
        inputStyle={"marginRight": "6px", "marginLeft": "16px"},
    ),
    dcc.Graph(id="error-chart", config={"displayModeBar": False}),
    html.Div([finding_card(t, d) for t, d in ERROR_FINDINGS],
              style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px", "marginTop": "24px"}),
], style=SECTION_STYLE)


benchmark_section = html.Section([
    html.H2("Benchmarking the forecast", style=H2_STYLE),
    html.P("Five methods scored on the same held-out period: three naive baselines "
           "(repeat demand from the previous hour, day, or week), ENTSO-E's own day-ahead "
           "forecast, and a tuned XGBoost model.", style=SUB_STYLE),
    dcc.RadioItems(
        id="metric-choice",
        options=[{"label": " " + METHOD_LABELS[k], "value": k} for k in METHOD_DATA],
        value="mae", inline=True,
        style={"marginBottom": "16px"},
        inputStyle={"marginRight": "6px", "marginLeft": "16px"},
    ),
    dcc.Graph(id="method-chart", config={"displayModeBar": False}),
    callout([html.B("XGBoost improves on ENTSO-E "), "by 19.8% on MAE, 35.7% on RMSE, and "
             "42.4% on the 95th-percentile error — it shrinks both the average and the "
             "worst-case misses."]),
], style=SECTION_STYLE)


shap_section = html.Section([
    html.H2("What the model leans on", style=H2_STYLE),
    html.P("SHAP analysis on the trained XGBoost model ranks feature importance by average "
           "impact on individual predictions.", style=SUB_STYLE),
    html.Div([
        html.Div([
            html.Span(rank, style={"fontFamily": MONO_FAMILY, "color": COLORS["gold"], "fontWeight": 600, "fontSize": "15px"}),
            html.Span(name, style={"fontSize": "14.5px"}),
            html.Span(badge, style={"fontFamily": MONO_FAMILY, "fontSize": "12.5px", "color": COLORS["ink_soft"],
                                     "background": COLORS["gold_soft"], "padding": "4px 10px", "borderRadius": "999px"}),
        ], style={"display": "grid", "gridTemplateColumns": "34px 1fr auto", "alignItems": "center", "gap": "14px",
                  **CARD_STYLE, "marginBottom": "12px"})
        for rank, name, badge in SHAP_FINDINGS
    ]),
    html.P("Both the most recent observed demand and the time of day carry substantial "
           "weight — the model is, in effect, learning a smoothed version of the daily "
           "load curve plus short-term momentum.", style={**SUB_STYLE, "marginTop": "16px", "marginBottom": "0"}),
], style=SECTION_STYLE)


quartile_section = html.Section([
    html.H2("The catch: performance varies by demand level", style=H2_STYLE),
    html.P("Splitting the test set into demand quartiles reveals something the aggregate "
           "numbers hide: ENTSO-E is more consistent — and often better — exactly when "
           "demand is highest.", style=SUB_STYLE),
    dcc.Graph(figure=make_quartile_chart(), config={"displayModeBar": False}),
    callout([html.B("Aggregate metrics can mislead. "), "ENTSO-E outperforms XGBoost in "
             "every demand quartile shown here, even though XGBoost wins on overall MAE — "
             "worth digging into further before deploying XGBoost as a full replacement "
             "rather than a complement to the official forecast."]),
], style=SECTION_STYLE)


extremes_section = html.Section([
    html.H2("The extremes", style=H2_STYLE),
    html.P("The three largest and smallest recorded demand hours in DK1 put the "
           "forecasting challenge in perspective.", style=SUB_STYLE),
    html.Div([
        html.Div([
            html.H3(title, style={"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "16px", "marginBottom": "12px"}),
            *[html.Div([html.Span(a), html.Span(b)],
                       style={"display": "flex", "justifyContent": "space-between", "fontSize": "13px",
                              "padding": "7px 0", "borderBottom": f"1px solid {COLORS['line']}",
                              "fontFamily": MONO_FAMILY})
              for a, b in rows],
        ]) for title, rows in EXTREMES.items()
    ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px"}),
], style=SECTION_STYLE)


reco_section = html.Section([
    html.H2("What this means for deployment", style=H2_STYLE),
    html.Div([
        html.Div([
            html.Span(f"{i+1:02d}", style={"fontFamily": MONO_FAMILY, "color": COLORS["gold"], "fontWeight": 600, "marginRight": "14px"}),
            html.Span(text),
        ], style={**CARD_STYLE, "display": "flex", "marginBottom": "14px"})
        for i, text in enumerate(RECOMMENDATIONS)
    ]),
], style=SECTION_STYLE)


footer = html.Footer([
    html.P("Source: ENTSO-E day-ahead load forecasts and actual demand for Denmark's DK1 "
           "and DK2 bidding zones, Jan 1 – Aug 24, 2026 (8,760 hourly rows, 5,646 with "
           "observed actual demand). Modeled with pandas, scikit-learn, XGBoost, LightGBM "
           "and SHAP; baselines use a chronological 80/20 train-test split validated with "
           "time-series cross-validation. LightGBM was also evaluated and performed "
           "comparably to XGBoost; XGBoost was carried forward for its more stable "
           "training behavior. All figures above are taken directly from the project "
           "notebook's computed output.",
           style={"fontSize": "12.5px", "color": COLORS["ink_soft"], "lineHeight": "1.6", "maxWidth": "74ch"}),
], style={**SECTION_STYLE, "borderBottom": "none"})


DemandForecasting.layout = html.Div([
    header, overview_section, patterns_section, error_section, benchmark_section,
    shap_section, quartile_section, extremes_section, reco_section, footer,
], style={"fontFamily": FONT_FAMILY, "color": COLORS["ink"], "background": COLORS["paper"]})


# APP
@DemandForecasting.callback(Output("error-chart", "figure"), Input("error-view", "value"))
def update_error_chart(view):
    return make_error_chart(view)


@DemandForecasting.callback(Output("method-chart", "figure"), Input("metric-choice", "value"))
def update_method_chart(metric):
    return make_method_chart(metric)

server = DemandForecasting
if __name__ == "__main__":
    DemandForecasting.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8051)),
        debug=False
    )