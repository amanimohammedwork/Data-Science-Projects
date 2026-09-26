from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import os

TOTAL_CUSTOMERS = 10000
CHURNED = 2037
RETAINED = TOTAL_CUSTOMERS - CHURNED

KPIS = [
    ("10,000", "Customers analyzed"),
    ("20.4%", "Churn rate"),
    ("51.5%", "Active members"),
    ("38.9", "Avg. customer age"),
    ("$76.5K", "Avg. account balance"),
]

CORRELATIONS = [
    ("Age", 0.29),
    ("Located in Germany", 0.17),
    ("Account balance", 0.12),
    ("Is active member", -0.16),
    ("Gender: male", -0.11),
]

SHAP_VALUES = [
    ("Age", 0.80, "> 0.80"),
    ("Number of products", 0.79, "≈ 0.79"),
    ("Active member", 0.43, "≈ 0.43"),
]

MODELS = {
    "xgb": {
        "label": "XGBoost",
        "accuracy": 0.8705,
        "precision": 0.7597,
        "recall": 0.4987,
        "f1": 0.6022,
        "cv_f1": 0.5934,
        "tn": 1545, "fp": 62, "fn": 197, "tp": 196,
        "note": ("XGBoost catches about half of all churners (recall 49.9%) "
                 "but is highly reliable when it does flag one "
                 "(precision 76.0%) — AUC 0.87."),
    },
    "log": {
        "label": "Logistic Regression",
        "accuracy": 0.728,
        "precision": 0.3953,
        "recall": 0.7252,
        "f1": 0.5117,
        "cv_f1": 0.4968,
        "tn": 1171, "fp": 436, "fn": 108, "tp": 285,
        "note": ("Logistic regression flags far more churners "
                 "(recall 72.5%) but with many more false alarms "
                 "(precision 39.5%) — over 3x XGBoost's false-positive count."),
    },
}

FINDINGS = [
    ("Age", "Customers aged 30–40 make up most of the retained base, but "
             "churn rises sharply relative to group size for customers "
             "aged 40–60."),
    ("Country", "France has the largest customer volume, but Germany "
                "churns at a disproportionately higher rate relative to "
                "its size than France or Spain."),
    ("Gender", "There are more male customers overall, but female "
               "customers churn at a higher rate and count than male "
               "customers."),
    ("Credit score", "Churned and retained customers share almost "
                      "identical median credit scores (~650) — except a "
                      "small tail of churned customers with scores below 400."),
]

RECOMMENDATIONS = [
    "Prioritize customers aged 45–60 — the segment where churn rises "
    "fastest relative to its size, and the single strongest predictive "
    "signal in the model.",
    "Investigate Germany specifically. It churns disproportionately to "
    "its customer base compared with France and Spain — worth a "
    "dedicated retention or service-quality review.",
    "Treat \u201cactive member\u201d status as an early-warning flag. It's the "
    "strongest protective factor after age, and the third most "
    "influential feature in the model.",
    "Use precision and recall deliberately. XGBoost's high precision "
    "(76%) suits a targeted, resource-limited campaign; a higher-recall "
    "model suits a broader, lower-cost one.",
]

COLORS = {
    "paper": "#F5F3EC",
    "card": "#FFFFFF",
    "ink": "#1B2A38",
    "ink_soft": "#5B6B78",
    "line": "#DED8C9",
    "risk": "#B54A3F",
    "risk_soft": "#F1DCD6",
    "safe": "#2F6E5B",
    "safe_soft": "#DCE9E2",
    "gold": "#B5852C",
    "gold_soft": "#F1E4C9",
}

FONT_FAMILY = "IBM Plex Sans, sans-serif"
HEADLINE_FAMILY = "Fraunces, serif"
MONO_FAMILY = "IBM Plex Mono, monospace"


# MODELS 
def make_donut():
    fig = go.Figure(
        data=[go.Pie(
            labels=["Churned", "Retained"],
            values=[CHURNED, RETAINED],
            hole=0.62,
            marker=dict(colors=[COLORS["risk"], COLORS["safe_soft"]]),
            textinfo="none",
            sort=False,
            direction="clockwise",
        )]
    )
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(
            text=f"<b>{CHURNED/TOTAL_CUSTOMERS:.1%}</b><br><span style='font-size:11px'>churned</span>",
            x=0.5, y=0.5, font=dict(size=20, color=COLORS["risk"], family=MONO_FAMILY),
            showarrow=False,
        )],
    )
    return fig


def make_correlation_chart():
    labels = [c[0] for c in CORRELATIONS]
    values = [c[1] for c in CORRELATIONS]
    colors = [COLORS["risk"] if v > 0 else COLORS["safe"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in values],
        textposition="outside",
        textfont=dict(family=MONO_FAMILY, size=12),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=40),
        height=260,
        xaxis=dict(zeroline=True, zerolinecolor=COLORS["ink_soft"], range=[-0.35, 0.35],
                   title="Correlation with churn", gridcolor=COLORS["line"]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_shap_chart():
    labels = [s[0] for s in SHAP_VALUES]
    values = [s[1] for s in SHAP_VALUES]
    text = [s[2] for s in SHAP_VALUES]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=COLORS["gold"],
        text=text, textposition="outside",
        textfont=dict(family=MONO_FAMILY, size=12),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=50),
        height=220,
        xaxis=dict(title="Mean |SHAP value| (top 3 of 12 features)", gridcolor=COLORS["line"], range=[0, 1]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_confusion_matrix(model_key):
    m = MODELS[model_key]
    z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
    labels = [["True negative", "False positive"], ["False negative", "True positive"]]
    text = [[f"{m['tn']:,}<br><span style='font-size:11px'>True negative</span>",
             f"{m['fp']:,}<br><span style='font-size:11px'>False positive</span>"],
            [f"{m['fn']:,}<br><span style='font-size:11px'>False negative</span>",
             f"{m['tp']:,}<br><span style='font-size:11px'>True positive</span>"]]
    colorscale = [[0, COLORS["safe_soft"]], [0.5, COLORS["gold_soft"]], [1, COLORS["risk_soft"]]]
    fig = go.Figure(go.Heatmap(
        z=[[0, 1], [1, 2]],  # just for distinct cell shading
        text=text,
        texttemplate="%{text}",
        textfont=dict(family=MONO_FAMILY, size=15, color=COLORS["ink"]),
        colorscale=colorscale,
        showscale=False,
        xgap=6, ygap=6,
    ))
    fig.update_layout(
        margin=dict(t=30, b=10, l=90, r=10),
        height=260,
        xaxis=dict(tickvals=[0, 1], ticktext=["Predicted: stays", "Predicted: churns"],
                   side="top", showgrid=False),
        yaxis=dict(tickvals=[0, 1], ticktext=["Actual: stays", "Actual: churns"],
                   showgrid=False, autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_score_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "font": {"family": MONO_FAMILY, "size": 40, "color": COLORS["ink"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLORS["ink_soft"]},
            "bar": {"color": COLORS["ink"], "thickness": 0.25},
            "bgcolor": COLORS["line"],
            "steps": [
                {"range": [0, 33], "color": COLORS["safe_soft"]},
                {"range": [33, 66], "color": COLORS["gold_soft"]},
                {"range": [66, 100], "color": COLORS["risk_soft"]},
            ],
            "borderwidth": 0,
        },
    ))
    fig.update_layout(
        height=220, margin=dict(t=30, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def illustrative_score(age, country, balance, active, gender):
    """Simplified, directional score built from the notebook's correlation
    coefficients. NOT the trained model's output."""
    score = 20.0
    score += ((age - 38.9) / 25) * 55          # age corr +0.29
    if country == "Germany":
        score += 16                            # Germany corr +0.17
    score += ((balance - 76486) / 90000) * 10  # balance corr +0.12
    score += -22 if active == "yes" else 6     # active_member corr -0.16
    score += -9 if gender == "M" else 6        # gender_male corr -0.11
    return max(3, min(95, round(score)))



# APP 

app = Dash(__name__)
app.title = "Bank Customer Churn — Findings Dashboard"

app.index_string = """
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
    "background": COLORS["card"],
    "border": f"1px solid {COLORS['line']}",
    "borderRadius": "10px",
    "padding": "16px 18px",
    "fontSize": "14.5px",
    "lineHeight": "1.55",
}

SECTION_STYLE = {
    "maxWidth": "980px",
    "margin": "0 auto",
    "padding": "40px 24px",
    "borderBottom": f"1px solid {COLORS['line']}",
}

H2_STYLE = {"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "26px", "marginBottom": "8px"}
SUB_STYLE = {"color": COLORS["ink_soft"], "fontSize": "14.5px", "maxWidth": "60ch", "marginBottom": "28px", "lineHeight": "1.55"}


def kpi_card(num, label):
    return html.Div([
        html.Span(num, style={"fontFamily": MONO_FAMILY, "fontSize": "22px", "fontWeight": 600, "display": "block"}),
        html.Span(label, style={"fontSize": "12px", "color": COLORS["ink_soft"]}),
    ], style={"background": COLORS["card"], "padding": "18px 16px"})


def metric_card(id_, label):
    return html.Div([
        html.Span(id="metric-" + id_, style={"fontFamily": MONO_FAMILY, "fontSize": "21px", "fontWeight": 600, "display": "block"}),
        html.Span(label, style={"fontSize": "11.5px", "color": COLORS["ink_soft"]}),
    ], style={"background": COLORS["card"], "padding": "16px"})


header = html.Header([
    html.P("Bank Customer Churn — Project Findings",
           style={"fontFamily": MONO_FAMILY, "fontSize": "12.5px", "color": COLORS["ink_soft"]}),
    html.H1("Who leaves the bank, and why",
            style={"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "44px",
                   "lineHeight": "1.1", "maxWidth": "14ch", "margin": "0 0 16px"}),
    html.P("An interactive summary of a churn analysis and prediction model built on "
           "10,000 retail banking customers — covering the exploratory findings, model "
           "performance, and the drivers behind churn.",
           style={"fontSize": "16.5px", "color": COLORS["ink_soft"], "maxWidth": "62ch", "lineHeight": "1.6"}),
    html.Div([kpi_card(n, l) for n, l in KPIS],
             style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "1px",
                    "background": COLORS["line"], "border": f"1px solid {COLORS['line']}",
                    "marginTop": "28px", "borderRadius": "10px", "overflow": "hidden"}),
], style={**SECTION_STYLE, "paddingTop": "50px"})


snapshot_section = html.Section([
    html.H2("The churn snapshot", style=H2_STYLE),
    html.P(f"Of the {TOTAL_CUSTOMERS:,} customers in the dataset, {CHURNED:,} churned and "
           f"{RETAINED:,} stayed — a base rate any predictive model needs to beat.",
           style=SUB_STYLE),
    html.Div([
        dcc.Graph(figure=make_donut(), config={"displayModeBar": False}, style={"width": "220px"}),
        html.Div([
            html.Div("Customers are split across three markets: France (50.1%), Germany "
                      "(25.1%) and Spain (24.8%). Gender skews slightly male — 54.6% men, "
                      "45.4% women.", style=CARD_STYLE),
            html.Div("Roughly 7 in 10 customers hold a credit card, and just over half "
                      "(51.5%) are classified as active members.",
                      style={**CARD_STYLE, "marginTop": "16px"}),
        ], style={"flex": 1, "marginLeft": "30px"}),
    ], style={"display": "flex", "alignItems": "center"}),
], style=SECTION_STYLE)


drivers_section = html.Section([
    html.H2("What correlates with leaving", style=H2_STYLE),
    html.P("Pearson correlation of each feature with churn. Age is the strongest single "
           "positive signal; being an active member is the strongest protective one.",
           style=SUB_STYLE),
    dcc.Graph(figure=make_correlation_chart(), config={"displayModeBar": False}),
    html.Div([
        html.Div([html.B(title + ". "), text], style=CARD_STYLE)
        for title, text in FINDINGS
    ], style={"display": "grid", "gridTemplateColumns": "repeat(2, 1fr)", "gap": "16px", "marginTop": "24px"}),
], style=SECTION_STYLE)


model_section = html.Section([
    html.H2("Predicting churn: two models compared", style=H2_STYLE),
    html.P("An XGBoost classifier was benchmarked against logistic regression, both tuned "
           "by grid search and scored on F1 on the held-out test set (2,000 customers).",
           style=SUB_STYLE),
    dcc.RadioItems(
        id="model-toggle",
        options=[{"label": " " + m["label"], "value": k} for k, m in MODELS.items()],
        value="xgb",
        inline=True,
        style={"marginBottom": "24px", "fontFamily": FONT_FAMILY},
        inputStyle={"marginRight": "6px", "marginLeft": "16px"},
    ),
    html.Div([
        metric_card("acc", "Accuracy"), metric_card("prec", "Precision"),
        metric_card("rec", "Recall"), metric_card("f1", "F1 (test)"),
        metric_card("cv", "Best CV F1"),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "1px",
              "background": COLORS["line"], "border": f"1px solid {COLORS['line']}",
              "borderRadius": "10px", "overflow": "hidden"}),
    html.Div([
        html.Div([
            html.P("Confusion matrix — test set, n = 2,000",
                   style={"fontSize": "13px", "color": COLORS["ink_soft"]}),
            dcc.Graph(id="confusion-matrix", config={"displayModeBar": False}),
            html.P(id="model-note", style={"fontSize": "13.5px", "color": COLORS["ink_soft"], "lineHeight": "1.6"}),
        ], style={"flex": 1, "marginRight": "20px"}),
        html.Div([
            html.B("Why this trade-off matters. "),
            "The false negatives are churners the model misses entirely. The false "
            "positives are wasted retention outreach. XGBoost trades some recall for much "
            "higher precision than logistic regression, which flags more churners overall "
            "but with far more false alarms.",
        ], style={**CARD_STYLE, "flex": 1, "alignSelf": "flex-start"}),
    ], style={"display": "flex", "marginTop": "28px"}),
], style=SECTION_STYLE)


shap_section = html.Section([
    html.H2("What drives the model's decisions", style=H2_STYLE),
    html.P("SHAP values on the XGBoost model rank feature importance by average impact on "
           "individual predictions. These are the top three of twelve engineered features.",
           style=SUB_STYLE),
    dcc.Graph(figure=make_shap_chart(), config={"displayModeBar": False}),
], style=SECTION_STYLE)


simulator_section = html.Section([
    html.H2("Try it: an illustrative risk score", style=H2_STYLE),
    html.P("A simplified, directional score built from the correlations above — not the "
           "trained model's actual output. It's here to make the relationships tangible, "
           "not to predict a real customer's risk.", style=SUB_STYLE),
    html.Div([
        html.Div([
            html.Label("Age", style={"fontSize": "13.5px", "color": COLORS["ink_soft"]}),
            dcc.Slider(id="sim-age", min=18, max=92, step=1, value=39,
                       marks={18: "18", 55: "55", 92: "92"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("Country", style={"fontSize": "13.5px", "color": COLORS["ink_soft"], "marginTop": "20px", "display": "block"}),
            dcc.Dropdown(id="sim-country",
                         options=[{"label": c, "value": c} for c in ["France", "Germany", "Spain"]],
                         value="France", clearable=False),

            html.Label("Account balance", style={"fontSize": "13.5px", "color": COLORS["ink_soft"], "marginTop": "20px", "display": "block"}),
            dcc.Slider(id="sim-balance", min=0, max=250000, step=1000, value=76000,
                       marks={0: "$0", 125000: "$125K", 250000: "$250K"},
                       tooltip={"placement": "bottom", "always_visible": True}),

            html.Label("Active member?", style={"fontSize": "13.5px", "color": COLORS["ink_soft"], "marginTop": "20px", "display": "block"}),
            dcc.RadioItems(id="sim-active",
                          options=[{"label": " Active", "value": "yes"}, {"label": " Inactive", "value": "no"}],
                          value="yes", inline=True, inputStyle={"marginRight": "6px", "marginLeft": "12px"}),

            html.Label("Gender", style={"fontSize": "13.5px", "color": COLORS["ink_soft"], "marginTop": "20px", "display": "block"}),
            dcc.RadioItems(id="sim-gender",
                          options=[{"label": " Female", "value": "F"}, {"label": " Male", "value": "M"}],
                          value="M", inline=True, inputStyle={"marginRight": "6px", "marginLeft": "12px"}),
        ], style={"flex": 1, "marginRight": "40px"}),

        html.Div([
            dcc.Graph(id="score-gauge", config={"displayModeBar": False}),
            html.Div("Directional only, built from the five correlation coefficients "
                      "above — not a real prediction.",
                      style={"fontSize": "12px", "color": COLORS["ink_soft"], "background": COLORS["gold_soft"],
                             "borderRadius": "8px", "padding": "10px 12px"}),
        ], style={"width": "260px"}),
    ], style={"display": "flex", "alignItems": "flex-start", "marginTop": "20px"}),
], style=SECTION_STYLE)


reco_section = html.Section([
    html.H2("What this means for retention", style=H2_STYLE),
    html.Div([
        html.Div([
            html.Span(f"{i+1:02d}", style={"fontFamily": MONO_FAMILY, "color": COLORS["gold"], "fontWeight": 600, "marginRight": "14px"}),
            html.Span(text),
        ], style={**CARD_STYLE, "display": "flex", "marginBottom": "14px"})
        for i, text in enumerate(RECOMMENDATIONS)
    ]),
], style=SECTION_STYLE)


app.layout = html.Div([
    header, snapshot_section, drivers_section, model_section,
    shap_section, simulator_section, reco_section,
], style={"fontFamily": FONT_FAMILY, "color": COLORS["ink"], "background": COLORS["paper"]})


# CALLBACK

@app.callback(
    Output("metric-acc", "children"),
    Output("metric-prec", "children"),
    Output("metric-rec", "children"),
    Output("metric-f1", "children"),
    Output("metric-cv", "children"),
    Output("confusion-matrix", "figure"),
    Output("model-note", "children"),
    Input("model-toggle", "value"),
)
def update_model(model_key):
    m = MODELS[model_key]
    return (
        f"{m['accuracy']:.1%}", f"{m['precision']:.1%}", f"{m['recall']:.1%}",
        f"{m['f1']:.1%}", f"{m['cv_f1']:.1%}",
        make_confusion_matrix(model_key),
        m["note"],
    )


@app.callback(
    Output("score-gauge", "figure"),
    Input("sim-age", "value"),
    Input("sim-country", "value"),
    Input("sim-balance", "value"),
    Input("sim-active", "value"),
    Input("sim-gender", "value"),
)
def update_score(age, country, balance, active, gender):
    score = illustrative_score(age, country, balance, active, gender)
    return make_score_gauge(score)

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )

# hiii