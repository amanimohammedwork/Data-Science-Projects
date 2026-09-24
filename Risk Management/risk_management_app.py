
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import os

# DATA
KPIS = [
    ("307,511", "Applications analyzed"),
    ("8.1%", "Default rate"),
    ("0.762", "Best model ROC-AUC"),
    ("24.4%", "Missing data, training set"),
    ("0.394", "KS-statistic"),
]

IMBALANCE_METRICS = [
    ("91.95%", "Accuracy", False),
    ("76.18%", "ROC-AUC", False),
    ("52.70%", "Precision", False),
    ("2.95%", "Recall", True),
    ("5.58%", "F1-Score", False),
    ("39.45%", "KS-statistic", False),
]


MODEL_DATA = {
    "roc": {"XGBoost": 0.7618, "Neural Network": 0.7476, "Logistic Regression": 0.7467, "Random Forest": 0.7452},
    "pr":  {"XGBoost": 0.2422, "Neural Network": 0.2271, "Logistic Regression": 0.2261, "Random Forest": 0.2176},
    "acc": {"XGBoost": 0.9195, "Logistic Regression": 0.9193, "Neural Network": 0.9193, "Random Forest": 0.8830},
    "ks":  {"XGBoost": 0.3945, "Random Forest": 0.3724, "Logistic Regression": 0.3695, "Neural Network": 0.3659},
}
MODEL_LABELS = {"roc": "ROC-AUC", "pr": "PR-AUC", "acc": "Accuracy", "ks": "KS-statistic"}

CORRELATIONS = [
    ("EXT_SOURCE_3", -0.179),
    ("EXT_SOURCE_2", -0.160),
    ("EXT_SOURCE_1", -0.099),
    ("Age (DAYS_BIRTH)", -0.078),
    ("Overdue installment ratio", 0.070),
    ("Credit-to-goods-price ratio", 0.069),
    ("Active bureau accounts", 0.067),
    ("Refused prior applications", 0.064),
]

SUBGROUP_DATA = {
    "gender": [("Male", 0.7575), ("Female", 0.7563)],
    "age": [("30-39", 0.7724), ("40-49", 0.7541), ("50-59", 0.7529), ("<30", 0.7338), ("60-69", 0.7200)],
    "income": [("Very High", 0.7646), ("High", 0.7645), ("Medium", 0.7616), ("Low", 0.7565)],
    "education": [("Higher education", 0.7678), ("Secondary / secondary special", 0.7549),
                  ("Incomplete higher", 0.7524), ("Lower secondary", 0.6903)],
    "housing": [("Office apartment", 0.7732), ("With parents", 0.7709), ("House / apartment", 0.7610),
                ("Municipal apartment", 0.7533), ("Co-op apartment", 0.7444), ("Rented apartment", 0.6898)],
    "marital": [("Civil marriage", 0.7689), ("Married", 0.7634), ("Single / not married", 0.7576),
                ("Widow", 0.7305), ("Separated", 0.7301)],
    "occupation": [
        ("HR staff", 0.9367), ("IT staff", 0.8966), ("Secretaries", 0.8120), ("Managers", 0.7869),
        ("Cooking staff", 0.7704), ("Waiters/barmen staff", 0.7580), ("Core staff", 0.7538),
        ("Medicine staff", 0.7556), ("Missing", 0.7558), ("Laborers", 0.7563), ("Sales staff", 0.7490),
        ("Accountants", 0.7442), ("High skill tech staff", 0.7427), ("Security staff", 0.7271),
        ("Drivers", 0.7269), ("Realty agents", 0.6964), ("Low-skill Laborers", 0.6802),
        ("Cleaning staff", 0.6899), ("Private service staff", 0.6577),
    ],
}
SUBGROUP_LABELS = {
    "gender": "Gender", "age": "Age group", "income": "Income group", "education": "Education",
    "housing": "Housing type", "marital": "Marital status", "occupation": "Occupation",
}
SUBGROUP_CALLOUTS = {
    "gender": "Virtually no gap between male (0.7575) and female (0.7563) applicants — gender doesn't "
              "appear to affect how well the model ranks risk.",
    "age": "A real spread: the model discriminates risk best for 30-39 year-olds (0.772) and worst "
           "for 60-69 year-olds (0.720) — a 5-point ROC-AUC gap worth investigating.",
    "income": "A mild, consistent gradient — the model works slightly better for higher-income "
              "applicants, from 0.757 (Low) up to 0.765 (Very High).",
    "education": "The widest 4-group gap in this dashboard: Higher education (0.768) vs Lower "
                 "secondary (0.690) — a 7.8-point difference.",
    "housing": "A large spread from Office apartment (0.773) down to Rented apartment (0.690) — "
               "renters are the group the model discriminates risk worst for.",
    "marital": "Civil marriage and Married applicants score highest (~0.76-0.77); Widowed and "
               "Separated applicants lowest (~0.73).",
    "occupation": "The widest spread of any grouping — but treat the top end with caution: HR staff "
                  "(0.937) and IT staff (0.897) are likely small subgroups, so those estimates may be "
                  "noisy. Private service staff (0.658) and Low-skill Laborers (0.680) sit clearly lowest.",
}

INCOME_SWEEP = [
    (69367.5, 0.082354), (84663.947368, 0.081056), (99960.394737, 0.080708), (115256.842105, 0.080557),
    (130553.289474, 0.080526), (145849.736842, 0.080551), (161146.184211, 0.080558), (176442.631579, 0.081168),
    (191739.078947, 0.081216), (207035.526316, 0.081168), (222331.973684, 0.081332), (237628.421053, 0.081153),
    (252924.868421, 0.080809), (268221.315789, 0.080405), (283517.763158, 0.080044), (298814.210526, 0.080079),
    (314110.657895, 0.080079), (329407.105263, 0.080430), (344703.552632, 0.080430), (360000.0, 0.080516),
]

RECOMMENDATIONS = [
    "Don't ship on accuracy alone. With an 8.1% base rate, track PR-AUC, recall, and KS-statistic — "
    "and pick a classification threshold deliberately based on the real cost of a missed default vs. "
    "a false alarm.",
    "Audit subgroup performance before deployment. The gaps between the strongest and weakest groups "
    "(by education, housing type, or occupation) are large enough to warrant a fairness review, "
    "especially for smaller subgroups where the estimate may be noisy.",
    "Protect the external bureau score pipeline. EXT_SOURCE_1/2/3 are the strongest individual "
    "predictors by a wide margin — any outage or degradation in that data source will disproportionately "
    "hurt model performance.",
    "Don't treat income as an independent risk lever. Underwriting policy that assumes \"higher income "
    "= lower risk\" in isolation isn't well supported by this model — income only matters through its "
    "correlation with other features.",
]

COLORS = {
    "paper": "#F5F3EC", "card": "#FFFFFF", "ink": "#1B2A38", "ink_soft": "#5B6B78",
    "line": "#DED8C9", "bad": "#B54A3F", "bad_soft": "#F1DCD6",
    "good": "#2F6E5B", "good_soft": "#DCE9E2", "gold": "#B5852C", "gold_soft": "#F1E4C9",
}
FONT_FAMILY = "IBM Plex Sans, sans-serif"
HEADLINE_FAMILY = "Fraunces, serif"
MONO_FAMILY = "IBM Plex Mono, monospace"


# VISUALISATIONS 

def make_model_chart(metric):
    vals = MODEL_DATA[metric]
    labels = list(vals.keys())
    values = list(vals.values())
    colors = [COLORS["good"] if l == "XGBoost" else COLORS["ink_soft"] for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(family=MONO_FAMILY, size=12),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=50), height=240,
        xaxis=dict(title=MODEL_LABELS[metric], gridcolor=COLORS["line"]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_correlation_chart():
    labels = [c[0] for c in CORRELATIONS]
    values = [c[1] for c in CORRELATIONS]
    colors = [COLORS["bad"] if v > 0 else COLORS["good"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:+.3f}" for v in values], textposition="outside",
        textfont=dict(family=MONO_FAMILY, size=11),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=50), height=320,
        xaxis=dict(zeroline=True, zerolinecolor=COLORS["ink_soft"],
                   title="Correlation with default (TARGET)", gridcolor=COLORS["line"]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_subgroup_chart(group):
    data = SUBGROUP_DATA[group]
    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [COLORS["good"] if v >= 0.75 else COLORS["gold"] if v >= 0.71 else COLORS["bad"] for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=colors,
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(family=MONO_FAMILY, size=11),
    ))
    fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=50), height=max(220, 34 * len(data)),
        xaxis=dict(title="ROC-AUC within subgroup", range=[0.6, 1.0], gridcolor=COLORS["line"]),
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


def make_whatif_chart():
    x = [d[0] for d in INCOME_SWEEP]
    y = [d[1] * 100 for d in INCOME_SWEEP]
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=COLORS["gold"], width=2.5),
        marker=dict(color=COLORS["gold"], size=6),
    ))
    fig.update_layout(
        margin=dict(t=20, b=40, l=50, r=20), height=300,
        xaxis=dict(title="Applicant income ($)", gridcolor=COLORS["line"], tickformat="$,.0f"),
        yaxis=dict(title="Predicted probability of default (%)", gridcolor=COLORS["line"]),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, color=COLORS["ink"]),
    )
    return fig


# APP

app = Dash(__name__)
app.title = "Home Credit Default Risk — Findings Dashboard"

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
    "background": COLORS["card"], "border": f"1px solid {COLORS['line']}",
    "borderRadius": "10px", "padding": "16px 18px", "fontSize": "14.5px", "lineHeight": "1.55",
}
SECTION_STYLE = {"maxWidth": "1000px", "margin": "0 auto", "padding": "40px 24px", "borderBottom": f"1px solid {COLORS['line']}"}
H2_STYLE = {"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "26px", "marginBottom": "8px"}
SUB_STYLE = {"color": COLORS["ink_soft"], "fontSize": "14.5px", "maxWidth": "66ch", "marginBottom": "28px", "lineHeight": "1.55"}


def kpi_card(num, label):
    return html.Div([
        html.Span(num, style={"fontFamily": MONO_FAMILY, "fontSize": "19px", "fontWeight": 600, "display": "block"}),
        html.Span(label, style={"fontSize": "11.5px", "color": COLORS["ink_soft"]}),
    ], style={"background": COLORS["card"], "padding": "18px 14px"})


def metric_spot(num, label, highlight=False):
    return html.Div([
        html.Span(num, style={"fontFamily": MONO_FAMILY, "fontSize": "18px", "fontWeight": 600, "display": "block",
                               "color": COLORS["bad"] if highlight else COLORS["ink"]}),
        html.Span(label, style={"fontSize": "10.5px", "color": COLORS["ink_soft"]}),
    ], style={"background": COLORS["card"], "padding": "14px", "textAlign": "center"})


def callout(children, warn=False):
    return html.Div(children, style={"background": COLORS["bad_soft"] if warn else COLORS["gold_soft"],
                                       "borderRadius": "10px", "padding": "18px 20px",
                                       "fontSize": "14.5px", "lineHeight": "1.6", "marginTop": "20px"})


header = html.Header([
    html.P("Home Credit Default Risk — Project Findings",
           style={"fontFamily": MONO_FAMILY, "fontSize": "12.5px", "color": COLORS["ink_soft"]}),
    html.H1("Ranking risk well isn't the same as catching defaulters",
            style={"fontFamily": HEADLINE_FAMILY, "fontWeight": 500, "fontSize": "38px",
                   "lineHeight": "1.1", "maxWidth": "17ch", "margin": "0 0 16px"}),
    html.P("An interactive summary of a credit-default prediction project on 307,511 loan applications "
           "— covering what drives default risk, how four models compare, whether the best model treats "
           "applicant groups consistently, and a class-imbalance trap that a headline accuracy score hides.",
           style={"fontSize": "16.5px", "color": COLORS["ink_soft"], "maxWidth": "64ch", "lineHeight": "1.6"}),
    html.Div([kpi_card(n, l) for n, l in KPIS],
             style={"display": "grid", "gridTemplateColumns": "repeat(5, 1fr)", "gap": "1px",
                    "background": COLORS["line"], "border": f"1px solid {COLORS['line']}",
                    "marginTop": "28px", "borderRadius": "10px", "overflow": "hidden"}),
], style={**SECTION_STYLE, "paddingTop": "50px"})


imbalance_section = html.Section([
    html.H2("The accuracy trap", style=H2_STYLE),
    html.P("Only 8.1% of applicants in the dataset actually default — so a model that never flags "
           "anyone as risky is already 92% \"accurate.\" That's exactly the trap the tuned XGBoost "
           "model falls into at the default 0.5 probability threshold:", style=SUB_STYLE),
    html.Div([metric_spot(n, l, h) for n, l, h in IMBALANCE_METRICS],
              style={"display": "grid", "gridTemplateColumns": "repeat(6, 1fr)", "gap": "1px",
                     "background": COLORS["line"], "border": f"1px solid {COLORS['line']}",
                     "borderRadius": "10px", "overflow": "hidden"}),
    callout([html.B("Recall of 2.95% means the model catches fewer than 3 in 100 real defaulters "),
             "at the standard classification threshold — despite a respectable ROC-AUC (0.76) and "
             "KS-statistic (0.39), which show the model can separate risk well when ranking applicants. "
             "The gap between ranking ability and default-threshold recall is a threshold problem, not "
             "a modeling-power problem."], warn=True),
], style=SECTION_STYLE)


model_section = html.Section([
    html.H2("Four models, compared", style=H2_STYLE),
    html.P("Logistic regression, random forest, XGBoost, and a small Keras neural network were each "
           "tuned via grid search (or trained directly, for the network) and scored on the same "
           "held-out validation split.", style=SUB_STYLE),
    dcc.RadioItems(
        id="model-metric",
        options=[{"label": " " + MODEL_LABELS[k], "value": k} for k in MODEL_DATA],
        value="roc", inline=True, style={"marginBottom": "16px"},
        inputStyle={"marginRight": "6px", "marginLeft": "16px"},
    ),
    dcc.Graph(id="model-chart", config={"displayModeBar": False}),
    callout([html.B("XGBoost wins on every metric "), "— highest ROC-AUC (0.762), PR-AUC (0.242), "
             "accuracy (91.95%) and KS-statistic (0.394) — but random forest's accuracy (88.3%) looks "
             "like an outlier low, largely a side effect of its best hyperparameters using "
             "class_weight=\"balanced\"."]),
], style=SECTION_STYLE)


drivers_section = html.Section([
    html.H2("What predicts default", style=H2_STYLE),
    html.P("Correlation of each engineered feature with the default target. The three external "
           "bureau scores are far and away the strongest signals — more so than anything engineered "
           "in-house.", style=SUB_STYLE),
    dcc.Graph(figure=make_correlation_chart(), config={"displayModeBar": False}),
    html.Div([html.B("External scores dominate. "), "EXT_SOURCE_1 is missing for 56.4% of applicants "
              "— the model has an explicit \"missing\" indicator feature for it, but any deployment "
              "needs a clear fallback for applicants without bureau history."],
             style={**CARD_STYLE, "marginTop": "20px"}),
], style=SECTION_STYLE)


fairness_section = html.Section([
    html.H2("Does the model treat every applicant group consistently?", style=H2_STYLE),
    html.P("ROC-AUC computed separately within each subgroup of the validation set — a quick read "
           "on whether the model discriminates risk equally well across different kinds of "
           "applicants, or whether it works much better for some groups than others.", style=SUB_STYLE),
    dcc.RadioItems(
        id="subgroup-choice",
        options=[{"label": " " + SUBGROUP_LABELS[k], "value": k} for k in SUBGROUP_DATA],
        value="gender", inline=True, style={"marginBottom": "16px"},
        inputStyle={"marginRight": "6px", "marginLeft": "16px"},
    ),
    dcc.Graph(id="subgroup-chart", config={"displayModeBar": False}),
    html.Div(id="subgroup-callout"),
], style=SECTION_STYLE)


whatif_section = html.Section([
    html.H2("What-if: does income alone move the needle?", style=H2_STYLE),
    html.P("Holding every other feature fixed and sweeping applicant income from the 5th to 95th "
           "percentile ($69K–$360K), the model's average predicted default probability barely "
           "moves.", style=SUB_STYLE),
    dcc.Graph(figure=make_whatif_chart(), config={"displayModeBar": False}),
    html.Div([html.B("Income is a weak standalone lever. "), "Predicted default risk stays within a "
              "narrow 0.080–0.082 band across the entire income sweep — the model has effectively "
              "learned that income alone, independent of the features it correlates with, doesn't "
              "move the needle much on its own."], style={**CARD_STYLE, "marginTop": "16px"}),
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
    html.P("Source: Home Credit Default Risk dataset (Kaggle competition), 307,511 training "
           "applications joined with bureau, previous-application, POS/cash, credit-card, and "
           "installment-payment history. Models — logistic regression, random forest, XGBoost, and "
           "a small Keras neural network — were trained on an 80/20 stratified split and evaluated "
           "on the held-out validation set. Subgroup ROC-AUC figures use the same validation set and "
           "predictions; subgroups with very few observations can produce noisy or undefined "
           "estimates and should be read with that caveat. All figures above are taken directly "
           "from the project notebook's printed output.",
           style={"fontSize": "12.5px", "color": COLORS["ink_soft"], "lineHeight": "1.6", "maxWidth": "76ch"}),
], style={**SECTION_STYLE, "borderBottom": "none"})


app.layout = html.Div([
    header, imbalance_section, model_section, drivers_section,
    fairness_section, whatif_section, reco_section, footer,
], style={"fontFamily": FONT_FAMILY, "color": COLORS["ink"], "background": COLORS["paper"]})


# CALLBACK APP

@app.callback(Output("model-chart", "figure"), Input("model-metric", "value"))
def update_model_chart(metric):
    return make_model_chart(metric)


@app.callback(
    Output("subgroup-chart", "figure"),
    Output("subgroup-callout", "children"),
    Input("subgroup-choice", "value"),
)
def update_subgroup_chart(group):
    fig = make_subgroup_chart(group)
    text = html.Div([html.B("Reading this view: "), SUBGROUP_CALLOUTS[group]],
                     style={"background": COLORS["bad_soft"], "borderRadius": "10px", "padding": "18px 20px",
                            "fontSize": "14.5px", "lineHeight": "1.6", "marginTop": "8px"})
    return fig, text


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8052)),
        debug=False
    )