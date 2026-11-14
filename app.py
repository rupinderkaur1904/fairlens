"""
FairLens - an AI fairness audit dashboard (UCS421 demonstration).

Run:  streamlit run app.py
All data is SYNTHETIC (see fairlens_core.py). Results illustrate mechanisms,
not real-world findings.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from fairlens_core import FEATURES, run_experiment_a, run_experiment_b

NAVY, CYAN, GREY, RED = "#2F4A43", "#8DB5A0", "#A7B3AE", "#B9775E"
GROUP_COLORS = {"Group A": NAVY, "Group B": CYAN, "A": NAVY, "B": CYAN}
METRIC_LABELS = {
    "accuracy": "Accuracy", "precision": "Precision", "recall_tpr": "Recall / TPR",
    "fpr": "False-positive rate", "fnr": "False-negative rate",
    "selection_rate": "Selection rate",
}

st.set_page_config(page_title="FairLens", layout="wide")
st.markdown(
    f"""<style>
    h1, h2, h3 {{ color: {NAVY}; }}
    .synthetic {{ background:#EDF3EE; border-radius:8px; padding:8px 14px;
                 color:{NAVY}; font-size:0.9rem; }}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def exp_a(share_b):
    return run_experiment_a(share_b_skewed=share_b)


@st.cache_resource(show_spinner=False)
def exp_b(threshold):
    return run_experiment_b(global_threshold=threshold)


def metric_table(report: pd.DataFrame) -> pd.DataFrame:
    t = report[list(METRIC_LABELS)].rename(columns=METRIC_LABELS)
    return (t * 100).round(1).astype(str) + "%"


def grouped_bars(reports: dict, metric: str, title: str):
    rows = []
    for cond, rep in reports.items():
        for g in ("Group A", "Group B"):
            rows.append({"Condition": cond, "Group": g, "value": rep.loc[g, metric] * 100})
    df = pd.DataFrame(rows)
    fig = px.bar(df, x="Condition", y="value", color="Group", barmode="group",
                 color_discrete_map=GROUP_COLORS, text=df["value"].round(1))
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(title=title, yaxis_title="%", yaxis_range=[0, 105],
                      plot_bgcolor="#FBFAF7", paper_bgcolor="#FBFAF7", height=360, margin=dict(t=50, b=10),
                      legend_title_text="")
    fig.update_yaxes(gridcolor="#DCE3DE")
    return fig


def metric_profile(report: pd.DataFrame, title: str):
    metrics = ["accuracy", "precision", "recall_tpr", "fpr", "fnr"]
    fig = go.Figure()
    for g in ("Group A", "Group B"):
        vals = [report.loc[g, m] * 100 for m in metrics]
        fig.add_bar(name=g, x=[METRIC_LABELS[m] for m in metrics], y=vals,
                    marker_color=GROUP_COLORS[g], text=[f"{v:.1f}%" for v in vals],
                    textposition="outside")
    fig.update_layout(title=title, barmode="group", yaxis_range=[0, 105],
                      plot_bgcolor="#FBFAF7", paper_bgcolor="#FBFAF7", height=360, margin=dict(t=50, b=10))
    fig.update_yaxes(gridcolor="#DCE3DE", title="%")
    return fig


# ---------------------------------------------------------------- header
st.title("FairLens — AI Fairness Audit Dashboard")
st.markdown(
    '<div class="synthetic"><b>Synthetic data.</b> Every candidate here is simulated. '
    "Both groups are drawn with the <i>same</i> distribution of true skill and the same "
    "chance of being qualified, so any gap in errors is created by the data pipeline or "
    "the model, not by real differences. These are demonstrations of mechanisms, not "
    "research findings.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Audit controls")
    exp = st.radio("Experiment", ["A · Dataset representation bias",
                                  "B · Algorithmic bias (data fixed)"])
    if exp.startswith("A"):
        share_pct = st.slider("Group B share of skewed training set (%)", 1, 40, 5)
    else:
        use_opt = st.checkbox("Use accuracy-optimal global threshold", value=True)
        thr = None if use_opt else st.slider("Global decision threshold", 0.20, 0.80, 0.50, 0.01)
    st.caption("Model A: gradient boosting · Model B: logistic regression · "
               "features: qualification score, years of experience, technical score. "
               "The group label is never a model input.")

# ---------------------------------------------------------------- Experiment A
if exp.startswith("A"):
    R = exp_a(share_pct / 100)
    sk, rp, test = R["skewed"], R["representative"], R["test"]
    st.header("Experiment A — Dataset representation bias")
    st.write("Same model, same features, same **balanced test set** (1,500 per group). "
             "Only the composition of the **training data** changes.")

    c1, c2 = st.columns(2)
    comp = pd.DataFrame({
        "Training set": [f"Skewed ({share_pct}% B)"] * 2 + ["Representative (50% B)"] * 2,
        "Group": ["A", "B", "A", "B"],
        "Candidates": [(sk["train"].group == "A").sum(), (sk["train"].group == "B").sum(),
                       (rp["train"].group == "A").sum(), (rp["train"].group == "B").sum()],
    })
    fig = px.bar(comp, x="Training set", y="Candidates", color="Group",
                 color_discrete_map=GROUP_COLORS, text="Candidates")
    fig.update_traces(textposition="inside")
    fig.update_layout(title="Training data distribution", plot_bgcolor="#FBFAF7", paper_bgcolor="#FBFAF7", height=330,
                      yaxis_range=[0, 4400])
    c1.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(test, x="qualification_score", color="group", barmode="overlay",
                       nbins=40, opacity=0.65, color_discrete_map=GROUP_COLORS)
    fig.update_layout(title="Test set: Group B is graded on a stricter scale",
                      plot_bgcolor="#FBFAF7", paper_bgcolor="#FBFAF7", height=330, xaxis_title="Qualification score")
    c2.plotly_chart(fig, use_container_width=True)

    a_b = sk["report"]
    gap = sk["gaps"]["tpr_gap (equal opportunity)"] * 100
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall accuracy (skewed)", f"{a_b.loc['Overall','accuracy']*100:.1f}%")
    k2.metric("Group B recall (skewed)", f"{a_b.loc['Group B','recall_tpr']*100:.1f}%",
              f"{(a_b.loc['Group B','recall_tpr']-a_b.loc['Group A','recall_tpr'])*100:.1f} pts vs A")
    k3.metric("Group B recall (representative)", f"{rp['report'].loc['Group B','recall_tpr']*100:.1f}%")
    k4.metric("Recall gap A−B", f"{gap:.1f} → {rp['gaps']['tpr_gap (equal opportunity)']*100:.1f} pts")

    c1, c2 = st.columns(2)
    c1.plotly_chart(grouped_bars({"Skewed": sk["report"], "Representative": rp["report"]},
                                 "recall_tpr", "Before vs after: recall (TPR) by group"),
                    use_container_width=True)
    c2.plotly_chart(grouped_bars({"Skewed": sk["report"], "Representative": rp["report"]},
                                 "fnr", "Before vs after: false-negative rate by group"),
                    use_container_width=True)

    t1, t2 = st.columns(2)
    t1.subheader("Skewed training"); t1.dataframe(metric_table(sk["report"]))
    t2.subheader("Representative training"); t2.dataframe(metric_table(rp["report"]))

    st.info(
        f"**Detected issue.** Trained on {share_pct}% Group B, the model learned Group A's "
        f"mapping from scores to suitability. Qualified Group B candidates, whose scores are "
        f"graded more strictly, are missed: Group B recall is "
        f"{a_b.loc['Group B','recall_tpr']*100:.1f}% vs {a_b.loc['Group A','recall_tpr']*100:.1f}% "
        f"for Group A, although overall accuracy looks acceptable "
        f"({a_b.loc['Overall','accuracy']*100:.1f}%). A representative training set narrows the "
        f"gap, but Group A accuracy changes too — rebalancing is a trade-off, not a guarantee."
    )

# ---------------------------------------------------------------- Experiment B
else:
    R = exp_b(thr)
    st.header("Experiment B — Algorithmic bias with the data held fixed")
    st.write("Balanced data: equal group sizes, equal skill, equal qualification rate. "
             "Group B records ~2 fewer years of experience for the same skill (career breaks). "
             "We change **only the algorithm design** and audit each version on the same test set.")

    c1, c2 = st.columns(2)
    tr = R["train"]
    fig = px.histogram(tr, x="years_experience", color="group", barmode="overlay", nbins=40,
                       opacity=0.65, color_discrete_map=GROUP_COLORS)
    fig.update_layout(title="Experience is a group-correlated proxy", plot_bgcolor="#FBFAF7", paper_bgcolor="#FBFAF7",
                      height=330, xaxis_title="Years of experience")
    c1.plotly_chart(fig, use_container_width=True)
    fig = px.histogram(pd.DataFrame({"score": R["B1"]["scores"], "group": R["test"]["group"]}),
                       x="score", color="group", barmode="overlay", nbins=40, opacity=0.65,
                       color_discrete_map=GROUP_COLORS)
    fig.add_vline(x=R["B1"]["threshold"], line_dash="dash", line_color=RED,
                  annotation_text=f"global threshold {R['B1']['threshold']:.2f}")
    fig.update_layout(title="One threshold applied to both groups' scores",
                      plot_bgcolor="#FBFAF7", paper_bgcolor="#FBFAF7", height=330, xaxis_title="Model score P(qualified)")
    c2.plotly_chart(fig, use_container_width=True)

    reps = {R[k]["label"]: R[k]["report"] for k in ("B1", "B2", "B3")}
    b1 = R["B1"]["report"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy A / B (baseline)",
              f"{b1.loc['Group A','accuracy']*100:.1f}% / {b1.loc['Group B','accuracy']*100:.1f}%")
    k2.metric("Recall A / B (baseline)",
              f"{b1.loc['Group A','recall_tpr']*100:.1f}% / {b1.loc['Group B','recall_tpr']*100:.1f}%")
    k3.metric("FPR A / B (baseline)",
              f"{b1.loc['Group A','fpr']*100:.1f}% / {b1.loc['Group B','fpr']*100:.1f}%")
    k4.metric("Recall gap, group thresholds",
              f"{R['B3']['gaps']['tpr_gap (equal opportunity)']*100:.1f} pts")

    st.plotly_chart(metric_profile(b1, "Baseline audit: near-equal accuracy hides unequal errors"),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(grouped_bars(reps, "recall_tpr", "Recall (TPR) under each design"),
                    use_container_width=True)
    c2.plotly_chart(grouped_bars(reps, "fpr", "False-positive rate under each design"),
                    use_container_width=True)

    summary = pd.DataFrame({
        lbl: {"Overall accuracy": f"{rep.loc['Overall','accuracy']*100:.1f}%",
              "Recall gap A−B (pts)": f"{(rep.loc['Group A','recall_tpr']-rep.loc['Group B','recall_tpr'])*100:.1f}",
              "FPR gap A−B (pts)": f"{(rep.loc['Group A','fpr']-rep.loc['Group B','fpr'])*100:.1f}",
              "Selection-rate gap (pts)": f"{(rep.loc['Group A','selection_rate']-rep.loc['Group B','selection_rate'])*100:.1f}"}
        for lbl, rep in reps.items()
    })
    st.subheader("Trade-off summary")
    st.caption("Baseline = all features + one accuracy-optimal threshold · Drop proxy = remove "
               "years of experience · Group thresholds = per-group thresholds equalising recall "
               "(equal opportunity)")
    st.dataframe(summary)
    with st.expander("Full per-group metrics"):
        for lbl, rep in reps.items():
            st.markdown(f"**{lbl}**"); st.dataframe(metric_table(rep))

    st.info(
        "**Detected issue.** With identical data, the baseline model leans on years of experience "
        f"(largest coefficient: {R['B1']['coef']['years_experience']}), a feature that is lower for "
        "Group B at the same skill. One global threshold then misses many more qualified Group B "
        "candidates (lower recall) while overall and per-group accuracy look almost equal. "
        "Dropping the proxy removes the gap but lowers accuracy; group-specific thresholds equalise "
        "recall but require using the group attribute at decision time, which raises legal and "
        "ethical questions. No single fix is 'fair' in every sense."
    )

st.caption("FairLens · UCS421 Ethics and Risk Mitigation in AI · synthetic data · "
           "metrics computed live with scikit-learn")
