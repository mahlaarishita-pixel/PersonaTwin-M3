import streamlit as st
import pandas as pd
import numpy as np
import json, uuid
from pathlib import Path
from datetime import datetime, timezone
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
RESP_FILE = DATA_DIR / "responses.csv"
SCENARIO_FILE = DATA_DIR / "scenario_responses.csv"

st.set_page_config(page_title="PersonaTwin | UK Financial Behaviour", page_icon="💷", layout="wide")

# ----------------------------- Visual identity -----------------------------
st.markdown("""
<style>
:root { --pt-purple:#9b8cff; --pt-teal:#39d6c5; }
.stApp { background: radial-gradient(ellipse at 12% 0%, rgba(92,72,190,.20), transparent 34%), radial-gradient(ellipse at 90% 8%, rgba(24,183,171,.13), transparent 28%), #0b1020; color:#edf2ff; }
[data-testid="stHeader"] { background:rgba(11,16,32,.82); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#111a30 0%,#0d1426 100%); border-right:1px solid rgba(155,140,255,.22); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#aebbd4; }
[data-testid="stSidebar"] h1,h1,h2,h3 { color:#f5f7ff !important; letter-spacing:-.035em; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding:9px 12px; border-radius:10px; margin:3px 0; background:rgba(255,255,255,.025); }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background:rgba(155,140,255,.14); }
p,li,label { color:#d4def1; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(30,43,72,.96),rgba(19,29,51,.96)); border:1px solid rgba(155,140,255,.24); border-radius:16px; padding:18px 20px; box-shadow:0 10px 28px rgba(0,0,0,.18); }
[data-testid="stMetricLabel"] { color:#aebbd4 !important; } [data-testid="stMetricValue"] { color:#fff !important; font-weight:800; }
.stButton button,.stDownloadButton button,[data-testid="stFormSubmitButton"] button { border:0; border-radius:11px; font-weight:700; background:linear-gradient(100deg,#7865e8,#19b9b0); color:white; box-shadow:0 7px 20px rgba(91,89,220,.22); }
.stButton button:hover,.stDownloadButton button:hover { filter:brightness(1.12); color:white; }
[data-testid="stForm"] { background:rgba(20,29,51,.72); border:1px solid rgba(155,140,255,.20); border-radius:18px; padding:22px; }
[data-testid="stExpander"] { background:rgba(24,36,62,.7); border:1px solid rgba(155,140,255,.18); border-radius:12px; }
[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,#8a78ff,#39d6c5); }
[data-testid="stDataFrame"] { border:1px solid rgba(155,140,255,.2); border-radius:12px; overflow:hidden; }
hr { border-color:rgba(155,140,255,.2); }
.pt-hero { padding:30px 32px; border-radius:22px; background:linear-gradient(115deg,rgba(90,70,185,.34),rgba(24,163,164,.19)),linear-gradient(145deg,#172440,#111a30); border:1px solid rgba(155,140,255,.30); box-shadow:0 18px 50px rgba(0,0,0,.22); margin:8px 0 22px; }
.pt-kicker { color:#7de7db; font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.pt-hero h1 { font-size:2.55rem; margin:.35rem 0 .45rem; color:#fff !important; }
.pt-hero p { color:#c7d4ec; font-size:1.04rem; max-width:760px; }
.pt-chip { display:inline-block; padding:6px 11px; border:1px solid rgba(57,214,197,.32); background:rgba(57,214,197,.09); color:#8cece1; border-radius:999px; font-size:.78rem; margin:4px 5px 0 0; }
.pt-section { padding:16px 18px; border-radius:15px; background:rgba(24,36,62,.62); border:1px solid rgba(155,140,255,.16); margin:8px 0 14px; }
@media (max-width:700px) { .pt-hero { padding:22px; } .pt-hero h1 { font-size:2rem; } }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Research-informed instrument -----------------------------
LIKERT = ["Strongly disagree", "Disagree", "Neither agree nor disagree", "Agree", "Strongly agree"]
FREQ = ["Never", "Rarely", "Sometimes", "Often", "Always"]

# Items marked OECD are reproduced/adapted from OECD/INFE Toolkit 2026.
# MaPS informs domain selection. GL items below are original project-written prompts
# inspired by the risk-tolerance construct, not the validated 13-item scale.
QUESTIONS = [
    ("B01", "Budgeting & Planning", "I make a plan to manage my income and expenses.", "OECD/INFE 2026, QF2 (adapted to first person)", "likert"),
    ("B02", "Budgeting & Planning", "I keep a note or digital record of my spending.", "OECD/INFE 2026, QF2 (adapted)", "likert"),
    ("B03", "Budgeting & Planning", "Before I buy something, I carefully consider whether I can afford it.", "OECD/INFE 2026, QS2 (wording retained)", "frequency"),
    ("S01", "Saving & Resilience", "In the past 12 months, I have actively put money aside for savings.", "OECD/INFE 2026, QF3 (adapted summary)", "likert"),
    ("S02", "Saving & Resilience", "I could cover an unexpected expense equivalent to one month of my income without borrowing.", "OECD/INFE 2026, QF4 (adapted to Likert)", "likert"),
    ("S03", "Saving & Resilience", "I am confident I could cope financially for a month if an emergency occurred.", "OECD/INFE 2026, QS3 (wording adapted)", "likert"),
    ("C01", "Credit & Debt Management", "I pay my bills and credit commitments on time.", "OECD/INFE 2026, QS2 (wording retained)", "frequency"),
    ("C02", "Credit & Debt Management", "I have too much debt right now.", "OECD/INFE 2026, QS1 (wording retained; reverse scored)", "likert_reverse"),
    ("C03", "Credit & Debt Management", "I understand the total cost of borrowing before using credit.", "Project-written item informed by MaPS credit-management domain", "likert"),
    ("R01", "Financial Risk Tolerance", "I would accept some chance of losing money for the possibility of a higher financial return.", "Project-written item informed by Grable & Lytton (1999) risk-tolerance construct", "likert"),
    ("R02", "Financial Risk Tolerance", "If an investment fell in value, I would be comfortable waiting rather than immediately selling.", "Project-written scenario attitude informed by Grable & Lytton (1999)", "likert"),
    ("R03", "Financial Risk Tolerance", "I prefer a guaranteed smaller return over an uncertain possibility of a larger return.", "Project-written item; reverse scored; informed by Grable & Lytton (1999)", "likert_reverse"),
    ("F01", "Financial Confidence", "I am confident in my ability to manage my finances.", "OECD/INFE 2026, QS3 (wording retained)", "likert"),
    ("F02", "Financial Confidence", "I keep a close personal watch on my financial affairs.", "OECD/INFE 2026, QS1 (wording retained)", "likert"),
    ("F03", "Financial Confidence", "I feel that my finances are under control.", "OECD/INFE 2026, QS2 (wording retained)", "likert"),
    ("L01", "Long-term Financial Planning", "I set long-term financial goals and strive to achieve them.", "OECD/INFE 2026, QS1 (wording retained)", "likert"),
    ("L02", "Long-term Financial Planning", "I value making an effort now to ensure financial security in the future.", "OECD/INFE 2026, QS1 (wording retained)", "likert"),
    ("L03", "Long-term Financial Planning", "I regularly review whether my savings and pension plans support my future goals.", "Project-written item informed by OECD/INFE and MaPS planning domains", "likert"),
]
DIMENSIONS = ["Budgeting & Planning", "Saving & Resilience", "Credit & Debt Management",
              "Financial Risk Tolerance", "Financial Confidence", "Long-term Financial Planning"]
Q_BY_ID = {q[0]: q for q in QUESTIONS}
OPTIONS = {
    "likert": LIKERT,
    "likert_reverse": LIKERT,
    "frequency": FREQ
}
SCORE = {v:i+1 for i,v in enumerate(LIKERT)}
FREQ_SCORE = {v:i+1 for i,v in enumerate(FREQ)}

SCENARIOS = {
    "S1": {
        "title": "Unexpected £500 expense",
        "prompt": "You receive an unexpected £500 bill. What would you most likely do first?",
        "options": ["Use emergency savings", "Use a credit card or borrow", "Reduce other spending and pay from income", "Ask family or friends for help"]
    },
    "S2": {
        "title": "Unexpected £1,000 windfall",
        "prompt": "You receive an unexpected £1,000. What would you most likely do?",
        "options": ["Save most or all of it", "Pay down existing debt", "Invest some or all of it", "Spend most or all of it"]
    },
    "S3": {
        "title": "Monthly budget pressure",
        "prompt": "Your essential costs rise and your monthly budget becomes tight. What would you most likely do?",
        "options": ["Reduce non-essential spending", "Use savings", "Rely on credit/overdraft", "Delay or miss a payment"]
    }
}
TARGET_MAP = {
    "Use emergency savings":"Savings",
    "Use a credit card or borrow":"Borrowing",
    "Reduce other spending and pay from income":"Budget adjustment",
    "Ask family or friends for help":"Informal support",
    "Save most or all of it":"Savings",
    "Pay down existing debt":"Debt repayment",
    "Invest some or all of it":"Investing",
    "Spend most or all of it":"Spending",
    "Reduce non-essential spending":"Budget adjustment",
    "Use savings":"Savings",
    "Rely on credit/overdraft":"Borrowing",
    "Delay or miss a payment":"Payment difficulty"
}
FEATURES = ["Budgeting & Planning","Saving & Resilience","Credit & Debt Management",
            "Financial Risk Tolerance","Financial Confidence","Long-term Financial Planning","Scenario"]

def load_csv(path):
    if path.exists():
        try: return pd.read_csv(path)
        except Exception: return pd.DataFrame()
    return pd.DataFrame()

def score_answers(answers):
    vals = {}
    for qid, value in answers.items():
        q = Q_BY_ID[qid]
        typ = q[4]
        if typ in ("likert", "likert_reverse"):
            score = SCORE[value]
        else:
            score = FREQ_SCORE[value]
        if typ == "likert_reverse":
            score = 6-score
        vals[qid] = score
    result = {}
    for dim in DIMENSIONS:
        ids = [q[0] for q in QUESTIONS if q[1] == dim]
        result[dim] = round(float(np.mean([vals[i] for i in ids]))*20, 1)
    return vals, result

def save_submission(meta, answers, scenario_answers, scores):
    respondent_id = str(uuid.uuid4())
    raw = {"respondent_id": respondent_id, "submitted_at_utc": datetime.now(timezone.utc).isoformat(), **meta}
    raw.update({f"answer_{k}":v for k,v in answers.items()})
    raw.update({f"rawscore_{k}":v for k,v in scores[0].items()})
    raw.update({f"score_{k}":v for k,v in scores[1].items()})
    df = load_csv(RESP_FILE)
    pd.concat([df, pd.DataFrame([raw])], ignore_index=True).to_csv(RESP_FILE, index=False)
    rows=[]
    for sid, answer in scenario_answers.items():
        rows.append({"respondent_id":respondent_id,"scenario_id":sid,
                     "Scenario":sid,"scenario_title":SCENARIOS[sid]["title"],
                     "scenario_response":answer,"target":TARGET_MAP[answer],
                     **scores[1]})
    sdf=load_csv(SCENARIO_FILE)
    pd.concat([sdf,pd.DataFrame(rows)],ignore_index=True).to_csv(SCENARIO_FILE,index=False)
    return respondent_id

def train_models(data):
    # A participant-level split prevents the same respondent appearing in train and test.
    X=data[FEATURES].copy()
    y=data["target"].astype(str)
    groups=data["respondent_id"]
    splitter=GroupShuffleSplit(n_splits=1,test_size=0.25,random_state=42)
    train_idx,test_idx=next(splitter.split(X,y,groups))
    Xtr,Xte=X.iloc[train_idx],X.iloc[test_idx]
    ytr,yte=y.iloc[train_idx],y.iloc[test_idx]
    numeric=DIMENSIONS
    prep=ColumnTransformer([
        ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),numeric),
        ("scenario",OneHotEncoder(handle_unknown="ignore"),["Scenario"])
    ])
    models={
        "Logistic Regression":Pipeline([("prep",prep),("model",LogisticRegression(max_iter=1500,class_weight="balanced"))]),
        "Random Forest":Pipeline([("prep",prep),("model",RandomForestClassifier(n_estimators=250,max_depth=6,min_samples_leaf=2,class_weight="balanced",random_state=42))])
    }
    results={}
    for name,model in models.items():
        model.fit(Xtr,ytr)
        pred=model.predict(Xte)
        p,r,f,_=precision_recall_fscore_support(yte,pred,average="weighted",zero_division=0)
        results[name]={"model":model,"accuracy":accuracy_score(yte,pred),"precision":p,"recall":r,"f1":f,"pred":pred,"actual":yte,"X_test":Xte}
    return results

def show_header():
    st.markdown("""
    <div class="pt-hero">
      <div class="pt-kicker">UK financial behaviour • Explainable AI</div>
      <h1>💷 PersonaTwin</h1>
      <p>Explore how people budget, save, manage credit and make everyday financial decisions — through a research-informed assessment and explainable machine learning.</p>
      <span class="pt-chip">18+ UK respondents</span><span class="pt-chip">6 behaviour dimensions</span><span class="pt-chip">Research prototype</span>
    </div>
    """, unsafe_allow_html=True)


def page_home():
    show_header()
    st.info("Research prototype — not financial advice, credit scoring, investment suitability assessment or a psychological diagnosis.")
    data=load_csv(RESP_FILE)
    scenarios=load_csv(SCENARIO_FILE)
    a,b,c=st.columns(3)
    a.metric("Completed assessments",len(data),help="Number of saved anonymous questionnaire submissions")
    b.metric("Scenario responses",len(scenarios),help="Three hypothetical scenario choices per completed assessment")
    c.metric("Financial dimensions",6,help="The six descriptive dimensions used in PersonaTwin")
    st.markdown('<div class="pt-section"><div class="pt-kicker">How it works</div><h2>From responses to financial behaviour insights</h2><p>PersonaTwin converts questionnaire responses into six descriptive indicators, then explores how those indicators relate to choices in everyday UK financial scenarios.</p></div>',unsafe_allow_html=True)
    x,y=st.columns(2)
    with x:
        st.markdown("#### Explore the assessment")
        st.write("Complete the anonymous questionnaire and respond to three realistic financial situations.")
        st.markdown("#### Understand your persona")
        st.write("Review your six dimension scores and visual profile.")
    with y:
        st.markdown("#### Explore the evidence")
        st.write("Review response patterns and scenario distributions as real submissions are collected.")
        st.markdown("#### Explain the model")
        st.write("Compare preliminary ML models and inspect feature contributions with SHAP when the dataset is adequate.")
    st.caption("Model performance will only be displayed after sufficient genuine respondent data has been collected. Voluntary convenience samples are not representative of all UK adults.")

def page_assessment():
    show_header()
    st.header("Financial Behaviour Assessment")
    st.write("For adults aged 18+ who currently live in the United Kingdom. Please do not enter your name, email, address, account details or other identifying information.")
    with st.form("assessment_form"):
        consent=st.checkbox("I am aged 18 or over, currently live in the UK, have read the project notice, and voluntarily agree to submit anonymous responses for an academic project.")
        age_band=st.selectbox("Age band",["Prefer not to say","18–24","25–34","35–44","45–54","55–64","65+"])
        region=st.selectbox("UK nation/region",["Prefer not to say","England","Scotland","Wales","Northern Ireland"])
        employment=st.selectbox("Current situation (optional)",["Prefer not to say","Employed","Self-employed","Student","Unemployed","Retired","Other"])
        st.divider()
        answers={}
        for dim in DIMENSIONS:
            st.subheader(dim)
            for qid,d,q,source,typ in QUESTIONS:
                if d==dim:
                    answers[qid]=st.radio(q,OPTIONS[typ],horizontal=True,key=f"q_{qid}")
                    with st.expander("Question source"):
                        st.caption(source)
        st.divider()
        st.subheader("Financial scenario responses")
        scenario_answers={}
        for sid,sc in SCENARIOS.items():
            st.markdown(f"**{sc['title']}**")
            st.write(sc["prompt"])
            scenario_answers[sid]=st.radio("Choose the response closest to what you would actually do:",sc["options"],key=f"sc_{sid}")
        submitted=st.form_submit_button("Submit anonymous assessment",type="primary",use_container_width=True)
    if submitted:
        if not consent:
            st.error("Please confirm consent and age eligibility before submitting.")
        elif len(answers)!=len(QUESTIONS):
            st.error("Please answer all questionnaire items.")
        else:
            scores=score_answers(answers)
            rid=save_submission({"age_band":age_band,"uk_region":region,"employment_status":employment},answers,scenario_answers,scores)
            st.session_state["last_scores"]=scores[1]
            st.session_state["last_rid"]=rid
            st.success("Your anonymous response has been saved on this app's local data store.")
            st.write("Your respondent reference:",rid[:8])
            st.info("Your responses are used for an academic research prototype. This local prototype does not provide financial recommendations.")

def page_persona():
    show_header()
    st.header("My Financial Persona")
    scores=st.session_state.get("last_scores")
    if scores is None:
        st.warning("Complete the Financial Assessment first to see your profile in this session.")
        return
    st.caption("Scores are descriptive indicators calculated from this project's research-informed questionnaire; they are not validated financial ratings.")
    c1,c2=st.columns(2)
    with c1:
        for dim,val in scores.items():
            st.metric(dim,f"{val:.0f}/100")
            st.progress(int(val))
    with c2:
        fig,ax=plt.subplots(figsize=(7,5),subplot_kw={"polar":True})
        names=list(scores.keys())
        vals=list(scores.values())
        angles=np.linspace(0,2*np.pi,len(names),endpoint=False).tolist()
        vals2=vals+[vals[0]]
        angles2=angles+[angles[0]]
        ax.plot(angles2,vals2,marker="o")
        ax.fill(angles2,vals2,alpha=.15)
        ax.set_xticks(angles)
        ax.set_xticklabels(["Budgeting","Saving","Credit","Risk","Confidence","Long-term"],fontsize=8)
        ax.set_ylim(0,100)
        ax.set_title("Financial Behaviour Profile")
        st.pyplot(fig)
        plt.close(fig)
    st.caption("A higher score indicates stronger endorsement of the positive direction of that dimension. Risk tolerance is intentionally not labelled as good or bad.")

def page_insights():
    show_header()
    st.header("Research Insights")
    responses=load_csv(RESP_FILE)
    scen=load_csv(SCENARIO_FILE)
    if responses.empty:
        st.info("No participant data has been collected yet. Complete the assessment or import an approved dataset.")
        return
    st.metric("Anonymous respondents",responses["respondent_id"].nunique())
    st.subheader("Dimension distributions")
    cols=[f"score_{d}" for d in DIMENSIONS if f"score_{d}" in responses.columns]
    if cols:
        summary=responses[cols].mean().rename(lambda x:x.replace("score_",""))
        st.bar_chart(summary)
    if not scen.empty:
        st.subheader("Observed scenario choices")
        st.bar_chart(scen["target"].value_counts())
        st.dataframe(scen.groupby(["scenario_title","target"]).size().rename("responses").reset_index(),use_container_width=True)
    st.caption("These are descriptive summaries of submitted responses, not estimates for the UK population. Sampling is voluntary and may be non-representative.")

def page_model():
    show_header()
    st.header("Machine Learning & Explainability")
    data=load_csv(SCENARIO_FILE)
    if data.empty:
        st.warning("No real scenario-response data is available yet.")
        return
    n=data["respondent_id"].nunique()
    st.metric("Unique respondents",n)
    if n<30:
        st.warning(f"Only {n} unique respondents have been collected. For this prototype, wait for at least 30 participants before attempting a preliminary model. Even 30 is not enough to claim robust generalisation.")
        st.dataframe(data.head(10),use_container_width=True)
        return
    if data["target"].nunique()<2:
        st.warning("At least two different observed response classes are required.")
        return
    if st.button("Train and evaluate models",type="primary"):
        try:
            results=train_models(data)
            summary=pd.DataFrame([{ "Model":name,"Accuracy":v["accuracy"],"Precision":v["precision"],"Recall":v["recall"],"F1":v["f1"]} for name,v in results.items()]).set_index("Model")
            st.subheader("Participant-level holdout evaluation")
            st.dataframe(summary.style.format("{:.3f}"),use_container_width=True)
            best=max(results,key=lambda k:results[k]["f1"])
            st.success(f"Best preliminary model by weighted F1: {best}")
            st.caption("A group-based split keeps each respondent entirely in either training or testing. Results remain preliminary and depend on sample size and representativeness.")
            st.subheader("Classification report")
            v=results[best]
            st.code(classification_report(v["actual"],v["pred"],zero_division=0))
            st.subheader("Model feature importance")
            model=results[best]["model"].named_steps["model"]
            fitted_prep=results[best]["model"].named_steps["prep"]
            try:
                feature_names=fitted_prep.get_feature_names_out()
            except Exception:
                feature_names=[f"Feature {i+1}" for i in range(len(getattr(model,"feature_importances_",[])))]
            if hasattr(model,"feature_importances_"):
                imp=pd.Series(model.feature_importances_,index=feature_names).sort_values()
                st.bar_chart(imp)
            else:
                st.info("Random Forest feature importance is available by selecting the Random Forest model when comparing models.")
            st.subheader("SHAP explanation")
            try:
                import shap
                if hasattr(model,"feature_importances_"):
                    transformed=fitted_prep.transform(v["X_test"])
                    explainer=shap.TreeExplainer(model)
                    shap_values=explainer.shap_values(transformed)
                    st.caption("SHAP values show how model features contribute to predictions on the held-out sample. They describe model behaviour, not causal effects.")
                    fig_shap,ax_shap=plt.subplots(figsize=(8,5))
                    shap.summary_plot(shap_values,transformed,feature_names=feature_names,show=False)
                    st.pyplot(plt.gcf())
                    plt.close("all")
                else:
                    st.info("Select Random Forest for the current TreeSHAP implementation.")
            except ImportError:
                st.info("Install the optional SHAP dependency with: pip install shap. Then rerun the app to enable TreeSHAP explanations.")
            except Exception as shap_error:
                st.warning(f"SHAP could not be generated for this sample: {shap_error}")
            st.caption("Model metrics and explanations are preliminary and depend on the size and representativeness of the collected UK respondent sample.")
        except Exception as e:
            st.error(f"Training could not be completed: {e}")

def page_research():
    show_header()
    st.header("Research & Methodology")
    st.markdown("""
### Research foundations
1. **OECD (2026), OECD/INFE Toolkit for Measuring Financial Literacy, Inclusion and Well-Being 2026.** Questionnaire items used/adapted include budgeting (QF2), active saving (QF3), expenditure shock (QF4), financial attitudes (QS1), financial behaviours (QS2), and confidence/resilience (QS3). The OECD toolkit is licensed CC BY 4.0. Adaptations are identified in the question-source notes.
2. **Money and Pensions Service (2022), Adult Financial Wellbeing Survey 2021.** Informs the UK context and the domains of active saving, credit management, spending tracking, pension saving, financial resilience and confidence.
3. **Grable, J. & Lytton, R. (1999), Financial risk tolerance revisited: the development of a risk assessment instrument, Financial Services Review, 8(3), 163–181.** Informs the risk-tolerance construct. The three risk items in this prototype are project-written prompts, not the original validated 13-item scale.
""")
    st.markdown("### Scoring")
    st.write("Likert and frequency responses are mapped to 1–5. Reverse-worded items are reverse scored. Each dimension is the mean of its three items, multiplied by 20 to give a 0–100 descriptive indicator.")
    st.markdown("### Data and modelling")
    st.write("Each participant contributes one anonymous questionnaire record and three scenario-response records. Models use the six dimension scores plus scenario identifier. A group-based train/test split is used to avoid placing the same participant in both sets.")
    st.markdown("### Ethics and limitations")
    st.write("- Participation is voluntary and restricted in this app to adults (18+) currently living in the UK.")
    st.write("- No names, emails, exact addresses, account numbers or sensitive financial credentials are requested.")
    st.write("- Responses are stored locally in CSV files by the app operator. For public deployment, configure secure storage, access controls, retention and an appropriate privacy notice.")
    st.write("- A convenience sample is not nationally representative. Do not generalise findings to all UK adults.")
    st.write("- Persona scores are project-defined descriptive indicators, not validated psychometric scales, financial advice or creditworthiness measures.")
    st.write("- Scenario choices are self-reported hypothetical intentions, not observed transactions or proof of future behaviour.")
    st.markdown("### Source links")
    st.markdown("- [OECD/INFE Toolkit 2026](https://doi.org/10.1787/92f2d439-en)")
    st.markdown("- [Money and Pensions Service — Adult Financial Wellbeing Survey 2021](https://maps.org.uk/en/publications/research/2021/financial-wellbeing-survey-2021)")
    st.markdown("- [Grable & Lytton (1999)](https://doi.org/10.1016/S1057-0810(99)00041-4)")
    st.markdown("- [Grable & Lytton scale retrospective (2015)](https://doi.org/10.61190/fsr.v24i2.3240)")
    st.caption("Research source access and use must follow each source's licence and any applicable permissions.")

def page_data():
    show_header()
    st.header("Data Management")
    st.write("Download the collected anonymous data for analysis. Do not share raw data publicly without checking consent and institutional requirements.")
    r=load_csv(RESP_FILE); s=load_csv(SCENARIO_FILE)
    a,b=st.columns(2)
    with a:
        st.metric("Assessment records",len(r))
        if not r.empty:
            st.download_button("Download questionnaire responses CSV",r.to_csv(index=False).encode("utf-8"),"personatwin_responses.csv","text/csv")
    with b:
        st.metric("Scenario records",len(s))
        if not s.empty:
            st.download_button("Download scenario responses CSV",s.to_csv(index=False).encode("utf-8"),"personatwin_scenarios.csv","text/csv")
    st.subheader("Import existing data")
    up=st.file_uploader("Upload scenario-response CSV using the app's schema",type=["csv"])
    if up is not None:
        imported=pd.read_csv(up)
        required={"respondent_id","Scenario","target",*DIMENSIONS}
        if required.issubset(imported.columns):
            imported.to_csv(SCENARIO_FILE,index=False)
            st.success("Scenario dataset imported.")
        else:
            st.error("CSV does not contain the required columns. Download a sample after collecting a response, or use the in-app collection workflow.")
    st.warning("This prototype stores files on the machine/container running Streamlit. Public cloud hosting may have ephemeral storage; connect a secure persistent database before real public deployment.")

pages={
    "Home":page_home,
    "Financial Assessment":page_assessment,
    "My Financial Persona":page_persona,
    "Financial Insights":page_insights,
    "Machine Learning & SHAP":page_model,
    "Research & Methodology":page_research,
    "Data Management":page_data
}
with st.sidebar:
    st.markdown("<div class='pt-kicker'>PERSONAL FINANCE • AI</div>",unsafe_allow_html=True)
    st.title("PersonaTwin")
    st.caption("UK Financial Behaviour Lab")
    selected=st.radio("Navigate",list(pages.keys()))
    st.divider()
    st.caption("Academic research prototype • Not financial advice")
pages[selected]()
