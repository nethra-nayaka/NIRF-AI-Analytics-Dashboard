import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from data_utils import load_data, prepare_data, profile_data, quality_report, find_col
from nlq_engine import answer_question
from sample_questions import QUESTIONS, run_sample_question

st.set_page_config(page_title="NIRF AI Analytics", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.main {background:#f7f9fc;}
.block-container {padding-top:1.5rem; max-width:1500px;}
[data-testid="stMetric"] {background:white;padding:16px;border:1px solid #e8edf4;border-radius:12px;}
.hero {padding:22px 26px;border-radius:16px;background:linear-gradient(110deg,#102a43,#176b87);color:white;margin-bottom:18px;}
.hero h1 {color:white;margin:0;}
.hero p {color:#e7f4fa;margin:5px 0 0 0;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data/FINAL_CLEANED.csv")

@st.cache_data(show_spinner="Loading NIRF dataset...")
def get_data(path):
    raw = load_data(path)
    return prepare_data(raw)

st.sidebar.title("🎓 NIRF Explorer")
st.sidebar.caption("AI-powered ranking analytics | 2017–2025")
uploaded = st.sidebar.file_uploader("Upload FINAL_CLEANED.csv if needed", type=["csv"])
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        df = prepare_data(raw)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    if not DATA_PATH.exists():
        st.warning("Place your FINAL_CLEANED.csv inside the project's data folder, or upload it using the sidebar.")
        st.code("NIRF_AI_Streamlit_Gemini/data/FINAL_CLEANED.csv")
        st.stop()
    try:
        df = get_data(str(DATA_PATH))
    except Exception as e:
        st.error(f"Dataset loading failed: {e}")
        st.stop()

if df.empty:
    st.error("The CSV has no usable records.")
    st.stop()

profile = profile_data(df)
years = sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df else []
categories = sorted(df["category"].dropna().astype(str).unique().tolist()) if "category" in df else []
states = sorted(df["state"].dropna().astype(str).unique().tolist()) if "state" in df else []

st.sidebar.divider()
page = st.sidebar.radio("Navigation", [
    "Overview", "Ranking Analysis", "Category Analysis",
    "State Analysis", "Institution Comparison", "Data Quality", "Gemini NLQ Chatbot", "40 Sample Questions"
])

st.markdown('<div class="hero"><h1>NIRF AI Analytics Platform</h1><p>Explore institutional rankings, performance trends and data insights with Google Gemini.</p></div>', unsafe_allow_html=True)
st.caption(f"Dataset: {len(df):,} rows × {len(df.columns)} columns | Years: {min(years) if years else 'N/A'}–{max(years) if years else 'N/A'}")

def filters(data, key_prefix):
    c1, c2, c3 = st.columns(3)
    out = data.copy()
    if categories:
        cat = c1.selectbox("Category", ["All"] + categories, key=f"{key_prefix}_cat")
        if cat != "All": out = out[out["category"] == cat]
    if years:
        yr = c2.selectbox("Year", ["All"] + years, key=f"{key_prefix}_year")
        if yr != "All": out = out[out["year"] == int(yr)]
    if states:
        state = c3.selectbox("State", ["All"] + states, key=f"{key_prefix}_state")
        if state != "All": out = out[out["state"] == state]
    return out

rank_col = "rank" if "rank" in df else None
score_col = "score" if "score" in df else None
inst_col = "institution" if "institution" in df else None

if page == "Overview":
    st.subheader("Dashboard overview")
    view = filters(df, "overview")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Records", f"{len(view):,}")
    m2.metric("Institutions", f"{view['institution'].nunique():,}" if "institution" in view else "N/A")
    m3.metric("Categories", f"{view['category'].nunique():,}" if "category" in view else "N/A")
    m4.metric("Average score", f"{view['score'].mean():.2f}" if "score" in view and view["score"].notna().any() else "N/A")
    left,right = st.columns(2)
    if "year" in view and "score" in view:
        trend = view.groupby("year", as_index=False)["score"].mean()
        left.plotly_chart(px.line(trend, x="year", y="score", markers=True, title="Average score by year"), use_container_width=True)
    if "category" in view:
        counts = view.groupby("category", as_index=False).size().sort_values("size", ascending=False)
        right.plotly_chart(px.bar(counts, x="category", y="size", title="Records by category"), use_container_width=True)
    if "state" in view:
        st.plotly_chart(px.bar(view["state"].value_counts().head(15).rename_axis("State").reset_index(name="Records"),
                               x="State", y="Records", title="Top 15 states by number of records"), use_container_width=True)
    st.dataframe(view.head(1000), use_container_width=True)

elif page == "Ranking Analysis":
    st.subheader("Ranking and score trends")
    view = filters(df, "ranking")
    a,b = st.columns(2)
    if "rank" in view and "year" in view:
        yearly = view.groupby("year", as_index=False)["rank"].mean()
        a.plotly_chart(px.line(yearly, x="year", y="rank", markers=True, title="Average rank by year (lower is better)"), use_container_width=True)
    if "score" in view and "year" in view:
        yearly = view.groupby("year", as_index=False)["score"].mean()
        b.plotly_chart(px.line(yearly, x="year", y="score", markers=True, title="Average score by year"), use_container_width=True)
    if "rank" in view and "institution" in view:
        top_n = st.slider("Number of institutions", 5, 50, 10, key="rank_topn")
        sort_df = view.dropna(subset=["rank"]).sort_values("rank").head(top_n)
        st.plotly_chart(px.bar(sort_df.sort_values("rank", ascending=False), x="rank", y="institution", orientation="h",
                               title=f"Top {top_n} institutions by rank", hover_data=["score"] if "score" in view else None), use_container_width=True)
    st.dataframe(view, use_container_width=True)

elif page == "Category Analysis":
    st.subheader("Category-wise performance")
    view = filters(df, "category")
    if "category" in view:
        if "score" in view:
            summary = view.groupby("category", as_index=False).agg(Mean_Score=("score","mean"), Median_Score=("score","median"), Records=("category","size"))
            c1,c2 = st.columns(2)
            c1.plotly_chart(px.bar(summary.sort_values("Mean_Score", ascending=False), x="category", y="Mean_Score", title="Mean score by category"), use_container_width=True)
            c2.plotly_chart(px.bar(summary, x="category", y="Records", title="Records by category"), use_container_width=True)
            st.dataframe(summary, use_container_width=True)
        else:
            st.info("No recognized score column was found. Category counts are shown instead.")
            st.dataframe(view["category"].value_counts().rename_axis("Category").reset_index(name="Records"))
    st.dataframe(view.head(1000), use_container_width=True)

elif page == "State Analysis":
    st.subheader("Geographical distribution")
    view = filters(df, "state")
    if "state" in view:
        summary = view.groupby("state", as_index=False).agg(Records=("state","size"))
        if "score" in view: summary = summary.merge(view.groupby("state", as_index=False)["score"].mean(), on="state")
        c1,c2 = st.columns(2)
        c1.plotly_chart(px.bar(summary.sort_values("Records", ascending=False).head(20), x="state", y="Records", title="Top states by institutional records"), use_container_width=True)
        if "score" in summary:
            c2.plotly_chart(px.bar(summary.sort_values("score", ascending=False).head(20), x="state", y="score", title="Average score by state"), use_container_width=True)
        st.dataframe(summary.sort_values("Records", ascending=False), use_container_width=True)
    else:
        st.info("State column not detected. Review column mapping in data_utils.py.")

elif page == "Institution Comparison":
    st.subheader("Compare institutions across years")
    if "institution" not in df:
        st.error("Institution name column was not detected.")
    else:
        names = sorted(df["institution"].dropna().astype(str).unique().tolist())
        selected = st.multiselect("Select institutions (up to 8)", names, default=names[:2], max_selections=8)
        view = df[df["institution"].isin(selected)].copy()
        if "year" in view:
            metric = "rank" if "rank" in view and st.radio("Compare using", ["Rank", "Score"], horizontal=True) == "Rank" else "score"
            if metric in view:
                st.plotly_chart(px.line(view, x="year", y=metric, color="institution", markers=True,
                                        title=f"Institution {metric} trend", hover_data=["category"] if "category" in view else None), use_container_width=True)
        st.dataframe(view.sort_values(["institution","year"]) if "year" in view else view, use_container_width=True)

elif page == "Data Quality":
    st.subheader("Data quality audit")
    report = quality_report(df)
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Duplicate rows", f"{int(df.duplicated().sum()):,}")
    c3.metric("Columns", len(df.columns))
    st.markdown("**Missing values by column**")
    st.dataframe(report, use_container_width=True)
    st.markdown("**Dataset profile**")
    st.json(profile)
    st.download_button("Download quality report", report.to_csv(index=False).encode("utf-8"), "nirf_quality_report.csv", "text/csv")

elif page == "Gemini NLQ Chatbot":
    st.subheader("Ask questions about NIRF data")
    st.write("Ask in natural language. Gemini converts your question into a safe structured query; Python calculates the answer from your CSV.")
    with st.expander("Example questions"):
        st.markdown("""
        - Which are the top 10 Engineering institutions in 2025?
        - Compare average scores across categories in 2024.
        - Show the score trend of IIT Madras from 2017 to 2025.
        - Which states have the most institutions in the top 100?
        - What is the average score in Karnataka?
        - Show missing values and duplicate records.
        """)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)
    question = st.chat_input("Ask a question about NIRF rankings...")
    if question:
        st.session_state.chat_history.append({"role":"user","content":question})
        with st.chat_message("user"): st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Gemini is analysing your question..."):
                try:
                    result = answer_question(question, df)
                    st.markdown(result["answer"])
                    if result.get("table") is not None:
                        st.dataframe(result["table"], use_container_width=True)
                    if result.get("chart") is not None:
                        st.plotly_chart(result["chart"], use_container_width=True)
                    st.session_state.chat_history.append({
                        "role":"assistant","content":result["answer"],
                        "table":result.get("table"),"chart":result.get("chart")
                    })
                except Exception as e:
                    msg = f"Could not process the question: {e}"
                    st.error(msg)
                    st.session_state.chat_history.append({"role":"assistant","content":msg})
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

elif page == "40 Sample Questions":
    st.subheader("Official NIRF Sample Question Explorer")
    st.write("Select any of the 40 questions from Annexure A. The calculations are performed with Python on your cleaned CSV, and the dashboard generates a relevant chart and result table.")

    years_available = sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df else []
    categories_available = sorted(df["category"].dropna().astype(str).unique().tolist()) if "category" in df else []
    states_available = sorted(df["state"].dropna().astype(str).unique().tolist()) if "state" in df else []
    default_cat = "Overall" if "Overall" in categories_available else (categories_available[0] if categories_available else "All")

    section_names = {
        1: "1. Time-series and trend analysis", 2: "2. Geospatial and regional analysis",
        3: "3. Univariate analysis", 4: "4. Bivariate analysis and correlation",
        5: "5. Multivariate analysis", 6: "6. Distribution and statistical comparison",
        7: "7. Ranking transitions and institutional mobility", 8: "8. Outlier detection and performance gaps"
    }
    section_choice = st.selectbox("Question category", list(section_names.values()))
    section_num = [k for k, v in section_names.items() if v == section_choice][0]
    qids = list(range((section_num-1)*5+1, section_num*5+1))
    qid = st.selectbox("Choose a sample question", qids, format_func=lambda q: f"Q{q}. {QUESTIONS[q]}")

    c1, c2, c3 = st.columns(3)
    with c1:
        cat_choice = st.selectbox("NIRF category", ["All"] + categories_available, index=(1 + categories_available.index(default_cat)) if default_cat in categories_available else 0)
    with c2:
        default_year_index = max(0, len(years_available)-2)
        selected_year = st.selectbox("Year / starting year", years_available, index=default_year_index) if years_available else 2024
    with c3:
        selected_year2 = st.selectbox("Comparison / ending year", years_available, index=(len(years_available)-1)) if years_available else selected_year

    c4, c5, c6 = st.columns(3)
    with c4:
        state_choice = st.selectbox("State filter", ["All"] + states_available)
    with c5:
        institution_names = sorted(df["institution"].dropna().astype(str).unique().tolist()) if "institution" in df else []
        institution_choice = st.selectbox("Institution (for Q1)", ["Select institution"] + institution_names)
        if institution_choice == "Select institution": institution_choice = None
    with c6:
        top_n_choice = st.slider("Number of institutions to display", 5, 50, 15)

    if st.button("Generate answer and chart", type="primary"):
        try:
            with st.spinner("Calculating from the NIRF dataset..."):
                result = run_sample_question(
                    qid, df, category=cat_choice, year=selected_year,
                    year2=selected_year2, institution=institution_choice,
                    state=state_choice, top_n=top_n_choice
                )
            st.success(f"Question {qid} completed")
            if result["note"]:
                st.info(result["note"])
            if result["figure"] is not None:
                st.plotly_chart(result["figure"], use_container_width=True)
            st.markdown("**Calculated result**")
            st.dataframe(result["table"], use_container_width=True)
            st.download_button(
                "Download this question's result as CSV",
                result["table"].to_csv(index=False).encode("utf-8"),
                f"nirf_question_{qid}_result.csv", "text/csv"
            )
        except Exception as e:
            st.error(f"Could not generate Question {qid}: {e}")

st.sidebar.divider()
st.sidebar.caption("NIRF AI Analytics | Streamlit + Gemini")
st.sidebar.download_button("Download filtered dataset", df.to_csv(index=False).encode("utf-8"), "nirf_filtered_data.csv", "text/csv")
