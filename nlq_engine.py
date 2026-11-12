import os, json, re, time
import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


FALLBACK_MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
]

def _generate_with_retry(client, prompt):
    models = [MODEL] + [
        m for m in FALLBACK_MODELS if m != MODEL
    ]

    last_error = None

    for model in models:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )
                if response.text:
                    return response

                raise RuntimeError("Gemini returned an empty response.")

            except Exception as e:
                last_error = e
                error_text = str(e).lower()

                temporary = any(
                    code in error_text
                    for code in ["429", "500", "502", "503", "504",
                                 "unavailable", "timeout", "temporarily"]
                )

                if temporary and attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                    continue

                if temporary:
                    break

                raise

    raise RuntimeError(
        "Gemini is temporarily unavailable across the configured "
        "models. Please try again later. Details: " + str(last_error)
    )


def _get_client():
    import os
    from dotenv import load_dotenv
    from google import genai

    # Load the local .env file
    load_dotenv()

    # First try the environment variable
    key = os.getenv("GEMINI_API_KEY")

    # If not found, try Streamlit secrets
    if not key:
        try:
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None

    if not key:
        raise RuntimeError(
            "Gemini API key not found. "
            "Check GEMINI_API_KEY in your .env file."
        )

    return genai.Client(api_key=key)


def _schema(df):
    return {
        "available_columns": list(map(str, df.columns)),
        "years": sorted(df["year"].dropna().astype(int).unique().tolist()) if "year" in df else [],
        "categories": sorted(df["category"].dropna().astype(str).unique().tolist()) if "category" in df else [],
        "states": sorted(df["state"].dropna().astype(str).unique().tolist())[:100] if "state" in df else [],
        "numeric_columns": list(df.select_dtypes(include="number").columns),
        "sample_rows": df.head(3).fillna("").astype(str).to_dict(orient="records"),
    }

def _make_plan(question, df):
    client = _get_client()
    context = _schema(df)
    prompt = f"""
You are an NLQ query planner for a NIRF ranking dataset. Return ONLY valid JSON.
Never answer using outside knowledge. Create a query plan using only the provided columns.
Allowed operations: top_n, aggregate, trend, institution_history, compare_institutions,
state_summary, correlation, distribution, data_quality, filter_table.
JSON shape:
{{
 "operation": "top_n|aggregate|trend|institution_history|compare_institutions|state_summary|correlation|distribution|data_quality|filter_table",
 "year": integer or null,
 "start_year": integer or null,
 "end_year": integer or null,
 "category": string or null,
 "state": string or null,
 "institution": string or null,
 "institutions": [strings],
 "metric": "score|rank|tlr|rpc|go|oi|perception|count",
 "group_by": "year|category|state|institution|zone",
 "limit": integer between 1 and 50,
 "chart": "bar|line|scatter|histogram|none",
 "filters": {{}},
 "reason": "short description"
}}
Only select fields and values that exist in the supplied dataset context. If question asks for top ranked, sort rank ascending. If it asks highest score, sort score descending. If a column is absent, do not use it.
Dataset context:
{json.dumps(context, ensure_ascii=False)}
User question: {question}
"""
    response = _generate_with_retry(client, prompt)
    raw = (response.text or "").strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.I).strip()
    plan = json.loads(raw)
    return plan

def _apply_filters(df, plan):
    x = df.copy()
    if plan.get("year") is not None and "year" in x:
        x = x[x["year"] == int(plan["year"])]
    if plan.get("start_year") is not None and "year" in x:
        x = x[x["year"] >= int(plan["start_year"])]
    if plan.get("end_year") is not None and "year" in x:
        x = x[x["year"] <= int(plan["end_year"])]
    for field in ["category","state"]:
        val = plan.get(field)
        if val and field in x:
            x = x[x[field].astype(str).str.contains(re.escape(str(val)), case=False, na=False)]
    inst = plan.get("institution")
    if inst and "institution" in x:
        x = x[x["institution"].astype(str).str.contains(re.escape(str(inst)), case=False, na=False)]
    insts = plan.get("institutions") or []
    if insts and "institution" in x:
        x = x[x["institution"].astype(str).isin(insts)]
    return x


def _safe_answer(question, plan, result_text):
    client = _get_client()

    prompt = f"""Explain this NIRF data result in clear, concise language.
Do not add facts, values or institutions that are not in the result.
Mention when the result is empty or limited by missing data.

Question: {question}
Query plan: {json.dumps(plan)}
Python result: {result_text}

Write a short interpretation and state the selected filters."""

    try:
        response = _generate_with_retry(client, prompt)
        return (response.text or "").strip() or (
            "The analysis completed successfully."
        )

    except Exception:
        return (
            "Your NIRF analysis was calculated successfully using Python. "
            "Gemini is temporarily unavailable, so an AI explanation "
            "could not be generated.\n\n"
            "Here are the calculated results:\n\n"
            + result_text[:5000]
        )

def answer_question(question, df):
    plan = _make_plan(question, df)
    allowed = {"top_n","aggregate","trend","institution_history","compare_institutions",
               "state_summary","correlation","distribution","data_quality","filter_table"}
    op = plan.get("operation")
    if op not in allowed:
        raise ValueError("Gemini returned an unsupported query operation.")
    data = _apply_filters(df, plan)
    metric = plan.get("metric", "score")
    if metric not in df.columns and metric != "count":
        metric = "score" if "score" in df else "rank" if "rank" in df else "count"
    limit = max(1, min(int(plan.get("limit") or 10), 50))
    chart = None
    table = None

    if op == "top_n":
        sort_metric = "rank" if "rank" in data and re.search(r"\b(rank|top|best)\b", question, re.I) else metric
        if sort_metric == "count": sort_metric = "score" if "score" in data else "rank"
        if sort_metric not in data: raise ValueError("Required numeric column is not available.")
        ascending = sort_metric == "rank"
        table = data.sort_values(sort_metric, ascending=ascending).head(limit)
        if "institution" in table and sort_metric in table:
            chart = px.bar(table.sort_values(sort_metric, ascending=not ascending),
                           x=sort_metric, y="institution", orientation="h",
                           title=f"Top {len(table)} institutions by {sort_metric}")
    elif op in {"trend","institution_history"}:
        group = "year" if "year" in data else None
        if not group: raise ValueError("Year column is not available for trend analysis.")
        if metric not in data: metric = "score" if "score" in data else "rank"
        table = data.groupby(group, as_index=False)[metric].mean().sort_values(group)
        chart = px.line(table, x=group, y=metric, markers=True, title=f"{metric.title()} trend by year")
    elif op in {"aggregate","state_summary","compare_institutions"}:
        group = plan.get("group_by") or ("category" if "category" in data else "state")
        if group not in data: raise ValueError(f"Grouping column '{group}' is unavailable.")
        if metric == "count":
            table = data.groupby(group, as_index=False).size().rename(columns={"size":"count"})
            y = "count"
        else:
            if metric not in data: raise ValueError(f"Metric '{metric}' is unavailable.")
            table = data.groupby(group, as_index=False)[metric].agg(["mean","median","count"]).reset_index()
            y = "mean"
        table = table.sort_values(y, ascending=(metric == "rank")).head(limit)
        chart = px.bar(table, x=group, y=y, title=f"{metric.title()} by {group}")
    elif op == "correlation":
        nums = data.select_dtypes(include="number")
        if nums.shape[1] < 2: raise ValueError("At least two numeric columns are required.")
        table = nums.corr(numeric_only=True).round(3).reset_index(names="parameter")
        chart = px.imshow(nums.corr(numeric_only=True), text_auto=True, title="Correlation matrix")
    elif op == "distribution":
        if metric not in data: raise ValueError(f"Metric '{metric}' is unavailable.")
        table = data[[metric]].dropna()
        chart = px.histogram(table, x=metric, title=f"Distribution of {metric}")
    elif op == "data_quality":
        table = pd.DataFrame({"column":df.columns.astype(str),
                              "missing":df.isna().sum().values,
                              "missing_percent":(df.isna().mean()*100).round(2).values,
                              "unique":df.nunique(dropna=True).values})
        table.loc[len(table)] = ["DUPLICATE_ROWS", int(df.duplicated().sum()), None, None]
    else:
        table = data.head(limit)

    result_text = table.head(50).to_string(index=False) if table is not None else "No tabular result."
    if table is not None and table.empty:
        answer = "No matching records were found. Try changing the year, category, state or institution name."
    else:
        answer = _safe_answer(question, plan, result_text)
    return {"answer":answer, "table":table, "chart":chart, "plan":plan}
