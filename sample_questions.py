"""Deterministic implementations of the 40 NIRF Annexure-A sample questions."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PARAMS = [c for c in ["tlr", "rpc", "go", "oi", "pr"]]
PARAM_LABELS = {"tlr": "TLR", "rpc": "RPC", "go": "GO", "oi": "OI", "pr": "PR"}

QUESTIONS = {
1: "How has the overall NIRF ranking of a selected institution changed from 2017 to 2025?",
2: "Which institutions have shown the greatest improvement in rank over the nine years?",
3: "Which institutions have experienced the largest decline in ranking?",
4: "How has the average score of institutions changed over time?",
5: "Which institutions have remained consistently in the top 10 across multiple years?",
6: "Which states have the highest number of institutions in the top 100?",
7: "How is the geographical distribution of top-ranked institutions changing from 2017 to 2025?",
8: "Which states have improved their representation in the top 100 over time?",
9: "How do the rankings of institutions in Karnataka compare with those in Tamil Nadu?",
10: "Which states have produced the largest number of newly appearing institutions in the rankings?",
11: "What is the distribution of NIRF scores for a selected year?",
12: "What is the distribution of overall ranks across all institutions?",
13: "What is the average, median, minimum, and maximum score in a given year?",
14: "What percentage of ranked institutions fall into each score quartile?",
15: "How does the spread of scores differ across NIRF categories?",
16: "How strongly are Research and Professional Practice (RPC) scores associated with overall scores?",
17: "How does Perception (PR) relate to overall scores?",
18: "Which NIRF parameter has the strongest correlation with the overall score?",
19: "How are TLR and RPC scores related across institutions?",
20: "Has the correlation between research scores and overall scores changed from 2017 to 2025?",
21: "How do the top five institutions compare across all five NIRF parameters?",
22: "Which institutions have high research scores but relatively low teaching scores?",
23: "Which institutions have high teaching scores but relatively low perception scores?",
24: "Which institutions have improved their overall scores despite declining scores in one or more parameters?",
25: "Which institutions have the largest difference between their strongest and weakest parameters?",
26: "How does the score distribution differ between the top 10 and the remaining institutions?",
27: "Are NIRF scores more dispersed in one year than another?",
28: "Which NIRF parameter exhibits the greatest variability across institutions?",
29: "How do the median and interquartile range of scores change over time?",
30: "How does the score distribution differ across states?",
31: "How many institutions moved from the 51–100 band into the top 50 between two selected years?",
32: "Which institutions entered or exited the top 10 between consecutive years?",
33: "What percentage of institutions remained in the same rank band from one year to the next?",
34: "Which institutions experienced the largest upward and downward rank movements?",
35: "How many institutions moved up, moved down, or retained their positions each year?",
36: "Which institutions have unusually high research scores relative to their overall scores?",
37: "Which institutions perform substantially better in research than in teaching?",
38: "Which institutions experienced an unusually large year-to-year score change?",
39: "Which institutions have a strong parameter score but a relatively poor overall rank?",
40: "Which institutions have unusually high ranking volatility compared with similar institutions?",
}


def _base(df, category=None, year=None, state=None):
    x = df.copy()
    if category and category != "All" and "category" in x:
        x = x[x.category.astype(str) == category]
    if year is not None and "year" in x:
        x = x[x.year == int(year)]
    if state and state != "All" and "state" in x:
        x = x[x.state.astype(str) == state]
    return x


def _need(x, cols):
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise ValueError("Dataset is missing required column(s): " + ", ".join(missing))


def _pair_years(df, category, year1, year2):
    x = _base(df, category=category)
    _need(x, ["year", "institution", "rank_num"])
    keys = ["institution"]
    if "category" in x.columns:
        keys.append("category")
    cols_a = keys + ["rank_num"]
    if "state" in x.columns:
        cols_a.append("state")
    a = x[x.year == int(year1)][cols_a].rename(columns={"rank_num":"rank_first", "state":"state_first"})
    b = x[x.year == int(year2)][keys + ["rank_num"]].rename(columns={"rank_num":"rank_second"})
    return a.merge(b, on=keys, how="inner")


def _chart_table(fig, table, note=""):
    return {"figure": fig, "table": table.reset_index(drop=True), "note": note}


def run_sample_question(qid, df, category="All", year=2025, year2=2025, institution=None, state="All", top_n=15):
    """Run one official sample question. All numeric results are calculated in pandas."""
    if qid not in QUESTIONS:
        raise ValueError("Choose a question number from 1 to 40.")
    x = _base(df, category=category, state=state)
    # data_utils.prepare_data standardizes rank_num to rank; retain a consistent alias here.
    if "rank_num" not in x.columns and "rank" in x.columns:
        x["rank_num"] = pd.to_numeric(x["rank"], errors="coerce")
    if "pr" not in x.columns and "perception" in x.columns:
        x["pr"] = pd.to_numeric(x["perception"], errors="coerce")
    y = int(year)
    y2 = int(year2)
    n = max(5, min(int(top_n), 50))
    category_specific_rank_questions = {1, 2, 3, 5, 6, 7, 8, 9, 12, 31, 32, 33, 34, 35, 40}
    if qid in category_specific_rank_questions and category == "All":
        raise ValueError("Select one NIRF category (for example Overall or Engineering). Rank positions from different categories should not be mixed.")
    _need(x, ["year", "institution"])
    fig = None
    table = pd.DataFrame()
    note = ""

    if qid == 1:
        _need(x, ["rank_num"])
        if not institution:
            raise ValueError("Select an institution to answer Question 1.")
        z = x[x.institution.astype(str).str.contains(str(institution), case=False, na=False)].copy()
        z = z[(z.year >= 2017) & (z.year <= 2025)].sort_values("year")
        if z.empty: raise ValueError("No matching institution found in the selected category.")
        fig = px.line(z, x="year", y="rank_num", markers=True, hover_data=["score"] if "score" in z else None, title=f"NIRF rank trend: {z.institution.iloc[0]}")
        fig.update_yaxes(autorange="reversed", title="Rank (lower is better)")
        table = z[[c for c in ["year","institution","rank_num","score"] if c in z]]
    elif qid in (2, 3, 34):
        _need(x, ["rank_num"])
        a, b = sorted(x.year.dropna().unique())[:1], sorted(x.year.dropna().unique())[-1:]
        if not a or not b: raise ValueError("Year values are unavailable.")
        pair = _pair_years(x, category, int(a[0]), int(b[0]))
        pair["rank_movement"] = pair.rank_first - pair.rank_second  # positive = moved up
        if qid == 2:
            table = pair.sort_values("rank_movement", ascending=False).head(n)
            title = "Institutions with greatest rank improvement"
        elif qid == 3:
            table = pair.sort_values("rank_movement").head(n)
            title = "Institutions with largest rank decline"
        else:
            up = pair.sort_values("rank_movement", ascending=False).head(n).assign(movement="Up")
            down = pair.sort_values("rank_movement").head(n).assign(movement="Down")
            table = pd.concat([up, down], ignore_index=True)
            title = "Largest upward and downward rank movements"
        fig = px.bar(table, x="rank_movement", y="institution", color="rank_movement", orientation="h", title=title)
        note = f"Compared institutions present in both {int(a[0])} and {int(b[0])}. Positive movement means rank improved."
    elif qid == 4:
        _need(x, ["score"])
        table = x.groupby("year", as_index=False).score.mean().sort_values("year")
        fig = px.line(table, x="year", y="score", markers=True, title="Average NIRF score over time")
    elif qid == 5:
        _need(x, ["rank_num"])
        z = x[x.rank_num <= 10]
        group_cols = ["institution"] + (["category"] if "category" in z.columns else [])
        table = z.groupby(group_cols, as_index=False).agg(years_in_top10=("year","nunique"), first_year=("year","min"), last_year=("year","max")).sort_values(["years_in_top10","institution"], ascending=[False,True])
        table = table[table.years_in_top10 > 1].head(n)
        fig = px.bar(table, x="institution", y="years_in_top10", title="Institutions repeatedly ranked in the top 10")
    elif qid in (6, 7, 8):
        _need(x, ["rank_num", "state"])
        z = x[x.rank_num <= 100]
        if qid == 6:
            z = z[z.year == y]
            table = z.groupby("state", as_index=False).institution.nunique().rename(columns={"institution":"top100_institutions"}).sort_values("top100_institutions", ascending=False).head(n)
            fig = px.bar(table, x="state", y="top100_institutions", title=f"Top 100 institutions by state — {y}")
        elif qid == 7:
            counts = z.groupby(["year","state"], as_index=False).institution.nunique().rename(columns={"institution":"top100_institutions"})
            states = counts.groupby("state").top100_institutions.sum().nlargest(8).index
            table = counts[counts.state.isin(states)]
            fig = px.line(table, x="year", y="top100_institutions", color="state", markers=True, title="Top-100 geographical distribution over time")
        else:
            yrs = sorted(z.year.dropna().unique())
            if len(yrs) < 2: raise ValueError("At least two years are required.")
            first, last = int(yrs[0]), int(yrs[-1])
            counts = z[z.year.isin([first,last])].groupby(["year","state"]).institution.nunique().unstack(fill_value=0)
            table = pd.DataFrame({"state":counts.columns, f"top100_{first}":counts.loc[first] if first in counts.index else 0, f"top100_{last}":counts.loc[last] if last in counts.index else 0}).reset_index(drop=True)
            table["change"] = table[f"top100_{last}"] - table[f"top100_{first}"]
            table = table.sort_values("change", ascending=False).head(n)
            fig = px.bar(table, x="state", y="change", title=f"Change in top-100 representation ({first} to {last})")
    elif qid == 9:
        _need(x, ["rank_num", "state"])
        z = x[(x.year == y) & (x.state.isin(["Karnataka","Tamil Nadu"]))]
        table = z.groupby("state", as_index=False).rank_num.agg(["mean","median","min","max","count"]).reset_index()
        fig = px.box(z, x="state", y="rank_num", points="all", title=f"Institution rank comparison: Karnataka vs Tamil Nadu ({y})")
        note = "Lower rank numbers indicate better positions; category selection controls comparability."
    elif qid == 10:
        _need(x, ["state"])
        if "first_year_in_category" in x:
            z = x[x.first_year_in_category == x.year]
        else:
            z = x.sort_values("year").drop_duplicates(["category","institution"], keep="first")
        table = z.groupby("state", as_index=False).institution.nunique().rename(columns={"institution":"new_institutions"}).sort_values("new_institutions", ascending=False).head(n)
        fig = px.bar(table, x="state", y="new_institutions", title="Newly appearing ranked institutions by state")
    elif qid == 11:
        _need(x, ["score"])
        z = x[x.year == y]
        table = z[["institution","score"]].dropna().sort_values("score", ascending=False)
        fig = px.histogram(z, x="score", nbins=30, title=f"NIRF score distribution — {y}")
    elif qid == 12:
        _need(x, ["rank_num"])
        z = x[x.rank_num.notna()]
        table = z[["institution","rank_num"]].sort_values("rank_num")
        fig = px.histogram(z, x="rank_num", nbins=30, title="Distribution of institution ranks")
    elif qid == 13:
        _need(x, ["score"])
        z = x[x.year == y].score.dropna()
        if z.empty: raise ValueError(f"No score records found for {y}.")
        table = pd.DataFrame({"statistic":["Average","Median","Minimum","Maximum"],"score":[z.mean(),z.median(),z.min(),z.max()]})
        fig = px.bar(table, x="statistic", y="score", title=f"Score summary statistics — {y}")
    elif qid == 14:
        _need(x, ["score"])
        z = x[x.year == y]
        if "score_quartile" in z:
            table = z.score_quartile.value_counts(dropna=False).rename_axis("quartile").reset_index(name="institutions")
        else:
            z = z.copy(); z["quartile"] = pd.qcut(z.score, 4, labels=["Q4 (lowest)","Q3","Q2","Q1 (highest)"])
            table = z.quartile.value_counts().rename_axis("quartile").reset_index(name="institutions")
        table["percentage"] = table.institutions / table.institutions.sum() * 100
        fig = px.pie(table, names="quartile", values="percentage", title=f"Score quartiles — {y}")
    elif qid == 15:
        _need(x, ["score","category"])
        z = x[x.year == y]
        table = z.groupby("category").score.agg(["count","mean","median","std","min","max"]).reset_index()
        fig = px.box(z, x="category", y="score", points="outliers", title=f"Score spread across categories — {y}")
    elif qid in (16,17,19):
        _need(x, ["score"])
        p = {16:"rpc",17:"pr",19:"tlr"}[qid]
        _need(x, [p])
        z = x[x.year == y][[p,"score","institution"]].dropna()
        corr = z[p].corr(z.score)
        table = pd.DataFrame({"parameter":[PARAM_LABELS[p]],"overall_score_correlation":[corr],"records":[len(z)]})
        fig = px.scatter(z, x=p, y="score", hover_name="institution", trendline="ols", title=f"{PARAM_LABELS[p]} vs overall score — r={corr:.3f}")
    elif qid == 18:
        _need(x, ["score"] + PARAMS)
        z = x[x.year == y]
        corr = z[PARAMS + ["score"]].corr(numeric_only=True)["score"].drop("score").sort_values(ascending=False)
        table = corr.rename_axis("parameter").reset_index(name="correlation_with_score")
        fig = px.bar(table, x="parameter", y="correlation_with_score", title=f"NIRF parameter correlation with overall score — {y}")
    elif qid == 20:
        _need(x, ["rpc","score"])
        rows=[]
        for yr, g in x.groupby("year"):
            d=g[["rpc","score"]].dropna()
            rows.append({"year":yr,"correlation":d.rpc.corr(d.score),"records":len(d)})
        table=pd.DataFrame(rows).sort_values("year")
        fig=px.line(table,x="year",y="correlation",markers=True,title="RPC–overall score correlation over time")
    elif qid == 21:
        _need(x, ["rank_num"] + PARAMS)
        z=x[x.year==y].nsmallest(5,"rank_num")
        table=z[["institution","rank_num"]+PARAMS]
        long=z.melt(id_vars=["institution"],value_vars=PARAMS,var_name="parameter",value_name="parameter_value")
        fig=px.line(long,x="parameter",y="parameter_value",color="institution",markers=True,title=f"Top five institutions across NIRF parameters — {y}")
    elif qid in (22,23):
        a,b=("rpc","tlr") if qid==22 else ("tlr","pr")
        _need(x,[a,b,"institution"])
        z=x[x.year==y].dropna(subset=[a,b]).copy()
        high=z[a].quantile(.75); low=z[b].quantile(.25)
        table=z[(z[a]>=high)&(z[b]<=low)].sort_values(a,ascending=False)[["institution",a,b]+(["rank_num"] if "rank_num" in z else [])].head(n)
        fig=px.scatter(z,x=a,y=b,hover_name="institution",title=f"High {a.upper()} and low {b.upper()} institutions — {y}")
        fig.add_hline(y=low,line_dash="dash"); fig.add_vline(x=high,line_dash="dash")
        note=f"High means at or above the 75th percentile; low means at or below the 25th percentile for the selected data."
    elif qid == 24:
        _need(x,["score_change","params_declined"])
        z=x[(x.year==y)&(x.score_change>0)&(x.params_declined>0)].copy()
        table=z.sort_values("score_change",ascending=False)[[c for c in ["institution","score","score_change","params_declined","rank_num"] if c in z]].head(n)
        fig=px.bar(table,x="score_change",y="institution",orientation="h",title=f"Score improved despite parameter declines — {y}")
    elif qid == 25:
        _need(x,["param_gap"])
        z=x[x.year==y]
        table=z.sort_values("param_gap",ascending=False)[[c for c in ["institution","param_gap","strongest_param","weakest_param"] if c in z]].head(n)
        fig=px.bar(table,x="param_gap",y="institution",orientation="h",title=f"Largest gap between strongest and weakest parameters — {y}")
    elif qid == 26:
        _need(x,["rank_num","score"])
        z=x[x.year==y].copy(); z["group"] = np.where(z.rank_num<=10,"Top 10","Remaining institutions")
        table=z.groupby("group").score.agg(["count","mean","median","std"]).reset_index()
        fig=px.box(z,x="group",y="score",points="outliers",title=f"Score distribution: top 10 vs remaining — {y}")
    elif qid == 27:
        _need(x,["score"])
        table=x.groupby("year").score.agg(count="count",mean="mean",std="std",median="median").reset_index()
        table["iqr"]=x.groupby("year").score.quantile(.75)-x.groupby("year").score.quantile(.25)
        fig=px.line(table,x="year",y="std",markers=True,title="Score dispersion (standard deviation) by year")
    elif qid == 28:
        _need(x,PARAMS)
        table=x[PARAMS].agg(["count","mean","std","var"]).T.reset_index(names="parameter").sort_values("std",ascending=False)
        fig=px.bar(table,x="parameter",y="std",title="Variability of NIRF parameters (standard deviation)")
    elif qid == 29:
        _need(x,["score"])
        table=x.groupby("year").score.agg(median="median",q1=lambda s:s.quantile(.25),q3=lambda s:s.quantile(.75)).reset_index()
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=table.year,y=table.q3,mode="lines",name="Q3 (75th percentile)"))
        fig.add_trace(go.Scatter(x=table.year,y=table.q1,mode="lines",name="Q1 (25th percentile)",fill="tonexty"))
        fig.add_trace(go.Scatter(x=table.year,y=table["median"],mode="lines+markers",name="Median"))
        fig.update_layout(title="Median and interquartile range of scores over time",xaxis_title="Year",yaxis_title="Score")
    elif qid == 30:
        _need(x,["score","state"])
        z=x[x.year==y]
        states=z.state.value_counts().head(12).index
        z=z[z.state.isin(states)]
        table=z.groupby("state").score.agg(["count","median","mean","std"]).reset_index()
        fig=px.box(z,x="state",y="score",points="outliers",title=f"Score distribution across states — {y}")
    elif qid in (31,32,33,35):
        _need(x,["rank_num"])
        yrs=sorted(x.year.dropna().unique())
        if qid==31:
            pair=_pair_years(x,category,y,y2)
            table=pair[(pair.rank_first.between(51,100))&(pair.rank_second<=50)].sort_values("rank_second")
            fig=px.bar(table,x="rank_second",y="institution",orientation="h",title=f"Moved from ranks 51–100 to top 50 ({y} to {y2})")
            note=f"Count: {len(table)} institutions present in both selected years and meeting the transition condition."
        elif qid==32:
            if y2 != y+1: y2=y+1
            pair=_pair_years(x,category,y,y2)
            entered=set(pair.loc[pair.rank_first>10,"institution"]) & set(pair.loc[pair.rank_second<=10,"institution"])
            exited=set(pair.loc[pair.rank_first<=10,"institution"]) & set(pair.loc[pair.rank_second>10,"institution"])
            # Include entrants/exits among all institutions in each year, including new/disappearing records.
            first=x[x.year==y]; second=x[x.year==y2]
            key_cols = ["institution"] + (["category"] if "category" in x.columns else [])
            first_top = set(map(tuple, first.loc[first.rank_num<=10, key_cols].drop_duplicates().to_numpy()))
            second_top = set(map(tuple, second.loc[second.rank_num<=10, key_cols].drop_duplicates().to_numpy()))
            entered = second_top - first_top
            exited = first_top - second_top
            table=pd.DataFrame([{**dict(zip(key_cols,v)),"transition":"Entered top 10"} for v in sorted(entered)]+[{**dict(zip(key_cols,v)),"transition":"Exited top 10"} for v in sorted(exited)])
            if table.empty:
                table = pd.DataFrame(columns=key_cols + ["transition"])
            counts = pd.DataFrame({"transition":["Entered top 10","Exited top 10"],"institutions":[len(entered),len(exited)]})
            fig=px.bar(counts,x="transition",y="institutions",color="transition",title=f"Top-10 entries and exits: {y} to {y2}")
        elif qid==33:
            pair=_pair_years(x,category,y,y2)
            def band(r):
                if r<=10:return "1-10"
                if r<=25:return "11-25"
                if r<=50:return "26-50"
                if r<=100:return "51-100"
                if r<=150:return "101-150"
                if r<=200:return "151-200"
                return "200+"
            pair["band_first"]=pair.rank_first.map(band); pair["band_second"]=pair.rank_second.map(band)
            pair["same_band"]=pair.band_first==pair.band_second
            pct=pair.same_band.mean()*100 if len(pair) else 0
            table=pd.DataFrame([{"year_from":y,"year_to":y2,"institutions_compared":len(pair),"same_band_count":int(pair.same_band.sum()),"same_band_percentage":pct}])
            fig=px.pie(pd.DataFrame({"status":["Same band","Changed band"],"institutions":[int(pair.same_band.sum()),int((~pair.same_band).sum())]}),names="status",values="institutions",title=f"Rank-band retention: {y} to {y2}")
        else:
            rows=[]
            for yr in yrs[1:]:
                prev=yr-1
                try: pair=_pair_years(x,category,prev,yr)
                except Exception: continue
                diff=pair.rank_first-pair.rank_second
                rows.append({"year":yr,"moved_up":int((diff>0).sum()),"moved_down":int((diff<0).sum()),"retained_rank":int((diff==0).sum()),"compared":len(pair)})
            table=pd.DataFrame(rows)
            if not table.empty:
                long=table.melt(id_vars="year",value_vars=["moved_up","moved_down","retained_rank"],var_name="movement",value_name="institutions")
                fig=px.bar(long,x="year",y="institutions",color="movement",barmode="group",title="Institutions moving up, down or retaining rank each year")
    elif qid == 36:
        _need(x,["rpc_residual_z"])
        z=x[x.year==y]
        table=z[z.rpc_residual_z>=2].sort_values("rpc_residual_z",ascending=False)[[c for c in ["institution","rpc","score","rpc_residual_z","rank_num"] if c in z]].head(n)
        fig=px.scatter(z,x="score",y="rpc",hover_name="institution",title=f"Research scores unusually high relative to overall score — {y}")
    elif qid == 37:
        _need(x,["rpc_minus_tlr"])
        z=x[x.year==y]
        table=z.sort_values("rpc_minus_tlr",ascending=False)[[c for c in ["institution","rpc","tlr","rpc_minus_tlr","rank_num"] if c in z]].head(n)
        fig=px.bar(table,x="rpc_minus_tlr",y="institution",orientation="h",title=f"Research score relative to teaching score — {y}")
    elif qid == 38:
        _need(x,["score_change"])
        z=x[x.year==y].copy()
        z["absolute_score_change"]=z.score_change.abs()
        table=z.sort_values("absolute_score_change",ascending=False)[[c for c in ["institution","score_change","score_change_z","score","prev_score"] if c in z]].head(n)
        fig=px.bar(table,x="score_change",y="institution",orientation="h",color="score_change",title=f"Largest year-to-year score changes — {y}")
    elif qid == 39:
        _need(x,PARAMS+["rank_num"])
        z=x[x.year==y].copy()
        long=z.melt(id_vars=["institution","rank_num"],value_vars=PARAMS,var_name="parameter",value_name="parameter_score")
        threshold=long.parameter_score.quantile(.75)
        poor_rank=long.rank_num.quantile(.5)
        table=long[(long.parameter_score>=threshold)&(long.rank_num>=poor_rank)].sort_values(["parameter_score","rank_num"],ascending=[False,False]).drop_duplicates("institution").head(n)
        fig=px.scatter(long,x="rank_num",y="parameter_score",color="parameter",hover_name="institution",title=f"Strong parameter scores with relatively poor ranks — {y}")
        fig.update_xaxes(autorange="reversed")
        note="Strong parameter = at/above the 75th percentile; relatively poor rank = rank number at/above the median."
    elif qid == 40:
        _need(x,["rank_volatility_sd"])
        z=x[x.year==y].copy() if "year" in x else x.copy()
        table=z.sort_values("rank_volatility_sd",ascending=False)[[c for c in ["institution","rank_volatility_sd","category","years_ranked"] if c in z]]
        dedup_cols = [c for c in ["institution", "category"] if c in table.columns]
        table=table.drop_duplicates(dedup_cols).head(n)
        fig=px.bar(table,x="rank_volatility_sd",y="institution",orientation="h",title="Institutions with highest ranking volatility")
        note="Volatility is the standard deviation of an institution's rank across the available years in its category."

    if fig is not None:
        fig.update_layout(height=520, margin=dict(l=20,r=20,t=65,b=35))
    if table is None or table.empty:
        note=(note+" No matching records were found for the selected filters.").strip()
    return _chart_table(fig, table if table is not None else pd.DataFrame(), note)
