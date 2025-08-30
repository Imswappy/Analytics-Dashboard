# =========================
# main.py  — with SQL Lab
# =========================
import io
import sqlite3
import time
from typing import Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st
from numerize.numerize import numerize
from streamlit_option_menu import option_menu

# -------------------------
# Page config + CSS + UI
# -------------------------
st.set_page_config(page_title="Descriptive Analytics", page_icon="🌎", layout="wide")
theme_plotly = None  # None or streamlit

# Custom CSS
try:
    with open('style.css') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# -------------------------
# Load Excel -> DataFrame
# -------------------------
df = pd.read_excel('data.xlsx', sheet_name='Sheet1')  # expects columns used below

# -------------------------
# Create SQLite (in-memory) and load df as 'dataset'
# -------------------------
conn = sqlite3.connect(":memory:")
df.to_sql('dataset', conn, index=False, if_exists='replace')

# Helper: get schema for display
def get_schema_markdown(connection: sqlite3.Connection, table: str = "dataset") -> str:
    cur = connection.cursor()
    cur.execute(f"PRAGMA table_info({table});")
    rows = cur.fetchall()
    if not rows:
        return "_No schema found_"
    cols = ["cid", "name", "type", "notnull", "dflt_value", "pk"]
    schema_df = pd.DataFrame(rows, columns=cols)
    schema_df = schema_df[["name", "type", "notnull", "pk"]]
    return schema_df.to_markdown(index=False)

# -------------------------
# Sidebar filters (original)
# -------------------------
st.sidebar.header("Please Filter Here:")
region = st.sidebar.multiselect(
    "Select the Region:",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)
location = st.sidebar.multiselect(
    "Select the Location:",
    options=df["Location"].unique(),
    default=df["Location"].unique(),
)
construction = st.sidebar.multiselect(
    "Select the Construction:",
    options=df["Construction"].unique(),
    default=df["Construction"].unique()
)
df_selection = df.query(
    "Region == @region & Location == @location & Construction == @construction"
)

# =========================
# Original Dash: Home
# =========================
def HomePage():
    with st.expander("🧭 My database"):
        shwdata = st.multiselect('Filter :', df_selection.columns, default=[])
        if shwdata:
            st.dataframe(df_selection[shwdata], use_container_width=True)
        else:
            st.dataframe(df_selection.head(100), use_container_width=True)

    total_investment = float(df_selection['Investment'].sum())
    investment_mode = float(df_selection['Investment'].mode())
    investment_mean = float(df_selection['Investment'].mean())
    investment_median = float(df_selection['Investment'].median())
    rating = float(df_selection['Rating'].sum())

    total1, total2, total3, total4, total5 = st.columns(5, gap='large')
    with total1:
        st.info('Total Investment', icon="🔍")
        st.metric(label='sum', value=f"{total_investment:,.0f}")
    with total2:
        st.info('Most frequently', icon="🔍")
        st.metric(label='Mode', value=f"{investment_mode:,.0f}")
    with total3:
        st.info('Investment Average', icon="🔍")
        st.metric(label='Mean', value=f"{investment_mean:,.0f}")
    with total4:
        st.info('Investment Margin', icon="🔍")
        st.metric(label='Median', value=f"{investment_median:,.0f}")
    with total5:
        st.info('Ratings', icon="🔍")
        st.metric(label='Rating', value=numerize(rating), help=f"Total rating: {rating}")
    st.markdown("""---""")

def Graphs():
    investment_by_businessType = (
        df_selection.groupby(by=["BusinessType"]).count()[["Investment"]].sort_values(by="Investment")
    )
    fig_investment = px.bar(
        investment_by_businessType,
        x="Investment",
        y=investment_by_businessType.index,
        orientation="h",
        title="Investment by Business Type",
        color_discrete_sequence=["#0083B8"] * len(investment_by_businessType),
        template="plotly_white",
    )
    fig_investment.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=(dict(showgrid=False)))

    investment_by_state = df_selection.groupby(by=["State"]).count()[["Investment"]]
    fig_state = px.line(
        investment_by_state,
        x=investment_by_state.index,
        orientation="v",
        y="Investment",
        title="Investment by Region",
        color_discrete_sequence=["#0083B8"] * len(investment_by_state),
        template="plotly_white",
    )
    fig_state.update_layout(xaxis=dict(tickmode="linear"),
                            plot_bgcolor="rgba(0,0,0,0)",
                            yaxis=(dict(showgrid=False)))
    left_column, right_column, center = st.columns(3)
    left_column.plotly_chart(fig_state, use_container_width=True)
    right_column.plotly_chart(fig_investment, use_container_width=True)

    with center:
        fig = px.pie(df_selection, values='Rating', names='State', title='Regions by Ratings')
        fig.update_layout(legend_title="Regions", legend_y=0.9)
        fig.update_traces(textinfo='percent+label', textposition='inside')
        st.plotly_chart(fig, use_container_width=True, theme=theme_plotly)

def ProgressBar():
    st.markdown(
        """<style>.stProgress > div > div > div > div { background-image: linear-gradient(to right, #99ff99 , #FFFF00)}</style>""",
        unsafe_allow_html=True,
    )
    target = 3_000_000_000
    current = df_selection['Investment'].sum()
    percent = round((current / target * 100))
    my_bar = st.progress(0)
    if percent > 100:
        st.subheader("Target 100 completed")
    else:
        st.write("you have ", percent, " % ", " of ", (format(target, ',d')), " TZS")
        for pc in range(percent):
            time.sleep(0.02)
            my_bar.progress(pc + 1, text="Target percentage")

# =========================
# NEW: SQL Lab
# =========================
Challenge = Dict[str, str]

def _challenges() -> List[Challenge]:
    # Each challenge has: id, title, prompt (question), solution (SQL)
    # All use table: dataset(Region, Location, Construction, Investment, Rating, BusinessType, State)
    return [
        {
            "id": "Q01",
            "title": "Totals by Region",
            "prompt": "Return total Investment and average Rating per Region, ordered by total Investment desc.",
            "solution": """
                SELECT
                  Region,
                  SUM(Investment) AS total_investment,
                  AVG(Rating)     AS avg_rating
                FROM dataset
                GROUP BY Region
                ORDER BY total_investment DESC;
            """,
        },
        {
            "id": "Q02",
            "title": "Top-3 BusinessTypes within each State (by total Investment)",
            "prompt": "For each State, list the top 3 BusinessType by total Investment.",
            "solution": """
                WITH sums AS (
                  SELECT State, BusinessType, SUM(Investment) AS tot_inv
                  FROM dataset
                  GROUP BY State, BusinessType
                ),
                ranked AS (
                  SELECT *,
                         ROW_NUMBER() OVER (PARTITION BY State ORDER BY tot_inv DESC) AS rn
                  FROM sums
                )
                SELECT State, BusinessType, tot_inv
                FROM ranked
                WHERE rn <= 3
                ORDER BY State, tot_inv DESC;
            """,
        },
        {
            "id": "Q03",
            "title": "Cumulative Pareto by Region",
            "prompt": "Regions that together contribute at least 25% of total Investment (cumulative over descending totals).",
            "solution": """
                WITH by_reg AS (
                  SELECT Region, SUM(Investment) AS tot
                  FROM dataset
                  GROUP BY Region
                ),
                totals AS (
                  SELECT SUM(tot) AS grand FROM by_reg
                ),
                ranked AS (
                  SELECT
                    b.Region,
                    b.tot,
                    SUM(b.tot) OVER (ORDER BY b.tot DESC) AS cum_tot,
                    (SELECT grand FROM totals) AS grand
                  FROM by_reg b
                )
                SELECT Region, tot, cum_tot, ROUND(100.0*cum_tot/grand,2) AS cum_pct
                FROM ranked
                WHERE cum_tot <= 0.25 * grand
                ORDER BY tot DESC;
            """,
        },
        {
            "id": "Q04",
            "title": "States above overall average Investment",
            "prompt": "List States whose average Investment is above the overall average Investment.",
            "solution": """
                WITH overall AS (SELECT AVG(Investment) AS overall_avg FROM dataset)
                SELECT State, AVG(Investment) AS avg_inv
                FROM dataset
                GROUP BY State
                HAVING avg_inv > (SELECT overall_avg FROM overall)
                ORDER BY avg_inv DESC;
            """,
        },
        {
            "id": "Q05",
            "title": "Pivot-style shares by BusinessType within a Location",
            "prompt": "For each Location, show total Investment per BusinessType and each BusinessType's % share of Location total.",
            "solution": """
                WITH base AS (
                  SELECT Location, BusinessType, SUM(Investment) AS tot
                  FROM dataset
                  GROUP BY Location, BusinessType
                ),
                loc AS (
                  SELECT Location, SUM(tot) AS loc_tot FROM base GROUP BY Location
                )
                SELECT
                  b.Location,
                  b.BusinessType,
                  b.tot AS business_investment,
                  ROUND(100.0*b.tot/l.loc_tot,2) AS pct_of_location
                FROM base b
                JOIN loc l USING(Location)
                ORDER BY b.Location, business_investment DESC;
            """,
        },
        {
            "id": "Q06",
            "title": "Regions covering all Construction types",
            "prompt": "Return Regions that contain EVERY Construction category present in the dataset.",
            "solution": """
                WITH all_c AS (SELECT COUNT(DISTINCT Construction) AS cnt FROM dataset)
                SELECT Region
                FROM dataset
                GROUP BY Region
                HAVING COUNT(DISTINCT Construction) = (SELECT cnt FROM all_c)
                ORDER BY Region;
            """,
        },
        {
            "id": "Q07",
            "title": "Outliers: Investment > mean + 2*std",
            "prompt": "Identify rows where Investment exceeds mean + 2*stddev of Investment (overall).",
            "solution": """
                WITH stats AS (
                  SELECT AVG(Investment) AS mu,
                         AVG(Investment*Investment) AS ex2
                  FROM dataset
                ),
                z AS (
                  SELECT
                    d.*,
                    s.mu,
                    s.ex2,
                    -- variance = E[X^2] - (E[X])^2
                    (s.ex2 - s.mu*s.mu) AS var,
                    CASE
                      WHEN (s.ex2 - s.mu*s.mu) <= 0 THEN 0
                      ELSE SQRT(s.ex2 - s.mu*s.mu)
                    END AS sd
                  FROM dataset d CROSS JOIN stats s
                )
                SELECT *
                FROM z
                WHERE Investment > mu + 2*sd
                ORDER BY Investment DESC;
            """,
        },
        {
            "id": "Q08",
            "title": "Rank Regions by avg Rating; tiebreak by total Investment",
            "prompt": "Return Regions ranked by avg Rating desc; break ties by total Investment desc.",
            "solution": """
                WITH agg AS (
                  SELECT Region,
                         AVG(Rating) AS avg_rating,
                         SUM(Investment) AS tot_inv
                  FROM dataset
                  GROUP BY Region
                )
                SELECT
                  Region, avg_rating, tot_inv,
                  RANK() OVER (ORDER BY avg_rating DESC, tot_inv DESC) AS rnk
                FROM agg
                ORDER BY rnk, Region;
            """,
        },
        {
            "id": "Q09",
            "title": "Best-rated BusinessType per State",
            "prompt": "For each State, return the BusinessType with highest average Rating (ties allowed).",
            "solution": """
                WITH rates AS (
                  SELECT State, BusinessType, AVG(Rating) AS avg_r
                  FROM dataset
                  GROUP BY State, BusinessType
                ),
                ranked AS (
                  SELECT *,
                         DENSE_RANK() OVER (PARTITION BY State ORDER BY avg_r DESC) AS dr
                  FROM rates
                )
                SELECT State, BusinessType, avg_r
                FROM ranked
                WHERE dr = 1
                ORDER BY State, BusinessType;
            """,
        },
        {
            "id": "Q10",
            "title": "Row-wise % difference vs State average",
            "prompt": "For each row, compute % difference of Investment vs the average Investment of its State.",
            "solution": """
                SELECT
                  *,
                  AVG(Investment) OVER (PARTITION BY State) AS state_avg,
                  ROUND(100.0*(Investment - AVG(Investment) OVER (PARTITION BY State))
                        / NULLIF(AVG(Investment) OVER (PARTITION BY State),0), 2) AS pct_diff_vs_state
                FROM dataset
                ORDER BY State, Investment DESC;
            """,
        },
        {
            "id": "Q11",
            "title": "Top 5 Locations by total Investment",
            "prompt": "Return the top 5 Locations by total Investment.",
            "solution": """
                SELECT Location, SUM(Investment) AS tot_inv
                FROM dataset
                GROUP BY Location
                ORDER BY tot_inv DESC
                LIMIT 5;
            """,
        },
        {
            "id": "Q12",
            "title": "Investment-weighted Rating by Region",
            "prompt": "Compute investment-weighted average Rating per Region.",
            "solution": """
                WITH w AS (
                  SELECT Region,
                         SUM(Rating * Investment) AS wsum,
                         SUM(Investment) AS invsum
                  FROM dataset
                  GROUP BY Region
                )
                SELECT Region,
                       CASE WHEN invsum = 0 THEN NULL ELSE ROUND(wsum*1.0/invsum, 4) END AS weighted_rating
                FROM w
                ORDER BY weighted_rating DESC;
            """,
        },
        {
            "id": "Q13",
            "title": "Co-occurring BusinessType pairs within a Location",
            "prompt": "Find BusinessType pairs that co-occur in the same Location (count of distinct co-occurring rows), exclude self-pairs.",
            "solution": """
                WITH base AS (
                  SELECT DISTINCT Location, BusinessType
                  FROM dataset
                )
                SELECT
                  a.Location,
                  MIN(a.BusinessType, b.BusinessType) AS bt1,
                  MAX(a.BusinessType, b.BusinessType) AS bt2,
                  COUNT(*) AS pair_count
                FROM base a
                JOIN base b
                  ON a.Location = b.Location AND a.BusinessType < b.BusinessType
                GROUP BY a.Location, bt1, bt2
                HAVING pair_count > 0
                ORDER BY pair_count DESC, a.Location;
            """,
        },
        {
            "id": "Q14",
            "title": "Dominant BusinessType by Location (>60% share)",
            "prompt": "Return Locations where a single BusinessType accounts for > 60% of Investment.",
            "solution": """
                WITH sums AS (
                  SELECT Location, BusinessType, SUM(Investment) AS bt_inv
                  FROM dataset
                  GROUP BY Location, BusinessType
                ),
                loc AS (
                  SELECT Location, SUM(bt_inv) AS loc_inv FROM sums GROUP BY Location
                ),
                shares AS (
                  SELECT s.Location, s.BusinessType, s.bt_inv, l.loc_inv,
                         1.0*s.bt_inv/l.loc_inv AS share
                  FROM sums s JOIN loc l USING(Location)
                )
                SELECT Location, BusinessType, ROUND(100.0*share,2) AS pct_share
                FROM shares
                WHERE share > 0.60
                ORDER BY pct_share DESC;
            """,
        },
        {
            "id": "Q15",
            "title": "Z-score of Investment within each Region",
            "prompt": "Compute z = (Investment - mean) / std within each Region (rows with std=0 => z NULL).",
            "solution": """
                SELECT
                  *,
                  AVG(Investment) OVER (PARTITION BY Region) AS mu,
                  AVG(Investment*Investment) OVER (PARTITION BY Region) AS ex2,
                  CASE
                    WHEN (AVG(Investment*Investment) OVER (PARTITION BY Region)
                          - (AVG(Investment) OVER (PARTITION BY Region))
                            * (AVG(Investment) OVER (PARTITION BY Region))) <= 0
                      THEN NULL
                    ELSE
                      (Investment - AVG(Investment) OVER (PARTITION BY Region))
                      / SQRT(AVG(Investment*Investment) OVER (PARTITION BY Region)
                             - (AVG(Investment) OVER (PARTITION BY Region))
                               * (AVG(Investment) OVER (PARTITION BY Region)))
                  END AS z_score
                FROM dataset
                ORDER BY Region, z_score DESC;
            """,
        },
        {
            "id": "Q16",
            "title": "HHI of BusinessType concentration by Region",
            "prompt": "Compute Herfindahl–Hirschman Index (HHI) of BusinessType Investment shares for each Region.",
            "solution": """
                WITH sums AS (
                  SELECT Region, BusinessType, SUM(Investment) AS bt_inv
                  FROM dataset
                  GROUP BY Region, BusinessType
                ),
                rtot AS (SELECT Region, SUM(bt_inv) AS r_inv FROM sums GROUP BY Region)
                SELECT
                  s.Region,
                  ROUND(10000 * SUM((1.0*s.bt_inv/r.r_inv)*(1.0*s.bt_inv/r.r_inv)), 2) AS HHI
                FROM sums s
                JOIN rtot r USING(Region)
                GROUP BY s.Region
                ORDER BY HHI DESC;
            """,
        },
        {
            "id": "Q17",
            "title": "Cumulative Investment by BusinessType within Region",
            "prompt": "For each Region, order BusinessTypes by total Investment and compute cumulative sum and cumulative %.",
            "solution": """
                WITH sums AS (
                  SELECT Region, BusinessType, SUM(Investment) AS bt_inv
                  FROM dataset
                  GROUP BY Region, BusinessType
                ),
                with_tot AS (
                  SELECT s.*,
                         SUM(bt_inv) OVER (PARTITION BY Region) AS region_tot,
                         SUM(bt_inv) OVER (PARTITION BY Region ORDER BY bt_inv DESC) AS cum_inv
                  FROM sums s
                )
                SELECT
                  Region, BusinessType, bt_inv,
                  cum_inv,
                  ROUND(100.0*cum_inv/region_tot,2) AS cum_pct
                FROM with_tot
                ORDER BY Region, bt_inv DESC;
            """,
        },
        {
            "id": "Q18",
            "title": "Counterintuitive: Highest-rated type has below-average Investment",
            "prompt": "Find States where the highest avg-rated BusinessType has total Investment below the State's avg Investment.",
            "solution": """
                WITH by_type AS (
                  SELECT State, BusinessType,
                         AVG(Rating) AS avg_r,
                         SUM(Investment) AS tot_inv_bt
                  FROM dataset
                  GROUP BY State, BusinessType
                ),
                top AS (
                  SELECT *,
                         DENSE_RANK() OVER (PARTITION BY State ORDER BY avg_r DESC) AS rnk
                  FROM by_type
                ),
                s_avg AS (
                  SELECT State, AVG(Investment) AS state_avg_inv
                  FROM dataset
                  GROUP BY State
                )
                SELECT t.State, t.BusinessType, t.avg_r, t.tot_inv_bt, s.state_avg_inv
                FROM top t
                JOIN s_avg s USING(State)
                WHERE rnk = 1 AND t.tot_inv_bt < s.state_avg_inv
                ORDER BY t.State;
            """,
        },
        {
            "id": "Q19",
            "title": "Dense rank of (Region, State, Construction) by rowcount",
            "prompt": "Count rows per (Region, State, Construction) and dense-rank counts within each Region.",
            "solution": """
                WITH counts AS (
                  SELECT Region, State, Construction, COUNT(*) AS n
                  FROM dataset
                  GROUP BY Region, State, Construction
                )
                SELECT
                  Region, State, Construction, n,
                  DENSE_RANK() OVER (PARTITION BY Region ORDER BY n DESC) AS dr
                FROM counts
                ORDER BY Region, dr, n DESC;
            """,
        },
        {
            "id": "Q20",
            "title": "Top 10% Investment as a View",
            "prompt": "Create a VIEW of rows in the top 10% by Investment (overall), then select from it.",
            "solution": """
                DROP VIEW IF EXISTS high_value_projects;
                WITH ranked AS (
                  SELECT *,
                         NTILE(10) OVER (ORDER BY Investment DESC) AS decile
                  FROM dataset
                )
                CREATE VIEW high_value_projects AS
                SELECT * FROM ranked WHERE decile=1;
                SELECT * FROM high_value_projects ORDER BY Investment DESC;
            """,
        },
    ]

def _run_sql(sql: str) -> pd.DataFrame:
    sql = sql.strip().rstrip(";")
    # Allow multiple statements (e.g., DROP VIEW; CREATE VIEW; SELECT ...)
    # We'll split by ';' safely: execute statements except final SELECT capturing last result
    cur = conn.cursor()
    result_df = None
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for i, stmt in enumerate(statements):
        # If it's a SELECT, fetch into DataFrame
        if stmt.lower().startswith("select"):
            result_df = pd.read_sql_query(stmt, conn)
        else:
            cur.execute(stmt)
            conn.commit()
    if result_df is None:
        # If no SELECT provided, show effect
        return pd.DataFrame({"message": ["Query executed. Add a SELECT to see results."]})
    return result_df

def _normalize_df(df_: pd.DataFrame) -> pd.DataFrame:
    if df_ is None or df_.empty:
        return df_
    # Sort columns alphabetically and rows by all columns for deterministic comparison
    cols = sorted(df_.columns.tolist())
    out = df_[cols].copy()
    try:
        out = out.sort_values(by=cols).reset_index(drop=True)
    except Exception:
        out = out.reset_index(drop=True)
    return out

def SqlLab():
    st.markdown("## 🧪 SQL Lab — Challenges on `dataset`")
    with st.expander("Table schema: `dataset`"):
        st.markdown(get_schema_markdown(conn, "dataset"))

    ch = _challenges()
    titles = [f"{c['id']} — {c['title']}" for c in ch]
    picked = st.selectbox("Pick a challenge:", titles, index=0)
    idx = titles.index(picked)
    challenge = ch[idx]

    st.markdown(f"**Problem:** {challenge['prompt']}")
    st.caption("Tip: You can run multiple SQL statements separated by semicolons.")

    starter = st.toggle("Insert starter SQL (SELECT * FROM dataset LIMIT 5)")
    default_sql = "SELECT * FROM dataset LIMIT 5;" if starter else ""

    user_sql = st.text_area("Your SQL", value=default_sql, height=200, key=f"sql_{challenge['id']}")
    c1, c2, c3 = st.columns([1,1,1])
    run_clicked = c1.button("▶ Run Query", use_container_width=True)
    show_solution = c2.toggle("Show Solution", value=False)
    check_ans = c3.button("✅ Check Against Solution", use_container_width=True)

    if run_clicked and user_sql.strip():
        try:
            out = _run_sql(user_sql)
            st.dataframe(out, use_container_width=True)
            # Download
            buff = io.StringIO()
            out.to_csv(buff, index=False)
            st.download_button("Download result as CSV", buff.getvalue(), file_name="query_result.csv")
        except Exception as e:
            st.error(f"SQL error: {e}")

    if show_solution:
        st.code(challenge['solution'].strip(), language="sql")
        if st.button("▶ Run Solution"):
            try:
                out = _run_sql(challenge['solution'])
                st.dataframe(out, use_container_width=True)
            except Exception as e:
                st.error(f"Solution execution error: {e}")

    if check_ans and user_sql.strip():
        try:
            user_df = _run_sql(user_sql)
            sol_df = _run_sql(challenge['solution'])
            u_norm = _normalize_df(user_df)
            s_norm = _normalize_df(sol_df)
            if u_norm.equals(s_norm):
                st.success("🎉 Correct! Your result matches the reference solution (order-insensitive).")
            else:
                st.warning("Results differ from the reference solution. Check columns, values, or duplicates.")
                with st.expander("See normalized comparison"):
                    st.write("Your (normalized):")
                    st.dataframe(u_norm, use_container_width=True)
                    st.write("Solution (normalized):")
                    st.dataframe(s_norm, use_container_width=True)
        except Exception as e:
            st.error(f"Could not validate: {e}")

# =========================
# Sidebar Menu 
# =========================
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Home", "Progress", "SQL Lab"],
        icons=["house", "eye", "database"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Home":
    try:
        HomePage()
        Graphs()
    except Exception:
        st.warning("One or more options are mandatory!")
elif selected == "Progress":
    try:
        ProgressBar()
        Graphs()
    except Exception:
        st.warning("One or more options are mandatory!")
elif selected == "SQL Lab":
    SqlLab()

# -------------------------
# Footer
# -------------------------
footer = """<style>
a:hover, a:active { color: red; background-color: transparent; text-decoration: underline; }
.footer { position: fixed; left: 0; height:5%; bottom: 0; width: 100%; background-color: #243946; color: white; text-align: center; }
</style>
<div class="footer">
<p>Developed with ❤ by Swapnil</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
