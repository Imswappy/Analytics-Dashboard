# 📊 Analytics Dashboard with SQL Lab

https://analyticsdashboard45.streamlit.app/

This project is an **interactive analytics dashboard** built with [Streamlit](https://streamlit.io/) and enhanced with **SQL learning functionality**.  
It allows you to explore and visualize data from an Excel dataset, apply filters, view summary metrics, interactive graphs, progress tracking, and even practice **20 advanced SQL challenges** directly within the app.

---

## 🚀 Features

### 🔹 Data Filtering
- Sidebar filters for **Region**, **Location**, and **Construction**
- Interactive dataframe preview with selected columns or full dataset
- Dynamic query updates in real time

### 🔹 Key Metrics
- **Total Investment**
- **Most Frequent Investment (Mode)**
- **Average Investment (Mean)**
- **Median Investment**
- **Total Ratings (summed and numerized)**

Metrics are displayed in **info cards with icons** for quick insights.

### 🔹 Graphical Visualizations
- **Bar Chart**: Investment by Business Type  
- **Line Chart**: Investment by State  
- **Pie Chart**: Ratings distribution by State  

All visualizations are powered by **Plotly Express** for interactivity.

### 🔹 Progress Tracker
- Custom **animated progress bar** showing percentage of target investment achieved  
- Updates dynamically with dataset values  
- Custom CSS gradient for better visualization  

### 🔹 SQL Lab (Learning Mode)
A dedicated **SQL Lab tab** with **20 advanced SQL challenges**:
- Uses dataset loaded into an **in-memory SQLite database**
- Challenges include **CTEs, window functions, cumulative sums, z-scores, Pareto analysis, HHI index, dense ranks, outlier detection, and more**
- Each challenge includes:
  - Problem description
  - Text editor for writing SQL queries (supports multiple statements)
  - Run queries and view results instantly
  - **Download results as CSV**
  - Toggle to view official **solution SQL code**
  - **Auto-validation**: Compares your result with solution (order-insensitive)

### 🔹 Custom Styling
- A `style.css` file adds a polished look & feel  
- Responsive, wide layout with Streamlit custom config  
- Sticky footer with developer credits  

---

## 📂 Project Structure

```
.
├── main.py          # Main Streamlit application with dashboard + SQL Lab
├── UI.py            # UI helper functions (if any future extension)
├── style.css        # Custom CSS styling
├── data.xlsx        # Dataset file used for dashboard & SQL Lab
└── requirements.txt # Dependencies for deployment
```

---

## ⚙️ Installation & Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/analytics-dashboard.git
   cd analytics-dashboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

4. The app will open in your browser at:
   ```
   http://localhost:8501
   ```

---

## 📑 Requirements

- Python 3.8+  
- Streamlit  
- Pandas  
- Plotly  
- OpenPyXL  
- Streamlit Option Menu  
- Numerize  

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🎯 SQL Challenges Included

Here are examples of the advanced SQL challenges included:

1. Totals by Region (SUM, AVG, ORDER BY)  
2. Top-3 BusinessTypes per State (CTE + ROW_NUMBER)  
3. Pareto Regions contributing 25% of Investment (Cumulative SUM)  
4. States above overall average Investment (HAVING)  
5. Pivot-style BusinessType shares per Location  
6. Regions covering all Construction types  
7. Outlier detection using mean + 2*stddev  
8. Ranking Regions by avg Rating & Investment  
9. Best-rated BusinessType per State  
10. Row-wise % difference vs State average  
11. Top 5 Locations by Investment  
12. Investment-weighted Rating per Region  
13. Co-occurring BusinessType pairs in a Location  
14. Dominant BusinessType (>60% share in a Location)  
15. Z-score of Investment within each Region  
16. HHI index (BusinessType concentration per Region)  
17. Cumulative Investment by BusinessType within Region  
18. States where best-rated BusinessType has below-average Investment  
19. Dense rank (Region, State, Construction) by rowcount  
20. Top 10% Investment projects (View + NTILE)  

---

## 🌐 Deployment

For deployment (e.g., Streamlit Cloud), include a `requirements.txt` file with:

```
streamlit
pandas
plotly
openpyxl
streamlit-option-menu
numerize
```

Push to GitHub and connect repo to Streamlit Cloud.

---

## 💡 Future Enhancements

- Add **score & timer system** for SQL Lab challenges  
- Support **multiple datasets / sheets** as SQL tables  
- Dark mode toggle for UI  
- Export SQL attempts history  


