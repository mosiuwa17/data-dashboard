import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="Interactive Data Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== TITLE ==========
st.title("📊 Interactive Data Dashboard")
st.markdown("Explore and visualize data with interactive charts and filters")

# ========== SIDEBAR ==========
st.sidebar.header("📁 Data Source")

# Data source selection
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Sample Data (Sales)", "Sample Data (Customer)", "Upload CSV"]
)

# ========== LOAD DATA FUNCTION ==========
@st.cache_data
def load_sample_data(dataset_type):
    """Load sample datasets"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    
    if dataset_type == "Sales":
        data = pd.DataFrame({
            'date': dates,
            'sales': np.random.normal(10000, 2000, len(dates)),
            'units_sold': np.random.randint(50, 500, len(dates)),
            'region': np.random.choice(['North', 'South', 'East', 'West'], len(dates)),
            'product': np.random.choice(['Product A', 'Product B', 'Product C'], len(dates))
        })
    else:  # Customer
        data = pd.DataFrame({
            'customer_id': range(1, 1001),
            'age': np.random.randint(18, 80, 1000),
            'spending_score': np.random.randint(1, 100, 1000),
            'annual_income': np.random.randint(20000, 150000, 1000),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 1000),
            'gender': np.random.choice(['Male', 'Female'], 1000)
        })
    
    return data

@st.cache_data
def load_uploaded_data(uploaded_file):
    """Load user-uploaded CSV file"""
    return pd.read_csv(uploaded_file)

# ========== LOAD DATA ==========
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        df = load_uploaded_data(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} rows")
    else:
        st.info("Please upload a CSV file to begin")
        st.stop()
else:
    dataset_type = "Sales" if data_source == "Sample Data (Sales)" else "Customer"
    df = load_sample_data(dataset_type)

# ========== DATA PREVIEW ==========
st.subheader("📋 Data Preview")
col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"**Rows:** {len(df)} | **Columns:** {len(df.columns)}")
with col2:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

st.dataframe(df.head(100), use_container_width=True)

# ========== SIDEBAR FILTERS ==========
st.sidebar.header("🔍 Filters")

# Filter by numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols[:2]:  # Show first 2 numeric columns as filters
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    filter_val = st.sidebar.slider(
        f"Filter {col}",
        min_val, max_val, (min_val, max_val)
    )
    df = df[(df[col] >= filter_val[0]) & (df[col] <= filter_val[1])]

# Filter by categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols[:2]:  # Show first 2 categorical columns
    unique_vals = df[col].dropna().unique().tolist()
    selected_vals = st.sidebar.multiselect(
        f"Filter {col}",
        unique_vals,
        default=unique_vals
    )
    if selected_vals:
        df = df[df[col].isin(selected_vals)]

# ========== KEY METRICS ==========
st.subheader("📈 Key Metrics")
if numeric_cols:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numeric Columns", len(numeric_cols))
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())

# ========== VISUALIZATIONS ==========
st.subheader("📊 Visualizations")

# Select chart type
chart_type = st.selectbox(
    "Select chart type:",
    ["Line Chart", "Bar Chart", "Scatter Plot", "Histogram", "Box Plot"]
)

# Select columns for visualization
if len(numeric_cols) >= 1:
    x_axis = st.selectbox("Select X-axis:", df.columns, index=0)
    y_axis = st.selectbox("Select Y-axis:", numeric_cols, index=0)
    
    if chart_type == "Line Chart":
        if 'date' in df.columns or pd.api.types.is_datetime64_any_dtype(df[x_axis]):
            fig = px.line(df, x=x_axis, y=y_axis, title=f"{y_axis} over {x_axis}")
        else:
            fig = px.line(df.sort_values(x_axis), x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}")
    
    elif chart_type == "Bar Chart":
        fig = px.bar(df, x=x_axis, y=y_axis, title=f"{y_axis} by {x_axis}", color=x_axis)
    
    elif chart_type == "Scatter Plot":
        if len(numeric_cols) >= 2:
            color_col = categorical_cols[0] if categorical_cols else None
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, title=f"{y_axis} vs {x_axis}")
        else:
            fig = px.scatter(df, x=x_axis, y=y_axis, title=f"{y_axis} vs {x_axis}")
    
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x_axis, nbins=30, title=f"Distribution of {x_axis}")
    
    elif chart_type == "Box Plot":
        if categorical_cols:
            fig = px.box(df, x=categorical_cols[0], y=y_axis, title=f"{y_axis} by {categorical_cols[0]}")
        else:
            fig = px.box(df, y=y_axis, title=f"Box Plot of {y_axis}")
    
    st.plotly_chart(fig, use_container_width=True)

# ========== CORRELATION MATRIX ==========
if len(numeric_cols) >= 2:
    st.subheader("🔗 Correlation Matrix")
    correlation = df[numeric_cols].corr()
    fig = px.imshow(
        correlation,
        text_auto=True,
        aspect="auto",
        title="Feature Correlation Matrix",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig, use_container_width=True)

# ========== STATISTICAL SUMMARY ==========
st.subheader("📊 Statistical Summary")
st.dataframe(df.describe(), use_container_width=True)

# ========== DOWNLOAD FILTERED DATA ==========
st.subheader("💾 Download Filtered Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)

# ========== FOOTER ==========
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with Streamlit | Data Dashboard for Exploration & Analysis"
    "</div>",
    unsafe_allow_html=True
)