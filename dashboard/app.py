"""
Streamlit Dashboard — AI-Powered No-Show & Customer Intelligence
Full multi-page dashboard with interactive Plotly charts and live predictor.
"""

import sys
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.predictor import NoShowPredictor
from src.retention import CustomerRetentionAnalyzer

# --- Page Config ---
st.set_page_config(
    page_title="Salon AI Intelligence",
    page_icon="💇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .metric-card h3 {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-card h1 {
        color: #e6f1ff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }

    .metric-card .delta {
        font-size: 0.8rem;
        margin-top: 4px;
    }

    .delta-positive { color: #64ffda; }
    .delta-negative { color: #ff6b6b; }

    .risk-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }

    .risk-LOW { background: #064e3b; color: #6ee7b7; }
    .risk-MEDIUM { background: #713f12; color: #fcd34d; }
    .risk-HIGH { background: #7c2d12; color: #fb923c; }
    .risk-CRITICAL { background: #7f1d1d; color: #fca5a5; }

    .strategy-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }

    .strategy-card h4 {
        color: #818cf8;
        margin-bottom: 12px;
    }

    .strategy-card p, .strategy-card strong {
        color: #cbd5e1;
    }

    .strategy-card strong {
        color: #e2e8f0;
    }

    .action-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #818cf8;
        border-radius: 8px;
        padding: 20px;
        margin-top: 16px;
        color: #cbd5e1;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# --- Data Loading ---
@st.cache_data(ttl=300)
def load_data():
    csv_path = PROJECT_ROOT / "data" / "bookings.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path, parse_dates=['booking_datetime'])
    df['outcome_binary'] = (df['outcome'] == 'No-Show').astype(int)
    df['booking_date'] = df['booking_datetime'].dt.date
    df['booking_month'] = df['booking_datetime'].dt.to_period('M').astype(str)
    df['noshow_rate_hist'] = df['past_noshow_count'] / df['past_visit_count'].clip(lower=1)
    return df


@st.cache_data(ttl=300)
def load_model_metadata():
    meta_path = PROJECT_ROOT / "models" / "model_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def load_model_comparison():
    path = PROJECT_ROOT / "models" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_feature_importance():
    path = PROJECT_ROOT / "models" / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


@st.cache_data(ttl=300)
def load_calibration_data():
    path = PROJECT_ROOT / "models" / "calibration_data.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource
def load_predictor():
    p = NoShowPredictor(str(PROJECT_ROOT / "models"))
    try:
        p.load_model()
    except Exception:
        return None
    return p


def assign_risk_tier(prob):
    if prob < 0.25:
        return 'LOW'
    elif prob < 0.50:
        return 'MEDIUM'
    elif prob < 0.70:
        return 'HIGH'
    return 'CRITICAL'


# --- Load data ---
df_raw = load_data()
if df_raw is None:
    st.error("⚠️ Data file not found. Run `python data/generate_data.py` first.")
    st.stop()

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/96/hair-salon.png", width=60)
st.sidebar.title("🧠 Salon AI")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Executive Overview", "🤖 AI Insights", "👥 Customer Behavior",
     "🔄 Retention Intelligence", "🎯 Live Predictor"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Global Filters")

# Date range filter
min_date = df_raw['booking_datetime'].min().date()
max_date = df_raw['booking_datetime'].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Branch filter
branches = st.sidebar.multiselect("Branches", df_raw['branch'].unique().tolist(), default=df_raw['branch'].unique().tolist())

# Service type filter
services = st.sidebar.multiselect("Service Types", df_raw['service_type'].unique().tolist(), default=df_raw['service_type'].unique().tolist())

# Apply filters
df = df_raw.copy()
if len(date_range) == 2:
    df = df[(df['booking_datetime'].dt.date >= date_range[0]) & (df['booking_datetime'].dt.date <= date_range[1])]
df = df[df['branch'].isin(branches)]
df = df[df['service_type'].isin(services)]


# ======================================================================
# PAGE 1: Executive Overview
# ======================================================================
if page == "📊 Executive Overview":
    st.title("📊 Executive Overview")
    st.markdown("Real-time KPIs and no-show trends across your salon network.")

    # KPI Cards
    total_bookings = len(df)
    noshow_rate = (df['outcome'] == 'No-Show').mean() * 100
    noshow_count = (df['outcome'] == 'No-Show').sum()
    # Realistic per-service pricing for a premium salon
    service_price_map = {
        'Haircut': 1500, 'Color': 4000, 'Keratin': 6000, 'Facial': 2500,
        'Manicure': 1000, 'Pedicure': 1200, 'Waxing': 800, 'Bridal': 20000,
    }
    noshow_df = df[df['outcome'] == 'No-Show']
    revenue_lost = noshow_df['service_type'].map(service_price_map).fillna(1500).sum()

    # Compute risk tiers for filtered data
    predictor_inst = load_predictor()
    high_risk_count = 0
    if predictor_inst:
        # Approximate high-risk using manual risk score
        df_risk = df.copy()
        df_risk['noshow_rate_h'] = df_risk['past_noshow_count'] / df_risk['past_visit_count'].clip(lower=1)
        df_risk['cancel_rate'] = df_risk['past_cancellation_count'] / df_risk['past_visit_count'].clip(lower=1)
        df_risk['risk_score'] = (
            df_risk['noshow_rate_h'] * 0.30 +
            (df_risk['booking_lead_time_hours'] < 3).astype(int) * 0.10 +
            (df_risk['booking_lead_time_hours'] > 168).astype(int) * 0.08 +
            (df_risk['past_visit_count'] == 0).astype(int) * 0.15 +
            (df_risk['hour_of_day'] >= 18).astype(int) * 0.05 +
            df_risk['payment_method'].isin(['Cash', 'Card on Arrival']).astype(int) * 0.12 +
            df_risk['cancel_rate'] * 0.20
        )
        high_risk_count = (df_risk['risk_score'] > 0.35).sum()

    # Calculate delta (vs previous period)
    if len(date_range) == 2:
        period_days = (date_range[1] - date_range[0]).days
        prev_start = date_range[0] - timedelta(days=period_days)
        prev_end = date_range[0] - timedelta(days=1)
        df_prev = df_raw[(df_raw['booking_datetime'].dt.date >= prev_start) &
                         (df_raw['booking_datetime'].dt.date <= prev_end)]
        df_prev = df_prev[df_prev['branch'].isin(branches) & df_prev['service_type'].isin(services)]
        prev_noshow_rate = (df_prev['outcome'] == 'No-Show').mean() * 100 if len(df_prev) > 0 else noshow_rate
        delta = noshow_rate - prev_noshow_rate
    else:
        delta = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Bookings</h3>
            <h1>{total_bookings:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        delta_class = "delta-negative" if delta > 0 else "delta-positive"
        delta_arrow = "▲" if delta > 0 else "▼"
        st.markdown(f"""
        <div class="metric-card">
            <h3>No-Show Rate</h3>
            <h1>{noshow_rate:.1f}%</h1>
            <div class="delta {delta_class}">{delta_arrow} {abs(delta):.1f}% vs prev period</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Est. Revenue Lost</h3>
            <h1>₹{revenue_lost:,.0f}</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>High-Risk Bookings</h3>
            <h1>{high_risk_count:,}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # No-show trend
    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.subheader("📈 No-Show Trend Over Time")
        agg_period = st.radio("Aggregation", ["Daily", "Weekly"], horizontal=True, key="trend_agg")

        trend_df = df.copy()
        if agg_period == "Weekly":
            trend_df['period'] = trend_df['booking_datetime'].dt.to_period('W').dt.start_time
        else:
            trend_df['period'] = trend_df['booking_datetime'].dt.date

        trend_data = trend_df.groupby('period').agg(
            total=('outcome', 'count'),
            noshows=('outcome_binary', 'sum'),
        ).reset_index()
        trend_data['noshow_rate'] = trend_data['noshows'] / trend_data['total'] * 100

        fig = px.line(
            trend_data, x='period', y='noshow_rate',
            labels={'period': 'Date', 'noshow_rate': 'No-Show Rate (%)'},
            template='plotly_dark',
        )
        fig.update_traces(line=dict(color='#818cf8', width=2), fill='tozeroy',
                          fillcolor='rgba(129,140,248,0.1)')
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("💰 Revenue Impact by Branch")
        service_price_map_branch = {
            'Haircut': 1500, 'Color': 4000, 'Keratin': 6000, 'Facial': 2500,
            'Manicure': 1000, 'Pedicure': 1200, 'Waxing': 800, 'Bridal': 20000,
        }
        noshow_branch_df = df[df['outcome'] == 'No-Show'].copy()
        noshow_branch_df['price'] = noshow_branch_df['service_type'].map(service_price_map_branch).fillna(1500)
        branch_impact = noshow_branch_df.groupby('branch')['price'].agg(noshows='count', revenue_lost='sum').reset_index()

        fig = px.bar(
            branch_impact.sort_values('revenue_lost', ascending=True),
            x='revenue_lost', y='branch', orientation='h',
            labels={'revenue_lost': 'Revenue Lost (₹)', 'branch': ''},
            template='plotly_dark',
            color='revenue_lost',
            color_continuous_scale='Reds',
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                          showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# PAGE 2: AI Insights
# ======================================================================
elif page == "🤖 AI Insights":
    st.title("🤖 AI Model Insights")

    model_meta = load_model_metadata()
    model_comp = load_model_comparison()
    feat_imp = load_feature_importance()
    cal_data = load_calibration_data()

    # Risk distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Risk Distribution")
        df_r = df.copy()
        df_r['noshow_rate_h'] = df_r['past_noshow_count'] / df_r['past_visit_count'].clip(lower=1)
        df_r['cancel_rate'] = df_r['past_cancellation_count'] / df_r['past_visit_count'].clip(lower=1)
        df_r['risk_score'] = (
            df_r['noshow_rate_h'] * 0.30 +
            (df_r['booking_lead_time_hours'] < 3).astype(int) * 0.10 +
            (df_r['booking_lead_time_hours'] > 168).astype(int) * 0.08 +
            (df_r['past_visit_count'] == 0).astype(int) * 0.15 +
            (df_r['hour_of_day'] >= 18).astype(int) * 0.05 +
            df_r['payment_method'].isin(['Cash', 'Card on Arrival']).astype(int) * 0.12 +
            df_r['cancel_rate'] * 0.20
        )
        df_r['risk_tier'] = df_r['risk_score'].apply(assign_risk_tier)

        tier_counts = df_r['risk_tier'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'], fill_value=0)
        colors = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#f97316', 'CRITICAL': '#ef4444'}

        fig = go.Figure(data=[go.Pie(
            labels=tier_counts.index,
            values=tier_counts.values,
            hole=0.55,
            marker=dict(colors=[colors[t] for t in tier_counts.index]),
            textinfo='label+percent',
            textfont=dict(size=14),
        )])
        fig.update_layout(template='plotly_dark', height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📊 SHAP Feature Importance (Top 15)")
        if feat_imp is not None:
            top15 = feat_imp.head(15).sort_values('shap_importance', ascending=True)
            fig = px.bar(
                top15, x='shap_importance', y='feature', orientation='h',
                labels={'shap_importance': 'Mean |SHAP Value|', 'feature': ''},
                template='plotly_dark',
                color='shap_importance', color_continuous_scale='Viridis',
            )
            fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance data not available. Run model training first.")

        # Risk heatmap
    st.subheader("🗺️ No-Show Rate Heatmap: Branch × Service Type")

    # ✅ Use ACTUAL no-show rate, not the manual risk score
    heatmap_data = df.groupby(['branch', 'service_type']).agg(
        noshow_rate=('outcome_binary', 'mean'),
        booking_count=('outcome_binary', 'count'),
    ).reset_index()

    # ✅ Only show cells with enough data (min 20 bookings)
    heatmap_data.loc[
        heatmap_data['booking_count'] < 20, 'noshow_rate'
    ] = np.nan

    heatmap_pivot = heatmap_data.pivot(
        index='branch', columns='service_type', values='noshow_rate'
    )

    # ✅ Reorder columns by avg no-show rate (lowest to highest)
    col_order = heatmap_pivot.mean().sort_values().index
    heatmap_pivot = heatmap_pivot[col_order]

    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x="Service Type", y="Branch", color="No-Show Rate"),
        color_continuous_scale='RdYlGn_r',
        template='plotly_dark',
        aspect='auto',
        text_auto='.1%',   # ✅ Show percentages in cells
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(
            title="No-Show Rate",
            tickformat='.0%',
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Model performance cards
    st.subheader("🏆 Model Performance")
    if model_meta:
        cols = st.columns(5)
        metric_items = [
            ("ROC-AUC", model_meta.get('roc_auc', 0)),
            ("F1 Score", model_meta.get('f1', 0)),
            ("Precision", model_meta.get('precision', 0)),
            ("Recall", model_meta.get('recall', 0)),
            ("Accuracy", model_meta.get('accuracy', 0)),
        ]
        for col, (name, val) in zip(cols, metric_items):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{name}</h3>
                    <h1>{val:.3f}</h1>
                </div>
                """, unsafe_allow_html=True)

    # Model comparison table
    if model_comp is not None:
        st.subheader("📋 Model Comparison")
        metric_cols = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
        styled_comp = (
            model_comp[['model_name'] + metric_cols]
            .style.format({c: '{:.4f}' for c in metric_cols})
            .highlight_max(
                subset=metric_cols,
                props='background-color: #0d9488; color: #ffffff; font-weight: bold',
            )
        )
        st.dataframe(styled_comp, use_container_width=True)

    # Calibration curve
    if cal_data:
        st.subheader("📐 Calibration Curves")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                  name='Perfect', line=dict(dash='dash', color='gray')))
        for model_name, data in cal_data.items():
            fig.add_trace(go.Scatter(
                x=data['mean_predicted'], y=data['fraction_positives'],
                mode='lines+markers', name=model_name,
            ))
        fig.update_layout(
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            template='plotly_dark', height=400,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# PAGE 3: Customer Behavior
# ======================================================================
elif page == "👥 Customer Behavior":
    st.title("👥 Customer Behavior Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔁 Repeat vs New Customers")
        repeat_counts = df['is_repeat_customer'].value_counts()
        labels = ['Repeat', 'New']
        fig = go.Figure(data=[go.Pie(
            labels=labels, values=[repeat_counts.get(True, 0), repeat_counts.get(False, 0)],
            hole=0.5,
            marker=dict(colors=['#818cf8', '#f472b6']),
            textinfo='label+percent+value',
        )])
        fig.update_layout(template='plotly_dark', height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⏱️ Booking Lead Time Distribution")
        fig = px.histogram(
            df, x='booking_lead_time_hours', nbins=60,
            labels={'booking_lead_time_hours': 'Lead Time (hours)'},
            template='plotly_dark', color_discrete_sequence=['#818cf8'],
        )
        # Vertical lines at key thresholds
        for hr, label, color in [(3, '3hr', '#ef4444'), (24, '24hr', '#f59e0b'), (168, '7 days', '#10b981')]:
            fig.add_vline(x=hr, line_dash='dash', line_color=color,
                          annotation_text=label, annotation_position='top')
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap: hour vs day of week
    st.subheader("🔥 No-Show Heatmap: Hour × Day of Week")
    heatmap_df = df.groupby(['hour_of_day', 'day_of_week']).agg(
        noshow_rate=('outcome_binary', 'mean'),
    ).reset_index()
    heatmap_pivot = heatmap_df.pivot(index='hour_of_day', columns='day_of_week', values='noshow_rate')
    heatmap_pivot.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig = px.imshow(
        heatmap_pivot, labels=dict(x="Day", y="Hour", color="No-Show Rate"),
        color_continuous_scale='YlOrRd', template='plotly_dark', aspect='auto',
    )
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("💳 No-Show Rate by Payment Method")
        pm_data = df.groupby('payment_method').agg(
            noshow_rate=('outcome_binary', 'mean'),
            count=('outcome_binary', 'count'),
        ).reset_index()
        pm_data['noshow_pct'] = pm_data['noshow_rate'] * 100

        fig = px.bar(
            pm_data.sort_values('noshow_pct', ascending=True),
            x='noshow_pct', y='payment_method', orientation='h',
            labels={'noshow_pct': 'No-Show Rate (%)', 'payment_method': ''},
            template='plotly_dark', color='noshow_pct',
            color_continuous_scale='RdYlGn_r',
            text='noshow_pct',
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🏷️ Customer Segment Distribution")
        df_seg = df.copy()
        df_seg['noshow_rate_h'] = df_seg['past_noshow_count'] / df_seg['past_visit_count'].clip(lower=1)

        def seg(row):
            v, nr = row['past_visit_count'], row['noshow_rate_h']
            if v >= 15 and nr < 0.1: return 'VIP'
            elif v >= 6: return 'Loyal'
            elif v >= 2: return 'At-Risk' if nr > 0.3 else 'Occasional'
            elif v >= 1: return 'At-Risk' if nr > 0.3 else 'Occasional'
            else: return 'New'

        df_seg['segment'] = df_seg.apply(seg, axis=1)
        seg_counts = df_seg['segment'].value_counts()

        fig = px.bar(
            x=seg_counts.index, y=seg_counts.values,
            labels={'x': 'Segment', 'y': 'Count'},
            template='plotly_dark',
            color=seg_counts.index,
            color_discrete_map={
                'VIP': '#818cf8', 'Loyal': '#34d399', 'Occasional': '#60a5fa',
                'At-Risk': '#fb923c', 'New': '#f472b6',
            },
        )
        fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# PAGE 4: Retention Intelligence
# ======================================================================
elif page == "🔄 Retention Intelligence":
    st.title("🔄 Retention Intelligence")

    retention = CustomerRetentionAnalyzer()
    cust_df = retention.build_customer_profiles(df)
    strategies = retention.generate_retention_strategies()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🫧 Customer Segments")
        seg_summary = retention.get_segment_summary()

        fig = px.scatter(
            seg_summary, x='avg_visits', y='avg_noshow_rate',
            size='count', color='segment',
            labels={'avg_visits': 'Avg Visits', 'avg_noshow_rate': 'Avg No-Show Rate',
                    'count': 'Customers'},
            template='plotly_dark', size_max=60,
            color_discrete_map={
                'VIP': '#818cf8', 'Loyal': '#34d399', 'Occasional': '#60a5fa',
                'At-Risk': '#fb923c', 'New': '#f472b6',
            },
            hover_data=['count', 'avg_ltv'],
        )
        fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("⚠️ Churn Risk Distribution")
        churn_counts = cust_df['churn_risk'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'], fill_value=0)
        colors_churn = {'LOW': '#10b981', 'MEDIUM': '#f59e0b', 'HIGH': '#ef4444'}

        fig = go.Figure(data=[go.Pie(
            labels=churn_counts.index, values=churn_counts.values,
            hole=0.5,
            marker=dict(colors=[colors_churn[c] for c in churn_counts.index]),
            textinfo='label+percent+value',
        )])
        fig.update_layout(template='plotly_dark', height=380, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Retention strategies
    st.subheader("🎯 Data-Backed Retention Strategies")
    for i, s in enumerate(strategies):
        st.markdown(f"""
        <div class="strategy-card">
            <h4>{'🚀' if i == 0 else '🌟' if i == 1 else '🛡️' if i == 2 else '💤'} {s['strategy_name']}</h4>
            <p><strong>Target:</strong> {s['target_segment']}</p>
            <p><strong>Rationale:</strong> {s['rationale']}</p>
            <p><strong>Action:</strong> {s['suggested_action']}</p>
            <p><strong>Projected Impact:</strong> {s.get('projected_impact', 'N/A')}</p>
        </div>
        """, unsafe_allow_html=True)

    # At-Risk customers table
    st.subheader("🚨 Top 20 At-Risk Customers")
    at_risk_df = retention.get_at_risk_customers(20)
    display_cols = ['customer_id', 'segment', 'visit_count', 'noshow_rate',
                    'days_since_last_visit', 'estimated_ltv', 'churn_risk', 'suggested_action']
    available_cols = [c for c in display_cols if c in at_risk_df.columns]
    st.dataframe(
        at_risk_df[available_cols].style.format({
            'noshow_rate': '{:.2%}', 'estimated_ltv': '₹{:,.0f}',
        }),
        use_container_width=True,
        height=500,
    )


# ======================================================================
# PAGE 5: Live Predictor
# ======================================================================
elif page == "🎯 Live Predictor":
    st.title("🎯 Live No-Show Predictor")
    st.markdown("Enter booking details below to get an AI-powered risk assessment.")

    predictor = load_predictor()
    if predictor is None:
        st.error("⚠️ Model not loaded. Run `python src/model_trainer.py` first.")
        st.stop()

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            service_type = st.selectbox("Service Type",
                ['Haircut', 'Color', 'Keratin', 'Facial', 'Manicure', 'Pedicure', 'Waxing', 'Bridal'])
            branch = st.selectbox("Branch", ['Science City', 'Memnagar', 'Sindhu Bhavan Road', 'Sabarmati', 'Chandkheda'])
            payment_method = st.selectbox("Payment Method",
                ['Online Prepaid', 'Card on Arrival', 'Cash', 'UPI'])
            staff_id = st.selectbox("Staff", [f"S{str(i).zfill(2)}" for i in range(1, 21)])

        with col2:
            booking_lead_time = st.slider("Lead Time (hours)", 0, 720, 24)
            day_of_week = st.selectbox("Day of Week",
                [("Monday", 0), ("Tuesday", 1), ("Wednesday", 2), ("Thursday", 3),
                 ("Friday", 4), ("Saturday", 5), ("Sunday", 6)],
                format_func=lambda x: x[0])[1]
            hour_of_day = st.slider("Appointment Hour", 0, 23, 14)

            svc_duration_map = {
                'Haircut': 45, 'Color': 120, 'Keratin': 150, 'Facial': 60,
                'Manicure': 40, 'Pedicure': 50, 'Waxing': 30, 'Bridal': 240,
            }
            duration = st.number_input("Duration (mins)", 10, 400,
                                       value=svc_duration_map.get(service_type, 60))

        with col3:
            past_visits = st.number_input("Past Visit Count", 0, 100, 3)
            past_cancellations = st.number_input("Past Cancellations", 0, 50, 0)
            past_noshows = st.number_input("Past No-Shows", 0, 20, 0)

        submitted = st.form_submit_button("🔮 Predict No-Show Risk", use_container_width=True)

    if submitted:
        booking = {
            'service_type': service_type,
            'branch': branch,
            'booking_lead_time_hours': booking_lead_time,
            'day_of_week': day_of_week,
            'hour_of_day': hour_of_day,
            'payment_method': payment_method,
            'past_visit_count': past_visits,
            'past_cancellation_count': past_cancellations,
            'past_noshow_count': past_noshows,
            'service_duration_mins': duration,
            'staff_id': staff_id,
        }

        result = predictor.predict(booking)
        prob = result['noshow_probability']
        tier = result['risk_tier']

        st.markdown("---")

        col_a, col_b = st.columns([1, 1])

        with col_a:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': '%', 'font': {'size': 48, 'color': 'white'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#818cf8'},
                    'bgcolor': '#1a1a2e',
                    'steps': [
                        {'range': [0, 25], 'color': '#064e3b'},
                        {'range': [25, 50], 'color': '#713f12'},
                        {'range': [50, 70], 'color': '#7c2d12'},
                        {'range': [70, 100], 'color': '#7f1d1d'},
                    ],
                },
                title={'text': 'No-Show Probability', 'font': {'color': 'white', 'size': 18}},
            ))
            fig.update_layout(
                template='plotly_dark', height=320,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Risk tier badge
            st.markdown(f"""
            <div style="text-align: center; margin-top: 20px;">
                <span class="risk-badge risk-{tier}">{tier} RISK</span>
            </div>
            """, unsafe_allow_html=True)

            # Risk factors
            st.markdown("#### 🔍 Top Risk Factors")
            for i, factor in enumerate(result['risk_factors'], 1):
                st.markdown(f"**{i}.** {factor}")

            # Recommended action
            st.markdown(f"""
            <div class="action-box">
                <strong>📋 Recommended Action:</strong><br>
                {result['recommended_action']}
            </div>
            """, unsafe_allow_html=True)


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.caption("🧠 Salon AI Intelligence v1.0 | Powered by XGBoost + SHAP")
