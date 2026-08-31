"""
Main Streamlit Application.
AQI 3-Day Forecaster Dashboard.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import pandas as pd

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import data_loader, predict

st.set_page_config(
    page_title="Lahore AQI Forecast",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for aesthetic improvements ---
st.markdown("""
<style>
    .big-font {
        font-size: 80px !important;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .medium-font {
        font-size: 30px !important;
        font-weight: 600;
        margin-top: 0px;
    }
    .card {
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .footer {
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #e0e0e0;
        color: #666;
        font-size: 14px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("🌫️ Lahore AQI Forecast")
    st.markdown("Predicting air quality up to 3 days ahead using machine learning.")
    
    # Sidebar
    st.sidebar.header("System Health")
    
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
        
    try:
        with st.spinner("Connecting to Model Registry and Feature Store..."):
            recent_features = data_loader.load_recent_features(hours_back=120)
            model, scaler, imputer, model_version, model_updated_at = data_loader.load_latest_model_and_scaler()
            
        st.sidebar.success("✅ Online & Connected")
        
        if not recent_features.empty:
            latest_row = recent_features.iloc[-1]
            latest_time = latest_row["fetched_at"]
            hours_ago = int((datetime.now(latest_time.tzinfo) - latest_time).total_seconds() / 3600)
            
            st.sidebar.markdown(f"**Last feature update:** {hours_ago} hours ago")
            st.sidebar.markdown(f"**Model version:** {model_version}")
            if model_updated_at != "Unknown":
                st.sidebar.markdown(f"**Model retrained:** {str(model_updated_at)[:10]}")
            else:
                st.sidebar.markdown("**Model retrained:** Offline Fallback")
        else:
            st.sidebar.warning("No recent features found.")
            st.warning("Feature Store returned empty data. Wait for the pipeline to run.")
            return

    except Exception as e:
        st.sidebar.error("❌ Offline / Error")
        st.error(f"Unable to load latest data — please try again in a few minutes. (Details: {e})")
        return
        
    # Get Current AQI
    current_aqi = latest_row.get("aqi")
    
    # Generate Forecast
    try:
        with st.spinner("Generating 3-Day Forecast..."):
            forecast_df = predict.generate_3day_forecast(model, scaler, imputer, recent_features)
    except Exception as e:
        st.error(f"Failed to generate forecast: {e}")
        return
        
    # --- TABS ---
    tab1, tab2 = st.tabs(["🔮 72-Hour Forecast", "📊 Historical Trends"])
    
    with tab1:
        # Forecast Alert Banner
        if not forecast_df.empty:
            max_aqi = forecast_df["predicted_aqi"].max()
            if max_aqi >= 300:
                st.error("⚠️ **HAZARDOUS AIR QUALITY EXPECTED** — Avoid all outdoor physical activities.")
            elif max_aqi >= 200:
                st.error("⚠️ **VERY UNHEALTHY AIR QUALITY EXPECTED** — Avoid prolonged outdoor exertion.")
            elif max_aqi >= 150:
                st.warning("⚠️ **UNHEALTHY AIR QUALITY EXPECTED** — Consider limiting outdoor activity, especially for sensitive groups.")
                
        # Current AQI Card
        if pd.notna(current_aqi):
            cat = predict.get_aqi_category(current_aqi)
            color = predict.get_aqi_color(cat)
            text_color = "black" if cat in ["Good", "Moderate", "Unknown"] else "white"
            
            st.markdown(f"""
            <div class="card" style="background-color: {color}; color: {text_color}; margin-bottom: 20px;">
                <p style="margin: 0; font-size: 24px;">Current AQI (Lahore)</p>
                <p class="big-font">{int(current_aqi)}</p>
                <p class="medium-font">{cat}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Current AQI is currently unavailable (Sensor may be down). Showing forecasts only.")

        st.subheader("📈 AQI Forecast Trend")
        
        if forecast_df.empty:
            st.warning("Not enough historical data to generate a forecast sequence.")
        else:
            # Create a Plotly chart
            fig = px.line(
                forecast_df, 
                x="forecast_timestamp", 
                y="predicted_aqi", 
                markers=True,
                labels={"forecast_timestamp": "Time", "predicted_aqi": "Predicted AQI"}
            )
            
            # Add color zones (Good, Moderate, Unhealthy, etc.) as background rects
            fig.add_hrect(y0=0, y1=50, fillcolor="#00e400", opacity=0.1, layer="below", line_width=0)
            fig.add_hrect(y0=50, y1=100, fillcolor="#ffff00", opacity=0.1, layer="below", line_width=0)
            fig.add_hrect(y0=100, y1=150, fillcolor="#ff7e00", opacity=0.1, layer="below", line_width=0)
            fig.add_hrect(y0=150, y1=200, fillcolor="#ff0000", opacity=0.1, layer="below", line_width=0)
            fig.add_hrect(y0=200, y1=300, fillcolor="#8f3f97", opacity=0.1, layer="below", line_width=0)
            fig.add_hrect(y0=300, y1=500, fillcolor="#7e0023", opacity=0.1, layer="below", line_width=0)
            
            fig.update_layout(
                height=400,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="AQI",
                xaxis_title="",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Forecast Table
            st.subheader("📋 Forecast Breakdown")
            display_df = forecast_df.copy()
            display_df["Time"] = display_df["forecast_timestamp"].dt.strftime("%a, %b %d - %I:%M %p")
            display_df["Predicted AQI"] = display_df["predicted_aqi"].round(0).astype(int)
            display_df["Category"] = display_df["aqi_category"]
            display_df = display_df[["Time", "Predicted AQI", "Category"]]
            
            def color_category(val):
                color = predict.get_aqi_color(val)
                text_color = "black" if val in ["Good", "Moderate", "Unknown"] else "white"
                return f"background-color: {color}; color: {text_color}"
                
            styled_df = display_df.style.map(color_category, subset=["Category"])
            st.dataframe(styled_df, use_container_width=True, hide_index=True)

    with tab2:
        st.subheader("Historical Model Accuracy")
        days_back = st.selectbox("Select time range:", [7, 14, 30], index=0)
        
        try:
            from src.feature_store import read_prediction_log
            log_df = read_prediction_log(days_back=days_back)
            
            if log_df.empty:
                st.info("No historical predictions logged yet. The automated training pipeline needs to run to start accumulating data.")
            else:
                # Plotly Chart overlay
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=log_df["forecast_timestamp"], y=log_df["actual_aqi"],
                    mode='lines', name='Actual AQI', line=dict(color='black', width=2)
                ))
                fig2.add_trace(go.Scatter(
                    x=log_df["forecast_timestamp"], y=log_df["predicted_aqi"],
                    mode='lines+markers', name='Predicted AQI', line=dict(color='#008080', dash='dash')
                ))
                
                fig2.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    yaxis_title="AQI",
                    xaxis_title="Date",
                    hovermode="x unified"
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                # Accuracy Metric
                log_df["diff"] = (log_df["predicted_aqi"] - log_df["actual_aqi"]).abs()
                within_15 = (log_df["diff"] <= 15).mean() * 100
                st.success(f"🎯 The model was within ±15 AQI points **{within_15:.1f}%** of the time over the last {days_back} days.")
                
        except Exception as e:
            st.warning(f"Unable to load prediction log: {e}")

    # --- FOOTER ---
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    
    with st.expander("ℹ️ How this works"):
        st.write("""
        This dashboard predicts air quality by leveraging a Machine Learning model trained on 
        thousands of historical weather and pollution data points. Every day, an automated pipeline 
        fetches the latest atmospheric conditions (like temperature, wind speed, and current PM2.5 levels) 
        and feeds them into the model to forecast the AQI up to 72 hours into the future.
        """)
        
    last_update = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st.markdown(f"**Data Sources:** [AQICN](https://aqicn.org/) & [OpenWeather](https://openweathermap.org/) | **Last Updated:** {last_update}")
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
