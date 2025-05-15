import streamlit as st
import numpy as np
import joblib
import time
import random
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json

# Define OpenWeatherMap API base URL
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

# Load trained model
model = joblib.load("fertilizer_model.pkl")

# Set page config
st.set_page_config(page_title="🌏Smart Soil Analysis System", layout="wide")

# Title and description
st.title("✨Smart Soil Analysis System")
st.write("🚨Real-time soil analysis and fertilizer recommendations")

# OpenWeatherMap API configuration
st.sidebar.title("🔑 API Configuration")
api_key = st.sidebar.text_input("OpenWeatherMap API Key", type="password", 
                               help="Get your free API key from https://openweathermap.org/api")

if not api_key:
    st.sidebar.warning("""
    ⚠️ Please enter your OpenWeatherMap API key to access weather data.
    
    To get an API key:
    1. Go to https://openweathermap.org/
    2. Sign up for a free account
    3. Verify your email address
    4. Wait 2 hours for API key activation
    5. Go to your account dashboard
    6. Find your API key under "My API Keys"
    """)
    st.stop()

# Test API connection
if st.sidebar.button("Test API Connection"):
    with st.sidebar:
        with st.spinner("Testing connection..."):
            try:
                test_params = {
                    "lat": 20.5937,  # Test with India's coordinates
                    "lon": 78.9629,
                    "appid": api_key,
                    "units": "metric"
                }
                response = requests.get(BASE_URL, params=test_params)
                if response.status_code == 200:
                    st.success("✅ API connection successful!")
                else:
                    data = response.json()
                    if response.status_code == 401:
                        st.error("""
                        ❌ Invalid API key. Please check:
                        1. Did you copy the entire API key?
                        2. Did you wait 2 hours after activation?
                        3. Is your account email verified?
                        """)
                    else:
                        st.error(f"❌ Error: {data.get('message', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Connection error: {str(e)}")

# Function to get real weather data
def get_real_weather_data(latitude, longitude):
    try:
        params = {
            "lat": latitude,
            "lon": longitude,
            "appid": api_key,
            "units": "metric"
        }
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if response.status_code == 200:
            forecast = []
            for item in data['list']:
                forecast.append({
                    "date": datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d %H:%M'),
                    "temperature": round(item['main']['temp'], 1),
                    "humidity": item['main']['humidity'],
                    "rainfall": item.get('rain', {}).get('3h', 0),
                    "wind_speed": round(item['wind']['speed'] * 3.6, 1),
                    "wind_direction": get_wind_direction(item['wind']['deg']),
                    "cloud_cover": item['clouds']['all'],
                    "pressure": round(item['main']['pressure'], 1),
                    "visibility": round(item['visibility'] / 1000, 1),
                    "description": item['weather'][0]['description'],
                    "icon": item['weather'][0]['icon']
                })
            return forecast
        else:
            if response.status_code == 401:
                st.error("""
                ❌ Invalid API key. Please check:
                1. Did you copy the entire API key?
                2. Did you wait 2 hours after activation?
                3. Is your account email verified?
                4. Try clicking 'Test API Connection' in the sidebar
                """)
            else:
                st.error(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

def get_wind_direction(degrees):
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = round(degrees / (360. / len(directions))) % len(directions)
    return directions[index]

# Extended Crop database with more crops and details
CROPS = {
    "Rice": {
        "N": 120, "P": 60, "K": 60, "pH": (5.5, 6.5),
        "growth_stages": ["Seedling", "Tillering", "Panicle Initiation", "Flowering", "Grain Filling"],
        "water_requirement": "High",
        "temperature_range": (20, 35),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Basmati", "Non-Basmati", "Hybrid"],
        "yield_potential": "4-6 tons/ha"
    },
    "Wheat": {
        "N": 100, "P": 50, "K": 50, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Tillering", "Stem Elongation", "Heading", "Ripening"],
        "water_requirement": "Medium",
        "temperature_range": (15, 25),
        "season": ["Rabi"],
        "varieties": ["Durum", "Bread Wheat", "Emmer"],
        "yield_potential": "3-5 tons/ha"
    },
    "Maize": {
        "N": 120, "P": 60, "K": 60, "pH": (5.8, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Tasseling", "Silking", "Maturity"],
        "water_requirement": "Medium",
        "temperature_range": (18, 32),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Sweet Corn", "Field Corn", "Popcorn"],
        "yield_potential": "5-8 tons/ha"
    },
    "Soybean": {
        "N": 20, "P": 60, "K": 80, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Pod Development", "Maturity"],
        "water_requirement": "Medium",
        "temperature_range": (20, 30),
        "season": ["Kharif"],
        "varieties": ["Black", "Yellow", "Green"],
        "yield_potential": "2-3 tons/ha"
    },
    "Cotton": {
        "N": 100, "P": 50, "K": 50, "pH": (5.8, 6.5),
        "growth_stages": ["Germination", "Vegetative", "Square Formation", "Flowering", "Boll Development"],
        "water_requirement": "Medium",
        "temperature_range": (20, 35),
        "season": ["Kharif"],
        "varieties": ["Upland", "Pima", "Egyptian"],
        "yield_potential": "2-3 bales/ha"
    },
    "Potato": {
        "N": 120, "P": 60, "K": 120, "pH": (5.0, 6.0),
        "growth_stages": ["Sprouting", "Vegetative", "Tuber Initiation", "Tuber Bulking", "Maturity"],
        "water_requirement": "High",
        "temperature_range": (15, 25),
        "season": ["Rabi"],
        "varieties": ["Russet", "Red", "White"],
        "yield_potential": "20-30 tons/ha"
    },
    "Tomato": {
        "N": 100, "P": 50, "K": 150, "pH": (5.5, 6.8),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Fruit Setting", "Harvesting"],
        "water_requirement": "Medium",
        "temperature_range": (20, 30),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Cherry", "Beefsteak", "Roma"],
        "yield_potential": "40-60 tons/ha"
    },
    "Sugarcane": {
        "N": 200, "P": 100, "K": 200, "pH": (6.0, 7.5),
        "growth_stages": ["Germination", "Tillering", "Grand Growth", "Maturity"],
        "water_requirement": "High",
        "temperature_range": (20, 35),
        "season": ["Kharif"],
        "varieties": ["Early", "Mid", "Late"],
        "yield_potential": "80-100 tons/ha"
    },
    "Millets": {
        "N": 60, "P": 30, "K": 30, "pH": (6.0, 7.5),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Grain Formation"],
        "water_requirement": "Low",
        "temperature_range": (20, 35),
        "season": ["Kharif"],
        "varieties": ["Pearl", "Finger", "Foxtail"],
        "yield_potential": "1.5-2.5 tons/ha"
    },
    "Pulses": {
        "N": 20, "P": 40, "K": 20, "pH": (6.0, 7.5),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Pod Formation"],
        "water_requirement": "Low",
        "temperature_range": (20, 30),
        "season": ["Rabi"],
        "varieties": ["Chickpea", "Lentil", "Pigeon Pea"],
        "yield_potential": "1-2 tons/ha"
    },
    "Oilseeds": {
        "N": 40, "P": 20, "K": 20, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Pod Formation"],
        "water_requirement": "Low",
        "temperature_range": (20, 30),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Mustard", "Sunflower", "Groundnut"],
        "yield_potential": "1.5-2.5 tons/ha"
    },
    "Vegetables": {
        "N": 80, "P": 40, "K": 60, "pH": (6.0, 7.0),
        "growth_stages": ["Germination", "Vegetative", "Flowering", "Fruit Setting"],
        "water_requirement": "High",
        "temperature_range": (15, 30),
        "season": ["Kharif", "Rabi"],
        "varieties": ["Leafy", "Root", "Fruit"],
        "yield_potential": "20-30 tons/ha"
    }
}

# Extended Fertilizer database with more brands and types
FERTILIZERS = {
    "Nitrogen": {
        "Urea": {"N": 46, "brands": ["IFFCO", "KRIBHCO", "Nagarjuna", "Chambal", "Tata", "Coromandel"]},
        "Ammonium Nitrate": {"N": 34, "brands": ["Coromandel", "Zuari", "GSFC", "RCF"]},
        "Ammonium Sulfate": {"N": 21, "brands": ["RCF", "GSFC", "IFFCO"]},
        "Calcium Ammonium Nitrate": {"N": 26, "brands": ["Yara", "Haifa", "ICL"]}
    },
    "Phosphorus": {
        "DAP": {"P": 46, "brands": ["IFFCO", "Coromandel", "Zuari", "Paradeep", "RCF"]},
        "SSP": {"P": 16, "brands": ["RCF", "GSFC", "IFFCO", "Coromandel"]},
        "Rock Phosphate": {"P": 30, "brands": ["Paradeep", "Jhamarkotra", "RSMML"]},
        "NPK Complex": {"P": 20, "brands": ["IFFCO", "Coromandel", "Zuari"]}
    },
    "Potassium": {
        "MOP": {"K": 60, "brands": ["IPL", "Zuari", "Coromandel", "IFFCO"]},
        "SOP": {"K": 50, "brands": ["IFFCO", "KRIBHCO", "Yara"]},
        "Potassium Nitrate": {"K": 44, "brands": ["Yara", "Haifa", "ICL"]}
    },
    "Micronutrients": {
        "Zinc Sulfate": {"Zn": 21, "brands": ["Coromandel", "Zuari", "IFFCO"]},
        "Boron": {"B": 11, "brands": ["Yara", "Haifa", "ICL"]},
        "Iron Chelate": {"Fe": 12, "brands": ["Yara", "Haifa", "ICL"]}
    }
}

# Function to simulate sensor readings with more parameters
def get_sensor_readings():
    readings = {
        "pH": round(random.uniform(5.0, 8.0), 1),
        "nitrogen": random.randint(20, 100),
        "phosphorus": random.randint(20, 100),
        "potassium": random.randint(20, 100),
        "moisture": round(random.uniform(30.0, 70.0), 1),
        "temperature": round(random.uniform(15.0, 35.0), 1),
        "humidity": random.randint(40, 90),
        "organic_matter": round(random.uniform(1.0, 5.0), 1),
        "ec": round(random.uniform(0.5, 3.0), 1),
        "soil_type": random.choice(["Sandy", "Loamy", "Clayey"]),
        "micronutrients": {
            "zinc": round(random.uniform(0.5, 2.0), 1),
            "iron": round(random.uniform(2.0, 10.0), 1),
            "manganese": round(random.uniform(1.0, 5.0), 1),
            "copper": round(random.uniform(0.2, 1.0), 1),
            "boron": round(random.uniform(0.2, 1.0), 1)
        }
    }
    return readings

# Function to get detailed weather forecast
def get_weather_forecast():
    today = datetime.now()
    forecast = []
    for i in range(7):
        date = today + timedelta(days=i)
        forecast.append({
            "date": date.strftime("%Y-%m-%d"),
            "temperature": round(random.uniform(15.0, 35.0), 1),
            "humidity": random.randint(40, 90),
            "rainfall": round(random.uniform(0, 20), 1),
            "wind_speed": round(random.uniform(0, 20), 1),
            "wind_direction": random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
            "cloud_cover": random.randint(0, 100),
            "uv_index": random.randint(1, 10),
            "pressure": round(random.uniform(1000, 1020), 1),
            "visibility": random.randint(5, 20),
            "dew_point": round(random.uniform(10, 25), 1)
        })
    return forecast

# Function to calculate fertilizer requirements with specific recommendations
def calculate_fertilizer_requirements(soil_readings, crop_type):
    crop_needs = CROPS[crop_type]
    recommendations = {
        "N": max(0, crop_needs["N"] - soil_readings["nitrogen"]),
        "P": max(0, crop_needs["P"] - soil_readings["phosphorus"]),
        "K": max(0, crop_needs["K"] - soil_readings["potassium"])
    }
    
    # Calculate specific fertilizer amounts
    fertilizer_details = {
        "Nitrogen": {
            "Urea": round(recommendations["N"] / 0.46, 1),
            "Ammonium Nitrate": round(recommendations["N"] / 0.34, 1),
            "Calcium Ammonium Nitrate": round(recommendations["N"] / 0.26, 1)
        },
        "Phosphorus": {
            "DAP": round(recommendations["P"] / 0.46, 1),
            "SSP": round(recommendations["P"] / 0.16, 1),
            "NPK Complex": round(recommendations["P"] / 0.20, 1)
        },
        "Potassium": {
            "MOP": round(recommendations["K"] / 0.60, 1),
            "SOP": round(recommendations["K"] / 0.50, 1),
            "Potassium Nitrate": round(recommendations["K"] / 0.44, 1)
        }
    }
    
    return recommendations, fertilizer_details

# Function to generate historical data with more parameters
def generate_historical_data(crop_type):
    dates = pd.date_range(end=datetime.now(), periods=12, freq='M')
    data = {
        'Date': dates,
        'Yield': [random.uniform(2.0, 4.0) for _ in range(12)],
        'Rainfall': [random.uniform(0, 200) for _ in range(12)],
        'Temperature': [random.uniform(15, 35) for _ in range(12)],
        'Fertilizer_Used': [random.uniform(100, 300) for _ in range(12)],
        'Soil_Moisture': [random.uniform(30, 70) for _ in range(12)],
        'Soil_pH': [random.uniform(5.0, 8.0) for _ in range(12)],
        'Pest_Incidence': [random.uniform(0, 100) for _ in range(12)],
        'Disease_Incidence': [random.uniform(0, 100) for _ in range(12)],
        'Market_Price': [random.uniform(1000, 5000) for _ in range(12)],
        'Labor_Cost': [random.uniform(500, 2000) for _ in range(12)]
    }
    return pd.DataFrame(data)

# Create sidebar for crop selection
st.sidebar.title("🌱 Crop Selection")
selected_crop = st.sidebar.selectbox("Select Crop Type", list(CROPS.keys()))

# Create sidebar for location input
st.sidebar.title("📍 Location")
latitude = st.sidebar.number_input("Latitude", value=20.5937, format="%.4f")
longitude = st.sidebar.number_input("Longitude", value=78.9629, format="%.4f")

# Display crop requirements
st.sidebar.markdown(f"""
### {selected_crop} Requirements:
- **Nitrogen (N)**: {CROPS[selected_crop]['N']} kg/ha
- **Phosphorus (P)**: {CROPS[selected_crop]['P']} kg/ha
- **Potassium (K)**: {CROPS[selected_crop]['K']} kg/ha
- **Optimal pH**: {CROPS[selected_crop]['pH'][0]} - {CROPS[selected_crop]['pH'][1]}
- **Water Requirement**: {CROPS[selected_crop]['water_requirement']}
- **Temperature Range**: {CROPS[selected_crop]['temperature_range'][0]}°C - {CROPS[selected_crop]['temperature_range'][1]}°C
- **Growing Season**: {', '.join(CROPS[selected_crop]['season'])}
- **Varieties**: {', '.join(CROPS[selected_crop]['varieties'])}
- **Yield Potential**: {CROPS[selected_crop]['yield_potential']}
""")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Soil Analysis", "Weather Forecast", "Historical Data"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Real-time Soil Analysis")
        
        if st.button("🔄 Get New Analysis"):
            with st.spinner("Analyzing soil..."):
                time.sleep(2)
                
                # Get sensor readings
                readings = get_sensor_readings()
                
                # Create soil parameter gauge charts
                fig_soil = make_subplots(rows=2, cols=3,
                                       specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                                             [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]])
                
                # Add gauges for key parameters
                fig_soil.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=readings['pH'],
                    title={'text': "pH Level"},
                    gauge={'axis': {'range': [5, 8]},
                          'bar': {'color': "darkblue"}}),
                    row=1, col=1)
                
                fig_soil.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=readings['moisture'],
                    title={'text': "Moisture %"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "green"}}),
                    row=1, col=2)
                
                fig_soil.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=readings['organic_matter'],
                    title={'text': "Organic Matter %"},
                    gauge={'axis': {'range': [0, 5]},
                          'bar': {'color': "brown"}}),
                    row=1, col=3)
                
                fig_soil.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=readings['nitrogen'],
                    title={'text': "Nitrogen (ppm)"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "blue"}}),
                    row=2, col=1)
                
                fig_soil.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=readings['phosphorus'],
                    title={'text': "Phosphorus (ppm)"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "purple"}}),
                    row=2, col=2)
                
                fig_soil.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=readings['potassium'],
                    title={'text': "Potassium (ppm)"},
                    gauge={'axis': {'range': [0, 100]},
                          'bar': {'color': "orange"}}),
                    row=2, col=3)
                
                fig_soil.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_soil, use_container_width=True)
                
                # Display detailed sensor readings
                st.markdown(f"""
                ### Current Soil Parameters:
                - **pH Level**: {readings['pH']}
                - **Nitrogen (N)**: {readings['nitrogen']} ppm
                - **Phosphorus (P)**: {readings['phosphorus']} ppm
                - **Potassium (K)**: {readings['potassium']} ppm
                - **Moisture**: {readings['moisture']}%
                - **Temperature**: {readings['temperature']}°C
                - **Humidity**: {readings['humidity']}%
                - **Organic Matter**: {readings['organic_matter']}%
                - **EC**: {readings['ec']} dS/m
                - **Soil Type**: {readings['soil_type']}
                
                ### Micronutrients:
                - **Zinc**: {readings['micronutrients']['zinc']} ppm
                - **Iron**: {readings['micronutrients']['iron']} ppm
                - **Manganese**: {readings['micronutrients']['manganese']} ppm
                - **Copper**: {readings['micronutrients']['copper']} ppm
                - **Boron**: {readings['micronutrients']['boron']} ppm
                """)
                
                # Calculate fertilizer requirements
                recommendations, fertilizer_details = calculate_fertilizer_requirements(readings, selected_crop)
                
                # Display fertilizer recommendations
                st.success(f"""
                ### 🌱 Fertilizer Recommendations for {selected_crop}:
                
                **Required Nutrients (kg/ha):**
                - Nitrogen (N): {recommendations['N']:.1f}
                - Phosphorus (P): {recommendations['P']:.1f}
                - Potassium (K): {recommendations['K']:.1f}
                
                **Recommended Fertilizers:**
                - **Urea**: {fertilizer_details['Nitrogen']['Urea']:.1f} kg/ha
                - **DAP**: {fertilizer_details['Phosphorus']['DAP']:.1f} kg/ha
                - **MOP**: {fertilizer_details['Potassium']['MOP']:.1f} kg/ha
                
                **Application Schedule:**
                - First application: At planting
                - Second application: During {CROPS[selected_crop]['growth_stages'][1]}
                - Third application: During {CROPS[selected_crop]['growth_stages'][2]}
                """)
                
                # Display soil health status
                st.info(f"""
                ### 🌍 Soil Health Status:
                - pH is {'optimal' if CROPS[selected_crop]['pH'][0] <= readings['pH'] <= CROPS[selected_crop]['pH'][1] else 'needs adjustment'}
                - Nitrogen level is {'sufficient' if readings['nitrogen'] >= CROPS[selected_crop]['N'] * 0.8 else 'low'}
                - Phosphorus level is {'sufficient' if readings['phosphorus'] >= CROPS[selected_crop]['P'] * 0.8 else 'low'}
                - Potassium level is {'sufficient' if readings['potassium'] >= CROPS[selected_crop]['K'] * 0.8 else 'low'}
                - Moisture level is {'optimal' if 40 <= readings['moisture'] <= 60 else 'needs adjustment'}
                - Organic matter is {'good' if readings['organic_matter'] >= 2.0 else 'low'}
                """)
    
    with col2:
        st.subheader("📝 Farming Tips")
        st.markdown("""
        ### Best Practices:
        1. **Soil Preparation**
           - Test soil before planting
           - Maintain proper pH levels
           - Ensure good drainage
           - Add organic matter if needed
        
        2. **Fertilizer Application**
           - Follow recommended dosages
           - Apply at right growth stages
           - Use organic fertilizers when possible
           - Consider split applications
        
        3. **Water Management**
           - Monitor soil moisture
           - Implement drip irrigation
           - Avoid over-watering
           - Consider weather forecasts
        
        4. **Sustainable Practices**
           - Practice crop rotation
           - Use cover crops
           - Implement integrated pest management
           - Maintain soil health
        """)

with tab2:
    st.subheader("🌤️ Detailed Weather Forecast")
    
    if st.button("🔄 Get Latest Weather Data"):
        with st.spinner("Fetching weather data..."):
            # Get real weather data
            forecast = get_real_weather_data(latitude, longitude)
            
            if forecast:
                # Create a DataFrame for the forecast
                forecast_df = pd.DataFrame(forecast)
                
                # Create subplots for weather parameters
                fig_weather = make_subplots(rows=2, cols=2,
                                          subplot_titles=("Temperature & Rainfall", "Humidity & Pressure",
                                                        "Wind Speed & Direction", "Weather Conditions"))
                
                # Add traces for each parameter
                fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['temperature'],
                                              name='Temperature', line=dict(color='red')), row=1, col=1)
                fig_weather.add_trace(go.Bar(x=forecast_df['date'], y=forecast_df['rainfall'],
                                          name='Rainfall'), row=1, col=1)
                
                fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['humidity'],
                                              name='Humidity', line=dict(color='blue')), row=1, col=2)
                fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['pressure'],
                                              name='Pressure', line=dict(color='green')), row=1, col=2)
                
                fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=forecast_df['wind_speed'],
                                              name='Wind Speed', line=dict(color='purple')), row=2, col=1)
                
                # Add weather icons
                weather_icons = []
                for icon in forecast_df['icon']:
                    weather_icons.append(f"https://openweathermap.org/img/wn/{icon}@2x.png")
                
                fig_weather.add_trace(go.Scatter(x=forecast_df['date'], y=[0]*len(forecast_df),
                                              name='Weather', mode='markers',
                                              marker=dict(size=20, symbol='circle'),
                                              hovertext=forecast_df['description']), row=2, col=2)
                
                fig_weather.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig_weather, use_container_width=True)
                
                # Display detailed forecast in a table
                st.subheader("Detailed Weather Forecast")
                forecast_table = forecast_df[['date', 'temperature', 'humidity', 'rainfall', 
                                           'wind_speed', 'wind_direction', 'description']]
                forecast_table.columns = ['Date & Time', 'Temperature (°C)', 'Humidity (%)', 
                                        'Rainfall (mm)', 'Wind Speed (km/h)', 'Wind Direction', 
                                        'Weather Condition']
                st.dataframe(forecast_table.style.format({
                    'Temperature (°C)': '{:.1f}',
                    'Humidity (%)': '{:.0f}',
                    'Rainfall (mm)': '{:.1f}',
                    'Wind Speed (km/h)': '{:.1f}'
                }))
                
                # Display current weather conditions
                current_weather = forecast[0]
                st.markdown(f"""
                ### Current Weather Conditions
                - **Temperature**: {current_weather['temperature']}°C
                - **Humidity**: {current_weather['humidity']}%
                - **Wind**: {current_weather['wind_speed']} km/h from {current_weather['wind_direction']}
                - **Pressure**: {current_weather['pressure']} hPa
                - **Visibility**: {current_weather['visibility']} km
                - **Condition**: {current_weather['description'].title()}
                """)
                
                # Add weather alerts if any
                st.subheader("🌾 Farming Weather Alerts")
                
                # Temperature-based alerts
                if current_weather['temperature'] > 35:
                    st.warning("""
                    ⚠️ **High Temperature Alert**
                    - Avoid fertilizer application during peak hours (10 AM - 4 PM)
                    - Consider early morning (5-7 AM) or late evening (5-7 PM) operations
                    - Increase irrigation frequency to prevent heat stress
                    - Monitor soil moisture more frequently
                    """)
                elif current_weather['temperature'] < 10:
                    st.warning("""
                    ⚠️ **Low Temperature Alert**
                    - Delay fertilizer application until temperatures rise
                    - Protect young plants with mulch or covers
                    - Consider using cold-resistant crop varieties
                    - Monitor for frost damage
                    """)
                
                # Rainfall-based alerts
                if current_weather['rainfall'] > 10:
                    st.warning("""
                    ⚠️ **Heavy Rainfall Alert**
                    - Postpone fertilizer application to prevent runoff
                    - Check drainage systems
                    - Monitor for waterlogging
                    - Prepare for potential disease outbreaks
                    - Consider foliar applications after rain stops
                    """)
                elif current_weather['rainfall'] == 0 and forecast_df['rainfall'].sum() < 5:
                    st.warning("""
                    ⚠️ **Dry Spell Alert**
                    - Increase irrigation frequency
                    - Consider drought-resistant crop varieties
                    - Apply mulch to conserve soil moisture
                    - Monitor soil moisture levels closely
                    - Schedule irrigation during cooler hours
                    """)
                
                # Wind-based alerts
                if current_weather['wind_speed'] > 30:
                    st.warning("""
                    ⚠️ **Strong Wind Alert**
                    - Postpone spraying operations
                    - Secure farm structures and equipment
                    - Protect young plants with windbreaks
                    - Delay fertilizer application to prevent drift
                    - Consider using granular fertilizers instead of sprays
                    """)
                elif current_weather['wind_speed'] < 5:
                    st.info("""
                    ℹ️ **Calm Wind Conditions**
                    - Ideal for spraying operations
                    - Good time for foliar applications
                    - Suitable for aerial spraying if needed
                    - Consider applying liquid fertilizers
                    """)
                
                # Humidity-based alerts
                if current_weather['humidity'] > 80:
                    st.warning("""
                    ⚠️ **High Humidity Alert**
                    - Increased risk of fungal diseases
                    - Monitor for pest infestations
                    - Consider preventive fungicide applications
                    - Ensure proper ventilation in greenhouses
                    - Avoid overhead irrigation
                    """)
                elif current_weather['humidity'] < 40:
                    st.warning("""
                    ⚠️ **Low Humidity Alert**
                    - Increase irrigation frequency
                    - Monitor for water stress
                    - Consider using shade nets
                    - Apply anti-transpirants if needed
                    - Schedule irrigation during early morning
                    """)
                
                # Combined condition alerts
                if current_weather['temperature'] > 30 and current_weather['humidity'] > 70:
                    st.warning("""
                    ⚠️ **Heat Stress Alert**
                    - High risk of heat stress in crops
                    - Increase irrigation frequency
                    - Consider using shade nets
                    - Monitor for wilting
                    - Apply anti-transpirants if needed
                    """)
                
                if current_weather['rainfall'] > 5 and current_weather['wind_speed'] > 20:
                    st.warning("""
                    ⚠️ **Storm Alert**
                    - Secure farm equipment and structures
                    - Postpone all field operations
                    - Check drainage systems
                    - Prepare for potential crop damage
                    - Monitor for waterlogging
                    """)
            else:
                st.error("Failed to fetch weather data. Please try again later.")

with tab3:
    st.subheader("📈 Historical Data Analysis")
    
    # Generate and display historical data
    historical_data = generate_historical_data(selected_crop)
    
    # Create subplots for historical trends
    fig_history = make_subplots(rows=2, cols=2,
                               subplot_titles=("Yield Trend", "Climate Data",
                                             "Soil Parameters", "Economic Indicators"))
    
    # Add traces for each parameter
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Yield'],
                                    name='Yield', line=dict(color='green')), row=1, col=1)
    
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Rainfall'],
                                    name='Rainfall', line=dict(color='blue')), row=1, col=2)
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Temperature'],
                                    name='Temperature', line=dict(color='red')), row=1, col=2)
    
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Soil_Moisture'],
                                    name='Soil Moisture', line=dict(color='brown')), row=2, col=1)
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Soil_pH'],
                                    name='Soil pH', line=dict(color='purple')), row=2, col=1)
    
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Market_Price'],
                                    name='Market Price', line=dict(color='orange')), row=2, col=2)
    fig_history.add_trace(go.Scatter(x=historical_data['Date'], y=historical_data['Labor_Cost'],
                                    name='Labor Cost', line=dict(color='gray')), row=2, col=2)
    
    fig_history.update_layout(height=800, showlegend=True)
    st.plotly_chart(fig_history, use_container_width=True)
    
    # Display statistics
    st.markdown("""
    ### Historical Statistics
    - Average Yield: {:.2f} tons/ha
    - Average Rainfall: {:.2f} mm
    - Average Temperature: {:.2f}°C
    - Average Fertilizer Usage: {:.2f} kg/ha
    - Average Soil Moisture: {:.2f}%
    - Average Soil pH: {:.2f}
    - Average Market Price: ₹{:.2f}/ton
    - Average Labor Cost: ₹{:.2f}/ha
    """.format(
        historical_data['Yield'].mean(),
        historical_data['Rainfall'].mean(),
        historical_data['Temperature'].mean(),
        historical_data['Fertilizer_Used'].mean(),
        historical_data['Soil_Moisture'].mean(),
        historical_data['Soil_pH'].mean(),
        historical_data['Market_Price'].mean(),
        historical_data['Labor_Cost'].mean()
    ))
