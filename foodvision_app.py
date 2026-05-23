"""
Food Vision Interface - Streamlit Web Application
Group 34 Project Prototype

A user-friendly web interface for food detection and nutrition analysis.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile
import os
from pathlib import Path
from foodvision_system import (
    FoodVisionSystem, 
    UserProfile, 
    DietaryRecommender,
    FoodNutritionDatabase
)

# Page configuration
st.set_page_config(
    page_title="Food Vision - AI Nutrition Analysis",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2E7D32;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .nutrition-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .food-item {
        background-color: #F1F8E9;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None

@st.cache_resource
def load_model(model_path: str):
    """Load the YOLO model with caching."""
    try:
        system = FoodVisionSystem(model_name=model_path)
        return system
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🍎 Food Vision System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Nutrition Analysis & Food Detection</p>', unsafe_allow_html=True)
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Model Configuration")
        model_file = st.selectbox(
            "Select Model",
            ["best (1).pt", "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
            help="Select the trained model to use for food detection"
        )
        
        model_path = model_file
        if not os.path.exists(model_path):
            st.warning(f"⚠️ Model file '{model_path}' not found. Please ensure the model file is in the current directory.")
            st.info("Using default YOLOv8n model for now.")
            model_path = "yolov8n.pt"
        
        # User profile settings
        st.subheader("👤 User Profile")
        age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["male", "female"], index=0)
        height_cm = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=175.0, step=1.0)
        weight_kg = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=75.0, step=0.5)
        activity_level = st.selectbox(
            "Activity Level",
            ["sedentary", "light", "moderate", "active", "very_active"],
            index=2,
            help="Sedentary: little/no exercise\nLight: light exercise 1-3 days/week\nModerate: moderate exercise 3-5 days/week\nActive: hard exercise 6-7 days/week\nVery Active: very hard exercise, physical job"
        )
        goal = st.selectbox(
            "Fitness Goal",
            ["weight_loss", "muscle_gain", "maintenance", "weight_gain"],
            index=0
        )
        
        # Load model button
        if st.button("🔄 Load/Reload Model", type="primary"):
            with st.spinner(f"Loading model {model_path}..."):
                st.session_state.model = load_model(model_path)
                if st.session_state.model:
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded successfully!")
                else:
                    st.session_state.model_loaded = False
                    st.error("❌ Failed to load model")
        
        # Show model status
        if st.session_state.model_loaded:
            st.success("✅ Model Ready")
        else:
            st.info("ℹ️ Click 'Load/Reload Model' to start")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📸 Analyze Food", "📊 User Profile", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Food Image")
        
        # Image input methods
        input_method = st.radio(
            "Choose input method:",
            ["📤 Upload Image", "📷 Camera Capture"],
            horizontal=True
        )
        
        uploaded_file = None
        
        if input_method == "📤 Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a photo of food to analyze"
            )
        else:
            uploaded_file = st.camera_input("Take a picture of your food")
        
        if uploaded_file is not None:
            # Load model if not loaded
            if not st.session_state.model_loaded:
                with st.spinner("Loading model for the first time..."):
                    st.session_state.model = load_model(model_path)
                    if st.session_state.model:
                        st.session_state.model_loaded = True
                    else:
                        st.error("❌ Failed to load model. Please check the model file.")
                        st.stop()
            
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Create user profile
            user_profile = UserProfile(
                age=int(age),
                gender=gender,
                height_cm=float(height_cm),
                weight_kg=float(weight_kg),
                activity_level=activity_level,
                goal=goal
            )
            
            # Analyze button
            if st.button("🔍 Analyze Food", type="primary", use_container_width=True):
                with st.spinner("Analyzing image... This may take a moment."):
                    # Save uploaded image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                        image.save(tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    try:
                        # Analyze the image
                        results = st.session_state.model.analyze_image(tmp_path, user_profile)
                        
                        if results is None:
                            st.error("❌ Analysis failed. Please try again with a different image.")
                        else:
                            # Display results in columns
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.subheader("📸 Detection Results")
                                # Convert annotated image to display
                                annotated_img = results['image']
                                annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                                st.image(annotated_img_rgb, caption="Detected Foods", use_container_width=True)
                            
                            with col2:
                                st.subheader("🔍 Detected Foods")
                                detections = results['detections']
                                
                                if len(detections) == 0:
                                    st.warning("⚠️ No food items detected in this image.")
                                else:
                                    for i, detection in enumerate(detections, 1):
                                        with st.expander(f"🍽️ {detection['food_name'].upper()}"):
                                            st.write(f"**Confidence:** {detection['confidence']:.1%}")
                                            st.write(f"**Weight:** {detection['weight_grams']:.1f}g")
                                            nutrition = detection['nutrition']
                                            st.write("**Nutrition:**")
                                            st.write(f"- Calories: {nutrition['calories']:.1f} kcal")
                                            st.write(f"- Protein: {nutrition['protein']:.1f}g")
                                            st.write(f"- Carbs: {nutrition['carbs']:.1f}g")
                                            st.write(f"- Fat: {nutrition['fat']:.1f}g")
                            
                            # Nutrition summary
                            st.markdown("---")
                            st.subheader("📊 Total Meal Nutrition")
                            
                            total_nutrition = results['total_nutrition']
                            
                            # Create columns for nutrition metrics
                            nut_col1, nut_col2, nut_col3, nut_col4, nut_col5 = st.columns(5)
                            
                            with nut_col1:
                                st.metric("Calories", f"{total_nutrition['calories']:.1f}", "kcal")
                            with nut_col2:
                                st.metric("Protein", f"{total_nutrition['protein']:.1f}", "g")
                            with nut_col3:
                                st.metric("Carbs", f"{total_nutrition['carbs']:.1f}", "g")
                            with nut_col4:
                                st.metric("Fat", f"{total_nutrition['fat']:.1f}", "g")
                            with nut_col5:
                                st.metric("Fiber", f"{total_nutrition['fiber']:.1f}", "g")
                            
                            # Personalized recommendations
                            st.markdown("---")
                            st.subheader("💡 Personalized Recommendations")
                            
                            recommender = DietaryRecommender(st.session_state.model.nutrition_db)
                            recommendations = recommender.recommend(
                                detections,
                                total_nutrition,
                                user_profile
                            )
                            
                            for rec in recommendations:
                                st.info(f"💡 {rec}")
                            
                            # Progress bar for daily calorie goal
                            st.markdown("---")
                            st.subheader("🎯 Daily Goal Progress")
                            
                            daily_target = user_profile.daily_calorie_target
                            meal_calories = total_nutrition['calories']
                            percentage = min((meal_calories / daily_target) * 100, 100)
                            
                            st.write(f"Daily Calorie Target: {daily_target:.0f} kcal")
                            st.write(f"This meal: {meal_calories:.0f} kcal ({percentage:.1f}% of daily target)")
                            st.progress(percentage / 100)
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        st.exception(e)
                    finally:
                        # Clean up temporary file
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
    
    with tab2:
        st.header("User Profile Information")
        
        user_profile = UserProfile(
            age=int(age),
            gender=gender,
            height_cm=float(height_cm),
            weight_kg=float(weight_kg),
            activity_level=activity_level,
            goal=goal
        )
        
        profile_summary = user_profile.get_profile_summary()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Information")
            st.write(f"**Age:** {profile_summary['age']} years")
            st.write(f"**Gender:** {profile_summary['gender'].title()}")
            st.write(f"**Height:** {profile_summary['height_cm']} cm")
            st.write(f"**Weight:** {profile_summary['weight_kg']} kg")
            st.write(f"**Goal:** {profile_summary['goal'].replace('_', ' ').title()}")
        
        with col2:
            st.subheader("Metabolic Information")
            st.metric("BMR (Basal Metabolic Rate)", f"{profile_summary['bmr']:.0f}", "kcal/day")
            st.metric("TDEE (Total Daily Energy Expenditure)", f"{profile_summary['tdee']:.0f}", "kcal/day")
            st.metric("Daily Calorie Target", f"{profile_summary['daily_calorie_target']:.0f}", "kcal/day")
        
        st.subheader("Macro Targets")
        macro_targets = profile_summary['macro_targets']
        st.write(f"**Protein:** {macro_targets['protein']}% of calories")
        st.write(f"**Carbohydrates:** {macro_targets['carbs']}% of calories")
        st.write(f"**Fat:** {macro_targets['fat']}% of calories")
    
    with tab3:
        st.header("About Food Vision System")
        
        st.markdown("""
        ### 🍎 Food Vision - AI-Powered Nutrition Analysis
        
        Food Vision is an intelligent system that uses advanced computer vision and machine learning 
        to analyze food images and provide comprehensive nutrition information.
        
        #### ✨ Features:
        - **Food Detection**: Uses YOLOv8 deep learning model to detect multiple food items in images
        - **Nutrition Analysis**: Calculates calories, protein, carbs, fat, and fiber for detected foods
        - **Portion Estimation**: Estimates portion sizes using geometric modeling
        - **Personalized Recommendations**: Provides dietary recommendations based on your profile and goals
        - **Goal Tracking**: Tracks progress toward your daily calorie and macro targets
        
        #### 🔧 Technology Stack:
        - **YOLOv8**: State-of-the-art object detection model
        - **Streamlit**: Modern web interface framework
        - **OpenCV**: Image processing and visualization
        - **Custom Nutrition Database**: Comprehensive food nutrition data
        
        #### 📝 How to Use:
        1. Configure your user profile in the sidebar
        2. Load your trained model (or use default)
        3. Upload a food image or take a photo
        4. Click "Analyze Food" to get instant nutrition analysis
        5. Review detected foods, nutrition breakdown, and personalized recommendations
        
        #### 👥 Group 34 Project Prototype
        
        For questions or support, please refer to the documentation.
        """)


if __name__ == "__main__":
    main()

