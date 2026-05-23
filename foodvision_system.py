"""
Food Vision System - Comprehensive AI-Powered Nutrition Analysis
Group 34 Project Prototype
Designed for Google Colab Execution

This script implements:
1. Object Detection using YOLOv8
2. Mock USDA Nutrition Database
3. Portion Estimation (Geometric Modeling Simulation)
4. Personalized Dietary Recommendations

USAGE IN GOOGLE COLAB:
---------------------
1. Create a new cell and run:
   !pip install ultralytics opencv-python-headless

2. Copy this entire script into a new cell and run it.

3. The script will automatically:
   - Download a sample food image
   - Run object detection using YOLOv8
   - Estimate portion sizes
   - Calculate nutrition information
   - Provide personalized recommendations

To use your own image:
- Upload an image to Colab
- Modify the image_path variable in main() function
"""

# ============================================================================
# ENVIRONMENT SETUP & IMPORTS
# ============================================================================
# IMPORTANT: Run this command in a separate cell first:
# !pip install ultralytics opencv-python-headless

import cv2
import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import urllib.request

# Handle Google Colab display - gracefully fallback if not in Colab
try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    # Fallback for local execution
    def cv2_imshow(img):
        """Fallback display function for non-Colab environments."""
        cv2.imshow('Food Vision Analysis', img)
        print("Image displayed in window. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ============================================================================
# MODULE 1: FOOD NUTRITION DATABASE (Mock USDA Food Data Central)
# ============================================================================

class FoodNutritionDatabase:
    """
    Module: Nutrition Database
    Mimics the USDA Food Data Central API with comprehensive food nutrition data.
    Maps YOLO class names to macro-nutrient information per 100g serving.
    """
    
    def __init__(self):
        """Initialize the mock USDA nutrition database."""
        # Comprehensive nutrition data per 100g
        # Format: {food_name: {'calories': kcal, 'protein': g, 'carbs': g, 'fat': g, 'fiber': g}}
        self.database = {
            # Fruits
            'apple': {'calories': 52, 'protein': 0.3, 'carbs': 14, 'fat': 0.2, 'fiber': 2.4, 'density': 0.6},
            'banana': {'calories': 89, 'protein': 1.1, 'carbs': 23, 'fat': 0.3, 'fiber': 2.6, 'density': 0.7},
            'orange': {'calories': 47, 'protein': 0.9, 'carbs': 12, 'fat': 0.1, 'fiber': 2.4, 'density': 0.6},
            
            # Vegetables
            'broccoli': {'calories': 34, 'protein': 2.8, 'carbs': 7, 'fat': 0.4, 'fiber': 2.6, 'density': 0.4},
            'carrot': {'calories': 41, 'protein': 0.9, 'carbs': 10, 'fat': 0.2, 'fiber': 2.8, 'density': 0.6},
            'pizza': {'calories': 266, 'protein': 11, 'carbs': 33, 'fat': 10, 'fiber': 2.3, 'density': 0.8},
            
            # Fast Food & Meals
            'sandwich': {'calories': 250, 'protein': 12, 'carbs': 30, 'fat': 9, 'fiber': 2.0, 'density': 0.7},
            'hamburger': {'calories': 295, 'protein': 17, 'carbs': 30, 'fat': 12, 'fiber': 1.5, 'density': 0.75},
            'hot dog': {'calories': 290, 'protein': 10, 'carbs': 4, 'fat': 26, 'fiber': 0, 'density': 0.8},
            'donut': {'calories': 452, 'protein': 5, 'carbs': 51, 'fat': 25, 'fiber': 1.5, 'density': 0.5},
            
            # Proteins
            'chicken': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'fiber': 0, 'density': 0.9},
            'steak': {'calories': 271, 'protein': 25, 'carbs': 0, 'fat': 19, 'fiber': 0, 'density': 0.95},
            'fish': {'calories': 206, 'protein': 22, 'carbs': 0, 'fat': 12, 'fiber': 0, 'density': 0.85},
            'egg': {'calories': 155, 'protein': 13, 'carbs': 1.1, 'fat': 11, 'fiber': 0, 'density': 0.7},
            
            # Grains & Carbs
            'bread': {'calories': 265, 'protein': 9, 'carbs': 49, 'fat': 3.2, 'fiber': 2.7, 'density': 0.4},
            'pasta': {'calories': 131, 'protein': 5, 'carbs': 25, 'fat': 1.1, 'fiber': 1.8, 'density': 0.5},
            'rice': {'calories': 130, 'protein': 2.7, 'carbs': 28, 'fat': 0.3, 'fiber': 0.4, 'density': 0.6},
            
            # Dairy
            'cheese': {'calories': 402, 'protein': 25, 'carbs': 1.3, 'fat': 33, 'fiber': 0, 'density': 0.9},
            'milk': {'calories': 42, 'protein': 3.4, 'carbs': 5, 'fat': 1, 'fiber': 0, 'density': 1.0},
            
            # Snacks & Others
            'cookie': {'calories': 488, 'protein': 6, 'carbs': 65, 'fat': 24, 'fiber': 1.5, 'density': 0.5},
            'cake': {'calories': 367, 'protein': 4.3, 'carbs': 53, 'fat': 16, 'fiber': 1.2, 'density': 0.4},
        }
    
    def get_nutrition(self, food_name: str) -> Optional[Dict]:
        """
        Retrieve nutrition information for a given food item.
        
        Args:
            food_name: Name of the food (should match YOLO class name)
            
        Returns:
            Dictionary with nutrition info per 100g, or None if not found
        """
        # Case-insensitive lookup
        food_lower = food_name.lower()
        
        # Try direct match first
        if food_lower in self.database:
            return self.database[food_lower]
        
        # Try partial matching for common variations
        for key in self.database.keys():
            if food_lower in key or key in food_lower:
                return self.database[key]
        
        # Default fallback for unknown foods
        return {
            'calories': 200, 'protein': 10, 'carbs': 25, 'fat': 8, 
            'fiber': 2, 'density': 0.7
        }
    
    def calculate_total_nutrition(self, food_name: str, weight_grams: float) -> Dict:
        """
        Calculate total nutrition based on weight.
        
        Args:
            food_name: Name of the food
            weight_grams: Weight in grams
            
        Returns:
            Dictionary with total nutrition values
        """
        nutrition_per_100g = self.get_nutrition(food_name)
        if nutrition_per_100g is None:
            return {}
        
        scale = weight_grams / 100.0
        return {
            'calories': round(nutrition_per_100g['calories'] * scale, 2),
            'protein': round(nutrition_per_100g['protein'] * scale, 2),
            'carbs': round(nutrition_per_100g['carbs'] * scale, 2),
            'fat': round(nutrition_per_100g['fat'] * scale, 2),
            'fiber': round(nutrition_per_100g['fiber'] * scale, 2),
            'weight_grams': round(weight_grams, 2)
        }


# ============================================================================
# MODULE 2: PORTION ESTIMATOR (Geometric Modeling Simulation)
# ============================================================================

class PortionEstimator:
    """
    Module: Portion Estimation
    Technical Challenge: Geometric Modeling for Portion Size Estimation
    
    Uses heuristic-based estimation using bounding box area relative to image size.
    Simulates depth estimation through density factors and coverage percentages.
    """
    
    def __init__(self, base_scale_factor: float = 200.0):
        """
        Initialize the portion estimator.
        
        Args:
            base_scale_factor: Base multiplier for converting area to weight (grams)
        """
        self.base_scale_factor = base_scale_factor
    
    def estimate_weight(
        self, 
        bbox: Tuple[float, float, float, float], 
        image_width: int, 
        image_height: int,
        food_name: str,
        nutrition_db: FoodNutritionDatabase
    ) -> float:
        """
        Estimate food weight in grams based on bounding box geometry.
        
        Logic:
        - Calculate bounding box coverage as percentage of image area
        - Apply food-specific density factor from nutrition database
        - Scale by base factor to estimate weight
        
        Args:
            bbox: Bounding box coordinates (x_center, y_center, width, height) normalized [0-1]
            image_width: Image width in pixels
            image_height: Image height in pixels
            food_name: Name of the detected food
            nutrition_db: Instance of FoodNutritionDatabase to get density
            
        Returns:
            Estimated weight in grams
        """
        try:
            # Extract bounding box dimensions (normalized coordinates)
            _, _, bbox_width, bbox_height = bbox
            
            # Convert normalized coordinates to pixel dimensions
            bbox_width_px = bbox_width * image_width
            bbox_height_px = bbox_height * image_height
            
            # Calculate bounding box area in pixels
            bbox_area_px = bbox_width_px * bbox_height_px
            
            # Calculate image total area
            image_area_px = image_width * image_height
            
            # Calculate coverage percentage
            coverage_percentage = (bbox_area_px / image_area_px) * 100
            
            # Get food-specific density factor from nutrition database
            nutrition_info = nutrition_db.get_nutrition(food_name)
            density_factor = nutrition_info.get('density', 0.7) if nutrition_info else 0.7
            
            # Estimate weight using heuristic formula:
            # weight = coverage% * base_scale * density_factor * area_ratio
            # Area ratio accounts for different image sizes
            area_ratio = (image_area_px / (640 * 640))  # Normalize to standard 640x640
            
            estimated_weight = (
                coverage_percentage * 
                self.base_scale_factor * 
                density_factor * 
                area_ratio * 
                0.1  # Calibration factor for realistic weights
            )
            
            # Apply reasonable bounds (10g to 1000g)
            estimated_weight = max(10.0, min(1000.0, estimated_weight))
            
            return estimated_weight
            
        except Exception as e:
            print(f"Error in portion estimation: {e}")
            # Return default weight estimate
            return 150.0


# ============================================================================
# MODULE 3: USER PROFILE
# ============================================================================

class UserProfile:
    """
    Module: User Profile Management
    Stores user metabolic information and dietary goals.
    """
    
    def __init__(
        self, 
        age: int = 30,
        gender: str = 'male',
        height_cm: float = 175.0,
        weight_kg: float = 75.0,
        activity_level: str = 'moderate',
        goal: str = 'weight_loss',
        daily_calorie_target: Optional[float] = None
    ):
        """
        Initialize user profile with metabolic data.
        
        Args:
            age: User age in years
            gender: 'male' or 'female'
            height_cm: Height in centimeters
            weight_kg: Current weight in kilograms
            activity_level: 'sedentary', 'light', 'moderate', 'active', 'very_active'
            goal: 'weight_loss', 'muscle_gain', 'maintenance', 'weight_gain'
            daily_calorie_target: Optional explicit calorie target (if None, calculated)
        """
        self.age = age
        self.gender = gender.lower()
        self.height_cm = height_cm
        self.weight_kg = weight_kg
        self.activity_level = activity_level.lower()
        self.goal = goal.lower()
        
        # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor Equation
        if self.gender == 'male':
            self.bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
        else:
            self.bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
        
        # Activity multipliers
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'active': 1.725,
            'very_active': 1.9
        }
        
        # Calculate TDEE (Total Daily Energy Expenditure)
        self.tdee = self.bmr * activity_multipliers.get(self.activity_level, 1.55)
        
        # Adjust TDEE based on goal
        goal_adjustments = {
            'weight_loss': -500,  # 500 calorie deficit
            'muscle_gain': 300,   # 300 calorie surplus
            'maintenance': 0,
            'weight_gain': 500
        }
        
        # Set daily calorie target
        if daily_calorie_target is None:
            self.daily_calorie_target = self.tdee + goal_adjustments.get(self.goal, 0)
        else:
            self.daily_calorie_target = daily_calorie_target
        
        # Macro targets (percentage of calories)
        if self.goal == 'weight_loss':
            self.macro_targets = {'protein': 30, 'carbs': 40, 'fat': 30}
        elif self.goal == 'muscle_gain':
            self.macro_targets = {'protein': 35, 'carbs': 45, 'fat': 20}
        else:
            self.macro_targets = {'protein': 25, 'carbs': 45, 'fat': 30}
    
    def get_profile_summary(self) -> Dict:
        """Return a summary dictionary of user profile."""
        return {
            'age': self.age,
            'gender': self.gender,
            'height_cm': self.height_cm,
            'weight_kg': self.weight_kg,
            'bmr': round(self.bmr, 2),
            'tdee': round(self.tdee, 2),
            'daily_calorie_target': round(self.daily_calorie_target, 2),
            'goal': self.goal,
            'macro_targets': self.macro_targets
        }


# ============================================================================
# MODULE 4: FOOD VISION SYSTEM (Main Controller)
# ============================================================================

class FoodVisionSystem:
    """
    Module: Main Food Vision System Controller
    Integrates YOLOv8 object detection with nutrition analysis.
    """
    
    def __init__(self, model_name: str = 'yolov8n.pt'):
        """
        Initialize the Food Vision System.
        
        Args:
            model_name: YOLO model name (yolov8n.pt for nano, faster inference)
        """
        print(f"Loading YOLO model: {model_name}...")
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
            print("YOLO model loaded successfully!")
        except ImportError:
            raise ImportError("Please install ultralytics: pip install ultralytics")
        except Exception as e:
            raise Exception(f"Error loading YOLO model: {e}")
        
        self.nutrition_db = FoodNutritionDatabase()
        self.portion_estimator = PortionEstimator()
        
        # Get class names from the model (works for both custom and default models)
        try:
            # Custom trained models will have their class names
            self.food_classes = list(self.model.names.values()) if hasattr(self.model, 'names') else []
            
            # If model doesn't have class names, use default COCO food classes
            if not self.food_classes:
                self.food_classes = [
                    'apple', 'banana', 'sandwich', 'pizza', 'hot dog', 'donut', 
                    'cake', 'orange', 'broccoli', 'carrot', 'hamburger'
                ]
        except:
            # Fallback to default food classes
            self.food_classes = [
                'apple', 'banana', 'sandwich', 'pizza', 'hot dog', 'donut', 
                'cake', 'orange', 'broccoli', 'carrot', 'hamburger'
            ]
        
        print(f"Model loaded with {len(self.food_classes)} food classes: {self.food_classes[:10]}...")
    
    def analyze_image(self, image_path: str, user_profile: UserProfile) -> Dict:
        """
        Analyze an image for food detection and nutrition calculation.
        
        Args:
            image_path: Path to the image file
            user_profile: UserProfile instance for personalized analysis
            
        Returns:
            Dictionary containing detection results and nutrition breakdown
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            image_height, image_width = image.shape[:2]
            
            # Run YOLO inference
            print("Running object detection...")
            results = self.model(image_path)
            
            # Parse results
            detections = []
            total_nutrition = {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0,
                'foods': []
            }
            
            # Process each detection
            for result in results:
                boxes = result.boxes
                
                for i, box in enumerate(boxes):
                    # Get class ID and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Filter low confidence detections
                    if confidence < 0.3:
                        continue
                    
                    # Get class name from model
                    class_name = self.model.names[class_id]
                    
                    # Accept all detections (custom trained models are already trained on food classes)
                    # Get bounding box (normalized coordinates)
                    bbox_xyxy = box.xyxy[0].cpu().numpy()
                    bbox_xywhn = box.xywhn[0].cpu().numpy()
                    
                    # Estimate portion weight
                    weight = self.portion_estimator.estimate_weight(
                        bbox_xywhn, image_width, image_height, 
                        class_name, self.nutrition_db
                    )
                    
                    # Get nutrition information
                    nutrition = self.nutrition_db.calculate_total_nutrition(
                        class_name, weight
                    )
                    
                    # Add to totals
                    total_nutrition['calories'] += nutrition.get('calories', 0)
                    total_nutrition['protein'] += nutrition.get('protein', 0)
                    total_nutrition['carbs'] += nutrition.get('carbs', 0)
                    total_nutrition['fat'] += nutrition.get('fat', 0)
                    total_nutrition['fiber'] += nutrition.get('fiber', 0)
                    
                    # Store detection info
                    detection = {
                        'food_name': class_name,
                        'confidence': confidence,
                        'bbox': bbox_xyxy.tolist(),
                        'weight_grams': weight,
                        'nutrition': nutrition
                    }
                    detections.append(detection)
                    total_nutrition['foods'].append(detection)
            
            # Visualize results
            visualized_image = self._visualize_detections(image, detections)
            
            return {
                'image': visualized_image,
                'detections': detections,
                'total_nutrition': total_nutrition,
                'image_dimensions': (image_width, image_height)
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _visualize_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        for detection in detections:
            food_name = detection['food_name']
            bbox = detection['bbox']
            calories = detection['nutrition'].get('calories', 0)
            confidence = detection['confidence']
            
            # Convert bbox from xyxy to integer coordinates
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label text
            label = f"{food_name}: {calories:.0f} kcal ({confidence:.2f})"
            
            # Calculate text size for background rectangle
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            cv2.rectangle(
                annotated_image, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                (0, 255, 0), 
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_image, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        
        return annotated_image


# ============================================================================
# MODULE 5: DIETARY RECOMMENDER
# ============================================================================

class DietaryRecommender:
    """
    Module: Personalized Dietary Recommendation Engine
    Implements hybrid filtering logic combining User Goals + Food Content analysis.
    """
    
    def __init__(self, nutrition_db: FoodNutritionDatabase):
        """
        Initialize the recommendation engine.
        
        Args:
            nutrition_db: Instance of FoodNutritionDatabase
        """
        self.nutrition_db = nutrition_db
    
    def recommend(
        self, 
        detected_foods: List[Dict], 
        total_nutrition: Dict,
        user_profile: UserProfile
    ) -> List[str]:
        """
        Generate personalized dietary recommendations.
        
        Args:
            detected_foods: List of detected food items
            total_nutrition: Total nutrition dictionary
            user_profile: UserProfile instance
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        total_calories = total_nutrition.get('calories', 0)
        total_protein = total_nutrition.get('protein', 0)
        total_carbs = total_nutrition.get('carbs', 0)
        total_fat = total_nutrition.get('fat', 0)
        
        # Compare against daily target (assuming this is one meal, estimate 30% of daily)
        meal_calorie_budget = user_profile.daily_calorie_target * 0.3
        
        # Calorie analysis
        if total_calories > meal_calorie_budget * 1.2:
            excess = total_calories - meal_calorie_budget
            recommendations.append(
                f"⚠️ Meal exceeds recommended calorie budget by {excess:.0f} kcal. "
                f"Consider reducing portion sizes or choosing lighter options."
            )
        elif total_calories < meal_calorie_budget * 0.7:
            recommendations.append(
                "✅ Meal is within calorie budget. Good portion control!"
            )
        
        # Goal-specific recommendations
        if user_profile.goal == 'weight_loss':
            # Check for high-calorie foods
            high_calorie_foods = [
                food for food in detected_foods 
                if food['nutrition'].get('calories', 0) > 300
            ]
            
            if high_calorie_foods:
                food_names = [f['food_name'] for f in high_calorie_foods]
                recommendations.append(
                    f"💡 For weight loss, consider healthier alternatives to: {', '.join(food_names)}. "
                    f"Try options with more protein and fiber."
                )
            
            # Check protein intake
            protein_percentage = (total_protein * 4 / total_calories * 100) if total_calories > 0 else 0
            if protein_percentage < 25:
                recommendations.append(
                    "💪 Increase protein intake for better satiety and muscle preservation during weight loss."
                )
        
        elif user_profile.goal == 'muscle_gain':
            # Check protein intake
            protein_percentage = (total_protein * 4 / total_calories * 100) if total_calories > 0 else 0
            if protein_percentage < 30:
                recommendations.append(
                    "💪 Add more protein-rich foods (chicken, eggs, lean meats) to support muscle growth."
                )
            
            if total_calories < meal_calorie_budget:
                recommendations.append(
                    f"📈 Consider adding healthy calorie-dense foods to meet muscle gain targets."
                )
        
        # Macro balance recommendations
        carb_percentage = (total_carbs * 4 / total_calories * 100) if total_calories > 0 else 0
        fat_percentage = (total_fat * 9 / total_calories * 100) if total_calories > 0 else 0
        
        if fat_percentage > 40:
            recommendations.append(
                "🥑 Meal is high in fat. Balance with more lean proteins and complex carbohydrates."
            )
        
        if carb_percentage > 60:
            recommendations.append(
                "🍞 Meal is high in carbs. Consider adding protein and healthy fats for better balance."
            )
        
        # Fiber check
        fiber = total_nutrition.get('fiber', 0)
        if fiber < 5:
            recommendations.append(
                "🌾 Add more fiber-rich foods (vegetables, whole grains) for better digestion and satiety."
            )
        
        # Default positive message if no issues
        if not recommendations:
            recommendations.append(
                "✅ Great meal choice! Well-balanced nutrition aligned with your goals."
            )
        
        return recommendations
    
    def suggest_alternatives(self, food_name: str, user_profile: UserProfile) -> List[str]:
        """
        Suggest healthier alternatives for a given food.
        
        Args:
            food_name: Name of the food to replace
            user_profile: UserProfile instance
            
        Returns:
            List of alternative food suggestions
        """
        current_nutrition = self.nutrition_db.get_nutrition(food_name)
        if not current_nutrition:
            return []
        
        alternatives = []
        
        # Weight loss alternatives
        if user_profile.goal == 'weight_loss':
            if current_nutrition['calories'] > 300:
                alternatives.extend([
                    "Grilled chicken breast",
                    "Steamed vegetables",
                    "Greek yogurt with berries",
                    "Salad with lean protein"
                ])
        
        return alternatives


# ============================================================================
# EXECUTION BLOCK (Main Function)
# ============================================================================

def download_sample_image(url: str, save_path: str = 'sample_food.jpg') -> str:
    """
    Download a sample food image from a URL.
    
    Args:
        url: URL of the image to download
        save_path: Local path to save the image
        
    Returns:
        Path to downloaded image
    """
    try:
        print(f"Downloading sample image from {url}...")
        urllib.request.urlretrieve(url, save_path)
        print(f"Image saved to {save_path}")
        return save_path
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def main():
    """
    Main execution function - Demo of the Food Vision System.
    """
    print("=" * 70)
    print("FOOD VISION SYSTEM - AI-Powered Nutrition Analysis")
    print("Group 34 Project Prototype")
    print("=" * 70)
    print()
    
    # Step 1: Download sample image
    # Using a publicly available food image
    sample_image_urls = [
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ae38?w=800",  # Pizza
        "https://images.unsplash.com/photo-1512058564366-18510be2db19?w=800",  # Hamburger
        "https://images.unsplash.com/photo-1579952363873-27f3bade9f55?w=800",  # Apple
    ]
    
    image_path = None
    for url in sample_image_urls:
        image_path = download_sample_image(url, 'sample_food.jpg')
        if image_path:
            break
    
    if not image_path:
        print("⚠️ Could not download sample image. Please provide a local image path.")
        print("You can upload an image to Colab and set image_path = 'your_image.jpg'")
        return
    
    # Step 2: Create user profile
    print("\nCreating user profile...")
    user_profile = UserProfile(
        age=28,
        gender='male',
        height_cm=180.0,
        weight_kg=80.0,
        activity_level='moderate',
        goal='weight_loss'
    )
    
    profile_summary = user_profile.get_profile_summary()
    print(f"User Profile:")
    print(f"  Goal: {profile_summary['goal']}")
    print(f"  Daily Calorie Target: {profile_summary['daily_calorie_target']:.0f} kcal")
    print(f"  TDEE: {profile_summary['tdee']:.0f} kcal")
    print()
    
    # Step 3: Initialize Food Vision System
    print("Initializing Food Vision System...")
    food_vision = FoodVisionSystem(model_name='yolov8n.pt')
    print()
    
    # Step 4: Analyze image
    print("Analyzing image for food detection and nutrition...")
    results = food_vision.analyze_image(image_path, user_profile)
    
    if results is None:
        print("❌ Analysis failed. Please check the error messages above.")
        return
    
    # Step 5: Display results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    # Display image with annotations
    print("\n📸 Annotated Image (with detections):")
    cv2_imshow(results['image'])
    
    # Print detection summary
    print(f"\n🔍 Detected Foods: {len(results['detections'])}")
    for i, detection in enumerate(results['detections'], 1):
        print(f"\n  {i}. {detection['food_name'].upper()}")
        print(f"     Confidence: {detection['confidence']:.2%}")
        print(f"     Estimated Weight: {detection['weight_grams']:.1f}g")
        nutrition = detection['nutrition']
        print(f"     Nutrition:")
        print(f"       • Calories: {nutrition['calories']:.1f} kcal")
        print(f"       • Protein: {nutrition['protein']:.1f}g")
        print(f"       • Carbs: {nutrition['carbs']:.1f}g")
        print(f"       • Fat: {nutrition['fat']:.1f}g")
    
    # Print total nutrition
    total = results['total_nutrition']
    print("\n" + "-" * 70)
    print("📊 TOTAL MEAL NUTRITION")
    print("-" * 70)
    print(f"  Calories: {total['calories']:.1f} kcal")
    print(f"  Protein: {total['protein']:.1f}g")
    print(f"  Carbohydrates: {total['carbs']:.1f}g")
    print(f"  Fat: {total['fat']:.1f}g")
    print(f"  Fiber: {total['fiber']:.1f}g")
    
    # Step 6: Get recommendations
    print("\n" + "-" * 70)
    print("💡 PERSONALIZED RECOMMENDATIONS")
    print("-" * 70)
    
    recommender = DietaryRecommender(food_vision.nutrition_db)
    recommendations = recommender.recommend(
        results['detections'],
        results['total_nutrition'],
        user_profile
    )
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n  {i}. {rec}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete! 🎉")
    print("=" * 70)


# Run the main function
if __name__ == "__main__":
    main()

