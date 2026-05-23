# Food Vision

AI-powered nutrition analysis from meal photos. Food Vision detects food items with **YOLOv8**, estimates portion size, calculates calories and macros from a built-in nutrition database, and provides personalized dietary recommendations based on your profile and goals.

**Group 34** — academic prototype for AI-powered meal analysis.

---

## Features

- **Food detection** — Multi-item recognition via Ultralytics YOLOv8 (custom-trained or pretrained weights)
- **Portion estimation** — Weight in grams from bounding-box geometry and food density factors
- **Nutrition breakdown** — Calories, protein, carbs, fat, and fiber per item and per meal
- **Personalized advice** — BMR/TDEE, daily targets, and goal-based recommendations (weight loss, muscle gain, etc.)
- **Web UI** — Streamlit app with image upload, camera capture, and profile settings

---

## Project structure

```
foodvision/
├── foodvision_system.py      # Core pipeline (detection, nutrition, recommendations)
├── foodvision_app.py         # Streamlit web interface
├── run_interface.py          # Launcher for the Streamlit app
├── train_food_detection_model.ipynb   # Fine-tune YOLOv8 on Roboflow dataset
└── requirements.txt
```

Model weights (e.g. `best (1).pt`) are not committed to the repo — place your trained `.pt` file in the project root after training.

---

## Requirements

- Python 3.8+
- pip

Optional but recommended for training and faster inference:

- NVIDIA GPU with CUDA (CPU works for the demo, but inference is slower)

---

## Installation

1. Clone or download this repository and open a terminal in the project folder.

2. Create a virtual environment (recommended):

   ```bash
   python -m venv venv
   ```

   **Windows (PowerShell):**

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**

   ```bash
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Quick start (web interface)

1. (Optional) Add your trained model as `best (1).pt` in the project root. If it is missing, the app falls back to `yolov8n.pt` (generic COCO classes, not food-specific).

2. Start the app:

   ```bash
   python run_interface.py
   ```

   Or directly:

   ```bash
   streamlit run foodvision_app.py
   ```

3. In the sidebar, configure your **user profile** and click **Load/Reload Model**.

4. On the **Analyze Food** tab, upload an image or use the camera, then click **Analyze Food**.

---

## Using the core library

You can run analysis from Python without Streamlit:

```python
from foodvision_system import FoodVisionSystem, UserProfile

user = UserProfile(
    age=30,
    gender="male",
    height_cm=175.0,
    weight_kg=75.0,
    activity_level="moderate",
    goal="weight_loss",
)

system = FoodVisionSystem(model_name="best (1).pt")  # or "yolov8n.pt"
results = system.analyze_image("path/to/meal.jpg", user)
print(results)
```

The script also supports **Google Colab** — see the header comments in `foodvision_system.py` for Colab setup (`pip install ultralytics opencv-python-headless`).

---

## Training a custom model

1. Open `train_food_detection_model.ipynb` in Jupyter or Google Colab.

2. Follow the notebook to download the **Roboflow** dataset (`foodvision-detection-acvpj`), configure `data.yaml`, and fine-tune **YOLOv8n**.

3. Export the best checkpoint (e.g. `best.pt`) and copy it to the project root as `best (1).pt` (or update the model name in `foodvision_app.py` / `run_interface.py`).

Training outputs under `runs/` are gitignored.

---

## Supported foods (nutrition database)

The in-memory nutrition database includes 20+ items, for example: apple, banana, orange, broccoli, carrot, pizza, sandwich, hamburger, hot dog, donut, chicken, steak, fish, egg, bread, pasta, rice, cheese, milk, cookie, cake.

Detected class names must match these keys for full nutrition data. Custom YOLO class names should align with the database or be extended in `FoodNutritionDatabase` inside `foodvision_system.py`.

---

## Limitations

- Portion weights are **heuristic** (bbox area + density), not clinical-grade.
- Nutrition values are **static lookups**, not a live USDA/API.
- Recommendations are **rule-based**, not medical advice.
- Best results require a **food-specific** trained model; the default `yolov8n.pt` is not tuned for this task.

---

## License

Academic / course project — check with your institution or group for redistribution terms if you plan to publish or reuse the work.
