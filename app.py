import os
import openai
import json
import shutil
import re
import cv2
import random

from flask import Flask, request, render_template_string, redirect, url_for, session

# Import backend modules from the "backend" folder.
from backend.floorplan_generator import FloorplanGenerator
from backend.floorplan_visualizer import FloorplanVisualizer
from backend.room_type_detector import RoomTypeDetector
from backend.perfect_plan_selector import PerfectPlanSelector
from backend.pretty_floorplan_maker import PrettyFloorplanMaker
from backend.first_floor_plan_generator import FirstFloorPlanGenerator
from backend.first_floor_enhancer import FirstFloorEnhancer
from backend.floorplan_rl_agent import FloorplanRLAgent

############################################
# 1) YOUR OPENAI API KEY
############################################
openai.api_key = "sk-...DuYA"  # Replace with your actual API key

# Configure Flask (no static folder override here since we use generated images)
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this for production

# ---------------------------
# Helper functions
# ---------------------------
def copy_json_for_png(src_png, src_dir, dst_dir):
    base = os.path.splitext(src_png)[0]
    src_json_path = os.path.join(src_dir, base + ".json")
    if os.path.isfile(src_json_path):
        shutil.copy2(src_json_path, dst_dir)

def rename_png_and_json(old_png, old_dir, new_png, new_dir):
    old_png_path = os.path.join(old_dir, old_png)
    new_png_path = os.path.join(new_dir, new_png)
    if os.path.exists(new_png_path):
        os.remove(new_png_path)
    os.rename(old_png_path, new_png_path)
    old_base = os.path.splitext(old_png)[0]
    new_base = os.path.splitext(new_png)[0]
    old_json = old_base + ".json"
    new_json = new_base + ".json"
    old_json_path = os.path.join(old_dir, old_json)
    new_json_path = os.path.join(new_dir, new_json)
    if os.path.exists(old_json_path):
        if os.path.exists(new_json_path):
            os.remove(new_json_path)
        os.rename(old_json_path, new_json_path)

def parse_floorplan_request(user_text: str) -> dict:
    system_content = """
You are a helpful AI that extracts floorplan specs from the user's text.
They might mention:
 - number of bedrooms (1..3)
 - number of washrooms (1..3)
 - garage or not
 - attached washrooms or not
If not stated, defaults: bedrooms=2, washrooms=1, has_garage=false, attached=false.
Return strictly valid JSON with 4 keys: bedrooms, washrooms, has_garage, has_attachedwashroom.
No extra text, just JSON.
"""
    user_prompt = f"User request:\n{user_text}\nExtract the parameters in JSON only."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0
        )
        content = response["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
    except Exception as e:
        print("OpenAI API error, using defaults:", e)
        data = {
            "bedrooms": 2,
            "washrooms": 1,
            "has_garage": False,
            "has_attachedwashroom": False
        }
    # Ensure valid values
    bedrooms = data.get("bedrooms", 2)
    if bedrooms not in [1, 2, 3]:
        bedrooms = 2
    washrooms = data.get("washrooms", 1)
    if washrooms not in [1, 2, 3]:
        washrooms = 1
    has_garage = bool(data.get("has_garage", False))
    has_attachedwashroom = bool(data.get("has_attachedwashroom", False))
    text_lower = user_text.lower()
    bed_match = re.search(r'(\d+)\s+bedroom', text_lower)
    if bed_match:
        bed_num = int(bed_match.group(1))
        if bed_num in [1, 2, 3]:
            bedrooms = bed_num
    wash_match = re.search(r'(\d+)\s+(?:washroom|bathroom)', text_lower)
    if wash_match:
        wash_num = int(wash_match.group(1))
        if wash_num in [1, 2, 3]:
            washrooms = wash_num
    # Override as per your logic.
    has_garage = True
    if bedrooms >= 2:
        has_attachedwashroom = True
    # Extract language if provided.
    language = "English"
    if "urdu" in text_lower:
        language = "Urdu"
    data["language"] = language
    data["bedrooms"] = bedrooms
    data["washrooms"] = washrooms
    data["has_garage"] = has_garage
    data["has_attachedwashroom"] = has_attachedwashroom
    return data

# ---------------------------
# Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session["description"] = request.form.get("description", "")
        session["language"] = request.form.get("language", "English")
        specs = parse_floorplan_request(session["description"])
        session["specs"] = specs
        # Initialize RL agent and store its Q-table in session.
        rl_agent = FloorplanRLAgent()
        session["rl_agent"] = rl_agent.q_table  # Q-table is a JSON-serializable dict.
        return redirect(url_for("ground_floor"))
    index_html = """
    <html>
      <head><title>Floorplan Generator</title></head>
      <body>
        <h2>Enter your floorplan description</h2>
        <form method="POST">
          <textarea name="description" rows="4" cols="50" placeholder="e.g., Generate a 3 bedroom, 2 washroom house with a garage."></textarea><br>
          <label>Select Language:</label>
          <select name="language">
            <option value="English">English</option>
            <option value="Urdu">Urdu</option>
          </select><br><br>
          <input type="submit" value="Generate Ground Floor Plans">
        </form>
      </body>
    </html>
    """
    return render_template_string(index_html)

@app.route("/ground_floor")
def ground_floor():
    specs = session.get("specs", {"bedrooms": 2, "washrooms": 1, "has_garage": True, "has_attachedwashroom": True, "language": "English"})
    language = specs.get("language", "English")
    # The ground-floor images will be generated by the backend and saved in the "pretty" folder.
    # For this route, we list the generated plans from the "pretty" folder.
    pretty_dir = "pretty"
    image_list = sorted([f for f in os.listdir(pretty_dir) if f.lower().endswith(".png")])[:3]
    html = """
    <html>
      <head>
        <title>Ground Floor Plans</title>
        <style>
          .img-select { border: 2px solid transparent; cursor: pointer; }
          .img-selected { border: 2px solid brown; }
        </style>
        <script>
          function selectImage(imgName) {
              var imgs = document.getElementsByClassName("img-select");
              for(var i=0; i<imgs.length; i++){
                  imgs[i].classList.remove("img-selected");
              }
              document.getElementById(imgName).classList.add("img-selected");
              document.getElementById("selected_image").value = imgName;
          }
        </script>
      </head>
      <body>
        <h2>Ground Floor Plans ({{ language }})</h2>
        {% for img in images %}
          <img id="{{ img }}" src="{{ url_for('static', filename='assets/ground/' + img) }}" class="img-select" width="150" onclick="selectImage('{{ img }}')">
        {% endfor %}
        <br><br>
        <form action="{{ url_for('generate_first_floor') }}" method="POST">
          <input type="hidden" id="selected_image" name="selected_image" value="">
          <input type="submit" value="Generate First Floor Plans">
        </form>
      </body>
    </html>
    """
    return render_template_string(html, images=image_list, language=language)

@app.route("/generate_first_floor", methods=["POST"])
def generate_first_floor():
    selected_ground = request.form.get("selected_image", "")
    if not selected_ground:
        return "Please select a ground-floor plan.", 400
    session["selected_ground"] = selected_ground
    chosen_json = os.path.join("pretty", f"{selected_ground[:-4]}.json")
    if not os.path.exists(chosen_json):
        return f"ERROR: Missing {chosen_json}", 400
    with open(chosen_json, "r") as f:
        chosen_floorplan_dict = json.load(f)
    floor1_dir = "output_floor1"
    if os.path.exists(floor1_dir):
        for oldf in os.listdir(floor1_dir):
            if oldf.lower().endswith((".png", ".json")):
                os.remove(os.path.join(floor1_dir, oldf))
    else:
        os.makedirs(floor1_dir, exist_ok=True)
    approach_list = [1, 2, 3]
    visualizer = FloorplanVisualizer()
    ff_width = FloorplanGenerator.FLOORPLAN_WIDTH
    ff_height = FloorplanGenerator.FLOORPLAN_HEIGHT
    for app_idx in approach_list:
        # Import FirstFloorPlanGenerator from backend.
        from backend.first_floor_plan_generator import FirstFloorPlanGenerator
        ff_gen = FirstFloorPlanGenerator(chosen_floorplan_dict, ff_width, ff_height)
        first_floor_dict = ff_gen.generate_first_floor_plan(approach=app_idx)
        outbase = f"first_floor_plan_{selected_ground[:-4]}_approach{app_idx}"
        out_png = os.path.join(floor1_dir, outbase + ".png")
        out_json = os.path.join(floor1_dir, outbase + ".json")
        visualizer.plot_with_boundaries(
            first_floor_dict,
            out_png,
            ff_width,
            ff_height,
            "English"  # Using English for first-floor visualization by default.
        )
        with open(out_json, "w") as jf:
            json.dump(first_floor_dict, jf)
    return redirect(url_for("first_floor"))

@app.route("/first_floor")
def first_floor():
    floor1_dir = "output_floor1"
    images = sorted([f for f in os.listdir(floor1_dir) if f.lower().endswith(".png")])
    html = """
    <html>
      <head>
        <title>First Floor Plans</title>
        <style>
          .img-select { border: 2px solid transparent; cursor: pointer; }
          .img-selected { border: 2px solid brown; }
        </style>
        <script>
          function selectImage(imgName) {
              var imgs = document.getElementsByClassName("img-select");
              for(var i=0; i<imgs.length; i++){
                  imgs[i].classList.remove("img-selected");
              }
              document.getElementById(imgName).classList.add("img-selected");
              document.getElementById("selected_first").value = imgName;
          }
        </script>
      </head>
      <body>
        <h2>First Floor Plans</h2>
        {% for img in images %}
          <img id="{{ img }}" src="{{ url_for('static', filename='assets/first/' + img) }}" class="img-select" width="150" onclick="selectImage('{{ img }}')">
        {% endfor %}
        <br><br>
        <form action="{{ url_for('summary') }}" method="POST">
          <input type="hidden" id="selected_first" name="selected_first" value="">
          <input type="submit" value="Finalize Selection">
        </form>
      </body>
    </html>
    """
    return render_template_string(html, images=images)

@app.route("/summary", methods=["POST"])
def summary():
    selected_first = request.form.get("selected_first", "")
    selected_ground = session.get("selected_ground", "")
    if not selected_first or not selected_ground:
        return "Selection incomplete.", 400
    summary_html = """
    <html>
      <head><title>Summary</title></head>
      <body>
        <h2>Final Summary</h2>
        <div style="border:1px solid black; padding:10px;">
          <h3>Ground Floor</h3>
          <img src="{{ url_for('static', filename='assets/ground/' + ground) }}" width="300">
        </div>
        <div style="border:1px solid black; padding:10px; margin-top:20px;">
          <h3>First Floor</h3>
          <img src="{{ url_for('static', filename='assets/first/' + first) }}" width="300">
        </div>
      </body>
    </html>
    """
    return render_template_string(summary_html, ground=selected_ground, first=selected_first)

@app.route("/feedback", methods=["POST"])
def feedback():
    selected_plan = int(request.form.get("selected_plan", 1))
    q_table = session.get("rl_agent", {})
    rl_agent = FloorplanRLAgent()
    rl_agent.q_table = q_table
    chosen_mutation_rate = rl_agent.get_mutation_rate()
    rl_agent.update(chosen_mutation_rate, reward=1)
    session["rl_agent"] = rl_agent.q_table
    return f"RL Agent updated with your feedback: {rl_agent.q_table}"

# ---------------------------
# Run the app
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
