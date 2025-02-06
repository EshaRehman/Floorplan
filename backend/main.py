# main.py

import os
import json
import shutil

from floorplan_generator import FloorplanGenerator
from floorplan_visualizer import FloorplanVisualizer
from room_type_detector import RoomTypeDetector
from perfect_plan_selector import PerfectPlanSelector
from pretty_floorplan_maker import PrettyFloorplanMaker

# Our new 1st-floor generator with 3 approaches
from first_floor_plan_generator import FirstFloorPlanGenerator

def copy_json_for_png(src_png, src_dir, dst_dir):
    """
    If 'src_png' = e.g. floorplan_37.png, look for floorplan_37.json in src_dir.
    If it exists, copy it to dst_dir. We keep the same base name.
    """
    base = os.path.splitext(src_png)[0]  # e.g. "floorplan_37"
    src_json_path = os.path.join(src_dir, base + ".json")
    if os.path.isfile(src_json_path):
        shutil.copy2(src_json_path, dst_dir)

def rename_png_and_json(old_png, old_dir, new_png, new_dir):
    """
    Renames old_png -> new_png in old_dir/new_dir.
    Then renames old_png.json -> new_png.json if present.
    If new files exist, remove them first (avoid WinError 183).
    """
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


if __name__ == "__main__":

    # ------------------------------------------------------------
    # 1) Ask user for floorplan specs
    # ------------------------------------------------------------
    while True:
        try:
            bedrooms = int(input("How many bedrooms do you want? (1, 2, or 3): "))
            if bedrooms not in [1, 2, 3]:
                raise ValueError
            break
        except ValueError:
            print("Invalid input. Please enter 1, 2, or 3.")

    washrooms = 1
    if bedrooms == 2:
        while True:
            try:
                washrooms = int(input("How many washrooms do you want? (1 or 2): "))
                if washrooms not in [1, 2]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter 1 or 2.")
    elif bedrooms == 3:
        while True:
            try:
                washrooms = int(input("How many washrooms do you want? (1, 2, or 3): "))
                if washrooms not in [1, 2, 3]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid input. Please enter 1, 2, or 3.")

    wants_garage = input("Do you want a garage? (y/n): ").strip().lower()
    has_garage = (wants_garage == 'y')

    wants_attachedwashroom = input("Do you want attached washrooms? (y/n): ").strip().lower()
    has_attachedwashroom = (wants_attachedwashroom == 'y')

    # ------------------------------------------------------------
    # 2) Build the list of rooms
    # ------------------------------------------------------------
    rooms = []
    for b in range(1, bedrooms + 1):
        rooms.append(f"Bedroom_{b}")
    for w in range(1, washrooms + 1):
        rooms.append(f"Washroom_{w}")
    rooms.append("Kitchen")
    if has_garage:
        rooms.append("Garage")

    # ------------------------------------------------------------
    # 3) Generate & visualize many floorplans (PNG + JSON)
    # ------------------------------------------------------------
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    generator = FloorplanGenerator(rooms=rooms, attached_washroom=has_attachedwashroom)
    visualizer = FloorplanVisualizer()

    num_floorplans = 40
    for i in range(num_floorplans):
        fp_dict = generator.genetic_algorithm()
        base_name = f"floorplan_{i+1}"
        png_path = os.path.join(output_dir, base_name + ".png")
        json_path = os.path.join(output_dir, base_name + ".json")

        # Save PNG
        visualizer.plot_with_boundaries(
            fp_dict,
            png_path,
            FloorplanGenerator.FLOORPLAN_WIDTH,
            FloorplanGenerator.FLOORPLAN_HEIGHT
        )
        # Save JSON
        with open(json_path, "w") as jf:
            json.dump(fp_dict, jf)

    # ------------------------------------------------------------
    # 4) Detect & label living rooms -> finaloutput
    # ------------------------------------------------------------
    detector = RoomTypeDetector(input_dir=output_dir, output_dir="finaloutput")
    detector.detect_and_label_images()

    # Copy matching JSON for each PNG in finaloutput
    final_out_pngs = [f for f in os.listdir("finaloutput") if f.lower().endswith(".png")]
    for pngf in final_out_pngs:
        copy_json_for_png(pngf, output_dir, "finaloutput")

    # ------------------------------------------------------------
    # 5) PerfectPlanSelector -> picks 3 => 'perfect'
    # ------------------------------------------------------------
    selector = PerfectPlanSelector(input_dir="finaloutput", output_dir="perfect")
    selector.select_connected_plans()

    # Copy JSON for those 3 perfect images
    perfect_pngs = [f for f in os.listdir("perfect") if f.lower().endswith(".png")]
    for pngf in perfect_pngs:
        copy_json_for_png(pngf, "finaloutput", "perfect")

    # ------------------------------------------------------------
    # 6) Make them pretty -> 'pretty'
    # ------------------------------------------------------------
    maker = PrettyFloorplanMaker(input_dir="perfect", output_dir="pretty")
    maker.make_pretty_floorplans()

    # Copy JSON for the final pretty images
    pretty_pngs = [f for f in os.listdir("pretty") if f.lower().endswith(".png")]
    for pngf in pretty_pngs:
        copy_json_for_png(pngf, "perfect", "pretty")

    # Now presumably we have 3 final PNG + JSON in 'pretty'.
    # Rename them to plan1, plan2, plan3
    pretty_pngs = sorted(
        f for f in os.listdir("pretty")
        if f.lower().endswith(".png")
    )
    for i, old_png in enumerate(pretty_pngs[:3], start=1):
        new_png = f"plan{i}.png"
        rename_png_and_json(old_png, "pretty", new_png, "pretty")

    print("\nFinal 3 plans in 'pretty' are now plan1.png/json, plan2.png/json, plan3.png/json.\n")

    # ------------------------------------------------------------
    # Ask user which plan => load JSON => generate 3 approaches
    # ------------------------------------------------------------
    ans = input("Do you want to generate the 1st-floor plan from plan1, plan2, or plan3? (y/n): ").strip().lower()
    if ans == 'y':
        while True:
            try:
                choice = int(input("Pick which plan (1, 2, or 3): "))
                if choice not in [1,2,3]:
                    raise ValueError
                break
            except ValueError:
                print("Invalid choice. Must be 1, 2, or 3.")

        chosen_json = os.path.join("pretty", f"plan{choice}.json")
        if not os.path.exists(chosen_json):
            print(f"ERROR: Missing {chosen_json}, can't proceed.")
        else:
            with open(chosen_json, "r") as f:
                chosen_floorplan_dict = json.load(f)

            # Clear old files in output_floor1
            floor1_dir = "output_floor1"
            if os.path.exists(floor1_dir):
                for oldf in os.listdir(floor1_dir):
                    if oldf.lower().endswith(".png") or oldf.lower().endswith(".json"):
                        os.remove(os.path.join(floor1_dir, oldf))
            else:
                os.makedirs(floor1_dir, exist_ok=True)

            # We want 3 FIRST-FLOOR IMAGES, each with a DIFFERENT approach (1,2,3).
            approach_list = [1, 2, 3]
            for app_idx in approach_list:
                ff_gen = FirstFloorPlanGenerator(
                    chosen_floorplan_dict,
                    FloorplanGenerator.FLOORPLAN_WIDTH,
                    FloorplanGenerator.FLOORPLAN_HEIGHT
                )
                # Generate a distinct approach
                first_floor_dict = ff_gen.generate_first_floor_plan(approach=app_idx)

                # Save them as e.g. first_floor_plan_2_approach1.png/.json
                outbase = f"first_floor_plan_{choice}_approach{app_idx}"
                out_png = os.path.join(floor1_dir, outbase + ".png")
                out_json = os.path.join(floor1_dir, outbase + ".json")

                visualizer.plot_with_boundaries(
                    first_floor_dict,
                    out_png,
                    FloorplanGenerator.FLOORPLAN_WIDTH,
                    FloorplanGenerator.FLOORPLAN_HEIGHT
                )
                with open(out_json, "w") as jf:
                    json.dump(first_floor_dict, jf)

            print(f"\nSaved 3 distinct first-floor plans (Approach #1, #2, #3) in '{floor1_dir}'.\n")

    print("All done!")
