# first_floor_plan_generator.py

import random
import cv2
import numpy as np

class FirstFloorPlanGenerator:
    """
    3 Approaches:
      #1) Garage->Storage; if >=2 bedrooms => rename one bedroom to Study; no balconies
      #2) Garage->Study; exactly 1 bedroom => carve out balcony
      #3) Garage->Storage, pick 1 bedroom=>Study, pick another=>Balcony

    After generating the plan via one of the approaches, the "Stairs" (if present)
    from the selected pretty floorplan are forced into the result.
    """

    def __init__(self, base_floorplan, floor_width, floor_height, source_image_path=None):
        """
        :param base_floorplan: dict with {room_name: {x, y, width, height}},
                               possibly including a "Stairs" key.
        :param floor_width: Overall floor width.
        :param floor_height: Overall floor height.
        :param source_image_path: Optional; if provided and if base_floorplan lacks "Stairs",
                                  this PNG will be loaded to extract the stairs rectangle.
        """
        self.floor_width = floor_width
        self.floor_height = floor_height

        # First, try to get stairs from the JSON dictionary.
        if "Stairs" in base_floorplan:
            self.base_stairs = base_floorplan["Stairs"].copy()
        else:
            self.base_stairs = None

        # Remove "Stairs" from the base floorplan (if present) so room-generation logic
        # works solely on room definitions.
        self.base_floorplan = {k: v for k, v in base_floorplan.items() if k != "Stairs"}

        # If stairs were not found in the JSON and a source image is provided,
        # attempt to extract the stairs rectangle from the image.
        if self.base_stairs is None and source_image_path:
            self.base_stairs = self._extract_stairs_from_image(source_image_path)

    def _extract_stairs_from_image(self, image_path):
        """
        Load the image from the provided path and try to locate the stairs rectangle.
        The stairs are drawn using the color (200, 100, 200) in BGR.
        A tolerance of ±10 is applied.
        Returns a dict {"x": x, "y": y, "width": w, "height": h} if found;
        otherwise, returns None.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Define lower and upper bounds for the stairs color.
        lower = np.array([200 - 10, 100 - 10, 200 - 10], dtype=np.uint8)
        upper = np.array([200 + 10, 100 + 10, 200 + 10], dtype=np.uint8)
        mask = cv2.inRange(img, lower, upper)
        # Find contours in the mask.
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        # Assume the largest contour corresponds to the stairs.
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 10:  # too small to be valid
            return None
        x, y, w, h = cv2.boundingRect(c)
        return {"x": x, "y": y, "width": w, "height": h}

    def generate_first_floor_plan(self, approach=None):
        """
        If approach is None => pick random from [1,2,3].
        Then, regardless of the approach used, force the "Stairs" to be exactly
        the same as in the selected pretty floorplan.
        """
        if approach is None:
            approach = random.choice([1, 2, 3])

        if approach == 1:
            plan = self._approach1()
        elif approach == 2:
            plan = self._approach2()
        else:
            plan = self._approach3()

        # Force the stairs into the generated plan using the preserved stairs data.
        if self.base_stairs is not None:
            plan["Stairs"] = self.base_stairs.copy()

        return plan

    def _approach1(self):
        plan = {}
        for rn, rect in self.base_floorplan.items():
            new_rect = dict(rect)
            if "Garage" in rn:
                new_name = rn.replace("Garage", "Storage")
                plan[new_name] = new_rect
            else:
                plan[rn] = new_rect

        # If there are at least 2 bedrooms, rename one to "Study"
        bedrooms = [k for k in plan if "Bedroom" in k]
        if len(bedrooms) >= 2:
            chosen_bed = random.choice(bedrooms)
            new_study = self._rename_key(chosen_bed, "Study")
            plan[new_study] = plan.pop(chosen_bed)
        return plan

    def _approach2(self):
        plan = {}
        for rn, rect in self.base_floorplan.items():
            new_rect = dict(rect)
            if "Garage" in rn:
                new_name = rn.replace("Garage", "Study")
                plan[new_name] = new_rect
            else:
                plan[rn] = new_rect

        # If there is at least one bedroom, carve out a balcony from one.
        bedrooms = [b for b in plan if "Bedroom" in b]
        if bedrooms:
            chosen = random.choice(bedrooms)
            self._carve_balcony_if_on_boundary(plan, chosen)
        return plan

    def _approach3(self):
        plan = {}
        for rn, rect in self.base_floorplan.items():
            new_rect = dict(rect)
            if "Garage" in rn:
                new_name = rn.replace("Garage", "Storage")
                plan[new_name] = new_rect
            else:
                plan[rn] = new_rect

        bedrooms = [b for b in plan if "Bedroom" in b]
        if bedrooms:
            chosen_study = random.choice(bedrooms)
            new_study = self._rename_key(chosen_study, "Study")
            plan[new_study] = plan.pop(chosen_study)
            bedrooms.remove(chosen_study)

            if bedrooms:
                chosen_balcony = random.choice(bedrooms)
                self._carve_balcony_if_on_boundary(plan, chosen_balcony)
        return plan

    def _rename_key(self, old_key, new_base):
        parts = old_key.split("_", 1)
        if len(parts) == 2:
            return new_base + "_" + parts[1]
        else:
            return new_base

    def _carve_balcony_if_on_boundary(self, plan, room_name, thickness=1):
        if room_name not in plan:
            return
        rect = plan[room_name]
        x, y = rect["x"], rect["y"]
        w, h = rect["width"], rect["height"]

        boundary_sides = []
        if x == 0:
            boundary_sides.append("left")
        if x + w == self.floor_width:
            boundary_sides.append("right")
        if y == 0:
            boundary_sides.append("bottom")
        if y + h == self.floor_height:
            boundary_sides.append("top")

        if not boundary_sides:
            return

        side = random.choice(boundary_sides)
        new_balc_name = room_name.replace("Bedroom", "Balcony").replace("Study", "Balcony")
        if new_balc_name == room_name:
            new_balc_name = "Balcony_" + room_name

        if side == "left" and w > thickness:
            balc = {"x": x, "y": y, "width": thickness, "height": h}
            rect["x"] += thickness
            rect["width"] -= thickness
            plan[new_balc_name] = balc
        elif side == "right" and w > thickness:
            balc = {"x": x + w - thickness, "y": y, "width": thickness, "height": h}
            rect["width"] -= thickness
            plan[new_balc_name] = balc
        elif side == "bottom" and h > thickness:
            balc = {"x": x, "y": y, "width": w, "height": thickness}
            rect["y"] += thickness
            rect["height"] -= thickness
            plan[new_balc_name] = balc
        elif side == "top" and h > thickness:
            balc = {"x": x, "y": y + h - thickness, "width": w, "height": thickness}
            rect["height"] -= thickness
            plan[new_balc_name] = balc
