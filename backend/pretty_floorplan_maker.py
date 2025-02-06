import os
import cv2
import numpy as np
import json
import random
from math import sqrt

class PrettyFloorplanMaker:
    """
    1) Loads each PNG + JSON from 'perfect' (self.input_dir)
    2) Finds the largest 'white-living-area' in the PNG
    3) Places a 'Stairs' rectangle in that living area
    4) Writes 'Stairs' to the dictionary (floor_dict["Stairs"])
    5) Saves updated image + dictionary in self.output_dir (e.g. 'pretty')
    Additionally, the area outside the black boundary is filled with background color FBF5F1
    and labeled as "Porch".
    """

    def __init__(self, input_dir="perfect", output_dir="pretty"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # The dimension of the stairs rectangle
        self.stairs_w = 15
        self.stairs_h = 15

        # Tolerance for color detection
        self.tol = 8

        # Known BGR references for certain rooms
        self.room_colors = [
            (233, 255, 255),  # washroom
            (237, 227, 197),  # garage
            (177, 243, 177),  # kitchen
            (166, 166, 244)   # bedroom
        ]
        
        # Porch background color (hex FBF5F1 -> BGR: (241, 245, 251))
        self.porch_color = (241, 245, 251)

    def make_pretty_floorplans(self):
        """
        1) Remove old .png/.json in self.output_dir.
        2) For each PNG in self.input_dir, read its JSON, place stairs, and then fill
           and label the area outside the floorplan as "Porch" with background color FBF5F1.
           Save the updated result.
        """
        # Remove old files in output_dir
        for f in os.listdir(self.output_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".json")):
                os.remove(os.path.join(self.output_dir, f))

        # Gather images
        image_files = [f for f in os.listdir(self.input_dir)
                       if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        for fname in image_files:
            path = os.path.join(self.input_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            base = os.path.splitext(fname)[0]
            dict_path = os.path.join(self.input_dir, base + ".json")

            # Load dictionary if present; otherwise, use an empty dict.
            floor_dict = {}
            if os.path.exists(dict_path):
                with open(dict_path, "r") as jf:
                    floor_dict = json.load(jf)

            # Recalculate stairs placement and porch filling/labeling.
            annotated, updated_dict = self._place_stairs_in_image(img, floor_dict)

            # Save updated image and JSON.
            out_img_path = os.path.join(self.output_dir, fname)
            cv2.imwrite(out_img_path, annotated)
            print(f"Saved plan with stairs => {out_img_path}")
            out_json_path = os.path.join(self.output_dir, base + ".json")
            with open(out_json_path, "w") as jf:
                json.dump(updated_dict, jf)

    def _place_stairs_in_image(self, img, floor_dict):
        """
        1) Find the largest living-area contour.
        2) Place a 'Stairs' rectangle and update floor_dict.
        3) Fill the region outside the black boundary with background color FBF5F1.
        4) Then, without using complex centroid computations, pick a porch-colored pixel
           (using cv2.findNonZero) and place the "Porch" label at that location with a small offset.
        5) Return the annotated image and updated dictionary.
        """
        annotated = img.copy()
        h, w = annotated.shape[:2]

        # Get floor mask: invert threshold, select largest contour, fill → floor_mask=255
        floor_mask = self._get_floorplan_mask(annotated)
        if floor_mask is None:
            return annotated, floor_dict

        color_mask = self._build_color_mask(annotated, floor_mask)

        # Determine living area within the floor mask.
        living_mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if floor_mask[y, x] == 255:
                    b, g, r = annotated[y, x]
                    if b >= 240 and g >= 240 and r >= 240:
                        living_mask[y, x] = 255

        cnts, _ = cv2.findContours(living_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return annotated, floor_dict

        largest_lr = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest_lr) < 10:
            return annotated, floor_dict

        # Place stairs using existing logic.
        annotated, floor_dict = self._try_place_stairs(annotated, largest_lr, color_mask, floor_dict)
        
        # ----- New Code for Porch Filling and Labeling -----
        # Fill the area outside the floor (the porch) with the porch background color.
        annotated[cv2.bitwise_not(floor_mask) == 255] = self.porch_color

        # Get all porch pixel coordinates.
        porch_pts = cv2.findNonZero(cv2.bitwise_not(floor_mask))
        if porch_pts is not None:
            # Convert to a list of (x, y) points.
            pts = [tuple(pt[0]) for pt in porch_pts]
            # Filter out points too close to the image boundaries (e.g., within 10 pixels)
            valid_pts = [pt for pt in pts if pt[0] > 10 and pt[0] < (w-10) and pt[1] > 10 and pt[1] < (h-10)]
            # If there are valid points, pick one randomly.
            if valid_pts:
                label_pt = random.choice(valid_pts)
            else:
                # Fallback: use the first available porch point.
                label_pt = pts[0]
            # Apply a small outward offset (e.g., 5 pixels) to ensure it is away from the boundary.
            final_pt = (label_pt[0] + 5, label_pt[1] + 5)
            cv2.putText(annotated, "Porch", (final_pt[0] - 20, final_pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        # ----- End New Code -----

        return annotated, floor_dict

    def _get_floorplan_mask(self, annotated):
        """
        Invert threshold → find largest external contour → fill → return floor mask.
        """
        gray = cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        big_cnts = [c for c in cnts if cv2.contourArea(c) > 2000]
        if not big_cnts:
            return None
        largest = max(big_cnts, key=cv2.contourArea)
        mask = np.zeros_like(black_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask

    def _build_color_mask(self, annotated, floor_mask):
        """
        Create a mask for known room colors.
        """
        h, w = annotated.shape[:2]
        total_mask = np.zeros((h, w), dtype=np.uint8)
        for (bC, gC, rC) in self.room_colors:
            lower = np.array([max(0, bC - self.tol), max(0, gC - self.tol), max(0, rC - self.tol)], dtype=np.uint8)
            upper = np.array([min(255, bC + self.tol), min(255, gC + self.tol), min(255, rC + self.tol)], dtype=np.uint8)
            mask = cv2.inRange(annotated, lower, upper)
            mask = cv2.bitwise_and(mask, floor_mask)
            total_mask = cv2.bitwise_or(total_mask, mask)
        return total_mask

    def _try_place_stairs(self, annotated, living_contour, color_mask, floor_dict):
        """
        Try a radial approach; if that fails, try a free-wall approach.
        """
        h, w = annotated.shape[:2]
        lr_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(lr_mask, [living_contour], -1, 255, -1)
        boundary_pts = living_contour.reshape(-1, 2)
        placed, annotated, floor_dict = self._radial_stairs(annotated, boundary_pts, lr_mask, color_mask, floor_dict)
        if not placed:
            placed2, annotated, floor_dict = self._free_wall_segment(annotated, boundary_pts, lr_mask, color_mask, floor_dict)
        return annotated, floor_dict

    def _radial_stairs(self, annotated, boundary_pts, lr_mask, color_mask, floor_dict):
        import math
        for (x1, y1) in boundary_pts:
            if self._neighbor_color(x1, y1, color_mask):
                continue
            for dist in range(5, 40, 5):
                for angle_deg in range(0, 360, 30):
                    rad = math.radians(angle_deg)
                    bx = int(x1 + dist * math.cos(rad))
                    by = int(y1 + dist * math.sin(rad))
                    if self._can_place_stairs_box(bx, by, annotated, lr_mask, color_mask):
                        annotated, floor_dict = self._draw_stairs(annotated, bx, by, floor_dict)
                        return True, annotated, floor_dict
        return False, annotated, floor_dict

    def _free_wall_segment(self, annotated, boundary_pts, lr_mask, color_mask, floor_dict):
        free_pts = []
        for (x1, y1) in boundary_pts:
            if not self._neighbor_color(x1, y1, color_mask):
                free_pts.append((x1, y1))
        if len(free_pts) < 2:
            return False, annotated, floor_dict

        segments = []
        cur = [free_pts[0]]
        for i in range(1, len(free_pts)):
            prev = free_pts[i - 1]
            this = free_pts[i]
            if self._distance(prev, this) < 2.0:
                cur.append(this)
            else:
                if len(cur) > 1:
                    segments.append(cur)
                cur = [this]
        if len(cur) > 1:
            segments.append(cur)
        if not segments:
            return False, annotated, floor_dict

        max_len = 0
        largest = None
        for seg in segments:
            dist = self._segment_length(seg)
            if dist > max_len:
                max_len = dist
                largest = seg
        if not largest or len(largest) < 2:
            return False, annotated, floor_dict

        seg_len = self._segment_length(largest)
        half = seg_len * 0.5
        dist_so_far = 0
        import math
        mx, my = largest[-1]
        for i in range(len(largest) - 1):
            d = self._distance(largest[i], largest[i + 1])
            if dist_so_far + d >= half:
                remain = half - dist_so_far
                ratio = remain / d
                mx = largest[i][0] + ratio * (largest[i + 1][0] - largest[i][0])
                my = largest[i][1] + ratio * (largest[i + 1][1] - largest[i][1])
                break
            dist_so_far += d

        # Determine local direction
        closest_idx = 0
        closest_d = 999999
        for i, pt in enumerate(largest):
            dd = self._distance(pt, (mx, my))
            if dd < closest_d:
                closest_d = dd
                closest_idx = i
        if closest_idx < len(largest) - 1:
            p0 = largest[closest_idx]
            p1 = largest[closest_idx + 1]
        else:
            p0 = largest[closest_idx - 1]
            p1 = largest[closest_idx]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 1e-3:
            return False, annotated, floor_dict
        nx = dx / length
        ny = dy / length
        lx = -ny
        ly = nx
        offset = 5
        bx = int(mx + offset * lx)
        by = int(my + offset * ly)
        if self._can_place_stairs_box(bx, by, annotated, lr_mask, color_mask):
            annotated, floor_dict = self._draw_stairs(annotated, bx, by, floor_dict)
            return True, annotated, floor_dict
        return False, annotated, floor_dict

    def _neighbor_color(self, x, y, color_mask):
        h, w = color_mask.shape
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if color_mask[ny, nx] == 255:
                    return True
        return False

    def _can_place_stairs_box(self, bx, by, annotated, lr_mask, color_mask):
        h, w = annotated.shape[:2]
        xB = bx + self.stairs_w
        yB = by + self.stairs_h
        if bx < 0 or by < 0 or xB > w or yB > h:
            return False
        sub_lr = lr_mask[by:yB, bx:xB]
        if cv2.countNonZero(sub_lr) < self.stairs_w * self.stairs_h:
            return False
        sub_color = color_mask[by:yB, bx:xB]
        if cv2.countNonZero(sub_color) > 0:
            return False
        return True

    def _draw_stairs(self, annotated, bx, by, floor_dict):
        """
        Draw the stairs on the image and store the stairs rectangle in the dictionary.
        """
        xB = bx + self.stairs_w
        yB = by + self.stairs_h
        cv2.rectangle(annotated, (bx, by), (xB, yB), (200, 100, 200), -1)
        label_x = bx + (self.stairs_w // 2) - 10
        label_y = by + (self.stairs_h // 2)
        cv2.putText(annotated, "Stairs", (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        floor_dict["Stairs"] = {
            "x": bx,
            "y": by,
            "width": self.stairs_w,
            "height": self.stairs_h
        }
        return annotated, floor_dict

    def _distance(self, a, b):
        import math
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _segment_length(self, seg):
        length = 0
        for i in range(len(seg) - 1):
            length += self._distance(seg[i], seg[i + 1])
        return length
