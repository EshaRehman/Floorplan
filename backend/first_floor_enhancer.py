import os
import cv2
import json
import numpy as np

class FirstFloorEnhancer:
    """
    This class uses the reference first-floor plan (from the 'pretty' folder)
    to detect the stairs region and living room centroid. It then processes all
    first-floor plan images in 'output_floor1' by drawing the same stairs (with
    the same size, shape, and position) and placing a "Living Room" label in the
    inner area. It also updates the corresponding JSON files.
    Additionally, the area outside the black boundary is filled with the background
    color FBF5F1.
    """
    
    def __init__(self, pretty_dir="pretty", first_floor_dir="output_floor1"):
        self.pretty_dir = pretty_dir
        self.first_floor_dir = first_floor_dir
        # The stairs are drawn in this color (BGR)
        self.stairs_color = (200, 100, 200)
        self.stairs_tol = 10  # tolerance for color detection
        self.living_label = "Living Room"
        # Background color for areas outside the floorplan (hex FBF5F1 => BGR: (241,245,251))
        self.porch_color = (241, 245, 251)
    
    def detect_stairs(self, img):
        """
        Detect the stairs region in the provided image by color thresholding.
        Returns a bounding rectangle (x, y, w, h) if found, else None.
        """
        lower = np.array([self.stairs_color[0] - self.stairs_tol,
                          self.stairs_color[1] - self.stairs_tol,
                          self.stairs_color[2] - self.stairs_tol])
        upper = np.array([self.stairs_color[0] + self.stairs_tol,
                          self.stairs_color[1] + self.stairs_tol,
                          self.stairs_color[2] + self.stairs_tol])
        mask = cv2.inRange(img, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < 10:
            return None
        x, y, w, h = cv2.boundingRect(c)
        return (x, y, w, h)
    
    def detect_living_area_centroid(self, img):
        """
        Detect the largest white region (living area) inside the floorplan.
        Returns the centroid (cx, cy) or None if not found.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, bin_img = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        floor_mask = np.zeros_like(gray)
        cv2.drawContours(floor_mask, [largest], -1, 255, -1)
        living_mask = np.zeros_like(gray)
        h, w = gray.shape
        for y in range(h):
            for x in range(w):
                if floor_mask[y, x] == 255:
                    b, g, r = img[y, x]
                    if b >= 240 and g >= 240 and r >= 240:
                        living_mask[y, x] = 255
        living_contours, _ = cv2.findContours(living_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not living_contours:
            return None
        living = max(living_contours, key=cv2.contourArea)
        M = cv2.moments(living)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    
    def _get_floorplan_mask(self, img):
        """
        Computes the floorplan mask from the given image.
        It thresholds the grayscale image and returns the filled mask of the largest contour.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(black_mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
        return mask

    def enhance_first_floor_plans(self, ref_plan_filename):
        """
        Using the reference plan filename (e.g. "plan1.png") from the 'pretty' folder,
        detect the stairs region and living area centroid. Then, for each first-floor
        plan image in self.first_floor_dir, draw the same stairs rectangle and add the
        "Living Room" label. Also, fill the area outside the floorplan with the porch
        color FBF5F1 and update the corresponding JSON file accordingly.
        """
        ref_path = os.path.join(self.pretty_dir, ref_plan_filename)
        ref_img = cv2.imread(ref_path)
        if ref_img is None:
            print(f"Reference image {ref_path} not found.")
            return
        stairs_rect = self.detect_stairs(ref_img)
        living_centroid = self.detect_living_area_centroid(ref_img)
        if stairs_rect is None:
            print("Stairs not detected in the reference image.")
            return
        if living_centroid is None:
            print("Living area not detected in the reference image.")
            return
        print(f"Detected stairs at {stairs_rect} and living room centroid at {living_centroid} in reference.")
        
        # Process each first-floor plan image in first_floor_dir
        for fname in os.listdir(self.first_floor_dir):
            if not fname.lower().endswith(".png"):
                continue
            fp_path = os.path.join(self.first_floor_dir, fname)
            img = cv2.imread(fp_path)
            if img is None:
                continue
            # Draw the stairs rectangle using the same coordinates
            x, y, w, h = stairs_rect
            cv2.rectangle(img, (x, y), (x + w, y + h), self.stairs_color, -1)
            label_x = x + (w // 2) - 10
            label_y = y + (h // 2)
            cv2.putText(img, "Stairs", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            # Place the "Living Room" label at the detected centroid
            cv2.putText(img, self.living_label, (living_centroid[0] - 20, living_centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
            
            # ----- New Code: Fill area outside floorplan with porch color -----
            mask = self._get_floorplan_mask(img)
            if mask is not None:
                porch_mask = cv2.bitwise_not(mask)
                img[porch_mask == 255] = self.porch_color
            # ----- End New Code -----
            
            cv2.imwrite(fp_path, img)
            print(f"Enhanced {fp_path}")
            # Update JSON file with stairs info
            base = os.path.splitext(fname)[0]
            json_path = os.path.join(self.first_floor_dir, base + ".json")
            if os.path.exists(json_path):
                with open(json_path, "r") as jf:
                    floor_dict = json.load(jf)
                floor_dict["Stairs"] = {"x": x, "y": y, "width": w, "height": h}
                with open(json_path, "w") as jf:
                    json.dump(floor_dict, jf)
                print(f"Updated JSON {json_path}")
