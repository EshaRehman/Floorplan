import os
import cv2
import numpy as np

class PerfectPlanSelector:
    def __init__(
        self,
        input_dir="finaloutput",
        output_dir="perfect",
        min_contour_area=200
    ):
        """
        :param input_dir:  Folder where labeled floorplans are found.
        :param output_dir: Folder to store the final chosen images.
        :param min_contour_area: Any contour below this is ignored when checking 'connected'.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Minimum contour area to consider a valid shape for connectivity
        self.min_contour_area = min_contour_area

    def select_connected_plans(self):
        """
        1) Gathers all images in self.input_dir.
        2) For each image, checks if it's connected (exactly one large black boundary).
        3) Among all connected images, measures each one's living-room area.
        4) Sorts them by living-room area (descending). Picks top 3.
        5) Copies those top 3 to self.output_dir.
        """
        # Remove old images in output_dir
        for f in os.listdir(self.output_dir):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(self.output_dir, f))

        # 1) Gather images
        image_files = [
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not image_files:
            print("No images found in input directory.")
            return

        # 2) Check connectivity + measure living-room area
        connected_plans = []
        for fname in image_files:
            path = os.path.join(self.input_dir, fname)
            image = cv2.imread(path)
            if image is None:
                continue

            if self.is_connected(image):
                # measure living room area
                living_area = self.get_living_room_area(image)
                if living_area is not None:
                    connected_plans.append((fname, living_area))
                else:
                    # If for some reason we can't measure living area, treat as 0
                    connected_plans.append((fname, 0))

        if not connected_plans:
            print("No connected floorplans found.")
            return

        # 3) Sort by living-room area, descending
        connected_plans.sort(key=lambda x: x[1], reverse=True)

        # 4) Pick top 3
        selected = connected_plans[:3]

        # 5) Copy them to output_dir
        for (filename, area) in selected:
            src_path = os.path.join(self.input_dir, filename)
            dst_path = os.path.join(self.output_dir, filename)
            img = cv2.imread(src_path)
            if img is not None:
                cv2.imwrite(dst_path, img)
                print(f"Copied '{filename}' to '{self.output_dir}' (living_area={area}).")

    def is_connected(self, image):
        """
        Return True if there's exactly 1 large contour => connected.
        1) Convert to gray
        2) Invert threshold: black => white
        3) Count how many large contours exist
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binimg = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(binimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Count how many have area > self.min_contour_area
        large = [c for c in cnts if cv2.contourArea(c) > self.min_contour_area]
        return (len(large) == 1)

    def get_living_room_area(self, image):
        """
        Measure the largest white region inside the black boundary = "living-room" area.

        Steps:
          1) Get the largest black boundary => mask the interior
          2) Among the interior, consider any pixel with b>=240,g>=240,r>=240 => white => living
          3) Find external contours => largest is living area => return its area
        """
        h, w = image.shape[:2]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # black => invert => largest boundary
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return 0

        # pick largest black boundary
        largest = max(cnts, key=cv2.contourArea)
        floorplan_mask = np.zeros_like(black_mask)
        cv2.drawContours(floorplan_mask, [largest], -1, 255, -1)

        # build living_mask => strictly white inside floorplan
        living_mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if floorplan_mask[y, x] == 255:
                    b, g, r = image[y, x]
                    if b>=240 and g>=240 and r>=240:
                        living_mask[y, x] = 255

        # find largest living contour => area
        lcnts, _ = cv2.findContours(living_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not lcnts:
            return 0

        largest_lr = max(lcnts, key=cv2.contourArea)
        return cv2.contourArea(largest_lr)
