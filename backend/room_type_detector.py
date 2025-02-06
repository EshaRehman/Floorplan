import os
import cv2
import numpy as np

class RoomTypeDetector:
    """
    1) Only selects images that have exactly 1 living room (strictly-white region
       inside the largest black boundary).
    2) Ensures at least 8 final images if possible. If fewer than 8 total images
       are available, it just saves whatever it can.
    """

    def __init__(self, input_dir="output", output_dir="finaloutput"):
        """
        :param input_dir:  Folder where raw floorplan images are found
        :param output_dir: Folder where final chosen floorplans are saved
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Strict near-white threshold for living-room detection
        self.living_room_lower = np.array([240, 240, 240], dtype=np.uint8)
        self.living_room_upper = np.array([255, 255, 255], dtype=np.uint8)

    def detect_and_label_images(self):
        """
        Main pipeline:
          1) Collect living-room info for each image (largest black boundary => find white).
          2) Separate images with exactly 1 LR from others.
          3) If we have >=8 in the 1-LR list, pick up to 10 from them.
             Otherwise, we take all from 1-LR and fill with 'other' images to reach 8 (if possible).
          4) Label the living room in those that have exactly 1 LR, then save everything in output_dir.
        """
        all_data = self._collect_living_room_info()
        if not all_data:
            print("No images found in input directory.")
            return

        # Step B: separate images that have exactly 1 LR from those that do not
        one_lr_list = [x for x in all_data if len(x[2]) == 1]
        other_list  = [x for x in all_data if len(x[2]) != 1]

        # Step C: if we have >=8 in one_lr_list, pick up to 10
        if len(one_lr_list) >= 6:
            final_plans = one_lr_list[:10]  # e.g. 8..10
        else:
            # fewer than 8 => take them all, fill from 'other_list'
            final_plans = list(one_lr_list)
            needed = 6 - len(final_plans)
            if needed > 0:
                final_plans.extend(other_list[:needed])

        if not final_plans:
            print("No final plans chosen. Nothing saved.")
            return

        # Step D: if final <8, we just do what we can
        if len(final_plans) < 6:
            print(f"Warning: only {len(final_plans)} floorplans in total (need >=8).")

        # Clear old files
        for f in os.listdir(self.output_dir):
            if f.lower().endswith((".png",".jpg",".jpeg")):
                os.remove(os.path.join(self.output_dir, f))

        # Step E: label + save
        for (filename, img, centroids) in final_plans:
            if len(centroids) == 1:  # label the 1 LR
                (cx, cy) = centroids[0]
                cv2.putText(
                    img,
                    "Living Room",
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0,0,0),
                    1,
                    cv2.LINE_AA
                )

            out_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(out_path, img)

    def _collect_living_room_info(self):
        """
        Return [(filename, image, [(cx,cy),...])] for each floorplan.
        """
        image_data_list = []
        for fname in os.listdir(self.input_dir):
            if not fname.lower().endswith((".png",".jpg",".jpeg")):
                continue
            path = os.path.join(self.input_dir, fname)
            img = cv2.imread(path)
            if img is None:
                continue

            living_rooms = self._find_living_rooms(img)
            image_data_list.append((fname, img, living_rooms))

        return image_data_list

    def _find_living_rooms(self, image):
        """
        1) Find largest black boundary => floorplan_mask
        2) Inside that => find strictly-white => living rooms
        3) Return centroid list
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # black => invert => external contour
        _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []

        # largest boundary => fill mask
        largest = max(cnts, key=cv2.contourArea)
        floorplan_mask = np.zeros_like(black_mask)
        cv2.drawContours(floorplan_mask, [largest], -1, 255, -1)

        # strictly white => inRange => living
        living_mask = cv2.inRange(image, self.living_room_lower, self.living_room_upper)
        # bitwise-and with floorplan_mask
        living_mask = cv2.bitwise_and(living_mask, floorplan_mask)

        # find living-room contours
        living_cnts, _ = cv2.findContours(living_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for c in living_cnts:
            area = cv2.contourArea(c)
            if area < 10:
                continue
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                centroids.append((cx, cy))

        return centroids
