import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
import random
import matplotlib
matplotlib.use('Agg')

class FloorplanVisualizer:
    @staticmethod
    def plot_with_boundaries(floorplan, save_path, width, height):
        """
        1) Draw each room as a rectangle with a distinct color
        2) Compute union => black boundary
        3) living_area => bounding box minus union
        4) Place exactly 1 door per room:
           - If bedroom/kitchen/garage => door on shared boundary with living room
           - If washroom => if it shares boundary with a study room or a bedroom => place door there,
             else place door on shared boundary with living room.
        The door is a small rectangle oriented with the wall.

        (We've added "Stairs":"plum" so if the dictionary has Stairs, it draws them.)
        """

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        ax.set_aspect("equal")
        ax.axis("off")

        # Define colors (added "Stairs": "plum")
        colors = {
                    "Garage":   "#C5D5E4",  # Soft pastel blue-gray for a calm and neutral garage
                    "Kitchen":  "#FFEDCC",  # Warm pastel peach (friendly and inviting kitchen)
                    "Bedroom":  "#FFB3C6",  # Soft pastel pink for a cozy and restful bedroom
                    "Washroom": "#C4E3CB",  # Mint pastel green for a fresh and clean feel
                    "Storage":  "#E9C9FF",  # Light lavender pastel for an elegant storage space
                    "Balcony":  "#FFF4B1",  # Warm pastel yellow for an open and sunny feel
                    "Study":    "#F6C5B6",  # Pastel coral peach to create a warm, productive atmosphere
                    "Door":     "#5C4033",  # Pastel purple for a soft, charming door presence
                    "Stairs":   "#A4D8C2"   # Mint-teal pastel for a refreshing yet distinct look
                }

        # --------------------------------------------------
        # 1) Create shapely polygons for each room
        # --------------------------------------------------
        room_polygons = {}
        for room_name, rect in floorplan.items():
            x, y = rect["x"], rect["y"]
            w, h = rect["width"], rect["height"]
            poly = Polygon([
                (x,     y),
                (x + w, y),
                (x + w, y + h),
                (x,     y + h)
            ])
            room_polygons[room_name] = poly

        # --------------------------------------------------
        # 2) Draw each room rectangle + label
        # --------------------------------------------------
        for room_name, poly in room_polygons.items():
            base_type = room_name.split("_")[0]
            color = colors.get(base_type, "gray")

            x_min, y_min, x_max, y_max = poly.bounds
            w = x_max - x_min
            h = y_max - y_min

            ax.add_patch(
                plt.Rectangle(
                    (x_min, y_min), w, h,
                    color=color, alpha=0.7
                )
            )
            # label in center
            cx = x_min + w/2
            cy = y_min + h/2
            ax.text(cx, cy, base_type,
                    color="black", fontsize=8,
                    ha="center", va="center")

        # --------------------------------------------------
        # 3) Union => black boundary
        # --------------------------------------------------
        union_poly = unary_union(list(room_polygons.values()))
        fused_poly = union_poly
        max_buf = 50
        step = 1
        for dist in range(step, max_buf+1, step):
            if (fused_poly.geom_type == "Polygon") and (len(fused_poly.interiors) == 0):
                break
            bigger = fused_poly.buffer(dist, join_style=2)
            fused_poly = bigger.buffer(-dist, join_style=2)

        # Draw black boundary
        if fused_poly.is_empty:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            return

        if fused_poly.geom_type == "Polygon":
            bx, by = fused_poly.exterior.xy
            ax.plot(bx, by, color="black", linewidth=2)
        else:
            for part in fused_poly.geoms:
                if part.geom_type == "Polygon":
                    shell = Polygon(part.exterior)
                    x_sh, y_sh = shell.exterior.xy
                    ax.plot(x_sh, y_sh, color="black", linewidth=2)

        # --------------------------------------------------
        # 4) living_area => bounding rectangle minus union
        # --------------------------------------------------
        bounding_poly = Polygon([(0,0), (width,0), (width,height), (0,height)])
        living_area_poly = bounding_poly.difference(union_poly)

        # --------------------------------------------------
        # 5) Place doors
        # --------------------------------------------------
        for room_name, poly in room_polygons.items():
            base_type = room_name.split("_")[0]

            if base_type == "Washroom":
                door_line = None
                # First try to find shared boundary with a Study room
                for other_name, other_poly in room_polygons.items():
                    if "Study" in other_name:
                        shared_line = poly.boundary.intersection(other_poly.boundary)
                        if not shared_line.is_empty and shared_line.length > 0.1:
                            door_line = shared_line
                            break
                # If no Study adjacent, try Bedroom adjacency
                if not door_line:
                    for other_name, other_poly in room_polygons.items():
                        if "Bedroom" in other_name:
                            shared_line = poly.boundary.intersection(other_poly.boundary)
                            if not shared_line.is_empty and shared_line.length > 0.1:
                                door_line = shared_line
                                break
                # If still not found, use living area boundary
                if not door_line:
                    door_line = poly.boundary.intersection(living_area_poly.boundary)
            else:
                # For other room types, door on adjacency with living area
                door_line = poly.boundary.intersection(living_area_poly.boundary)

            if door_line and not door_line.is_empty:
                valid_door_line = None
                if fused_poly.geom_type == "Polygon":
                    valid_door_line = door_line.difference(fused_poly.exterior)
                elif fused_poly.geom_type == "MultiPolygon":
                    for part in fused_poly.geoms:
                        if part.geom_type == "Polygon":
                            valid_door_line = door_line.difference(part.exterior)
                            if not valid_door_line.is_empty:
                                break

                if valid_door_line and not valid_door_line.is_empty:
                    door_rect = FloorplanVisualizer._construct_door_rectangle(valid_door_line)
                    if door_rect is not None:
                        x_coords, y_coords = door_rect.exterior.xy
                        ax.add_patch(
                            plt.Polygon(
                                list(zip(x_coords, y_coords)),
                                color=colors["Door"]
                            )
                        )

        # Save
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    @staticmethod
    def _construct_door_rectangle(door_line, door_len=0.4, door_thick=0.1):
        """
        door_line can be a LineString or MultiLineString.
        1) Pick the longest line segment
        2) Find its midpoint param
        3) Build a small rectangle around that midpoint, oriented with the line direction.
        Returns a shapely Polygon or None.
        """
        if door_line.is_empty:
            return None

        if door_line.geom_type == "LineString":
            line = door_line
        elif door_line.geom_type == "MultiLineString":
            max_len = 0
            best_line = None
            for l in door_line.geoms:
                if l.length > max_len:
                    max_len = l.length
                    best_line = l
            if not best_line or best_line.length < 0.01:
                return None
            line = best_line
        else:
            return None

        if line.length < 0.01:
            return None

        mid_param = line.length / 2
        mid_pt = line.interpolate(mid_param)

        epsilon = 0.01
        p1 = line.interpolate(mid_param - epsilon)
        p2 = line.interpolate(mid_param + epsilon)
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        line_len = (dx*dx + dy*dy)**0.5
        if line_len < 1e-6:
            return None

        nx = dx / line_len
        ny = dy / line_len

        left_nx = -ny
        left_ny = nx

        halfL = door_len * 0.5
        halfT = door_thick * 0.5

        mx, my = mid_pt.x, mid_pt.y

        c1x = mx - halfL*nx + halfT*left_nx
        c1y = my - halfL*ny + halfT*left_ny

        c2x = mx + halfL*nx + halfT*left_nx
        c2y = my + halfL*ny + halfT*left_ny

        c3x = mx + halfL*nx - halfT*left_nx
        c3y = my + halfL*ny - halfT*left_nx

        c4x = mx - halfL*nx - halfT*left_ny
        c4y = my - halfL*ny - halfT*left_ny

        return Polygon([(c1x,c1y), (c2x,c2y), (c3x,c3y), (c4x,c4y)])

    @staticmethod
    def _pick_door_location(geom):
        """
        Old approach: pick a single point (circle).
        Not used now, but kept for reference.
        """
        if geom.is_empty:
            return None

        if geom.geom_type == "LineString":
            line = geom
        elif geom.geom_type == "MultiLineString":
            max_len = 0
            best_line = None
            for l in geom.geoms:
                if l.length>0:
                    if l.length>max_len:
                        max_len = l.length
                        best_line = l
            if best_line is None or best_line.length<0.01:
                return None
            line = best_line
        else:
            return None

        if line.length<0.01:
            return None

        midparam = line.length/2
        return line.interpolate(midparam)
