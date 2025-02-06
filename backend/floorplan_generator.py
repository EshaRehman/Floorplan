import random

class FloorplanGenerator:
    ROOMS = ["Garage", "Kitchen", "Bedroom", "Washroom"]
    FLOORPLAN_WIDTH = 20
    FLOORPLAN_HEIGHT = 20
    POPULATION_SIZE = 10
    GENERATIONS = 50
    MUTATION_RATE = 0.1  # Default mutation rate

    def __init__(self, rooms=None, attached_washroom=False, rl_agent=None):
        """
        Initialize the floorplan generator.

        Args:
         - rooms (list): List of rooms to include in the floorplan.
         - attached_washroom (bool): Whether washrooms must be adjacent to bedrooms.
         - rl_agent: (Optional) An RL agent that provides a mutation rate.
        """
        self.rooms = rooms if rooms else self.ROOMS
        self.attached_washroom = attached_washroom
        self.rl_agent = rl_agent

    @staticmethod
    def check_overlap(room1, room2):
        """
        Check if two rooms overlap.
        """
        return not (
            room1["x"] + room1["width"] <= room2["x"] or
            room2["x"] + room2["width"] <= room1["x"] or
            room1["y"] + room1["height"] <= room2["y"] or
            room2["y"] + room2["height"] <= room1["y"]
        )

    @staticmethod
    def check_min_gap(room1, room2, min_gap=3):
        """
        Check if there is at least `min_gap` units of space between two rooms.
        """
        return (
            room1["x"] + room1["width"] + min_gap <= room2["x"] or
            room2["x"] + room2["width"] + min_gap <= room1["x"] or
            room1["y"] + room1["height"] + min_gap <= room2["y"] or
            room2["y"] + room2["height"] + min_gap <= room1["y"]
        )

    @staticmethod
    def is_flush_adjacent(r1, r2):
        """
        Return True if two rectangles share a full vertical or horizontal boundary.
        """
        r1_left,  r1_right  = r1["x"], r1["x"] + r1["width"]
        r1_bottom, r1_top    = r1["y"], r1["y"] + r1["height"]
        r2_left,  r2_right  = r2["x"], r2["x"] + r2["width"]
        r2_bottom, r2_top    = r2["y"], r2["y"] + r2["height"]

        if abs(r1_right - r2_left) < 0.0001:
            if min(r1_top, r2_top) - max(r1_bottom, r2_bottom) > 0:
                return True
        if abs(r2_right - r1_left) < 0.0001:
            if min(r1_top, r2_top) - max(r1_bottom, r2_bottom) > 0:
                return True
        if abs(r1_top - r2_bottom) < 0.0001:
            if min(r1_right, r2_right) - max(r1_left, r2_left) > 0:
                return True
        if abs(r2_top - r1_bottom) < 0.0001:
            if min(r1_right, r2_right) - max(r1_left, r2_left) > 0:
                return True
        return False

    def initialize_population(self):
        population = []
        while len(population) < self.POPULATION_SIZE:
            floorplan = {}
            valid = True
            for room in self.rooms:
                if "Bedroom" in room:
                    width, height = random.randint(4, 5), random.randint(4, 5)
                elif "Kitchen" in room:
                    width, height = random.randint(3, 4), random.randint(3, 4)
                elif "Washroom" in room:
                    width, height = random.randint(2, 3), random.randint(2, 3)
                else:
                    width, height = random.randint(4, 6), random.randint(4, 6)
                placed = False
                for _ in range(50):
                    x = random.randint(0, self.FLOORPLAN_WIDTH - width)
                    y = random.randint(0, self.FLOORPLAN_HEIGHT - height)
                    room_rect = {"x": x, "y": y, "width": width, "height": height}
                    if any(self.check_overlap(room_rect, existing_room)
                           for existing_room in floorplan.values()):
                        continue
                    if "Bedroom" in room:
                        bedroom_conflict = False
                        for existing_name, existing_rect in floorplan.items():
                            if "Bedroom" in existing_name:
                                if self.is_flush_adjacent(room_rect, existing_rect):
                                    bedroom_conflict = True
                                    break
                        if bedroom_conflict:
                            continue
                    if self.attached_washroom and "Washroom" in room:
                        if not self._place_adjacent_to_bedroom(floorplan, room, room_rect):
                            continue
                    elif not ("Bedroom" in room or "Washroom" in room) and any(
                        not self.check_min_gap(room_rect, existing_room, min_gap=3)
                        for existing_room in floorplan.values()):
                        continue
                    floorplan[room] = room_rect
                    placed = True
                    break
                if not placed:
                    valid = False
                    break
            if valid:
                population.append(floorplan)
        return population

    def _place_adjacent_to_bedroom(self, floorplan, washroom_name, room_rect):
        for existing_room_name, existing_rect in floorplan.items():
            if "Bedroom" in existing_room_name:
                if existing_rect.get("has_washroom_attached", False):
                    continue
                adjacency_options = [
                    {"x": existing_rect["x"] - room_rect["width"],
                     "y": existing_rect["y"],
                     "width": room_rect["width"],
                     "height": room_rect["height"]},
                    {"x": existing_rect["x"] + existing_rect["width"],
                     "y": existing_rect["y"],
                     "width": room_rect["width"],
                     "height": room_rect["height"]},
                    {"x": existing_rect["x"],
                     "y": existing_rect["y"] - room_rect["height"],
                     "width": room_rect["width"],
                     "height": room_rect["height"]},
                    {"x": existing_rect["x"],
                     "y": existing_rect["y"] + existing_rect["height"],
                     "width": room_rect["width"],
                     "height": room_rect["height"]}
                ]
                for option in adjacency_options:
                    if any(self.check_overlap(option, existing_r) for existing_r in floorplan.values()):
                        continue
                    existing_rect["has_washroom_attached"] = True
                    room_rect.update(option)
                    return True
        return False

    def genetic_algorithm(self):
        # Initialize the population.
        population = self.initialize_population()
        # Use the RL agent's mutation rate if provided.
        mutation_rate = self.rl_agent.get_mutation_rate() if self.rl_agent is not None else self.MUTATION_RATE

        # Define a simple fitness function: maximize total room area.
        def fitness(floorplan):
            return sum(rect["width"] * rect["height"] for rect in floorplan.values())

        # Tournament selection helper.
        def tournament_selection(pop, tournament_size=3):
            participants = random.sample(pop, tournament_size)
            return max(participants, key=fitness)

        # Crossover operator: for each room, choose one parent's gene.
        def crossover(parent1, parent2):
            child = {}
            for room in parent1.keys():
                if random.random() < 0.5:
                    child[room] = parent1[room].copy()
                else:
                    child[room] = parent2[room].copy()
            return child

        # Mutation operator: perturb room parameters.
        def mutate(floorplan, mutation_rate):
            mutated = {}
            for room, rect in floorplan.items():
                new_rect = rect.copy()
                if random.random() < mutation_rate:
                    new_rect["x"] += random.randint(-1, 1)
                    new_rect["y"] += random.randint(-1, 1)
                    new_rect["width"] += random.randint(-1, 1)
                    new_rect["height"] += random.randint(-1, 1)
                    # Enforce boundaries.
                    if new_rect["x"] < 0:
                        new_rect["x"] = 0
                    if new_rect["y"] < 0:
                        new_rect["y"] = 0
                    if new_rect["x"] + new_rect["width"] > self.FLOORPLAN_WIDTH:
                        new_rect["width"] = self.FLOORPLAN_WIDTH - new_rect["x"]
                    if new_rect["y"] + new_rect["height"] > self.FLOORPLAN_HEIGHT:
                        new_rect["height"] = self.FLOORPLAN_HEIGHT - new_rect["y"]
                mutated[room] = new_rect
            return mutated

        # Full GA loop.
        for _ in range(self.GENERATIONS):
            new_population = []
            while len(new_population) < self.POPULATION_SIZE:
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)
                child = crossover(parent1, parent2)
                child = mutate(child, mutation_rate)
                new_population.append(child)
            population = new_population

        # Return the best floorplan.
        best_plan = max(population, key=fitness) if population else {}
        return best_plan
