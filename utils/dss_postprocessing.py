import os

# ensure is in parent directory
try:
    os.chdir("../../RUG-HandRec/")
except:
    pass

from PIL import Image
from collections import defaultdict

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
from utils.transforms import remove_ornaments, set_adaptive_threshold
import matplotlib.pyplot as plt

import numpy as np
import math

from typing import List, Tuple, Optional


class Point:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

    def __repr__(self):
        return f'Point({self.x}, {self.y}, "{self.label}")'


def find_centers_of_characters(img: np.ndarray, distance_threshold=0):
    unique_values = np.unique(img)
    unique_values = unique_values[unique_values != 0]  # Exclude background value (0)

    points = []

    for value in unique_values:
        binary_img = np.where(img == value, value, 0).astype(img.dtype)

        # Label connected components
        labeled_img, num_components = label(binary_img)

        # Calculate centers
        value_centers = center_of_mass(binary_img, labeled_img, range(1, num_components + 1))

        # Convert coordinates to integers
        int_centers = [tuple(map(int, center)) for center in value_centers]

        # Merge centers that are close together
        merged_centers = []
        for idx, center in enumerate(int_centers):
            if center in merged_centers:
                continue

            # Find the centers that are within the distance threshold
            distances = cdist([center], int_centers)[0]
            close_centers_idx = np.where(distances <= distance_threshold)[0]

            # Calculate the new combined center
            new_center = np.mean([int_centers[i] for i in close_centers_idx], axis=0)
            merged_centers.append(Point(int(new_center[0]), int(new_center[1]), value))

        points.extend(merged_centers)

    return points


def plot_points_on_image(img: np.ndarray, points, label_to_color):
    plt.figure(figsize=(10, 10))  # Set the figure size, you can adjust this as needed
    plt.imshow(img, cmap='gray')  # Display the image in grayscale

    for point in points:
        plt.scatter(point.y, point.x, s=5, color=label_to_color[str(point.label)])

    plt.show()  # show the plot


def greate_graph(points):
    graph = defaultdict(list)
    for i, point in enumerate(points):
        for j, other_point in enumerate(points):
            if i == j:
                continue
            angle = math.atan2(other_point.y - point.y, other_point.x - point.x)
            graph[point].append((other_point, angle))
    return graph


# TODO rewrite using Point class
def distance_between_points(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def find_starting_point(points: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    try:
        anchor_point = (1500, 0)
        return min(points, key=lambda point: distance_between_points(anchor_point, point))
    except ValueError:
        print("No starting point found")
        return None


def find_next_point_in_line(current_point: Tuple[int, int], points: List[Tuple[int, int]],
                            angle_threshold=50) -> Optional[Tuple[int, int]]:
    points_on_left = [point for point in points if point[0] < current_point[0] and abs(
        point[1] - current_point[1]) < angle_threshold]
    return min(points_on_left, key=lambda point: distance_between_points(current_point, point),
               default=None)


def find_next_line_start(points: List[Tuple[int, int]], last_point: Tuple[int, int]) -> Optional[
    Tuple[int, int]]:
    remaining_points = [point for point in points if
                        point[1] > last_point[1]]  # Filter points below the last point
    return find_starting_point(remaining_points)


def print_characters(points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    points = [(point.y, point.x) for point in points]
    ordered_points = []
    start_point = find_starting_point(points)
    if start_point is None:
        return ordered_points

    ordered_points.append(start_point)
    points.remove(start_point)

    while points:
        next_point = find_next_point_in_line(ordered_points[-1], points)
        if next_point is not None:
            ordered_points.append(next_point)
            points.remove(next_point)
        else:
            next_line_start = find_next_line_start(points, ordered_points[-1])
            if next_line_start is not None:
                ordered_points.append(next_line_start)
                points.remove(next_line_start)
            else:
                break

    return ordered_points


def get_label_from_coords(points, center) -> Point:
    for point in points:
        if point.y == center[0] and point.x == center[1]:
            return hebrew_chars[int(point.label - 1)]
    return None


def transcribe_image(image) -> str:
    binary_image = np.where(image > 0, 1, 0)
    allowed_pixels = remove_ornaments(binary_image, set_adaptive_threshold(binary_image))
    image_without_ornaments = np.where(allowed_pixels > 0, image, 0)

    points = find_centers_of_characters(image_without_ornaments)
    ordered_centers = print_characters(points)
    indices = list(range(1, len(ordered_centers) + 1))

    final_string = ""

    starting_point = find_starting_point(ordered_centers)
    for center, number in zip(ordered_centers, indices):
        if center[0] - starting_point[0] < image.shape[0] * 0.3:
            final_string += str(get_label_from_coords(points, center))
        else:
            final_string += "\n"
        starting_point = center

    return final_string


if __name__ == '__main__':
    hebrew_chars = ["א", "ב", "ג", "ד", "ה", "ו", "ז", "ח", "ט", "י",
                    "כ", "ל", "מ", "נ", "ס", "ע", "פ", "צ", "ק", "ר",
                    "ש", "ת", "ך", "ם", "ן", "ף", "ץ"]

    masks_folder = "experiments/Masks/"
    masks = os.listdir(masks_folder)
    random_mask = np.random.choice(masks)
    image = np.array(Image.open(masks_folder + random_mask))

    # show image
    plt.imshow(image, cmap='gray')
    plt.show()

    # transcribe image
    transcript = transcribe_image(image)
    print(transcript)
