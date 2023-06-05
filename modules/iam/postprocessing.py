
from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
import re

def label2char(label: int):
    code = label + ord(" ")
    return chr(code)

class Point:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

    def __repr__(self):
        return f'Point({self.x}, {self.y}, "{self.label}")'

def walk_char(acc, i, j, visited_symbols=[], visited_acc=None):
    stack = [(i, j)]
    while stack:
        i, j = stack.pop()
        # check if i and j are within bounds
        if i < 0 or i >= acc.shape[0] or j < 0 or j >= acc.shape[1]:
            continue

        # if visited or reached a background pixel then return
        if acc[i, j] == 0 or visited_acc[i, j] == 1:
            continue

        visited_acc[i, j] = 1  # visited
        visited_symbols.append((i, j))

        stack.append((i - 1, j))
        stack.append((i + 1, j))
        stack.append((i, j - 1))
        stack.append((i, j + 1))


def set_adaptive_threshold(acc):
    num_pixels_per_symbol = []
    visited_acc = np.zeros_like(acc)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if acc[i, j] == 1 and visited_acc[i, j] == 0:
                visited_positions = []
                walk_char(acc, i, j, visited_positions, visited_acc=visited_acc)
                num_pixels_per_symbol.append(len(visited_positions))

    num_pixels_per_symbol = np.sort(num_pixels_per_symbol)

    fourier = np.fft.fft(num_pixels_per_symbol)
    freqs = np.fft.fftfreq(len(num_pixels_per_symbol))
    fourier[freqs > 0.1] = 0
    num_pixels_per_symbol = np.fft.ifft(fourier)
    num_pixels_per_symbol = np.real(num_pixels_per_symbol)
    num_pixels_per_symbol[num_pixels_per_symbol < 0] = 0
    num_pixels_per_symbol = np.round(num_pixels_per_symbol).astype(np.uint8)

    threshold = num_pixels_per_symbol[-1]

    return threshold


def remove_ornaments(page, threshold=10):
    acc = page.copy()
    visited_acc = np.zeros_like(acc)
    delete_list = []

    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            if acc[i, j] == 1 and visited_acc[i, j] == 0:

                visited_positions = []
                walk_char(acc, i, j, visited_positions, visited_acc=visited_acc)

                if len(visited_positions) < threshold:
                    delete_list.extend(visited_positions)

            visited_acc[i, j] = 1  # visited

    for i, j in delete_list:
        acc[i, j] = 0
    acc = (acc > 0)
    acc = acc.astype(np.uint8)
    return acc

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

def construct_graph(points, dist_x=50, dist_y=250):
    graph = {}
    for u in points:
        graph[u] = []
        for v in points:
            if u == v:
                continue

            d_x = abs(u.x - v.x)
            d_y = abs(u.y - v.y)
            
            if d_x <= dist_x and d_y <= dist_y:
                graph[u].append(v)
    return graph

def plot_graph(img: np.ndarray, graph, label_to_color):
    plt.figure(figsize=(10, 10))  # Set the figure size, you can adjust this as needed
    plt.imshow(-img, cmap='CMRmap')  # Display the image in grayscale

    for u, neighbors in graph.items():
        for v in neighbors:
            plt.plot([u.y, v.y], [u.x, v.x], color=label_to_color[str(u.label)], alpha=0.1)

    # tick y every 50 pixels
    plt.yticks(np.arange(0, img.shape[0], 50))
    plt.show()  # show the plot

def estimate_line_spacing(img):
    histogram = np.sum(img, axis=1)
    histogram[histogram < histogram.mean()] = 0

    # Compute the frequencies using np.fft.fftfreq
    num_samples = len(histogram)
    frequencies = np.fft.fftfreq(num_samples)

    # Perform a Fourier Transform on the histogram
    fft_result = np.fft.fft(histogram)
    fft_magnitude = np.abs(fft_result)

    # Exclude the first element (corresponding to the DC component)
    frequencies = frequencies[1:]
    fft_magnitude = fft_magnitude[1:]

    # Find the indices of the peaks corresponding to the highest frequencies
    peak_indices = np.argsort(-fft_magnitude)
    peak_wavelengths = 1 / np.abs(frequencies[peak_indices])
    # Filter out values in peak_wavelengths that dont have decimals
    peak_wavelengths = peak_wavelengths[np.where(peak_wavelengths % 1 != 0)]

    top_k_wavelengths = peak_wavelengths[:3]
    line_spacing = np.mean(top_k_wavelengths)

    return line_spacing

def transpose_points(points):
    return [Point(point.y, point.x, point.label) for point in points]

def plot_lines(img: np.ndarray, lines, label_to_color):
    plt.figure(figsize=(10, 10))  # Set the figure size, you can adjust this as needed
    plt.imshow(-img, cmap='CMRmap')  # Display the image in grayscale

    i2color = {i: color for i, color in enumerate(label_to_color.values())}

    for j, line in enumerate(lines):
        for i in range(len(line) - 1):
            u = line[i]
            v = line[i + 1]
            plt.plot([u.y, v.y], [u.x, v.x], color=i2color[j], alpha=1)

    # plt.axis('off')  # remove the axis
    plt.show()  # show the plot

def find_best_line(graph, line):
    """
    Recursively find the best line 
    Args:
        graph: the graph
        line: the current line so far starting with [p0, p1, ..., pn] sorted by x coordinate
    Returns:
        the best line starting with [starting_point], fitness of the line
    """

    def get_started_points(line):
        # Mark the ends of the line
        starting_points = [line[-1]]
        if len(line) != 1:
            starting_points.append(line[-2])
        return starting_points
    
    def point_in_bounds(point, line):
        # Line bounds for pruning 
        min_x = min([p.x for p in line])
        max_x = max([p.x for p in line])
        return point.x >= min_x and point.x <= max_x

    prev_starting_points = None
    starting_points = get_started_points(line)

    while prev_starting_points != starting_points:
        for starting_point in starting_points:
            neighbors = [p for p in graph[starting_point] if not point_in_bounds(p, line)]

            if len(neighbors) == 0:
                continue
            closest_neighbor = min(neighbors, key=lambda p: abs(p.x - starting_point.x))
            new_line = line + [closest_neighbor]
            new_line = sorted(new_line, key=lambda p: p.x)
            line = new_line

        prev_starting_points = starting_points
        starting_points = get_started_points(line)
        
    return line

def extract_lines_from_graph(graph):
    lines = []
    ranks = []

    _graph = graph.copy()        
    while len(_graph) > 0:
        # take point with max y
        starting_point = min(_graph.keys(), key=lambda p: p.x)
        # starting_point = random.choice(list(_graph.keys()))
        line = find_best_line(_graph, [starting_point]) 
        rank = np.mean([p.y for p in line])
        lines.append(line)
        ranks.append(rank)

        # remove points from graph
        for point in line:
            # remove references to this point
            for neighbors in _graph.values():
                if point in neighbors:
                    neighbors.remove(point)
            del _graph[point]

    # sort lines by rank
    lines = [line for _, line in sorted(zip(ranks, lines), key=lambda pair: pair[0])]

    return lines

def extract_lines(img):
    binary_image = np.where(img > 0, 1, 0)
    points = find_centers_of_characters(img)
    lines = [points]
    return lines

def transcribe_image(img):
    # img = img.max(0)
    img = img.astype(int)
    # Take mode across axis 0
    votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=ord("~")-ord(" ")+1), axis=0, arr=img)
    votes[0] = 0
    avg_counts = np.mean(votes, axis=1)
    # Remove outliers
    votes = np.where(votes > avg_counts[:, None], votes, 0)
    img = np.argmax(votes, axis=0)
    img = img.tolist()
    transcription = [label2char(int(label)) for label in img]
    transcription = ''.join(transcription)
    # Remove redudant chars for example, heeelllloooo -> hello
    out = ''
    for i, c in enumerate(transcription):
        if i == 0 or c != transcription[i - 1]:
            out += c
    return out