# main-file
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from skimage.measure import EllipseModel
import cv2
#from symmetry import detect_symmetry 

def read_csv_(csv_path):
    np_path_XYs = np.genfromtxt(csv_path, delimiter=',')
    path_XYs = []
    for i in np.unique(np_path_XYs[:, 0]):
        npXYs = np_path_XYs[np_path_XYs[:, 0] == i][:, 1:]
        XYs = []
        for j in np.unique(npXYs[:, 0]):
            XY = npXYs[npXYs[:, 0] == j][:, 1:]
            XYs.append(XY)
        path_XYs.append(XYs)
    return path_XYs

def get_contour(points):
    points = points.reshape((-1, 1, 2)).astype(np.int32)
    # Get the contour using OpenCV
    contour = cv2.approxPolyDP(points, 0.02 * cv2.arcLength(points, True), True)
    return contour.reshape(-1, 2)  # Flatten contour to a 2D array

def is_right_angle(v1, v2):
    """Check if the angle between vectors v1 and v2 is close to 90 degrees"""
    v1 = v1.flatten() 
    v2 = v2.flatten()
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (magnitude_v1 * magnitude_v2)
    return np.isclose(cosine_angle, 0, atol=1e-2)

def similarity_rectangle(points):
    if len(points) < 4:
        return 0
    contour = get_contour(points)
    if len(contour) != 4:
        return 0
    v1 = contour[1] - contour[0]
    v2 = contour[2] - contour[1]
    v3 = contour[3] - contour[2]
    v4 = contour[0] - contour[3]
    angles = [
        is_right_angle(v1, v2),
        is_right_angle(v2, v3),
        is_right_angle(v3, v4),
        is_right_angle(v4, v1)
    ]
    angle_score = sum(angles) / len(angles)
    side_lengths = [
        np.linalg.norm(v1),
        np.linalg.norm(v2),
        np.linalg.norm(v3),
        np.linalg.norm(v4)
    ]
    length_score = (
        (np.isclose(side_lengths[0], side_lengths[2], atol=1e-2) +
         np.isclose(side_lengths[1], side_lengths[3], atol=1e-2)) / 2
    )
    # Overall similarity score
    overall_score = (angle_score + length_score) / 2
    return overall_score

def calculate_angle(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))
    return angle

def similarity_line(points, angle_threshold=167):
    if len(points) < 2:
        return 0
    contour = get_contour(points)
    # Check contour length
    if len(contour) == 2:
        return 1 
    if len(contour) == 3:
        p1, p2, p3 = contour[0], contour[1], contour[2]
        angle = calculate_angle(p1, p2, p3)
        # Check if the angle is greater than the threshold
        if 180-angle > angle_threshold:
            return 1
    return 0

def similarity_circle(points):
    if len(points) < 3:
        return 0
    contour = get_contour(points)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity

def similarity_square(points):
    if len(points) < 4:
        return 0
    # Convert points to a contour format
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    contour = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    if len(contour) != 4:
        return 0
    # Compute bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    # Compute aspect ratio to check if the shape is a square
    aspect_ratio = w / float(h)
    # Check if aspect ratio is close to 1 (indicating a square)
    if 0.9 <= aspect_ratio <= 1.1:
        return 1
    return 0

def similarity_star(points):
    contour = get_contour(points)
    if len(contour)==10:
      return 1
    if len(points) < 5:
        return 0
    contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    contour = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull.size == 0:
        return 0
    try:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0
        num_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(start) - np.array(far))
            c = np.linalg.norm(np.array(end) - np.array(far))
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
            if angle <= np.pi / 2:  # angle less than 90 degrees
                num_defects += 1
        return 1 if num_defects == 5 else 0
    except cv2.error as e:
        print(f"OpenCV error: {e}")
        return 0

#def similarity_ellipse(points):
    if len(points) < 5:
        return 0
    # Fit an ellipse using OpenCV
    try:
        # Ensure points are in the correct format for cv2.fitEllipse
        points = np.array(points, dtype=np.float32)
        ellipse_params = cv2.fitEllipse(points)
        # Extract ellipse parameters
        center, (major_axis, minor_axis), angle = ellipse_params
        # Generate ellipse points using the parameters
        theta = np.linspace(0, 2 * np.pi, len(points))
        a = major_axis / 2
        b = minor_axis / 2
        x0, y0 = center
        angle_rad = np.deg2rad(angle)
        x_fit = x0 + a * np.cos(theta) * np.cos(angle_rad) - b * np.sin(theta) * np.sin(angle_rad)
        y_fit = y0 + a * np.cos(theta) * np.sin(angle_rad) + b * np.sin(theta) * np.cos(angle_rad)
        ellipse_fit = np.column_stack((x_fit, y_fit))

        # Compute the fitting error
        distances = np.linalg.norm(points[:, None] - ellipse_fit[None, :], axis=2)
        min_distances = np.min(distances, axis=1)
        error = np.mean(min_distances)

        # Define a threshold for similarity
        threshold = 5
        return 1 if error < threshold else 0
    except Exception as e:
        print(f"Error fitting ellipse: {e}")

#def similarity_saturn_ring(points):
    if len(points) < 10:
        return 0
    main_ellipse = regularize_ellipse(points)
    try:
        num_rings = 3
        ring_width = 0.1
        rings = []
        for i in range(num_rings):
            scale = 1 + i * ring_width
            ring_ellipse = regularize_ellipse(points) * scale
            rings.append(ring_ellipse)
        errors = []
        for ring in rings:
            distances = np.linalg.norm(points[:, None] - ring[None, :], axis=2)
            min_distances = np.min(distances, axis=1)
            error = np.mean(min_distances)
            errors.append(error)

        # Combine errors to assess overall similarity
        overall_error = np.mean(errors)
        threshold = 10 
        return 1 if overall_error < threshold else 0
    except Exception as e:
        print(f"Error fitting Saturn rings: {e}")
        return 0

def process_shapes(paths_XYs, threshold=0.74):
    processed_paths = []
    for path in paths_XYs:
        for points in path:
            if len(points) < 2:
                continue

            # Calculate symmetry lines
            symmetry_info = detect_symmetry(points)
            reflectional_lines = symmetry_info["Reflectional Symmetry Lines"]
            rotational_lines_180 = symmetry_info["Rotational Symmetry Lines (180°)"]
            rotational_lines_90 = symmetry_info["Rotational Symmetry Lines (90°)"]

            print(f"Reflectional Symmetry Lines: {reflectional_lines}")
            print(f"Rotational Symmetry Lines (180°): {rotational_lines_180}")
            print(f"Rotational Symmetry Lines (90°): {rotational_lines_90}")

            # Calculate similarity probabilities for different shapes
            line_prob = similarity_line(points)
            circle_prob = similarity_circle(points)
            rectangle_prob = similarity_rectangle(points)
            square_prob = similarity_square(points)
            star_prob = similarity_star(points)
            #ellipse_prob = similarity_ellipse(points)
            #saturn_ring_prob = similarity_saturn_ring(points)

            # Determine the maximum similarity probability
            max_prob = max(line_prob, circle_prob, rectangle_prob, square_prob, star_prob)#, ellipse_prob, saturn_ring_prob)

            print(f"line_prob: {line_prob}, circle_prob: {circle_prob}, rectangle_prob: {rectangle_prob}, square_prob: {square_prob}, star_prob: {star_prob}")#, ellipse_prob: {ellipse_prob}, saturn_ring_prob: {saturn_ring_prob}")

            # Regularize the shape based on the highest similarity probability
            if max_prob < threshold:
                processed_paths.append([points])
            elif max_prob == line_prob:
                processed_paths.append([regularize_line(points)])
            elif max_prob == circle_prob:
                processed_paths.append([regularize_circle(points)])
            elif max_prob == rectangle_prob:
                processed_paths.append([regularize_rectangle(points)])
            elif max_prob == square_prob:
                processed_paths.append([regularize_square(points)])
            elif max_prob == star_prob:
                processed_paths.append([regularize_star(points)])
            #elif max_prob == ellipse_prob:
                #processed_paths.append([regularize_ellipse(points)])
            #elif max_prob == saturn_ring_prob:
                #processed_paths.append([regularize_saturn_ring(points)])
                
            # If symmetry lines are detected
            if reflectional_lines > 0:
                print(f"Shape has {reflectional_lines} reflectional symmetry lines.")
            if rotational_lines_180 > 0:
                print(f"Shape has {rotational_lines_180} rotational symmetry lines at 180°.")
            if rotational_lines_90 > 0:
                print(f"Shape has {rotational_lines_90} rotational symmetry lines at 90°.")
                
    return processed_paths

def plot(paths_XYs, title, ax):
    colours = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan', 'magenta']
    for i, XYs in enumerate(paths_XYs):
        c = colours[i % len(colours)]
        for XY in XYs:
            ax.plot(XY[:, 0], XY[:, 1], c=c, linewidth=2)
    ax.set_aspect('equal')
    ax.set_title(title)

csv_path1 = "occlusion2.csv"
csv_path2 = "occlusion2_sol.csv"

output_data1 = read_csv_(csv_path1)
expected_output_data = read_csv_(csv_path2)

processed_data = process_shapes(output_data1)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

plot(output_data1, "Original Data", ax1)
plot(processed_data, "Processed Data", ax2)
plot(expected_output_data, "Expected Output", ax3)

plt.show()