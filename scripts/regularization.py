# regularization.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial import distance
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from skimage.measure import EllipseModel
import cv2

def regularize_line(points):
    if len(points) < 2:
        return points
    # Keep the start and end points the same
    start_point = points[0]
    end_point = points[-1]
    total_distance = np.linalg.norm(end_point - start_point)
    # Calculate the direction vector for the straight line
    direction = (end_point - start_point) / total_distance
    # Generate new points along the straight line
    new_points = []
    for i in range(len(points)):
        # Calculate the distance from the start point for the current point
        current_distance = np.linalg.norm(points[i] - start_point)
        # Calculate the new point along the straight line
        new_point = start_point + current_distance * direction
        new_points.append(new_point)
    return np.array(new_points)
    
def regularize_circle(points):
    if len(points) < 3:
        return points
    def calc_R(xc, yc):
        return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2)
    def residuals(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    center_estimate = np.mean(points, axis=0)
    center = least_squares(residuals, center_estimate).x
    Ri = calc_R(*center)
    R = Ri.mean()
    theta = np.linspace(0, 2 * np.pi, len(points))
    x_fit = center[0] + R * np.cos(theta)
    y_fit = center[1] + R * np.sin(theta)
    return np.column_stack((x_fit, y_fit))

def regularize_rectangle(points):
    if len(points) < 4:
        return points
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    rect_points = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]])
    return rect_points

def regularize_square(points):
    if len(points) < 4:
        return points
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    side = max(x_max - x_min, y_max - y_min)
    sqr_points = np.array([[x_min, y_min], [x_min + side, y_min], [x_min + side, y_min + side], [x_min, y_min + side], [x_min, y_min]])
    return sqr_points


def find_star_vertices(points, angle_threshold=0.1):
    vertices = []
    num_points = len(points)
    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        p3 = points[(i + 2) % num_points]
        # Vectors
        v1 = p2 - p1
        v2 = p3 - p2
        # Compute angle between vectors
        angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
        if angle > angle_threshold:
            vertices.append(p2)
    return vertices


def regularize_star(points):
    vertices = find_star_vertices(points)
    # Create an ordered list of vertices 
    ordered_vertices = np.array(vertices)
    ordered_vertices = np.append(ordered_vertices, [ordered_vertices[0]], axis=0)  # Close the shape 
    return ordered_vertices


def regularize_ellipse(points):
    if len(points) < 5:
        return points
    # Fit ellipse using OpenCV
    ellipse_params = cv2.fitEllipse(np.array(points, dtype=np.float32))
    center, (major_axis, minor_axis), angle = ellipse_params
    # Generate ellipse points
    theta = np.linspace(0, 2 * np.pi, len(points))
    a = major_axis / 2
    b = minor_axis / 2
    x0, y0 = center
    x_fit = x0 + a * np.cos(theta) * np.cos(np.deg2rad(angle)) - b * np.sin(theta) * np.sin(np.deg2rad(angle))
    y_fit = y0 + a * np.cos(theta) * np.sin(np.deg2rad(angle)) + b * np.sin(theta) * np.cos(np.deg2rad(angle))
    return np.column_stack((x_fit, y_fit))

def regularize_saturn_ring(points):
    if len(points) < 10:
        return points
    body_ellipse = regularize_ellipse(points)
    num_rings = 3
    ring_width = 0.1
    ring_ellipses = []
    for i in range(num_rings):
        scale = 1 + i * ring_width
        ring_ellipse = body_ellipse * scale
        ring_ellipses.append(ring_ellipse)
    return np.vstack([body_ellipse] + ring_ellipses)