# symmetry.py
import numpy as np

def detect_symmetry(points):
    reflectional_symmetry_lines = detect_reflectional_symmetry_lines(points)
    rotational_symmetry_lines_180 = detect_rotational_symmetry_lines(points, np.pi)  # 180 degrees
    rotational_symmetry_lines_90 = detect_rotational_symmetry_lines(points, np.pi / 2)  # 90 degrees
    symmetry_results = {
        "Reflectional Symmetry Lines": reflectional_symmetry_lines,
        "Rotational Symmetry Lines (180°)": rotational_symmetry_lines_180,
        "Rotational Symmetry Lines (90°)": rotational_symmetry_lines_90,
    }
    return symmetry_results

def detect_reflectional_symmetry_lines(points):
    centroid = np.mean(points, axis=0)
    count = 0
    for angle in np.linspace(0, np.pi, 180):  # Test every 1 degree
        reflection_matrix = np.array([
            [np.cos(2 * angle), np.sin(2 * angle)],
            [np.sin(2 * angle), -np.cos(2 * angle)]
        ])
        reflected_points = np.dot(points - centroid, reflection_matrix) + centroid
        sorted_points = np.sort(points, axis=0)
        sorted_reflected = np.sort(reflected_points, axis=0)
        differences = np.linalg.norm(sorted_points - sorted_reflected, axis=1)
        if np.all(differences < 1e-6):
            count += 1
    return count

def detect_rotational_symmetry_lines(points, angle):
    centroid = np.mean(points, axis=0)
    count = 0
    for i in range(0, 360, int(np.degrees(angle))):
        rad_angle = np.radians(i)
        rotation_matrix = np.array([
            [np.cos(rad_angle), -np.sin(rad_angle)],
            [np.sin(rad_angle),  np.cos(rad_angle)]
        ])
        rotated_points = np.dot(points - centroid, rotation_matrix.T) + centroid
        sorted_points = np.sort(points, axis=0)
        sorted_rotated = np.sort(rotated_points, axis=0)
        differences = np.linalg.norm(sorted_points - sorted_rotated, axis=1)
        if np.all(differences < 1e-6):
            count += 1
    return count