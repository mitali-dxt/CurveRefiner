#completion.py
import numpy as np
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

def handle_disconnected_occlusions(points):
    if len(points) < 2:
        return points
    start_point = points[0]
    end_point = points[-1]
    lin_reg = LinearRegression()
    lin_reg.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    new_x = np.linspace(start_point[0], end_point[0], num=500)
    new_y = lin_reg.predict(new_x.reshape(-1, 1))
    completed_points = np.vstack((points, np.column_stack((new_x, new_y))))
    return completed_points

def complete_curve(points, occlusion_type='connected'):
    if occlusion_type == 'connected':
        x = points[:, 0]
        y = points[:, 1]
        f = interp1d(x, y, kind='linear')
        new_x = np.linspace(x[0], x[-1], num=500)
        new_y = f(new_x)
        completed_points = np.column_stack((new_x, new_y))
        return completed_points
    elif occlusion_type == 'disconnected':
        completed_points = handle_disconnected_occlusions(points)
        return completed_points