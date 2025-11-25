import numpy as np
from numpy.typing import ArrayLike

class ControllerState:
    def __init__(self):
        self.v_integral = 0.0
        self.delta_integral = 0.0
        self.prev_v_error = 0.0
        self.prev_delta_error = 0.0
        self.dt = 0.1

ctrl_state = ControllerState()

def find_closest_point(position: ArrayLike, path: ArrayLike) -> int:
    """Find the index of the closest point on the path."""
    distances = np.linalg.norm(path - position, axis=1)
    return np.argmin(distances)

def pure_pursuit_steering(state: ArrayLike, path: ArrayLike, closest_idx: int, wheelbase: float) -> float:
    """
    Pure Pursuit algorithm for path tracking.
    Returns desired steering angle.
    """
    position = state[0:2]
    heading = state[4]
    velocity = max(abs(state[3]), 0.1)
    
    if velocity < 10.0:
        lookahead_distance = np.clip(8.0 + 0.3 * velocity, 8.0, 20.0)
    else:
        lookahead_distance = np.clip(8.0 + 0.35 * velocity, 8.0, 18.0)
    
    n_points = len(path)
    cumulative_dist = 0.0
    lookahead_idx = closest_idx
    
    for i in range(1, n_points):
        idx = (closest_idx + i) % n_points
        prev_idx = (closest_idx + i - 1) % n_points
        cumulative_dist += np.linalg.norm(path[idx] - path[prev_idx])
        if cumulative_dist >= lookahead_distance:
            lookahead_idx = idx
            break
    
    lookahead_point = path[lookahead_idx]
    
    dx = lookahead_point[0] - position[0]
    dy = lookahead_point[1] - position[1]
    
    local_x = dx * np.cos(-heading) - dy * np.sin(-heading)
    local_y = dx * np.sin(-heading) + dy * np.cos(-heading)
    
    ld = np.sqrt(local_x**2 + local_y**2)
    if ld < 0.1:
        return 0.0
    
    steering = np.arctan(2.0 * wheelbase * local_y / (ld * ld))
    
    return steering

def detect_s_curve(path: ArrayLike, idx: int) -> bool:
    n = len(path)
    window = 12
    
    turn_directions = []
    
    for i in range(5, 20, 5):
        check_idx = (idx + i) % n
        idx_before = (check_idx - window) % n
        idx_after = (check_idx + window) % n
        
        v1 = path[check_idx] - path[idx_before]
        v2 = path[idx_after] - path[check_idx]
        
        angle1 = np.arctan2(v1[1], v1[0])
        angle2 = np.arctan2(v2[1], v2[0])
        
        angle_diff = np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1))
        
        if abs(angle_diff) > 0.12:
            turn_directions.append(np.sign(angle_diff))
    
    if len(turn_directions) >= 2:
        for i in range(len(turn_directions) - 1):
            if turn_directions[i] * turn_directions[i + 1] < 0:
                return True
    
    return False

def compute_target_velocity(path: ArrayLike, idx: int) -> float:
    n = len(path)
    
    window = 15
    idx_before = (idx - window) % n
    idx_after = (idx + window) % n
    
    v1 = path[idx] - path[idx_before]
    v2 = path[idx_after] - path[idx]
    
    angle1 = np.arctan2(v1[1], v1[0])
    angle2 = np.arctan2(v2[1], v2[0])
    angle_diff = abs(np.arctan2(np.sin(angle2 - angle1), np.cos(angle2 - angle1)))
    
    max_angle = angle_diff
    for i in range(1, 20):
        future_idx = (idx + i * 2) % n
        future_before = (future_idx - window) % n
        future_after = (future_idx + window) % n
        fv1 = path[future_idx] - path[future_before]
        fv2 = path[future_after] - path[future_idx]
        fa1 = np.arctan2(fv1[1], fv1[0])
        fa2 = np.arctan2(fv2[1], fv2[0])
        future_angle = abs(np.arctan2(np.sin(fa2 - fa1), np.cos(fa2 - fa1)))
        max_angle = max(max_angle, future_angle)
    
    in_s_curve = detect_s_curve(path, idx)
    
    is_super_sharp = max_angle > 0.35
    
    if is_super_sharp:
        curvature_factor = np.exp(-7.0 * max_angle)
        target_v = 25.0 + 50.0 * curvature_factor
        return np.clip(target_v, 22.0, 75.0)
    elif in_s_curve:
        curvature_factor = np.exp(-6.0 * max_angle)
        target_v = 32.0 + 53.0 * curvature_factor
        return np.clip(target_v, 28.0, 85.0)
    else:
        curvature_factor = np.exp(-5.0 * max_angle)
        target_v = 40.0 + 60.0 * curvature_factor
        return np.clip(target_v, 35.0, 100.0)

def velocity_pid(current_v: float, target_v: float) -> float:
    error = target_v - current_v
    
    ctrl_state.v_integral += error * ctrl_state.dt
    ctrl_state.v_integral = np.clip(ctrl_state.v_integral, -10.0, 10.0)
    
    derivative = (error - ctrl_state.prev_v_error) / ctrl_state.dt
    ctrl_state.prev_v_error = error
    
    if error < 0:
        kp, ki, kd = 8.0, 0.6, 0.35
    else:
        kp, ki, kd = 6.5, 0.5, 0.25
    
    accel = kp * error + ki * ctrl_state.v_integral + kd * derivative
    
    accel = np.clip(accel, -20.0, 20.0)
    
    return accel

def steering_pid(current_delta: float, target_delta: float, current_v: float = None) -> float:
    error = target_delta - current_delta
    error = np.arctan2(np.sin(error), np.cos(error))
    
    ctrl_state.delta_integral += error * ctrl_state.dt
    ctrl_state.delta_integral = np.clip(ctrl_state.delta_integral, -0.3, 0.3)
    
    raw_derivative = (error - ctrl_state.prev_delta_error) / ctrl_state.dt
    if not hasattr(ctrl_state, 'filtered_delta_derivative'):
        ctrl_state.filtered_delta_derivative = 0.0
    alpha = 0.3
    ctrl_state.filtered_delta_derivative = alpha * raw_derivative + (1 - alpha) * ctrl_state.filtered_delta_derivative
    derivative = ctrl_state.filtered_delta_derivative
    
    ctrl_state.prev_delta_error = error
    
    kp, ki, kd = 7.0, 0.35, 0.4
    steering_rate = kp * error + ki * ctrl_state.delta_integral + kd * derivative
    
    max_rate = 0.4
    
    return np.clip(steering_rate, -max_rate, max_rate)

def controller(state: ArrayLike, parameters: ArrayLike, racetrack, raceline: ArrayLike) -> ArrayLike:
    closest_idx = find_closest_point(state[0:2], raceline)
    
    wheelbase = parameters[0]
    desired_steering = pure_pursuit_steering(state, raceline, closest_idx, wheelbase)
    
    desired_velocity = compute_target_velocity(raceline, closest_idx)
    
    return np.array([desired_steering, desired_velocity])

def lower_controller(state: ArrayLike, desired: ArrayLike, parameters: ArrayLike) -> ArrayLike:
    desired_delta = desired[0]
    desired_v = desired[1]
    
    current_delta = state[2]
    current_v = abs(state[3])
    
    acceleration = velocity_pid(current_v, desired_v)
    steering_rate = steering_pid(current_delta, desired_delta, current_v)
    
    max_steering_rate = parameters[9]
    min_steering_rate = parameters[7]
    max_accel = parameters[10]
    min_accel = parameters[8]
    
    steering_rate = np.clip(steering_rate, min_steering_rate, max_steering_rate)
    acceleration = np.clip(acceleration, min_accel, max_accel)
    
    return np.array([steering_rate, acceleration])
