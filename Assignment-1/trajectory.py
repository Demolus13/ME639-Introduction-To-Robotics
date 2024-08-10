import numpy as np
import matplotlib.pyplot as plt

# Robot arm parameters
L1 = 1.0
L2 = 1.0

def inverse_kinematics(x, y):
    """
    Compute the inverse kinematics for a 2D robot arm with two links.
    Returns the joint angles theta1 and theta2.
    """
    d = np.sqrt(x**2 + y**2)
    if d > (L1 + L2):
        return None, None

    # Law of cosines
    cos_theta2 = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    sin_theta2 = np.sqrt(1 - cos_theta2**2)
    theta2 = np.arctan2(sin_theta2, cos_theta2)
    
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * sin_theta2
    theta1 = np.arctan2(y, x) - np.arctan2(k2, k1)
    
    return theta1, theta2

def forward_kinematics(theta1, theta2):
    """
    Compute the forward kinematics for a 2D robot arm with two links.
    Returns the (x, y) coordinates of the end-effector.
    """
    if theta1 is None or theta2 is None:
        return None, None
    
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y

# Define trajectories
t = np.linspace(0, 2 * np.pi, 100)

# 1. Straight Line
x_line = np.linspace(0, 1.5, 100)
y_line = 1.5 * x_line + 0.5

# 2. Circle
x_circle = 1.0 * np.cos(t)
y_circle = 1.0 * np.sin(t)

# 3. Sine Wave
x_sine = np.linspace(0, 2 * np.pi, 100)
y_sine = 0.5 * np.sin(x_sine)

# Calculate end-effector coordinates from trajectories
x_line_fk, y_line_fk = [], []
x_circle_fk, y_circle_fk = [], []
x_sine_fk, y_sine_fk = [], []

for (x, y) in zip(x_line, y_line):
    theta1, theta2 = inverse_kinematics(x, y)
    x_fk, y_fk = forward_kinematics(theta1, theta2)
    x_line_fk.append(x_fk)
    y_line_fk.append(y_fk)

for (x, y) in zip(x_circle, y_circle):
    theta1, theta2 = inverse_kinematics(x, y)
    x_fk, y_fk = forward_kinematics(theta1, theta2)
    x_circle_fk.append(x_fk)
    y_circle_fk.append(y_fk)

for (x, y) in zip(x_sine, y_sine):
    theta1, theta2 = inverse_kinematics(x, y)
    x_fk, y_fk = forward_kinematics(theta1, theta2)
    x_sine_fk.append(x_fk)
    y_sine_fk.append(y_fk)

# Plot results
plt.figure(figsize=(15, 5))

# Plot for Straight Line
plt.subplot(1, 3, 1)
plt.plot(x_line, y_line, label='Original Trajectory', linewidth=2)
plt.plot(x_line_fk, y_line_fk, label='Recalculated Trajectory', linestyle='--', linewidth=2)
plt.title('Straight Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')

# Plot for Circle
plt.subplot(1, 3, 2)
plt.plot(x_circle, y_circle, label='Original Trajectory', linewidth=2)
plt.plot(x_circle_fk, y_circle_fk, label='Recalculated Trajectory', linestyle='--', linewidth=2)
plt.title('Circle')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')

# Plot for Sine Wave
plt.subplot(1, 3, 3)
plt.plot(x_sine, y_sine, label='Original Trajectory', linewidth=2)
plt.plot(x_sine_fk, y_sine_fk, label='Recalculated Trajectory', linestyle='--', linewidth=2)
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

