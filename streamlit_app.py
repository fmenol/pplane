import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import streamlit as st
import os

# # Install Streamlit
# !pip install streamlit

# # Install localtunnel
# !npm install -g localtunnel

# Start Streamlit
def start_streamlit():
    os.system('streamlit run app.py &')

# Start localtunnel
def start_localtunnel():
    os.system('lt --port 8501')

# Default differential equations
dx_dt_eq = "(1 + a * y**2) / (1 + y**2) - gx * x"
dy_dt_eq = "(1 + a * y**2) / (1 + y**2) - gc * x * y - gy * y"

# Function to safely evaluate equations
def safe_eval(expr, **kwargs):
    allowed_names = {'x', 'y', 'a', 'gx', 'gy', 'gc', 'np'}
    code = compile(expr, "<string>", "eval")
    for name in code.co_names:
        if name not in allowed_names:
            raise NameError(f"The use of '{name}' is not allowed")
    return eval(code, {"__builtins__": {}, 'np': np}, kwargs)

# Define the differential equations
def dx_dt(x, y, a, gx, eq):
    return safe_eval(eq, x=x, y=y, a=a, gx=gx)

def dy_dt(x, y, a, gx, gy, gc, eq):
    return safe_eval(eq, x=x, y=y, a=a, gx=gx, gy=gy, gc=gc)

def system(t, z, a, gx, gy, gc, dx_eq, dy_eq):
    x, y = z
    return [dx_dt(x, y, a, gx, dx_eq), dy_dt(x, y, a, gx, gy, gc, dy_eq)]

# Function to find intersections of nullclines
def find_intersections(a, gx, gy, gc, dx_eq, dy_eq):
    def equations(p):
        x, y = p
        return (dx_dt(x, y, a, gx, dx_eq), dy_dt(x, y, a, gx, gy, gc, dy_eq))

    # Initial guesses for fsolve (can be adjusted if needed)
    guesses = [(i, j) for i in np.linspace(0, 40, 5) for j in np.linspace(0, 3, 5)]

    # Find intersections
    intersections = []
    for guess in guesses:
        intersection, infodict, ier, mesg = fsolve(equations, guess, full_output=True)
        if ier == 1:
            intersections.append(intersection)

    # Remove duplicates and points outside the range
    intersections = np.array(intersections)
    unique_intersections = []
    for point in intersections:
        if not any(np.allclose(point, uniq_point, atol=1e-2) for uniq_point in unique_intersections):
            unique_intersections.append(point)
    return np.array(unique_intersections)

# Function to compute the Jacobian matrix
def jacobian(x, y, a, gx, gy, gc, dx_eq, dy_eq):
    eps = 1e-6
    J = np.zeros((2, 2))
    J[0, 0] = (dx_dt(x + eps, y, a, gx, dx_eq) - dx_dt(x - eps, y, a, gx, dx_eq)) / (2 * eps)
    J[0, 1] = (dx_dt(x, y + eps, a, gx, dx_eq) - dx_dt(x, y - eps, a, gx, dx_eq)) / (2 * eps)
    J[1, 0] = (dy_dt(x + eps, y, a, gx, gy, gc, dy_eq) - dy_dt(x - eps, y, a, gx, gy, gc, dy_eq)) / (2 * eps)
    J[1, 1] = (dy_dt(x, y + eps, a, gx, gy, gc, dy_eq) - dy_dt(x, y - eps, a, gx, gy, gc, dy_eq)) / (2 * eps)
    return J

# Function to calculate eigenvalues
def calculate_eigenvalues(x, y, a, gx, gy, gc, dx_eq, dy_eq):
    J = jacobian(x, y, a, gx, gy, gc, dx_eq, dy_eq)
    return np.linalg.eigvals(J)

# Create a function to plot the nullclines, vector field, and trajectories
def plot_phase_plane(a, gx, gy, gc, x_range, y_range, x0, y0, timestep, total_time, dx_eq, dy_eq):
    x = np.linspace(0, x_range, 400)
    y = np.linspace(0, y_range, 400)
    X, Y = np.meshgrid(x, y)

    # Calculate nullclines
    dx = np.vectorize(dx_dt)(X, Y, a, gx, dx_eq)
    dy = np.vectorize(dy_dt)(X, Y, a, gx, gy, gc, dy_eq)

    # Plot the nullclines
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, dx, levels=[0], colors='r', linewidths=2, linestyles='dashed')
    plt.contour(X, Y, dy, levels=[0], colors='b', linewidths=2, linestyles='dashed')

    # Add labels for nullclines
    red_patch = mpatches.Patch(color='red', label='dx/dt = 0')
    blue_patch = mpatches.Patch(color='blue', label='dy/dt = 0')

    # Calculate the vector field
    U = np.vectorize(dx_dt)(X, Y, a, gx, dx_eq)
    V = np.vectorize(dy_dt)(X, Y, a, gx, gy, gc, dy_eq)

    # Plot the vector field using streamplot
    speed = np.sqrt(U**2 + V**2)
    strm = plt.streamplot(X, Y, U, V, color=speed, linewidth=1, cmap='coolwarm', arrowstyle='->')

    # Find and plot intersections of nullclines
    intersections = find_intersections(a, gx, gy, gc, dx_eq, dy_eq)
    plt.plot(intersections[:, 0], intersections[:, 1], 'go', label='Intersections')

    # Simulate and plot trajectories
    t_span = (0, total_time)
    t_eval = np.arange(0, total_time, timestep)
    sol = solve_ivp(system, t_span, [x0, y0], args=(a, gx, gy, gc, dx_eq, dy_eq), method='LSODA', t_eval=t_eval)
    plt.plot(sol.y[0], sol.y[1], 'k-', label='Trajectory')

    plt.xlim(0, x_range)
    plt.ylim(0, y_range)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Phase Plane with a={a}, gx={gx}, gy={gy}, gc={gc}')
    plt.grid(True)
    plt.gca().set_facecolor('white')
    plt.legend(handles=[red_patch, blue_patch, mpatches.Patch(color='green', label='Intersections'), mpatches.Patch(color='black', label='Trajectory')], loc='upper right')
    st.pyplot(plt.gcf())

    # Display intersections and eigenvalues
    if len(intersections) > 0:
        result_html = "<b>Intersections and Eigenvalues:</b><br>"
        for i, (x, y) in enumerate(intersections):
            eigenvalues = calculate_eigenvalues(x, y, a, gx, gy, gc, dx_eq, dy_eq)
            result_html += f"Intersection {i+1}: (x={x:.2f}, y={y:.2f})<br>"
            result_html += f"Eigenvalues: {eigenvalues[0]:.2f}, {eigenvalues[1]:.2f}<br>"
        st.markdown(result_html, unsafe_allow_html=True)
    else:
        st.markdown("<b>No intersections found.</b>", unsafe_allow_html=True)

st.title("Phase Plane Analysis")

# Create sliders for parameters
a = st.slider('a', 0.0, 50.0, 50.0, 0.1)
gx = st.slider('gx', 0.0, 50.0, 1.0, 0.1)
gy = st.slider('gy', 0.0, 50.0, 1.0, 0.1)
gc = st.slider('gc', 0.0, 50.0, 1.0, 0.1)
x_range = st.slider('x range', 1, 50, 40, 1)
y_range = st.slider('y range', 1, 50, 3, 1)
x0 = st.slider('x0', 0.0, 40.0, 1.0, 0.1)
y0 = st.slider('y0', 0.0, 3.0, 1.0, 0.1)
timestep = st.slider('timestep', 0.001, 1.0, 0.01, 0.001)
total_time = st.slider('total time', 1, 100, 10, 1)
dx_eq = st.text_input('dx/dt =', dx_dt_eq)
dy_eq = st.text_input('dy/dt =', dy_dt_eq)

# Plot the phase plane based on user input
plot_phase_plane(a, gx, gy, gc, x_range, y_range, x0, y0, timestep, total_time, dx_eq, dy_eq)

# Start Streamlit and localtunnel in separate threads
from threading import Thread
t1 = Thread(target=start_streamlit)
t2 = Thread(target=start_localtunnel)
t1.start()
t2.start()