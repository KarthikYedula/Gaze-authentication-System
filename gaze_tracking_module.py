import cv2
import numpy as np
import random
from GazeTracking.gaze_tracking import GazeTracking
import threading
import time


is_tracking = False  # Global variable to track gaze authentication status


gaze = GazeTracking()

def create_connected_maze_with_display(size=5):
    """
    Generate a maze with simple, high-contrast colors for easy analysis.
    """
    shape = (size * 2 + 1, size * 2 + 1)
    maze = np.ones(shape, dtype=np.uint8) * 255  # White paths
    visited = np.zeros(shape, dtype=bool)
    start_x, start_y = 1, 1
    stack = [(start_x, start_y)]
    visited[start_y, start_x] = True
    maze_coords = [(start_x, start_y)]

    directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
    while stack:
        x, y = stack[-1]
        found = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 < nx < shape[1] and 0 < ny < shape[0] and not visited[ny, nx]:
                maze[(y + ny) // 2, (x + nx) // 2] = 0  # Path
                maze[ny, nx] = 0  # Path
                visited[ny, nx] = True
                stack.append((nx, ny))
                maze_coords.append((nx, ny))
                found = True
                break
        if not found:
            stack.pop()

    # Resize the maze for display
    maze = cv2.resize(maze, (300, 300), interpolation=cv2.INTER_NEAREST)
    maze = cv2.cvtColor(maze, cv2.COLOR_GRAY2BGR)  # Convert to color image

    # Customize colors
    wall_color = (0, 0, 0)  # Black walls
    path_color = (255, 255, 255)  # White paths
    maze[np.where((maze == [255, 255, 255]).all(axis=2))] = path_color
    maze[np.where((maze == [0, 0, 0]).all(axis=2))] = wall_color

    # Mark start and end points
    start_point = (15, 15)
    end_point = (285, 285)
    start_color = (0, 255, 0)  # Green start point
    end_color = (255, 0, 0)  # Red end point
    cv2.circle(maze, start_point, 7, start_color, -1)  # Start point in green
    cv2.circle(maze, end_point, 7, end_color, -1)  # End point in red

    cv2.putText(maze, "Start", (start_point[0] - 15, start_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, start_color, 1)
    cv2.putText(maze, "End", (end_point[0] - 15, end_point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, end_color, 1)

    return maze, maze_coords



def authenticate_gaze(terminate_signal, maze_coords, gaze_coords):
    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    maze, maze_coords[:] = create_connected_maze_with_display(size=5)
    print(f"Maze coordinates: {maze_coords}")

    try:
        while not terminate_signal():
            ret, frame = webcam.read()
            if not ret:
                print("Error: Unable to capture frame.")
                continue

            gaze.refresh(frame)
            left_pupil = gaze.pupil_left_coords()
            if left_pupil:
                gaze_coords.append((int(left_pupil[0]), int(left_pupil[1])))
                print(f"Left pupil detected at: {left_pupil}")

            cv2.imshow("Webcam Feed", gaze.annotated_frame())
            cv2.imshow("Maze", maze)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        webcam.release()
        cv2.destroyAllWindows()
        print("Gaze tracking terminated.")


def scale_maze_coordinates(maze_coords, maze_size=(300, 300), grid_size=5):
    """
    Scale maze grid coordinates to pixel coordinates for comparison.
    :param maze_coords: List of maze coordinates in grid units.
    :param maze_size: Size of the maze in pixels (width, height).
    :param grid_size: Size of the maze grid (number of cells in one dimension).
    :return: Scaled coordinates in pixel units.
    """
    pixel_coords = []
    cell_width = maze_size[0] / (grid_size * 2 + 1)
    cell_height = maze_size[1] / (grid_size * 2 + 1)
    
    for x, y in maze_coords:
        pixel_x = int(x * cell_width)
        pixel_y = int(y * cell_height)
        pixel_coords.append((pixel_x, pixel_y))
    
    return pixel_coords


def compare_coordinates(maze_coords, gaze_coords, maze_size=(300, 300), grid_size=5):
    """
    Compare scaled maze and gaze coordinates to determine authentication success.
    """
    scaled_maze_coords = scale_maze_coordinates(maze_coords, maze_size, grid_size)
    print(f"Scaled Maze coordinates: {scaled_maze_coords}")
    print(f"Gaze coordinates: {gaze_coords}")

    threshold = 50  # Pixel proximity threshold
    matches = 0

    for g_coord in gaze_coords:
        for m_coord in scaled_maze_coords:
            if np.linalg.norm(np.array(g_coord) - np.array(m_coord)) < threshold:
                matches += 1
                break

    print(f"Matches: {matches}")
    return matches >= 5  # Require at least 5 matches








    







