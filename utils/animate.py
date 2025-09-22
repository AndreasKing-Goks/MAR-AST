import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.animation import FFMpegWriter

class ShipTrajectoryAnimator:
    def __init__(self, test_ship_df, obs_ship_df, test_route, obs_route,
                 waypoint_reached_times, waypoints, map_obj,
                 radius_of_acceptance, timestamps, interval=500,
                 test_drawer=None, obs_drawer=None,
                 test_headings=None, obs_headings=None):
        """
        Initializes the ShipTrajectoryAnimator.

        Parameters:
        - test_ship_df, obs_ship_df: DataFrames containing east/north positions for each timestep.
        - test_route, obs_route: Dict with keys 'east', 'north' for ship routes.
        - waypoint_reached_times: List of timestamps when waypoints become visible.
        - intermediate_waypoints: List of tuples [(east, north), ...] for obstacle ship.
        - map_obj: Object with method plot_obstacle(ax) for plotting static map obstacles.
        - radius_of_acceptance: Radius around waypoints to plot.
        - timestamps: List or array of simulation timestamps corresponding to positions.
        - interval: Animation update interval in milliseconds.
          example:
          interval=1000 -> Animations update every 1 second.
          interval=500  -> Animations update every 0.5 second.
          interval=100  -> Animations update every 0.1 second.
        """
        self.test_ship_df = test_ship_df
        self.obs_ship_df = obs_ship_df
        self.test_route = test_route
        self.obs_route = obs_route
        self.waypoint_reached_times = waypoint_reached_times
        self.waypoints = waypoints
        self.always_visible_waypoints = [waypoints[0], waypoints[-1]]
        self.intermediate_waypoints = waypoints[1:-1]
        self.map = map_obj
        self.radius_of_acceptance = radius_of_acceptance
        self.timestamps = timestamps
        self.interval = interval
        self.test_drawer = test_drawer
        self.obs_drawer = obs_drawer
        self.test_yaw_rad = np.radians(test_headings) if test_headings is not None else None
        self.obs_yaw_rad = np.radians(obs_headings) if obs_headings is not None else None

        # Prepare the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5.5))
        self.setup_plot()

    def setup_plot(self):
        # Dynamic time counter
        self.time_text = self.ax.text(0.01, 0.97, '', transform=self.ax.transAxes,
                              fontsize=12, verticalalignment='top')
        
        # Axes settings
        self.ax.set_xlim(0, 20000)
        self.ax.set_ylim(0, 10000)
        self.ax.set_aspect('equal')
        self.ax.grid(color='0.8', linestyle='-', linewidth=0.5)
        self.ax.set_title('Animated Ship Trajectory')
        self.ax.set_xlabel('East position (m)')
        self.ax.set_ylabel('North position (m)')

        # Plot static map obstacles
        self.map.plot_obstacle(self.ax)

        # Initialize ship lines
        self.test_ship_line, = self.ax.plot([], [], color='blue', label='Test Ship')
        self.obs_ship_line, = self.ax.plot([], [], color='red', label='Obstacle Ship')
        
        # Initialize ship outlines using Line2D
        self.test_ship_outline, = self.ax.plot([], [], color='blue', linewidth=1.5)
        self.obs_ship_outline, = self.ax.plot([], [], color='red', linewidth=1.5)
        
        # Initialize route points
        self.ax.plot(self.test_route['east'], self.test_route['north'], '--', color='blue', alpha=0.6)
        self.ax.scatter(self.test_route['east'], self.test_route['north'], marker='x', color='blue', alpha=0.6)

        # Containers for dynamic waypoint plotting
        self.always_visible_markers = []
        self.always_visible_circles = []
        self.waypoint_markers = []  # for dynamic/intermediate
        self.waypoint_circles = []

        for idx, (east, north) in enumerate(self.always_visible_waypoints):
            color = 'green' if idx == 1 else 'red'  # goal = green, start = red
            marker, = self.ax.plot(east, north, 'x', color=color, markersize=10)
            self.always_visible_markers.append(marker)

            circle = Circle(
                (east, north),
                self.radius_of_acceptance,
                color=color,
                alpha=0.3,
                fill=True
            )
            self.ax.add_patch(circle)
            self.always_visible_circles.append(circle)

            self.ax.legend()
            
            if idx == 0:
                self.ax.text(east, north + 200, 'START', color='red', ha='center', fontsize=10)
            elif idx == 1:
                self.ax.text(east, north + 200, 'GOAL', color='green', ha='center', fontsize=10)


    def init_animation(self):
        # Reset ship lines
        self.test_ship_line.set_data([], [])
        self.obs_ship_line.set_data([], [])

        # Clear waypoint markers and circles
        for marker in self.waypoint_markers:
            marker.remove()
        self.waypoint_markers.clear()

        for circle in self.waypoint_circles:
            circle.remove()
        self.waypoint_circles.clear()

        # Add START and GOAL labels for the test ship
        test_start_east = self.test_route['east'][0]
        test_start_north = self.test_route['north'][0]
        test_goal_east = self.test_route['east'][-1]
        test_goal_north = self.test_route['north'][-1]

        self.ax.text(test_start_east, test_start_north + 200, 'START', color='blue',
                    ha='center', fontsize=10, fontweight='bold')
        self.ax.text(test_goal_east, test_goal_north + 200, 'GOAL', color='blue',
                    ha='center', fontsize=10, fontweight='bold')

        return self.test_ship_line, self.obs_ship_line

    def animate(self, i):
        # print(f"Animating frame {i}, timestamp={self.timestamps[i]:.2f}s")
        current_time = self.timestamps[i]

        # Update ship trajectories
        self.test_ship_line.set_data(
            self.test_ship_df['east position [m]'][:i],
            self.test_ship_df['north position [m]'][:i]
        )
        self.obs_ship_line.set_data(
            self.obs_ship_df['east position [m]'][:i],
            self.obs_ship_df['north position [m]'][:i]
        )
        
        # Update test ship outline
        if self.test_drawer and self.test_yaw_rad is not None and i < len(self.test_ship_df):
            east = self.test_ship_df['east position [m]'].iloc[i]
            north = self.test_ship_df['north position [m]'].iloc[i]
            heading = self.test_yaw_rad[i]

            x_local, y_local = self.test_drawer.local_coords()
            x_rot, y_rot = self.test_drawer.rotate_coords(x_local, y_local, heading)
            x_final, y_final = self.test_drawer.translate_coords(x_rot, y_rot, north, east)

            self.test_ship_outline.set_data(y_final, x_final)  # East (x) is horizontal, North (y) is vertical

        # Update obstacle ship outline
        if self.obs_drawer and self.obs_yaw_rad is not None and i < len(self.obs_ship_df):
            east = self.obs_ship_df['east position [m]'].iloc[i]
            north = self.obs_ship_df['north position [m]'].iloc[i]
            heading = self.obs_yaw_rad[i]

            x_local, y_local = self.obs_drawer.local_coords()
            x_rot, y_rot = self.obs_drawer.rotate_coords(x_local, y_local, heading)
            x_final, y_final = self.obs_drawer.translate_coords(x_rot, y_rot, north, east)

            self.obs_ship_outline.set_data(y_final, x_final)

        self.time_text.set_text(f"Time: {current_time:.1f} s")

        return (self.test_ship_line, self.obs_ship_line,
                *self.waypoint_markers, *self.waypoint_circles,
                self.test_ship_outline, self.obs_ship_outline,
                self.time_text)

    def run_realtime(self):
        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.timestamps),
            init_func=self.init_animation,
            blit=True,
            interval=self.interval,
            repeat=False
        )

        plt.show()
    
    def run(self, fps=None, auto_show=False):
        """
        Run the animation in real time.

        Parameters:
        - fps: Optional. Frames per second for the live animation. If None, uses default interval.
        """
        if fps is not None:
            self.interval = 1000 / fps  # override interval

        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.timestamps),
            init_func=self.init_animation,
            blit=True,
            interval=self.interval,
            repeat=False
        )

        if auto_show:
            plt.show()

        
    def save(self, video_path, fps=2):
        """
        Save the animation to a video file.

        Parameters:
        - filename: Output file name (e.g., 'output.mp4')
        - fps: Frames per second in the video (use 2 for real-time if timestep is 0.5s)
        """
        print(f"Saving animation to '{video_path}' at {fps} FPS...")
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.timestamps),
            init_func=self.init_animation,
            blit=True,
            interval=self.interval,
            repeat=False
        )

        writer = FFMpegWriter(fps=fps)
        self.animation.save(video_path, writer=writer)
        print("Animation saved successfully.")
        
class RLShipTrajectoryAnimator:
    def __init__(self, test_ship_df, obs_ship_df, test_route, obs_route,
                 waypoint_sampling_times, waypoints, map_obj,
                 radius_of_acceptance, timestamps, interval=500,
                 test_drawer=None, obs_drawer=None,
                 test_headings=None, obs_headings=None,
                 is_collision_imminent_list=None,
                 is_collision_list=None):
        """
        Initializes the ShipTrajectoryAnimator.

        Parameters:
        - test_ship_df, obs_ship_df: DataFrames containing east/north positions for each timestep.
        - test_route, obs_route: Dict with keys 'east', 'north' for ship routes.
        - waypoint_reached_times: List of timestamps when waypoints become visible.
        - intermediate_waypoints: List of tuples [(east, north), ...] for obstacle ship.
        - map_obj: Object with method plot_obstacle(ax) for plotting static map obstacles.
        - radius_of_acceptance: Radius around waypoints to plot.
        - timestamps: List or array of simulation timestamps corresponding to positions.
        - interval: Animation update interval in milliseconds.
          example:
          interval=1000 -> Animations update every 1 second.
          interval=500  -> Animations update every 0.5 second.
          interval=100  -> Animations update every 0.1 second.
        """
        self.test_ship_df = test_ship_df
        self.obs_ship_df = obs_ship_df
        self.test_route = test_route
        self.obs_route = obs_route
        self.waypoint_sampling_times = waypoint_sampling_times
        self.waypoints = waypoints
        self.always_visible_waypoints = [waypoints[0], waypoints[-1]]
        self.intermediate_waypoints = waypoints[1:-1]
        self.map = map_obj
        self.radius_of_acceptance = radius_of_acceptance
        self.timestamps = timestamps
        self.interval = interval
        self.test_drawer = test_drawer
        self.obs_drawer = obs_drawer
        self.test_yaw_rad = np.radians(test_headings) if test_headings is not None else None
        self.obs_yaw_rad = np.radians(obs_headings) if obs_headings is not None else None
        self.visible_waypoint_indices = set()

        # Prepare the figure
        self.fig, self.ax = plt.subplots(figsize=(10, 5.5))
        self.setup_plot()
        
        # For warning purposes
        self.is_collision_imminent_list = None if is_collision_imminent_list == [] else is_collision_imminent_list
        self.is_collision_list = None if is_collision_list == [] else is_collision_list

    def setup_plot(self):
        # Dynamic time counter
        self.time_text = self.ax.text(0.01, 0.97, '', transform=self.ax.transAxes,
                              fontsize=12, verticalalignment='top')
        
        # Axes settings
        self.ax.set_xlim(0, 20000)
        self.ax.set_ylim(0, 10000)
        self.ax.set_aspect('equal')
        self.ax.grid(color='0.8', linestyle='-', linewidth=0.5)
        self.ax.set_title("Top View of Ships' Trajectory")
        self.ax.set_xlabel('East position (m)')
        self.ax.set_ylabel('North position (m)')

        # Plot static map obstacles
        self.map.plot_obstacle(self.ax)

        # Initialize ship lines
        self.test_ship_line, = self.ax.plot([], [], color='blue', label='Test Ship')
        self.obs_ship_line, = self.ax.plot([], [], color='red', label='Obstacle Ship')
        
        # Initialize ship outlines using Line2D
        self.test_ship_outline, = self.ax.plot([], [], color='blue', linewidth=1.5)
        self.obs_ship_outline, = self.ax.plot([], [], color='red', linewidth=1.5)
        
        # Initialize route points
        self.ax.plot(self.test_route['east'], self.test_route['north'], '--', color='blue', alpha=0.6)
        self.ax.scatter(self.test_route['east'], self.test_route['north'], marker='x', color='blue', alpha=0.6)

        # Containers for dynamic waypoint plotting
        self.always_visible_markers = []
        self.always_visible_circles = []
        self.waypoint_markers = []  # for dynamic/intermediate
        self.waypoint_circles = []

        for idx, (east, north) in enumerate(self.always_visible_waypoints):
            color = 'green' if idx == 1 else 'red'  # goal = green, start = red
            marker, = self.ax.plot(east, north, 'x', color=color, markersize=10)
            self.always_visible_markers.append(marker)

            circle = Circle(
                (east, north),
                self.radius_of_acceptance,
                color=color,
                alpha=0.3,
                fill=True
            )
            self.ax.add_patch(circle)
            self.always_visible_circles.append(circle)

            self.ax.legend()
            
            if idx == 0:
                self.ax.text(east, north + 200, 'START', color='red', ha='center', fontsize=10)
            elif idx == 1:
                self.ax.text(east, north + 200, 'GOAL', color='green', ha='center', fontsize=10)
        
        # Dynamic time counter
        self.time_text = self.ax.text(0.01, 0.97, '', transform=self.ax.transAxes,
                                    fontsize=12, verticalalignment='top')

        # Status alert textbox (collision or COLAV)
        self.status_text = self.ax.text(0.01, 0.92, '', transform=self.ax.transAxes,
                                        fontsize=12, verticalalignment='top',
                                        horizontalalignment='left',
                                        fontweight='bold')


    def init_animation(self):
        # Reset ship lines
        self.test_ship_line.set_data([], [])
        self.obs_ship_line.set_data([], [])

        # Clear waypoint markers and circles
        for marker in self.waypoint_markers:
            marker.remove()
        self.waypoint_markers.clear()

        for circle in self.waypoint_circles:
            circle.remove()
        self.waypoint_circles.clear()

        # Add START and GOAL labels for the test ship
        test_start_east = self.test_route['east'][0]
        test_start_north = self.test_route['north'][0]
        test_goal_east = self.test_route['east'][-1]
        test_goal_north = self.test_route['north'][-1]

        self.ax.text(test_start_east, test_start_north + 200, 'START', color='blue',
                    ha='center', fontsize=10, fontweight='bold')
        self.ax.text(test_goal_east, test_goal_north + 200, 'GOAL', color='blue',
                    ha='center', fontsize=10, fontweight='bold')

        return self.test_ship_line, self.obs_ship_line

    def animate(self, i):
        # print(f"Animating frame {i}, timestamp={self.timestamps[i]:.2f}s")
        current_time = self.timestamps[i]

        # Update ship trajectories
        self.test_ship_line.set_data(
            self.test_ship_df['east position [m]'][:i],
            self.test_ship_df['north position [m]'][:i]
        )
        self.obs_ship_line.set_data(
            self.obs_ship_df['east position [m]'][:i],
            self.obs_ship_df['north position [m]'][:i]
        )
        
        # Update test ship outline
        if self.test_drawer and self.test_yaw_rad is not None and i < len(self.test_ship_df):
            east = self.test_ship_df['east position [m]'].iloc[i]
            north = self.test_ship_df['north position [m]'].iloc[i]
            heading = self.test_yaw_rad[i]

            x_local, y_local = self.test_drawer.local_coords()
            x_rot, y_rot = self.test_drawer.rotate_coords(x_local, y_local, heading)
            x_final, y_final = self.test_drawer.translate_coords(x_rot, y_rot, north, east)

            self.test_ship_outline.set_data(y_final, x_final)  # East (x) is horizontal, North (y) is vertical

        # Update obstacle ship outline
        if self.obs_drawer and self.obs_yaw_rad is not None and i < len(self.obs_ship_df):
            east = self.obs_ship_df['east position [m]'].iloc[i]
            north = self.obs_ship_df['north position [m]'].iloc[i]
            heading = self.obs_yaw_rad[i]

            x_local, y_local = self.obs_drawer.local_coords()
            x_rot, y_rot = self.obs_drawer.rotate_coords(x_local, y_local, heading)
            x_final, y_final = self.obs_drawer.translate_coords(x_rot, y_rot, north, east)

            self.obs_ship_outline.set_data(y_final, x_final)

        # Reveal intermediate waypoints progressively
        for idx, reached_time in enumerate(self.waypoint_sampling_times):
            # Skip if already added
            if current_time >= reached_time and idx not in self.visible_waypoint_indices:
                east, north = self.intermediate_waypoints[idx]

                marker, = self.ax.plot(east, north, 'x', color='red', markersize=10)
                self.waypoint_markers.append(marker)

                circle = Circle(
                    (east, north),
                    self.radius_of_acceptance,
                    color='red',
                    alpha=0.3,
                    fill=True
                )
                self.ax.add_patch(circle)
                self.waypoint_circles.append(circle)
                
                self.visible_waypoint_indices.add(idx)

        self.time_text.set_text(f"Time: {current_time:.1f} s")
        
        # Update status text
        status = ""
        if self.is_collision_list is not None and i < len(self.is_collision_list):
            if self.is_collision_list[i]:
                status = "SHIP COLLISION!"
                self.status_text.set_text(status)
                self.status_text.set_color('red')
            elif self.is_collision_imminent_list is not None and self.is_collision_imminent_list[i]:
                status = "SUT COLAV activated!"
                self.status_text.set_text(status)
                self.status_text.set_color('blue')
            else:
                self.status_text.set_text("")
        
        return (self.test_ship_line, self.obs_ship_line,
                *self.waypoint_markers, *self.waypoint_circles,
                self.test_ship_outline, self.obs_ship_outline,
                self.time_text, self.status_text)

    def run_realtime(self):
        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.timestamps),
            init_func=self.init_animation,
            blit=True,
            interval=self.interval,
            repeat=False
        )

        plt.show()
    
    def run(self, fps=None):
        """
        Run the animation in real time.

        Parameters:
        - fps: Optional. Frames per second for the live animation. If None, uses default interval.
        
        """
        if fps is not None:
            self.interval = 1000 / fps  # override interval

        # Create animation
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.timestamps),
            init_func=self.init_animation,
            blit=True,
            interval=self.interval,
            repeat=False
        )
        
        # Show animation
        plt.show()

        
    def save(self, video_path, fps=2):
        """
        Save the animation to a video file.

        Parameters:
        - filename: Output file name (e.g., 'output.mp4')
        - fps: Frames per second in the video (use 2 for real-time if timestep is 0.5s)
        """
        print(f"Saving animation to '{video_path}' at {fps} FPS...")
        self.animation = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.timestamps),
            init_func=self.init_animation,
            blit=True,
            interval=self.interval,
            repeat=False
        )

        writer = FFMpegWriter(fps=fps)
        self.animation.save(video_path, writer=writer)
        print("Animation saved successfully.")
