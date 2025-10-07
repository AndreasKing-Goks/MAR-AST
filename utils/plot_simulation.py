import matplotlib.pyplot as plt
import numpy as np

def center_plot_window():
    try:
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend().lower()

        if "tkagg" in backend:
            # For TkAgg (Tkinter)
            manager.window.update_idletasks()
            screen_width = manager.window.winfo_screenwidth()
            screen_height = manager.window.winfo_screenheight()
            window_width = manager.window.winfo_width()
            window_height = manager.window.winfo_height()
            pos_x = int((screen_width - window_width) / 2)
            pos_y = int((screen_height - window_height) / 2)
            manager.window.geometry(f"+{pos_x}+{pos_y}")

        elif "qt" in backend:
            # For QtAgg, Qt5Agg, qtagg, etc.
            screen = manager.window.screen().availableGeometry()
            screen_width, screen_height = screen.width(), screen.height()
            window_width = manager.window.width()
            window_height = manager.window.height()
            pos_x = int((screen_width - window_width) / 2)
            pos_y = int((screen_height - window_height) / 2)
            manager.window.move(pos_x, pos_y)

        else:
            print(f"Centering not supported for backend: {backend}")

    except Exception as e:
        print("Could not reposition the plot window:", e)

def plot_ship_status(asset, result_df, show=False):
    fig_2, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    plt.figure(fig_2.number)  # Ensure it's the current figure
    axes = axes.flatten()  # Flatten the 2D array for easier indexing

    # Center plotting
    center_plot_window()
    
    # Plot 2.1:Speed
    speed = np.sqrt(result_df['forward speed [m/s]']**2 + result_df['sideways speed [m/s]']**2)
    axes[0].plot(result_df['time [s]'], speed)
    axes[0].axhline(y=asset.ship_model.desired_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
    axes[0].set_title(str(asset.info.name_tag + ' Speed [m/s]'))
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Forward Speed (m/s)')
    axes[0].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[0].set_xlim(left=0)

    # Plot 2.2: Rudder Angle
    axes[1].plot(result_df['time [s]'], result_df['rudder angle [deg]'])
    axes[1].set_title(asset.info.name_tag + ' Rudder angle [deg]')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Rudder angle [deg]')
    axes[1].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(-31,31)

    # Plot 2.3: Cross Track error
    axes[2].plot(result_df['time [s]'], result_df['cross track error [m]'])
    axes[2].set_title(asset.info.name_tag + ' Cross Track Error [m]')
    axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Cross track error (m)')
    axes[2].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[2].set_xlim(left=0)
    axes[2].set_ylim(-501,501)

    # Plot 2.4: Propeller Shaft Speed
    axes[3].plot(result_df['time [s]'], result_df['propeller shaft speed [rpm]'])
    axes[3].set_title(asset.info.name_tag + ' Propeller Shaft Speed [rpm]')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Propeller Shaft Speed (rpm)')
    axes[3].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[3].set_xlim(left=0)

    # Plot 2.5: Power vs Available Power
    if asset.ship_model.ship_machinery_model.operating_mode in ('PTO', 'MEC'):
        axes[4].plot(result_df['time [s]'], result_df['power me [kw]'], label="Power")
        axes[4].plot(result_df['time [s]'], result_df['available power me [kw]'], label="Available Power")
        axes[4].set_title(asset.info.name_tag + " Power vs Available Mechanical Power [kw]")
        axes[4].set_xlabel('Time (s)')
        axes[4].set_ylabel('Power (kw)')
        axes[4].legend()
        axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[4].set_xlim(left=0)
    elif asset.ship_model.ship_machinery_model.operating_mode == 'PTI':
        axes[4].plot(result_df['time [s]'], result_df['power electrical [kw]'], label="Power")
        axes[4].plot(result_df['time [s]'], result_df['available power electrical [kw]'], label="Available Power")
        axes[4].set_title(asset.info.name_tag + " Power vs Available Power Electrical [kw]")
        axes[4].set_xlabel('Time (s)')
        axes[4].set_ylabel('Power (kw)')
        axes[4].legend()
        axes[4].grid(color='0.8', linestyle='-', linewidth=0.5)
        axes[4].set_xlim(left=0)

    # Plot 2.6: Fuel Consumption
    axes[5].plot(result_df['time [s]'], result_df['fuel consumption [kg]'])
    axes[5].set_title(asset.info.name_tag + ' Fuel Consumption [kg]')
    axes[5].set_xlabel('Time (s)')
    axes[5].set_ylabel('Fuel Consumption (kg)')
    axes[5].grid(color='0.8', linestyle='-', linewidth=0.5)
    axes[5].set_xlim(left=0)

    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Show
    if show is True:
        plt.show()
        
def plot_ship_and_real_map(assets,
                           result_dfs,
                           land_gdf,
                           ocean_gdf,
                           water_gdf,
                           coast_gdf,
                           frame_gdf):
    
    # Center plotting
    center_plot_window()
    
    fig, ax = plt.subplots(figsize=(14, 8))

    # draw order: land first, then ocean/water, then coast lines, then frame boundary
    if not land_gdf.empty:
        land_gdf.plot(ax=ax, facecolor="#e8e4d8", edgecolor="#b5b2a6", linewidth=0.4, zorder=1)
    if not ocean_gdf.empty:
        ocean_gdf.plot(ax=ax, facecolor="#d9f2ff", edgecolor="#bde9ff", linewidth=0.4, alpha=0.95, zorder=2)
    if not water_gdf.empty:
        water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
    if not coast_gdf.empty:
        coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.0, zorder=3)

    # fit exactly to frame bounds, remove margins/axes so the basemap fills the figure
    minx, miny, maxx, maxy = frame_gdf.total_bounds
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_aspect("equal"); ax.set_axis_off(); ax.margins(0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    # ax.set_title("SIT simulation over real map", fontsize=12)

    # Plot 1.1: Ship trajectory with sampled route
    # Test ship
    color = ['blue', 'red', 'green', 'yellow', 'pink', 'brown', 'black', 'white', 'gray']
    
    for i, asset in enumerate(assets):
        plt.plot(result_dfs[i]['east position [m]'].to_numpy(), result_dfs[i]['north position [m]'].to_numpy(), label=asset.info.name_tag) # Trajectories
        plt.scatter(asset.ship_model.auto_pilot.navigate.east, asset.ship_model.auto_pilot.navigate.north, marker='x', color=color[i])  # Waypoints
        plt.plot(asset.ship_model.auto_pilot.navigate.east, asset.ship_model.auto_pilot.navigate.north, linestyle='--', color=color[i])  # Waypoints Line
        for x, y in zip(asset.ship_model.ship_drawings[1], asset.ship_model.ship_drawings[0]):
            plt.plot(x, y, color=color[i])

    plt.title('Ship Trajectory')
    plt.xlabel('East position (m)')
    plt.ylabel('North position (m)')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.grid(color='0.8', linestyle='-', linewidth=0.5)

    # Adjust layout for better spacing
    plt.tight_layout()