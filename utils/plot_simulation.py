import matplotlib.pyplot as plt
import numpy as np

def wrap_angle_deg(x):
    # wrap to (-pi, pi]
    return (x + 180) % (360) - 180

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

def plot_ship_status(asset, result_df, plot_env_load=True, show=False):
    def _ax_style(ax, *, xlabel=False, ylabel=None, xlim_left=0, ypct_series=None):
        ax.grid(color='0.85', linestyle='-', linewidth=0.5)
        ax.set_xlim(left=xlim_left)
        if ylabel:
            ax.set_ylabel(ylabel)
        if xlabel:
            ax.set_xlabel('Time (s)')
        if ypct_series is not None:
            # percentile-based y-limits to avoid one spike flattening the trace
            lo, hi = np.nanpercentile(ypct_series, [1, 99])
            if np.isfinite(lo) and np.isfinite(hi) and lo != hi:
                pad = 0.05 * (hi - lo)
                ax.set_ylim(lo - pad, hi + pad)

    # Global readability
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.1,
    })

    # ---------- FIGURE 1: SHIP STATUS ----------
    fig_1, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 8), constrained_layout=True)
    axes = axes.flatten()
    plt.figure(fig_1.number)

    # Center plotting
    try:
        center_plot_window()
    except Exception:
        pass

    t = result_df['time [s]']
    nm = str(asset.info.name_tag)

    # 1. Speed (use resultant to match your code)
    speed = np.sqrt(result_df['forward speed [m/s]']**2 + result_df['sideways speed [m/s]']**2)
    axes[0].plot(t, speed, label='Speed')
    axes[0].axhline(y=asset.ship_model.desired_speed, color='red', linestyle='--', linewidth=1.5, label='Desired Forward Speed')
    axes[0].set_title(f'{nm} Speed')
    _ax_style(axes[0], xlabel=True, ylabel='Speed (m/s)')
    axes[0].legend(loc='upper right', frameon=False)

    # 2. Yaw angle
    axes[1].plot(t, wrap_angle_deg(result_df['yaw angle [deg]']))
    axes[1].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
    axes[1].set_title(f'{nm} Yaw Angle')
    _ax_style(axes[1], xlabel=True, ylabel='Yaw angle (deg)')
    axes[1].set_ylim(-180.0, 180.0)

    # 3. Rudder angle
    axes[2].plot(t, result_df['rudder angle [deg]'])
    axes[2].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
    axes[2].set_title(f'{nm} Rudder Angle')
    _ax_style(axes[2], xlabel=True, ylabel='Rudder angle (deg)')
    axes[2].set_ylim(-np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max), np.rad2deg(asset.ship_model.ship_machinery_model.rudder_ang_max))

    # 4. Cross-track error
    axes[3].plot(t, result_df['cross track error [m]'])
    axes[3].axhline(y=0.0, color='red', linestyle='--', linewidth=1.5)
    axes[3].set_title(f'{nm} Cross-Track Error')
    _ax_style(axes[3], xlabel=True, ylabel='Cross-track error (m)')
    axes[3].set_ylim(-asset.ship_model.cross_track_error_tolerance, asset.ship_model.cross_track_error_tolerance)

    # 5. Propeller shaft speed
    axes[4].plot(t, result_df['propeller shaft speed [rpm]'])
    axes[4].set_title(f'{nm} Propeller Shaft Speed')
    _ax_style(axes[4], xlabel=True, ylabel='Shaft speed (rpm)')
    
    # 6. Thrust Force
    axes[5].plot(t, result_df['thrust force [kN]'])
    axes[5].set_title(f'{nm} Thrust force')
    _ax_style(axes[5], xlabel=True, ylabel='Thrust force (kN)')

    # 7. Power vs available power (mode-dependent)
    ax6 = axes[6]
    mode = asset.ship_model.ship_machinery_model.operating_mode
    if mode in ('PTO', 'MEC'):
        ax6.plot(t, result_df['power me [kw]'], label='Power')
        ax6.plot(t, result_df['available power me [kw]'], color='red', linestyle='--', label='Available Power')
        ax6.set_title(f'{nm} Power vs Available Mechanical Power')
        _ax_style(ax6, xlabel=True, ylabel='Power (kW)')
        ax6.legend(frameon=False)
    elif mode == 'PTI':
        ax6.plot(t, result_df['power electrical [kw]'], label='Power')
        ax6.plot(t, result_df['available power electrical [kw]'], color='red', linestyle='--', label='Available Power')
        ax6.set_title(f'{nm} Power vs Available Electrical Power')
        _ax_style(ax6, xlabel=True, ylabel='Power (kW)')
        ax6.legend(frameon=False)
    else:
        ax6.set_visible(False)

    # 8. Fuel consumption
    axes[7].plot(t, result_df['fuel consumption [kg]'])
    axes[7].set_title(f'{nm} Fuel Consumption')
    _ax_style(axes[7], xlabel=True, ylabel='Fuel (kg)')

    # ---------- FIGURES 2–4: ENVIRONMENT LOADS (optional) ----------
    if plot_env_load:
        # --- WAVES: Fx, Fy, Mz ---
        fig_w, aw = plt.subplots(1, 3, figsize=(16, 5), sharex=True, constrained_layout=True)
        cols = ['wave force north [N]', 'wave force east [N]']
        max_value = result_df[cols].abs().to_numpy().max()
        
        aw[0].plot(t, result_df['wave force north [N]'])
        aw[0].set_title('Wave Force North'); _ax_style(aw[0], ylabel='Force (N)', ypct_series=result_df['wave force north [N]'])
        aw[0].set_ylim(-max_value, max_value)

        aw[1].plot(t, result_df['wave force east [N]'])
        aw[1].set_title('Wave Force East'); _ax_style(aw[1], ypct_series=result_df['wave force east [N]'])
        aw[1].set_ylim(-max_value, max_value)

        aw[2].plot(t, result_df['wave moment [Nm]'])
        aw[2].set_title('Wave Moment'); _ax_style(aw[2], xlabel=True, ylabel='Moment (Nm)', ypct_series=result_df['wave moment [Nm]'])

        fig_w.suptitle(f'Wave loads on {nm}', y=1.02)

        # --- WIND: speed, dir, Fx, Fy, Mz ---
        fig_wind, axw = plt.subplots(2, 3, figsize=(18, 8), sharex=True, constrained_layout=True)
        axw = axw.ravel()
        
        cols = ['wind force north [N]', 'wind force east [N]']
        max_value = result_df[cols].abs().to_numpy().max()

        axw[0].plot(t, result_df['wind speed [m/s]'])
        axw[0].set_title('Wind Speed'); _ax_style(axw[0], ylabel='Speed (m/s)', ypct_series=result_df['wind speed [m/s]'])

        axw[1].plot(t, result_df['wind dir [deg]'])
        axw[1].set_title('Wind Direction'); _ax_style(axw[1], ypct_series=result_df['wind dir [deg]'])
        axw[1].set_ylim(-180, 180)

        axw[2].axis('off')  # spacer to make a clean 2x3 grid

        axw[3].plot(t, result_df['wind force north [N]'])
        axw[3].set_title('Wind Force North'); _ax_style(axw[3], ylabel='Force (N)', ypct_series=result_df['wind force north [N]'])
        axw[3].set_ylim(-max_value, max_value)

        axw[4].plot(t, result_df['wind force east [N]'])
        axw[4].set_title('Wind Force East'); _ax_style(axw[4], ypct_series=result_df['wind force east [N]'])
        axw[4].set_ylim(-max_value, max_value)

        axw[5].plot(t, result_df['wind moment [Nm]'])
        axw[5].set_title('Wind Moment'); _ax_style(axw[5], xlabel=True, ylabel='Moment (Nm)', ypct_series=result_df['wind moment [Nm]'])

        fig_wind.suptitle(f'Wind field & loads on {nm}', y=1.02)

        # --- CURRENT: speed, dir ---
        fig_c, ac = plt.subplots(1, 2, figsize=(12, 5), sharex=True, constrained_layout=True)

        ac[0].plot(t, result_df['current speed [m/s]'])
        ac[0].set_title('Current Speed'); _ax_style(ac[0], ylabel='Speed (m/s)', ypct_series=result_df['current speed [m/s]'])

        ac[1].plot(t, result_df['current dir [deg]'])
        ac[1].set_title('Current Direction'); _ax_style(ac[1], xlabel=True, ylabel='Angle in NED (deg)')
        ac[1].set_ylim(-180, 180)

        fig_c.suptitle(f'Current field on {nm}', y=1.02)

    if show:
        plt.show()
        
def plot_ship_and_real_map(assets, result_dfs, map_gdfs=None, show=False):
    fig, ax = plt.subplots(figsize=(16, 9))  # temp; we’ll resize to aspect
    
    if map_gdfs is not None:
        frame_gdf, ocean_gdf, land_gdf, coast_gdf, water_gdf = map_gdfs
        if not land_gdf.empty:
            land_gdf.plot(ax=ax, facecolor="#e8e4d8", edgecolor="#b5b2a6", linewidth=0.4, zorder=1)
        if not ocean_gdf.empty:
            ocean_gdf.plot(ax=ax, facecolor="#d9f2ff", edgecolor="#bde9ff", linewidth=0.4, alpha=0.95, zorder=2)
        if not water_gdf.empty:
            water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
        if not coast_gdf.empty:
            coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.0, zorder=3)

        # --- fit to frame & keep aspect ---
        minx, miny, maxx, maxy = frame_gdf.total_bounds
        dx, dy = (maxx - minx), (maxy - miny)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Resize figure to match map aspect (no letterbox)
        fig_w = 16.0
        fig_h = fig_w * (dy / dx) if dx > 0 else 9.0
        fig.set_size_inches(fig_w, fig_h, forward=True)

    # Full-bleed canvas
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_position([0, 0, 1, 1])
    
    # -------- enforce 1:1 scale from DATA --------
    # If no map was set, autoscale to the plotted data first
    ax.relim(); ax.autoscale_view()
    # Now lock aspect so 1 unit in x == 1 unit in y
    ax.set_aspect('equal', adjustable='box')   # robust when limits come from data

    # --- ships & routes ---
    palette = ['#1f77b4', '#d62728', '#2ca02c', '#bcbd22', '#e377c2', '#8c564b', '#000000', '#7f7f7f', '#17becf']
    for i, asset in enumerate(assets):
        c = palette[i % len(palette)]
        ax.plot(result_dfs[i]['east position [m]'].to_numpy(),
                result_dfs[i]['north position [m]'].to_numpy(),
                lw=2, alpha=0.95, label=asset.info.name_tag, zorder=5)
        if asset.ship_model.auto_pilot is not None:
            ax.plot(asset.ship_model.auto_pilot.navigate.east,
                    asset.ship_model.auto_pilot.navigate.north,
                    linestyle='--', lw=1.2, alpha=0.85, color=c, zorder=5)
            ax.scatter(asset.ship_model.auto_pilot.navigate.east,
                    asset.ship_model.auto_pilot.navigate.north,
                    marker='x', s=28, linewidths=1.2, color=c, zorder=6)
        for x, y in zip(asset.ship_model.ship_drawings[1], asset.ship_model.ship_drawings[0]):
            ax.plot(x, y, color=c, lw=1.0, zorder=7)

    lg = ax.legend(loc='upper right', frameon=True)
    lg.get_frame().set_alpha(0.95)
    
    ax.set_axis_on()
    fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.07)
    ax.set_title('Ship Trajectory')
    ax.set_xlabel('East position (m)')
    ax.set_ylabel('North position (m)')
    ax.grid(True, linewidth=0.5, alpha=0.4)

    if show:
        plt.show()