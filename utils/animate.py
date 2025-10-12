import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle, Polygon

# ---------- static helpers for the basemap ----------

def draw_real_map(ax, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf):
    # draw order: land -> ocean/water -> coasts
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.plot(ax=ax, facecolor="#e8e4d8", edgecolor="#b5b2a6", linewidth=0.4, zorder=1)
    if ocean_gdf is not None and not ocean_gdf.empty:
        ocean_gdf.plot(ax=ax, facecolor="#d9f2ff", edgecolor="#bde9ff", linewidth=0.4, alpha=0.95, zorder=2)
    if water_gdf is not None and not water_gdf.empty:
        water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
    if coast_gdf is not None and not coast_gdf.empty:
        coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.0, zorder=3)

    # fit to frame bounds
    if frame_gdf is not None and not frame_gdf.empty:
        minx, miny, maxx, maxy = frame_gdf.total_bounds
        ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0)
    ax.grid(False)


# ---------- animator with GDF basemap on the left ----------

class Animator:
    def __init__(self,
                 assets,                 # list of ships: see previous message
                 env,                    # dict (wave/current/wind) for focus ship
                 map_gdfs,
                 focus_ship_index=0,     # System under test index under assets
                 interval_ms=500,
                 status=None):

        self.assets = assets
        self.t = assets[focus_ship_index].simulation_results['time [s]']
        self.interval = interval_ms
        self.focus = focus_ship_index
        
        land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf = map_gdfs

        # Environment (same as before)
        dy = assets[focus_ship_index].ship_model.simulation_results['wave force north [N]']
        dx = assets[focus_ship_index].ship_model.simulation_results['wave force east [N]']
        self.wave_dir_deg  = np.rad2deg(np.arctan2(dy, dx))
        self.wave_mag      = np.sqrt(dy**2 + dx**2)
        
        self.curr_dir_deg  = assets[focus_ship_index].ship_model.simulation_results['current dir [deg]']
        self.curr_speed    = assets[focus_ship_index].ship_model.simulation_results['current speed [m/s]']
        
        self.wind_dir_deg  = assets[focus_ship_index].ship_model.simulation_results['wind dir [deg]']
        self.wind_speed    = assets[focus_ship_index].ship_model.simulation_results['wind speed [m/s]']

        status = status or {}
        self.colav_active  = np.asarray(status.get('colav_active',  []))
        self.collision     = np.asarray(status.get('collision',     []))
        self.nav_failure   = np.asarray(status.get('nav_failure',   []))
        self.grounding     = np.asarray(status.get('grounding',     []))

        # Precompute headings (deg->rad)
        for a in self.assets:
            a['headings_rad'] = (np.radians(a['headings_deg'])
                                 if a.get('headings_deg') is not None else None)

        # --- Figure & layout
        self.fig = plt.figure(figsize=(15, 7.5))
        gs = GridSpec(nrows=3, ncols=2, width_ratios=[2.2, 1.0], height_ratios=[1.0, 1.0, 0.35], figure=self.fig)

        self.ax_map = self.fig.add_subplot(gs[:, 0])
        self.ax_polar_wave = self.fig.add_subplot(gs[0, 1], projection='polar')
        self.ax_polar_curr = self.fig.add_subplot(gs[1, 1], projection='polar')
        self.ax_status     = self.fig.add_subplot(gs[2, 1])
        self.ax_status.set_axis_off()

        # --- Draw static basemap once (replaces self.map)
        draw_real_map(self.ax_map, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf)
        self.ax_map.set_title("Top View: All Ships")  # optional title; axes stay off otherwise

        # --- Prepare dynamic artists on top of map
        self.ship_trails, self.ship_outlines, self.route_handles, self.route_scats = [], [], [], []
        for a in self.assets:
            color = a.get('color', None)
            trail, = self.ax_map.plot([], [], '-', linewidth=1.8, color=color, zorder=10)   # dynamic
            outline, = self.ax_map.plot([], [], linewidth=1.5, color=color, zorder=12)     # dynamic
            self.ship_trails.append(trail)
            self.ship_outlines.append(outline)
            if a.get('route') is not None:
                r = a['route']
                h, = self.ax_map.plot(r['east'], r['north'], '--', alpha=0.5, color=color, zorder=6)  # static
                s = self.ax_map.scatter(r['east'], r['north'], marker='x', alpha=0.6, color=color, zorder=7)
                self.route_handles.append(h); self.route_scats.append(s)

        # Timestamp
        self.time_text = self.ax_map.text(0.01, 0.97, '', transform=self.ax_map.transAxes,
                                          fontsize=12, va='top', zorder=20)

        # --- Polars & status
        self._setup_polar(self.ax_polar_wave, "Wave load & heading",
                          self._max_safe([self.wave_mag, self.curr_speed, self.wind_speed]))
        self._setup_polar(self.ax_polar_curr, "Current & Wind",
                          self._max_safe([self.wave_mag, self.curr_speed, self.wind_speed]))
        self.wave_ship_line, = self.ax_polar_wave.plot([], [], linewidth=2)
        self.wave_vec_line,  = self.ax_polar_wave.plot([], [], linewidth=2)
        self.curr_ship_line, = self.ax_polar_curr.plot([], [], linewidth=2)
        self.curr_vec_line,  = self.ax_polar_curr.plot([], [], linewidth=2)     # current
        self.wind_vec_line,  = self.ax_polar_curr.plot([], [], linewidth=2, linestyle='--')  # wind
        self.wave_ship_icon = None
        self.curr_ship_icon = None

        # Status boxes (text + hatch, not color-only)
        self.status_boxes = []
        labels = ["COLAV", "Collision", "Nav Failure", "Grounding"]
        for i, lab in enumerate(labels):
            x0, w = i/4.0, 1/4.0
            rect = Rectangle((x0, 0.05), w-0.02, 0.9, fill=False, linewidth=1.2)
            self.ax_status.add_patch(rect)
            txt = self.ax_status.text(x0 + w/2 - 0.01, 0.5, f"{lab}\nINACTIVE",
                                      ha='center', va='center', fontsize=10)
            self.status_boxes.append((rect, txt))
        self.ax_status.set_xlim(0, 1); self.ax_status.set_ylim(0, 1)

    # ---------- setup helpers ----------
    def _setup_polar(self, ax, title, max_r):
        ax.set_title(title)
        ax.set_rlim(0, max(1.0, max_r))
        ax.set_rlabel_position(225)
        ax.grid(alpha=0.3)

    # ---------- animation core ----------
    def init_animation(self):
        for t in self.ship_trails: t.set_data([], [])
        for o in self.ship_outlines: o.set_data([], [])
        self.time_text.set_text('')
        for line in [self.wave_ship_line, self.wave_vec_line,
                     self.curr_ship_line, self.curr_vec_line, self.wind_vec_line]:
            line.set_data([], [])
        self._clear_ship_icon(self.ax_polar_wave)
        self._clear_ship_icon(self.ax_polar_curr)
        for _, txt in self.status_boxes:
            txt.set_text(txt.get_text().split('\n')[0] + "\nINACTIVE")
        return (*self.ship_trails, *self.ship_outlines,
                self.wave_ship_line, self.wave_vec_line,
                self.curr_ship_line, self.curr_vec_line, self.wind_vec_line,
                self.time_text)

    def animate(self, i):
        # Left: trails + outlines
        for a, trail, outline in zip(self.assets, self.ship_trails, self.ship_outlines):
            df = a['df']
            trail.set_data(df['east position [m]'][:i], df['north position [m]'][:i])
            if a.get('drawer') and a.get('headings_rad') is not None and i < len(df):
                east  = df['east position [m]'].iloc[i]
                north = df['north position [m]'].iloc[i]
                psi   = a['headings_rad'][i]
                xL, yL = a['drawer'].local_coords()
                xR, yR = a['drawer'].rotate_coords(xL, yL, psi)
                xF, yF = a['drawer'].translate_coords(xR, yR, north, east)
                outline.set_data(yF, xF)

        # Right: polars for focus ship
        focus = self.assets[self.focus]
        hdg_deg = focus['headings_deg'][i] if focus.get('headings_deg') is not None else 0.0
        hdg = np.deg2rad(hdg_deg)

        # Wave panel
        if self.wave_mag.size:
            self._set_ray(self.wave_ship_line, hdg, 0.8*self.ax_polar_wave.get_rmax())
            if self.wave_dir_deg.size:
                self._set_ray(self.wave_vec_line, np.deg2rad(self.wave_dir_deg[i]), self.wave_mag[i])
        self._draw_ship_icon(self.ax_polar_wave, hdg)

        # Current+Wind panel
        self._set_ray(self.curr_ship_line, hdg, 0.8*self.ax_polar_curr.get_rmax())
        if self.curr_speed.size and self.curr_dir_deg.size:
            self._set_ray(self.curr_vec_line, np.deg2rad(self.curr_dir_deg[i]), self.curr_speed[i])
        if self.wind_speed.size and self.wind_dir_deg.size:
            self._set_ray(self.wind_vec_line, np.deg2rad(self.wind_dir_deg[i]), self.wind_speed[i])

        # Time + status
        self.time_text.set_text(f"Time: {self.t[i]:.1f} s")
        self._status_update(0, self._get(self.colav_active, i), "ACTIVE", "INACTIVE")
        self._status_update(1, self._get(self.collision,    i), "YES",    "NO")
        self._status_update(2, self._get(self.nav_failure,  i), "ACTIVE", "INACTIVE")
        self._status_update(3, self._get(self.grounding,    i), "YES",    "NO")

        return (*self.ship_trails, *self.ship_outlines,
                self.wave_ship_line, self.wave_vec_line,
                self.curr_ship_line, self.curr_vec_line, self.wind_vec_line,
                self.time_text)

    # ---------- small helpers ----------
    @staticmethod
    def _max_safe(seq_list):
        vals = []
        for s in seq_list:
            if hasattr(s, 'size') and s.size:
                vals.append(np.nanmax(s))
        return max(vals) if vals else 1.0

    @staticmethod
    def _set_ray(line, theta, r):
        line.set_data([theta, theta], [0, max(0, r)])

    def _draw_ship_icon(self, ax_polar, heading_rad):
        self._clear_ship_icon(ax_polar)
        r = 0.25 * ax_polar.get_rmax(); dth = np.deg2rad(18)
        verts = [(heading_rad, r), (heading_rad - dth, 0.6*r), (heading_rad + dth, 0.6*r)]
        xy = np.array([(rho*np.cos(th), rho*np.sin(th)) for th, rho in verts])
        patch = Polygon(xy, closed=True, fill=False, linewidth=1.5)
        ax_polar.add_patch(patch)
        if ax_polar is self.ax_polar_wave: self.wave_ship_icon = patch
        else: self.curr_ship_icon = patch

    def _clear_ship_icon(self, ax_polar):
        ref = self.wave_ship_icon if ax_polar is self.ax_polar_wave else self.curr_ship_icon
        if ref is not None: ref.remove()
        if ax_polar is self.ax_polar_wave: self.wave_ship_icon = None
        else: self.curr_ship_icon = None

    @staticmethod
    def _get(arr, i):  # safe boolean fetch
        return bool(arr[i]) if hasattr(arr, 'size') and arr.size and i < arr.size else False

    def _status_update(self, idx, on, on_text, off_text):
        rect, txt = self.status_boxes[idx]
        label = txt.get_text().split('\n')[0]
        txt.set_text(f"{label}\n{on_text if on else off_text}")
        rect.set_fill(on); rect.set_alpha(0.15 if on else 0.0); rect.set_hatch('////' if on else '')

    # ---------- public ----------
    def run(self, fps=None):
        if fps is not None: self.interval = 1000 / fps
        self.animation = FuncAnimation(self.fig, self.animate,
                                       frames=len(self.t),
                                       init_func=self.init_animation,
                                       blit=True, interval=self.interval, repeat=False)
        plt.show()

    def save(self, path, fps=2):
        self.animation = FuncAnimation(self.fig, self.animate,
                                       frames=len(self.t),
                                       init_func=self.init_animation,
                                       blit=True, interval=self.interval, repeat=False)
        writer = FFMpegWriter(fps=fps)
        self.animation.save(path, writer=writer)
