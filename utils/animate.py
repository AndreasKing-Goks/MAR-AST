import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon, Rectangle
from matplotlib import gridspec as mpl_gs

# =========================
# Shared helpers
# =========================

def draw_real_map(ax, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf):
    # Draw order
    if land_gdf is not None and not land_gdf.empty:
        land_gdf.plot(ax=ax, facecolor="#e8e4d8", edgecolor="#b5b2a6", linewidth=0.4, zorder=1)
    if ocean_gdf is not None and not ocean_gdf.empty:
        ocean_gdf.plot(ax=ax, facecolor="#d9f2ff", edgecolor="#bde9ff", linewidth=0.4, alpha=0.95, zorder=2)
    if water_gdf is not None and not water_gdf.empty:
        water_gdf.plot(ax=ax, facecolor="#a0c8f0", edgecolor="#74a8d8", linewidth=0.4, alpha=0.95, zorder=2)
    if coast_gdf is not None and not coast_gdf.empty:
        coast_gdf.plot(ax=ax, color="#2f7f3f", linewidth=1.0, zorder=3)

    # Keep the full frame extent (avoid aspect cropping)
    if frame_gdf is not None and not frame_gdf.empty:
        minx, miny, maxx, maxy = land_gdf.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    ax.set_autoscale_on(False)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_anchor('C')
    ax.set_axis_off()
    ax.margins(0)
    ax.grid(False)

def setup_nav_polar(ax, rmax):
    ax.set_rlim(0, max(1.0, rmax))
    ax.set_theta_zero_location('N')   # 0° at North
    ax.set_theta_direction(-1)        # CW positive
    ax.set_thetagrids([0,45,90,135,180,225,270,315],
                      labels=['0','45°','90°','135°','180°','-135°','-90°','-45°'])
    ax.set_rlabel_position(225)
    ax.grid(alpha=0.3)

def _wrap180(a_deg):  # [-180, 180]
    return ((a_deg + 180.0) % 360.0) - 180.0

def _math_to_nav(deg_math):  # 0E/CCW+ -> 0N/CW+
    return _wrap180(90.0 - deg_math)

def ship_vertices_polar(ship_draw, heading_rad_nav, ax, size_frac=0.25):
    """Vertices in (theta, r) for a ShipDraw outline, rotated by heading."""
    x_loc, y_loc = ship_draw.local_coords()
    rmax = ax.get_rmax()
    scale = (size_frac * rmax) / (ship_draw.l / 2.0)
    x = x_loc * scale; y = y_loc * scale
    theta_math = np.arctan2(y, x)
    theta_nav  = (np.pi/2.0) - theta_math + heading_rad_nav
    r = np.hypot(x, y)
    return np.column_stack([theta_nav, r])

# -------- Window positioning (backend-safe-ish) --------
def get_screen_size(fig):
    m = fig.canvas.manager
    # Tk
    try:
        return m.window.winfo_screenwidth(), m.window.winfo_screenheight()
    except Exception:
        pass
    # Qt
    try:
        scr = m.window.screen() if hasattr(m.window, "screen") else m.window.windowHandle().screen()
        geo = scr.availableGeometry()
        return geo.width(), geo.height()
    except Exception:
        pass
    # Fallback
    return 1920, 1080

def move_figure(fig, x, y, w=None, h=None):
    m = fig.canvas.manager
    backend = plt.get_backend().lower()
    try:
        if 'tk' in backend:
            if w is None or h is None:
                m.window.wm_geometry(f"+{int(x)}+{int(y)}")
            else:
                m.window.geometry(f"{int(w)}x{int(h)}+{int(x)}+{int(y)}")
        elif 'qt' in backend:
            if w is not None and h is not None:
                m.window.setGeometry(int(x), int(y), int(w), int(h))
            else:
                m.window.move(int(x), int(y))
        elif 'wx' in backend:
            m.window.SetPosition((int(x), int(y)))
            if w is not None and h is not None:
                m.window.SetSize((int(w), int(h)))
    except Exception:
        # Last-ditch: ignore failures silently
        pass

def animate_side_by_side(fig_left, fig_right, left_frac=0.68, height_frac=0.92, gap_px=12, show=True):
    """Center both windows on screen, same height, side-by-side."""
    sw, sh = get_screen_size(fig_left)
    H = int(sh * height_frac)
    W_left  = int(sw * left_frac) - gap_px // 2
    W_right = int(sw * (1 - left_frac)) - gap_px // 2
    total_w = W_left + gap_px + W_right
    x0 = (sw - total_w) // 2
    y0 = (sh - H) // 2
    move_figure(fig_left,  x0,              y0, W_left,  H)
    move_figure(fig_right, x0 + W_left + gap_px, y0, W_right, H)
    if show:
        plt.show()

# =========================
# 1) MAP-ONLY ANIMATOR (with status strip)
# =========================

class MapAnimator:
    """
    Map-only animation with basemap + ships + status strip (for one chosen asset).
    """
    def __init__(self, assets, map_gdfs, interval_ms=500, status_asset_index=0):
        self.assets = assets
        self.interval = interval_ms
        self.focus_idx = status_asset_index
        self.stop_requested = False
        land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf = map_gdfs

        def sr(a, k): return np.asarray(a.ship_model.simulation_results[k])
        self.t = sr(self.assets[0], 'time [s]')  # assume aligned

        # Cache series per asset
        self.series = []
        for a in assets:
            s = a.ship_model
            sim = s.simulation_results
            self.series.append({
                'east':  np.asarray(sim['east position [m]']),
                'north': np.asarray(sim['north position [m]']),
                'hdg':   np.radians(np.asarray(sim['yaw angle [deg]'])),
                'label': getattr(getattr(a, 'info', None), 'name_tag', 'Ship'),
                'drawer': getattr(s, 'draw', None),
                'route_e': getattr(getattr(s, 'auto_pilot', None), 'navigate', None).east
                           if hasattr(getattr(s, 'auto_pilot', None), 'navigate') else None,
                'route_n': getattr(getattr(s, 'auto_pilot', None), 'navigate', None).north
                           if hasattr(getattr(s, 'auto_pilot', None), 'navigate') else None,
            })

        # Status arrays for the focus ship
        fm = self.assets[self.focus_idx].ship_model
        self.colav = np.asarray(getattr(fm, 'colav_active_array', []))
        self.colli = np.asarray(getattr(fm, 'collision_array', []))
        self.navfl = np.asarray(getattr(fm, 'nav_failure_array', []))
        self.ground= np.asarray(getattr(fm, 'grounding_array', []))

        # ---- Figure: map + status strip (taller than before)
        self.fig = plt.figure(figsize=(17.5, 9.8), constrained_layout=True)
        gs = self.fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1.0, 0.12], hspace=0.0)
        self.ax_map   = self.fig.add_subplot(gs[0, 0])
        self.ax_stat  = self.fig.add_subplot(gs[1, 0]); self.ax_stat.set_axis_off()

        draw_real_map(self.ax_map, land_gdf, ocean_gdf, water_gdf, coast_gdf, frame_gdf)
        self.ax_map.set_title("Ship Trajectory")

        # Static routes
        for s in self.series:
            if s['route_e'] is not None and s['route_n'] is not None:
                self.ax_map.plot(s['route_e'], s['route_n'], '--', alpha=0.55, zorder=6)
                self.ax_map.scatter(s['route_e'], s['route_n'], marker='x', alpha=0.7, zorder=7)

        # Dynamic artists
        self.trails, self.outlines = [], []
        for s in self.series:
            tr, = self.ax_map.plot([], [], '-', lw=1.8, zorder=10, label=s['label'])
            ol, = self.ax_map.plot([], [], lw=1.4, zorder=12)
            self.trails.append(tr); self.outlines.append(ol)
        self.ax_map.legend(loc='upper right')
        self.time_text = self.ax_map.text(0.01, 0.97, '', transform=self.ax_map.transAxes, fontsize=12, va='top')

        # Status boxes (bottom)
        self.boxes = []
        for i, lab in enumerate(["COLAV", "Collision", "Nav Failure", "Grounding"]):
            x0, w = i/4.0, 1/4.0
            rect = Rectangle((x0, 0.05), w-0.02, 0.9, fill=False, lw=1.2)
            self.ax_stat.add_patch(rect)
            txt = self.ax_stat.text(x0 + w/2 - 0.01, 0.5, f"{lab}\nINACTIVE", ha='center', va='center', fontsize=10)
            self.boxes.append((rect, txt))
        self.ax_stat.set_xlim(0, 1); self.ax_stat.set_ylim(0, 1)

        # 'q' to stop
        self.fig.canvas.mpl_connect('key_press_event', lambda e: self.request_stop() if e.key == 'q' else None)

    def request_stop(self):
        self.stop_requested = True
        try: self.animation.event_source.stop()
        except Exception: pass

    def _status(self, idx, on, on_text, off_text):
        rect, txt = self.boxes[idx]
        label = txt.get_text().split('\n')[0]
        txt.set_text(f"{label}\n{on_text if on else off_text}")
        rect.set_fill(on); rect.set_alpha(0.15 if on else 0.0); rect.set_hatch('////' if on else '')

    @staticmethod
    def _on(arr, i): return bool(arr[i]) if hasattr(arr, 'size') and i < arr.size else False

    def init_animation(self):
        for tr in self.trails: tr.set_data([], [])
        for ol in self.outlines: ol.set_data([], [])
        self.time_text.set_text('')
        # reset status
        for _, txt in self.boxes:
            txt.set_text(txt.get_text().split('\n')[0] + "\nINACTIVE")
        return (*self.trails, *self.outlines, self.time_text)

    def animate(self, i):
        if self.stop_requested:
            return (*self.trails, *self.outlines, self.time_text)
        for s, tr, ol in zip(self.series, self.trails, self.outlines):
            tr.set_data(s['east'][:i], s['north'][:i])
            if s['drawer'] is not None and i < len(s['east']):
                east, north, psi = s['east'][i], s['north'][i], s['hdg'][i]
                xL, yL = s['drawer'].local_coords()
                xR, yR = s['drawer'].rotate_coords(xL, yL, psi)
                xF, yF = s['drawer'].translate_coords(xR, yR, north, east)
                ol.set_data(yF, xF)  # (East, North)
        self.time_text.set_text(f"Time: {self.t[i]:.1f} s")

        # status for focus ship
        self._status(0, self._on(self.colav, i), "ACTIVE", "INACTIVE")
        self._status(1, self._on(self.colli, i), "YES",    "NO")
        self._status(2, self._on(self.navfl, i), "ACTIVE", "INACTIVE")
        self._status(3, self._on(self.ground, i), "YES",    "NO")

        return (*self.trails, *self.outlines, self.time_text)

    def run(self, fps=None, show=False, repeat=False):
        if fps is not None: self.interval = 1000 / fps
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                       init_func=self.init_animation, blit=True,
                                       interval=self.interval, repeat=repeat)
        if show:
            plt.show()

    def save(self, path, fps=2):
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                       init_func=self.init_animation, blit=True,
                                       interval=self.interval, repeat=False)
        writer = FFMpegWriter(fps=fps)
        self.animation.save(path, writer=writer)

# =========================
# 2) POLAR-ONLY ANIMATOR (unchanged)
# =========================

class PolarAnimator:
    """
    Three polar panels (wave/current/wind) for one focus ship.
    Each axis gets its own rmax based on its own data.
    """
    def __init__(self, focus_asset, interval_ms=500, rpad=0.05, rmin=1.0):
        self.a = focus_asset
        self.interval = interval_ms

        sim = self.a.ship_model.simulation_results
        self.t = np.asarray(sim['time [s]'])
        self.hdg_rad = np.radians(np.asarray(sim['yaw angle [deg]']))

        # Wave: from forces (math -> nav)
        dy = np.asarray(sim['wave force north [N]'])
        dx = np.asarray(sim['wave force east [N]'])
        wave_deg_math = np.rad2deg(np.arctan2(dy, dx))
        self.wave_dir_deg = _math_to_nav(wave_deg_math)
        self.wave_mag     = np.sqrt(dx**2 + dy**2)

        # Current & wind (assumed already 0N/CW+; convert here if not)
        self.curr_dir_deg = np.asarray(sim['current dir [deg]'])
        self.curr_speed   = np.asarray(sim['current speed [m/s]'])
        self.wind_dir_deg = np.asarray(sim['wind dir [deg]'])
        self.wind_speed   = np.asarray(sim['wind speed [m/s]'])

        # ----- per-axis rmax (with small padding) -----
        def _rmax(arr):
            if hasattr(arr, "size") and arr.size:
                m = float(np.nanmax(arr))
                return max(rmin, m * (1.0 + rpad))
            return rmin

        self.rmax_wave = _rmax(self.wave_mag)
        self.rmax_curr = _rmax(self.curr_speed)
        self.rmax_wind = _rmax(self.wind_speed)

        # Figure
        self.fig = plt.figure(figsize=(6.5, 9), constrained_layout=True)
        self.ax_wave  = self.fig.add_subplot(311, projection='polar')
        self.ax_curr  = self.fig.add_subplot(312, projection='polar')
        self.ax_wind  = self.fig.add_subplot(313, projection='polar')

        # Apply per-axis nav setup
        setup_nav_polar(self.ax_wave, self.rmax_wave); self.ax_wave.set_title("Wave & heading")
        setup_nav_polar(self.ax_curr, self.rmax_curr); self.ax_curr.set_title("Current & heading")
        setup_nav_polar(self.ax_wind, self.rmax_wind); self.ax_wind.set_title("Wind & heading")

        # Rays + ship icons
        self.wave_ship, = self.ax_wave.plot([], [], lw=2)
        self.wave_vec,  = self.ax_wave.plot([], [], lw=2)
        self.curr_ship, = self.ax_curr.plot([], [], lw=2)
        self.curr_vec,  = self.ax_curr.plot([], [], lw=2)
        self.wind_ship, = self.ax_wind.plot([], [], lw=2)
        self.wind_vec,  = self.ax_wind.plot([], [], lw=2)

        self.ship_icon_wave = None
        self.ship_icon_curr = None
        self.ship_icon_wind = None

        self.time_text = self.fig.text(0.02, 0.98, '', ha='left', va='top')

    @staticmethod
    def _set_ray(line, theta, r):
        line.set_data([theta, theta], [0, max(0, r)])

    def init_animation(self):
        for ln in [self.wave_ship, self.wave_vec, self.curr_ship, self.curr_vec, self.wind_ship, self.wind_vec]:
            ln.set_data([], [])
        for icon in [self.ship_icon_wave, self.ship_icon_curr, self.ship_icon_wind]:
            try:
                if icon is not None: icon.remove()
            except Exception:
                pass
        self.ship_icon_wave = self.ship_icon_curr = self.ship_icon_wind = None
        self.time_text.set_text('')
        return self.wave_ship, self.wave_vec, self.curr_ship, self.curr_vec, self.wind_ship, self.wind_vec

    def animate(self, i):
        hdg = self.hdg_rad[i]

        # Wave panel
        self._set_ray(self.wave_ship, hdg, 0.8*self.ax_wave.get_rmax())
        if self.wave_mag.size and self.wave_dir_deg.size:
            self._set_ray(self.wave_vec, np.deg2rad(self.wave_dir_deg[i]), self.wave_mag[i])
        verts = ship_vertices_polar(getattr(self.a.ship_model, 'draw', None) or self.a.ship_model,
                                    hdg, self.ax_wave, 0.25)
        if self.ship_icon_wave is None:
            self.ship_icon_wave = Polygon(verts, closed=True, fill=False, lw=1.5)
            self.ship_icon_wave.set_transform(self.ax_wave.transData); self.ax_wave.add_patch(self.ship_icon_wave)
        else:
            self.ship_icon_wave.set_xy(verts)

        # Current panel
        self._set_ray(self.curr_ship, hdg, 0.8*self.ax_curr.get_rmax())
        if self.curr_speed.size and self.curr_dir_deg.size:
            self._set_ray(self.curr_vec, np.deg2rad(self.curr_dir_deg[i]), self.curr_speed[i])
        verts = ship_vertices_polar(getattr(self.a.ship_model, 'draw', None) or self.a.ship_model,
                                    hdg, self.ax_curr, 0.25)
        if self.ship_icon_curr is None:
            self.ship_icon_curr = Polygon(verts, closed=True, fill=False, lw=1.5)
            self.ship_icon_curr.set_transform(self.ax_curr.transData); self.ax_curr.add_patch(self.ship_icon_curr)
        else:
            self.ship_icon_curr.set_xy(verts)

        # Wind panel
        self._set_ray(self.wind_ship, hdg, 0.8*self.ax_wind.get_rmax())
        if self.wind_speed.size and self.wind_dir_deg.size:
            self._set_ray(self.wind_vec, np.deg2rad(self.wind_dir_deg[i]), self.wind_speed[i])
        verts = ship_vertices_polar(getattr(self.a.ship_model, 'draw', None) or self.a.ship_model,
                                    hdg, self.ax_wind, 0.25)
        if self.ship_icon_wind is None:
            self.ship_icon_wind = Polygon(verts, closed=True, fill=False, lw=1.5)
            self.ship_icon_wind.set_transform(self.ax_wind.transData); self.ax_wind.add_patch(self.ship_icon_wind)
        else:
            self.ship_icon_wind.set_xy(verts)

        self.time_text.set_text(f"t = {self.t[i]:.1f} s")
        return self.wave_ship, self.wave_vec, self.curr_ship, self.curr_vec, self.wind_ship, self.wind_vec

    def run(self, fps=None, show=False, repeat=False):
        if fps is not None: self.interval = 1000 / fps
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                       init_func=self.init_animation, blit=True,
                                       interval=self.interval, repeat=repeat)
        if show:
            plt.show()

    def save(self, path, fps=2):
        self.animation = FuncAnimation(self.fig, self.animate, frames=len(self.t),
                                       init_func=self.init_animation, blit=True,
                                       interval=self.interval, repeat=False)
        writer = FFMpegWriter(fps=fps)
        self.animation.save(path, writer=writer)

