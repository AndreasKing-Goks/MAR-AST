import matplotlib.pyplot as plt

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
