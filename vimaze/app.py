import logging

import customtkinter as ctk

from vimaze.configs import solver_app_options
from vimaze.maze import Maze

logging.basicConfig(level=logging.DEBUG)


class SolverApp:
    def __init__(self):
        logging.debug("Initializing SolverApp")

        self.root = ctk.CTk()

        self.root.title(solver_app_options['window']['title'])
        self.root.geometry(
            f"{solver_app_options['window']['window_width']}x{solver_app_options['window']['window_height']}")

        for index, row in enumerate(solver_app_options['grid_config']['rows']):
            self.root.grid_rowconfigure(index, weight=row['weight'], minsize=row['minsize'])

        for index, col in enumerate(solver_app_options['grid_config']['cols']):
            self.root.grid_columnconfigure(index, weight=col['weight'], minsize=col['minsize'])

        self.maze_rows_str = ctk.StringVar()
        self.maze_cols_str = ctk.StringVar()

        for frame_name, frame_config in solver_app_options['frames'].items():
            frame = ctk.CTkFrame(self.root, corner_radius=frame_config['corner_radius'],
                                 border_width=frame_config['border_width'], fg_color=frame_config['bg'])
            frame.grid(row=frame_config['grid_options']['row'], column=frame_config['grid_options']['column'],
                       sticky=frame_config['grid_options']['sticky'])
            frame.grid_propagate(False)

            if frame_name == "controls_frame":
                self.controls_frame = frame
                self.tabview = None
                self.add_tabs(frame_config.get('tabs', []))
            elif frame_name == "maze_frame":
                self.maze_frame = frame

        for canvas_name, canvas_config in solver_app_options['canvases'].items():
            frame = self.maze_frame
            if canvas_name == "maze_canvas":
                frame = self.maze_frame

            canvas = ctk.CTkCanvas(frame, bg=canvas_config['bg'],
                                   width=canvas_config['width'],
                                   height=canvas_config['height'])
            canvas.pack(fill=canvas_config['pack_config']['fill'], expand=canvas_config['pack_config']['expand'])

            if canvas_name == "maze_canvas":
                self.maze_canvas = canvas

        self.root.bind("<Configure>", self.on_resize)

        self.maze = Maze(self.maze_canvas)

    def on_resize(self, event):
        # logging.debug(f"Resizing SolverApp {event}")

        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        self.controls_frame.configure(
            width=max(solver_app_options['frames']['controls_frame']['width'], int(window_width * 0.4)),
            height=max(solver_app_options['frames']['controls_frame']['height'], window_height))

        self.maze_frame.configure(
            width=max(solver_app_options['frames']['maze_frame']['width'], int(window_width * 0.6)),
            height=max(solver_app_options['frames']['maze_frame']['height'], window_height))

    def add_tabs(self, tabs_config):
        if not self.controls_frame:
            return

        container = ctk.CTkFrame(self.controls_frame, fg_color="green",
                                 width=solver_app_options['frames']['controls_frame']['width'], )
        container.pack(expand=True, fill="y", pady=20)  # Add some padding at the top and bottom
        container.pack_propagate(False)

        # Create a CTkTabview inside the controls_frame
        self.tabview = ctk.CTkTabview(container)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs dynamically
        for tab_config in tabs_config:
            tab_name = tab_config.get('name', 'Tab')
            tab = self.tabview.add(tab_name)  # Add a new tab
            self.add_controls(tab, tab_config.get('controls', []))  # Add controls to the tab

    def add_controls(self, parent, controls_config):
        for control_config in controls_config:
            control_type = control_config.get('type')
            if control_type == 'button':
                self.add_button(parent, control_config)
            elif control_type == 'slider':
                self.add_slider(parent, control_config)
            elif control_type == 'input':
                self.add_input(parent, control_config)
            elif control_type == 'dropdown':
                self.add_dropdown(parent, control_config)

    def add_button(self, parent, config):
        """Add a button to the parent frame."""
        button = ctk.CTkButton(parent, text=config.get('text', 'Button'),
                               command=lambda: self.handle_button_click(config.get('command')))
        button.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

    def add_slider(self, parent, config):
        """Add a slider to the parent frame."""
        label = ctk.CTkLabel(parent, text=config.get('label', 'Slider'))
        label.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

        slider = ctk.CTkSlider(parent, from_=config.get('from_', 0), to=config.get('to', 100),
                               command=lambda value: self.handle_slider_change(config.get('command'), value))
        slider.set(config.get('default_value', 50))
        slider.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

    def add_input(self, parent, config):
        """Add an input field to the parent frame."""
        label = ctk.CTkLabel(parent, text=config.get('label', 'Input'))
        label.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

        text_variable = ctk.StringVar()
        if config['key'] == 'maze_rows':
            text_variable = self.maze_rows_str
        elif config['key'] == 'maze_cols':
            text_variable = self.maze_cols_str

        input_field = ctk.CTkEntry(parent, textvariable=text_variable)
        input_field.insert(0, config.get('default_value', ''))
        input_field.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

    def add_dropdown(self, parent, config):
        """Add a dropdown menu to the parent frame with label-value pairs."""
        # Add a label for the dropdown
        label = ctk.CTkLabel(parent, text=config.get('label', 'Dropdown'))
        label.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

        # Extract labels and values from the configuration
        options = config.get('options', [])
        labels = [option['label'] for option in options]  # List of display labels
        values = {option['label']: option['value'] for option in options}  # Map labels to values

        # Create the dropdown with display labels
        dropdown = ctk.CTkOptionMenu(parent, values=labels,
                                     command=lambda label_value: self.handle_dropdown_change(
                                         config.get('command'), values[label_value]  # Pass the corresponding value
                                     ))

        # Set the default value (find the label corresponding to the default value)
        default_value = config.get('default_value', '')
        default_label = next((option['label'] for option in options if option['value'] == default_value), '')
        dropdown.set(default_label)

    # Common button handlers

    def handle_button_click(self, command):
        """Handle button click events."""
        logging.debug(f"Button clicked: {command}")
        # Add your logic here based on the command

        if command == "gen_display_algo_maze":
            self.gen_display_algo_maze()

    def handle_slider_change(self, command, value):
        """Handle slider value changes."""
        logging.debug(f"Slider changed: {command} = {value}")
        # Add your logic here based on the command

    def handle_dropdown_change(self, command, value):
        """Handle dropdown value changes."""
        logging.debug(f"Dropdown changed: {command} = {value}")
        # Add your logic here based on the command

        if command == "set_maze_gen_algorithm":
            self.set_maze_gen_algorithm(value)

    # Handler functions for the controls

    def set_maze_gen_algorithm(self, value):
        logging.debug(f"Setting maze generation algorithm to: {value}")
        # Add your logic here

        self.maze.set_maze_gen_algorithm(value)

    def gen_display_algo_maze(self):
        logging.debug(f"Generating maze: {int(self.maze_rows_str.get()), int(self.maze_cols_str.get())}")
        # Add your logic here

        self.maze.gen_algo_maze(int(self.maze_rows_str.get()), int(self.maze_cols_str.get()))
        self.maze_canvas.delete("all")
        self.maze.display_maze()

    def run(self):
        logging.debug("Running SolverApp")
        self.root.mainloop()
