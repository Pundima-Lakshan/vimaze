import logging

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from vimaze.configs import solver_app_options
from vimaze.maze import Maze
from tkinter import messagebox

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

        # Input variables
        self.maze_rows_str = ctk.StringVar()
        self.maze_cols_str = ctk.StringVar()
        self.animation_speed_str = ctk.StringVar()
        self.costs_str = ctk.StringVar()
        self.maze_start_pos_str = ctk.StringVar()
        self.maze_end_pos_str = ctk.StringVar()
        
        # Image processing variables
        self.image_path_str = ctk.StringVar()
        self.processor_type_str = ctk.StringVar(value="Standard Processor")
        self.invert_binary_str = ctk.StringVar(value="false")
        self.wall_threshold_str = ctk.StringVar(value="127")
        self.cell_size_str = ctk.StringVar(value="20")
        self.debug_mode = ctk.BooleanVar(value=False)

        for frame_name, frame_config in solver_app_options['frames'].items():
            frame = ctk.CTkFrame(self.root, corner_radius=frame_config['corner_radius'],
                                 border_width=frame_config['border_width'], fg_color=frame_config['bg'])
            frame.grid(row=frame_config['grid_options']['row'], rowspan=frame_config['grid_options']['rowspan'],
                       column=frame_config['grid_options']['column'],
                       sticky=frame_config['grid_options']['sticky'])
            frame.grid_propagate(False)

            if frame_name == "controls_frame":
                self.controls_frame = frame
                self.tabview = None
                self.add_tabs(frame_config.get('tabs', []))
            elif frame_name == "maze_frame":
                self.maze_frame = frame
            elif frame_name == 'animate_frame':
                self.animate_frame = frame
                self.add_controls(frame, frame_config.get('controls', []))
            elif frame_name == 'cost_frame':
                self.cost_frame = frame
                self.add_controls(frame, frame_config.get('controls', []))

        for canvas_name, canvas_config in solver_app_options['canvases'].items():
            frame = self.maze_frame
            if canvas_name == "maze_canvas":
                frame = self.maze_frame

            canvas = ctk.CTkCanvas(frame, bg=canvas_config['bg'], borderwidth=0,
                                   width=canvas_config['width'],
                                   height=canvas_config['height'])
            canvas.pack(fill=canvas_config['pack_config']['fill'], expand=canvas_config['pack_config']['expand'])

            if canvas_name == "maze_canvas":
                self.maze_canvas = canvas
        
        # self.maze_canvas.bind("<Motion>", doSomething)
        self.root.bind("<Configure>", self.on_resize)

        self.maze = Maze(self.maze_canvas, self)

    def run(self):
        logging.debug("Running SolverApp")
        self.root.mainloop()

    def on_resize(self, event):
        # logging.debug(f"Resizing SolverApp {event}")

        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        self.controls_frame.configure(
            width=solver_app_options['frames']['controls_frame']['width'],
            height=max(solver_app_options['frames']['controls_frame']['height'], window_height))

        self.maze_frame.configure(
            width=max(solver_app_options['frames']['maze_frame']['width'],
                      int(window_width - solver_app_options['frames']['controls_frame']['width'])),
            height=max(solver_app_options['frames']['maze_frame']['height'], window_height))

    def add_tabs(self, tabs_config):
        if not self.controls_frame:
            return

        container = ctk.CTkFrame(self.controls_frame, fg_color=solver_app_options['frames']['controls_frame']['bg'],
                                 width=solver_app_options['frames']['controls_frame']['width'], )
        container.pack(expand=True, fill="y", pady=20)  # Add some padding at the top and bottom
        container.pack_propagate(False)

        # Create a CTkTabview inside the controls_frame
        self.tabview = ctk.CTkTabview(container, fg_color=solver_app_options['frames']['controls_frame']['bg'])
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

        text_variable = None
        if config['key'] == 'maze_rows':
            text_variable = self.maze_rows_str
        elif config['key'] == 'maze_cols':
            text_variable = self.maze_cols_str
        elif config['key'] == 'animation_speed':
            text_variable = self.animation_speed_str
        elif config['key'] == 'costs':
            text_variable = self.costs_str
        elif config['key'] == 'maze_start_pos':
            text_variable = self.maze_start_pos_str
        elif config['key'] == 'maze_end_pos':
            text_variable = self.maze_end_pos_str
        elif config['key'] == 'image_path':
            text_variable = self.image_path_str
        elif config['key'] == 'invert_binary':
            text_variable = self.invert_binary_str
        elif config['key'] == 'wall_threshold':
            text_variable = self.wall_threshold_str
        elif config['key'] == 'cell_size':
            text_variable = self.cell_size_str
        else:
            text_variable = ctk.StringVar()

        input_field = ctk.CTkEntry(parent, textvariable=text_variable)
        input_field.delete(0, 'end')  # Clear any existing text
        input_field.insert(0, config.get('default_value', ''))
        input_field.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding
        
    def add_dropdown(self, parent, config):
        """Add a dropdown menu to the parent frame."""
        # Add a label for the dropdown
        label = ctk.CTkLabel(parent, text=config.get('label', 'Dropdown'))
        label.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

        dropdown = ctk.CTkOptionMenu(parent, values=config.get('values', []),
                                     command=lambda value: self.handle_dropdown_change(config.get('command'), value))
        dropdown.set(config.get('default_value', ''))
        dropdown.pack(pady=5, padx=10, fill="x")  # Fill horizontally with padding

    # Common button handlers
    def handle_button_click(self, command):
        """Handle button click events."""
        logging.debug(f"Button clicked: {command}")

        if command == "gen_display_algo_maze":
            self.gen_display_algo_maze()
        elif command == 'animate_last_action':
            self.animate_last_action()
        elif command == 'stop_animation':
            self.stop_animation()
        elif command == 'solve_maze':
            self.solve_display_maze()
        elif command == 'select_maze_image':
            self.select_maze_image()
        elif command == 'process_maze_image':
            self.process_maze_image()
        
    def handle_slider_change(self, command, value):
        """Handle slider value changes."""
        logging.debug(f"Slider changed: {command} = {value}")

    def handle_dropdown_change(self, command, value):
        """Handle dropdown value changes."""
        logging.debug(f"Dropdown changed: {command} = {value}")

        if command == "set_maze_gen_algorithm":
            self.set_maze_gen_algorithm(value)
        elif command == "set_maze_solving_algorithm":
            self.set_maze_solving_algorithm(value)
        elif command == "set_processor_type":
            self.processor_type_str.set(value)
            
    def select_maze_image(self):
        """
        Open a file dialog to select a maze image.
        """
        file_path = filedialog.askopenfilename(
            title="Select Maze Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")],
            parent=self.root  # Use the main window as parent
        )
        
        # Update the path variable if a file was selected
        if file_path:
            self.image_path_str.set(file_path)
            logging.debug(f"Selected image: {file_path}")
            
            # Also update the input field directly
            for tab in self.tabview.winfo_children():
                if isinstance(tab, ctk.CTkFrame):
                    for widget in tab.winfo_children():
                        if isinstance(widget, ctk.CTkEntry) and widget.cget("textvariable") == self.image_path_str:
                            widget.delete(0, 'end')
                            widget.insert(0, file_path)
    
    def process_maze_image(self):
        """
        Process the selected maze image and initialize the maze.
        """
        image_path = self.image_path_str.get()
        
        if not image_path:
            messagebox.showerror("Error", "Please select an image file first.")
            return
        
        try:
            # Get processor type
            processor_type = "simple" if self.processor_type_str.get() == "Simple Processor" else "standard"
               
            # Configure the maze image processor through the maze instance
            self.maze.init_from_image_with_params(
                image_path, 
                processor_type=processor_type,
                invert_binary=(self.invert_binary_str.get().lower() == "true"),
                wall_threshold=int(self.wall_threshold_str.get()) if self.wall_threshold_str.get().isdigit() else 127,
                cell_size=int(self.cell_size_str.get()) if self.cell_size_str.get().isdigit() else 20,
                debug_mode=self.debug_mode.get()
            )
            
            # Show success message
            messagebox.showinfo("Success", f"Maze loaded successfully. Size: {self.maze.rows}x{self.maze.cols}")
            
        except Exception as e:
            # Display error message
            messagebox.showerror("Error", f"Failed to process maze image: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full traceback for debugging
        
    def toggle_debug_mode(self):
        """
        Toggle debug mode for image processing.
        """
        current_value = self.debug_mode.get()
        self.debug_mode.set(not current_value)
        # Show message to user
        if current_value:
            messagebox.showinfo("Debug Mode", "Debug mode enabled. Debug images will be saved to the 'debug' folder.")
        else:
            messagebox.showinfo("Debug Mode", "Debug mode disabled.")

    # Handler functions for the controls

    def set_maze_gen_algorithm(self, value):
        logging.debug(f"Setting maze generation algorithm to: {value}")

        self.maze.set_maze_gen_algorithm(value)

    def set_maze_solving_algorithm(self, value):
        logging.debug(f"Setting maze solving algorithm to: {value}")

        self.maze.set_maze_solving_algorithm(value)

    def gen_display_algo_maze(self):
        logging.debug(f"Generating maze: {int(self.maze_rows_str.get()), int(self.maze_cols_str.get())}")

        self.maze.gen_algo_maze(int(self.maze_rows_str.get()), int(self.maze_cols_str.get()))
        self.maze_canvas.delete("all")
        self.maze.display_maze()

    def animate_last_action(self):
        logging.debug(f"Animation last action")

        self.maze.animator.animate(int(self.animation_speed_str.get()))

    def stop_animation(self):
        logging.debug(f"Animation last action")

        self.maze.animator.stop_animation()

    def solve_display_maze(self):
        logging.debug(f"Start solving maze")

        start_pos = tuple(int(x) for x in self.maze_start_pos_str.get().split(", "))
        end_pos = tuple(int(x) for x in self.maze_end_pos_str.get().split(", "))

        self.maze.solve_maze((start_pos[0], start_pos[1]), (end_pos[0], end_pos[1]))
        self.maze_canvas.delete("all")
        self.maze.display_maze()
        self.maze.display_path(self.maze.solver.solved_path)
