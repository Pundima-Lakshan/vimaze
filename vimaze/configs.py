maze_generation_options = {
    'defaults': {
        'rows': 10,
        'cols': 10,
    },
    'algorithms': {
        'prim': {
            'name': 'Prim\'s',
        },
    }
}

maze_solving_options = {
    'algorithms': {
        'dfs': {
            'name': 'DFS',
        },
    }
}

maze_display_options = {
    'speed': 5,
    'cell_color': "white",
    'cell_outline': "white smoke",
    'wall_color': "blue",
    'outline_width': 2,
    'cell_size': 20,
    'offset': 10,
}

solver_app_options = {
    'window': {
        'title': "Maze Solver",
        'window_width': 920,
        'window_height': 740
    },
    'grid_config': {
        'rows': [
            {
                'weight': 1,
                'minsize': 400
            },
            {
                'weight': 1,
                'minsize': 200
            },
        ],
        'cols': [
            {
                'weight': 1,
                'minsize': 600
            },
            {
                'weight': 1,
                'minsize': 300
            },
        ],
    },
    'frames': {
        'controls_frame': {
            'grid_options': {
                'row': 0,
                'rowspan': 1,
                'column': 1,
                'sticky': "nsew"
            },
            'corner_radius': 0,
            'border_width': 0,
            'width': 300,
            'height': 350,
            'bg': "gainsboro",
            'tabs': [
                {
                    'name': 'Generation',
                    'controls': [
                        {
                            'type': 'dropdown',
                            'label': 'Algorithm',
                            'values': ['Prim\'s'],
                            'default_value': '',
                            'command': 'set_maze_gen_algorithm'
                        },
                        {
                            'type': 'input',
                            'key': 'maze_rows',
                            'label': 'No of rows',
                            'default_value': 10,
                        },
                        {
                            'type': 'input',
                            'key': 'maze_cols',
                            'label': 'No of columns',
                            'default_value': 10,
                        },
                        {
                            'type': 'button',
                            'text': 'Generate Maze',
                            'command': 'gen_display_algo_maze',  # Callback function name
                        },
                    ]
                },
                {
                    'name': 'Solving',
                    'controls': [
                        {
                            'type': 'dropdown',
                            'label': 'Algorithm',
                            'values': ['DFS', 'BFS'],
                            'default_value': '',
                            'command': 'set_maze_solving_algorithm'
                        },
                        {
                            'type': 'input',
                            'key': 'maze_start_pos',
                            'label': 'Start Position (0, 0)',
                            'default_value': '0, 0',
                        },
                        {
                            'type': 'input',
                            'key': 'maze_end_pos',
                            'label': 'End Position (9, 9)',
                            'default_value': '9, 9',
                        },
                        {
                            'type': 'button',
                            'text': 'Solve Maze',
                            'command': 'solve_maze',  # Callback function name
                        },
                    ]
                },
                {
                    'name': 'Image',
                    'controls': [
                        {
                            'type': 'button',
                            'text': 'Generate Maze',
                            'command': 'generate_maze',  # Callback function name
                        },
                    ]
                },
            ],
        },
        'animate_frame': {
            'grid_options': {
                'row': 1,
                'rowspan': 1,
                'column': 1,
                'sticky': "nsew"
            },
            'corner_radius': 0,
            'border_width': 0,
            'width': 300,
            'height': 200,
            'bg': "gainsboro",
            'controls': [
                {
                    'type': 'input',
                    'key': 'animation_speed',
                    'label': 'Delay in ms',
                    'default_value': 50,
                },
                {
                    'type': 'button',
                    'text': 'Animate Last Operation',
                    'command': 'animate_last_action',  # Callback function name
                },
                {
                    'type': 'button',
                    'text': 'Stop animation',
                    'command': 'stop_animation',  # Callback function name
                },
            ]
        },
        'cost_frame': {
            'grid_options': {
                'row': 1,
                'rowspan': 1,
                'column': 0,
                'sticky': "nsew"
            },
            'corner_radius': 0,
            'border_width': 0,
            'width': 300,
            'height': 200,
            'bg': "gainsboro",
            'controls': [
                {
                    'type': 'input',
                    'key': 'costs',
                    'label': 'Costs (operation, algorithm, cost)',
                    'default_value': '',
                },
            ]
        },
        'maze_frame': {
            'grid_options': {
                'row': 0,
                'rowspan': 1,
                'column': 0,
                'sticky': "nsew"
            },
            'corner_radius': 0,
            'border_width': 0,
            'width': 600,
            'height': 400,
            'bg': "gainsboro"
        },
    },
    'canvases': {
        'maze_canvas': {
            'width': 600,
            'height': 600,
            'bg': "sky blue",
            'pack_config': {
                'fill': "both",
                'expand': True
            },
        },
    },
}

maze_animator_options = {
    'generation': {
        'prims': {
            'action_colors': {
                'visited_update': 'white',
                'frontier_update': 'pink',
                'frontier_select': 'yellow',
                'maze_cell_select': 'green',
                'maze_cell_deselect': 'white',
                'node_connect': 'white'
            }
        }
    }
}