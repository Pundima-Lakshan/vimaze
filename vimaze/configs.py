maze_generation_options = {
    'defaults': {
        'rows': 10,
        'cols': 10,
    },
    'algorithms': {
        'prim': {
            'name': 'Prim\'s',
        },
        'binary': {
            'name': 'Binary',
        },
        'kruskal': {
            'name': 'Kruskal',
        },
        'rec_backtrack': {
            'name': 'Recursive Backtracking',
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
        'window_height': 620
    },
    'grid_config': {
        'rows': [
            {
                'weight': 1,
                'minsize': 500
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
                'column': 1,
                'sticky': "nsew"
            },
            'corner_radius': 0,
            'border_width': 0,
            'width': 300,
            'height': 600,
            'bg': "blue",
            'tabs': [
                {
                    'name': 'Generation',
                    'controls': [
                        {
                            'type': 'dropdown',
                            'label': 'Algorithm',
                            'options': [
                                {'label': maze_generation_options['algorithms']['prim']['name'], 'value': 'prim'},
                                {'label': maze_generation_options['algorithms']['binary']['name'], 'value': 'binary'},
                                {'label': maze_generation_options['algorithms']['kruskal']['name'], 'value': 'kruskal'},
                                {'label': maze_generation_options['algorithms']['rec_backtrack']['name'],
                                 'value': 'rec_backtrack'},
                            ],
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
                            'type': 'button',
                            'text': 'Generate Maze',
                            'command': 'generate_maze',  # Callback function name
                        },
                        {
                            'type': 'slider',
                            'label': 'Speed',
                            'from_': 1,
                            'to': 10,
                            'default_value': 5,
                            'command': 'set_speed',  # Callback function name
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
                {
                    'name': 'Image2',
                    'controls': [
                        {
                            'type': 'button',
                            'text': 'Generate Maze',
                            'command': 'generate_maze',  # Callback function name
                        },
                    ]
                }
            ],
        },
        'maze_frame': {
            'grid_options': {
                'row': 0,
                'column': 0,
                'sticky': "nsew"
            },
            'corner_radius': 0,
            'border_width': 0,
            'width': 600,
            'height': 600,
            'bg': "green"
        },
    },
    'canvases': {
        'maze_canvas': {
            'width': 600,
            'height': 600,
            'bg': "red",
            'pack_config': {
                'fill': "both",
                'expand': True
            },
        },
    },
}
