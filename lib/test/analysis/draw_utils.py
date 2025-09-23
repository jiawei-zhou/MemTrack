COLOR = {}
LINE_STYLE = {}
MARKER_STYLE = ['o', 'v', '<', '*', 'D', 'x', '.', 'x', '<', '.']
def color_line(trackers):
    color = {}
    line = {}
    color_style = ((1, 0, 0),
            (0, 1, 0),
            (1, 0.5, 0.5),
            (1, 0, 1),
            (0  , 162/255, 232/255),
            (0.5, 0.5, 0.5),
            (0, 0, 1),
            (0, 1, 1),
            (136/255, 0  , 21/255),
            (255/255, 127/255, 39/255),
            (177/255,238/255,13/255),
            (1,0,127/255),
            (102/255,178/255,1),
            (0, 0, 0))
    line_style = ['-', '--', ':', '-', '--', ':', '-', '--', ':', '-','--','-',':']
    for id,track in enumerate(trackers):
        color[track.name] = color_style[id]
        line[track.name] = line_style[id]
    COLOR.update(color)
    LINE_STYLE.update(line)
    return COLOR