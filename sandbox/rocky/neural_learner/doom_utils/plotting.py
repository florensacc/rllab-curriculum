import numpy as np


def parse_map(content):
    if isinstance(content, bytes):
        content = content.decode()
    parts = content.replace('\n', '').replace(' ', '').split('}')

    parsed_parts = []
    for part in parts:
        if len(part) > 0:
            part = part.split('{')
            type_part = part[0].split(';')[-1]
            attrs = dict([x.split('=') for x in part[1].split(';') if len(x) > 0])
        parsed_parts.append(dict(attrs, klass=type_part))
    return parsed_parts


def get_vertices(parsed_map):
    vertices = []
    for part in parsed_map:
        if part['klass'] == 'vertex':
            vertices.append((int(part['x']), int(part['y'])))
    vertices = np.asarray(vertices)
    return vertices


def get_target_coords(parsed_map, side_length):
    target_coords = []
    for idx, part in enumerate(parsed_map):
        if part['klass'] == 'vertex':
            if int(part['x']) % side_length != 0 or int(part['y']) % side_length != 0:
                target_coords.append((int(part['x']), int(part['y'])))
    return np.mean(target_coords, axis=0)


def plot_textmap(parsed_map, agent_trajs, out_width, out_height, side_length, padding=20):
    import gizeh
    import cv2

    vertices = get_vertices(parsed_map)

    min_x, min_y = np.min(vertices, axis=0)
    max_x, max_y = np.max(vertices, axis=0)

    height = max_x - min_x
    width = max_y - min_y

    surface = gizeh.Surface(width=width, height=height, bg_color=(1, 1, 1))

    def rescale_point(x, y):
        tx = int((int(x) - min_x) / (max_x - min_x) * (height - 2 * padding) + padding)
        ty = int((int(y) - min_y) / (max_y - min_y) * (width - 2 * padding) + padding)
        return ty, tx

    vertices = []

    for idx, part in enumerate(parsed_map):
        if part['klass'] == 'vertex':
            vertices.append(part)
        if part['klass'] == 'linedef':
            v1 = vertices[int(part['v1'])]
            v2 = vertices[int(part['v2'])]

            pt1 = (int(v1['x']), int(v1['y']))
            pt2 = (int(v2['x']), int(v2['y']))
            if pt1[0] % side_length == 0 and \
                                    pt1[1] % side_length == 0 and \
                                    pt2[0] % side_length == 0 and \
                                    pt2[1] % side_length == 0:
                pt1 = rescale_point(*pt1)
                pt2 = rescale_point(*pt2)
                line = gizeh.polyline([pt1, pt2], stroke=(0, 0, 0), stroke_width=5, line_cap='round')
                line.draw(surface)

        elif part['klass'] == 'thing':
            if int(part['type']) == 1:
                # character
                sx, sy = int(part['x']), int(part['y'])

    tx, ty = get_target_coords(parsed_map, side_length=side_length)

    start_color = (0, 0, 1, 0.5)
    end_color = (1, 0, 0, 0.5)

    # colors = [
    #     (0.3, 0.3, 0.3),
    #     (0.3, 0.3, 0.3),
    #     # (0, 0, 1),
    #     # (0, 1, 0, 0.5),
    # ]

    for traj in agent_trajs:
        if len(traj) > 1:

            traj = np.asarray(traj)

            # from sandbox.rocky.neural_learner.scripts.visual_nav.piecewise_linear import prune_traj
            # traj = prune_traj(traj)

            def draw_arrow(surface, pt1, pt2, color, arrow_length=10, arrow_angle=np.pi / 9):
                pt1 = np.asarray(pt1)
                pt2 = np.asarray(pt2)
                dx, dy = pt2 - pt1
                ang = np.arctan2(dy, dx)
                ang1 = np.pi / 2 - ang - arrow_angle
                ang2 = np.pi / 2 - ang + arrow_angle

                dx1 = np.sin(ang1) * arrow_length
                dy1 = np.cos(ang1) * arrow_length
                dx2 = np.sin(ang2) * arrow_length
                dy2 = np.cos(ang2) * arrow_length

                ang_pt1 = [pt2[0] - dx1, pt2[1] - dy1]
                ang_pt2 = [pt2[0] - dx2, pt2[1] - dy2]

                gizeh.polyline([pt1, pt2], stroke=color, stroke_width=2).draw(surface)

                if dx ** 2 + dy ** 2 > arrow_length ** 2:
                    gizeh.polyline([pt2, ang_pt1, ang_pt2], fill=color, stroke=color, stroke_width=1).draw(surface)

            for idx, (pt1, pt2) in enumerate(zip(traj[:-1], traj[1:])):
                color = np.asarray(end_color) * idx / len(traj) + np.asarray(start_color) * (1 - idx / len(traj))
                draw_arrow(surface, rescale_point(*pt1), rescale_point(*pt2), color=color)
        else:
            gizeh.circle(r=5, xy=rescale_point(*traj[0]), fill=start_color).draw(surface)
                # gizeh.polyline(
                #     [rescale_point(*pt1), rescale_point(*pt2)],
                #     # [rescale_point(x, y) for x, y in traj],
                #     stroke=color,
                #     stroke_width=5,
                #     close_path=False,
                #     line_cap='round',
                #     line_join='round',
                # ).draw(surface)

    margin = 10

    # Draw the start region
    # sx, sy = agent_trajs[0][0]
    # gizeh.polyline(
    #     [
    #         rescale_point(sx - side_length / 2 + margin, sy - side_length / 2 + margin),
    #         rescale_point(sx - side_length / 2 + margin, sy + side_length / 2 - margin),
    #         rescale_point(sx + side_length / 2 - margin, sy + side_length / 2 - margin),
    #         rescale_point(sx + side_length / 2 - margin, sy - side_length / 2 + margin),
    #     ],
    #     stroke=(0, 0, 1, 0.5),
    #     stroke_width=5,
    #     line_cap='round',
    #     close_path=True,
    # ).draw(surface)
    # Draw the goal region
    gizeh.polyline(
        [
            rescale_point(tx - side_length / 2 + margin, ty - side_length / 2 + margin),
            rescale_point(tx - side_length / 2 + margin, ty + side_length / 2 - margin),
            rescale_point(tx + side_length / 2 - margin, ty + side_length / 2 - margin),
            rescale_point(tx + side_length / 2 - margin, ty - side_length / 2 + margin),
        ],
        stroke=(1, 0, 0, 0.5),
        stroke_width=5,
        line_cap='round',
        close_path=True,
    ).draw(surface)

    img = surface.get_npimage()

    img = cv2.resize(img, (out_width, out_height))
    return img, surface
