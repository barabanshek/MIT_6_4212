# Bookkeeping
import numpy as np

#
# BrickPlanner class: plan brick layout based on the floorplan and manipulator location
# Floorplan definitions:
#    - wall brick:
#       - vertical segment: '|' (vertical line)
#       - horizontal segment: '- ' (dash and space)
#    - empty space in horizontal direction:
#       - two consequent ' ' (space) sharacters, same as horizontal segment, but with no brick
#    - empty space in vertical direction:
#       - single ' ' (space) sharacter, same as vertical segment, but with no brick
#    - frames/coordinates:
#       - everything is in the World frame
#       - manipulator is the origin (0, 0, 0)
#
# Floorplan example:
#       floorplan = """
#           |― ― ― ― ― |
#           |          |
#           |          |
#           |          |
#           |― ― ― ― ― |
#
#       """
#
class Planner:
    def __init__(self, floorplan, brick_dim, margin, height):
        self.floorplan_ = floorplan
        self.brick_dim_ = brick_dim

        #
        fp_parsed = self.parse_floorplan(floorplan)
        fp_poses = self.parsed_floorplan_2_brick_poses(fp_parsed, self.brick_dim_, margin)

        #
        all_layers_bricks = []
        for i in range(0, height, 1):
            all_layers_bricks = all_layers_bricks + [[f[0], f[1], brick_dim[2]/2 + brick_dim[2] * i, f[2]] for f in fp_poses]
        self.poses_ = np.array(all_layers_bricks)

    # Parse floorplan `floorplan`, returns [X, Y, <orientation>];
    # <orientation>: 0 - vertical, 1 - horizontal
    kBrickOrientationVertical = 0
    kBrickOrientationHorizontal = 1
    kBrickEmpty = -1
    def parse_floorplan(self, floorplan):
        print("Parsing floorplan: ", floorplan)

        # parse
        i = -1
        j = 0
        bricks = []
        robot = None
        k = 0
        while k < len(floorplan) - 1:
            s = floorplan[k]
            s1 = floorplan[k+1]

            if s == ' ' and s1 == ' ':
                bricks.append([i, j, Planner.kBrickEmpty])
                j = j + 1
                k = k + 2
            elif s == '\n':
                i = i + 1
                j = 0
                k = k + 1
            elif s == '|':
                bricks.append([i, j, Planner.kBrickOrientationVertical])
                j = j + 1
                k = k + 1
            elif s == '-' and s1 == ' ':
                bricks.append([i, j, Planner.kBrickOrientationHorizontal])
                j = j + 1
                k = k + 2
            elif s == '*' and s1 == ' ':
                robot = [i, j]
                j = j + 1
                k = k + 2
            else:
                assert False, "Incorrect floorplan, error in line " + str(i) + " at position " + str(j)

        #print(robot)
        #print(bricks)

        return bricks

    # Transform parsed floorplan into brick poses;
    # `margin` is the gap size between the neighboring bricks
    def parsed_floorplan_2_brick_poses(self, parsed_floorplan, brick_dim, margin):
        brick_dim_short = min(brick_dim[0], brick_dim[1])
        brick_dim_long = max(brick_dim[0], brick_dim[1])

        # Generate for all other bricks
        orientation_horizontal_prev = None
        orientation_vertical_prev = None
        i_prev = -1
        x = 0
        y = 0
        base_poses = []
        for b in parsed_floorplan:
            orientation = b[2]

            # Handle vertical move
            if not i_prev == b[0]:
                y = 0

                # Account for prev brick
                if not orientation_vertical_prev == None:
                    if orientation_vertical_prev == Planner.kBrickOrientationVertical:
                        x = x + brick_dim_long/2
                    elif orientation_vertical_prev == Planner.kBrickOrientationHorizontal:
                        x = x + brick_dim_short/2
                    else:
                        # in vertical move, gap is treated as empty vertical brick
                        x = x + brick_dim_long/2

                # Account for current block
                if orientation == Planner.kBrickOrientationVertical:
                    x = x + brick_dim_long/2
                elif orientation == Planner.kBrickOrientationHorizontal:
                    x = x + brick_dim_short/2
                else:
                    # in vertical move, gap is treated as empty vertical brick
                    x = x + brick_dim_long/2

                # Add margin
                x = x + margin

                #
                orientation_vertical_prev = orientation
                orientation_horizontal_prev = None

            # Handle horizontal move:
            #           use the same x as defined by the left most brick
            # Account for prev brick
            if not orientation_horizontal_prev == None:
                if orientation_horizontal_prev == Planner.kBrickOrientationVertical:
                    y = y + brick_dim_short/2
                elif orientation_horizontal_prev == Planner.kBrickOrientationHorizontal:
                    y = y + brick_dim_long/2
                else:
                    # in horizontal move, gap is treated as empty horizontal brick
                    y = y + brick_dim_long/2

                # Account for current brick
                if orientation == Planner.kBrickOrientationVertical:
                    y = y + brick_dim_short/2
                elif orientation == Planner.kBrickOrientationHorizontal:
                    y = y + brick_dim_long/2
                else:
                    # in horizontal move, gap is treated as empty horizontal brick
                    y = y + brick_dim_long/2

                # Add margin
                y = y + margin
            else:
                y = 0

            #
            orientation_horizontal_prev = orientation

            # Append brick pose
            if not orientation == Planner.kBrickEmpty:
                base_poses.append([x, y, orientation])

            #
            #
            i_prev = b[0]

        return base_poses

    # Shift walls w.r.t origin
    def shift(self, x, y):
        self.poses_ = self.poses_ + np.array([x, y, 0, 0])

    def get_total_bricks_n(self):
        return self.poses_.shape[0]

    def get_brick_poses(self):
        return self.poses_
