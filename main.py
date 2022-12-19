from robot import Robot


def main(path):

    robot = Robot(path)
    robot.view()
    #robot.reset()
    #robot.move_debug([1, 1, 1, 1, 1, 1])


if __name__ == '__main__':
    main()
