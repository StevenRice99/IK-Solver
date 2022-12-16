from helpers import Robot


def main(path):

    robot = Robot(path, 0.1)
    #robot.view()
    #robot.reset()
    #robot.move_debug([1, 1, 1, 1, 1, 1])
    #robot.snap([1, 1, 1, 1, 1, 1])
    #robot.move_debug([-1, -1, -1, -1, -1, -1])

    pair = robot.generate_pair()
    print(pair)
    print(robot)

if __name__ == '__main__':
    main()
