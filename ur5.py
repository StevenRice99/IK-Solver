import os

from main import main


def ur5():
    main(f"{os.getcwd()}/universal_robots_ur5e/ur5e.xml")


if __name__ == '__main__':
    ur5()
