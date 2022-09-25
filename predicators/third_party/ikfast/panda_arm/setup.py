"""Note: this subdirectory is modified from the pybullet-planning repository
by Caelan Garrett (https://github.com/caelan/pybullet-planning/)."""

import sys

from compile import compile_ikfast


def main():
    # lib name template: 'ikfast_<robot name>'
    sys.argv[:] = sys.argv[:1] + ['build']
    robot_name = 'panda_arm'
    compile_ikfast(module_name='ikfast_{}'.format(robot_name),
                   cpp_filename='ikfast_{}.cpp'.format(robot_name))


if __name__ == '__main__':
    main()
