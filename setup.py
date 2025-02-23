from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ros2_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'), glob('models/*.[eon]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='taiki',
    maintainer_email='taiki@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolo_node = ros2_yolo.yolo_node:main'
        ],
    },
)
