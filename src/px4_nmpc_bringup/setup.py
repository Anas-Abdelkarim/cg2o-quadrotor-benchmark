from glob import glob
from setuptools import find_packages, setup

package_name = 'px4_nmpc_bringup'

setup(
    name=package_name,
    version='3.5.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Research User',
    maintainer_email='research@example.com',
    description='Launch files for the PX4 NMPC obstacle-avoidance project.',
    license='BSD-3-Clause',
)
