from setuptools import find_packages, setup

package_name = 'px4_nmpc_controller'

setup(
    name=package_name,
    version='3.5.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='Research User',
    maintainer_email='research@example.com',
    description='ROS 2 node that connects the NMPC core to PX4 Offboard trajectory/attitude/body-rate control.',
    license='BSD-3-Clause',
    entry_points={'console_scripts': ['nmpc_controller = px4_nmpc_controller.controller_node:main']},
)
