from setuptools import find_packages, setup

package_name = 'px4_nmpc_tools'
setup(
    name=package_name,
    version='3.5.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    zip_safe=True,
    maintainer='Research User',
    maintainer_email='research@example.com',
    description='Experiment, visualization, monitoring and Gazebo utilities for the PX4 NMPC project.',
    license='BSD-3-Clause',
    entry_points={'console_scripts': [
        'nmpc_experiment = px4_nmpc_tools.experiment_manager:main',
        'nmpc_visualizer = px4_nmpc_tools.nmpc_visualizer:main',
        'nmpc_monitor = px4_nmpc_tools.nmpc_monitor:main',
        'nmpc_mission = px4_nmpc_tools.mission_cli:main',
    ]},
)
