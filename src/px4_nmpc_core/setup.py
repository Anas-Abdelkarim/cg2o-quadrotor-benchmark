from setuptools import find_packages, setup

package_name = 'px4_nmpc_core'

setup(
    name=package_name,
    version='3.5.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'casadi'],
    zip_safe=True,
    maintainer='Research User',
    maintainer_email='research@example.com',
    description='ROS-independent NMPC, planning, and math library for PX4 obstacle avoidance.',
    license='BSD-3-Clause',
)
