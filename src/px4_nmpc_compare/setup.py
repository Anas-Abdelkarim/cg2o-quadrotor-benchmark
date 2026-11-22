from setuptools import setup, find_packages

package_name = 'px4_nmpc_compare'
setup(
    name=package_name,
    version='3.5.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML', 'matplotlib'],
    zip_safe=True,
    maintainer='Research User',
    maintainer_email='research@example.com',
    description='Benchmark model, paired cg2o/IPOPT replay and tuning evaluation of the quadrotor benchmark.',
    license='BSD-3-Clause',
)
