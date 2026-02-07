import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'multiagent_pursuit_evasion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    scripts=['scripts/main_executor'],
    install_requires=[
        'setuptools',
        'numpy',
        'scipy',
        'torch',
    ],
    zip_safe=True,
    maintainer='llanesc',
    maintainer_email='christian.llanes@gatech.edu',
    description='Multi-agent pursuit-evasion on Crazyflie quadrotors with learned policies',
    license='MIT',
    tests_require=['pytest'],
)
