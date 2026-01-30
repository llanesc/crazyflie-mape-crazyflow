import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'multiagent_pursuit_evasion'

# Collect all JSON and PT files from models directory
model_files = []
for policy_type in ['ffn', 'acmpc']:
    model_dir = os.path.join('models', policy_type)
    if os.path.exists(model_dir):
        # Collect JSON config files
        json_files = glob(os.path.join(model_dir, '*.json'))
        # Collect checkpoint files (.pt)
        pt_files = glob(os.path.join(model_dir, '*.pt'))
        all_files = json_files + pt_files
        if all_files:
            model_files.append(
                (os.path.join('share', package_name, 'models', policy_type), all_files)
            )

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
    ] + model_files,
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
