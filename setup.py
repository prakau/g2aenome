from setuptools import setup, find_packages

setup(
    name='g2aenome_dr_kaushik_adv_poc',
    version='0.1.0',
    author='Dr. Prashant Kaushik & AI Assistant',
    author_email='your_email@example.com',  # Replace with a valid email
    description='Genomic & Agri-Phenomic AI Nexus - Advanced Proof-of-Concept',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/g2aenome_dr_kaushik_adv_poc',  # Replace with your GitHub repo URL
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        # Dependencies will be read from requirements.txt
        # For a more robust setup, parse requirements.txt here
        # or list them explicitly.
        # For now, keeping it simple.
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'g2aenome_pipeline=main_pipeline:main', # Example if main_pipeline.py has a main() function
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
