import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='stockanalysis',
    version='0.1.0',
    author='Evbor',
    author_email='',
    description=('A package for analyzing stock price time-series. '
                 'Contains a CLI tool for downloading End of Day US stock data '
                 'and SEC filings. As well as a pre-configured ML pipeline '
                 'to train and validate ML models of stock price time-series.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Evbor/Stock_Analysis',
    packages=['stockanalysis'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
    ],
    install_requires=[
        'tensorflow>=2.2.0',
        'pandas>=1.0.4',
        'requests>=2.23.0',
        'lxml>=4.5.1',
        'beautifulsoup4 >= 4.9.1',
        'click>=7.1.2',
        'spacy>=2.2.4,<3.0.0'
    ],
    python_requires='>=3',
    package_data={
        'stockanalysis': ['default_config.pickle']
    },
    entry_points='''
        [console_scripts]
        stockanalysis=stockanalysis.command_line:stockanalysis
    '''
)
