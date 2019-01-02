from setuptools import setup, find_packages

setup(name='transformer-word-segmenter',

      version='0.1-SNAPSHOT',

      url='https://github.com/GlassyWing/transformer-word-segmenter',

      license='Apache License 2.0',

      author='Manlier',

      author_email='dengjiaxim@gmail.com',

      description='word segmenter base on bidirectional transformer (Transformer encoder)',

      packages=find_packages(where='.', exclude=('examples', 'tests'), include=('segmenter',)),

      long_description=open('README.md', encoding="utf-8").read(),

      zip_safe=False,

      install_requires=['pandas', 'matplotlib', 'keras', 'keras-transformer', 'tensorflow-hub'],

      )
