#!/bin/bash
source scl_source enable devtoolset-6
source loadLSST.bash
setup lsst_sims
pip install nose
pip install coveralls
eups declare gasp -r ${TRAVIS_BUILD_DIR} -t current
setup gasp
cd ${TRAVIS_BUILD_DIR}
scons opt=3
nosetests -s --with-coverage --cover-package=desc.gasp
