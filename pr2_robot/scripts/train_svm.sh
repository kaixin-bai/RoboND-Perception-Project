#! /bin/bash
# This script safely launches ros nodes with buffer time to allow param server population
PKG_ROOT=$(rospack find pr2_robot)
DATA_PATH="${PKG_ROOT}/config/training_set.sav"
MODEL_PATH="${PKG_ROOT}/config/model.sav"
python ${PKG_ROOT}/src/pr2_robot/svm_classifier.py ${MODEL_PATH} ${DATA_PATH} --train --test
