# libgp - Gaussian process library for Machine Learning
# Copyright (c) 2013, Manuel Blum <mblum@informatik.uni-freiburg.de>
# All rights reserved.

ADD_EXECUTABLE(gpdense gp_example_dense.cc)
TARGET_LINK_LIBRARIES(gpdense gp)

ADD_EXECUTABLE(gptrain gp_example_train.cc)
TARGET_LINK_LIBRARIES(gptrain gp)

ADD_EXECUTABLE(gptrain_time gp_example_train_time_add.cc)
TARGET_LINK_LIBRARIES(gptrain_time gp)
