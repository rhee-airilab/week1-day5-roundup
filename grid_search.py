#!/usr/bin/env python
# coding: utf-8
#%%
import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit, ParameterGrid

#%%
from mnist_data import load_mnist_t10k
from mnist_classifier import MnistClassifier


#%%
data_dir = '../Day3_MNIST/mnist'
t_images, t_labels = load_mnist_t10k(data_dir)
t_images = t_images / 127.0 - 1.0
t_images = np.reshape(t_images,[-1,28*28])
test_size  = 10000

#%%
param_grid = dict(  hidden_units=[30], \
                    batch_size=[128], \
                    learning_rate=[0.05], \
                    max_epoch=[100], \
                    activation=['gated','relu',], \
                    input_dropout=[0.2], \
                    hidden_dropout=[0.5] )

#%%
grid = GridSearchCV(estimator=MnistClassifier(), param_grid=param_grid, n_jobs=1, pre_dispatch=None, cv=3)
grid_result = grid.fit(t_images,t_labels)
print('Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))

#%%
means = grid_result.cv_results_['mean_test_score']
stds  = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

#%%
for mean, stdev, param in zip(means, stds, params):
    print('%f (%f) with: %r' % (mean, stdev, param))

