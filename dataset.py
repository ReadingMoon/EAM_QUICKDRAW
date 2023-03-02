# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import numpy as np
import os
import random
import constants
import pickle


# This code is an abstraction for the Google QUICKDRAW dataset,
columns = 28
rows = 28

def get_training(fold):
    return _get_segment(_TRAINING_SEGMENT, fold)

def get_filling(fold):
    return _get_segment(_FILLING_SEGMENT, fold)

def get_testing(fold, noised = False):
    return _get_segment(_TESTING_SEGMENT, fold, noised)

def _get_segment(segment, fold, noised = False):
    if (_get_segment.data is None) \
        or (_get_segment.noised is None) \
            or (_get_segment.labels is None):
        _get_segment.data, _get_segment.noised, _get_segment.labels = \
            _load_dataset(constants.data_path)
    print('Delimiting segment of data.')
    total = len(_get_segment.labels)
    training = total*constants.nn_training_percent
    filling = total*constants.am_filling_percent
    testing = total*constants.am_testing_percent
    step = total / constants.n_folds
    i = fold * step
    j = i + training
    k = j + filling
    l = k + testing
    i = int(i)
    j = int(j) % total
    k = int(k) % total
    l = int(l) % total
    n, m = None, None
    if segment == _TRAINING_SEGMENT:
        n, m = i, j
    elif segment == _FILLING_SEGMENT:
        n, m = j, k
    elif segment == _TESTING_SEGMENT:
        n, m = k, l

    data = constants.get_data_in_range(_get_segment.noised, n, m) \
            if noised \
                else constants.get_data_in_range(_get_segment.data, n, m)
    labels = constants.get_data_in_range(_get_segment.labels, n, m)
    return data, labels

_get_segment.data = None
_get_segment.noised = None
_get_segment.labels = None

def noised(data, percent):
    print(f'Adding {percent}% noise to data.')
    copy = np.zeros(data.shape, dtype=float)
    n = 0
    for i in range(len(copy)):
        copy[i] = _noised(data[i], percent)
        n += 1
        constants.print_counter(n, 10000, step=100)
    return copy

def _noised(image, percent):
    copy = np.array([row[:] for row in image])
    total = round(columns*rows*percent/100.0)
    noised = []
    while len(noised) < total:
        i = random.randrange(rows)
        j = random.randrange(columns)
        if (i, j) in noised:
            continue
        value = random.randrange(255)
        copy[i,j] = value
        noised.append((i,j))
    return copy       

_TRAINING_SEGMENT = 0
_FILLING_SEGMENT = 1
_TESTING_SEGMENT = 2

def _load_dataset(path):
    data, noised_data, labels = _preprocessed_dataset(path)
    if (data is None) or (noised_data is None) or (labels is None):
        data, labels = _load_quickdraw(path)
        noised_data = noised(data, constants.noise_percent)
        data, noised_data, labels = _shuffle(data, noised_data, labels)
        _save_dataset(data, noised_data, labels, path)
    return data, noised_data, labels

def _preprocessed_dataset(path):
    data_fname = os.path.join(path, constants.prep_data_fname)
    noised_fname = os.path.join(path, constants.pred_noised_data_fname)
    labels_fname = os.path.join(path, constants.prep_labels_fname)
    data = None
    noised = None
    labels = None
    try:
        data = np.load(data_fname)
        noised = np.load(noised_fname)
        labels = np.load(labels_fname).astype('int')
        print('Preprocessed dataset exists, so it is used.')
    except:
        print('Preprocessed dataset does not exist.')
    return data, noised, labels

def _save_dataset(data, noised, labels, path):
    print('Saving preprocessed dataset')
    data_fname = os.path.join(path, constants.prep_data_fname)
    noised_fname = os.path.join(path, constants.pred_noised_data_fname)
    labels_fname = os.path.join(path, constants.prep_labels_fname)
    np.save(data_fname, data)
    np.save(noised_fname, noised)
    np.save(labels_fname, labels)

def _load_quickdraw(path):
    """Load QUICKDRAW data from `path`"""
    labels_path = os.path.join(path, 'labels.pkl')
    images_path = os.path.join(path, 'images.pkl')

    pkl_file1 = open('labels.pkl', 'rb')
    labels = pickle.load(pkl_file1)
    pkl_file1.close()
    
    pkl_file2 = open('images.pkl', 'rb')
    images = pickle.load(pkl_file2)
    pkl_file2.close()
    return images, labels

def _shuffle(data, noised, labels):
    print('Shuffling data and labels')
    tuples = [(data[i], noised[i], labels[i]) for i in range(len(labels))]
    random.shuffle(tuples)
    data = np.array([p[0] for p in tuples])
    noised = np.array([p[1] for p in tuples])
    labels = np.array([p[2] for p in tuples], dtype=int)
    return data, noised, labels