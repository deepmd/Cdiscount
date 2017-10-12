import pandas as pd
import numpy as np
import os
from collections import defaultdict
from tqdm import *
import bcolz
import random as rn
import tensorflow as tf
import keras.backend as K
import datetime


# Create dictionaries for quick lookup of category_id to category_idx mapping.
def make_category_tables(categories_df):
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat


def make_val_set(categories_df, train_offsets_df, split_percentage=0.2, drop_percentage=0.):
    # Create a random train/validation split
    '''We split on products, not on individual images. 
    Since some of the categories only have a few products, we do the split separately for each category. 
    This creates two new tables, one for the training images and one for the validation images. 
    There is a row for every single image, so if a product has more than one image it occurs more than once in the table.'''
    df = train_offsets_df
    cat2idx, idx2cat = make_category_tables(categories_df) 
    
    # Find the product_ids for each category.
    category_dict = defaultdict(list)
    for ir in tqdm(df.itertuples()):
        category_dict[ir[4]].append(ir[0])

    train_list = []
    val_list = []
    with tqdm(total=len(df)) as pbar:
        for category_id, product_ids in category_dict.items():
            if any(df.category_id == category_id):
                category_idx = cat2idx[category_id]            
                # Randomly remove products to make the dataset smaller.
                keep_size = int(len(product_ids) * (1. - drop_percentage))
                if keep_size < len(product_ids):
                    product_ids = np.random.choice(product_ids, keep_size, replace=False)

                # Randomly choose the products that become part of the validation set.
                val_size = int(len(product_ids) * split_percentage)
                if val_size > 0:
                    val_ids = np.random.choice(product_ids, val_size, replace=False)
                else:
                    val_ids = []

                # Create a new row for each image.
                for product_id in product_ids:
                    row = [product_id, category_idx]
                    for img_idx in range(df.loc[product_id, "num_imgs"]):
                        if product_id in val_ids:
                            val_list.append(row + [img_idx])
                        else:
                            train_list.append(row + [img_idx])
                    pbar.update()
                               
    columns = ["product_id", "category_idx", "img_idx"]
    train_df = pd.DataFrame(train_list, columns=columns)
    val_df = pd.DataFrame(val_list, columns=columns)
    return train_df, val_df

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]

def get_run_name(weights_file, model_name):
    dt = datetime.datetime.now()
    while True:
        run_name = '{0}-{1:%Y-%m-%d-%H%M}'.format(model_name, dt)
        if not os.path.isfile(weights_file.format(run_name)):
            return run_name
        dt = dt + datetime.timedelta(minutes=-1)

def set_results_reproducible():
    '''https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development'''
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    return