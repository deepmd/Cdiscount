from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import pandas as pd
import numpy as np
from tqdm import *
import tensorflow as tf
import keras.backend as K
import cv2
import queue
import threading
import csv
import io
import gzip
import shutil

def make_multi_crop_from(im, input_size, crop_size):
    cordinates = [((0, 0), (crop_size-1, crop_size-1)),
        ((input_size-crop_size, 0), (input_size-1, crop_size-1)),
        ((0, input_size-crop_size), (crop_size-1, input_size-1)),
        ((input_size-crop_size, input_size-crop_size), (input_size-1, input_size-1))]    
    flipped_im = cv2.flip(im, 1)
    mc = []
    for c1, c2 in cordinates:
        x1, y1 = c1
        x2, y2 = c2
        mc.append(im[y1:y2+1, x1:x2+1])
        mc.append(flipped_im[y1:y2+1, x1:x2+1])
    im = cv2.resize(im, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    flipped_im = cv2.resize(flipped_im, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
    mc.append(im)
    mc.append(flipped_im)
    return np.array(mc)


def data_loader(q, data, input_size, crop_size, num_test_products):
    num_crop = 10
    test_datagen = ImageDataGenerator() #ImageDataGenerator(preprocessing_function=preprocess_input)
    for n, d in enumerate(data):
        if n == num_test_products:
            break
        product_id = d["_id"]
        num_imgs = len(d["imgs"])
        images = np.zeros((num_imgs*num_crop, crop_size, crop_size, 3), dtype=K.floatx())        
        for i in range(num_imgs):
            bson_img = d["imgs"][i]["picture"]
            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(input_size, input_size))
            x = img_to_array(img)
            x = test_datagen.random_transform(x)
            x = test_datagen.standardize(x)
            mc = make_multi_crop_from(x, input_size, crop_size)            
            for j, crop_im in enumerate(mc):            
                # Add the image to the batch.
                images[i*num_crop+j] = crop_im
        q.put((product_id, images))

def predictor(q, out_q, graph, model, num_test_products, idx2cat):
    for i in tqdm(range(num_test_products)):
        product_id, images = q.get()
        with graph.as_default():
            predictions = model.predict_on_batch(images)
            avg_pred = predictions.mean(axis=0)
            cat_idx = np.argmax(avg_pred)
            out_q.put((product_id, list(avg_pred), idx2cat[cat_idx]))

def output_writer(out_q, num_test_products, prediction_csv_writer, submission_csv_writer):
    for i in range(num_test_products):
        product_id, predictions, category = out_q.get()
        prediction_csv_writer.writerow([product_id] + predictions)
        submission_csv_writer.writerow([product_id] + [category])

def generate_submit(model, data, input_size, crop_size, num_test_products, idx2cat,
                    submission_path, prediction_path, q_size=50, out_q_size=50):

    with open('{}.csv'.format(prediction_path), 'w', newline='') as prediction_file, \
         open('{}.csv'.format(submission_path), 'w') as submission_file:
        prediction_csv_writer = csv.writer(prediction_file)
        prediction_csv_writer.writerow(['_id'] + [idx2cat[i] for i in idx2cat])
        submission_csv_writer = csv.writer(submission_file)
        submission_csv_writer.writerow(['_id', 'category_id'])

        graph = tf.get_default_graph()
            
        q = queue.Queue(maxsize=q_size)
        out_q = queue.Queue(maxsize=out_q_size)
        t1 = threading.Thread(target=data_loader, name='DataLoader',
                            args=(q, data, input_size, crop_size, num_test_products, ))
        t2 = threading.Thread(target=predictor, name='Predictor', 
                            args=(q, out_q, graph, model, num_test_products, idx2cat, ))
        t3 = threading.Thread(target=output_writer, name='OutputWriter', 
                            args=(out_q, num_test_products, prediction_csv_writer, submission_csv_writer, ))
        print('Start predicting on samples...')
        t1.start()
        t2.start()
        t3.start()
        # Wait for all threads to finish
        t1.join()
        t2.join()
        t3.join()

    print('Compressing submission file...')
    with open('{}.csv'.format(submission_path), 'rb') as f_in, gzip.open('{}.csv.gz'.format(submission_path), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print('Done')