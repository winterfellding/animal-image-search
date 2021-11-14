import tensorflow as tf
from tensorflow.keras.preprocessing import image
import  numpy as np
import os
import json

model = tf.keras.applications.mobilenet_v2.MobileNetV2()

def prepare_image(file):
    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

def gen_vector(img):
    pre_process_img = prepare_image(img)
    return model.predict(pre_process_img)

def list_all_images(base_path):
    all_files = []
    dirs = os.listdir(base_path)
    for dir in dirs:
        sub_path = os.path.join(base_path, dir)
        if os.path.isdir(sub_path):
            files = os.listdir(sub_path)
            for file in files:
                if file.endswith('.jpeg'):
                    all_files.append(os.path.join(sub_path, file))
    return all_files

if __name__ == '__main__':
    data_base_path = '/Users/winterfall/Desktop/animal/raw-img'
    all_files = list_all_images(data_base_path)
    all_vectors = {}
    for file in all_files:
        vector = gen_vector(file).tolist()
        file_obj_key = file[len(data_base_path) + 1:]
        print(file_obj_key)
        all_vectors[file_obj_key] = vector
    with open('vec_data.txt', 'w') as f:
        f.write(json.dumps(all_vectors))