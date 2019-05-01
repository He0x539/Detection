import cv2
# model download
import tarfile
import os
import six.moves.urllib as urllib
# using model
import tensorflow as tf
import numpy as np
import sys

if sys.version_info[0] >= 3:
    import PySimpleGUI as sg
else:
    import PySimpleGUI27 as sg
import cv2 as cv
from PIL import Image
import io

import sys

# labels
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class Detection():


    def __init__(self, name):
        self.name = name
        self.LANGUAGE = None
        self.COUNTRY = None
        self.CITY = None

        self.category_index_ENG = None
        self.category_index_SPA = None
        self.category_index_SWE = None
        self.category_index_CUSTOM = None

    def change_language(self, new_lang):
        self.LANGUAGE = new_lang
        print("Language set to " + self.LANGUAGE)

    def set_location_info(self, country, city):
        self.COUNTRY = country
        self.CITY = city

    def create_capture(self, device):
        capture = cv2.VideoCapture(device)
        print("created webcam-capture")
        return capture

    def model_download(self, MODEL_FILE, DOWNLOAD_BASE, PATH_TO_TENSORFLOW_MODEL):

        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())

                print("Coco Tensorflow model downloaded and extracted ")

    def load_custom_label(self, NUM_CLASSES, PATH):

        # CUSTOM
        label_mapping_custom = label_map_util.load_labelmap(PATH)
        categories_CUSTOM = label_map_util.convert_label_map_to_categories(label_mapping_custom,
                                                                           max_num_classes=NUM_CLASSES,
                                                                           use_display_name=True)
        self.category_index_CUSTOM = label_map_util.create_category_index(categories_CUSTOM)

        print('Loaded custom labels')

    def load_labels(self, NUM_CLASSES):

        # if self.LANGUAGE == 'eng':
        PATH_TO_ENG = os.path.join('object_detection', 'data', 'mscoco_label_map.pbtxt')
        # if self.LANGUAGE == 'spa':
        PATH_TO_SPA = os.path.join('object_detection', 'data', 'Custom_labels_spa2.pbtxt')
        # if self.LANGUAGE == 'swe':
        PATH_TO_SWE = os.path.join('object_detection', 'data', 'Custom_labels_swe.pbtxt')

        # loads mapping between predicted class and name of class in different languages

        # ENG
        label_mapping_eng = label_map_util.load_labelmap(PATH_TO_ENG)
        categories_ENG = label_map_util.convert_label_map_to_categories(label_mapping_eng, max_num_classes=NUM_CLASSES,
                                                                        use_display_name=True)
        self.category_index_ENG = label_map_util.create_category_index(categories_ENG)

        # SPA
        label_mapping_spa = label_map_util.load_labelmap(PATH_TO_SPA)
        categories_SPA = label_map_util.convert_label_map_to_categories(label_mapping_spa, max_num_classes=NUM_CLASSES,
                                                                        use_display_name=True)
        self.category_index_SPA = label_map_util.create_category_index(categories_SPA)

        # SWE
        label_mapping_swe = label_map_util.load_labelmap(PATH_TO_SWE)
        categories_SWE = label_map_util.convert_label_map_to_categories(label_mapping_swe, max_num_classes=NUM_CLASSES,
                                                                        use_display_name=True)
        self.category_index_SWE = label_map_util.create_category_index(categories_SWE)

        print("Labels loaded")
        # return category_index

    def run_detection(self, PATH_TO_TENSORFLOW_MODEL, capture):

        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_TENSORFLOW_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        print(" Model loaded")

        ############### UI

        sg.ChangeLookAndFeel('Black')

        # define the window layout
        layout = [[sg.Text('Object Detection Translator', size=(40, 1), justification='center', font='Helvetica 20')],
                   [sg.Text('User location: ' + self.COUNTRY + ' : ' + self.CITY, size=(40, 1), justification='center', font='Helvetica 20')],
                   [sg.Text('Default language set to: ' + self.LANGUAGE, size=(40, 1), justification='center', font='Helvetica 20')],
                  [sg.Image(filename='', key='image')],

                  [sg.ReadButton('Swedish', size=(10, 1), font='Any 14', border_width=0),
                   sg.RButton('English', size=(10, 1), font='Any 14', border_width=0),
                   sg.RButton('Spanish', size=(10, 1), font='Any 14', border_width=0),
                   sg.RButton('Browse', size=(10, 1), font='Any 14', border_width=0),
                   sg.RButton('Exit', size=(10, 1), font='Helvetica 14', border_width=0)
                   ]]
        # pad=((200, 0), 3),

        # create the window and show it without the plot
        window = sg.Window('Object Detection Translator',
                           location=(800, 400))
        window.Layout(layout).Finalize()

        ###############

        with detection_graph.as_default():
            with tf.Session(graph=detection_graph) as sess:
                # take input from webcam
                while True:
                    # handle labels

                    if self.LANGUAGE == 'English':
                        category_index = self.category_index_ENG
                    elif self.LANGUAGE == 'Spanish':
                        category_index = self.category_index_SPA
                    elif self.LANGUAGE == 'Swedish':
                        category_index = self.category_index_SWE
                    elif self.LANGUAGE == 'Custom':
                        category_index = self.category_index_CUSTOM

                    ret, image_np = capture.read()

                    # UI
                    button, values = window.ReadNonBlocking()

                    if button is 'Exit' or values is None:
                        sys.exit(0)
                    elif button == 'Browse':
                        event, (filename,) = sg.Window('Get language file').Layout(
                            [[sg.Text('Filename')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()]]).Read()
                        self.LANGUAGE = 'custom'
                        print(filename)
                        # window.Close()
                        self.load_custom_label(90, filename)

                    elif button == 'English':
                        self.LANGUAGE = 'English'
                        print("lang = eng")
                    elif button == 'Spanish':
                        self.LANGUAGE = 'Spanish'
                        print("lang = spa")
                    elif button == 'Swedish':
                        self.LANGUAGE = 'Swedish'
                        print("lang = swe")

                    #

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Each box represents a part of the image where a particular object was detected.
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Each score represent how level of confidence for each of the objects.
                    # Score is shown on the result image, together with the class label.
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        line_thickness=4)

                    # gray = cv.cvtColor(image_np, cv.COLOR_BGR2GRAY)
                    # flip colors so that they are displayed correctly
                    color = cv.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                    img = Image.fromarray(color)  # create PIL image from frame(image_np)
                    bio = io.BytesIO()  # binary temp save
                    img.save(bio, format='PNG')  # save image as png to temp save
                    imgbytes = bio.getvalue()  # this can be used by OpenCV
                    window.FindElement('image').Update(data=imgbytes)

                run_detection()
