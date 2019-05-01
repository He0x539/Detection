
import Detection
import GeoLoc


class Main:

    def __init__(self, name):
        print("init")
        self.name = "__main__"

    print("running")
    # input for change lang
    LANGUAGE = 'eng'
    # input for model download
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
    # Path to object detection model.
    PATH_TO_TENSORFLOW_MODEL = MODEL_NAME + '/frozen_inference_graph.pb'
    model = None
    # Label mapping
    NUM_CLASSES = 90

    # test
    detector = Detection.Detection('Bob')
    # set default based on location of user
    geoloc = GeoLoc.LanguageFetch()
    language = geoloc.getLanguage_based_on_loc()
    print('returned lang: '+language)
    country = geoloc.getCountry()
    city = geoloc.getCity()

    detector.change_language(language)
    detector.set_location_info(country, city)

    # setup
    capture = detector.create_capture(0)

    # check if we already have a model
    # if model is None:
    # print("Model was None")
    model = detector.model_download(MODEL_FILE, DOWNLOAD_BASE, PATH_TO_TENSORFLOW_MODEL)
    # else:
    # print("Model was already loaded")
    #detector.model_load(PATH_TO_TENSORFLOW_MODEL)

    detector.load_labels(NUM_CLASSES)

    detector.run_detection(PATH_TO_TENSORFLOW_MODEL, capture)


# if we break loop to change language
    #detector.change_language('eng')



