from styx_msgs.msg import TrafficLight
import rospy
import os
import numpy as np
import yaml
import tensorflow as tf
#####################################################################
#   file: TlClassifier.py
#####################################################################

TL_THRESHOLD = 0.5 # Minimum score for light

MODELS_FOLDER = '/tl_perception_models/'

class TLClassifier(object):

    def __init__(self):
        rospy.logwarn("entry: TLClassifier:Init")
        curr_dir    = os.path.dirname(os.path.realpath(__file__))
        item_green  = {'id': 1, 'name': 'Green'}
        item_red    = {'id': 2, 'name': 'Red'}
        item_yellow = {'id': 3, 'name': 'Yellow'}
        #self.config_data = ''
        self.site        = True

        self.label_dict = {1: item_green, 2: item_red, 3: item_yellow}

        config_yaml = rospy.get_param("/traffic_light_config")
        config_data = yaml.load(config_yaml)
        self.site  =  config_data['is_site']
        print('*******is_site**********')
        print(self.site)

# The graphs were generated using TensorFlow on image data available from coldknight.
        if self.site == False:
            self.model_filename = 'simulator_frozen_inference_graph.pb'
        else:
            self.model_filename = 'real_frozen_inference_graph.pb'

        model_path  = curr_dir + MODELS_FOLDER + self.model_filename
        self.build_model_graph(model_path)

        print("Loading ")
        print(model_path)
        print("Classifier ready")


    def detect_traffic_light(self, scores, biggest_score_idx,
                             classes,
                             detected_light):
        if scores[biggest_score_idx] > TL_THRESHOLD:
            rospy.logwarn("Current traffic light is: {}"
                          .format(self.label_dict[classes[biggest_score_idx]]['name']))
            if classes[biggest_score_idx] == 1:
                detected_light = TrafficLight.GREEN
            elif classes[biggest_score_idx] == 2:
                detected_light = TrafficLight.RED
            elif classes[biggest_score_idx] == 3:
                detected_light = TrafficLight.YELLOW
        else:
            rospy.logwarn("Not defined")
            rospy.logwarn(scores[biggest_score_idx])
        return detected_light


    def build_model_graph(self, model_path):
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                saved_graph = fid.read()
                od_graph_def.ParseFromString(saved_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.model_graph)

        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.scores  = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.model_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.model_graph.get_tensor_by_name('num_detections:0')    
        self.boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')

    def classify(self, image):

        detected_light = TrafficLight.UNKNOWN

        image_expanded = np.expand_dims(image, axis=0)
        with self.model_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.boxes, self.scores, self.classes, self.num_detections], 
                                                          feed_dict={self.image_tensor: image_expanded})

        scores = np.squeeze(scores)

        return self.detect_traffic_light(scores, scores.argmax(),
                                         np.squeeze(classes).astype(np.int32),
detected_light)

