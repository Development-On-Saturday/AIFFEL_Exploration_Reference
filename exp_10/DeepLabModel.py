from utils import *

class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME ='frozen_inference_graph'

    #__init__()에서 모델 구조를 직접 구현하는 대신, tar file에서 읽어들인
    #그래프구조 graph_def를 tf.compat.v1.import_graph_def를 통해 불러들여 활용한다.
    def __init__(self, tarball_path):
        self.graph=tf.Graph()
        graph_def=None
        tar_file=tarfile.open(tarball_path)

        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.compat.v1.GraphDef.FromString(file_handle.read())
                break
        tar_file.close()

        with self.graph.as_default():
            tf.compat.v1.import_graph_def(graph_def, name='')

        self.sess=tf.compat.v1.Session(graph=self.graph)

    #이미지를 전처리하여 Tensorflow 입력으로 사용 가능한 Shape의 numpy array로 변환한다.
    def preprocess(self, img_orig):
        height, width = img_orig.shape[:2]
        resize_ratio = 1.0*self.INPUT_SIZE/max(width, height)
        target_size = (int(resize_ratio*width), int(resize_ratio*height))
        resized_image = cv2.resize(img_orig, target_size)
        resized_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        img_input = resized_rgb
        return img_input

    def run(self, image):
        img_input=self.preprocess(image)

        #Tensorflow V1 에서는 model(input) 방식이 아니라 sess.run(feed_dict={input...}) 방식을 활용한다.
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME:[img_input]})
        seg_map = batch_seg_map[0]
        return cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR), seg_map
