#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Single Shot Detection (SSD)

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tensorflow as tf
import imutils

class SSD:
    def start(self):
        self.cv_image = None
        # Carregar o modelo SSD
        self.model_path = 'ssd_bola_model.pb'
        # with tf.io.gfile.GFile(self.model_path, 'rb') as f:
        #     graph_def = tf.compat.v1.GraphDef()
        #     graph_def.ParseFromString(f.read())

        # # Importar o grafo do TensorFlow
        # with tf.compat.v1.Session() as self.sess:
        #     self.sess.graph.as_default()
        #     tf.import_graph_def(graph_def, name='')
        
        # Subscrever o tópico da imagem
        rospy.Subscriber("/camera0/usb_cam/image_raw", Image, self.image_callback)

    # Função de callback para a mensagem de imagem ROS
    def image_callback(self, msg):
        global cv_image
        try:
            self.cv_image = CvBridge().imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)
        return

    # # Função para processar a detecção
    # def detect_ball(self, image):
    #     rows, cols, _ = image.shape

    #     # Redimensionar a imagem para o tamanho esperado pelo modelo SSD
    #     input_image = cv2.resize(image, (300, 300))
    #     input_image = input_image[:, :, [2, 1, 0]]  # BGR para RGB
    #     input_image = np.expand_dims(input_image, axis=0)

    #     # Executar inferência do modelo SSD
    #     output = self.sess.run(self.sess.graph.get_tensor_by_name('detection_out:0'), feed_dict={
    #         'image_tensor:0': input_image
    #     })

    #     # Processar os resultados da detecção
    #     num_detections = int(output[0, 0, 0, 2])
    #     for i in range(num_detections):
    #         class_id = int(output[0, 0, i, 1])
    #         score = float(output[0, 0, i, 2])
    #         bbox = [float(v) for v in output[0, 0, i, 3:]]

    #         # Verificar se a detecção é uma bola e se a confiança é maior que um limiar (por exemplo, 0.5)
    #         if class_id == 1 and score > 0.5:
    #             x1 = int(bbox[1] * cols)
    #             y1 = int(bbox[0] * rows)
    #             x2 = int(bbox[3] * cols)
    #             y2 = int(bbox[2] * rows)

    #             # Desenhar o bounding box e o rótulo da classe na imagem
    #             cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(image, f"Bola: {score:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #     return image

    def detect_ball(self, img, BGR=False):
        # blue ball
        self.firstColorLower = (82, 110, 80)
        self.firstColorUpper = (153, 255, 255)

        self.secondColorLower = (82, 110, 80)
        self.secondColorUpper = (153, 255, 255)

        # grab the current frame
        frame = img
        if BGR:
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # if we are viewing a video and we did not grab a frame,
        # then we have reached the end of the video
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        self.mask0 = cv2.inRange(hsv, self.firstColorLower, self.firstColorUpper)
        self.mask0 = cv2.erode(self.mask0, None, iterations=2)
        self.mask0 = cv2.dilate(self.mask0, None, iterations=2)

        self.mask1 = cv2.inRange(hsv, self.secondColorLower, self.secondColorUpper)
        self.mask1 = cv2.erode(self.mask1, None, iterations=2)
        self.mask1 = cv2.dilate(self.mask1, None, iterations=2)

        mask = self.mask0 | self.mask1

        mask = cv2.dilate(mask, None, iterations=10)
        mask = cv2.erode(mask, None, iterations=10)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            self.st.update(tp, x, y)
            self.filter_pub(filter=False)
            self.pub.publish(self.StereoPose)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # only proceed if the radius meets a minimum size
            if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius),
                    (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
            # update the points queue
            self.pts.appendleft(center)
            font                   = cv2.FONT_HERSHEY_SIMPLEX
            fontScale              = 1
            fontColor              = (255,0,0)
            thickness              = 1
            lineType               = 1
            
            bottomLeftCornerOfText = (40,60)
            cv2.putText(frame,'x: {:3.2f}'.format(self.st.cam0.Px), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        
            bottomLeftCornerOfText = (40,100)
            cv2.putText(frame,'y: {:3.2f}'.format(self.st.cam0.Py), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)

            bottomLeftCornerOfText = (40,140)
            cv2.putText(frame,'z: {:3.2f}'.format(self.st.cam0.Pz), 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                thickness,
                lineType)
        # loop over the set of tracked points
        key = cv2.waitKey(1) & 0xFF
        return frame
    
    def video_ball_detection(self):
        while not rospy.is_shutdown():
            if self.cv_image is not None:
                # Detectar bola na imagem
                result_image = self.detect_ball(self.cv_image)

                # Mostrar a imagem com a detecção da bola
                cv2.imshow("Ball Detection", result_image)
        cv2.destroyAllWindows()


def main():
    rospy.init_node('SSD_ball_detection', anonymous=True)
    node = SSD()
    node.start()

    # Continuar processando as imagens até que o nó seja encerrado
    node.video_ball_detection()

if __name__ == '__main__':
    main()