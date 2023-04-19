#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import tensorflow as tf
from imutils.video import VideoStream

model = cv2.dnn.readNetFromDarknet('darknet/cfg/yolov4.cfg', 'darknet/yolov4.weights')
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def detect_ball(image):
    # Redimensione a imagem para a entrada da rede neural
    input_image = cv2.dnn.blobFromImage(image, 1/255, (608, 608), swapRB=True)

    # Passe a imagem pela rede neural
    model.setInput(input_image)
    output_layer_names = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layer_names)

    # Analise as saídas da rede neural para encontrar a caixa delimitadora da bola
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.5:
                center_x, center_y, width, height = detection[:4] * np.array([image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
                left, top = int(center_x - width/2), int(center_y - height/2)
                boxes.append([left, top, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Desenhe a caixa delimitadora da bola na imagem
    if len(boxes) > 0:
        index = np.argmax(confidences)
        left, top, width, height = boxes[index]
        right, bottom = left + width, top + height
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        # Retorne a posição da bola em pixels
        ball_x = (left + right) / 2
        ball_y = (top + bottom) / 2
        return int(ball_x), int(ball_y)
    else:
        return None

v1 = VideoStream(2).start()

def show(image):
    ball_pos = detect_ball(image)

    if ball_pos is not None:
        print(f'A bola está na posição (x={ball_pos[0]}, y={ball_pos[1]})')
    else:
        print('A bola não foi detectada na imagem')

    cv2.imshow('Imagem', image)

while True:
    fr = v1.read()
    if fr is not None:
        show(fr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
