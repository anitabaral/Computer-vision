import os
import cv2
import argparse

import numpy as np


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=False, help="path to input image")
    ap.add_argument(
        "-p", "--prototxt", required=True, help="path to Caffe 'deploy' prototxt file"
    )
    ap.add_argument(
        "-m", "--model", required=True, help="path to Caffe pre-trained model"
    )
    ap.add_argument(
        "-c",
        "--confidence",
        type=float,
        required=True,
        help="minimum probability to filter weak detections",
    )
    args = vars(ap.parse_args())
    print(args)
    return args

def main(args):
    dirname, filenaem = os.path.split(os.path.abspath(__file__))
    prototxt = os.path.join(dirname, args["prototxt"])
    model = os.path.join(dirname, args["model"])
    image = os.path.join(dirname, args["image"])

    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    image = cv2.imread(image)
    (height, width) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= args["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            text =f"{(confidence * 100):.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
    cv2.imshow("Output", image)
    cv2.waitKey(0)

if __name__=="__main__":
    main(parse_arguments())