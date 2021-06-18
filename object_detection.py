import numpy as np
from cv2 import cv2
import time
from gtts import gTTS 
import os
import matplotlib.pyplot as plt
from PIL import Image
from flask import Flask, send_file
import io
def yolo():
    #camera = cv2.VideoCapture(0)
    h, w = None, None

    
    """
    End of:
    Reading stream video from camera
    """


    """
    Start of:
    Loading YOLO v3 network 
    """

    with open('yolo-coco-data/coco.names') as f:    
        labels = [line.strip() for line in f]

    network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

    layers_names_all = network.getLayerNames()

    layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

    probability_minimum = 0.5

    threshold = 0.3

    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    """
    End of:
    Loading YOLO v3 network
    """


    """
    Start of:
    Reading frames in the loop
    """

    start_time = time.time()
    seconds = 10
    img = cv2.imread('Car.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread('Person.png')
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    print('Our two images are')
    # plt.imshow(img)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    
    print('Objects are: ')
    all_objects = []
    k = 1
    while (k):
        #_, frame = camera.read()
        frame = img
        if(k == 2):
            frame = img2
        if w is None or h is None:
            h, w = frame.shape[:2]

        current_time = time.time()
        elapsed_time = current_time - start_time    
        
        if elapsed_time > seconds:
            # Releasing camera
            #camera.release()
            # Destroying all opened OpenCV windows
            cv2.destroyAllWindows()
            
            break
        """
        Start of:
            Getting blob from current frame
            """
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        """
        End of:
        Getting blob from current frame
        """
        
        """
        Start of:
        Implementing Forward pass
        """
    
        network.setInput(blob)  # setting blob as input to the network
        #start = time.time()
        output_from_network = network.forward(layers_names_output)
        #end = time.time()
        #print('Current frame took {:.5f} seconds'.format(end - start))

        """
        End of:
            Implementing Forward pass
        """

        """
        Start of:
        Getting bounding boxes
        """

        bounding_boxes = []
        confidences = []
        class_numbers = []
        for result in output_from_network:
            # Going through all detections from current output layer
            for detected_objects in result:
                # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
                # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
                # Getting value of probability for defined class
                confidence_current = scores[class_current]
                
                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)
            
        
        if(k == 2):
            k = 0
        if(k == 1):
            k = 2
        
        """
        End of:
        Getting bounding boxes
        """

        """
        Start of:
        Non-maximum suppression
        """
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        """
        End of:
        Non-maximum suppression
        """

        """
        Start of:
        Drawing bounding boxes and labels
        """
        
        if len(results) > 0:
        # Going through indexes of results
            for i in results.flatten():
            #speech output
                obj = labels[int(class_numbers[i])]
                #if obj in all_objects:
                #    continue
                #else:
                all_objects.append(obj)
                #text = "There is a "+obj+" in front of you."
                #language = 'en'
                #speech = gTTS(text = text, lang = language, slow = False)
                #speech.save("text.wav")
                #os.system("text.wav")
    
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                colour_box_current = colours[class_numbers[i]].tolist()

                #cv2.rectangle(frame, (x_min, y_min),
                #          (x_min + box_width, y_min + box_height),
                #          colour_box_current, 2)
                        # plt.imshow(cv2.rectangle(frame, (x_min, y_min),
                        #           (x_min + box_width, y_min + box_height),
                        #           colour_box_current, 2))
                        # plt.show()
                
                arr = cv2.rectangle(frame, (x_min, y_min),(x_min + box_width, y_min + box_height),colour_box_current, 2)
                img = Image.fromarray(arr.astype('uint8'))

                # create file-object in memory
                file_object = io.BytesIO()

                # write PNG in file-object
                img.save(file_object, 'PNG')

                # move to beginning of file so `send_file()` it will read from start    
                file_object.seek(0)

                return send_file(file_object, mimetype='image/PNG')

                print(obj)
                print(x_min)
                print(y_min)
                print(box_width)
                print(box_height)
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])
            
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

        #print(len(all_objects))
        
        """
        End of:
        Drawing bounding boxes and labels
        """

        """
        Start of:
        Showing processed frames in OpenCV Window
        """
        cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
        # Pay attention! 'cv2.imshow' takes images in BGR format
        cv2.imshow('YOLO v3 Real Time Detections', frame)

        # Breaking the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        """
        End of:
        Showing processed frames in OpenCV Window
        """

    """
    End of:
    Reading frames in the loop
    """

    
    for i in all_objects:
        print(i)
    # Releasing camera
    #camera.release()
    # Destroying all opened OpenCV windows
    cv2.destroyAllWindows()

    return send_file(file_object, mimetype='image/PNG')