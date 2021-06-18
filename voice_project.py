from flask import Flask, send_file
import io
import numpy as np
import cv2
import time
from gtts import gTTS 
import os
import matplotlib.pyplot as plt

import glob
import argparse
import matplotlib
import numpy as np

# Keras / TensorFlow

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images

from PIL import Image  
import PIL



def yolo(img):
    #camera = cv2.VideoCapture(0)
    h, w = None, None
    img_original = img
    frame_information = []             # details for every bounding box
    
    #img = cv2.imread('Car_People.jpeg')
    #img = cv2.imread('Fan_Mobile.jpeg')

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = image_resize(img, width = 640)
    h = img.shape[0]
    w = img.shape[1]
    ratio = w/h
    image = np.zeros((480, 640, 3), dtype='uint8')
    image = image - 1
    if(ratio < 640/480):
        img = cv2.resize(img, (int(480*ratio), 480), interpolation = cv2.INTER_AREA)

        for i in range(480):
            for j in range(int(480*ratio)):
                image[i][j] = img[i][j]

    else:
        img = cv2.resize(img, (640, int(640/ratio)), interpolation = cv2.INTER_AREA)
        for i in range(int(640/ratio)):
            for j in range(640):
                image[i][j] = img[i][j]


    im = Image.fromarray(image)
    im.save("static/Frame.png")  
    

    img = img_original
    img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
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
    #img = cv2.imread('Kitchen.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img2 = cv2.imread('Fan_Mobile.jpeg')
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#    print('Our two images are')
#     plt.imshow(img)
#     plt.show()
#     plt.imshow(img2)
#     plt.show()
    
    print('Objects are: ')
    all_objects = []
    loop = 1
    
 
    object_id = 1                      # object id
    
        # Argument Parser


    parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')

    parser.add_argument('--model', default='nyu.h5', type=str, help='Trained Keras model file.')

    parser.add_argument('--input', default='static/Frame.png', type=str, help='Input filename or folder.')

    #args = parser.parse_args()
    args, unknown = parser.parse_known_args()
    # Custom object needed for inference and training

    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

    print('Loading model...')

    # Load model into GPU / CPU
    model = load_model(args.model, custom_objects=custom_objects, compile=False)

    print('\nModel loaded ({0}).'.format(args.model))

    # Input images
    inputs = load_images( glob.glob(args.input) )
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)
    print("PREDICTION COMPLETE")
    depth_array = np.squeeze(outputs)

    im = Image.fromarray(depth_array * 255)
    print(depth_array)
    im = im.convert('RGB')
    im.save("static/DepthMap.png")  
    print('DEPTHMAP')

    while (loop):
        #_, frame = camera.read()
        print("OBJECT DETECTION")

        
        frame = img

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

        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)


        network.setInput(blob)  # setting blob as input to the network
        #start = time.time()
        output_from_network = network.forward(layers_names_output)
        #end = time.time()
        #print('Current frame took {:.5f} seconds'.format(end - start))



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
            
        

    
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

        
        if len(results) > 0:
        # Going through indexes of results
            for i in results.flatten():
            #speech output
                single_frame = []
                
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
#                 plt.imshow(cv2.rectangle(frame, (x_min, y_min),
#                           (x_min + box_width, y_min + box_height),
#                           colour_box_current, 2))
#                 plt.show()
                
                im = Image.fromarray(cv2.rectangle(frame, (x_min, y_min),(x_min + box_width, y_min + box_height),colour_box_current, 2))

                single_frame.append(object_id)
                single_frame.append(obj)
                single_frame.append((int(x_min/2),int(y_min/2),int(box_width/2),int(box_height/2)))
#                 print(obj)
#                 print(x_min)
#                 print(y_min)
#                 print(box_width)
#                 print(box_height)
                
                frame_information.append(single_frame)
                object_id += 1
                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])
            
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
            im.save("static/Detection.png")
            print('DETECTION')
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
        
        loop = 0
        """
        End of:
        Showing processed frames in OpenCV Window
        """

    """
    End of:
    Reading frames in the loop
    """
    k = 0
    distances = []
    frames = []


    for box in frame_information:

        img = depth_array

        depth_sum = 0

        x_min = box[2][0]
        y_min = box[2][1]
        width = box[2][2]
        height = box[2][3]

        copy = np.zeros((height, width))
        frames.append(copy)
        minimum = 1
        for i in range(y_min, y_min + height):
            for j in range(x_min, x_min + width):
                depth_sum += img[i][j]
                if(minimum > img[i][j]):
                    minimum = img[i][j]

                copy[i-y_min][j-x_min] = img[i][j]

        distances.append(minimum)

        #plt.imshow(copy, cmap='gray')
        #plt.show()

        k+=1
    if(not distances):
        return 
    
    closest = distances[0]
    index = 0
    for i in range(len(distances)):
        if(closest > distances[i]):
            closest = distances[i]
            index = i

    print('\n\nClosest object:')
    print(frame_information[index])
    #plt.imshow(frames[index])
    #plt.show()
    
    language = 'en'
    text=[]
    text='A ' 
    closest_obj = frame_information[index][1]
    text += closest_obj
    text += ' is in front of you'
    speech = gTTS(text=text, lang=language, slow = False)
    print(type(speech))
    speech.save('text.wav')
    
    #os.system('text.wav')
    

    # Releasing camera
    #camera.release()
    # Destroying all opened OpenCV windows
    cv2.destroyAllWindows()
    #return send_file('text.wav')

    img = Image.fromarray(img_original.astype('uint8'))

    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)

    return
    #return send_file(file_object, mimetype='image/PNG')
    #return str(frame_information[index])


def run():
    cap = cv2.VideoCapture(0)

    while(True):

        ret, frame = cap.read()

        gray = frame

        if cap is None or not cap.isOpened():
            print('Camera Not Working')
            break
        
        return yolo(gray)

        cv2.imshow('frame',gray)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        break

    cap.release()
    cv2.destroyAllWindows()

