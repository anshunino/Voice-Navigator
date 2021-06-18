import numpy as np
import cv2
import time
from gtts import gTTS 
import os
import matplotlib.pyplot as plt
import os
import glob
import argparse
import matplotlib
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt
from PIL import Image  
import PIL

#reading images 
#img1=cv2.imread('Kitchen.png',1)

def yolo(img):
    
    h, w = None, None
    
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

    
    #loading classes and model
    classesFile = 'coco.names'
    with open(classesFile) as f:
        classes=[line.strip() for line in f]

    modelConfig = 'yolov3.cfg'
    modelWeights = 'yolov3.weights'

    net = cv2.dnn.readNetFromDarknet(modelConfig , modelWeights)

    #initializing parameters
    confThreshold = 0.5
    nmsThreshold = 0.3
    colours = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # depth map
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
    depth_array = np.squeeze(outputs)

    im = Image.fromarray(depth_array * 255)
    print(depth_array)
    im = im.convert('RGB')
    im.save("static/DepthMap.png")  
    print('DEPTHMAP')
   
    # yolo object detection

    # getting ending layers
    def getOutputsName(net):
        layerNames = net.getLayerNames()
        return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # converting image to an input blob
    blob1 = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), swapRB = True, crop=False)


    blob=blob1
    objects=[]

    #implementing forward pass
    net.setInput(blob)
    start = time.time()
    outputFromNet = net.forward(getOutputsName(net))  
    end = time.time()
    print('Current frame took {:.5f} seconds'.format(end - start))

    #getting bounding boxes
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in outputFromNet:
        for detectedObjects in result:
            scores = detectedObjects[5:]
            classCurrent = np.argmax(scores)
            confidenceCurrent = scores[classCurrent]

            #removing boxes with confidence score less than confThreshold
            if confidenceCurrent > confThreshold:
                imgHeight = img.shape[0]
                imgWidth = img.shape[1]
                x_center = int(detectedObjects[0] * imgWidth)
                y_center = int(detectedObjects[1] * imgHeight)
                box_width = int(detectedObjects[2] * imgWidth)
                box_height = int(detectedObjects[3] * imgHeight)

                class_numbers.append(classCurrent)
                confidences.append(float(confidenceCurrent))
                bounding_boxes.append([x_center, y_center, box_width, box_height])

    #performing non maximum suppression 
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, confThreshold, nmsThreshold)

    #drawing predicted bounding boxes
    if len(results) >0:
        o=[]
        for i in results.flatten():
            obj = classes[int(class_numbers[i])]
            o.append(obj)
            x_center, y_center = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            x_min = int(x_center - box_width/2)
            y_min = int(y_center - box_height/2)
            colour_box_current = colours[class_numbers[i]].tolist()
            cv2.rectangle(img, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            text_box_current = '{}: {:.4f}'.format(classes[int(class_numbers[i])],confidences[i])
            im = Image.fromarray(cv2.putText(img, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2))

        im.save("static/Detection.png")
        print('DETECTION')
        objects.append(o)


    # coonverting image returned from depth map to grey scale
    b=depth_array
    uint_img = np.array(b*255).astype('uint8')
    grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)

    # finding max pixel value in each bbs(returned from yolo model)
    if len(results) >0:
        o={}
        y_ratio = b.shape[0]/img.shape[0]
        x_ratio = b.shape[1]/img.shape[1]
        for i in results.flatten():
            obj = classes[int(class_numbers[i])]
            x_center, y_center = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            x_min = (x_center - box_width/2) * x_ratio
            y_min = (y_center - box_height/2) * y_ratio
            x_min = int(x_min)
            y_min = int(y_min)
            box_width = int(box_width*x_ratio)
            box_height = int(box_height*y_ratio)
            m=b[y_min,x_min]
            for i in range(x_min, x_min+box_width):
                for j in range(y_min, y_min+box_height):
                    m=min(m,b[j][i])
            o[obj]=m

    # sorting to get objects in increasing order of distance from user  
    depthAns = sorted(o.items(), key = lambda kv:(kv[1], kv[0])) 


    # generating audio file

    language = 'en'
    text=[]
    text='Objects in increasing order of distance from user are '
    for i in range(0,len(depthAns)):
        if(len(depthAns) == 1):
            text='The object infront of the user is a ' + depthAns[0][0] 
            break

        if(i == len(depthAns)-1):
            text=text+'and '+depthAns[i][0] + ' '
        else:
            text+=depthAns[i][0]+', '

    print(text)
    speech = gTTS(text=text, lang=language, slow = False)
    speech.save('static/text.wav')
    text_file = open("static/Output.txt", "w")
    text_file.write("%s" % text)
    text_file.close()

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