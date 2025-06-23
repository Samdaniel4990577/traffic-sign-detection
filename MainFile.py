#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import streamlit as st
from PIL import Image,ImageFilter
import matplotlib.image as mpimg
from tensorflow.keras.layers import Dense, Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
import streamlit as st

#====================== READ A INPUT IMAGE =========================

filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)
plt.title('Original Image') 
plt.axis ('off')
plt.show()

st.image(img)
    
    
#============================ PREPROCESS =================================

#==== RESIZE IMAGE ====

resized_image = cv2.resize(img,(300,300))
img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(resized_image)
plt.axis ('off')
plt.show()
   
             
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1,cmap='gray')
plt.axis ('off')
plt.show()
    
    
# ============== FEATURE EXTRACTION ==============


#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("====================================")
print("        Feature Extraction          ")
print("====================================")
print()
print(features_extraction)

# ==== LBP =========

import cv2
import numpy as np
from matplotlib import pyplot as plt
   
      
def find_pixel(imgg, center, x, y):
    new_value = 0
    try:
        if imgg[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value
   
# Function for calculating LBP
def lbp_calculated_pixel(imgg, x, y):
    center = imgg[x][y]
    val_ar = []
    val_ar.append(find_pixel(imgg, center, x-1, y-1))
    val_ar.append(find_pixel(imgg, center, x-1, y))
    val_ar.append(find_pixel(imgg, center, x-1, y + 1))
    val_ar.append(find_pixel(imgg, center, x, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y + 1))
    val_ar.append(find_pixel(imgg, center, x + 1, y))
    val_ar.append(find_pixel(imgg, center, x + 1, y-1))
    val_ar.append(find_pixel(imgg, center, x, y-1))
    power_value = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_value[i]
    return val
   
   
height, width, _ = img.shape
   
img_gray_conv = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   
img_lbp = np.zeros((height, width),np.uint8)
   
for i in range(0, height):
    for j in range(0, width):
        img_lbp[i, j] = lbp_calculated_pixel(img_gray_conv, i, j)

plt.imshow(img_lbp, cmap ="gray")
plt.title("LBP")
plt.show()

    
#============================ 5. IMAGE SPLITTING ===========================

import os 

from sklearn.model_selection import train_test_split

test= os.listdir('TrafficSign/images')
train = os.listdir('TrafficSign/images')

#       
dot1= []
labels1 = [] 
for img11 in test:
        # print(img)
        img_1 = mpimg.imread('TrafficSign/images//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(1)


for img11 in train:
        # print(img)
        img_1 = mpimg.imread('TrafficSign/images//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(2)


x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)

print()
print("-------------------------------------")
print("       IMAGE SPLITTING               ")
print("-------------------------------------")
print()


print("Total no of data        :",len(dot1))
print("Total no of test data   :",len(x_train))
print("Total no of train data  :",len(x_test))

    
# ====================== CLASSIFICATION ================

# ==== RESNET ==

from keras.utils import to_categorical

from tensorflow.keras.models import Sequential

import tensorflow as tf
from tensorflow.keras import layers, models

# Define the identity block for ResNet
def identity_block(x, filters):
    f1, f2 = filters

    # Shortcut
    shortcut = x

    # First component of main path
    x = layers.Conv2D(f1, (1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second component of main path
    x = layers.Conv2D(f2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Third component of main path
    x = layers.Conv2D(filters[0], (1, 1), padding='valid')(x)
    x = layers.BatchNormalization()(x)

    # Add the shortcut to the output
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# Define the convolutional block for ResNet
def convolutional_block(x, filters, s=2):
    f1, f2 = filters

    # Shortcut
    shortcut = x

    # First component of main path
    x = layers.Conv2D(f1, (1, 1), strides=(s, s))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Second component of main path
    x = layers.Conv2D(f2, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Third component of main path
    x = layers.Conv2D(f1, (1, 1))(x)
    x = layers.BatchNormalization()(x)

    # Shortcut path
    shortcut = layers.Conv2D(f1, (1, 1), strides=(s, s))(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    # Add the shortcut to the output
    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x

# Build the ResNet model
input_shape = (50, 50, 3)
input_tensor = tf.keras.layers.Input(shape=input_shape)

x = layers.ZeroPadding2D((3, 3))(input_tensor)
x = layers.Conv2D(64, (7, 7), strides=(2, 2))(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.ZeroPadding2D((1, 1))(x)
x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

x = convolutional_block(x, [64, 256])
x = identity_block(x, [64, 256])
x = identity_block(x, [64, 256])

x = convolutional_block(x, [128, 512])
x = identity_block(x, [128, 512])
x = identity_block(x, [128, 512])
x = identity_block(x, [128, 512])

x = convolutional_block(x, [256, 1024])
x = identity_block(x, [256, 1024])
x = identity_block(x, [256, 1024])
x = identity_block(x, [256, 1024])
x = identity_block(x, [256, 1024])

x = convolutional_block(x, [512, 2048])
x = identity_block(x, [512, 2048])
x = identity_block(x, [512, 2048])

# x = layers.AveragePooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(1, activation='softmax')(x)

model = models.Model(inputs=input_tensor, outputs=x)

# Compile the model
model.compile(optimizer='adam', loss='mae')

# Display the model summary
model.summary()

y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)


x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]


history = model.fit(x_train2,y_train1,batch_size=50,epochs=2,verbose=1)

print("--------------------------------------------")
print("                     Resnet                 ")
print("--------------------------------------------")
print()
loss=history.history['loss']
error_res =max(loss)*10
acc_res=100-error_res
print()
print("1.Accuracy is :",acc_res,'%')
print()
print("2.Loss is     :",error_res)
print()


    
    
    
#  ---- DEEP CNN


# Build the CNN model
model = models.Sequential()

# Convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(50, 50, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='softmax'))  # 10 classes for CIFAR-10

# Compile the model
model.compile(optimizer='adam', loss='mae')

# Train the model
history= model.fit(x_train2, y_train1, epochs=10, verbose=1)



loss=history.history['loss']
error_dcnn =max(loss) 
acc_dcnn=100-error_dcnn
print()
print("1.Accuracy is :",acc_dcnn,'%')
print()
print("2.Loss is     :",error_dcnn)
print()




# ================ YOLO v-8

import cv2
import numpy as np

def detect_objects(image_path, config_path, weights_path):
    net = cv2.dnn.readNet(config_path, weights_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    img = cv2.imread(image_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(class_ids), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



import xml.etree.ElementTree as ET
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimg
import matplotlib.pyplot as plt 


inpimg = mpimg.imread(filename)

aa= filename.split('/')

aa3 = aa[len(aa)-1]


ff = 'TrafficSign/annotations/'+str(aa3[0:len(aa3)-4])+'.xml'



import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os

def draw_bounding_boxes(image_path, annotations_folder='annotations'):
    # Get the image filename without extension
    image_filename = os.path.basename(image_path)
    image_name, _ = os.path.splitext(image_filename)

    # Construct the path to the XML annotation file (assuming it's in the annotations folder)
    xml_path =ff

    # Check if the XML file exists
    if not os.path.exists(xml_path):
        print(f"Error: XML file {xml_path} not found.")
        return

    # Open the image file
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Loop through all 'object' elements in the XML
    for obj in root.findall('object'):
        # Get the bounding box coordinates
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        
        # Extract the coordinates (xmin, ymin, xmax, ymax)
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        # Draw the bounding box on the image
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Optionally, annotate the image with the object class name
        draw.text((xmin, ymin), name, fill="red")

    # Show or save the image with bounding boxes
    # img.show()
    plt.imshow(img)
    img.save(f'output_with_bboxes_{image_name}.jpg')

# Example usage
image_path = filename
draw_bounding_boxes(image_path)



# ff = filename[0:len(filename)-4]+str('.xml')
xml_data = open(ff, 'r').read()  # Read file
root = ET.XML(xml_data)  # Parse XML

data = []
cols = []
vval = []
cols1 = []
vval2 = []

for i, child in enumerate(root):
    data.append([subchild.text for subchild in child])
    cols.append(child.tag)
    vval2.append([subchildds.text for subchildds in child])
    vval.append([subchildd.tag for subchildd in child])



import pandas as pd

df = pd.DataFrame(data).T  
df.columns = cols  

df1 = pd.DataFrame(vval).T  
df1.columns = cols 


# import xmltodict
# import pandas as pd

# xml_data = open(ff, 'r').read() 
# xmlDict = xmltodict.parse(xml_data)  

# colsss = xmlDict['annotation']

# try:    
#     AZ = inpimg 
#     for iii in range(len(colsss['object'])):  # Access 'object' as a list of objects
#         Dims = colsss['object'][iii]['bndbox']
        
#         D1 = int(Dims['xmax'])
#         D2 = int(Dims['xmin'])
#         D3 = int(Dims['ymax'])
#         D4 = int(Dims['ymin'])
        
#         import cv2
#         AZ = cv2.rectangle(AZ, (D1, D4), (D2, D3), (255,0,0), 3)
# except:
#     # If there is only one object, handle it accordingly
#     Dims = colsss['object']['bndbox']  # Here 'object' should not be accessed with a string key
#     D1 = int(Dims['xmax'])
#     D2 = int(Dims['xmin'])
#     D3 = int(Dims['ymax'])
#     D4 = int(Dims['ymin'])
    
#     import cv2
#     AZ = cv2.rectangle(inpimg, (D1, D4), (D2, D3), (255,0,0), 3)

# plt.imshow(AZ)
# plt.title('DETECTED IMAGE')
# plt.show()


try:
    DD = df['object']
    print("------------------------------")
    print(" IDENTIFIED  = ",DD[0])
    print("------------------------------")

    # print(DD[0])
except:
    print(' ')
    aa = df['object'][:1]
    print(aa)

res = DD[0]

if  res =="trafficlight" or res == "Traffic sign":
    print("Red indicates the vehicles to stop, yellow indicates the vehicles to slow down and get ready to stop, and the green light indicates the vehicles to go ahead.")
    
elif res == "Stop":
    print("Standard stop signs are red octagons with “STOP” printed in white letters. When you see one at any corner or intersection, know that you must stop and proceed only if the way ahead is clear, and after obeying any rules regarding right-of-way.")
    

elif res == "Speedlimit":
    print("Speed limits on road traffic, as used in most countries, set the legal maximum speed at which vehicles may travel on a given stretch of road. Speed limits are generally indicated on a traffic sign reflecting the maximum permitted speed, expressed as kilometres per hour (km/h) or miles per hour (mph) or both")
    

elif res == "Crosswalk":
    print("Do NOT start to cross when you see a steady orange hand symbol or orange DON'T WALK signal Wait until the white walking person symbol or the white WALK signal is lit before starting to cross. Always look to see if there is a push button for the pedestrian signal.")
    
    
