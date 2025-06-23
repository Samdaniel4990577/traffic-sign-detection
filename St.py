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
import base64
import cv2

# ================ Background image ===

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:38px;font-family:Caveat, sans-serif;">{"Traffic Signal Detection and Recognition using DL and IOT"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('1.avif')


#====================== READ A INPUT IMAGE =========================


# st.title("An Improved Traffic Sign Recognition and Road Lane Detection for Self Driving Cars Using YOLO-v8")

# st.text(" For Traffic Sign language")

# inp_img = st.file_uploader("UPLOAD  IMAGE", type=[".jpg",".png"])
# # st.text(Human_img)

# if inp_img==None:
#     st.text("Browse")
# else:
#     # filename = askopenfilename()
#     img = mpimg.imread(inp_img)
#     plt.imshow(img)
#     plt.title('Original Image') 
#     plt.axis ('off')
#     plt.show()
    
#     st.image(img)
    
    
#     #============================ PREPROCESS =================================
    
#     #==== RESIZE IMAGE ====
    
#     resized_image = cv2.resize(img,(300,300))
#     img_resize_orig = cv2.resize(img,((50, 50)))
    
#     fig = plt.figure()
#     plt.title('RESIZED IMAGE')
#     plt.imshow(resized_image)
#     plt.axis ('off')
#     plt.show()
       
             
#     #==== GRAYSCALE IMAGE ====
    
    
    
#     SPV = np.shape(img)
    
#     try:            
#         gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
        
#     except:
#         gray1 = img_resize_orig
       
#     fig = plt.figure()
#     plt.title('GRAY SCALE IMAGE')
#     plt.imshow(gray1,cmap='gray')
#     plt.axis ('off')
#     plt.show()
    
    

    
        
import streamlit as st
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

# Streamlit file uploader for selecting the image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image using PIL
    image_path = uploaded_file.name
    img = Image.open(uploaded_file)
    st.image(img)
    
    # Extract the filename without the extension
    aa = image_path.split('/')
    aa3 = aa[len(aa)-1]
    
    # Define the corresponding XML path
    ff = 'TrafficSign/annotations/'+str(aa3[0:len(aa3)-4])+'.xml'

    # Check if XML exists
    if not os.path.exists(ff):
        st.error(f"Error: XML file {ff} not found.")
    else:
        # Function to draw bounding boxes
        def draw_bounding_boxes(image, xml_path):
            draw = ImageDraw.Draw(image)
            
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

            return image
        
        # Call the function to draw bounding boxes
        img_with_bboxes = draw_bounding_boxes(img, ff)
        
        # Display the image with bounding boxes
        st.image(img_with_bboxes, caption="Image with Bounding Boxes", use_column_width=True)
        
        # Save the resulting image (optional)
        img_with_bboxes.save(f'output_with_bboxes_{aa3}')
        st.success("Bounding boxes have been drawn and saved successfully!")

    
        
        
    # import xml.etree.ElementTree as ET
    # from tkinter.filedialog import askopenfilename
    # import matplotlib.image as mpimg
    # import matplotlib.pyplot as plt
    # import cv2
    # import pandas as pd
    # import xmltodict
    
    # # Read the input image
    # inpimg = mpimg.imread(inp_img)  # Read image using matplotlib
    # inpimg = inpimg.copy()  # Make the image writable (to avoid OpenCV's "readonly" error)
    
    # # Extract file name (without extension)
    aa = uploaded_file.name
    ff = 'TrafficSign/annotations/' + str(aa[0:len(aa)-4]) + '.xml'
    st.text(ff)
    
    # # Read XML file
    # xml_data = open(ff, 'r').read()
    # root = ET.XML(xml_data)  # Parse XML
    
    # data = []
    # cols = []
    # vval = []
    # cols1 = []
    # vval2 = []
    
    # for i, child in enumerate(root):
    #     data.append([subchild.text for subchild in child])
    #     cols.append(child.tag)
    #     vval2.append([subchildds.text for subchildds in child])
    #     vval.append([subchildd.tag for subchildd in child])
    
    # # Use xmltodict to parse XML
    # xml_data = open(ff, 'r').read()
    # xmlDict = xmltodict.parse(xml_data)
    
    # # Extract object information from XML
    # colsss = xmlDict['annotation']
    
    # # Try to draw bounding boxes on the image
    # try:    
    #     AZ = inpimg  # Make a writable copy of the image
    #     for iii in range(0, len(colsss['object'])):
    #         Dims = colsss['object'][iii]['bndbox']
    
    #         D1 = int(Dims['xmax'])
    #         D2 = int(Dims['xmin'])
    #         D3 = int(Dims['ymax'])
    #         D4 = int(Dims['ymin'])
    
    #         AZ = cv2.rectangle(AZ, (D1, D4), (D2, D3), (255, 0, 0), 3)
    # except:
    #     # Handle single object case
    #     Dims = colsss['object']['bndbox']
    #     D1 = int(Dims['xmax'])
    #     D2 = int(Dims['xmin'])
    #     D3 = int(Dims['ymax'])
    #     D4 = int(Dims['ymin'])
        
    #     AZ = cv2.rectangle(inpimg, (D1, D4), (D2, D3), (255, 0, 0), 3)
    
    # # Display the image with bounding boxes
    # plt.imshow(AZ)
    # st.image(AZ)
    # plt.title('DETECTED IMAGE')
    # plt.show()
        
        
    
    
    # import xml.etree.ElementTree as ET
    # from PIL import Image, ImageDraw
    # import os
    
    # def draw_bounding_boxes(image_path, annotations_folder='annotations'):
    #     # Get the image filename without extension
    #     image_filename = os.path.basename(image_path)
    #     image_name, _ = os.path.splitext(image_filename)
    
    #     # Construct the path to the XML annotation file (assuming it's in the annotations folder)
    #     xml_path =ff
    
    #     # Check if the XML file exists
    #     if not os.path.exists(xml_path):
    #         print(f"Error: XML file {xml_path} not found.")
    #         return
    
    #     # Open the image file
    #     img = Image.open(image_path)
    #     draw = ImageDraw.Draw(img)
    
    #     # Parse the XML file
    #     tree = ET.parse(xml_path)
    #     root = tree.getroot()
    
    #     # Loop through all 'object' elements in the XML
    #     for obj in root.findall('object'):
    #         # Get the bounding box coordinates
    #         name = obj.find('name').text
    #         bndbox = obj.find('bndbox')
            
    #         # Extract the coordinates (xmin, ymin, xmax, ymax)
    #         xmin = int(bndbox.find('xmin').text)
    #         ymin = int(bndbox.find('ymin').text)
    #         xmax = int(bndbox.find('xmax').text)
    #         ymax = int(bndbox.find('ymax').text)
            
    #         # Draw the bounding box on the image
    #         draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    
    #         # Optionally, annotate the image with the object class name
    #         draw.text((xmin, ymin), name, fill="red")
    
    #     # Show or save the image with bounding boxes
    #     # img.show()
    #     plt.imshow(img)
    #     st.image(img)
    #     img.save(f'output_with_bboxes_{image_name}.jpg')
    
    # # Example usage
    # image_path = inp_img.name
    # draw_bounding_boxes(image_path)
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # inpimg = mpimg.imread(inp_img)
    
    # aa = inp_img.name
    
    # # st.text(aa)
    
    # # e
    # # aa= Human_img.split('/')
    
    # # aa3 = aa[len(aa)-1]
    # # st.text(aa3)
    
    
    # ff = 'TrafficSign/annotations/'+str(aa[0:len(aa)-4])+'.xml'
    # st.text(ff)
    
    # # ff = filename[0:len(filename)-4]+str('.xml')
    # xml_data = open(ff, 'r').read()  # Read file
    # root = ET.XML(xml_data)  # Parse XML
    
    # data = []
    # cols = []
    # vval = []
    # cols1 = []
    # vval2 = []
    
    # for i, child in enumerate(root):
    #     data.append([subchild.text for subchild in child])
    #     cols.append(child.tag)
    #     vval2.append([subchildds.text for subchildds in child])
    #     vval.append([subchildd.tag for subchildd in child])
    
    
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
   # res = DD[0]
    
    if  res =="trafficlight" or res == "Traffic sign":
        st.text("red indicates the vehicles to stop, yellow indicates the vehicles to slow down and get ready to stop, and the green light indicates the vehicles to go ahead.")
        a ="red indicates the vehicles to stop, yellow indicates the vehicles to slow down and get ready to stop, and the green light indicates the vehicles to go ahead."
    elif res == "Stop":
        st.text("Standard stop signs are red octagons with “STOP” printed in white letters. When you see one at any corner or intersection, know that you must stop and proceed only if the way ahead is clear, and after obeying any rules regarding right-of-way.")
        a = "Standard stop signs are red octagons with “STOP” printed in white letters. When you see one at any corner or intersection, know that you must stop and proceed only if the way ahead is clear, and after obeying any rules regarding right-of-way."
    
    elif res == "Speedlimit":
        st.text("Speed limits on road traffic, as used in most countries, set the legal maximum speed at which vehicles may travel on a given stretch of road. Speed limits are generally indicated on a traffic sign reflecting the maximum permitted speed, expressed as kilometres per hour (km/h) or miles per hour (mph) or both")
        a ="Speed limits on road traffic, as used in most countries, set the legal maximum speed at which vehicles may travel on a given stretch of road. Speed limits are generally indicated on a traffic sign reflecting the maximum permitted speed, expressed as kilometres per hour (km/h) or miles per hour (mph) or both"
    
    elif res == "Crosswalk":
        st.text("Do NOT start to cross when you see a steady orange hand symbol or orange DON'T WALK signal Wait until the white walking person symbol or the white WALK signal is lit before starting to cross. Always look to see if there is a push button for the pedestrian signal.")
        a ="Do NOT start to cross when you see a steady orange hand symbol or orange DON'T WALK signal Wait until the white walking person symbol or the white WALK signal is lit before starting to cross. Always look to see if there is a push button for the pedestrian signal."        
 
    
 
    from gtts import gTTS
        
    predictedtext = a
        
    language = 'en'
        
    myobj = gTTS(text=predictedtext, lang=language, slow=False)
    
    print("Text to speech conversion starting ...........")
    print()
    
    import os
    
    myobj.save("result.mp3")
    
    os.system("result.mp3")     
     

    audio_file = open('result.mp3', 'rb')
    audio_bytes = audio_file.read()
    
    st.audio(audio_bytes, format='audio/mp3')
    
    sample_rate = 44100  # 44100 samples per second
    seconds = 2  # Note duration of 2 seconds
    frequency_la = 440  # Our played note will be 440 Hz
    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, seconds * sample_rate, False)
    # Generate a 440 Hz sine wave
    note_la = np.sin(frequency_la * t * 2 * np.pi)
    
    st.audio(note_la, sample_rate=sample_rate)









