import numpy as np
import cv2
import urllib.request
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime
 
def get_date():
    date_time = datetime.now() # auto generated
    date_time = date_time.strftime("%G_")+date_time.strftime("%m_")+date_time.strftime("%d")
    date_time = str(date_time)
    return date_time

# resize image for im.show visualizing purposes
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

def draw_text(img, text, pos):
    pil_image = Image.fromarray(img) # convert image to pil
    draw = ImageDraw.Draw(pil_image) # create image draw object
    font = ImageFont.truetype("Aperto_Modules/fonts/Raleway-SemiBold.ttf", 50) # must be filepath to directory if on local machine
    
    # Write the text to the image
    draw.text(pos, text, font=font, fill="black")
    
    # convert PIL image back to CV2
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return img

# draw the rectangles to form tables with text to go inside...
def draw_tables(img_blank, values):
    
    # Set the width and height of each rectangle
    rect_width = int(img_blank.shape[1]/2)
    rect_height = 60

    # Set the font and scale for the text
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2

    # Iterate over the 5 rows and 2 columns to draw the rectangles and text
    index = 0
    print(len(values))
    x_offset = 0
    y_offset= 90
    left_column_shrink_val = 1.5 # value: 2 is 50%, > 2 decreases size of left side. < 2 increases size of left side.
    for row in range(int(len(values)/2)):
        for col in range(2):
            # Calculate the top-left corner of the rectangle
            x1 = col * rect_width + x_offset
            y1 = row * rect_height + y_offset
            # Calculate the bottom-right corner of the rectangle
            x2 = x1 + rect_width
            y2 = y1 + rect_height
            
            # Adjust the coordinates for the left column
            if col == 0:
                x2 = int(x1 + (rect_width*2) /left_column_shrink_val)
            
            if col== 1:
                x1 = col * int((rect_width*2)/left_column_shrink_val) + x_offset
                
            # Draw the rectangle on the blank image
            cv2.rectangle(img_blank, (x1, y1), (x2, y2), (0, 0, 0), thickness=1)
            # Calculate the position of the text
            text_x = x1 + 10
            text_y = y1 + int(rect_height/2) + 10
            
            # Get the value to display from the array
            index = row * 2 + col
            value = values[index]
            # Draw the text on the image
            cv2.putText(img_blank, value, (text_x, text_y), font, scale, (0, 0, 0), thickness)
            index = index+1        
    # return the resulting image to main function
    return img_blank

def draw_image_stats(values):
    # start with blank canvas...
    height = int(90+((len(values)/2)*60))
    img_blank = np.zeros((height,500,3), dtype=np.uint8)
    img_blank.fill(255)
    img_blank = draw_tables(img_blank, values)

    date_string = get_date()
    img_blank = draw_text(img_blank, date_string, pos=(15, 15))

    cv2.imshow('Athlete Report',img_blank)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img_blank

    # gallery_fname = event['value']['fields']['filename']['stringValue']+"_"+get_date()+"_Kinogram.jpg"
    # gallery_fname = "Sprinting_Photo_Sequence_"+event['timeCreated']+".jpg"
    # #gallery_fname = "test_kinogram.jpg"
    # print("successfully compiled to gallery")

    # cv2.imwrite("/tmp/"+gallery_fname,vis_3)