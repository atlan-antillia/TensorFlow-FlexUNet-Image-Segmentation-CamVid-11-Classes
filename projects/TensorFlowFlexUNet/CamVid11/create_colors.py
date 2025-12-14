import cv2
import os
import numpy as np
import shutil

def create_color_map(output_dir):

  print("<html><body>")

  """
  rgb_map = 
  {
  (0,0,0):0,
  (237,237,237):1,
  (181,0,0):2,
  (135,135,135):3,
  (189,107,0):4,
  (128,0,128):5,
  (31,123,22):6,
  (6,0,130):7, 
  (0,168,255):8,
  (240,255,0):9,}
  """

  COLORS = {
  "Outdoorstructures":(237,237,237),
  "Buildings":(181,0,0),
  "Pavedground":(135,135,135),
  "Non-pavedground":(189,107,0),
  "Traintracks":(128,0,128),
  "Plants":(31,123,22),
  "Wheeledvehicles":(6,0,130),
  "Water":(0,168,255),
  "People":(240,255,0)
  }

  keys = COLORS.keys()
  print("<table border=1 style='border-collapse:collapse;' cellpadding='5'>")
  print("<caption>Drone images 9 classes</caption>")
  print("<tr><th>Indexed Color</th><<th>Color</th><th>RGB</th><th>Class</th></tr>")
  index = 0
  for key in keys:
      color = COLORS[key]
      #print("{}  color {}".format(key, color))             

      w = 40
      h = 25

      (r,g,b) = color
      bgr_color = (b,g,r) 
      index += 1
      image = np.zeros((h, w, 3), dtype=np.uint8)
      cv2.rectangle(image, (0, 0), (w, h), bgr_color, cv2.FILLED, cv2.LINE_AA)
      filename = key + ".png"
      output_file = os.path.join(output_dir, filename)
      cv2.imwrite(output_file, image)
    
      print("<tr><td>{}</td><td with='80' height='auto'><img src='{}' widith='40' height='25'</td><td>{}</td><td>{}</td></tr>".format(index,output_file, color, key) )
  
  print("</table>") 
  print("</body></html>")

output_dir = "./color_class_mapping/"
if os.path.exists(output_dir):
   shutil.rmtree(output_dir)
os.makedirs(output_dir)

create_color_map(output_dir)
