import cv2
import mediapipe as mp
import os
import json

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

PATH_VIDEOS = 'Videos etiquetados'
videos = os.listdir(PATH_VIDEOS)

annotated_list = []

for video in videos:
  vidcap = cv2.VideoCapture(PATH_VIDEOS + '/' + video)
  success,image = vidcap.read()
  count = 0
  IMAGE_FILES = []
  label=video.replace('.mp4','')
  os.makedirs('Frames/'+label, exist_ok=True)
  FRAMES_DIR='Frames/'+label
  os.makedirs('Annotated/' + label, exist_ok=True)
  ANNOTATED_DIR = 'Annotated/' + label
  while success:
    filename=  FRAMES_DIR+"/frame_"+ label +"_"+ str(count)+".jpg"
    cv2.imwrite(filename, image)     # save frame as JPEG file
    success,image = vidcap.read()
    #print('Read a new frame: ', success)
    count += 1
    IMAGE_FILES.append(filename)

  with mp_hands.Hands(
      static_image_mode=True,
      max_num_hands=1,#21/08/21 use 1 hand
      min_detection_confidence=0.5) as hands:
    for idx, file in enumerate(IMAGE_FILES):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

      # Print handedness and draw hand landmarks on the image.
      #print('Handedness:', results.multi_handedness)

      if not results.multi_hand_landmarks:
        continue
      image_height, image_width, _ = image.shape
      annotated_image = image.copy()
      dict_nodes={}
      for hand_landmarks in results.multi_hand_landmarks:
        '''
        print('hand_landmarks:', hand_landmarks)
        print(
            f'Index finger tip coordinates: (',
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
        )
        '''
        #dict_nodes["image_width"] = image_width
        #dict_nodes["image_height"] = image_height
        for nodes_hand in range(len(mp_hands.HandLandmark._member_names_)):
          dict_nodes[mp_hands.HandLandmark._member_names_[nodes_hand]]={'x': hand_landmarks.landmark[nodes_hand].x,
                                                                        'y': hand_landmarks.landmark[nodes_hand].y}

        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      cv2.imwrite(ANNOTATED_DIR+'/annotated_image_' + label +"_"+ str(idx) + '.png', cv2.flip(annotated_image, 1))
      img_dir=".//"+label+'/annotated_image_' + label +"_"+ str(idx) + '.png'
      annotated_list.append(img_dir)

      json_object = json.dumps(dict_nodes, indent=4)
      with open(ANNOTATED_DIR+'/annotated_nodes_' + label +"_"+ str(idx) + '.json', "w") as outfile:
        outfile.write(json_object)
      json_dir = ".//" + label + '/annotated_nodes_' + label + "_" + str(idx) + '.json'
      annotated_list.append(json_dir)


textfile = open("Annotated/summary.txt", "w")
for element in annotated_list:
    textfile.write(element + "\n")
textfile.close()