import mediapipe as mp
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
colors = [(245,117,16), (117,245,16), (16,117,245)]

def media_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable - True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, mp_drawing.DrawingSpec(color = (10, 103, 110), thickness = 1, circle_radius = 1), mp_drawing.DrawingSpec(color = (242, 228, 42), thickness = 1, circle_radius = 1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def points2folder():
    Data_Path = os.path.join("Point Data")
    actions = np.array(['confidence %2.0f' %(i) + '%' for i in range(40,101,5)])
    tot_seq = 30
    seq_len = 30


    for action in actions: 
        for seq in range(tot_seq):
            try: 
                os.makedirs(os.path.join(Data_Path, action, str(seq)))
            except:
                pass


    with mp_holistic.Holistic(min_detection_confidence = 0.8, min_tracking_confidence = 0.2) as holistic:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            for action in actions:
                for seq in range(tot_seq):
                    for frames in range(seq_len):
                        ret, frame = cap.read()

                        image, results = media_detection(frame, holistic)

                        draw_landmarks(image, results)

                        if frames == 0:
                            cv2.putText(img=image, text='Starting Collection', org=(120, 200), 
                                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                                        color=(0, 0, 255), thickness=4, lineType=cv2.LINE_AA)
                            cv2.putText(image, "Collecting frames for {} video number{}".format(action, seq),
                                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('Data cam', image)
                            cv2.waitKey(2000)
                        else:
                            cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, seq),
                                        (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                            cv2.imshow('Data cam', image)

                        KeyPoints = extract_keypoints(results)
                        npy_path = os.path.join(Data_Path, action, str(seq), str(frames))
                        np.save(npy_path, KeyPoints)

                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            break
            break
        cap.release()
        cv2.destroyAllWindows()


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame


def draw_sentence_on_frame(sentence, image):
    # Draw the new recognized sentence
    cv2.putText(image, sentence, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return image



def testModel(path):
    actions = np.array(['confidence %2.0f' %(i) + '%' for i in range(30,101,5)])
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    m1 = load_model(path)

    # 1. New detection variables
    sequence = []
    sentence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = media_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = m1.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                cv2.putText(image, actions[np.argmax(res)], (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (196, 35, 56), 2, cv2.LINE_AA)

                
                
            #3. Viz logic
                '''if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                image = draw_sentence_on_frame(' '.join(sentence), image)'''
            
            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    with mp_holistic.Holistic(min_detection_confidence = 0.8, min_tracking_confidence = 0.2) as holistic:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()

            image, results = media_detection(frame, holistic)
            
            draw_landmarks(image, results)

            cv2.imshow('Test Window', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()