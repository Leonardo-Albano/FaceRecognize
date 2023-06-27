import pickle
import cv2
import time
import FaceModule as fm

cap = cv2.VideoCapture(1)
pTime = 0
detector = fm.FaceModule()

print("Loading Encoded File..")
file = open("EncodeFile.p", "rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, userIds = encodeListKnownWithIds
print(userIds)
print("Encoded File Loaded.")


while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img)
    if len(faces) != 0:
        print(faces)

    imgSmall = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    # --------- 50:35 - Face Recognition With Real Time Database ---------
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)