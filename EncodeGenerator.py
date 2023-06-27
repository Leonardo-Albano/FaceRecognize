import cv2
import face_recognition
import pickle
import os
# import FaceModule as fm

# detector = fm.FaceModule
folderPath = 'Photos'
pathList = os.listdir(folderPath)
imgList = []
userIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    userIds.append(os.path.splitext(path)[0])
print(userIds)

def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        # img, encode = detector.findFaceMesh(img, False)[0]
        encodeList.append(encode)

    return encodeList

print("Encoding Started")
encodeListKnow = findEncodings(imgList)
encodeListKnowWithIds = [encodeListKnow, userIds]
# print(len(encodeListKnow))
print("Encoding Complete")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnowWithIds, file)
file.close()
print("File Saved")
