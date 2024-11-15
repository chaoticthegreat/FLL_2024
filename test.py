import cv2, queue, threading
from roboflow import Roboflow






class VideoCapture:

    def __init__(self, source, cap):
        self.cap = cv2.VideoCapture(source, cap)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        # self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)

        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon=True
        t.start()
    def _reader(self):

        while True:
            # print("hello")
            ret, frame = self.cap.read(1)
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)
    def read(self):
        return self.q.get()




cap = VideoCapture('/dev/video0', cv2.CAP_V4L)
rf = Roboflow(api_key="riNaV6Wbv5Gc3Nkq9NLl")
version = rf.workspace().project("rock-paper-scissors-sxsw").version(14)




x = 0
while x<1000:
    x+=1
    frame = cap.read()
    cv2.imshow('frame', frame)
    cv2.imwrite(filename="test.jpg",img=frame)
    prediction = version.model.predict("test.jpg")
    print(prediction.json())


    for bounding_box in prediction.json()['predictions']:
        x0 = bounding_box['x'] - bounding_box['width'] / 2
        x1 = bounding_box['x'] + bounding_box['width'] / 2
        y0 = bounding_box['y'] - bounding_box['height'] / 2
        y1 = bounding_box['y'] + bounding_box['height'] / 2

        start = (int(x0), int(y0))
        end = (int(x1), int(y1))
        cv2.rectangle(frame, start,end, color = (100,0,0), thickness=3)

        cv2.putText(frame, bounding_box['class'], (int(x0), int(y0) - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255, 255, 255), thickness = 2)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
cap.release()