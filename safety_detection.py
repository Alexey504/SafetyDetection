from ultralytics import YOLO
import cv2
import cvzone
import math

# Настройка камеры
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# для видео
cap = cv2.VideoCapture('ppe.mp4')

model = YOLO('safety_model.pt')

class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest',
               'machinery', 'vehicle']
my_color = (0, 0, 255)

while True:
    # Запуск камеры
    sucsess, img = cap.read()
    results = model(img, stream=True)
    # Нахожждение границ
    for res in results:
        boxes = res.boxes
        for box in boxes:
            # отрисовка границ
            # x1, y1, x2, y2 = map(int, box.xyxy[0])
            # print(x1, y1, x2, y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 3)

            # отрисовка границ(2 вариант)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            # cvzone.cornerRect(img, (x1, y1, w, h))


            # вероятность и отрисовка вероятности
            conf = math.ceil((box.conf[0] * 100)) / 100
            # print(conf)

            # имя класса
            cls = int(box.cls[0])
            curren_class = class_names[cls]

            if conf > 0.5:
                # проверка наличия защитных средств
                if curren_class in ('Hardhat', 'Mask', 'Safety Cone', 'Safety Vest'):
                    my_color = (0, 255, 0)
                elif curren_class in ('NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'):
                    my_color = (0, 0, 255)
                else:
                    my_color = (255, 0, 0)


                cvzone.putTextRect(img, f'{class_names[cls]} {conf}', (max(0, x1), max(35, y1)),
                                   scale=2, thickness=1, colorB=my_color, colorT=(255, 255, 255), colorR=my_color)
                cv2.rectangle(img, (x1, y1), (x2, y2), my_color, 3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
