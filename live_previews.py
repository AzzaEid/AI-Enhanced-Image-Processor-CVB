       
import cv2
import numpy as np
from ultralytics import YOLO

# def detect_humans(image: np.ndarray):
#     # تحميل النموذج
#     model = YOLO('yolov8n.pt')

#     # إجراء الكشف
#     results = model(image)

#     # تصفية النتائج
#     filtered_boxes = []
#     for result in results:
#         boxes = result.boxes
#         for box in boxes:
#             confidence = box.conf[0]  # الثقة
#             if confidence > 0.5 and box.cls[0] == 0:  # 0 هو ID الشخص في COCO
#                 x1, y1, x2, y2 = box.xyxy[0]
#                 filtered_boxes.append((x1, y1, x2, y2, confidence))

#     # رسم النتائج على الصورة
#     for (x1, y1, x2, y2, confidence) in filtered_boxes:
#         cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
#         cv2.putText(image, f'Person: {confidence:.2f}', (int(x1), int(y1) - 10), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
#     cv2.imshow('Detected Persons', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     #return image  # إرجاع الصورة مع النتائج

# # استخدام الدالة
# if __name__ == "__main__":
#     # هنا يمكنك إرسال الصورة ككائن NumPy
#     image_path = 'path/to/stitched_image.jpg'  # فقط مثال لتحميل صورة
#     image = cv2.imread(image_path)  # قراءة الصورة

#     # استدعاء الدالة للكشف عن الأشخاص
#     output_image = detect_humans(image)

#     # عرض الصورة مع النتائج
#     cv2.imshow('Detected Persons', output_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

class live():
    def __init__(self):
        self.vaild = True
    
    def canny(self):
        print("hereeee")
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Could not open video device")
        # هيك أخذت غن بوت الكاميرا 
        while True:
            ret, frame = video.read() # بمسك فريم فريم عشان أعالجه
            if not ret or frame is None:
                print("No frame captured")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # يفضل الرمادي في شغلنا

            edges = cv2.Canny(gray, 100, 200)

            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # رجعت من رمادي عشان بحتاج يكون عندي 3 تشانلز

            combined = np.hstack((frame, edges_3d)) # بحتاج 3 تشانلز عشان اعرف ادمج الفريم الاصلي (3 تشانلز) مع الايدج الي توصلتله
            cv2.imshow('Video, Edges', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        video.release()
        cv2.destroyAllWindows()

    def DoG(self):
        morph_shape = cv2.MORPH_RECT
        size = 5
        mph_mask = cv2.getStructuringElement(morph_shape, (size, size))
        print("hereeee")
        video = cv2.VideoCapture(0)
        if not video.isOpened():
            print("Could not open video device")
        # هيك أخذت غن بوت الكاميرا 
        while True:
            ret, frame = video.read() # بمسك فريم فريم عشان أعالجه
            if not ret or frame is None:
                print("No frame captured")
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # يفضل الرمادي في شغلنا
            gaussian_1 = cv2.GaussianBlur(gray, (31,31), 3)
            gaussian_2 = cv2.GaussianBlur(gray, (31,31), 5)
            DoG = gaussian_1- gaussian_2
            edges = cv2.morphologyEx(DoG, cv2.MORPH_OPEN, mph_mask)
            
            edges_3d = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB) # رجعت من رمادي عشان بحتاج يكون عندي 3 تشانلز

            combined = np.hstack((frame, edges_3d)) # بحتاج 3 تشانلز عشان اعرف ادمج الفريم الاصلي (3 تشانلز) مع الايدج الي توصلتله
            cv2.imshow('Video, Edges', combined)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        video.release()
        cv2.destroyAllWindows()

    def detect_humans(self):
        # تحميل النموذج
        model = YOLO('yolo11n.pt')
        video = cv2.VideoCapture(0)
        while True:
            ret, frame = video.read() # بمسك فريم فريم عشان أعالجه
            if not ret or frame is None:
                print("No frame captured")
                break
            # إجراء الكشف
            # results = model(image)
            results = model(source=frame, show=False, conf=0.5, device='cpu')

            # تصفية الكائنات لتشمل البشر فقط (الفئة 0 في COCO dataset)
            # human_detections = [d for d in detections if int(d) == 0]
            h_count = 0
            for result in results:
                for box in result.boxes:
                    # الحصول على إحداثيات الصندوق
                    x1, y1, x2, y2 = box.xyxy[0]  # إحداثيات الصندوق
                    conf = box.conf[0]  # الثقة
                    cls = int(box.cls[0])  # فئة الكائن

                    if cls == 0 and conf > 0.2:  # فئة 0 هي "شخص"
                        h_count += 1
                        # رسم الصندوق
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'Person {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # عرض الإطار مع الكائنات المكتشفة
            cv2.imshow('Human Detection', frame)

            # الخروج عند الضغط على مفتاح 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # إغلاق الكاميرا
        video.release()
        cv2.destroyAllWindows()

