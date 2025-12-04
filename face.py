import face_recognition
import cv2

# Load the image file
HuGe_image = face_recognition.load_image_file("E:\人工智能\人脸识别\Hu.jpg")
PengYuYan_image = face_recognition.load_image_file("E:\人工智能\人脸识别\Peng.jpg")



# Find face locations in the image
HuGe_locations = face_recognition.face_locations(HuGe_image)
PengYuYan_locations = face_recognition.face_locations(PengYuYan_image)

HuGe_encodings = face_recognition.face_encodings(HuGe_image, HuGe_locations)[0]
PengYuYan_encodings = face_recognition.face_encodings(PengYuYan_image, PengYuYan_locations)[0]


known_faces = [
    HuGe_encodings,
    PengYuYan_encodings
]

# 加载未知图片
unknown_image = face_recognition.load_image_file("E:\人工智能\人脸识别\_1.jpg")
# 提取未知图片的特征向量
unknown_face_encodings = face_recognition.face_encodings(unknown_image)



# 处理每一张未知人脸
for unknown_face_encoding in unknown_face_encodings:
    # 比较未知人脸特征向量与已知人脸特征向量
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding,tolerance=0.4)
     # 输出匹配结果
    if results[0]:
        print("未知人脸与胡歌匹配")
    elif results[1]:
        print("未知人脸与彭于晏匹配")
    else:
        print("未知人脸与已知面孔不匹配")

    # 获取未知人脸位置
    face_locations = face_recognition.face_locations(unknown_image)
    #绘制人脸框
    for (top, right, bottom, left), result in zip(face_locations, results):
        color = (0, 255, 0) if result else (0, 0, 255)
        cv2.rectangle(unknown_image, (left, top), (right, bottom), color, 2)

    # 显示标记的未知图像
    cv2.imshow("Unknown Image", cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()



