import face_recognition
import cv2

# 加载已知人脸的图片文件
HuGe_image = face_recognition.load_image_file("E:\人工智能\人脸识别\Hu.jpg")  # 加载胡歌的人脸图片
PengYuYan_image = face_recognition.load_image_file("E:\人工智能\人脸识别\Peng.jpg")  # 加载彭于晏的人脸图片

# 定位图片中的人脸位置（返回人脸的上下左右坐标）
HuGe_locations = face_recognition.face_locations(HuGe_image)  # 定位胡歌图片中的人脸
PengYuYan_locations = face_recognition.face_locations(PengYuYan_image)  # 定位彭于晏图片中的人脸

# 提取人脸的特征编码（128维向量，作为人脸的唯一标识），取第一个检测到的人脸特征[0]
HuGe_encodings = face_recognition.face_encodings(HuGe_image, HuGe_locations)[0]  # 提取胡歌人脸的特征编码
PengYuYan_encodings = face_recognition.face_encodings(PengYuYan_image, PengYuYan_locations)[0]  # 提取彭于晏人脸的特征编码

# 构建已知人脸特征编码列表（用于后续比对）
known_faces = [
    HuGe_encodings,  # 胡歌的人脸特征
    PengYuYan_encodings  # 彭于晏的人脸特征
]

# 加载需要识别的未知人脸图片
unknown_image = face_recognition.load_image_file("E:\人工智能\人脸识别\_1.jpg")
# 提取未知图片中所有人脸的特征编码（可能包含多张人脸）
unknown_face_encodings = face_recognition.face_encodings(unknown_image)

# 遍历未知图片中检测到的每一张人脸特征
for unknown_face_encoding in unknown_face_encodings:
    # 将未知人脸特征与已知人脸特征库比对，tolerance=0.4为匹配阈值（值越小匹配越严格）
    results = face_recognition.compare_faces(known_faces, unknown_face_encoding, tolerance=0.4)

    # 输出匹配结果
    if results[0]:
        print("未知人脸与胡歌匹配")
    elif results[1]:
        print("未知人脸与彭于晏匹配")
    else:
        print("未知人脸与已知面孔不匹配")

    # 获取未知图片中所有人脸的位置坐标
    face_locations = face_recognition.face_locations(unknown_image)

    # 遍历人脸位置，绘制人脸框（匹配成功为绿色，失败为红色）
    for (top, right, bottom, left), result in zip(face_locations, results):
        color = (0, 255, 0) if result else (0, 0, 255)  # 匹配成功：绿色(0,255,0)，失败：红色(0,0,255)
        cv2.rectangle(unknown_image, (left, top), (right, bottom), color, 2)  # 绘制人脸矩形框，线宽为2

    # 显示标记后的图片（注意：face_recognition读取的是RGB格式，OpenCV显示需转为BGR格式）
    cv2.imshow("Unknown Image", cv2.cvtColor(unknown_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)  # 等待按键输入后关闭窗口（按任意键继续）
    cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口