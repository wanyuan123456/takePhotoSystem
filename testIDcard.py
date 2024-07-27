# import wx
# import cv2
# import pytesseract
# from PIL import Image
# from io import BytesIO
#
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 替换为你的实际安装路径
# print(pytesseract.get_tesseract_version())
#
# class MyFrame(wx.Frame):
#     def __init__(self, parent, title):
#         wx.Frame.__init__(self, parent, title=title, size=(800, 600))
#
#         # 创建主面板
#         panel = wx.Panel(self)
#
#         # 创建摄像头显示的画布
#         self.camera_display = wx.StaticBitmap(panel)
#
#         # 创建按钮
#         self.capture_btn = wx.Button(panel, label="拍照并识别")
#         self.Bind(wx.EVT_BUTTON, self.on_capture, self.capture_btn)
#
#         # 设置布局
#         sizer = wx.BoxSizer(wx.VERTICAL)
#         sizer.Add(self.camera_display, 1, wx.EXPAND | wx.ALL, 5)
#         sizer.Add(self.capture_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
#         panel.SetSizer(sizer)
#
#         # 初始化摄像头
#         self.cap = cv2.VideoCapture(0)
#
#         # 开启摄像头捕捉
#         self.capture_camera()
#
#     def capture_camera(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # 将OpenCV的BGR图像转换为RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # 将图像转换为wxPython可用的格式
#             h, w = frame.shape[:2]
#             bitmap = wx.Bitmap.FromBuffer(w, h, frame)
#             # 在画布上显示摄像头捕捉的图像
#             self.camera_display.SetBitmap(bitmap)
#         # 每20毫秒更新一次画面
#         wx.CallLater(int(1000/12), self.capture_camera)
#
#     def on_capture(self, event):
#         ret, frame = self.cap.read()
#         if ret:
#             # 将OpenCV的BGR图像转换为RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # 将图像转换为PIL格式，方便后续使用Tesseract识别文本
#             pil_image = Image.fromarray(frame)
#             # 使用Tesseract进行OCR识别身份证信息
#             extracted_text = pytesseract.image_to_string(pil_image, lang='chi_sim')
#             # 显示识别结果（这里假设直接打印出来，实际应用中可以根据需要处理）
#             print("识别结果：", extracted_text)
#
#
# app = wx.App(False)
# frame = MyFrame(None, "身份证信息识别")
# frame.Show(True)
# app.MainLoop()
#
# # 释放摄像头
# frame.cap.release()
# cv2.destroyAllWindows()

#==============================================
# import wx
# import cv2
# import pytesseract
# import pandas as pd
# from PIL import Image
# from io import BytesIO
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 替换为你的实际安装路径
# print(pytesseract.get_tesseract_version())
#
#
# class MyFrame(wx.Frame):
#     def __init__(self, parent, title):
#         wx.Frame.__init__(self, parent, title=title, size=(800, 600))
#
#         panel = wx.Panel(self)
#
#         # 创建按钮
#         self.capture_btn = wx.Button(panel, label="拍照并识别")
#         self.Bind(wx.EVT_BUTTON, self.on_capture, self.capture_btn)
#
#         # 初始化摄像头
#         self.cap = cv2.VideoCapture(0)
#         self.camera_display = wx.StaticBitmap(panel)
#
#         # 设置布局
#         sizer = wx.BoxSizer(wx.VERTICAL)
#         sizer.Add(self.camera_display, 1, wx.EXPAND | wx.ALL, 5)
#         sizer.Add(self.capture_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 20)
#         panel.SetSizer(sizer)
#
#         # 显示摄像头画面
#         self.capture_camera()
#
#     def capture_camera(self):
#         ret, frame = self.cap.read()
#         if ret:
#             # 将OpenCV的BGR图像转换为RGB
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # 转换为wxPython的Bitmap格式
#             h, w = frame.shape[:2]
#             bitmap = wx.Bitmap.FromBuffer(w, h, frame)
#             # 在界面上显示摄像头捕捉的图像
#             self.camera_display.SetBitmap(bitmap)
#         # 每60毫秒更新一次画面
#         wx.CallLater(60, self.capture_camera)
#
#     def on_capture(self, event):
#         ret, frame = self.cap.read()
#         if ret:
#             # 将OpenCV的BGR图像转换为RGB
#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             # 使用Tesseract进行OCR识别身份证信息
#             extracted_text = self.ocr_id_card(frame_rgb)
#             # 将识别结果保存到Excel文件
#             self.save_to_excel(extracted_text)
#
#     def ocr_id_card(self, frame):
#         # 在这里实现使用Tesseract进行身份证OCR识别的代码
#         # 假设这里直接返回了识别的文本信息字典
#         pil_image = Image.fromarray(frame)
#         text = pytesseract.image_to_string(pil_image, lang='chi_sim')
#         # 这里可以根据具体的身份证信息格式进行解析，提取姓名、地址、身份证号等信息
#         # 返回一个包含识别信息的字典
#         # return {
#         #     '姓名': '张三',
#         #     '地址': '北京市朝阳区',
#         #     '身份证号': '123456789012345678',
#         #     '人脸信息': '人脸特征数据'
#         # }
#         print("text的内容是：", text)
#         return {
#             '姓名': text.split("\n\n")[0],
#             '地址': text.split("\n\n")[1],
#             '身份证号': text.split("\n\n")[2],
#             '人脸信息': '人脸特征数据'
#         }
#
#     def save_to_excel(self, extracted_text):
#         # 在这里实现将识别到的信息保存到Excel文件的代码
#         df = pd.DataFrame([extracted_text])
#         df.to_excel('id_card_info.xlsx', index=False)
#         print("身份证信息已保存到 id_card_info.xlsx 文件")
#
#
# app = wx.App(False)
# frame = MyFrame(None, "身份证信息识别")
# frame.Show(True)
# app.MainLoop()
#
# # 释放摄像头资源
# frame.cap.release()
# cv2.destroyAllWindows()



#============================================================打印图片处理后的图片============================================
# import cv2
#
# def preprocess_image(image):
#     # 图像裁剪
#     # cropped_image = image[100:500, 200:600]
#     cropped_image = image
#
#     # 灰度化
#     gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#
#     # 二值化
#     _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#     # 去噪
#     denoised_image = cv2.fastNlMeansDenoising(binary_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
#
#     return denoised_image
#
# # 读取图像
# image_path = r"C:\Users\jisuanji\Desktop\IDcard.jpg"
# image = cv2.imread(image_path)
#
# # 图像预处理
# preprocessed_image = preprocess_image(image)
#
# # 显示预处理后的图像
# cv2.imshow("Preprocessed Image", preprocessed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#===========================================实际的提取身份证id卡号过程======================================
import pytesseract
import cv2
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 替换为你的实际安装路径
# print(pytesseract.get_tesseract_version())
#
# def extract_id_number(image_path):
#     # 读取图像
#     image = cv2.imread(image_path)
#
#     # 转换为灰度图像
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # 二值化
#     _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#
#     cv2.imshow("binary Image", binary)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     # 识别文本
#     text = pytesseract.image_to_string(binary, lang='chi_sim')
#
#     # 去除空格和非数字字符
#     id_number = ''.join(filter(str.isdigit, text))
#
#     return id_number
#
# # 身份证图像路径
# image_path = r"C:\Users\jisuanji\Desktop\IDcard.jpg"
#
# # 提取身份证号码
# id_number = extract_id_number(image_path)
#
# # 打印结果
# print("身份证号码：", id_number)

import numpy as np
import wx
import cv2
import re
import os
from main import OCR
from face_demo import FaceRecognition
class GetIdCard(wx.Frame):
    def __init__(self, parent, title, video_capture):
        super(GetIdCard, self).__init__(parent, title=title, size=(800, 640))
        self.idcardCapture = video_capture
        self.idcard_panel = wx.Panel(self, size=(640, 360))
        self.idcard_btn = wx.Button(self, label="拍照并识别")  # 创建按钮
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.idcard_panel, 3, wx.EXPAND)
        vbox.Add(self.idcard_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 20)
        self.Bind(wx.EVT_BUTTON, self.idcard_on_capture, self.idcard_btn)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(1000 // 24)  # 设置定时器间隔，这里假设每秒显示 24 帧
        self.SetSizer(vbox)

    def on_timer(self, event):
        ret, frame = self.idcardCapture.read()
        frame = cv2.flip(frame, 1)
        if ret:
            h, w = frame.shape[:2]
            image = wx.ImageFromBuffer(w, h,
                                       cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))
            panel_width, panel_heigh = self.idcard_panel.GetSize()
            image = image.Scale(panel_width, panel_heigh, wx.IMAGE_QUALITY_HIGH)
            bitmap = wx.Bitmap(image)
            dc = wx.ClientDC(self.idcard_panel)
            dc.DrawBitmap(bitmap, 0, 0)

    def idcard_on_capture(self, event):

        file_path = None
        xname = None
        idnum = None
        # i = 0  #用来计数当前班级第i个同学进行身份证拍照检索信息
        ret, frame = self.idcardCapture.read()
        if ret:
            srcimg = cv2.flip(frame, 0)
            myocr = OCR()
            results = myocr.det_rec(srcimg)
            match = re.search(r'姓名(.+)', results[0]['text'])  # 判断是否有文本
            if match is None:
                srcimg = cv2.flip(srcimg, 0)
                results = myocr.det_rec(srcimg)

            for i, res in enumerate(results):
                point = res['location'].astype(int)
                cv2.polylines(srcimg, [point], True, (0, 0, 255), thickness=2)
                if re.search(r'公民身份号码(\d+)', res['text']):
                    # 在这里处理捕获到的图像frame，例如保存到文件或进行进一步的处理
                    idnum = re.search(r'公民身份号码(\d+)', res['text']).group(1)
                    print("idnum身份证号码:", idnum)
                    output_folder = r'C:\Users\jisuanji\Desktop\idcard\real'
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    file_name = f'idcard_{idnum}.jpg'
                    file_path = os.path.join(output_folder, file_name)
                    self.id_file_path = file_path
                    cv2.imwrite(file_path, frame)
                if re.search(r'名(.+)', res['text']):
                    # 在这里处理捕获到的图像frame，例如保存到文件或进行进一步的处理
                    xname = re.search(r'名(.+)', res['text']).group(1)
                    print("xname姓名:", xname)
                # print(res['text'])
            # wx.MessageBox("身份证拍照成功！", "提示", wx.OK | wx.ICON_INFORMATION)
            if idnum and xname:
                frame = wx.Frame(None, -1, "Message Dialog Example")
                # 创建一个消息对话框
                dlg = wx.MessageDialog(frame, "该生采集身份证信息成功！", "提示", wx.OK | wx.ICON_INFORMATION)
                # 显示对话框
                result = dlg.ShowModal()
                # 根据用户的操作做相应处理
                if result == wx.ID_OK:
                    # 用户点击了确定按钮
                    dlg.Destroy()  # 销毁对话框
                    self.Close()
            else:
                frame = wx.Frame(None, -1, "Message Dialog Example")
                # 创建一个消息对话框
                dlg = wx.MessageDialog(frame, "该生采集身份证信息出错！！", "提示", wx.OK | wx.ICON_INFORMATION)
                # 显示对话框
                result = dlg.ShowModal()
                # 根据用户的操作做相应处理
                if result == wx.ID_OK:
                    # 用户点击了确定按钮
                    dlg.Destroy()  # 销毁对话框


class CompanyIdAndPhoto():
    def __init__(self,path1,path2):
        self.path1 = path1   # id_path
        self.path2 = path2   # phpto_path
    def company(self):
        id_img = cv2.imdecode(np.fromfile(self.path1, dtype=np.uint8),
                              -1)  # 身份证照片
        # id_img = cv2.rotate(id_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.imdecode(np.fromfile(self.path2, dtype=np.uint8), -1)  # 现场照片
        face_recognitio = FaceRecognition()
        print("检测ing")
        faces_img = list()
        results = face_recognitio.detect(img)
        for result in results:
            print('人脸框坐标：{}'.format(result["bbox"]))
            print('人脸五个关键点：{}'.format(result["kps"]))
        results2 = face_recognitio.detect(id_img)
        # print(results)
        for result in results2:
            print('人脸框坐标：{}'.format(result["bbox"]))
            print('人脸五个关键点：{}'.format(result["kps"]))

        ide, _ = face_recognitio.feature_compare(results[0]["embedding"], results2[0]["embedding"],
                                                 0.7)  # default = 0.7
        if ide:
            print('同一人')
            return '匹配正确'
        else:
            print('匹配错误')
            return '匹配错误'

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)  # 打开默认摄像头，如果有多个摄像头，可以尝试不同的索引
    app = wx.App()
    frame = GetIdCard(None, title='ID Card Capture', video_capture=capture)
    frame.Show()
    app.MainLoop()


