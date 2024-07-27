# -*- coding: utf-8 -*-
import wx
import wx.grid as wg
import cv2
import pandas as pd
import onnxruntime
import numpy as np
import os
from hivisionai.hycv.vision import add_background
from src.face_judgement_align import face_number_and_angle_detection,align_face,idphoto_cutting
from beautyPlugin.MakeBeautiful import makeBeautiful
from src.imageTransform import standard_photo_resize, hollowOutFix, get_modnet_matting, draw_picture_dots, detect_distance
from PIL import Image
from screeninfo import get_monitors
import time


from face_demo import FaceRecognition
import pyclipper               #  需要pip
from shapely.geometry import Polygon   #需要pip
from keys import alphabetChinese as alphabet
import re
import warnings
warnings.filterwarnings('ignore', message="FutureWarning")


from main import OCR
APP_TITLE = u'拍照'
standard_size = (622, 413)  # （413,295） 为一寸的   （622,413）为两寸的
class mainFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, APP_TITLE)
        self.SetBackgroundColour(wx.Colour(240, 240, 240))
        self.SetSize((1500, 920))
        self.Center()
        self.timer = wx.Timer(self)
        # 镜头
        # self.number = wx.StaticText(self, -1, u'当前拍照序号:', size=(300, 30))
        self.preview = wx.Panel(self, -1, style=wx.SUNKEN_BORDER)
        self.preview.SetBackgroundColour(wx.Colour(0, 0, 0))
        self.preview.SetMinSize((690, 920))
        # 按钮
        self.button = wx.Button(self, -1, u'开启摄像头', size=(150, 30))
        self.pic_but = wx.Button(self, -1, u'拍照', size=(150, 30))
        self.CSV_but = wx.Button(self, -1, u'读取表格', size=(150, 30))
        self.IDcard_but = wx.Button(self, -1, u'身份证拍照', size=(150, 30))
        # 创建Grid 表格
        self.grid = wg.Grid(self)
        self.grid.CreateGrid(1, 1)  # 初始化时设置为0行0列，之后根据数据动态调整
        self.grid.SetMinSize((400, 300))
        font = wx.Font(35, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.grid.SetDefaultCellFont(font)
        # self.scrolled_window = wx.ScrolledWindow(self)
        # self.scrolled_window.SetMinSize((400, 100))
        # 弹窗
        self.selected_image_index = None
        self.dialog = None
        self.photo_count = 0
        self.path = None
        self.index = 0
        self.sub_index = 0
        #第二个窗口展示学生照片
        self.dialog_showtoStu = None
        #布局
        sizer_tool = wx.BoxSizer(wx.VERTICAL)
        self.on_mouse_selected_image = None

        sizer_but = wx.BoxSizer()
        sizer_but.Add(self.button, 0, wx.ALL, 5)
        sizer_but.Add(self.pic_but, 0, wx.ALL, 5)
        sizer_but.Add(self.CSV_but, 0, wx.ALL, 5)
        sizer_but.Add(self.IDcard_but, 0, wx.ALL, 5)

        sizer_tool.Add(sizer_but, 0, flag=wx.ALIGN_CENTER | wx.ALL, border=5)  # 按钮居中

        sizer_pin = wx.BoxSizer(wx.VERTICAL)  # 竖屏
        sizer_pin.Add(self.preview, 1, wx.EXPAND | wx.LEFT | wx.TOP | wx.BOTTOM, 5)
        # sizer_pinAndbut.Add(sizer_tool, 0, wx.EXPAND | wx.ALL, 0)

        sizer_max = wx.BoxSizer(wx.HORIZONTAL)  # 横屏
        sizer_max.Add(sizer_pin, 0, wx.EXPAND | wx.ALL, 0)

        show_grid = wx.BoxSizer(wx.VERTICAL)
        show_grid.Add(self.grid, 1, wx.EXPAND | wx.ALL, 10)
        # show_grid.Add(self.scrolled_window, 0, wx.EXPAND | wx.ALL, 10)

        show_name = wx.BoxSizer(wx.HORIZONTAL)
        self.name_label = wx.StaticText(self, -1, label="姓名:", size=(150, 70), style=wx.ALIGN_CENTER)
        font = wx.Font(40, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.name_label.SetFont(font)
        self.name_text = wx.TextCtrl(self, value="  ")
        self.name_text.SetMinSize((250, -1))
        font = wx.Font(45, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.name_text.SetFont(font)
        show_name.Add(self.name_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        show_name.Add(self.name_text, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)

        show_number = wx.BoxSizer(wx.HORIZONTAL)
        self.number_label = wx.StaticText(self, -1, label="身份证号码:", size=(-1, 70), style=wx.ALIGN_CENTER)
        font = wx.Font(40, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.number_label.SetFont(font)
        self.number_text = wx.TextCtrl(self, value="  ")
        self.number_text.SetMinSize((580, -1))
        font = wx.Font(45, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.number_text.SetFont(font)
        show_number.Add(self.number_label, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        show_number.Add(self.number_text, 0, wx.ALIGN_CENTER_VERTICAL| wx.ALL, 10)

        show_stu = wx.BoxSizer(wx.VERTICAL)
        show_stu.Add(show_name, 1, wx.EXPAND | wx.ALL, 0)
        show_stu.Add(show_number, 1, wx.EXPAND | wx.ALL, 0)

        show_gridAndstuTool = wx.BoxSizer(wx.VERTICAL)   # 竖屏
        show_gridAndstuTool.Add(show_grid, 2, wx.EXPAND | wx.ALL, 0)
        show_gridAndstuTool.Add(sizer_tool, 0, wx.EXPAND | wx.ALL, 0)
        show_gridAndstuTool.Add(show_stu, 1, wx.EXPAND | wx.ALL, 0)

        sizer_max.Add(show_gridAndstuTool, 1, wx.EXPAND | wx.ALL, 1)

        self.capture = None
        self.SetAutoLayout(True)
        self.SetSizer(sizer_max)
        self.Layout()

        self.Bind(wx.EVT_BUTTON, self.PhotoIDcard, self.IDcard_but)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.Bind(wx.EVT_BUTTON, self.on_button_click, self.button)
        self.Bind(wx.EVT_BUTTON, self.on_picture, self.pic_but)
        self.Bind(wx.EVT_BUTTON, self.load_csv, self.CSV_but)
        # self.grid.Bind(wx.grid.EVT_GRID_CELL_LEFT_CLICK, self.on_select_cell)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.preview.Bind(wx.EVT_LEFT_DOWN, self.on_picture)
        self.temp_static_bitmap = None
        self.temp_save_stu = 0

        self.idcardwindow = None
#==========================================================================拍照识别身份证信息===================
    def PhotoIDcard(self, event):
        cap = cv2.VideoCapture(0)  # 打开摄像头
        self.idcardwindow = GetIdCard(self, "身份证信息提取界面", video_capture=cap, callback=self.on_idcard_captured, tempIdPhoto_folder=self.save_folder)
        # self.idcardwindow.Show()
        # cap.release()
        # cv2.destroyAllWindows()
    def on_idcard_captured(self, file_path, xname, idnum):          # 回调函数
        self.file_path = file_path
        self.idxname = xname
        self.ididnum = idnum
        self.name_text.SetValue(self.idxname)
        self.number_text.SetValue(self.ididnum)
        # self.second_window.name_text.SetValue(self.idxname + '_' + self.ididnum)
        self.second_window.name_text.SetValue(self.idxname)
        self.df.loc[self.index, "姓名"] = self.idxname
        self.df.loc[self.index, "身份证号"] = self.ididnum
        if self.path is not None:          # 以下内容将保存df内容，同时展示在grid上面
            self.df.to_excel(self.path, index=False)
            # 将DataFrame转换为二维列表，并将内容添加到grid中
            rows = self.df.values.tolist()
            header = self.df.columns.tolist()
            if self.grid.GetNumberCols() < len(header):
                self.grid.AppendCols(len(header) - self.grid.GetNumberCols())  # 添加对应数量的列

            for col_idx, col_name in enumerate(header):
                self.grid.SetColLabelValue(col_idx, col_name)
                if col_name == '身份证号':
                    self.grid.SetColSize(col_idx, 500)  # 设置每个表格的大小
                else:
                    self.grid.SetColSize(col_idx, 300)  # 设置每个表格的大小
            if self.grid.GetNumberRows() < len(rows):
                self.grid.AppendRows(len(rows) - self.grid.GetNumberRows())  # 添加对应数量的行

            for row_idx, row_data in enumerate(rows):
                for col_idx, cell_value in enumerate(row_data):
                    self.grid.SetCellValue(row_idx, col_idx, str(cell_value))
                    self.grid.SetRowSize(row_idx, 100)  # default = 30
        #===========================重新给设立一下光标焦点======================
        self.SetFocus()  # 先获取焦点
        self.grid.SelectRow(self.index)  # 选中当前一行
        self.grid.SetGridCursor(self.index, 0)
        print(f"Captured info: {file_path}, {xname}, {idnum}")
#=======================================================================================
    def on_button_click(self, event):
        if self.capture is None:
            self.capture = cv2.VideoCapture(1)
            # width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            # height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print("Camera resolution: {} x {}".format(width, height))
            self.capture.set(3, 3414)
            self.capture.set(4, 1920)

            self.timer.Start(int(1000 / 24))  # 24 frames per second
            self.button.SetLabel("停止照相")
            self.second_window = AnotherFrame(self, "Second Window", self.capture)
        else:
            #_, self.image = self.capture.read()
            self.capture.release()
            self.capture = None
            self.timer.Stop()
            self.button.SetLabel("开始照相")

    def on_close(self, event):
        if self.capture is not None:
            self.capture.release()
        if self.path is not None:
            self.df.to_excel(self.path, index=False)
        self.Destroy()


    def on_timer(self, event):

        ret, frame = self.capture.read()
        if ret:
            height, width = frame.shape[:2]
            # print("height的长度:", height)
            # print("width的长度:", width)
            image = wx.ImageFromBuffer(width, height, cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))  #cv2.flip(frame, 1)设置镜像
            # print("width:", width)
            # print("height:", height)
            panel_width, panel_height = self.preview.GetSize()
            image = image.Scale(panel_width, panel_height, wx.IMAGE_QUALITY_HIGH)   # 两行用来控制视频显示器的铺满整个布局
            # image = image.Scale(600, 500, wx.IMAGE_QUALITY_HIGH)  #
            bmp = wx.Bitmap(image)
            dc = wx.ClientDC(self.preview)
            dc.DrawBitmap(bmp, 0, 0)


    def load_csv(self, event):
        # 先清空原来表格的数据和设置
        self.grid.ClearGrid()
        self.index = 0
        self.image_1 = None
        self.image_2 = None
        self.name_text.SetValue(' ')
        self.number_text.SetValue(' ')
        self.second_window.name_text.SetValue('请读取表格！ ')
        openFileDialog = wx.FileDialog(self, "Open Excel File", wildcard="Excel files (*.xlsx)|*.xlsx",
                                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        if openFileDialog.ShowModal() == wx.ID_OK:
            self.path = openFileDialog.GetPath()
            self.df = pd.read_excel(self.path)  # 使用pandas加载Excel
            # 确保整列都是字符串类型
            # self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)   # iloc是针对列，loc是针对字符串标签 会出现警告
            self.df['身份证号'] = self.df['身份证号'].astype(str)
            self.df['比对结果'] = self.df['比对结果'].astype(str)
            # print("表格的路径地址为：", self.path)
            self.save_folder = os.path.split(self.path)[0]
            # print("表格的上一个路径地址为：", self.save_folder)

            self.OriSaveFolder = os.path.join(self.save_folder, "Original")
            self.FinalFolder = os.path.join(self.save_folder, "480_640")
            if not os.path.exists(self.OriSaveFolder):
                # 如果文件夹不存在，则创建文件夹
                os.makedirs(self.OriSaveFolder)
            if not os.path.exists(self.FinalFolder):
                # 如果文件夹不存在，则创建文件夹
                os.makedirs(self.FinalFolder)


            # 将DataFrame转换为二维列表，并将内容添加到grid中
            rows = self.df.values.tolist()
            header = self.df.columns.tolist()
            if self.grid.GetNumberCols() < len(header):
                self.grid.AppendCols(len(header) - self.grid.GetNumberCols())  # 添加对应数量的列

            for col_idx, col_name in enumerate(header):
                self.grid.SetColLabelValue(col_idx, col_name)
                if col_name=='身份证号':
                    self.grid.SetColSize(col_idx, 500)   # 设置每个表格的大小
                else:
                    self.grid.SetColSize(col_idx, 300)  # 设置每个表格的大小

            if self.grid.GetNumberRows() < len(rows):
                self.grid.AppendRows(len(rows) - self.grid.GetNumberRows())  # 添加对应数量的行

            for row_idx, row_data in enumerate(rows):
                for col_idx, cell_value in enumerate(row_data):
                    self.grid.SetCellValue(row_idx, col_idx, str(cell_value))
                    self.grid.SetRowSize(row_idx, 100)   # default = 30

                    # 获取行数

            num_rows = self.df.shape[0]
            if num_rows == 0:   #如果读取的表格没有行（第一次读取)
                self.SetFocus()  # 先获取焦点
                self.grid.SelectRow(self.index)  # 选中一行   如果表格里面没有数据，则光标默认在第一行，并且self.index=0
                self.grid.SetGridCursor(self.index, 0)
            else:
                self.SetFocus()  # 先获取焦点
                self.grid.SelectRow(num_rows - 1)  # 选中下一行 如果表格里面有数据，则光标默认到最后一名同学的位置上
                # print("self.index:", self.index)
                self.grid.SetGridCursor(num_rows - 1, 0)  # 设置光标位置到下一行第一个单元格
                self.index = num_rows            # 因为可能继续往这个班级添加信息，因此需要将self.index设置为最后一个同学的索引
                # print("self.index:", self.index)
            print(f"Loaded {num_rows} rows from Excel file.")
            grid_size = self.grid.GetBestSize()



    def on_select_cell(self, event):
        selected_row = event.GetRow()
        # print("选中的行为：", selected_row)
        name = self.grid.GetCellValue(selected_row, 1)  # 获取选中行的姓名数据
        number = self.grid.GetCellValue(selected_row, 2)  # 获取选中行的数字数据
        self.name_text.SetValue(name)  # 在文本框中显示姓名数据
        self.number_text.SetValue(number)  # 在文本框中显示数字数据
        self.second_window.name_text.SetValue(name)
        self.index = selected_row
        # print("选中的行为index：", self.index)
        self.SetFocus()  # 先获取焦点
        self.grid.SelectRow(selected_row)  # 选中下一行
        self.grid.SetGridCursor(selected_row, 0)  # 设置光标位置到下一行第一个单元格


    def on_picture(self, event):
        #self.capture.release()
        self.image_1 = None
        self.image_2 = None
        if self.dialog is None:
            self.dialog = wx.Dialog(self, title="图片展示", size=(1500, 1000),  style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
            # display_count = wx.Display.GetCount()
            # if display_count > 1:
            #     second_display = wx.Display(1)  # 假设第二个显示器是索引为1的显示器
            #     second_frame_position = second_display.GetClientArea().GetPosition()
            #     self.dialog.SetPosition(second_frame_position)
            # print("self.photo_count:", self.photo_count)

        sizer = wx.BoxSizer(wx.HORIZONTAL)  # 创建水平方向的 BoxSizer 对象 ==
        while (self.photo_count < 2):
            if self.photo_count < 1:
                _, self.image_1 = self.capture.read()
                print("image_1的shape：", self.image_1.shape)
                num = self.df.iloc[self.index, 1]
                Stuname = f"{num}_1.jpg"
                savepicFolder = os.path.join(self.OriSaveFolder, Stuname)
                try:
                    cv2.imwrite(savepicFolder, self.image_1, [cv2.IMWRITE_JPEG_QUALITY, 100])
                except Exception as e:
                    print(f"保存图片时发生异常：{e}")
                # print("dsadsadad489612323")
                self.photo_count += 1
                time.sleep(0.5)
            else:
                _, self.image_2 = self.capture.read()
                print("image_2的shape：", self.image_2.shape)
                num = self.df.iloc[self.index, 1]
                Stuname = f"{num}_2.jpg"
                savepicFolder = os.path.join(self.OriSaveFolder, Stuname)
                try:
                    cv2.imwrite(savepicFolder, self.image_2, [cv2.IMWRITE_JPEG_QUALITY, 100])
                except Exception as e:
                    print(f"保存图片时发生异常：{e}")
                # print("dsadsadaddsadaf1951261231")
                self.photo_count += 1


        self.faces, self.dets, self.landmark = face_number_and_angle_detection(self.image_1)  #
        self.faces_2, self.dets_2, self.landmark_2 = face_number_and_angle_detection(self.image_2)
        print("faces:", self.faces)
        print("faces222:", self.faces_2)
        if not isinstance(self.faces, int) and not isinstance(self.faces_2, int):
            self.image_1 = get_modnet_matting(self.image_1,sess=onnxruntime.InferenceSession("src/hivision_modnet.onnx"))
            self.image_1 = hollowOutFix(self.image_1)  # 抠图洞洞修补
            print("输入前的self.iamge1：", self.image_1.shape)
            self.image_cut1,_,_ = \
                idphoto_cutting(faces=self.faces, head_measure_ratio=0.16, standard_size=standard_size, head_height_ratio=0.40, origin_png_image=self.image_1,
                                origin_png_image_pre=self.image_1, rotation_params={"L1": None, "L2": None, "L3": None, "clockwise": None, "drawed_image": None},
                                top_distance_max=0.08, top_distance_min=0.01)
            print("image_cut1:", self.image_cut1.shape)
            self.image_2 = get_modnet_matting(self.image_2, sess=onnxruntime.InferenceSession("src/hivision_modnet.onnx"))
            self.image_2 = hollowOutFix(self.image_2)  # 抠图洞洞修补
            self.image_cut2, _, _ = \
                idphoto_cutting(faces=self.faces, head_measure_ratio=0.16, standard_size=standard_size, head_height_ratio=0.40,
                                origin_png_image=self.image_2,
                                origin_png_image_pre=self.image_2,
                                rotation_params={"L1": None, "L2": None, "L3": None, "clockwise": None,
                                                 "drawed_image": None},
                                top_distance_max=0.08, top_distance_min=0.02)
            print("image_cut2:", self.image_cut2.shape)
            height, width, channels = self.image_cut1.shape
            image = wx.ImageFromBuffer(width, height, cv2.cvtColor(self.image_cut1, cv2.COLOR_BGR2RGB))

            height, width, channels = self.image_cut2.shape
            image_cam = wx.ImageFromBuffer(width, height, cv2.cvtColor(self.image_cut2, cv2.COLOR_BGR2RGB))

            bitmap = image.ConvertToBitmap()
            bitmap_cam = image_cam.ConvertToBitmap()

            self.static_bitmap = wx.StaticBitmap(self.dialog, -1, bitmap, (30, 30), (bitmap.GetWidth(), bitmap.GetHeight()))
            self.static_bitmap_cam = wx.StaticBitmap(self.dialog, -1, bitmap_cam, (30, 30), (bitmap_cam.GetWidth(), bitmap_cam.GetHeight()))

            sizer.Add(self.static_bitmap, 0, wx.ALL, 5)
            sizer.Add(self.static_bitmap_cam, 0, wx.ALL, 5)


            # self.choose_but = wx.Button(self.dialog, label="选中照片")
            self.aliButton = wx.Button(self.dialog, label='旋转矫正')
            self.beautyButton = wx.Button(self.dialog, label='美颜')
            self.cuttingButton = wx.Button(self.dialog, label='抠图')
            self.okButton = wx.Button(self.dialog, label="确定并保存当前照片")
            # self.rephotoButton = CircularButton(self.dialog, "重拍", size=(50, 50))
            self.compIdcardAndPhotoBut = wx.Button(self.dialog, label="照片智能对比")
            self.manual = wx.Button(self.dialog,label="人工对比通过")
            self.vbox = wx.BoxSizer(wx.VERTICAL)
            self.vbox.Add(sizer, 1, wx.ALIGN_CENTER | wx.ALL, 5)

            vbox_sizer = wx.BoxSizer(wx.HORIZONTAL)
            # vbox_sizer.Add(self.choose_but, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            vbox_sizer.Add(self.aliButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            vbox_sizer.Add(self.beautyButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            vbox_sizer.Add(self.cuttingButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            vbox_sizer.Add(self.okButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            # vbox_sizer.Add(self.rephotoButton, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            vbox_sizer.Add(self.compIdcardAndPhotoBut, 0, wx.ALIGN_CENTER | wx.ALL, 5)
            vbox_sizer.Add(self.manual, 0, wx.ALIGN_CENTER | wx.ALL, 5)

            self.vbox.Add(vbox_sizer, 1, wx.ALIGN_CENTER | wx.ALL, 10)

            # self.choose_but.Bind(wx.EVT_BUTTON, self.on_choose)
            self.aliButton.Bind(wx.EVT_BUTTON, self.on_align)
            self.cuttingButton.Bind(wx.EVT_BUTTON, self.img_cutting)
            self.beautyButton.Bind(wx.EVT_BUTTON, self.on_beauty)
            self.okButton.Bind(wx.EVT_BUTTON, self.on_ok_button_click)
            self.dialog.Bind(wx.EVT_CLOSE, self.on_dclose)
            # self.rephotoButton.Bind(wx.EVT_BUTTON, self.rephoto_click)
            self.compIdcardAndPhotoBut.Bind(wx.EVT_BUTTON, self.compIdcardAndPhoto_click)
            self.manual.Bind(wx.EVT_BUTTON, self.manual_click)

            self.static_bitmap.Bind(wx.EVT_LEFT_DOWN, lambda event: self.on_image_click(event, 0, self.static_bitmap, self.static_bitmap_cam))           # 绑定点击事件
            self.static_bitmap.Bind(wx.EVT_MOTION, lambda event: self.on_mouse_move(static_bitmap=self.static_bitmap))
            self.static_bitmap_cam.Bind(wx.EVT_LEFT_DOWN, lambda event: self.on_image_click_cam(event, 1, self.static_bitmap, self.static_bitmap_cam))  # 绑定点击事件
            self.static_bitmap_cam.Bind(wx.EVT_MOTION, lambda event: self.on_mouse_move(static_bitmap=self.static_bitmap_cam))
            self.dialog.SetSizer(self.vbox)
            self.dialog.ShowModal()
        else:
            self.photo_count = 0
            frame = wx.Frame(None, -1, "Message Dialog Example")
            # 创建一个消息对话框
            dlg = wx.MessageDialog(frame, "识别到不止一个人，请重新拍照！", "提示", wx.OK | wx.ICON_INFORMATION)
            # 显示对话框
            result = dlg.ShowModal()
            # 根据用户的操作做相应处理
            if result == wx.ID_OK:
                # 用户点击了确定按钮
                dlg.Destroy()  # 销毁对话框

    def on_mouse_move(self, static_bitmap):
        global second_frame_position
        display_count = wx.Display.GetCount()
        if display_count > 1:
            second_display = wx.Display(1)  # 假设第二个显示器是索引为1的显示器
            second_frame_position = second_display.GetClientArea().GetPosition()

        if self.dialog_showtoStu is None:
            self.dialog_showtoStu = wx.Dialog(self, title="图片展示", size=(900, 900))
        sizer_showtoStu = wx.BoxSizer(wx.VERTICAL)

        bitmap = static_bitmap.GetBitmap()  # 将位图对象转换为图像对象
        image = bitmap.ConvertToImage()  # 如上 image 是第一个窗口下的两张图片中任意一张

        if self.temp_static_bitmap is None:
            self.temp_static_bitmap = wx.StaticBitmap(self.dialog_showtoStu, wx.ID_ANY, wx.Bitmap(image))
            sizer_showtoStu.Add(self.temp_static_bitmap, 0, wx.ALL, 5)
            self.dialog_showtoStu.SetSizer(sizer_showtoStu)
            self.dialog_showtoStu.SetPosition(second_frame_position)
            self.dialog_showtoStu.Show()
        else:
            on_mouse_image_bitmap = self.temp_static_bitmap.GetBitmap()
            on_mouse_image = on_mouse_image_bitmap.ConvertToImage()
            if not self.images_equals(image, on_mouse_image):
                self.temp_static_bitmap.Destroy()  # 该步骤是需要的
                self.temp_static_bitmap = wx.StaticBitmap(self.dialog_showtoStu, wx.ID_ANY, wx.Bitmap(image))
                sizer_showtoStu.Add(self.temp_static_bitmap, 0, wx.ALL, 5)
                self.dialog_showtoStu.SetSizer(sizer_showtoStu)
                self.dialog_showtoStu.SetPosition(second_frame_position)
                self.dialog_showtoStu.Show()

            else: return

    def images_equals(self, image1,image2):   #判断两个控件上面的image是否相同

        if image1.IsOk() and image2.IsOk():
            if image1.GetSize() != image2.GetSize():
                return False
            data1 = image1.GetData()
            data2 = image2.GetData()
            if data1 != data2:
                return False
            return True
        else:
            return False

    def on_dclose(self, event):

        # self.static_bitmap.Destroy()
        # self.static_bitmap_cam.Destroy()
        if self.temp_static_bitmap:
            self.temp_static_bitmap.Destroy()  # 销毁之前的对象
            self.temp_static_bitmap = None  # 置空对象
        if self.dialog_showtoStu:
            self.dialog_showtoStu.Destroy()
            self.dialog_showtoStu = None
        self.dialog.Destroy()
        self.dialog = None  # 重置对话框对象为None
        self.photo_count = 0
    def img_cutting(self, event):
        height, width = self.fimg.shape[:2]
        result_image_hd = np.uint8(add_background(self.fimg, bgr=(212, 140, 86)))

        if self.selected_image_index == 0:
            self.image_1 = np.uint8(add_background(self.image_1, bgr=(212, 140, 86)))
            self.image_1 = get_modnet_matting(self.image_1,
                                              sess=onnxruntime.InferenceSession("src/hivision_modnet.onnx"))
            self.image_1 = hollowOutFix(self.image_1)  # 抠图洞洞修补

        else:
            self.image_2 = np.uint8(add_background(self.image_2, bgr=(212, 140, 86)))
            self.image_2 = get_modnet_matting(self.image_2,
                                              sess=onnxruntime.InferenceSession("src/hivision_modnet.onnx"))
            self.image_2 = hollowOutFix(self.image_2)  # 抠图洞洞修补

        # cv2.imwrite("result_image_hd.png", result_image_hd)
        self.fimg = result_image_hd
        image = wx.ImageFromBuffer(width, height, cv2.cvtColor(result_image_hd, cv2.COLOR_BGR2RGB))
        if self.selected_image_index == 0:
            self.static_bitmap.SetBitmap(wx.Bitmap(image))
        else:
            self.static_bitmap_cam.SetBitmap(wx.Bitmap(image))

    def on_beauty(self, event):
        if self.selected_image_index == 0:
            self.image_1 = makeBeautiful(self.image_1, self.landmark, 2, 2, 5, 4)
            self.image_cut1 = makeBeautiful(self.fimg, self.landmark, 2, 2, 5, 4)
            img = self.image_cut1
        else:
            self.image_2 = makeBeautiful(self.image_2, self.landmark_2, 2, 2, 5, 4)
            self.image_cut2 = makeBeautiful(self.fimg, self.landmark_2, 2, 2, 5, 4)
            img = self.image_cut2
        height, width = img.shape[:2]
        self.fimg = img
        image = wx.ImageFromBuffer(width, height, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 假设 image 是通过 wx.ImageFromBuffer 创建的 wxPython 图像对象
        width = image.GetWidth()
        height = image.GetHeight()
        # 获取图像数据
        buffer = image.GetData()
        # 将图像数据转换为 NumPy 数组
        image_array = np.frombuffer(buffer, dtype=np.uint8)
        image_array = image_array.reshape((height, width, 3))  # 假设是 RGB 图像
        # print(image_array.shape)
        image_array = wx.ImageFromBuffer(width, height, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.selected_image_index == 0:
            self.static_bitmap.SetBitmap(wx.Bitmap(image_array))
        else: self.static_bitmap_cam.SetBitmap(wx.Bitmap(image_array))

    def on_align(self, event):
        if self.selected_image_index == 0:
            rotated_img, eye_center, angle = align_face(self.image_1, self.landmark)
        else:
            rotated_img, eye_center, angle = align_face(self.image_2, self.landmark_2)
        rotated_img, _, _ = \
            idphoto_cutting(faces=self.faces, head_measure_ratio=0.2, standard_size=standard_size, head_height_ratio=0.45,
                            origin_png_image=rotated_img,
                            origin_png_image_pre=rotated_img,
                            rotation_params={"L1": None, "L2": None, "L3": None, "clockwise": None,
                                             "drawed_image": None},
                            top_distance_max=0.08, top_distance_min=0.02)
        height, width = rotated_img.shape[:2]
        image = wx.ImageFromBuffer(width, height, cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
        self.fimg = rotated_img

        if self.selected_image_index == 0:
            self.static_bitmap.SetBitmap(wx.Bitmap(image))
        else:
            self.static_bitmap_cam.SetBitmap(wx.Bitmap(image))

    def on_ok_button_click(self, event):
        if self.dialog is not None:
            if self.selected_image_index == 0:  #第一张图片
                bitmap = self.static_bitmap.GetBitmap()
                # 将位图对象转换为图像对象
                image = bitmap.ConvertToImage()
                # 获取图像数据
                image_data = image.GetData()
                # 创建 PIL 图像对象
                pil_image = Image.frombuffer('RGB', (image.GetWidth(), image.GetHeight()), image_data)
                # 设置DPI
                dpi = (300, 300)  # 设置为300 DPI
                # 修改图像分辨率
                resized_image = pil_image.resize((480, 640))
                num = self.df.iloc[self.index, 1]
                Stuname = f"{num}.jpg"
                filename = os.path.join(self.FinalFolder, Stuname)
                # 保存修改后的图像
                pil_image.save(filename, dpi=dpi)  # 替换为你想要保存图像的路径和文件名
                # bitmap.SaveFile(filename, wx.BITMAP_TYPE_JPEG)
                self.static_bitmap.Destroy()
            else:# 第二张图片
                bitmap = self.static_bitmap_cam.GetBitmap()
                # 将位图对象转换为图像对象
                image = bitmap.ConvertToImage()
                # 获取图像数据
                image_data = image.GetData()
                # 创建 PIL 图像对象
                pil_image = Image.frombuffer('RGB', (image.GetWidth(), image.GetHeight()), image_data)
                # 设置DPI
                dpi = (300, 300)  # 设置为300 DPI
                # 修改图像分辨率
                resized_image = pil_image.resize((413, 622))
                num = self.df.iloc[self.index, 1]
                Stuname = f"{num}.jpg"
                filename = os.path.join(self.FinalFolder, Stuname)
                pil_image.save(filename, dpi=dpi)
                # bitmap.SaveFile(filename, wx.BITMAP_TYPE_JPEG)
                self.static_bitmap_cam.Destroy()

            # self.df.iloc[self.index, 4] = '已拍照'

            # # 将DataFrame转换为二维列表，并将内容添加到grid中
            # rows = self.df.values.tolist()
            #
            # for row_idx, row_data in enumerate(rows):
            #     for col_idx, cell_value in enumerate(row_data):
            #         self.grid.SetCellValue(row_idx, col_idx, str(cell_value))

            # current_row = self.grid.GetGridCursorRow()
            # print("indexxxxx", self.sub_index)
            # if current_row == self.grid.NumberRows - 1:  # 如果当前行已经是最后一行
            #     if(self.sub_index >0):
            #         for i in range(self.sub_index):
            #             self.grid.MovePageDown()
            #
            #     elif(self.sub_index <0):
            #         for i in range(abs(self.sub_index)):
            #             self.grid.MovePageUp()
            #     else:
            #         self.grid.MovePageDown()
            #     # self.grid.MovePageDown()
            #     self.SetFocus()  # 先获取焦点
            #     self.grid.SelectRow(self.temp_save_stu)  # 选中下一行
            #     print("self.temp_save_stu的值:", self.temp_save_stu)
            #     self.grid.SetGridCursor(self.temp_save_stu, 0)  # 设置光标位置到下一行第一个单元格
            # else:  # 移动焦点到下一行的第一个单元格
            #     next_row = current_row + 1
            #     if(self.sub_index >0):
            #         for i in range(self.sub_index):
            #             self.grid.MovePageDown()
            #
            #     elif(self.sub_index <0):
            #         for i in range(abs(self.sub_index)):
            #             self.grid.MovePageUp()
            #     else:
            #         self.grid.MovePageDown()
            #     # self.grid.MovePageDown()
            #     self.SetFocus()  # 先获取焦点
            #     self.grid.SelectRow(next_row)  # 选中下一行
            #     self.grid.SetGridCursor(next_row, 0)  # 设置光标位置到下一行第一个单元格

            self.grid.ForceRefresh()  # 刷新表格显示
            self.dialog.Destroy()  # 销毁已关闭的对话框 (主要是展示照片的框架)
            self.dialog = None  # 重置对话框对象为None
            self.dialog_showtoStu.Destroy()  # 销毁已关闭的对话框  （第二个窗口的显示框）
            self.dialog_showtoStu = None  # 重置对话框对象为None
            self.temp_static_bitmap.Destroy()  # 销毁之前的对象    （第二个窗口上，当鼠标移植到特定位置的时候，展示给学生老师选中的图）
            self.temp_static_bitmap = None  # 置空对象
        else:
            print("Dialog 未正常初始化")


        self.index = self.index + 1   # 需要更改遍历全部是否拍摄完整
        self.photo_count = 0     # 用来计数当前存了是否有两张照片
        # old_index = self.index
        # print("index 大小", self.index)
        # print("deshape 大小", self.df.shape[0])

        # 获取特定列的名称    -----------判断是否全部列是否为对应的值
        # column_name = "是否已拍照"
        # 获取特定列的所有值
        # column_values = self.df[column_name]
        # print("column_values全部值为：", column_values)
        # 判断列中的值是否都为同一个字符串"True"
        # all_true = all(value == "已拍照" for value in column_values)
        # print("all-true:", all_true)

        # self.sub_index = 0
        # for index, row in self.df.iterrows():   # 判断某一行的那一列不是“已拍照”时候
        #     if row[column_name] != "已拍照":
        #         self.temp_save_stu = index
        #         self.sub_index = self.temp_save_stu - old_index
        #         print("相差的索引：", self.sub_index)
        #         print("找到不符合条件的行，索引为：", index)
        #         break  # 找到后立即退出循环
        # print("找到不符合条件的行，索引为：", self.temp_save_stu)
        # if self.index + 1 > self.df.shape[0]:
        # if all_true:
        #     self.photo_count = 0
        #     frame = wx.Frame(None, -1, "Message Dialog Example")
        #     # 创建一个消息对话框
        #     dlg = wx.MessageDialog(frame, "该班级已全部完成拍照", "提示", wx.OK | wx.ICON_INFORMATION)
        #     # 显示对话框
        #     result = dlg.ShowModal()
        #     # 根据用户的操作做相应处理
        #     if result == wx.ID_OK:
        #         # 用户点击了确定按钮
        #         dlg.Destroy()  # 销毁对话框
            # self.name_text.SetValue("已完成！")
            # self.number_text.SetValue("-1")
        self.df.to_excel(self.path, index=False)
        # else:
        #
        #     if self.index + 1 > self.df.shape[0]:
        #         self.index = self.temp_save_stu
        #     self.photo_count = 0
        #     Namestu = self.df.iloc[self.index, 1]
        #     self.name_text.SetValue(Namestu)
        #     self.second_window.name_text.SetValue(Namestu)
        #     Number = self.df.iloc[self.index, 2]
        #     self.number_text.SetValue(str(Number))


    def on_image_click(self, event, index, static_bitmap, static_bitmap_cam):
        if self.dialog is None:
            self.dialog = wx.Dialog(self, title="图片展示", size=(1000, 500))
        self.selected_image_index = index
        # if index == 0:
        #     # self.set_image_transparent(static_bitmap)
        #     self.set_image_opaque(static_bitmap_cam)
        # self.refresh_images()  # 刷新图片显示，以显示选中效果
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))  # 更改鼠标指针为手型
        # wx.CallAfter(self.on_choose)   # 同时触发选中图片事件
        if self.selected_image_index is not None:
            if self.selected_image_index == 0:
                # self.set_image_opaque(static_bitmap=self.static_bitmap)
                self.static_bitmap_cam.Destroy()
                self.fimg = self.image_cut1

                # x, y = self.static_bitmap.GetPosition()
                width, height = self.static_bitmap.GetSize()
                # 计算新的位置
                new_x = (self.dialog.GetSize()[0] - width) // 2
                new_y = (self.dialog.GetSize()[1] - height) // 2 - 60  # 根据需求进行调整
                # 设置新的位置
                self.static_bitmap.SetPosition((new_x, new_y))

            elif self.selected_image_index == 1:
                # self.set_image_opaque(static_bitmap=self.static_bitmap_cam)
                self.static_bitmap.Destroy()
                self.fimg = self.image_cut2

                # x, y = self.static_bitmap_cam.GetPosition()
                width, height = self.static_bitmap_cam.GetSize()
                # 计算新的位置
                new_x = (self.dialog.GetSize()[0] - width) // 2
                new_y = (self.dialog.GetSize()[1] - height) // 2 - 60 # 根据需求进行调整
                # 设置新的位置
                self.static_bitmap_cam.SetPosition((new_x, new_y))

    def on_image_click_cam(self, event, index, static_bitmap, static_bitmap_cam):
        if self.dialog is None:
            self.dialog = wx.Dialog(self, title="图片展示", size=(1000, 500))
        self.selected_image_index = index
        # if index == 1:
        #     # self.set_image_transparent(static_bitmap_cam)
        #     self.set_image_opaque(static_bitmap)
        # self.refresh_images()  # 刷新图片显示，以显示选中效果
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))  # 更改鼠标指针为手型
        # wx.CallAfter(self.on_choose)  # 同时触发选中图片事件
        if self.selected_image_index is not None:
            if self.selected_image_index == 0:
                # self.set_image_opaque(static_bitmap=self.static_bitmap)
                self.static_bitmap_cam.Destroy()
                self.fimg = self.image_cut1

                # x, y = self.static_bitmap.GetPosition()
                width, height = self.static_bitmap.GetSize()
                # 计算新的位置
                new_x = (self.dialog.GetSize()[0] - width) // 2
                new_y = (self.dialog.GetSize()[1] - height) // 2 - 60  # 根据需求进行调整
                # 设置新的位置
                self.static_bitmap.SetPosition((new_x, new_y))

            elif self.selected_image_index == 1:
                # self.set_image_opaque(static_bitmap=self.static_bitmap_cam)
                self.static_bitmap.Destroy()
                self.fimg = self.image_cut2

                # x, y = self.static_bitmap_cam.GetPosition()
                width, height = self.static_bitmap_cam.GetSize()
                # 计算新的位置
                new_x = (self.dialog.GetSize()[0] - width) // 2
                new_y = (self.dialog.GetSize()[1] - height) // 2 - 60 # 根据需求进行调整
                # 设置新的位置
                self.static_bitmap_cam.SetPosition((new_x, new_y))

    # def refresh_images(self):
    #     for child in self.dialog.GetChildren():
    #         if isinstance(child, wx.StaticBitmap):
    #             child.Refresh()

    def set_image_transparent(self, static_bitmap):
        bitmap = static_bitmap.GetBitmap()
        image = bitmap.ConvertToImage()

        if not image.HasAlpha():  # 检查是否已经有alpha通道
            image.InitAlpha()
            for x in range(image.GetWidth()):
                for y in range(image.GetHeight()):
                    alpha = image.GetAlpha(x, y)
                    image.SetAlpha(x, y, 128)  # 设置半透明度为128
        else:
            for x in range(image.GetWidth()):
                for y in range(image.GetHeight()):
                    alpha = image.GetAlpha(x, y)
                    image.SetAlpha(x, y, 128)  # 设置半透明度为128

        static_bitmap.SetBitmap(wx.Bitmap(image))

    def set_image_opaque(self, static_bitmap):
        bitmap = static_bitmap.GetBitmap()
        image = bitmap.ConvertToImage()

        if not image.HasAlpha():  # 检查是否已经有alpha通道
            image.InitAlpha()
            for x in range(image.GetWidth()):
                for y in range(image.GetHeight()):
                    image.SetAlpha(x, y, 255)  # 设置完全不透明
        else:
            for x in range(image.GetWidth()):
                for y in range(image.GetHeight()):
                    image.SetAlpha(x, y, 255)  # 设置完全不透明
        static_bitmap.SetBitmap(wx.Bitmap(image))

    # def rephoto_click(self, event):
    #     wx.CallAfter(self.on_dclose, event)
    #     wx.CallAfter(self.on_picture, event)
    #
    #     # # 在这里处理按钮点击事件
    #     # frame = wx.Frame(None, -1, "Message Dialog Example")
    #     # # 创建一个消息对话框
    #     # dlg = wx.MessageDialog(frame, "重拍成功", "提示", wx.OK | wx.ICON_INFORMATION)
    #     # # 显示对话框
    #     # result = dlg.ShowModal()
    #     # # 根据用户的操作做相应处理
    #     # if result == wx.ID_OK:
    #     #     # 用户点击了确定按钮
    #     #     dlg.Destroy()  # 销毁对话框

    def compIdcardAndPhoto_click(self, event):
        if self.dialog is not None:
            if self.selected_image_index == 0:  #第一张图片
                bitmap = self.static_bitmap.GetBitmap()
                # 将位图对象转换为图像对象
                image = bitmap.ConvertToImage()
                # 获取图像数据
                image_data = image.GetData()
                # 创建 PIL 图像对象
                pil_image = Image.frombuffer('RGB', (image.GetWidth(), image.GetHeight()), image_data)
                # 设置DPI
                dpi = (300, 300)  # 设置为300 DPI
                Stuname = f"{self.idxname}_{self.ididnum}.jpg"
                tempPhoto_folder = os.path.join(self.save_folder, "temp_Photo")
                if not os.path.exists(tempPhoto_folder):
                    os.makedirs(tempPhoto_folder)
                filename = os.path.join(tempPhoto_folder, Stuname)
                pil_image.save(filename, dpi=dpi)
                comp = CompanyIdAndPhoto(path1=self.file_path, path2=filename)
                mess = comp.company()
                print("mess的结果", mess)
                if mess:
                    self.df.loc[self.index, "比对结果"] = "正确"
                    wx.MessageBox("智能匹配结果正确！", "提示", wx.OK | wx.ICON_INFORMATION)
                    print("智能匹配结果正确")
                else:
                    self.df.loc[self.index, "比对结果"] = "错误"
                    wx.MessageBox("智能匹配失败，请人工对比！", "提示", wx.OK | wx.ICON_INFORMATION)
                    print("智能匹配失败，请人工对比")
            else:# 第二张图片
                bitmap = self.static_bitmap_cam.GetBitmap()
                # 将位图对象转换为图像对象
                image = bitmap.ConvertToImage()
                # 获取图像数据
                image_data = image.GetData()
                # 创建 PIL 图像对象
                pil_image = Image.frombuffer('RGB', (image.GetWidth(), image.GetHeight()), image_data)
                # 设置DPI
                dpi = (300, 300)  # 设置为300 DPI
                Stuname = f"{self.idxname}_{self.ididnum}.jpg"
                tempPhoto_folder = os.path.join(self.save_folder, "temp_Photo")
                if not os.path.exists(tempPhoto_folder):
                    os.makedirs(tempPhoto_folder)
                filename = os.path.join(tempPhoto_folder, Stuname)
                pil_image.save(filename, dpi=dpi)
                comp = CompanyIdAndPhoto(path1=self.file_path, path2=filename)
                mess = comp.company()
                print("mess的结果", mess)   #  True or False  匹配正确或错误
                if mess:
                    self.df.loc[self.index, "比对结果"] = "正确"
                    wx.MessageBox("智能匹配结果正确！", "提示", wx.OK | wx.ICON_INFORMATION)
                    print("智能匹配结果正确")
                else:
                    self.df.loc[self.index, "比对结果"] = "错误"
                    wx.MessageBox("智能匹配失败，请人工对比！", "提示", wx.OK | wx.ICON_INFORMATION)
                    print("智能匹配失败，请人工对比")
            #================================将匹配结果写入到excel和grid中========================
            # self.df.loc[self.index, "比对结果"] = "正确"
            if self.path is not None:
                self.df.to_excel(self.path, index=False)
                # 将DataFrame转换为二维列表，并将内容添加到grid中
                rows = self.df.values.tolist()
                header = self.df.columns.tolist()
                if self.grid.GetNumberCols() < len(header):
                    self.grid.AppendCols(len(header) - self.grid.GetNumberCols())  # 添加对应数量的列

                for col_idx, col_name in enumerate(header):
                    self.grid.SetColLabelValue(col_idx, col_name)
                    if col_name == '身份证号':
                        self.grid.SetColSize(col_idx, 500)  # 设置每个表格的大小
                    else:
                        self.grid.SetColSize(col_idx, 300)  # 设置每个表格的大小
                if self.grid.GetNumberRows() < len(rows):
                    self.grid.AppendRows(len(rows) - self.grid.GetNumberRows())  # 添加对应数量的行

                for row_idx, row_data in enumerate(rows):
                    for col_idx, cell_value in enumerate(row_data):
                        self.grid.SetCellValue(row_idx, col_idx, str(cell_value))
                        self.grid.SetRowSize(row_idx, 100)  # default = 30
        # return '匹配正确'

    def manual_click(self, event):
        self.df.loc[self.index, "比对结果"] = "正确"
        if self.path is not None:
            self.df.to_excel(self.path, index=False)
            # 将DataFrame转换为二维列表，并将内容添加到grid中
            rows = self.df.values.tolist()
            header = self.df.columns.tolist()
            if self.grid.GetNumberCols() < len(header):
                self.grid.AppendCols(len(header) - self.grid.GetNumberCols())  # 添加对应数量的列

            for col_idx, col_name in enumerate(header):
                self.grid.SetColLabelValue(col_idx, col_name)
                if col_name == '身份证号':
                    self.grid.SetColSize(col_idx, 500)  # 设置每个表格的大小
                else:
                    self.grid.SetColSize(col_idx, 300)  # 设置每个表格的大小
            if self.grid.GetNumberRows() < len(rows):
                self.grid.AppendRows(len(rows) - self.grid.GetNumberRows())  # 添加对应数量的行

            for row_idx, row_data in enumerate(rows):
                for col_idx, cell_value in enumerate(row_data):
                    self.grid.SetCellValue(row_idx, col_idx, str(cell_value))
                    self.grid.SetRowSize(row_idx, 100)  # default = 30
        wx.MessageBox("人工对比完成，请点击“确定并保存当前照片”", "提示", wx.OK | wx.ICON_INFORMATION)
        print("人工匹配结果正确")
        return '匹配正确'


class AnotherFrame(wx.Frame):
    def __init__(self, parent, title, capture):
        super(AnotherFrame, self).__init__(parent, title=title, size=(1080, 1920))
        self.capture_ano = capture
        self.video_panel = wx.Panel(self, size=(640, 480))

        vbox = wx.BoxSizer(wx.VERTICAL)
        # 将静态位图控件添加到垂直布局中
        vbox.Add(self.video_panel, 3, wx.EXPAND)
        # 设置 panel 的布局

        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(1000 // 24)  # 设置定时器间隔，这里假设每秒显示 24 帧

        self.name_text = wx.TextCtrl(self, value="请读取表格！", style=wx.TE_MULTILINE)
        self.name_text.SetMinSize((400, 200))
        font = wx.Font(160, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.name_text.SetFont(font)
        vbox.Add(self.name_text, 1, wx.EXPAND, 0)
        self.SetSizer(vbox)

        monitors = get_monitors()
        display_count = wx.Display.GetCount()
        if display_count > 1:
            second_display = wx.Display(1)  # 假设第二个显示器是索引为1的显示器
            second_frame_position = second_display.GetClientArea().GetPosition()
            self.SetPosition(second_frame_position)

            second_monitor = monitors[1]
            self.SetPosition((second_monitor.x, second_monitor.y))  # 设置窗口位置为第二个显示器的起始坐标
            self.SetSize((second_monitor.width, second_monitor.height))  # 设置窗口大小为第二个显示器的分辨率
            # self.ShowFullScreen(True)
            self.Show()


    def on_timer(self, event):
        ret, frame = self.capture_ano.read()
        if ret:
            h, w = frame.shape[:2]
            image = wx.ImageFromBuffer(w, h, cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))  #cv2.flip(frame, 1)设置镜像
            panel_width, panel_heigh = self.video_panel.GetSize()
            image = image.Scale(panel_width, panel_heigh, wx.IMAGE_QUALITY_HIGH)
            # bitmap = wx.Bitmap.FromBuffer(w, h, frame)
            bitmap = wx.Bitmap(image)
            dc = wx.ClientDC(self.video_panel)
            dc.DrawBitmap(bitmap, 0, 0)


class CircularButton(wx.Button):
    def __init__(self, parent, label, size):
        super(CircularButton, self).__init__(parent, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)  # 设置背景样式为wx.BG_STYLE_PAINT
        self.label = label
        # font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        # self.label.SetFont(font)
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        dc.Clear()
        dc.SetBrush(wx.Brush(wx.Colour(169, 169, 169)))  # 设置按钮颜色为灰色
        dc.SetPen(wx.TRANSPARENT_PEN)
        rect = self.GetClientRect()
        dc.DrawEllipse(rect.x, rect.y, rect.width, rect.height)
        dc.SetTextForeground(wx.WHITE)
        dc.DrawLabel(self.label, rect, alignment=wx.ALIGN_CENTER)


#=================================================身份证识别==================================
class GetIdCard(wx.Frame):
    def __init__(self, parent, title, video_capture, callback, tempIdPhoto_folder):
        super(GetIdCard, self).__init__(parent, title=title, size=(800, 640))
        self.idcardCapture = video_capture
        self.idcard_panel = wx.Panel(self, size=(640, 360))
        self.idcard_btn = wx.Button(self, label="拍照并识别")  # 创建按钮
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.idcard_panel, 3, wx.EXPAND)
        vbox.Add(self.idcard_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 20)
        # vbox.Add(self.idcard_panel, 1, wx.EXPAND | wx.ALL, 5)
        # vbox.Add(self.idcard_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 20)
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.timer.Start(1000 // 24)  # 设置定时器间隔，这里假设每秒显示 24 帧
        self.Bind(wx.EVT_BUTTON, self.idcard_on_capture1, self.idcard_btn)
        self.SetSizer(vbox)
        self.id_file_path = None
        self.callback = callback
        self.tempIdPhoto_folder = tempIdPhoto_folder
        self.Show()
    def on_timer(self, event):
        ret, frame = self.idcardCapture.read()
        frame = cv2.flip(frame, 1)
        if ret:
            h, w = frame.shape[:2]
            image = wx.ImageFromBuffer(w, h,
                                       cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB))  # cv2.flip(frame, 1)设置镜像
            panel_width, panel_heigh = self.idcard_panel.GetSize()
            image = image.Scale(panel_width, panel_heigh, wx.IMAGE_QUALITY_HIGH)
            bitmap = wx.Bitmap(image)
            dc = wx.ClientDC(self.idcard_panel)
            dc.DrawBitmap(bitmap, 0, 0)

    def idcard_on_capture1(self, event):
        file_path = None
        xname = None
        idnum = None
        # i = 0  #用来计数当前班级第i个同学进行身份证拍照检索信息
        ret, frame = self.idcardCapture.read()
        if ret:
            srcimg = cv2.flip(frame,0)
            myocr = OCR()
            results = myocr.det_rec(srcimg)
            match = re.search(r'姓名(.+)', results[0]['text'])    #判断是否有文本
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
                    output_folder = os.path.join(self.tempIdPhoto_folder, "temp_IdPhoto")
                    if not os.path.exists(output_folder):
                        os.makedirs(output_folder)
                    file_name = f'idcard_{idnum}.jpg'
                    file_path = os.path.join(output_folder, file_name)
                    # self.id_file_path = file_path
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
                    if self.callback:                      #  只有调用成功了才可以callback回调给mainFrame
                        self.callback(file_path, xname, str(idnum))
                    self.Close()
                    self.idcardCapture.release()
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


#=========================================================对比照片和人是否一致==============================
class CompanyIdAndPhoto():
    def __init__(self, path1, path2):
        self.path1 = path1   # id_path  身份证的照片
        self.path2 = path2   # phpto_path   现场拍的照片
    def company(self):
        id_img = cv2.imdecode(np.fromfile(self.path1, dtype=np.uint8),-1)  # 身份证照片
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
            return True  # '匹配正确'
        else:
            print('匹配错误')
            return False   #'匹配错误'

class mainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)  # APP_TITLE = u'拍照'
        self.Frame = mainFrame(None)
        self.Frame.Show()

        # # 显示第二个窗口在另一个显示器上
        # display_count = wx.Display.GetCount()
        # if display_count > 1:
        #     second_display = wx.Display(1)  # 假设第二个显示器是索引为1的显示器
        #     second_frame_position = second_display.GetClientArea().GetPosition()
        #     self.OtherFrame = AnotherFrame(main_frame = self.Frame, parent=None, title='学生窗口')
        #     self.OtherFrame.SetPosition(second_frame_position)
        #     self.OtherFrame.Show()
        return True

if __name__ == "__main__":
    app = mainApp()
    app.MainLoop()