#!/usr/bin/env python
import wx
import random
import csv
import cv2
import sys
import numpy as np

import os
import platform
import math

import lane_find as lf


class DrawFrame(wx.Frame):

    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)

        self.CreateMenu()
        self.CreateWidgets()
        self.CreateBindings()
        self.SetupLayout()
        image_path = "output_images/test4_undistorted.jpg"
        # create the image holder:
        # image = wx.Image(image_path, wx.BITMAP_TYPE_ANY)
        # bmp = self.scale_image_to_bitmap(
        #     image, image.GetWidth() * .25, image.GetHeight() * .25)
        # self.image_src.SetBitmap(bmp)

        self.img = cv2.imread(image_path)
        self.img = self.img[:, :, ::-1]
        image = self.NumpyArrayToWxImage(self.img)
        bmp = self.scale_image_to_bitmap(
            image, image.GetWidth() * .25, image.GetHeight() * .25)
        self.image_src.SetBitmap(bmp)
        self.M, self.Minv = lf.get_perspective_transform_matrices()

        self.Refresh()

        self.Show()
        # self.filename = os.path.abspath(
        #     "../../data/raw/swerving_full copy/driving_log.csv")
        # self.LoadData()

    def CreateMenu(self):
        # Create the menubar
        menuBar = wx.MenuBar()

        # and a menu
        menu = wx.Menu()

        # add an item to the menu, using \tKeyName automatically
        # creates an accelerator, the third param is some help text
        # that will show up in the statusbar
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit this simple sample")

        # # bind the menu event to an event handler
        # self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_EXIT)

        # # and put the menu on the menubar
        # item = wx.MenuItem(menu, wx.ID_ANY, "&Open", "Open file")
        # self.Bind(wx.EVT_MENU, self.OnFileOpen, item)
        # menu.Append(item)
        # item = wx.MenuItem(menu, wx.ID_ANY, "&Save", "Save file")
        # self.Bind(wx.EVT_MENU, self.OnFileSave, item)
        # menu.Append(item)
        # self.SetMenuBar(menuBar)
        # menuBar.Append(menu, "&File")

    def CreateWidgets(self):
        self.panel = wx.Panel(self)

        self.image_src = wx.StaticBitmap(self)
        self.image_dst = wx.StaticBitmap(self)
        self.text_grad_x = wx.StaticText(self, label="Gradient X Thresholds")
        self.text_grad_y = wx.StaticText(self, label="Gradient Y Thresholds")
        self.text_mag = wx.StaticText(self, label="Magnitude  Thresholds")
        self.text_dir = wx.StaticText(self, label="Direction Thresholds")

        self.slider_grad_x_min = wx.Slider(self,
                                           value=46,
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL)
        self.slider_grad_x_max = wx.Slider(self,
                                           value=82,
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL)
        self.slider_grad_y_min = wx.Slider(self,
                                           value=14,
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL)
        self.slider_grad_y_max = wx.Slider(self,
                                           value=166,
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL)
        self.slider_mag_min = wx.Slider(self,
                                        value=17,
                                        minValue=0,
                                        maxValue=255,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL)
        self.slider_mag_max = wx.Slider(self,
                                        value=255,
                                        minValue=0,
                                        maxValue=255,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL)

        self.slider_dir_min = wx.Slider(self,
                                        value=0,
                                        minValue=0,
                                        maxValue=360,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL)
        self.slider_dir_max = wx.Slider(self,
                                        value=math.degrees(0.6632),
                                        minValue=0,
                                        maxValue=360,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL)

    def CreateBindings(self):
        self.slider_grad_x_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_grad_x_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_grad_y_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_grad_y_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_mag_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_mag_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_dir_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_dir_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)

    def SetupLayout(self):
        image_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        image_sizer.Add(self.image_src, 1, wx.CENTER)
        image_sizer.Add(self.image_dst, 1, wx.CENTER)
        self.Sizer.Add(image_sizer, 1, wx.CENTER)

        self.Sizer.Add(self.text_grad_x, 1, wx.CENTER)
        self.Sizer.Add(self.slider_grad_x_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_grad_x_max, 1, wx.EXPAND)
        self.Sizer.Add(self.text_grad_y, 1, wx.CENTER)
        self.Sizer.Add(self.slider_grad_y_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_grad_y_max, 1, wx.EXPAND)
        self.Sizer.Add(self.text_mag, 1, wx.CENTER)
        self.Sizer.Add(self.slider_mag_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_mag_max, 1, wx.EXPAND)
        self.Sizer.Add(self.text_dir, 1, wx.CENTER)
        self.Sizer.Add(self.slider_dir_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_dir_max, 1, wx.EXPAND)
        # Set font sizes
        # self.steering_angle_st.SetFont(
        #     wx.FFont(20, wx.FONTFAMILY_SWISS, wx.FONTFLAG_BOLD))
        # self.image_index_st.SetFont(
        #     wx.FFont(20, wx.FONTFAMILY_SWISS, wx.FONTFLAG_BOLD))

    def OnClose(self, evt):
        self.Close()

    def OnScrollSlider(self, event):
        self.Refresh()

    def scale_image_to_bitmap(self, image, width, height):
        image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
        return image.ConvertToBitmap()

    def Refresh(self):
        grad_x_threshold = (self.slider_grad_x_min.GetValue(),
                            self.slider_grad_x_max.GetValue())
        grad_y_threshold = (self.slider_grad_y_min.GetValue(),
                            self.slider_grad_y_max.GetValue())
        mag_threshold = (self.slider_mag_min.GetValue(),
                         self.slider_mag_max.GetValue())
        # Convert to radians
        dir_threshold = (math.radians(self.slider_dir_min.GetValue()),
                         math.radians(self.slider_dir_max.GetValue()))

        print(grad_x_threshold)
        print(grad_y_threshold)
        print(mag_threshold)
        print(dir_threshold)

        dst_image = lf.prepare_img(self.img,
                                   self.M,
                                   ksize=9,
                                   grad_x_threshold=grad_x_threshold,
                                   grad_y_threshold=grad_x_threshold,
                                   mag_threshold=mag_threshold,
                                   dir_threshold=dir_threshold)
        dst_image *= 255
        image = self.NumpyArrayToWxImage(dst_image)
        bmp = self.scale_image_to_bitmap(
            image, image.GetWidth() * .25, image.GetHeight() * .25)
        self.image_dst.SetBitmap(bmp)
        self.panel.Layout()

    def OnExit(self, evt):
        self.close()

    def OnFileOpen(self, evt):
        path = ""
        dlg = wx.FileDialog(self, "Choose A CSV File",
                            ".", "", "*.csv", wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
        else:
            dlg.Destroy()
            return

        dlg.Destroy()
        self.filename = path
        self.LoadData()

    def OnFileSave(self, evt):
        self.SaveData()

    def LoadData(self):
        self.lines = []
        if not os.path.isfile(self.filename):
            return
        self.image_dir = os.path.dirname(self.filename) + '/IMG/'

        with open(self.filename, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if not line:
                    continue
                # if first time opening this file,
                # add the selected column as unselected.
                if(len(line) == 7):
                    line.append(0)
                self.lines.append(line)
        self.slider.SetMax(len(self.lines))
        self.image_index = 0
        self.Refresh()

    def SaveData(self):
        with open(self.filename, 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for line in self.lines:
                writer.writerow(line)

    def NumpyArrayToWxImage(self, nparray):
        if (len(np.shape(nparray)) == 2):
            nparray = np.dstack((nparray, nparray, nparray))

        height, width, color = np.shape(nparray)
        image = wx.EmptyImage(width, height)
        image.SetData(nparray.tostring())
        return image


app = wx.App(False)
F = DrawFrame(None, title="Image Thresholds tool", size=(700, 700))
app.MainLoop()
