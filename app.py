#!/usr/bin/env python
import csv
import math
import os
import platform
import random
import sys

import numpy as np
import wx

import cv2
from binary_image import BinaryImage, SourceType


class DrawFrame(wx.Frame):

    def __init__(self, *args, **kwargs):
        wx.Frame.__init__(self, *args, **kwargs)
        # Change as necessary
        self.source_type = SourceType.V_CHANNEL

        self._create_menu()
        self.create_widgets()
        self.create_bindings()
        self.setup_layout()
        self.grad_x_threshold = None
        self.grad_y_threshold = None
        self.mag_threshold = None
        self.dir_threshold = None

        self.Show()


    def _create_menu(self):
        # Create the menubar
        menuBar = wx.MenuBar()

        menu = wx.Menu()
        item = menu.Append(wx.ID_ANY, "&Open Image", "Open file")
        self.Bind(wx.EVT_MENU, self.OnFileOpen, item)

        item = menu.Append(wx.ID_ANY, "&Save Config", "Save the thresholds to config file.")
        self.Bind(wx.EVT_MENU, self.OnFileSave, item)
        
        menu.Append(wx.ID_EXIT, "E&xit\tAlt-X", "Exit the app")
        self.Bind(wx.EVT_MENU, self.OnClose, id=wx.ID_EXIT)

        menuBar.Append(menu, "&File")
        self.SetMenuBar(menuBar)



    def create_widgets(self):
        binary_image = BinaryImage(self.source_type)

        self.panel = wx.Panel(self)

        self.image_src = wx.StaticBitmap(self)
        self.image_color = wx.StaticBitmap(self)
        self.image_binary = wx.StaticBitmap(self)

        self.text_grad_x = wx.StaticText(self, label="Gradient X Thresholds")
        self.text_grad_y = wx.StaticText(self, label="Gradient Y Thresholds")
        self.text_mag = wx.StaticText(self, label="Magnitude  Thresholds")
        self.text_dir = wx.StaticText(self, label="Direction Thresholds")

        self.slider_grad_x_min = wx.Slider(self,
                                           value=binary_image.grad_x_threshold[0],
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_grad_x_max = wx.Slider(self,
                                           value=binary_image.grad_x_threshold[1],
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_grad_y_min = wx.Slider(self,
                                           value=binary_image.grad_y_threshold[0],
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_grad_y_max = wx.Slider(self,
                                           value=binary_image.grad_y_threshold[1],
                                           minValue=0,
                                           maxValue=255,
                                           pos=wx.DefaultPosition,
                                           size=wx.DefaultSize,
                                           style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_mag_min = wx.Slider(self,
                                        value=binary_image.mag_threshold[0],
                                        minValue=0,
                                        maxValue=255,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_mag_max = wx.Slider(self,
                                        value=binary_image.mag_threshold[1],
                                        minValue=0,
                                        maxValue=255,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL | wx.SL_LABELS)

        self.slider_dir_min = wx.Slider(self,
                                        value=math.degrees(
                                            binary_image.dir_threshold[0]),
                                        minValue=0,
                                        maxValue=360,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider_dir_max = wx.Slider(self,
                                        value=math.degrees(
                                            binary_image.dir_threshold[1]),
                                        minValue=0,
                                        maxValue=360,
                                        pos=wx.DefaultPosition,
                                        size=wx.DefaultSize,
                                        style=wx.SL_HORIZONTAL | wx.SL_LABELS)

    def create_bindings(self):
        self.slider_grad_x_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_grad_x_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_grad_y_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_grad_y_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_mag_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_mag_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_dir_min.Bind(wx.EVT_SCROLL, self.OnScrollSlider)
        self.slider_dir_max.Bind(wx.EVT_SCROLL, self.OnScrollSlider)

    def setup_layout(self):
        self.Sizer = wx.BoxSizer(wx.VERTICAL)
        
        image_sizer = wx.BoxSizer(wx.HORIZONTAL)
        image_sizer.Add(self.image_src, 5, wx.CENTER)
        image_sizer.AddSpacer(5)
        image_sizer.Add(self.image_color, 5, wx.CENTER)
        image_sizer.AddSpacer(5)
        image_sizer.Add(self.image_binary, 5, wx.CENTER)
        image_sizer.AddSpacer(5)

        self.Sizer.Add(image_sizer, 1, wx.CENTER)
        self.Sizer.AddSpacer(10)
        self.Sizer.Add(self.text_grad_x, 1, wx.CENTER)
        self.Sizer.Add(self.slider_grad_x_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_grad_x_max, 1, wx.EXPAND)
        self.Sizer.AddSpacer(10)
        self.Sizer.Add(self.text_grad_y, 1, wx.CENTER)
        self.Sizer.Add(self.slider_grad_y_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_grad_y_max, 1, wx.EXPAND)
        self.Sizer.AddSpacer(10)
        self.Sizer.Add(self.text_mag, 1, wx.CENTER)
        self.Sizer.Add(self.slider_mag_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_mag_max, 1, wx.EXPAND)
        self.Sizer.AddSpacer(10)
        self.Sizer.Add(self.text_dir, 1, wx.CENTER)
        self.Sizer.Add(self.slider_dir_min, 1, wx.EXPAND)
        self.Sizer.Add(self.slider_dir_max, 1, wx.EXPAND)

    def OnClose(self, evt):
        self.Close()

    def OnScrollSlider(self, event):
        self.refresh()

    def scale_image_to_bitmap(self, image, width, height):
        image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
        return image.ConvertToBitmap()

    def refresh(self):
        self.grad_x_threshold = (self.slider_grad_x_min.GetValue(),
                            self.slider_grad_x_max.GetValue())
        self.grad_y_threshold = (self.slider_grad_y_min.GetValue(),
                            self.slider_grad_y_max.GetValue())
        self.mag_threshold = (self.slider_mag_min.GetValue(),
                         self.slider_mag_max.GetValue())
        # Convert to radians
        self.dir_threshold = (math.radians(self.slider_dir_min.GetValue()),
                         math.radians(self.slider_dir_max.GetValue()))

        print("grad x threshold = {}".format(self.grad_x_threshold))
        print("grad y threshold = {}".format(self.grad_y_threshold))
        print("magnitude threshold = {}".format(self.mag_threshold))
        print("directional threshold = {}".format(self.dir_threshold))

        # Always use a new object so different settings don't 
        # get merged together via previous image merging.
        binary_image = BinaryImage(self.source_type)

        binary_image.set_thresholds(grad_x_threshold=self.grad_x_threshold,
                          grad_y_threshold=self.grad_y_threshold,
                          mag_threshold=self.mag_threshold,
                          dir_threshold=self.dir_threshold)
        #process the image
        print("Process image")
        binary = binary_image.process_image(self.img)
        binary = binary * 255
        self.display_image(binary.astype(np.uint8), self.image_binary)
        color_image = binary_image.source_channel
        self.display_image(color_image.astype(np.uint8), self.image_color)
        self.panel.Layout()

    def display_image(self, image, widget):
        wximage = self.NumpyArrayToWxImage(image)
        bmp = self.scale_image_to_bitmap(
            wximage, wximage.GetWidth() * .25, wximage.GetHeight() * .25)
        widget.SetBitmap(bmp)

    def OnExit(self, evt):
        self.close()

    def OnFileOpen(self, evt):
        path = ""
        dlg = wx.FileDialog(self, "Choose an image File",
                            ".", "", "*.jpg", wx.FD_OPEN)
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
        self.img = cv2.imread(self.filename)
        self.img = self.img[:, :, ::-1]
        image = self.NumpyArrayToWxImage(self.img)
        bmp = self.scale_image_to_bitmap(
            image, image.GetWidth() * .25, image.GetHeight() * .25)
        self.image_src.SetBitmap(bmp)
        self.refresh()

    def SaveData(self):
        bi = BinaryImage(self.source_type)
        bi.set_thresholds(grad_x_threshold=self.grad_x_threshold,
                            grad_y_threshold=self.grad_y_threshold,
                            mag_threshold=self.mag_threshold,
                            dir_threshold=self.dir_threshold)
        bi.save_thresholds()

    def NumpyArrayToWxImage(self, nparray):
        if (len(np.shape(nparray)) == 2):
            nparray = np.dstack((nparray, nparray, nparray))

        height, width, color = np.shape(nparray)
        image = wx.EmptyImage(width, height)
        image.SetData(nparray.tostring())
        return image


app = wx.App(False)
F = DrawFrame(None, title="Image Thresholds tool", size=(1200, 900))
app.MainLoop()
