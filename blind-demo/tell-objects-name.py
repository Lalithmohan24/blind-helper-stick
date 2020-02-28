#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import Jetson.GPIO as GPIO
import speech_recognition as sr
import os,sys

import jetson.utils
from subprocess import Popen
import argparse
import os,sys

# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage())

parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)
channel = 11
GPIO.setup(channel, GPIO.IN)


try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# create the camera and display
font = jetson.utils.cudaFont()
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()

def speech():
   r = sr.Recognizer()
   with sr.Microphone() as source:
      print("Speak Anything :")
      audio = r.adjust_for_ambient_noise(source)
      audio = r.listen(source)
      try:
         text = r.recognize_google(audio)
         if text == "what do you see":
            print("You said : {}".format(text))
            os.system('espeak "{} is front of you"'.format(class_desc))
      except:
         print("Sorry could not recognize what you said")                
         os.system('espeak "Sorry could not recognize what you said"')
# process frames until user exits
while display.IsOpen():
	# capture the image
	img, width, height = camera.CaptureRGBA()

	# classify the image
	class_idx, confidence = net.Classify(img, width, height)

	# find the object description
	class_desc = net.GetClassDesc(class_idx)

	# overlay the result on the image	
	font.OverlayText(img, width, height, "{:05.2f}% {:s}".format(confidence * 100, class_desc), 5, 5, font.White, font.Gray40)
	
	# render the image
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

        if GPIO.input(channel) == False:
           print("working")
           speech()
        else:
           print("nothing")

	net.PrintProfilerTimes()


#        if class_desc == "water bottle":
#        Popen(['espeak', class_desc])

#              os.system('espeak "water bottle is front of you"')
#              os.system('espeak "are you thresty"')
#        elif class_desc == "biscuit":

#              os.system('espeak "biscuit is front of you"')
#              os.system('espeak "are you hungry"')
	# print out performance info


