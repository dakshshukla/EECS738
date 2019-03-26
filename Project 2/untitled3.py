# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:16:52 2019

@author: daksh
"""

#import sys
#
#def input_with_tab(text):
#    #text = ""
#    while True:
#        char = sys.stdin.read(1)  # read 1 character from stdin
#        if char == '\t':  # if a tab was read
#            text_out = text
#            break
#        elif char == '\b': # backspace
#            text_out = ""
#            break
##        text += char
#
#    return text_out
#
#
#text_out = input_with_tab('Hello')




from pynput.keyboard import Key, Listener

def on_press(key):
    print('{0} pressed'.format(
        key))

def on_release(key):
    print('{0} release'.format(
        key))
    if key == Key.esc:
        # Stop listener
        return False

# Collect events until released
with Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()