#!/usr/bin/env python3

from pytube import YouTube

url = 'https://www.youtube.com/watch?v=_rfgJ8ljxPk'
#url = input('Please input video url: ')
YouTube(url).streams.first().download()
