#
# -*- coding: utf-8 -*-
#

#play locally
import winsound
winsound.PlaySound(r'C:\Windows\media\Speech On.wav', winsound.SND_FILENAME | winsound.SND_ASYNC)

#send by socket
import websocket
try:
    ws = websocket.create_connection("ws://192.168.1.40/ws")
    ws.send("on_buzzz")
    ws.close()
finally:
    pass

# verify sound has enougth time to play
import time
time.sleep(1)
