import pyfirmata
import cv2
from time import sleep

def right():
    board.digital[in1].write(1)
    board.digital[in2].write(0)
    board.digital[in3].write(1)
    board.digital[in4].write(0)
    
def left():
    board.digital[in1].write(0)
    board.digital[in2].write(1)
    board.digital[in3].write(0)
    board.digital[in4].write(1)
    
def reverse():
    board.digital[in1].write(0)
    board.digital[in2].write(1)
    board.digital[in3].write(1)
    board.digital[in4].write(0)
        
def forward():
    board.digital[in1].write(1)
    board.digital[in2].write(0)
    board.digital[in3].write(0)
    board.digital[in4].write(1)

def stop():
    enA.write(0)
    enB.write(0)

def run():
    enA.write(0.5)
    enB.write(0.5)

# if __name__ == '__main__':

board = pyfirmata.Arduino('/dev/ttyUSB1')

led = 13
in1 = 9
in2 = 8
in3 = 3
in4 = 4
enA = board.digital[10]
enA.mode = pyfirmata.PWM
enB = board.digital[5]
enB.mode = pyfirmata.PWM
# try:
#     while True:
#     # board.digital[led].write(0)
#     # sleep(1)
#     # board.digital[led].write(1)
#     # sleep(1)
#         right()
#         enA.write(0.5)
#         enB.write(0.5)
# except KeyboardInterrupt:
#     stop()

