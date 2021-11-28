import pyfirmata
# import cv2
import time

# def right():
#     board.digital[in1].write(1)
#     board.digital[in2].write(0)
#     board.digital[in3].write(1)
#     board.digital[in4].write(0)
    
# def left():
#     board.digital[in1].write(0)
#     board.digital[in2].write(1)
#     board.digital[in3].write(0)
#     board.digital[in4].write(1)
    
# def reverse():
#     board.digital[in1].write(0)
#     board.digital[in2].write(1)
#     board.digital[in3].write(1)
#     board.digital[in4].write(0)
        
# def forward():
#     board.digital[in1].write(1)
#     board.digital[in2].write(0)
#     board.digital[in3].write(0)
#     board.digital[in4].write(1)

# def stop():
#     enA.write(0)
#     enB.write(0)

# def run():
#     enA.write(0.5)
#     enB.write(0.5)

# if __name__ == '__main__':

board = pyfirmata.Arduino('/dev/ttyUSB0')

spd = board.digital[2]
spd.mode = pyfirmata.INPUT
spd.enable_reporting()

# Start iterator to receive input data
it = pyfirmata.util.Iterator(board)
it.start()



while True:

    start_time = 0
    end_time = 0
    steps = 0
    steps_old = 0
    temp = 0
    rps = 0
    start_time = time.time()
    end_time = start_time + 1
    # print(time.strftime('%X %x %Z',time.localtime(start_time)))
    # print(time.strftime('%X %x %Z',time.localtime(end_time)))

    while(time.time() < end_time):
        print('here1')
        if(spd.read):
            print('here2')
            steps = steps + 1
            # while(spd.read):
            #     print('stuck here')
            #     pass
    print('here3')
    temp = steps - steps_old
    steps_old = steps
    rps = temp/20
    print(rps)    
            

# led = 13
# in1 = 9
# in2 = 8
# in3 = 3
# in4 = 4
# enA = board.digital[10]
# enA.mode = pyfirmata.PWM
# enB = board.digital[5]
# enB.mode = pyfirmata.PWM
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

