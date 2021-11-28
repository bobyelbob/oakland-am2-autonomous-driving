import serial
import time

arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, timeout=.1)


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline().decode('utf-8').rstrip()
    return data


while True:
    # num = input("Enter a number: ")
    x = str(2)
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline().decode('utf-8').rstrip()
    # num=("2")
    # value = write_read(num)
    print(data)