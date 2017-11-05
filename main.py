import serial

device = "COM5"
baudrate = 9600

arduino_port = serial.Serial(device,
                           baudrate = baudrate,
                           parity=serial.PARITY_ODD,
                           stopbits=serial.STOPBITS_TWO,
                           bytesize=serial.SEVENBITS,
                           rtscts   = False,
                           dsrdtr   = False,
                           xonxoff  = False,
                           timeout  = 25)


isStarted = False
while True:
    file = open("testE.csv", 'a')
    rawLine = arduino_port.readline()
    line = str(rawLine)[2:-5]
    splitLine = line.split(',')
    index = splitLine[len(splitLine)-1]
    if (index == "1"):
        if (isStarted):
            file.write("\n" + line + ",")
        else:
            file.write(line + ",")
            isStarted = True
    else:
        file.write(line + ",")
    print(line + ",")
    file.close()



