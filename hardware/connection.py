import serial
from time import sleep

def recv(serial):
    while True:
        data = serial.read_all()
        if data == '':
            continue
        else:
            break
        sleep(0.02)
    return data

def send(serial, send_data):
    if (serial.isOpen()):
        serial.write(send_data)  # 编码
        # print("发送成功", send_data)
    else:
        print("发送失败！")

if __name__ == '__main__':
    serial = serial.Serial('COM7', 9600, timeout=0.5)
    if serial.isOpen() :
        print("open success")
    else :
        print("open failed")

    #这里如果不加上一个while True，程序执行一次就自动跳出了
    while True:
        a = input("输入要发送的数据：")
        if a == 'quit':
            exit(print("退出程序"))
        send(int(a).to_bytes(4, 'big', signed = True)) #big表示最高位在前
        sleep(0.2)  # 起到一个延时的效果
