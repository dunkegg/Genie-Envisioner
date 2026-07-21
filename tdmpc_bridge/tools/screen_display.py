import time
import rospy
import random
import serial
from std_msgs.msg import Int32

T = 2

if __name__== '__main__':
    ser = serial.Serial("/dev/ttyUSB1", 9600)

    rospy.init_node('key_screen', anonymous=True)
    rate = rospy.Rate(10)

    pub_key = rospy.Publisher('key', Int32, queue_size=10)

    s_pre = '$001,'
    s_end = '#'

    while not rospy.is_shutdown() and ser.is_open:

        print("===============")
        
        num_target = random.randint(1000, 9999)
        s_target = str(num_target)

        s = bytes((s_pre + s_target + s_end).encode()).hex()
        send_data = bytes.fromhex(s)
        print("num : ", num_target)
        ser.write(send_data)
        
        target_msg = Int32()
        target_msg.data = int(num_target)
        pub_key.publish(target_msg)

        time.sleep(T)