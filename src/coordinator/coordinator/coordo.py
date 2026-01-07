import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from datetime import datetime
from geometry_msgs.msg import PointStamped

STATE_STOP = 0
STATE_ROTATE = 1
STATE_WANDERER = 2
STATE_LOCK_IN = 3

class Coordo(Node):
    def __init__(self):
        super().__init__('coordo')
        self.ball_subscriber = self.create_subscription(PointStamped, '/ball_3d', self.ball_scan_callback, 10)
        self.state_publisher = self.create_publisher(UInt8, '/coordinator/state', 10)
        self.point_publisher = self.create_publisher(PointStamped, '/coordinator/point', 10)
        # timer_period = 0.5 # seconds
        self.timer = self.create_timer(1/6, self.send_state)
        self.i = 0
        self.start_timer = datetime.now()
        self.state = STATE_ROTATE
        print(f'[INFO] {datetime.now()} : Démarrage avec STATE_ROTATE')
        msg = UInt8()
        msg.data = self.state
        self.state_publisher.publish(msg)

    def ball_scan_callback(self, msg):
        prev_state = self.state
        self.state = STATE_LOCK_IN
        
        if prev_state != self.state : print(f'[INFO] {datetime.now()} : Changement d\'état STATE_LOCK_IN')
        
        msg_state = UInt8()
        msg_state.data = self.state
        self.state_publisher.publish(msg_state)
        
        self.point_publisher.publish(msg)

        ''' 
            envoyer le
        '''

    def send_state(self):
        msg = UInt8()

        if (self.state == STATE_ROTATE):
            now = datetime.now()
            delta = now - self.start_timer
            
            if delta.seconds >= 20:
                self.state = STATE_WANDERER
                msg.data = self.state
                self.state_publisher.publish(msg)
                print(f'[INFO] {datetime.now()} : Changement d\'état STATE_WANDERER')
                self.start_timer = datetime.now()

        elif (self.state == STATE_WANDERER):
            now = datetime.now()
            delta = now - self.start_timer

            if delta.seconds >= 60:
                self.state = STATE_ROTATE
                msg.data = self.state
                self.state_publisher.publish(msg)
                print(f'[INFO] {datetime.now()} : Changement d\'état STATE_ROTATE')
                self.start_timer = datetime.now()
        
        elif (self.state == STATE_LOCK_IN): pass

        # msg = String()
        # msg.data = f'Hello, world! {self.i}'
        # self.publisher_.publish(msg)
        # self.get_logger().info(f'Publishing: "{msg.data}"')
        # self.i += 1

def main(args=None):
    rclpy.init(args=args) # Initialize the ROS2 Python system
    node = Coordo() # Create an instance of the Listener node
    rclpy.spin(node) # Keep the node running, listening for messages
    node.destroy_node() # Cleanup when the node is stopped
    rclpy.shutdown() # It cleans up all ROS2 resources used by the node

if __name__ == '__main__':
    main()