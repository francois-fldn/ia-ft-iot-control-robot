import rclpy
from rclpy.node import Node
from std_msgs.msg import UInt8
from datetime import datetime
from geometry_msgs.msg import PointStamped
import logging

STATE_STOP = 0
STATE_ROTATE = 1
STATE_WANDERER = 2
STATE_LOCK_IN = 3
STATE_GO = 4

def state_to_str(state_int):
    states = ['STATE_STOP', 'STATE_ROTATE', 'STATE_WANDERER', 'STATE_LOCK_IN', 'STATE_GO']
    return states[state_int]

class Coordo(Node):
    def __init__(self):
        super().__init__('coordo')

        self.ball_subscriber = self.create_subscription(PointStamped, '/ball_3d', self.ball_scan_callback, 10)
        self.state_publisher = self.create_publisher(UInt8, '/coordinator/state', 10)
        self.point_publisher = self.create_publisher(PointStamped, '/coordinator/point', 10)
        
        self.timer = self.create_timer(1/6, self.send_state)

        self.i = 0

        self.start_timer = datetime.now()

        self.state = STATE_STOP
        self.goal_achieved = False

        msg = UInt8()
        msg.data = self.state
        self.state_publisher.publish(msg)

        self.logger = logging.getLogger("ORCHESTRATOR")
        logging.basicConfig(
            level = logging.INFO,
            format = "[ORCHESTRATOR] [%(levelname)s] %(message)s"
            )
        
        self.logger.info(f'Démarrage avec {state_to_str(self.state)}')

    def ball_scan_callback(self, msg):
        prev_state = self.state

        if (self.goal_achieved): 
            return

        if (msg.point.x == 1000):
            return

        if (msg.point.x != 1000): 
            # print(f"coordonnees recues : {msg.point.x}, {msg.point.y}, {msg.point.z}")
            self.state = STATE_LOCK_IN
            if (msg.point.x <= 0.28):
                self.state = STATE_GO
                self.start_timer = datetime.now()

        if prev_state != self.state : self.logger.info(f'Changement d\'état {state_to_str(self.state)}')
        
        msg_state = UInt8()
        msg_state.data = self.state
        self.state_publisher.publish(msg_state)
        
        self.point_publisher.publish(msg)

        ''' 
            envoyer le
        '''

    def send_state(self):
        msg = UInt8()

        if (self.state == STATE_STOP):
            if (self.goal_achieved): return 
            now = datetime.now()
            delta = now - self.start_timer

            if delta.seconds >= 10: # une petite attente le temps que la simu se lance
                self.state = STATE_ROTATE
                msg.data = self.state
                self.state_publisher.publish(msg)
                self.logger.info(f'Changement d\'état {state_to_str(self.state)}')
                self.start_timer = datetime.now()


        if (self.state == STATE_ROTATE):
            now = datetime.now()
            delta = now - self.start_timer
            
            if delta.seconds >= 20:
                self.state = STATE_WANDERER
                msg.data = self.state
                self.state_publisher.publish(msg)
                self.logger.info(f'Changement d\'état {state_to_str(self.state)}')
                self.start_timer = datetime.now()


        elif (self.state == STATE_WANDERER):
            now = datetime.now()
            delta = now - self.start_timer

            if delta.seconds >= 60:
                self.state = STATE_ROTATE
                msg.data = self.state
                self.state_publisher.publish(msg)
                self.logger.info(f'Changement d\'état {state_to_str(self.state)}')
                self.start_timer = datetime.now()


        elif (self.state == STATE_LOCK_IN): pass

        elif (self.state == STATE_GO):
            now = datetime.now()
            delta = now - self.start_timer

            if delta.seconds >= 2:
                self.state = STATE_STOP
                msg.data = self.state
                self.state_publisher.publish(msg)
                self.logger.info(f'Changement d\'état {state_to_str(self.state)}')
                self.goal_achieved = True

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