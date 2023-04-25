#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import face_recognition
import time


class FaceRecognitionNode(Node):

    def __init__(self):
        super().__init__("face_recognition_node")
        #self.counter_ = 0
        self.get_logger().info("Face recognition node started.")
        #self.create_timer(0.5, self.image_callback(self.msg))
        
        # Load known face encodings
        self.known_face_encodings = []
        self.known_face_names = []
        # Replace with your own known face images and names
        image1 = face_recognition.load_image_file("/home/logan/i9robot_ws/src/i9robot_camera/known_faces/Logan/logan.jpg")
        encoding1 = face_recognition.face_encodings(image1)[0]
        self.known_face_encodings.append(encoding1)
        self.known_face_names.append("Logan Naidoo")
        image2 = face_recognition.load_image_file("/home/logan/i9robot_ws/src/i9robot_camera/known_faces/Anis/anis.png")
        encoding2 = face_recognition.face_encodings(image2)[0]
        self.known_face_encodings.append(encoding2)
        self.known_face_names.append("Anis")
        
        # Initialize camera
        # self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()
        
        # FPS timer variables
        self.start_time = time.time()
        self.num_frames = 0
        self.fps = 0
        
        # Create subscriber and publisher
        self.image_sub = self.create_subscription(Image,"camera/image_raw",self.image_callback,10)
        self.image_pub = self.create_publisher(Image,"face_recognition/output",10)

    def image_callback(self, msg):
        # Update the logger counter
        #self.counter_ += 1
        #self.get_logger().info("Loop: " + str(self.counter_))
        # Convert ROS message to OpenCV image
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        
        # Find faces in frame
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        # Recognize faces
        face_names = []
        for face_encoding in face_encodings:
            # Compare face encoding with known face encodings
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            
            # Find best match
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            
            face_names.append(name)
        
        # Draw rectangles and names on frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Display FPS
        self.num_frames += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1:
            self.fps = self.num_frames / elapsed_time
            self.start_time = time.time()
            self.num_frames = 0
        cv2.putText(frame, f"FPS: {int(self.fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        # Publish output
        output_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        self.image_pub.publish(output_msg)
        #cv2.imshow("Face Recognition", frame)


def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode() # MODIFY NAME
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
