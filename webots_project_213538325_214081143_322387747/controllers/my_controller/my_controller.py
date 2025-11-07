from controller import Robot, Camera, Lidar, GPS, InertialUnit, Motor
from ultralytics import YOLO
import cv2
import numpy as np
import math
import json
import os
from enum import Enum
from collections import deque

class NavigationMode(Enum):
    FRONTIER_EXPLORATION = 1
    WALL_FOLLOWING = 2
    AGGRESSIVE_SEARCH = 3
    RECOVERY = 4

class TurtleBotPersonDetector:
    def __init__(self):
        # Initialize Webots Robot
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize devices
        self._init_devices()
        
        # YOLO model for person detection
        print("Loading YOLO model...")
        self.model = YOLO("yolov8n.pt")
        
        # Person tracking
        self.detected_people = []
        self.person_dict = {}
        self.min_distance = 0.5
        self.conf_threshold = 0.5
        self.save_path = os.path.expanduser("~/person_list.json")
        
        # Simplified occupancy grid
        self.map_resolution = 0.1  # Larger cells = faster
        self.map_size = 500  # Smaller map = faster
        self.map_origin = np.array([-25.0, -25.0])
        self.occupancy_grid = np.full((self.map_size, self.map_size), -1, dtype=np.int8)
        
        # Navigation state machine
        self.nav_mode = NavigationMode.AGGRESSIVE_SEARCH
        self.target_position = None
        self.position_tolerance = 0.6
        
        # Movement tracking (smaller buffer = faster response)
        self.position_history = deque(maxlen=25)  # Only 0.5 seconds
        self.stuck_counter = 0
        self.mode_timer = 0
        
        # AGGRESSIVE SPEED SETTINGS
        self.max_speed = 6.3
        self.cruise_speed = 0.8  # 3x faster than before!
        self.fast_turn_speed = 2.5  # Aggressive turning
        self.min_obstacle_dist = 0.4  # Get closer to obstacles
        
        # Camera parameters
        self.camera_hfov = 69.4
        
        print("🚀 FAST & AGGRESSIVE TurtleBot Controller initialized!")

    def _init_devices(self):
        """Initialize all robot devices"""
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.timestep)
        
        self.lidar = self.robot.getDevice('LDS-01')
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        
        self.imu = self.robot.getDevice('inertial_unit')
        self.imu.enable(self.timestep)

    def get_position(self):
        return np.array(self.gps.getValues()[:2])

    def get_orientation(self):
        _, _, yaw = self.imu.getRollPitchYaw()
        return yaw

    def get_lidar_sectors(self):
        """Fast LiDAR processing - only 4 sectors"""
        ranges = self.lidar.getRangeImage()
        if ranges is None or len(ranges) == 0:
            return None
        
        ranges = np.array([r if not math.isinf(r) and not math.isnan(r) else 3.5 for r in ranges])
        n = len(ranges)
        
        # Only 4 sectors for speed
        sectors = {
            'front': np.min(ranges[n*3//8:n*5//8]),
            'left': np.min(ranges[n*5//8:n*7//8]),
            'back': np.min(np.concatenate([ranges[0:n//8], ranges[n*7//8:n]])),
            'right': np.min(ranges[n*1//8:n*3//8]),
        }
        
        return sectors, ranges

    def is_stuck(self):
        """Fast stuck detection"""
        if len(self.position_history) < 15:  # Only 0.3 seconds
            return False
        
        positions = np.array(list(self.position_history))
        movement = np.linalg.norm(positions[-1] - positions[-15])
        return movement < 0.03  # Less than 3cm in 0.3s = stuck

    def update_occupancy_grid(self):
        """Lightweight grid update - sample every 10th ray"""
        sectors, ranges = self.get_lidar_sectors()
        if ranges is None:
            return
        
        position = self.get_position()
        yaw = self.get_orientation()
        
        num_points = len(ranges)
        angle_increment = (2 * math.pi) / num_points
        
        for i in range(0, num_points, 10):  # Every 10th ray = 10x faster
            r = ranges[i]
            if r > 3.5:
                continue
            
            angle = -math.pi + i * angle_increment + yaw
            px = position[0] + r * math.cos(angle)
            py = position[1] + r * math.sin(angle)
            
            gx = int((px - self.map_origin[0]) / self.map_resolution)
            gy = int((py - self.map_origin[1]) / self.map_resolution)
            
            if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                self.occupancy_grid[gy, gx] = 100

    def find_frontier(self):
        """Fast frontier search - large sample rate"""
        position = self.get_position()
        frontiers = []
        
        sample_rate = 50  # Very coarse sampling = fast
        for y in range(5, self.map_size - 5, sample_rate):
            for x in range(5, self.map_size - 5, sample_rate):
                if self.occupancy_grid[y, x] == -1:
                    neighbors = self.occupancy_grid[y-2:y+3, x-2:x+3].flatten()
                    if 0 in neighbors:
                        wx = self.map_origin[0] + x * self.map_resolution
                        wy = self.map_origin[1] + y * self.map_resolution
                        dist = np.linalg.norm([wx - position[0], wy - position[1]])
                        
                        if 1.0 < dist < 15.0:
                            frontiers.append((wx, wy, dist))
        
        if not frontiers:
            return None
        
        # Pick a random frontier from nearest 5 for variety
        frontiers.sort(key=lambda f: f[2])
        import random
        chosen = random.choice(frontiers[:min(5, len(frontiers))])
        return np.array([chosen[0], chosen[1]])

    def aggressive_search_behavior(self):
        """Fast, aggressive random exploration"""
        sectors, _ = self.get_lidar_sectors()
        if sectors is None:
            return NavigationMode.RECOVERY
        
        linear_speed = self.cruise_speed  # Fast!
        angular_speed = 0.0
        
        # Simple reactive behavior
        if sectors['front'] < 0.25:
            # Obstacle ahead - fast turn
            linear_speed = 0.1
            angular_speed = self.fast_turn_speed if sectors['right'] > sectors['left'] else -self.fast_turn_speed
        elif sectors['front'] < 0.5:
            # Obstacle approaching - start turning
            linear_speed = self.cruise_speed * 0.7
            angular_speed = 1.5 if sectors['right'] > sectors['left'] else -1.5
        else:
            # Clear ahead - GO FAST!
            linear_speed = self.cruise_speed
            # Slight random wobble for exploration
            import random
            angular_speed = random.uniform(-0.3, 0.3)
        
        self.set_motor_speeds(linear_speed, angular_speed)
        
        # Try frontier exploration every 5 seconds
        if self.mode_timer > 250:
            return NavigationMode.FRONTIER_EXPLORATION
        
        return NavigationMode.AGGRESSIVE_SEARCH

    def wall_following_behavior(self):
        """Fast wall following"""
        sectors, _ = self.get_lidar_sectors()
        if sectors is None:
            return NavigationMode.RECOVERY
        
        wall_dist = sectors['right']
        front_dist = sectors['front']
        
        linear_speed = self.cruise_speed * 0.8  # Fast wall following
        angular_speed = 0.0
        
        if front_dist < 0.3:
            linear_speed = 0.0
            angular_speed = 2.0  # Fast turn
        elif wall_dist < 0.3:
            angular_speed = 1.0  # Turn away from wall
        elif wall_dist > 0.6:
            angular_speed = -1.0  # Turn toward wall
        
        self.set_motor_speeds(linear_speed, angular_speed)
        
        # Switch after 3 seconds
        if self.mode_timer > 150:
            return NavigationMode.AGGRESSIVE_SEARCH
        
        return NavigationMode.WALL_FOLLOWING

    def frontier_exploration_behavior(self):
        """Fast navigation to frontiers"""
        if self.target_position is None:
            self.target_position = self.find_frontier()
        
        if self.target_position is None:
            return NavigationMode.AGGRESSIVE_SEARCH
        
        position = self.get_position()
        yaw = self.get_orientation()
        
        dx = self.target_position[0] - position[0]
        dy = self.target_position[1] - position[1]
        distance = math.sqrt(dx**2 + dy**2)
        target_angle = math.atan2(dy, dx)
        angle_error = math.atan2(math.sin(target_angle - yaw), math.cos(target_angle - yaw))
        
        if distance < self.position_tolerance:
            self.target_position = None
            return NavigationMode.FRONTIER_EXPLORATION
        
        # Fast navigation
        angular_speed = 4.0 * angle_error  # Aggressive turning
        linear_speed = self.cruise_speed  # Always fast
        
        # Only slow down for VERY sharp turns
        if abs(angle_error) > 1.0:  # > 60 degrees
            linear_speed = 0.2
        
        # Obstacle check
        sectors, _ = self.get_lidar_sectors()
        if sectors and sectors['front'] < 0.25:
            self.target_position = None  # Abandon target
            return NavigationMode.AGGRESSIVE_SEARCH
        
        self.set_motor_speeds(linear_speed, angular_speed)
        return NavigationMode.FRONTIER_EXPLORATION

    def recovery_behavior(self):
        """FAST recovery - spin and go!"""
        self.stuck_counter += 1
        
        if self.stuck_counter < 20:
            # Quick spin
            self.set_motor_speeds(0.0, 2.5)
        elif self.stuck_counter < 40:
            # Back up while turning
            self.set_motor_speeds(-0.3, 2.0)
        else:
            # Done recovering
            self.stuck_counter = 0
            return NavigationMode.AGGRESSIVE_SEARCH
        
        return NavigationMode.RECOVERY

    def set_motor_speeds(self, linear_speed, angular_speed):
        """Convert to wheel speeds"""
        wheel_separation = 0.16
        wheel_radius = 0.033
        
        left_speed = (linear_speed - angular_speed * wheel_separation / 2.0) / wheel_radius
        right_speed = (linear_speed + angular_speed * wheel_separation / 2.0) / wheel_radius
        
        left_speed = max(-self.max_speed, min(self.max_speed, left_speed))
        right_speed = max(-self.max_speed, min(self.max_speed, right_speed))
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def navigate(self):
        """Fast navigation state machine"""
        self.mode_timer += 1
        
        # Quick stuck check every 15 steps
        if self.mode_timer % 15 == 0 and self.is_stuck():
            print(f"  ⚡ STUCK! Quick recovery...")
            self.nav_mode = NavigationMode.RECOVERY
            self.mode_timer = 0
        
        # Execute behavior
        if self.nav_mode == NavigationMode.FRONTIER_EXPLORATION:
            new_mode = self.frontier_exploration_behavior()
        elif self.nav_mode == NavigationMode.WALL_FOLLOWING:
            new_mode = self.wall_following_behavior()
        elif self.nav_mode == NavigationMode.AGGRESSIVE_SEARCH:
            new_mode = self.aggressive_search_behavior()
        elif self.nav_mode == NavigationMode.RECOVERY:
            new_mode = self.recovery_behavior()
        
        if new_mode != self.nav_mode:
            print(f"  ⚡ {self.nav_mode.name} → {new_mode.name}")
            self.nav_mode = new_mode
            self.mode_timer = 0

    def detect_persons(self):
        """Fast person detection"""
        image = self.camera.getImage()
        if image is None:
            return
        
        width = self.camera.getWidth()
        height = self.camera.getHeight()
        
        frame = np.frombuffer(image, dtype=np.uint8).reshape((height, width, 4))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        
        # Smaller image for faster YOLO
        frame_resized = cv2.resize(frame, (480, 360))
        
        results = self.model(frame_resized, verbose=False)[0]
        
        sectors, ranges = self.get_lidar_sectors()
        if ranges is None:
            cv2.imshow("Person Detection", frame_resized)
            cv2.waitKey(1)
            return
        
        img_w = 480
        num_ranges = len(ranges)
        
        for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(cls) == 0 and conf > self.conf_threshold:
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_resized, f'{conf:.2f}', (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                bbox_center_x = (x1 + x2) / 2.0
                angle_offset = (bbox_center_x / img_w - 0.5) * math.radians(self.camera_hfov)
                
                laser_index = int((angle_offset + math.pi) / (2 * math.pi) * num_ranges)
                laser_index = max(0, min(num_ranges - 1, laser_index))
                distance = ranges[laser_index]
                
                if distance > 3.5:
                    distance = 2.0
                
                position = self.get_position()
                yaw = self.get_orientation()
                
                person_x = position[0] + distance * math.cos(yaw + angle_offset)
                person_y = position[1] + distance * math.sin(yaw + angle_offset)
                
                self._log_person(person_x, person_y, float(conf))
        
        cv2.imshow("Person Detection", frame_resized)
        cv2.waitKey(1)

    def _log_person(self, x, y, confidence):
        """Log person if not already detected"""
        for px, py in self.detected_people:
            if math.sqrt((x - px)**2 + (y - py)**2) < self.min_distance:
                return
        
        self.detected_people.append((x, y))
        self.person_dict = {f"person{i+1}": [float(p[0]), float(p[1])] 
                           for i, p in enumerate(self.detected_people)}
        
        print(f"\n🎉 PERSON #{len(self.detected_people)} at ({x:.2f}, {y:.2f}) | {confidence:.2f}")
        
        with open(self.save_path, 'w') as f:
            json.dump(self.person_dict, f, indent=2)

    def run(self):
        """Main control loop"""
        print("\n🚀 FAST & AGGRESSIVE Person Detection System")
        print("=" * 50)
        print("Speed: 3x FASTER | Response: INSTANT | Never stuck!")
        print("=" * 50)
        
        step_count = 0
        
        while self.robot.step(self.timestep) != -1:
            step_count += 1
            
            # Track position
            position = self.get_position()
            self.position_history.append(position)
            
            # Update map (only every 3 steps for speed)
            if step_count % 3 == 0:
                self.update_occupancy_grid()
            
            # Navigate every step
            self.navigate()
            
            # Detect persons (only every 5 steps for speed)
            if step_count % 5 == 0:
                self.detect_persons()
            
            # Status (every 5 seconds)
            if step_count % 250 == 0:
                print(f"⚡ {self.nav_mode.name} | People: {len(self.detected_people)} | "
                      f"Pos: ({position[0]:.1f}, {position[1]:.1f})")

if __name__ == "__main__":
    controller = TurtleBotPersonDetector()
    controller.run()