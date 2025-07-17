import pygame
import numpy as np
from collections import deque
import threading
import time
import random
import torch 
from nodivide.config import DEVICE
from nodivide.model import IMUToTrajectoryNet, load_model
from nodivide.dataset import get_mean_and_std

def model_predict(imu_data):
    """将6维IMU数据转换为2维坐标（模拟函数）"""
    # 这里使用简单转换：取前两维数据并缩放
    x = imu_data[0] * 0.1 + imu_data[1] * 0.05
    y = imu_data[2] * 0.1 + imu_data[3] * 0.05
    return np.array([x, y])

class TrajectoryVisualizer:
    def __init__(self):
        # 数据缓冲区
        self.imu_data_buffer = deque(maxlen=10000)  # 存储原始IMU数据
        self.trajectory = deque(maxlen=10000)     # 存储转换后的2D轨迹
        self.model = load_model()
        self.imu_mean, self.imu_std = get_mean_and_std()
        
        # 显示设置
        self.window_size = (1600, 800)
        self.origin = np.array([self.window_size[0]//4, self.window_size[1]//2])
        self.scale = 100  # 像素/单位
        
        # 线程控制
        self.running = True
        self.data_ready = threading.Event()
        
        # 初始化PyGame
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("Real-time Trajectory Visualization")
        self.clock = pygame.time.Clock()
    
    def normalize_imu_data(self, imu_data):
        imu_data = torch.FloatTensor(imu_data)
        if len(imu_data) == 6:
            imu_data = (imu_data - self.imu_mean) / self.imu_std
        return imu_data

    def model_predict(self, imu_data, last_point=np.array([0, 0])):
        # self.model.eval()
        # with torch.no_grad():
        #     input_tensor = self.normalize_imu_data(imu_data).unsqueeze(0).to(DEVICE)
        #     output = self.model(input_tensor)
        #     output = output.cpu().numpy().squeeze()
        output = np.random.uniform(-10, 10, (len(imu_data), 2))
        output_point = output[-1, :2] / 200 + last_point
        return output_point
    
    def data_processing_thread(self):
        while self.running:
            # 模拟200fps数据输入
            time.sleep(0.005)
            imu_data = np.random.rand(6)
            self.imu_data_buffer.append(imu_data)
            
            if len(self.imu_data_buffer) >= 100:
                last_point = self.trajectory[-1] if self.trajectory else np.array([0, 0])
                output_point = self.model_predict(self.imu_data_buffer, last_point)
                self.trajectory.append(output_point)
                self.data_ready.set()  # 通知有新数据
    
    def draw_trajectory(self):
        """绘制轨迹到屏幕"""
        self.screen.fill((0, 0, 0))  # 黑色背景
        
        # 绘制坐标轴
        pygame.draw.line(self.screen, (100, 100, 100), 
                         (0, self.origin[1]), 
                         (self.window_size[0], self.origin[1]), 1)
        pygame.draw.line(self.screen, (100, 100, 100), 
                         (self.origin[0], 0), 
                         (self.origin[0], self.window_size[1]), 1)
        
        # 绘制轨迹
        if len(self.trajectory) > 1:
            points = []
            trajectory_copy = list(self.trajectory)
            for point in trajectory_copy:
                # 转换坐标系：物理坐标 -> 屏幕坐标
                screen_x = int(self.origin[0] + point[0] * self.scale)
                screen_y = int(self.origin[1] - point[1] * self.scale)  # Y轴向上为正
                points.append((screen_x, screen_y))
            
            # 绘制路径线
            if len(points) > 1:
                pygame.draw.lines(self.screen, (0, 255, 0), False, points, 2)
            
            # 绘制最新位置
            if points:  # Check if points list is not empty
                pygame.draw.circle(self.screen, (255, 0, 0), points[-1], 5)
        
        # 显示数据统计
        font = pygame.font.SysFont(None, 24)
        text = font.render(f"Points: {len(self.trajectory)} | FPS: {int(self.clock.get_fps())}", True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        pygame.display.flip()
    
    def run(self):
        # 启动数据处理线程
        processing_thread = threading.Thread(target=self.data_processing_thread)
        processing_thread.daemon = True
        processing_thread.start()
        
        # 主渲染循环
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            # 有新数据时重绘
            if self.data_ready.is_set():
                self.draw_trajectory()
                self.data_ready.clear()
            
            self.clock.tick(60)  # 限制60FPS渲染
        
        pygame.quit()

if __name__ == "__main__":
    visualizer = TrajectoryVisualizer()
    visualizer.run()