try:
    from worker_comm import stop_program
except ImportError:
    from irobot_edu_sdk.utils import stop_program

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note
import time

name = "CapstoneRobot1"

robot = Create3(Bluetooth(name))
sensor_data = [0] * 7
@event(robot.when_bumped, [True, False])
async def bumped(robot):
    print('Left bump sensor hit')
    
@event(robot.when_bumped, [False, True])
async def bumped(robot):
    print('Right bump sensor hit')
    print(sensor_data)
@event(robot.when_play)
async def play(robot):
    while True:
        sensors = (await robot.get_ir_proximity()).sensors
        sensor_data[0] = sensors[0]
        sensor_data[1] = sensors[1]
        sensor_data[2] = sensors[2]
        sensor_data[3] = sensors[3]
        sensor_data[4] = sensors[4]
        sensor_data[5] = sensors[5]
        # print("Ir sensor input:", sensors)
        await hand_over()
        
@event(robot.when_play)
async def play(robot):
    for i in range(20, 1):
        await robot.play_note(i*80, 0.2)
    await robot.stop_sound()
    for i in range(20):
        await robot.play_note(200, 0.1)
        time.sleep(1)
    print("Connected to robot with name", name)
    print("Try using the bumps sensors")
    while True:
        #sensors = (await robot.get_ir_proximity()).sensors
        # print("Ir sensor input:", sensors)
        await hand_over()
robot.play()

