try:
    from worker_comm import stop_program
except ImportError:
    from irobot_edu_sdk.utils import stop_program

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note

#backend0 = Bluetooth('')
#backend1 = Bluetooth('ROOT')
name = "CapstoneRobot1"

robot = Create3(Bluetooth(name))

@event(robot.when_play)
async def play(robot):
    for i in range(20, 1):
        await robot.play_note(i*80, 0.2)
    await robot.stop_sound()
    print("Connected to robot with name", name)
    stop_program()
robot.play()

