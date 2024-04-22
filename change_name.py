try:
    from worker_comm import stop_program
except ImportError:
    from irobot_edu_sdk.utils import stop_program

from irobot_edu_sdk.backend.bluetooth import Bluetooth
from irobot_edu_sdk.robots import event, hand_over, Color, Robot, Root, Create3
from irobot_edu_sdk.music import Note

#backend0 = Bluetooth('')
#backend1 = Bluetooth('ROOT')
robot = Create3(Bluetooth())
newName = "CapstoneRobot1"
@event(robot.when_play)
async def play(robot):
    old_name = await robot.get_name()
    await robot.set_name(newName)
    print('Renamed: ', old_name, '->', newName)
    stop_program()
robot.play()

