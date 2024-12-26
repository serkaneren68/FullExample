import pybullet as p
import pybullet_data
import time
import math

# PyBullet bağlantısı
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.8)

basic_gun_pos = [0,0,0]
basic_gun = p.loadURDF("basic_gun.urdf",basic_gun_pos[0],basic_gun_pos[1],basic_gun_pos[2])

num_joints = p.getNumJoints(basic_gun)
print(num_joints)

for joint in range(num_joints):
    print(p.getJointInfo(basic_gun, joint))

# p.setJointMotorControl2(
#         bodyUniqueId=basic_gun,
#         jointIndex=2,
#         controlMode=p.POSITION_CONTROL,
#         targetPosition=0.5,
#         force=100
#     )
p.setJointMotorControl2(
        bodyUniqueId=basic_gun,
        jointIndex=2,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=0.01,
        force=100
    )
p.setJointMotorControl2(
        bodyUniqueId=basic_gun,
        jointIndex=2,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=-0.01,
        force=100
    )

 