import pybullet as p
import pybullet_data
import time

# PyBullet simülasyonunu başlat
p.connect(p.GUI)

# PyBullet örnek veri dizinini ekle
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Yerçekimini ayarla
p.setGravity(0, 0, -9.8)

# Zemin ekle
plane_id = p.loadURDF("plane.urdf")

# URDF dosyasını yükle (güncel URDF dosyanızın doğru yolunu yazın)
robot_id = p.loadURDF("Stationary2.urdf", useFixedBase=True)

# Simülasyon adım süresi
timestep = 1. / 240.
p.setTimeStep(timestep)

# Tüm eklemler için slider oluştur
num_joints = p.getNumJoints(robot_id)
slider_ids = []  # Slider ID'lerini saklamak için
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)
    joint_name = joint_info[1].decode('utf-8')
    joint_type = joint_info[2]

    # Eklemin tipini kontrol et: Revolute veya Continuous
    if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC or joint_type == 0:
        lower_limit = joint_info[8]
        upper_limit = joint_info[9]

        # Continuous eklemler için limitleri geniş tutalım
        if joint_type == p.JOINT_REVOLUTE:
            lower_limit = joint_info[0]  

            upper_limit = 3.14  # +180 derece

        # Slider oluştur
        slider_id = p.addUserDebugParameter(
            paramName=joint_name,
            rangeMin=lower_limit,
            rangeMax=upper_limit,
            startValue=0
        )
        slider_ids.append((slider_id, joint_index))

# Simülasyonu çalıştır
while True:
    # Slider değerlerini oku ve eklemleri kontrol et
    for slider_id, joint_index in slider_ids:
        # Slider'dan kullanıcı tarafından ayarlanmış değeri al
        target_position = p.readUserDebugParameter(slider_id)

        # Eklem pozisyonunu kontrol et
        p.setJointMotorControl2(
            bodyIndex=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position
        )

    # Simülasyonu ilerlet
    p.stepSimulation()
    time.sleep(timestep)
