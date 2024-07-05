import pybullet as p
import pybullet_data
import time
import os
from PIL import Image

# Connect to PyBullet
os.environ["DISPLAY"] = ":99"
physics_client_id = p.connect(p.GUI) 
# physics_client_id = p.connect(p.DIRECT)#, options="--opengl2") 

# Set additional search path for pybullet_data
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the plane
plane_id = p.loadURDF("plane.urdf")

# Set gravity
p.setGravity(0, 0, -9.81)

# Create visual shape with transparency
half_extents = [0.5, 0.5, 0.5]
visual_id = p.createVisualShape(p.GEOM_BOX,
                                halfExtents=half_extents,
                                rgbaColor=(0.7, 0.7, 0.7, 0.3),  # 50% transparency
                                physicsClientId=physics_client_id)

# Create collision shape
collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)

# Create multi-body with the visual and collision shape
pose = (0, 0, 1)
orientation = p.getQuaternionFromEuler([0, 0, 0])
box_id = p.createMultiBody(baseMass=1,
                           baseCollisionShapeIndex=collision_id,
                           baseVisualShapeIndex=visual_id,
                           basePosition=pose,
                           baseOrientation=orientation,
                           physicsClientId=physics_client_id)


# Directory to save images
image_dir = "images"
os.makedirs(image_dir, exist_ok=True)

# Set the camera
p.resetDebugVisualizerCamera(cameraDistance=2,
                             cameraYaw=0,
                             cameraPitch=-40,
                             cameraTargetPosition=[0, 0, 0])



# Run the simulation and save images
for i in range(100):
    p.stepSimulation()

    # Capture image
    width, height, rgb_img, depth_img, seg_img = p.getCameraImage(640, 480, 
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Save the image
    image_path = os.path.join(image_dir, f"frame_{i:04d}.png")
    img = Image.fromarray(rgb_img)
    img.save(image_path)

# Disconnect from PyBullet
p.disconnect()

