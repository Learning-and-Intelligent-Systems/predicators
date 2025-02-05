import trimesh
import subprocess
import os
import random


def add_comment_to_obj(file_path, comment):
    # Read the original content of the file
    with open(file_path, 'r') as file:
        content = file.readlines()

    # Prepare the comment line
    comment_line = f"# {comment}\n"

    # Add the comment at the top of the content
    new_content = [comment_line] + content

    # Write the new content back to the file
    with open(file_path, 'w') as file:
        file.writelines(new_content)

def retrieve_comment_from_obj(file_path):
    # Open the file and read the first line
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()

    # Check if the first line is a comment
    if first_line.startswith('#'):
        # Remove the '#' and split the comment by comma
        comment = first_line[1:].strip()
        values = comment.split(',')

        if len(values) == 2:
            variable1 = values[0].strip()
            variable2 = values[1].strip()
            return float(variable1), float(variable2)
        else:
            raise ValueError("Comment does not contain exactly two comma-separated values")
    else:
        raise ValueError("The first line is not a comment")


def _generate_rings(rings_to_create: int, outer_radius=None, tubular_radius=None, start=0):
    # Default: 0.015, 0.005
    max_large_outer_radius = 0.06
    min_large_outer_radius = 0.037
    max_small_outer_radius = 0.037
    min_small_outer_radius = 0.012

    min_tubular_radius = 0.004
    max_tubular_radius = 0.008
    min_radius_difference = 0.008

    random_geometry = False

    for _ in range(start, rings_to_create):

        if random_geometry:
            outer_radius = None
            tubular_radius = None

        if outer_radius is None or tubular_radius is None:
            random_geometry = True
            while True:

                if random.uniform(0, 1) < 0.5:
                    outer_radius = random.uniform(min_large_outer_radius, max_large_outer_radius)
                else:
                    outer_radius = random.uniform(min_small_outer_radius, max_small_outer_radius)

                tubular_radius = random.uniform(min_tubular_radius, max_tubular_radius)

                if (outer_radius - tubular_radius) < min_radius_difference:
                    continue
                else:
                    break

        radial_segments = 50
        tubular_segments = 50

        print("CREATING RING MESH")
        # Create the torus mesh using trimesh
        torus_mesh = trimesh.creation.torus(major_radius=outer_radius, minor_radius=tubular_radius,
                                            major_sections=radial_segments, minor_sections=tubular_segments)

        # Save the torus mesh as a .obj file
        torus_mesh.export('rings/torus.obj')
        print("DECONSTRUCTING RING CONVEX HULL")
        subprocess.run(
            ["VHACD.exe", "rings/torus.obj", "-h", "256", "-d", "7", "-r", "100000", "-e",
             "0.001", "-v",
             "64"])
        try:
            os.remove(f"rings/ring_{_}.obj")
            os.remove("decomp.mtl")
            os.remove("decomp.stl")
        except Exception:
            pass


        filepath = f"rings/ring_{_}.obj"
        os.rename("decomp.obj", filepath)
        add_comment_to_obj(filepath, f"{outer_radius},{tubular_radius}")

        obj_file_path = filepath
        o, t = retrieve_comment_from_obj(obj_file_path)
        print(f"Outer Radius 1: {o}")
        print(f"Tubular Radius: {t}")

        assert o == outer_radius and t == tubular_radius, f"geometries do not match! {o},{outer_radius}\n{t},{tubular_radius}"

        print("RING OBJECT CREATED!")


_generate_rings(1000, start=0)
