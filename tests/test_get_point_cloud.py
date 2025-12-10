import pyrealsense2 as rs
import numpy as np
import time 
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2
from sensor_msgs.msg import PointField

print("demarrage")
start_t = time.time()
pc = rs.pointcloud()
# points = rs.points()
pipe = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 6)

pipe.start(config)

decimate = rs.decimation_filter(8)
align = rs.align(rs.stream.color)
print(f"temps d'initialisation {time.time() - start_t} seconde")

start = time.time()
#end = time.time()
i = 0
print("debut de la boucle")
while (True):
	frames = pipe.wait_for_frames()
	
	aligned_frames = align.process(frames)
	aligned_depth = aligned_frames.get_depth_frame()
	aligned_colors = aligned_frames.get_color_frame()
	
	
	points = pc.calculate(aligned_depth)
	# decimated = decimate.process(aligned_frames).as_frameset()
	# pc.map_to(aligned_colors)
	
	verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
	# tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 3)

	'''
	color_image = np.asanyarray(aligned_colors.get_data())
	h, w, _ = color_image.shape
	colors = []
	for uv in tex_coords:
		u = min(max(int(uv[0] * w), 0), w-1)
		v = min(max(int(uv[1] * h), 0), h-1)
		colors.append(color_image[v,u])
	colors = np.array(colors, dtype=np.uint8)
'''

	color_image = np.asanyarray(aligned_colors.get_data())
	color_image_flat = color_image.reshape(-1,3)
	r = color_image_flat[:,0].astype(np.uint32)
	g = color_image_flat[:,1].astype(np.uint32)
	b = color_image_flat[:,2].astype(np.uint32)
	rgb = np.left_shift(r, 16) | np.left_shift(g, 8) | b
	rgb.dtype = np.float32

	'''
	points_with_colors = np.hstack([verts, colors.astype(np.float32)])
	print(points_with_colors)
	'''
	
	dtype_cloud = [('x', np.float32), ('y', np.float32), ('z', np.float32), ('rgb', np.float32)]
	cloud_data = np.zeros(verts.shape[0], dtype=dtype_cloud)
	cloud_data['x'] = verts[:, 0]
	cloud_data['y'] = verts[:, 1]
	cloud_data['z'] = verts[:, 2]
	cloud_data['rgb'] = rgb
		
	print(cloud_data)
	
	header = Header()
	header.frame_id = "camera_link"  # Remplace par le nom de ta frame TF
	# header.stamp = ... (idéalement, utilise l'horloge ROS ou le timestamp realsense)

	point_cloud_msg = point_cloud2.create_cloud(
		 header,
		 fields=[
		     PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
		     PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
		     PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
		     PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
		 ],
		 points=cloud_data
	)
	
	#print(colors)
	# for j in colors:
	# 	print(j)
	# valid = verts[:,2] > 0
	# verts_filtered = verts[valid]
	# tex_coords_filtered = tex_coords[valid]

	# print(tex_coords)
	# print(np.sum(valid))

	# depth_raw = frames.get_depth_frame()

	# start_t = time.time()
	# print(f"temps de decimation magnitude 8 pour du 424x240 {time.time() - start_t} seconde")
	# depth = decimated.get_depth_frame()

	# print(f"before : {depth_raw.get_width()}x{depth_raw.get_height()}")
	# print(f"after : {depth.get_width()}x{depth.get_height()}")

	# aligner les frames (meme si pas besoin)
	# decimated = decimate.process(aligned_frames).as_frameset()

	# recuperer les 2 frames, profondeur et couleur
		
	# calculer le nuage de points et ajouter les couleurs

	# Récupère les points sous forme (N, 3)

	# print(verts)
	# # rien que ca ca permet de div par 3 a peu pres le nombre de points

	# print(verts.size/valid.size)
	# print(valid.size)

	# 	# Coordonnées texture (pour extraire les couleurs)
	# texcoords = np.asanyarray(points.get_texture_coordinates())
	# texcoords = texcoords.view(np.float32).reshape(-1, 2)

	# 	# Image couleur
	# color_image = np.asanyarray(aligned_colors.get_data())

	# 	# Couleurs associées aux points
	# h, w, _ = color_image.shape
	# colors = []

	# for u, v in texcoords:
	# 	x = min(max(int(u * w), 0), w-1)
	# 	y = min(max(int(v * h), 0), h-1)
	# 	colors.append(color_image[y, x])

	# 	colors = np.array(colors)
	# print(colors)

	# print(f"===Frame {i} id: {depth.get_frame_number()}===")
	# print(f"TimeStamp : {depth.get_timestamp()}")
	# pointcloud=pc.calculate(depth)
	# print(pointcloud)
	# end = time.time()
	i+=1

print(f"temps total pour 6 frames : {time.time()-start}")
pipe.stop()

# faudra que je set un timer 1/6 self.timer_callback

'''
verts = np.asanyarray(points.get_vertices())
xyz = verts.view(np.float32).reshape(-1, 3)

# Build ROS2 message
msg = self.numpy_to_pointcloud2(xyz, frame_id="camera_link")

# Publish
self.pub.publish(msg)
'''
