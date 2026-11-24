extends SceneTree
## Inspect local Z-up triangle geometry with a Vulkan camera, without booting the title.
## godot --rendering-method mobile --rendering-driver vulkan --script scene.gd -- mesh.obj [frame.png]

var model_count := 0
var scene_bounds: Array[Vector3] = []
var view_height := 1.0
const GENERATE_TANGENTS := 8  # GLTFDocument import flag for normal-mapped surfaces.
var camera := Camera3D.new()
var focus := Vector3.ZERO
var radius := 1.0
var yaw := 0.65
var elevation := 0.58
var triangles := 0
var frame_count := 0
var output := ""
var gpu_ms: Array[float] = []
var wall_ms: Array[float] = []
var last_us := 0
var source: String
var benchmark_done := false
var framing: Dictionary = {}

func _initialize() -> void:
	call_deferred("setup")

func setup() -> void:
	var args := OS.get_cmdline_user_args()
	if args.is_empty() or args.size() > 2:
		return fail("Expected mesh.obj and an optional output PNG")
	source = args[0]
	if args.size() == 2:
		output = args[1]
		if output.get_extension().to_lower() != "png":
			return fail("The output image must have a .png extension")
	if RenderingServer.get_current_rendering_driver_name().to_lower() != "vulkan":
		return fail("A Vulkan rendering driver is required")
	root.title = "VIctory | " + source.get_file()
	root.size = Vector2i(1280, 720)
	var stage := Node3D.new()
	root.add_child(stage)
	if source.get_extension().to_lower() == "json":
		var scene = JSON.parse_string(FileAccess.get_file_as_string(source))
		if not scene is Dictionary or not scene.get("models") is Array:
			return fail("Expected a scene manifest containing a models array")
		framing = scene.get("camera", {})
		yaw = deg_to_rad(float(scene.get("yaw_degrees", 37)))
		elevation = deg_to_rad(float(scene.get("elevation_degrees", 28)))
		for entry in scene.models:
			var model := load_model(source.get_base_dir().path_join(entry.file))
			if model == null:
				return fail("Cannot read " + str(entry.file))
			stage.add_child(model)
			model.position = vector(entry.get("position", [0, 0, 0]))
			var q: Array = entry.get("rotation", [0, 0, 0, 1])
			model.quaternion = Quaternion(q[0], q[1], q[2], q[3]).normalized()
			model.scale = vector(entry.get("scale", [1, 1, 1]))
			measure_model(model)
	else:
		var model := load_model(source)
		if model == null:
			return fail("Cannot read " + source)
		stage.add_child(model)
		measure_model(model)
	if scene_bounds.is_empty():
		return fail("Scene has no surfaces")
	var box := AABB(scene_bounds[0], Vector3.ZERO)
	for point in scene_bounds:
		box = box.expand(point)
	focus = box.get_center()
	radius = maxf(0.1, box.size.length() * 1.75)
	root.add_child(camera)
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.current = true
	update_camera()
	var projected := Rect2()
	var first := true
	for point in scene_bounds:
		var local := camera.to_local(point)
		var xy := Vector2(local.x, local.y)
		projected = Rect2(xy, Vector2.ZERO) if first else projected.expand(xy)
		first = false
	var center := projected.get_center()
	focus += camera.basis.x * center.x + camera.basis.y * center.y
	view_height = maxf(0.01, maxf(projected.size.y, projected.size.x * 720.0 / 1280.0) * 1.08)
	if framing.has("target"):
		focus = vector(framing.target)
	if framing.has("height"):
		view_height = maxf(0.01, float(framing.height))
	update_camera()
	var environment := Environment.new()
	environment.background_mode = Environment.BG_COLOR
	environment.background_color = Color("#253745")
	environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.ambient_light_color = Color.WHITE
	environment.ambient_light_energy = 0.70
	camera.environment = environment
	var sun := DirectionalLight3D.new()
	root.add_child(sun)
	sun.rotation_degrees = Vector3(-52, 27, 0)
	sun.light_energy = 1.1
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = camera.far
	RenderingServer.viewport_set_measure_render_time(root.get_viewport_rid(), true)
	RenderingServer.frame_post_draw.connect(after_draw)
	root.window_input.connect(handle_input)

# Manifest transforms are in glTF coordinates (Y up), including model origins.
func vector(values: Array) -> Vector3:
	return Vector3(values[0], values[1], values[2])

func load_model(path: String) -> Node3D:
	match path.get_extension().to_lower():
		"glb", "gltf":
			var loader := GLTFDocument.new()
			var state := GLTFState.new()
			if loader.append_from_file(path, state, GENERATE_TANGENTS) != OK:
				return null
			return loader.generate_scene(state) as Node3D
		"obj":
			var mesh := read_obj(path)
			if mesh == null:
				return null
			var instance := MeshInstance3D.new()
			instance.mesh = mesh
			var material := StandardMaterial3D.new()
			material.vertex_color_use_as_albedo = true
			material.cull_mode = BaseMaterial3D.CULL_DISABLED
			material.roughness = 0.91
			instance.material_override = material
			return instance
	return null

func measure_model(model: Node3D) -> void:
	model_count += 1
	var meshes := model.find_children("*", "MeshInstance3D", true, false)
	if model is MeshInstance3D:
		meshes.append(model)
	for node in meshes:
		var mesh := node.mesh as ArrayMesh
		if mesh == null:
			continue
		var bounds := mesh.get_aabb()
		for corner in 8:
			scene_bounds.append(node.to_global(bounds.get_endpoint(corner)))
		for surface in mesh.get_surface_count():
			if mesh.surface_get_primitive_type(surface) == Mesh.PRIMITIVE_TRIANGLES:
				var indices := mesh.surface_get_array_index_len(surface)
				triangles += (indices if indices > 0 else mesh.surface_get_array_len(surface)) / 3

func read_obj(path: String) -> ArrayMesh:
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		return null
	var positions := PackedVector3Array()
	var vertices := PackedVector3Array()
	var normals := PackedVector3Array()
	var colors := PackedColorArray()
	# Distinct colours identify surfaces explicitly grouped by the exporter.
	var palette := [Color("#9fbea7"), Color("#e3ac76"), Color("#91c4de"), Color("#cdb486")]
	var color: Color = palette[0]
	var group_colors := {}
	while file.get_position() < file.get_length():
		var line := file.get_line().split("#", true, 1)[0].strip_edges().replace("\t", " ")
		var fields := line.split(" ", false)
		if fields.is_empty():
			continue
		match fields[0]:
			"v":
				if fields.size() < 4:
					return null
				var position := Vector3(float(fields[1]), float(fields[3]), -float(fields[2]))
				if not position.is_finite():
					return null
				positions.append(position)
			"g", "o", "usemtl":
				if not group_colors.has(line):
					group_colors[line] = palette[group_colors.size() % palette.size()]
				color = group_colors[line]
				if line.ends_with("_road_0x2000"):
					color = Color("#dea55a")
				elif line.ends_with("_other"):
					color = Color("#86bda4")
			"f":
				if fields.size() < 4:
					return null
				var face := PackedInt32Array()
				for field in fields.slice(1):
					var index := int(field.get_slice("/", 0))
					index = index - 1 if index > 0 else positions.size() + index
					if index < 0 or index >= positions.size():
						return null
					face.append(index)
				for i in range(1, face.size() - 1):
					var a := positions[face[0]]
					var b := positions[face[i]]
					var c := positions[face[i + 1]]
					var normal := (b - a).cross(c - a).normalized()
					if normal == Vector3.ZERO:
						continue
					# Navigation polygons can have either winding.
					if normal.y < 0:
						normal = -normal
					vertices.append_array(PackedVector3Array([a, b, c]))
					for j in 3:
						normals.append(normal)
						colors.append(color)
	if vertices.is_empty():
		return null
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_NORMAL] = normals
	arrays[Mesh.ARRAY_COLOR] = colors
	var result := ArrayMesh.new()
	result.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return result

func update_camera() -> void:
	var direction := Vector3(sin(yaw) * cos(elevation), sin(elevation), cos(yaw) * cos(elevation))
	camera.position = focus + direction * radius
	camera.near = maxf(0.005, radius / 2000.0)
	camera.far = radius * 4.0
	camera.size = view_height
	camera.look_at(focus)

func handle_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and event.button_mask & MOUSE_BUTTON_MASK_LEFT:
		yaw -= event.relative.x * 0.008
		elevation = clampf(elevation + event.relative.y * 0.008, -1.5, 1.5)
		update_camera()
	elif event is InputEventMouseButton and event.pressed:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			view_height = maxf(0.01, view_height / 1.18)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			view_height *= 1.18
		update_camera()
	elif event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		quit()

func after_draw() -> void:
	if output.is_empty() or benchmark_done:
		return
	frame_count += 1
	var now := Time.get_ticks_usec()
	if frame_count > 30:
		var elapsed := RenderingServer.viewport_get_measured_render_time_gpu(root.get_viewport_rid())
		if elapsed > 0:
			gpu_ms.append(elapsed)
		wall_ms.append((now - last_us) / 1000.0)
	last_us = now
	if frame_count == 150:
		benchmark_done = true
		call_deferred("save_result")

func save_result() -> void:
	var image := root.get_texture().get_image()
	if image == null or image.save_png(output) != OK:
		return fail("Cannot save frame to " + output)
	wall_ms.sort()
	gpu_ms.sort()
	var report := {"mesh": source.get_file(), "triangles": triangles, "models": model_count, "driver":
		RenderingServer.get_current_rendering_driver_name(), "adapter": RenderingServer.get_video_adapter_name(),
		"size": [image.get_width(), image.get_height()], "samples": wall_ms.size(),
		"wall_median_ms": wall_ms[wall_ms.size() / 2], "wall_p95_ms": wall_ms[ceili(wall_ms.size() * 0.95) - 1],
		"gpu_samples": gpu_ms.size()}
	if not gpu_ms.is_empty():
		report.gpu_median_ms = gpu_ms[gpu_ms.size() / 2]
		report.gpu_p95_ms = gpu_ms[ceili(gpu_ms.size() * 0.95) - 1]
	var metrics := FileAccess.open(output.get_basename() + ".json", FileAccess.WRITE)
	if metrics == null:
		return fail("Cannot write metrics for " + output)
	metrics.store_line(JSON.stringify(report, "  "))
	print(JSON.stringify(report))
	quit()

func fail(message: String) -> void:
	printerr(message)
	quit(1)
