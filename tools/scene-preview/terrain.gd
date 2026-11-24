extends RefCounted
## Inspect the base-colour blend of exported four-layer terrain materials.
const SHADER := preload("terrain.gdshader")
const UV_FLAGS := [Mesh.ARRAY_FORMAT_TEX_UV, Mesh.ARRAY_FORMAT_TEX_UV2, Mesh.ARRAY_FORMAT_CUSTOM0]

static func texture_file(entry: Variant, directory: String, cache: Dictionary) -> Texture2D:
	if not entry is Dictionary or not entry.get("png") is String:
		return null
	var relative: String = entry.png
	var path := directory.path_join(relative).simplify_path()
	if relative.is_absolute_path() or not path.begins_with(directory.trim_suffix("/") + "/"):
		return null
	if cache.has(path):
		return cache[path]
	var image := Image.new()
	if image.load(path) != OK or image.generate_mipmaps() != OK:
		return null
	var texture := ImageTexture.create_from_image(image)
	cache[path] = texture
	return texture

static func build(material: Dictionary, directory: String, textures: Dictionary) -> ShaderMaterial:
	var extras = material.get("extras")
	if not extras is Dictionary:
		return null
	var terrain = extras.get("multilayer")
	if not terrain is Dictionary or terrain.get("family") != "four_layer" or material.get("alphaMode", "OPAQUE") != "OPAQUE":
		return null
	var layers = terrain.get("layers")
	if not layers is Array or layers.size() != 4:
		return null
	var sets := PackedInt32Array()
	var transforms := PackedVector4Array()
	for layer in layers:
		if not layer is Dictionary or layer.get("uv_source") != "material constants":
			return null
		var uv := ["TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2"].find(layer.get("uv_set"))
		var scale = layer.get("uv_scale")
		var offset = layer.get("uv_offset")
		if scale is float or scale is int:
			scale = [scale, scale]
		if uv < 0 or not scale is Array or scale.size() != 2 or not offset is Array or offset.size() != 2:
			return null
		for value in scale + offset:
			if not (value is float or value is int):
				return null
		var transform := Vector4(scale[0], scale[1], offset[0], offset[1])
		if not transform.is_finite():
			return null
		sets.append(uv)
		transforms.append(transform)
	var weights = terrain.get("weights")
	if not weights is Dictionary:
		return null
	var lookup := texture_file(weights.get("texture"), directory, textures)
	if lookup == null:
		return null
	var result := ShaderMaterial.new()
	result.shader = SHADER
	result.set_shader_parameter("lookup", lookup)
	result.set_shader_parameter("uv_sets", sets)
	result.set_shader_parameter("transforms", transforms)
	result.set_shader_parameter("double_sided", material.get("doubleSided", false))
	for i in layers.size():
		var colour := texture_file(layers[i].get("colour"), directory, textures)
		if colour == null:
			return null
		result.set_shader_parameter("colour%d" % i, colour)
	return result

static func apply(model: Node3D, state: GLTFState, path: String) -> void:
	var descriptions: Array = state.json.get("materials", [])
	var imported := state.get_materials()
	var candidates := {}
	for i in mini(descriptions.size(), imported.size()):
		var extras = descriptions[i].get("extras")
		if not extras is Dictionary:
			continue
		var info = extras.get("multilayer")
		if info is Dictionary and info.get("family") == "four_layer":
			candidates[imported[i]] = descriptions[i]
	if candidates.is_empty():
		return
	var directory := path.get_base_dir()
	while not FileAccess.file_exists(directory.path_join("manifest.jsonl")):
		var parent := directory.get_base_dir()
		if directory == parent or parent.is_empty():
			directory = path.get_base_dir()
			break
		directory = parent
	var replacements := {}
	var textures := {}
	var nodes := model.find_children("*", "MeshInstance3D", true, false)
	if model is MeshInstance3D:
		nodes.append(model)
	for node: MeshInstance3D in nodes:
		var mesh := node.mesh as ArrayMesh
		if mesh == null:
			continue
		for i in mesh.get_surface_count():
			var source := mesh.surface_get_material(i)
			if not candidates.has(source):
				continue
			if not replacements.has(source):
				replacements[source] = build(candidates[source], directory, textures)
			var result: ShaderMaterial = replacements[source]
			if result == null:
				continue
			var needed: int = Mesh.ARRAY_FORMAT_TEX_UV
			for uv in result.get_shader_parameter("uv_sets"):
				needed |= UV_FLAGS[uv]
			if (mesh.surface_get_format(i) & needed) == needed:
				node.set_surface_override_material(i, result)
