# Vulkan scene preview

This optional playground draws local models with Godot 4 and its Vulkan mobile renderer. VIctory itself does not use Godot. The playground is for inspecting decoded geometry, transforms and exported materials independently from the title runtime.

From this directory, run `./run.sh /absolute/path/to/scene.json`. Set `GODOT_BIN` when Godot is outside PATH. Drag with the left mouse button to orbit, use the wheel to zoom, and press Escape to close. The interactive preview caps its frame rate at 60.

The input may be a Z-up OBJ, a glTF/GLB, or a JSON object containing a `models` array. Each array entry takes a `file` path relative to the manifest and optional glTF-space `position`, `rotation` (quaternion xyzw) and `scale` arrays. OBJ input supports vertices and polygon faces with diagnostic colours per group. GLB input retains the exported materials. Scene manifests can specify `yaw_degrees` and `elevation_degrees` for the initial camera. A `camera` object can override the automatic framing using a glTF-space `target` and a positive orthographic `height` in scene units.

Pass an absolute output PNG path as the second argument to benchmark the view. The tool turns off vsync, allows 30 warm-up frames, and times 120 rendered frames. The JSON next to the image records the adapter, viewport GPU timestamps and wall intervals between frame completions. Keep assets, captures and measurements in a private directory outside this repository.
