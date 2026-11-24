shader_type spatial;
render_mode cull_disabled;

uniform sampler2D lookup : repeat_disable, filter_linear_mipmap;
uniform sampler2D colour0 : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D colour1 : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D colour2 : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform sampler2D colour3 : source_color, repeat_enable, filter_linear_mipmap_anisotropic;
uniform vec4 transforms[4]; // xy scale, zw offset
uniform int uv_sets[4];
uniform bool double_sided;
varying vec2 third_uv;

void vertex() {
	// Godot imports glTF TEXCOORD_2 in the first two components of CUSTOM0.
	third_uv = CUSTOM0.xy;
}

vec2 layer_uv(int layer, vec2 first, vec2 second, vec2 third) {
	vec2 uv = first;
	if (uv_sets[layer] == 1) {
		uv = second;
	} else if (uv_sets[layer] == 2) {
		uv = third;
	}
	return uv * transforms[layer].xy + transforms[layer].zw;
}

void fragment() {
	if (!double_sided && !FRONT_FACING) {
		discard;
	}
	vec3 weights = texture(lookup, UV).rgb;
	float remainder = max(0.0, 1.0 - weights.r - weights.g - weights.b);
	ALBEDO = weights.r * texture(colour0, layer_uv(0, UV, UV2, third_uv)).rgb
		+ weights.g * texture(colour1, layer_uv(1, UV, UV2, third_uv)).rgb
		+ weights.b * texture(colour2, layer_uv(2, UV, UV2, third_uv)).rgb
		+ remainder * texture(colour3, layer_uv(3, UV, UV2, third_uv)).rgb;
}
