@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read_write> C: array<f32>;

@group(0) @binding(3) var<uniform> dims: vec3<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
	let row = gid.y;
	let col = gid.x;

	let M = dims.x;
	let K = dims.y;
	let N = dims.z;

	if (row >= M || col >= N) {
		return;
	}

	var acc = 0.0;

	for (var i = 0u; i < K; i = i + 1u) {
		acc = acc + A[row * K + i] * B[i * N + col];
	}

	C[row * N + col] = acc;
}
