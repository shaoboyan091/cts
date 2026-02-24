export const description = `
Pipeline creation validation tests for immediate data size mismatches.

Validates that creating a pipeline fails if the shader uses immediate data
larger than the immediateSize specified in the pipeline layout, or larger than
maxImmediateSize if layout is 'auto'.
`;

import { makeTestGroup } from '../../../../common/framework/test_group.js';
import { getGPU } from '../../../../common/util/navigator_gpu.js';
import { assert, range, supportsImmediateData } from '../../../../common/util/util.js';
import { AllFeaturesMaxLimitsGPUTest } from '../../../gpu_test.js';
import * as vtu from '../validation_test_utils.js';

export const g = makeTestGroup(AllFeaturesMaxLimitsGPUTest);

/**
 * Generate shader code for a given stage with the specified immediate data size.
 * If size is 0, the shader has no immediate data.
 */
function makeShaderCode(size: number, stage: 'compute' | 'vertex' | 'fragment'): string {
  if (size === 0) {
    switch (stage) {
      case 'compute':
        return `@compute @workgroup_size(1) fn main_compute() {}`;
      case 'vertex':
        return `@vertex fn main_vertex() -> @builtin(position) vec4<f32> { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }`;
      case 'fragment':
        return `@fragment fn main_fragment() -> @location(0) vec4<f32> { return vec4<f32>(0.0, 1.0, 0.0, 1.0); }`;
    }
  }
  const numFields = size / 4;
  const fields = range(numFields, i => `m${i}: u32`).join(', ');
  const structDecl = `struct Immediates { ${fields} }\nvar<immediate> data: Immediates;`;
  switch (stage) {
    case 'compute':
      return `${structDecl}\nfn use_data() { _ = data.m0; }\n@compute @workgroup_size(1) fn main_compute() { use_data(); }`;
    case 'vertex':
      return `${structDecl}\n@vertex fn main_vertex() -> @builtin(position) vec4<f32> { _ = data.m0; return vec4<f32>(0.0, 0.0, 0.0, 1.0); }`;
    case 'fragment':
      return `${structDecl}\n@fragment fn main_fragment() -> @location(0) vec4<f32> { _ = data.m0; return vec4<f32>(0.0, 1.0, 0.0, 1.0); }`;
  }
}

/**
 * Describes the shader's immediate data size relative to the pipeline layout's immediateSize.
 * 'none' through 'larger' are compared against the fixed layout size (kFixedLayoutSize).
 * 'atLimit' and 'exceedLimit' are compared against the device's maxImmediateSize
 * and use 'auto' layout.
 */
type ShaderVsLayoutSize =
  | 'none'
  | 'smaller'
  | 'equal'
  | 'larger_small'
  | 'larger'
  | 'atLimit' // shader size == maxImmediateSize (auto layout)
  | 'exceedLimit'; // shader size > maxImmediateSize (auto layout)

/** Fixed layout size used for numeric (non-limit) test cases. */
const kFixedLayoutSize = 16;

/** Resolve a shader-vs-layout size category to an actual byte size. */
function resolveShaderSize(category: ShaderVsLayoutSize, maxImmediateSize: number): number {
  switch (category) {
    case 'none':
      return 0;
    case 'smaller':
      return kFixedLayoutSize - 4;
    case 'equal':
      return kFixedLayoutSize;
    case 'larger_small':
      return kFixedLayoutSize + 4;
    case 'larger':
      return kFixedLayoutSize * 2;
    // The following compare against the device's maxImmediateSize, not the fixed layout size.
    case 'atLimit':
      return maxImmediateSize;
    case 'exceedLimit':
      return maxImmediateSize + 4;
  }
}

/** Whether a shader-vs-layout category requires an 'auto' pipeline layout. */
function usesAutoLayout(category: ShaderVsLayoutSize): boolean {
  return category === 'atLimit' || category === 'exceedLimit';
}

g.test('pipeline_creation_immediate_size_mismatch,compute')
  .desc(
    `
    Validate that creating a compute pipeline fails if the shader uses
    immediate data larger than the immediateSize specified in the pipeline layout,
    or larger than maxImmediateSize if layout is 'auto'.
    Also validates that using less or equal size is allowed.
    `
  )
  .params(u =>
    u.combine('isAsync', [true, false]).combine('shaderSizeVsLayout', [
      'smaller',
      'equal',
      'larger_small',
      'larger',
      'atLimit', // shader size == maxImmediateSize (compared to device limit)
      'exceedLimit', // shader size > maxImmediateSize (compared to device limit)
    ] as const)
  )
  .fn(t => {
    t.skipIf(!supportsImmediateData(getGPU(t.rec)), 'Immediate data not supported');

    const { isAsync, shaderSizeVsLayout } = t.params;
    const maxImmediateSize = t.device.limits.maxImmediateSize;
    assert(maxImmediateSize !== undefined);

    const resolvedSize = resolveShaderSize(shaderSizeVsLayout, maxImmediateSize);

    // Validate non-exceeding sizes fit within device limits.
    if (shaderSizeVsLayout !== 'exceedLimit') {
      assert(
        resolvedSize <= maxImmediateSize,
        `shader size (${resolvedSize}) must be <= maxImmediateSize (${maxImmediateSize})`
      );
    }

    // Build pipeline layout.
    let layout: GPUPipelineLayout | 'auto';
    let validSize: number;

    if (usesAutoLayout(shaderSizeVsLayout)) {
      layout = 'auto';
      validSize = maxImmediateSize;
    } else {
      layout = t.device.createPipelineLayout({
        bindGroupLayouts: [],
        immediateSize: kFixedLayoutSize,
      });
      validSize = kFixedLayoutSize;
    }

    const shouldError = resolvedSize > validSize;
    const code = makeShaderCode(resolvedSize, 'compute');

    vtu.doCreateComputePipelineTest(t, isAsync, !shouldError, {
      layout,
      compute: { module: t.device.createShaderModule({ code }) },
    });
  });

g.test('pipeline_creation_immediate_size_mismatch,render')
  .desc(
    `
    Validate that creating a render pipeline fails if the shader uses
    immediate data larger than the immediateSize specified in the pipeline layout,
    or larger than maxImmediateSize if layout is 'auto'.
    Tests vertex and fragment stages independently.
    `
  )
  .params(u =>
    u.combine('isAsync', [true, false]).combineWithParams([
      // Shader immediate size vs explicit layout size (kFixedLayoutSize)
      { vertexShaderVsLayout: 'equal', fragmentShaderVsLayout: 'equal' },
      { vertexShaderVsLayout: 'smaller', fragmentShaderVsLayout: 'smaller' },
      { vertexShaderVsLayout: 'larger_small', fragmentShaderVsLayout: 'larger_small' },
      { vertexShaderVsLayout: 'larger', fragmentShaderVsLayout: 'larger' },
      // Shader immediate size vs device maxImmediateSize (auto layout) — vertex only
      { vertexShaderVsLayout: 'atLimit', fragmentShaderVsLayout: 'none' },
      { vertexShaderVsLayout: 'exceedLimit', fragmentShaderVsLayout: 'none' },
      // Shader immediate size vs device maxImmediateSize (auto layout) — fragment only
      { vertexShaderVsLayout: 'none', fragmentShaderVsLayout: 'atLimit' },
      { vertexShaderVsLayout: 'none', fragmentShaderVsLayout: 'exceedLimit' },
      // Shader immediate size vs device maxImmediateSize (auto layout) — both stages
      { vertexShaderVsLayout: 'atLimit', fragmentShaderVsLayout: 'atLimit' },
    ] as const)
  )
  .fn(t => {
    t.skipIf(!supportsImmediateData(getGPU(t.rec)), 'Immediate data not supported');

    const { isAsync, vertexShaderVsLayout, fragmentShaderVsLayout } = t.params;
    const maxImmediateSize = t.device.limits.maxImmediateSize;
    assert(maxImmediateSize !== undefined);

    const resolvedVertexSize = resolveShaderSize(vertexShaderVsLayout, maxImmediateSize);
    const resolvedFragmentSize = resolveShaderSize(fragmentShaderVsLayout, maxImmediateSize);

    // Validate non-exceeding sizes fit within device limits.
    if (vertexShaderVsLayout !== 'exceedLimit') {
      assert(
        resolvedVertexSize <= maxImmediateSize,
        `vertex shader size (${resolvedVertexSize}) must be <= maxImmediateSize (${maxImmediateSize})`
      );
    }
    if (fragmentShaderVsLayout !== 'exceedLimit') {
      assert(
        resolvedFragmentSize <= maxImmediateSize,
        `fragment shader size (${resolvedFragmentSize}) must be <= maxImmediateSize (${maxImmediateSize})`
      );
    }

    // Build pipeline layout.
    let layout: GPUPipelineLayout | 'auto';
    let validSize: number;

    if (usesAutoLayout(vertexShaderVsLayout) || usesAutoLayout(fragmentShaderVsLayout)) {
      layout = 'auto';
      validSize = maxImmediateSize;
    } else {
      layout = t.device.createPipelineLayout({
        bindGroupLayouts: [],
        immediateSize: kFixedLayoutSize,
      });
      validSize = kFixedLayoutSize;
    }

    const vertexExceedsLimit = resolvedVertexSize > validSize;
    const fragmentExceedsLimit = resolvedFragmentSize > validSize;
    const shouldError = vertexExceedsLimit || fragmentExceedsLimit;

    const vertexCode = makeShaderCode(resolvedVertexSize, 'vertex');
    const fragmentCode = makeShaderCode(resolvedFragmentSize, 'fragment');

    vtu.doCreateRenderPipelineTest(t, isAsync, !shouldError, {
      layout,
      vertex: { module: t.device.createShaderModule({ code: vertexCode }) },
      fragment: {
        module: t.device.createShaderModule({ code: fragmentCode }),
        targets: [{ format: 'rgba8unorm' }],
      },
    });
  });
