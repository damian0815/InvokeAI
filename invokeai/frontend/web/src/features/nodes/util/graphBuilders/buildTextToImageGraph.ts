import { RootState } from 'app/store/store';
import {
  CompelInvocation,
  Graph,
  EdgeConnection,
  IterateInvocation,
  LatentsToImageInvocation,
  NoiseInvocation,
  ParamFloatInvocation,
  ParamIntInvocation,
  RandomIntInvocation,
  RangeOfSizeInvocation,
  TextToLatentsInvocation,
} from 'services/api';
import { NonNullableGraph } from 'features/nodes/types/types';
import { addControlNetToLinearGraph } from '../addControlNetToLinearGraph';
import {
  seedWeightsToArray,
  stringToSeedWeightsArray,
} from 'common/util/seedWeightPairs';

const POSITIVE_CONDITIONING = 'positive_conditioning';
const NEGATIVE_CONDITIONING = 'negative_conditioning';
const TEXT_TO_LATENTS = 'text_to_latents';
const LATENTS_TO_IMAGE = 'latents_to_image';
const NOISE = 'noise';
const RANDOM_INT = 'rand_int';
const FIXED_INT = 'int';
const RANGE_OF_SIZE = 'range_of_size';
const ITERATE = 'iterate';

/**
 * Builds the Text to Image tab graph.
 */
export const buildTextToImageGraph = (state: RootState): Graph => {
  const {
    positivePrompt,
    negativePrompt,
    model,
    cfgScale: cfg_scale,
    scheduler,
    steps,
    width,
    height,
    iterations,
    seed,
    shouldRandomizeSeed,
    shouldGenerateVariations,
    seedWeights,
    variationAmount,
  } = state.generation;

  const graph: NonNullableGraph = {
    nodes: {},
    edges: [],
  };

  // Create the conditioning, t2l and l2i nodes
  const positiveConditioningNode: CompelInvocation = {
    id: POSITIVE_CONDITIONING,
    type: 'compel',
    prompt: positivePrompt,
    model,
  };

  const negativeConditioningNode: CompelInvocation = {
    id: NEGATIVE_CONDITIONING,
    type: 'compel',
    prompt: negativePrompt,
    model,
  };

  const textToLatentsNode: TextToLatentsInvocation = {
    id: TEXT_TO_LATENTS,
    type: 't2l',
    cfg_scale,
    model,
    scheduler,
    steps,
  };

  const latentsToImageNode: LatentsToImageInvocation = {
    id: LATENTS_TO_IMAGE,
    type: 'l2i',
    model,
  };

  // Add to the graph
  graph.nodes[POSITIVE_CONDITIONING] = positiveConditioningNode;
  graph.nodes[NEGATIVE_CONDITIONING] = negativeConditioningNode;
  graph.nodes[TEXT_TO_LATENTS] = textToLatentsNode;
  graph.nodes[LATENTS_TO_IMAGE] = latentsToImageNode;

  // Connect them
  graph.edges.push({
    source: { node_id: POSITIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: TEXT_TO_LATENTS,
      field: 'positive_conditioning',
    },
  });

  graph.edges.push({
    source: { node_id: NEGATIVE_CONDITIONING, field: 'conditioning' },
    destination: {
      node_id: TEXT_TO_LATENTS,
      field: 'negative_conditioning',
    },
  });

  graph.edges.push({
    source: { node_id: TEXT_TO_LATENTS, field: 'latents' },
    destination: {
      node_id: LATENTS_TO_IMAGE,
      field: 'latents',
    },
  });

  /**
   * Now we need to handle iterations and random seeds. There are four possible scenarios:
   * - Single iteration, explicit seed
   * - Single iteration, random seed
   * - Multiple iterations, explicit seed
   * - Multiple iterations, random seed
   *
   * They all have different graphs and connections.
   */

  let nodeIdCounter = 0;

  /** Make a seed node, either using a fixed seed or (if seed is undefined) a random number.
   * Return an EdgeConnection representing the output seed. */
  function makeSeedNode(seed: number | undefined): EdgeConnection {
    if (seed === undefined) {
      const randomIntNode: RandomIntInvocation = {
        id: RANDOM_INT + nodeIdCounter++,
      };
      graph.nodes[randomIntNode.id] = randomIntNode;
      return { node_id: randomIntNode.id, field: 'a' };
    } else {
      const intParamNode: ParamIntInvocation = {
        id: FIXED_INT + nodeIdCounter++,
      };
      graph.nodes[intParamNode.id] = intParamNode;
      return { node_id: intParamNode.id, field: 'a' };
    }
  }

  /** Make a noise node, using node `seedSourceId` as a seed source (assuming the seed can be accessed as a field
   * labelled 'a'). Return the NoiseInvocation node id. */
  function makeNoiseNode(
    seedOrSeedSourceNode: EdgeConnection | number
  ): EdgeConnection {
    const noiseNode: NoiseInvocation = {
      id: NOISE + nodeIdCounter++,
      type: 'noise',
      seed:
        typeof seedOrSeedSourceNode === 'number'
          ? seedOrSeedSourceNode
          : undefined,
      width,
      height,
    };
    graph.nodes[noiseNode.id] = noiseNode;
    if (typeof seedOrSeedSourceNode !== 'number') {
      graph.edges.push({
        source: seedOrSeedSourceNode as EdgeConnection,
        destination: {
          node_id: noiseNode.id,
          field: 'seed',
        },
      });
    }
    return { node_id: noiseNode.id, field: 'noise' };
  }

  /** Make an Iterate node that iterates over `size` ints starting from `startSource` and
   * incrementing by 1 each iteration. */
  function makeIterateRangeNode(
    startSource: EdgeConnection,
    size: number
  ): EdgeConnection {
    const rangeOfSizeNode: RangeOfSizeInvocation = {
      id: RANGE_OF_SIZE + nodeIdCounter++,
      size: size,
    };
    const iterateNode: IterateInvocation = {
      id: ITERATE + nodeIdCounter++,
    };
    graph.nodes[rangeOfSizeNode.id] = rangeOfSizeNode;
    graph.nodes[iterateNode.id] = iterateNode;
    // Connect start node to start
    graph.edges.push({
      source: startSource,
      destination: { node_id: RANGE_OF_SIZE, field: 'start' },
    });

    // Connect range of size to iterate
    graph.edges.push({
      source: { node_id: rangeOfSizeNode.id, field: 'collection' },
      destination: {
        node_id: iterateNode.id,
        field: 'collection',
      },
    });
    return { node_id: iterateNode.id, field: 'item' };
  }

  function makeBlendNode(
    latentsA: EdgeConnection,
    latentsB: EdgeConnection,
    weight: number
  ): EdgeConnection {
    const blendNode: BlendLatentsInvocation = {
      id: BLEND + nodeIdCounter++,
      weight: weight,
    };
    graph.nodes[blendNode.id] = blendNode;
    graph.edges.push({
      source: latentsA,
      destination: {
        node_id: blendNode.id,
        field: 'latents_a',
      },
    });
    graph.edges.push({
      source: latentsB,
      destination: {
        node_id: blendNode.id,
        field: 'latents_b',
      },
    });
    return { node_id: blendNode.id, field: '' };
  }

  let initialNoiseSource = undefined;

  if (iterations <= 1) {
    // Single iteration
    const seedSource = makeSeedNode(shouldRandomizeSeed ? undefined : seed);
    initialNoiseSource = makeNoiseNode(seedSource);
  } else {
    // Multiple iterations
    const startSeedSource = makeSeedNode(
      shouldRandomizeSeed ? undefined : seed
    );
    const iteratingSeedSource = makeIterateRangeNode(
      startSeedSource,
      iterations
    );
    initialNoiseSource = makeNoiseNode(iteratingSeedSource);
  }

  let noiseSource = initialNoiseSource;
  if (shouldGenerateVariations) {
    const parsedSeedWeights = stringToSeedWeightsArray(seedWeights);
    parsedSeedWeights.forEach((seedWeightPair) => {
      const seed = seedWeightPair[0];
      const weight: number = seedWeightPair[1];
      const nextNoiseSource = makeNoiseNode(seed);
      noiseSource = makeBlendNode(noiseSource, nextNoiseSource, weight);
    });
  }

  const finalNoiseSource = noiseSource;

  // Connect noise to t2l
  graph.edges.push({
    source: finalNoiseSource,
    destination: {
      node_id: TEXT_TO_LATENTS,
      field: 'noise',
    },
  });

  addControlNetToLinearGraph(graph, TEXT_TO_LATENTS, state);

  return graph;
};
