import { Box, Divider, Flex, Image } from "@invoke-ai/ui-library";
import { useStore } from '@nanostores/react';
import type { Dimensions } from "@xyflow/react";
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { useAppSelector } from "app/store/storeHooks";
import { selectLastSelectedItem, selectTokenizationDisplayMode } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ImageDTO } from "services/api/types";
import { useImageDTO } from "services/api/endpoints/images";
import { useDebouncedMetadata } from "services/api/hooks/useDebouncedMetadata";

import { fitDimsToContainer } from "./common";
import { TokenizationToolbar } from "./TokenizationToolbar";

// ============================================================================
// Types
// ============================================================================

type TokenMetadata = {
  tokens: string[];
  attentionMapsImageName: string;
};

type AttentionMapCoordinates = {
  x: number;
  y: number;
};

// ============================================================================
// Utilities
// ============================================================================

const extractTokens = (metadata: any, key: 'positive' | 'negative'): string[] | undefined => {
  const allPromptsUnfilteredTokens = JSON.parse(metadata["tokenization"]["value"]);
  if (!allPromptsUnfilteredTokens) {
    return undefined;
  }
  
  let promptUnfilteredTokens = allPromptsUnfilteredTokens[key];
  if (!promptUnfilteredTokens || promptUnfilteredTokens.length === 0) {
    return undefined;
  }
  
  // TODO: support multi-chunk (long prompts) - for now just use the first chunk
  if (Array.isArray(promptUnfilteredTokens[0])) {
    promptUnfilteredTokens = promptUnfilteredTokens[0];
  }
  
  // Drop '<bos>' and '<eos>' tokens if present
  return promptUnfilteredTokens.filter((token: string) => token !== '<bos>' && token !== '<eos>');
};

const getHeatmapColor = (value: number): string => {
  // Heatmap: blue (0) -> purple (0.5) -> red (1) with alpha fade in first 20%
  let r: number; let g: number; let b: number; let alpha: number;

  const alphaFadeInRange = 0.2;
  alpha = value < alphaFadeInRange ? value / alphaFadeInRange : 1;
  
  if (value < 0.5) {
    // Blue to Purple (0 to 0.5)
    const t = value * 2;
    r = Math.round(128 * t);
    g = 0;
    b = 255;
  } else {
    // Purple to Red (0.5 to 1)
    const t = (value - 0.5) * 2;
    r = Math.round(128 + 127 * t);
    g = 0;
    b = Math.round(255 * (1 - t));
  }
  
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
};

// ============================================================================
// Custom Hooks
// ============================================================================

const useAttentionMapData = (imageUrl: string | undefined) => {
  const crossOrigin = useStore($crossOrigin);
  const [imageData, setImageData] = useState<ImageData | null>(null);
  
  useEffect(() => {
    if (!imageUrl) {
      return;
    }

    const img = new window.Image();
    img.crossOrigin = crossOrigin || 'anonymous';
    img.src = imageUrl;
    
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        return;
      }

      ctx.drawImage(img, 0, 0);
      setImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));
    };
  }, [imageUrl, crossOrigin]);
  
  return imageData;
};

const useLuminanceCalculator = (
  attentionMapData: ImageData | null,
  tokenCount: number
) => {
  const calculateLuminance = useCallback((x: number, y: number): number[] => {
    if (!attentionMapData || tokenCount === 0) {
      return [];
    }

    const attentionWidth = attentionMapData.width;
    const attentionHeightPerToken = attentionMapData.height / tokenCount;
    const luminanceValues: number[] = [];

    for (let tokenIdx = 0; tokenIdx < tokenCount; tokenIdx++) {
      const tokenAttentionY = Math.floor(tokenIdx * attentionHeightPerToken + y);
      
      if (x >= 0 && x < attentionWidth && 
          tokenAttentionY >= 0 && tokenAttentionY < attentionMapData.height) {
        
        const pixelIndex = (tokenAttentionY * attentionWidth + x) * 4;
        const r = attentionMapData.data[pixelIndex] ?? 0;
        const g = attentionMapData.data[pixelIndex + 1] ?? 0;
        const b = attentionMapData.data[pixelIndex + 2] ?? 0;
        
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        luminanceValues.push(luminance);
      } else {
        luminanceValues.push(0);
      }
    }
    
    return luminanceValues;
  }, [attentionMapData, tokenCount]);

  return calculateLuminance;
};

// ============================================================================
// Sub-Components
// ============================================================================

const TokenList = memo(({ 
  tokens, 
  luminanceValues, 
  hoveredTokenIdx,
  onTokenHover
}: { 
  tokens: string[];
  luminanceValues: number[];
  hoveredTokenIdx: number | null;
  onTokenHover: (index: number | null) => void;
}) => {
  return (
    <Box mb={2} overflowY="auto">
      <Flex gap={2} flexWrap="wrap">
        {tokens.map((token, index) => {
          const luminance = Math.pow(luminanceValues[index] || 0, 1);
          const isHighlighted = hoveredTokenIdx === index;
          
          return (
            <Box
              key={index}
              px={2}
              py={1}
              borderRadius="base"
              bg={luminance > 0 ? getHeatmapColor(luminance) : "base.800"}
              color="white"
              fontSize="xs"
              maxW="max-content"
              transition="all 0.1s ease"
              onMouseEnter={() => onTokenHover(index)}
              onMouseLeave={() => onTokenHover(null)}
              cursor="pointer"
              border="2px solid"
              borderColor={isHighlighted ? "yellow.400" : "transparent"}
              transform={isHighlighted ? "scale(1.1)" : "scale(1)"}
              zIndex={isHighlighted ? 10 : 1}
            >
              {token}
            </Box>
          );
        })}
      </Flex>
    </Box>
  );
});
TokenList.displayName = 'TokenList';

const AttentionMapOverlay = memo(({ 
  tokenIdx,
  attentionMapImageUrl,
  attentionMapData,
  tokenCount,
  fittedDims
}: {
  tokenIdx: number;
  attentionMapImageUrl: string;
  attentionMapData: ImageData;
  tokenCount: number;
  fittedDims: Dimensions;
}) => {
  const crossOrigin = useStore($crossOrigin);
  
  return (
    <canvas
      ref={(overlayCanvas: HTMLCanvasElement | null) => {
        if (!overlayCanvas) {
          return;
        }
        
        const width = attentionMapData.width;
        const heightPerToken = attentionMapData.height / tokenCount;
        
        overlayCanvas.width = width;
        overlayCanvas.height = heightPerToken;
        
        const ctx = overlayCanvas.getContext('2d');
        if (!ctx) {
          return;
        }
        
        const sourceY = tokenIdx * heightPerToken;
        const img = new window.Image();
        img.crossOrigin = crossOrigin || 'anonymous';
        img.src = attentionMapImageUrl;
        
        img.onload = () => {
          ctx.drawImage(
            img,
            0, sourceY, width, heightPerToken,
            0, 0, width, heightPerToken
          );
          
          // Apply yellow tint for visibility
          const imageData = ctx.getImageData(0, 0, width, heightPerToken);
          const data = imageData.data;
          
          for (let i = 0; i < data.length; i += 4) {
            const luminance = data[i] ?? 0;
            data[i] = Math.min(255, luminance * 1.5);     // R
            data[i + 1] = Math.min(255, luminance * 1.5); // G
            data[i + 2] = 0;                               // B - no blue = yellow
            data[i + 3] = Math.min(255, luminance * 2);   // A
          }
          
          ctx.putImageData(imageData, 0, 0);
        };
      }}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: `${fittedDims.width}px`,
        height: `${fittedDims.height}px`,
        pointerEvents: 'none',
        opacity: 0.8,
        imageRendering: 'pixelated'
      }}
    />
  );
});
AttentionMapOverlay.displayName = 'AttentionMapOverlay';

const AttentionMapCrosshair = memo(({ 
  x, 
  y 
}: { 
  x: number; 
  y: number;
}) => (
  <Box
    position="absolute"
    left={`${x}px`}
    top={`${y}px`}
    w="1px"
    h="1px"
    pointerEvents="none"
    _before={{
      content: '""',
      position: 'absolute',
      left: '-5px',
      top: '0',
      width: '11px',
      height: '1px',
      bg: 'red.500',
    }}
    _after={{
      content: '""',
      position: 'absolute',
      left: '0',
      top: '-5px',
      width: '1px',
      height: '11px',
      bg: 'red.500',
    }}
  />
));
AttentionMapCrosshair.displayName = 'AttentionMapCrosshair';

const AttentionMapColumn = memo(({ 
  columnIndex,
  startTokenIdx,
  tokensInColumn,
  attentionMapImageUrl,
  attentionMapData,
  attentionHeightPerToken,
  hoveredTokenIdx,
  crosshairPos,
  onTokenHover
}: {
  columnIndex: number;
  startTokenIdx: number;
  tokensInColumn: number;
  attentionMapImageUrl: string;
  attentionMapData: ImageData;
  attentionHeightPerToken: number;
  hoveredTokenIdx: number | null;
  crosshairPos: AttentionMapCoordinates | null;
  onTokenHover: (index: number | null) => void;
}) => {
  const crossOrigin = useStore($crossOrigin);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const endTokenIdx = startTokenIdx + tokensInColumn;
  const columnHeight = tokensInColumn * attentionHeightPerToken;

  // Draw the attention map slice once on mount or when data changes
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    canvas.width = attentionMapData.width;
    canvas.height = columnHeight;

    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const img = new window.Image();
    img.crossOrigin = crossOrigin || 'anonymous';
    img.src = attentionMapImageUrl;

    img.onload = () => {
      const sourceY = startTokenIdx * attentionHeightPerToken;
      ctx.drawImage(
        img,
        0, sourceY, attentionMapData.width, columnHeight,
        0, 0, attentionMapData.width, columnHeight
      );
    };
  }, [attentionMapImageUrl, attentionMapData, columnHeight, startTokenIdx, attentionHeightPerToken, crossOrigin]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const y = e.clientY - rect.top;
    const localTokenIdx = Math.floor((y / columnHeight) * tokensInColumn);
    const globalTokenIdx = startTokenIdx + localTokenIdx;
    onTokenHover(globalTokenIdx);
  }, [columnHeight, tokensInColumn, startTokenIdx, onTokenHover]);

  const handleMouseLeave = useCallback(() => {
    onTokenHover(null);
  }, [onTokenHover]);

  return (
    <Box position="relative" flexShrink={0}>
      <canvas
        ref={canvasRef}
        style={{
          width: `${attentionMapData.width}px`,
          height: `${columnHeight}px`,
          display: 'block',
          cursor: 'pointer'
        }}
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
      />
      
      {/* Yellow border around hovered token's attention map */}
      {hoveredTokenIdx !== null && hoveredTokenIdx >= startTokenIdx && hoveredTokenIdx < endTokenIdx && (
        <Box
          position="absolute"
          left={0}
          top={`${(hoveredTokenIdx - startTokenIdx) * attentionHeightPerToken}px`}
          width="100%"
          height={`${attentionHeightPerToken}px`}
          border="2px solid"
          borderColor="yellow.400"
          pointerEvents="none"
          boxSizing="border-box"
        />
      )}
      
      {/* Crosshairs for each token */}
      {crosshairPos && Array.from({ length: tokensInColumn }, (_, localTokenIdx) => {
        const yOffset = localTokenIdx * attentionHeightPerToken;
        return (
          <AttentionMapCrosshair
            key={startTokenIdx + localTokenIdx}
            x={crosshairPos.x}
            y={yOffset + crosshairPos.y}
          />
        );
      })}
    </Box>
  );
});
AttentionMapColumn.displayName = 'AttentionMapColumn';

const AttentionMapGrid = memo(({ 
  attentionMapImageUrl,
  attentionMapData,
  tokenCount,
  hoveredTokenIdx,
  crosshairPos,
  maxHeight,
  onTokenHover
}: {
  attentionMapImageUrl: string;
  attentionMapData: ImageData;
  tokenCount: number;
  hoveredTokenIdx: number | null;
  crosshairPos: AttentionMapCoordinates | null;
  maxHeight: number;
  onTokenHover: (index: number | null) => void;
}) => {
  const tokensPerColumn = 8;
  const columnCount = Math.ceil(tokenCount / tokensPerColumn);
  const attentionHeightPerToken = attentionMapData.height / tokenCount;

  return (
    <Flex gap={2} flexWrap="wrap" maxH={maxHeight} overflow="auto">
      {Array.from({ length: columnCount }, (_, colIdx) => {
        const startTokenIdx = colIdx * tokensPerColumn;
        const endTokenIdx = Math.min(startTokenIdx + tokensPerColumn, tokenCount);
        const tokensInColumn = endTokenIdx - startTokenIdx;
        
        return (
          <AttentionMapColumn
            key={colIdx}
            columnIndex={colIdx}
            startTokenIdx={startTokenIdx}
            tokensInColumn={tokensInColumn}
            attentionMapImageUrl={attentionMapImageUrl}
            attentionMapData={attentionMapData}
            attentionHeightPerToken={attentionHeightPerToken}
            hoveredTokenIdx={hoveredTokenIdx}
            crosshairPos={crosshairPos}
            onTokenHover={onTokenHover}
          />
        );
      })}
    </Flex>
  );
});
AttentionMapGrid.displayName = 'AttentionMapGrid';

const InteractiveImage = memo(({ 
  imageDTO,
  fittedDims,
  hoveredTokenIdx,
  attentionMapImageUrl,
  attentionMapData,
  tokenCount,
  onCrosshairChange,
  onLuminanceChange
}: {
  imageDTO: ImageDTO;
  fittedDims: Dimensions;
  hoveredTokenIdx: number | null;
  attentionMapImageUrl: string | undefined;
  attentionMapData: ImageData | null;
  tokenCount: number;
  onCrosshairChange: (pos: AttentionMapCoordinates | null) => void;
  onLuminanceChange: (values: number[]) => void;
}) => {
  const crossOrigin = useStore($crossOrigin);
  const calculateLuminance = useLuminanceCalculator(attentionMapData, tokenCount);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    if (!attentionMapData || tokenCount === 0) {
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert from display coordinates to image coordinates to attention map coordinates (8x downsampled)
    const scaleX = imageDTO.width / rect.width;
    const scaleY = imageDTO.height / rect.height;
    const attentionX = Math.floor((x * scaleX) / 8);
    const attentionY = Math.floor((y * scaleY) / 8);

    onCrosshairChange({ x: attentionX, y: attentionY });
    onLuminanceChange(calculateLuminance(attentionX, attentionY));
  }, [attentionMapData, tokenCount, imageDTO, calculateLuminance, onCrosshairChange, onLuminanceChange]);

  const handleMouseLeave = useCallback(() => {
    onCrosshairChange(null);
    onLuminanceChange([]);
  }, [onCrosshairChange, onLuminanceChange]);

  return (
    <Box position="relative">
      <Image
        id="image"
        src={imageDTO.image_url}
        fallbackSrc={imageDTO.thumbnail_url}
        crossOrigin={crossOrigin}
        w={fittedDims.width}
        h={fittedDims.height}
        maxW="full"
        maxH="full"
        objectFit="cover"
        objectPosition="top left"
        onMouseMove={handleMouseMove}
        onMouseLeave={handleMouseLeave}
        cursor="crosshair"
      />
      
      {/* Overlay attention map when hovering over a token */}
      {hoveredTokenIdx !== null && attentionMapImageUrl && attentionMapData && (
        <AttentionMapOverlay
          tokenIdx={hoveredTokenIdx}
          attentionMapImageUrl={attentionMapImageUrl}
          attentionMapData={attentionMapData}
          tokenCount={tokenCount}
          fittedDims={fittedDims}
        />
      )}
    </Box>
  );
});
InteractiveImage.displayName = 'InteractiveImage';

// ============================================================================
// Main Component
// ============================================================================

const ImageTokenizationContent = memo(({ 
  image, 
  metadata, 
  fittedDims 
}: { 
  image: ImageDTO; 
  metadata: any; 
  fittedDims: Dimensions;
}) => {
  const tokenizationDisplayMode = useAppSelector(selectTokenizationDisplayMode);
  
  // Extract token metadata
  const tokens = useMemo(() => 
    tokenizationDisplayMode ? extractTokens(metadata, tokenizationDisplayMode) : undefined,
    [metadata, tokenizationDisplayMode]
  );
  
  const attentionMapImageName = useMemo(() => 
    metadata["attention_maps"]["collection"][tokenizationDisplayMode === 'positive' ? 1 : 0]["image_name"],
    [metadata, tokenizationDisplayMode]
  );
  
  const attentionMapsDTO = useImageDTO(attentionMapImageName);
  const attentionMapData = useAttentionMapData(attentionMapsDTO?.image_url);
  
  // State
  const [luminanceValues, setLuminanceValues] = useState<number[]>([]);
  const [crosshairPos, setCrosshairPos] = useState<AttentionMapCoordinates | null>(null);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  
  const tokenCount = tokens?.length || 0;

  if (!tokens || !attentionMapsDTO || !attentionMapData) {
    return null;
  }

  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <TokenList 
        tokens={tokens}
        luminanceValues={luminanceValues}
        hoveredTokenIdx={hoveredTokenIdx}
        onTokenHover={setHoveredTokenIdx}
      />
      
      <Flex gap={4} alignItems="flex-start" position="relative">
        <InteractiveImage
          imageDTO={image}
          fittedDims={fittedDims}
          hoveredTokenIdx={hoveredTokenIdx}
          attentionMapImageUrl={attentionMapsDTO.image_url}
          attentionMapData={attentionMapData}
          tokenCount={tokenCount}
          onCrosshairChange={setCrosshairPos}
          onLuminanceChange={setLuminanceValues}
        />
        
        <AttentionMapGrid
          attentionMapImageUrl={attentionMapsDTO.image_url}
          attentionMapData={attentionMapData}
          tokenCount={tokenCount}
          hoveredTokenIdx={hoveredTokenIdx}
          crosshairPos={crosshairPos}
          maxHeight={fittedDims.height}
          onTokenHover={setHoveredTokenIdx}
        />
      </Flex>
    </Flex>
  );
});
ImageTokenizationContent.displayName = 'ImageTokenizationContent';

// ============================================================================
// Root Component (unchanged)
// ============================================================================

export const ImageTokenization = memo(() => {
  const [rect, setRect] = useState<DOMRect | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const imageDTO = useImageDTO(lastSelectedItem?.type === 'image' ? lastSelectedItem?.id : null);
  const { metadata } = useDebouncedMetadata(imageDTO?.image_name);

  const measureNode = useCallback((node: HTMLDivElement) => {
    if (node) {
      ref.current = node;
      const boundingRect = node.getBoundingClientRect();
      setRect(boundingRect);
    }
  }, []);

  const fittedDims = useMemo<Dimensions>(() => {
    if (!rect || !imageDTO) {
      return { width: 0, height: 0 };
    }
    return fitDimsToContainer(rect, imageDTO);
  }, [imageDTO, rect]);

  return (
    <Flex flexDir="column" w="full" h="full" overflow="hidden" gap={2} position="relative">
      <TokenizationToolbar />
      <Divider />
      <Flex w="full" h="full" position="relative">
        <Box ref={measureNode} w="full" h="full" overflow="hidden">
          {imageDTO && metadata && metadata["attention_maps"] && metadata["tokenization"] && (
            <ImageTokenizationContent 
              image={imageDTO} 
              metadata={metadata} 
              fittedDims={fittedDims} 
            />
          )}
        </Box>
      </Flex>
    </Flex>
  );
});
ImageTokenization.displayName = 'ImageTokenization';
