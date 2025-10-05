import { Box, Divider, Flex, Image } from "@invoke-ai/ui-library";
import { useStore } from '@nanostores/react';
import type { Dimensions } from "@xyflow/react";
import { $crossOrigin } from 'app/store/nanostores/authToken';
import { useAppSelector } from "app/store/storeHooks";
import { selectLastSelectedItem, selectTokenizationDisplayMode } from 'features/gallery/store/gallerySelectors';
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useImageDTO } from "services/api/endpoints/images";
import { useDebouncedMetadata } from "services/api/hooks/useDebouncedMetadata";

import { fitDimsToContainer } from "./common";
import { TokenizationToolbar } from "./TokenizationToolbar";

export const ImageTokenization = memo(() => {

  const [rect, setRect] = useState<DOMRect | null>(null);
  const ref = useRef<HTMLDivElement | null>(null);

  const lastSelectedItem = useAppSelector(selectLastSelectedItem);
  const imageDTO = useImageDTO(lastSelectedItem?.type === 'image' ? lastSelectedItem?.id : null);
  const { metadata, isLoading } = useDebouncedMetadata(imageDTO?.image_name);

  // Ref callback runs synchronously when the DOM node is attached, ensuring we have a measurement before
  // the tokenization content is rendered.
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
          {imageDTO && metadata && metadata["attention_maps"] && metadata["tokenization"] && <ImageTokenizationContent image={imageDTO} metadata={metadata} fittedDims={fittedDims} />}
        </Box>
      </Flex>
    </Flex>
  );


});


const Tokens = ({ 
  tokens, 
  luminanceValues, 
  highlightedTokenIdx,
  onTokenHover
}: { 
  tokens: string[] | undefined; 
  luminanceValues: number[];
  highlightedTokenIdx: number | null;
  onTokenHover: (index: number | null) => void;
}) => {
  const getHeatmapColor = (value: number): string => {
    // Heatmap: blue (0) -> purple (0.5) -> red (1)
    // Blue: (0, 0, 255), Purple: (128, 0, 255), Red: (255, 0, 0)
    // First 10% fades alpha from 0 to 100%
    let r: number; let g: number; let b: number; let alpha: number;

    // Alpha fade in first 20%
    const alphaFadeInRange = 0.2;
    if (value < alphaFadeInRange) {
      alpha = value/alphaFadeInRange; // 0-0.1 maps to 0-1
    } else {
      alpha = 1;
    }
    
    if (value < 0.5) {
      // Blue to Purple (0 to 0.5)
      const t = value * 2; // Normalize to 0-1
      r = Math.round(128 * t);
      g = 0;
      b = 255;
    } else {
      // Purple to Red (0.5 to 1)
      const t = (value - 0.5) * 2; // Normalize to 0-1
      r = Math.round(128 + 127 * t);
      g = 0;
      b = Math.round(255 * (1 - t));
    }
    
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  return <Box mb={2} overflowY="auto">
    <Flex gap={2} flexWrap="wrap">
      {tokens && tokens.map((token, index) => {
        const luminance = Math.pow(luminanceValues[index] || 0, 1);
        const bgColor = luminance > 0 ? getHeatmapColor(luminance) : "base.800";
        const textColor = "white";
        const isHighlighted = highlightedTokenIdx === index;
        
        return (
          <Box
            key={index}
            px={2}
            py={1}
            borderRadius="base"
            bg={bgColor}
            color={textColor}
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
  </Box>;
}

const ImageTokenizationContent = memo(({ image, metadata, fittedDims }: { image: any; metadata: any; fittedDims: Dimensions }) => {
  const crossOrigin = useStore($crossOrigin);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [luminanceValues, setLuminanceValues] = useState<number[]>([]);
  const [attentionMapData, setAttentionMapData] = useState<ImageData | null>(null);
  const [crosshairPos, setCrosshairPos] = useState<{ x: number; y: number } | null>(null);
  const [hoveredTokenIdx, setHoveredTokenIdx] = useState<number | null>(null);
  const tokenizationDisplayMode = useAppSelector(selectTokenizationDisplayMode);

  const attentionMapsDTO = useImageDTO(metadata["attention_maps"]["collection"][tokenizationDisplayMode === 'positive' ? 1 : 0]["image_name"]);

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
    // drop '<bos>' and '<eos>' tokens if present
    return promptUnfilteredTokens.filter((token: string) => token !== '<bos>' && token !== '<eos>');
  }

  const tokens = tokenizationDisplayMode ? extractTokens(metadata, tokenizationDisplayMode) : undefined;
  // Assumption: attention map height is num_tokens * (image height / 8) (8x downsampled)
  const tokenCount = tokens?.length || 0;

  // Load attention maps image onto canvas when available
  useEffect(() => {
    if (!attentionMapsDTO?.image_url || !canvasRef.current) {
      return;
    }

    const img = new window.Image();
    img.crossOrigin = crossOrigin || 'anonymous';
    img.src = attentionMapsDTO.image_url;
    
    img.onload = () => {
      const canvas = canvasRef.current;
      if (!canvas) {
        return;
      }

      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      if (!ctx) {
        return;
      }

      ctx.drawImage(img, 0, 0);
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      setAttentionMapData(imageData);
    };
  }, [attentionMapsDTO?.image_url, crossOrigin]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLImageElement>) => {
    if (!attentionMapData || tokenCount === 0 || !image) {
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert from display coordinates to image coordinates
    const scaleX = image.width / rect.width;
    const scaleY = image.height / rect.height;
    const imageX = x * scaleX;
    const imageY = y * scaleY;

    // Convert to attention map coordinates (8x downsampled)
    const attentionX = Math.floor(imageX / 8);
    const attentionY = Math.floor(imageY / 8);

    // Update crosshair position
    setCrosshairPos({ x: attentionX, y: attentionY });

    // Get attention map dimensions
    const attentionWidth = attentionMapData.width;
    const attentionHeightPerToken = attentionMapData.height / tokenCount;

    // Extract luminance for each token
    const newLuminanceValues: number[] = [];
    for (let tokenIdx = 0; tokenIdx < tokenCount; tokenIdx++) {
      const tokenAttentionY = Math.floor(tokenIdx * attentionHeightPerToken + attentionY);
      
      // Bounds check
      if (attentionX >= 0 && attentionX < attentionWidth && 
          tokenAttentionY >= 0 && tokenAttentionY < attentionMapData.height) {
        
        const pixelIndex = (tokenAttentionY * attentionWidth + attentionX) * 4;
        const r = attentionMapData.data[pixelIndex] ?? 0;
        const g = attentionMapData.data[pixelIndex + 1] ?? 0;
        const b = attentionMapData.data[pixelIndex + 2] ?? 0;
        
        // Calculate relative luminance (normalized to 0-1)
        const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        newLuminanceValues.push(luminance);
      } else {
        newLuminanceValues.push(0);
      }
    }
    
    setLuminanceValues(newLuminanceValues);
  }, [attentionMapData, tokenCount, image]);

  const handleMouseLeave = useCallback(() => {
    setLuminanceValues([]);
    setCrosshairPos(null);
  }, []);

  // Split tokens into chunks of 8 for column display
  const tokensPerColumn = 8;
  const columnCount = Math.ceil(tokenCount / tokensPerColumn);
  const attentionHeightPerToken = attentionMapData ? attentionMapData.height / tokenCount : 0;

  // Store refs to column canvases
  const columnCanvasRefs = useRef<(HTMLCanvasElement | null)[]>([]);

  // Draw column canvases when attention map data is loaded
  useEffect(() => {
    if (!attentionMapsDTO?.image_url || !attentionMapData) {
return;
}

    const img = new window.Image();
    img.crossOrigin = crossOrigin || 'anonymous';
    img.src = attentionMapsDTO.image_url;

    img.onload = () => {
      for (let colIdx = 0; colIdx < columnCount; colIdx++) {
        const canvas = columnCanvasRefs.current[colIdx];
        if (!canvas) {
continue;
}

        const startTokenIdx = colIdx * tokensPerColumn;
        const endTokenIdx = Math.min(startTokenIdx + tokensPerColumn, tokenCount);
        const tokensInColumn = endTokenIdx - startTokenIdx;
        const columnHeight = tokensInColumn * attentionHeightPerToken;

        canvas.width = attentionMapData.width;
        canvas.height = columnHeight;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
continue;
}

        const sourceY = startTokenIdx * attentionHeightPerToken;
        ctx.drawImage(
          img,
          0, sourceY, attentionMapData.width, columnHeight,
          0, 0, attentionMapData.width, columnHeight
        );
      }
    };
  }, [attentionMapsDTO?.image_url, attentionMapData, columnCount, tokensPerColumn, tokenCount, attentionHeightPerToken, crossOrigin]);

  return (
    <Flex flexDir="column" w="full" h="full" gap={2}>
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {tokens && <Tokens 
        tokens={tokens} 
        luminanceValues={luminanceValues}
        highlightedTokenIdx={hoveredTokenIdx}
        onTokenHover={setHoveredTokenIdx}
      />}
      <Flex gap={4} alignItems="flex-start" position="relative">
        <Box position="relative">
          <Image
            id="image"
            src={image.image_url}
            fallbackSrc={image.thumbnail_url}
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
          {hoveredTokenIdx !== null && attentionMapData && (
            <canvas
              ref={(overlayCanvas: HTMLCanvasElement | null) => {
                if (!overlayCanvas || !attentionMapsDTO) {
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
                
                const sourceY = hoveredTokenIdx * heightPerToken;
                const img = new window.Image();
                img.crossOrigin = crossOrigin || 'anonymous';
                img.src = attentionMapsDTO.image_url;
                img.onload = () => {
                  // Draw the grayscale attention map
                  ctx.drawImage(
                    img,
                    0, sourceY, width, heightPerToken,
                    0, 0, width, heightPerToken
                  );
                  
                  // Apply yellow tint to make it more visible
                  const imageData = ctx.getImageData(0, 0, width, heightPerToken);
                  const data = imageData.data;
                  
                  for (let i = 0; i < data.length; i += 4) {
                    const luminance = data[i] ?? 0; // Grayscale, so R=G=B
                    // Convert to yellow gradient
                    data[i] = Math.min(255, luminance * 1.5);     // R - boost red
                    data[i + 1] = Math.min(255, luminance * 1.5); // G - boost green
                    data[i + 2] = 0;                               // B - no blue = yellow
                    data[i + 3] = Math.min(255, luminance * 2);   // A - boost alpha for visibility
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
          )}
        </Box>
        {attentionMapsDTO && attentionMapData && (
          <Flex gap={2} flexWrap="wrap" maxH={fittedDims.height} overflow="auto">
            {Array.from({ length: columnCount }, (_, colIdx) => {
              const startTokenIdx = colIdx * tokensPerColumn;
              const endTokenIdx = Math.min(startTokenIdx + tokensPerColumn, tokenCount);
              const tokensInColumn = endTokenIdx - startTokenIdx;
              const columnHeight = tokensInColumn * attentionHeightPerToken;
              
              return (
                <Box key={colIdx} position="relative" flexShrink={0}>
                  <canvas
                    ref={(canvas) => {
                      columnCanvasRefs.current[colIdx] = canvas;
                    }}
                    style={{
                      width: `${attentionMapsDTO.width}px`,
                      height: `${columnHeight}px`,
                      display: 'block',
                      cursor: 'pointer'
                    }}
                    onMouseMove={(e) => {
                      // Calculate which token in this column is being hovered
                      const rect = e.currentTarget.getBoundingClientRect();
                      const y = e.clientY - rect.top;
                      const localTokenIdx = Math.floor((y / columnHeight) * tokensInColumn);
                      const globalTokenIdx = startTokenIdx + localTokenIdx;
                      setHoveredTokenIdx(globalTokenIdx);
                    }}
                    onMouseLeave={() => {
                      setHoveredTokenIdx(null);
                    }}
                  />
                  {/* Yellow border around the hovered token's attention map */}
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
                  {crosshairPos && Array.from({ length: tokensInColumn }, (_, localTokenIdx) => {
                    const globalTokenIdx = startTokenIdx + localTokenIdx;
                    const yOffset = localTokenIdx * attentionHeightPerToken;
                    
                    return (
                      <Box
                        key={globalTokenIdx}
                        position="absolute"
                        left={`${crosshairPos.x}px`}
                        top={`${yOffset + crosshairPos.y}px`}
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
                    );
                  })}
                </Box>
              );
            })}
          </Flex>
        )}
      </Flex>
    </Flex>
  );
});
ImageTokenization.displayName = 'ImageTokenization';
